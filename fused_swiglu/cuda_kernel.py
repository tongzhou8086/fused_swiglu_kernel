"""CUDA launcher for the b42-based save_factors fused kernel.

Mirrors the swiglu_fused/swiglu/matmul_silu_mul.py pattern: pycuda
loads the .cu and JIT-compiles into a cubin cached on disk.  We expose
a single Python function `matmul_save_factors(x, W_packed)` that
returns `(out, factors)`.
"""
import os
import numpy as np
import torch
import pycuda.driver as drv

from ._pycuda_loader import get_module_jit, SM_ARCH
from . import _tma_utils as tma

DTYPE = torch.bfloat16
_HERE = os.path.dirname(os.path.abspath(__file__))
_CU_PATH = os.path.join(_HERE, "_matmul_save_factors.cu")
_CUBIN   = os.path.join(_HERE, f"_matmul_save_factors_{SM_ARCH}.cubin")

BM = 128
NW = 8
CTA_GROUP = 2
NUM_SMS = 148   # B200

# Match the (BN, BK, NS, GSM) launchers emitted in the .cu file.
_DEFAULT_CONFIG = (256, 64, 6, 32)        # mirrors b42's tuned production config


def _get_mod():
    return get_module_jit(_CU_PATH, _CUBIN, ["-arch=sm_100a", "-DLB_MIN_BLOCKS=1"])


def _kname(bn, bk, ns, gsm, persistent: bool = False):
    pers = "_pers" if persistent else ""
    return f"matmul_save_factors{pers}_bm{BM}_bn{bn}_bk{bk}_ns{ns}_gsm{gsm}"


def _smem_bytes(bn, bk, ns):
    """Dynamic SMEM size for the K-loop ring (epilogue aliases on top)."""
    bn_local = bn // CTA_GROUP
    return ns * (BM + bn_local) * bk * 2


_tmap_cache: dict = {}


def _setup(A, W_packed, bn, bk):
    key = (A.data_ptr(), W_packed.data_ptr(), bn, bk)
    hit = _tmap_cache.get(key)
    if hit is not None:
        return hit
    M, K = A.shape
    _, twoN = W_packed.shape
    A_tmap = tma.build_tma_2d(A.data_ptr(),        M,    K,    BM, 64, tma.SWIZZLE_128B)
    B_tmap = tma.build_tma_2d(W_packed.data_ptr(), K,    twoN, bk, 64, tma.SWIZZLE_128B)
    _tmap_cache[key] = (A_tmap, B_tmap)
    return A_tmap, B_tmap


def _pad_M(M: int) -> int:
    """Round M up to the next multiple of (2 * BM) so the cluster pairs
    along M tile cleanly."""
    step = 2 * BM
    return ((M + step - 1) // step) * step


# Padded-tensor cache keyed by ptr so we don't re-pad on hot paths.
_padA_cache: dict = {}
_padW_cache: dict = {}


def _maybe_pad_A(A, M_pad):
    """If A's M doesn't tile cleanly, return a zero-padded copy of shape (M_pad, K)."""
    M, K = A.shape
    if M == M_pad:
        return A
    key = (A.data_ptr(), M_pad, K)
    hit = _padA_cache.get(key)
    if hit is not None and hit.shape == (M_pad, K):
        # Refresh contents — A may have been overwritten.
        hit[:M].copy_(A)
        hit[M:].zero_()
        return hit
    A_pad = torch.zeros(M_pad, K, device=A.device, dtype=A.dtype)
    A_pad[:M].copy_(A)
    _padA_cache[key] = A_pad
    return A_pad


def matmul_save_factors(A, W_packed, config=_DEFAULT_CONFIG, persistent: bool = False):
    """Computes `out = left * silu(gate)` and `factors = [silu | left·silu']`.

    A          : [M, K]      bf16 row-major
    W_packed   : [K, 2*N]    bf16 row-major, chunk-interleaved
    config     : (BN, BK, NS, GSM) — must match a launcher in the .cu file
    persistent : if True, use the persistent-grid variant (grid = NUM_SMS).
                 Only NS=4 and NS=7 have persistent launchers compiled.

    Returns:
      out      : [M, N]     bf16 row-major
      factors  : [M, 2*N]   bf16 row-major, chunked layout matching W_packed

    NOTE: this kernel uses cta_group::2 cluster MMA, which requires the
    M-tile count to be even (and the persistent variant requires the
    cluster-tile count to be well-defined).  At M=11136, M/BM=87 is odd,
    so we pad M to the next 2*BM multiple (11264) and slice the output
    back.  Costs ~1.1 % extra compute at this shape.
    """
    bn, bk, ns, gsm = config
    M, K = A.shape
    _, twoN = W_packed.shape
    assert twoN % 2 == 0
    N = twoN // 2
    assert twoN % bn == 0
    assert K % bk == 0
    if persistent:
        assert ns in (4, 7), f"persistent variant only compiled for NS=4,7 (got NS={ns})"

    M_pad = _pad_M(M)
    assert M_pad % BM == 0 and (M_pad // BM) % 2 == 0, f"M_pad={M_pad}"

    A_use = _maybe_pad_A(A, M_pad)

    out_pad = torch.empty(M_pad, N,    device="cuda", dtype=DTYPE)
    fac_pad = torch.empty(M_pad, twoN, device="cuda", dtype=DTYPE)

    A_tmap, B_tmap = _setup(A_use, W_packed, bn, bk)
    mod = _get_mod()
    fn  = mod.get_function(_kname(bn, bk, ns, gsm, persistent=persistent))

    smem = _smem_bytes(bn, bk, ns)
    if smem > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)

    if persistent:
        grid_x = NUM_SMS                                                      # 148 on B200
    else:
        grid_x = (M_pad // BM) * (twoN // bn)                                 # one CTA per tile
    grid  = (grid_x, 1, 1)
    block = (NW * 32, 1, 1)
    fn(A_tmap, B_tmap,
       np.intp(out_pad.data_ptr()), np.intp(fac_pad.data_ptr()),
       np.int32(M_pad), np.int32(N), np.int32(K),
       block=block, grid=grid, shared=smem)

    if M_pad == M:
        return out_pad, fac_pad
    return out_pad[:M], fac_pad[:M]
