"""CUDA launcher for the Path B (K-loop / epilogue overlap) variant.

K-loop runs concurrently with the previous tile's epilogue via:
  - Double-buffered TMEM (2 × BLOCK_N = 512 cols)
  - Warp role split: warp 0 = TMA (continuous), warp 1 (CTA 0) = MMA
    (continuous, alternates TMEM halves), warps 4..7 = epilogue
  - Per-tile mbarriers (mma_done_per_tile[2], epi_done_per_tile[2])
  - NS=4 only (SMEM budget: 128 KB ring + ~100 KB staging ≈ 228 KB cap)
"""
import os
import numpy as np
import torch
import pycuda.driver as drv

from ._pycuda_loader import get_module_jit, SM_ARCH
from . import _tma_utils as tma

DTYPE = torch.bfloat16
_HERE = os.path.dirname(os.path.abspath(__file__))
_CU_PATH = os.path.join(_HERE, "_matmul_save_factors_b.cu")
_CUBIN   = os.path.join(_HERE, f"_matmul_save_factors_b_{SM_ARCH}.cubin")

BM = 128
NW = 8
CTA_GROUP = 2
NUM_SMS = 148   # B200

_DEFAULT_CONFIG = (256, 64, 4, 16)        # Path B is NS=4 only


def _get_mod():
    return get_module_jit(_CU_PATH, _CUBIN, ["-arch=sm_100a", "-DLB_MIN_BLOCKS=1"])


def _kname(bn, bk, ns, gsm):
    return f"matmul_save_factors_b_bm{BM}_bn{bn}_bk{bk}_ns{ns}_gsm{gsm}"


def _smem_bytes(bn, bk, ns):
    """Ring + dedicated OUT_sh + dedicated FAC_sh (non-aliased, NO padding)."""
    bn_local = bn // CTA_GROUP
    bn_half  = bn // 2
    ring     = ns * (BM + bn_local) * bk * 2
    out_sh   = BM * bn_half * 2
    fac_sh   = BM * bn * 2
    return ring + out_sh + fac_sh


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
    step = 2 * BM
    return ((M + step - 1) // step) * step


_padA_cache: dict = {}


def _maybe_pad_A(A, M_pad):
    M, K = A.shape
    if M == M_pad:
        return A
    key = (A.data_ptr(), M_pad, K)
    hit = _padA_cache.get(key)
    if hit is not None and hit.shape == (M_pad, K):
        hit[:M].copy_(A)
        hit[M:].zero_()
        return hit
    A_pad = torch.zeros(M_pad, K, device=A.device, dtype=A.dtype)
    A_pad[:M].copy_(A)
    _padA_cache[key] = A_pad
    return A_pad


def matmul_save_factors_b(A, W_packed, config=_DEFAULT_CONFIG):
    bn, bk, ns, gsm = config
    assert ns == 4, "Path B is NS=4 only"
    M, K = A.shape
    _, twoN = W_packed.shape
    assert twoN % 2 == 0
    N = twoN // 2
    assert twoN % bn == 0
    assert K % bk == 0

    M_pad = _pad_M(M)
    A_use = _maybe_pad_A(A, M_pad)

    out_pad = torch.empty(M_pad, N,    device="cuda", dtype=DTYPE)
    fac_pad = torch.empty(M_pad, twoN, device="cuda", dtype=DTYPE)

    A_tmap, B_tmap = _setup(A_use, W_packed, bn, bk)
    mod = _get_mod()
    fn  = mod.get_function(_kname(bn, bk, ns, gsm))

    smem = _smem_bytes(bn, bk, ns)
    if smem > 0:
        fn.set_attribute(drv.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, smem)

    grid  = (NUM_SMS, 1, 1)
    block = (NW * 32, 1, 1)
    fn(A_tmap, B_tmap,
       np.intp(out_pad.data_ptr()), np.intp(fac_pad.data_ptr()),
       np.int32(M_pad), np.int32(N), np.int32(K),
       block=block, grid=grid, shared=smem)

    if M_pad == M:
        return out_pad, fac_pad
    return out_pad[:M], fac_pad[:M]
