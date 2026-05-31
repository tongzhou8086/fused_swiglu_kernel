"""B200 SwiGLU projection experiments with packed W1 layout.

This file intentionally keeps only the two active variants:

1. ``packed_save_factors_inplace``
   Forward computes ``out`` and saves packed backward factors.  Backward
   overwrites those factors with ``grad_de`` and uses cuBLAS for the two GEMMs.

2. ``packed_no_save_cublas_recompute``
   Forward saves no activation.  Backward recomputes ``x @ W1`` with cuBLAS,
   overwrites that temporary preactivation with ``grad_de``, then uses cuBLAS
   for the two GEMMs.

The packed weight layout is ``[left chunk 0 | gate chunk 0 | left chunk 1 |
gate chunk 1 | ...]`` over the output dimension.  That lets each CTA compute
left/gate together with one wide accumulator.
"""

from __future__ import annotations

import functools
import math

import torch
import triton
import triton.language as tl


BLOCK_SIZE_M = 128
BLOCK_SIZE_N_HALF = 128
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 32
NUM_WARPS = 8
NUM_STAGES = 4
SAVE_NUM_STAGES = 4
WARP_SPECIALIZE = True
USE_TILE_ID_C = False

BWD_FACTORS_BLOCK_SIZE_M = 64
BWD_FACTORS_BLOCK_SIZE_N_HALF = 128
BWD_FACTORS_NUM_WARPS = 4

PACK_BLOCK_K = 16
PACK_NUM_WARPS = 8


def _tma_alloc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _ensure_allocator() -> None:
    triton.set_allocator(_tma_alloc)


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _packed_grad_input_weight(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    grad_de: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x2 = x.view(-1, x.shape[-1])
    grad_x = grad_de @ packed_weight.t()
    grad_weight = x2.t().to(packed_weight.dtype) @ grad_de
    return grad_x, grad_weight


@triton.jit
def _compute_pid(
    tile_id,
    num_pid_in_group: tl.constexpr,
    num_pid_m: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M_
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M_)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _fused_swiglu_wide_packed_kernel(
    a_ptr,
    bp_ptr,
    c_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    USE_TILE_ID_C_: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    bp_desc = tl.make_tensor_descriptor(
        bp_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N2],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )

    tile_id_c = start_pid - NUM_SMS
    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n2 = pid_n * BLOCK_SIZE_N2

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N2), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])
            b = bp_desc.load([offs_k, offs_n2])
            acc = tl.dot(a, b, acc)

        acc3 = tl.reshape(acc, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
        acc3 = tl.permute(acc3, (0, 2, 1))
        left, gate = tl.split(acc3)
        out = left * (gate * tl.sigmoid(gate))

        if USE_TILE_ID_C_:
            tile_id_c += NUM_SMS
            pid_m, pid_n = _compute_pid(
                tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
            )
            offs_m = pid_m * BLOCK_SIZE_M_

        c_desc.store(
            [offs_m, pid_n * BLOCK_SIZE_N_HALF_],
            out.to(c_ptr.dtype.element_ty),
        )


@triton.jit
def _fused_swiglu_wide_packed_save_factors_kernel(
    a_ptr,
    bp_ptr,
    c_ptr,
    factors_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    bp_desc = tl.make_tensor_descriptor(
        bp_ptr,
        shape=[K, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N2],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    factors_desc = tl.make_tensor_descriptor(
        factors_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n = pid_n * BLOCK_SIZE_N_HALF_
        offs_n2 = pid_n * BLOCK_SIZE_N2

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N2), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])
            b = bp_desc.load([offs_k, offs_n2])
            acc = tl.dot(a, b, acc)

        acc3 = tl.reshape(acc, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
        acc3 = tl.permute(acc3, (0, 2, 1))
        left, gate = tl.split(acc3)

        sig = tl.sigmoid(gate)
        silu = gate * sig
        silu_prime = sig + silu * (1.0 - sig)
        factor_gate = left * silu_prime

        factors_desc.store(
            [offs_m, offs_n2],
            silu.to(factors_ptr.dtype.element_ty),
        )
        factors_desc.store(
            [offs_m, offs_n2 + BLOCK_SIZE_N_HALF_],
            factor_gate.to(factors_ptr.dtype.element_ty),
        )
        c_desc.store(
            [offs_m, offs_n],
            (left * silu).to(c_ptr.dtype.element_ty),
        )


@triton.jit
def _swiglu_packed_grad_de_from_preact_kernel(
    preact_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_
    offs_n2 = pid_n * BLOCK_SIZE_N2

    preact_desc = tl.make_tensor_descriptor(
        preact_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    grad_desc = tl.make_tensor_descriptor(
        grad_de_ptr,
        shape=[M, N_HALF * 2],
        strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2],
    )

    preact = preact_desc.load([offs_m, offs_n2]).to(tl.float32)
    preact3 = tl.reshape(preact, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
    preact3 = tl.permute(preact3, (0, 2, 1))
    left, gate = tl.split(preact3)

    dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
    sig = tl.sigmoid(gate)
    silu = gate * sig
    silu_prime = sig + silu * (1.0 - sig)
    grad_left = dy * silu
    grad_gate = dy * left * silu_prime
    grad_de = tl.cat(grad_left, grad_gate, dim=1)
    grad_desc.store([offs_m, offs_n2], grad_de.to(grad_de_ptr.dtype.element_ty))


@triton.jit
def _swiglu_packed_grad_de_from_factors_ptr_kernel(
    factors_ptr,
    dy_ptr,
    grad_de_ptr,
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_
    offs_n2 = pid_n * BLOCK_SIZE_N2
    rows = offs_m + tl.arange(0, BLOCK_SIZE_M_)
    cols = tl.arange(0, BLOCK_SIZE_N_HALF_)

    dy = tl.load(
        dy_ptr + rows[:, None] * N_HALF + (offs_n + cols)[None, :]
    ).to(tl.float32)
    factor_left = tl.load(
        factors_ptr + rows[:, None] * (N_HALF * 2) + (offs_n2 + cols)[None, :]
    ).to(tl.float32)
    factor_gate = tl.load(
        factors_ptr
        + rows[:, None] * (N_HALF * 2)
        + (offs_n2 + BLOCK_SIZE_N_HALF_ + cols)[None, :]
    ).to(tl.float32)
    tl.store(
        grad_de_ptr + rows[:, None] * (N_HALF * 2) + (offs_n2 + cols)[None, :],
        (dy * factor_left).to(grad_de_ptr.dtype.element_ty),
    )
    tl.store(
        grad_de_ptr
        + rows[:, None] * (N_HALF * 2)
        + (offs_n2 + BLOCK_SIZE_N_HALF_ + cols)[None, :],
        (dy * factor_gate).to(grad_de_ptr.dtype.element_ty),
    )


@triton.jit
def _pack_swiglu_weight_chunked_kernel(
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    N_HALF: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N_HALF_: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    N2: tl.constexpr = N_HALF * 2

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N_HALF_ + tl.arange(0, BLOCK_N_HALF_)
    dst_base = pid_n * BLOCK_N_HALF_ * 2

    src_left = offs_k[:, None] * N2 + offs_n[None, :]
    src_gate = offs_k[:, None] * N2 + (N_HALF + offs_n)[None, :]
    dst_left = (
        offs_k[:, None] * N2
        + (dst_base + tl.arange(0, BLOCK_N_HALF_))[None, :]
    )
    dst_gate = (
        offs_k[:, None] * N2
        + (dst_base + BLOCK_N_HALF_ + tl.arange(0, BLOCK_N_HALF_))[None, :]
    )
    mask = (offs_k[:, None] < K) & (offs_n[None, :] < N_HALF)

    left = tl.load(weight_ptr + src_left, mask=mask)
    gate = tl.load(weight_ptr + src_gate, mask=mask)
    tl.store(out_ptr + dst_left, left, mask=mask)
    tl.store(out_ptr + dst_gate, gate, mask=mask)


def pack_swiglu_weight_chunked_into(
    weight: torch.Tensor,
    out: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Pack [K, 2*N] into chunk-interleaved output using torch copies."""
    assert weight.is_contiguous()
    assert out.is_contiguous()
    assert out.shape == weight.shape and out.dtype == weight.dtype
    k, n2 = weight.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % block_n_half == 0

    chunks = n_half // block_n_half
    left = weight[:, :n_half].view(k, chunks, block_n_half)
    gate = weight[:, n_half:].view(k, chunks, block_n_half)
    packed = out.view(k, chunks, 2, block_n_half)
    packed[:, :, 0, :].copy_(left)
    packed[:, :, 1, :].copy_(gate)
    return out


def pack_swiglu_weight_chunked_into_triton(
    weight: torch.Tensor,
    out: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Pack [K, 2*N] with a Triton copy kernel."""
    assert weight.is_cuda and weight.is_contiguous()
    assert out.is_cuda and out.is_contiguous()
    assert out.shape == weight.shape and out.dtype == weight.dtype
    assert block_n_half == BLOCK_SIZE_N_HALF

    k, n2 = weight.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % block_n_half == 0

    grid = (triton.cdiv(k, PACK_BLOCK_K), triton.cdiv(n_half, block_n_half))
    _pack_swiglu_weight_chunked_kernel[grid](
        weight,
        out,
        k,
        n_half,
        BLOCK_K=PACK_BLOCK_K,
        BLOCK_N_HALF_=block_n_half,
        num_warps=PACK_NUM_WARPS,
    )
    return out


def pack_swiglu_weight_chunked(
    weight: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Pack [K, 2*N] SwiGLU weight into chunk-interleaved layout."""
    if weight.is_cuda:
        return pack_swiglu_weight_chunked_into_triton(
            weight, torch.empty_like(weight), block_n_half
        )
    return pack_swiglu_weight_chunked_into(
        weight, torch.empty_like(weight), block_n_half
    )


def pack_swiglu_weight_chunked_torch(
    weight: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Pack [K, 2*N] using regular torch copies."""
    return pack_swiglu_weight_chunked_into(weight, torch.empty_like(weight), block_n_half)


def unpack_swiglu_weight_chunked_torch(
    packed_weight: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Undo chunk-interleaved packing back to [K, 2*N]."""
    assert packed_weight.is_contiguous()
    k, n2 = packed_weight.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % block_n_half == 0

    out = torch.empty_like(packed_weight, requires_grad=False)
    chunks = n_half // block_n_half
    packed = packed_weight.view(k, chunks, 2, block_n_half)
    left = out[:, :n_half].view(k, chunks, block_n_half)
    gate = out[:, n_half:].view(k, chunks, block_n_half)
    left.copy_(packed[:, :, 0, :])
    gate.copy_(packed[:, :, 1, :])
    return out


def pack_swiglu_linear_weight(
    linear_weight: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Pack a Linear weight [2*N, K] into internal [K, 2*N]."""
    assert linear_weight.ndim == 2
    return pack_swiglu_weight_chunked_torch(
        linear_weight.t().contiguous(), block_n_half
    )


def unpack_swiglu_linear_weight(
    packed_weight: torch.Tensor,
    block_n_half: int = BLOCK_SIZE_N_HALF,
) -> torch.Tensor:
    """Return a Linear weight [2*N, K] from internal packed storage."""
    assert packed_weight.ndim == 2
    return unpack_swiglu_weight_chunked_torch(
        packed_weight.contiguous(), block_n_half
    ).t().contiguous()


def fused_swiglu_wide_packed(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    use_tile_id_c: bool = USE_TILE_ID_C,
) -> torch.Tensor:
    """No-save forward: compute ``swiglu(x @ packed_weight)``."""
    _ensure_allocator()
    assert x.is_cuda and packed_weight.is_cuda
    assert x.is_contiguous() and packed_weight.is_contiguous()

    m, k = x.shape
    k2, n2 = packed_weight.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % BLOCK_SIZE_N_HALF == 0

    out = torch.empty((m, n_half), device=x.device, dtype=x.dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    _fused_swiglu_wide_packed_kernel[grid](
        x,
        packed_weight,
        out,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=BLOCK_SIZE_K,
        GROUP_SIZE_M_=GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        USE_TILE_ID_C_=use_tile_id_c,
        FLATTEN=True,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


def fused_swiglu_wide_packed_save_factors(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    factors_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward that saves packed backward factors for the fast training path."""
    _ensure_allocator()
    assert x.is_cuda and packed_weight.is_cuda
    assert x.is_contiguous() and packed_weight.is_contiguous()

    m, k = x.shape
    k2, n2 = packed_weight.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % BLOCK_SIZE_N_HALF == 0

    out = torch.empty((m, n_half), device=x.device, dtype=x.dtype)
    if factors_dtype is None:
        factors_dtype = x.dtype
    factors = torch.empty((m, n2), device=x.device, dtype=factors_dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    _fused_swiglu_wide_packed_save_factors_kernel[grid](
        x,
        packed_weight,
        out,
        factors,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=BLOCK_SIZE_K,
        GROUP_SIZE_M_=GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        FLATTEN=True,
        num_warps=NUM_WARPS,
        num_stages=SAVE_NUM_STAGES,
    )
    return out, factors


def swiglu_packed_grad_de_cublas_recompute(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Recompute preactivation with cuBLAS, then overwrite it with grad_de."""
    assert x.is_cuda and packed_weight.is_cuda and grad_out.is_cuda
    if not x.is_contiguous():
        x = x.contiguous()
    if not packed_weight.is_contiguous():
        packed_weight = packed_weight.contiguous()
    preact = x @ packed_weight
    return swiglu_packed_grad_de_from_preact_inplace(preact, grad_out)


def swiglu_packed_grad_de_from_preact_inplace(
    preact: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Overwrite packed preactivation with packed grad_de and return it."""
    _ensure_allocator()
    assert preact.is_cuda and grad_out.is_cuda
    assert preact.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = preact.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0

    grid = (triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),)
    _swiglu_packed_grad_de_from_preact_kernel[grid](
        preact,
        grad_out,
        preact,
        m,
        n_half,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        num_warps=NUM_WARPS,
    )
    return preact


def swiglu_packed_grad_de_from_factors_inplace(
    factors: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """Overwrite packed factors with packed grad_de and return the same tensor."""
    _ensure_allocator()
    assert factors.is_cuda and grad_out.is_cuda
    assert factors.is_contiguous()
    if not grad_out.is_contiguous():
        grad_out = grad_out.contiguous()

    m, n2 = factors.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert grad_out.shape == (m, n_half)
    assert n_half % BLOCK_SIZE_N_HALF == 0

    grid = (
        triton.cdiv(m, BWD_FACTORS_BLOCK_SIZE_M)
        * triton.cdiv(n_half, BWD_FACTORS_BLOCK_SIZE_N_HALF),
    )
    _swiglu_packed_grad_de_from_factors_ptr_kernel[grid](
        factors,
        grad_out,
        factors,
        m,
        n_half,
        BLOCK_SIZE_M_=BWD_FACTORS_BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BWD_FACTORS_BLOCK_SIZE_N_HALF,
        num_warps=BWD_FACTORS_NUM_WARPS,
    )
    return factors


class _FusedSwiGLUPackedCublasRecomputeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        ctx.save_for_backward(x, packed_weight)
        ctx.x_shape = x.shape
        return fused_swiglu_wide_packed(x, packed_weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, packed_weight = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_packed_grad_de_cublas_recompute(
            x, packed_weight, grad_out
        )
        grad_x, grad_weight = _packed_grad_input_weight(x, packed_weight, grad_de)
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_wide_packed_cublas_recompute_autograd(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    """No-save forward; backward recomputes preactivation with cuBLAS."""
    return _FusedSwiGLUPackedCublasRecomputeFn.apply(x, packed_weight)


class _FusedSwiGLUPackedSaveFactorsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_weight: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        out, factors = fused_swiglu_wide_packed_save_factors(x, packed_weight)
        ctx.save_for_backward(x, packed_weight, factors)
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, packed_weight, factors = ctx.saved_tensors
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_de = swiglu_packed_grad_de_from_factors_inplace(factors, grad_out)
        ctx.maybe_clear_saved_tensors()
        grad_x, grad_weight = _packed_grad_input_weight(x, packed_weight, grad_de)
        return grad_x.view(ctx.x_shape), grad_weight


def fused_swiglu_wide_packed_save_factors_autograd(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
) -> torch.Tensor:
    """Fast training path: save backward factors and reuse them in-place."""
    return _FusedSwiGLUPackedSaveFactorsFn.apply(x, packed_weight)


class PackedSwiGLULinear(torch.nn.Module):
    """Minimal packed SwiGLU projection module for the retained variants."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        block_n_half: int = BLOCK_SIZE_N_HALF,
        backward_mode: str = "save_factors",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        assert backward_mode in {"save_factors", "no_save_cublas_recompute"}
        assert hidden_features % block_n_half == 0
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.block_n_half = block_n_half
        self.backward_mode = backward_mode
        self.weight = torch.nn.Parameter(
            torch.empty(in_features, hidden_features * 2, device=device, dtype=dtype)
        )
        self.reset_parameters()

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        *,
        block_n_half: int = BLOCK_SIZE_N_HALF,
        backward_mode: str = "save_factors",
    ) -> "PackedSwiGLULinear":
        assert linear.bias is None
        assert linear.out_features % 2 == 0
        module = cls(
            linear.in_features,
            linear.out_features // 2,
            block_n_half=block_n_half,
            backward_mode=backward_mode,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        module.load_linear_weight_(linear.weight)
        return module

    @torch.no_grad()
    def reset_parameters(self) -> None:
        normal_weight = torch.empty(
            self.hidden_features * 2,
            self.in_features,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        torch.nn.init.normal_(
            normal_weight[: self.hidden_features],
            std=1.0 / math.sqrt(self.in_features),
        )
        torch.nn.init.kaiming_normal_(normal_weight[self.hidden_features :])
        self.load_linear_weight_(normal_weight)

    @torch.no_grad()
    def load_linear_weight_(self, linear_weight: torch.Tensor) -> None:
        expected = (self.hidden_features * 2, self.in_features)
        assert tuple(linear_weight.shape) == expected
        packed = pack_swiglu_linear_weight(linear_weight.detach().contiguous())
        self.weight.copy_(packed.to(device=self.weight.device, dtype=self.weight.dtype))

    def linear_weight(self) -> torch.Tensor:
        with torch.no_grad():
            return unpack_swiglu_linear_weight(self.weight, self.block_n_half)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        weight = self.linear_weight()
        if not keep_vars:
            weight = weight.detach()
        destination[prefix + "weight"] = weight

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        key = prefix + "weight"
        original = state_dict.get(key)
        if original is not None:
            normal_shape = (self.hidden_features * 2, self.in_features)
            packed_shape = (self.in_features, self.hidden_features * 2)
            if tuple(original.shape) == normal_shape:
                state_dict[key] = pack_swiglu_linear_weight(
                    original.detach().contiguous(), self.block_n_half
                )
            elif tuple(original.shape) != packed_shape:
                error_msgs.append(
                    f"size mismatch for {key}: copying a param with shape "
                    f"{tuple(original.shape)} from checkpoint, expected "
                    f"{normal_shape} linear or {packed_shape} packed"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if original is not None:
            state_dict[key] = original

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[-1])
        if not x.is_contiguous():
            x = x.contiguous()
        if self.backward_mode == "no_save_cublas_recompute":
            out = fused_swiglu_wide_packed_cublas_recompute_autograd(x, self.weight)
        else:
            out = fused_swiglu_wide_packed_save_factors_autograd(x, self.weight)
        return out.view(*x_shape[:-1], self.hidden_features)

