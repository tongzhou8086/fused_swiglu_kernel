"""Fused SwiGLU forward kernel optimization project.

Goal: improve the forward kernel used by ``save_factors_inplace`` — a
wide-accumulator GEMM with an activation epilogue and a factor side
store ``[M, 2N] bf16``.  Baseline is the colleague's Triton kernel
``_fused_swiglu_wide_packed_save_factors_kernel`` from upstream
``swiglu_fused/swiglu/swiglu_layer/fused_swiglu_wide_packed.py``,
copied here as ``triton_baseline.py``.
"""

from .triton_baseline import (
    fused_swiglu_wide_packed,
    fused_swiglu_wide_packed_save_factors,
    pack_swiglu_weight_chunked_torch,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N_HALF,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
    NUM_WARPS,
    NUM_STAGES,
    SAVE_NUM_STAGES,
    WARP_SPECIALIZE,
)
