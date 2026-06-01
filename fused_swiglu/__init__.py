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

# CUDA implementation (b42-based).  Compiled on first use.
try:
    from .cuda_kernel import matmul_save_factors as cuda_matmul_save_factors
    HAS_CUDA_KERNEL = True
except Exception as _e:
    HAS_CUDA_KERNEL = False

# x64-load variant (wider tcgen05.ld).
try:
    from .cuda_kernel_x64 import matmul_save_factors_x64 as cuda_matmul_save_factors_x64
    HAS_CUDA_KERNEL_X64 = True
except Exception as _e:
    HAS_CUDA_KERNEL_X64 = False

# tanh-form sigmoid variant (1 SFU op per element instead of 2).
try:
    from .cuda_kernel_tanh import matmul_save_factors_tanh as cuda_matmul_save_factors_tanh
    HAS_CUDA_KERNEL_TANH = True
except Exception as _e:
    HAS_CUDA_KERNEL_TANH = False

# No-SMEM-staging variant (direct TMEM→regs→GMEM, uncoalesced writes).
try:
    from .cuda_kernel_nostg import matmul_save_factors_nostg as cuda_matmul_save_factors_nostg
    HAS_CUDA_KERNEL_NOSTG = True
except Exception as _e:
    HAS_CUDA_KERNEL_NOSTG = False
