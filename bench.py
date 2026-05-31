"""Benchmark the forward fused SwiGLU kernel with factor side store.

The kernel under test is `_fused_swiglu_wide_packed_save_factors_kernel`
(invoked via the `fused_swiglu_wide_packed_save_factors` wrapper).  It
does, in one persistent + flatten + warp-specialized Triton kernel:

    [left | gate] = A @ W_packed                        (wide-acc GEMM)
    sig            = sigmoid(gate)                       (in registers)
    silu           = gate * sig                          (in registers)
    silu_prime     = sig + silu * (1 - sig)              (in registers)
    out            = left * silu                         → [M, N]    main store
    factors        = [silu | left * silu_prime]          → [M, 2N]   side store

We compare to:

  cuBLAS GEMM only            x @ W_packed → [M, 2N]   (no activation, no side store)
  Triton T1 (no side store)   fused_swiglu_wide_packed → [M, N]
  cuBLAS + torch.compile act  matches save_factors' inputs/outputs but
                              without the factor side store

All at the colleague's shape M=11136 K=3584 N=14336 (BF16, B200).
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F
import triton.testing as tt

import fused_swiglu as fs


# ── Shape ────────────────────────────────────────────────────────────────
M, K, N = 11136, 3584, 14336
DTYPE = torch.bfloat16
torch.manual_seed(0)

# A single forward GEMM [M, K] @ [K, 2N] does 2·M·K·2N FLOPs.
FLOPS_FWD = 2 * M * K * (2 * N)
B200_PEAK = 2250e12   # BF16 dense, TFLOPS×1e12


# ── Inputs ───────────────────────────────────────────────────────────────
def make_inputs():
    x        = torch.randn(M, K,     device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
    W_normal = torch.randn(K, 2 * N, device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
    W_packed = fs.pack_swiglu_weight_chunked_torch(W_normal)
    return x, W_normal, W_packed


# ── Reference: fp32 SwiGLU on chunked-layout preact ──────────────────────
def swiglu_eager_chunked(preact_chunked):
    """Eager swiglu on a CHUNKED preact [M, 2N]."""
    BNH = fs.BLOCK_SIZE_N_HALF
    M_, N2 = preact_chunked.shape
    N_ = N2 // 2
    n_chunks = N_ // BNH
    p = preact_chunked.view(M_, n_chunks, 2, BNH)
    return (p[:, :, 0, :] * F.silu(p[:, :, 1, :])).reshape(M_, N_)


def factors_ref_chunked(preact_chunked):
    """Reference factors = [silu(gate) | left * silu'(gate)] in chunked layout."""
    BNH = fs.BLOCK_SIZE_N_HALF
    M_, N2 = preact_chunked.shape
    N_ = N2 // 2
    n_chunks = N_ // BNH
    p = preact_chunked.view(M_, n_chunks, 2, BNH)
    left, gate = p[:, :, 0, :], p[:, :, 1, :]
    sig         = torch.sigmoid(gate)
    silu        = gate * sig
    silu_prime  = sig + silu * (1.0 - sig)
    factor_gate = left * silu_prime
    # Pack back into chunked [silu | factor_gate] layout
    factors = torch.empty_like(preact_chunked)
    f = factors.view(M_, n_chunks, 2, BNH)
    f[:, :, 0, :] = silu
    f[:, :, 1, :] = factor_gate
    return factors


# ── Correctness ──────────────────────────────────────────────────────────
def correctness(x, W_normal, W_packed):
    print(f"=== correctness (M={M}  K={K}  N={N}) ===")
    # fp32 reference, computed via the chunked-preact path so the layouts match.
    preact32 = (x.float() @ W_packed.float())
    out_ref  = swiglu_eager_chunked(preact32)
    fac_ref  = factors_ref_chunked(preact32)

    out_k, fac_k = fs.fused_swiglu_wide_packed_save_factors(x, W_packed)
    err_out = (out_k.float() - out_ref).abs().max().item()
    err_fac = (fac_k.float() - fac_ref).abs().max().item()
    atol = max(1.0, K ** 0.5 / 16)
    ok = err_out <= atol and err_fac <= atol
    print(f"  save_factors  : out max_abs={err_out:.3e}  factors max_abs={err_fac:.3e}  "
          f"atol={atol:.2f}  →  {'OK' if ok else 'FAIL'}")

    # T1 sanity check.
    out_t1 = fs.fused_swiglu_wide_packed(x, W_packed)
    err_t1 = (out_t1.float() - out_ref).abs().max().item()
    print(f"  T1            : out max_abs={err_t1:.3e}  →  {'OK' if err_t1 <= atol else 'FAIL'}")
    return ok and err_t1 <= atol


# ── Bench helpers ────────────────────────────────────────────────────────
def bench(fn, label):
    fn(); torch.cuda.synchronize()
    ms, _, _ = tt.do_bench(fn, warmup=20, rep=200, quantiles=(0.5, 0.0, 1.0))
    tflops = FLOPS_FWD / (ms / 1e3) / 1e12
    pct = tflops * 1e12 / B200_PEAK * 100
    print(f"  {label:<48s}  {ms:7.3f} ms   {tflops:7.1f} TFLOPS   {pct:5.1f}% peak")
    return ms, tflops


def main():
    print(f"device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"shape  : M={M}  K={K}  N={N}   2N={2*N}")
    print(f"FLOPs  : {FLOPS_FWD/1e12:.3f} T  (one [M,K]·[K,2N] GEMM)")
    print()

    x, W_normal, W_packed = make_inputs()
    if not correctness(x, W_normal, W_packed):
        print("\nCorrectness FAILED — bailing out before timing.")
        sys.exit(1)

    print()
    print("=== forward kernel timings ===")
    print(f"  {'variant':<48s}  {'time':>7}        {'TFLOPS':>7}   {'% peak':>6}")
    print(f"  {'-'*48}  {'-'*7}        {'-'*7}   {'-'*6}")

    # Ceilings & references.
    bench(lambda: x @ W_packed,
          "cuBLAS GEMM only (no activation, no save)")
    bench(lambda: fs.fused_swiglu_wide_packed(x, W_packed),
          "Triton T1 (fused: GEMM + activation, no save)")

    # The kernel we care about — the one save_factors_inplace uses.
    bench(lambda: fs.fused_swiglu_wide_packed_save_factors(x, W_packed),
          "Triton save_factors (GEMM + activation + factor save) ← target")

    # Our b42-based CUDA kernel doing the same work.
    if fs.HAS_CUDA_KERNEL:
        # Warm + JIT-compile if not yet cached.
        fs.cuda_matmul_save_factors(x, W_packed)
        torch.cuda.synchronize()
        bench(lambda: fs.cuda_matmul_save_factors(x, W_packed),
              "CUDA save_factors (b42-based, our version)")

    # cuBLAS + compile, no factor save (cheapest baseline that doesn't save).
    @torch.compile
    def compiled_act(preact):
        return swiglu_eager_chunked(preact)
    def cublas_plus_compiled():
        return compiled_act(x @ W_packed)
    bench(cublas_plus_compiled,
          "cuBLAS + torch.compile activation (no save)")


if __name__ == "__main__":
    main()
