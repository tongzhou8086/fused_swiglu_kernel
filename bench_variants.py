"""A/B bench of CUDA save_factors variants vs Triton baseline.

Includes:
  - Triton baseline (target)
  - cuda_matmul_save_factors        (x32 TMEM loads, int4 stores)        ← current best
  - cuda_matmul_save_factors_x64    (x64 TMEM loads, int4 stores)        ← new variant

Each CUDA variant is correctness-checked vs Triton, then timed at its
configured (BN, BK, NS, GSM) settings.  Uses median timing via
triton.testing.do_bench with a long rep window.
"""
import math
import sys
import torch
import torch.nn.functional as F
import triton.testing as tt

import fused_swiglu as fs


M, K, N = 11136, 3584, 14336
DTYPE = torch.bfloat16
torch.manual_seed(0)

FLOPS_FWD = 2 * M * K * (2 * N)
B200_PEAK = 2250e12

x        = torch.randn(M, K,     device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_normal = torch.randn(K, 2 * N, device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_packed = fs.pack_swiglu_weight_chunked_torch(W_normal)


def check(name, fn):
    out_c, fac_c = fn()
    out_t, fac_t = fs.fused_swiglu_wide_packed_save_factors(x, W_packed)
    err_out = (out_c.float() - out_t.float()).abs().max().item()
    err_fac = (fac_c.float() - fac_t.float()).abs().max().item()
    ok = err_out == 0 and err_fac == 0
    tag = "OK (bit-identical)" if ok else f"MISMATCH err_out={err_out:.3e} err_fac={err_fac:.3e}"
    print(f"  validate {name:<40s} → {tag}")
    return ok


def bench(name, fn):
    fn(); torch.cuda.synchronize()
    ms, _, _ = tt.do_bench(fn, warmup=50, rep=500, quantiles=(0.5, 0.0, 1.0))
    tflops = FLOPS_FWD / (ms / 1e3) / 1e12
    pct = tflops * 1e12 / B200_PEAK * 100
    print(f"  {name:<48s}  {ms:7.3f} ms   {tflops:7.1f} TFLOPS   {pct:5.1f}% peak")
    return ms


print(f"device : {torch.cuda.get_device_name(0)}")
print(f"shape  : M={M}  K={K}  N={N}")
print()
print("=== correctness ===")

# Baseline (x32)
ok_base = check("CUDA (x32 ld, persistent NS=4 GSM=16)",
                lambda: fs.cuda_matmul_save_factors(x, W_packed, (256, 64, 4, 16), persistent=True))
# x64 variant
ok_x64 = check("CUDA (x64 ld, persistent NS=4 GSM=16)",
               lambda: fs.cuda_matmul_save_factors_x64(x, W_packed, (256, 64, 4, 16), persistent=True))

if not (ok_base and ok_x64):
    print("\nCorrectness FAILED — bailing.")
    sys.exit(1)

print()
print("=== timings ===")
print(f"  {'variant':<48s}  {'time':>7}        {'TFLOPS':>7}   {'% peak':>6}")
print(f"  {'-'*48}  {'-'*7}        {'-'*7}   {'-'*6}")

# Triton target
bench("Triton baseline (target)",
      lambda: fs.fused_swiglu_wide_packed_save_factors(x, W_packed))

# Baseline (x32, persistent, best config NS=7 GSM=16 found earlier)
bench("CUDA x32  PERS NS=7 GSM=16  (prior best)",
      lambda: fs.cuda_matmul_save_factors(x, W_packed, (256, 64, 7, 16), persistent=True))
bench("CUDA x32  PERS NS=4 GSM=16",
      lambda: fs.cuda_matmul_save_factors(x, W_packed, (256, 64, 4, 16), persistent=True))

# x64 variant
bench("CUDA x64  PERS NS=7 GSM=16",
      lambda: fs.cuda_matmul_save_factors_x64(x, W_packed, (256, 64, 7, 16), persistent=True))
bench("CUDA x64  PERS NS=4 GSM=16",
      lambda: fs.cuda_matmul_save_factors_x64(x, W_packed, (256, 64, 4, 16), persistent=True))
