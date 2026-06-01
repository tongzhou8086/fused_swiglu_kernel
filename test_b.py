"""Quick correctness + smoke bench of the Path B kernel."""
import math
import sys

import torch
import triton.testing as tt
import fused_swiglu as fs


M, K, N = 11136, 3584, 14336
DTYPE = torch.bfloat16
torch.manual_seed(0)

x        = torch.randn(M, K,     device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_normal = torch.randn(K, 2 * N, device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_packed = fs.pack_swiglu_weight_chunked_torch(W_normal)

print("compiling and launching Path B kernel ...")
out_b, fac_b = fs.cuda_matmul_save_factors_b(x, W_packed, (256, 64, 4, 16))
out_t, fac_t = fs.fused_swiglu_wide_packed_save_factors(x, W_packed)

err_out = (out_b.float() - out_t.float()).abs().max().item()
err_fac = (fac_b.float() - fac_t.float()).abs().max().item()
atol = max(1.0, K ** 0.5 / 16)
ok = err_out <= atol and err_fac <= atol
print(f"out err vs Triton    : max_abs = {err_out:.3e}")
print(f"factors err vs Triton: max_abs = {err_fac:.3e}")
print(f"atol = {atol:.2f}  →  {'OK' if ok else 'FAIL'}")

if not ok:
    sys.exit(1)

# Quick timing
def b_fn():
    return fs.cuda_matmul_save_factors_b(x, W_packed, (256, 64, 4, 16))

# Warm up
for _ in range(5):
    b_fn()
torch.cuda.synchronize()

ms, _, _ = tt.do_bench(b_fn, warmup=200, rep=1500, quantiles=(0.5, 0.0, 1.0))
print(f"\nPath B: {ms:.3f} ms median")
