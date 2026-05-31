"""Quick smoke test: compile + run the CUDA kernel once, validate correctness."""
import math
import sys

import torch
import fused_swiglu as fs


M, K, N = 11136, 3584, 14336
DTYPE = torch.bfloat16
torch.manual_seed(0)

x        = torch.randn(M, K,     device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_normal = torch.randn(K, 2 * N, device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_packed = fs.pack_swiglu_weight_chunked_torch(W_normal)

print(f"compiling and launching CUDA kernel ...")
out_c, fac_c = fs.cuda_matmul_save_factors(x, W_packed)
print(f"  out shape:     {tuple(out_c.shape)}  dtype={out_c.dtype}")
print(f"  factors shape: {tuple(fac_c.shape)}  dtype={fac_c.dtype}")

# Compare to Triton's save_factors output.
out_t, fac_t = fs.fused_swiglu_wide_packed_save_factors(x, W_packed)

err_out = (out_c.float() - out_t.float()).abs().max().item()
err_fac = (fac_c.float() - fac_t.float()).abs().max().item()
print(f"  out err vs Triton    : max_abs = {err_out:.3e}")
print(f"  factors err vs Triton: max_abs = {err_fac:.3e}")

atol = max(1.0, K ** 0.5 / 16)
ok = err_out <= atol and err_fac <= atol
print(f"  atol = {atol:.2f}  →  {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
