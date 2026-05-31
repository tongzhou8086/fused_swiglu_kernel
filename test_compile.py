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

print(f"compiling and launching CUDA kernel (non-persistent) ...")
# Use NS=4 since that's the only NS shared between non-persistent and persistent launchers.
cfg = (256, 64, 4, 8)
out_c, fac_c = fs.cuda_matmul_save_factors(x, W_packed, config=cfg)
print(f"  out shape:     {tuple(out_c.shape)}  dtype={out_c.dtype}")
print(f"  factors shape: {tuple(fac_c.shape)}  dtype={fac_c.dtype}")

# Compare to Triton's save_factors output.
out_t, fac_t = fs.fused_swiglu_wide_packed_save_factors(x, W_packed)

err_out = (out_c.float() - out_t.float()).abs().max().item()
err_fac = (fac_c.float() - fac_t.float()).abs().max().item()
print(f"  non-persistent out err vs Triton    : max_abs = {err_out:.3e}")
print(f"  non-persistent factors err vs Triton: max_abs = {err_fac:.3e}")
ok1 = err_out == 0 and err_fac == 0

print()
print(f"compiling and launching CUDA kernel (persistent) ...")
out_p, fac_p = fs.cuda_matmul_save_factors(x, W_packed, config=cfg, persistent=True)
err_out_p = (out_p.float() - out_t.float()).abs().max().item()
err_fac_p = (fac_p.float() - fac_t.float()).abs().max().item()
print(f"  persistent out err vs Triton    : max_abs = {err_out_p:.3e}")
print(f"  persistent factors err vs Triton: max_abs = {err_fac_p:.3e}")
ok2 = err_out_p == 0 and err_fac_p == 0

atol = max(1.0, K ** 0.5 / 16)
ok = ok1 and ok2
print()
print(f"atol = {atol:.2f}  →  {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
