"""Pattern check: does the mismatch fall on a column-stride or row-stride pattern?"""
import math
import torch
import fused_swiglu as fs


M, K, N = 11136, 3584, 14336
DTYPE = torch.bfloat16
torch.manual_seed(0)

x        = torch.randn(M, K,     device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_normal = torch.randn(K, 2 * N, device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_packed = fs.pack_swiglu_weight_chunked_torch(W_normal)

out_t, fac_t = fs.fused_swiglu_wide_packed_save_factors(x, W_packed)
out_c, fac_c = fs.cuda_matmul_save_factors(x, W_packed, config=(256, 64, 4, 8), persistent=False)

diff = (out_c.float() - out_t.float()).abs()

# Count mismatches per column in row 0.
print("OUT row 0 — match status per 32-col block:")
print("  col range          : matches / 32")
for c0 in range(0, 256, 32):
    bad = (diff[0, c0:c0+32] > 1e-3).sum().item()
    print(f"  [{c0:3d}, {c0+32:3d})         : {32 - bad:2d} / 32  ({'OK' if bad == 0 else f'{bad} bad'})")

# Same for column 0 across rows.
print("\nOUT col 0 — match status per 32-row block:")
for r0 in range(0, 256, 32):
    bad = (diff[r0:r0+32, 0] > 1e-3).sum().item()
    print(f"  [{r0:3d}, {r0+32:3d})         : {32 - bad:2d} / 32  ({'OK' if bad == 0 else f'{bad} bad'})")

# Look at the specific structure: is there a Z-shaped or interleaved pattern?
print("\nOUT row 0 — first 64 cols, diff values:")
for c in range(0, 64, 8):
    vals = [f"{diff[0, c+i].item():.3f}" for i in range(8)]
    print(f"  cols [{c:2d}-{c+7:2d}]: {vals}")