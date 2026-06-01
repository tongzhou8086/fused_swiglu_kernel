"""Diagnose TMA-store errors: where exactly do CUDA and Triton differ?"""
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

diff_out = (out_c.float() - out_t.float()).abs()
diff_fac = (fac_c.float() - fac_t.float()).abs()

print(f"OUT shape {tuple(out_c.shape)}  max_abs={diff_out.max():.3e}")
print(f"FAC shape {tuple(fac_c.shape)}  max_abs={diff_fac.max():.3e}")

# Where are the non-zero diffs?
mask_out = diff_out > 1e-3
mask_fac = diff_fac > 1e-3
print(f"\nrows with any OUT mismatch: {mask_out.any(dim=1).sum().item()} / {M}")
print(f"cols with any OUT mismatch: {mask_out.any(dim=0).sum().item()} / {N}")
print(f"rows with any FAC mismatch: {mask_fac.any(dim=1).sum().item()} / {M}")
print(f"cols with any FAC mismatch: {mask_fac.any(dim=0).sum().item()} / {2 * N}")

# Pattern: print the first few bad indices.
print("\nfirst 10 bad OUT cells (row, col, cuda, triton):")
bad_rows, bad_cols = torch.nonzero(mask_out, as_tuple=True)
for i in range(min(10, len(bad_rows))):
    r, c = bad_rows[i].item(), bad_cols[i].item()
    print(f"  ({r:5d}, {c:5d})  cuda={out_c[r, c].item():+.4f}  triton={out_t[r, c].item():+.4f}")

print("\nfirst 10 bad FAC cells (row, col, cuda, triton):")
bad_rows, bad_cols = torch.nonzero(mask_fac, as_tuple=True)
for i in range(min(10, len(bad_rows))):
    r, c = bad_rows[i].item(), bad_cols[i].item()
    print(f"  ({r:5d}, {c:5d})  cuda={fac_c[r, c].item():+.4f}  triton={fac_t[r, c].item():+.4f}")

# Check: are mismatches clustered by tile?
BM = 128
BN_HALF = 128
BLOCK_N = 256
print("\nOUT mismatch density by 128x128 tile (M=87 tiles, N_HALF=112 tiles):")
n_tile_m = M // BM         # 87 (some unused)
n_tile_n = N // BN_HALF    # 112
sums = mask_out.view(n_tile_m, BM, n_tile_n, BN_HALF).sum(dim=(1, 3))
# Show density
nz_tiles = (sums > 0).sum().item()
total_tiles = n_tile_m * n_tile_n
print(f"  tiles with any mismatch: {nz_tiles} / {total_tiles}")
print(f"  per-tile mismatch counts (first 10 nonzero):")
for tm_id, tn_id in zip(*torch.nonzero(sums, as_tuple=True)[:10]):
    pass
nz_idx = torch.nonzero(sums, as_tuple=False)
for i in range(min(10, nz_idx.shape[0])):
    tm, tn = nz_idx[i, 0].item(), nz_idx[i, 1].item()
    print(f"    tile (M={tm}, N={tn}): {sums[tm, tn].item()} bad elements")
