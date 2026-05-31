"""Sweep over the CUDA kernel's compiled launcher configs."""
import math
import torch
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

# Match the (BN, BK, NS) combinations the .cu file emits launchers for.
CONFIGS = []
for (bn, bk, ns) in [(256, 64, 7), (256, 64, 6), (256, 64, 5), (256, 64, 4), (256, 128, 3)]:
    for gsm in (1, 4, 8, 16, 32):
        CONFIGS.append((bn, bk, ns, gsm))

print(f"shape: M={M}  K={K}  N={N}")
print(f"sweep: {len(CONFIGS)} CUDA configs (rep=500ms / warmup=50ms / median)")
print()

results = []
for i, cfg in enumerate(CONFIGS, 1):
    bn, bk, ns, gsm = cfg
    try:
        fs.cuda_matmul_save_factors(x, W_packed, config=cfg)
        torch.cuda.synchronize()
        ms, _, _ = tt.do_bench(
            lambda: fs.cuda_matmul_save_factors(x, W_packed, config=cfg),
            warmup=50, rep=500, quantiles=(0.5, 0.0, 1.0))
        tflops = FLOPS_FWD / (ms / 1e3) / 1e12
        pct = tflops * 1e12 / B200_PEAK * 100
        print(f"  [{i:2d}/{len(CONFIGS)}] BN={bn} BK={bk:3d} NS={ns} GSM={gsm:2d}  "
              f"{ms:7.3f} ms   {tflops:7.1f} TFLOPS   {pct:5.1f}% peak",
              flush=True)
        results.append((ms, tflops, pct, cfg))
    except Exception as e:
        print(f"  [{i:2d}/{len(CONFIGS)}] BN={bn} BK={bk:3d} NS={ns} GSM={gsm:2d}  "
              f"FAILED: {type(e).__name__}", flush=True)

results.sort(key=lambda r: r[0])
print()
print("=== top 5 ===")
for ms, tflops, pct, cfg in results[:5]:
    print(f"  BN={cfg[0]} BK={cfg[1]:3d} NS={cfg[2]} GSM={cfg[3]:2d}  "
          f"{ms:.3f} ms   {tflops:.1f} TFLOPS   {pct:.1f}% peak")

print()
print("vs Triton save_factors @ 1.798 ms / 1273 TFLOPS / 56.6% peak (from latest bench.py):")
best_ms = results[0][0]
print(f"  best CUDA: {best_ms:.3f} ms — "
      f"{'beats Triton by {:.0f} µs'.format((1.798 - best_ms)*1000) if best_ms < 1.798 else 'still {:.0f} µs slower than Triton'.format((best_ms - 1.798)*1000)}")
