"""Rigorous A/B bench: N independent do_bench runs per variant, randomized order.

Earlier ad-hoc benches showed ~50-80 µs of run-to-run variance for the
SAME kernel — bigger than the per-variant deltas I was measuring.  This
script:
  1. Re-runs each variant N times within the SAME process (so module
     compilation, allocator state, etc. are amortized).
  2. Randomizes the order of variants in each round so no variant
     systematically gets the cold cache.
  3. Reports min / median / max / σ across the N rounds.

The min is the most useful single number — it's the "best achievable
on warm hardware with the cache cooperative", which is what production
will see in steady state.
"""
from __future__ import annotations
import math
import random
import statistics
import sys
import torch
import triton.testing as tt
import fused_swiglu as fs


M, K, N = 11136, 3584, 14336
DTYPE = torch.bfloat16
torch.manual_seed(0)

FLOPS_FWD = 2 * M * K * (2 * N)
B200_PEAK = 2250e12

N_ROUNDS = 5   # 5 rounds × 4 variants = 20 do_bench calls

x        = torch.randn(M, K,     device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_normal = torch.randn(K, 2 * N, device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_packed = fs.pack_swiglu_weight_chunked_torch(W_normal)


def make_runner(name, fn):
    return (name, fn)


VARIANTS = [
    make_runner("Triton            (baseline)",
                lambda: fs.fused_swiglu_wide_packed_save_factors(x, W_packed)),
    make_runner("CUDA x32 PERS NS=7 GSM=16",
                lambda: fs.cuda_matmul_save_factors(x, W_packed, (256, 64, 7, 16), persistent=True)),
    make_runner("CUDA x32 PERS NS=4 GSM=16",
                lambda: fs.cuda_matmul_save_factors(x, W_packed, (256, 64, 4, 16), persistent=True)),
    make_runner("CUDA x64 PERS NS=7 GSM=16",
                lambda: fs.cuda_matmul_save_factors_x64(x, W_packed, (256, 64, 7, 16), persistent=True)),
    make_runner("CUDA x64 PERS NS=4 GSM=16",
                lambda: fs.cuda_matmul_save_factors_x64(x, W_packed, (256, 64, 4, 16), persistent=True)),
    make_runner("CUDA tanh PERS NS=7 GSM=16",
                lambda: fs.cuda_matmul_save_factors_tanh(x, W_packed, (256, 64, 7, 16), persistent=True)),
    make_runner("CUDA tanh PERS NS=4 GSM=16",
                lambda: fs.cuda_matmul_save_factors_tanh(x, W_packed, (256, 64, 4, 16), persistent=True)),
]


def time_once(fn) -> float:
    """Returns median ms via triton.do_bench."""
    fn(); torch.cuda.synchronize()
    ms, _, _ = tt.do_bench(fn, warmup=50, rep=500, quantiles=(0.5, 0.0, 1.0))
    return ms


# Compile / warm every variant once.
print("warming up all variants ...", flush=True)
for name, fn in VARIANTS:
    fn()
torch.cuda.synchronize()

# Run N rounds in randomized order.
results: dict[str, list[float]] = {name: [] for name, _ in VARIANTS}
random.seed(0)

for r in range(N_ROUNDS):
    order = list(VARIANTS)
    random.shuffle(order)
    print(f"\nround {r+1}/{N_ROUNDS} (order: {[n.split()[0] + ' ' + (n.split()[1] if len(n.split()) > 1 else '') for n, _ in order]})", flush=True)
    for name, fn in order:
        ms = time_once(fn)
        results[name].append(ms)
        print(f"  {name:<36s}  {ms:7.3f} ms", flush=True)

# Report.
print()
print("=" * 110)
print(f"  {'variant':<36s}  {'min':>7}  {'median':>7}  {'max':>7}  {'σ':>5}  {'TFLOPS @ min':>13}  {'% peak @ min':>12}")
print("-" * 110)
for name, _ in VARIANTS:
    times = results[name]
    mn   = min(times)
    med  = statistics.median(times)
    mx   = max(times)
    sd   = statistics.stdev(times) if len(times) > 1 else 0.0
    tflops = FLOPS_FWD / (mn / 1e3) / 1e12
    pct = tflops * 1e12 / B200_PEAK * 100
    print(f"  {name:<36s}  {mn:7.3f}  {med:7.3f}  {mx:7.3f}  {sd*1000:5.0f}  {tflops:13.1f}  {pct:12.1f}")

print()
# Pairwise comparisons against the baseline (Triton)
trit_mn = min(results["Triton            (baseline)"])
print("vs Triton (using each variant's best/min):")
for name, _ in VARIANTS:
    if "Triton" in name: continue
    mn = min(results[name])
    delta = (mn - trit_mn) * 1000  # µs
    pct_slow = (mn / trit_mn - 1) * 100
    print(f"  {name:<36s}  {mn:7.3f} ms  ({delta:+7.0f} µs / {pct_slow:+5.1f}%)")
