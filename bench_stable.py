"""Stable bench with LONG do_bench windows.

Earlier observation: do_bench medians were varying ~30 µs between calls
because the GPU takes a few rounds to fully warm/clock up.  This script:
  1. Long warmup-everything pass (each variant called many times in
     a global loop).
  2. ONE do_bench per variant with warmup=500ms, rep=3000ms.
  3. No randomization, no rounds — just trust the median over ~1500
     iterations within a 3-second window.
"""
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


VARIANTS = [
    ("Triton            (baseline)",
     lambda: fs.fused_swiglu_wide_packed_save_factors(x, W_packed)),
    ("CUDA x32 PERS NS=7 GSM=16",
     lambda: fs.cuda_matmul_save_factors(x, W_packed, (256, 64, 7, 16), persistent=True)),
    ("CUDA x32 PERS NS=4 GSM=16",
     lambda: fs.cuda_matmul_save_factors(x, W_packed, (256, 64, 4, 16), persistent=True)),
    ("CUDA x64 PERS NS=7 GSM=16",
     lambda: fs.cuda_matmul_save_factors_x64(x, W_packed, (256, 64, 7, 16), persistent=True)),
    ("CUDA x64 PERS NS=4 GSM=16",
     lambda: fs.cuda_matmul_save_factors_x64(x, W_packed, (256, 64, 4, 16), persistent=True)),
    ("CUDA tanh PERS NS=7 GSM=16",
     lambda: fs.cuda_matmul_save_factors_tanh(x, W_packed, (256, 64, 7, 16), persistent=True)),
    ("CUDA tanh PERS NS=4 GSM=16",
     lambda: fs.cuda_matmul_save_factors_tanh(x, W_packed, (256, 64, 4, 16), persistent=True)),
]

# Heavy global warmup: each variant runs in a tight loop for ~5 seconds total.
print(f"global warmup: 5s of mixed calls to boost clocks ...", flush=True)
import time
t0 = time.time()
i = 0
while time.time() - t0 < 5.0:
    _, fn = VARIANTS[i % len(VARIANTS)]
    fn()
    i += 1
torch.cuda.synchronize()
print(f"  ran {i} calls during warmup", flush=True)
print()

# Single long-window do_bench per variant.
print(f"=== timings (do_bench warmup=500ms rep=3000ms) ===")
print(f"  {'variant':<36s}  {'median':>7}  {'min':>7}  {'max':>7}  {'TFLOPS @ med':>12}  {'% peak @ med':>12}")
print("-" * 100)
for name, fn in VARIANTS:
    ms_med, ms_min, ms_max = tt.do_bench(
        fn, warmup=500, rep=3000, quantiles=(0.5, 0.0, 1.0))
    tflops = FLOPS_FWD / (ms_med / 1e3) / 1e12
    pct = tflops * 1e12 / B200_PEAK * 100
    print(f"  {name:<36s}  {ms_med:7.3f}  {ms_min:7.3f}  {ms_max:7.3f}  {tflops:12.1f}  {pct:12.1f}",
          flush=True)
