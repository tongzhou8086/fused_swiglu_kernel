# fused_swiglu_kernel

Optimization project for the forward fused SwiGLU kernel with factor side store
— the kernel that powers `save_factors_inplace` in the colleague's
[`swiglu_fused`](../swiglu_fused) repo.

## What this kernel does

Given inputs `A [M, K]` and `W_packed [K, 2N]` (chunk-interleaved layout),
the kernel computes in one launch:

```
[left | gate] = A @ W_packed                   # wide-acc bf16 → fp32 GEMM
sig           = sigmoid(gate)                   # in registers
silu          = gate * sig                      # in registers
silu_prime    = sig + silu * (1 - sig)          # in registers
out           = left * silu                     → [M, N]   main store
factors       = [silu | left * silu_prime]      → [M, 2N]  side store
```

`out` is the FFN-stage activation; `factors` is the precomputed
backward state that lets the backward pass skip an SFU recompute.

Both outputs are bf16.  The `factors` side store is what makes this
kernel a training-favorable fusion: in exchange for ~113 µs of extra
HBM write tax on forward, the backward pass becomes a cheap elementwise
`grad_de = grad_out * factors` instead of a SFU-bound recompute from
`preact`.

## Why we think there's headroom

From the upstream analysis (see [`swiglu_fused/README.md`](../swiglu_fused/README.md)):

- cuBLAS NN GEMM at this shape : **73 % of B200 BF16 peak**
- Triton fused (save_factors)  : **58 % of peak  (80 % of cuBLAS NN)**

The gap is a mix of:

1. **Activation absorption overhead** (~234 µs vs the GEMM ceiling).
   Should be ~0 with perfect persistent-grid pipelining where tile T's
   epilogue overlaps tile T+1's K-loop.
2. **Factor-side-store tax** (~113 µs).  Should also be ~0 in
   principle for the same reason.

If we can close even half of those, we'd pull save_factors from 1.75 ms
toward 1.50 ms at this shape — a 14 % speedup on a kernel that's
already in the production critical path.

## Where the optimization experience comes from

This project is set up to lean on adjacent work:

- [`~/projects/mmcomposer/tutorial`](../mmcomposer/tutorial) — the 12-chapter
  optimization ladder for B200 matmul.  Specifically chapters 8–12
  cover wide-acc, warp specialization, cluster MMA, hoisted descriptors,
  and the autotuning pattern.
- [`~/projects/mymatmul/mymatmul/gpu/blackwell/_matmul_b42_gsm.cu`](../mymatmul) —
  b42, our hand-tuned production matmul kernel at the equivalent shape.
  Runs at ~95 % of cuBLAS NN on B200 with `cta_group::2`, tcgen05 MMA,
  K-major B, and chunked-CTA-swizzle for L2 reuse.

The Triton baseline doesn't use `cta_group::2`-cluster MMA; b42 does.
That's one of the structural levers we may pull if we end up writing
a custom CUDA version.

## Shape

```
M = 11136   (token count — colleague's training shape)
K =  3584   (d_model)
N = 14336   (d_ff per gated branch; full W is [K, 2N] = [3584, 28672])
```

K is short relative to M and N — the "FFN-stage" regime where fusion
pays off most.

## Layout

```
fused_swiglu_kernel/
  fused_swiglu/
    triton_baseline.py             — colleague's Triton kernel (target)
    _matmul_save_factors.cu        — CUDA production kernel (b42-based, int4 stores, persistent)
    _matmul_save_factors_x64.cu    — variant: x64 TMEM loads (tested, NO gain)
    _matmul_save_factors_tanh.cu   — variant: tanh-form sigmoid (tested, regression)
    cuda_kernel.py                 — Python launcher for the production CUDA kernel
    cuda_kernel_x64.py             — launcher for x64 variant
    cuda_kernel_tanh.py            — launcher for tanh variant
  bench.py                         — initial smoke bench
  bench_stable.py                  — CANONICAL bench (long warmup + rep, rigorous)
  bench_rigorous.py                — multi-round randomized bench (kept for reference)
  bench_variants.py / bench_autotune.py / bench_cuda_sweep.py — older sweeps
  artifacts/                       — Triton PTX dumps for inspection
```

## Run

```bash
srun -p dedicated --gres=gpu:nvidia_b200:1 --time=00:10:00 \
    ~/miniconda3/bin/python bench_stable.py
```

## Final results (Path A complete)

Measured via `bench_stable.py` (5s global warmup, 3s rep per variant):

| variant | median | gap vs Triton |
|---|---|---|
| Triton baseline                | 1.804 ms | — |
| **CUDA x32 PERS NS=7 GSM=16**  | **1.917 ms** | **+113 µs (+6.3%)** ← production |
| CUDA x64 PERS NS=7 GSM=16      | 1.925 ms | +121 µs |
| CUDA tanh PERS NS=7 GSM=16     | 1.976 ms | +172 µs (regression) |

### What Path A actually bought us

The journey through the optimization ladder, with honest attribution:

| step | description | actual delta |
|---|---|---|
| First cut | b42 main loop + 4-warp save_factors epilogue | 2.003 ms (−204 µs vs cold start) |
| 8-warp Phase 1 | 4 row × 2 col warp split | 1.961 ms (~within noise) |
| Persistent grid | outer tile loop, mbar phases preserved across tiles | **1.917 ms (−44 µs, real win)** |
| x64 TMEM loads | one load each for left/gate vs two | **no signal** (~8 µs slower median) |
| tanh-form sigmoid | replace divide-form for sigmoid | **regression −60 µs** |

**Only persistent grid was a real, statistically significant win** — the rest fell within
run-to-run variance once measured rigorously (long warmup + 3s rep windows).

### What's left

The remaining 113 µs gap (6.3%) is the cost of our serial K-loop → epilogue → next K-loop
chain.  Triton's persistent + FLATTEN compiler scheduling overlaps tile T's epilogue with
tile T+1's K-loop; our hand-written CUDA does not.  Closing this requires a Path B
restructure:

- Double-buffered TMEM (so MMA can write tile T+1 while epilogue reads tile T)
- Per-tile MMA-done mbarrier (not the CTA-wide one we have)
- Split warp roles: warps 0 (TMA) and 1 (MMA) keep working across tiles independently;
  epilogue warps 2..7 consume tiles asynchronously with their own per-tile signals.

This is ~1-2 days of careful work and not yet attempted.

### Benchmarking lessons learned

- **GPU warmup matters a lot.** B200 takes ~1-2 seconds of mixed workload to fully
  boost clocks.  `do_bench`'s default `warmup=25ms` is far too short.
- **Run multiple variants and use long `rep`.**  `bench_stable.py`'s 5s global warmup +
  3s per-variant rep gives medians stable to within ~10 µs.
- **A single `do_bench` call right after compilation lies.**  Earlier ad-hoc
  comparisons were confounded by 30-80 µs of warmup-state variance; once that was
  controlled for, several "optimizations" evaporated.
