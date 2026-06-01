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
    triton_baseline.py                 — colleague's Triton kernel (target)
    _matmul_save_factors.cu            — Path A: b42-based, int4 stores, persistent
    _matmul_save_factors_x64.cu        — variant: x64 TMEM loads (tested, no gain)
    _matmul_save_factors_tanh.cu       — variant: tanh-form sigmoid (regression)
    _matmul_save_factors_nostg.cu      — variant: no SMEM staging (regression)
    _matmul_save_factors_b.cu          — Path B: K-loop / epilogue OVERLAP
    _matmul_save_factors_b_nostg.cu    — Path B + no SMEM staging (regression)
    _matmul_save_factors_b_tmast.cu    — Path B + TMA stores ← CURRENT BEST
    cuda_kernel*.py                    — Python launchers (one per .cu)
  bench_stable.py                      — CANONICAL bench (long warmup + 3s rep)
  dump_triton_ptx.py                   — dumps + analyses Triton's PTX for comparison
  artifacts/                           — Triton PTX / TTGIR dumps
```

## Run

```bash
srun -p dedicated --gres=gpu:nvidia_b200:1 --time=00:10:00 \
    ~/miniconda3/bin/python bench_stable.py
```

## Final results

Measured via `bench_stable.py` (5s global warmup, 3s rep per variant) on B200:

| variant | median | TFLOPS | % peak | gap vs Triton |
|---|---|---|---|---|
| Triton baseline                       | 1.788 ms | 1280 | 56.9 % | — |
| **CUDA Path B + TMA stores NS=4 GSM=16** | **1.849 ms** | **1237** | **55.0 %** | **+61 µs (+3.4 %)** ← current best |
| CUDA Path B (int4 stores) NS=4 GSM=16 | 1.858 ms | 1232 | 54.8 % | +70 µs |
| CUDA Path A (x32) NS=7 GSM=16         | 1.909 ms | 1199 | 53.3 % | +121 µs |

We close ~55 % of the original CUDA-vs-Triton gap and end at **96.6 % of Triton's perf**, bit-identical (max_abs = 0.0).

### The journey — what actually mattered

| step | description | actual delta | verdict |
|---|---|---|---|
| Path A first cut       | b42 main loop + 4-warp save_factors epilogue       | 2.003 ms | — |
| 8-warp Phase 1         | 4 row × 2 col warp split for the epilogue          | 1.961 ms | within noise |
| Persistent grid        | outer tile loop, mbar phases preserved             | 1.917 ms | **−44 µs, real win** |
| x64 TMEM loads         | wider tcgen05.ld vs two narrow loads               | ~no change | no signal |
| tanh-form sigmoid      | one SFU op instead of two                          | +60 µs    | regression |
| Drop SMEM staging      | direct TMEM→regs→GMEM (uncoalesced stores)         | +400 µs   | regression |
| **Path B: K-loop / epilogue overlap** | double-buffered TMEM, per-tile mbars   | **−45 µs, real win** | the big win |
| Early `epi_done` arrive | move arrive between Phase 1 and Phase 2           | **−28 µs gap, real win** | sync audit |
| Path B + no SMEM staging | NS=7 ring, direct GMEM stores                    | +170 µs   | uncoalesced stores cost more than ring-depth gains |
| **Path B + TMA stores** (this commit) | async cp.async.bulk.tensor.2d stores      | **−9 µs, real win** | matches Triton's epilogue |
| Path B + TMA stores + SWIZZLE_128B | swizzled SMEM writes for TMA stores      | **dead end** | cuTensorMap rejects box_width > 64 bf16 |

The two real architectural wins — **persistent grid** and **K-loop/epilogue overlap** — together account for ~80 µs of the 140 µs original gap.  The TMA store swap inside the overlap framework picks up another ~10 µs.  Several plausible-looking changes (x64 loads, tanh, dropping SMEM staging) turned out to be noise or regressions once measured rigorously.

### Why we stopped — the remaining 61 µs

After PTX-dumping Triton (see `dump_triton_ptx.py` + `artifacts/save_factors.ptx`):

| feature | Triton | Path B + TMAst | match? |
|---|---|---|---|
| K-pipeline NS                        | 4 | 4 | ✓ |
| Persistent grid + GROUP_SIZE_M       | yes | yes | ✓ |
| Warp-specialized                     | yes | yes | ✓ |
| SMEM staging in epilogue             | yes (32 `st.shared`) | yes | ✓ |
| TMA stores per tile                  | 3 (`cp.async.bulk.tensor.2d`) | 2 (combined FAC) | ≈ |
| SWIZZLE for TMA stores               | `SWIZZLE_NONE` (forced by box width) | `SWIZZLE_NONE` | ✓ |
| **MMA grouping**                     | **`cta_group::1` (single-CTA)** | **`cta_group::2` (cluster)** | **✗** |

The only structural difference left is **single-CTA MMA vs cluster MMA**.  At this shape the cluster's A-multicast benefit apparently doesn't pay for the cluster sync / multicast-arrive overhead.  Switching to single-CTA would be a 200+ line restructure with uncertain payoff — we explicitly decided this is a dead end for this optimization round and stopped at 96.6 % of Triton.

### Dead ends explored (so we don't re-explore them)

1. **`SWIZZLE_128B` TMA stores** — cuTensorMap requires inner-box-dim ≤ 128 bytes (= 64 bf16).  Our 128-bf16-wide stores would need to split into 6 stores per tile (vs 2 today).  Triton itself uses `SWIZZLE_NONE` for the same reason — verified by box-width probing.
2. **No SMEM staging in epilogue** — frees ~100 KB SMEM (NS=4 → NS=7 ring fits) but the TMEM `32x32b_x32` layout puts one row per lane, so direct GMEM stores hit 32 different cachelines per warp.  Lost more than gained (+170 µs net).
3. **`cta_group::1` (single-CTA) MMA** — believed to be where Triton's remaining edge lives, but not pursued; significant rewrite, uncertain payoff at our shape.
4. **TMA stores on non-overlap kernel** (commit `9df0e84`, reverted in `40bd120`) — was 172 µs WORSE than int4 stores because `wait_group<0>` stalled the whole CTA between tiles with no other work to hide behind.  Only became a win once layered onto Path B's overlap.

### Benchmarking lessons learned

- **GPU warmup matters a lot.**  B200 takes ~1-2 seconds of mixed workload to fully boost clocks.  `do_bench`'s default `warmup=25ms` is far too short.
- **Run multiple variants and use long `rep`.**  `bench_stable.py`'s 5s global warmup + 3s per-variant rep gives medians stable to within ~10 µs.
- **A single `do_bench` call right after compilation lies.**  Earlier ad-hoc comparisons were confounded by 30-80 µs of warmup-state variance; once that was controlled for, several "optimizations" evaporated.
- **Compare GAPS within the same run**, not absolute medians across runs.  GPU thermal/clock state drifts; the gap to Triton in the same run is the right signal.
- **PTX-dump the baseline before speculating.**  We avoided two more dead ends (NS=7, SWIZZLE_128B speculation) once we saw Triton's actual settings.
