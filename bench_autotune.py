"""Wide Triton autotune sweep for the save_factors forward kernel.

Mutates the module-level constants in fused_swiglu.triton_baseline before
each invocation so Triton JIT compiles a fresh kernel per config.  Each
unique (BM, BNH, BK, GSM, NW, NS) combination is timed via
triton.testing.do_bench at the colleague's production shape.

Reports:
  - per-config TFLOPS
  - top-5 winners
  - whether anything materially beats the colleague's hardcoded
    (128, 128, 64, 32, 8, 4) baseline.
"""
from __future__ import annotations

import math
import sys
import traceback

import torch
import torch.nn.functional as F
import triton.testing as tt

# Mutating module globals before each call.
import fused_swiglu.triton_baseline as fbk
import fused_swiglu as fs

# ── Shape ────────────────────────────────────────────────────────────────
M, K, N = 11136, 3584, 14336
DTYPE = torch.bfloat16
torch.manual_seed(0)

FLOPS_FWD = 2 * M * K * (2 * N)
B200_PEAK = 2250e12


# ── Inputs (built once) ──────────────────────────────────────────────────
x        = torch.randn(M, K,     device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_normal = torch.randn(K, 2 * N, device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_packed = fs.pack_swiglu_weight_chunked_torch(W_normal)


# ── Sweep grid ───────────────────────────────────────────────────────────
# Tuning notes:
# - BM divides M (=11136 = 2^6 · 3 · 29).  64 and 128 fit; 256 does not.
# - BNH divides N (=14336 = 2^11 · 7).  All powers of 2 up to 256 fit.
# - BK  divides K (=3584 = 2^9 · 7).  32, 64, 128, 256 all fit.
# - GSM is the Triton-style cluster-tile group-size; bigger → more A reuse
#   in L2, smaller → flatter walk.  Chapter-12 lesson: prune GSM > grid_m.
# - NS = num_stages drives the software pipeline depth.  Bigger costs SMEM.
# - NW = num_warps; 4 vs 8 mainly affects epilogue + warp-specialization.

# Focused grid: keep BM=128 (baseline already uses this; BM=64 is much smaller
# and unlikely to win at M=11136 where we have plenty of M-tiles); drop the
# extremes of BK (32 was too small in the early sweep; 256 likely SMEM-blows-up).
# This is ~144 configs, ~25 min sweep at ~5 cfg/min.
CONFIGS = []
for BM in (128,):
    for BNH in (64, 128, 256):
        for BK in (64, 128):
            for GSM in (8, 16, 32):
                for NW in (4, 8):
                    for NS in (3, 4, 5, 6):
                        CONFIGS.append(dict(BM=BM, BNH=BNH, BK=BK, GSM=GSM, NW=NW, NS=NS))

# Drop equivalent configs: GSM clamps to grid_m_tiles = M/BM (CTA-tile granularity).
def prune(cfgs):
    out = []
    seen = set()
    for c in cfgs:
        grid_m = M // c["BM"]
        gsm_eff = min(c["GSM"], grid_m)
        key = (c["BM"], c["BNH"], c["BK"], gsm_eff, c["NW"], c["NS"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


CONFIGS = prune(CONFIGS)
print(f"sweep: {len(CONFIGS)} unique configs", flush=True)
print(f"shape: M={M}  K={K}  N={N}   FLOPs/call = {FLOPS_FWD/1e12:.3f} T", flush=True)
print()


# ── Per-config bench ─────────────────────────────────────────────────────
def time_config(cfg) -> tuple[float, str]:
    """Returns (ms_median, error_msg).  ms is float('inf') if anything failed."""
    # Mutate the module globals.  Triton JIT keys on constexpr values, so
    # changing these here creates a new compile per unique combination.
    fbk.BLOCK_SIZE_M       = cfg["BM"]
    fbk.BLOCK_SIZE_N_HALF  = cfg["BNH"]
    fbk.BLOCK_SIZE_K       = cfg["BK"]
    fbk.GROUP_SIZE_M       = cfg["GSM"]
    fbk.NUM_WARPS          = cfg["NW"]
    fbk.NUM_STAGES         = cfg["NS"]
    fbk.SAVE_NUM_STAGES    = cfg["NS"]   # wrapper passes this for the save_factors path

    try:
        out_k, fac_k = fbk.fused_swiglu_wide_packed_save_factors(x, W_packed)
        torch.cuda.synchronize()
    except Exception as e:
        return float("inf"), f"compile/launch failed: {type(e).__name__}: {e}"

    try:
        def _fn():
            return fbk.fused_swiglu_wide_packed_save_factors(x, W_packed)
        # do_bench takes ms windows for warmup and rep.
        # Bumped rep=500ms (~270 iters per ~1.8 ms kernel) for tight CIs;
        # quantiles=(0.5, 0.0, 1.0) returns (median, min, max) — we take
        # the median below.  Triton internally averages within the rep
        # window and applies the requested quantile across multiple
        # restarts, so this gives a robust per-config number.
        ms, _, _ = tt.do_bench(_fn, warmup=50, rep=500, quantiles=(0.5, 0.0, 1.0))
        return ms, ""
    except Exception as e:
        return float("inf"), f"bench failed: {type(e).__name__}: {e}"


# ── Baseline (hardcoded) — restored when sweep ends ─────────────────────
BASELINE = dict(BM=128, BNH=128, BK=64, GSM=32, NW=8, NS=4)


# ── Main sweep ───────────────────────────────────────────────────────────
results = []   # list of (ms, tflops, pct, cfg, err)
n = len(CONFIGS)
for i, cfg in enumerate(CONFIGS, 1):
    ms, err = time_config(cfg)
    if ms == float("inf"):
        tflops = 0.0
        pct = 0.0
        print(f"  [{i:3d}/{n}] BM={cfg['BM']:3d} BNH={cfg['BNH']:3d} BK={cfg['BK']:3d} "
              f"GSM={cfg['GSM']:2d} NW={cfg['NW']} NS={cfg['NS']}  SKIP  ({err[:60]})",
              flush=True)
    else:
        tflops = FLOPS_FWD / (ms / 1e3) / 1e12
        pct = tflops * 1e12 / B200_PEAK * 100
        mark = ""
        if cfg["BM"]  == BASELINE["BM"]  and cfg["BNH"] == BASELINE["BNH"] and \
           cfg["BK"]  == BASELINE["BK"]  and cfg["GSM"] == BASELINE["GSM"] and \
           cfg["NW"]  == BASELINE["NW"]  and cfg["NS"]  == BASELINE["NS"]:
            mark = "  ← BASELINE"
        print(f"  [{i:3d}/{n}] BM={cfg['BM']:3d} BNH={cfg['BNH']:3d} BK={cfg['BK']:3d} "
              f"GSM={cfg['GSM']:2d} NW={cfg['NW']} NS={cfg['NS']}  "
              f"{ms:7.3f} ms   {tflops:7.1f} TFLOPS   {pct:5.1f}% peak{mark}",
              flush=True)
    results.append((ms, tflops, pct, cfg, err))


# ── Report ───────────────────────────────────────────────────────────────
results_ok = [r for r in results if r[0] != float("inf")]
results_ok.sort(key=lambda r: r[0])

print()
print(f"=== top 10 configs (lowest ms first) ===")
print()
for i, (ms, tflops, pct, cfg, _) in enumerate(results_ok[:10], 1):
    print(f"  #{i:2d}  BM={cfg['BM']:3d} BNH={cfg['BNH']:3d} BK={cfg['BK']:3d} "
          f"GSM={cfg['GSM']:2d} NW={cfg['NW']} NS={cfg['NS']}  "
          f"{ms:7.3f} ms   {tflops:7.1f} TFLOPS   {pct:5.1f}% peak")

baseline_result = next(
    (r for r in results_ok
     if all(r[3][k] == BASELINE[k] for k in BASELINE)),
    None
)
print()
if baseline_result:
    bms, btf, bpct, _, _ = baseline_result
    best_ms, best_tf, best_pct, best_cfg, _ = results_ok[0]
    print(f"baseline (BM=128 BNH=128 BK=64 GSM=32 NW=8 NS=4): "
          f"{bms:.3f} ms   {btf:.1f} TFLOPS   {bpct:.1f}% peak")
    print(f"best                                             : "
          f"{best_ms:.3f} ms   {best_tf:.1f} TFLOPS   {best_pct:.1f}% peak")
    delta = bms - best_ms
    if delta > 0.020:
        print(f"speedup vs baseline: {bms/best_ms:.3f}×  ({delta*1000:.0f} µs faster)")
    elif delta > 0:
        print(f"essentially tied with baseline (within {delta*1000:.0f} µs)")
    else:
        print(f"baseline is best (no improvement)")
