"""Annotate save_factors.ptx with a high-level map of the warp-specialized
pipeline. Pure text munging — we never reorder PTX, we only insert PTX
line comments (`//`) so the file remains valid PTX.

Sections added:
  - file-level header explaining the 3 worker loops
  - banners above $L__BB0_7  (MMA group K-loop)
                 $L__BB0_13 (TMA-LOAD warp K-loop)
                 $L__BB0_20 (epilogue / compute warps)
  - inline comments at the TMA stores, TMA loads, MMA issues, TMEM ops,
    epilogue waits, and the persistent stride.
"""
from __future__ import annotations

import os
import re

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SRC = os.path.join(ART, "save_factors.ptx")
DST = os.path.join(ART, "save_factors.annotated.ptx")

HEADER = """\
// ============================================================================
//  ANNOTATED PTX — _fused_swiglu_wide_packed_save_factors_kernel
//  (Triton-generated, NVIDIA Blackwell B200, BF16, warp-specialized)
//
//  Config (from compile site):
//    BLOCK_M=128  BLOCK_N_HALF=128  BLOCK_K=64  GROUP_M=32
//    num_warps=8  num_stages=4  WARP_SPECIALIZE=True  FLATTEN=True
//    persistent grid: stride = NUM_SMS = 148
//
// ----------------------------------------------------------------------------
//  WARP PARTITION  (the answer to "which warp does what")
// ----------------------------------------------------------------------------
//
//  The CTA launches with MORE than num_warps=8 warps. The entry block at
//  lines 34-39 splits them by warp_id into two physical pools:
//
//      warp_id = %tid.x >> 5
//      if (warp_id < 8) goto BB0_18    // DEFAULT pool  — heavy regs
//      else             goto BB0_1     // SPECIALIZED   — lightweight regs
//
//  ┌────────────────────────────────────────────────────────────────────────┐
//  │ POOL A — DEFAULT, warps 0..7   (256 threads, setmaxnreg.inc 240)       │
//  │   path: BB0_18 → BB0_20 (the persistent tile loop, Depth=1)            │
//  │                                                                        │
//  │   ROLE  =  EPILOGUE  +  ALL THREE TMA STORES                           │
//  │     - tcgen05.ld.32x32b.x64  (drain TMEM accumulator → registers)      │
//  │     - SwiGLU math (mul / ex2.approx / fma)                             │
//  │     - st.shared.v4.b32       (stage results into SMEM)                 │
//  │     - cp.async.bulk.tensor.2d.global.shared    ← TMA STORE #1, #2, #3  │
//  │                                                                        │
//  │   private sync: bar.sync 0, 256   (only these 8 warps)                 │
//  └────────────────────────────────────────────────────────────────────────┘
//
//  ┌────────────────────────────────────────────────────────────────────────┐
//  │ POOL B — SPECIALIZED, warps 8+  (setmaxnreg.dec 24, register-light)    │
//  │   path: BB0_1 → BB0_2 → brx.idx on a per-warp role tag in SMEM         │
//  │                                                                        │
//  │   Each specialized warp reads byte [smem + warp_id] and brx.idx into:  │
//  │     role 0 → BB0_5  → BB0_7   ★ MMA WARP                               │
//  │     role 1 → BB0_11 → BB0_13  ★ TMA-LOAD WARP                          │
//  │     role 2 → BB0_17           (idle, no work)                          │
//  │     role 3 → BB0_28           (exit)                                   │
//  │                                                                        │
//  │   ★ ROLE 0  — MMA WARP  (issues all tcgen05.mma):                      │
//  │       - mbarrier.try_wait.parity on "load done"                        │
//  │       - 4× tcgen05.mma.cta_group::1.kind::f16  (async into TMEM)       │
//  │       - tcgen05.commit.mbarrier (signal accumulator ready)             │
//  │       private sync: bar.warp.sync -1   (warp-local, no CTA stall)      │
//  │                                                                        │
//  │   ★ ROLE 1  — TMA-LOAD WARP  (issues all cp.async.bulk loads):         │
//  │       - mbarrier.try_wait.parity on "buffer empty"                     │
//  │       - mbarrier.arrive.expect_tx 49152  (= 128·64·2·3 bytes)          │
//  │       - 3× cp.async.bulk.tensor.2d.shared::cta.global                  │
//  │           load X tile, W_lo tile, W_hi tile (3 TMAs fire in parallel)  │
//  │       private sync: bar.sync 3, 64   (= 2 warps, count=64 threads)     │
//  └────────────────────────────────────────────────────────────────────────┘
//
// ----------------------------------------------------------------------------
//  HOW THEY OVERLAP
// ----------------------------------------------------------------------------
//
//                            time ─────────────────────────────────────►
//   TMA-LOAD warp  (BB0_13) │ load tile T+1 k=0 │ k=1 │ k=2 │ k=3 │ ...
//   MMA warp       (BB0_7)  │ (wait load)   mma│ mma │ mma │ mma │ ...
//   Compute warps  (BB0_20) │ tcgen05.ld T │ SwiGLU │ store#1│ wait │ store#2│ wait │ store#3
//
//  The three pools run on PHYSICALLY DIFFERENT warps and never share a
//  CTA-wide barrier. The epilogue's bar.sync 0,256 only pulls in the 8
//  compute warps; the load warp lives on bar 3; the MMA warp uses only its
//  own warp-local bar.warp.sync. Communication is purely via mbarriers in
//  shared memory.
//
//  The 3 TMA stores are serial relative to EACH OTHER (they reuse a single
//  SMEM staging buffer, gated by cp.async.bulk.wait_group.read 0), but the
//  load+MMA pipeline for tile T+1 runs concurrently with that entire
//  store-wait-store-wait-store chain. The `wait_group.read 0` only waits
//  for SMEM-read drain, NOT for HBM commit, so the next store can fire as
//  soon as the staging buffer is reusable (~hundreds of cycles).
//
//  Net result: save_factors adds 2 extra TMA stores per tile but hides
//  them behind the next tile's load+MMA — costs ~nothing on the critical
//  path vs the no-save variant.
// ============================================================================

"""

# (regex matched on the line, comment to inject ABOVE that line)
ABOVE = [
    # Persistent tile stride
    (re.compile(r'^\s*add\.s32\s+\S+,\s+\S+,\s+148\s*;'),
     "// --- persistent loop: advance tile_id by NUM_SMS=148 -----------------"),

    # TMEM accumulator pull (epilogue reads tile T's accumulator out of TMEM)
    (re.compile(r'^\s*tcgen05\.ld\.sync\.aligned\.32x32b\.x64\.b32'),
     "// --- tcgen05.ld: pull accumulator from TMEM into registers (epilogue start)"),
    (re.compile(r'^\s*tcgen05\.wait::ld\.sync\.aligned'),
     "// --- wait for TMEM load to be visible in registers"),

    # mbarrier.arrive that releases TMEM for the next MMA to overwrite
    (re.compile(r'^\s*@%p25\s+mbarrier\.arrive\.shared::cta\.b64\s+_,\s+\[%r306\]'),
     "// --- release TMEM buffer: tile T+1's MMA may now overwrite it"),
]

# (line number from your inspection, comment to inject ABOVE that line)
ABOVE_LINE = {
    34: "// === ENTRY: split CTA into two warp pools by warp_id ===============",
    37: "// %p1 = (warp_id < 8)  → split point between DEFAULT and SPECIALIZED",
    38: "// warps 0..7  go to BB0_18 (DEFAULT pool: epilogue + TMA stores)",
    39: "// warps 8..N go to BB0_1  (SPECIALIZED pool: MMA warp + TMA-load warp)",
    47: "// DEFAULT pool: bump per-thread register count to 240 (heavy compute)",
    1667: "// --- EPILOGUE store #1: stage factors[:,0:128] to SMEM, fence, TMA-store",
    1672: "// TMA STORE #1: factors lo half  (S2G, 128x128 bf16, %rd30 = factors descriptor)",
    1679: "// wait for SMEM staging buffer to drain (read-side) so we can reuse it for store #2",
    1705: "// --- EPILOGUE store #2: stage factors[:,128:256] to SMEM, fence, TMA-store",
    1709: "// n-coord OR'd with 128 -> second half of the factors tile",
    1711: "// TMA STORE #2: factors hi half  (same %rd30 descriptor, n+128)",
    1780: "// wait again — same SMEM staging buffer reused for the out tile",
    1838: "// --- EPILOGUE store #3: stage out[:,0:128] to SMEM, fence, TMA-store",
    1844: "// TMA STORE #3: out tile        (%rd31 = out descriptor; the 'main' store)",
    1846: "// commit store #3 and FALL THROUGH — no wait. The compute warps return to the",
    1847: "// loop; meanwhile tile T+1's MMAs (BB0_7) and TMA loads (BB0_13) have already",
}

# Section banners above labels
BANNERS = {
    '$L__BB0_18:': """\
// ============================================================================
//  $L__BB0_18 — DEFAULT POOL ENTRY  (warps 0..7, 256 threads, regs=240)
//
//  This is the entry point for the 8 "heavy" compute warps. They do:
//    1. Allocate TMEM, build 2 tensormap descriptors (for `out` and `factors`)
//       in SMEM and fence them out — lines 40..476.
//    2. Fall through to BB0_20 below (the persistent tile loop).
//
//  These 8 warps own ALL the TMA STORES and the TMEM accumulator read.
//  They do NOT issue MMA and do NOT issue TMA loads — those live on the
//  specialized warps in POOL B.
// ============================================================================
""",
    '$L__BB0_1:': """\
// ============================================================================
//  $L__BB0_1  — SPECIALIZED POOL ENTRY  (warps 8+, regs=24, lightweight)
//
//  All "extra" WS warps land here. They share one entry but split into
//  individual roles via a per-warp tag in SMEM:
//      %r20 = global_smem + warp_id
//      ld.shared.b8 %r18, [%r20+229464]     // each warp reads its OWN byte
//      brx.idx %r18, [BB0_5, BB0_11, BB0_17, BB0_28]
//  so different specialized warps end up in different roles:
//      role 0 → BB0_5 → BB0_7   = MMA WARP   (issues tcgen05.mma)
//      role 1 → BB0_11 → BB0_13 = TMA-LOAD WARP (issues cp.async.bulk loads)
//      role 2 → BB0_17           = idle
//      role 3 → BB0_28           = exit
// ============================================================================
""",
    '$L__BB0_7:':  """\
// ============================================================================
//  $L__BB0_7  — ★ MMA WARP, K-loop body  (role 0; tcgen05.mma issuer)
//
//  Owner: one specialized warp from POOL B (warp_id ≥ 8) with role tag 0.
//  Does NOT touch HBM; only TMEM. Async-issues mma and moves on.
//
//  Loop: 56 iters/tile (K=3584 / BK=64).
//    1. mbarrier.try_wait.parity on LOAD-DONE barrier — wait for TMA-load
//       warp (BB0_13) to fill this stage's SMEM (X + W_lo + W_hi).
//    2. Issue 4× tcgen05.mma.cta_group::1.kind::f16 — async, fire-and-forget
//       into TMEM. No stall on epilogue/HBM stores.
//    3. Last K-iter (k==55): tcgen05.commit.mbarrier → signals POOL A's
//       epilogue that the accumulator is ready to drain.
//
//  Private sync: bar.warp.sync -1 (warp-local). Never blocks other pools.
// ============================================================================
""",
    '$L__BB0_13:': """\
// ============================================================================
//  $L__BB0_13 — ★ TMA-LOAD WARP, K-loop body  (role 1; cp.async.bulk issuer)
//
//  Owner: ~2 specialized warps from POOL B with role tag 1.
//  (bar.sync 3, 64 ⇒ count=64 threads ⇒ 2 warps participate.)
//
//  Loop: 56 iters/tile. Per iter:
//    1. mbarrier.try_wait.parity on BUFFER-EMPTY barrier — wait for the
//       MMA warp to be done reading this stage's SMEM.
//    2. mbarrier.arrive.expect_tx 49152 — declare expected byte count:
//         49152 = 128 (BLOCK_M) × 64 (BLOCK_K) × 2 (bf16) × 3 tensors
//                = X tile + W_lo tile + W_hi tile.
//    3. Three back-to-back cp.async.bulk.tensor.2d.shared::cta.global —
//       X, W_lo, W_hi. All three issue in parallel; the mbarrier collects
//       completions when the byte count matches.
//
//  Private sync: bar.sync 3, 64 — a separate named barrier so the epilogue
//  pool's bar.sync 0, 256 does NOT pull this warp in. THAT is what lets
//  loads for tile T+1 stream while POOL A's stores for tile T serialize.
// ============================================================================
""",
    '$L__BB0_20:': """\
// ============================================================================
//  $L__BB0_20 — POOL A's persistent tile loop  (warps 0..7, the 8 compute
//               warps; Depth=1 outer loop, one iter per output tile).
//
//  Per output tile, these 8 warps do:
//    1. tcgen05.ld.32x32b.x64 — drain TMEM accumulator into registers.
//       This is what's waiting on the MMA warp's commit-mbarrier above.
//    2. mbarrier.arrive — release TMEM so MMA warp can overwrite for T+1.
//    3. SwiGLU math on registers (mul / ex2.approx / fma).
//    4. Three rounds of: st.shared (stage) → fence → bar.sync 0,256 →
//       cp.async.bulk.tensor.2d.global.shared (TMA STORE) → commit →
//       wait_group.read 0.
//          store #1: factors[:, 0:128]
//          store #2: factors[:, 128:256]
//          store #3: out[:, 0:128]
//       Serial because all three reuse the same SMEM staging buffer.
//
//  bar.sync 0, 256 only syncs these 8 warps; the MMA warp (BB0_7) and
//  TMA-load warp (BB0_13) advance freely on tile T+1 during the entire
//  store chain below.
// ============================================================================
""",
}

def annotate():
    with open(SRC) as f:
        lines = f.readlines()

    out = [HEADER]
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Label-based banners
        for label, banner in BANNERS.items():
            if stripped.startswith(label):
                out.append(banner)
                break

        # Line-number-based inline notes
        if i in ABOVE_LINE:
            out.append("\t" + ABOVE_LINE[i] + "\n")

        # Regex-based inline notes
        for pat, note in ABOVE:
            if pat.search(line):
                out.append("\t" + note + "\n")
                break

        out.append(line)

    with open(DST, "w") as f:
        f.writelines(out)

    print(f"wrote {DST}  ({sum(1 for _ in open(DST))} lines, original {len(lines)})")


if __name__ == "__main__":
    annotate()
