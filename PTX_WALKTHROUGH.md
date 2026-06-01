# How `_fused_swiglu_wide_packed_save_factors_kernel` runs on Blackwell

A walkthrough of the Triton-generated PTX for the fused SwiGLU forward
kernel that also saves the backward "factors". Goal of this doc: explain
how warp specialization splits the work so that three TMA stores per output
tile cost essentially nothing — they hide behind the next tile's compute.

The full annotated PTX is at `artifacts/save_factors.annotated.ptx`
(~2750 lines). Below we extract only the load-bearing instructions.

---

## Setup

- **Kernel**: `_fused_swiglu_wide_packed_save_factors_kernel`
  (`fused_swiglu/triton_baseline.py`)
- **Hardware**: NVIDIA Blackwell B200, BF16
- **Triton config**:
  `BLOCK_M=128, BLOCK_N_HALF=128, BLOCK_K=64, GROUP_M=32,
   num_warps=8, num_stages=4, WARP_SPECIALIZE=True, FLATTEN=True`
- **Persistent grid**: stride = `NUM_SMS = 148`
- **Per tile output**: one `out` tile (128×128 bf16) + one `factors` tile
  (128×256 bf16) for the backward fast-path

## The 30-second picture

The CTA launches with **12 warps**, not 8 — the `num_warps=8` you pass to
Triton is only the *compute-pool* size. With `WARP_SPECIALIZE=True`,
Triton's WS pass silently adds 4 extra lightweight warps for the
specialized roles. You can confirm this on the compiled object:

```python
compiled = jit_kern.warmup(...)
print(compiled.metadata.num_warps)   # → 12
```

The 12 warps are split by `warp_id` into two physical pools that run
truly in parallel:

```
                                     time ─────────────────────────────────────►
POOL B: TMA-LOAD warp   (role 1)   │ load T+1 k=0 │ k=1 │ k=2 │ k=3 │ ...
POOL B: MMA warp        (role 0)   │ (wait load) mma│ mma │ mma │ mma │ ...
POOL A: 8 compute warps (epilogue) │ tcgen05.ld T │ SwiGLU │ store#1│ wait │ store#2│ wait │ store#3
```

| Pool | Warps | Register budget | Job |
|---|---|---|---|
| **A — Default** | warps 0..7 (8 warps) | `setmaxnreg.inc 240` | TMEM accumulator drain, SwiGLU math, all 3 TMA **stores** |
| **B — Specialized, role 0** | warp 8 (1 warp) | `setmaxnreg.dec 24` | Issue `tcgen05.mma` |
| **B — Specialized, role 1** | warps 9..10 (2 warps) | `setmaxnreg.dec 24` | Issue 3 TMA **loads** per K-iter (X, W_lo, W_hi) |
| **B — Specialized, role 2** | warp 11 (1 warp) | `setmaxnreg.dec 24` | Idle (WS pad — typical Triton convention adds extras in groups of 4) |

(`bar.sync 0, 256` later in the epilogue uses thread count = 8 warps × 32 = 256, and `bar.sync 3, 64` uses 2 warps × 32 = 64 — that's how we know the participating warp count of each barrier.)

They communicate exclusively via shared-memory **mbarriers** — never via
a CTA-wide barrier. The result: the three TMA stores in the epilogue
serialize through one SMEM staging buffer, but the load+MMA pipeline for
tile T+1 runs concurrently with the entire store chain.

---

## 1. Entry: split the CTA into two warp pools

```ptx
// %tid.x → warp_id → pool
mov.u32           %r1, %tid.x;
shr.u32           %r2, %r1, 5;                // warp_id = tid >> 5
shfl.sync.idx.b32 %r3, %r2, 0, 31, -1;        // broadcast lane-0 warp_id
setp.lt.u32       %p1, %r3, 8;                // %p1 = (warp_id < 8)
@%p1 bra          $L__BB0_18;                 // warps 0..7 → DEFAULT (Pool A)
bra.uni           $L__BB0_1;                  // warps 8..N → SPECIALIZED (Pool B)
```

Pool A goes to **`BB0_18`** (epilogue + stores). Pool B goes to **`BB0_1`**,
which further partitions each warp into a role by reading a per-warp tag
from SMEM:

```ptx
$L__BB0_2:
    setmaxnreg.dec.sync.aligned.u32 24;       // specialized warps: lightweight
    ld.shared.b8    %r18, [%r20+229464];      // each warp reads ITS OWN byte
                                              // (%r20 = smem + warp_id)
    brx.idx %r18, $L_brx_0;                   // role dispatch:
        // role 0 → BB0_5  → BB0_7   (MMA warp)
        // role 1 → BB0_11 → BB0_13  (TMA-LOAD warp)
        // role 2 → BB0_17           (idle)
        // role 3 → BB0_28           (exit)
```

That's how a single PTX entry-point implements 3+ different worker roles
without runtime overhead — each warp's role is decided once, by reading
a different SMEM byte.

---

## 2. Pool B / role 1 — the TMA-LOAD warp (`BB0_13`)

Two specialized warps live here. Their entire job per K-iter is: wait
until the previous stage's SMEM buffer has been drained by the MMA warp,
then fire three parallel TMA loads into the next stage.

```ptx
$L__BB0_13:                       // K-loop body for the load warp
    // 1. Wait for "buffer empty" from the MMA warp.
    mbarrier.try_wait.parity.shared::cta.b64  complete, [%r43], %r979;
    @!complete bra.uni waitLoop;

    // 2. Declare expected bytes for the producer-consumer mbarrier:
    //    49152 = 128 (BLOCK_M) * 64 (BLOCK_K) * 2 (bf16) * 3 tensors
    //          = X tile + W_lo tile + W_hi tile
    bar.sync 3, 64;                           // sync this pool only (2 warps)
    @%p6 mbarrier.arrive.expect_tx.shared::cta.b64  _, [%r44], 49152;
    bar.sync 3, 64;

    // 3. Three back-to-back TMA loads, all targeting the same mbarrier.
    //    They fire in parallel; the mbarrier flips when 49152 bytes have
    //    landed in SMEM.
    @%p7 cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes
            [%r45], [%rd9,  {%r46, %r983}], [%r44];          // X tile
    @%p8 cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes
            [%r47], [%rd10, {%r48, %r46}],  [%r44];          // W_lo tile
    @%p8 cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes
            [%r49], [%rd10, {%r50, %r46}],  [%r44];          // W_hi tile
```

Note `bar.sync 3, 64` — **named barrier 3, count 64**. Only this pool's 2
warps × 32 threads participate. Pool A's `bar.sync 0, 256` cannot stall
this warp.

---

## 3. Pool B / role 0 — the MMA warp (`BB0_7`)

One specialized warp lives here. Per K-iter it waits on the load
mbarrier, fires 4 async MMAs into TMEM, and moves on. It never touches
HBM and never blocks on the epilogue.

```ptx
$L__BB0_7:                        // K-loop body for the MMA warp
    bar.warp.sync -1;             // warp-local sync; never blocks others

    // 1. Wait for the TMA-load warp to fill this stage's SMEM.
    mbarrier.try_wait.parity.shared::cta.b64  complete, [%r68], %r973;
    @!complete bra.uni waitLoop;

    // 2. Issue 4× tcgen05.mma into TMEM accumulator. Async — fire & forget.
    @%p15 tcgen05.mma.cta_group::1.kind::f16 [%r69+0], %rd11, %rd12, %r70, %p92;
    @%p15 tcgen05.mma.cta_group::1.kind::f16 [%r69+0], %rd13, %rd14, %r70, %p16;
    @%p15 tcgen05.mma.cta_group::1.kind::f16 [%r69+0], %rd15, %rd16, %r70, %p16;
    @%p15 tcgen05.mma.cta_group::1.kind::f16 [%r69+0], %rd17, %rd18, %r70, %p16;

    // 3. On the LAST K-iter of a tile (k == 55), commit the accumulator-
    //    ready mbarrier so the epilogue pool can drain TMEM.
    @%p15 tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64
          [%r71];
```

Because `tcgen05.mma` is fully async (the issue is just an enqueue into
the tensor core's TMEM pipeline), this warp can race ahead of the
epilogue store chain. By the time Pool A is on store #2 for tile T,
this warp may already be enqueueing MMAs for tile T+1.

---

## 4. Pool A — the epilogue + TMA stores (`BB0_20`)

The 8 compute warps own everything that touches `out` and `factors` in
HBM. Per output tile, they:

1. Drain TMEM accumulator into registers.
2. Run SwiGLU math (silu + multiply + scale).
3. Stage results into SMEM and fire 3 TMA stores, one at a time.

### 4a. Drain TMEM into registers

```ptx
// 64-wide TMEM read, 32-bit elements (full accumulator slab).
tcgen05.ld.sync.aligned.32x32b.x64.b32  {%r176,...,%r239}, [%r240+0];
tcgen05.wait::ld.sync.aligned;
tcgen05.ld.sync.aligned.32x32b.x64.b32  {%r241,...,%r304}, [%r305+0];
tcgen05.wait::ld.sync.aligned;

// Release the TMEM buffer back to the MMA warp — tile T+1's MMAs may
// now safely overwrite it.
@%p25 mbarrier.arrive.shared::cta.b64  _, [%r306];
cp.async.bulk.wait_group.read 0;
```

After the second `mbarrier.arrive`, Pool B's MMA warp can begin tile T+1
with no further coordination from Pool A.

### 4b. The three TMA stores (the centerpiece)

All three stores follow the same pattern: convert bf16 → write to SMEM →
fence → barrier → TMA → commit → wait. The wait between stores is
`wait_group.read 0` — it waits only for the **SMEM read** to drain so the
buffer can be reused, **not** for the HBM commit.

```ptx
// === TMA STORE #1: factors[:, 0:128] =====================================
fence.proxy.async.shared::cta;
bar.sync 0, 256;                                          // sync only Pool A
@%p58 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
          [%rd30, {%r347, %r990}], [%r348];               // %rd30 = factors descr
cp.async.bulk.commit_group;
cp.async.bulk.wait_group.read 0;                          // wait SMEM drain
bar.sync 0, 256;

// === TMA STORE #2: factors[:, 128:256] ===================================
//   ... restage SAME SMEM buffer with the second half of factors ...
or.b32  %r381, %r347, 128;                                // n-coord OR 128
fence.proxy.async.shared::cta;
bar.sync 0, 256;
@%p59 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
          [%rd30, {%r381, %r990}], [%r348];               // same descr, n+128
cp.async.bulk.commit_group;
cp.async.bulk.wait_group.read 0;
bar.sync 0, 256;

// === TMA STORE #3: out[:, 0:128] (the "main" store) ======================
//   ... restage SAME SMEM buffer with the SwiGLU output ...
fence.proxy.async.shared::cta;
bar.sync 0, 256;
@%p60 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
          [%rd31, {%r414, %r990}], [%r348];               // %rd31 = out descr
cp.async.bulk.commit_group;
// NO wait here — fall through to the next tile.
```

Why the stores are serial: all three reuse the **same** SMEM staging
region (the `st.shared.v4.b32 [%r307..%r342]` slots get rewritten for
each store). To make them parallel we'd need 3 distinct staging regions
and 3 mbarriers — not worth it, since the compute pipeline already hides
the latency.

---

## 5. Why this hides the cost of save_factors

Three observations make the design work:

1. **Disjoint barriers.** Pool A syncs on `bar 0` (count 256); Pool B's
   TMA-load warp syncs on `bar 3` (count 64); the MMA warp uses only
   `bar.warp.sync -1`. No instruction in the epilogue can stall Pool B.

2. **`wait_group.read 0`, not `write 0`.** Between TMA stores the
   epilogue waits only for the **SMEM-read** drain (a few hundred cycles),
   not for HBM commit (tens of microseconds). Pool A is never blocked on
   global memory.

3. **MMA is fully async.** `tcgen05.mma` is an enqueue, not a compute.
   Pool B's MMA warp can stay several K-iters ahead of where Pool A is
   draining. By the time Pool A starts the epilogue for tile T, Pool B
   has already issued some MMAs for tile T+1.

Together: the 2 extra TMA stores for `factors` add HBM bandwidth but
**zero critical-path latency** relative to the no-save variant. That's
why the benchmark shows save_factors at parity with the unsaved forward.

---

## File map

| File | What it is |
|---|---|
| `dump_triton_ptx.py` | Compiles the kernel via `JITFunction.warmup` and writes `artifacts/save_factors.{ptx,ttgir}` |
| `annotate_ptx.py` | Re-runs over the dumped PTX, injecting the section banners and inline notes that produced `artifacts/save_factors.annotated.ptx` |
| `artifacts/save_factors.ptx` | Raw Triton output |
| `artifacts/save_factors.annotated.ptx` | Full PTX with annotations (still valid PTX — all insertions are `//` comments) |
| `PTX_WALKTHROUGH.md` | This doc |
