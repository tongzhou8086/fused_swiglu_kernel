"""Dump Triton's generated PTX for save_factors and analyze the schedule.

We're hunting two things:
  1. Persistent + flatten pattern: an outer tile loop wrapping the K-loop.
  2. Side-store placement: where the `st.global` for `factors` lands
     relative to the next tile's K-loop mma instructions.  If the
     compiler successfully pipelined the epilogue, the factor stores
     of tile T should issue while tile T+1's K-loop mma is already
     running — visible in PTX as stores between mma blocks instead of
     forming a serial chain at the K-loop tail.

We extract PTX via the JIT kernel's cache (Triton populates it after the
first launch).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import triton

import fused_swiglu.triton_baseline as fbk
import fused_swiglu as fs


M, K, N = 11136, 3584, 14336
DTYPE = torch.bfloat16
torch.manual_seed(0)

x        = torch.randn(M, K,     device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_normal = torch.randn(K, 2 * N, device="cuda", dtype=DTYPE) * (1.0 / math.sqrt(K))
W_packed = fs.pack_swiglu_weight_chunked_torch(W_normal)
out      = torch.empty(M, N,     device="cuda", dtype=DTYPE)
factors  = torch.empty(M, 2 * N, device="cuda", dtype=DTYPE)

jit_kern = fbk._fused_swiglu_wide_packed_save_factors_kernel

# JITFunction.warmup compiles the kernel at the given constexpr config
# and returns the CompiledKernel which has `.asm['ptx']`.
import torch
device = torch.cuda.current_device()
import pycuda.driver as drv
drv.init()
num_sms = drv.Device(device).get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)

compiled = jit_kern.warmup(
    x, W_packed, out, factors,
    M, N, K,
    NUM_SMS            = num_sms,
    BLOCK_SIZE_M_      = fbk.BLOCK_SIZE_M,
    BLOCK_SIZE_N_HALF_ = fbk.BLOCK_SIZE_N_HALF,
    BLOCK_SIZE_K_      = fbk.BLOCK_SIZE_K,
    GROUP_SIZE_M_      = fbk.GROUP_SIZE_M,
    WARP_SPECIALIZE_   = fbk.WARP_SPECIALIZE,
    FLATTEN            = True,
    num_warps          = fbk.NUM_WARPS,
    num_stages         = fbk.SAVE_NUM_STAGES,
    grid               = (1,),
)
print(f"compiled artifact type: {type(compiled).__name__}")
print(f"asm keys: {list(compiled.asm.keys())}")
print()

artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)

ptx = compiled.asm.get("ptx", "")
ttgir = compiled.asm.get("ttgir", "")

ptx_path = os.path.join(artifacts_dir, f"save_factors.ptx")
with open(ptx_path, "w") as f:
    f.write(ptx)
print(f"PTX: {len(ptx)} bytes  →  {ptx_path}")

if ttgir:
    ttgir_path = os.path.join(artifacts_dir, f"save_factors.ttgir")
    with open(ttgir_path, "w") as f:
        f.write(ttgir)
    print(f"TTGIR: {len(ttgir)} bytes  →  {ttgir_path}")

# ── Quick PTX analysis ──
import re

with open(os.path.join(artifacts_dir, "save_factors.ptx")) as f:
    ptx = f.read()

print()
print("=== PTX statistics ===")
print(f"  total length            : {len(ptx)} chars")
print(f"  total lines             : {ptx.count(chr(10))}")
# Instruction-count proxies.
def count(pat):
    return len(re.findall(pat, ptx))

print()
print("=== instruction counts ===")
print(f"  tcgen05.mma   (tensor MMA)    : {count(r'tcgen05\.mma')}")
print(f"  tcgen05.ld    (TMEM loads)    : {count(r'tcgen05\.ld')}")
print(f"  cp.async.bulk (TMA loads/stores): {count(r'cp\.async\.bulk')}")
print(f"  st.global     (HBM stores)    : {count(r'st\.global')}")
print(f"  st.shared     (SMEM stores)   : {count(r'st\.shared')}")
print(f"  ld.global     (HBM loads)     : {count(r'ld\.global')}")
print(f"  ld.shared     (SMEM loads)    : {count(r'ld\.shared')}")
print(f"  bra           (branches)      : {count(r'(?<!\\.)\\bbra\\b')}")
print(f"  mbarrier      (sync ops)      : {count(r'mbarrier')}")

# Look for evidence of persistent loop (outer loop wrapping inner K-loop).
# A persistent grid kernel typically has a BACKWARD branch labeled with
# something like LOOP_BEG / NUM_SMS / tile_id update.
print()
print("=== loop structure detection ===")
# Heuristic: persistent kernels have multiple distinct backward branches.
backward_branches = re.findall(r'^\$L__BB\d+_(\d+):', ptx, flags=re.MULTILINE)
unique_labels = sorted(set(backward_branches))
print(f"  basic blocks (labels): {len(unique_labels)}")

# Search for the obvious pattern: an outer-loop variable being incremented
# by NUM_SMS = 148 (B200 SM count).
if re.search(r'add\.s32.*148', ptx):
    print("  FOUND add by 148 → suggests persistent loop incrementing tile_id by NUM_SMS")
else:
    print("  no obvious 'add by 148' pattern (didn't find persistent stride)")

# Look for st.global.v8 / st.global.v4 — these are the wide stores Triton
# uses for the epilogue when materializing factors.
print()
print("=== wide-store inventory (epilogue writes) ===")
print(f"  st.global.v8.b32 (32-byte stores)  : {count(r'st\.global\.v8\.b32')}")
print(f"  st.global.v4.b32 (16-byte stores)  : {count(r'st\.global\.v4\.b32')}")
print(f"  st.global.v2.b32 ( 8-byte stores)  : {count(r'st\.global\.v2\.b32')}")
print(f"  st.global.b32    ( 4-byte stores)  : {count(r'(?<!\\.v[28]\\.b32 |\\.v4\\.b32 )st\\.global\\.b32')}")

# Locate where these stores live relative to the mma instructions.
# Specifically: are there mmas AFTER the last st.global?  If so, the
# stores aren't all at the tail; they're being interleaved with the K-loop.
print()
print("=== mma vs store interleaving ===")
mma_positions   = [m.start() for m in re.finditer(r'tcgen05\.mma', ptx)]
store_positions = [m.start() for m in re.finditer(r'st\.global', ptx)]
if mma_positions and store_positions:
    first_mma  = mma_positions[0]
    last_mma   = mma_positions[-1]
    first_store = store_positions[0]
    last_store  = store_positions[-1]
    print(f"  first mma      at char {first_mma}")
    print(f"  last mma       at char {last_mma}")
    print(f"  first st.global at char {first_store}")
    print(f"  last st.global  at char {last_store}")
    if first_store < last_mma:
        # Stores started before the last mma — interleaving.
        intervening_mmas = sum(1 for p in mma_positions if first_store < p)
        print(f"  → {intervening_mmas} mmas APPEAR AFTER first st.global")
        print(f"    (suggests epilogue stores ARE interleaved with later mmas — persistent pipeline)")
    else:
        print(f"  → all st.global appear AFTER all mma")
        print(f"    (suggests epilogue is on the critical path after the K-loop ends)")
