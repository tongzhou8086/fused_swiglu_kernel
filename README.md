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
    __init__.py            # exports the wrappers and tile constants
    triton_baseline.py     # copy of upstream save_factors kernel + helpers
  bench.py                 # forward-kernel comparison at the target shape
  README.md
```

## Run

```bash
srun -p dedicated --gres=gpu:nvidia_b200:1 --time=00:10:00 \
    ~/miniconda3/bin/python bench.py
```
