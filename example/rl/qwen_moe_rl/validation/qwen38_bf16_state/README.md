# BF16 policy and Adam state qualification

`training/qwen38_bf16_optimizer.py` stores the policy in BF16 and both Adam
moments in CPU BF16. Reduced gradients, accumulation, clipping, and bounded
optimizer arithmetic use FP32. Counter-based stochastic rounding applies to
all three persistent writes. It keeps no full FP32 master or moment copy and
has no disk fallback. The CPU state allocator refuses an insufficient budget.

Run `test_bf16_numerics.py` in a temporary working directory with the example's
`training` directory on `PYTHONPATH`. It checks independent NumPy update and
rounding equations, uneven-shard/chunk invariance, small updates, nonfinite
inputs, and the state budget. Torch 2.13's reduction interception is explicitly
version and source guarded; other Torch implementations need review.

This source passed the CPU reference, real four-GPU HF/FSDP tests, and a real
18-node/72-GPU two-update FSDP microtest on SM100/Torch 2.13. At world72 one
independent CPU/GPU FP32 operation differed by one ULP at an exact stochastic
rounding threshold; replaying the actual GPU equations was bit exact. These
tests do not constitute completed Qwen3.8-2.4T RL training or convergence proof.

The opt-in `qwen38_bf16_training_runtime` additionally streams the pinned
checkpoint directly into 72 BF16 FSDP shards, subdivides expert and vocabulary
wraps, and keeps original FP32 rotary buffers across the real verl dtype cast.
Its phase manager releases gradients and partially offloads masters before
native allocation. Restoring training requires the real native cleanup proof.
The original rLLM/verl configuration path was checked with reference and critic
disabled, one TP8/attention-DP9/global-EP72 rollout world, and two GRPO updates.
The final policy hook requires actual step-2 synchronization and fresh native
probes before reporting success. The matching world adapter and SM100 native
build remain separate prerequisites; checkpoints are disabled until a bounded
state writer is provided.


The bounded checkpoint path keeps attention/normalization, router, shared
expert, and individual expert chunks in separate non-reentrant checkpoints.
It recreates normalized inputs from the same saved residual storage instead
of retaining a normalized activation for every chunk. Parameter names and
mathematical routing and loss are preserved. The exact installed HF decoder
source is guarded before this adaptation is enabled.

In a real four-GPU, 32-chunk-per-layer test, whole-decoder recomputation kept
32 expert chunks gathered concurrently. The bounded path kept one active
chunk; one additional pending backward all-gather remained. An empty explicit
backward prefetch list in Torch 2.13 falls back to its default lookahead, so
this profile does not claim that backward prefetch is disabled. Budget both
weight buffers and a separate FP32 reduce-scatter input.

Tiny HF forward logits were bit exact. FP32 CPU gradients agreed to about
2e-7 relative L2; BF16 backward accumulation order changed, producing about
0.5% CPU and 1.1% four-GPU relative L2 gradient differences. All trainable
parameters retained gradients, and the real four-GPU BF16 optimizer completed
two updates with source-lease offload/restore. These are limited qualification
results, not a full 2.4T memory peak, RL completion, or convergence result.
The two `qualify_*checkpoint*` scripts use the checked-in official config
fixture and random tiny weights; they never open official weight payloads.


`qwen38_length_guards` is enabled by explicit real-token bounds in the recipe.
It rejects out-of-range trace steps and merged rows before the original
converter can truncate, then checks all converted IDs and masks. Synthetic
CPU boundary tests exercise the actual installed converter; no tool calls or
rewards are manufactured by those tests. The separate GPU 8K single-layer GDN
measurement does not certify a full-model 8K training peak, so the initial
complete-model recipe uses a 4K real-token bound.

`Q38_MICRO_BOUNDARY_RELEASE_CACHE=1` enables the dedicated training hook that
synchronizes CUDA and releases empty allocator segments before each original
micro forward. Live BF16 masters and FP32 gradient slots are preserved. Its
JSON audit records the actual micro index, accumulated gradient contributions,
parameter `.grad` shadows, allocated/reserved CUDA bytes, and process CPU RSS.
The gradient-cache limit covers FP32 slots only; activation, all-gather,
reduce-scatter and allocator memory need their own headroom. Process RSS is
not a container cgroup limit or peak, and a static state plan is not a measured
full-model memory result.
