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
