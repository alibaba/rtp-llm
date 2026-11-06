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
