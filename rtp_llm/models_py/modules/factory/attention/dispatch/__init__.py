"""Dynamic decode attention backend dispatch.

During CUDA graph capture, pick the best backend per decode graph for each bs
bucket, using support filtering and real-machine performance comparison.
Fixed-priority first-match cannot pick the best backend when the winner flips
with bs (measured in roughly 78% of cases); this package replaces that with a
data-driven approach.

Module layering:
  - selector: pure CPU selection logic, unit-testable in CI.
  - backend_bench: real capture-path timing (GPU).
  - backend_selector: startup-time orchestration and TP broadcast (GPU + NCCL).
"""
