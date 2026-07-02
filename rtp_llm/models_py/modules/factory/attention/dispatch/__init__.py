"""Dynamic decode attention backend dispatch.

During CUDA graph capture, pick the best backend per decode graph for each bs
bucket, using two-stage filtering: the precision gate decides which backends are
usable, and performance comparison decides which is faster. Fixed-priority
first-match cannot pick the best backend when the winner flips with bs (measured
in roughly 78% of cases the winner flips with bs); this package replaces that
with a data-driven approach.

Module layering:
  - precision_metrics / decode_gate / plan / selector: pure CPU, no GPU or heavy
    rtp_llm dependencies, unit-testable in CI.
  - backend_bench: real capture-path timing (GPU).
  - backend_selector: startup-time orchestration and TP broadcast (GPU + NCCL).
"""
