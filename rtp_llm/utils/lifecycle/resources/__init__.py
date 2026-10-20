"""Backend GPU-resource adapters used by model/engine lifecycle hooks.

Adapters do not own HTTP routing, lifecycle RPCs or instance orchestration.
Keep this package inert so control-only callers never initialize CUDA/NCCL.
"""
