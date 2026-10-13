"""Sleep/wake control and shared bounded-shutdown coordination.

The controller owns phase orchestration; resources owns backend GPU operations.
Lease, validation, status and timing are shared without importing either layer.
Import the needed submodule explicitly: importing this package must not load
gRPC clients, model code or CUDA/NCCL adapters.
"""
