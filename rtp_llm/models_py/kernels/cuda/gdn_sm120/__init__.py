"""SM120 GDN prefill kernels adapted from FlashInfer 0.7.0.

Keep only CP prefill with page checkpoints, using the existing FlashInfer
utilities and scratch allocator and CUTLASS DSL 4.4.2 store operations.
"""
