"""Shared speculative-decoding model helpers.

Keep package import side-effect free: the proposer depends on compiled RTP
bindings, while the torch reference oracle is intentionally CPU-only.
"""
