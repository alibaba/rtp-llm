"""DeepSeek-V4 decode-only ops & metadata.

This subpackage hosts the decode-path pieces — sparse attn op, indexer
op, KV-write op, and the per-step attention metadata builder. Strictly
**decode-only**: nothing in here touches prefill code, so PD-disagg can
later split prefill/decode services on a clean cleavage line.

Sub-modules:
- ``sparse_attn_decode_op`` — per-step batched MQA + per-head sink
  (currently TileLang BF16; FP8 swap deferred to Phase 4 because
  FlashMLA's sparse decode requires ``is_fp8_kvcache=True``).
- ``indexer_decode_op`` — paged FP8 MQA logits + topk via DeepGEMM,
  matching ``base/cuda/indexer_op.py``'s V3.2 path.
- ``kv_write_decode_op`` — per-token append into SWA / compressed-K
  buffers via ``slot_mapping``.
- ``decode_attn_metadata`` — per-step builder for slot mappings,
  block tables, indexer topk buffer.
"""
