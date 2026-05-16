"""DeepSeek-V4 decode-only ops & metadata.

Sub-modules:
- ``indexer_decode_op`` — paged FP8 MQA logits + topk via DeepGEMM
- ``kv_write_decode_op`` — per-token append into pool via slot_mapping
- ``decode_attn_metadata`` — per-step builder for slot mappings,
  block tables, indexer topk buffer
- ``forward`` — decode forward helpers
"""
