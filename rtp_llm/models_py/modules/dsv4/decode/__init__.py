"""DeepSeek-V4 decode-only ops & metadata.

Sub-modules:
- ``fp8_sparse_attn_decode_op`` — FP8 KV sparse attn decode
- ``fp8_kv_quant_decode_op`` — FP8 KV dequant for decode
- ``indexer_decode_op`` — paged FP8 MQA logits + topk via DeepGEMM
- ``forward`` — decode forward helpers
"""
