"""DSV4 FP8 compressor pool layout constants.

Shared by ``compressor_fp8.py`` (writer) and ``_indexer_score_fp8.py``
(reader). Mirrors the C++ side (``DSV4CacheConfig.h``):

  * ``KV_HEAD_DIM=512`` / ``KV_ENTRY_BYTES=584`` — CSA / HCA FP8 KV slot
    (448 fp8 NoPE + 64 bf16 RoPE + 8 UE8M0 scales).
  * ``INDEXER_HEAD_DIM=128`` / ``INDEXER_ENTRY_BYTES=132`` — indexer FP8
    KV slot (128 fp8 K + 4-byte fp32 scale).
"""

KV_HEAD_DIM = 512
KV_ROPE_HEAD_DIM = 64
KV_NOPE_HEAD_DIM = KV_HEAD_DIM - KV_ROPE_HEAD_DIM  # 448
KV_TOKEN_DATA_SIZE = KV_NOPE_HEAD_DIM + KV_ROPE_HEAD_DIM * 2  # 448 fp8 + 128 bf16 = 576
KV_SCALES_PER_TOKEN = 8  # 7 real UE8M0 + 1 pad
KV_ENTRY_BYTES = KV_TOKEN_DATA_SIZE + KV_SCALES_PER_TOKEN  # 584

INDEXER_HEAD_DIM = 128
INDEXER_ENTRY_BYTES = 132  # 128 fp8 + 4-byte fp32 scale

DSV4_KERNEL_TOKENS_PER_BLOCK = 256
