"""Construction-level UT for the FP8 ``__init__`` dispatch in
``Attention`` (Phase A of the FP8 KV cache rewrite).

Validates that ``kv_cache_dtype="FP8"`` swaps the BF16 ``Compressor`` /
``Indexer`` for ``CompressorFP8`` / ``IndexerFP8``, and that the nested
indexer compressor is also the FP8 variant. Compressed-block dequant
math itself is covered by ``test_compressor_fp8_class.py`` /
``test_compressor_fp8_reader.py``; this UT just gates the wiring.
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4.attention import Attention
from rtp_llm.models_py.modules.dsv4.compressor import Compressor
from rtp_llm.models_py.modules.dsv4.compressor_fp8 import CompressorFP8
from rtp_llm.models_py.modules.dsv4.indexer import Indexer
from rtp_llm.models_py.modules.dsv4.indexer_fp8 import IndexerFP8


def _make(ratio: int, dtype):
    """Tiny synthetic Attention. ``head_dim=512`` is required by the FP8
    compressor (584B layout asserts head_dim==512). Other dims kept small
    to keep init fast."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return Attention(
            layer_id=0,
            dim=128,
            n_heads=4,
            q_lora_rank=64,
            head_dim=512,
            rope_head_dim=64,
            o_lora_rank=64,
            o_groups=2,
            window_size=8,
            compress_ratio=ratio,
            compress_rope_theta=10000.0,
            rope_theta=10000.0,
            rope_factor=1.0,
            beta_fast=32,
            beta_slow=1,
            original_seq_len=0,
            max_batch_size=2,
            max_seq_len=64,
            index_n_heads=4,
            index_head_dim=128,
            index_topk=4,
            norm_eps=1e-6,
            weights=None,
            prefix="",
            tp_size=1,
            tp_rank=0,
            kv_cache_dtype=dtype,
        )
    finally:
        torch.set_default_dtype(prev)


def test_bf16_csa_uses_legacy_classes():
    """ratio=4 + dtype=None → Compressor + Indexer (BF16 legacy)."""
    a = _make(ratio=4, dtype=None)
    assert isinstance(a.compressor, Compressor) and not isinstance(
        a.compressor, CompressorFP8
    ), f"expected legacy Compressor, got {type(a.compressor).__name__}"
    assert isinstance(a.indexer, Indexer) and not isinstance(
        a.indexer, IndexerFP8
    ), f"expected legacy Indexer, got {type(a.indexer).__name__}"
    assert not a._kv_cache_is_fp8


def test_fp8_csa_uses_fp8_classes():
    """ratio=4 + dtype=FP8 → CompressorFP8(head_dim=512) +
    IndexerFP8 + nested CompressorFP8(head_dim=128)."""
    a = _make(ratio=4, dtype="FP8")
    assert a._kv_cache_is_fp8
    assert isinstance(
        a.compressor, CompressorFP8
    ), f"expected CompressorFP8, got {type(a.compressor).__name__}"
    assert (
        a.compressor.head_dim == 512
    ), f"CSA compressor head_dim must be 512; got {a.compressor.head_dim}"
    assert isinstance(
        a.indexer, IndexerFP8
    ), f"expected IndexerFP8, got {type(a.indexer).__name__}"
    # Nested indexer compressor must be CompressorFP8 too — it writes the
    # 132B INDEXER pool that DeepGEMM ``fp8_paged_mqa_logits`` consumes.
    nested = a.indexer.compressor
    assert isinstance(
        nested, CompressorFP8
    ), f"expected nested CompressorFP8, got {type(nested).__name__}"
    assert nested.head_dim == 128, (
        f"indexer nested compressor head_dim must be 128 (132B layout); "
        f"got {nested.head_dim}"
    )


def test_fp8_hca_no_indexer():
    """ratio=128 + dtype=FP8 → CompressorFP8 only (HCA layers have no indexer)."""
    a = _make(ratio=128, dtype="FP8")
    assert isinstance(a.compressor, CompressorFP8)
    assert a.compressor.head_dim == 512
    assert a.indexer is None


def test_fp8_swa_only_no_compressor():
    """ratio=0 + dtype=FP8 → no compressor, no indexer (SWA-only layer)."""
    a = _make(ratio=0, dtype="FP8")
    assert a._kv_cache_is_fp8
    assert a.compressor is None
    assert a.indexer is None


if __name__ == "__main__":
    test_bf16_csa_uses_legacy_classes()
    test_fp8_csa_uses_fp8_classes()
    test_fp8_hca_no_indexer()
    test_fp8_swa_only_no_compressor()
    print("OK")
