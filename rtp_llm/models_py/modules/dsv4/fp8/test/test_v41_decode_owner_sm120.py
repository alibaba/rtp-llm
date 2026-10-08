"""Prefill-to-decode qualification of the upstream V4.1 attention owner."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8._v41_decode_attention_sm120 import (
    sparse_decode_attention,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import rms_norm, rope_only
from rtp_llm.models_py.modules.dsv4.fp8.decode.compute_qkv import decode_compute_qkv
from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
    build_decode_metadata_fp8,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
    make_layer,
    output_composition,
    prepare,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_decode_attention_sm120 import (
    oracle,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache


def decode_metadata(layer, cache, starts, tables):
    return build_decode_metadata_fp8(
        torch.tensor(starts, device="cuda", dtype=torch.int32),
        q_len=1,
        window_size=128,
        head_dim=512,
        max_seq_len=layer._rope_max_seq_len,
        compress_ratios=[],
        index_topk=512,
        device=torch.device("cuda"),
        paged_block_tables=tables,
        paged_pool_entries_per_block={
            tag: layout.entries for tag, layout in cache.layouts.items()
        },
        paged_pool_tokens_per_block={tag: 256 for tag in cache.layouts},
        swa_single_block_per_request=True,
    )


def prefill(layer, cache, x, lengths, prefixes=None, tables=None):
    positions, tables, _ = prepare(layer, x, cache, lengths, prefixes, tables)
    layer(x, positions, cache, tables)
    return tables


class DecodeOwnerSM120Test(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41221)

    @torch.inference_mode()
    def test_query_normalization_contract_and_v4_default(self):
        layer = make_layer(gen=0)
        x = torch.randn(2, 1, 512, device="cuda", dtype=torch.bfloat16)
        positions = torch.tensor([3, 129], device="cuda")
        freqs = layer.freqs_cis[positions]
        qr = rms_norm(layer.wq_a(x), layer.q_norm, layer.eps)
        q = layer.wq_b(qr).view(2, 4, 512)
        expected = rope_only(q.clone(), freqs, 64).view(2, 1, 4, 512)
        actual = decode_compute_qkv(layer, x, positions)
        torch.testing.assert_close(actual.qr, qr, rtol=0, atol=0)
        torch.testing.assert_close(actual.q, expected, rtol=0, atol=0)
        del layer.skip_post_q_norm
        legacy = decode_compute_qkv(layer, x, positions)
        normalized = q.float() * torch.rsqrt(
            q.float().square().mean(-1, keepdim=True) + layer.eps
        )
        normalized = rope_only(normalized, freqs, 64).bfloat16().view_as(expected)
        torch.testing.assert_close(legacy.q, normalized, rtol=0.01, atol=0.016)
        self.assertGreater(
            (legacy.q.float() - expected.float()).abs().max().item(), 0.5
        )
        torch.testing.assert_close(actual.kv, legacy.kv, rtol=0, atol=0)

    @torch.inference_mode()
    def test_prefill_then_eight_decode_steps_bytes_and_cpu_attention(self):
        for ratio in (0, 1, 2):
            for gen in (0, 1):
                with self.subTest(ratio=ratio, gen=gen):
                    torch.manual_seed(41222)
                    layer = make_layer(2 if ratio else 0, ratio, gen=gen)
                    torch.manual_seed(41222)
                    reference = make_layer(2 if ratio else 0, ratio, gen=gen)
                    cache = NativeTaggedCache([layer], gen=gen)
                    ref_cache = NativeTaggedCache([reference], gen=gen)
                    starts = [3, 125]
                    prompt = torch.randn(
                        sum(starts), 512, device="cuda", dtype=torch.bfloat16
                    )
                    tables = prefill(layer, cache, prompt, starts)
                    prefill(reference, ref_cache, prompt, starts, tables=tables)
                    for step in range(8):
                        x = torch.randn(2, 1, 512, device="cuda", dtype=torch.bfloat16)
                        metadata = decode_metadata(layer, cache, starts, tables)
                        captured = []

                        def checked_attention(q, sp, ss, sink, gp=None, gs=None):
                            actual = sparse_decode_attention(q, sp, ss, sink, gp, gs)
                            swa = V41PagedCache(
                                cache.groups[0, "swa_kv"].kv_cache_base, "swa_kv", gen
                            )
                            sources = [(swa, ss)]
                            if ratio:
                                tag = layer._global_region()
                                global_cache = V41PagedCache(
                                    cache.groups[0, tag].kv_cache_base, tag, gen
                                )
                                sources.append((global_cache, gs))
                                expected = torch.arange(512, device="cuda")[
                                    None
                                ].expand(2, -1)
                                visible = (metadata.start_pos + 1) // ratio
                                expected = torch.where(
                                    expected < visible[:, None], expected, -1
                                ).int()
                                torch.testing.assert_close(
                                    layer._shared_attention["topk"][layer.layer_id],
                                    expected,
                                    rtol=0,
                                    atol=0,
                                )
                            torch.testing.assert_close(
                                actual.cpu(),
                                oracle(q, sink, sources),
                                rtol=0.01,
                                atol=0.004,
                            )
                            captured.append(actual.clone())
                            return actual

                        with patch(
                            "rtp_llm.models_py.modules.dsv4.fp8._v41_decode_attention_sm120.sparse_decode_attention",
                            side_effect=checked_attention,
                        ):
                            output = layer.forward_decode(x, metadata, cache)
                        expected_output = output_composition(
                            layer,
                            captured[0].view(2, 4, 512),
                            layer.freqs_cis[metadata.position_ids.long()],
                        )
                        torch.testing.assert_close(
                            output[:, 0], expected_output, rtol=0, atol=0
                        )
                        self.assertTrue(bool(torch.isfinite(output).all()))
                        prefill(reference, ref_cache, x[:, 0], [1, 1], starts, tables)
                        for key, raw in cache.storage.items():
                            torch.testing.assert_close(
                                raw,
                                ref_cache.storage[key],
                                rtol=0,
                                atol=0,
                                msg=f"ratio={ratio} gen={gen} step={step} pool={key}",
                            )
                        starts = [s + 1 for s in starts]
                        self.assertIsNone(layer._kv_cache)
                        self.assertIsNone(layer._block_tables_by_type)

    @torch.inference_mode()
    def test_shared_source_consumer_preserves_producer_bytes(self):
        for ratio in (1, 2):
            shared = {}
            source = make_layer(20, ratio, shared, gen=0)
            consumer = make_layer(21, ratio, shared, source=20, gen=0)
            cache = NativeTaggedCache([source, consumer], gen=0)
            prompt = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
            tables = prefill(source, cache, prompt, [5])
            prefill(consumer, cache, prompt, [5], tables=tables)
            for start in range(5, 9):
                x = torch.randn(1, 1, 512, device="cuda", dtype=torch.bfloat16)
                metadata = decode_metadata(source, cache, [start], tables)
                source.forward_decode(x, metadata, cache)
                global_keys = shared["global"][20]
                selected = shared["topk"][20]
                before = {
                    key: value.clone()
                    for key, value in cache.storage.items()
                    if key[0] == 0
                }
                output = consumer.forward_decode(x * 2, metadata, cache)
                self.assertTrue(bool(torch.isfinite(output).all()))
                self.assertIs(shared["global"][20], global_keys)
                self.assertIs(shared["topk"][20], selected)
                for key, raw in before.items():
                    torch.testing.assert_close(cache.storage[key], raw, rtol=0, atol=0)

    @torch.inference_mode()
    def test_last_page_position_and_rejection_before_writes(self):
        layer = make_layer(2, 2, gen=0)
        cache = NativeTaggedCache([layer], gen=0)
        prompt = torch.randn(255, 512, device="cuda", dtype=torch.bfloat16)
        tables = prefill(layer, cache, prompt, [255])
        x = torch.randn(1, 1, 512, device="cuda", dtype=torch.bfloat16)
        metadata = decode_metadata(layer, cache, [255], tables)
        output = layer.forward_decode(x, metadata, cache)
        self.assertTrue(bool(torch.isfinite(output).all()))
        before = {key: value.clone() for key, value in cache.storage.items()}
        bad_slots = {
            **metadata.pool_write_slot_mappings,
            "swa_kv": metadata.pool_write_slot_mappings["swa_kv"] + 1,
        }
        cases = [
            (x, decode_metadata(layer, cache, [256], tables)),
            (x, replace(metadata, is_cuda_graph=True)),
            (x, replace(metadata, paged_pool_tokens_per_block={"swa_kv": 128})),
            (x, replace(metadata, pool_write_slot_mappings=bad_slots)),
            (x, replace(metadata, swa_global_slots=metadata.swa_global_slots + 1)),
            (x, replace(metadata, position_ids=metadata.position_ids - 1)),
            (x, replace(metadata, req_id_per_token=metadata.req_id_per_token + 1)),
            (x.expand(1, 2, 512).contiguous(), metadata),
            (x, decode_metadata(layer, cache, [0], tables)),
            (x, None),
        ]
        for value, bad in cases:
            with self.assertRaises((ValueError, NotImplementedError)):
                layer.forward_decode(value, bad, cache)
            for key, raw in before.items():
                torch.testing.assert_close(cache.storage[key], raw, rtol=0, atol=0)
            self.assertIsNone(layer._kv_cache)
            self.assertIsNone(layer._block_tables_by_type)


if __name__ == "__main__":
    unittest.main()
