"""Two-page eager attention, with native retained rings and growing GLOBAL pools."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.decode.forward import build_metadata_eager
from rtp_llm.models_py.modules.dsv4.fp8._v41_decode_attention_sm120 import (
    sparse_decode_attention,
)
from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_attention_sm120 import (
    sparse_prefill_attention,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
    contraction_oracle,
    make_layer,
    prepare,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_decode_attention_sm120 import (
    oracle,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_decode_owner_sm120 import (
    decode_metadata,
    prefill,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache


def two_page_tables(cache, batch=2):
    return {
        tag: torch.tensor(
            ([[6], [4]] if tag in ("swa_kv", "csa_state") else [[7, 2], [5, 3]])[
                :batch
            ],
            device="cuda",
            dtype=torch.int32,
        )
        for tag in cache.layouts
    }


class ContextOwnerSM120Test(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41512)

    def assert_same_cache(self, cache, reference):
        for key, raw in cache.storage.items():
            torch.testing.assert_close(raw, reference.storage[key], rtol=0, atol=0)
            layout = cache.layouts[key[1]]
            self.assertTrue(bool((raw[:, layout.stride_bytes :] == 173).all()))
            self.assertEqual(raw[0, : layout.stride_bytes].count_nonzero().item(), 0)

    def test_eager_builder_selects_explicit_ring_mode(self):
        args = SimpleNamespace(
            v41_config={},
            max_seq_len=512,
            window_size=128,
            head_dim=512,
            compress_ratios=[0],
            n_layers=1,
            index_topk=512,
        )
        starts = torch.tensor([256, 511], device="cuda", dtype=torch.int32)
        table = torch.tensor([[7], [3]], device="cuda", dtype=torch.int32)
        inputs = {
            "swa_kv": SimpleNamespace(
                is_target_verify=False,
                sequence_lengths=starts,
                kv_cache_kernel_block_id_device=table,
            )
        }
        for entries in (128, 130):
            metadata = build_metadata_eager(
                args,
                inputs,
                torch.device("cuda"),
                {"swa_kv": (entries, 256, 1)},
                fp8_kv_cache=True,
            )
            expected = [
                [
                    page * entries + pos % entries
                    for pos in range(start - 127, start + 1)
                ]
                for page, start in ((7, 256), (3, 511))
            ]
            self.assertEqual(metadata.swa_global_slots.tolist(), expected)
            self.assertEqual(
                metadata.pool_write_slot_mappings["swa_kv"].tolist(),
                [row[-1] for row in expected],
            )
        args.v41_config = None
        legacy = build_metadata_eager(
            args,
            inputs,
            torch.device("cuda"),
            {"swa_kv": (128, 256, 1)},
            fp8_kv_cache=True,
        )
        self.assertTrue(bool((legacy.swa_global_slots[1] == -1).all()))

    @torch.inference_mode()
    def test_cold_prefill_and_eight_decode_steps_across_pages(self):
        for ratio in (0, 1, 2):
            for gen in (0, 1):
                with self.subTest(ratio=ratio, gen=gen):
                    torch.manual_seed(41512)
                    layer = make_layer(2 if ratio else 0, ratio, gen=gen)
                    torch.manual_seed(41512)
                    reference = make_layer(2 if ratio else 0, ratio, gen=gen)
                    cache = NativeTaggedCache([layer], gen=gen, pages=8)
                    ref_cache = NativeTaggedCache([reference], gen=gen, pages=8)
                    tables = two_page_tables(cache)
                    starts = [253, 504]
                    prompt = torch.randn(sum(starts), 512, device="cuda").bfloat16()
                    prefill(layer, cache, prompt, starts, tables=tables)
                    prefill(reference, ref_cache, prompt, starts, tables=tables)
                    for _ in range(8):
                        x = torch.randn(2, 1, 512, device="cuda").bfloat16()
                        metadata = decode_metadata(layer, cache, starts, tables)
                        checked = []

                        def attention(q, sp, ss, sink, gp=None, gs=None):
                            actual = sparse_decode_attention(q, sp, ss, sink, gp, gs)
                            sources = [
                                (
                                    V41PagedCache(
                                        cache.groups[0, "swa_kv"].kv_cache_base,
                                        "swa_kv",
                                        gen,
                                    ),
                                    ss,
                                )
                            ]
                            if ratio:
                                tag = layer._global_region()
                                sources.append(
                                    (
                                        V41PagedCache(
                                            cache.groups[0, tag].kv_cache_base, tag, gen
                                        ),
                                        gs,
                                    )
                                )
                            torch.testing.assert_close(
                                actual.cpu(),
                                oracle(q, sink, sources),
                                rtol=0.01,
                                atol=0.004,
                            )
                            checked.append(True)
                            return actual

                        with patch(
                            "rtp_llm.models_py.modules.dsv4.fp8._v41_decode_attention_sm120.sparse_decode_attention",
                            side_effect=attention,
                        ):
                            output = layer.forward_decode(x, metadata, cache)
                        self.assertEqual(len(checked), 1)
                        self.assertTrue(bool(torch.isfinite(output).all()))
                        prefill(reference, ref_cache, x[:, 0], [1, 1], starts, tables)
                        self.assert_same_cache(cache, ref_cache)
                        starts = [value + 1 for value in starts]

    @torch.inference_mode()
    def test_multi_page_continuation_and_shared_source(self):
        for ratio in (1, 2):
            shared = {}
            source = make_layer(20, ratio, shared, gen=0)
            consumer = make_layer(21, ratio, shared, source=20, gen=0)
            cache = NativeTaggedCache([source, consumer], gen=0, pages=8)
            tables = two_page_tables(cache)
            starts = [125, 129]
            prompt = torch.randn(sum(starts), 512, device="cuda").bfloat16()
            prefill(source, cache, prompt, starts, tables=tables)
            prefill(consumer, cache, prompt, starts, tables=tables)
            lengths = [258, 257]
            x = torch.randn(sum(lengths), 512, device="cuda").bfloat16()
            checked = []

            def attention(q, kv, indices, lengths, sinks):
                actual = sparse_prefill_attention(q, kv, indices, lengths, sinks)
                rows = [
                    row
                    for row in (0, 127, 128, 254, 255, 256, len(q) - 1)
                    if row < len(q)
                ]
                expected = contraction_oracle(
                    q[rows], kv, indices[rows], lengths[rows], sinks
                )
                torch.testing.assert_close(
                    actual[rows].cpu(), expected, rtol=0.01, atol=0.004
                )
                checked.append(True)
                return actual

            with patch(
                "rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_attention_sm120.sparse_prefill_attention",
                side_effect=attention,
            ):
                prefill(source, cache, x, lengths, starts, tables)
                before = {
                    key: raw.clone()
                    for key, raw in cache.storage.items()
                    if key[0] == 0
                }
                prefill(consumer, cache, x * 2, lengths, starts, tables)
            self.assertTrue(checked)
            for key, raw in before.items():
                torch.testing.assert_close(cache.storage[key], raw, rtol=0, atol=0)
            ends = [a + b for a, b in zip(starts, lengths)]
            source.forward_decode(
                x[:2, None].contiguous(),
                decode_metadata(source, cache, ends, tables),
                cache,
            )

    @torch.inference_mode()
    def test_all_required_pages_validated_before_writes(self):
        layer = make_layer(2, 2, gen=0)
        cache = NativeTaggedCache([layer], gen=0, pages=8)
        tables = two_page_tables(cache)
        x = torch.randn(600, 512, device="cuda").bfloat16()
        positions, _, _ = prepare(layer, x, cache, [300, 300], tables=tables)
        before = {key: raw.clone() for key, raw in cache.storage.items()}
        for tag in ("swa_kv", "csa_state", "csa_kv", "indexer_kv"):
            col = 0 if tag in ("swa_kv", "csa_state") else 1
            for value in (0, -1, 8, int(tables[tag][1, 0])):
                invalid = {key: tensor.clone() for key, tensor in tables.items()}
                invalid[tag][0, col] = value
                with self.assertRaisesRegex(ValueError, "block table"):
                    layer(x, positions, cache, invalid)
                for key, raw in before.items():
                    torch.testing.assert_close(cache.storage[key], raw, rtol=0, atol=0)
            invalid = dict(tables)
            invalid[tag] = (
                tables[tag].expand(-1, 2).contiguous()
                if col == 0
                else tables[tag][:, :1].contiguous()
            )
            with self.assertRaisesRegex(ValueError, "block table"):
                prepare(layer, x, cache, [300, 300], tables=invalid)
        with self.assertRaises(NotImplementedError):
            prepare(layer, x, cache, [513, 87], tables=tables)
        for key, raw in before.items():
            torch.testing.assert_close(cache.storage[key], raw, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
