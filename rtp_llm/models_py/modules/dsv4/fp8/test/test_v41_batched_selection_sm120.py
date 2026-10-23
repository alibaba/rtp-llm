"""Batched fused index selection must match independent request execution."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_indexer as indexer
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.native_l20_transfer import _native_tables
from rtp_llm.models_py.modules.dsv4.native_pp_execution import l20_tables
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import V41CacheIdentity


class V41BatchedTablesTest(unittest.TestCase):
    def test_uneven_allocations_preserve_page_order_and_reserved_capacity(self):
        first, second = V41CacheIdentity(1, 1, 0), V41CacheIdentity(2, 2, 0)
        dispatch = SimpleNamespace(
            spans=(SimpleNamespace(identity=first), SimpleNamespace(identity=second)),
            tables={
                "hca_kv": torch.tensor([[3, 1, 4], [2, 0, 0]], dtype=torch.int32),
                "indexer_kv": torch.tensor([[5, 2], [3, 0]], dtype=torch.int32),
            },
        )
        snapshots = {tag: table.clone() for tag, table in dispatch.tables.items()}
        expected = {
            first: {"hca_kv": (3, 1, 4), "indexer_kv": (5, 2)},
            second: {"hca_kv": (2,), "indexer_kv": (3,)},
        }
        actual = l20_tables(dispatch)
        self.assertEqual(actual, expected)
        for tables in actual.values():
            self.assertEqual(_native_tables(tables), tables)
        for tag, table in dispatch.tables.items():
            self.assertTrue(torch.equal(table, snapshots[tag]))

    def test_padding_removal_does_not_hide_invalid_allocated_pages(self):
        identity = V41CacheIdentity(1, 1, 0)
        for pages in ([1, 0, 2, 0], [0, 1, 0, 0], [1, -1, 0, 0]):
            with self.subTest(pages=pages):
                dispatch = SimpleNamespace(
                    spans=(SimpleNamespace(identity=identity),),
                    tables={
                        "hca_kv": torch.tensor([pages], dtype=torch.int32),
                        "indexer_kv": torch.tensor([[1, 0]], dtype=torch.int32),
                    },
                )
                with self.assertRaisesRegex(ValueError, "reserved page zero"):
                    _native_tables(l20_tables(dispatch)[identity])


class V41BatchedSelectionTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "SM120 hardware is required")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        self.assertTrue(indexer.is_supported(torch.device("cuda"), 32, 128))
        torch.manual_seed(41987)

    @torch.inference_mode()
    def check_requests(self, counts, *, interleaved=False):
        keys = [
            indexer.PrefillIndexerKeys(
                *indexer.quantize_indexer_k_reference(
                    torch.randn(n, 128, device="cuda", dtype=torch.bfloat16)
                )
            )
            for n in (1025, 513)
        ]
        total = sum(counts)
        qr = torch.randn(total, 32 * 128, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(total, 32, device="cuda", dtype=torch.bfloat16)
        req_ids = torch.repeat_interleave(
            torch.arange(2, device="cuda"), torch.tensor(counts, device="cuda")
        )
        visible = torch.cat(
            [
                torch.linspace(0, len(k), count, device="cuda").long()
                for k, count in zip(keys, counts)
            ]
        )
        freqs = torch.polar(
            torch.ones(2050, 32, device="cuda"),
            torch.randn(2050, 32, device="cuda"),
        )
        if interleaved:
            order = torch.argsort(torch.arange(total, device="cuda") % 3, stable=True)
            qr, x, req_ids, visible = (
                value[order] for value in (qr, x, req_ids, visible)
            )

        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                positions = (visible * ratio - 1).clamp_min(0)

                def select(request_keys, rows, ids, slices=None):
                    owner = SimpleNamespace(
                        _shared_attention={
                            "global": {0: [(None, k) for k in request_keys]}
                        },
                        is_index_source=True,
                        kv_source_layer_id=0,
                        layer_id=0,
                        index_topk=512,
                        index_n_heads=32,
                        index_head_dim=128,
                        compress_ratio=ratio,
                        index_wq=None,
                        # Isolate selection from batch-dependent projection rounding.
                        _lin=lambda _, query: query.clone(),
                        index_weights=torch.eye(
                            32, device="cuda", dtype=torch.bfloat16
                        ),
                        freqs_cis=freqs,
                        rope_head_dim=64,
                    )
                    result = AttentionV41FP8._select_indices(
                        owner,
                        x[rows],
                        qr[rows],
                        positions[rows],
                        ids,
                        request_row_slices=slices,
                    )
                    self.assertIs(owner._shared_attention["topk"][0], result)
                    return result

                slices = (
                    None
                    if interleaved
                    else [slice(0, counts[0]), slice(counts[0], total)]
                )
                actual = select(keys, slice(None), req_ids, slices)
                expected = torch.empty_like(actual)
                for request, key in enumerate(keys):
                    rows = torch.where(req_ids == request)[0]
                    if not rows.numel():
                        continue
                    expected[rows] = select([key], rows, torch.zeros_like(rows))
                # Native TopK leaves output order unspecified.
                self.assertTrue(
                    torch.equal(actual.sort(-1).values, expected.sort(-1).values)
                )
                self.assertEqual(actual.dtype, torch.int32)
                for row in range(total):
                    selected = actual[row][actual[row] >= 0]
                    limit = min(
                        int((positions[row] + 1) // ratio), len(keys[int(req_ids[row])])
                    )
                    self.assertEqual(selected.numel(), min(512, limit))
                    self.assertEqual(selected.unique().numel(), selected.numel())
                    self.assertTrue(bool((selected < limit).all()))

    def test_contiguous_requests(self):
        self.check_requests((7, 5))

    def test_interleaved_requests(self):
        self.check_requests((7, 5), interleaved=True)

    def test_empty_first_request(self):
        self.check_requests((0, 5))


if __name__ == "__main__":
    unittest.main()
