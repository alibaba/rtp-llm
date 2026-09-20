"""CPU regressions for prepared request slices in V4.1 prefill selection.

Only the GPU scorer/selector entry points are replaced with CPU oracles. The
real request loop, chunking, candidate publication/masking and output writes
run unchanged. No CUDA allocation or native RMSNorm is used.
"""

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import attention_v41 as attention
from rtp_llm.models_py.modules.dsv4.fp8.attention import PrefillMeta


def _slices(lengths):
    result, offset = [], 0
    for count in lengths:
        result.append(slice(offset, offset + count))
        offset += count
    return tuple(result)


def _guard_dynamic_rows(stack):
    original = torch.where

    def where(*args, **kwargs):
        if len(args) == 1 and not kwargs:
            raise AssertionError("One-argument torch.where synchronizes dynamic rows")
        return original(*args, **kwargs)

    stack.enter_context(patch.object(torch, "where", side_effect=where))
    stack.enter_context(
        patch.object(torch, "nonzero", side_effect=AssertionError("Dynamic nonzero"))
    )
    stack.enter_context(
        patch.object(
            torch.Tensor, "nonzero", side_effect=AssertionError("Tensor.nonzero")
        )
    )


def _bounds(positions, width, ratio):
    end = ((positions + 1).clamp_min(0) // ratio).clamp_max(width).int()
    return torch.zeros_like(end), end


class PrefillRequestRowsCPU(unittest.TestCase):
    def setUp(self):
        # Import native bindings normally, then prevent any lazy CUDA gate
        # from treating these CPU-only fixtures as GPU tests.
        guard = patch.object(torch.cuda, "is_available", return_value=False)
        guard.start()
        self.addCleanup(guard.stop)

    def test_cp_uses_padded_local_lengths_without_reading_device_metadata(self):
        forbidden = Mock()
        forbidden.detach.side_effect = AssertionError("Unexpected device readback")
        forbidden.cpu.side_effect = AssertionError("Unexpected device readback")
        forbidden.tolist.side_effect = AssertionError("Unexpected device readback")
        for lengths in ((4, 0, 6), (0, 0, 0), (2,)):
            with self.subTest(lengths=lengths):
                common = SimpleNamespace(
                    cp_on=True,
                    cp_ctx=SimpleNamespace(
                        chunk_lengths_per_req=lengths,
                        input_lengths_global_host=(7, 0, 11),
                    ),
                    batch_size=len(lengths),
                    seqlen=sum(lengths),
                    input_lengths=forbidden,
                    prefix_lengths=forbidden,
                    req_id_per_token=forbidden,
                )
                self.assertEqual(
                    attention._prefill_request_row_slices(common), _slices(lengths)
                )

    def test_non_cp_new_rows_exclude_prefix_lengths(self):
        for lengths, prefixes in (
            ([0, 3, 0, 5], [28672, 28672, 1024, 11]),
            ([0], [28672]),
            ([9], [0]),
        ):
            with self.subTest(lengths=lengths):
                common = SimpleNamespace(
                    cp_on=False,
                    cp_ctx=None,
                    batch_size=len(lengths),
                    seqlen=sum(lengths),
                    input_lengths=torch.tensor(lengths, dtype=torch.int32),
                    prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
                )
                self.assertEqual(
                    attention._prefill_request_row_slices(common), _slices(lengths)
                )

    def test_malformed_lengths_raise_instead_of_misassigning_requests(self):
        for cp_on in (False, True):
            for lengths, batch, total in (
                ((3, 4), 3, 7),
                ((3, -1), 2, 2),
                ((3, 4), 2, 8),
                ((0, 0), 2, 1),
            ):
                with self.subTest(cp=cp_on, lengths=lengths, batch=batch, total=total):
                    common = SimpleNamespace(
                        cp_on=cp_on,
                        cp_ctx=SimpleNamespace(chunk_lengths_per_req=lengths),
                        batch_size=batch,
                        seqlen=total,
                        input_lengths=torch.tensor(lengths, dtype=torch.int32),
                    )
                    with self.assertRaises(ValueError):
                        attention._prefill_request_row_slices(common)

    def test_missing_host_layout_keeps_legacy_fallback_but_single_is_known(self):
        for cp_on in (False, True):
            common = SimpleNamespace(
                cp_on=cp_on,
                cp_ctx=SimpleNamespace(chunk_lengths_per_req=None),
                batch_size=3,
                seqlen=7,
                input_lengths=None,
            )
            self.assertIsNone(attention._prefill_request_row_slices(common))
            common.batch_size = 1
            self.assertEqual(
                attention._prefill_request_row_slices(common), (slice(0, 7),)
            )

    def test_selection_rejects_noncontiguous_or_incomplete_supplied_slices(self):
        owner = SimpleNamespace(
            is_index_source=True,
            kv_source_layer_id=20,
            layer_id=20,
            index_topk=2,
            compress_ratio=1,
            _shared_attention={"global": {20: [(None, torch.zeros(5, 2))] * 2}},
        )
        for supplied in (
            (slice(0, 4),),
            (slice(0, 1), slice(2, 4)),
            (slice(0, 3), slice(2, 4)),
            (slice(0, 2), slice(2, 5)),
            (slice(0, 2), slice(2, 4, 2)),
        ):
            with self.subTest(slices=supplied):
                with self.assertRaises(ValueError):
                    attention.AttentionV41FP8._select_indices(
                        owner,
                        torch.zeros(4, 2),
                        torch.zeros(4, 2),
                        torch.arange(4),
                        torch.tensor([0, 0, 1, 1]),
                        request_row_slices=supplied,
                    )

    def test_builder_attaches_slices_and_restores_compression_ratio(self):
        common = PrefillMeta(
            seqlen=8,
            seqlen_full=32,
            rd=64,
            device=torch.device("cpu"),
            cp_ctx=SimpleNamespace(chunk_lengths_per_req=(2, 0, 6)),
            cp_on=True,
            freqs_cis=torch.empty(8, 32, dtype=torch.complex64),
            topk_idxs=torch.empty(8, 512, dtype=torch.int32),
            sp_int=28672,
            any_cont=True,
            row_seqlens_full=torch.tensor([32]),
            batch_size=3,
            input_lengths=torch.tensor([2, 0, 5]),  # last row has CP padding
        )
        owner = attention.AttentionV41FP8.__new__(attention.AttentionV41FP8)
        torch.nn.Module.__init__(owner)
        owner.compress_ratio = 2

        def base(*args, **kwargs):
            self.assertEqual(owner.compress_ratio, 0)
            self.assertIsNone(kwargs["reuse_common_meta"])
            return common

        with patch.object(
            attention.AttentionFP8, "_build_shared_prefill_meta", side_effect=base
        ) as parent:
            actual = owner._build_shared_prefill_meta(
                torch.empty(8, 1), 28672, reuse_common_meta=object()
            )
        parent.assert_called_once()
        self.assertEqual(actual.request_row_slices, _slices([2, 0, 6]))
        self.assertIs(actual.freqs_cis, common.freqs_cis)
        self.assertEqual(owner.compress_ratio, 2)
        self.assertIsNone(common.request_row_slices)

    def _dense_case(self, *, fast, publish, fused_topk, ratio):
        lengths, key_counts = [0, 3, 0, 5], [7, 9, 1, 13]
        rows = sum(lengths)
        requests = torch.repeat_interleave(torch.arange(4), torch.tensor(lengths))
        visible = torch.tensor([0, 1, 8, 2, 3, 7, 10, 12])
        positions = visible * ratio - 1
        qr = torch.arange(1, rows + 1).float()[:, None].expand(-1, 2).clone()
        keys = [
            (
                None,
                attention.prefill_indexer.PrefillIndexerKeys(
                    torch.zeros(n, 64, dtype=torch.int8),
                    torch.zeros(n, dtype=torch.int32),
                ),
            )
            for n in key_counts
        ]
        shared = {"global": {20: keys}}
        if not publish:
            shared["candidates"] = (torch.arange(rows) % 4).int()[:, None]
        attn = SimpleNamespace(
            is_index_source=True,
            index_source_layer_id=20 if publish else 26,
            kv_source_layer_id=20,
            layer_id=20 if publish else 26,
            index_topk=2,
            index_n_heads=1,
            index_head_dim=2,
            rope_head_dim=0,
            compress_ratio=ratio,
            index_wq=None,
            index_weights=torch.ones(1, 2),
            freqs_cis=torch.ones(32, 1, dtype=torch.complex64),
            _lin=lambda weight, value: value,
            v41_config=dict(
                candidate_source_layer_id=20,
                candidate_topk_blocks=1,
                candidate_block_size=2,
            ),
            _shared_attention=shared,
        )
        chunk_calls, destinations, bound_calls = [], [], []
        original_candidates = attention._apply_prefill_candidates

        def bounds(pos, width, compression):
            self.assertGreater(pos.numel(), 0)
            self.assertEqual(pos.untyped_storage().data_ptr(), positions.data_ptr())
            bound_calls.append(pos.tolist())
            return _bounds(pos, width, compression)

        def score(q, sf, k, k_sf, weights, vis, **kwargs):
            if fast:
                self.assertEqual(q.untyped_storage().data_ptr(), qr.data_ptr())
                self.assertIsNotNone(kwargs.get("bounds"))
                torch.testing.assert_close(kwargs["bounds"][1], vis)
            columns = torch.arange(k.shape[0])
            sign = (q[:, 0, 0].int() % 2 == 0).float() * 2 - 1
            return ((columns[None] + 1) * sign[:, None]).masked_fill(
                columns[None] >= vis[:, None], -torch.inf
            )

        def candidate_step(state, logits, vis, selected_rows, *args):
            if fast:
                self.assertIsInstance(selected_rows, slice)
            chunk_calls.append(selected_rows)
            return original_candidates(state, logits, vis, selected_rows, *args)

        def select(logits, vis, topk, *, out, bounds):
            destinations.append(out)
            if fast:
                self.assertIsNotNone(out)
            else:
                self.assertIsNone(out)
            if not fused_topk:
                return None
            values, result = logits.topk(topk, dim=-1)
            result = result.int().masked_fill(~values.isfinite(), -1)
            return result if out is None else out.copy_(result)

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"DSV41_FUSED_PREFILL_METADATA": "1"})
            )
            stack.enter_context(
                patch.object(attention, "rope_only", side_effect=lambda q, *a: q)
            )
            stack.enter_context(
                patch.object(
                    attention.indexer_q_fusion, "try_fused_indexer_q", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_indexer,
                    "quantize_indexer_q",
                    side_effect=lambda q: (q, torch.zeros(rows, 1)),
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_indexer, "logits_chunk_rows", return_value=2
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_indexer, "score_indexer_chunk", side_effect=score
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_metadata, "try_score_bounds", side_effect=bounds
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_topk, "try_select_tokens", side_effect=select
                )
            )
            stack.enter_context(
                patch.object(
                    attention, "_apply_prefill_candidates", side_effect=candidate_step
                )
            )
            if fast:
                _guard_dynamic_rows(stack)
            actual = attention.AttentionV41FP8._select_indices(
                attn,
                qr,
                qr,
                positions,
                requests,
                request_row_slices=_slices(lengths) if fast else None,
            )
        expected = torch.full((rows, 2), -1, dtype=torch.int32)
        for row, length in enumerate(visible.tolist()):
            allowed = list(range(min(length, key_counts[int(requests[row])])))
            if not publish:
                allowed = [col for col in allowed if col // 2 == row % 4]
            if (row + 1) % 2 == 0:
                allowed.reverse()
            chosen = allowed[:2]
            expected[row, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if publish:
            expected_candidates = torch.tensor(
                [[(n - 1) // 2 if n else -1] for n in visible.tolist()],
                dtype=torch.int32,
            )
            torch.testing.assert_close(shared["candidates"], expected_candidates)
        if fast:
            self.assertEqual(
                chunk_calls,
                [slice(0, 2), slice(2, 3), slice(3, 5), slice(5, 7), slice(7, 8)],
            )
            self.assertEqual(
                bound_calls, [positions[:3].tolist(), positions[3:].tolist()]
            )
            for out in destinations:
                self.assertEqual(out.untyped_storage().data_ptr(), actual.data_ptr())
        self.assertEqual(len(destinations), 5)
        return actual, shared["candidates"]

    def test_dense_source_slices_match_scalar_oracle_and_old_where_path(self):
        for ratio in (1, 2):
            for fused in (False, True):
                with self.subTest(ratio=ratio, fused_topk=fused):
                    old = self._dense_case(
                        fast=False, publish=True, fused_topk=fused, ratio=ratio
                    )
                    new = self._dense_case(
                        fast=True, publish=True, fused_topk=fused, ratio=ratio
                    )
                    for actual, expected in zip(new, old):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dense_consumer_slices_align_candidate_rows_and_output_views(self):
        for ratio in (1, 2):
            old = self._dense_case(
                fast=False, publish=False, fused_topk=True, ratio=ratio
            )
            new = self._dense_case(
                fast=True, publish=False, fused_topk=True, ratio=ratio
            )
            for actual, expected in zip(new, old):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sparse_slices_reuse_request_plans_and_write_output_views(self):
        lengths, key_counts = [3, 0, 4], [1537, 99, 1793]
        rows = sum(lengths)
        visible = torch.tensor([0, 3, 513, 1, 10, 777, 1201])
        requests = torch.repeat_interleave(torch.arange(3), torch.tensor(lengths))
        qr = torch.arange(1, rows + 1).float()[:, None]
        candidates = torch.stack(
            [torch.arange(128).roll(row * 7) for row in range(rows)]
        ).int()
        candidates[:, 2] = -1
        shared = {
            "global": {
                20: [
                    (
                        None,
                        attention.prefill_indexer.PrefillIndexerKeys(
                            torch.zeros(n, 64, dtype=torch.int8),
                            torch.zeros(n, dtype=torch.int32),
                        ),
                    )
                    for n in key_counts
                ]
            },
            "candidates": candidates,
        }
        attn = SimpleNamespace(
            is_index_source=True,
            index_source_layer_id=26,
            kv_source_layer_id=20,
            layer_id=26,
            index_topk=512,
            index_n_heads=32,
            index_head_dim=128,
            rope_head_dim=0,
            compress_ratio=1,
            index_wq=None,
            index_weights=torch.ones(32, 1),
            freqs_cis=torch.ones(2048, 1, dtype=torch.complex64),
            _lin=lambda weight, value: value.expand(-1, 32 * 128),
            v41_config=dict(
                candidate_source_layer_id=20,
                candidate_topk_blocks=128,
                candidate_block_size=8,
            ),
            _shared_attention=shared,
        )
        destinations, plans, probes = [], [], []

        def prepare(cand, vis, width, block_size):
            self.assertEqual(cand.untyped_storage().data_ptr(), candidates.data_ptr())
            self.assertGreater(cand.shape[0], 0)
            logical = torch.full((cand.shape[0], 1024), -1, dtype=torch.int32)
            ends = []
            for row, (blocks, n) in enumerate(zip(cand.tolist(), vis.tolist())):
                allowed = sorted(
                    {
                        b * 8 + j
                        for b in blocks
                        for j in range(8)
                        if b >= 0 and b * 8 + j < min(n, width)
                    }
                )
                logical[row, : len(allowed)] = torch.tensor(allowed, dtype=torch.int32)
                ends.append(len(allowed))
            plan = SimpleNamespace(
                logical=logical, end=torch.tensor(ends, dtype=torch.int32), nbytes=128
            )
            plans.append(plan)
            return plan

        def supported(q, sf, keys, weights, cand, pos, *args):
            self.assertEqual(q.shape[0], 1)
            self.assertEqual(cand.untyped_storage().data_ptr(), candidates.data_ptr())
            probes.append(len(keys))
            return True

        def score(q, sf, keys, weights, plan):
            sign = (q[:, 0, 0].int() % 2 == 0).float() * 2 - 1
            return (plan.logical.float() * sign[:, None]).masked_fill(
                plan.logical < 0, -torch.inf
            )

        def select(logits, end):
            values, selected = logits.topk(512, dim=-1)
            return selected.int().masked_fill(~values.isfinite(), -1)

        def remap(selected, plan, *, logits, out):
            self.assertIsNotNone(out)
            result = plan.logical.gather(1, selected.long().clamp_min(0)).masked_fill(
                selected < 0, -1
            )
            destinations.append(out)
            return out.copy_(result)

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {
                        "DSV41_FUSED_PREFILL_METADATA": "1",
                        "DSV41_SPARSE_PREFILL_PLAN_MAX_BYTES": "4096",
                    },
                )
            )
            stack.enter_context(
                patch.object(attention, "rope_only", side_effect=lambda q, *a: q)
            )
            stack.enter_context(
                patch.object(
                    attention.indexer_q_fusion, "try_fused_indexer_q", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_indexer,
                    "quantize_indexer_q",
                    side_effect=lambda q: (q, torch.zeros(rows, 32)),
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_metadata, "try_score_bounds", side_effect=_bounds
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_deepselect, "is_available", return_value=True
                )
            )
            stack.enter_context(
                patch.object(
                    attention.sparse_prefill_indexer,
                    "is_supported",
                    side_effect=supported,
                )
            )
            stack.enter_context(
                patch.object(attention.sparse_prefill_indexer, "MAX_CHUNK_ROWS", 2)
            )
            stack.enter_context(
                patch.object(
                    attention.sparse_prefill_indexer,
                    "prepare_plan",
                    side_effect=prepare,
                )
            )
            stack.enter_context(
                patch.object(
                    attention.sparse_prefill_indexer, "score", side_effect=score
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_deepselect,
                    "try_select_sparse_tokens",
                    side_effect=select,
                )
            )
            stack.enter_context(
                patch.object(
                    attention.sparse_prefill_indexer, "remap", side_effect=remap
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_indexer,
                    "score_indexer_chunk",
                    side_effect=AssertionError("Dense fallback"),
                )
            )
            _guard_dynamic_rows(stack)
            outputs = []
            for layer in (26, 32, 38, 39):
                attn.layer_id = layer
                outputs.append(
                    attention.AttentionV41FP8._select_indices(
                        attn,
                        qr,
                        qr,
                        visible - 1,
                        requests,
                        request_row_slices=_slices(lengths),
                    )
                )
        expected = torch.full((rows, 512), -1, dtype=torch.int32)
        for row, (blocks, n) in enumerate(zip(candidates.tolist(), visible.tolist())):
            allowed = sorted(
                {
                    b * 8 + j
                    for b in blocks
                    for j in range(8)
                    if b >= 0 and b * 8 + j < n
                },
                reverse=(row + 1) % 2 == 0,
            )[:512]
            expected[row, : len(allowed)] = torch.tensor(allowed, dtype=torch.int32)
        for actual in outputs:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(plans), 4)
        self.assertEqual(probes, [1537, 1793] * 4)
        self.assertEqual({key[0] for key in shared["prefill_sparse_plans"][1]}, {0, 2})
        for layer_index, output in enumerate(outputs):
            for view in destinations[layer_index * 4 : (layer_index + 1) * 4]:
                self.assertEqual(view.untyped_storage().data_ptr(), output.data_ptr())


if __name__ == "__main__":
    unittest.main()
