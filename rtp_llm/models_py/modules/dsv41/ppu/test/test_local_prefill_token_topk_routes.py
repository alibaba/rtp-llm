"""Phase and real selector wiring; the CPU double does not prove native math."""

import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.ppu import attention_layer as attention
from rtp_llm.models_py.modules.dsv41.ppu import prefill_token_topk as token
from rtp_llm.models_py.modules.dsv41.ppu.prefill import build_step_context
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_prefill_topk import reference


def _state():
    return NS(
        prefill=True, idle=False, cache=NS(sharded=False),
        attn=NS(is_cuda_graph=False),
        runtime=NS(model=NS(
            _is_speculative=False, _is_decode_role=False, _prefill_cp_size=1,
            parallelism_config=NS(tp_size=1),
        )),
    )


def _scores(positions, width):
    output = torch.full((positions.numel(), width), -2, dtype=torch.bfloat16)
    values = (torch.arange(512, dtype=torch.int16) + 0x3F00).view(torch.bfloat16)
    for row, position in enumerate(positions.tolist()):
        columns = torch.arange(512) + (position % 17) * 3
        output[row, columns] = values
    columns = torch.arange(width)
    return output.masked_fill(columns[None] >= (positions + 1)[:, None], -torch.inf)


def _unit(*, layer=2, candidate_source=-1):
    unit = attention.SharedPPUAttention.__new__(attention.SharedPPUAttention)
    nn.Module.__init__(unit)
    unit.geometry = NS(index_topk=512, index_n_heads=32, index_head_dim=128,
                       ratio=lambda source: 1)
    unit.layer_id, unit.kv_source_layer_id, unit.index_source_layer_id = layer, 0, layer
    unit.is_index_source = True
    unit.v41_config = dict(candidate_source_layer_id=candidate_source,
                          candidate_block_size=8, candidate_topk_blocks=4)
    positions = torch.tensor([990, 991, 992, 993])
    # Real request rows are deliberately interleaved; out[chunk] is a copy.
    requests = torch.tensor([1, 0, 1, 0])
    context = build_step_context(positions, requests, torch.tensor([0, 2, 4], dtype=torch.int32))
    unit._project_indexer = lambda x, qr, positions, shared, **kw: (
        torch.zeros(4, 32, 128), torch.ones(4, 32),
    )
    recorded = []

    def score(query, keys, weights, **kwargs):
        value = _scores(kwargs["context"].positions, keys.values.shape[0])
        recorded.append(value)
        return value

    unit.ops = NS(index_score=score, decode_kv=lambda packed, **kw: kw["out"].zero_())
    cache = NS(lengths=(1024, 1024), descriptors=lambda source: {"indexer_kv": None},
               read_source=lambda source, rid, global_entries: (
                   None, torch.zeros(global_entries, 68, dtype=torch.uint8),
               ))
    return unit, context, cache, recorded


def _run(*, local=False, prefill=True, layer=2, candidate_source=-1, shared=None, selector=None):
    unit, context, cache, recorded = _unit(layer=layer, candidate_source=candidate_source)
    shared = {} if shared is None else shared
    with patch.object(attention, "_quantized_indexer_query", side_effect=lambda q: NS(values=q, scales=None)):
        with patch.object(token, "try_select_tokens", side_effect=selector) as probe:
            output = unit._select(
                torch.zeros(4, 8), torch.zeros(4, 8), context, cache, shared,
                prefill=prefill, local_token_topk=local,
            )
    return output, shared, recorded, probe


def _canonical_double(logits, visible, **kwargs):
    if kwargs.get("disposable") is not True:
        raise AssertionError("the caller did not certify disposable private logits")
    selected = reference(logits, visible)
    selected = selected.masked_fill(selected < 0, logits.shape[1]).sort(-1).values
    return selected.masked_fill(selected == logits.shape[1], -1)


class LocalTokenTopKRoutes(unittest.TestCase):
    @torch.inference_mode()
    def test_phase_guard_excludes_nonlocal_or_observable_readers(self):
        with patch.dict(os.environ, {token.ENV: "1"}), patch.object(
            attention, "is_ppu", return_value=True,
        ), patch("torch.cuda.is_current_stream_capturing", return_value=False):
            guard = attention._local_prefill_token_topk_enabled
            self.assertTrue(guard(_state(), None))
            self.assertTrue(guard(_state(), NS(cp_size=1, kv_cache_sharded=False)))
            for flag in ("idle", "verify", "proposal", "is_graph", "is_graph_commit"):
                state = _state()
                setattr(state, flag, True)
                self.assertFalse(guard(state, None))
            for reason in ("decode", "cache", "sharded", "attn_graph", "spec", "role", "tp", "cp"):
                state = _state()
                model = state.runtime.model
                if reason == "decode": state.prefill = False
                if reason == "cache": state.cache = None
                if reason == "sharded": state.cache.sharded = True
                if reason == "attn_graph": state.attn.is_cuda_graph = True
                if reason == "spec": model._is_speculative = True
                if reason == "role": model._is_decode_role = True
                if reason == "tp": model.parallelism_config.tp_size = 8
                if reason == "cp": model._prefill_cp_size = 8
                self.assertFalse(guard(state, None), reason)
            self.assertFalse(guard(_state(), NS(cp_size=8, kv_cache_sharded=False)))
            self.assertFalse(guard(_state(), NS(cp_size=1, kv_cache_sharded=True)))
            self.assertFalse(guard(_state(), None, diagnostic={}))
            self.assertFalse(guard(_state(), None, packed_plan=object()))
            with patch("torch.cuda.is_current_stream_capturing", return_value=True):
                self.assertFalse(guard(_state(), None))
            with patch.object(attention, "is_ppu", return_value=False):
                self.assertFalse(guard(_state(), None))
            with torch.enable_grad():
                self.assertFalse(guard(_state(), None))
            os.environ.pop(token.ENV)
            self.assertFalse(guard(_state(), None))

    @torch.inference_mode()
    def test_native_results_scatter_into_interleaved_parent_rows(self):
        control, _, _, old = _run()
        candidate, _, _, native = _run(local=True, selector=_canonical_double)
        self.assertEqual(old.call_count, 0)
        self.assertEqual(native.call_count, 2)
        self.assertTrue(torch.equal(control.sort(-1).values, candidate.sort(-1).values))
        self.assertTrue((candidate >= 0).all())
        self.assertFalse(torch.equal(candidate[0], candidate[1]))

    @torch.inference_mode()
    def test_candidate_publication_precedes_disposable_selection(self):
        control, old_shared, _, _ = _run(candidate_source=2)
        shared = {}

        def select(logits, visible, **kwargs):
            rows = visible - 991
            self.assertTrue((shared["candidates"][rows] >= 0).all())
            result = _canonical_double(logits, visible, **kwargs)
            # An illegal later pooling consumer would now see different scores.
            logits.fill_(-torch.inf)
            return result

        candidate, _, _, native = _run(local=True, candidate_source=2, shared=shared, selector=select)
        self.assertEqual(native.call_count, 2)
        self.assertTrue(torch.equal(shared["candidates"], old_shared["candidates"]))
        self.assertTrue(torch.equal(control.sort(-1).values, candidate.sort(-1).values))

    @torch.inference_mode()
    def test_candidate_mask_precedes_native_selection(self):
        candidates = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7],
                                   [8, 9, 10, 11], [12, 13, 14, 15]], dtype=torch.int32)
        control, _, _, _ = _run(layer=3, candidate_source=2, shared={"candidates": candidates.clone()})
        observed = []

        def select(logits, visible, **kwargs):
            self.assertEqual(logits.isfinite().sum(-1).tolist(), [32, 32])
            observed.append(True)
            return _canonical_double(logits, visible, **kwargs)

        candidate, _, _, _ = _run(local=True, layer=3, candidate_source=2,
                                 shared={"candidates": candidates.clone()}, selector=select)
        self.assertEqual(len(observed), 2)
        self.assertTrue(torch.equal(control.sort(-1).values, candidate.sort(-1).values))

    @torch.inference_mode()
    def test_dependency_fallback_and_decode_keep_the_old_selection(self):
        control, _, _, _ = _run()
        fallback, _, _, rejected = _run(local=True, selector=lambda *a, **kw: None)
        self.assertEqual(rejected.call_count, 2)
        self.assertTrue(torch.equal(control, fallback))
        decoded, _, _, probe = _run(local=True, prefill=False,
                                    selector=Mock(side_effect=AssertionError("Decode admission")))
        self.assertEqual(probe.call_count, 0)
        self.assertTrue(torch.equal(control, decoded))

    @torch.inference_mode()
    def test_sparse_selection_receives_local_plan_end_and_scatter(self):
        class CudaGateTensor(torch.Tensor):
            # Only the branch predicate is doubled. All storage remains CPU;
            # scoring, native selection and remap below are explicit doubles.
            @property
            def is_cuda(self):
                return True

        def run(local):
            unit, context, cache, _ = _unit(layer=24, candidate_source=20)
            unit.kv_source_layer_id = 20
            unit.v41_config["candidate_topk_blocks"] = 2048
            cache.lengths = (32768, 32768)
            shared = {"candidates": torch.zeros(4, 2048, dtype=torch.int32)}
            plans, observed = [], []

            def plan(shared, key, *args, **kwargs):
                result = NS(sparse_columns=1024, end=torch.tensor([513, 400], dtype=torch.int32),
                            bias=(key[0] + 1) * 1000)
                plans.append(result)
                return result

            def score(q, scales, keys, weights, plan):
                return _scores(torch.tensor([990, 991]), plan.sparse_columns)

            def remap(selected, plan, **kwargs):
                valid = (selected >= 0) & (selected < plan.end[:, None])
                logits = kwargs.get("logits")
                if logits is not None:
                    values = logits.gather(1, selected.long().clamp(0, logits.shape[1] - 1))
                    valid &= values.isfinite()
                return torch.where(valid, selected + plan.bias, -1).int()

            def select(logits, visible, **kwargs):
                self.assertEqual(visible.data_ptr(), plans[-1].end.data_ptr())
                self.assertEqual(visible.tolist(), [513, 400])
                observed.append(True)
                selected = _canonical_double(logits, visible, **kwargs)
                self.assertTrue(((selected < 0) | (selected < visible[:, None])).all())
                return selected

            with patch.dict(os.environ, {attention.PREFILL_SPARSE_FP4_SCORE_ENV: "1"}), patch.object(
                attention, "is_ppu", return_value=True,
            ), patch.object(attention.sparse_prefill_indexer, "_get_deep_gemm", return_value=NS()), patch.object(
                attention, "_prefill_sparse_plan", side_effect=plan,
            ), patch.object(attention.sparse_prefill_indexer, "score", side_effect=score), patch.object(
                attention.sparse_prefill_indexer, "remap", side_effect=remap,
            ), patch.object(attention, "_quantized_indexer_query", side_effect=lambda q: NS(values=q, scales=None)), patch.object(
                token, "try_select_tokens", side_effect=select,
            ):
                output = unit._select(
                    torch.zeros(4, 8).as_subclass(CudaGateTensor), torch.zeros(4, 8),
                    context, cache, shared, prefill=True, local_token_topk=local,
                )
            return output, observed

        control, old = run(False)
        candidate, native = run(True)
        self.assertEqual(old, [])
        self.assertEqual(len(native), 2)
        self.assertTrue(torch.equal(control.sort(-1).values, candidate.sort(-1).values))


if __name__ == "__main__":
    unittest.main()
