"""Check scores, top-k visibility, and live Graph replay for prefill scoring."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.qsa.indexer_prefill_score import try_prefill_score
from rtp_llm.models_py.modules.qwen4_exp.qsa import indexer as indexer_module
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import Qwen4ExpQSAIndexer


class IndexerPrefillScoreTest(unittest.TestCase):
    def inputs(self, length=131, blocks=37, strided=False):
        torch.manual_seed(73)
        shape = (2, length, 4, 128)
        if strided:
            q = torch.randn(2, length, 8, 128, device="cuda", dtype=torch.bfloat16)[
                :, :, ::2
            ]
            k = torch.randn(2, blocks * 2, 128, device="cuda", dtype=torch.bfloat16)[
                :, ::2
            ]
        else:
            q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(2, blocks, 128, device="cuda", dtype=torch.bfloat16)
        return q, k

    def reference(self, q, k):
        return torch.relu(
            torch.matmul(q.float(), k.float().transpose(-1, -2).unsqueeze(1)).transpose(
                -1, -2
            )
        ).sum(-1) / math.sqrt(128)

    def test_ragged_strided_scores_match_fp64_reference(self):
        for strided in (False, True):
            q, k = self.inputs(strided=strided)
            actual = try_prefill_score(q, k)
            self.assertIsNotNone(actual)
            expected = torch.relu(
                torch.matmul(
                    q.double(), k.double().transpose(-1, -2).unsqueeze(1)
                ).transpose(-1, -2)
            ).sum(-1) / math.sqrt(128)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual.double(), expected, atol=5e-6, rtol=5e-6)

    def test_long_prefill_preserves_topk_selection_and_invisible_blocks(self):
        q, k = self.inputs(length=4093, blocks=1023)
        actual = try_prefill_score(q, k)
        expected = self.reference(q, k)
        complete = (torch.arange(q.shape[1], device=q.device) + 1) // 4
        visible = torch.arange(k.shape[1], device=q.device)[None] < complete[:, None]
        actual = actual.masked_fill(~visible[None], float("-inf"))
        expected = expected.masked_fill(~visible[None], float("-inf"))
        self.assertTrue(torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
        actual_top = actual.topk(512, dim=-1).indices.sort(-1).values
        expected_top = expected.topk(512, dim=-1).indices.sort(-1).values
        self.assertTrue(torch.equal(actual_top, expected_top))

    def test_graph_replay_refreshes_queries_and_keys(self):
        q, k = self.inputs(strided=True)
        try_prefill_score(q, k)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                try_prefill_score(q, k)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_prefill_score(q, k)
        for scale in (0.1, 0.5, 1.0):
            q.copy_(torch.randn_like(q) * scale)
            k.copy_(torch.randn_like(k) * scale)
            graph.replay()
            torch.testing.assert_close(
                actual, self.reference(q, k), atol=5e-6, rtol=5e-6
            )
        if torch.version.hip is None and torch.cuda.get_device_capability() == (12, 0):
            with torch.inference_mode():
                self._check_prefill_score_core()

    def _check_prefill_score_core(self):
        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpAttention
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.segmented_prefill import (
            GraphSegment,
            _qsa_prefill_model_supported,
            _qsa_prefill_request_supported,
            _QsaPrefillScoreCore,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(91)
        indexers = tuple(
            Qwen4ExpQSAIndexer(
                torch.randn(640, 8, device=device, dtype=torch.bfloat16),
                torch.randn(128, device=device, dtype=torch.bfloat16) * 0.02,
                torch.randn(128, device=device, dtype=torch.bfloat16) * 0.02,
                n_heads=4,
                kv_heads=1,
                head_dim=128,
                token_budget=2048,
                compress_ratio=4,
                norm_eps=1e-6,
            )
            for _ in range(2)
        )
        owner = SimpleNamespace(
            device=device,
            current=None,
            executing=True,
            recording=True,
            qsa_prefill_active=True,
            qsa_epoch=0,
            pool=torch.cuda.graph_pool_handle(),
            capture_stream=torch.cuda.Stream(device=device),
        )
        core = _QsaPrefillScoreCore(owner, indexers)
        for indexer in indexers:
            self.assertIsNone(indexer._prefill_score_core)
            indexer._prefill_score_core = core
            self.assertNotIn("forward_causal_from_projected", indexer.__dict__)
        contexts = {
            length: SimpleNamespace(
                capacity=length, live_tokens=length, qsa_regions={}, qsa_captured=set()
            )
            for length in (2048, 8192)
        }

        def inputs(length):
            return SimpleNamespace(
                input_ids=torch.zeros(length, dtype=torch.int32),
                input_hiddens=None,
                multimodal_inputs=SimpleNamespace(
                    multimodal_features=[],
                    mm_extra_input=[],
                    mm_features_spans=None,
                    mm_features_locs=None,
                ),
                embedding_inputs=SimpleNamespace(
                    text_tokens_mask=None, combo_tokens_type_ids=None
                ),
                attention_inputs={
                    "full": SimpleNamespace(
                        is_prefill=True,
                        is_target_verify=False,
                        is_cuda_graph=False,
                        is_s_padded=False,
                        context_parallel_info=None,
                        cache_store_inputs=None,
                        input_lengths=torch.tensor([length], dtype=torch.int32),
                        prefix_lengths=torch.tensor([0], dtype=torch.int32),
                    )
                },
            )

        def cat_count(spy):
            return sum(
                len(call.args[0]) == 1 and call.kwargs.get("dim") == 0
                for call in spy.call_args_list
            )

        def poison(context):
            for region in context.qsa_regions.values():
                for tensor in region.arguments[:2]:
                    tensor.fill_(float("nan"))
                region.arguments[2].fill_(-1)
                region.output.fill_(-913)

        def eager_region(region):
            region.output.copy_(region._qsa_function(*region.arguments))
            return region.output

        for length, seed in (
            (8192, 73),
            (2048, 97),
            (8192, 101),
            (2048, 0),
            (8192, 103),
        ):
            owner.current = context = contexts[length]
            owner.recording = not context.qsa_regions
            owner.qsa_epoch += 1
            owner.qsa_prefill_active = _qsa_prefill_request_supported(
                inputs(length), length
            )
            self.assertTrue(owner.qsa_prefill_active)
            torch.manual_seed(seed)
            hidden = torch.randn(length, 8, device=device, dtype=torch.bfloat16)
            if seed == 0:
                hidden.zero_()
            angle = torch.randn(1, length, 32, device=device).repeat(1, 1, 2)
            cosine, sine = angle.cos().bfloat16(), angle.sin().bfloat16()
            references = []
            for indexer in indexers:
                q, raw = indexer.project_raw(hidden)
                keys = indexer.pooled_block_keys_all(raw.unsqueeze(0), cosine, sine)
                attention = SimpleNamespace(
                    qsa_indexer=indexer, qsa_rope_config=None, is_mtp_draft=False
                )
                runtime = SimpleNamespace(
                    main_inputs=inputs(length).attention_inputs["full"],
                    host_values=lambda tensor: tensor.tolist(),
                    write_prefill_indexer_cache=lambda *args, **kwargs: (
                        [length],
                        cosine,
                        sine,
                        {"block_keys": keys},
                    ),
                )

                def call():
                    return Qwen4ExpAttention._qsa_selected_indices(
                        attention, hidden, runtime, projected=(q, raw)
                    )

                with (
                    patch.object(indexer, "_prefill_score_core", None),
                    patch.object(torch, "cat", wraps=torch.cat) as cat,
                ):
                    expected = call()
                self.assertEqual(cat_count(cat), 1)
                references.append(expected)
                poison(context)
                with patch.object(torch, "cat", wraps=torch.cat) as cat:
                    actual = call()
                self.assertEqual(cat_count(cat), 0)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # Stand in for a consumer which finishes before the next replay.
                actual.fill_(-317)
                poison(context)
                with patch.object(core, "_execute", side_effect=eager_region):
                    prepared = call()
                torch.testing.assert_close(prepared, expected, rtol=0, atol=0)
                prepared.fill_(-319)
                poison(context)
                torch.testing.assert_close(call(), expected, rtol=0, atol=0)
            if seed:
                self.assertFalse(torch.equal(*references))
            self.assertEqual(len(context.qsa_regions), 1)
            self.assertEqual(
                context.qsa_captured, {id(indexer) for indexer in indexers}
            )
            owner.recording = False
        self.assertNotEqual(
            next(iter(contexts[2048].qsa_regions.values())).output.data_ptr(),
            next(iter(contexts[8192].qsa_regions.values())).output.data_ptr(),
        )

        # A borrowed view is one-use and tied to the current owner epoch/context.
        selected = indexer.forward_causal_from_projected(
            indexer.project(hidden)[0].unsqueeze(0),
            raw.unsqueeze(0),
            cosine,
            sine,
            precomputed_block_keys=keys,
        ).squeeze(0)
        self.assertFalse(core.borrowed_selection(indexer, selected.t()))
        region = next(iter(context.qsa_regions.values()))
        core._borrowed = context, owner.qsa_epoch, indexer, region.output
        self.assertTrue(core.borrowed_selection(indexer, selected))
        self.assertFalse(core.borrowed_selection(indexer, selected))
        core._borrowed = context, owner.qsa_epoch - 1, indexer, region.output
        self.assertFalse(core.borrowed_selection(indexer, selected))
        core._borrowed = contexts[2048], owner.qsa_epoch, indexer, region.output
        self.assertFalse(core.borrowed_selection(indexer, selected))
        core._borrowed = context, owner.qsa_epoch, indexer, region.output
        self.assertFalse(core.borrowed_selection(indexers[0], selected))
        core._borrowed = context, owner.qsa_epoch, indexer, region.output
        self.assertFalse(core.borrowed_selection(indexer, selected.clone()))
        core.check_bindings()

        # No capture-time construction during a live request; original cat remains.
        owner.current = SimpleNamespace(
            capacity=length, live_tokens=length, qsa_regions={}, qsa_captured=set()
        )
        with (
            patch.object(
                GraphSegment, "__init__", side_effect=AssertionError("late capture")
            ),
            patch.object(torch, "cat", wraps=torch.cat) as cat,
        ):
            torch.testing.assert_close(call(), expected, rtol=0, atol=0)
        self.assertEqual(cat_count(cat), 1)
        self.assertFalse(owner.current.qsa_regions)

        pc = SimpleNamespace(
            role_type="PDFUSION",
            tp_size=2,
            ep_size=1,
            dp_size=1,
            pp_size=1,
            get_attn_tp_size=lambda: pc.tp_size,
            get_ffn_tp_size=lambda: pc.tp_size,
        )
        model = SimpleNamespace(
            _ordinary_decode_graph_plan=True,
            _capture_mtp_target_hidden=False,
            config=SimpleNamespace(is_mtp=False, _qwen4_full_fp8_nvfp4=True),
            parallelism_config=pc,
        )
        for tp_size in (2, 4, 8):
            with self.subTest(tp_size=tp_size), patch.object(pc, "tp_size", tp_size):
                self.assertTrue(_qsa_prefill_model_supported(model))
        for target, field, value in (
            (pc, "role_type", "PREFILL"),
            (pc, "role_type", "DECODE"),
            (pc, "tp_size", 1),
            (pc, "tp_size", 3),
            (pc, "tp_size", 16),
            (pc, "get_attn_tp_size", lambda: 4),
            (pc, "get_ffn_tp_size", lambda: 4),
            (pc, "ep_size", 2),
            (model, "_ordinary_decode_graph_plan", False),
            (model, "_capture_mtp_target_hidden", True),
            (model.config, "is_mtp", True),
        ):
            with patch.object(target, field, value):
                self.assertFalse(_qsa_prefill_model_supported(model))
        bad = inputs(length)
        meta = bad.attention_inputs["full"]
        for target, field, value in (
            (bad, "input_ids", torch.empty(0, dtype=torch.int32)),
            (bad, "input_hiddens", torch.ones(1)),
            (meta, "input_lengths", torch.tensor([length // 2] * 2, dtype=torch.int32)),
            (meta, "input_lengths", torch.tensor([length // 4] * 4, dtype=torch.int32)),
            (meta, "input_lengths", torch.tensor([length - 1, 1], dtype=torch.int32)),
            (meta, "prefix_lengths", torch.tensor([1], dtype=torch.int32)),
            (meta, "is_s_padded", True),
            (meta, "is_target_verify", True),
            (meta, "is_prefill", False),
            (meta, "is_cuda_graph", True),
            (meta, "context_parallel_info", object()),
            (meta, "cache_store_inputs", object()),
            (bad.multimodal_inputs, "multimodal_features", [torch.ones(1)]),
            (bad.multimodal_inputs, "mm_extra_input", [torch.ones(1)]),
            (bad.multimodal_inputs, "mm_features_spans", torch.ones(1)),
            (bad.multimodal_inputs, "mm_features_locs", torch.ones(1)),
            (bad.embedding_inputs, "text_tokens_mask", torch.ones(length)),
            (bad.embedding_inputs, "combo_tokens_type_ids", torch.ones(length)),
        ):
            with patch.object(target, field, value):
                self.assertFalse(_qsa_prefill_request_supported(bad, length))
        owner.current = context
        owner.qsa_prefill_active = False
        with patch.object(torch, "cat", wraps=torch.cat) as cat:
            torch.testing.assert_close(call(), expected, rtol=0, atol=0)
        self.assertEqual(cat_count(cat), 1)

    def test_automatic_selection_and_unsupported_inputs_fall_back(self):
        q, k = self.inputs()
        self.assertIsNotNone(try_prefill_score(q, k))
        self.assertIsNone(try_prefill_score(q[:, :1], k))
        self.assertIsNone(try_prefill_score(q.float(), k.float()))
        with patch("torch.version.hip", "hip"):
            self.assertIsNone(try_prefill_score(q, k))
        with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
            self.assertIsNone(try_prefill_score(q, k))

    def reference_selection(self, q, keys, past_length):
        batch, length = q.shape[:2]
        positions = past_length + torch.arange(length, device=q.device)
        complete = (positions + 1) // 4
        scores = self.reference(q, keys)
        visible = torch.arange(keys.shape[1], device=q.device)[None] < complete[:, None]
        values, blocks = scores.masked_fill(~visible[None], float("-inf")).topk(
            min(512, keys.shape[1]), dim=-1
        )
        tokens = blocks[..., None] * 4 + torch.arange(4, device=q.device)
        tokens = torch.where(values[..., None] > float("-inf"), tokens, -1)
        output = torch.full(
            (batch, length, 2051), -1, device=q.device, dtype=torch.int32
        )
        output[..., : tokens.shape[-2] * 4] = tokens.flatten(-2).int()
        tail = complete[:, None] * 4 + torch.arange(3, device=q.device)
        output[..., 2048:] = torch.where(tail <= positions[:, None], tail, -1).int()
        return output

    def test_bounded_fallback_keeps_absolute_positions_ties_and_tail(self):
        for past_length, tied in ((0, False), (5, False), (0, True), (5, True)):
            with self.subTest(past_length=past_length, tied=tied):
                q, _ = self.inputs(length=131, strided=True)
                raw = torch.randn(2, 131, 128, device=q.device, dtype=q.dtype)
                past = torch.randn(2, past_length, 128, device=q.device, dtype=q.dtype)
                if tied:
                    q.zero_()
                cosine = torch.ones(
                    2, past_length + 131, 64, device=q.device, dtype=q.dtype
                )
                sine = torch.zeros_like(cosine)
                indexer = Qwen4ExpQSAIndexer(
                    torch.zeros(640, 8, device=q.device, dtype=q.dtype),
                    torch.zeros(128, device=q.device, dtype=q.dtype),
                    torch.zeros(128, device=q.device, dtype=q.dtype),
                    n_heads=4,
                    kv_heads=1,
                    head_dim=128,
                    token_budget=2048,
                    compress_ratio=4,
                    norm_eps=1e-6,
                )
                keys = indexer.pooled_block_keys_all(
                    torch.cat((past, raw), 1), cosine, sine
                )
                expected = self.reference_selection(q, keys, past_length)
                bytes_per_query = 2 * keys.shape[1] * (2 * 4 + 2) * 4
                with (
                    patch(
                        "rtp_llm.models_py.kernels.cuda.qsa.indexer_prefill_score.try_prefill_score",
                        return_value=None,
                    ),
                    patch.object(
                        indexer_module,
                        "_MAX_SCORE_WORKSPACE_BYTES",
                        bytes_per_query * 7,
                    ),
                    patch.object(torch, "matmul", wraps=torch.matmul) as matmul,
                ):
                    actual = indexer.forward_causal_from_projected(
                        q,
                        raw,
                        cosine,
                        sine,
                        past if past_length else None,
                        precomputed_block_keys=keys if not past_length else None,
                    )
                self.assertGreater(matmul.call_count, 1)
                self.assertTrue(
                    all(call.args[0].shape[1] <= 7 for call in matmul.call_args_list)
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
