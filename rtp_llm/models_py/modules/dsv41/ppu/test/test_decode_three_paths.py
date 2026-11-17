"""Native score equivalence, live graph logits and Verify epilogue ownership."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant, graph_indexer
from rtp_llm.models_py.modules.dsv41.ppu.graph_metadata import finish_verify_outputs
from rtp_llm.models_py.modules.dsv41.ppu.graph_runtime import graph_lm_head
from rtp_llm.utils.model_weight import W


class DecodeThreePathsContracts(unittest.TestCase):
    def test_target_aux_wrapper_preserves_graph_logits_and_current_aux(self):
        from rtp_llm.models_py.model_desc.deepseek_v41_base_model import DeepSeekV4Model
        from rtp_llm.ops import PyModelOutputs

        hidden = torch.randn(4, 8)
        logits = torch.randn(4, 16)
        confidence = torch.randn(4, 1)
        original = PyModelOutputs(hidden)
        original.logits = logits
        original.dspark_confidence_hidden_logits = confidence
        owner = NS(
            _captures_aux_hidden=True,
            v4=NS(_mtp_hidden_buffer=torch.empty(4, 24)),
            _forward_impl=lambda *_: original,
        )
        inputs = NS(input_ids=torch.zeros(4, dtype=torch.int32))
        for _ in range(3):
            aux = torch.randn(4, 24)
            owner.get_mtp_target_hidden_states = lambda rows: aux
            result = DeepSeekV4Model.forward(owner, inputs)
            self.assertIs(result, original)
            self.assertEqual(result.logits.data_ptr(), logits.data_ptr())
            self.assertEqual(result.hidden_states.data_ptr(), hidden.data_ptr())
            self.assertEqual(
                result.dspark_confidence_hidden_logits.data_ptr(), confidence.data_ptr()
            )
            self.assertEqual(result.mtp_target_hidden_states.data_ptr(), aux.data_ptr())


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_GRAPH_DEVICE") == "1" and torch.cuda.is_available(),
    "requires explicit PPU device qualification",
)
class DecodeThreePathsDevice(unittest.TestCase):
    @torch.inference_mode()
    def test_batched_journal_graph_exception_restore_and_42_launches(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_packed_attention import (
            GraphPackedAttention,
        )

        adapter = GraphPackedAttention.__new__(GraphPackedAttention)
        keys = [(0, "swa_kv"), (0, "global_kv_1"), (1, "swa_kv"), (1, "global_kv_1")]
        widths = [528, 288, 528, 288]
        adapter.views = {
            key: torch.randint(
                256, (2, 128, 1, width), device="cuda", dtype=torch.uint8
            )
            for key, width in zip(keys, widths)
        }
        payloads = {
            key: torch.randint(256, (32, width), device="cuda", dtype=torch.uint8)
            for key, width in zip(keys, widths)
        }
        adapter.journal = NS(rows=32, payloads=payloads)
        pages = [adapter.views[key][0, :32, 0] for key in keys]
        backups = [torch.empty_like(page) for page in pages]
        adapter._tail_transaction_active = False
        adapter._tail_transaction = NS(
            pages=pages,
            backups=backups,
            page_ptrs=torch.tensor(
                [p.data_ptr() for p in pages], dtype=torch.int64, device="cuda"
            ),
            backup_ptrs=torch.tensor(
                [p.data_ptr() for p in backups], dtype=torch.int64, device="cuda"
            ),
            sizes=torch.tensor(
                [p.numel() for p in pages], dtype=torch.int32, device="cuda"
            ),
            max_size=max(p.numel() for p in pages),
        )
        observed = [torch.empty_like(page) for page in pages]

        def forward(observe=True):
            with adapter.verify_tail_transaction():
                for layer in range(40):
                    start = (layer % 2) * 2
                    with adapter._stage_verify_tail(keys[start : start + 2]):
                        if observe:
                            for i in (start, start + 1):
                                observed[i].copy_(pages[i])

        with patch.dict(os.environ, {"RTP_V41_PPU_DECODE_LAUNCH_REUSE": "1"}):
            before = {key: value.clone() for key, value in adapter.views.items()}
            with self.assertRaisesRegex(RuntimeError, "consumer failed"):
                with adapter.verify_tail_transaction():
                    with adapter._stage_verify_tail(keys[:2]):
                        raise RuntimeError("consumer failed")
            for key in keys:
                self.assertTrue(torch.equal(adapter.views[key], before[key]))
            self.assertFalse(adapter._tail_transaction_active)
            forward()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            for _ in range(3):
                for value in adapter.views.values():
                    value.random_(0, 256)
                for value in payloads.values():
                    value.random_(0, 256)
                before = {key: value.clone() for key, value in adapter.views.items()}
                graph.replay()
                torch.cuda.synchronize()
                for key, got in zip(keys, observed):
                    self.assertTrue(torch.equal(got, payloads[key]))
                    self.assertTrue(torch.equal(adapter.views[key], before[key]))
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                forward(observe=False)
                torch.cuda.synchronize()
            path = (
                Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
                / "batched-journal.trace.json"
            )
            profile.export_chrome_trace(str(path))
            kernels = [
                e["name"]
                for e in json.loads(path.read_text())["traceEvents"]
                if e.get("cat") == "kernel"
            ]
            self.assertEqual(len(kernels), 42, kernels)
            self.assertEqual(sum("copy_reserved_pages" in name for name in kernels), 2)

    @torch.inference_mode()
    def test_paged_prefix_matches_sparse_scores_topk_and_live_graph(self):
        device = torch.device("cuda", torch.cuda.current_device())
        _quant.prepare_e2m1_codebooks(device)
        rows, page, capacity = 32, 128, 164096
        pages = (capacity + page - 1) // page
        pool = torch.empty((pages + 1, page * 68), dtype=torch.uint8, device=device)
        physical = torch.arange(pages + 1, device=device)[:, None]
        token = torch.arange(page, device=device)[None, :]
        payload = torch.where((token + physical) % 3 == 0, 0x22, 0x44)
        scales = 126 + ((token + 2 * physical) % 3)
        pool[:, : page * 64].copy_(payload.repeat_interleave(64, dim=1))
        pool[:, page * 64 :].copy_(scales.repeat_interleave(4, dim=1))
        table = torch.arange(1, pages + 1, dtype=torch.int32, device=device).repeat(
            rows, 1
        )
        query = torch.randn((rows, 32, 128), dtype=torch.bfloat16, device=device) * 0.1
        weights = torch.randn((rows, 32), dtype=torch.bfloat16, device=device) * 0.1
        visible = torch.tensor(
            [0, 7, 512, 65535, 65536, 131072, 131073, capacity] * 4,
            dtype=torch.int64,
            device=device,
        )
        valid = torch.ones(rows, dtype=torch.bool, device=device)
        cache = NS(
            max_seq_len=capacity,
            tables={"indexer_kv": table},
            entries={"indexer_kv": page},
            spans={"indexer_kv": page},
            pool=lambda *_: pool,
        )

        def forward(enabled):
            with patch.dict(os.environ, {"RTP_V41_PPU_INDEXER_PAGED_PREFIX": enabled}):
                return list(
                    graph_indexer._indexer_score_tiles(
                        query,
                        weights,
                        cache=cache,
                        source_layer=0,
                        ratio=1,
                        visible=visible,
                        valid_mask=valid,
                        fused_tiles=True,
                        tile_metadata_cache={},
                    )
                )

        def check(actual):
            reference = forward("0")
            self.assertEqual(
                [t[0].shape for t in actual], [t[0].shape for t in reference]
            )
            for got, expected in zip(actual, reference):
                torch.testing.assert_close(got[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(got[1], expected[1], rtol=0, atol=0)
            got = graph_indexer._select_verify_topk(
                actual, visible, valid, 512, bf16_tail_provenance=True
            )
            expected = graph_indexer._select_verify_topk(
                reference, visible, valid, 512, bf16_tail_provenance=True
            )
            # Native TopK deliberately leaves tied IDs unordered. The fixture
            # has many equal BF16 keys: compare its selected score multiset,
            # uniqueness/domain, ordinal rows and legacy candidate publication.
            dense = torch.cat([t[0] for t in actual], dim=1)
            for picks in (got, expected):
                for row, length in enumerate(visible.tolist()):
                    live = picks[row][picks[row] >= 0].long()
                    self.assertEqual(live.unique().numel(), live.numel())
                    self.assertTrue((live < length).all().item())
                    if not valid[row].item():
                        self.assertEqual(live.numel(), 0)
                    elif length <= 512:
                        torch.testing.assert_close(
                            live, torch.arange(length, device=device), rtol=0, atol=0
                        )

            def values(picks):
                return (
                    torch.where(
                        picks >= 0,
                        dense.gather(1, picks.clamp_min(0).long()),
                        -torch.inf,
                    )
                    .sort(1)
                    .values
                )

            torch.testing.assert_close(values(got), values(expected), rtol=0, atol=0)
            torch.testing.assert_close(
                graph_indexer._candidate_tiles(actual, visible, 8, 512),
                graph_indexer._candidate_tiles(reference, visible, 8, 512),
                rtol=0,
                atol=0,
            )

        check(forward("1"))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = forward("1")
        for iteration in range(3):
            query.normal_(0, 0.1)
            weights.normal_(0, 0.1)
            visible.add_(1).clamp_max_(capacity)
            valid[::3] = iteration != 1
            table[5, 2] = 0 if iteration == 1 else 3
            graph.replay()
            check(actual)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def test_lm_head_graph_live_values_fp32_and_separate_bf16(self):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        for rows in (24, 32):
            hidden = (
                torch.randn((rows, 5120), device="cuda", dtype=torch.bfloat16) * 0.1
            )
            weight = (
                torch.randn((4096, 5120), device="cuda", dtype=torch.bfloat16) * 0.1
            )
            model = NS(
                _is_speculative=True,
                _is_decode_role=True,
                weight=NS(global_weights={W.lm_head: weight}),
            )
            with patch.dict(os.environ):
                os.environ.pop("RTP_V41_PPU_GRAPH_LM_HEAD", None)
                os.environ.pop("RTP_V41_PPU_DRAFT_LM_HEAD_BF16", None)
                default_logits = graph_lm_head(model, hidden, draft=True)
                self.assertEqual(default_logits.dtype, torch.float32)
                torch.testing.assert_close(
                    default_logits,
                    rtp_llm_ops.cublas_gemm_bf16_bf16_fp32(hidden, weight),
                    rtol=0,
                    atol=0,
                )
                os.environ["RTP_V41_PPU_GRAPH_LM_HEAD"] = "0"
                self.assertIsNone(graph_lm_head(model, hidden, draft=True))
            for bf16 in ("0", "1"):
                with patch.dict(
                    os.environ,
                    {
                        "RTP_V41_PPU_GRAPH_LM_HEAD": "1",
                        "RTP_V41_PPU_DRAFT_LM_HEAD_BF16": bf16,
                    },
                ):
                    graph_lm_head(model, hidden, draft=True)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = graph_lm_head(model, hidden, draft=True)
                    for _ in range(3):
                        hidden.normal_(0, 0.1)
                        graph.replay()
                        expected = (
                            torch.mm(hidden, weight.t())
                            if bf16 == "1"
                            else rtp_llm_ops.cublas_gemm_bf16_bf16_fp32(hidden, weight)
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        # Keep the qualified M=24/32 GEMM geometry: changing M
                        # selects another native reduction tile (FP32 rounding
                        # then differs). Still exercise permutation/duplicates.
                        selected = torch.arange(rows - 1, -1, -1, device="cuda")
                        selected[0] = selected[1]
                        selected_reference = (
                            torch.mm(hidden.index_select(0, selected), weight.t())
                            if bf16 == "1"
                            else rtp_llm_ops.cublas_gemm_bf16_bf16_fp32(
                                hidden.index_select(0, selected), weight
                            )
                        )
                        torch.testing.assert_close(
                            actual.index_select(0, selected),
                            selected_reference,
                            rtol=0,
                            atol=0,
                        )
                        owned = actual.float().clone()
                        saved = owned.clone()
                        hidden.zero_()
                        graph.replay()
                        torch.testing.assert_close(owned, saved, rtol=0, atol=0)
                    # Target remains FP32 even in the draft-BF16 experiment.
                    self.assertEqual(graph_lm_head(model, hidden).dtype, torch.float32)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def test_verify_epilogue_live_bytes_counters_and_one_launch(self):
        rows, width = 32, 5120
        hidden = torch.empty((rows, width), device="cuda", dtype=torch.bfloat16)
        aux = torch.empty((rows, 15360), device="cuda", dtype=torch.bfloat16)
        valid = torch.ones(rows, dtype=torch.bool, device="cuda")
        inputs = NS(
            input_ids=torch.arange(rows, device="cuda", dtype=torch.int32),
            engram_token_windows=torch.arange(rows * 4, device="cuda").reshape(rows, 4),
        )
        impl = NS(
            consumed_ids=torch.empty_like(inputs.input_ids),
            consumed_windows=torch.empty_like(inputs.engram_token_windows),
            cache=NS(
                completions=torch.tensor([2**32], device="cuda", dtype=torch.int64)
            ),
        )
        runtime = NS(
            layer_calls=torch.arange(40, device="cuda", dtype=torch.int64) + 2**32,
            moe_calls=torch.arange(40, device="cuda", dtype=torch.int64) + 2**33,
            engram_calls=torch.arange(40, device="cuda", dtype=torch.int64) + 2**34,
            _engram_layer_mask=torch.arange(40, device="cuda") % 3 == 0,
        )

        def forward():
            return finish_verify_outputs(
                hidden, aux, valid, inputs, impl, runtime, bulk_counters=True
            )

        self.assertTrue(forward())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.assertTrue(forward())
        for pattern in ("live", "mixed", "idle"):
            source, source_aux = torch.randn_like(hidden), torch.randn_like(aux)
            for tensor in (source, source_aux):
                tensor[:, 0] = float("nan")
                tensor[:, 1] = float("inf")
                tensor[:, 2] = -0.0
            hidden.copy_(source)
            aux.copy_(source_aux)
            valid.fill_(pattern != "idle")
            if pattern == "mixed":
                valid[::2] = False
            before = [
                t.clone()
                for t in (
                    runtime.layer_calls,
                    runtime.moe_calls,
                    runtime.engram_calls,
                    impl.cache.completions,
                )
            ]
            inputs.input_ids.add_(3)
            inputs.engram_token_windows.add_(5)
            graph.replay()
            expected = torch.where(valid[:, None], source, 0)
            expected_aux = torch.where(valid[:, None], source_aux, 0)
            self.assertTrue(
                torch.equal(hidden.view(torch.uint8), expected.view(torch.uint8))
            )
            self.assertTrue(
                torch.equal(aux.view(torch.uint8), expected_aux.view(torch.uint8))
            )
            torch.testing.assert_close(
                impl.consumed_ids, inputs.input_ids, rtol=0, atol=0
            )
            torch.testing.assert_close(
                impl.consumed_windows, inputs.engram_token_windows, rtol=0, atol=0
            )
            for got, old in zip((runtime.layer_calls, runtime.moe_calls), before):
                torch.testing.assert_close(got, old + 1, rtol=0, atol=0)
            torch.testing.assert_close(
                runtime.engram_calls,
                before[2] + runtime._engram_layer_mask.long(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                impl.cache.completions, before[3] + 1, rtol=0, atol=0
            )
        alias = NS(
            consumed_ids=inputs.input_ids,
            consumed_windows=impl.consumed_windows,
            cache=impl.cache,
        )
        self.assertFalse(
            finish_verify_outputs(
                hidden, aux, valid, inputs, alias, runtime, bulk_counters=True
            )
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            forward()
            torch.cuda.synchronize()
        path = (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
            / "verify-epilogue.trace.json"
        )
        profile.export_chrome_trace(str(path))
        kernels = [
            e["name"]
            for e in json.loads(path.read_text())["traceEvents"]
            if e.get("cat") == "kernel"
        ]
        self.assertEqual(len(kernels), 1, kernels)
