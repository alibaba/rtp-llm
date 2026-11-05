import gc
import json
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.base.common.fp8_linear import (
    make_projection,
    project_fp8_pair,
)
from rtp_llm.models_py.modules.hybrid.gated_residual import grouped_rms_norm
from rtp_llm.models_py.modules.qwen4_exp.ple.ple_conv_triton import try_ple_conv
from rtp_llm.models_py.modules.qwen4_exp.ple.ple_gate_triton import try_ple_gate
from rtp_llm.models_py.modules.qwen4_exp.test.support.ple_gate import (
    PLEProjectionFixture,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PleGateTest(PLEProjectionFixture, unittest.TestCase):
    def tensors(self, rows):
        torch.manual_seed(7300 + rows)
        shapes = [(1, rows, 10240), (1, rows, 2560), (1, rows, 10240)]
        return [torch.randn(s, device="cuda", dtype=torch.bfloat16) for s in shapes] + [
            torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07
            for _ in range(3)
        ]

    def reference(self, key, value, hidden, gk, gq, gc):
        kn = grouped_rms_norm(key, gk, 2560, 1e-6).unflatten(-1, (4, 2560))
        qn = grouped_rms_norm(hidden, gq, 2560, 1e-6).unflatten(-1, (4, 2560))
        gate = (kn * qn).sum(-1, keepdim=True) / math.sqrt(2560)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
        return gated, grouped_rms_norm(gated, gc, 2560, 1e-6)

    def test_all_materialization_boundaries(self):
        for rows in (1, 4, 16, 127, 128, 257, 4096):
            tensors = self.tensors(rows)
            for actual, expected in zip(
                try_ple_gate(*tensors, 1e-6), self.reference(*tensors)
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_batched_page_quant_and_projected_gate_are_exact(self):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            scaled_fp8_per_tensor_quant,
        )
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple_page_projection import (
            project_pages_pair,
            quantize_pages,
        )

        layer = self.layer()
        for rows in (128, 129, 257, 4093, 4095, 4096, 12287, 12288):
            torch.manual_seed(7820 + rows)
            x = torch.randn(1, rows, 2560, device="cuda", dtype=torch.bfloat16)
            x[:, : min(rows, 128)] *= 0.125
            quantized, scales = quantize_pages(
                x.squeeze(0), layer.key_projection.scale, layer.value_projection.scale
            )
            keys, values = [], []
            for start in range(0, rows, 128):
                page = x[:, start : start + 128].contiguous()
                padded = torch.empty(
                    ((page.shape[1] + 15) // 16 * 16, 2560),
                    device="cuda",
                    dtype=torch.float8_e4m3fn,
                )
                q, scale = scaled_fp8_per_tensor_quant(page.squeeze(0), output=padded)
                torch.testing.assert_close(
                    quantized[start : start + q.shape[0]].view(torch.uint8),
                    q.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    scales[0][start // 128 : start // 128 + 1], scale, rtol=0, atol=0
                )
                keys.append(layer.key_projection(page))
                values.append(layer.value_projection(page))
            key, value, sf = project_pages_pair(
                x, layer.key_projection, layer.value_projection
            )
            hidden = torch.randn(1, rows, 10240, device="cuda", dtype=torch.bfloat16)
            expected = try_ple_gate(
                torch.cat(keys, 1),
                torch.cat(values, 1),
                hidden,
                layer.norm_key,
                layer.norm_query,
                layer.norm_conv,
                layer.norm_eps,
            )
            actual = try_ple_gate(
                key,
                value,
                hidden,
                layer.norm_key,
                layer.norm_query,
                layer.norm_conv,
                layer.norm_eps,
                projection_scales=sf,
            )
            for candidate, reference in zip(actual, expected):
                torch.testing.assert_close(candidate, reference, rtol=0, atol=0)

    def test_shared_page_quant_keeps_both_projection_scales(self):
        for rows in (1, 17, 128):
            x = torch.randn(1, rows, 2560, device="cuda", dtype=torch.bfloat16)
            projections = [
                make_projection(
                    torch.randn(n, 2560, device="cuda").to(torch.float8_e4m3fn),
                    torch.tensor([scale], device="cuda"),
                )
                for n, scale in ((10240, 0.03), (2560, 0.07))
            ]
            expected = [p(x) for p in projections]
            actual = project_fp8_pair(x, *projections)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_batched_graph_conv_keeps_prefix_and_candidate_states(self):
        for batch, queries in ((1, 1), (1, 4), (4, 4), (8, 4)):
            key, value, hidden, gk, gq, gc = self.tensors(batch * queries)
            key = key.reshape(batch, queries, 10240)
            value = value.reshape(batch, queries, 2560)
            hidden = hidden.reshape(batch, queries, 10240)
            initial = torch.randn(batch, 9, 10240, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(10240, 1, 4, device="cuda", dtype=torch.bfloat16) * 0.1

            def call():
                gated, normed = try_ple_gate(key, value, hidden, gk, gq, gc, 1e-6)
                return try_ple_conv(gated, normed, initial, weight, 3), normed

            for _ in range(3):
                call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual, candidate = call()
            for seed in (7331, 7332):
                torch.manual_seed(seed)
                key.normal_()
                value.normal_()
                hidden.normal_()
                initial.normal_()
                before = initial.clone()
                gated, normed = self.reference(key, value, hidden, gk, gq, gc)
                expected = gated + F.silu(
                    F.conv1d(
                        torch.cat((initial, normed), 1).transpose(1, 2),
                        weight,
                        groups=10240,
                        dilation=3,
                    )
                ).transpose(1, 2)
                graph.replay()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(candidate, normed, rtol=0, atol=0)
                torch.testing.assert_close(initial, before, rtol=0, atol=0)

    def test_all_page_snapshots_tail_and_prefix(self):
        layer = self.layer()
        for length in (1, 7, 127, 128, 129, 257, 4096):
            hidden = torch.randn(1, length, 10240, device="cuda", dtype=torch.bfloat16)
            embeddings = torch.randn(
                1, length, 2560, device="cuda", dtype=torch.bfloat16
            )
            tokens = torch.zeros(1, length + 2, device="cuda", dtype=torch.int64)
            for initial in (
                None,
                torch.randn(1, 9, 10240, device="cuda", dtype=torch.bfloat16),
            ):
                expected, states = self.serial(
                    layer, hidden, tokens, embeddings, initial
                )
                actual, snapshots = layer.prefill_page_snapshots(
                    hidden, tokens, 128, initial, embeddings=embeddings
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(len(states), len(snapshots))
                for a, b in zip(snapshots, states):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PleSourceBf16BatchRouteTest(unittest.TestCase):
    """The real BF16 producer contract, independent of the FP8 fixture above."""

    @classmethod
    def setUpClass(cls):
        if torch.version.hip is not None or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("The candidate is restricted to SM120")

    def layer(self):
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpPLELayer

        torch.manual_seed(81420)
        layer = Qwen4ExpPLELayer(
            SimpleNamespace(ngram_size=3, context_len=2),
            torch.randn(10240, 2560, device="cuda", dtype=torch.bfloat16) * 0.02,
            torch.randn(2560, 2560, device="cuda", dtype=torch.bfloat16) * 0.02,
            torch.randn(10240, 1, 4, device="cuda", dtype=torch.bfloat16) * 0.1,
            *[
                torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07
                for _ in range(3)
            ],
            hc_mult=4,
            hidden_size=2560,
            conv_kernel_size=4,
            norm_eps=1e-6,
        )
        self.assertIsNone(layer.key_projection)
        self.assertIsNone(layer.value_projection)
        return layer

    def tensor_formula_reference(self, layer, hidden, embeddings, initial):
        # Existing GPU grouped_rms_norm helper plus unfused BF16 tensor formula.
        # This is neither a CPU nor an FP64 oracle, and it is not the old fused gate.
        state = initial
        stage = {
            k: []
            for k in (
                "key",
                "value",
                "gated",
                "normed",
                "conv",
                "silu",
                "inner",
                "outer",
                "states",
            )
        }
        for start in range(0, hidden.shape[1], 128):
            h = hidden[:, start : start + 128]
            page = embeddings[:, start : start + 128].contiguous()
            key = F.linear(page, layer.key_proj)
            value = F.linear(page, layer.value_proj)
            kn = grouped_rms_norm(key, layer.norm_key, 2560, 1e-6).unflatten(
                -1, (4, 2560)
            )
            qn = grouped_rms_norm(h, layer.norm_query, 2560, 1e-6).unflatten(
                -1, (4, 2560)
            )
            gate = (kn * qn).sum(-1, keepdim=True) / math.sqrt(2560)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
            normed = grouped_rms_norm(gated, layer.norm_conv, 2560, 1e-6)
            history = (
                F.pad(normed.transpose(1, 2), (9, 0)).transpose(1, 2)
                if state is None
                else torch.cat((state, normed), 1)
            )
            conv_bf16 = F.conv1d(
                history.transpose(1, 2),
                layer.conv_weight,
                groups=10240,
                dilation=3,
            )
            self.assertEqual(conv_bf16.dtype, torch.bfloat16)
            silu_bf16 = F.silu(conv_bf16)
            self.assertEqual(silu_bf16.dtype, torch.bfloat16)
            inner_bf16 = gated + silu_bf16.transpose(1, 2)
            outer_bf16 = h + inner_bf16
            state = history[:, -9:]
            for name, tensor in (
                ("key", key),
                ("value", value),
                ("gated", gated),
                ("normed", normed),
                ("conv", conv_bf16),
                ("silu", silu_bf16),
                ("inner", inner_bf16),
                ("outer", outer_bf16),
                ("states", state),
            ):
                stage[name].append(tensor)
        return stage

    def old_serial_reference(self, layer, hidden, embeddings, initial, tokens):
        """Capture the actual feature=0 production page route, not a reimplementation."""
        stage = {k: [] for k in ("key", "value", "gated", "normed", "conv", "silu")}
        linear, conv1d, silu = F.linear, F.conv1d, F.silu
        gate_call = try_ple_gate
        projection_calls = []

        def record_linear(page, weight, *args, **kwargs):
            which = "key" if len(projection_calls) % 2 == 0 else "value"
            actual = linear(page, weight, *args, **kwargs)
            stage[which].append(actual)
            projection_calls.append(which)
            return actual

        def record_gate(*args, **kwargs):
            self.assertEqual(args[0].shape, (1, 128, 10240))
            actual = gate_call(*args, **kwargs)
            self.assertIsNotNone(actual)
            stage["gated"].append(actual[0])
            stage["normed"].append(actual[1])
            return actual

        def record_conv(*args, **kwargs):
            actual = conv1d(*args, **kwargs)
            self.assertEqual(actual.dtype, torch.bfloat16)
            stage["conv"].append(actual)
            return actual

        def record_silu(*args, **kwargs):
            actual = silu(*args, **kwargs)
            self.assertEqual(actual.dtype, torch.bfloat16)
            stage["silu"].append(actual)
            return actual

        with patch.dict(os.environ, {"RTP_QWEN4_BF16_PLE_BATCH": "0"}), patch(
            "torch.nn.functional.linear", side_effect=record_linear
        ), patch(
            "rtp_llm.models_py.modules.qwen4_exp.ple.ple_gate_triton.try_ple_gate",
            side_effect=record_gate,
        ), patch(
            "torch.nn.functional.conv1d", side_effect=record_conv
        ), patch(
            "torch.nn.functional.silu", side_effect=record_silu
        ):
            inner, states = layer.prefill_page_snapshots(
                hidden,
                tokens,
                128,
                initial,
                embeddings=embeddings,
                add_residual=False,
            )
        self.assertEqual(projection_calls, ["key", "value"] * 64)
        self.assertEqual(len(stage["gated"]), 64)
        self.assertEqual(len(stage["conv"]), 64)
        self.assertEqual(len(stage["silu"]), 64)
        with patch.dict(os.environ, {"RTP_QWEN4_BF16_PLE_BATCH": "0"}):
            outer, outer_states = layer.prefill_page_snapshots(
                hidden,
                tokens,
                128,
                initial,
                embeddings=embeddings,
                add_residual=True,
            )
        self.assert_exact(outer, hidden + inner)
        for state, outer_state in zip(states, outer_states):
            self.assert_exact(state, outer_state)
        stage.update(inner=[inner], outer=[outer], states=states)
        return stage

    def stage_tensor(self, stage, name):
        return torch.cat(
            stage[name], 2 if name in ("conv", "silu") else 0 if name == "states" else 1
        )

    def tensor_formula_diagnostics(self, actual, formula, names):
        result = {}
        for name in names:
            left, right = self.stage_tensor(actual, name), self.stage_tensor(
                formula, name
            )
            delta = (left.float() - right.float()).abs()
            result[name] = {
                "mismatched": int(torch.count_nonzero(left != right).item()),
                "numel": left.numel(),
                "max_abs_fp32_diagnostic": float(delta.max().item()),
                "max_rel_fp32_diagnostic": float(
                    (delta / right.float().abs().clamp_min(1e-30)).max().item()
                ),
            }
        return result

    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def assert_page_cache(self, snapshots, expected, prefix, context):
        # The model-private caller writes BF16 PLE state and int64 n-gram context.
        # These are distinct from the attention group's FP8 KV payload/scales.
        count = len(snapshots)
        pages = prefix // 128 + torch.arange(count, device="cuda")
        physical = 2 + (torch.arange(count, device="cuda") * 17) % 67
        table = torch.zeros((1, 129), device="cuda", dtype=torch.int32)
        table[0, pages] = physical.to(torch.int32)
        blocks = table[0, pages].long()
        actual_pool = torch.full(
            (73, 9, 10240), -3, device="cuda", dtype=torch.bfloat16
        )
        expected_pool = actual_pool.clone()
        actual_pool.index_copy_(0, blocks, torch.cat(snapshots, 0))
        expected_pool.index_copy_(0, physical, torch.cat(expected, 0))
        self.assert_exact(actual_pool, expected_pool)
        actual_context = torch.full((73, 2), -19, device="cuda", dtype=torch.int64)
        expected_context = actual_context.clone()
        values = torch.stack(
            [context[0, end : end + 2] for end in range(128, 8193, 128)]
        )
        actual_context.index_copy_(0, blocks, values)
        expected_context.index_copy_(0, physical, values)
        self.assert_exact(actual_context, expected_context)
        self.assertEqual(
            int(torch.count_nonzero((actual_pool + 3).abs().sum((1, 2))).item()), count
        )

    def test_real_bf16_two_chunks_all_stages_and_every_page(self):
        layer = self.layer()
        initial = None
        context_ids = torch.tensor([[7, 11]], device="cuda", dtype=torch.int64)
        linear, gate_call, conv_call = F.linear, try_ple_gate, try_ple_conv
        for chunk in range(2):
            torch.manual_seed(81421 + chunk)
            hidden = torch.randn(1, 8192, 10240, device="cuda", dtype=torch.bfloat16)
            embeddings = torch.randn(1, 8192, 2560, device="cuda", dtype=torch.bfloat16)
            tokens = torch.cat(
                (context_ids, torch.arange(8192, device="cuda").reshape(1, -1)), 1
            )
            formula = self.tensor_formula_reference(layer, hidden, embeddings, initial)
            old = self.old_serial_reference(layer, hidden, embeddings, initial, tokens)
            old_diagnostics = self.tensor_formula_diagnostics(
                old,
                formula,
                (
                    "key",
                    "value",
                    "gated",
                    "normed",
                    "conv",
                    "silu",
                    "inner",
                    "outer",
                    "states",
                ),
            )
            self.assertEqual(old_diagnostics["key"]["mismatched"], 0)
            self.assertEqual(old_diagnostics["value"]["mismatched"], 0)
            if chunk == 0:
                # Reproduce A's oracle discrepancy on the unchanged OLD fused page route.
                self.assertGreater(old_diagnostics["gated"]["mismatched"], 0)
            before = [
                x.clone()
                for x in (
                    hidden,
                    embeddings,
                    layer.key_proj,
                    layer.value_proj,
                    layer.conv_weight,
                )
            ]
            initial_before = None if initial is None else initial.clone()
            calls, candidate = [], {}

            def checked_linear(page, weight, *args, **kwargs):
                index = len(calls)
                which = "key" if index % 2 == 0 else "value"
                self.assertEqual(tuple(page.shape), (1, 128, 2560))
                self.assertEqual(
                    page.stride(),
                    embeddings[:, (index // 2) * 128 : (index // 2 + 1) * 128]
                    .contiguous()
                    .stride(),
                )
                self.assertTrue(page.is_contiguous())
                self.assertEqual(
                    weight.data_ptr(),
                    (layer.key_proj if which == "key" else layer.value_proj).data_ptr(),
                )
                actual = linear(page, weight, *args, **kwargs)
                self.assert_exact(actual, old[which][index // 2])
                calls.append(which)
                return actual

            def checked_gate(*args, **kwargs):
                self.assertEqual(args[0].shape, (1, 8192, 10240))
                actual = gate_call(*args, **kwargs)
                self.assertIsNotNone(actual)
                for name, tensor in zip(("gated", "normed"), actual):
                    self.assert_exact(tensor, self.stage_tensor(old, name))
                    candidate[name] = [tensor]
                return actual

            def checked_conv(gated, normed, state, weight, dilation, *, residual=None):
                self.assertEqual(dilation, 3)
                self.assertEqual(weight.shape, (10240, 1, 4))
                inner = conv_call(gated, normed, state, weight, dilation)
                self.assert_exact(inner, self.stage_tensor(old, "inner"))
                candidate["inner"] = [inner]
                actual = conv_call(
                    gated, normed, state, weight, dilation, residual=residual
                )
                self.assert_exact(
                    actual,
                    self.stage_tensor(
                        old, "outer" if residual is not None else "inner"
                    ),
                )
                return actual

            with patch.dict(os.environ, {"RTP_QWEN4_BF16_PLE_BATCH": "1"}), patch(
                "torch.nn.functional.linear", side_effect=checked_linear
            ), patch(
                "rtp_llm.models_py.modules.qwen4_exp.ple.ple_gate_triton.try_ple_gate",
                side_effect=checked_gate,
            ), patch(
                "rtp_llm.models_py.modules.qwen4_exp.ple.ple_conv_triton.try_ple_conv",
                side_effect=checked_conv,
            ):
                output, snapshots = layer.prefill_page_snapshots(
                    hidden,
                    tokens,
                    128,
                    initial,
                    embeddings=embeddings,
                    add_residual=True,
                )
            self.assertEqual(calls, ["key", "value"] * 64)
            self.assertEqual(len(snapshots), 64)
            self.assert_exact(output, self.stage_tensor(old, "outer"))
            for actual, expected in zip(snapshots, old["states"]):
                self.assertEqual(actual.shape, (1, 9, 10240))
                self.assert_exact(actual, expected)
            candidate.update(outer=[output], states=snapshots)
            names = ("gated", "normed", "inner", "outer", "states")
            new_diagnostics = self.tensor_formula_diagnostics(candidate, formula, names)
            self.assertEqual(
                new_diagnostics, {name: old_diagnostics[name] for name in names}
            )
            print(
                "PLE_BF16_BATCH_ORACLE_DIAGNOSTIC="
                + json.dumps(
                    {
                        "chunk": chunk,
                        "prefix_tokens": chunk * 8192,
                        "formula_contract": "existing GPU grouped_rms_norm helper + BF16 tensor gate/native conv formula; not CPU/FP64",
                        "acceptance_oracle": "feature=0 actual OLD serial page route, same fused try_ple_gate and native conv",
                        "old_vs_formula": old_diagnostics,
                        "candidate_vs_formula": new_diagnostics,
                        "candidate_vs_old": "BITEXACT all projections/gated/normed/inner/outer/all page states",
                    },
                    sort_keys=True,
                )
            )
            for actual, expected in zip(
                (
                    hidden,
                    embeddings,
                    layer.key_proj,
                    layer.value_proj,
                    layer.conv_weight,
                ),
                before,
            ):
                self.assert_exact(actual, expected)
                self.assertNotEqual(
                    output.untyped_storage().data_ptr(),
                    actual.untyped_storage().data_ptr(),
                )
            if initial is not None:
                self.assert_exact(initial, initial_before)
                self.assertNotEqual(
                    output.untyped_storage().data_ptr(),
                    initial.untyped_storage().data_ptr(),
                )
            self.assert_page_cache(snapshots, old["states"], chunk * 8192, tokens)
            initial = snapshots[-1].clone()
            context_ids = tokens[:, -2:].clone()
            del formula, old, candidate, before, output, snapshots, hidden, embeddings

    @unittest.skipUnless(
        os.environ.get("RTP_QWEN4_PLE_BATCH_BENCH", "0") == "1",
        "Optional unmocked CUDA-event/allocator receipt; requires the GPU lock",
    )
    def test_unmocked_component_timing_and_allocator_receipt(self):
        gc.collect()
        torch.cuda.current_stream().synchronize()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            layer = self.layer()
            torch.manual_seed(81500)
            hidden = torch.randn(1, 8192, 10240, device="cuda", dtype=torch.bfloat16)
            embedding = torch.randn(1, 8192, 2560, device="cuda", dtype=torch.bfloat16)
            initial = torch.randn(1, 9, 10240, device="cuda", dtype=torch.bfloat16)
            tokens = torch.zeros((1, 8194), device="cuda", dtype=torch.int64)
            # Strong local references keep every persistent fixture allocation alive.
            persistent = (layer, hidden, embedding, initial, tokens)
            receipt = {
                "scope": "unmocked complete PLE component; synthetic source-BF16; not E2E",
                "shape": [1, 8192, 10240],
                "page_size": 128,
                "state_dtype": "bfloat16",
                "ngram_context_dtype": "int64",
                "warmup_runs": 3,
                "measure_runs": 7,
                "variant_order": ["0", "1"],
                "gc_initially_enabled": gc_was_enabled,
                "gc_disabled_during_warmup_measurement": True,
                "gc_explicit_before_each_sample": True,
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                "variants": {},
            }
            for enabled in ("0", "1"):
                with patch.dict(os.environ, {"RTP_QWEN4_BF16_PLE_BATCH": enabled}):
                    gc.collect()
                    torch.cuda.current_stream().synchronize()
                    for _ in range(3):
                        output, snapshots = layer.prefill_page_snapshots(
                            hidden,
                            tokens,
                            128,
                            initial,
                            embeddings=embedding,
                            add_residual=True,
                        )
                        del output, snapshots
                    torch.cuda.current_stream().synchronize()
                    samples = []
                    for _ in range(7):
                        gc.collect()
                        torch.cuda.current_stream().synchronize()
                        before_allocated = torch.cuda.memory_allocated()
                        before_reserved = torch.cuda.memory_reserved()
                        torch.cuda.reset_peak_memory_stats()
                        start, end = torch.cuda.Event(
                            enable_timing=True
                        ), torch.cuda.Event(enable_timing=True)
                        start.record()
                        output, snapshots = layer.prefill_page_snapshots(
                            hidden,
                            tokens,
                            128,
                            initial,
                            embeddings=embedding,
                            add_residual=True,
                        )
                        end.record()
                        end.synchronize()
                        after_return_allocated = torch.cuda.memory_allocated()
                        snapshot_storages = {
                            s.untyped_storage().data_ptr(): s.untyped_storage().nbytes()
                            for s in snapshots
                        }
                        sample = {
                            "cuda_event_ms": start.elapsed_time(end),
                            "allocated_after_return_bytes": after_return_allocated,
                            "retained_allocated_delta_bytes": after_return_allocated
                            - before_allocated,
                            "snapshot_payload_bytes": sum(
                                s.numel() * s.element_size() for s in snapshots
                            ),
                            "snapshot_unique_storage_bytes": sum(
                                snapshot_storages.values()
                            ),
                            "snapshot_unique_storage_count": len(snapshot_storages),
                            "allocated_before_bytes": before_allocated,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_allocated_delta_bytes": torch.cuda.max_memory_allocated()
                            - before_allocated,
                            "reserved_before_bytes": before_reserved,
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        }
                        self.assertEqual(len(snapshots), 64)
                        del output, snapshots
                        torch.cuda.current_stream().synchronize()
                        after_delete_allocated = torch.cuda.memory_allocated()
                        sample.update(
                            allocated_after_delete_bytes=after_delete_allocated,
                            allocated_after_delete_delta_bytes=after_delete_allocated
                            - before_allocated,
                            memory_sample_valid=(
                                after_return_allocated >= before_allocated
                                and after_delete_allocated == before_allocated
                            ),
                        )
                        samples.append(sample)
                    receipt["variants"][enabled] = samples
            self.assertEqual(len(persistent), 5)
            receipt["all_memory_samples_valid"] = all(
                s["memory_sample_valid"]
                for samples in receipt["variants"].values()
                for s in samples
            )
            print(
                "PLE_BF16_BATCH_COMPONENT_BENCH=" + json.dumps(receipt, sort_keys=True)
            )
            self.assertTrue(
                receipt["all_memory_samples_valid"],
                "Allocator receipt is contaminated or retains unexpected allocations",
            )
        finally:
            if gc_was_enabled:
                gc.enable()
            gc.collect()
            torch.cuda.current_stream().synchronize()

    def test_default_off_tail_and_graph_fallback(self):
        layer = self.layer()
        for length in (127, 128, 129, 257, 8191, 8193):
            hidden = torch.randn(1, length, 10240, device="cuda", dtype=torch.bfloat16)
            embedding = torch.randn(
                1, length, 2560, device="cuda", dtype=torch.bfloat16
            )
            for value in ("0", "1"):
                with patch.dict(os.environ, {"RTP_QWEN4_BF16_PLE_BATCH": value}), patch(
                    "torch.nn.functional.linear"
                ) as linear:
                    self.assertIsNone(
                        layer._try_bf16_prefill_page_snapshots(
                            hidden, embedding, 128, None, False
                        )
                    )
                    linear.assert_not_called()
            if length <= 257:
                tokens = torch.zeros((1, length + 2), device="cuda", dtype=torch.int64)
                with patch.dict(os.environ, {"RTP_QWEN4_BF16_PLE_BATCH": "0"}):
                    expected, expected_states = layer.prefill_page_snapshots(
                        hidden, tokens, 128, embeddings=embedding, add_residual=True
                    )
                with patch.dict(os.environ, {"RTP_QWEN4_BF16_PLE_BATCH": "1"}):
                    actual, states = layer.prefill_page_snapshots(
                        hidden, tokens, 128, embeddings=embedding, add_residual=True
                    )
                self.assert_exact(actual, expected)
                self.assertEqual(len(states), len(expected_states))
                for state, expected_state in zip(states, expected_states):
                    self.assert_exact(state, expected_state)
        hidden = torch.empty((1, 8192, 10240), device="cuda", dtype=torch.bfloat16)
        embedding = torch.empty((1, 8192, 2560), device="cuda", dtype=torch.bfloat16)
        for enabled, capture in (("0", False), ("1", True)):
            with patch.dict(os.environ, {"RTP_QWEN4_BF16_PLE_BATCH": enabled}), patch(
                "torch.cuda.is_current_stream_capturing", return_value=capture
            ), patch("torch.nn.functional.linear") as linear:
                self.assertIsNone(
                    layer._try_bf16_prefill_page_snapshots(
                        hidden, embedding, 128, None, False
                    )
                )
                linear.assert_not_called()


if __name__ == "__main__":
    unittest.main()
