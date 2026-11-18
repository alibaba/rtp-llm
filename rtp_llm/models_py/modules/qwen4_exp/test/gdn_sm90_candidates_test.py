import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.qwen4_exp.gdn_decode_triton import (
    _run_packed_decode,
    try_packed_decode,
)
from rtp_llm.models_py.triton_kernels.fla.block import store_ssm_state_to_block_map
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule_fwd
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd


class GdnSm90DispatchTest(unittest.TestCase):
    def test_cp_dependency_pairs(self):
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
            _sm90_cp_available,
        )

        try:
            for flashinfer_version, cutlass_version, supported in (
                ("0.6.9+8c4f4dcf", "4.4.2", True),
                ("0.7.3", "4.8.0", True),
                ("0.6.9+8c4f4dcf", "4.8.0", False),
                ("0.7.3", "4.4.2", False),
            ):
                with self.subTest(
                    flashinfer=flashinfer_version, cutlass=cutlass_version
                ), mock.patch.dict(
                    "sys.modules",
                    {
                        "flashinfer": SimpleNamespace(__version__=flashinfer_version),
                        "cutlass": SimpleNamespace(__version__=cutlass_version),
                    },
                ):
                    _sm90_cp_available.cache_clear()
                    self.assertEqual(_sm90_cp_available(), supported)
        finally:
            _sm90_cp_available.cache_clear()

    def test_extended_sm90_prefill_default_and_override(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNetPrefill,
        )
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
            Qwen4Sm120GdnPrefill,
        )

        package = "rtp_llm.models_py.modules.qwen4_exp"
        packed = SimpleNamespace(is_cuda=True, device="cuda:0")
        inputs = SimpleNamespace(input_lengths=None, prefix_lengths=None)
        for enabled in (None, "0", "1"):
            with self.subTest(enabled=enabled), mock.patch.dict(
                "os.environ", {"RTP_QWEN4_SM90_GDN_EXTENDED": enabled or "1"}
            ), mock.patch.object(
                Qwen3NextGatedDeltaNetPrefill, "__init__", return_value=None
            ):
                if enabled is None:
                    os.environ.pop("RTP_QWEN4_SM90_GDN_EXTENDED", None)
                candidate = Qwen4Sm120GdnPrefill()
            self.assertEqual(candidate.sm90_cp_extended, enabled != "0")
            output, fallback = object(), object()
            helper = mock.Mock(return_value=output)
            with (
                mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
                mock.patch(
                    "torch.cuda.get_device_properties",
                    return_value=SimpleNamespace(multi_processor_count=132),
                ),
                mock.patch(
                    f"{package}.gdn_prefill_adapter._sm90_cp_available",
                    return_value=True,
                ),
                mock.patch.object(
                    Qwen3NextGatedDeltaNetPrefill, "_fla", return_value=fallback
                ),
                mock.patch.dict(
                    "sys.modules",
                    {
                        f"{package}.gdn_prefill_cp_sm90": SimpleNamespace(
                            try_sm90_prefill=helper
                        )
                    },
                ),
            ):
                actual = candidate._fla(packed, None, None, None, 64, inputs, None)
            self.assertIs(actual, output if enabled != "0" else fallback)
            self.assertEqual(helper.call_count, int(enabled != "0"))

    def test_cp_prefill_is_enabled_by_default(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNetPrefill,
        )
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
            Qwen4Sm120GdnPrefill,
        )

        with mock.patch.object(
            Qwen3NextGatedDeltaNetPrefill, "__init__", return_value=None
        ):
            candidate = Qwen4Sm120GdnPrefill()
        self.assertTrue(candidate.sm90_cp_prefill)
        self.assertTrue(candidate.sm90_grouped_prepare)
        output = object()
        helper = mock.Mock(return_value=output)
        packed = SimpleNamespace(is_cuda=True, device="cuda:0")
        package = "rtp_llm.models_py.modules.qwen4_exp"
        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            mock.patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=78),
            ),
            mock.patch(
                f"{package}.gdn_prefill_adapter._sm90_cp_available", return_value=True
            ),
            mock.patch.dict(
                "sys.modules",
                {
                    f"{package}.gdn_prefill_cp_sm90": SimpleNamespace(
                        try_sm90_prefill=helper
                    )
                },
            ),
        ):
            actual = candidate._fla(packed, None, None, None, 64, None, None)
        self.assertIs(actual, output)
        helper.assert_called_once_with(
            candidate, packed, None, None, None, 64, None, None
        )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "requires SM90",
)
class GdnSm90CandidateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261112)

    def assertBytesEqual(self, value, expected):
        self.assertTrue(
            torch.equal(
                value.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def checked_packed_decode(self, *args, **kwargs):
        result = try_packed_decode(*args, **kwargs)
        self.assertIsNotNone(result, "eligible input fell back to native recurrence")
        return result

    @torch.inference_mode()
    def test_target_verify_model_entry_and_graph(self):
        from rtp_llm.models_py.modules.qwen4_exp.test.gdn_composition_test import (
            GdnDecodeConvGateSM90Test,
        )

        helper = GdnDecodeConvGateSM90Test
        with mock.patch.dict("os.environ"):
            os.environ.pop("RTP_QWEN4_SM90_GDN_EXTENDED", None)
            original_layer, candidate_layer = helper.make_pair(tp=2)
        candidate_layer.decode_gdn._packed_decode = try_packed_decode
        self.assertTrue(candidate_layer.decode_gdn._packed_verify)
        for steps in (2, 3, 4):
            with self.subTest(steps=steps):
                hidden, initial, inputs, meta = helper.make_decode_case(
                    candidate_layer, 2, tokens=steps
                )
                initial = initial.repeat(2, 1)
                inputs.kv_cache_kernel_block_id_device = torch.arange(
                    1, 2 * (steps + 2) + 1, device="cuda", dtype=torch.int32
                ).view(2, steps + 2)
                reference_state, candidate_state = initial.clone(), initial.clone()
                expected = helper.run_decode(
                    original_layer, hidden, reference_state, inputs, meta
                )
                with mock.patch.object(
                    candidate_layer.decode_gdn,
                    "_packed_decode",
                    wraps=self.checked_packed_decode,
                ) as selected:
                    actual = helper.run_decode(
                        candidate_layer, hidden, candidate_state, inputs, meta
                    )
                self.assertEqual(selected.call_count, 1)
                self.assertBytesEqual(actual, expected)
                self.assertBytesEqual(candidate_state, reference_state)
                inputs.is_cuda_graph = True
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = helper.run_decode(
                        candidate_layer, hidden, candidate_state, inputs, meta
                    )
                for _ in range(3):
                    hidden.normal_()
                    reference_state.copy_(initial)
                    candidate_state.copy_(initial)
                    expected = helper.run_decode(
                        original_layer, hidden, reference_state, inputs, meta
                    )
                    graph.replay()
                    self.assertBytesEqual(actual, expected)
                    self.assertBytesEqual(candidate_state, reference_state)
                graph.reset()

    @torch.inference_mode()
    def test_fp8_projection_decode_and_verify(self):
        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            has_deep_gemm,
            is_deep_gemm_e8m0_used,
        )
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
            CudaFp8GEMMLinear,
        )
        from rtp_llm.models_py.modules.qwen4_exp.test.gdn_composition_test import (
            GdnDecodeConvGateSM90Test,
        )
        from rtp_llm.utils.model_weight import W

        if not has_deep_gemm() or is_deep_gemm_e8m0_used():
            self.skipTest("requires the SM90 FP8 block-scale linear backend")
        helper = GdnDecodeConvGateSM90Test
        template, _ = helper.make_pair(tp=2)
        weights = dict(template.weights)
        packed_weight = torch.randn(8192, 2560, device="cuda").to(torch.float8_e4m3fn)
        weights[W.linear_attn_qkvz_w] = packed_weight.reshape(2560, 8192)
        weights[W.linear_attn_qkvz_s] = torch.full(
            (20, 64), 0.02, device="cuda", dtype=torch.float32
        )
        layers = []
        for enabled in ("0", None):
            with mock.patch.dict(
                "os.environ", {"RTP_QWEN4_SM90_GDN_EXTENDED": enabled or "1"}
            ):
                if enabled is None:
                    os.environ.pop("RTP_QWEN4_SM90_GDN_EXTENDED", None)
                layer = Qwen3NextGatedDeltaNet(
                    template.linear_attn_config,
                    template.parallelism_config,
                    dict(weights),
                    1e-6,
                    quant_config=Fp8BlockWiseQuantConfig(is_quanted=True),
                    norm_activation="sigmoid",
                ).eval()
            layer.tp_all_reduce = lambda value, group: value
            self.assertFalse(layer._qkvz_ba_fused)
            self.assertIsInstance(layer.in_proj_qkvz, CudaFp8GEMMLinear)
            layers.append(layer)
        original_layer, candidate_layer = layers
        self.assertIs(candidate_layer.decode_gdn._packed_decode, try_packed_decode)
        candidate_layer.decode_gdn._packed_decode = self.checked_packed_decode
        for steps in (1, 2, 4):
            with self.subTest(steps=steps):
                hidden, initial, inputs, meta = helper.make_decode_case(
                    candidate_layer, 2, tokens=steps
                )
                initial = initial.repeat(2, 1)
                inputs.kv_cache_kernel_block_id_device = torch.arange(
                    1, 2 * (steps + 2) + 1, device="cuda", dtype=torch.int32
                ).view(2, steps + 2)
                reference_state, candidate_state = initial.clone(), initial.clone()
                expected = helper.run_decode(
                    original_layer, hidden, reference_state, inputs, meta
                )
                actual = helper.run_decode(
                    candidate_layer, hidden, candidate_state, inputs, meta
                )
                self.assertBytesEqual(actual, expected)
                self.assertBytesEqual(candidate_state, reference_state)
                inputs.is_cuda_graph = True
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = helper.run_decode(
                        candidate_layer, hidden, candidate_state, inputs, meta
                    )
                hidden.normal_()
                reference_state.copy_(initial)
                candidate_state.copy_(initial)
                expected = helper.run_decode(
                    original_layer, hidden, reference_state, inputs, meta
                )
                graph.replay()
                self.assertBytesEqual(actual, expected)
                self.assertBytesEqual(candidate_state, reference_state)
                graph.reset()

    @torch.inference_mode()
    def test_verify_graph_refreshes_metadata_and_committed_state(self):
        from rtp_llm.models_py.modules.qwen4_exp.test.gdn_composition_test import (
            GdnDecodeConvGateSM90Test,
        )

        helper = GdnDecodeConvGateSM90Test
        with mock.patch.dict("os.environ", {"RTP_QWEN4_SM90_GDN_EXTENDED": "1"}):
            original_layer, candidate_layer = helper.make_pair(tp=2)
        candidate_layer.decode_gdn._packed_decode = self.checked_packed_decode
        for steps in (2, 3, 4):
            with self.subTest(steps=steps):
                hidden, initial, inputs, meta = helper.make_decode_case(
                    candidate_layer, 2, tokens=steps
                )
                initial = initial.repeat(8, 1)
                reference_state, candidate_state = initial.clone(), initial.clone()
                table = torch.zeros(2, 12, device="cuda", dtype=torch.int32)
                inputs.kv_cache_kernel_block_id_device = table
                inputs.is_cuda_graph = True
                lengths = inputs.sequence_lengths_plus_1_device
                lengths.copy_(torch.tensor([63, 127], device="cuda", dtype=torch.int32))
                source_slots = [1, 2]
                next_slot = 3

                def prepare_table():
                    nonlocal next_slot
                    table.zero_()
                    for row, length in enumerate(lengths.tolist()):
                        source_col, target_col = (length - 2) // 64, (length - 1) // 64
                        table[row, source_col] = source_slots[row]
                        destinations = torch.arange(
                            next_slot,
                            next_slot + steps,
                            device="cuda",
                            dtype=torch.int32,
                        )
                        if source_col == target_col:
                            destinations[0] = source_slots[row]
                        table[row, target_col : target_col + steps] = destinations
                        next_slot += steps

                prepare_table()
                helper.run_decode(
                    candidate_layer, hidden, candidate_state, inputs, meta
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = helper.run_decode(
                        candidate_layer, hidden, candidate_state, inputs, meta
                    )
                candidate_state.copy_(initial)
                for turn in range(4):
                    if turn == 3:
                        lengths[-1] = 0
                        table[-1].zero_()
                    hidden.normal_()
                    expected = helper.run_decode(
                        original_layer, hidden, reference_state, inputs, meta
                    )
                    graph.replay()
                    self.assertBytesEqual(actual, expected)
                    self.assertBytesEqual(candidate_state, reference_state)
                    if turn == 3:
                        break
                    accepted = [turn % steps + 1, steps - turn % steps]
                    for row, count in enumerate(accepted):
                        target_col = (int(lengths[row]) - 1) // 64
                        source_slots[row] = int(table[row, target_col + count - 1])
                    lengths.add_(
                        torch.tensor(accepted, device="cuda", dtype=torch.int32)
                    )
                    prepare_table()
                graph.reset()

    def test_extended_decode_attachment_with_quantized_projection(self):
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet
        from rtp_llm.models_py.modules.qwen4_exp.test.gdn_composition_test import (
            GdnDecodeConvGateSM90Test,
        )
        from rtp_llm.utils.model_weight import W

        reference, _ = GdnDecodeConvGateSM90Test.make_pair(tp=2)
        for enabled in (None, "0", "1"):
            with self.subTest(enabled=enabled):
                weights = dict(reference.weights)
                weights[W.linear_attn_qkvz_s] = torch.ones(1, device="cuda")
                factory = SimpleNamespace(
                    create_linear=mock.Mock(
                        side_effect=lambda *a, **kw: torch.nn.Identity()
                    ),
                    create_linear_from_weights=mock.Mock(
                        side_effect=lambda *a, **kw: torch.nn.Identity()
                    ),
                )
                with mock.patch.dict(
                    "os.environ", {"RTP_QWEN4_SM90_GDN_EXTENDED": enabled or "1"}
                ):
                    if enabled is None:
                        os.environ.pop("RTP_QWEN4_SM90_GDN_EXTENDED", None)
                    layer = Qwen3NextGatedDeltaNet(
                        reference.linear_attn_config,
                        reference.parallelism_config,
                        weights,
                        1e-6,
                        norm_activation="sigmoid",
                        linear_factory=factory,
                    )
                self.assertFalse(layer._qkvz_ba_fused)
                self.assertEqual(
                    getattr(layer.decode_gdn, "_packed_decode", None)
                    is try_packed_decode,
                    enabled != "0",
                )
                factory.create_linear.assert_not_called()

    def test_inference_output_projection_reads_current_weight(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
            CudaF16Linear,
        )
        from rtp_llm.models_py.modules.qwen4_exp.gdn_projection import Sm90GdnLinear

        with torch.inference_mode():
            weight = (
                torch.randint(-3, 4, (3072, 2560), device="cuda").to(torch.bfloat16)
                * 0.125
            )
        native = CudaF16Linear(weight)
        candidate = Sm90GdnLinear(native)
        value = (
            torch.randint(-3, 4, (8192, 3072), device="cuda").to(torch.bfloat16) * 0.125
        )
        self.assertTrue(candidate.weight.is_inference())
        self.assertEqual(candidate.weight.data_ptr(), native.weight.data_ptr())
        self.assertBytesEqual(candidate(value), native(value))
        self.assertIsNone(candidate._packed)
        self.assertEqual(candidate._plans, {})
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = candidate(value)
        for _ in range(3):
            with torch.inference_mode():
                native.weight.add_(0.125)
            graph.replay()
            self.assertBytesEqual(result, native(value))
            self.assertBytesEqual(candidate(value), native(value))
        graph.reset()
        with torch.inference_mode():
            candidate.weight = candidate.weight.clone()
        self.assertBytesEqual(candidate(value), native(value))

    def test_grouped_value_preparation_preserves_rounding_and_tail_guards(self):
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
            _sm90_cp_available,
        )

        if not _sm90_cp_available():
            self.skipTest("requires the qualified FlashInfer CP dependencies")
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_cp_sm90 import (
            _copy_value_alpha_beta,
            _prepare_cp,
        )

        for rows in (1, 3, 17, 2049):
            with self.subTest(rows=rows):
                packed = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
                projected = torch.randn(rows, 8240, device="cuda", dtype=torch.bfloat16)
                a, b = projected[:, -48:].split(24, dim=-1)
                a[0].fill_(30)
                b[0, ::2].fill_(-30)
                alog = torch.randn(24, device="cuda", dtype=torch.float32)
                bias = torch.randn(24, device="cuda", dtype=torch.bfloat16)
                saved = (packed.clone(), projected.clone())
                outputs = []
                for tokens, heads in ((16, 1), (4, 8)):
                    value = torch.full(
                        (rows + 2, 24, 128),
                        7,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    alpha = torch.full(
                        (rows + 2, 24), 7, device="cuda", dtype=torch.float32
                    )
                    beta = torch.full_like(alpha, 7)
                    _copy_value_alpha_beta[
                        ((rows + tokens - 1) // tokens, 24 // heads)
                    ](
                        packed,
                        a,
                        b,
                        alog,
                        bias,
                        value[1:-1],
                        alpha[1:-1],
                        beta[1:-1],
                        rows,
                        a.stride(0),
                        8,
                        24,
                        TOKENS=tokens,
                        HEADS=heads,
                        num_warps=4,
                    )
                    for tensor in (value, alpha, beta):
                        self.assertTrue(torch.all(tensor[0] == 7).item())
                        self.assertTrue(torch.all(tensor[-1] == 7).item())
                    outputs.append((value, alpha, beta))
                for actual, expected in zip(outputs[1], outputs[0]):
                    self.assertBytesEqual(actual, expected)
                self.assertBytesEqual(packed, saved[0])
                self.assertBytesEqual(projected, saved[1])

        rows = 32768
        packed = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
        projected = torch.randn(rows, 8240, device="cuda", dtype=torch.bfloat16)
        a, b = projected[:, -48:].split(24, dim=-1)
        expected = _prepare_cp(packed, a, b, alog, bias, 8, 24)
        actual = _prepare_cp(packed, a, b, alog, bias, 8, 24, group_heads=True)
        for value, reference in zip(actual, expected):
            self.assertBytesEqual(value, reference)

    def test_decode_pages_null_padding_and_strided_pool(self):
        batch = 8
        for page, steps in ((64, 1), (128, 1), (64, 2), (64, 3), (64, 4), (128, 4)):
            with self.subTest(page=page, steps=steps):
                tokens, width = batch * steps, steps + 3
                qkv = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
                projection = torch.randn(
                    tokens, 8240, device="cuda", dtype=torch.bfloat16
                )
                b, a = projection[:, -48:].split(24, dim=-1)
                alog = torch.randn(24, device="cuda", dtype=torch.float32)
                bias = torch.randn(24, device="cuda", dtype=torch.bfloat16)
                lengths = torch.tensor(
                    [0, 2, page, page + 1, page + 2, 2 * page, 2 * page + 1, 2],
                    device="cuda",
                    dtype=torch.int32,
                )
                mapping = torch.arange(
                    1, batch * width + 1, device="cuda", dtype=torch.int32
                ).view(batch, width)
                mapping[-1, 0] = 0
                raw = (
                    torch.randn(
                        batch * width + 1,
                        24 * 128 * 128 + 512,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    * 0.1
                )
                original = raw.clone()
                state = raw[:, : 24 * 128 * 128].view(batch * width + 1, 24, 128, 128)
                q, k, v = qkv.view(batch, steps, 40, 128).split((8, 8, 24), dim=2)
                g, beta = fused_gdn_gating(alog, a, b, bias)
                expected, _ = fused_recurrent_gated_delta_rule_fwd(
                    q,
                    k,
                    v,
                    g.view(batch, steps, 24),
                    beta.view(batch, steps, 24),
                    128**-0.5,
                    state,
                    block_map=mapping,
                    seq_size_per_block=page,
                    sequence_lengths=lengths,
                    use_qk_l2norm_in_kernel=True,
                )
                expected_state = raw.clone()
                raw.copy_(original)
                result = try_packed_decode(
                    qkv,
                    a,
                    b,
                    alog,
                    bias,
                    state,
                    mapping,
                    lengths,
                    page,
                    sequence_length=steps,
                )
                self.assertIsNotNone(result)
                self.assertBytesEqual(result, expected)
                self.assertBytesEqual(raw, expected_state)
                if steps == 4:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_output = try_packed_decode(
                            qkv,
                            a,
                            b,
                            alog,
                            bias,
                            state,
                            mapping,
                            lengths,
                            page,
                            sequence_length=steps,
                        )
                    raw.copy_(original)
                    graph.replay()
                    self.assertBytesEqual(graph_output, expected)
                    self.assertBytesEqual(raw, expected_state)
                padded = torch.empty(tokens, 5128, device="cuda", dtype=torch.bfloat16)[
                    :, :5120
                ]
                self.assertIsNone(
                    try_packed_decode(
                        padded,
                        a,
                        b,
                        alog,
                        bias,
                        state,
                        mapping,
                        lengths,
                        page,
                        sequence_length=steps,
                    )
                )
                self.assertIsNone(
                    try_packed_decode(
                        qkv.half(),
                        a,
                        b,
                        alog,
                        bias,
                        state,
                        mapping,
                        lengths,
                        page,
                        sequence_length=steps,
                    )
                )
                self.assertIsNone(
                    try_packed_decode(
                        qkv,
                        a,
                        b,
                        alog,
                        bias,
                        state,
                        mapping,
                        lengths,
                        32,
                        sequence_length=steps,
                    )
                )

    @torch.inference_mode()
    def test_decode_shared_quantizer_preserves_both_scales(self):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            scaled_fp8_per_tensor_quant,
            sgl_per_token_group_quant_fp8,
        )
        from rtp_llm.models_py.modules.qwen4_exp.gdn_decode_fp8 import (
            _quantize_ba,
            _quantize_qkvz,
        )

        with mock.patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL="legacy"):
            for batch in (16, 17, 32, 64):
                x = torch.randn(batch, 2560, device="cuda", dtype=torch.bfloat16)
                for gain in (1.0, 0.0, 1.0e-38, 1.0e-9, float("inf"), float("nan")):
                    x.normal_().mul_(gain)
                    q, scales = sgl_per_token_group_quant_fp8(
                        x,
                        128,
                        eps=1e-4,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                    )
                    padded = torch.empty(
                        ((batch + 15) // 16 * 16, 2560),
                        device=x.device,
                        dtype=q.dtype,
                    )
                    ba_q, ba_scale = scaled_fp8_per_tensor_quant(x, output=padded)
                    actual_q, actual_scales, partial = _quantize_qkvz(x)
                    actual_ba, actual_ba_scale = _quantize_ba(x, partial)
                    for actual, expected in zip(
                        (actual_q, actual_scales, actual_ba, actual_ba_scale),
                        (q, scales, ba_q, ba_scale),
                    ):
                        self.assertBytesEqual(actual, expected)

    def test_decode_small_batch_value_tile(self):
        for batch in (16, 17, 32, 63, 64, 65, 70, 100, 128):
            with self.subTest(batch=batch):
                qkv = torch.randn(batch, 5120, device="cuda", dtype=torch.bfloat16)
                a = torch.randn(batch, 24, device="cuda", dtype=torch.bfloat16)
                b = torch.randn_like(a)
                alog = torch.randn(24, device="cuda", dtype=torch.float32)
                bias = torch.randn(24, device="cuda", dtype=torch.bfloat16)
                lengths = torch.full((batch,), 65, device="cuda", dtype=torch.int32)
                lengths[0] = 0
                lengths[1] = 66
                mapping = torch.arange(
                    1, 2 * batch + 1, device="cuda", dtype=torch.int32
                ).view(batch, 2)
                states = (
                    torch.randn(
                        2 * batch + 1, 24, 128, 128, device="cuda", dtype=torch.float32
                    )
                    * 0.05
                )
                initial = states.clone()
                expected = _run_packed_decode(
                    qkv, a, b, alog, bias, states, mapping, lengths, 64, value_tile=8
                )
                expected_state = states.clone()
                for tuning in (
                    {},
                    {"tune_h20": True},
                    {"tune_h20_bf16": True},
                ):
                    states.copy_(initial)
                    output = try_packed_decode(
                        qkv,
                        a,
                        b,
                        alog,
                        bias,
                        states,
                        mapping,
                        lengths,
                        64,
                        **tuning,
                    )
                    self.assertBytesEqual(output, expected)
                    self.assertBytesEqual(states, expected_state)

    def test_prefill_checkpoint_tail_and_prefix(self):
        for page, rows, prefix in (
            (64, 1, 0),
            (64, 64, 0),
            (64, 65, 0),
            (128, 63, 0),
            (128, 128, 0),
            (128, 129, 0),
            (64, 2049, 128),
            (128, 2049, 256),
        ):
            with self.subTest(page=page, rows=rows, prefix=prefix):
                q = l2norm_fwd(
                    torch.randn(1, rows, 8, 128, device="cuda", dtype=torch.bfloat16)
                )
                k = l2norm_fwd(torch.randn_like(q))
                v = (
                    torch.randn(1, rows, 24, 128, device="cuda", dtype=torch.bfloat16)
                    * 0.1
                )
                g = -torch.rand(1, rows, 24, device="cuda", dtype=torch.float32) * 0.1
                beta = torch.rand(1, rows, 24, device="cuda", dtype=torch.bfloat16)
                initial = (
                    torch.randn(1, 24, 128, 128, device="cuda", dtype=torch.float32)
                    * 0.1
                )
                cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
                prefixes = torch.tensor([prefix], device="cuda", dtype=torch.int32)
                count = (prefix + rows + page - 1) // page + 1
                mapping = torch.arange(
                    count, 0, -1, device="cuda", dtype=torch.int32
                ).view(1, count)
                states = (
                    torch.randn(
                        count + 1, 24, 128, 128, device="cuda", dtype=torch.float32
                    )
                    * 0.1
                )
                original = states.clone()
                _, expected, _, final, _, h, _ = chunk_gated_delta_rule_fwd(
                    q, k, v, g, beta, 128**-0.5, initial, True, cu
                )
                store_ssm_state_to_block_map(
                    h, final, prefixes, cu, mapping, states, page, 64
                )
                if page == 64 and rows > 64:
                    first_page = mapping[0, prefix // page]
                    self.assertBytesEqual(states[first_page], h[0, 1])
                expected_state, expected_final = states.clone(), final.clone()
                states.copy_(original)
                _, result, _, final, _, _, _ = chunk_gated_delta_rule_fwd(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    128**-0.5,
                    initial,
                    True,
                    cu,
                    page_state_store=(states, mapping, prefixes, page),
                )
                self.assertBytesEqual(result, expected)
                self.assertBytesEqual(final, expected_final)
                self.assertBytesEqual(states, expected_state)

    def test_cp_initial_state_and_prefix_checkpoint_slots(self):
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
            _sm90_cp_available,
        )

        if not _sm90_cp_available():
            self.skipTest("requires the qualified FlashInfer CP dependencies")
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_cp_sm90 import _run

        rows, prefix = 65, 128
        for page in (64, 128):
            with self.subTest(page=page):
                q = l2norm_fwd(
                    torch.randn(1, rows, 8, 128, device="cuda", dtype=torch.bfloat16)
                )
                k = l2norm_fwd(torch.randn_like(q))
                v = (
                    torch.randn(1, rows, 24, 128, device="cuda", dtype=torch.bfloat16)
                    * 0.1
                )
                g = -torch.rand(1, rows, 24, device="cuda") * 0.1
                beta = torch.rand(1, rows, 24, device="cuda", dtype=torch.bfloat16)
                mapping = torch.tensor([[5, 3, 2, 4]], device="cuda", dtype=torch.int32)
                pool = torch.randn(6, 24, 128, 128, device="cuda") * 0.1
                original = pool.clone()
                initial = pool[mapping[0, (prefix - 1) // page]].unsqueeze(0).clone()
                cumulative = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
                prefixes = torch.tensor([prefix], device="cuda", dtype=torch.int32)
                _, expected, _, final, _, h, _ = chunk_gated_delta_rule_fwd(
                    q, k, v, g, beta, 128**-0.5, initial, True, cumulative
                )
                store_ssm_state_to_block_map(
                    h, final, prefixes, cumulative, mapping, pool, page, 64
                )
                expected_pool = pool.clone()
                pool.copy_(original)
                actual = _run(
                    q.squeeze(0),
                    k.squeeze(0),
                    v.squeeze(0),
                    g.exp().squeeze(0),
                    beta.float().squeeze(0),
                    pool,
                    mapping[:, prefix // page :],
                    cumulative,
                    page,
                    cp_chunk_len=64,
                    initial_state=initial,
                )
                first, end = prefix // page, (prefix + rows + page - 1) // page
                comparisons = [(actual, expected.squeeze(0))]
                for slot in mapping[0, first:end].tolist():
                    comparisons.append((pool[slot], expected_pool[slot]))
                for result, reference in comparisons:
                    difference = (
                        (result.float() - reference.float()).square().mean().sqrt()
                    )
                    scale = reference.float().square().mean().sqrt()
                    self.assertLess((difference / scale).item(), 0.005)
                    self.assertTrue(torch.isfinite(result).all().item())
                untouched = torch.ones(pool.shape[0], device="cuda", dtype=torch.bool)
                untouched[mapping[0, first:end]] = False
                self.assertBytesEqual(pool[untouched], original[untouched])

    @torch.inference_mode()
    def test_cp_prefill_production_entry_and_rejections(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNetPrefill,
            Qwen3NextMetadata,
        )
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
            Qwen4Sm120GdnPrefill,
            _sm90_cp_available,
        )
        from rtp_llm.models_py.modules.qwen4_exp.test.gdn_composition_test import (
            GdnDecodeConvGateSM90Test,
        )

        if not _sm90_cp_available():
            self.skipTest("requires the qualified FlashInfer CP dependencies")
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_cp_sm90 import (
            try_sm90_prefill,
        )

        template, _ = GdnDecodeConvGateSM90Test.make_pair(tp=2)
        cases = (
            (8192, 0, 64),
            (32768, 0, 64),
            (2048, 64, 64),
            (2049, 128, 128),
            (4800, 3200, 64),
            (5200, 4800, 64),
            (32768, 64, 64),
            (32768, 128, 128),
        )
        for rows, prefix, page in cases:
            with self.subTest(rows=rows, prefix=prefix, page=page):
                with mock.patch.dict(
                    "os.environ", {"RTP_QWEN4_SM90_GDN_EXTENDED": "0"}
                ):
                    if prefix:
                        os.environ.pop("RTP_QWEN4_SM90_GDN_EXTENDED", None)
                    owner = Qwen4Sm120GdnPrefill(
                        template.linear_attn_config,
                        template.parallelism_config,
                        template.weights,
                    )
                self.assertEqual(owner.sm90_cp_extended, bool(prefix))
                packed = (
                    torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16) * 0.1
                )
                gates = torch.randn(rows, 48, device="cuda", dtype=torch.bfloat16) * 0.1
                b, a = gates.split(24, dim=-1)
                pages = (prefix + rows + page - 1) // page + 1
                table = torch.arange(
                    pages, 0, -1, device="cuda", dtype=torch.int32
                ).view(1, pages)
                converter = owner.linear_cache_converter
                initial = torch.full(
                    (pages + 1, converter.block_size_bytes + 128),
                    -91,
                    device="cuda",
                    dtype=torch.int8,
                )
                owner._get_ssm_states(initial).normal_(std=0.05)
                reference_cache, candidate_cache = initial.clone(), initial.clone()
                inputs = SimpleNamespace(
                    is_cuda_graph=False,
                    input_lengths=torch.tensor([rows], dtype=torch.int32),
                    prefix_lengths=torch.tensor([prefix], dtype=torch.int32),
                    prefix_lengths_device=torch.tensor(
                        [prefix], device="cuda", dtype=torch.int32
                    ),
                    cu_seqlens_device=torch.tensor(
                        [0, rows], device="cuda", dtype=torch.int32
                    ),
                    kv_cache_kernel_block_id_device=table,
                )
                metadata = Qwen3NextMetadata()
                expected = Qwen3NextGatedDeltaNetPrefill._fla(
                    owner, packed, b, a, reference_cache, page, inputs, metadata
                )
                with mock.patch.object(
                    Qwen3NextGatedDeltaNetPrefill,
                    "_fla",
                    side_effect=AssertionError("eligible Prefill fell back"),
                ):
                    actual = owner._fla(
                        packed, b, a, candidate_cache, page, inputs, metadata
                    )
                first, end = prefix // page, (prefix + rows + page - 1) // page
                comparisons = [(actual, expected)]
                reference_pool = owner._get_ssm_states(reference_cache)
                candidate_pool = owner._get_ssm_states(candidate_cache)
                written = table[0, first:end]
                for slot in written.tolist():
                    comparisons.append((candidate_pool[slot], reference_pool[slot]))
                for value, reference in comparisons:
                    error = (value.float() - reference.float()).square().mean().sqrt()
                    scale = reference.float().square().mean().sqrt().clamp_min(1e-12)
                    self.assertLess((error / scale).item(), 0.005)
                    self.assertTrue(torch.isfinite(value).all().item())
                untouched = torch.ones(pages + 1, device="cuda", dtype=torch.bool)
                untouched[written] = False
                self.assertBytesEqual(candidate_cache[untouched], initial[untouched])
                self.assertBytesEqual(
                    converter.get_conv_state_tensor(candidate_cache),
                    converter.get_conv_state_tensor(initial),
                )

                if rows != 2048:
                    continue
                snapshot = candidate_cache.clone()
                invalid_rows = torch.empty(
                    2047, 5120, device="cuda", dtype=torch.bfloat16
                )
                too_many_rows = torch.empty(
                    32769, 5120, device="cuda", dtype=torch.bfloat16
                )
                for changes in (
                    {"prefix_lengths": torch.tensor([prefix + 1])},
                    {"kv_cache_kernel_block_id_device": table[:, :1]},
                    {"kv_cache_kernel_block_id_device": table.long()},
                    {"is_cuda_graph": True},
                ):
                    invalid = SimpleNamespace(**(vars(inputs) | changes))
                    self.assertIsNone(
                        try_sm90_prefill(
                            owner,
                            packed,
                            b,
                            a,
                            candidate_cache,
                            page,
                            invalid,
                            metadata,
                        )
                    )
                for invalid in (invalid_rows, too_many_rows):
                    self.assertIsNone(
                        try_sm90_prefill(
                            owner,
                            invalid,
                            b,
                            a,
                            candidate_cache,
                            page,
                            inputs,
                            metadata,
                        )
                    )
                self.assertBytesEqual(candidate_cache, snapshot)

    def test_projection_graph_streams_and_weight_refresh(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
            CudaF16Linear,
        )
        from rtp_llm.models_py.modules.qwen4_exp.gdn_projection import (
            Sm90GdnLinear,
            _library,
            _prefill_projection,
        )

        for columns, inner in ((8240, 2560), (2560, 3072)):
            weight = (
                torch.randint(-3, 4, (inner, columns), device="cuda").to(torch.bfloat16)
                * 0.125
            )
            native = CudaF16Linear(weight)
            candidate = Sm90GdnLinear(native)
            for rows in (16, 32, 64, 8192, 32768):
                value = (
                    torch.randint(-3, 4, (rows, inner), device="cuda").to(
                        torch.bfloat16
                    )
                    * 0.125
                )
                expected = native(value)
                self.assertBytesEqual(candidate(value), expected)
                h20 = torch.cuda.get_device_properties().multi_processor_count == 78
                if rows == 8192 and h20:
                    storage = torch.empty(
                        rows * inner + 1, device="cuda", dtype=torch.bfloat16
                    )
                    unaligned = storage[1:].view(rows, inner)
                    unaligned.copy_(value)
                    selected = (
                        candidate._packed if columns == 2560 else candidate.weight
                    )
                    uncached = torch.empty_like(expected)
                    _prefill_projection[(rows // 64 * ((columns + 127) // 128),)](
                        unaligned,
                        selected,
                        uncached,
                        columns,
                        inner,
                        *selected.stride(),
                        num_warps=4,
                        num_stages=3,
                    )
                    self.assertBytesEqual(candidate(unaligned), uncached)
                    self.assertBytesEqual(candidate(unaligned), uncached)
                if (
                    h20
                    and _library() is not None
                    and rows != 8192
                    and not (rows == 32 and columns == 2560)
                ):
                    packed = (
                        columns == 8240
                        and rows <= 64
                        or columns == 2560
                        and rows >= 8192
                    )
                    self.assertTrue(candidate._plans[(rows, packed)].valid)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = candidate(value)
                graph.replay()
                self.assertBytesEqual(actual, expected)
                graph.reset()
                streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                ready = torch.cuda.Event()
                ready.record()
                outputs = []
                for stream in streams:
                    stream.wait_event(ready)
                    with torch.cuda.stream(stream):
                        outputs.append(candidate(value))
                for stream in streams:
                    stream.synchronize()
                for result in outputs:
                    self.assertBytesEqual(result, expected)
            native.weight.add_(0.125)
            value = torch.ones(16, inner, device="cuda", dtype=torch.bfloat16)
            self.assertBytesEqual(candidate(value), native(value))
            candidate.weight = native.weight.clone()
            self.assertBytesEqual(candidate(value), native(value))
            tail = value[:3]
            self.assertBytesEqual(candidate(tail), native(tail))
            with torch.inference_mode():
                candidate.weight = native.weight.clone()
            self.assertBytesEqual(candidate(value), native(value))


if __name__ == "__main__":
    # Qualification requires all SM90 cases to execute, including optional CP.
    required = "--require-qualification" in sys.argv
    if required:
        sys.argv.remove("--require-qualification")
        if len(sys.argv) != 1:
            raise SystemExit(
                "SM90 qualification requires the complete unfiltered suite"
            )
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
            _sm90_cp_available,
        )

        if not (
            torch.cuda.is_available()
            and torch.version.hip is None
            and torch.cuda.get_device_capability() == (9, 0)
            and _sm90_cp_available()
        ):
            raise SystemExit(
                "SM90 qualification requires a supported FlashInfer/CUTLASS DSL pair"
            )
    result = unittest.main(verbosity=2, exit=False).result
    if required and (not result.testsRun or result.skipped):
        raise SystemExit("SM90 qualification requires executed tests without skips")
    raise SystemExit(not result.wasSuccessful())
