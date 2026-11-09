"""F5: one input quantization for the indexer window's two fp8 projections.

Plan-window trace evidence (sm90, per-layer ``qwen4.qsa.indexer`` windows,
T~1386-1568): every layer quantizes the SAME ``[T, 2560]`` hidden states twice
-- once for the attention gate projection ``[2560, 3072]`` and once for the
indexer q/k projection ``[2560, 640]`` -- paying two ``zero_``/absmax/quant
chains and two ``_scaled_mm`` launches.

The deployed projections keep their own per-tensor weight scales (the gate
and indexer storage scales differ), so a single ``_scaled_mm`` with one
``scale_b`` cannot reproduce both GEMMs exactly; the rowwise-scale fallback
changes the cuBLAS per-tensor epilogue rounding. The fusion therefore shares
only the INPUT quantization (one quant chain, one scale accumulator reset) and
keeps both ``_scaled_mm`` calls with their own weight scales, which is
bit-exact against the two-projection reference for the quantized bytes and for
every output element.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpAttention
from rtp_llm.models_py.model_desc.qwen4_exp_runtime import qsa_step_plan
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_step_plan import (
    has_nonzero_prefill_prefix,
)
from rtp_llm.models_py.modules.base.common.fp8_linear import (
    Fp8Linear,
    project_fp8_pair,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import Qwen4ExpQSAIndexer

HIDDEN = 2560
GATE_OUT = 3072
QK_OUT = 640
QUANT_TARGET = "rtp_llm.models_py.kernels.cuda.fp8_kernel.scaled_fp8_per_tensor_quant"


def _capture_dynamic_quant(record):
    """Wrap ``scaled_fp8_per_tensor_quant``, recording bytes and scale value."""
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
        scaled_fp8_per_tensor_quant as real,
    )

    def wrapped(rows, scale=None, output=None):
        quantized, out_scale = real(rows, scale=scale, output=output)
        record.append(
            (
                quantized[: rows.shape[0]].detach().clone(),
                float(out_scale.detach().clone().item()),
            )
        )
        return quantized, out_scale

    return patch(QUANT_TARGET, wrapped)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaQkProjectionFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(107202)
        qsa_step_plan.clear_step_plan_cache()

    def fixture(self, heads=4):
        """Deployed plan-window geometry: gate [3072, 2560], indexer q/k
        [(heads + 1) * 128, 2560] with distinct per-tensor weight scales."""

        def weight(shape):
            return (torch.randn(shape, device="cuda") * 0.1).to(torch.float8_e4m3fn)

        gate = Fp8Linear(
            weight((GATE_OUT, HIDDEN)), torch.tensor([0.119713], device="cuda")
        )
        indexer = Qwen4ExpQSAIndexer(
            weight(((heads + 1) * 128, HIDDEN)),
            torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.07,
            torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.07,
            n_heads=heads,
            kv_heads=1,
            head_dim=128,
            token_budget=2048,
            compress_ratio=4,
            norm_eps=1e-6,
            projection_scale=torch.tensor([0.03125], device="cuda"),
        )
        return gate, indexer

    def attention_fixture(self, heads=4):
        gate, indexer = self.fixture(heads=heads)
        attention = Qwen4ExpAttention.__new__(Qwen4ExpAttention)
        torch.nn.Module.__init__(attention)
        attention.qsa_indexer = indexer
        attention.gate = gate
        attention.qsa_rope_config = SimpleNamespace()
        attention.is_mtp_draft = False
        return attention, gate, indexer

    def runtime_fixture(self, is_prefill=True):
        inputs = SimpleNamespace(
            is_prefill=is_prefill, is_target_verify=False, is_cuda_graph=False
        )
        runtime = SimpleNamespace(
            main_cache=None,
            main_inputs=inputs,
            rollback_side_cache=Mock(),
            finalize_side_cache=Mock(),
            validate_before_projection=Mock(),
        )

        class SelectionComplete(Exception):
            pass

        fmha = SimpleNamespace(
            begin_qsa_cache_transaction=Mock(),
            qsa_main_cache_mutation_started=Mock(return_value=False),
            set_selected_indices=Mock(side_effect=SelectionComplete),
        )
        return runtime, fmha, SelectionComplete

    # ---- shared input quantization: bit-exact vs the two-projection path ----

    def test_plan_window_rows_share_one_input_quantization(self):
        gate, indexer = self.fixture()
        for rows in (4, 25):
            # The small-row domain: both paths quantize through the same small
            # triton quantizer, so only the output contract is observable.
            with self.subTest(rows=rows):
                x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
                expected_gate = gate(x)
                expected_q, expected_k = indexer.project_raw(x)
                actual_gate, projected_qk = project_fp8_pair(
                    x, gate, indexer.qk_projection
                )
                actual_q, actual_k = indexer.split_projected(projected_qk)
                for actual, expected in (
                    (actual_gate, expected_gate),
                    (actual_q, expected_q),
                    (actual_k, expected_k),
                ):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for rows in (26, 33, 64, 512, 1390, 1525, 1568, 4092):
            # The dynamic per-tensor quant domain the plan window runs in.
            with self.subTest(rows=rows):
                x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
                reference = []
                with _capture_dynamic_quant(reference):
                    expected_gate = gate(x)
                    expected_q, expected_k = indexer.project_raw(x)
                self.assertEqual(len(reference), 2)
                self.assertEqual(reference[0][0].shape, (rows, HIDDEN))
                # Both reference chains quantize identical bytes of the same
                # input; the fused path quantizes those bytes exactly once.
                self.assertTrue(
                    torch.equal(
                        reference[0][0].view(torch.uint8),
                        reference[1][0].view(torch.uint8),
                    )
                )
                fused = []
                with _capture_dynamic_quant(fused):
                    actual_gate, projected_qk = project_fp8_pair(
                        x, gate, indexer.qk_projection
                    )
                self.assertEqual(len(fused), 1)
                self.assertTrue(
                    torch.equal(
                        fused[0][0].view(torch.uint8),
                        reference[0][0].view(torch.uint8),
                    )
                )
                self.assertEqual(fused[0][1], reference[0][1])
                actual_q, actual_k = indexer.split_projected(projected_qk)
                for actual, expected in (
                    (actual_gate, expected_gate),
                    (actual_q, expected_q),
                    (actual_k, expected_k),
                ):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_pair_keeps_two_gemms_with_each_projection_weight_scale(self):
        gate, indexer = self.fixture()
        self.assertNotEqual(
            float(gate.scale.item()), float(indexer.qk_projection.scale.item())
        )
        x = torch.randn(1525, HIDDEN, device="cuda", dtype=torch.bfloat16)
        with patch.object(torch, "_scaled_mm", wraps=torch._scaled_mm) as gemm:
            project_fp8_pair(x, gate, indexer.qk_projection)
        # The two weight scales differ, so the pair must keep two GEMMs, each
        # consuming its own scale_b tensor (both scale domains preserved).
        self.assertEqual(gemm.call_count, 2)
        self.assertEqual(
            [tuple(call.args[1].shape) for call in gemm.call_args_list],
            [(HIDDEN, GATE_OUT), (HIDDEN, QK_OUT)],
        )
        self.assertIs(gemm.call_args_list[0].kwargs["scale_b"], gate.scale)
        self.assertIs(
            gemm.call_args_list[1].kwargs["scale_b"], indexer.qk_projection.scale
        )

    def test_weights_are_reused_without_per_call_concatenation(self):
        gate, indexer = self.fixture()
        x = torch.randn(1525, HIDDEN, device="cuda", dtype=torch.bfloat16)
        gate_weight = gate.weight.detach().clone()
        qk_weight = indexer.qk_projection.weight.detach().clone()

        def gemm_weight_ptrs():
            with patch.object(torch, "_scaled_mm", wraps=torch._scaled_mm) as gemm:
                project_fp8_pair(x, gate, indexer.qk_projection)
            self.assertEqual(gemm.call_count, 2)
            return [call.args[1].data_ptr() for call in gemm.call_args_list]

        first, second = gemm_weight_ptrs(), gemm_weight_ptrs()
        # The GEMM operands are the projections' own weight tensors on every
        # call: no per-call (or cached) weight concatenation is involved.
        self.assertEqual(first, second)
        self.assertEqual(first[0], gate.weight.data_ptr())
        self.assertEqual(first[1], indexer.qk_projection.weight.data_ptr())
        self.assertTrue(torch.equal(gate.weight, gate_weight))
        self.assertTrue(torch.equal(indexer.qk_projection.weight, qk_weight))

    # ---- owner routing: the plan window reaches the shared-quant path ----

    def test_owner_routes_plan_window_rows_through_the_shared_quant(self):
        attention, gate, indexer = self.attention_fixture()
        runtime, fmha, SelectionComplete = self.runtime_fixture(is_prefill=True)
        for rows, expected in (
            (1, 1),
            (2, 1),
            (32, 1),
            (33, 1),
            (1525, 1),
            (4092, 1),
            (32768, 1),
            (0, 0),
            (32784, 0),
        ):
            with self.subTest(rows=rows):
                x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
                with patch(
                    "rtp_llm.models_py.modules.base.common.fp8_linear.project_fp8_pair",
                    wraps=project_fp8_pair,
                ) as pair, patch.object(
                    attention, "_qsa_selected_indices", return_value=None
                ):
                    with self.assertRaises(SelectionComplete):
                        attention(
                            x, fmha, None, runtime.main_inputs, qsa_runtime=runtime
                        )
                    self.assertEqual(pair.call_count, expected)

    def test_pair_declines_unsupported_geometry(self):
        gate, indexer = self.fixture()
        x = torch.randn(1525, HIDDEN, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(project_fp8_pair(x.float(), gate, indexer.qk_projection))
        biased = Fp8Linear(
            gate.weight,
            gate.scale,
            torch.zeros(GATE_OUT, device="cuda", dtype=torch.bfloat16),
        )
        self.assertIsNone(project_fp8_pair(x, biased, indexer.qk_projection))
        projection = indexer.qk_projection
        handle = projection.register_forward_hook(lambda module, args, result: result)
        try:
            self.assertIsNone(project_fp8_pair(x, gate, projection))
        finally:
            handle.remove()

    # ---- per-layer prefix verdicts read the frozen per-step host facts ----

    def test_prefix_verdicts_read_frozen_host_facts(self):
        attention, _, _ = self.attention_fixture()
        x = torch.randn(128, HIDDEN, device="cuda", dtype=torch.bfloat16)
        for prefix_value in (0, 128):
            with self.subTest(prefix=prefix_value):
                qsa_step_plan.clear_step_plan_cache()
                inputs = SimpleNamespace(
                    is_prefill=True,
                    is_target_verify=False,
                    is_cuda_graph=False,
                    prefix_lengths=torch.tensor([prefix_value], dtype=torch.int32),
                    input_lengths=torch.tensor([128], dtype=torch.int32),
                )
                rope_cos = torch.ones(1, 128, 64, device="cuda", dtype=torch.bfloat16)
                rope_sin = torch.zeros(1, 128, 64, device="cuda", dtype=torch.bfloat16)
                runtime = SimpleNamespace(main_inputs=inputs)
                runtime.write_prefill_indexer_cache = Mock(
                    return_value=([128], rope_cos, rope_sin, {})
                )
                runtime.select_prefix_reuse_prefill_tokens = Mock(
                    return_value=torch.zeros(
                        128, 2051, dtype=torch.int32, device="cuda"
                    )
                )
                with patch(
                    "rtp_llm.models_py.model_desc.qwen4_exp.has_nonzero_prefill_prefix",
                    wraps=has_nonzero_prefill_prefix,
                ) as frozen:
                    attention._qsa_selected_indices(x, runtime)
                    # Both per-layer check sites (the prefix-reuse gate and the
                    # prefill dispatch) read the frozen verdict.
                    self.assertEqual(frozen.call_count, 2)
                    computes = qsa_step_plan.step_plan_cache_stats()["fact_computes"]
                    attention._qsa_selected_indices(x, runtime)
                    self.assertEqual(frozen.call_count, 4)
                    # The mirror-facts chain computed once; the second layer
                    # reuses it without another any()/item() host pass.
                    self.assertEqual(
                        qsa_step_plan.step_plan_cache_stats()["fact_computes"],
                        computes,
                    )
                if prefix_value:
                    self.assertEqual(
                        runtime.select_prefix_reuse_prefill_tokens.call_count, 2
                    )
                    runtime.write_prefill_indexer_cache.assert_not_called()
                else:
                    self.assertEqual(runtime.write_prefill_indexer_cache.call_count, 2)
                    runtime.select_prefix_reuse_prefill_tokens.assert_not_called()

    def test_missing_prefix_mirror_declines_without_crash(self):
        inputs = SimpleNamespace(input_lengths=torch.tensor([4], dtype=torch.int32))
        self.assertFalse(has_nonzero_prefill_prefix(inputs))


if __name__ == "__main__":
    unittest.main()
