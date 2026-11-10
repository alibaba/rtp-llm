"""CPU integration coverage of the model's prepared QSA projection wiring.

The actual Qwen4ExpAttention constructor/forward, PreparedFp8Pair.project and
legacy project_fp8_pair run. Only the base-attention construction, hardware
predicates and numerical kernels are mocked. The candidate's native projection
is one mocked compute boundary; the legacy path still executes its Python
quantization/GEMM dispatch against separate mocks. Real CPU tensors preserve shape,
stride, owner and version behavior. Selection normally stops at the FMHA
handoff; one override case runs the real selection router as well.

This is dispatch evidence, not CUDA numerical, launch-count or latency evidence.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch

from rtp_llm.models_py.model_desc import qwen4_exp as model
from rtp_llm.models_py.model_desc.qwen4_exp_runtime import qsa_step_plan
from rtp_llm.models_py.modules.base.common import fp8_linear, qsa_fp8_pair
from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_gqa_impl import SparseGqaFmhaImpl


class SelectionComplete(Exception):
    """Stop after the actual model hands the selection to FMHA."""


class CustomSparseAdapter(SparseGqaFmhaImpl):
    pass


class CpuSourceGuard:
    """CPU double for native metadata validation; CUDA tests exercise the real guard."""

    def __init__(self, sources):
        self.tokens = tuple(self.token(tensor) for tensor in sources)
        self.device = sources[0].device

    @staticmethod
    def token(tensor):
        inference = torch.is_inference(tensor)
        return (
            tensor._cdata,
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            inference,
            None if inference else tensor._version,
        )

    def unchanged(self, sources):
        return self.tokens == tuple(self.token(tensor) for tensor in sources)


class QSAThreeKernelDispatchTest(unittest.TestCase):
    ROWS = 128

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Pretend the CPU-owned weights are supported CUDA weights only at the
        # Python hardware boundary. No CUDA tensor or CUDA kernel is created.
        self.stack.enter_context(
            patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            )
        )
        self.stack.enter_context(
            patch("torch.cuda.get_device_capability", return_value=(9, 0))
        )
        self.stack.enter_context(patch("torch.cuda.is_available", return_value=True))
        self.native_available = self.stack.enter_context(
            patch.object(
                qsa_fp8_pair, "_native_projection_available", return_value=True
            )
        )
        self.stack.enter_context(
            patch.object(
                qsa_fp8_pair, "_create_fp8_source_guard", side_effect=CpuSourceGuard
            )
        )
        self.capture = self.stack.enter_context(
            patch("torch.cuda.is_current_stream_capturing", return_value=False)
        )
        self.stack.enter_context(patch.object(torch.version, "hip", None))
        self.native_project = self.stack.enter_context(
            patch.object(
                qsa_fp8_pair,
                "_project_fp8_prefill_native",
                side_effect=self._native_project,
            )
        )
        self.old_quant = self.stack.enter_context(
            patch.object(
                fp8_linear, "_quantize_projection_input", side_effect=self._old_quant
            )
        )
        self.gemm = self.stack.enter_context(
            patch.object(torch, "_scaled_mm", side_effect=self._gemm)
        )
        self.legacy = self.stack.enter_context(
            patch.object(
                fp8_linear, "project_fp8_pair", wraps=fp8_linear.project_fp8_pair
            )
        )
        self.scalar_forward = self.stack.enter_context(
            patch.object(
                fp8_linear.Fp8Linear,
                "forward",
                autospec=True,
                side_effect=lambda owner, x, **kwargs: torch.zeros(
                    (x.shape[0], owner.weight.shape[0]),
                    dtype=torch.bfloat16,
                    device="cpu",
                ),
            )
        )
        self.x = torch.zeros((self.ROWS, 2560), dtype=torch.bfloat16, device="cpu")
        self.gate = self._projection(3072)
        self.inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            input_lengths=torch.tensor([self.ROWS], dtype=torch.int32, device="cpu"),
            prefix_lengths=torch.tensor([5120], dtype=torch.int32, device="cpu"),
        )
        self.cache = object()
        self.runtime = SimpleNamespace(
            main_cache=self.cache,
            main_inputs=self.inputs,
            indexer_kv_cache=SimpleNamespace(seq_size_per_block=128),
            indexer_state_cache=SimpleNamespace(seq_size_per_block=128),
            is_mtp_draft=False,
            rollback_side_cache=Mock(),
            finalize_side_cache=Mock(),
            validate_before_projection=Mock(),
            select_prefix_reuse_prefill_tokens=Mock(return_value=None),
        )
        self.fmha = self._fmha(SparseGqaFmhaImpl)
        self.attention = self._attention()
        self.pair = self.attention._qsa_fp8_pair
        self.assertIsInstance(self.pair, qsa_fp8_pair.PreparedFp8Pair)
        self.project = self.stack.enter_context(
            patch.object(self.pair, "project", wraps=self.pair.project)
        )

    @staticmethod
    def _projection(width):
        return fp8_linear.Fp8Linear(
            torch.zeros((width, 2560), dtype=torch.float8_e4m3fn, device="cpu"),
            torch.tensor([0.125], dtype=torch.float32, device="cpu"),
            role="PREFILL",
        )

    @staticmethod
    def _fmha(cls):
        fmha = cls.__new__(cls)
        fmha.begin_qsa_cache_transaction = Mock()
        fmha.qsa_main_cache_mutation_started = Mock(return_value=False)
        fmha.validate_qsa_before_side_write = Mock()
        fmha.set_mtp_draft_mode = Mock()
        fmha.set_selected_indices = Mock(side_effect=SelectionComplete)
        fmha.set_trusted_selected_indices = Mock(side_effect=SelectionComplete)
        return fmha

    def _attention(self, *, draft=False, enabled=True):
        config = SimpleNamespace(
            is_mtp=draft,
            enable_qwen4_qsa=enabled,
            attn_config=SimpleNamespace(
                indexer_head_num=4, rope_config=SimpleNamespace(dim=0)
            ),
            _qwen4_indexer_head_dim=128,
            _qwen4_indexer_compress_ratio=4,
            _qwen4_indexer_budget=2048,
            _qwen4_indexer_kv_heads=1,
        )
        weights = {
            model.W.qwen4_indexer_qk_proj_w: torch.zeros(
                (640, 2560), dtype=torch.float8_e4m3fn, device="cpu"
            ),
            model.W.qwen4_indexer_qk_proj_w
            + ".qwen4_fp8_storage_scale": torch.tensor(
                [0.03125], dtype=torch.float32, device="cpu"
            ),
            model.W.qwen4_indexer_q_ln_gamma: torch.zeros(128, device="cpu"),
            model.W.qwen4_indexer_k_ln_gamma: torch.zeros(128, device="cpu"),
        }

        def base_init(owner, *args, **kwargs):
            torch.nn.Module.__init__(owner)
            owner.gate = self.gate

        with patch.object(model.Qwen3NextAttention, "__init__", new=base_init):
            return model.Qwen4ExpAttention(
                config.attn_config, SimpleNamespace(), weights, 1e-6, config
            )

    @staticmethod
    def _native_project(x, weight_t, column_scales, guard, sources):
        if not (
            x.device == guard.device
            and x.dtype == torch.bfloat16
            and x.ndim == 2
            and x.shape[1] == 2560
            and x.shape[0] > 0
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and not x.requires_grad
        ) or not guard.unchanged(sources):
            return None
        padded = (x.shape[0] + 15) // 16 * 16
        return torch.zeros(
            (padded, weight_t.shape[1]), dtype=torch.bfloat16, device="cpu"
        )

    @staticmethod
    def _old_quant(rows, *, role, scale=None, output=None):
        return (
            (
                output
                if output is not None
                else torch.empty_like(rows, dtype=torch.float8_e4m3fn)
            ),
            torch.ones(1, dtype=torch.float32, device="cpu"),
        )

    @staticmethod
    def _gemm(quantized, weight, **kwargs):
        return torch.zeros(
            (quantized.shape[0], weight.shape[1]), dtype=torch.bfloat16, device="cpu"
        )

    def _run(self, *, fmha=None, real_selection=False):
        with ExitStack() as stack:
            selection = None
            if not real_selection:
                selection = stack.enter_context(
                    patch.object(
                        self.attention, "_qsa_selected_indices", return_value=None
                    )
                )
            with self.assertRaises(SelectionComplete):
                self.attention(
                    self.x,
                    fmha or self.fmha,
                    self.cache,
                    self.inputs,
                    qsa_runtime=self.runtime,
                )
            return selection

    def test_constructor_prepares_main_owner_and_forward_uses_real_pair(self):
        self.assertIs(self.pair.first, self.attention.gate)
        self.assertIs(self.pair.second, self.attention.qsa_indexer.qk_projection)
        selection = self._run()
        self.project.assert_called_once_with(
            self.x,
            first=self.attention.gate,
            second=self.attention.qsa_indexer.qk_projection,
        )
        self.legacy.assert_not_called()
        self.native_project.assert_called_once_with(
            self.x,
            self.pair.weight_t,
            self.pair.weight_scales,
            self.pair._source_guard,
            self.pair._source_tensors,
        )
        self.gemm.assert_not_called()
        q, k = selection.call_args.kwargs["projected"]
        self.assertEqual(q.shape, (self.ROWS, 4, 128))
        self.assertEqual(k.shape, (self.ROWS, 128))
        self.assertEqual((q.stride(0), k.stride(0)), (3712, 3712))
        self.runtime.rollback_side_cache.assert_called_once()

    def test_constructor_does_not_prepare_draft_or_disabled_qsa(self):
        with patch.object(
            fp8_linear, "prepare_fp8_pair", wraps=fp8_linear.prepare_fp8_pair
        ) as prepare:
            draft = self._attention(draft=True)
            disabled = self._attention(enabled=False)
        prepare.assert_not_called()
        self.assertIsNone(draft._qsa_fp8_pair)
        self.assertIsNone(disabled._qsa_fp8_pair)
        self.assertIsNone(disabled.qsa_indexer)

    def test_unavailable_native_build_keeps_old_pair_without_packing(self):
        self.native_available.return_value = False
        with patch.object(qsa_fp8_pair, "PreparedFp8Pair") as pack, patch.object(
            qsa_fp8_pair, "prepare_fp8_pair", wraps=qsa_fp8_pair.prepare_fp8_pair
        ) as prepare:
            self.attention = self._attention()
        prepare.assert_called_once_with(
            self.attention.gate, self.attention.qsa_indexer.qk_projection
        )
        pack.assert_not_called()
        self.assertIsNone(self.attention._qsa_fp8_pair)
        self._run()
        self.project.assert_not_called()
        self.native_project.assert_not_called()
        self.legacy.assert_called_once_with(
            self.x, self.attention.gate, self.attention.qsa_indexer.qk_projection
        )
        self.assertEqual(self.gemm.call_count, 2)

    def test_phase_and_custom_adapter_boundaries_use_original_pair(self):
        for case in ("draft", "graph", "verify", "cold", "decode", "adapter"):
            with self.subTest(case=case):
                self.project.reset_mock()
                self.legacy.reset_mock()
                self.native_project.reset_mock()
                self.attention.is_mtp_draft = case == "draft"
                self.runtime.is_mtp_draft = case == "draft"
                self.inputs.is_cuda_graph = case == "graph"
                self.inputs.is_target_verify = case == "verify"
                self.inputs.is_prefill = case != "decode"
                self.inputs.prefix_lengths.fill_(0 if case == "cold" else 5120)
                fmha = (
                    self._fmha(CustomSparseAdapter) if case == "adapter" else self.fmha
                )
                self._run(fmha=fmha)
                self.project.assert_not_called()
                self.native_project.assert_not_called()
                self.legacy.assert_called_once()

    def test_missing_or_declined_prepared_pair_falls_back_to_real_old_pair(self):
        for mode in ("missing", "capture"):
            with self.subTest(mode=mode):
                self.attention._qsa_fp8_pair = None if mode == "missing" else self.pair
                self.x = torch.zeros((self.ROWS, 2560), dtype=torch.bfloat16)
                self.capture.return_value = mode == "capture"
                self.legacy.reset_mock()
                self.native_project.reset_mock()
                self.gemm.reset_mock()
                selection = self._run()
                self.legacy.assert_called_once()
                self.native_project.assert_not_called()
                self.assertEqual(self.gemm.call_count, 2)
                self.assertIn("projected", selection.call_args.kwargs)

    def test_all_positive_row_sizes_share_projection_and_fresh_validation_scope(self):
        epochs = []
        outer_epoch = getattr(qsa_step_plan._SIDE_VALIDATION_LOCAL, "entries", None)

        def observe_validation(*args):
            epoch = getattr(qsa_step_plan._SIDE_VALIDATION_LOCAL, "entries", None)
            self.assertIsInstance(epoch, dict)
            self.assertTrue(all(epoch is not previous for previous in epochs))
            epochs.append(epoch)

        def run_attention(inputs, fmha):
            self.assertIs(inputs, self.inputs)
            return self._run(fmha=fmha)

        # Execute the real model-forward scope around the real attention
        # router. Only the remaining model layers/PLE and communicator are
        # absent; validation must remain inside a fresh epoch at every size.
        owner = SimpleNamespace(ple_layers={}, _forward_with_ple_prefetch=run_attention)
        self.fmha.validate_qsa_before_side_write.side_effect = observe_validation
        with patch(
            "rtp_llm.models_py.distributed.flashinfer_all_reduce."
            "get_flashinfer_allreduce",
            return_value=None,
        ):
            sizes = (1, 127, 128, 2048, 2049, 8193, 32769)
            for rows in sizes:
                with self.subTest(rows=rows):
                    self.x = torch.zeros(
                        (rows, 2560), dtype=torch.bfloat16, device="cpu"
                    )
                    self.inputs.input_lengths.fill_(rows)
                    for mock in (
                        self.project,
                        self.native_project,
                        self.legacy,
                        self.old_quant,
                        self.gemm,
                        self.fmha.validate_qsa_before_side_write,
                    ):
                        mock.reset_mock()
                    selection = model.Qwen4ExpModel.forward(
                        owner, self.inputs, self.fmha
                    )
                    self.fmha.validate_qsa_before_side_write.assert_called_once_with(
                        self.runtime, self.attention.qsa_indexer, self.x
                    )
                    self.assertIs(
                        getattr(qsa_step_plan._SIDE_VALIDATION_LOCAL, "entries", None),
                        outer_epoch,
                    )
                    q, k = selection.call_args.kwargs["projected"]
                    self.assertEqual(q.shape, (rows, 4, 128))
                    self.assertEqual(k.shape, (rows, 128))
                    self.project.assert_called_once_with(
                        self.x,
                        first=self.attention.gate,
                        second=self.attention.qsa_indexer.qk_projection,
                    )
                    self.native_project.assert_called_once_with(
                        self.x,
                        self.pair.weight_t,
                        self.pair.weight_scales,
                        self.pair._source_guard,
                        self.pair._source_tensors,
                    )
                    self.legacy.assert_not_called()
                    self.gemm.assert_not_called()
                    # Reshape may canonicalize the irrelevant leading stride
                    # of a singleton row. Storage identity/offsets still prove
                    # both outputs are views into the same packed projection.
                    self.assertEqual(
                        q.untyped_storage().data_ptr(), k.untyped_storage().data_ptr()
                    )
                    self.assertEqual(
                        (q.storage_offset(), k.storage_offset()), (3072, 3584)
                    )
                    if rows > 1:
                        self.assertEqual((q.stride(0), k.stride(0)), (3712, 3712))
        self.assertEqual(len(epochs), len(sizes))

    def test_rebound_gate_or_indexer_owner_declines_without_using_stale_weights(self):
        for owner in ("gate", "indexer"):
            with self.subTest(owner=owner):
                self.attention.gate = self.pair.first
                self.attention.qsa_indexer.qk_projection = self.pair.second
                if owner == "gate":
                    self.attention.gate = self._projection(3072)
                else:
                    self.attention.qsa_indexer.qk_projection = self._projection(640)
                self.legacy.reset_mock()
                self.native_project.reset_mock()
                self._run()
                self.native_project.assert_not_called()
                self.legacy.assert_called_once_with(
                    self.x,
                    self.attention.gate,
                    self.attention.qsa_indexer.qk_projection,
                )
                self.assertIs(
                    self.project.call_args.kwargs["first"], self.attention.gate
                )
                self.assertIs(
                    self.project.call_args.kwargs["second"],
                    self.attention.qsa_indexer.qk_projection,
                )

    def test_changed_weight_scale_version_uses_current_original_pair(self):
        self.attention.gate.scale.mul_(2)
        self._run()
        self.native_project.assert_called_once()
        self.legacy.assert_called_once()
        self.assertIs(
            self.gemm.call_args_list[0].kwargs["scale_b"], self.attention.gate.scale
        )

    def test_indexer_method_overrides_are_not_bypassed_by_either_pair(self):
        for name in ("project", "project_raw", "split_projected"):
            with self.subTest(name=name), patch.object(
                self.attention.qsa_indexer, name
            ):
                self.project.reset_mock()
                self.legacy.reset_mock()
                selection = self._run()
                self.project.assert_not_called()
                self.legacy.assert_not_called()
                self.assertNotIn("projected", selection.call_args.kwargs)

    def test_project_raw_override_reaches_real_selection_router(self):
        q = torch.zeros((self.ROWS, 4, 128), dtype=torch.bfloat16)
        k = torch.zeros((self.ROWS, 128), dtype=torch.bfloat16)
        with patch.object(
            self.attention.qsa_indexer, "project_raw", return_value=(q, k)
        ) as project_raw:
            self._run(real_selection=True)
        self.project.assert_not_called()
        self.legacy.assert_not_called()
        project_raw.assert_called_once_with(self.x)
        self.runtime.select_prefix_reuse_prefill_tokens.assert_called_once()
        call = self.runtime.select_prefix_reuse_prefill_tokens.call_args
        self.assertIs(call.args[0], q)
        self.assertIs(call.args[1], k)
        self.assertTrue(call.kwargs["raw_query"])

    def test_projection_override_and_hook_keep_module_dispatch(self):
        expected_gate = torch.zeros((self.ROWS, 3072), dtype=torch.bfloat16)
        with patch.object(
            self.attention.gate, "forward", return_value=expected_gate
        ) as forward:
            selection = self._run()
        forward.assert_called_once_with(self.x)
        self.native_project.assert_not_called()
        self.assertNotIn("projected", selection.call_args.kwargs)
        hook = Mock(return_value=None)
        handle = self.attention.gate.register_forward_hook(hook)
        try:
            selection = self._run()
        finally:
            handle.remove()
        hook.assert_called_once()
        self.assertNotIn("projected", selection.call_args.kwargs)

    def test_execution_error_propagates_and_does_not_trigger_fallback(self):
        self.native_project.side_effect = RuntimeError("projection launch failed")
        with patch.object(self.attention, "_qsa_selected_indices") as selection:
            with self.assertRaisesRegex(RuntimeError, "projection launch failed"):
                self.attention(
                    self.x, self.fmha, self.cache, self.inputs, qsa_runtime=self.runtime
                )
        self.legacy.assert_not_called()
        selection.assert_not_called()
        self.runtime.rollback_side_cache.assert_called_once()
        self.runtime.finalize_side_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
