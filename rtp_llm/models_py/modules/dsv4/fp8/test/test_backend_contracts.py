"""CPU contracts for provider boundaries; numerical GPU coverage stays separate."""

import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import (
    attention_backend,
    backend_selection,
    sm120_projection,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention_backend import SparsePrefillInputs
from rtp_llm.models_py.modules.dsv4.fp8.sm120_attention_backend import SM120_ATTENTION


def source_fn(filename, name, env):
    path = Path(__file__).resolve().parents[1] / filename
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), env)
    return env[name]


class BackendContractsTest(unittest.TestCase):
    def setUp(self):
        for cached in (
            backend_selection._is_cuda_runtime,
            backend_selection._is_sm120_index,
            backend_selection._require_sm120_providers,
            attention_backend._backend_for_cuda,
        ):
            cached.cache_clear()
            self.addCleanup(cached.cache_clear)

    def test_selection_uses_operand_device_without_loading_kernel_providers(self):
        device = torch.device("cuda:7")
        with patch.object(
            backend_selection, "is_cuda", return_value=True
        ), patch.object(
            backend_selection, "is_sm120", side_effect=lambda d: d == device
        ) as arch, patch.object(
            torch.cuda, "current_device", side_effect=AssertionError("implicit device")
        ):
            for _ in range(2):
                self.assertIs(
                    attention_backend.get_sparse_attention_backend(device),
                    SM120_ATTENTION,
                )
                self.assertTrue(backend_selection.is_sm120_device(device))
                self.assertIsNot(
                    attention_backend.get_sparse_attention_backend(
                        torch.device("cuda:1")
                    ),
                    SM120_ATTENTION,
                )
                self.assertIsNot(
                    attention_backend.get_sparse_attention_backend(torch.device("cpu")),
                    SM120_ATTENTION,
                )
            self.assertEqual(
                [call.args[0] for call in arch.call_args_list],
                [device, torch.device("cuda:1")],
            )

    def test_indexless_alias_tracks_current_device_and_caches_canonical_index(self):
        with patch.object(
            backend_selection, "is_cuda", return_value=True
        ), patch.object(
            backend_selection, "is_sm120", side_effect=lambda d: d.index == 1
        ) as arch, patch.object(
            torch.cuda, "current_device", side_effect=[1, 0, 1, 0]
        ) as current:
            alias = torch.device("cuda")
            self.assertIs(
                attention_backend.get_sparse_attention_backend(alias), SM120_ATTENTION
            )
            self.assertIsNot(
                attention_backend.get_sparse_attention_backend(alias), SM120_ATTENTION
            )
            self.assertTrue(backend_selection.is_sm120_device(alias))
            self.assertFalse(backend_selection.is_sm120_device(alias))
            self.assertEqual(current.call_count, 4)
            self.assertEqual(
                [call.args[0].index for call in arch.call_args_list], [1, 0]
            )

    def test_non_nvidia_cuda_alias_does_not_probe_device(self):
        # PPU and ROCm also use torch's cuda device spelling.
        with patch.object(
            backend_selection, "is_cuda", return_value=False
        ), patch.object(
            torch.cuda, "current_device", side_effect=AssertionError("non-NVIDIA probe")
        ), patch.object(
            backend_selection, "is_sm120"
        ) as arch:
            for device in (
                torch.device("cuda"),
                torch.device("cuda:0"),
                torch.device("cpu"),
            ):
                self.assertIsNot(
                    attention_backend.get_sparse_attention_backend(device),
                    SM120_ATTENTION,
                )
                self.assertFalse(backend_selection.is_sm120_device(device))
            arch.assert_not_called()

    def test_only_exact_sm120_selects_sm120(self):
        from rtp_llm.models_py.utils import arch

        with patch.object(
            backend_selection, "is_cuda", return_value=True
        ), patch.object(arch, "is_cuda", return_value=True), patch.object(
            arch,
            "get_sm",
            side_effect=lambda d: {0: (12, 0), 1: (12, 1), 2: (10, 0), 3: (9, 0)}[
                d.index
            ],
        ):
            for index in range(4):
                self.assertEqual(
                    attention_backend.get_sparse_attention_backend(
                        torch.device("cuda", index)
                    )
                    is SM120_ATTENTION,
                    index == 0,
                )

    def test_shared_quant_guard_rejects_duck_typed_sm120_fallback(self):
        check = source_fn("attention.py", "_can_reuse_qkv_input_quant", {})
        linear = types.SimpleNamespace(
            quantize_input=Mock(), forward_quantized=Mock(), K=128, scale_ue8m0=True
        )
        owner = types.SimpleNamespace(wq_a=linear, wkv=linear)
        for backend, expected in (
            (attention_backend.SparseAttentionBackend(), True),
            (SM120_ATTENTION, False),
        ):
            with patch.object(
                attention_backend, "get_sparse_attention_backend", return_value=backend
            ):
                self.assertEqual(check(owner, torch.device("cpu")), expected)

    def test_head_dependent_prefix_and_positive_decode_window_guards(self):
        for heads in (8, 16, 32, 64, 128):
            self.assertTrue(SM120_ATTENTION.preserve_packed_prefix(heads))
            self.assertFalse(SM120_ATTENTION.uses_flash_mla_prefill(heads))
        self.assertTrue(SM120_ATTENTION.uses_flash_mla_prefill(12))
        for window in (0, -1):
            with self.assertRaises(ValueError):
                SM120_ATTENTION.validate_decode_window(window)
        SM120_ATTENTION.validate_decode_window(128)

    def test_default_prefill_keeps_flashmla_views_lengths_and_sink(self):
        q = torch.randn(5, 2, 4, dtype=torch.bfloat16)
        kv = torch.randn(7, 1, 4, dtype=torch.bfloat16)
        indices = torch.zeros(5, 1, 3, dtype=torch.int32)
        lengths = torch.tensor([1, 2, 3, 2, 1], dtype=torch.int32)
        sink = torch.randn(2)
        kernel = Mock(side_effect=lambda **kw: (kw["q"].clone(), None, None))
        module = types.SimpleNamespace(flash_mla_sparse_fwd=kernel)
        with patch.dict(
            sys.modules, {"rtp_llm.models_py.utils.flash_mla_legacy": module}
        ):
            prepared = attention_backend.SparseAttentionBackend().prepare_prefill(
                q=q,
                kv=kv,
                indices=indices,
                topk_length=lengths,
                attn_sink=sink,
                softmax_scale=0.5,
            )
            output = prepared.forward(1, 4)
        self.assertFalse(prepared.request_aligned)
        torch.testing.assert_close(output, q[1:4], rtol=0, atol=0)
        args = kernel.call_args.kwargs
        self.assertEqual(args["q"].data_ptr(), q[1:].data_ptr())
        self.assertIs(args["kv"], kv)
        self.assertIs(args["attn_sink"], sink)
        torch.testing.assert_close(args["topk_length"], lengths[1:4])
        self.assertEqual(args["sm_scale"], 0.5)

    def test_sm120_paged_prefill_keeps_live_padding_and_length_mask(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import sm120_sparse_mla

        q = torch.zeros(3, 8, 512, dtype=torch.bfloat16)
        cache = torch.zeros(2, 64, 584, dtype=torch.uint8)
        indices = torch.tensor([[65, -1], [70, 71], [80, -1]], dtype=torch.int32)
        lengths = torch.tensor([1, 2, 1], dtype=torch.int32)
        sink = torch.arange(8, dtype=torch.bfloat16)
        kernel = Mock(side_effect=lambda **kw: kw["out"].fill_(9))
        with patch.object(sm120_sparse_mla, "run", kernel):
            prepared = SM120_ATTENTION.prepare_prefill(
                n_heads=8,
                q=q,
                kv=None,
                indices=indices,
                topk_length=lengths,
                head_dim=512,
                softmax_scale=0.25,
                attn_sink=sink,
                prefill_workspace=None,
                backend_inputs=SparsePrefillInputs(swa_cache=cache),
            )
            output = prepared.forward(1, 3)
        self.assertTrue(prepared.request_aligned)
        torch.testing.assert_close(output, torch.full_like(q[1:3], 9))
        args = kernel.call_args.kwargs
        self.assertIs(args["swa_cache"], cache)
        torch.testing.assert_close(args["swa_lens"], lengths[1:3])
        self.assertEqual(args["swa_indices"].shape, (2, 128))
        self.assertTrue(torch.all(args["swa_indices"][0, 2:] == 70))
        self.assertTrue(torch.all(args["swa_indices"][1, 1:] == 80))
        torch.testing.assert_close(args["sinks"], sink.float())
        self.assertTrue(args["is_prefill"])
        # Each chunk observes the source sink, even for BF16 test fixtures.
        old_sink = args["sinks"].clone()
        sink.add_(1)
        with patch.object(sm120_sparse_mla, "run", kernel):
            prepared.forward(0, 1)
        torch.testing.assert_close(kernel.call_args.kwargs["sinks"], old_sink + 1)

    def test_compressor_projection_preserves_shape_precision_and_small_batch_padding(
        self,
    ):
        weight = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        x = torch.arange(8, dtype=torch.bfloat16).reshape(1, 2, 4)
        expected = torch.nn.functional.linear(x.float(), weight.float())
        for sm120, rows in ((False, 2), (True, 8)):
            with self.subTest(sm120=sm120):
                gemm = Mock(side_effect=lambda a, b: a.float() @ b.float().t())
                linear = source_fn(
                    "compressor.py",
                    "_linear_bf16_bf16_fp32",
                    {"torch": torch, "_CUBLAS_GEMM_BF16_BF16_FP32": gemm},
                )
                with patch.object(
                    backend_selection, "is_sm120_device", return_value=sm120
                ):
                    output = linear(x, weight)
                self.assertEqual(gemm.call_args.args[0].shape, (rows, 4))
                self.assertEqual(output.dtype, torch.float32)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_indexer_projection_preserves_fixed_rows_and_bf16_rounding(self):
        x = torch.arange(36, dtype=torch.bfloat16).reshape(3, 3, 4)
        weight = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        mm = torch.mm

        def gemm(a, b, *, out_dtype):
            self.assertEqual(a.shape[0], 8)
            self.assertEqual(out_dtype, torch.float32)
            return mm(a.float(), b.float())

        with patch.object(torch, "mm", side_effect=gemm) as calls:
            output = sm120_projection.indexer_decode(x, weight)
        self.assertEqual(calls.call_count, 2)
        expected = torch.nn.functional.linear(x.float(), weight.float()).to(x.dtype)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_invalid_cp_chunks_fail_before_projection_provider_import(self):
        with self.assertRaisesRegex(ValueError, "CP request chunks"):
            sm120_projection.compressor_prefill(
                torch.zeros(3, 4), torch.zeros(2, 4), Mock(), (1, 1)
            )

    def _provider_modules(self, blockwise=True, sparse_mla=True, grouped_fp4=True):
        blockwise_mod = types.ModuleType(
            "rtp_llm.models_py.kernels.cuda.sm120_fp8_blockwise"
        )
        blockwise_mod.sm120_blockwise_fp8_available = Mock(return_value=blockwise)
        sparse_mla_mod = types.ModuleType(
            "rtp_llm.models_py.modules.dsv4.fp8.sm120_sparse_mla"
        )
        sparse_mla_mod.sparse_mla_available = Mock(return_value=sparse_mla)
        grouped_fp4_mod = types.ModuleType(
            "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_grouped_fp4"
        )
        grouped_fp4_mod.sm120_grouped_fp4_available = Mock(return_value=grouped_fp4)
        stubs = {
            blockwise_mod.__name__: blockwise_mod,
            sparse_mla_mod.__name__: sparse_mla_mod,
            grouped_fp4_mod.__name__: grouped_fp4_mod,
        }
        return stubs, (blockwise_mod, sparse_mla_mod, grouped_fp4_mod)

    def test_non_sm120_device_imports_and_probes_no_provider(self):
        stubs, _ = self._provider_modules()
        for module in stubs.values():
            for probe in vars(module).values():
                if isinstance(probe, Mock):
                    probe.side_effect = AssertionError("provider probed")
        with patch.object(
            backend_selection, "is_cuda", return_value=True
        ), patch.object(backend_selection, "is_sm120", return_value=False), patch.dict(
            sys.modules, stubs
        ):
            self.assertIsNone(
                backend_selection.require_sm120_providers(torch.device("cuda", 0))
            )
            self.assertIsNone(
                backend_selection.require_sm120_providers(torch.device("cpu"))
            )

    def test_complete_provider_set_admits_and_caches(self):
        stubs, modules = self._provider_modules()
        with patch.object(
            backend_selection, "is_cuda", return_value=True
        ), patch.object(backend_selection, "is_sm120", return_value=True), patch.dict(
            sys.modules, stubs
        ):
            device = torch.device("cuda", 0)
            self.assertIsNone(backend_selection.require_sm120_providers(device))
            self.assertIsNone(backend_selection.require_sm120_providers(device))
        modules[0].sm120_blockwise_fp8_available.assert_called_once_with(0)
        modules[1].sparse_mla_available.assert_called_once_with()
        modules[2].sm120_grouped_fp4_available.assert_called_once_with()

    def test_each_missing_provider_names_only_itself(self):
        names = (
            "the SM120 FP8 blockwise GEMM binding",
            "trtllm_batch_decode_sparse_mla_dsv4 with is_prefill",
            "the FlashInfer fused-MoE workspace API",
        )
        for missing in range(3):
            with self.subTest(missing=names[missing]):
                backend_selection._require_sm120_providers.cache_clear()
                available = [True, True, True]
                available[missing] = False
                stubs, _ = self._provider_modules(*available)
                with patch.object(
                    backend_selection, "is_cuda", return_value=True
                ), patch.object(
                    backend_selection, "is_sm120", return_value=True
                ), patch.dict(
                    sys.modules, stubs
                ):
                    with self.assertRaises(RuntimeError) as raised:
                        backend_selection.require_sm120_providers(
                            torch.device("cuda", 0)
                        )
                message = str(raised.exception)
                self.assertIn("cuda13_sm120_dsv4", message)
                self.assertIn(names[missing], message)
                for other in range(3):
                    if other != missing:
                        self.assertNotIn(names[other], message)

    def test_all_missing_providers_names_all_three(self):
        stubs, _ = self._provider_modules(False, False, False)
        with patch.object(
            backend_selection, "is_cuda", return_value=True
        ), patch.object(backend_selection, "is_sm120", return_value=True), patch.dict(
            sys.modules, stubs
        ):
            with self.assertRaises(RuntimeError) as raised:
                backend_selection.require_sm120_providers(torch.device("cuda", 0))
        message = str(raised.exception)
        for name in (
            "the SM120 FP8 blockwise GEMM binding",
            "trtllm_batch_decode_sparse_mla_dsv4 with is_prefill",
            "the FlashInfer fused-MoE workspace API",
        ):
            self.assertIn(name, message)


if __name__ == "__main__":
    unittest.main()
