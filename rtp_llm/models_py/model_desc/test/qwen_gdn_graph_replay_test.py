"""ROCm Qwen replay contracts, independent of NVIDIA-only model imports."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3_next import (
    HybridAttentionType,
    Qwen3NextGatedDeltaNetDecode,
    Qwen3NextMetadata,
    Qwen3NextModel,
    _GdnDecodeGraphReplay,
    _is_cuda_graph_forward,
    _QwenGdnGraphDelegate,
    _QwenGraphAttentionImpls,
)


class QwenGdnGraphReplayTest(unittest.TestCase):
    def test_decode_reuses_indices_per_shared_layout_not_per_layer(self):
        decode = object.__new__(Qwen3NextGatedDeltaNetDecode)
        torch.nn.Module.__init__(decode)
        decode.local_num_k_heads, decode.local_num_v_heads, decode.head_k_dim = (
            2,
            8,
            128,
        )
        decode.alog = torch.zeros(8)
        decode.dt_bias = torch.zeros(8)
        decode._get_ssm_states = lambda state: state
        decode._is_target_verify = lambda *args: False
        decode._get_bs_from_attenion_input = lambda *args: (1, 1)
        state = torch.zeros(5, 8, 128, 128)
        lengths = torch.tensor([2], dtype=torch.int32)
        tables = [torch.ones(1, 2, dtype=torch.int32) for _ in range(3)]
        prefix = "rtp_llm.models_py.model_desc.qwen3_next."
        with patch.object(torch.version, "hip", "rocm"), patch(
            prefix + "is_aiter_flydsl_gdn_decode_supported", return_value=True
        ), patch(
            prefix + "prepare_aiter_flydsl_gdn_decode_state_indices",
            side_effect=lambda meta: tuple(
                torch.ones(1, dtype=torch.int32) for _ in range(3)
            ),
        ) as prepare, patch(
            prefix + "aiter_flydsl_gdn_decode", return_value=torch.zeros(1, 1, 8, 128)
        ):
            for forward in range(2):
                metadata = Qwen3NextMetadata(None, False, is_cuda_graph=True)
                for layer in range(24):
                    table = tables[layer % 3]
                    inputs = SimpleNamespace(
                        kv_cache_kernel_block_id_device=table.view_as(table),
                        kv_cache_kernel_block_id=table.view_as(table),
                        sequence_lengths_plus_1_device=lengths,
                        sequence_lengths=lengths,
                    )
                    decode._fla(
                        torch.zeros(1, 12 * 128),
                        torch.zeros(1, 8),
                        torch.zeros(1, 8),
                        state,
                        1024,
                        inputs,
                        metadata,
                    )
                self.assertEqual(prepare.call_count, 3 * (forward + 1))
                self.assertEqual(len(metadata.aiter_flydsl_gdn_decode_indices), 3)

    def test_bound_buckets_read_updated_tables_and_reject_unknown_bucket(self):
        delegate = Mock()
        replay = _QwenGdnGraphDelegate({"linear": [(4, 8)]}, "full", delegate)
        buckets = []
        for size in (1, 2):
            lengths = torch.ones(size, dtype=torch.int32)
            group = SimpleNamespace(
                sequence_lengths=lengths,
                kv_cache_kernel_block_id=torch.ones(size, 2, dtype=torch.int32),
                sequence_lengths_plus_1_device=lengths.clone(),
            )
            inputs = SimpleNamespace(attention_inputs={"full": group, "linear": group})
            replay.bind_graph_inputs(inputs)
            buckets.append(group)
        for group in buckets:
            replay.prepare_cuda_graph(group)
            group.kv_cache_kernel_block_id[0, 0] = 4
            with self.assertRaisesRegex(RuntimeError, "invalid state block"):
                replay.prepare_cuda_graph(group)
            group.kv_cache_kernel_block_id[0, 0] = 1
        self.assertEqual(delegate.prepare_cuda_graph.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, "not registered"):
            replay.prepare_cuda_graph(
                SimpleNamespace(sequence_lengths=torch.ones(3, dtype=torch.int32))
            )

    def test_shared_padding_buffer_is_cleared_once(self):
        lengths = torch.tensor([1, 0], dtype=torch.int32)
        group = SimpleNamespace(
            sequence_lengths=lengths,
            kv_cache_kernel_block_id=torch.ones(2, 2, dtype=torch.int32),
            sequence_lengths_plus_1_device=lengths.clone(),
        )
        replay = _QwenGdnGraphDelegate(
            {"linear1": [(4, 8)], "linear2": [(4, 8)]}, "full", None
        )
        replay.bind_graph_inputs(
            SimpleNamespace(
                attention_inputs={"full": group, "linear1": group, "linear2": group}
            )
        )
        with patch.object(_GdnDecodeGraphReplay, "prepare_cuda_graph") as prepare:
            replay.prepare_cuda_graph(group)
        self.assertEqual(
            [call.kwargs["clear_padding"] for call in prepare.call_args_list],
            [True, False],
        )

    @unittest.skipIf(
        torch.version.hip is not None,
        "Run the shared C++ replay contract on CUDA: main's ROCm graph shim "
        "destroys a static Python module after interpreter shutdown",
    )
    def test_real_cpp_runner_calls_model_validator_on_every_replay(self):
        from rtp_llm.ops.compute_ops import (
            PyAttentionInputs,
            PyModelInputs,
            PyModelOutputs,
            get_typemeta,
        )

        relative = Path("rtp_llm/cpp/cuda_graph/tests/libtest_cuda_graph_runner.so")
        root = Path(__file__).resolve().parents[4]
        library = root / relative
        if not library.exists():
            library = root / "bazel-bin" / relative
        spec = importlib.util.spec_from_file_location(
            "libtest_cuda_graph_runner", library
        )
        extension = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = extension
        spec.loader.exec_module(extension)

        class Model:
            def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
                replay = _QwenGdnGraphDelegate({"linear": [(4, 8)]}, "linear", None)
                return _QwenGraphAttentionImpls({"linear": replay}, replay)

            def forward(self, inputs, fmha_impl=None):
                fmha_impl.bind_graph_inputs(inputs)
                lengths = inputs.attention_inputs[
                    "linear"
                ].sequence_lengths_plus_1_device
                return PyModelOutputs(inputs.input_hiddens + lengths[:, None])

        def make_live(block_id):
            attn = PyAttentionInputs()
            attn.is_prefill = False
            attn.is_target_verify = False
            attn.prefix_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
            attn.input_lengths = torch.ones(1, dtype=torch.int32).pin_memory()
            attn.sequence_lengths = torch.ones(1, dtype=torch.int32).pin_memory()
            attn.sequence_lengths_plus_1_device = torch.tensor(
                [2], device="cuda", dtype=torch.int32
            )
            attn.decode_cu_seqlens_device = torch.arange(
                2, device="cuda", dtype=torch.int32
            )
            attn.cu_seqlens = torch.zeros(2, dtype=torch.int32).pin_memory()
            attn.cu_seqlens_device = attn.cu_seqlens.cuda()
            attn.cu_kv_seqlens_device = torch.zeros_like(attn.cu_seqlens_device)
            attn.context_total_kv_length = 1
            attn.total_tokens = 1
            attn.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
            attn.padding_offset = torch.zeros(1, device="cuda", dtype=torch.int32)
            attn.kv_cache_kernel_block_id = torch.tensor(
                [[block_id]], dtype=torch.int32
            ).pin_memory()
            attn.kv_cache_kernel_block_id_device = attn.kv_cache_kernel_block_id.cuda()
            attn.kv_cache_block_id = attn.kv_cache_kernel_block_id
            attn.kv_cache_block_id_device = attn.kv_cache_kernel_block_id_device
            inputs = PyModelInputs()
            inputs.input_ids = torch.zeros(1, device="cuda", dtype=torch.int32)
            inputs.input_hiddens = torch.zeros(
                1, 4, device="cuda", dtype=torch.bfloat16
            )
            inputs.attention_inputs = {"linear": attn, "full": attn}
            return inputs

        runner = extension.CudaGraphRunner()
        runner.init_decode(Model(), 4, 8, 8, 8, [2], ["linear", "full"])
        valid = make_live(1)
        self.assertTrue(runner.canRun(valid))
        result = runner.forward(valid)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            result.hidden_states, torch.full_like(result.hidden_states, 2)
        )
        for block_id in (-1, 0, 4):
            invalid = make_live(block_id)
            self.assertTrue(runner.canRun(invalid))
            with self.assertRaisesRegex(RuntimeError, "invalid state block IDs"):
                runner.forward(invalid)

    def make_inputs(self):
        return SimpleNamespace(
            sequence_lengths=torch.tensor([1024, 0], dtype=torch.int32),
            sequence_lengths_plus_1_device=torch.tensor([1025, 999], dtype=torch.int32),
            kv_cache_kernel_block_id=torch.tensor([[1, 2], [0, 0]], dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.tensor(
                [[1, 2], [0, 0]], dtype=torch.int32
            ),
        )

    def test_refresh_and_invalid_live_blocks(self):
        inputs = self.make_inputs()
        delegate = Mock()
        hook = _GdnDecodeGraphReplay([(4, 1024)], delegate)
        hook.prepare_cuda_graph(inputs)
        self.assertEqual(inputs.sequence_lengths_plus_1_device.tolist(), [1025, 0])
        delegate.prepare_cuda_graph.assert_called_once_with(inputs)
        for bad_id in (0, -1, 4, 100):
            with self.subTest(bad_id=bad_id):
                inputs.kv_cache_kernel_block_id[0, 1] = bad_id
                with self.assertRaisesRegex(RuntimeError, "invalid state block IDs"):
                    hook.prepare_cuda_graph(inputs)
        self.assertEqual(delegate.prepare_cuda_graph.call_count, 1)

    def test_pool_bounds_are_per_capture(self):
        inputs = self.make_inputs()
        _GdnDecodeGraphReplay([(4, 1024)]).prepare_cuda_graph(inputs)
        with self.assertRaisesRegex(RuntimeError, "invalid state block IDs"):
            _GdnDecodeGraphReplay([(2, 1024)]).prepare_cuda_graph(inputs)

    def test_host_view_cache_tracks_bucket_switch_and_inplace_updates(self):
        first = self.make_inputs()
        second = self.make_inputs()
        hook = _GdnDecodeGraphReplay([(4, 1024)])
        hook.prepare_cuda_graph(first)
        hook.prepare_cuda_graph(second)
        first.kv_cache_kernel_block_id[0, 1] = 4
        with self.assertRaisesRegex(RuntimeError, "invalid state block IDs"):
            hook.prepare_cuda_graph(first)
        first.kv_cache_kernel_block_id[0, 1] = 2
        hook.prepare_cuda_graph(first)

    def test_invalid_padding_layout(self):
        inputs = self.make_inputs()
        for lengths in ([0, 10], [-1, 0]):
            inputs.sequence_lengths[:] = torch.tensor(lengths)
            with self.assertRaisesRegex(RuntimeError, "positive live rows"):
                _GdnDecodeGraphReplay([(4, 1024)]).prepare_cuda_graph(inputs)

    def test_cuda_dispatch_unchanged(self):
        model = object.__new__(Qwen3NextModel)
        impl = object()
        with patch.object(torch.version, "hip", None), patch.object(
            GptModelBase, "prepare_fmha_impl", return_value=impl
        ):
            self.assertIs(model.prepare_fmha_impl(object(), True), impl)

    def test_rocm_wraps_one_full_attention_callback_for_linear_validation(self):
        model = object.__new__(Qwen3NextModel)
        torch.nn.Module.__init__(model)
        model.layers = [SimpleNamespace(layer_type=HybridAttentionType.LINEAR)]
        model.kv_cache = SimpleNamespace(
            get_layer_cache_groups=lambda idx: [
                SimpleNamespace(
                    tag="linear",
                    kv_cache_base=torch.empty(4, 1),
                    seq_size_per_block=1024,
                )
            ]
        )
        groups = {
            tag: SimpleNamespace(
                is_cuda_graph=False, is_prefill=False, is_target_verify=False
            )
            for tag in ("full", "linear")
        }
        inputs = SimpleNamespace(attention_inputs=groups)
        full = object()
        with patch.object(torch.version, "hip", "rocm"), patch.object(
            GptModelBase, "prepare_fmha_impl", return_value={"full": full}
        ), patch(
            "rtp_llm.models_py.model_desc.qwen3_next._is_aiter_flydsl_gdn_decode_disabled",
            return_value=False,
        ):
            impls = model.prepare_fmha_impl(inputs, True)
            groups["linear"].is_target_verify = True
            target_impls = model.prepare_fmha_impl(inputs, True)
        self.assertIs(target_impls["full"], full)
        self.assertIsNone(target_impls.replay)
        self.assertEqual(set(impls), {"full"})
        self.assertIs(impls["full"].delegate, full)
        self.assertEqual(impls["full"].bounds_by_tag, {"linear": [(4, 1024)]})
        self.assertTrue(all(group.is_cuda_graph for group in groups.values()))
        # The next C++ -> Python call may produce fresh wrappers whose scalar
        # fields still carry defaults. Graph identity must remain explicit.
        for group in groups.values():
            group.is_cuda_graph = False
        with patch.object(torch.version, "hip", "rocm"):
            self.assertTrue(_is_cuda_graph_forward(inputs, impls))
            self.assertFalse(_is_cuda_graph_forward(inputs, {"full": full}))

    @unittest.skipUnless(torch.version.hip is not None, "requires ROCm")
    def test_model_callback_with_real_flydsl_graph_replay(self):
        from rtp_llm.models_py.triton_kernels.fla.aiter_flydsl_gdn_decode import (
            AiterFlydslGdnDecodeStateMetadata,
            aiter_flydsl_gdn_decode,
            prepare_aiter_flydsl_gdn_decode_state_indices,
        )

        torch.manual_seed(37)
        inputs = self.make_inputs()
        inputs.sequence_lengths_plus_1_device = (
            inputs.sequence_lengths_plus_1_device.cuda()
        )
        inputs.kv_cache_kernel_block_id_device = inputs.kv_cache_kernel_block_id.cuda()
        hook = _QwenGdnGraphDelegate({"linear": [(4, 1024)]}, "linear", None)
        hook.bind_graph_inputs(SimpleNamespace(attention_inputs={"linear": inputs}))
        hook.prepare_cuda_graph(inputs)
        state = torch.randn(4, 32, 128, 128, device="cuda", dtype=torch.float32)
        kwargs = {
            "q": torch.randn(2, 1, 16, 128, device="cuda", dtype=torch.bfloat16),
            "k": torch.randn(2, 1, 16, 128, device="cuda", dtype=torch.bfloat16),
            "v": torch.randn(2, 1, 32, 128, device="cuda", dtype=torch.bfloat16),
            "a": torch.randn(2, 32, device="cuda", dtype=torch.bfloat16),
            "b": torch.randn(2, 32, device="cuda", dtype=torch.bfloat16),
            "A_log": torch.randn(32, device="cuda", dtype=torch.float32),
            "dt_bias": torch.randn(32, device="cuda", dtype=torch.bfloat16),
        }
        metadata = AiterFlydslGdnDecodeStateMetadata(
            block_map=inputs.kv_cache_kernel_block_id_device,
            host_block_map=inputs.kv_cache_kernel_block_id,
            block_map_width=2,
            sequence_lengths_plus_1=inputs.sequence_lengths_plus_1_device,
            host_sequence_lengths=inputs.sequence_lengths,
            seq_size_per_block=1024,
            state_pool_size=4,
        )

        def run(pool):
            read, write, _ = prepare_aiter_flydsl_gdn_decode_state_indices(metadata)
            return aiter_flydsl_gdn_decode(
                **kwargs, state=pool, read_indices=read, write_indices=write
            )

        run(state.clone())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(state)
        for length in (1023, 1024, 1025):
            inputs.sequence_lengths[0] = length
            # Emulate the existing runner copying only live GPU lengths;
            # poison the tail to ensure the model callback is responsible.
            inputs.sequence_lengths_plus_1_device.copy_(
                torch.tensor([length + 1, 999], device="cuda", dtype=torch.int32)
            )
            hook.prepare_cuda_graph(inputs)
            expected_state = state.clone()
            expected = run(expected_state)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
            self.assertEqual(output[1].count_nonzero().item(), 0)
        inputs.kv_cache_kernel_block_id[0, 1] = 4
        with self.assertRaisesRegex(RuntimeError, "invalid state block IDs"):
            hook.prepare_cuda_graph(inputs)


if __name__ == "__main__":
    unittest.main()
