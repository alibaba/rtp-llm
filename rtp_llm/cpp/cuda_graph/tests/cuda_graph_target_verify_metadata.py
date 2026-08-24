import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

try:
    from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
except ModuleNotFoundError:
    runner_so = os.environ.get("CUDA_GRAPH_TEST_RUNNER_SO")
    if not runner_so:
        raise
    spec = importlib.util.spec_from_file_location(
        "libtest_cuda_graph_runner", runner_so
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CUDA Graph test runner from {runner_so}")
    runner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner_module)
    CudaGraphRunner = runner_module.CudaGraphRunner
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    tokenspeed_mla_impl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    decode_query_length,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_impl import (
    TokenSpeedMlaDecodeImpl,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    rtp_llm_ops,
)


class _MetadataProbeModel:
    def __init__(self) -> None:
        self.capture_metadata = []
        self.replay_host_metadata = []
        self.replay_device_metadata = []
        self.initial_sequence_lengths_plus_1 = None
        self.attention_configs = AttentionConfigs()
        self.attention_configs.use_mla = True
        self.attention_configs.is_sparse = False
        self.attention_configs.kv_cache_dtype = KvCacheDataType.BASE
        self.attention_configs.head_num = 12
        self.attention_configs.kv_lora_rank = 512
        self.attention_configs.rope_head_dim = 64
        self.attention_configs.kernel_tokens_per_block = 64

    def prepare_fmha_impl(self, inputs, is_cuda_graph):
        attention = inputs.attention_inputs
        query_length = decode_query_length(attention)
        self.capture_metadata.append(
            (
                attention.total_tokens,
                attention.input_lengths_host.tolist(),
                attention.is_target_verify,
                is_cuda_graph,
                query_length,
                TokenSpeedMlaDecodeImpl.support(self.attention_configs, attention),
            )
        )
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        params.fill_params(
            attention.prefix_lengths_host,
            torch.empty(0, dtype=torch.int32),
            attention.input_lengths_host,
            attention.kv_cache_kernel_block_id_host,
            self.attention_configs.kernel_tokens_per_block,
            False,
        )
        metadata = SimpleNamespace(
            params=params,
            query_length=query_length,
            batch_size=attention.input_lengths.numel(),
        )

        def prepare_cuda_graph(current_attention):
            self.replay_host_metadata.append(
                (
                    current_attention.prefix_lengths_host.tolist(),
                    current_attention.input_lengths_host.tolist(),
                    current_attention.sequence_lengths_host.tolist(),
                )
            )
            self.replay_device_metadata.append(
                tuple(
                    tensor.clone()
                    for tensor in (
                        current_attention.input_lengths,
                        current_attention.prefix_lengths,
                        current_attention.sequence_lengths,
                        current_attention.sequence_lengths_plus_1_d,
                    )
                )
            )
            params.fill_params(
                current_attention.prefix_lengths_host,
                torch.empty(0, dtype=torch.int32),
                current_attention.input_lengths_host,
                current_attention.kv_cache_kernel_block_id_host,
                self.attention_configs.kernel_tokens_per_block,
                True,
            )

        metadata.prepare_cuda_graph = prepare_cuda_graph
        return metadata

    def forward(self, inputs, _fmha_impl=None):
        if self.initial_sequence_lengths_plus_1 is None:
            self.initial_sequence_lengths_plus_1 = (
                inputs.attention_inputs.sequence_lengths_plus_1_d.cpu()
            )
        # Keep a real CUDA operation in the graph while avoiding model weights.
        return PyModelOutputs(inputs.input_hiddens + 1)


class _SequenceHostProbeModel:
    def __init__(self) -> None:
        self.capture_sequence_lengths = []
        self.replay_sequence_lengths = []
        self.replay_sequence_lengths_device = []
        self.replay_sequence_lengths_plus_1 = []

    def prepare_fmha_impl(self, inputs, _is_cuda_graph):
        sequence_lengths_host = inputs.attention_inputs.sequence_lengths_host
        self.capture_sequence_lengths.append(
            (sequence_lengths_host.tolist(), sequence_lengths_host.is_pinned())
        )
        metadata = SimpleNamespace()

        def prepare_cuda_graph(current_attention):
            current_sequence_lengths_host = (
                current_attention.sequence_lengths_host
            )
            self.replay_sequence_lengths.append(
                (
                    current_sequence_lengths_host.tolist(),
                    current_sequence_lengths_host.is_pinned(),
                )
            )
            self.replay_sequence_lengths_device.append(
                current_attention.sequence_lengths.clone()
            )
            self.replay_sequence_lengths_plus_1.append(
                current_attention.sequence_lengths_plus_1_d.clone()
            )

        metadata.prepare_cuda_graph = prepare_cuda_graph
        return metadata

    def forward(self, inputs, _fmha_impl=None):
        return PyModelOutputs(inputs.input_hiddens + 1)


class CudaGraphTargetVerifyMetadataTest(unittest.TestCase):
    @staticmethod
    def _build_replay_inputs(batch_size: int, q_len: int) -> PyModelInputs:
        inputs = PyModelInputs()
        attention = PyAttentionInputs()
        total_tokens = batch_size * q_len
        inputs.input_ids = torch.arange(total_tokens, dtype=torch.int32, device="cuda")
        inputs.input_hiddens = torch.zeros(
            (total_tokens, 16), dtype=torch.float16, device="cuda"
        )
        attention.input_lengths = torch.full(
            (batch_size,), q_len, dtype=torch.int32, device="cuda"
        )
        attention.input_lengths_host = torch.full(
            (batch_size,), q_len, dtype=torch.int32
        ).pin_memory()
        prefixes = torch.tensor([126, 255], dtype=torch.int32)[:batch_size]
        attention.prefix_lengths = prefixes.cuda()
        attention.prefix_lengths_host = prefixes.pin_memory()
        attention.sequence_lengths = prefixes.cuda()
        attention.sequence_lengths_host = prefixes.clone().pin_memory()
        attention.sequence_lengths_plus_1_d = (prefixes + 1).cuda()
        attention.decode_cu_seqlens_d = (
            torch.arange(batch_size + 1, dtype=torch.int32, device="cuda") * q_len
        )
        block_table = torch.arange(
            batch_size * 6, dtype=torch.int32, device="cuda"
        ).reshape(batch_size, 6)
        attention.kv_cache_kernel_block_id_device = block_table
        attention.kv_cache_kernel_block_id_host = block_table.cpu()
        attention.kv_cache_block_id_device = block_table
        attention.kv_cache_block_id_host = attention.kv_cache_kernel_block_id_host
        attention.cu_seqlens = attention.decode_cu_seqlens_d
        attention.cu_kv_seqlens = attention.decode_cu_seqlens_d.clone()
        attention.padding_offset = torch.zeros(
            total_tokens, dtype=torch.int32, device="cuda"
        )
        attention.is_prefill = True
        attention.is_target_verify = True
        attention.total_tokens = total_tokens
        attention.context_total_kv_length = int((prefixes + q_len).sum())
        inputs.attention_inputs = attention
        return inputs

    @staticmethod
    def _build_decode_replay_inputs(sequence_lengths) -> PyModelInputs:
        batch_size = len(sequence_lengths)
        inputs = PyModelInputs()
        attention = PyAttentionInputs()
        inputs.input_ids = torch.arange(
            batch_size, dtype=torch.int32, device="cuda"
        )
        inputs.input_hiddens = torch.zeros(
            (batch_size, 16), dtype=torch.float16, device="cuda"
        )
        attention.input_lengths = torch.ones(
            batch_size, dtype=torch.int32, device="cuda"
        )
        attention.input_lengths_host = torch.ones(
            batch_size, dtype=torch.int32
        ).pin_memory()
        sequence_lengths_host = torch.tensor(
            sequence_lengths, dtype=torch.int32
        ).pin_memory()
        attention.sequence_lengths = sequence_lengths_host.cuda()
        attention.sequence_lengths_host = sequence_lengths_host
        attention.sequence_lengths_plus_1_d = attention.sequence_lengths + 1
        attention.decode_cu_seqlens_d = torch.arange(
            batch_size + 1, dtype=torch.int32, device="cuda"
        )
        block_table = torch.arange(
            batch_size * 6, dtype=torch.int32, device="cuda"
        ).reshape(batch_size, 6)
        attention.kv_cache_kernel_block_id_device = block_table
        attention.kv_cache_kernel_block_id_host = block_table.cpu().pin_memory()
        attention.kv_cache_block_id_device = block_table
        attention.kv_cache_block_id_host = (
            attention.kv_cache_kernel_block_id_host
        )
        attention.cu_seqlens = attention.decode_cu_seqlens_d
        attention.cu_kv_seqlens = attention.decode_cu_seqlens_d.clone()
        attention.padding_offset = torch.zeros(
            batch_size, dtype=torch.int32, device="cuda"
        )
        attention.is_prefill = False
        attention.total_tokens = batch_size
        inputs.attention_inputs = attention
        return inputs

    def test_runner_mirrors_decode_sequence_lengths_on_host(self):
        model = _SequenceHostProbeModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            hidden_size=16,
            max_seq_len=384,
            tokens_per_block=64,
            kernel_tokens_per_block=64,
            decode_capture_batch_sizes=[1, 8],
            max_context_batch_size=8,
        )

        self.assertIn(([382], True), model.capture_sequence_lengths)
        self.assertIn(([382] * 8, True), model.capture_sequence_lengths)
        replay_inputs = self._build_decode_replay_inputs([126, 255])
        self.assertTrue(runner.canRun(replay_inputs))
        outputs = runner.forward(replay_inputs)
        torch.cuda.synchronize()

        self.assertEqual(
            model.replay_sequence_lengths[-1], ([126, 255] + [0] * 6, True)
        )
        torch.testing.assert_close(
            model.replay_sequence_lengths_device[-1].cpu(),
            torch.tensor([126, 255] + [0] * 6, dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model.replay_sequence_lengths_plus_1[-1].cpu(),
            torch.tensor([127, 256] + [1] * 6, dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            outputs.hidden_states,
            torch.ones_like(outputs.hidden_states),
        )

    def test_async_prepare_uses_stream_wait_without_blocking_cpu(self):
        model = _SequenceHostProbeModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            hidden_size=16,
            max_seq_len=384,
            tokens_per_block=64,
            kernel_tokens_per_block=64,
            decode_capture_batch_sizes=[2],
            max_context_batch_size=2,
        )

        first_inputs = self._build_decode_replay_inputs([126, 255])
        self.assertTrue(runner.canRun(first_inputs))

        # An unrecorded forward event must be a no-op on the first async
        # prepare. Complete that prepare before consuming it on the main stream.
        first_prepare_stream = torch.cuda.Stream()
        with torch.cuda.stream(first_prepare_stream):
            runner.prepareAttentionInputs(first_inputs)
            first_prepare_done = first_prepare_stream.record_event()
        first_prepare_done.synchronize()
        runner.forward(first_inputs)
        torch.cuda.synchronize()

        second_inputs = self._build_decode_replay_inputs([63, 127])
        self.assertTrue(runner.canRun(second_inputs))
        prepare_stream = torch.cuda.Stream()

        # Put the next replay behind a long-running kernel. Async prepare on a
        # second stream must enqueue a wait and return while this gate is still
        # pending; a CPU event synchronize would make gate_done query true.
        torch.cuda._sleep(1_000_000_000)
        gate_done = torch.cuda.current_stream().record_event()
        runner.forward(first_inputs)
        self.assertFalse(gate_done.query())

        with torch.cuda.stream(prepare_stream):
            runner.prepareAttentionInputs(second_inputs)
            prepare_done = prepare_stream.record_event()

        self.assertFalse(gate_done.query())
        prepare_done.synchronize()
        self.assertTrue(gate_done.query())
        self.assertEqual(model.replay_sequence_lengths[-1], ([63, 127], True))

    def test_runner_publishes_rectangular_query_shape(self):
        model = _MetadataProbeModel()
        runner = CudaGraphRunner()
        with mock.patch.dict(
            os.environ, {"RTP_MLA_DECODE_KERNEL": "auto"}
        ), mock.patch.object(
            tokenspeed_mla_impl, "_is_tokenspeed_blackwell", return_value=True
        ), mock.patch.object(
            tokenspeed_mla_impl, "_load_tokenspeed_mla", return_value=True
        ), mock.patch.object(
            tokenspeed_mla_impl,
            "tokenspeed_mla_kernel_supported",
            return_value=True,
        ):
            runner.init_decode(
                model,
                hidden_size=16,
                max_seq_len=384,
                tokens_per_block=64,
                kernel_tokens_per_block=64,
                decode_capture_batch_sizes=[1, 8],
                num_tokens_per_bs=4,
                is_target_verify=True,
                max_context_batch_size=8,
            )

        self.assertIn((4, [4], True, True, 4, True), model.capture_metadata)
        self.assertIn((32, [4] * 8, True, True, 4, True), model.capture_metadata)
        self.assertTrue(
            all(total_tokens > 0 for total_tokens, *_ in model.capture_metadata)
        )
        self.assertIsNotNone(model.initial_sequence_lengths_plus_1)
        torch.testing.assert_close(
            model.initial_sequence_lengths_plus_1,
            torch.full_like(model.initial_sequence_lengths_plus_1, 381),
            rtol=0,
            atol=0,
        )
        replay_inputs = self._build_replay_inputs(batch_size=2, q_len=4)
        self.assertTrue(runner.canRun(replay_inputs))
        outputs = runner.forward(replay_inputs)
        torch.cuda.synchronize()

        self.assertEqual(
            model.replay_host_metadata[-1],
            (
                [126, 255] + [380] * 6,
                [4] * 8,
                [126, 255] + [379] * 6,
            ),
        )
        expected_device_metadata = (
            [4] * 8,
            [126, 255] + [380] * 6,
            [126, 255] + [379] * 6,
            [127, 256] + [381] * 6,
        )
        for actual, expected in zip(
            model.replay_device_metadata[-1], expected_device_metadata
        ):
            torch.testing.assert_close(
                actual.cpu(),
                torch.tensor(expected, dtype=torch.int32),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            outputs.hidden_states,
            torch.ones_like(outputs.hidden_states),
        )

    def test_runner_falls_back_before_target_verify_metadata_overflow(self):
        model = _MetadataProbeModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            hidden_size=16,
            max_seq_len=384,
            tokens_per_block=64,
            kernel_tokens_per_block=64,
            decode_capture_batch_sizes=[1],
            num_tokens_per_bs=4,
            is_target_verify=True,
            max_context_batch_size=1,
        )
        replay_inputs = self._build_replay_inputs(batch_size=1, q_len=4)
        overflow_prefix = 381
        replay_inputs.attention_inputs.prefix_lengths.fill_(overflow_prefix)
        replay_inputs.attention_inputs.prefix_lengths_host.fill_(overflow_prefix)
        replay_inputs.attention_inputs.sequence_lengths.fill_(overflow_prefix)
        replay_inputs.attention_inputs.sequence_lengths_plus_1_d.fill_(
            overflow_prefix + 1
        )

        self.assertFalse(runner.canRun(replay_inputs))

    def test_runner_falls_back_when_query_length_differs_from_capture(self):
        model = _MetadataProbeModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            hidden_size=16,
            max_seq_len=384,
            tokens_per_block=64,
            kernel_tokens_per_block=64,
            decode_capture_batch_sizes=[1],
            num_tokens_per_bs=4,
            is_target_verify=True,
            max_context_batch_size=1,
        )
        replay_inputs = self._build_replay_inputs(batch_size=1, q_len=3)

        self.assertFalse(runner.canRun(replay_inputs))


if __name__ == "__main__":
    unittest.main()
