import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

try:
    from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import (
        CudaGraphRunner,
        bind_model_cache_inputs,
    )
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
    bind_model_cache_inputs = runner_module.bind_model_cache_inputs
from rtp_llm.models_py.model_desc import module_base
from rtp_llm.models_py.model_desc.kimi_k3_eagle3 import KimiK3Eagle3Model
from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel
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
            current_sequence_lengths_host = current_attention.sequence_lengths_host
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


def _draft_planner(attention, is_cuda_graph=False):
    # Keep the production host planner and TokenSpeed GPU metadata conversion;
    # omit model weights and the architecture-specific attention kernel.
    planner = TokenSpeedMlaDecodeImpl.__new__(TokenSpeedMlaDecodeImpl)
    planner.seq_size_per_block = 64
    planner.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
    planner.fmha_impl = tokenspeed_mla_impl._TokenSpeedDecodeMetadata(
        64,
        attention.input_lengths.numel(),
        384,
        is_cuda_graph,
        torch.device("cuda"),
    )
    planner.prepare(attention)
    return planner


class _DraftPageTableProbeModel:
    def initialize(self, _resources):
        return True

    def prepare_fmha_impl(self, inputs, is_cuda_graph):
        return _draft_planner(inputs.attention_inputs, is_cuda_graph)

    def forward(self, inputs, fmha_impl=None):
        # The captured kernel consumes the real planner's first page ID.
        return PyModelOutputs(
            inputs.input_hiddens + fmha_impl.fmha_impl.block_tables[0, 0]
        )


class CudaGraphTargetVerifyMetadataTest(unittest.TestCase):
    @classmethod
    def _draft_inputs(cls, draft_first_page=50):
        inputs = cls._build_decode_replay_inputs([70])
        attention = inputs.attention_inputs
        tables = [
            torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int32),
            torch.tensor([[30, 31, 32, 33, 34, 35]], dtype=torch.int32),
            torch.tensor([[draft_first_page, 51, 52, 53, 54, 55]], dtype=torch.int32),
        ]
        attention.kv_cache_kernel_block_id_host_by_group = [
            t.pin_memory() for t in tables
        ]
        attention.kv_cache_kernel_block_id_device_by_group = [t.cuda() for t in tables]
        attention.kv_cache_kernel_block_id_host = tables[0].pin_memory()
        attention.kv_cache_kernel_block_id_device = (
            attention.kv_cache_kernel_block_id_device_by_group[0]
        )
        attention.kv_cache_layer_to_group = torch.tensor(
            [2], dtype=torch.int32
        ).pin_memory()
        attention.kv_cache_layer_to_group_host = attention.kv_cache_layer_to_group
        inputs.attention_inputs = attention
        return inputs

    def test_draft_planner_selects_group_before_eager_construction(self):
        for model_cls in (KimiK3MtpModel, KimiK3Eagle3Model):
            for graph_capture in (False, True):
                with self.subTest(
                    model=model_cls.__name__, graph_capture=graph_capture
                ):
                    model = model_cls.__new__(model_cls)
                    torch.nn.Module.__init__(model)
                    model.config = model.parallelism_config = model.weight = (
                        model.fmha_config
                    ) = None
                    model.cuda_graph_fmha_workspaces = {}
                    inputs = self._draft_inputs()

                    def metadata_only_factory(_c, _p, _w, attn, _f, graph):
                        planner = _draft_planner(attn, graph)
                        return SimpleNamespace(
                            fmha_impl=planner.fmha_impl, fmha_params=planner.fmha_params
                        )

                    with mock.patch.object(
                        module_base.AttnImplFactory,
                        "get_fmha_impl",
                        side_effect=metadata_only_factory,
                    ):
                        planner = model.prepare_fmha_impl(inputs, graph_capture)
                    torch.cuda.synchronize()
                    self.assertEqual(
                        planner.fmha_impl.block_tables[0, :2].tolist(), [50, 51]
                    )

    def test_cpp_draft_binding_selects_host_and_device_before_planning(self):
        inputs = self._draft_inputs()
        attention = bind_model_cache_inputs(
            _DraftPageTableProbeModel(), inputs.attention_inputs, 1
        )
        self.assertEqual(
            attention.kv_cache_kernel_block_id_host[0, :2].tolist(), [50, 51]
        )
        self.assertEqual(
            attention.kv_cache_kernel_block_id_device[0, :2].tolist(), [50, 51]
        )

    def test_cpp_target_binding_preserves_full_group_when_first_layer_is_linear(self):
        inputs = self._draft_inputs()
        inputs.attention_inputs.kv_cache_layer_to_group = torch.tensor(
            [1, 0], dtype=torch.int32
        )
        inputs.attention_inputs.kv_cache_layer_to_group_host = (
            inputs.attention_inputs.kv_cache_layer_to_group
        )
        attention = bind_model_cache_inputs(
            _DraftPageTableProbeModel(), inputs.attention_inputs, 0
        )
        self.assertEqual(
            attention.kv_cache_kernel_block_id_host[0, :2].tolist(), [10, 11]
        )

    def test_draft_graph_replay_uses_selected_live_host_table(self):
        model = _DraftPageTableProbeModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            hidden_size=16,
            max_seq_len=384,
            tokens_per_block=64,
            kernel_tokens_per_block=64,
            decode_capture_batch_sizes=[1],
            max_context_batch_size=1,
            kv_cache_layer_to_group=[2],
            kv_cache_group_num=3,
        )
        # Reuse the same captured graph with two different draft allocations.
        for first_page in (50, 72):
            with self.subTest(first_page=first_page):
                inputs = self._draft_inputs(first_page)
                inputs.attention_inputs = bind_model_cache_inputs(
                    model, inputs.attention_inputs, 1
                )
                self.assertTrue(runner.canRun(inputs))
                output = runner.forward(inputs)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output.hidden_states,
                    torch.full_like(output.hidden_states, first_page),
                )

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
        inputs.input_ids = torch.arange(batch_size, dtype=torch.int32, device="cuda")
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
        attention.kv_cache_block_id_host = attention.kv_cache_kernel_block_id_host
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
        with mock.patch.object(
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

    def test_ktp_idle_rank_keeps_target_verify_physical_rows(self):
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
        replay_inputs.ktp_local_real_batch = 0
        replay_inputs.ktp_common_physical_batch = 1
        replay_inputs.ktp_use_cuda_graph = True
        replay_inputs.ktp_valid_row_mask = torch.zeros(
            4, dtype=torch.int32, device="cuda"
        )

        self.assertTrue(runner.canRun(replay_inputs))
        outputs = runner.forward(replay_inputs)
        torch.cuda.synchronize()

        self.assertEqual(tuple(outputs.hidden_states.shape), (4, 16))

    def test_ktp_ordinary_decode_still_trims_to_owner_local_rows(self):
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
        replay_inputs = self._build_decode_replay_inputs([126, 255])
        replay_inputs.ktp_local_real_batch = 1
        replay_inputs.ktp_common_physical_batch = 2
        replay_inputs.ktp_use_cuda_graph = True
        replay_inputs.ktp_valid_row_mask = torch.tensor(
            [1, 0], dtype=torch.int32, device="cuda"
        )

        self.assertTrue(runner.canRun(replay_inputs))
        outputs = runner.forward(replay_inputs)
        torch.cuda.synchronize()

        self.assertEqual(tuple(outputs.hidden_states.shape), (1, 16))

    def test_decode_batch_above_capture_range_falls_back_to_eager(self):
        model = _SequenceHostProbeModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            hidden_size=16,
            max_seq_len=384,
            tokens_per_block=64,
            kernel_tokens_per_block=64,
            decode_capture_batch_sizes=[1],
            max_context_batch_size=2,
        )
        replay_inputs = self._build_decode_replay_inputs([126, 255])

        self.assertFalse(runner.canRun(replay_inputs))

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
