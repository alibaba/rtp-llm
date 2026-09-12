import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import triton
import triton.language as tl

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
    _TokenSpeedDecodeMetadata,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import (
    LinearReplayInputs,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    rtp_llm_ops,
)


_LINEAR_REPLAY_FIELDS = (
    "slot_ids",
    "slot_generations",
    "active_block_ids",
    "prev_accept_lengths",
    "history_valid_lengths",
    "history_epochs",
    "verify_epochs",
    "init_kinds",
    "state_read_block_ids",
    "anchor_processed_lengths",
)


@triton.jit
def _visit_live_replay_slots(Slots, Visits):
    slot = tl.load(Slots + tl.program_id(0))
    if slot >= 0:
        tl.atomic_add(Visits + slot, 1)


class _LinearReplayProbeModel:
    def __init__(self, steps, groups):
        self.steps = steps
        self.groups = groups
        self.live_slot_visits = torch.zeros(32, dtype=torch.int32, device="cuda")
        self.outputs_by_batch = {}
        self.capture_dtypes = []
        self.prepared_snapshots = []

    @staticmethod
    def pack(replay):
        columns = []
        for name in _LINEAR_REPLAY_FIELDS:
            value = getattr(replay, name)
            columns.append(value.T if value.ndim == 2 else value.unsqueeze(1))
        return torch.cat([value.to(torch.int64) for value in columns], dim=1)

    def prepare_fmha_impl(self, inputs, _is_cuda_graph):
        replay = inputs.attention_inputs.linear_replay
        self.capture_dtypes.append(
            tuple(getattr(replay, name).dtype for name in _LINEAR_REPLAY_FIELDS)
        )

        def prepare_cuda_graph(attention):
            self.prepared_snapshots.append(self.pack(attention.linear_replay))

        return SimpleNamespace(prepare_cuda_graph=prepare_cuda_graph)

    def forward(self, inputs, _fmha_impl=None):
        replay = inputs.attention_inputs.linear_replay
        batch = replay.slot_ids.numel()
        _visit_live_replay_slots[(batch,)](replay.slot_ids, self.live_slot_visits)
        metadata = self.pack(replay)
        padding = torch.zeros(
            (batch, 16 - metadata.shape[1]), dtype=torch.int64, device="cuda"
        )
        output = torch.cat((metadata, padding), dim=1).repeat_interleave(
            self.steps, dim=0
        )
        self.outputs_by_batch[batch] = output
        # Runner output storage follows the model's FP16 dtype. Retain the
        # complete int64 probe separately and emit exactly representable bits.
        return PyModelOutputs(output.remainder(1024).to(inputs.input_hiddens.dtype))


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
                    current_attention.prefix_lengths_host,
                    current_attention.input_lengths_host.tolist(),
                    current_attention.sequence_lengths_host,
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
        metadata.prepare_cuda_graph = prepare_cuda_graph
        return metadata

    def forward(self, inputs, _fmha_impl=None):
        if self.initial_sequence_lengths_plus_1 is None:
            self.initial_sequence_lengths_plus_1 = (
                inputs.attention_inputs.sequence_lengths_plus_1_d.cpu()
            )
        # Keep a real CUDA operation in the graph while avoiding model weights.
        return PyModelOutputs(inputs.input_hiddens + 1)


class _DevicePlannerProbeModel:
    def __init__(self):
        self.metadata_by_batch = {}
        self.capture_table_widths = []
        self.replay_pointers = []

    @staticmethod
    def pointers(metadata):
        return tuple(
            tensor.data_ptr()
            for tensor in (
                metadata.positions_d,
                metadata.batch_indice_d,
                metadata.slot_mapping,
                metadata.block_tables,
                metadata.seq_lens,
            )
        )

    def prepare_fmha_impl(self, inputs, _is_cuda_graph):
        attention = inputs.attention_inputs
        batch = attention.prefix_lengths.numel()
        metadata = _TokenSpeedDecodeMetadata(
            64, batch, 384, True, attention.prefix_lengths.device
        )
        metadata.plan_device(
            attention.prefix_lengths, attention.kv_cache_kernel_block_id_device, 4
        )
        self.metadata_by_batch[batch] = metadata
        self.capture_table_widths.append(
            attention.kv_cache_kernel_block_id_device.size(1)
        )

        def prepare_cuda_graph(current):
            if current.prefix_lengths_host is not None:
                raise AssertionError("target prefix host mirror must be absent")
            if current.sequence_lengths_host is not None:
                raise AssertionError("target sequence host mirror must be absent")
            metadata.plan_device(
                current.prefix_lengths,
                current.kv_cache_kernel_block_id_device,
                4,
                forbid_realloc=True,
            )
            self.replay_pointers.append(self.pointers(metadata))

        return SimpleNamespace(metadata=metadata, prepare_cuda_graph=prepare_cuda_graph)

    def forward(self, inputs, fmha_impl=None):
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs, True)
        metadata = fmha_impl.metadata
        values = torch.stack(
            (
                metadata.positions_d,
                metadata.seq_lens.repeat_interleave(4),
                metadata.slot_mapping,
                metadata.batch_indice_d,
            ),
            dim=1,
        ).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(
            torch.cat((values, torch.zeros_like(inputs.input_hiddens[:, 4:])), dim=1)
        )


class _SequenceHostProbeModel:
    def __init__(self) -> None:
        self.capture_sequence_lengths = []
        self.replay_sequence_lengths = []
        self.replay_sequence_lengths_device = []
        self.replay_sequence_lengths_plus_1 = []

    def prepare_fmha_impl(self, inputs, _is_cuda_graph):
        sequence_lengths_host = inputs.attention_inputs.sequence_lengths_host
        self.capture_sequence_lengths.append(
            None
            if sequence_lengths_host is None
            else (sequence_lengths_host.tolist(), sequence_lengths_host.is_pinned())
        )
        metadata = SimpleNamespace()

        def prepare_cuda_graph(current_attention):
            current_sequence_lengths_host = current_attention.sequence_lengths_host
            self.replay_sequence_lengths.append(
                None
                if current_sequence_lengths_host is None
                else (
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
    # Keep the production TokenSpeed device metadata planner;
    # omit model weights and the architecture-specific attention kernel.
    planner = TokenSpeedMlaDecodeImpl.__new__(TokenSpeedMlaDecodeImpl)
    planner.seq_size_per_block = 64
    planner.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
    planner.fmha_impl = tokenspeed_mla_impl._TokenSpeedDecodeMetadata(
        64,
        attention.input_lengths.numel(),
        384,
        is_cuda_graph,
        torch.device("cuda", torch.cuda.current_device()),
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

    def test_linear_replay_metadata_survives_padding_and_batch_changes(self):
        steps, groups = 4, 3
        model = _LinearReplayProbeModel(steps, groups)
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            hidden_size=16,
            max_seq_len=384,
            tokens_per_block=64,
            kernel_tokens_per_block=64,
            decode_capture_batch_sizes=[8],
            num_tokens_per_bs=steps,
            is_target_verify=True,
            max_context_batch_size=8,
            linear_replay_group_num=groups,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            model.live_slot_visits, torch.zeros_like(model.live_slot_visits)
        )
        expected_dtypes = tuple(
            (
                torch.int64
                if name in ("slot_generations", "history_epochs", "verify_epochs")
                else torch.int32
            )
            for name in _LINEAR_REPLAY_FIELDS
        )
        self.assertTrue(model.capture_dtypes)
        self.assertTrue(
            all(dtypes == expected_dtypes for dtypes in model.capture_dtypes)
        )
        visits = torch.zeros(32, dtype=torch.int32)
        for round_id, slots in enumerate(([3, 7], [9], [11, 3]), start=1):
            batch = len(slots)
            inputs = self._build_replay_inputs(batch, steps)
            replay = LinearReplayInputs()
            for field, dtype in zip(_LINEAR_REPLAY_FIELDS, expected_dtypes):
                if field in ("active_block_ids", "state_read_block_ids"):
                    backing = torch.full(
                        (groups, batch + 4), -77, dtype=dtype, device="cuda"
                    )
                    tensor = backing[:, 1 : batch + 1]
                    tensor.copy_(
                        torch.arange(groups * batch, dtype=dtype, device="cuda")
                        .reshape(groups, batch)
                        .add_(
                            1000 * round_id
                            + (100 if field == "active_block_ids" else 200)
                        )
                    )
                    self.assertFalse(tensor.is_contiguous())
                else:
                    offset = 100 * round_id + _LINEAR_REPLAY_FIELDS.index(field)
                    if dtype == torch.int64:
                        offset += 1 << 42
                    tensor = torch.arange(batch, dtype=dtype, device="cuda") + offset
                setattr(replay, field, tensor)
            replay.slot_ids.copy_(torch.tensor(slots, dtype=torch.int32, device="cuda"))
            replay.prev_accept_lengths.fill_(3)
            replay.history_valid_lengths.fill_(steps)
            replay.init_kinds.zero_()
            inputs.attention_inputs.linear_replay = replay
            expected = model.pack(replay).clone()
            self.assertTrue(runner.canRun(inputs))
            self.assertEqual(runner.getCurrentRealGraphSize(), 8)
            output = runner.forward(inputs).hidden_states
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output.reshape(batch, steps, 16)[:, 0, : expected.shape[1]],
                expected.remainder(1024).to(output.dtype),
                rtol=0,
                atol=0,
            )
            prepared = model.prepared_snapshots[-1]
            torch.testing.assert_close(prepared[:batch], expected, rtol=0, atol=0)
            captured_output = model.outputs_by_batch[8].reshape(8, steps, 16)[:, 0]
            torch.testing.assert_close(
                captured_output[:batch, : expected.shape[1]], expected, rtol=0, atol=0
            )
            padded = captured_output[batch:]
            # Slot -1 masks padding even when ignored group/epoch fields retain
            # their previous backing values. No stale request may visit a slot.
            self.assertTrue(torch.all(padded[:, 0] == -1).item())
            self.assertTrue(torch.all(padded[:, 5:7] == 0).item())
            self.assertTrue(torch.all(padded[:, 9] == 0).item())
            for slot in slots:
                visits[slot] += 1
            torch.testing.assert_close(
                model.live_slot_visits.cpu(), visits, rtol=0, atol=0
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

    def test_decode_host_mirror_recovers_after_device_only_round_and_batch_shrink(self):
        model = _SequenceHostProbeModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            hidden_size=16,
            max_seq_len=384,
            tokens_per_block=64,
            kernel_tokens_per_block=64,
            decode_capture_batch_sizes=[8],
            max_context_batch_size=8,
        )

        # Reuse one graph throughout: a missing source invalidates its prior
        # host mirror, and a smaller restored batch must clear the old rows.
        rounds = (
            ([126, 255, 63, 191], True),
            ([127, 256, 64, 192], False),
            ([31, 95], True),
        )
        for lengths, has_host_mirror in rounds:
            with self.subTest(lengths=lengths, has_host_mirror=has_host_mirror):
                inputs = self._build_decode_replay_inputs(lengths)
                if not has_host_mirror:
                    # Pybind accepts an empty source tensor; a cleared C++
                    # destination is exposed to the planner as None.
                    inputs.attention_inputs.sequence_lengths_host = torch.empty(
                        0, dtype=torch.int32
                    )
                self.assertTrue(runner.canRun(inputs))
                self.assertEqual(runner.getCurrentRealGraphSize(), 8)
                outputs = runner.forward(inputs)
                torch.cuda.synchronize()

                padding = 8 - len(lengths)
                expected_host = (
                    (lengths + [0] * padding, True) if has_host_mirror else None
                )
                self.assertEqual(model.replay_sequence_lengths[-1], expected_host)
                torch.testing.assert_close(
                    model.replay_sequence_lengths_device[-1].cpu(),
                    torch.tensor(lengths + [0] * padding, dtype=torch.int32),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    model.replay_sequence_lengths_plus_1[-1].cpu(),
                    torch.tensor(
                        [length + 1 for length in lengths] + [1] * padding,
                        dtype=torch.int32,
                    ),
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
                None,
                [4] * 8,
                None,
            ),
        )
        expected_device_metadata = (
            [4] * 8,
            [126, 255] + [0] * 6,
            [126, 255] + [0] * 6,
            [127, 256] + [1] * 6,
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
        # A valid allocator reserves the next page for prefix 381 + query 4.
        # The capacity decision must depend on this host-known shape only.
        replay_inputs.attention_inputs.kv_cache_kernel_block_id_device = torch.zeros(
            (1, 7), dtype=torch.int32, device="cuda"
        )
        replay_inputs.attention_inputs.prefix_lengths_host = torch.empty(0, dtype=torch.int32)
        replay_inputs.attention_inputs.sequence_lengths_host = torch.empty(0, dtype=torch.int32)

        self.assertFalse(runner.canRun(replay_inputs))

    def test_runner_accepts_device_only_lengths_and_rejects_unsafe_tables(self):
        model = _MetadataProbeModel()
        runner = CudaGraphRunner()
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
            kv_cache_group_num=2,
        )
        replay_inputs = self._build_replay_inputs(batch_size=2, q_len=4)
        attention = replay_inputs.attention_inputs
        # Allow a row stride larger than the live width, as group views and
        # narrowed runtime allocation tables need not be fully contiguous.
        table = torch.arange(24, dtype=torch.int32, device="cuda").reshape(2, 12)
        table = table[:, :6]
        attention.kv_cache_kernel_block_id_device = table
        attention.kv_cache_kernel_block_id_device_by_group = [table, table]
        # PyWrappedModel represents target verification as context: sequence
        # lengths are empty and decode-only cumulative lengths are absent.
        attention.sequence_lengths = torch.empty(0, dtype=torch.int32, device="cuda")
        attention.decode_cu_seqlens_d = torch.empty(0, dtype=torch.int32, device="cuda")
        for host_value in (torch.empty(0, dtype=torch.int32), attention.prefix_lengths):
            with self.subTest(host_is_device=host_value.is_cuda):
                # A CUDA tensor in a legacy host field must never cause D2H.
                attention.prefix_lengths_host = host_value
                attention.sequence_lengths_host = host_value
                self.assertTrue(runner.canRun(replay_inputs))
                runner.forward(replay_inputs)
                torch.cuda.synchronize()
                self.assertEqual(model.replay_host_metadata[-1], (None, [4] * 8, None))
                torch.testing.assert_close(
                    model.replay_device_metadata[-1][1][:2],
                    attention.prefix_lengths,
                    rtol=0,
                    atol=0,
                )

        bad_tables = (
            torch.zeros((2, 7), dtype=torch.int32, device="cuda"),
            torch.zeros((9, 6), dtype=torch.int32, device="cuda"),
            torch.zeros((2, 12), dtype=torch.int32, device="cuda")[:, ::2],
            torch.zeros((2, 6), dtype=torch.int64, device="cuda"),
        )
        for bad_table in bad_tables:
            with self.subTest(shape=bad_table.shape, stride=bad_table.stride()):
                # Group 0 still fits. Every other group must be checked before
                # prepare can enqueue an oversized strided D2D copy.
                attention.kv_cache_kernel_block_id_device_by_group = [table, bad_table]
                self.assertFalse(runner.canRun(replay_inputs))
        attention.kv_cache_kernel_block_id_device_by_group = [table]
        self.assertFalse(runner.canRun(replay_inputs))

        attention.kv_cache_kernel_block_id_device_by_group = [table, table]
        attention.kv_cache_kernel_block_id_host = torch.zeros((9, 6), dtype=torch.int32)
        self.assertFalse(runner.canRun(replay_inputs))

    def test_device_planner_survives_cpp_graph_padding_and_live_prefix_changes(self):
        model = _DevicePlannerProbeModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            hidden_size=16,
            max_seq_len=384,
            # Physical-page rounding gives eight capture columns for six
            # context pages, exercising the same overcapacity as SP reserve.
            tokens_per_block=256,
            kernel_tokens_per_block=64,
            decode_capture_batch_sizes=[8],
            num_tokens_per_bs=4,
            is_target_verify=True,
            max_context_batch_size=8,
        )
        self.assertTrue(all(width == 8 for width in model.capture_table_widths))
        metadata = model.metadata_by_batch[8]
        self.assertEqual(metadata.block_tables.shape, (8, 6))
        pointers = model.pointers(metadata)

        for prefixes, first_page in (([126, 189], 1), ([61], 9)):
            with self.subTest(prefixes=prefixes):
                batch = len(prefixes)
                inputs = self._build_replay_inputs(batch, 4)
                attention = inputs.attention_inputs
                prefix = torch.tensor(prefixes, dtype=torch.int32)
                table = torch.arange(batch * 4, dtype=torch.int32).reshape(batch, 4)
                table += first_page
                attention.prefix_lengths = prefix.cuda()
                attention.prefix_lengths_host = torch.empty(0, dtype=torch.int32)
                attention.sequence_lengths = torch.empty(
                    0, dtype=torch.int32, device="cuda"
                )
                attention.sequence_lengths_host = torch.empty(0, dtype=torch.int32)
                attention.input_lengths_host = torch.empty(0, dtype=torch.int32)
                attention.sequence_lengths_plus_1_d = (prefix + 1).cuda()
                attention.decode_cu_seqlens_d = torch.empty(0, dtype=torch.int32, device="cuda")
                attention.kv_cache_kernel_block_id_device = table.cuda()
                attention.kv_cache_kernel_block_id_host = table.pin_memory()
                with mock.patch.object(
                    torch.Tensor, "cpu", side_effect=AssertionError("D2H")
                ), mock.patch.object(
                    torch.Tensor, "tolist", side_effect=AssertionError("host read")
                ), mock.patch.object(
                    torch.Tensor, "item", side_effect=AssertionError("host scalar")
                ):
                    self.assertTrue(runner.canRun(inputs))
                    outputs = runner.forward(inputs)
                torch.cuda.synchronize()
                self.assertEqual(runner.getCurrentRealGraphSize(), 8)
                self.assertEqual(model.replay_pointers[-1], pointers)
                positions = (prefix[:, None] + torch.arange(4)).flatten()
                rows = torch.arange(batch).repeat_interleave(4)
                slots = table[rows, positions // 64] * 64 + positions % 64
                expected = torch.stack(
                    (positions, (prefix + 4).repeat_interleave(4), slots, rows), dim=1
                ).to(torch.float16)
                torch.testing.assert_close(
                    outputs.hidden_states[:, :4].cpu(), expected, rtol=0, atol=0
                )
                expected_table = torch.zeros((8, 6), dtype=torch.int32)
                for row, length in enumerate(prefixes):
                    pages = (length + 4 + 63) // 64
                    expected_table[row, :pages] = table[row, :pages]
                torch.testing.assert_close(
                    metadata.block_tables.cpu(), expected_table, rtol=0, atol=0
                )

    def test_runner_conservatively_falls_back_at_partial_capture_page(self):
        model = _MetadataProbeModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            hidden_size=16,
            max_seq_len=383,
            tokens_per_block=64,
            kernel_tokens_per_block=64,
            decode_capture_batch_sizes=[1],
            num_tokens_per_bs=4,
            is_target_verify=True,
            max_context_batch_size=1,
        )
        replay_inputs = self._build_replay_inputs(batch_size=1, q_len=4)
        replay_inputs.attention_inputs.prefix_lengths_host = torch.empty(0, dtype=torch.int32)
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
