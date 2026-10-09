import copy
import gc
import os
import unittest
import weakref
from typing import Optional

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import (
    CudaGraphRunner,
    DirtyCudaGraphCaptureError,
)
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    CudaGraphSelectionMode,
)
from rtp_llm.ops.compute_ops import (
    BertEmbeddingInputs,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
)

GROUP_TAGS = ["full", "aux"]
HIDDEN_SIZE = 4
TOKENS_PER_BLOCK = 8
KERNEL_BLOCK_TABLE_WIDTH = 8


class TaggedBlockTableModel:
    """Small graph-safe model whose output exposes both tag-local block tables."""

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: str | None = None,
    ):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        full_id = attention_inputs["full"].kv_cache_kernel_block_id_device[0, 0]
        aux_id = attention_inputs["aux"].kv_cache_kernel_block_id_device[0, 0]
        signature = (full_id + 16 * aux_id).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + signature)


class TaggedSequenceLengthModel:
    """Expose the cumulative lengths used by a tagged captured graph."""

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        full_inputs = inputs.attention_inputs["full"]
        signature = torch.stack(
            (
                full_inputs.cu_seqlens_device[-1],
                full_inputs.cu_kv_seqlens_device[-1],
                full_inputs.input_lengths_device.sum(),
                full_inputs.sequence_lengths_plus_1_device[-1],
            )
        ).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + signature)


class TaggedDecodePaddingModel:
    """Expose metadata that must describe rounded decode rows as safe dummies."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        full_inputs = inputs.attention_inputs["full"]
        signature = torch.stack(
            (
                full_inputs.sequence_lengths_plus_1_device.sum(),
                full_inputs.sequence_lengths_plus_1_device[-1],
                full_inputs.decode_cu_seqlens_device[-1],
                full_inputs.decode_cu_seqlens_device[-2],
            )
        ).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + signature)


class StaticInputTailModel:
    """Expose stale hidden rows retained by a reused graph input buffer."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        tail_signature = inputs.input_hiddens[-1].sum()
        return PyModelOutputs(inputs.input_hiddens + tail_signature)


class StaticTokenMetadataTailModel:
    """Expose stale token, MRoPE-position and hidden rows after graph shrink."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        tail_signature = torch.stack(
            (
                inputs.input_ids[-1] + inputs.input_hiddens[-1].sum().to(torch.int32),
                inputs.combo_position_ids[-3],
                inputs.combo_position_ids[-2],
                inputs.combo_position_ids[-1],
            )
        ).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + tail_signature)


class AuxiliaryOutputModel:
    """Return a second output whose view must follow the selected graph bucket."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        hidden_states = inputs.input_hiddens + 1
        target_features = torch.cat(
            (inputs.input_hiddens + 2, inputs.input_hiddens + 3), dim=1
        )
        return PyModelOutputs(hidden_states, target_features)


class FailOnceGraphPrepare:
    def __init__(self) -> None:
        self.call_count = 0

    def prepare_cuda_graph(self, inputs) -> None:
        self.call_count += 1
        if self.call_count == 1:
            raise RuntimeError("injected prepare_cuda_graph failure")


class RetryableGraphPrepareModel(TaggedBlockTableModel):
    def __init__(self) -> None:
        self.graph_prepare = FailOnceGraphPrepare()

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return self.graph_prepare


class CapturedHostLengthModel:
    """Model a HIP Graph kernel that directly consumes mapped host metadata."""

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        # FusedRopeKVCacheDecodeOp on ROCm passes the captured pinned-host
        # input/sequence-length pointers directly to a HIP kernel. A captured
        # non-blocking H2D models the same lifetime contract without tying this
        # runner regression to one attention implementation.
        first_length = inputs.attention_inputs["full"].input_lengths.to(
            device=inputs.input_hiddens.device, non_blocking=True
        )[0]
        return PyModelOutputs(inputs.input_hiddens + first_length)


class TextOnlyMultimodalCapableModel:
    """Qwen3-VL-like graph model used without request-side multimodal payload."""

    input_hiddens_numel = -1
    cuda_graph_selection_mode = None

    def __init__(self) -> None:
        self.bert_buffer_shapes: list[tuple[int, int]] = []

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: str | None = None,
    ):
        self.cuda_graph_selection_mode = cuda_graph_selection_mode
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        self.input_hiddens_numel = inputs.input_hiddens.numel()
        position_ids = inputs.bert_embedding_inputs.combo_position_ids
        token_type_ids = inputs.bert_embedding_inputs.combo_tokens_type_ids
        self.bert_buffer_shapes.append(
            (
                0 if position_ids is None else position_ids.numel(),
                0 if token_type_ids is None else token_type_ids.numel(),
            )
        )
        # Qwen3-VL reads these fields on every forward. They stay empty for a
        # pure-text request, which must not make model-level eligibility fail.
        _ = inputs.embedding_inputs.text_tokens_mask
        _ = inputs.multimodal_inputs.multimodal_features
        _ = inputs.multimodal_inputs.mm_features_locs
        _ = inputs.multimodal_inputs.mm_extra_input
        token_values = inputs.input_ids.to(torch.bfloat16).unsqueeze(1)
        if (
            inputs.combo_position_ids is not None
            and inputs.combo_position_ids.numel() > 0
        ):
            token_values = token_values + inputs.combo_position_ids.view(-1, 3)[:, :1]
        hidden_offsets = torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device=inputs.input_ids.device
        ).unsqueeze(0)
        return PyModelOutputs(token_values * 10 + hidden_offsets)


class BertWeightAwareModel(TextOnlyMultimodalCapableModel):
    """Expose both fixed embedding tables and request-owned dynamic IDs."""

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        base = super().forward(inputs, fmha_impl).hidden_states
        position_bias = inputs.bert_embedding_inputs.position_encoding.index_select(
            0, inputs.bert_embedding_inputs.combo_position_ids.to(torch.int64)
        )
        token_type_bias = (
            inputs.bert_embedding_inputs.token_type_embedding.index_select(
                0, inputs.bert_embedding_inputs.combo_tokens_type_ids.to(torch.int64)
            )
        )
        return PyModelOutputs(base + position_bias + token_type_bias)


class PrefillPaddingBlockTableModel(TextOnlyMultimodalCapableModel):
    """Make any non-zero padding-row block id visible in every real output."""

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        base = super().forward(inputs, fmha_impl).hidden_states
        padding_signature = (
            inputs.attention_inputs["full"].kv_cache_kernel_block_id_device[-1].sum()
        )
        return PyModelOutputs(base + padding_signature.to(base.dtype))


class _InjectablePrepareImpl:
    def __init__(self) -> None:
        self.fail = False

    def prepare_cuda_graph(self, _attention_inputs) -> None:
        if self.fail:
            raise RuntimeError("injected prepare failure")


class InjectablePrepareFailureModel(TextOnlyMultimodalCapableModel):
    def __init__(self) -> None:
        super().__init__()
        self.impl = _InjectablePrepareImpl()

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: str | None = None,
    ):
        return self.impl


class InjectableCaptureBodyFailureModel(TextOnlyMultimodalCapableModel):
    """Raise only after graph capture has begun (the fourth forward call)."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        self.forward_calls += 1
        # The datatype probe and two eager warmups run first; call four is
        # inside graphCaptureBegin()/capture_end().
        if self.forward_calls == 4:
            raise RuntimeError("injected capture-body failure")
        return super().forward(inputs, fmha_impl)


class TaggedBlockRowModel:
    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        full = inputs.attention_inputs["full"].kv_cache_kernel_block_id_device
        aux = inputs.attention_inputs["aux"].kv_cache_kernel_block_id_device
        signature = (full.sum(dim=1) + 16 * aux.sum(dim=1)).to(
            inputs.input_hiddens.dtype
        )
        return PyModelOutputs(inputs.input_hiddens + signature.unsqueeze(1))


class DraftBlockTableModel:
    """Only draft owns layers; compatibility tags must not become consumers."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        attention = inputs.attention_inputs
        if isinstance(attention, dict):
            attention = attention["draft"]
        signature = attention.kv_cache_kernel_block_id_device[:, 0].to(
            inputs.input_hiddens.dtype
        )
        return PyModelOutputs(inputs.input_hiddens + signature.unsqueeze(1))


def _tag_attention_inputs(
    common: PyAttentionInputs, tags: list[str], values: dict[str, int]
) -> dict[str, PyAttentionInputs]:
    tagged = {}
    for tag in tags:
        tag_inputs = copy.copy(common)
        host_blocks = torch.full_like(
            common.kv_cache_kernel_block_id, values[tag], device="cpu"
        ).pin_memory()
        device_blocks = host_blocks.cuda()
        tag_inputs.kv_cache_kernel_block_id = host_blocks
        tag_inputs.kv_cache_kernel_block_id_device = device_blocks
        tag_inputs.kv_cache_block_id = host_blocks
        tag_inputs.kv_cache_block_id_device = device_blocks
        tagged[tag] = tag_inputs
    return tagged


def _build_common_inputs(
    attention_inputs: PyAttentionInputs,
    tags: list[str],
    values: dict[str, int],
    batch_size: int,
    token_count: int,
    block_count: int,
) -> PyModelInputs:
    inputs = PyModelInputs()
    inputs.input_ids = torch.arange(token_count, dtype=torch.int32, device="cuda")
    inputs.input_hiddens = torch.zeros(
        (token_count, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    )

    attention_inputs.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
    attention_inputs.padding_offset = torch.zeros(
        token_count, dtype=torch.int32, device="cuda"
    )
    attention_inputs.total_tokens = token_count
    attention_inputs.kv_cache_kernel_block_id = torch.zeros(
        (batch_size, block_count), dtype=torch.int32
    ).pin_memory()
    attention_inputs.kv_cache_kernel_block_id_device = (
        attention_inputs.kv_cache_kernel_block_id.cuda()
    )
    attention_inputs.kv_cache_block_id = attention_inputs.kv_cache_kernel_block_id
    attention_inputs.kv_cache_block_id_device = (
        attention_inputs.kv_cache_kernel_block_id_device
    )
    inputs.attention_inputs = _tag_attention_inputs(attention_inputs, tags, values)
    return inputs


def _with_mrope_positions(inputs: PyModelInputs) -> PyModelInputs:
    token_count = inputs.input_ids.numel()
    inputs.combo_position_ids = (
        torch.arange(token_count, dtype=torch.int32, device="cuda")
        .unsqueeze(1)
        .expand(-1, 3)
        .contiguous()
    )
    return inputs


def _build_decode_inputs(
    tags: list[str],
    values: dict[str, int],
    batch_size: int = 2,
    block_count: int = 1,
) -> PyModelInputs:
    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = False
    attention_inputs.is_target_verify = False
    attention_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
    attention_inputs.input_lengths = torch.ones(
        batch_size, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths = torch.ones(
        batch_size, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths_plus_1_device = torch.full(
        (batch_size,), 2, dtype=torch.int32, device="cuda"
    )
    attention_inputs.decode_cu_seqlens_device = torch.arange(
        batch_size + 1, dtype=torch.int32, device="cuda"
    )
    attention_inputs.cu_seqlens = torch.zeros(
        batch_size + 1, dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens_device = attention_inputs.cu_seqlens.cuda()
    attention_inputs.cu_kv_seqlens_device = torch.zeros_like(
        attention_inputs.cu_seqlens_device
    )
    attention_inputs.context_total_kv_length = batch_size
    return _build_common_inputs(
        attention_inputs,
        tags,
        values,
        batch_size=batch_size,
        token_count=batch_size,
        block_count=block_count,
    )


def _build_prefill_inputs(
    tags: list[str], values: dict[str, int], seq_len: int | list[int] = 4
) -> PyModelInputs:
    seq_lens = [seq_len] if isinstance(seq_len, int) else seq_len
    token_count = sum(seq_lens)
    cu_seqlens = [0]
    for length in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + length)

    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = True
    attention_inputs.is_target_verify = False
    attention_inputs.input_lengths = torch.tensor(
        seq_lens, dtype=torch.int32
    ).pin_memory()
    attention_inputs.prefix_lengths = torch.zeros(
        len(seq_lens), dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens = torch.tensor(
        cu_seqlens, dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens_device = attention_inputs.cu_seqlens.cuda()
    attention_inputs.cu_kv_seqlens_device = attention_inputs.cu_seqlens_device.clone()
    attention_inputs.context_total_kv_length = token_count
    return _build_common_inputs(
        attention_inputs,
        tags,
        values,
        batch_size=len(seq_lens),
        token_count=token_count,
        block_count=max(1, (max(seq_lens) + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK),
    )


def _build_target_verify_inputs(
    tags: list[str],
    values: dict[str, int],
    batch_size: int = 1,
    query_len: int = 5,
    prefix_len: int = 11,
    is_prefill: bool = True,
) -> PyModelInputs:
    token_count = batch_size * query_len

    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = is_prefill
    attention_inputs.is_target_verify = True
    attention_inputs.input_lengths = torch.full(
        (batch_size,), query_len, dtype=torch.int32
    ).pin_memory()
    attention_inputs.prefix_lengths = torch.full(
        (batch_size,), prefix_len, dtype=torch.int32
    ).pin_memory()
    attention_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
    attention_inputs.sequence_lengths_plus_1_device = (
        attention_inputs.prefix_lengths.cuda() + 1
    )

    cu_q = torch.arange(0, token_count + 1, query_len, dtype=torch.int32).pin_memory()
    attention_inputs.cu_seqlens = cu_q
    attention_inputs.cu_seqlens_device = cu_q.cuda()
    attention_inputs.cu_kv_seqlens_device = torch.arange(
        0,
        batch_size * (query_len + prefix_len) + 1,
        query_len + prefix_len,
        dtype=torch.int32,
        device="cuda",
    )
    attention_inputs.decode_cu_seqlens = torch.arange(
        batch_size + 1, dtype=torch.int32
    ).pin_memory()
    attention_inputs.decode_cu_seqlens_device = (
        attention_inputs.decode_cu_seqlens.cuda()
    )

    attention_inputs.context_total_kv_length = batch_size * (query_len + prefix_len)

    block_count = (prefix_len + query_len + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
    return _build_common_inputs(
        attention_inputs,
        tags,
        values,
        batch_size=batch_size,
        token_count=token_count,
        block_count=block_count,
    )


class TestCudaGraphTaggedCache(unittest.TestCase):
    def test_draft_placeholder_map_preserves_eager_graph_contract(self):
        tags = ["unused", "draft", "other"]
        model = DraftBlockTableModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [2],
            tags,
        )
        for value, order in ((2, tags), (5, list(reversed(tags)))):
            inputs = _build_decode_inputs(
                order, {"unused": 31, "draft": value, "other": 47}
            )
            expected = model.forward(inputs).hidden_states.clone()
            self.assertTrue(runner.canRun(inputs))
            output = runner.forward(inputs)
            torch.cuda.synchronize()
            torch.testing.assert_close(output.hidden_states, expected)
            torch.testing.assert_close(
                output.hidden_states, torch.full_like(output.hidden_states, value)
            )
        self.assertFalse(
            runner.canRun(
                _build_decode_inputs(["draft", "other"], {"draft": 2, "other": 47})
            )
        )
        self.assertFalse(
            runner.canRun(
                _build_decode_inputs(
                    tags + ["extra"],
                    {"unused": 31, "draft": 2, "other": 47, "extra": 99},
                )
            )
        )

    def test_single_draft_group_uses_direct_graph_inputs(self):
        model = DraftBlockTableModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [2],
            ["draft"],
        )
        for value in (2, 5):
            inputs = _build_decode_inputs(["draft"], {"draft": value})
            inputs.attention_inputs = inputs.attention_inputs["draft"]
            expected = model.forward(inputs).hidden_states.clone()
            self.assertTrue(runner.canRun(inputs))
            output = runner.forward(inputs)
            torch.cuda.synchronize()
            torch.testing.assert_close(output.hidden_states, expected)

    def test_decode_heterogeneous_block_width_replay(self) -> None:
        for tags in (GROUP_TAGS, list(reversed(GROUP_TAGS))):
            for bpk in (4, 128):
                with self.subTest(tags=tags, bpk=bpk):
                    runner = CudaGraphRunner()
                    runner.init_decode(
                        TaggedBlockRowModel(),
                        HIDDEN_SIZE,
                        TOKENS_PER_BLOCK,
                        bpk,
                        [2],
                        tags,
                    )
                    for value in (1, 2):
                        inputs = _build_decode_inputs(tags, {"full": 0, "aux": value})
                        tagged = inputs.attention_inputs
                        rows = torch.stack(
                            (
                                torch.full((bpk,), value, dtype=torch.int32),
                                torch.full((bpk,), value + 1, dtype=torch.int32),
                            )
                        ).pin_memory()
                        tagged["full"].kv_cache_kernel_block_id = rows
                        tagged["full"].kv_cache_kernel_block_id_device = rows.cuda()
                        inputs.attention_inputs = tagged
                        self.assertTrue(runner.canRun(inputs))
                        output = runner.forward(inputs)
                        torch.cuda.synchronize()
                        expected = (
                            torch.tensor(
                                [
                                    bpk * value + 16 * value,
                                    bpk * (value + 1) + 16 * value,
                                ],
                                dtype=output.hidden_states.dtype,
                                device="cuda",
                            )
                            .unsqueeze(1)
                            .expand_as(output.hidden_states)
                        )
                        torch.testing.assert_close(output.hidden_states, expected)

                    oversized = _build_decode_inputs(tags, {"full": 0, "aux": 1})
                    tagged = oversized.attention_inputs
                    rows = torch.ones((2, bpk + 1), dtype=torch.int32).pin_memory()
                    tagged["full"].kv_cache_kernel_block_id = rows
                    tagged["full"].kv_cache_kernel_block_id_device = rows.cuda()
                    oversized.attention_inputs = tagged
                    self.assertTrue(runner.canRun(oversized))
                    output = runner.forward(oversized)
                    torch.cuda.synchronize()
                    expected = torch.full_like(output.hidden_states, bpk + 16)
                    torch.testing.assert_close(output.hidden_states, expected)

    def _assert_replay_signature(
        self, runner: CudaGraphRunner, inputs: PyModelInputs, expected: int
    ) -> None:
        self.assertTrue(runner.canRun(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected_output = torch.full_like(output.hidden_states, expected)
        torch.testing.assert_close(output.hidden_states, expected_output)

    def test_prepare_sync_policy_is_role_scoped(self) -> None:
        # Run in a fresh process for each async environment: the production
        # helper intentionally caches these startup switches. Observe the
        # actual wait scopes, not elapsed time or unrelated copy synchronizations.
        stream_async = any(
            os.environ.get(name) == "1"
            for name in ("RTP_LLM_STREAM_ASYNC", "RTP_LLM_MTP_ASYNC_PREPARE")
        )
        for role in ("decode", "target_verify", "embedding", "generation"):
            runner = CudaGraphRunner()
            if role in ("decode", "target_verify"):
                runner.init_decode(
                    TaggedBlockTableModel(),
                    HIDDEN_SIZE,
                    TOKENS_PER_BLOCK,
                    KERNEL_BLOCK_TABLE_WIDTH,
                    [2],
                    GROUP_TAGS,
                    role == "target_verify",
                    2 if role == "target_verify" else 1,
                )
                inputs = (
                    _build_target_verify_inputs(
                        GROUP_TAGS,
                        {"full": 1, "aux": 2},
                        batch_size=2,
                        query_len=2,
                        prefix_len=1,
                    )
                    if role == "target_verify"
                    else _build_decode_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
                )
            else:
                if role == "generation":
                    runner.init_generation_prefill(
                        TextOnlyMultimodalCapableModel(),
                        2,
                        TOKENS_PER_BLOCK,
                        KERNEL_BLOCK_TABLE_WIDTH,
                        [4],
                        HIDDEN_SIZE,
                        GROUP_TAGS,
                        0,
                    )
                else:
                    runner.init_prefill(
                        TaggedBlockTableModel(),
                        2,
                        TOKENS_PER_BLOCK,
                        KERNEL_BLOCK_TABLE_WIDTH,
                        [4],
                        HIDDEN_SIZE,
                        GROUP_TAGS,
                    )
                inputs = _build_prefill_inputs(
                    GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=[2, 2]
                )

            for skip_sync in (False, True):
                with self.subTest(role=role, skip_sync=skip_sync):
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU]
                    ) as prof:
                        self.assertTrue(
                            runner.prepare(inputs, skip_forward_event_sync=skip_sync)
                        )
                    scopes = {event.key for event in prof.key_averages()}
                    prefix = "cuda_graph.prepareAttentionInputs("
                    self.assertEqual(
                        prefix + "wait_forward_event)" in scopes,
                        role != "generation" and (not skip_sync or not stream_async),
                    )
                    for event in (
                        "generation_prefill_wait_staging_event)",
                        "generation_prefill_wait_forward_event)",
                    ):
                        self.assertEqual(prefix + event in scopes, role == "generation")
                    torch.cuda.synchronize()

    def test_decode_tag_validation_and_replay_updates(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedBlockTableModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [2],
            GROUP_TAGS,
        )

        self._assert_replay_signature(
            runner,
            _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1}),
            18,
        )
        self._assert_replay_signature(
            runner,
            _build_decode_inputs(list(reversed(GROUP_TAGS)), {"full": 5, "aux": 3}),
            53,
        )

        self.assertFalse(runner.canRun(_build_decode_inputs(["full"], {"full": 2})))
        self.assertFalse(
            runner.canRun(
                _build_decode_inputs(
                    ["full", "aux", "extra"],
                    {"full": 2, "aux": 1, "extra": 9},
                )
            )
        )
        self.assertFalse(
            runner.canRun(
                _build_decode_inputs(["full", "wrong"], {"full": 2, "wrong": 1})
            )
        )

    def test_failed_async_prepare_is_retried_by_forward(self) -> None:
        model = RetryableGraphPrepareModel()
        runner = CudaGraphRunner()
        runner.init_decode(
            model,
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [2],
            GROUP_TAGS,
        )
        inputs = _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1})
        self.assertTrue(runner.canRun(inputs))

        with self.assertRaisesRegex(
            RuntimeError, "injected prepare_cuda_graph failure"
        ):
            runner.prepareAttentionInputs(inputs)
        self.assertEqual(model.graph_prepare.call_count, 1)

        output = runner.forward(inputs)
        torch.cuda.synchronize()
        self.assertEqual(model.graph_prepare.call_count, 2)
        torch.testing.assert_close(
            output.hidden_states,
            torch.full_like(output.hidden_states, 18),
        )

    def test_prefill_tagged_capture_and_replay_updates(self) -> None:
        runner = CudaGraphRunner()
        runner.init_prefill(
            TaggedBlockTableModel(),
            2,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [4],
            HIDDEN_SIZE,
            GROUP_TAGS,
        )

        self._assert_replay_signature(
            runner,
            _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2}),
            33,
        )
        self._assert_replay_signature(
            runner,
            _build_prefill_inputs(list(reversed(GROUP_TAGS)), {"full": 4, "aux": 3}),
            52,
        )

    def test_prefill_host_metadata_is_stable_without_device_sync(self) -> None:
        runner = CudaGraphRunner()
        runner.init_prefill(
            TaggedSequenceLengthModel(),
            2,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [4],
            HIDDEN_SIZE,
            GROUP_TAGS,
        )

        snapshots: list[tuple[torch.Tensor, int]] = []
        patterns = ([1], [1, 1], [1, 2], [2, 2]) * 4
        for seq_lens in patterns:
            inputs = _build_prefill_inputs(
                GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=list(seq_lens)
            )
            self.assertTrue(runner.prepare(inputs))
            output = runner.forward(inputs)
            # Keep the first outputs before the runner reuses its graph output
            # buffer, but intentionally do not synchronize the device between
            # prepare/replay iterations.
            snapshots.append((output.hidden_states.clone(), sum(seq_lens)))

        torch.cuda.synchronize()
        for snapshot, token_count in snapshots:
            expected_signature = torch.tensor(
                [token_count, token_count, token_count, 0],
                dtype=snapshot.dtype,
                device=snapshot.device,
            )
            torch.testing.assert_close(
                snapshot, expected_signature.unsqueeze(0).expand_as(snapshot)
            )

    @unittest.skipUnless(
        torch.version.hip is not None, "ROCm-specific host-pointer ABI"
    )
    @unittest.skipUnless(
        hasattr(torch.cuda, "_sleep"), "requires an asynchronous GPU delay"
    )
    def test_rocm_replay_protects_captured_host_metadata_without_user_sync(
        self,
    ) -> None:
        runner = CudaGraphRunner()
        runner.init_prefill(
            CapturedHostLengthModel(),
            2,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [4],
            HIDDEN_SIZE,
            GROUP_TAGS,
        )

        first_inputs = _build_prefill_inputs(
            GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=[1, 3]
        )
        second_inputs = _build_prefill_inputs(
            GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=[3, 1]
        )

        # Delay the first graph so an unsafe CPU overwrite of its captured
        # pinned-host source is deterministic. There is intentionally no
        # torch.cuda.synchronize() between the two forward calls.
        torch.cuda._sleep(300_000_000)
        self.assertTrue(runner.prepare(first_inputs))
        first_output = runner.forward(first_inputs).hidden_states.clone()
        self.assertTrue(runner.prepare(second_inputs))
        second_output = runner.forward(second_inputs).hidden_states.clone()
        torch.cuda.synchronize()

        torch.testing.assert_close(first_output, torch.ones_like(first_output))
        torch.testing.assert_close(second_output, torch.full_like(second_output, 3))

    def test_prepare_failure_can_retry_and_replay(self) -> None:
        runner = CudaGraphRunner()
        model = InjectablePrepareFailureModel()
        runner.init_generation_prefill(
            model,
            2,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [TOKENS_PER_BLOCK],
            HIDDEN_SIZE,
            GROUP_TAGS,
            0,
        )
        inputs = _build_prefill_inputs(
            GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=[2, 2]
        )

        model.impl.fail = True
        with self.assertRaisesRegex(RuntimeError, "injected prepare failure"):
            runner.prepare(inputs)

        model.impl.fail = False
        self.assertTrue(runner.prepare(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected = inputs.input_ids.to(torch.bfloat16).unsqueeze(1) * 10 + torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(0)
        torch.testing.assert_close(output.hidden_states, expected)

    def test_dirty_capture_failure_is_fail_closed(self) -> None:
        # A failed capture can poison allocator/stream state by design. The
        # dedicated Bazel targets below run this test alone in a disposable
        # process; normal tagged-cache suites skip it.
        if os.environ.get("RTP_LLM_RUN_DIRTY_CAPTURE_TEST") != "1":
            self.skipTest("requires an isolated process for a dirty CUDA capture")

        def trigger_dirty_capture():
            runner = CudaGraphRunner()
            model = InjectableCaptureBodyFailureModel()
            model_ref = weakref.ref(model)
            with self.assertRaisesRegex(
                DirtyCudaGraphCaptureError, "injected capture-body failure"
            ):
                runner.init_generation_prefill(
                    model,
                    1,
                    TOKENS_PER_BLOCK,
                    KERNEL_BLOCK_TABLE_WIDTH,
                    [TOKENS_PER_BLOCK],
                    HIDDEN_SIZE,
                    GROUP_TAGS,
                    0,
                )
            self.assertEqual(model.forward_calls, 4)
            return model_ref

        retained_model = trigger_dirty_capture()
        gc.collect()
        # The dirty factory deliberately retains the runner and all graph-owned
        # memory instead of returning it to a live allocator. The model's
        # strong reference is an observable retention sentinel for that
        # fail-closed ownership decision.
        self.assertIsNotNone(retained_model())

    def test_generation_prefill_padding_row_uses_reserved_block_zero(self) -> None:
        runner = CudaGraphRunner()
        model = PrefillPaddingBlockTableModel()
        runner.init_generation_prefill(
            model,
            1,
            16,
            KERNEL_BLOCK_TABLE_WIDTH,
            [16],
            HIDDEN_SIZE,
            GROUP_TAGS,
            0,
        )

        # Nine tokens select a two-page capture table. Real rows carry nonzero
        # request blocks; the graph-owned positive-length padding row must stay
        # all-zero so the attention kernel writes only allocator-reserved page 0.
        inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 7, "aux": 9}, seq_len=9)
        self.assertTrue(runner.canRun(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected = inputs.input_ids.to(torch.bfloat16).unsqueeze(1) * 10 + torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(0)
        torch.testing.assert_close(output.hidden_states, expected)

    def test_generation_prefill_uses_bucket_capacity_without_ratio_gate(self) -> None:
        runner = CudaGraphRunner()
        model = TextOnlyMultimodalCapableModel()
        runner.init_generation_prefill(
            model,
            2,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [4, TOKENS_PER_BLOCK],
            HIDDEN_SIZE,
            GROUP_TAGS,
            3,
        )
        self.assertIs(
            model.cuda_graph_selection_mode,
            CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH,
        )

        # PREPARE and FORWARD use the same position-id contract. Reject before
        # prepareAttentionInputs can attempt to copy a missing mRoPE tensor.
        missing_positions = _build_prefill_inputs(
            GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=[2, 2]
        )
        self.assertFalse(runner.canPrepare(missing_positions))
        self.assertFalse(runner.prepare(missing_positions))
        self.assertFalse(runner.canRun(missing_positions))

        # Exercise the smaller exact bucket first.
        inputs = _build_prefill_inputs(
            GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=[2, 2]
        )
        _with_mrope_positions(inputs)
        self.assertTrue(runner.canRun(inputs))
        self.assertEqual(runner.getCurrentRealGraphSize(), 4)
        # Exercise the production split prepare -> forward path. This used to
        # throw because PyWrappedModel only put mRoPE IDs in nested attention
        # inputs while the graph runner validates the top-level tensor.
        self.assertTrue(runner.prepare(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected = (
            inputs.input_ids.to(torch.bfloat16).unsqueeze(1)
            + inputs.combo_position_ids[:, :1]
        ) * 10 + torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(
            0
        )
        torch.testing.assert_close(output.hidden_states, expected)

        # Six real tokens are served by the eight-token graph. There is no
        # padding-ratio gate: any positive token count within the bucket range
        # remains eligible, and selection advances to the next captured bucket.
        larger_inputs = _build_prefill_inputs(
            GROUP_TAGS, {"full": 3, "aux": 4}, seq_len=[3, 3]
        )
        _with_mrope_positions(larger_inputs)
        self.assertTrue(runner.canRun(larger_inputs))
        self.assertEqual(runner.getCurrentRealGraphSize(), TOKENS_PER_BLOCK)
        larger_output = runner.forward(larger_inputs)
        torch.cuda.synchronize()
        larger_expected = (
            larger_inputs.input_ids.to(torch.bfloat16).unsqueeze(1)
            + larger_inputs.combo_position_ids[:, :1]
        ) * 10 + torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(
            0
        )
        torch.testing.assert_close(larger_output.hidden_states, larger_expected)

        # Generation prefill starts from token IDs and must not allocate the
        # decode/MTP-only input_hiddens scratch buffer.
        self.assertEqual(model.input_hiddens_numel, 0)

        reject_cases: list[tuple[str, PyModelInputs, str]] = []

        prefixed = _with_mrope_positions(
            _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=[2, 2])
        )
        prefixed.attention_inputs["full"].prefix_lengths[0] = 1
        prefixed.attention_inputs["aux"].prefix_lengths[0] = 1
        reject_cases.append(("prefix", prefixed, "prefix_cache_not_supported"))

        multimodal_inputs = _build_prefill_inputs(
            GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=[2, 2]
        )
        _with_mrope_positions(multimodal_inputs)
        multimodal_inputs.multimodal_inputs.multimodal_features = [
            torch.ones((1, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
        ]
        multimodal_inputs.multimodal_inputs.mm_features_locs = torch.tensor(
            [0], dtype=torch.int32, device="cuda"
        )
        multimodal_inputs.embedding_inputs.text_tokens_mask = torch.ones(
            4, dtype=torch.int32, device="cuda"
        )
        reject_cases.append(
            (
                "multimodal",
                multimodal_inputs,
                "multimodal_input_not_supported",
            )
        )

        token_type_inputs = _with_mrope_positions(
            _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=[2, 2])
        )
        token_type_inputs.embedding_inputs.combo_tokens_type_ids = torch.zeros(
            4, dtype=torch.int32, device="cuda"
        )
        reject_cases.append(
            (
                "token_type_input",
                token_type_inputs,
                "token_type_input_not_supported",
            )
        )

        too_many_requests = _with_mrope_positions(
            _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=[1, 1, 1])
        )
        reject_cases.append(
            (
                "request_count",
                too_many_requests,
                "request_count_exceed_capture_limit",
            )
        )
        too_many_tokens = _with_mrope_positions(
            _build_prefill_inputs(
                GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=TOKENS_PER_BLOCK + 1
            )
        )
        reject_cases.append(
            (
                "token_count",
                too_many_tokens,
                "input_tokens_exceed_capture_limit",
            )
        )

        for name, rejected_inputs, expected_status in reject_cases:
            with self.subTest(rejection=name):
                self.assertFalse(runner.canRun(rejected_inputs))
                self.assertEqual(runner.getGenerationPrefillStatus(), expected_status)

    def test_generation_prefill_without_combo_position_ids(self) -> None:
        runner = CudaGraphRunner()
        model = TextOnlyMultimodalCapableModel()
        runner.init_generation_prefill(
            model,
            1,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [TOKENS_PER_BLOCK],
            HIDDEN_SIZE,
            GROUP_TAGS,
            0,
        )
        self.assertFalse(runner.captureSessionMayBeDirty())
        self.assertTrue(model.bert_buffer_shapes)
        self.assertEqual(set(model.bert_buffer_shapes), {(0, 0)})

        inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        self.assertTrue(runner.canRun(inputs))

        metadata_only = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        # PyModelInputs.input_ids is a torch::Tensor field. Its generated
        # pybind setter rejects None on ROCm, so clear it through the test
        # wrapper to preserve the undefined-tensor PREPARE contract.
        runner.clear_input_ids(metadata_only)
        self.assertTrue(runner.canPrepare(metadata_only))
        self.assertFalse(runner.canRun(metadata_only))
        self.assertEqual(runner.getGenerationPrefillStatus(), "input_metadata_invalid")

        scalar_lengths = _build_prefill_inputs(
            GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=1
        )
        runner.make_input_lengths_scalar(scalar_lengths)
        self.assertFalse(runner.canRun(scalar_lengths))
        self.assertEqual(runner.getGenerationPrefillStatus(), "input_metadata_invalid")

        scalar_input_ids = _build_prefill_inputs(
            GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=1
        )
        runner.make_input_ids_scalar(scalar_input_ids)
        self.assertFalse(runner.canRun(scalar_input_ids))
        self.assertEqual(runner.getGenerationPrefillStatus(), "input_metadata_invalid")

        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected = inputs.input_ids.to(torch.bfloat16).unsqueeze(1) * 10 + torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(0)
        torch.testing.assert_close(output.hidden_states, expected)

    def test_prefill_zero_dimensional_metadata_falls_back(self) -> None:
        runner = CudaGraphRunner()
        runner.init_prefill(
            TextOnlyMultimodalCapableModel(),
            1,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [TOKENS_PER_BLOCK],
            HIDDEN_SIZE,
            GROUP_TAGS,
        )

        scalar_lengths = _build_prefill_inputs(
            GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=1
        )
        runner.make_input_lengths_scalar(scalar_lengths)
        self.assertFalse(runner.canRun(scalar_lengths))

        scalar_input_ids = _build_prefill_inputs(
            GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=1
        )
        runner.make_input_ids_scalar(scalar_input_ids)
        self.assertFalse(runner.canRun(scalar_input_ids))

    def test_legacy_embedding_keeps_capture_buffers_and_eligibility(self) -> None:
        weights = torch.ones((2, 1), dtype=torch.bfloat16, device="cuda")
        for has_positions, has_token_types in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(positions=has_positions, token_types=has_token_types):
                runner = CudaGraphRunner()
                model = TextOnlyMultimodalCapableModel()
                runner.init_prefill(
                    model,
                    2,
                    TOKENS_PER_BLOCK,
                    KERNEL_BLOCK_TABLE_WIDTH,
                    [4],
                    HIDDEN_SIZE,
                    GROUP_TAGS,
                    weights if has_positions else None,
                    weights if has_token_types else None,
                )
                # Main always allocates both max_seq_len * max_batch buffers;
                # both are sliced per bucket only when position weights exist.
                full_capacity = 2 * TOKENS_PER_BLOCK
                expected_shapes = {(full_capacity, full_capacity)}
                if has_positions:
                    expected_shapes.add((4, 4))
                self.assertEqual(set(model.bert_buffer_shapes), expected_shapes)

                inputs = _build_prefill_inputs(
                    GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=[2, 2]
                )
                # Request-owned BERT IDs are not a legacy graph eligibility
                # requirement. The generation-only validation must not leak in.
                self.assertTrue(runner.canPrepare(inputs))
                self.assertTrue(runner.canRun(inputs))

    def test_legacy_embedding_refreshes_bert_ids_during_prepare(self) -> None:
        runner = CudaGraphRunner()
        position_encoding = torch.tensor(
            [[3], [7]], dtype=torch.bfloat16, device="cuda"
        )
        token_type_embedding = torch.tensor(
            [[5], [11]], dtype=torch.bfloat16, device="cuda"
        )
        runner.init_prefill(
            BertWeightAwareModel(),
            1,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [TOKENS_PER_BLOCK],
            HIDDEN_SIZE,
            GROUP_TAGS,
            position_encoding,
            token_type_embedding,
        )
        for ids in ([0, 1, 0, 1], [1, 0, 1, 0]):
            inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
            positions = torch.tensor(ids, dtype=torch.int32, device="cuda")
            token_types = 1 - positions
            inputs.bert_embedding_inputs = BertEmbeddingInputs(
                positions, position_encoding, token_types, token_type_embedding, 1.0
            )
            self.assertTrue(runner.canRun(inputs))
            # No separate metadata-only prepare: forward calls prepareInputs(),
            # including the legacy attention-stage BERT ID copies.
            output = runner.forward(inputs)
            torch.cuda.synchronize()
            expected = (
                inputs.input_ids.to(torch.bfloat16).unsqueeze(1) * 10
                + torch.arange(
                    HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
                ).unsqueeze(0)
                + position_encoding.index_select(0, positions.to(torch.int64))
                + token_type_embedding.index_select(0, token_types.to(torch.int64))
            )
            torch.testing.assert_close(output.hidden_states, expected)

    def test_generation_prefill_installs_embedding_weights_before_capture(self) -> None:
        runner = CudaGraphRunner()
        model = BertWeightAwareModel()
        position_encoding = torch.tensor(
            [[3], [7]], dtype=torch.bfloat16, device="cuda"
        )
        token_type_embedding = torch.tensor(
            [[5], [11]], dtype=torch.bfloat16, device="cuda"
        )
        runner.init_generation_prefill(
            model,
            1,
            4 * TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [TOKENS_PER_BLOCK],
            HIDDEN_SIZE,
            GROUP_TAGS,
            0,
            position_encoding,
            token_type_embedding,
        )
        self.assertTrue(model.bert_buffer_shapes)
        self.assertEqual(
            set(model.bert_buffer_shapes),
            {(TOKENS_PER_BLOCK, TOKENS_PER_BLOCK)},
        )

        outputs = []
        for position_ids, token_type_ids in (
            ([0, 1, 0, 1], [0, 0, 1, 1]),
            ([1, 0, 1, 0], [1, 1, 0, 0]),
        ):
            inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
            position_ids_tensor = torch.tensor(
                position_ids, dtype=torch.int32, device="cuda"
            )
            token_type_ids_tensor = torch.tensor(
                token_type_ids, dtype=torch.int32, device="cuda"
            )
            inputs.bert_embedding_inputs = BertEmbeddingInputs(
                position_ids_tensor,
                position_encoding,
                token_type_ids_tensor,
                token_type_embedding,
                1.0,
            )
            self.assertTrue(runner.prepare(inputs))
            self.assertTrue(runner.canRun(inputs))
            output = runner.forward(inputs)
            torch.cuda.synchronize()
            expected = (
                inputs.input_ids.to(torch.bfloat16).unsqueeze(1) * 10
                + torch.arange(
                    HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
                ).unsqueeze(0)
                + position_encoding.index_select(0, position_ids_tensor.to(torch.int64))
                + token_type_embedding.index_select(
                    0, token_type_ids_tensor.to(torch.int64)
                )
            )
            torch.testing.assert_close(output.hidden_states, expected)
            outputs.append(output.hidden_states.clone())
        self.assertFalse(torch.equal(outputs[0], outputs[1]))

    def test_duplicate_capture_tag_is_rejected(self) -> None:
        runner = CudaGraphRunner()
        with self.assertRaisesRegex(
            RuntimeError, "duplicate CUDA graph KV cache tag=full"
        ):
            runner.init_decode(
                TaggedBlockTableModel(),
                HIDDEN_SIZE,
                TOKENS_PER_BLOCK,
                KERNEL_BLOCK_TABLE_WIDTH,
                [1],
                ["full", "full"],
            )

    def test_target_verify_validates_exact_tag_set(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedBlockTableModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [2],
            GROUP_TAGS,
            True,
        )

        valid = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
        )
        self.assertTrue(runner.canRun(valid))

        missing = _build_target_verify_inputs(
            ["full"], {"full": 2}, batch_size=2, query_len=1, prefix_len=1
        )
        self.assertFalse(runner.canRun(missing))

        wrong = _build_target_verify_inputs(
            ["full", "wrong"],
            {"full": 2, "wrong": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
        )
        self.assertFalse(runner.canRun(wrong))

        non_prefill = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
            is_prefill=False,
        )
        self.assertFalse(runner.canRun(non_prefill))

    def test_target_verify_uses_reserved_block_dummy_rows(self) -> None:
        scenarios = (
            (4, 5, 11, (4, 3, 2, 1, 4)),
            # Production shape: DSpark proposes 7+1 tokens and a live batch of
            # six replays the graph-eight bucket on MI308X.
            (8, 8, 17, (8, 6, 2, 8)),
        )
        for graph_size, query_len, prefix_len, batch_sizes in scenarios:
            runner = CudaGraphRunner()
            runner.init_decode(
                TaggedSequenceLengthModel(),
                HIDDEN_SIZE,
                64,
                KERNEL_BLOCK_TABLE_WIDTH,
                [graph_size],
                GROUP_TAGS,
                True,
                query_len,
            )

            # Exercise both growth and shrink on the same graph instance. The
            # production failure appeared only after a full bucket lost a request.
            for batch_size in batch_sizes:
                with self.subTest(graph_size=graph_size, batch_size=batch_size):
                    inputs = _build_target_verify_inputs(
                        GROUP_TAGS,
                        {"full": 2, "aux": 1},
                        batch_size=batch_size,
                        query_len=query_len,
                        prefix_len=prefix_len,
                    )
                    self.assertTrue(runner.canRun(inputs))
                    self.assertEqual(runner.getCurrentRealGraphSize(), graph_size)

                    output = runner.forward(inputs)
                    torch.cuda.synchronize()
                    total_kv_length = batch_size * (query_len + prefix_len)
                    expected_signature = torch.tensor(
                        [
                            graph_size * query_len,
                            total_kv_length + (graph_size - batch_size) * query_len,
                            graph_size * query_len,
                            prefix_len + 1 if batch_size == graph_size else query_len,
                        ],
                        dtype=output.hidden_states.dtype,
                        device=output.hidden_states.device,
                    )
                    torch.testing.assert_close(
                        output.hidden_states,
                        expected_signature.unsqueeze(0).expand_as(output.hidden_states),
                    )

    def test_target_verify_clears_static_token_metadata_after_shrink(self) -> None:
        query_len = 8
        runner = CudaGraphRunner()
        runner.init_decode(
            StaticTokenMetadataTailModel(),
            HIDDEN_SIZE,
            64,
            KERNEL_BLOCK_TABLE_WIDTH,
            [8],
            GROUP_TAGS,
            True,
            query_len,
            3,
        )

        # Seed every graph-capacity row with non-zero request data. A later
        # replay with six live requests must not let rows 48..63 retain it.
        full_inputs = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=8,
            query_len=query_len,
            prefix_len=17,
        )
        full_inputs.input_ids.fill_(91)
        full_inputs.input_hiddens.fill_(7)
        full_inputs.combo_position_ids = torch.full(
            (8 * query_len * 3,), 73, dtype=torch.int32, device="cuda"
        )
        self.assertTrue(runner.canRun(full_inputs))
        runner.forward(full_inputs)
        torch.cuda.synchronize()

        shrunk_inputs = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=6,
            query_len=query_len,
            prefix_len=17,
        )
        shrunk_inputs.combo_position_ids = torch.arange(
            1,
            6 * query_len * 3 + 1,
            dtype=torch.int32,
            device="cuda",
        )
        self.assertTrue(runner.canRun(shrunk_inputs))
        self.assertEqual(runner.getCurrentRealGraphSize(), 8)

        output = runner.forward(shrunk_inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(output.hidden_states, shrunk_inputs.input_hiddens)

    def test_hybrid_wider_staging_row_replays_bounded_intersection(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedBlockTableModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            KERNEL_BLOCK_TABLE_WIDTH,
            [4],
            GROUP_TAGS,
        )

        # The shared staging row is wider than the model-local capture row.
        # Upstream replay semantics permit that and copy only the intersection.
        inputs = _build_decode_inputs(
            GROUP_TAGS, {"full": 5, "aux": 3}, batch_size=3, block_count=KERNEL_BLOCK_TABLE_WIDTH + 2
        )
        self.assertTrue(runner.canRun(inputs))
        self.assertTrue(runner.canPrepare(inputs))
        runner.updateBlockTables(inputs)
        self._assert_replay_signature(runner, inputs, 53)

    def test_topology_real_and_fake_allocator_bounds(self) -> None:
        # Enum values: LINEAR=0, FULL=1, SWA=2. Only FULL expands physical IDs.
        self.assertEqual(CudaGraphRunner.topologyWidths([8, 32], [8, 8], [1, 1], 64, 0, 1), [8, 4])
        self.assertEqual(CudaGraphRunner.topologyWidths([8, 32], [8, 8], [1, 1], 64, 1, 1), [12, 4])
        self.assertEqual(CudaGraphRunner.topologyWidths([512], [64], [1], 513, 0, 1), [16, 8])
        self.assertEqual(CudaGraphRunner.topologyWidths([8], [2], [0], 64, 1, 1), [8, 1])
        self.assertEqual(CudaGraphRunner.topologyWidths([8], [2], [0], 64, 9, 1), [16, 1])
        self.assertEqual(CudaGraphRunner.topologyWidths([8], [2], [2], 64, 9, 1), [10, 1])
        self.assertEqual(CudaGraphRunner.topologyWidths([512], [64], [1], 1, 0, 5), [8, 40])
        self.assertEqual(CudaGraphRunner.topologyWidths([8], [8], [1], 1, 0, 0), [1, 1])
        for sequence, reserve, fake in ((0, 0, 1), (2**63 - 1, 1, 1), (1, 0, 2**63)):
            with self.subTest(sequence=sequence, reserve=reserve, fake=fake):
                with self.assertRaises(RuntimeError):
                    CudaGraphRunner.topologyWidths([8], [2], [1], sequence, reserve, fake)

    def test_topology_width_end_to_end_preserves_last_column(self) -> None:
        for tags in (GROUP_TAGS, list(reversed(GROUP_TAGS))):
            for reserve, fake_count, width in ((0, 1, 8), (1, 1, 12), (0, 5, 20)):
                with self.subTest(tags=tags, reserve=reserve, fake_count=fake_count):
                    spans = {"full": 8, "aux": 32}
                    real_width, fake_width = CudaGraphRunner.topologyWidths(
                        [spans[tag] for tag in tags], [8, 8], [1, 1], 64, reserve, fake_count
                    )
                    self.assertEqual(max(real_width, fake_width), width)
                    runner = CudaGraphRunner()
                    runner.init_decode(
                        TaggedBlockRowModel(), HIDDEN_SIZE, 64, width, [4], tags
                    )
                    inputs = _build_decode_inputs(
                        tags, {"full": 2, "aux": 3}, batch_size=3, block_count=width
                    )
                    self._assert_replay_signature(runner, inputs, width * 50)

                    shorter = _build_decode_inputs(
                        tags, {"full": 1, "aux": 1}, batch_size=3, block_count=1
                    )
                    self._assert_replay_signature(runner, shorter, 17)

    def test_cacheless_runner_uses_explicit_width(self) -> None:
        for width in (1, KERNEL_BLOCK_TABLE_WIDTH):
            with self.subTest(width=width):
                runner = CudaGraphRunner()
                runner.init_decode(
                    TaggedBlockTableModel(), HIDDEN_SIZE, 64, width, [4], GROUP_TAGS
                )
                inputs = _build_decode_inputs(GROUP_TAGS, {"full": 5, "aux": 3}, block_count=width)
                self._assert_replay_signature(runner, inputs, 53)

        with self.assertRaisesRegex(RuntimeError, "positive kernel block table width"):
            runner = CudaGraphRunner()
            runner.init_decode(TaggedBlockTableModel(), HIDDEN_SIZE, 64, 0, [4], GROUP_TAGS)

    def test_generation_prefill_keeps_host_width_gate(self) -> None:
        runner = CudaGraphRunner()
        runner.init_generation_prefill(
            TextOnlyMultimodalCapableModel(),
            1,
            TOKENS_PER_BLOCK,
            1,
            [TOKENS_PER_BLOCK],
            HIDDEN_SIZE,
            GROUP_TAGS,
            0,
        )
        inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        self.assertTrue(runner.canRun(inputs))

        wide = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        wide_tags = {}
        for tag, attention in inputs.attention_inputs.items():
            attention = copy.copy(attention)
            host = torch.full((1, 2), 7, dtype=torch.int32).pin_memory()
            attention.kv_cache_kernel_block_id = host
            attention.kv_cache_kernel_block_id_device = host.cuda()
            wide_tags[tag] = attention
        wide.attention_inputs = wide_tags
        self.assertFalse(runner.canRun(wide))
        self.assertEqual(runner.getGenerationPrefillStatus(), "graph_input_shape_mismatch")
        self.assertFalse(runner.canPrepare(wide))

    def test_decode_clears_rounded_batch_sequence_metadata(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedDecodePaddingModel(),
            HIDDEN_SIZE,
            64,
            KERNEL_BLOCK_TABLE_WIDTH,
            [4],
            GROUP_TAGS,
        )

        full_inputs = _build_decode_inputs(
            GROUP_TAGS, {"full": 2, "aux": 1}, batch_size=4
        )
        self.assertTrue(runner.canRun(full_inputs))
        runner.forward(full_inputs)
        torch.cuda.synchronize()

        inputs = _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1}, batch_size=3)
        self.assertTrue(runner.canRun(inputs))
        self.assertEqual(runner.getCurrentRealGraphSize(), 4)

        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected_signature = torch.tensor(
            [7, 1, 4, 3],
            dtype=output.hidden_states.dtype,
            device=output.hidden_states.device,
        )
        torch.testing.assert_close(
            output.hidden_states,
            expected_signature.unsqueeze(0).expand_as(output.hidden_states),
        )

    def test_decode_clears_hidden_rows_after_batch_shrink(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            StaticInputTailModel(),
            HIDDEN_SIZE,
            64,
            KERNEL_BLOCK_TABLE_WIDTH,
            [4],
            GROUP_TAGS,
        )

        full_inputs = _build_decode_inputs(
            GROUP_TAGS, {"full": 2, "aux": 1}, batch_size=4
        )
        full_inputs.input_hiddens[-1].fill_(7)
        self.assertTrue(runner.canRun(full_inputs))
        runner.forward(full_inputs)
        torch.cuda.synchronize()

        inputs = _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1}, batch_size=3)
        self.assertTrue(runner.canRun(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.hidden_states,
            torch.zeros_like(output.hidden_states),
        )

    def test_auxiliary_output_view_follows_each_graph_bucket(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            AuxiliaryOutputModel(),
            HIDDEN_SIZE,
            64,
            KERNEL_BLOCK_TABLE_WIDTH,
            [1, 4],
            GROUP_TAGS,
        )

        for batch_size in (4, 1, 3, 1, 4):
            with self.subTest(batch_size=batch_size):
                inputs = _build_decode_inputs(
                    GROUP_TAGS, {"full": 2, "aux": 1}, batch_size=batch_size
                )
                inputs.input_hiddens.copy_(
                    torch.arange(
                        batch_size * HIDDEN_SIZE,
                        dtype=inputs.input_hiddens.dtype,
                        device=inputs.input_hiddens.device,
                    ).reshape(batch_size, HIDDEN_SIZE)
                )
                self.assertTrue(runner.canRun(inputs))
                output = runner.forward(inputs)
                torch.cuda.synchronize()

                expected = torch.cat(
                    (inputs.input_hiddens + 2, inputs.input_hiddens + 3), dim=1
                )
                self.assertEqual(
                    (batch_size, HIDDEN_SIZE * 2),
                    tuple(output.mtp_target_hidden_states.shape),
                )
                torch.testing.assert_close(output.mtp_target_hidden_states, expected)


if __name__ == "__main__":
    unittest.main()
