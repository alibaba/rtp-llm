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
                full_inputs.prefix_lengths_device.sum(),
            )
        ).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + signature)


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
        block_count=1,
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
    def _assert_replay_signature(
        self, runner: CudaGraphRunner, inputs: PyModelInputs, expected: int
    ) -> None:
        self.assertTrue(runner.canRun(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected_output = torch.full_like(output.hidden_states, expected)
        torch.testing.assert_close(output.hidden_states, expected_output)

    def test_decode_tag_validation_and_replay_updates(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedBlockTableModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
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
            _build_decode_inputs(GROUP_TAGS, {"full": 5, "aux": 3}),
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

    def test_prefill_tagged_capture_and_replay_updates(self) -> None:
        runner = CudaGraphRunner()
        runner.init_prefill(
            TaggedBlockTableModel(),
            2,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
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
            _build_prefill_inputs(GROUP_TAGS, {"full": 4, "aux": 3}),
            52,
        )

    def test_prefill_host_metadata_is_stable_without_device_sync(self) -> None:
        runner = CudaGraphRunner()
        runner.init_prefill(
            TaggedSequenceLengthModel(),
            2,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
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
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
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
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
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
                    TOKENS_PER_BLOCK,
                    TOKENS_PER_BLOCK,
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
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
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
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
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
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
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
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
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
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
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
                TOKENS_PER_BLOCK,
                TOKENS_PER_BLOCK,
                [1],
                ["full", "full"],
            )

    def test_target_verify_validates_exact_tag_set(self) -> None:
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedBlockTableModel(),
            HIDDEN_SIZE,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
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

    def test_target_verify_clears_rounded_batch_sequence_lengths(self) -> None:
        query_len = 5
        prefix_len = 11
        runner = CudaGraphRunner()
        runner.init_decode(
            TaggedSequenceLengthModel(),
            HIDDEN_SIZE,
            64,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            GROUP_TAGS,
            True,
            query_len,
        )

        for batch_size in (1, 2, 4):
            with self.subTest(batch_size=batch_size):
                inputs = _build_target_verify_inputs(
                    GROUP_TAGS,
                    {"full": 2, "aux": 1},
                    batch_size=batch_size,
                    query_len=query_len,
                    prefix_len=prefix_len,
                )
                self.assertTrue(runner.canRun(inputs))
                self.assertEqual(runner.getCurrentRealGraphSize(), 4)

                output = runner.forward(inputs)
                torch.cuda.synchronize()
                total_query_length = batch_size * query_len
                total_kv_length = batch_size * (query_len + prefix_len)
                expected_signature = torch.tensor(
                    [
                        total_query_length,
                        total_kv_length,
                        total_query_length,
                        batch_size * prefix_len,
                    ],
                    dtype=output.hidden_states.dtype,
                    device=output.hidden_states.device,
                )
                torch.testing.assert_close(
                    output.hidden_states,
                    expected_signature.unsqueeze(0).expand_as(output.hidden_states),
                )


if __name__ == "__main__":
    unittest.main()
