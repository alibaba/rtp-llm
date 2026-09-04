import copy
import unittest
from typing import Optional

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    CudaGraphSelectionMode,
)
from rtp_llm.ops.compute_ops import (
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


class TextOnlyMultimodalCapableModel:
    """Qwen3-VL-like graph model used without request-side multimodal payload."""

    input_hiddens_numel = -1
    cuda_graph_selection_mode = None

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
    """Expose model-owned embedding weights that must exist during capture."""

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        base = super().forward(inputs, fmha_impl).hidden_states
        position_bias = inputs.bert_embedding_inputs.position_encoding[0, 0]
        token_type_bias = inputs.bert_embedding_inputs.token_type_embedding[0, 0]
        return PyModelOutputs(base + position_bias + token_type_bias)


class InputEmbeddingOverlayModel(TextOnlyMultimodalCapableModel):
    """Exercise the production fixed-buffer input embedding overlay."""

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        base = super().forward(inputs, fmha_impl).hidden_states
        return PyModelOutputs(GptModelBase.apply_input_embeddings(self, base, inputs))


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

    def test_generative_prefill_uses_bucket_capacity_without_ratio_gate(self) -> None:
        runner = CudaGraphRunner()
        model = TextOnlyMultimodalCapableModel()
        runner.init_generative_prefill(
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
            CudaGraphSelectionMode.PREFILL_GRAPH,
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

        # Generative prefill starts from token IDs and must not allocate the
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
                self.assertEqual(runner.getPrefillStatus(), expected_status)

    def test_generative_prefill_without_combo_position_ids(self) -> None:
        runner = CudaGraphRunner()
        model = TextOnlyMultimodalCapableModel()
        runner.init_generative_prefill(
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

        inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        self.assertTrue(runner.canRun(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected = inputs.input_ids.to(torch.bfloat16).unsqueeze(1) * 10 + torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(0)
        torch.testing.assert_close(output.hidden_states, expected)

    def test_generative_prefill_installs_embedding_weights_before_capture(self) -> None:
        runner = CudaGraphRunner()
        model = BertWeightAwareModel()
        position_encoding = torch.full((2, 1), 3, dtype=torch.bfloat16, device="cuda")
        token_type_embedding = torch.full(
            (2, 1), 5, dtype=torch.bfloat16, device="cuda"
        )
        runner.init_generative_prefill(
            model,
            1,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [TOKENS_PER_BLOCK],
            HIDDEN_SIZE,
            GROUP_TAGS,
            0,
            position_encoding,
            token_type_embedding,
        )

        inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        self.assertTrue(runner.canRun(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected = inputs.input_ids.to(torch.bfloat16).unsqueeze(1) * 10 + torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(0)
        torch.testing.assert_close(output.hidden_states, expected + 8)

    def test_generative_prefill_replays_dynamic_input_embeddings(self) -> None:
        runner = CudaGraphRunner()
        runner.init_generative_prefill(
            InputEmbeddingOverlayModel(),
            2,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4, TOKENS_PER_BLOCK],
            HIDDEN_SIZE,
            GROUP_TAGS,
            0,
        )

        inputs = _build_prefill_inputs(
            GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=[2, 2]
        )
        first_override = torch.tensor(
            [[101.0, 102.0, 103.0, 104.0], [201.0, 202.0, 203.0, 204.0]],
            dtype=torch.float32,
            device="cuda",
        )
        inputs.input_embeddings = [first_override]
        inputs.input_embeddings_locs = torch.tensor([1], dtype=torch.int32)
        self.assertTrue(runner.prepare(inputs))
        output = runner.forward(inputs)
        torch.cuda.synchronize()
        expected = inputs.input_ids.to(torch.bfloat16).unsqueeze(1) * 10 + torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(0)
        expected[1:3] = first_override.to(torch.bfloat16)
        torch.testing.assert_close(output.hidden_states, expected)

        # Reuse the same four-token capture with a different interval topology,
        # including a public-API 1-D embedding and dtype conversion.
        changed = _build_prefill_inputs(GROUP_TAGS, {"full": 3, "aux": 4}, seq_len=3)
        changed.input_embeddings = [
            torch.full((HIDDEN_SIZE,), 7.0, dtype=torch.float32, device="cuda"),
            torch.full((1, HIDDEN_SIZE), 9.0, dtype=torch.float16, device="cuda"),
        ]
        changed.input_embeddings_locs = torch.tensor([0, 2], dtype=torch.int64)
        self.assertTrue(runner.canRun(changed))
        changed_output = runner.forward(changed)
        torch.cuda.synchronize()
        changed_expected = changed.input_ids.to(torch.bfloat16).unsqueeze(
            1
        ) * 10 + torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(
            0
        )
        changed_expected[0] = 7
        changed_expected[2] = 9
        torch.testing.assert_close(changed_output.hidden_states, changed_expected)

        # The graph kernel consumes and clears metadata; stale intervals must
        # not leak into a following request that has no input embeddings.
        plain = _build_prefill_inputs(GROUP_TAGS, {"full": 5, "aux": 6}, seq_len=3)
        self.assertTrue(runner.canRun(plain))
        plain_output = runner.forward(plain)
        torch.cuda.synchronize()
        plain_expected = plain.input_ids.to(torch.bfloat16).unsqueeze(
            1
        ) * 10 + torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(
            0
        )
        torch.testing.assert_close(plain_output.hidden_states, plain_expected)

        # Async preparation may be abandoned before forward. Stage an override
        # in the tail of the largest bucket, then replace it with a smaller
        # plain request. Cleanup must cover the shared backing allocation, not
        # only the currently selected bucket view.
        abandoned = _build_prefill_inputs(
            GROUP_TAGS, {"full": 7, "aux": 8}, seq_len=TOKENS_PER_BLOCK
        )
        abandoned.input_embeddings = [
            torch.full((1, HIDDEN_SIZE), 77.0, dtype=torch.bfloat16, device="cuda")
        ]
        abandoned.input_embeddings_locs = torch.tensor(
            [TOKENS_PER_BLOCK - 1], dtype=torch.int32
        )
        self.assertTrue(runner.prepare(abandoned))

        smaller_plain = _build_prefill_inputs(
            GROUP_TAGS, {"full": 9, "aux": 10}, seq_len=3
        )
        self.assertTrue(runner.prepare(smaller_plain))
        smaller_output = runner.forward(smaller_plain)
        torch.cuda.synchronize()
        smaller_expected = smaller_plain.input_ids.to(torch.bfloat16).unsqueeze(
            1
        ) * 10 + torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(
            0
        )
        torch.testing.assert_close(smaller_output.hidden_states, smaller_expected)

        large_plain = _build_prefill_inputs(
            GROUP_TAGS, {"full": 11, "aux": 12}, seq_len=TOKENS_PER_BLOCK
        )
        self.assertTrue(runner.canRun(large_plain))
        large_output = runner.forward(large_plain)
        torch.cuda.synchronize()
        large_expected = large_plain.input_ids.to(torch.bfloat16).unsqueeze(
            1
        ) * 10 + torch.arange(
            HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).unsqueeze(
            0
        )
        torch.testing.assert_close(large_output.hidden_states, large_expected)

    def test_generative_prefill_rejects_invalid_input_embedding_metadata(self) -> None:
        runner = CudaGraphRunner()
        runner.init_generative_prefill(
            InputEmbeddingOverlayModel(),
            1,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            HIDDEN_SIZE,
            GROUP_TAGS,
            0,
        )

        invalid = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        invalid.input_embeddings = [
            torch.ones((2, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
        ]
        invalid.input_embeddings_locs = torch.tensor([3], dtype=torch.int32)
        self.assertFalse(runner.canRun(invalid))
        self.assertEqual(runner.getPrefillStatus(), "input_metadata_invalid")

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
