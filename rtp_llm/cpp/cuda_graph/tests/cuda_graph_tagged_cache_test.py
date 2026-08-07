import copy
import unittest

import torch

from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    get_typemeta,
    rtp_llm_ops,
)

GROUP_TAGS = ["full", "aux"]
HIDDEN_SIZE = 4
TOKENS_PER_BLOCK = 8


class TaggedBlockTableModel:
    """Small graph-safe model whose output exposes both tag-local block tables."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        attention_inputs = inputs.attention_inputs
        full_id = attention_inputs["full"].kv_cache_kernel_block_id_device[0, 0]
        aux_id = attention_inputs["aux"].kv_cache_kernel_block_id_device[0, 0]
        signature = (full_id + 16 * aux_id).to(inputs.input_hiddens.dtype)
        return PyModelOutputs(inputs.input_hiddens + signature)


class TaggedSequenceLengthModel:
    """Expose the cumulative lengths used by a tagged captured graph."""

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
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


class BertEmbeddingModel:
    """Run the real Bert embedding op across capture and replay."""

    def __init__(self, word_embedding: torch.Tensor):
        self.word_embedding = word_embedding

    def prepare_fmha_impl(self, inputs: PyModelInputs, is_cuda_graph: bool = False):
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        bert_inputs = inputs.bert_embedding_inputs
        token_count = inputs.input_hiddens.size(0)
        output = torch.empty_like(inputs.input_hiddens)
        rtp_llm_ops.embedding_bert(
            output,
            inputs.input_ids[:token_count],
            self.word_embedding,
            bert_inputs.combo_position_ids,
            bert_inputs.position_encoding,
            bert_inputs.combo_tokens_type_ids,
            bert_inputs.token_type_embedding,
            bert_inputs.input_embedding_scalar,
            None,
        )
        return PyModelOutputs(output)


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
    tags: list[str], values: dict[str, int], seq_len: int = 4
) -> PyModelInputs:
    attention_inputs = PyAttentionInputs()
    attention_inputs.is_prefill = True
    attention_inputs.is_target_verify = False
    attention_inputs.input_lengths = torch.tensor(
        [seq_len], dtype=torch.int32
    ).pin_memory()
    attention_inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32).pin_memory()
    attention_inputs.cu_seqlens = torch.tensor(
        [0, seq_len], dtype=torch.int32
    ).pin_memory()
    attention_inputs.cu_seqlens_device = attention_inputs.cu_seqlens.cuda()
    attention_inputs.cu_kv_seqlens_device = attention_inputs.cu_seqlens_device.clone()
    attention_inputs.context_total_kv_length = seq_len
    return _build_common_inputs(
        attention_inputs,
        tags,
        values,
        batch_size=1,
        token_count=seq_len,
        block_count=1,
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

    def test_multimodal_prefill_falls_back_from_cuda_graph(self) -> None:
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

        baseline_inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        self.assertTrue(runner.canRun(baseline_inputs))

        mask_inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        mask_inputs.embedding_inputs.text_tokens_mask = torch.ones(
            4, dtype=torch.int32, device="cuda"
        )
        mixed_mask_inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        mixed_mask_inputs.embedding_inputs.text_tokens_mask = torch.tensor(
            [0, 1, 0, 1], dtype=torch.int32, device="cuda"
        )
        all_zero_mask_inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        all_zero_mask_inputs.embedding_inputs.text_tokens_mask = torch.zeros(
            4, dtype=torch.int32, device="cuda"
        )
        empty_mask_inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        empty_mask_inputs.embedding_inputs.text_tokens_mask = torch.empty(
            0, dtype=torch.int32, device="cuda"
        )
        feature_inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        feature_inputs.multimodal_inputs.multimodal_features = [
            torch.zeros((1, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
        ]
        location_inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        location_inputs.multimodal_inputs.mm_features_locs = torch.tensor(
            [1], dtype=torch.int32, device="cuda"
        )
        extra_input = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
        extra_input.multimodal_inputs.mm_extra_input = [
            torch.zeros((1, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
        ]

        for signal, inputs in (
            ("all_one_text_tokens_mask", mask_inputs),
            ("mixed_text_tokens_mask", mixed_mask_inputs),
            ("all_zero_text_tokens_mask", all_zero_mask_inputs),
            ("multimodal_features", feature_inputs),
            ("mm_features_locs", location_inputs),
            ("mm_extra_input", extra_input),
        ):
            with self.subTest(signal=signal):
                self.assertFalse(runner.canRun(inputs))

        with self.subTest(contract="prepared_state_cleared_after_multimodal_fallback"):
            prepared_inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2})
            self.assertTrue(runner.canRun(prepared_inputs))
            runner.prepareAttentionInputs(prepared_inputs)
            self.assertFalse(runner.canRun(mask_inputs))

            # A rejected request must clear the prepared state so the next clean
            # request prepares fresh mirrors and replays its own block-table data.
            recovery_inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 4, "aux": 3})
            self._assert_replay_signature(runner, recovery_inputs, 52)

        # An empty mask is semantically absent and remains graph-safe.
        self.assertTrue(runner.canRun(empty_mask_inputs))

    def test_decode_request_owned_inputs_disable_cuda_graph(self) -> None:
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

        def _decode_inputs_with_mask() -> PyModelInputs:
            fresh = _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1})
            fresh.embedding_inputs.text_tokens_mask = torch.ones(
                2, dtype=torch.int32, device="cuda"
            )
            return fresh

        self._assert_replay_signature(
            runner, _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1}), 18
        )
        self.assertFalse(runner.canRun(_decode_inputs_with_mask()))

        with self.subTest(signal="multimodal_features"):
            case = _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1})
            case.multimodal_inputs.multimodal_features = [
                torch.zeros((1, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
            ]
            self.assertFalse(runner.canRun(case))

        with self.subTest(signal="mm_features_locs"):
            case = _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1})
            case.multimodal_inputs.mm_features_locs = torch.tensor(
                [1], dtype=torch.int32, device="cuda"
            )
            self.assertFalse(runner.canRun(case))

        with self.subTest(signal="mm_extra_input"):
            case = _build_decode_inputs(GROUP_TAGS, {"full": 2, "aux": 1})
            case.multimodal_inputs.mm_extra_input = [
                torch.zeros((1, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
            ]
            self.assertFalse(runner.canRun(case))

    def test_prefill_replay_updates_bert_ids_and_uses_capture_parameters(self) -> None:
        capture_scalar = 0.5
        word_embedding = torch.arange(
            16 * HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        ).reshape(16, HIDDEN_SIZE)
        position_encoding = (
            (torch.arange(16, dtype=torch.bfloat16, device="cuda").unsqueeze(1) * 100)
            .expand(-1, HIDDEN_SIZE)
            .contiguous()
        )
        token_type_embedding = (
            (torch.arange(8, dtype=torch.bfloat16, device="cuda").unsqueeze(1) * 10)
            .expand(-1, HIDDEN_SIZE)
            .contiguous()
        )
        runner = CudaGraphRunner()
        runner.init_prefill(
            BertEmbeddingModel(word_embedding),
            2,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            HIDDEN_SIZE,
            GROUP_TAGS,
            position_encoding,
            token_type_embedding,
            capture_scalar,
        )

        def build_inputs(
            input_ids: list[int],
            position_ids: list[int],
            token_type_ids: list[int],
            *,
            provide_position_table: bool = True,
        ) -> PyModelInputs:
            inputs = _build_prefill_inputs(
                GROUP_TAGS, {"full": 1, "aux": 2}, seq_len=len(input_ids)
            )
            inputs.input_ids = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
            # EmbeddingExecutor always populates this generic field. Bert uses
            # the separately captured/copied bert_embedding_inputs IDs, so the
            # generic field must not disable graph replay.
            inputs.embedding_inputs.combo_tokens_type_ids = torch.tensor(
                token_type_ids, dtype=torch.int32, device="cuda"
            )
            inputs.bert_embedding_inputs.combo_position_ids = torch.tensor(
                position_ids, dtype=torch.int32, device="cuda"
            )
            if provide_position_table:
                # Request-owned tables and scalar are deliberately different:
                # replay must keep the parameters baked into capture.
                inputs.bert_embedding_inputs.position_encoding = (
                    position_encoding + 1000
                )
            else:
                inputs.bert_embedding_inputs.position_encoding = torch.empty(
                    0, dtype=torch.bfloat16, device="cuda"
                )
            inputs.bert_embedding_inputs.combo_tokens_type_ids = torch.tensor(
                token_type_ids, dtype=torch.int32, device="cuda"
            )
            inputs.bert_embedding_inputs.token_type_embedding = (
                token_type_embedding + 1000
            )
            inputs.bert_embedding_inputs.input_embedding_scalar = 9.0
            return inputs

        replay_cases = (
            ([1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]),
            ([4, 3, 2, 1], [8, 7, 6, 5], [1, 2, 3, 4]),
        )
        for input_ids, position_ids, token_type_ids in replay_cases:
            with self.subTest(input_ids=input_ids):
                inputs = build_inputs(input_ids, position_ids, token_type_ids)
                self.assertTrue(runner.canRun(inputs))
                output = runner.forward(inputs)
                torch.cuda.synchronize()
                expected = (
                    word_embedding[torch.tensor(input_ids, device="cuda")]
                    * capture_scalar
                    + position_encoding[torch.tensor(position_ids, device="cuda")]
                    + token_type_embedding[torch.tensor(token_type_ids, device="cuda")]
                )
                torch.testing.assert_close(output.hidden_states, expected)

        with self.subTest(contract="prepared_attention_inputs_copy_bert_ids"):
            prepared_inputs = build_inputs([5, 6, 7, 8], [4, 3, 2, 1], [1, 2, 3, 4])
            self.assertTrue(runner.canRun(prepared_inputs))
            runner.prepareAttentionInputs(prepared_inputs)
            prepared_output = runner.forward(prepared_inputs)
            torch.cuda.synchronize()
            prepared_expected = (
                word_embedding[torch.tensor([5, 6, 7, 8], device="cuda")]
                * capture_scalar
                + position_encoding[torch.tensor([4, 3, 2, 1], device="cuda")]
                + token_type_embedding[torch.tensor([1, 2, 3, 4], device="cuda")]
            )
            torch.testing.assert_close(prepared_output.hidden_states, prepared_expected)

        with self.subTest(contract="failed_eligibility_clears_prepared_state"):
            stale_inputs = build_inputs([1, 2, 3, 4], [1, 1, 1, 1], [1, 1, 1, 1])
            self.assertTrue(runner.canRun(stale_inputs))
            runner.prepareAttentionInputs(stale_inputs)

            invalid_after_prepare = build_inputs(
                [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]
            )
            invalid_after_prepare.bert_embedding_inputs.combo_position_ids = (
                torch.empty(0, dtype=torch.int32, device="cuda")
            )
            self.assertFalse(runner.canRun(invalid_after_prepare))

            fresh_inputs = build_inputs([8, 7, 6, 5], [8, 7, 6, 5], [4, 3, 2, 1])
            self.assertTrue(runner.canRun(fresh_inputs))
            fresh_output = runner.forward(fresh_inputs)
            torch.cuda.synchronize()
            fresh_expected = (
                word_embedding[torch.tensor([8, 7, 6, 5], device="cuda")]
                * capture_scalar
                + position_encoding[torch.tensor([8, 7, 6, 5], device="cuda")]
                + token_type_embedding[torch.tensor([4, 3, 2, 1], device="cuda")]
            )
            torch.testing.assert_close(fresh_output.hidden_states, fresh_expected)

        # A shorter request replays the next captured bucket and only copies
        # current_seq_len IDs, including after a full-bucket request.
        with self.subTest(contract="short_request_bucket"):
            short_inputs = build_inputs([6, 7], [9, 10], [2, 3])
            self.assertTrue(runner.canRun(short_inputs))
            short_output = runner.forward(short_inputs)
            torch.cuda.synchronize()
            short_expected = (
                word_embedding[torch.tensor([6, 7], device="cuda")] * capture_scalar
                + position_encoding[torch.tensor([9, 10], device="cuda")]
                + token_type_embedding[torch.tensor([2, 3], device="cuda")]
            )
            torch.testing.assert_close(short_output.hidden_states, short_expected)

        # Dynamic IDs are copied according to the captured Bert graph capability,
        # not according to whether the request repeats the capture-time table.
        with self.subTest(contract="capture_tables_are_constants"):
            inputs_without_position_table = build_inputs(
                [2, 3, 4, 5],
                [5, 6, 7, 8],
                [1, 2, 3, 4],
                provide_position_table=False,
            )
            self.assertTrue(runner.canRun(inputs_without_position_table))
            output = runner.forward(inputs_without_position_table)
            torch.cuda.synchronize()
            expected = (
                word_embedding[torch.tensor([2, 3, 4, 5], device="cuda")]
                * capture_scalar
                + position_encoding[torch.tensor([5, 6, 7, 8], device="cuda")]
                + token_type_embedding[torch.tensor([1, 2, 3, 4], device="cuda")]
            )
            torch.testing.assert_close(output.hidden_states, expected)

        invalid_inputs = []

        missing_bert_ids = build_inputs([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4])
        missing_bert_ids.bert_embedding_inputs.combo_position_ids = torch.empty(
            0, dtype=torch.int32, device="cuda"
        )
        missing_bert_ids.bert_embedding_inputs.combo_tokens_type_ids = torch.empty(
            0, dtype=torch.int32, device="cuda"
        )
        invalid_inputs.append(("missing_bert_ids", missing_bert_ids))

        short_position_ids = build_inputs([1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4])
        invalid_inputs.append(("short_position_ids", short_position_ids))

        short_token_type_ids = build_inputs([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3])
        invalid_inputs.append(("short_token_type_ids", short_token_type_ids))

        int64_position_ids = build_inputs([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4])
        int64_position_ids.bert_embedding_inputs.combo_position_ids = (
            int64_position_ids.bert_embedding_inputs.combo_position_ids.to(torch.int64)
        )
        invalid_inputs.append(("int64_position_ids", int64_position_ids))

        cpu_token_type_ids = build_inputs([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4])
        cpu_token_type_ids.bert_embedding_inputs.combo_tokens_type_ids = (
            cpu_token_type_ids.bert_embedding_inputs.combo_tokens_type_ids.cpu()
        )
        invalid_inputs.append(("cpu_token_type_ids", cpu_token_type_ids))

        noncontiguous_position_ids = build_inputs(
            [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]
        )
        noncontiguous_position_ids.bert_embedding_inputs.combo_position_ids = (
            torch.arange(8, dtype=torch.int32, device="cuda")[::2]
        )
        invalid_inputs.append(
            ("noncontiguous_position_ids", noncontiguous_position_ids)
        )

        for name, invalid in invalid_inputs:
            with self.subTest(name=name):
                self.assertFalse(runner.canRun(invalid))

        # Establish a valid graph selection, then mutate the request without
        # another canRun call. prepareInputs must still reject the malformed
        # buffer before scheduling a D2D copy.
        with self.subTest(contract="direct_forward_rejects_mutated_ids"):
            direct_invalid = build_inputs([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4])
            self.assertTrue(runner.canRun(direct_invalid))
            direct_invalid.bert_embedding_inputs.combo_position_ids = torch.tensor(
                [1, 2, 3], dtype=torch.int32, device="cuda"
            )
            with self.assertRaisesRegex(RuntimeError, "Bert position/type IDs"):
                runner.forward(direct_invalid)

        with self.subTest(contract="recovery_after_rejected_replay"):
            recovery_inputs = build_inputs([4, 3, 2, 1], [5, 6, 7, 8], [1, 2, 3, 4])
            self.assertTrue(runner.canRun(recovery_inputs))
            recovery_output = runner.forward(recovery_inputs)
            torch.cuda.synchronize()
            recovery_expected = (
                word_embedding[torch.tensor([4, 3, 2, 1], device="cuda")]
                * capture_scalar
                + position_encoding[torch.tensor([5, 6, 7, 8], device="cuda")]
                + token_type_embedding[torch.tensor([1, 2, 3, 4], device="cuda")]
            )
            torch.testing.assert_close(recovery_output.hidden_states, recovery_expected)

    def test_prefill_capture_with_only_one_bert_table_uses_non_bert_inputs(
        self,
    ) -> None:
        embedding_table = torch.zeros(
            (16, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )
        for name, position_encoding, token_type_embedding in (
            ("position_only", embedding_table, None),
            ("token_type_only", None, embedding_table),
        ):
            with self.subTest(name=name):
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
                    position_encoding,
                    token_type_embedding,
                )
                self._assert_replay_signature(
                    runner,
                    _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2}),
                    33,
                )

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

        masked = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
        )
        masked.embedding_inputs.text_tokens_mask = torch.ones(
            2, dtype=torch.int32, device="cuda"
        )
        self.assertFalse(runner.canRun(masked))

        multimodal = _build_target_verify_inputs(
            GROUP_TAGS,
            {"full": 2, "aux": 1},
            batch_size=2,
            query_len=1,
            prefix_len=1,
        )
        multimodal.multimodal_inputs.multimodal_features = [
            torch.zeros((1, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda")
        ]
        self.assertFalse(runner.canRun(multimodal))

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
