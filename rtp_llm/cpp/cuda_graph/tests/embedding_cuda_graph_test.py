"""Embedding graph preparation, input validation and request-changing replay."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.cpp.cuda_graph.tests.cuda_graph_input_test_utils import (
    GROUP_TAGS,
    HIDDEN_SIZE,
    TOKENS_PER_BLOCK,
    EmbeddingGraphTestModel,
    _build_common_inputs,
    _build_prefill_inputs,
    _tag_attention_inputs,
)
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
    rtp_llm_ops,
)


class BertEmbeddingModel(EmbeddingGraphTestModel):
    """Prepare real BERT embeddings outside capture with model-owned weights."""

    def __init__(self, word_embedding, position_encoding, token_type_embedding, scalar):
        self.word_embedding = word_embedding
        self.position_encoding = position_encoding
        self.token_type_embedding = token_type_embedding
        self.scalar = scalar

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        return None

    def validate_embedding_graph_inputs(self, inputs):
        if not super().validate_embedding_graph_inputs(inputs):
            return False
        count = inputs.input_ids.numel()
        return all(
            tensor is not None
            and tensor.numel() >= count
            and tensor.dtype == torch.int32
            and tensor.is_cuda
            and tensor.is_contiguous()
            for tensor in (
                inputs.bert_embedding_inputs.combo_position_ids,
                inputs.bert_embedding_inputs.combo_tokens_type_ids,
            )
        )

    def prepare_transformer_inputs(self, inputs):
        bert = inputs.bert_embedding_inputs
        output = torch.empty(
            (inputs.input_ids.numel(), HIDDEN_SIZE),
            dtype=self.word_embedding.dtype,
            device=inputs.input_ids.device,
        )
        rtp_llm_ops.embedding_bert(
            output,
            inputs.input_ids,
            self.word_embedding,
            bert.combo_position_ids,
            self.position_encoding,
            bert.combo_tokens_type_ids,
            self.token_type_embedding,
            self.scalar,
            None,
        )
        return output

    def forward(self, inputs, fmha_impl=None):
        return PyModelOutputs(inputs.input_hiddens)


class TransformerOnlyCudaGraphEmbeddingModel(EmbeddingGraphTestModel):

    def __init__(self):
        self.prepare_calls = 0

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        return None

    def validate_embedding_graph_inputs(self, inputs):
        return True  # This mock explicitly supports visual input preparation.

    def prepare_transformer_inputs(self, inputs):
        assert not torch.cuda.is_current_stream_capturing()
        self.prepare_calls += 1
        hidden = (
            inputs.input_ids.to(torch.bfloat16)
            .unsqueeze(1)
            .expand(-1, HIDDEN_SIZE)
            .clone()
        )
        locs = inputs.multimodal_inputs.mm_features_locs
        if locs is not None:
            for feature, loc in zip(
                inputs.multimodal_inputs.multimodal_features, locs.tolist()
            ):
                hidden[loc : loc + feature.size(0)].copy_(feature)
        return hidden

    def forward(self, inputs, fmha_impl=None):
        hidden = inputs.input_hiddens
        # Including the tail in this reduction detects stale bucket padding.
        return PyModelOutputs(hidden * 2 + hidden.sum(dim=0))


class TestEmbeddingCudaGraph(unittest.TestCase):
    def test_embedding_runner_does_not_interpret_business_flags(self):
        model = TransformerOnlyCudaGraphEmbeddingModel()
        model.custom_policy_enabled = (
            True  # Only the model's validator defines its input contract.
        )
        runner = CudaGraphRunner()
        runner.init_prefill(model, 2, 8, 8, [8], HIDDEN_SIZE, GROUP_TAGS)
        inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2}, 4)
        self.assertTrue(
            runner.canRun(inputs)
        )  # No custom metadata is required by this model.
        result = runner.forward(inputs).hidden_states.clone()
        hidden = (
            inputs.input_ids.to(torch.bfloat16).unsqueeze(1).expand(-1, HIDDEN_SIZE)
        )
        torch.testing.assert_close(result, hidden * 2 + hidden.sum(dim=0))

    def test_embedding_graph_honors_model_metadata_validation(self):
        class SegmentValidatedModel(TransformerOnlyCudaGraphEmbeddingModel):
            def validate_embedding_graph_inputs(self, inputs):
                for attn in inputs.attention_inputs.values():
                    metadata = attn.host_model_metadata
                    if len(metadata) != 2:
                        return False
                    starts, counts = metadata
                    lengths = attn.input_lengths
                    if any(t.is_cuda or t.dtype != torch.int32 or t.numel() != lengths.numel()
                           for t in metadata):
                        return False
                    if bool(((starts < 0) | (counts < 0) | (starts + counts > lengths)).any()):
                        return False
                return True


        model = SegmentValidatedModel()
        runner = CudaGraphRunner()
        runner.init_prefill(model, 2, 8, 8, [8], HIDDEN_SIZE, GROUP_TAGS)
        inputs = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2}, 4)
        self.assertFalse(runner.canRun(inputs))

        self.assertIsInstance(inputs.attention_inputs, dict)
        self.assertEqual(set(inputs.attention_inputs), set(GROUP_TAGS))

        def set_bounds(starts, counts, device="cpu"):
            # Each getter exposes references into the current C++ map. Fetch
            # fresh references after every assignment replaces that map.
            tagged = inputs.attention_inputs
            for attention in tagged.values():
                attention.host_model_metadata = [
                    torch.tensor(starts, dtype=torch.int32, device=device),
                    torch.tensor(counts, dtype=torch.int32, device=device),
                ]
            # The binding copies the primary input when assigning the mapping.
            # Reassign it so both the primary input and every tag are refreshed.
            inputs.attention_inputs = tagged

        set_bounds([1], [2])
        self.assertTrue(runner.canRun(inputs))
        for starts, counts in (([1], [5]), ([1], [-1]), ([], [])):
            with self.subTest(starts=starts, counts=counts):
                set_bounds(starts, counts)
                self.assertFalse(runner.canRun(inputs))
        set_bounds([1], [2], device="cuda")
        self.assertFalse(runner.canRun(inputs))
        # A rejected request must not prevent a later valid request replaying.
        set_bounds([1], [2])
        self.assertTrue(runner.canRun(inputs))
        runner.prepareAttentionInputs(inputs)
        actual = runner.forward(inputs).hidden_states.clone()
        torch.cuda.synchronize()
        hidden = inputs.input_ids.to(torch.bfloat16).unsqueeze(1).expand(-1, HIDDEN_SIZE)
        torch.testing.assert_close(actual, hidden * 2 + hidden.sum(dim=0))

    def test_transformer_only_32_sequences_exceed_single_sequence_capacity(self):
        runner = CudaGraphRunner()
        model = TransformerOnlyCudaGraphEmbeddingModel()
        runner.init_prefill(model, 32, 512, 8, [6144], HIDDEN_SIZE, GROUP_TAGS)
        self.assertEqual(model.prepare_calls, 0)
        # Alternate a packed 32-item batch and a short batch in the same graph.
        # Every real sequence fits 512; the aggregate batch is much larger.
        for lengths in ([170] * 32, [12], [180] * 32):
            attention = PyAttentionInputs()
            attention.is_prefill = True
            attention.input_lengths = torch.tensor(
                lengths, dtype=torch.int32
            ).pin_memory()
            attention.prefix_lengths = torch.zeros(
                len(lengths), dtype=torch.int32
            ).pin_memory()
            attention.cu_seqlens = torch.tensor(
                [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                dtype=torch.int32,
            ).pin_memory()
            attention.cu_seqlens_device = attention.cu_seqlens.cuda()
            attention.cu_kv_seqlens_device = attention.cu_seqlens_device.clone()
            attention.context_total_kv_length = sum(lengths)
            inputs = _build_common_inputs(
                attention,
                GROUP_TAGS,
                {"full": 1, "aux": 2},
                len(lengths),
                sum(lengths),
                1,
            )
            # Match EmbeddingExecutor's CUDA metadata path, including an empty
            # host cu_seqlens. Reading input_lengths through a CPU pointer here
            # used to SIGSEGV before the graph could replay.
            attn_inputs = attention
            attn_inputs.input_lengths = attn_inputs.input_lengths.cuda()
            attn_inputs.prefix_lengths = attn_inputs.prefix_lengths.cuda()
            attn_inputs.cu_seqlens = torch.empty(0, dtype=torch.int32).pin_memory()
            inputs.attention_inputs = _tag_attention_inputs(
                attn_inputs, GROUP_TAGS, {"full": 1, "aux": 2}
            )
            self.assertTrue(runner.canRun(inputs))
            runner.prepareAttentionInputs(inputs)
            result = runner.forward(inputs).hidden_states.clone()
            torch.cuda.synchronize()
            hidden = (
                inputs.input_ids.to(torch.bfloat16).unsqueeze(1).expand(-1, HIDDEN_SIZE)
            )
            torch.testing.assert_close(result, hidden * 2 + hidden.sum(dim=0))

        over_capacity = _build_prefill_inputs(GROUP_TAGS, {"full": 1, "aux": 2}, 6145)
        self.assertFalse(runner.canRun(over_capacity))

    def test_transformer_only_replays_request_owned_visual_inputs(self):
        model = TransformerOnlyCudaGraphEmbeddingModel()
        runner = CudaGraphRunner()
        runner.init_prefill(
            model,
            2,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4, 8],
            HIDDEN_SIZE,
            GROUP_TAGS,
        )
        self.assertEqual(model.prepare_calls, 0)
        for seq_len, loc, value in [
            (4, 0, 9),
            (6, 3, -2),
            (1, 0, 5),
            (3, 1, 7),
            (4, 2, 1),
        ]:
            with self.subTest(seq_len=seq_len, loc=loc, value=value):
                inputs = _build_prefill_inputs(
                    GROUP_TAGS, {"full": 1, "aux": 2}, seq_len
                )
                inputs.multimodal_inputs.multimodal_features = [
                    torch.full(
                        (1, HIDDEN_SIZE), value, dtype=torch.bfloat16, device="cuda"
                    )
                ]
                inputs.multimodal_inputs.mm_features_locs = torch.tensor(
                    [loc], dtype=torch.int32
                )
                inputs.embedding_inputs.text_tokens_mask = torch.ones(
                    seq_len, dtype=torch.int32, device="cuda"
                )
                self.assertTrue(runner.canRun(inputs))
                # Also exercise the production path that plans before forward.
                runner.prepareAttentionInputs(inputs)
                before = model.prepare_calls
                result = runner.forward(inputs).hidden_states.clone()
                torch.cuda.synchronize()
                self.assertEqual(model.prepare_calls, before + 1)
                hidden = (
                    inputs.input_ids.to(torch.bfloat16)
                    .unsqueeze(1)
                    .expand(-1, HIDDEN_SIZE)
                    .clone()
                )
                hidden[loc] = value
                torch.testing.assert_close(result, hidden * 2 + hidden.sum(dim=0))

    def test_prefill_prepares_bert_embeddings_outside_graph_with_model_parameters(
        self,
    ) -> None:
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
            BertEmbeddingModel(
                word_embedding, position_encoding, token_type_embedding, capture_scalar
            ),
            2,
            TOKENS_PER_BLOCK,
            TOKENS_PER_BLOCK,
            [4],
            HIDDEN_SIZE,
            GROUP_TAGS,
            position_encoding,
            token_type_embedding,
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
            # the request's bert_embedding_inputs IDs outside capture, so the
            # generic field must not disable graph replay.
            inputs.embedding_inputs.combo_tokens_type_ids = torch.tensor(
                token_type_ids, dtype=torch.int32, device="cuda"
            )
            inputs.bert_embedding_inputs.combo_position_ids = torch.tensor(
                position_ids, dtype=torch.int32, device="cuda"
            )
            if provide_position_table:
                # Request-owned tables and scalar are deliberately different:
                # the producer must use its own model parameters.
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

        # A shorter request prepares only its real embeddings, including after
        # a full-bucket request; the Runner clears the fixed buffer's unused tail.
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

        # Model parameters are owned by the graph-external embedding producer.
        with self.subTest(contract="embedding_producer_owns_parameters"):
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
            with self.assertRaisesRegex(
                RuntimeError,
                r"combo_position_ids must have at least one id per token",
            ):
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


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class EmbeddingPrefillGraphTest(unittest.TestCase):
    def test_encoder_graph_replans_layout_and_isolates_padding(self):
        self._check_encoder_graph(causal=False)

    def test_causal_encoder_graph_replans_layout_and_isolates_padding(self):
        self._check_encoder_graph(causal=True)

    def _check_encoder_graph(self, causal):
        if torch.version.hip is None:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                PyFlashinferPrefillImpl as DevicePrefillImpl,
            )
        else:
            from rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter import (
                AiterPrefillImplNonAsm as DevicePrefillImpl,
            )
        from rtp_llm.ops import KvCacheDataType
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        capacity, dim = 8, 64
        configs = SimpleNamespace(
            is_causal=causal,
            kernel_tokens_per_block=8,
            kv_cache_dtype=KvCacheDataType.BASE,
            head_num=1,
            kv_head_num=1,
            size_per_head=dim,
            dtype=torch.bfloat16,
        )
        inputs = PyAttentionInputs()
        inputs.input_lengths = torch.tensor([8, 0], dtype=torch.int32).pin_memory()
        inputs.prefix_lengths = torch.zeros(2, dtype=torch.int32).pin_memory()
        impl = DevicePrefillImpl(
            configs,
            inputs,
            is_cuda_graph=True,
            token_capacity=capacity,
            max_seq_len=capacity,
        )
        qkv = torch.randn(capacity, 3 * dim, dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            impl.forward(qkv, None)
            impl.forward(qkv, None)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = impl.forward(qkv, None)
        pointer = impl.fmha_impl.cu_seqlens_device.data_ptr()
        for lengths in ([3, 2], [8, 0], [2, 0], [1, 5], [4, 4]):
            with self.subTest(lengths=lengths):
                torch.cuda.synchronize()
                inputs.input_lengths.copy_(torch.tensor(lengths, dtype=torch.int32))
                impl.prepare_cuda_graph(inputs)
                qkv.normal_()
                real_tokens = sum(lengths)
                # Adversarial padding would contaminate a real sequence if its
                # length were incorrectly extended to the token bucket.
                qkv[real_tokens:].fill_(20)
                graph.replay()
                torch.cuda.synchronize()
                self.assertEqual(impl.fmha_impl.cu_seqlens_device.data_ptr(), pointer)
                offset = 0
                for length in lengths:
                    if length:
                        q, k, v = qkv[offset : offset + length].chunk(3, dim=-1)
                        expected = torch.nn.functional.scaled_dot_product_attention(
                            q[None, None],
                            k[None, None],
                            v[None, None],
                            is_causal=causal,
                        )[0, 0]
                        torch.testing.assert_close(
                            output[offset : offset + length],
                            expected,
                            atol=2e-2,
                            rtol=2e-2,
                        )
                    offset += length


if __name__ == "__main__":
    unittest.main()
