"""GPU contract tests for the embedding_bert binding."""

from unittest import SkipTest, TestCase, main

import torch
from bert_embedding_test_utils import (
    _VOCAB_SIZE,
    _ZERO_LAYER_HIDDEN_SIZE,
    _build_inputs,
    _build_word_embedding,
)

from rtp_llm.ops import rtp_llm_ops
from rtp_llm.ops.compute_ops import PyModelInputs


class BertEmbeddingOpTest(TestCase):
    """Covers the embedding_bert op contract with a plain word-embedding table."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        self.weight = _build_word_embedding()

    def _embed(
        self,
        weight: torch.Tensor,
        inputs: PyModelInputs,
        mask,
        *,
        input_ids: torch.Tensor | None = None,
    ):
        bert_inputs = inputs.bert_embedding_inputs
        tokens = inputs.input_ids if input_ids is None else input_ids
        output = torch.empty(
            (tokens.size(0), weight.size(1)), dtype=weight.dtype, device=weight.device
        )
        rtp_llm_ops.embedding_bert(
            output,
            tokens,
            weight,
            bert_inputs.combo_position_ids,
            bert_inputs.position_encoding,
            bert_inputs.combo_tokens_type_ids,
            bert_inputs.token_type_embedding,
            bert_inputs.input_embedding_scalar,
            mask,
        )
        return output

    def test_embedding_bert_skips_masked_oov_hashes(self):
        inputs = _build_inputs()
        inputs.input_ids = torch.tensor(
            [123456, 2, -3, 741852], dtype=torch.int32, device="cuda"
        )
        inputs.embedding_inputs.text_tokens_mask = torch.tensor(
            [0, 1, 0, 0], dtype=torch.int32, device="cuda"
        )
        bert_inputs = inputs.bert_embedding_inputs
        bert_inputs.position_encoding = torch.arange(
            16, dtype=torch.float16, device="cuda"
        ).reshape(4, 4)
        bert_inputs.token_type_embedding = torch.tensor(
            [[0.5, 1.0, 1.5, 2.0]], dtype=torch.float16, device="cuda"
        )
        actual = self._embed(
            self.weight, inputs, inputs.embedding_inputs.text_tokens_mask
        )
        masked_rows = torch.tensor([0, 2, 3], device="cuda")
        expected_masked_rows = (
            bert_inputs.position_encoding[masked_rows]
            + bert_inputs.token_type_embedding[0]
        )
        torch.testing.assert_close(actual[masked_rows], expected_masked_rows)
        # Construct the text-row oracle from detached values and explicit
        # scalar arithmetic rather than the module's embedding expression.
        token_weight = self.weight.detach().clone()[2]
        scalar = float(inputs.bert_embedding_inputs.input_embedding_scalar)
        expected_text_row = (
            torch.mul(token_weight, scalar)
            + bert_inputs.position_encoding[1]
            + bert_inputs.token_type_embedding[0]
        )
        torch.testing.assert_close(actual[1], expected_text_row)

    def test_embedding_bert_rejects_invalid_output(self):
        inputs = _build_inputs()
        bert_inputs = inputs.bert_embedding_inputs
        output = torch.empty(
            (4, _ZERO_LAYER_HIDDEN_SIZE), dtype=torch.float32, device="cuda"
        )
        with self.assertRaisesRegex(RuntimeError, "output dtype must match"):
            rtp_llm_ops.embedding_bert(
                output,
                inputs.input_ids,
                self.weight,
                bert_inputs.combo_position_ids,
                bert_inputs.position_encoding,
                bert_inputs.combo_tokens_type_ids,
                bert_inputs.token_type_embedding,
                bert_inputs.input_embedding_scalar,
                inputs.embedding_inputs.text_tokens_mask,
            )

    def test_embedding_bert_mask_semantics_cover_bf16_and_multiple_warps(self):
        hidden_size = 768
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.weight = _build_word_embedding(
                    dtype=dtype, hidden_size=hidden_size
                )
                inputs = _build_inputs(dtype=dtype, hidden_size=hidden_size)
                inputs.input_ids = torch.tensor(
                    [0, 123456, -3, _VOCAB_SIZE - 1],
                    dtype=torch.int32,
                    device="cuda",
                )
                mask = torch.tensor([1, 0, 0, 1], dtype=torch.int32, device="cuda")

                actual = self._embed(self.weight, inputs, mask)

                torch.testing.assert_close(actual[[0, 3]], self.weight[[0, -1]])
                torch.testing.assert_close(
                    actual[1:3],
                    torch.zeros((2, hidden_size), dtype=dtype, device="cuda"),
                )

    def test_embedding_bert_rejects_int64_input_ids(self):
        inputs = _build_inputs()
        with self.assertRaisesRegex(RuntimeError, "input_ids must be int32"):
            self._embed(
                self.weight,
                inputs,
                inputs.embedding_inputs.text_tokens_mask,
                input_ids=inputs.input_ids.to(torch.int64),
            )

    def test_embedding_bert_rejects_empty_position_or_type_table(self):
        def clear_position_table(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.position_encoding = torch.empty(
                (0, _ZERO_LAYER_HIDDEN_SIZE),
                dtype=torch.float16,
                device="cuda",
            )

        def clear_token_type_table(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.token_type_embedding = torch.empty(
                (0, _ZERO_LAYER_HIDDEN_SIZE),
                dtype=torch.float16,
                device="cuda",
            )

        cases = [
            (
                "position_encoding",
                clear_position_table,
                "position embedding table must not be empty",
            ),
            (
                "token_type_embedding",
                clear_token_type_table,
                "token type embedding table must not be empty",
            ),
        ]
        for name, mutate, expected_message in cases:
            with self.subTest(name=name):
                inputs = _build_inputs()
                mutate(inputs)
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    self._embed(
                        self.weight,
                        inputs,
                        inputs.embedding_inputs.text_tokens_mask,
                    )

    def test_embedding_bert_rejects_short_position_or_type_ids(self):
        def shorten_position_ids(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.combo_position_ids = torch.zeros(
                3, dtype=torch.int32, device="cuda"
            )

        def shorten_token_type_ids(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.combo_tokens_type_ids = torch.zeros(
                3, dtype=torch.int32, device="cuda"
            )

        cases = [
            (
                "combo_position_ids",
                shorten_position_ids,
                "combo_position_ids must have at least one id per token, got 3 vs 4",
            ),
            (
                "combo_tokens_type_ids",
                shorten_token_type_ids,
                "combo_tokens_type_ids must have at least one id per token, got 3 vs 4",
            ),
        ]
        for name, mutate, expected_message in cases:
            with self.subTest(name=name):
                inputs = _build_inputs()
                mutate(inputs)
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    self._embed(
                        self.weight,
                        inputs,
                        inputs.embedding_inputs.text_tokens_mask,
                    )

    def test_embedding_bert_rejects_cpu_ids_or_tables(self):
        def move_position_ids(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.combo_position_ids = (
                inputs.bert_embedding_inputs.combo_position_ids.cpu()
            )

        def move_token_type_ids(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.combo_tokens_type_ids = (
                inputs.bert_embedding_inputs.combo_tokens_type_ids.cpu()
            )

        def move_position_table(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.position_encoding = (
                inputs.bert_embedding_inputs.position_encoding.cpu()
            )

        def move_token_type_table(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.token_type_embedding = (
                inputs.bert_embedding_inputs.token_type_embedding.cpu()
            )

        cases = [
            ("combo_position_ids", move_position_ids),
            ("combo_tokens_type_ids", move_token_type_ids),
            ("position_encoding", move_position_table),
            ("token_type_embedding", move_token_type_table),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                inputs = _build_inputs()
                mutate(inputs)
                with self.assertRaisesRegex(
                    RuntimeError, rf"{name} must be a CUDA tensor"
                ):
                    self._embed(
                        self.weight,
                        inputs,
                        inputs.embedding_inputs.text_tokens_mask,
                    )

    def test_embedding_bert_rejects_table_hidden_size_or_dtype_mismatch(self):
        def change_position_hidden_size(inputs: PyModelInputs) -> None:
            current = inputs.bert_embedding_inputs.position_encoding
            inputs.bert_embedding_inputs.position_encoding = torch.zeros(
                (current.size(0), _ZERO_LAYER_HIDDEN_SIZE + 1),
                dtype=torch.float16,
                device="cuda",
            )

        def change_token_type_hidden_size(inputs: PyModelInputs) -> None:
            current = inputs.bert_embedding_inputs.token_type_embedding
            inputs.bert_embedding_inputs.token_type_embedding = torch.zeros(
                (current.size(0), _ZERO_LAYER_HIDDEN_SIZE + 1),
                dtype=torch.float16,
                device="cuda",
            )

        def change_position_dtype(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.position_encoding = (
                inputs.bert_embedding_inputs.position_encoding.float()
            )

        def change_token_type_dtype(inputs: PyModelInputs) -> None:
            inputs.bert_embedding_inputs.token_type_embedding = (
                inputs.bert_embedding_inputs.token_type_embedding.float()
            )

        cases = [
            (
                "position_hidden_size",
                change_position_hidden_size,
                r"position_encoding\.size\(1\).*5 vs 4",
            ),
            (
                "token_type_hidden_size",
                change_token_type_hidden_size,
                r"token_type_embedding\.size\(1\).*5 vs 4",
            ),
            (
                "position_dtype",
                change_position_dtype,
                "position_encoding dtype must match",
            ),
            (
                "token_type_dtype",
                change_token_type_dtype,
                "token_type_embedding dtype must match",
            ),
        ]
        for name, mutate, expected_message in cases:
            with self.subTest(name=name):
                inputs = _build_inputs()
                mutate(inputs)
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    self._embed(
                        self.weight,
                        inputs,
                        inputs.embedding_inputs.text_tokens_mask,
                    )

    def test_embedding_bert_all_zero_mask_with_single_token(self):
        inputs = _build_inputs()
        inputs.input_ids = torch.tensor([123456], dtype=torch.int32, device="cuda")
        inputs.bert_embedding_inputs.combo_position_ids = torch.tensor(
            [0], dtype=torch.int32, device="cuda"
        )
        inputs.bert_embedding_inputs.combo_tokens_type_ids = torch.tensor(
            [0], dtype=torch.int32, device="cuda"
        )
        mask = torch.zeros(1, dtype=torch.int32, device="cuda")

        actual = self._embed(self.weight, inputs, mask)

        torch.testing.assert_close(
            actual,
            inputs.bert_embedding_inputs.position_encoding[0:1]
            + inputs.bert_embedding_inputs.token_type_embedding[0],
        )

    def test_embedding_bert_rejects_invalid_masks(self):
        inputs = _build_inputs()
        cases = [
            (
                "dtype",
                lambda: torch.ones(4, dtype=torch.bool, device="cuda"),
                "text_tokens_mask must be int32",
            ),
            (
                "device",
                lambda: torch.ones(4, dtype=torch.int32),
                "text_tokens_mask must be a CUDA tensor",
            ),
            (
                "contiguous",
                lambda: torch.ones(8, dtype=torch.int32, device="cuda")[::2],
                "text_tokens_mask must be contiguous",
            ),
            (
                "dimension",
                lambda: torch.ones((2, 2), dtype=torch.int32, device="cuda"),
                "text_tokens_mask must be a 1D tensor",
            ),
            (
                "length",
                lambda: torch.ones(3, dtype=torch.int32, device="cuda"),
                "text_tokens_mask must have one id per token, got 3 vs 4",
            ),
        ]

        for name, mask_factory, expected_message in cases:
            with self.subTest(name=name):
                invalid_mask = mask_factory()
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    self._embed(self.weight, inputs, invalid_mask)

    def test_embedding_bert_accepts_empty_mask_as_absent(self):
        inputs = _build_inputs()
        empty_mask = torch.empty(0, dtype=torch.int32, device="cuda")

        actual = self._embed(self.weight, inputs, empty_mask)
        expected = self._embed(self.weight, inputs, None)

        torch.testing.assert_close(actual, expected)

    def test_embedding_bert_applies_scalar_only_to_unmasked_word_embeddings(self):
        inputs = _build_inputs()
        bert_inputs = inputs.bert_embedding_inputs
        bert_inputs.input_embedding_scalar = 2.0
        bert_inputs.position_encoding = torch.arange(
            16, dtype=torch.float16, device="cuda"
        ).reshape(4, 4)
        bert_inputs.token_type_embedding = torch.tensor(
            [[0.5, 1.0, 1.5, 2.0]], dtype=torch.float16, device="cuda"
        )
        mask = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")

        actual = self._embed(self.weight, inputs, mask)
        word_embeddings = self.weight
        expected = (
            word_embeddings[inputs.input_ids.long()] * 2.0
            + bert_inputs.position_encoding
            + bert_inputs.token_type_embedding[0]
        )
        masked_rows = mask == 0
        expected[masked_rows] = (
            bert_inputs.position_encoding[masked_rows]
            + bert_inputs.token_type_embedding[0]
        )

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    main()
