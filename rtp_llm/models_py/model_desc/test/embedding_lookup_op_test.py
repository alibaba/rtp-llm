from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


class EmbeddingLookupOpTest(TestCase):
    """CUDA contract tests for the shared embedding bindings."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        self.device = torch.device("cuda")
        rows = torch.arange(16, dtype=torch.float32, device=self.device).unsqueeze(1)
        columns = torch.arange(32, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.weight = (rows * 0.5 + columns / 64).to(torch.bfloat16)

    def _generic_embedding(
        self,
        position_ids=None,
        token_type_ids=None,
        mask=None,
        *,
        input_dtype=torch.int32,
        output_dtype=None,
    ):
        input_ids = torch.tensor([1, 2, 3, 4], dtype=input_dtype, device=self.device)
        output = torch.empty(
            (4, 32),
            dtype=self.weight.dtype if output_dtype is None else output_dtype,
            device=self.device,
        )
        rtp_llm_ops.embedding(
            output,
            input_ids,
            self.weight,
            position_ids,
            token_type_ids,
            mask,
        )
        return output

    def test_generic_embedding_rejects_malformed_mask(self):
        cases = (
            (
                "short_mask",
                dict(mask=torch.ones(3, dtype=torch.int32, device=self.device)),
                "text_tokens_mask must have one id per token",
            ),
            (
                "mask_dtype",
                dict(mask=torch.ones(4, dtype=torch.bool, device=self.device)),
                "text_tokens_mask must be int32",
            ),
        )

        for name, kwargs, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, message):
                    self._generic_embedding(**kwargs)

    def test_generic_embedding_ignores_position_and_token_type_ids(self):
        expected = self._generic_embedding()
        position_ids = torch.zeros((2, 2), dtype=torch.int64)
        token_type_ids = torch.zeros(1, dtype=torch.bool)

        actual = self._generic_embedding(position_ids, token_type_ids)

        torch.testing.assert_close(actual, expected)

    def test_generic_embedding_applies_exact_mask(self):
        mask = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device=self.device)

        actual = self._generic_embedding(mask=mask)

        torch.testing.assert_close(actual[[0, 2]], self.weight[[1, 3]])
        torch.testing.assert_close(actual[[1, 3]], torch.zeros_like(actual[[1, 3]]))

    def test_generic_embedding_rejects_int64_input_ids(self):
        with self.assertRaisesRegex(RuntimeError, "input_ids must be int32"):
            self._generic_embedding(input_dtype=torch.int64)

    def test_generic_embedding_rejects_output_dtype_mismatch(self):
        with self.assertRaisesRegex(
            RuntimeError, "output dtype must match embedding weight dtype"
        ):
            self._generic_embedding(output_dtype=torch.float16)

    def _bert_inputs(self, tokens=4):
        ids = torch.arange(tokens, dtype=torch.int32, device=self.device)
        return dict(
            output=torch.empty_like(self.weight[:tokens]),
            input=ids,
            weight=self.weight,
            combo_position_ids=ids,
            position_encoding=self.weight,
            combo_tokens_type_ids=torch.zeros_like(ids),
            token_type_embedding=self.weight[:1],
        )

    def test_bert_lookup_rejects_invalid_inputs(self):
        args = self._bert_inputs()
        cases = [
            ("input", args["input"].long(), "input_ids must be int32"),
            ("output", args["output"].float(), "output dtype must match"),
        ]
        for name in ("combo_position_ids", "combo_tokens_type_ids"):
            ids = args[name]
            cases.extend(
                [
                    (name, ids[:-1], f"{name} must have at least one id per token"),
                    (name, ids.cpu(), f"{name} must be a CUDA tensor"),
                ]
            )
        for name in ("position_encoding", "token_type_embedding"):
            table = args[name]
            cases.extend(
                [
                    (name, table[:0], "embedding table must not be empty"),
                    (name, table.cpu(), f"{name} must be a CUDA tensor"),
                    (name, table.float(), f"{name} dtype must match"),
                    (name, table[:, :-1].contiguous(), rf"{name}\.size\(1\)"),
                ]
            )
        for name, value, message in cases:
            with self.subTest(name=name, message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    rtp_llm_ops.embedding_bert(**{**args, name: value})

    def test_bert_lookup_rejects_invalid_masks(self):
        args = self._bert_inputs()
        mask = torch.ones_like(args["input"])
        for invalid, message in (
            (mask.bool(), "must be int32"),
            (mask.cpu(), "must be a CUDA tensor"),
            (mask.repeat(2)[::2], "must be contiguous"),
            (mask.reshape(2, 2), "must be a 1D tensor"),
            (mask[:-1], "must have one id per token"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    RuntimeError, f"text_tokens_mask {message}"
                ):
                    rtp_llm_ops.embedding_bert(**args, text_tokens_mask=invalid)

    def test_bert_lookup_absent_empty_and_all_masked(self):
        args = self._bert_inputs(tokens=1)
        args["input"] = torch.ones_like(args["input"])
        mask = torch.zeros_like(args["input"])
        for value in (None, mask[:0]):
            with self.subTest(mask=value):
                rtp_llm_ops.embedding_bert(**args, text_tokens_mask=value)
                torch.testing.assert_close(
                    args["output"][0], self.weight[1] + 2 * self.weight[0]
                )
        args["input"].fill_(123456)
        rtp_llm_ops.embedding_bert(**args, text_tokens_mask=mask)
        torch.testing.assert_close(args["output"][0], 2 * self.weight[0])

    def test_embedding_bert_bf16_multiwarp_rows_remain_distinct(self):
        token_count = 65
        input_ids = (
            torch.arange(token_count, dtype=torch.int32, device=self.device) % 16
        )
        position_ids = (
            torch.arange(token_count, dtype=torch.int32, device=self.device) % 32
        )
        token_type_ids = (
            torch.arange(token_count, dtype=torch.int32, device=self.device) % 4
        )
        position_rows = torch.arange(
            32, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        type_rows = torch.arange(4, dtype=torch.float32, device=self.device).unsqueeze(
            1
        )
        position_table = (position_rows * 0.25).expand(-1, 32).to(torch.bfloat16)
        token_type_table = (type_rows * 0.125).expand(-1, 32).to(torch.bfloat16)
        mask = torch.ones(token_count, dtype=torch.int32, device=self.device)
        mask[1::3] = 0
        output = torch.empty(
            (token_count, 32), dtype=torch.bfloat16, device=self.device
        )

        rtp_llm_ops.embedding_bert(
            output,
            input_ids,
            self.weight,
            position_ids,
            position_table,
            token_type_ids,
            token_type_table,
            0.5,
            mask,
        )

        expected = (
            position_table[position_ids.long()]
            + token_type_table[token_type_ids.long()]
        )
        text_rows = mask.bool()
        expected[text_rows] += self.weight[input_ids[text_rows].long()] * 0.5
        torch.testing.assert_close(output, expected)

    def test_embedding_bert_masked_visual_ids_preserve_valid_type_embeddings(self):
        input_ids = torch.tensor(
            [1, -1, -3, 2147483647, -2147483648, 999999, -1, 4],
            dtype=torch.int32,
            device=self.device,
        )
        token_type_ids = torch.tensor(
            [1, -1, -3, 2, 2147483647, -2147483648, 1, 0],
            dtype=torch.int32,
            device=self.device,
        )
        position_ids = torch.arange(8, dtype=torch.int32, device=self.device)
        mask = torch.tensor(
            [1, 0, 0, 0, 0, 0, 0, 1], dtype=torch.int32, device=self.device
        )
        safe_type_ids = torch.tensor(
            [1, 0, 0, 0, 0, 0, 1, 0], dtype=torch.long, device=self.device
        )
        text_rows = mask.bool()

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                # Exercise multiple warps per row at the BERT hidden size.
                hidden_size = 768
                weight = self.weight.to(dtype).repeat(
                    1, hidden_size // self.weight.size(1)
                )
                position_table = (
                    torch.arange(8, dtype=torch.float32, device=self.device)
                    .unsqueeze(1)
                    .expand(-1, hidden_size)
                    / 32
                ).to(dtype)
                token_type_table = (
                    torch.tensor([[0.125], [0.875]], dtype=dtype, device=self.device)
                    .expand(-1, hidden_size)
                    .contiguous()
                )
                output = torch.empty((8, hidden_size), dtype=dtype, device=self.device)

                rtp_llm_ops.embedding_bert(
                    output,
                    input_ids,
                    weight,
                    position_ids,
                    position_table,
                    token_type_ids,
                    token_type_table,
                    0.5,
                    mask,
                )

                expected = position_table + token_type_table[safe_type_ids]
                expected[text_rows] = (
                    weight[input_ids[text_rows].long()] * 0.5
                    + position_table[text_rows]
                    + token_type_table[token_type_ids[text_rows].long()]
                )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    main()
