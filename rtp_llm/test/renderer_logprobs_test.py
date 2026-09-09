import math
from unittest import IsolatedAsyncioTestCase, TestCase, main

import torch

from rtp_llm.openai.api_datatype import ChatCompletionRequest, ChatMessage, RoleEnum
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    StreamStatus,
    StreamStatusSync,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput


class Tokenizer:
    def decode(self, ids):
        return "".join("abc!"[int(i)] for i in ids)


class RendererLogprobsTest(TestCase):
    def setUp(self):
        self.renderer = CustomChatRenderer.__new__(CustomChatRenderer)
        self.renderer.tokenizer = Tokenizer()
        self.request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            logprobs=True,
            top_logprobs=2,
        )

    def test_all_accepted_tokens_have_their_own_probability(self):
        for status_type in (StreamStatus, StreamStatusSync):
            with self.subTest(status_type=status_type):
                status = status_type(self.request)
                self.renderer._append_log_probs(
                    status,
                    torch.tensor(
                        [
                            [0.1, 0.6, 0.3],
                            [0.7, 0.2, 0.1],
                            [0.2, 0.3, 0.5],
                        ]
                    ),
                    torch.tensor([[1, 0, 2]]),
                )
                status.output_ids = [1, 0, 2]
                result = self.renderer._take_log_probs(status)
                self.assertEqual([p.token for p in result], ["b", "a", "c"])
                for item, probability in zip(result, [0.6, 0.7, 0.5]):
                    self.assertAlmostEqual(
                        item.logprob, math.log(probability), places=6
                    )
                    self.assertEqual(len(item.top_logprobs), 2)
                self.assertIsNone(self.renderer._take_log_probs(status))

    def test_buffered_tokens_keep_probabilities_until_text_is_emitted(self):
        status = StreamStatus(self.request)
        self.renderer._append_log_probs(
            status, torch.tensor([[0.1, 0.9]]), torch.tensor([1])
        )
        self.renderer._append_log_probs(
            status, torch.tensor([[0.8, 0.2]]), torch.tensor([0])
        )
        status.output_ids = [1, 0]
        result = self.renderer._take_log_probs(status)
        self.assertEqual([p.token for p in result], ["b", "a"])
        self.assertAlmostEqual(result[0].logprob, math.log(0.9), places=6)

    def test_eos_and_tokens_after_eos_are_not_returned(self):
        status = StreamStatus(self.request)
        self.renderer._append_log_probs(
            status,
            torch.tensor(
                [
                    [0.2, 0.6, 0.2],
                    [0.1, 0.1, 0.8],
                    [0.6, 0.2, 0.2],
                ]
            ),
            torch.tensor([1, 2, 0]),
        )
        status.output_ids = [1]  # The renderer has removed EOS and its suffix.
        result = self.renderer._take_log_probs(status)
        self.assertEqual([p.token for p in result], ["b"])
        self.assertIsNone(self.renderer._take_log_probs(status))

    def test_omitted_and_zero_top_logprobs_preserve_one_candidate(self):
        for status_type in (StreamStatus, StreamStatusSync):
            for top_logprobs in (None, 0):
                with self.subTest(status_type=status_type, top_logprobs=top_logprobs):
                    self.request.top_logprobs = top_logprobs
                    status = status_type(self.request)
                    self.renderer._append_log_probs(
                        status, torch.tensor([[0.8, 0.2]]), torch.tensor([1])
                    )
                    status.output_ids = [1]
                    result = self.renderer._take_log_probs(status)
                    self.assertEqual(result[0].token, "b")
                    self.assertAlmostEqual(result[0].logprob, math.log(0.2), places=6)
                    self.assertEqual([p.token for p in result[0].top_logprobs], ["a"])

    def test_candidates_are_limited_by_nonzero_probabilities_per_token(self):
        self.request.top_logprobs = 20
        for status_type in (StreamStatus, StreamStatusSync):
            with self.subTest(status_type=status_type):
                status = status_type(self.request)
                self.renderer._append_log_probs(
                    status,
                    torch.tensor([[0.0, 1.0, 0.0], [0.6, 0.0, 0.4]]),
                    torch.tensor([1, 2]),
                )
                status.output_ids = [1, 2]
                result = self.renderer._take_log_probs(status)
                self.assertEqual([p.token for p in result[0].top_logprobs], ["b"])
                self.assertEqual([p.token for p in result[1].top_logprobs], ["a", "c"])
                self.assertTrue(
                    all(
                        math.isfinite(p.logprob)
                        for item in result
                        for p in item.top_logprobs
                    )
                )

    def test_zero_selected_probability_uses_existing_sentinel(self):
        status = StreamStatus(self.request)
        self.renderer._append_log_probs(
            status, torch.tensor([[0.0, 1.0]]), torch.tensor([0])
        )
        status.output_ids = [0]
        result = self.renderer._take_log_probs(status)
        self.assertEqual(result[0].logprob, -9999)

    def test_missing_rows_fail_instead_of_using_last_token_probability(self):
        status = StreamStatus(self.request)
        with self.assertRaisesRegex(ValueError, "one row"):
            self.renderer._append_log_probs(
                status, torch.tensor([[0.2, 0.8]]), torch.tensor([0, 1])
            )

    def test_sync_status_consumes_the_supplied_token_tensor(self):
        status = StreamStatusSync(self.request)
        status.update_output_sync(
            torch.tensor([[0, 1]]), 2, lambda *_: None, lambda ids, _: ids
        )
        self.assertEqual(status.output_ids, [0, 1])

    def test_disabled_logprobs_requires_no_probability_tensor(self):
        self.request.logprobs = False
        status = StreamStatus(self.request)
        self.renderer._append_log_probs(status, None, torch.tensor([1]))
        self.assertIsNone(self.renderer._take_log_probs(status))

    def test_sync_mtp_delta_serializes_all_token_scores(self):
        self.renderer.eos_token_id = 3
        self.renderer.max_seq_len = 100
        self.renderer.stop_words_id_list = []
        self.renderer.get_all_extra_stop_word_ids_list = lambda: []
        status = StreamStatusSync(self.request)
        delta = self.renderer._update_single_status_sync(
            status,
            1,
            2,
            0,
            torch.tensor([[[0.1, 0.8, 0.1], [0.6, 0.2, 0.2]]]),
            torch.tensor([[1, 0]]),
            10,
            [],
            [],
            True,
        )
        response = self.renderer._generate_stream_response_sync([delta])
        self.assertEqual(response.choices[0].delta.content, "ba")
        self.assertEqual(
            [p.token for p in response.choices[0].logprobs.content], ["b", "a"]
        )


class RendererLogprobsAsyncTest(IsolatedAsyncioTestCase):
    async def test_mtp_chunk_with_eos_keeps_only_visible_token_scores(self):
        renderer = CustomChatRenderer.__new__(CustomChatRenderer)
        renderer.tokenizer = Tokenizer()
        renderer.eos_token_id = 3
        renderer.max_seq_len = 100
        renderer.stop_words_id_list = []
        renderer.get_all_extra_stop_word_ids_list = lambda: []
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            logprobs=True,
            top_logprobs=2,
        )
        status = StreamStatus(request)
        output = GenerateOutput(
            output_ids=torch.tensor([[1, 0, 3]]),
            all_probs=torch.tensor(
                [[[0.1, 0.7, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]]]
            ),
            aux_info=AuxInfo(input_len=1, output_len=3),
            finished=True,
        )
        delta = await renderer._update_single_status(status, output, 10, [], [], True)
        self.assertEqual(delta.output_str, "ba")
        self.assertEqual([p.token for p in delta.logprobs], ["b", "a"])
        self.assertIsNone(await renderer._generate_log_probs(status, output))


if __name__ == "__main__":
    main()
