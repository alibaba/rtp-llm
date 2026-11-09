from types import SimpleNamespace
from typing import List

import torch
from unittest import TestCase, main

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.model_config import ModelConfig as PyModelConfig
from rtp_llm.ops import FfnDisAggregateConfig, ModelConfig, PDSepConfig, RuntimeConfig
from rtp_llm.pipeline import Pipeline
from rtp_llm.utils.base_model_datatypes import GenerateOutput, GenerateOutputs
from rtp_llm.utils.word_util import get_stop_word_slices


class StopWordTest(TestCase):
    def _test_single(
        self,
        text: str,
        token_buffer: str,
        expected_text: str,
        expected_token_buffer: str,
        stop_word_str_list: List[str],
        is_final_response: bool,
        return_incremental: bool,
        print_stop_words: bool,
    ):
        generate_config = GenerateConfig(
            return_incremental=return_incremental, print_stop_words=print_stop_words
        )
        generate_output = GenerateOutput(finished=is_final_response)
        stop_word_str_slices = get_stop_word_slices(stop_word_str_list)

        actual_text, actual_token_buffer = Pipeline.process_stop_str(
            generate_config,
            generate_output,
            text,
            "",
            stop_word_str_list,
            stop_word_str_slices,
            token_buffer,
        )

        should_finish = is_final_response or any(
            [stop_word in token_buffer + text for stop_word in stop_word_str_list]
        )

        # Check generate_output.finished and the actual_text, actual_token_buffer
        self.assertEqual(
            actual_text,
            expected_text,
            f"actual_text '{actual_text}' != expected_text '{expected_text}'",
        )
        self.assertEqual(
            actual_token_buffer,
            expected_token_buffer,
            f"actual_token_buffer '{actual_token_buffer}' != expected_token_buffer '{expected_token_buffer}'",
        )
        self.assertEqual(
            generate_output.finished,
            should_finish,
            f"generate_output.finished '{generate_output.finished}' != should_finish '{is_final_response}'",
        )

    def test_part_match(self):
        self._test_single(
            text="hello how are you",
            token_buffer="",
            expected_text="hello how ",
            expected_token_buffer="",
            stop_word_str_list=["are you ok"],
            is_final_response=False,
            return_incremental=False,
            print_stop_words=False,
        )

        self._test_single(
            text="hello how are you",
            token_buffer="",
            expected_text="hello how are you",
            expected_token_buffer="",
            stop_word_str_list=["are you ok"],
            is_final_response=False,
            return_incremental=False,
            print_stop_words=True,
        )

        self._test_single(
            text="hello how are you",
            token_buffer="",
            expected_text="hello how ",
            expected_token_buffer="are you",
            stop_word_str_list=["are you ok"],
            is_final_response=False,
            return_incremental=True,
            print_stop_words=False,
        )

        self._test_single(
            text="hello how are you",
            token_buffer="",
            expected_text="hello how ",
            expected_token_buffer="are you",
            stop_word_str_list=["are you ok"],
            is_final_response=False,
            return_incremental=True,
            print_stop_words=True,
        )

        # final response should not execute part match
        for return_incremental in [False, True]:
            for print_stop_words in [False, True]:
                self._test_single(
                    text="hello how are you",
                    token_buffer="",
                    expected_text="hello how are you",
                    expected_token_buffer="",
                    stop_word_str_list=["are you ok"],
                    is_final_response=True,
                    return_incremental=return_incremental,
                    print_stop_words=print_stop_words,
                )

    def test_middle_match(self):
        for is_final_response in [False, True]:
            for return_incremental in [False, True]:
                self._test_single(
                    text="hello how are you",
                    token_buffer="",
                    expected_text="hello ",
                    expected_token_buffer="",
                    stop_word_str_list=["how are"],
                    is_final_response=is_final_response,
                    return_incremental=return_incremental,
                    print_stop_words=False,
                )
                self._test_single(
                    text="hello how are you",
                    token_buffer="",
                    expected_text="hello how are",
                    expected_token_buffer="",
                    stop_word_str_list=["how are"],
                    is_final_response=is_final_response,
                    return_incremental=return_incremental,
                    print_stop_words=True,
                )

    def test_inc_match(self):
        for is_final_response in [False, True]:
            self._test_single(
                text=" are you",
                token_buffer="hello how",
                expected_text="",
                expected_token_buffer="",
                stop_word_str_list=["hello how are"],
                is_final_response=is_final_response,
                return_incremental=True,
                print_stop_words=False,
            )
            self._test_single(
                text=" are you",
                token_buffer="hello how",
                expected_text="hello how are",
                expected_token_buffer="",
                stop_word_str_list=["hello how are"],
                is_final_response=is_final_response,
                return_incremental=True,
                print_stop_words=True,
            )
            self._test_single(
                text=" are you",
                token_buffer="hello how",
                expected_text="hello ",
                expected_token_buffer="",
                stop_word_str_list=["how are"],
                is_final_response=is_final_response,
                return_incremental=True,
                print_stop_words=False,
            )
            self._test_single(
                text=" are you",
                token_buffer="hello how",
                expected_text="hello how are",
                expected_token_buffer="",
                stop_word_str_list=["how are"],
                is_final_response=is_final_response,
                return_incremental=True,
                print_stop_words=True,
            )

    def test_multi_match(self):
        # multi match should choose first match stop words
        for is_final_response in [False, True]:
            for return_incremental in [False, True]:
                self._test_single(
                    text="hello how are you",
                    token_buffer="",
                    expected_text="hello ",
                    expected_token_buffer="",
                    stop_word_str_list=["you", "how", "are"],
                    is_final_response=is_final_response,
                    return_incremental=return_incremental,
                    print_stop_words=False,
                )
                self._test_single(
                    text="hello how are you",
                    token_buffer="",
                    expected_text="hello how",
                    expected_token_buffer="",
                    stop_word_str_list=["you", "how", "are"],
                    is_final_response=is_final_response,
                    return_incremental=return_incremental,
                    print_stop_words=True,
                )

    def test_multi_part_match(self):
        # stop words match first
        for is_final_response in [False, True]:
            for return_incremental in [False, True]:
                self._test_single(
                    text="hello how are you",
                    token_buffer="",
                    expected_text="hello how are ",
                    expected_token_buffer="",
                    stop_word_str_list=["you", "are you ok"],
                    is_final_response=is_final_response,
                    return_incremental=return_incremental,
                    print_stop_words=False,
                )
                self._test_single(
                    text="hello how are you",
                    token_buffer="",
                    expected_text="hello how are you",
                    expected_token_buffer="",
                    stop_word_str_list=["you", "are you ok"],
                    is_final_response=is_final_response,
                    return_incremental=return_incremental,
                    print_stop_words=True,
                )

        # multi part match use match most stop words
        is_final_response = False
        self._test_single(
            text="hello how are you",
            token_buffer="",
            expected_text="hello how ",
            expected_token_buffer="",
            stop_word_str_list=["you ok", "are you ok"],
            is_final_response=is_final_response,
            return_incremental=False,
            print_stop_words=False,
        )
        self._test_single(
            text="hello how are you",
            token_buffer="",
            expected_text="hello how are you",
            expected_token_buffer="",
            stop_word_str_list=["you ok", "are you ok"],
            is_final_response=is_final_response,
            return_incremental=False,
            print_stop_words=True,
        )
        for print_stop_words in [False, True]:
            self._test_single(
                text="hello how are you",
                token_buffer="",
                expected_text="hello how ",
                expected_token_buffer="are you",
                stop_word_str_list=["you ok", "are you ok"],
                is_final_response=is_final_response,
                return_incremental=True,
                print_stop_words=print_stop_words,
            )


class ThinkingStopWordTest(TestCase):
    def test_decode_paths_keep_think_stops(self):
        class Tokenizer:
            is_fast = True

            def decode(self, tokens, skip_special_tokens=False, **kwargs):
                return "".join(
                    chr(t)
                    for t in tokens
                    if not (skip_special_tokens and t == ord("|"))
                )

            def batch_decode(self, batches, **kwargs):
                return [self.decode(tokens, **kwargs) for tokens in batches]

            def convert_ids_to_tokens(self, tokens, **kwargs):
                return [chr(t) for t in tokens]

            def convert_tokens_to_string(self, tokens):
                return "".join(tokens)

        pipeline = Pipeline.__new__(Pipeline)
        pipeline.tokenizer = Tokenizer()
        pipeline._special_tokens = SimpleNamespace(eos_token_id=0)
        config = GenerateConfig(
            in_think_mode=True,
            end_think_token_ids=[ord("|")],
            return_incremental=True,
            is_streaming=True,
        )
        states, buffers, all_tokens = [], [], []
        emitted = ""
        for chunk in ["ST", "OP|answerST", "OPtail"]:
            output = GenerateOutput(
                output_ids=torch.tensor([list(map(ord, chunk))]), finished=False
            )
            texts, _, states, buffers, all_tokens = pipeline.decode_incremental_tokens(
                config,
                GenerateOutputs(generate_outputs=[output]),
                ["STOP"],
                get_stop_word_slices(["STOP"]),
                [],
                [],
                states,
                buffers,
                all_tokens,
            )
            emitted += texts[0]
        self.assertEqual(emitted, "STOP|answer")
        self.assertTrue(output.finished)

        config.return_incremental = False
        config.skip_special_tokens = True
        output = GenerateOutput(
            output_ids=torch.tensor([list(map(ord, "STOP|answerSTOPtail"))]),
            finished=False,
        )
        texts, _, _ = pipeline.decode_non_incremental_tokens(
            config,
            GenerateOutputs(generate_outputs=[output]),
            ["STOP"],
            get_stop_word_slices(["STOP"]),
            [],
            [],
            [],
        )
        self.assertEqual(texts, ["STOPanswer"])
        self.assertTrue(output.finished)

    def test_token_stop_preserves_reasoning_and_marker(self):
        config = GenerateConfig(in_think_mode=True, end_think_token_ids=[8, 9])
        output = GenerateOutput(finished=False)
        for tokens, expected in [
            ([7], [7]),
            ([7, 8], [7, 8]),
            ([7, 8, 9], [7, 8, 9]),
            ([7, 8, 9, 5, 7], [7, 8, 9, 5]),
        ]:
            with self.subTest(tokens=tokens):
                self.assertEqual(
                    Pipeline.process_stop_id(
                        config, output, tokens, [[7], [8, 9]], [[7], [8], [8, 9]]
                    ),
                    expected,
                )

    def test_cumulative_text_stops_only_in_content(self):
        config = GenerateConfig(in_think_mode=True)
        for text, boundary, expected, finished in [
            ("STOP", 4, "STOP", False),
            ("STOP</think>answerSTOP", 12, "STOP</think>answer", True),
        ]:
            with self.subTest(text=text):
                output = GenerateOutput(finished=False)
                result, buffer = Pipeline.process_stop_str(
                    config,
                    output,
                    text,
                    text,
                    ["STOP"],
                    get_stop_word_slices(["STOP"]),
                    "",
                    content_start=boundary,
                )
                self.assertEqual(result, expected)
                self.assertEqual(buffer, "")
                self.assertEqual(output.finished, finished)

    def test_incremental_text_stop_after_transition(self):
        config = GenerateConfig(in_think_mode=True, return_incremental=True)
        output = GenerateOutput(finished=False)
        result, buffer = Pipeline.process_stop_str(
            config,
            output,
            "STOP</think>answerST",
            "STOP</think>answerST",
            ["STOP"],
            get_stop_word_slices(["STOP"]),
            "",
            content_start=12,
        )
        self.assertEqual(result, "STOP</think>answer")
        self.assertEqual(buffer, "ST")
        self.assertFalse(output.finished)
        result, buffer = Pipeline.process_stop_str(
            config,
            output,
            "OPextra",
            "STOP</think>answerSTOPextra",
            ["STOP"],
            get_stop_word_slices(["STOP"]),
            buffer,
            content_start=12,
        )
        self.assertEqual(result, "")
        self.assertEqual(buffer, "")
        self.assertTrue(output.finished)


if __name__ == "__main__":
    main()
