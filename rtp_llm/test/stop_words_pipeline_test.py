from typing import List
from unittest import TestCase, main

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.frontend.frontend_worker import FrontendWorker
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.base_model_datatypes import GenerateOutput
from rtp_llm.utils.word_util import get_stop_word_slices


class StopWordTest(TestCase):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.config = GptInitModelParameters(
            head_num=1, size_per_head=128, layer_num=1, max_seq_len=32, vocab_size=1024
        )
        self.backend_rpc_server_visitor = BackendRPCServerVisitor(self.config, False)
        self.frontend_worker = FrontendWorker(
            self.config,
            None,
            self.backend_rpc_server_visitor,
        )

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

        actual_text, actual_token_buffer = self.frontend_worker.process_stop_str(
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


if __name__ == "__main__":
    main()
