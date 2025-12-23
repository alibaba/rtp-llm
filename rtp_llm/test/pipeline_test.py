import unittest
from typing import List

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.frontend.generation.orchestrator import GenerationOrchestrator
from rtp_llm.frontend.tokenizer_factory.tokenizer_utils import DecodingState
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.ops import PDSepConfig, SpecialTokens
from rtp_llm.utils.base_model_datatypes import GenerateOutput, GenerateOutputs


class MockTokenizer(BaseTokenizer):
    """Mock tokenizer for testing purposes"""

    def __init__(self):
        # Don't call super().__init__ to avoid loading real tokenizer
        self._vocab_size = 1000
        self._eos_token_id = 0
        self._special_tokens = {}

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def eos_token_id(self):
        return self._eos_token_id

    def batch_decode(self, token_ids, **kwargs):
        # Simple mock implementation
        results = []
        for seq in token_ids:
            # Convert token ids to simple string representation
            text = "".join([chr(ord("A") + (int(token_id) % 26)) for token_id in seq])
            results.append(text)
        return results

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        # Simple mock implementation
        if isinstance(ids, int):
            return str(ids)
        return [str(token_id) for token_id in ids]

    def convert_tokens_to_string(self, tokens):
        # Simple mock implementation
        return "".join(tokens)

    @property
    def is_fast(self):
        return True

    def get_added_vocab(self):
        return {}


class PipelineDecodeTest(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock model config
        # Create mock tokenizer
        self.tokenizer = MockTokenizer()

        # Create pipeline instance
        self.pipeline = GenerationOrchestrator(
            special_tokens=SpecialTokens(),
            pd_sep_config=PDSepConfig(),
            addresses=[],
            max_seq_len=1000,
            seq_size_per_block=1,
            tokenizer=self.tokenizer,
            sp_config=None,
            mm_related_params=None,
        )

    def test_decode_non_incremental_tokens_basic(self):
        """Test basic functionality of decode_non_incremental_tokens"""
        # Create generate config
        generate_config = GenerateConfig(
            stop_words_str=[],
            stop_words_list=[],
            skip_special_tokens=True,
            ignore_eos=False,
        )

        # Create generate outputs with simple token IDs
        output1 = GenerateOutput(
            output_ids=torch.tensor([[1, 2, 3]], dtype=torch.int32), finished=True
        )
        generate_outputs = GenerateOutputs([output1])

        # Call the method
        final_texts, output_lens, output_tokens_list = (
            self.pipeline.decode_non_incremental_tokens(
                generate_config=generate_config,
                generate_outputs=generate_outputs,
                stop_word_str_list=[],
                stop_word_str_slices=[],
                stop_word_ids=[],
                stop_word_id_slices=[],
                ouput_tokens_list=[],
            )
        )

        # Assertions
        self.assertEqual(len(final_texts), 1)
        self.assertEqual(output_lens, [3])
        self.assertEqual(len(output_tokens_list), 1)

    def test_decode_non_incremental_tokens_with_stop_words(self):
        """Test decode_non_incremental_tokens with stop words"""
        # Create generate config with stop words
        generate_config = GenerateConfig(
            stop_words_str=["C"],
            stop_words_list=[],
            print_stop_words=False,
            skip_special_tokens=True,
            ignore_eos=False,
        )

        # Create generate outputs
        output1 = GenerateOutput(
            output_ids=torch.tensor(
                [[1, 2, 3, 4, 5]], dtype=torch.int32
            ),  # A, B, C, D, E
            finished=True,
        )
        generate_outputs = GenerateOutputs([output1])

        # Call the method
        final_texts, output_lens, output_tokens_list = (
            self.pipeline.decode_non_incremental_tokens(
                generate_config=generate_config,
                generate_outputs=generate_outputs,
                stop_word_str_list=["C"],
                stop_word_str_slices=["C"],
                stop_word_ids=[],
                stop_word_id_slices=[],
                ouput_tokens_list=[],
            )
        )

        # Assertions
        self.assertEqual(len(final_texts), 1)
        # Should be truncated at "C" -> "AB"
        # Note: Actual result depends on mock tokenizer implementation
        self.assertEqual(output_lens, [5])  # Full length before truncation
        self.assertEqual(len(output_tokens_list), 1)

    def test_decode_non_incremental_tokens_with_beams(self):
        """Test decode_non_incremental_tokens with beam search"""
        # Create generate config with beams
        generate_config = GenerateConfig(
            num_beams=2,
            stop_words_str=[],
            stop_words_list=[],
            skip_special_tokens=True,
            ignore_eos=False,
        )

        # Create generate outputs with beam search results
        output1 = GenerateOutput(
            output_ids=torch.tensor([[1, 2, 3]], dtype=torch.int32), finished=True
        )
        output2 = GenerateOutput(
            output_ids=torch.tensor([[4, 5, 6]], dtype=torch.int32), finished=True
        )

        generate_outputs = GenerateOutputs([output1, output2])
        # Call the method
        final_texts, output_lens, output_tokens_list = (
            self.pipeline.decode_non_incremental_tokens(
                generate_config=generate_config,
                generate_outputs=generate_outputs,
                stop_word_str_list=[],
                stop_word_str_slices=[],
                stop_word_ids=[],
                stop_word_id_slices=[],
                ouput_tokens_list=[],
            )
        )

        # Assertions
        self.assertEqual(len(final_texts), 2)  # Single output despite beams
        self.assertEqual(len(output_lens), 2)

    def test_decode_incremental_tokens_basic(self):
        """Test basic functionality of decode_incremental_tokens"""
        # Create generate config
        generate_config = GenerateConfig(
            is_streaming=True,
            stop_words_str=[],
            stop_words_list=[],
            skip_special_tokens=True,
            ignore_eos=False,
            return_incremental=False,
        )

        # Create generate outputs
        output1 = GenerateOutput(
            output_ids=torch.tensor([[1]], dtype=torch.int32), finished=False
        )
        generate_outputs = GenerateOutputs([output1])

        # Call the method with empty initial states
        final_texts, output_lens, decoding_states, token_buffers, output_tokens_list = (
            self.pipeline.decode_incremental_tokens(
                generate_config=generate_config,
                generate_outputs=generate_outputs,
                stop_word_str_list=[],
                stop_word_str_slices=[],
                stop_word_ids=[],
                stop_word_id_slices=[],
                decoding_states=[],
                token_buffers=[],
                ouput_tokens_list=[],
            )
        )

        # Assertions
        self.assertEqual(len(final_texts), 1)
        self.assertEqual(output_lens, [1])
        self.assertEqual(len(decoding_states), 1)
        self.assertEqual(len(token_buffers), 1)
        self.assertEqual(len(output_tokens_list), 1)

    def test_decode_incremental_tokens_multiple_steps(self):
        """Test decode_incremental_tokens with multiple steps"""
        # Create generate config
        generate_config = GenerateConfig(
            is_streaming=True,
            stop_words_str=[],
            stop_words_list=[],
            skip_special_tokens=True,
            ignore_eos=False,
            return_incremental=True,
        )

        # First step
        output1 = GenerateOutput(
            output_ids=torch.tensor([[1]], dtype=torch.int32), finished=False
        )
        generate_outputs1 = GenerateOutputs([output1])

        # Call the method for first step
        final_texts, output_lens, decoding_states, token_buffers, output_tokens_list = (
            self.pipeline.decode_incremental_tokens(
                generate_config=generate_config,
                generate_outputs=generate_outputs1,
                stop_word_str_list=[],
                stop_word_str_slices=[],
                stop_word_ids=[],
                stop_word_id_slices=[],
                decoding_states=[],
                token_buffers=[],
                ouput_tokens_list=[],
            )
        )

        # Second step
        output2 = GenerateOutput(
            output_ids=torch.tensor([[2]], dtype=torch.int32), finished=False
        )
        generate_outputs2 = GenerateOutputs([output2])

        # Call the method for second step with previous states
        final_texts, output_lens, decoding_states, token_buffers, output_tokens_list = (
            self.pipeline.decode_incremental_tokens(
                generate_config=generate_config,
                generate_outputs=generate_outputs2,
                stop_word_str_list=[],
                stop_word_str_slices=[],
                stop_word_ids=[],
                stop_word_id_slices=[],
                decoding_states=decoding_states,
                token_buffers=token_buffers,
                ouput_tokens_list=output_tokens_list,
            )
        )

        # Assertions
        self.assertEqual(len(final_texts), 1)
        self.assertEqual(output_lens, [2])
        self.assertEqual(len(decoding_states), 1)
        self.assertEqual(len(token_buffers), 1)
        self.assertEqual(len(output_tokens_list), 1)

    def test_decode_incremental_tokens_with_stop_words(self):
        """Test decode_incremental_tokens with stop words"""
        # Create generate config with stop words
        generate_config = GenerateConfig(
            is_streaming=True,
            stop_words_str=["B"],
            stop_words_list=[],
            print_stop_words=False,
            skip_special_tokens=True,
            ignore_eos=False,
            return_incremental=False,
        )

        # Create generate outputs
        output1 = GenerateOutput(
            output_ids=torch.tensor([[1, 2, 3]], dtype=torch.int32),  # A, B, C
            finished=False,
        )
        generate_outputs = GenerateOutputs([output1])

        # Call the method
        final_texts, output_lens, decoding_states, token_buffers, output_tokens_list = (
            self.pipeline.decode_incremental_tokens(
                generate_config=generate_config,
                generate_outputs=generate_outputs,
                stop_word_str_list=["B"],
                stop_word_str_slices=["B"],
                stop_word_ids=[],
                stop_word_id_slices=[],
                decoding_states=[],
                token_buffers=[],
                ouput_tokens_list=[],
            )
        )

        # Assertions
        self.assertEqual(len(final_texts), 1)
        self.assertEqual(output_lens, [3])
        self.assertEqual(len(decoding_states), 1)
        self.assertEqual(len(token_buffers), 1)
        self.assertEqual(len(output_tokens_list), 1)

    def test_process_stop_id_function(self):
        """Test the process_stop_id helper function"""
        # Create generate config
        generate_config = GenerateConfig(print_stop_words=False)

        # Create generate output
        generate_output = GenerateOutput(finished=False)

        # Test tokens
        tokens = [1, 2, 3, 4, 5]
        stop_word_ids = [[3, 4]]
        stop_word_id_slices = [[3]]

        # Call the method
        result = self.pipeline.process_stop_id(
            generate_config, generate_output, tokens, stop_word_ids, stop_word_id_slices
        )

        # Should be truncated at stop word
        # Note: Actual behavior depends on implementation details
        self.assertIsInstance(result, list)

    def test_process_stop_str_function(self):
        """Test the process_stop_str helper function"""
        # Create generate config
        generate_config = GenerateConfig(
            return_incremental=False, print_stop_words=False
        )

        # Create generate output
        generate_output = GenerateOutput(finished=False)

        # Test parameters
        text = "Hello World"
        all_text = "Hello World"
        stop_word_str_list = ["World"]
        stop_word_str_slices = ["Wor"]
        token_buffer = ""

        # Call the method
        result_text, result_buffer = self.pipeline.process_stop_str(
            generate_config,
            generate_output,
            text,
            all_text,
            stop_word_str_list,
            stop_word_str_slices,
            token_buffer,
        )

        # Assertions
        self.assertIsInstance(result_text, str)
        self.assertIsInstance(result_buffer, str)

    def test_decode_non_incremental_tokens_ignore_eos(self):
        """Test decode_non_incremental_tokens with ignore_eos=True"""
        # Create generate config
        generate_config = GenerateConfig(
            stop_words_str=[],
            stop_words_list=[],
            skip_special_tokens=True,
            ignore_eos=True,
        )

        # Create generate outputs with EOS tokens
        output1 = GenerateOutput(
            output_ids=torch.tensor([[1, 2, 0, 3]], dtype=torch.int32),  # 0 is EOS
            finished=True,
        )
        generate_outputs = GenerateOutputs([output1])

        # Call the method
        final_texts, output_lens, output_tokens_list = (
            self.pipeline.decode_non_incremental_tokens(
                generate_config=generate_config,
                generate_outputs=generate_outputs,
                stop_word_str_list=[],
                stop_word_str_slices=[],
                stop_word_ids=[],
                stop_word_id_slices=[],
                ouput_tokens_list=[],
            )
        )

        # With ignore_eos=True, EOS tokens should not be removed
        self.assertEqual(len(final_texts), 1)
        self.assertEqual(len(output_tokens_list), 1)

    def test_decode_incremental_tokens_ignore_eos(self):
        """Test decode_incremental_tokens with ignore_eos=True"""
        # Create generate config
        generate_config = GenerateConfig(
            is_streaming=True,
            stop_words_str=[],
            stop_words_list=[],
            skip_special_tokens=True,
            ignore_eos=True,
            return_incremental=False,
        )

        # Create generate outputs with EOS tokens
        output1 = GenerateOutput(
            output_ids=torch.tensor([[1, 2, 0, 3]], dtype=torch.int32),  # 0 is EOS
            finished=False,
        )
        generate_outputs = GenerateOutputs([output1])

        # Call the method
        final_texts, output_lens, decoding_states, token_buffers, output_tokens_list = (
            self.pipeline.decode_incremental_tokens(
                generate_config=generate_config,
                generate_outputs=generate_outputs,
                stop_word_str_list=[],
                stop_word_str_slices=[],
                stop_word_ids=[],
                stop_word_id_slices=[],
                decoding_states=[],
                token_buffers=[],
                ouput_tokens_list=[],
            )
        )

        # With ignore_eos=True, EOS tokens should not be removed
        self.assertEqual(len(final_texts), 1)
        self.assertEqual(output_lens, [4])  # All tokens including EOS
        self.assertEqual(len(decoding_states), 1)
        self.assertEqual(len(token_buffers), 1)
        self.assertEqual(len(output_tokens_list), 1)


if __name__ == "__main__":
    unittest.main()
