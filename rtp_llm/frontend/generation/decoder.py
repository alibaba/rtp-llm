from typing import Any, List, Tuple

import torch

from rtp_llm.frontend.tokenizer_factory.tokenizer_utils import (
    DecodingState,
    IncrementDecodingUtils,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.ops import SpecialTokens
from rtp_llm.utils.word_util import (
    batch_remove_padding_eos,
    get_stop_word_slices,
    match_stop_words,
    remove_padding_eos_with_numpy,
    truncate_response_with_stop_words,
    truncate_token_with_stop_word_id,
)


class GenerationDecoder:
    """Decode generation outputs and handle stop words."""

    def __init__(self, tokenizer: BaseTokenizer, special_tokens: SpecialTokens) -> None:
        self.tokenizer = tokenizer
        self._special_tokens = special_tokens

    @staticmethod
    def process_stop_id(
        generate_config,
        generate_output,
        tokens,
        stop_word_ids: List[List[int]],
        stop_word_id_slices: List[List[int]],
    ):
        if not generate_config.print_stop_words:
            if not generate_output.finished:
                tokens = truncate_token_with_stop_word_id(tokens, stop_word_id_slices)
            else:
                tokens = truncate_token_with_stop_word_id(tokens, stop_word_ids)
        return tokens

    @staticmethod
    def process_stop_str(
        generate_config,
        generate_output,
        text: str,
        all_text: str,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        token_buffer: str,
        **kwargs: Any,
    ):
        if generate_config.return_incremental:
            text = token_buffer + text

        if stop_word_str_list:
            stop_idx, stop_len = match_stop_words(text, stop_word_str_list)
            if stop_idx != -1:
                if not generate_config.print_stop_words:
                    text = text[:stop_idx]
                else:
                    text = text[: stop_idx + stop_len]
                token_buffer = ""
                generate_output.finished = True

        if generate_output.finished:
            return text, token_buffer

        if generate_config.return_incremental or not generate_config.print_stop_words:
            trunc_text = truncate_response_with_stop_words(
                text, stop_word_str_slices, generate_config.is_streaming, True
            )
            if generate_config.return_incremental:
                token_buffer = text[len(trunc_text) :]
            text = trunc_text

        return text, token_buffer

    def decode_non_incremental_tokens(
        self,
        generate_config,
        generate_outputs,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        stop_word_ids: List[int],
        stop_word_id_slices: List[int],
        ouput_tokens_list: List[torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[List[str], List[int]]:
        tokens_lists_for_decode_input = []
        output_lens = []
        token_lists_to_decode = []
        if generate_config.has_num_beams():
            all_output_ids = torch.cat(
                [go.output_ids for go in generate_outputs.generate_outputs], dim=0
            )
            all_output_ids_np = all_output_ids.cpu().numpy()
            if not generate_config.ignore_eos:
                processed_tokens_np_list = batch_remove_padding_eos(
                    all_output_ids_np, self._special_tokens.eos_token_id
                )
                tokens_lists_for_decode_input = [
                    tokens.tolist() for tokens in processed_tokens_np_list
                ]
            else:
                tokens_lists_for_decode_input = all_output_ids_np.tolist()
        else:
            if len(ouput_tokens_list) == 0:
                ouput_tokens_list = [
                    torch.empty(0, dtype=torch.int32)
                    for _ in range(len(generate_outputs.generate_outputs))
                ]
            for i, generate_output in enumerate(generate_outputs.generate_outputs):
                if len(ouput_tokens_list[i]) == 0:
                    ouput_tokens_list[i] = generate_output.output_ids
                else:
                    ouput_tokens_list[i] = torch.cat(
                        (ouput_tokens_list[i], generate_output.output_ids), dim=1
                    )
                    generate_output.output_ids = ouput_tokens_list[i]
                tokens = generate_output.output_ids.cpu().numpy().flatten()
                if not generate_config.ignore_eos:
                    tokens = remove_padding_eos_with_numpy(
                        tokens, self._special_tokens.eos_token_id
                    )
                else:
                    tokens = tokens.reshape(-1)
                tokens_lists_for_decode_input.append(tokens)
        for i, generate_output in enumerate(generate_outputs.generate_outputs):
            tokens_list = tokens_lists_for_decode_input[i]
            output_lens.append(len(tokens_list))
            processed_tokens = GenerationDecoder.process_stop_id(
                generate_config,
                generate_output,
                tokens_list,
                stop_word_ids,
                stop_word_id_slices,
            )
            token_lists_to_decode.append(processed_tokens)

        decoded_batch = self.tokenizer.batch_decode(
            token_lists_to_decode,
            skip_special_tokens=generate_config.skip_special_tokens,
            **kwargs,
        )
        newly_decoded_texts = [text.rstrip("\uFFFD") for text in decoded_batch]
        all_texts = newly_decoded_texts

        final_texts = []
        for i in range(len(all_texts)):
            processed_text, _ = GenerationDecoder.process_stop_str(
                generate_config,
                generate_outputs.generate_outputs[i],
                newly_decoded_texts[i],
                all_texts[i],
                stop_word_str_list,
                stop_word_str_slices,
                "",
                **kwargs,
            )

            if generate_config.out_prefix:
                processed_text = generate_config.out_prefix + processed_text

            final_texts.append(processed_text)

        return (final_texts, output_lens, ouput_tokens_list)

    def decode_incremental_tokens(
        self,
        generate_config,
        generate_outputs,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        stop_word_ids: List[int],
        stop_word_id_slices: List[int],
        decoding_states: List[DecodingState],
        token_buffers: List[str],
        ouput_tokens_list: List[torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[List[str], List[int]]:
        num_outputs = len(generate_outputs.generate_outputs)
        if len(token_buffers) == 0:
            token_buffers = [""] * num_outputs

        if len(decoding_states) == 0:
            decoding_states = [DecodingState() for _ in range(num_outputs)]

        if len(ouput_tokens_list) == 0:
            ouput_tokens_list = [
                torch.empty(0, dtype=torch.int32) for _ in range(num_outputs)
            ]

        newly_decoded_texts = []
        all_texts = []
        output_lens = []
        ignore_eos = generate_config.ignore_eos
        for i, generate_output in enumerate(generate_outputs.generate_outputs):
            ouput_tokens_list[i] = torch.cat(
                (ouput_tokens_list[i], generate_output.output_ids), dim=1
            )
            full_tokens_tensor = ouput_tokens_list[i]
            tokens_np = full_tokens_tensor.cpu().numpy().flatten()
            if not ignore_eos:
                tokens_list = remove_padding_eos_with_numpy(
                    tokens_np, self._special_tokens.eos_token_id
                ).tolist()
            else:
                tokens_list = tokens_np.tolist()

            output_lens.append(len(tokens_list))

            processed_tokens = GenerationDecoder.process_stop_id(
                generate_config,
                generate_output,
                tokens_list,
                stop_word_ids,
                stop_word_id_slices,
            )
            new_text = IncrementDecodingUtils.detokenize_incrementally(
                self.tokenizer, processed_tokens, decoding_states[i]
            )
            decoding_states[i].all_text += new_text

            text_to_return = (
                new_text
                if generate_config.return_incremental
                else decoding_states[i].all_text
            )
            newly_decoded_texts.append(text_to_return)
            all_texts.append(decoding_states[i].all_text)

        final_texts = []
        for i in range(len(all_texts)):
            processed_text, token_buffers[i] = GenerationDecoder.process_stop_str(
                generate_config,
                generate_outputs.generate_outputs[i],
                newly_decoded_texts[i],
                all_texts[i],
                stop_word_str_list,
                stop_word_str_slices,
                token_buffers[i],
                **kwargs,
            )

            if generate_config.out_prefix:
                processed_text = generate_config.out_prefix + processed_text

            final_texts.append(processed_text)

        return (
            final_texts,
            output_lens,
            decoding_states,
            token_buffers,
            ouput_tokens_list,
        )

    @staticmethod
    def stop_word_slices(stop_words):
        return get_stop_word_slices(stop_words)
