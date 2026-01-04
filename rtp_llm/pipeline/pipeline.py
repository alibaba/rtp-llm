import asyncio
import logging
import queue
import threading
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Tuple, Union

import torch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.frontend.tokenizer_factory.tokenizer_utils import (
    DecodingState,
    IncrementDecodingUtils,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.ops import SpecialTokens, SpeculativeExecutionConfig, VitSeparation
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
    GenerateResponse,
)
from rtp_llm.utils.multimodal_util import MultimodalInput
from rtp_llm.utils.time_util import current_time_ms
from rtp_llm.utils.util import AtomicCounter
from rtp_llm.utils.word_util import (
    batch_remove_padding_eos,
    get_stop_word_slices,
    match_stop_words,
    remove_padding_eos_with_numpy,
    truncate_response_with_stop_words,
    truncate_token_with_stop_word_id,
)

request_counter = AtomicCounter()


class Pipeline(object):
    def __init__(
        self,
        special_tokens: SpecialTokens,  # SpecialTokens from ModelConfig
        pd_sep_config,  # PDSepConfig from ops
        addresses: list[str],  # RPC addresses for data parallel communication
        max_seq_len: int,  # max_seq_len_ from ModelConfig
        seq_size_per_block: int,  # seq_size_per_block_ from ModelConfig
        tokenizer: Optional[BaseTokenizer],
        sp_config: Optional[SpeculativeExecutionConfig] = None,
        mm_related_params: Optional[
            Any
        ] = None,  # mm_related_params from ModelConfig (optional)
        grpc_config: Optional[Any] = None,  # grpc_config from PyEnvConfigs (optional)
        vit_separation: Optional[VitSeparation] = None,  # Optional VitSeparation
    ):
        self.pd_sep_config = pd_sep_config
        self.tokenizer = tokenizer
        self._special_tokens: SpecialTokens = special_tokens
        self._mm_token: str = ""
        if mm_related_params:
            self._mm_token = mm_related_params.special_tokens.get(
                "default_mm_token", ""
            )

        self.backend_rpc_server_visitor = BackendRPCServerVisitor(
            max_seq_len=max_seq_len,
            seq_size_per_block=seq_size_per_block,
            pd_sep_config=pd_sep_config,
            addresses=addresses,
            sp_config=sp_config,
            grpc_config=grpc_config,
            vit_separation=vit_separation,
        )

    def encode(self, prompt: str):
        assert self.tokenizer is not None
        return self.tokenizer.encode(prompt)

    def decode(self, token_id: int):
        assert self.tokenizer is not None
        return self.tokenizer.decode([token_id])

    @staticmethod
    def create_generate_config(
        generate_config: Union[GenerateConfig, Dict[str, Any]],
        vocab_size: int,
        special_tokens: Any,
        tokenizer: BaseTokenizer,
        generate_env_config,
        **kwargs: Any
    ) -> GenerateConfig:
        if isinstance(generate_config, dict):
            config = GenerateConfig.create_generate_config(generate_config, **kwargs)
        else:
            # 认为是从frontend_worker传递进来的，不需要再处理一遍
            config = generate_config
        config.add_special_tokens(special_tokens)
        config.convert_select_tokens(vocab_size, tokenizer)
        config.add_thinking_params(tokenizer, generate_env_config)
        config.add_stop_ids_from_str(tokenizer)
        return config

    def __call__(
        self, prompt: str, urls: Optional[List[str]] = None, **kwargs: Any
    ) -> Iterator[GenerateResponse]:
        # if not multimodal model, just pass [[]] * len(prompt)
        return self.pipeline(prompt, urls=urls, **kwargs)

    def pipeline(
        self,
        prompt: str,
        request_id: int = None,
        urls: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Iterator[GenerateResponse]:

        q = queue.Queue()

        async def generator():
            res = None
            try:
                res = self.pipeline_async(prompt, request_id, urls, **kwargs)
                async for x in res:
                    q.put(x)
                q.put(None)
            except Exception as e:
                q.put(e)
            finally:
                # if pipline break, should call aclose() to remove async_generator task from loop
                if res is not None:
                    res.aclose()

        def start_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(generator())

        backgroud_thread = threading.Thread(target=start_loop)
        backgroud_thread.start()
        try:
            while True:
                try:
                    r = q.get(timeout=0.01)
                    if r is None:
                        break
                    if isinstance(r, Exception):
                        raise r
                    yield r
                except queue.Empty:
                    continue
        finally:
            backgroud_thread.join()

    @torch.inference_mode()
    def pipeline_async(  # type: ignore
        self,
        prompt: str,
        request_id: int = None,
        urls: Optional[List[str]] = None,
        **kwargs: Any
    ) -> AsyncGenerator[GenerateResponse, None]:
        begin_time = current_time_ms()

        if request_id == None:
            request_id = request_counter.increment()

        generate_config_json = kwargs.pop("generate_config", {})
        generate_config = self.create_generate_config(
            generate_config_json,
            len(self.tokenizer),
            self._special_tokens,
            self.tokenizer,
            **kwargs
        )
        mm_inputs = [MultimodalInput(url) for url in urls] if urls is not None else []

        if len(prompt) == 0:
            raise FtRuntimeException(
                ExceptionType.EMPTY_PROMPT_ERROR,
                "prompt should have at least one token!",
            )
        if type(prompt) is not str:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "expect string prompt, actual: " + str(prompt),
            )
        token_ids = self.tokenizer.encode(prompt)

        if generate_config.sp_advice_prompt != "":
            generate_config.sp_advice_prompt_token_ids = self.tokenizer.encode(
                generate_config.sp_advice_prompt
            )

        kmonitor.report(
            GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time
        )
        kmonitor.report(GaugeMetrics.NUM_BEAMS_METRIC, generate_config.max_num_beams())
        kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(token_ids))
        return self.generate_stream(
            request_id, token_ids, mm_inputs, generate_config, **kwargs
        )

    @staticmethod
    def process_stop_id(
        generate_config: GenerateConfig,
        generate_output: GenerateOutput,
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
        generate_config: GenerateConfig,
        generate_output: GenerateOutput,
        text: str,
        all_text: str,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        token_buffer: str,
        **kwargs: Any
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
        generate_config: GenerateConfig,
        generate_outputs: GenerateOutputs,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        stop_word_ids: List[int],
        stop_word_id_slices: List[int],
        ouput_tokens_list: List[torch.Tensor],
        **kwargs: Any
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
            processed_tokens = Pipeline.process_stop_id(
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
            **kwargs
        )
        newly_decoded_texts = [text.rstrip("\uFFFD") for text in decoded_batch]
        all_texts = newly_decoded_texts

        final_texts = []
        for i in range(len(all_texts)):
            processed_text, _ = Pipeline.process_stop_str(
                generate_config,
                generate_outputs.generate_outputs[i],
                newly_decoded_texts[i],
                all_texts[i],
                stop_word_str_list,
                stop_word_str_slices,
                "",
                **kwargs
            )

            if generate_config.out_prefix:
                processed_text = generate_config.out_prefix + processed_text

            final_texts.append(processed_text)

        return (final_texts, output_lens, ouput_tokens_list)

    def decode_incremental_tokens(
        self,
        generate_config: GenerateConfig,
        generate_outputs: GenerateOutputs,
        stop_word_str_list: List[str],
        stop_word_str_slices: List[str],
        stop_word_ids: List[int],
        stop_word_id_slices: List[int],
        decoding_states: List[DecodingState],
        token_buffers: List[str],
        ouput_tokens_list: List[torch.Tensor],
        **kwargs: Any
    ) -> Tuple[List[str], List[int]]:
        """处理增量解码的逻辑。"""
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

            processed_tokens = Pipeline.process_stop_id(
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
            processed_text, token_buffers[i] = Pipeline.process_stop_str(
                generate_config,
                generate_outputs.generate_outputs[i],
                newly_decoded_texts[i],
                all_texts[i],
                stop_word_str_list,
                stop_word_str_slices,
                token_buffers[i],
                **kwargs
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

    @torch.inference_mode()
    async def generate_stream(
        self,
        request_id: int,
        token_ids: List[int],
        mm_inputs: List[MultimodalInput],
        generate_config: GenerateConfig,
        **kwargs: Any
    ) -> AsyncGenerator[GenerateResponse, None]:
        token_type_ids = []

        token_ids = torch.tensor(token_ids, dtype=torch.int)

        input = GenerateInput(
            request_id=request_id,
            token_ids=token_ids,
            mm_inputs=mm_inputs,
            generate_config=generate_config,
            tokenizer=self.tokenizer,
            token_type_ids=token_type_ids,
        )

        stop_word_strs = generate_config.stop_words_str
        stop_word_str_slices = get_stop_word_slices(stop_word_strs)
        stop_word_ids = generate_config.stop_words_list
        stop_word_id_slices = get_stop_word_slices(stop_word_ids)

        stream: AsyncGenerator[GenerateOutputs, None] = (
            await self.backend_rpc_server_visitor.enqueue(input)
        )

        decoding_states: List[DecodingState] = []
        ouput_tokens_list: List[torch.Tensor] = []
        token_buffers: List[str] = []
        generate_outputs_cache = GenerateOutputs()

        # TODO(xinfei.sxf) add batch and stop test
        async for generate_outputs in stream:
            if not generate_outputs_cache.generate_outputs:
                generate_outputs_cache.generate_outputs = (
                    generate_outputs.generate_outputs
                )
            else:
                generate_outputs_cache.generate_outputs = [
                    out if out.finished else generate_outputs.generate_outputs[i]
                    for i, out in enumerate(generate_outputs_cache.generate_outputs)
                ]
            assert len(generate_outputs_cache.generate_outputs) == len(
                generate_outputs.generate_outputs
            )
            begin_time = current_time_ms()
            is_incremental = (
                not generate_config.has_num_beams() and generate_config.is_streaming
            )
            if is_incremental:
                (
                    generate_texts,
                    output_lens,
                    decoding_states,
                    token_buffers,
                    ouput_tokens_list,
                ) = self.decode_incremental_tokens(
                    generate_config,
                    generate_outputs_cache,
                    stop_word_strs,
                    stop_word_str_slices,
                    stop_word_ids,
                    stop_word_id_slices,
                    decoding_states,
                    token_buffers,
                    ouput_tokens_list,
                    **kwargs
                )
            else:
                (
                    generate_texts,
                    output_lens,
                    ouput_tokens_list,
                ) = self.decode_non_incremental_tokens(
                    generate_config,
                    generate_outputs_cache,
                    stop_word_strs,
                    stop_word_str_slices,
                    stop_word_ids,
                    stop_word_id_slices,
                    ouput_tokens_list,
                    **kwargs
                )

            kmonitor.report(
                GaugeMetrics.POST_PIPELINE_RT_METRIC, current_time_ms() - begin_time
            )

            yield GenerateResponse(
                generate_outputs=generate_outputs_cache, generate_texts=generate_texts
            )
            if (
                all(
                    output.finished
                    for output in generate_outputs_cache.generate_outputs
                )
                and generate_config.aux_info
            ):
                kmonitor.report(
                    GaugeMetrics.FT_ITERATE_COUNT_METRIC,
                    generate_outputs_cache.generate_outputs[0].aux_info.iter_count,
                )
                if len(output_lens) > 0:
                    kmonitor.report(
                        GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC,
                        sum(output_lens) / len(output_lens),
                    )
                break
