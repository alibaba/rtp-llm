import copy
import functools
import json
import os
from typing import List

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPartTypeEnum,
    FinisheReason,
    RoleEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    OutputDelta,
    RenderedInputs,
    RendererParams,
    StreamStatus,
)
from rtp_llm.openai.renderers.llava_renderer import Conversation, SeparatorStyle
from rtp_llm.utils.base_model_datatypes import GenerateOutput
from rtp_llm.utils.fuser import fetch_remote_file_to_local
from rtp_llm.utils.multimodal_util import MMUrlType


class InternVLConversation(Conversation):
    def render_messages(
        self, messages: List[ChatMessage], video_frame_num: int = 8
    ) -> PromptWithMMInput:
        prompt: str = ""
        urls: List[str] = []
        types: List[MMUrlType] = []

        if messages[0].role != RoleEnum.system:
            prompt = self.roles[RoleEnum.system] + self.system_content + self.seps[0]

        for index, message in enumerate(messages):
            if isinstance(message.content, str):
                prompt += (
                    f"{self.roles[message.role]}{self.connector[0]}{message.content}"
                )
            elif isinstance(message.content, list):
                now_prompt = ""
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert isinstance(content_part.text, str)
                        now_prompt += content_part.text
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert content_part.image_url != None
                        urls.append(content_part.image_url.url)
                        now_prompt += "<image>\n"
                        types.append(MMUrlType.IMAGE)
                    elif content_part.type == ContentPartTypeEnum.video_url:
                        assert content_part.video_url != None
                        urls.append(content_part.video_url.url)
                        now_prompt += "".join(
                            [f"Frame{i+1}: <image>\n" for i in range(video_frame_num)]
                        )
                        types.append(MMUrlType.VIDEO)

                prompt += f"{self.roles[message.role]}" + self.connector[0] + now_prompt
            if self.sep_style == SeparatorStyle.TWO:
                prompt += self.seps[index % 2]
            else:
                prompt += self.seps[0]
        prompt += self.roles[RoleEnum.assistant] + self.connector[1]
        return PromptWithMMInput(prompt, urls, types)


conv_internlm2 = InternVLConversation(
    system_content="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
    roles={
        RoleEnum.user: "<|im_start|>user\n",
        RoleEnum.assistant: "<|im_start|>assistant\n",
        RoleEnum.system: "<|im_start|>system\n",
    },
    sep_style=SeparatorStyle.SINGLE,
    seps=["<|im_end|>"],
    connector=["", ""],
)

conv_phi3 = InternVLConversation(
    system_content="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
    roles={
        RoleEnum.user: "<|user|>\n",
        RoleEnum.assistant: "<|assistant|>\n",
        RoleEnum.system: "<|system|>\n",
    },
    sep_style=SeparatorStyle.SINGLE,
    seps=["<|end|>"],
    connector=["", ""],
)

conv_internvl2_5 = InternVLConversation(
    system_content="你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。",
    roles={
        RoleEnum.user: "<|im_start|>user\n",
        RoleEnum.assistant: "<|im_start|>assistant\n",
        RoleEnum.system: "<|im_start|>system\n",
    },
    sep_style=SeparatorStyle.SINGLE,
    seps=["<|im_end|>\n"],
    connector=["", ""],
)

conv_templates = {
    "Hermes-2": conv_internlm2,
    "internlm2-chat": conv_internlm2,
    "phi3-chat": conv_phi3,
    "internvl2_5": conv_internvl2_5,
}


from typing import Any, Optional

from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig


class InternVLRenderer(CustomChatRenderer):
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
        generate_env_config: GenerateEnvConfig,
        render_config: Optional[RenderConfig] = None,
        ckpt_path: Optional[str] = None,
        misc_config: Optional[Any] = None,
        vit_config: Optional[Any] = None,
    ):
        super().__init__(
            tokenizer,
            renderer_params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )
        self.roles = {RoleEnum.user: "USER", RoleEnum.assistant: "ASSISTANT"}
        self.video_frame_num = 8

    @staticmethod
    def _removes_decoder_leading_space(
        decoded_prev_token: str, decoded_string: str
    ) -> bool:
        return bool(
            decoded_string.startswith(" ")
            and (not decoded_prev_token or not decoded_prev_token.startswith(" "))
        )

    @classmethod
    def _normalize_decoded_delta(
        cls, decoded_prev_token: str, decoded_string: str
    ) -> str:
        """Apply InternVL's context-sensitive leading-space normalization."""
        if cls._removes_decoder_leading_space(decoded_prev_token, decoded_string):
            return decoded_string[len(decoded_prev_token) + 1 :]
        return decoded_string[len(decoded_prev_token) :]

    @staticmethod
    def _account_for_invisible_leading_space(
        status: StreamStatus, removed_leading_space: bool
    ) -> None:
        """Account for a standalone space token swallowed by normalization."""
        if (
            removed_leading_space
            and status.pending_logprobs
            and status.pending_logprobs[0].token == " "
        ):
            status.pending_logprobs.pop(0)
            # This counter is also the logprob ledger's accounted-token cursor.
            status.emitted_logprob_token_count += 1

    def _trim_pending_logprobs_to_visible_text(
        self, status: StreamStatus, visible_delta: str
    ) -> None:
        """Trim logprobs using the same decoded-text semantics as InternVL."""
        if not status.pending_logprobs:
            return
        if not visible_delta:
            status.pending_logprobs = []
            return

        output_ids = list(status.output_ids)
        context_count = status.emitted_logprob_token_count
        pending_count = len(status.pending_logprobs)
        if context_count + pending_count > len(output_ids):
            return

        # Accounted invisible tokens have not entered last_output_ids, but they
        # remain part of the decoder context for the pending visible prefix.
        previous_ids = (
            list(status.prev_token_id)
            + output_ids[len(status.last_output_ids) : context_count]
        )
        pending_ids = output_ids[context_count : context_count + pending_count]
        decoded_prev_token = self.tokenizer.decode(previous_ids)
        keep_count = 0
        found_visible_prefix = False
        for index in range(1, pending_count + 1):
            candidate_string = self.tokenizer.decode(previous_ids + pending_ids[:index])
            candidate_delta = self._normalize_decoded_delta(
                decoded_prev_token, candidate_string
            )
            if candidate_delta == visible_delta or candidate_delta.startswith(
                visible_delta
            ):
                keep_count = index
                found_visible_prefix = True
                break
            if visible_delta.startswith(candidate_delta):
                keep_count = index
                found_visible_prefix = True

        # If a context-sensitive tokenizer cannot reproduce the visible prefix,
        # retaining the records is safer than guessing away a sampled token.
        if found_visible_prefix:
            status.pending_logprobs = status.pending_logprobs[:keep_count]

    def _render_messages(self, messages: List[ChatMessage]) -> PromptWithMMInput:
        # Use checkpoint path from model_config
        ckpt_path: str = self.model_config.checkpoint_path
        config_path = os.path.join(fetch_remote_file_to_local(ckpt_path), "config.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
                template_name = config_json["template"]
                conv_template = conv_templates[template_name]
                prefix = "<s>" if template_name == "internlm2-chat" else ""
                res = conv_template.render_messages(messages, 8)
                res.prompt = prefix + res.prompt
                return res
        else:
            raise Exception("no config.json found")

    async def _update_single_status(
        self,
        status: StreamStatus,
        output: GenerateOutput,
        max_new_tokens: int,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
    ) -> OutputDelta:
        if status.finish_reason != None:
            return await self._create_empty_delta(status.output.aux_info)
        status.update_output(
            output,
            functools.partial(self._check_finish_reason, max_new_tokens=max_new_tokens),
            self._remove_stop_word_ids,
        )
        self._accumulate_log_probs_from_tensors(
            status,
            output.output_ids,
            output.token_logprobs,
            output.top_logprob_token_ids,
            output.top_logprobs,
        )
        decoded_prev_token = self.tokenizer.decode(status.prev_token_id)
        decoded_string = self.tokenizer.decode(status.tokens_to_decode)
        decoded_string, trimmed_incomplete_utf8 = self._prepare_decoded_string_for_emit(
            status, decoded_string, is_streaming
        )
        if decoded_string is None:
            return await self._create_empty_delta(output.aux_info)
        # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
        removed_leading_space = self._removes_decoder_leading_space(
            decoded_prev_token, decoded_string
        )
        status.delta_output_string = self._normalize_decoded_delta(
            decoded_prev_token, decoded_string
        )
        self._account_for_invisible_leading_space(status, removed_leading_space)

        # Process stop words: truncate complete stop words, detect partial stop words
        untruncated_delta = status.delta_output_string
        status.delta_output_string, should_buffer = self._process_stop_words(
            untruncated_delta,
            stop_words_str,
            stop_word_slice_list,
            is_streaming,
            status,
        )
        if trimmed_incomplete_utf8 or len(status.delta_output_string) < len(
            untruncated_delta
        ):
            self._trim_pending_logprobs_to_visible_text(
                status, status.delta_output_string
            )

        if should_buffer:
            return await self._create_empty_delta(output.aux_info)

        # Build delta output
        if len(status.delta_output_string) > 0:
            current_logprobs = self._take_pending_logprobs(status)
            status.update_result()
            delta = OutputDelta(
                output_str=status.delta_output_string,
                logprobs=current_logprobs,
                input_length=output.aux_info.input_len,
                output_length=output.aux_info.output_len,
                reuse_length=output.aux_info.reuse_len,
            )
            status.delta_output_string = ""
            return delta
        else:
            # A decoder-only leading space can make a sampled token temporarily
            # invisible. Keep its probability pending until later text makes the
            # token visible. At termination there is no later text, so discard it
            # instead of emitting logprobs with an empty content delta.
            if status.finish_reason is not None:
                self._trim_pending_logprobs_to_visible_text(status, "")
            return await self._create_empty_delta(output.aux_info)

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        messages = copy.deepcopy(request.messages)
        prompt_and_mm_input = self._render_messages(messages)
        input_ids = self.tokenizer.encode(prompt_and_mm_input.prompt)
        return RenderedInputs(
            input_ids=input_ids,
            input_urls=prompt_and_mm_input.urls,
            rendered_prompt=prompt_and_mm_input.prompt,
            input_urls_type=prompt_and_mm_input.mm_types,
        )


register_renderer("internvl", InternVLRenderer)
