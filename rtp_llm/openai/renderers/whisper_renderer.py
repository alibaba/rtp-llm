"""Whisper chat renderer: audio_url content parts -> forced decoder prompt.

Whisper has no chat template — the decoder prompt is the fixed forced-token
sequence SOT -> language -> task -> (notimestamps). The audio is request
side state for cross attention (mm_skip_token_expansion), so no placeholder
token is needed in input_ids.

Language/task control: the text content part, when present, is parsed as
"<lang>[:task]" (e.g. "en", "zh:transcribe", "auto:translate"); defaults are
"en" and "transcribe". "auto" falls back to "en" until language detection
(Phase B) is wired.
"""

import json
import logging
import os
from typing import List, Optional

from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ContentPart,
    ContentPartTypeEnum,
    RoleEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import CustomChatRenderer, RenderedInputs
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.util import check_with_info

logger = logging.getLogger(__name__)

_SOT_TOKEN = "<|startoftranscript|>"
_NOTIMESTAMPS_TOKEN = "<|notimestamps|>"
_DEFAULT_LANG = "en"
_DEFAULT_TASK = "transcribe"
_KNOWN_TASKS = {"transcribe", "translate"}


class WhisperRenderer(BasicRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generation_cfg = self._load_generation_cfg()

    def _load_generation_cfg(self) -> dict:
        ckpt_path = self.ckpt_path or ""
        candidates = []
        if ckpt_path:
            candidates.append(os.path.join(ckpt_path, "generation_config.json"))
        for path in candidates:
            if os.path.exists(path):
                with open(path) as reader:
                    return json.load(reader)
        logger.warning("whisper renderer: generation_config.json not found at %s", candidates)
        return {}

    def _lang_token_id(self, lang: str) -> Optional[int]:
        lang_to_id = self._generation_cfg.get("lang_to_id") or {}
        token = f"<|{lang}|>"
        if token in lang_to_id:
            return lang_to_id[token]
        return None

    def _task_token_id(self, task: str) -> Optional[int]:
        # task_to_id keys are plain task names ("transcribe"/"translate"),
        # unlike lang_to_id whose keys are <|lang|> special-token strings.
        task_to_id = self._generation_cfg.get("task_to_id") or {}
        if task in task_to_id:
            return task_to_id[task]
        return None

    def _parse_lang_task(self, request: ChatCompletionRequest):
        lang, task = _DEFAULT_LANG, _DEFAULT_TASK
        for message in request.messages:
            if isinstance(message.content, list):
                for part in message.content:
                    if part.type == ContentPartTypeEnum.text and part.text and part.text.strip():
                        text = part.text.strip().lower()
                        if ":" in text:
                            maybe_lang, maybe_task = text.split(":", 1)
                        else:
                            maybe_lang, maybe_task = text, ""
                        if self._lang_token_id(maybe_lang) is not None:
                            lang = maybe_lang
                        if maybe_task in _KNOWN_TASKS:
                            task = maybe_task
                        break
        return lang, task

    def _extract_audio_urls(self, request: ChatCompletionRequest) -> List[str]:
        urls: List[str] = []
        for message in request.messages:
            if isinstance(message.content, list):
                for content_part in message.content:
                    if content_part.type == ContentPartTypeEnum.audio_url:
                        check_with_info(
                            content_part.audio_url is not None,
                            f"audio_url should not be none in {content_part}",
                        )
                        urls.append(content_part.audio_url.url)
        return urls

    def _forced_prompt(self, lang: str, task: str, return_timestamps: bool) -> str:
        prompt = f"{_SOT_TOKEN}<|{lang}|><|{task}|>"
        if not return_timestamps:
            prompt += _NOTIMESTAMPS_TOKEN
        return prompt

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        for message in request.messages:
            if message.role not in [RoleEnum.user]:
                raise Exception(f"unsupported role: {message.role} in whisper request")
        audio_urls = self._extract_audio_urls(request)
        if not audio_urls:
            raise Exception("whisper request must contain at least one audio_url content part")
        lang, task = self._parse_lang_task(request)
        if lang == "auto":
            lang = _DEFAULT_LANG  # language detection arrives with Phase B
        prompt = self._forced_prompt(lang, task, return_timestamps=False)
        input_ids = self.tokenizer.encode(prompt)
        sot_id = self._generation_cfg.get("decoder_start_token_id", 50258)
        no_ts_id = self._generation_cfg.get("no_timestamps_token_id", 50364)
        lang_id = self._lang_token_id(lang)
        task_id = self._task_token_id(task)
        check_with_info(
            lang_id is not None and task_id is not None,
            f"failed to resolve whisper forced tokens for lang={lang} task={task}",
        )
        forced_ids = [sot_id, lang_id, task_id, no_ts_id]
        # The whisper tokenizer post-processor pads prefix/suffix specials
        # (e.g. [50258, 50364, ..., 50257]) when encoding raw text, so only
        # accept its output when it is exactly the forced sequence.
        if input_ids != forced_ids:
            input_ids = forced_ids
        return RenderedInputs(
            rendered_prompt=prompt,
            input_ids=input_ids,
            input_urls=audio_urls,
            input_urls_type=[MMUrlType.AUDIO] * len(audio_urls),
        )


register_renderer("whisper", WhisperRenderer)
