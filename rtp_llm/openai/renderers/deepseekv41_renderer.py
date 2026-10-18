"""V4.1 recipe protocol with canonical checkpoint image metadata."""

import base64
import importlib.util
import os
import sys
import uuid
from collections.abc import Mapping

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.openai.api_datatype import (
    DeltaMessage,
    FinisheReason,
    FunctionCall,
    ToolCall,
)
from rtp_llm.openai.reasoning_effort import normalize_v41_reasoning_effort
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import (
    OutputDelta,
    RenderedInputs,
    StreamStatus,
)
from rtp_llm.openai.renderers.deepseekv4_renderer import DeepseekV4Renderer
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv41_detector import (
    DeepSeekV41Detector,
)
from rtp_llm.openai.renderers.v41_recipe import (
    convert_request,
    render_request,
    resolve_thinking,
)
from rtp_llm.openai.renderers.v41_stream import V41StreamParser
from rtp_llm.utils.base_model_datatypes import MMUrlType

IMAGE_PLACEHOLDER = "<\uff5cdeepseek_image\uff5c>"


def _image_records_to_urls(prompt_tokens, images, image_token_id):
    """Map recipe media records to the generic MultimodalInput URL contract.

    This frontend helper only validates placeholder cardinality and preserves
    media bytes/URLs. Decode, resize, grid planning and token-span construction
    stay inside the ViT backend.
    """
    if sum(token == image_token_id for token in prompt_tokens) != len(images):
        raise ValueError(
            "image placeholder count does not match the supplied image records"
        )
    urls = []
    for record in images:
        if not isinstance(record, Mapping):
            raise ValueError("V4.1 image records must be mappings")
        source = record.get("source")
        source = source if isinstance(source, Mapping) else {}
        inline = record.get("data")
        if inline is None:
            inline = source.get("data")
        url = source.get("url") or record.get("url")
        if inline is None and isinstance(url, str) and url.startswith("data:"):
            header, separator, inline = url.partition(",")
            if not separator or not header.endswith(";base64"):
                raise ValueError("image data URLs require base64 encoding")
        if inline is not None:
            if isinstance(inline, str):
                inline = base64.b64decode(inline, validate=True)
            if not isinstance(inline, bytes):
                raise ValueError("V4.1 inline image must be bytes or base64 text")
            urls.append(
                "data:application/octet-stream;base64,"
                + base64.b64encode(inline).decode("ascii")
            )
            continue
        if not isinstance(url, str) or not url:
            raise ValueError("image record has no supported data or URL source")
        urls.append(url)
    return urls


class V41StreamStatus(StreamStatus):
    def __init__(self, request, options):
        super().__init__(request)
        self.call_prefix = uuid.uuid4().hex
        self.parser = V41StreamParser(
            thinking=options.thinking,
            parse_tools=bool(options.tools),
            force_tools=options.force_tools,
            json_output=options.json_output,
            stop=options.stop,
        )


class DeepseekV41Renderer(DeepseekV4Renderer):
    detector_class = DeepSeekV41Detector
    dsml_tool_calls_marker = "<\uff5cDSML\uff5c calls>"
    parses_user_stop_sequences = True
    # Recipe rendering/tokenization is CPU-only. Image decode/preprocessing is
    # owned by the normal multimodal engine, not by this frontend thread.
    render_chat_in_thread = True

    def _load_encoding_module(self, ckpt_path: str):
        script = os.path.join(ckpt_path, "encoding", "encoding.py")
        if not os.path.isfile(script):
            raise FileNotFoundError(f"V4.1 checkpoint encoder is missing: {script}")
        spec = importlib.util.spec_from_file_location(
            "rtp_deepseek_v41_encoding", script
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load V4.1 checkpoint encoder: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.image_token_id = V41Config.from_path(ckpt_path).image_token_id
        return module

    def _create_detector(self, request):
        if self._active_tools_for_request(request):
            return DeepSeekV41Detector(
                self.encoding_module,
                "thinking" if self.in_think_mode(request) else "chat",
            )
        return None

    def _normalize_reasoning_effort(self, effort):
        try:
            return normalize_v41_reasoning_effort(effort)
        except ValueError as error:
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS, str(error)
            ) from error

    @staticmethod
    def _request_dict(request):
        return request.model_dump(mode="json", exclude_unset=True, exclude_none=True)

    def resolve_thinking_mode(self, request):
        from rtp_llm.config.generate_config import ThinkingMode

        return (
            ThinkingMode.ENABLED
            if self.in_think_mode(request)
            else ThinkingMode.DISABLED
        )

    def in_think_mode(self, request):
        try:
            return resolve_thinking(self._request_dict(request), self.think_mode)[0]
        except (TypeError, ValueError) as error:
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS, str(error)
            ) from error

    def _recipe_request(self, request):
        try:
            return convert_request(
                self._request_dict(request), default_thinking=self.think_mode
            )
        except (TypeError, ValueError) as error:
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS, str(error)
            ) from error

    def _encode_request(self, request):
        prompt, images = render_request(self._recipe_request(request))
        return prompt, {"images": images}

    def _build_prompt(self, request):
        return self._encode_request(request)[0]

    def render_chat(self, request):
        prompt, media = self._encode_request(request)
        try:
            input_ids = list(self.tokenizer.encode(prompt))
            urls = _image_records_to_urls(
                input_ids, media["images"], self.image_token_id
            )
        except (TypeError, ValueError) as error:
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS, str(error)
            ) from error
        return RenderedInputs(
            input_ids=input_ids,
            rendered_prompt=prompt,
            input_urls=urls,
            input_urls_type=[MMUrlType.IMAGE] * len(urls),
        )

    def _build_tool_call_structural_tag(self, request):
        try:
            return super()._build_tool_call_structural_tag(request)
        except ValueError as error:
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS, str(error)
            ) from error

    def apply_chat_completion_constraints(self, request, config):
        # Required/named choice specifies a generation prefix in recipe. It is
        # not an implicit strict grammar. Explicit RTP grammar fields survive.
        self._recipe_request(request)

    async def _create_status_list(self, n, request):
        options = self._recipe_request(request)
        return [V41StreamStatus(request, options) for _ in range(n)]

    def _split_reasoning_text_and_content(self, item, think_status):
        if isinstance(item.output_str, DeltaMessage):
            return item.output_str
        return super()._split_reasoning_text_and_content(item, think_status)

    async def _generate_stream_response(self, items, think_status_list):
        response = await super()._generate_stream_response(items, think_status_list)
        # Recipe's CompletionUsage has no reasoning-token breakdown. Retokenizing
        # parsed deltas cannot recover counts of the original generated IDs.
        response.usage.completion_tokens_details = None
        return response

    async def render_response_stream(self, output_generator, request, generate_config):
        try:
            async for response in super().render_response_stream(
                output_generator, request, generate_config
            ):
                yield response
        finally:
            # Parser-side stops must promptly release the backend request, as
            # must caller cancellation and errors from the token source.
            close = getattr(output_generator, "aclose", None)
            if close is not None:
                await close()

    @staticmethod
    def _protocol_message(status, events):
        content, reasoning, calls = [], [], {}
        for event in events:
            if event.kind == "content":
                content.append(event.text)
            elif event.kind == "reasoning":
                reasoning.append(event.text)
            elif event.kind == "tool":
                calls[event.index] = ToolCall(
                    index=event.index,
                    id=f"call_{status.call_prefix}_{event.index}",
                    type="function",
                    function=FunctionCall(name=event.text, arguments=""),
                )
            elif event.kind == "arguments":
                if event.index not in calls:
                    calls[event.index] = ToolCall(
                        index=event.index,
                        type="function",
                        function=FunctionCall(name=None, arguments=""),
                    )
                calls[event.index].function.arguments += event.text
        return DeltaMessage(
            content="".join(content) or None,
            reasoning_content="".join(reasoning) or None,
            tool_calls=list(calls.values()) or None,
        )

    async def _process_single_token_delta(
        self,
        status,
        delta_text,
        output,
        stop_words_str,
        stop_word_slice_list,
        is_streaming,
    ):
        text = status.delta_output_string + delta_text
        text, should_buffer = self._process_stop_words(
            text, stop_words_str, stop_word_slice_list, is_streaming, status
        )
        status.delta_output_string = text if should_buffer else ""
        if should_buffer:
            return None
        events = status.parser.feed(text)
        if status.parser.stopped:
            status.finish_reason = FinisheReason.stop
        if not events:
            return None
        return OutputDelta(
            output_str=self._protocol_message(status, events),
            logprobs=await self._generate_log_probs(status, output),
            input_length=output.aux_info.input_len,
            output_length=output.aux_info.output_len,
            reuse_length=output.aux_info.reuse_len,
        )

    async def _flush_buffer(
        self, buffer_list, stop_words_str, is_streaming, think_status_list
    ):
        items = []
        for status in buffer_list:
            events = status.parser.feed(status.delta_output_string)
            events.extend(status.parser.finish())
            status.delta_output_string = ""
            if status.parser.stopped:
                status.finish_reason = FinisheReason.stop
            aux = status.output.aux_info
            items.append(
                OutputDelta(
                    self._protocol_message(status, events),
                    None,
                    aux.input_len,
                    aux.output_len,
                    aux.reuse_len,
                )
            )
        return await self._generate_stream_response(items, think_status_list)

    async def _generate_final(self, buffer_list, request, think_status_list):
        for status in buffer_list:
            reason = status.finish_reason
            reason = reason.value if reason is not None else None
            status.finish_reason = FinisheReason(status.parser.finish_reason(reason))
        response = await super()._generate_final(
            buffer_list, request, think_status_list
        )
        response.usage.completion_tokens_details = None
        return response


register_renderer("deepseek_v41", DeepseekV41Renderer)
register_renderer("deepseek_v41_dspark", DeepseekV41Renderer)
