"""V4.1 recipe protocol with canonical checkpoint image metadata."""

import hashlib
import importlib.util
import uuid
from pathlib import Path

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.models.multimodal.deepseek_v41_processor import (
    V41ImageProcessorConfig,
    prepare_vl_inputs,
)
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

    def _load_encoding_module(self, ckpt_path: str):
        path = Path(ckpt_path) / "encoding" / "encoding.py"
        if not path.is_file():
            raise FileNotFoundError(f"V4.1 encoding module is missing: {path}")
        identity = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = importlib.util.spec_from_file_location(
            f"rtp_dsv41_encoding_{identity}", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load V4.1 encoding module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in ("encode_messages", "parse_message_from_completion_text"):
            if not callable(getattr(module, name, None)):
                raise ImportError(f"V4.1 encoding module is missing {name}")
        self.encoding_sha256 = identity
        self.image_processor_config = V41ImageProcessorConfig.from_model_config(
            V41Config.from_path(ckpt_path)
        )
        return module

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

    def prepare_v41_inputs(self, request, *, url_loader=None, output_budget=None):
        """Prepare request-owned canonical IDs, masks, images and content hashes.

        The backend must transport this metadata through its typed V4.1 input
        contract. It must not discard it via the generic RenderedInputs path.
        """
        prompt, media = self._encode_request(request)
        if output_budget is None:
            output_budget = request.max_completion_tokens
            if output_budget is None:
                output_budget = request.max_tokens or 0
        try:
            return prepare_vl_inputs(
                prompt,
                media["images"],
                self.tokenizer,
                self.image_processor_config,
                url_loader=url_loader,
                output_budget=output_budget,
            )
        except (TypeError, ValueError) as error:
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS, str(error)
            ) from error

    def render_chat(self, request):
        prepared = self.prepare_v41_inputs(request).validate()
        return RenderedInputs(
            input_ids=list(prepared.token_ids),
            rendered_prompt=prepared.prompt,
            v41_inputs=prepared,
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
