"""V4.1 checkpoint encoding and exact numeric effort handling."""

import hashlib
import importlib.util
from pathlib import Path

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.models.multimodal.deepseek_v41_processor import (
    V41ImageProcessorConfig,
    prepare_vl_inputs,
)
from rtp_llm.openai.reasoning_effort import normalize_v41_reasoning_effort
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import RenderedInputs
from rtp_llm.openai.renderers.deepseekv4_renderer import DeepseekV4Renderer
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv41_detector import (
    DeepSeekV41Detector,
)


class DeepseekV41Renderer(DeepseekV4Renderer):
    detector_class = DeepSeekV41Detector
    dsml_tool_calls_marker = "<\uff5cDSML\uff5c calls>"

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

    def _prepare_encoding_inputs(self, request):
        request = request.model_copy(deep=True)
        for message in request.messages:
            if isinstance(message.content, list):
                parts = [
                    (
                        part.model_dump(mode="json", exclude_none=True)
                        if hasattr(part, "model_dump")
                        else part
                    )
                    for part in message.content
                ]
                for part in parts:
                    if part.get("type") not in ("text", "image_url"):
                        raise FtRuntimeException(
                            ExceptionType.INVALID_PARAMS,
                            "V4.1 supports text and image content parts only",
                        )
                    if any(
                        value is not None
                        for value in (part.get("preprocess_config") or {}).values()
                    ):
                        raise FtRuntimeException(
                            ExceptionType.INVALID_PARAMS,
                            "V4.1 image preprocessing is fixed by the model config",
                        )
                message.content = parts
        return super()._prepare_encoding_inputs(request)

    def _encode_request(self, request):
        messages, config = self._prepare_encoding_inputs(request)
        try:
            return self.encoding_module.encode_messages(
                messages, **config, return_multi_modal_data=True
            )
        except (TypeError, ValueError) as error:
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS, str(error)
            ) from error

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


register_renderer("deepseek_v41", DeepseekV41Renderer)
