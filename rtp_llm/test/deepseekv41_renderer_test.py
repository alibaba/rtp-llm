import json
import os
import unittest
from pathlib import Path

from pydantic import ValidationError
from transformers import PreTrainedTokenizerFast

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.dash_sc.codec import DashScParameterError, parse_other_params
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.reasoning_effort import (
    normalize_v41_reasoning_effort,
    validate_reasoning_effort_for_model,
)
from rtp_llm.openai.renderers.deepseekv4_renderer import DeepseekV4Renderer
from rtp_llm.openai.renderers.deepseekv41_renderer import DeepseekV41Renderer
from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv41_detector import (
    DeepSeekV41Detector,
)


class V41RequestTest(unittest.TestCase):
    def test_strict_json_types_and_legacy_validation(self):
        for stream in (True, False):
            for value in (None, 1, 25, 50, 75, 100, "low", "high", "xhigh", "max"):
                request = ChatCompletionRequest.model_validate_json(
                    json.dumps(
                        {"messages": [], "reasoning_effort": value, "stream": stream}
                    )
                )
                self.assertIs(type(request.reasoning_effort), type(value))
                validate_reasoning_effort_for_model(
                    request.reasoning_effort, "deepseek_v41"
                )
            for value in (True, False, 1.0, {}, []):
                with self.subTest(value=value), self.assertRaises(ValidationError):
                    ChatCompletionRequest.model_validate_json(
                        json.dumps({"messages": [], "reasoning_effort": value})
                    )
            for value in ("1", "50", "medium", "none", 0, 101):
                with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "reasoning_effort"
                ):
                    normalize_v41_reasoning_effort(value)
        with self.assertRaises(ValueError):
            validate_reasoning_effort_for_model(50, "deepseek_v4")
        legacy = DeepseekV4Renderer.__new__(DeepseekV4Renderer)
        self.assertIsNone(legacy._normalize_reasoning_effort("low"))
        self.assertEqual(legacy._normalize_reasoning_effort("xhigh"), "max")

    def test_dash_wire_preserves_integer_and_named_max(self):
        request = predict_v2_pb2.ModelInferRequest()
        request.parameters["reasoning_effort"].int64_param = 37
        self.assertEqual(
            parse_other_params(request, model_type="deepseek_v41").reasoning_effort, 37
        )
        request.parameters["reasoning_effort"].string_param = "max"
        self.assertEqual(
            parse_other_params(request, model_type="deepseek_v41").reasoning_effort, 100
        )
        self.assertEqual(parse_other_params(request).reasoning_effort, "xhigh")
        for value in ("37", "medium", "HIGH"):
            request.parameters["reasoning_effort"].string_param = value
            with self.assertRaises(DashScParameterError):
                parse_other_params(request, model_type="deepseek_v41")

    def test_dsml_names_and_every_stream_split(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="lookup",
                    parameters={
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                ),
            )
        ]
        text = '<\uff5cDSML\uff5c calls><\uff5cDSML\uff5c invoke name="lookup"><\uff5cDSML\uff5c parameter name="key" string="true">value</\uff5cDSML\uff5c parameter></\uff5cDSML\uff5c invoke></\uff5cDSML\uff5c calls>'
        parsed = DeepSeekV41Detector().detect_and_parse(text, tools)
        self.assertEqual(len(parsed.calls), 1)
        self.assertEqual(json.loads(parsed.calls[0].parameters), {"key": "value"})
        for split in range(1, len(text)):
            detector = DeepSeekV41Detector()
            first = detector.parse_streaming_increment(text[:split], tools)
            second = detector.parse_streaming_increment(text[split:], tools)
            emitted = first.calls + second.calls
            arguments = "".join(call.parameters for call in emitted)
            with self.subTest(split=split):
                self.assertEqual(json.loads(arguments), {"key": "value"})


class V41OfficialEncodingTest(unittest.TestCase):
    def test_json_to_checkpoint_encoder(self):
        checkpoint = Path(os.environ["DSV41_REFERENCE_CHECKPOINT"])
        fast = PreTrainedTokenizerFast(
            tokenizer_file=str(checkpoint / "tokenizer.json")
        )

        class Tokenizer:
            def encode(self, text):
                return fast.encode(text, add_special_tokens=False)

        renderer = DeepseekV41Renderer.__new__(DeepseekV41Renderer)
        renderer.encoding_module = renderer._load_encoding_module(str(checkpoint))
        renderer.tokenizer = Tokenizer()
        renderer.think_mode = False
        messages = [
            {"role": "system", "content": "First instruction"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "Second instruction"},
        ]
        for thinking, effort in (
            (False, None),
            (True, None),
            (True, "low"),
            (True, 1),
            (True, 100),
        ):
            with self.subTest(thinking=thinking, effort=effort):
                request = ChatCompletionRequest.model_validate_json(
                    json.dumps(
                        {
                            "messages": messages,
                            "reasoning_effort": effort,
                            "enable_thinking": thinking,
                        }
                    )
                )
                actual = renderer.render_chat(request)
                expected = renderer.encoding_module.encode_messages(
                    messages,
                    thinking_mode="thinking" if thinking else "chat",
                    reasoning_effort=normalize_v41_reasoning_effort(effort),
                    drop_thinking=True,
                    add_default_bos_token=True,
                )
                self.assertEqual(actual.rendered_prompt, expected)
                self.assertEqual(
                    actual.input_ids, fast.encode(expected, add_special_tokens=False)
                )


if __name__ == "__main__":
    unittest.main()
