"""Regression coverage for UTF-8 tool-call arguments across MTP steps."""

import json
from typing import Optional
from unittest import IsolatedAsyncioTestCase, main

import torch

from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    GPTFunctionDefinition,
    GPTToolDefinition,
    RoleEnum,
)
from rtp_llm.openai.renderers.custom_renderer import RendererParams
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv32_detector import (
    DeepSeekV32Detector,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF


class SpaceMergeByteTokenizer:
    """Minimal byte tokenizer modeling Qwen-style space-merged CJK tokens."""

    def __init__(self):
        self._next_id = 1
        self._id_to_bytes = {}
        self._bytes_to_id = {}

    def _get_id(self, raw: bytes) -> int:
        if raw not in self._bytes_to_id:
            self._bytes_to_id[raw] = self._next_id
            self._id_to_bytes[self._next_id] = raw
            self._next_id += 1
        return self._bytes_to_id[raw]

    def encode(self, text: str):
        tokens = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\uff5c":
                tokens.append(self._get_id(ch.encode("utf-8")))
                i += 1
            elif ord(ch) < 128:
                if ch == " " and i + 1 < len(text) and _is_cjk(text[i + 1]):
                    nxt = text[i + 1].encode("utf-8")
                    tokens.append(self._get_id(b" " + nxt[:1]))
                    tokens.extend(self._get_id(bytes([byte])) for byte in nxt[1:])
                    i += 2
                else:
                    tokens.append(self._get_id(ch.encode("utf-8")))
                    i += 1
            elif _is_cjk(ch):
                tokens.extend(
                    self._get_id(bytes([byte])) for byte in ch.encode("utf-8")
                )
                i += 1
            else:
                tokens.append(self._get_id(ch.encode("utf-8")))
                i += 1
        return tokens

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        raw = b"".join(self._id_to_bytes.get(token_id, b"") for token_id in token_ids)
        return raw.decode("utf-8", errors="replace")


DSML = "\uff5cDSML\uff5c"


def _tool_call_text(todos) -> str:
    return (
        f"<{DSML}function_calls>\n"
        f'<{DSML}invoke name="todo_write">\n'
        f'<{DSML}parameter name="todos" string="true">'
        + json.dumps(todos, ensure_ascii=False)
        + f"</{DSML}parameter>\n"
        f"</{DSML}invoke>\n"
        f"</{DSML}function_calls>"
    )


class MtpToolCallSegmentationTest(IsolatedAsyncioTestCase):
    """Drive the production renderer across split UTF-8 tool arguments."""

    def setUp(self):
        class TestRenderer(ReasoningToolBaseRenderer):
            def _setup_chat_template(self):
                self.chat_template = "test"

            def in_think_mode(self, request: ChatCompletionRequest) -> bool:
                return False

            def _create_detector(
                self, request: ChatCompletionRequest
            ) -> Optional[BaseFormatDetector]:
                return DeepSeekV32Detector() if request.tools else None

        self.tokenizer = SpaceMergeByteTokenizer()
        self.renderer = TestRenderer(
            tokenizer=self.tokenizer,
            renderer_params=RendererParams(
                model_type="deepseek_v32",
                max_seq_len=2048,
                eos_token_id=0,
                stop_word_ids_list=[],
            ),
            generate_env_config=GenerateEnvConfig(),
        )
        self.request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="test")],
            tools=[
                GPTToolDefinition(
                    type="function",
                    function=GPTFunctionDefinition(
                        name="todo_write",
                        description="write todos",
                        parameters={
                            "type": "object",
                            "properties": {"todos": {"type": "string"}},
                            "required": ["todos"],
                        },
                    ),
                )
            ],
        )
        self.todos = [
            {"content": "检查环境 可乐 薯片 饼干", "status": "进行中"},
            {"content": "启动服务", "status": "待处理"},
        ]
        self.token_ids = self.tokenizer.encode(_tool_call_text(self.todos))

    async def _stream_arguments(self, chunk_size: int):
        (status,) = await self.renderer._create_status_list(1, self.request)
        names = {}
        arguments = {}

        for offset in range(0, len(self.token_ids), chunk_size):
            chunk = self.token_ids[offset : offset + chunk_size]
            aux_info = AuxInfo()
            aux_info.input_len = 0
            aux_info.output_len = len(status.output_ids_list) + len(chunk)
            aux_info.reuse_len = 0

            output = GenerateOutput()
            output.output_ids = torch.tensor([chunk], dtype=torch.int64)
            output.aux_info = aux_info
            delta = await self.renderer._update_single_status(
                status,
                output,
                max_new_tokens=len(self.token_ids) + 1,
                stop_words_str=[],
                stop_word_slice_list=[],
                is_streaming=True,
            )

            message = delta.output_str
            for tool_call in getattr(message, "tool_calls", None) or []:
                index = tool_call.index
                function = tool_call.function
                if function.name:
                    names[index] = function.name
                if function.arguments:
                    arguments[index] = arguments.get(index, "") + function.arguments

        return names, arguments

    async def test_tool_arguments_survive_single_and_multi_token_steps(self):
        for chunk_size in (1, 2, 3, 4, 5, 6, 8, 16):
            with self.subTest(chunk_size=chunk_size):
                names, arguments = await self._stream_arguments(chunk_size)
                self.assertEqual(names, {0: "todo_write"})
                self.assertEqual(
                    json.loads(json.loads(arguments[0])["todos"]),
                    self.todos,
                    f"chunk_size={chunk_size} corrupted arguments: {arguments!r}",
                )


if __name__ == "__main__":
    main()
