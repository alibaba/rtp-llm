"""K3/K3.1 HTTP contracts with real renderers and native grammar matching.

Only the tokenizer and model token source are fixtures. Checkpoint-backed
tokenization and vision execution remain in the checkpoint smoke targets.
"""

import copy
import json
import math
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch
import xgrammar as xgr
from fastapi.testclient import TestClient

from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyEnvConfigs,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.frontend.shutdown_manager import FrontendShutdownManager
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.kimi_k3_renderer import KimiK3Renderer
from rtp_llm.ops import SpecialTokens
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs
from rtp_llm.utils.concurrency_controller import ConcurrencyController

RESPONSE_CLOSE = "<|close|>response<|sep|>"
MESSAGE_CLOSE = "<|close|>message<|sep|>"
THINK_TO_RESPONSE = "<|close|>think<|sep|><|open|>response<|sep|>"
TOOLS_OPEN = "<|open|>tools<|sep|>"
TOOLS_CLOSE = "<|close|>tools<|sep|>"
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look up a query.",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
            "additionalProperties": False,
        },
    },
}
TOOL_DECLARATIONS = {
    "tools": {"tools": [TOOL]},
    "functions": {"functions": [TOOL["function"]]},
    "dynamic": {
        "messages": [
            {"role": "system", "content": "", "tools": [TOOL]},
            {"role": "user", "content": "question"},
        ]
    },
}
FORCED_TOOL_CHOICES = (
    "required",
    {"type": "function", "function": {"name": "lookup"}},
)
AUTO_TOOL_REPLY = (
    RESPONSE_CLOSE
    + TOOLS_OPEN
    + '<|open|>call tool="lookup" index="1"<|sep|>'
    + '<|open|>argument key="q" type="string"<|sep|>x<|close|>argument<|sep|>'
    + "<|close|>call<|sep|>"
    + TOOLS_CLOSE
    + MESSAGE_CLOSE
)
FORCED_TOOL_REPLY = (
    RESPONSE_CLOSE
    + TOOLS_OPEN
    + '<|open|>call tool="lookup" index="1"<|sep|>'
    + '<|open|>json type="object"<|sep|>{"q":"x"}<|close|>json<|sep|>'
    + "<|close|>call<|sep|>"
    + TOOLS_CLOSE
    + MESSAGE_CLOSE
)
FORMATS = (
    ("omitted", None, "ok"),
    ("text", {"type": "text"}, "ok"),
    ("json_object", {"type": "json_object"}, '{"value":1}'),
    (
        "json_schema",
        {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        },
        '{"value":1}',
    ),
    ("regex", {"type": "regex", "pattern": "ok"}, "ok"),
    ("ebnf", {"type": "ebnf", "grammar": 'root ::= "ok"'}, "ok"),
    (
        "structural_tag",
        {
            "type": "structural_tag",
            "structural_tag": {
                "type": "structural_tag",
                "format": {"type": "const_string", "value": "ok"},
            },
        },
        "ok",
    ),
)


class ProtocolTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.vocab = [chr(i) for i in range(128)] + [
            "<|open|>",
            "<|close|>",
            "<|sep|>",
            "<|end_of_msg|>",
            "think",
            "response",
            "message",
            "tools",
        ]
        self.ids = {text: index for index, text in enumerate(self.vocab)}
        self.pieces = sorted(self.ids, key=len, reverse=True)
        self.template_calls = []

    def __len__(self):
        return len(self.vocab)

    def encode(self, text, **kwargs):
        ids = []
        while text:
            piece = next(piece for piece in self.pieces if text.startswith(piece))
            ids.append(self.ids[piece])
            text = text[len(piece) :]
        return ids

    def decode(self, ids, **kwargs):
        return "".join(self.vocab[int(token)] for token in ids)

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token)

    def tokenize(self, text):
        return [self.vocab[index] for index in self.encode(text)]

    def apply_chat_template(self, messages, *, tools=None, thinking=False, **kwargs):
        self.template_calls.append(
            copy.deepcopy({"messages": messages, "tools": tools, "thinking": thinking})
        )
        channel = "think" if thinking else "response"
        return self.encode(f"question<|open|>{channel}<|sep|>")


class TokenSource:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []
        self.reply = "ok" + RESPONSE_CLOSE + MESSAGE_CLOSE
        self.chunk_size = 3

    async def enqueue(self, generate_input):
        self.calls.append(generate_input)
        config = generate_input.generate_config
        ids = self.tokenizer.encode(self.reply)[: config.max_new_tokens]
        content_start = 0
        if config.return_logprobs and config.in_think_mode:
            boundary_size = len(config.end_think_token_ids)
            assert boundary_size > 0
            content_start = next(
                (
                    i + boundary_size
                    for i in range(len(ids) - boundary_size + 1)
                    if ids[i : i + boundary_size] == config.end_think_token_ids
                ),
                len(ids),
            )

        async def generate():
            for start in range(0, len(ids), self.chunk_size):
                end = min(start + self.chunk_size, len(ids))
                scores = {}
                if config.return_logprobs:
                    score_start = max(start, min(end, content_start))
                    scored_ids = ids[score_start:end]
                    scores.update(
                        logprobs_offset=score_start - start,
                        logprobs_count=len(scored_ids),
                    )
                    if scored_ids:
                        scores.update(
                            token_logprobs=torch.full(
                                (len(scored_ids),), math.log(0.75)
                            ),
                            top_logprob_token_ids=torch.tensor(
                                [
                                    [token, 1 if token != 1 else 2]
                                    for token in scored_ids
                                ],
                                dtype=torch.int32,
                            )[:, : config.top_logprobs],
                            top_logprobs=torch.tensor(
                                [[math.log(0.75), math.log(0.25)]]
                            ).expand(len(scored_ids), -1)[:, : config.top_logprobs],
                        )
                yield GenerateOutputs(
                    [
                        GenerateOutput(
                            output_ids=torch.tensor([ids[start:end]], dtype=torch.int),
                            finished=torch.tensor([end == len(ids)]),
                            aux_info=AuxInfo(
                                input_len=generate_input.input_length,
                                output_len=end,
                                step_output_len=end - start,
                            ),
                            **scores,
                        )
                    ]
                )

        return generate()

    async def batch_enqueue(self, inputs):
        results = []
        for item in inputs:
            stream = await self.enqueue(item)
            chunks = [chunk async for chunk in stream]
            if len(chunks) != 1:
                raise AssertionError("batch source must return one chunk")
            results.append(chunks[0])
        return results


class KimiK3OpenaiEndpointTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ProtocolTokenizer()
        self.backend = TokenSource(self.tokenizer)
        self.compiler = xgr.GrammarCompiler(
            xgr.TokenizerInfo(self.tokenizer.vocab, stop_token_ids=[0]), max_threads=1
        )
        self.compiled = {}
        self.clients = []

    def tearDown(self):
        for client in self.clients:
            client.close()

    def client(self, model):
        config = SimpleNamespace(
            generate_env_config=GenerateEnvConfig(),
            max_seq_len=8192,
            model_name=model,
            model_type=model,
            special_tokens=SpecialTokens(),
            template_type=None,
            ckpt_path="",
            render_config=RenderConfig(),
        )
        endpoint = OpenaiEndpoint(
            config,
            PyMiscellaneousConfig(),
            VitConfig(),
            self.tokenizer,
            self.backend,
        )
        self.assertIsInstance(endpoint.chat_renderer, KimiK3Renderer)
        env = PyEnvConfigs()
        env.server_config.ip = "127.0.0.1"
        env.server_config.start_port = 0
        env.model_args.model_type = model
        with (
            patch("rtp_llm.frontend.frontend_server.AccessLogger"),
            patch("rtp_llm.frontend.frontend_server.kmonitor.init"),
            patch(
                "rtp_llm.frontend.frontend_server.get_global_controller",
                return_value=ConcurrencyController(max_concurrency=4),
            ),
        ):
            frontend = FrontendServer(0, 0, env)
        frontend._openai_endpoint = endpoint
        frontend._frontend_worker = SimpleNamespace(
            is_streaming=lambda request: bool(request.get("stream", False))
        )
        owner = FrontendApp.__new__(FrontendApp)
        owner.frontend_server = frontend
        owner.shutdown_manager = FrontendShutdownManager()
        owner.separated_frontend = True
        owner.server_config = SimpleNamespace(http_port=0)
        owner.grpc_client = None
        client = TestClient(owner.create_app())
        self.clients.append(client)
        return client, frontend

    @staticmethod
    def payload(**fields):
        return {"messages": [{"role": "user", "content": "question"}], **fields}

    def accepts(self, config, text):
        self.assertIsNone(config.response_format)
        self.assertIsNotNone(config.structural_tag)
        key = json.dumps(config.structural_tag, sort_keys=True)
        if key not in self.compiled:
            self.compiled[key] = self.compiler.compile_structural_tag(key)
        matcher = xgr.GrammarMatcher(
            self.compiled[key], terminate_without_stop_token=True
        )
        return matcher.accept_string(text) and matcher.is_terminated()

    def response(self, client, payload, path="/v1/chat/completions"):
        reply = client.post(path, json=payload)
        self.assertEqual(reply.status_code, 200, reply.text)
        if not payload.get("stream"):
            body = reply.json()
            self.assertNotIn("error_code", body, body)
            return body
        frames = [
            json.loads(line[6:])
            for line in reply.text.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
        self.assertTrue(frames, reply.text)
        choices = [choice for frame in frames for choice in frame.get("choices", [])]
        self.assertEqual(
            sum(choice.get("finish_reason") is not None for choice in choices), 1
        )
        message = {"content": "", "reasoning_content": "", "tool_calls": []}
        probabilities = []
        for choice in choices:
            delta = choice["delta"]
            for name in ("content", "reasoning_content"):
                message[name] += delta.get(name) or ""
            message["tool_calls"].extend(delta.get("tool_calls") or [])
            probabilities.extend((choice.get("logprobs") or {}).get("content") or [])
        return {
            "choices": [
                {
                    "message": message,
                    "finish_reason": choices[-1]["finish_reason"],
                    "logprobs": {"content": probabilities},
                }
            ],
            "frames": frames,
        }

    def test_response_formats_reach_the_backend_with_valid_grammar(self):
        for model in ("kimi_k3", "kimi_k31"):
            client, frontend = self.client(model)
            cases = list(product((False, True), (False, True), FORMATS[:1]))
            if model == "kimi_k3":
                # Both models share the renderer. Exercise each additional
                # format with and without reasoning on the streamed path.
                cases.extend(
                    (thinking, True, fmt)
                    for thinking, fmt in product((False, True), FORMATS[1:])
                )
            for thinking, stream, (name, response_format, answer) in cases:
                with self.subTest(
                    model=model, thinking=thinking, stream=stream, format=name
                ):
                    prefix = "r" + THINK_TO_RESPONSE if thinking else ""
                    self.backend.reply = (
                        prefix + answer + RESPONSE_CLOSE + MESSAGE_CLOSE
                    )
                    payload = self.payload(
                        enable_thinking=thinking, thinking_budget=8, stream=stream
                    )
                    if response_format is not None:
                        payload["response_format"] = response_format
                    response = self.response(client, payload)
                    choice = response["choices"][0]
                    message = choice["message"]
                    self.assertEqual(message.get("content"), answer)
                    self.assertEqual(
                        message.get("reasoning_content") or "", "r" if thinking else ""
                    )
                    self.assertEqual(choice["finish_reason"], "stop")
                    if thinking and not stream:
                        self.assertEqual(
                            response["usage"]["completion_tokens_details"][
                                "reasoning_tokens"
                            ],
                            len(self.tokenizer.tokenize("r")),
                        )
                    config = self.backend.calls[-1].generate_config
                    self.assertTrue(self.accepts(config, self.backend.reply))
                    self.assertFalse(self.accepts(config, prefix + AUTO_TOOL_REPLY))
                    self.assertEqual(config.in_think_mode, thinking)
                    self.assertEqual(config.max_thinking_tokens, 8 if thinking else 0)
                    self.assertEqual(
                        frontend._global_controller.get_available_concurrency(),
                        frontend._global_controller.max_concurrency,
                    )

    def test_forced_tools_override_response_formats_at_the_endpoint(self):
        for model in ("kimi_k3", "kimi_k31"):
            client, _ = self.client(model)
            cases = list(
                product(FORCED_TOOL_CHOICES, (False, True), (False, True), FORMATS[:1])
            )
            if model == "kimi_k3":
                # Once a tool is forced the renderer replaces the response
                # format with its tool grammar. Check precedence once per format.
                cases.extend(("required", True, True, fmt) for fmt in FORMATS[1:])
            for tool_choice, thinking, stream, (name, response_format, _) in cases:
                with self.subTest(
                    model=model,
                    tool_choice=tool_choice,
                    thinking=thinking,
                    stream=stream,
                    format=name,
                ):
                    prefix = "r" + THINK_TO_RESPONSE if thinking else ""
                    self.backend.reply = prefix + FORCED_TOOL_REPLY
                    payload = self.payload(
                        tools=[TOOL],
                        tool_choice=tool_choice,
                        enable_thinking=thinking,
                        stream=stream,
                    )
                    if response_format is not None:
                        payload["response_format"] = response_format
                    choice = self.response(client, payload)["choices"][0]
                    self.assertFalse(choice["message"].get("content"))
                    self.assertEqual(choice["finish_reason"], "tool_calls")
                    calls = [
                        item.get("function", {})
                        for item in choice["message"]["tool_calls"]
                    ]
                    self.assertEqual(
                        [item["name"] for item in calls if item.get("name")], ["lookup"]
                    )
                    arguments = "".join(item.get("arguments") or "" for item in calls)
                    self.assertEqual(json.loads(arguments), {"q": "x"})
                    config = self.backend.calls[-1].generate_config
                    self.assertTrue(self.accepts(config, self.backend.reply))
                    self.assertFalse(
                        self.accepts(
                            config, prefix + "plain" + RESPONSE_CLOSE + MESSAGE_CLOSE
                        )
                    )
                    self.assertFalse(
                        self.accepts(
                            config,
                            self.backend.reply.replace(
                                'tool="lookup"', 'tool="unknown"'
                            ),
                        )
                    )

    def test_batch_chat_render_and_single_share_configs_without_cross_request_state(
        self,
    ):
        self.backend.chunk_size = 100000
        for model in ("kimi_k3", "kimi_k31"):
            client, _ = self.client(model)
            for thinking in (False, True):
                with self.subTest(model=model, thinking=thinking):
                    self.backend.calls.clear()
                    prefix = "r" + THINK_TO_RESPONSE if thinking else ""
                    self.backend.reply = prefix + "ok" + RESPONSE_CLOSE + MESSAGE_CLOSE
                    payload = self.payload(
                        enable_thinking=thinking,
                        temperature=0.7,
                        top_p=0.95,
                        thinking_budget=9,
                        seed=42,
                        max_completion_tokens=128,
                    )
                    single_response = self.response(
                        client, payload, "/chat/completions"
                    )
                    self.assertEqual(
                        single_response["usage"]["prompt_tokens"],
                        len(self.tokenizer.encode("question")),
                    )
                    single = self.backend.calls[-1]
                    render = client.post("/v1/chat/render", json=payload)
                    self.assertEqual(render.status_code, 200, render.text)
                    batch = client.post(
                        "/v1/batch/chat/completions",
                        json={"requests": [payload, payload]},
                    )
                    self.assertEqual(batch.status_code, 200, batch.text)
                    self.assertEqual(len(batch.json()["responses"]), 2)
                    for response in batch.json()["responses"]:
                        self.assertEqual(response["usage"], single_response["usage"])
                    self.assertEqual(len(self.backend.calls), 3)
                    for item in self.backend.calls:
                        self.assertEqual(
                            item.token_ids.tolist()[0], render.json()["input_ids"]
                        )
                        for field in (
                            "temperature",
                            "top_p",
                            "max_new_tokens",
                            "max_thinking_tokens",
                            "in_think_mode",
                            "random_seed",
                            "structural_tag",
                        ):
                            self.assertEqual(
                                getattr(item.generate_config, field),
                                getattr(single.generate_config, field),
                                field,
                            )
                            self.assertEqual(
                                getattr(item.generate_config, field),
                                render.json()["generate_config"][field],
                                field,
                            )
                    self.assertIsNot(
                        self.backend.calls[1].generate_config,
                        self.backend.calls[2].generate_config,
                    )
                    with self.assertRaisesRegex(
                        ValueError, "does not support streaming"
                    ):
                        client.post(
                            "/v1/batch/chat/completions",
                            json={"requests": [{**payload, "stream": True}]},
                        )
                    self.assertEqual(len(self.backend.calls), 3)

    def test_extra_config_token_limit_preserves_top_level_precedence(self):
        for model in ("kimi_k3", "kimi_k31"):
            client, _ = self.client(model)
            for fields, expected in (
                ({}, 7),
                ({"max_tokens": 5}, 5),
                ({"max_completion_tokens": 3}, 3),
                ({"max_tokens": 5, "max_completion_tokens": 3}, 3),
            ):
                with self.subTest(model=model, fields=fields):
                    response = client.post(
                        "/v1/chat/render",
                        json=self.payload(
                            enable_thinking=False,
                            extra_configs={"max_new_tokens": 7},
                            **fields,
                        ),
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(
                        response.json()["generate_config"]["max_new_tokens"], expected
                    )

    def test_auto_none_and_declared_tool_sources_use_matching_grammar(self):
        plain_reply = "ok" + RESPONSE_CLOSE + MESSAGE_CLOSE
        for model in ("kimi_k3", "kimi_k31"):
            client, _ = self.client(model)
            for source, choice, stream in product(
                TOOL_DECLARATIONS, (None, "auto", "none"), (False, True)
            ):
                with self.subTest(
                    model=model, source=source, choice=choice, stream=stream
                ):
                    self.backend.reply = (
                        plain_reply if choice == "none" else AUTO_TOOL_REPLY
                    )
                    payload = self.payload(
                        **TOOL_DECLARATIONS[source],
                        enable_thinking=False,
                        stream=stream,
                    )
                    if choice is not None:
                        payload["tool_choice"] = choice
                    response_choice = self.response(client, payload)["choices"][0]
                    config = self.backend.calls[-1].generate_config
                    self.assertTrue(self.accepts(config, self.backend.reply))
                    if choice == "none":
                        self.assertFalse(response_choice["message"].get("tool_calls"))
                        self.assertFalse(self.accepts(config, AUTO_TOOL_REPLY))
                    else:
                        self.assertEqual(response_choice["finish_reason"], "tool_calls")
                        self.assertTrue(response_choice["message"].get("tool_calls"))
                        self.assertTrue(self.accepts(config, plain_reply))

    def test_legacy_and_dynamic_tools_can_be_selected_explicitly(self):
        self.backend.reply = FORCED_TOOL_REPLY
        for model in ("kimi_k3", "kimi_k31"):
            client, _ = self.client(model)
            for source, choice in product(
                ("functions", "dynamic"), FORCED_TOOL_CHOICES
            ):
                with self.subTest(model=model, source=source, choice=choice):
                    body = self.response(
                        client,
                        self.payload(
                            **TOOL_DECLARATIONS[source],
                            tool_choice=choice,
                            enable_thinking=False,
                        ),
                    )
                    self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
                    self.assertTrue(
                        self.accepts(
                            self.backend.calls[-1].generate_config, self.backend.reply
                        )
                    )

    def test_partial_text_and_custom_template_take_their_own_render_paths(self):
        for model in ("kimi_k3", "kimi_k31"):
            client, _ = self.client(model)
            self.backend.reply = "ok" + RESPONSE_CLOSE + MESSAGE_CLOSE
            body = self.response(
                client,
                self.payload(
                    enable_thinking=False,
                    messages=[
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "prefix", "partial": True},
                    ],
                ),
            )
            self.assertEqual(body["choices"][0]["message"]["content"], "ok")
            self.assertTrue(
                self.tokenizer.decode(self.backend.calls[-1].token_ids[0]).endswith(
                    "<|open|>response<|sep|>prefix"
                )
            )
            for response_format, answer, thinking in (
                ({"type": "text"}, "ok", False),
                ({"type": "json_object"}, "{}", False),
                ({"type": "text"}, "ok", True),
            ):
                with self.subTest(
                    model=model,
                    empty_partial=response_format["type"],
                    thinking=thinking,
                ):
                    self.backend.reply = (
                        ("r" + THINK_TO_RESPONSE if thinking else "")
                        + answer
                        + RESPONSE_CLOSE
                        + MESSAGE_CLOSE
                    )
                    body = self.response(
                        client,
                        self.payload(
                            enable_thinking=thinking,
                            response_format=response_format,
                            messages=[
                                {"role": "user", "content": "question"},
                                {"role": "assistant", "content": "", "partial": True},
                            ],
                        ),
                    )
                    self.assertEqual(body["choices"][0]["message"]["content"], answer)
                    self.assertTrue(
                        self.accepts(
                            self.backend.calls[-1].generate_config, self.backend.reply
                        )
                    )
            self.backend.reply = "ok"
            body = self.response(
                client,
                self.payload(
                    enable_thinking=False, user_template="{{ messages[0].content }}"
                ),
            )
            self.assertEqual(body["choices"][0]["message"]["content"], "ok")
            config = self.backend.calls[-1].generate_config
            self.assertIsNone(config.structural_tag)
            self.assertEqual(
                self.tokenizer.decode(self.backend.calls[-1].token_ids[0]), "question"
            )

    def test_nonempty_partial_with_effective_thinking_is_rejected(self):
        for model in ("kimi_k3", "kimi_k31"):
            client, _ = self.client(model)
            for thinking_fields in (
                {},
                {"enable_thinking": True},
                {"enable_thinking": False, "thinking": {"type": "enabled"}},
            ):
                with self.subTest(model=model, thinking_fields=thinking_fields):
                    count = len(self.backend.calls)
                    reply = client.post(
                        "/v1/chat/completions",
                        json=self.payload(
                            messages=[
                                {"role": "user", "content": "question"},
                                {
                                    "role": "assistant",
                                    "content": "prefix",
                                    "partial": True,
                                },
                            ],
                            **thinking_fields,
                        ),
                    )
                    self.assertEqual(reply.json().get("error_code"), 605, reply.text)
                    self.assertEqual(reply.json().get("message"), "INVALID_PARAMS")
                    self.assertEqual(len(self.backend.calls), count)

    def test_partial_structured_response_is_rejected_instead_of_restarting_grammar(
        self,
    ):
        for model in ("kimi_k3", "kimi_k31"):
            client, _ = self.client(model)
            for name, response_format, _ in FORMATS:
                if name in ("omitted", "text"):
                    continue
                with self.subTest(model=model, format=name):
                    count = len(self.backend.calls)
                    reply = client.post(
                        "/v1/chat/completions",
                        json=self.payload(
                            enable_thinking=False,
                            response_format=response_format,
                            messages=[
                                {"role": "user", "content": "question"},
                                {
                                    "role": "assistant",
                                    "content": '{"value":',
                                    "partial": True,
                                },
                            ],
                        ),
                    )
                    self.assertEqual(reply.json().get("error_code"), 605, reply.text)
                    self.assertEqual(reply.json().get("message"), "INVALID_PARAMS")
                    self.assertEqual(len(self.backend.calls), count)

    def test_invalid_partial_messages_cannot_change_protocol_state(self):
        partial = {"role": "assistant", "content": "prefix", "partial": True}
        for model in ("kimi_k3", "kimi_k31"):
            client, _ = self.client(model)
            for fields in (
                {"messages": [{**partial, "role": "user"}]},
                {"messages": [partial, {"role": "user", "content": "later"}]},
                {
                    "messages": [
                        {**partial, "content": [{"type": "text", "text": "prefix"}]}
                    ]
                },
                {"messages": [{**partial, "content": "<|open|>tools<|sep|>"}]},
                {"messages": [{**partial, "content": "prefix<|media_pad|>"}]},
                {"messages": [{**partial, "tools": [TOOL]}]},
                {"messages": [{**partial, "reasoning_content": "thinking prefix"}]},
                {
                    "messages": [
                        {
                            **partial,
                            "function_call": {"name": "lookup", "arguments": "{}"},
                        }
                    ]
                },
                {
                    "messages": [
                        {
                            **partial,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        }
                    ]
                },
                {"messages": [partial], "json_format": True},
                {"messages": [partial], "tools": [TOOL], "tool_choice": "required"},
            ):
                with self.subTest(model=model, fields=fields):
                    count = len(self.backend.calls)
                    reply = client.post(
                        "/v1/chat/completions",
                        json=self.payload(enable_thinking=False, **fields),
                    )
                    self.assertEqual(reply.json().get("error_code"), 605, reply.text)
                    self.assertEqual(reply.json().get("message"), "INVALID_PARAMS")
                    self.assertEqual(len(self.backend.calls), count)

    def test_custom_template_structured_partial_and_unsupported_tools_are_rejected(
        self,
    ):
        for model in ("kimi_k3", "kimi_k31"):
            client, _ = self.client(model)
            for fields in (
                {
                    "messages": [
                        {"role": "assistant", "content": '{"value":', "partial": True}
                    ],
                    "response_format": {"type": "json_object"},
                },
                {
                    "messages": [
                        {"role": "assistant", "content": "prefix", "partial": True}
                    ],
                    "json_format": True,
                },
                {"functions": [TOOL["function"]], "tool_choice": "required"},
                {
                    "messages": [{"role": "system", "content": "", "tools": [TOOL]}],
                    "tool_choice": "required",
                },
            ):
                with self.subTest(model=model, fields=fields):
                    count = len(self.backend.calls)
                    reply = client.post(
                        "/v1/chat/completions",
                        json=self.payload(
                            enable_thinking=False,
                            user_template="{{ messages[0].content }}",
                            **fields,
                        ),
                    )
                    self.assertEqual(reply.json().get("error_code"), 605, reply.text)
                    self.assertEqual(len(self.backend.calls), count)

    def test_logprobs_stream_usage_and_length_finish_reach_http(self):
        for model in ("kimi_k3", "kimi_k31"):
            client, frontend = self.client(model)
            for thinking in (False, True):
                for stream in (False, True):
                    with self.subTest(model=model, thinking=thinking, stream=stream):
                        self.backend.reply = (
                            ("r" + THINK_TO_RESPONSE if thinking else "")
                            + "ok"
                            + RESPONSE_CLOSE
                            + MESSAGE_CLOSE
                        )
                        payload = self.payload(
                            enable_thinking=thinking,
                            stream=stream,
                            logprobs=True,
                            top_logprobs=2,
                        )
                        if stream:
                            payload["stream_options"] = {"include_usage": True}
                        body = self.response(client, payload)
                        choice = body["choices"][0]
                        self.assertEqual(choice["message"]["content"], "ok")
                        self.assertEqual(
                            choice["message"].get("reasoning_content") or "",
                            "r" if thinking else "",
                        )
                        rows = choice["logprobs"]["content"]
                        self.assertEqual("".join(row["token"] for row in rows), "ok")
                        for row in rows:
                            self.assertAlmostEqual(
                                row["logprob"], math.log(0.75), places=6
                            )
                            self.assertEqual(len(row["top_logprobs"]), 2)
                        config = self.backend.calls[-1].generate_config
                        self.assertTrue(config.return_logprobs)
                        self.assertEqual(config.top_logprobs, 2)
                        self.assertEqual(config.return_all_probs, 0)
                        if stream:
                            frames = body["frames"]
                            for frame in frames:
                                for stream_choice in frame["choices"]:
                                    if "reasoning_content" in stream_choice["delta"]:
                                        self.assertFalse(stream_choice.get("logprobs"))
                            self.assertEqual(frames[-1]["choices"], [])
                            self.assertIn("usage", frames[-1])
                        self.assertEqual(
                            frontend._global_controller.get_available_concurrency(),
                            frontend._global_controller.max_concurrency,
                        )
            self.backend.reply = "abcdef" + RESPONSE_CLOSE + MESSAGE_CLOSE
            for stream in (False, True):
                with self.subTest(model=model, stream=stream, finish="length"):
                    body = self.response(
                        client,
                        self.payload(
                            enable_thinking=False,
                            stream=stream,
                            logprobs=True,
                            max_completion_tokens=3,
                        ),
                    )
                    self.assertEqual(body["choices"][0]["finish_reason"], "length")
                    self.assertEqual(body["choices"][0]["message"]["content"], "abc")
                    self.assertEqual(
                        "".join(
                            row["token"]
                            for row in body["choices"][0]["logprobs"]["content"]
                        ),
                        "abc",
                    )

    def test_invalid_requests_are_rejected_before_enqueue(self):
        for model in ("kimi_k3", "kimi_k31"):
            client, frontend = self.client(model)
            for fields in (
                {"temperature": 1.1},
                {"top_p": 0.5},
                {"presence_penalty": 1},
                {"frequency_penalty": 1},
                {"max_tokens": 0},
                {"max_completion_tokens": -1},
                {"n": 2},
                {"top_logprobs": -1},
                {"response_format": {"type": "unknown"}},
                {"response_format": {"type": "json_schema"}},
                {"tool_choice": "required"},
                {
                    "tools": [TOOL],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "unknown"},
                    },
                },
                {"extra_configs": {"structural_tag": {}}},
                {
                    "messages": [
                        {"role": "tool", "tool_call_id": "missing", "content": "x"}
                    ]
                },
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video_url",
                                    "video_url": {"url": "data:video/mp4,AA"},
                                }
                            ],
                        }
                    ]
                },
            ):
                with self.subTest(model=model, fields=fields):
                    count = len(self.backend.calls)
                    reply = client.post(
                        "/v1/chat/completions",
                        json=self.payload(enable_thinking=False, **fields),
                    )
                    self.assertTrue(
                        reply.status_code >= 400 or reply.json().get("error_code"),
                        reply.text,
                    )
                    self.assertEqual(len(self.backend.calls), count)
                    self.assertEqual(
                        frontend._global_controller.get_available_concurrency(),
                        frontend._global_controller.max_concurrency,
                    )


if __name__ == "__main__":
    unittest.main()
