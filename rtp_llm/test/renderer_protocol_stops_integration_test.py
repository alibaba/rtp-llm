"""Real endpoint/renderers with the checked-in Qwen tokenizer and scripted tokens.

Only model execution/RPC is substituted. The source deliberately keeps producing
after a protocol boundary, so renderer cancellation and engine stop configuration
are checked independently without requiring model weights or a serving process.
"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.config.model_config import ModelConfig, update_tokenizer_special_tokens
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.qwen_tokenizer import QWenTokenizer
from rtp_llm.openai.api_datatype import ChatCompletionRequest, FinisheReason
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.qwen_renderer import QwenRenderer
from rtp_llm.ops import SpecialTokens
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


class ScriptedBackend:
    def __init__(self, chunks):
        self.chunks = chunks
        self.inputs = []
        self.yielded_chunks = 0
        self.closed = False

    async def enqueue(self, inputs):
        self.inputs.append(inputs)

        async def source():
            output_tokens = 0
            try:
                for index, tokens in enumerate(self.chunks):
                    output_tokens += len(tokens)
                    self.yielded_chunks += 1
                    yield GenerateOutputs(
                        generate_outputs=[
                            GenerateOutput(
                                output_ids=torch.tensor(tokens, dtype=torch.int32),
                                finished=index + 1 == len(self.chunks),
                                aux_info=AuxInfo(
                                    input_len=inputs.token_ids.shape[-1],
                                    output_len=output_tokens,
                                ),
                            )
                        ]
                    )
            finally:
                self.closed = True

        return source()


class RendererProtocolStopsIntegrationTest(unittest.IsolatedAsyncioTestCase):
    def make_endpoint(self, backend, *, force, env_stop=None):
        vocab = (
            Path(__file__).parent
            / "model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )
        # These are existing Qwen vocabulary tokens exposed through normal HF
        # tokenizer metadata, not stops manually injected into a renderer.
        tokenizer = QWenTokenizer(str(vocab.parent))
        tokenizer.tokenizer.bos_token = "<|extra_1|>"
        tokenizer.tokenizer.additional_special_tokens = ["<|extra_0|>"]
        model = ModelConfig()
        model.model_type = "qwen"
        model.model_name = "qwen"
        model.max_seq_len = 8192
        model.ckpt_path = ""
        model.render_config = RenderConfig()
        model.generate_env_config = GenerateEnvConfig()
        # This is the object populated by --force_stop_words/FORCE_STOP_WORDS.
        model.generate_env_config.force_stop_words = force
        model.generate_env_config.stop_words_str = json.dumps(env_stop or [])
        model.generate_env_config.stop_words_list = "[]"
        model.special_tokens = SpecialTokens()
        # Qwen's normal model stops plus a configurable model default. The
        # BasicRenderer-only tokenizer extras above are intentionally separate.
        model.special_tokens.stop_words_id_list = [tokenizer.encode("MODEL_STOP")]
        update_tokenizer_special_tokens(model.special_tokens, tokenizer)
        endpoint = OpenaiEndpoint(
            model,
            PyMiscellaneousConfig(),
            VitConfig(),
            tokenizer,
            backend,
        )
        self.assertIsInstance(endpoint.chat_renderer, QwenRenderer)
        self.assertIsInstance(endpoint.template_renderer, BasicRenderer)
        self.assertIsNot(endpoint.chat_renderer, endpoint.template_renderer)
        return endpoint

    async def complete(self, endpoint, request):
        response = await endpoint.chat_completion_async(
            1, request, SimpleNamespace(headers={})
        )
        frames = [frame async for frame in response]
        complete = await response.gen_complete_response_once()
        return frames, complete

    async def test_force_stop_words_preserves_qwen_function_protocol(self):
        for env_stops in ([], ["DEPLOY_END"]):
            for split_observation in (False, True):
                with self.subTest(env_stops=env_stops, split=split_observation):
                    backend = ScriptedBackend([])
                    endpoint = self.make_endpoint(
                        backend, force=True, env_stop=env_stops
                    )
                    tokenizer = endpoint.tokenizer
                    observation = [37763, 367, 25]
                    self.assertEqual(tokenizer.decode(observation), "Observation:")
                    stop_chunks = (
                        [[token] for token in observation]
                        if split_observation
                        else [observation]
                    )
                    backend.chunks = [
                        tokenizer.encode("Thought: search\nAction:"),
                        tokenizer.encode(' lookup\nAction Input: {"city":"杭州"}\n'),
                        *stop_chunks,
                        tokenizer.encode(
                            " fabricated result\nThought: retry\nAction: delete_account"
                            '\nAction Input: {"id":1}'
                        ),
                        [tokenizer.eos_token_id],
                    ]
                    request = ChatCompletionRequest.model_validate(
                        {
                            "messages": [{"role": "user", "content": "lookup 杭州"}],
                            "functions": [
                                {
                                    "name": "lookup",
                                    "description": "Look up a city",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"city": {"type": "string"}},
                                        "required": ["city"],
                                    },
                                }
                            ],
                            "stream": True,
                            "max_tokens": 512,
                        }
                    )
                    self.assertFalse(endpoint.chat_renderer.in_think_mode(request))
                    self.assertFalse(request.tools)
                    rendered = endpoint.render_chat(request)
                    self.assertIn(
                        "Observation: the result of the action",
                        rendered.rendered_prompt,
                    )
                    _, result = await self.complete(endpoint, request)
                    choice = result.choices[0]
                    self.assertEqual(choice.finish_reason, FinisheReason.function_call)
                    self.assertIsNotNone(choice.message.function_call)
                    self.assertEqual(choice.message.function_call.name, "lookup")
                    self.assertEqual(
                        json.loads(choice.message.function_call.arguments),
                        {"city": "杭州"},
                    )
                    self.assertEqual(backend.yielded_chunks, 2 + len(stop_chunks))
                    self.assertTrue(backend.closed)
                    self.assertIn(
                        observation, backend.inputs[0].generate_config.stop_words_list
                    )

    async def test_user_template_preserves_active_basic_renderer_extras(self):
        # Exercise both BasicRenderer registration sources without FORCE_STOP_WORDS.
        for marker in ("<|extra_0|>", "<|extra_1|>"):
            with self.subTest(marker=marker):
                backend = ScriptedBackend([])
                endpoint = self.make_endpoint(backend, force=False)
                tokenizer = endpoint.tokenizer
                stop_ids = tokenizer.encode(marker)
                self.assertIn(
                    stop_ids,
                    endpoint.template_renderer.get_all_extra_stop_word_ids_list(),
                )
                self.assertNotIn(
                    stop_ids, endpoint.chat_renderer.get_all_extra_stop_word_ids_list()
                )
                backend.chunks = [
                    tokenizer.encode("visible"),
                    stop_ids,
                    tokenizer.encode("hidden next turn"),
                    [tokenizer.eos_token_id],
                ]
                request = ChatCompletionRequest.model_validate(
                    {
                        "messages": [{"role": "user", "content": "hello"}],
                        "user_template": "{{ messages[0].content }}",
                        "stream": True,
                        "max_tokens": 64,
                    }
                )
                self.assertEqual(endpoint.render_chat(request).rendered_prompt, "hello")
                frames, result = await self.complete(endpoint, request)
                self.assertEqual(result.choices[0].message.content, "visible")
                self.assertEqual(result.choices[0].finish_reason, FinisheReason.stop)
                self.assertEqual(
                    "".join(frame.choices[0].delta.content or "" for frame in frames),
                    "visible",
                )
                self.assertEqual(backend.yielded_chunks, 2)
                self.assertTrue(backend.closed)
                self.assertIn(
                    stop_ids, backend.inputs[0].generate_config.stop_words_list
                )

    async def test_force_preserves_qwen_tokenizer_protocol_boundaries(self):
        for marker in ("<|im_start|>", "<|im_end|>"):
            with self.subTest(marker=marker):
                backend = ScriptedBackend([])
                endpoint = self.make_endpoint(backend, force=True)
                tokenizer = endpoint.tokenizer
                stop_ids = tokenizer.encode(marker)
                self.assertIn(stop_ids, tokenizer.stop_words_id_list)
                backend.chunks = [
                    tokenizer.encode("visible"),
                    stop_ids,
                    tokenizer.encode("hidden next turn"),
                    [tokenizer.eos_token_id],
                ]
                request = ChatCompletionRequest.model_validate(
                    {
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                        "max_tokens": 64,
                    }
                )
                _, result = await self.complete(endpoint, request)
                self.assertEqual(result.choices[0].message.content, "visible")
                self.assertEqual(result.choices[0].finish_reason, FinisheReason.stop)
                self.assertEqual(backend.yielded_chunks, 2)
                self.assertTrue(backend.closed)
                self.assertIn(
                    stop_ids, backend.inputs[0].generate_config.stop_words_list
                )

    async def test_force_keeps_configured_and_request_stop_semantics(self):
        for force in (False, True):
            for terminator in ("ENV_STOP", "REQUEST_STOP", "EXTRA_STOP"):
                with self.subTest(force=force, terminator=terminator):
                    backend = ScriptedBackend([])
                    endpoint = self.make_endpoint(
                        backend, force=force, env_stop=["ENV_STOP"]
                    )
                    tokenizer = endpoint.tokenizer
                    prefix = "visible MODEL_STOP still visible "
                    backend.chunks = [
                        tokenizer.encode(prefix),
                        tokenizer.encode(terminator),
                        tokenizer.encode("hidden"),
                        [tokenizer.eos_token_id],
                    ]
                    request = ChatCompletionRequest.model_validate(
                        {
                            "messages": [{"role": "user", "content": "hello"}],
                            "stream": True,
                            "stop": "REQUEST_STOP",
                            "extra_configs": {
                                "stop_words_list": [tokenizer.encode("EXTRA_STOP")]
                            },
                            "max_tokens": 128,
                        }
                    )
                    _, result = await self.complete(endpoint, request)
                    self.assertEqual(
                        result.choices[0].message.content,
                        prefix if force else "visible ",
                    )
                    config = backend.inputs[0].generate_config
                    for stop in ("ENV_STOP", "REQUEST_STOP", "EXTRA_STOP"):
                        self.assertIn(tokenizer.encode(stop), config.stop_words_list)
                    if force:
                        self.assertNotIn(
                            tokenizer.encode("MODEL_STOP"), config.stop_words_list
                        )
                    else:
                        self.assertIn(
                            tokenizer.encode("MODEL_STOP"), config.stop_words_list
                        )
                    self.assertTrue(backend.closed)


if __name__ == "__main__":
    unittest.main()
