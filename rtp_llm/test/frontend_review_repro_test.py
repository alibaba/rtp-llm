"""Regression coverage for stop boundaries and K3 continuation contracts."""

import asyncio
import unittest
from types import SimpleNamespace

import torch
import xgrammar as xgr

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.config.py_config_modules import GenerateEnvConfig, VitConfig
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import RESPONSE_OPEN, THINK_OPEN
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    FinisheReason,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.custom_renderer import CustomChatRenderer, RendererParams
from rtp_llm.openai.renderers.kimi_k3_renderer import KimiK3Renderer
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
)


class _Tokenizer:
    """Small vocabulary with native-style, indivisible XTML control tokens."""

    def __init__(self, extra_pieces=()):
        self.vocab = [chr(index) for index in range(128)] + [
            "<|open|>",
            "<|close|>",
            "<|sep|>",
            "think",
            "response",
            "message",
        ]
        self.vocab[0] = "<eos>"
        self.vocab.extend(extra_pieces)
        self.ids = {text: index for index, text in enumerate(self.vocab)}
        self.pieces = sorted(self.ids, key=len, reverse=True)

    def encode(self, text, **_kwargs):
        result = []
        while text:
            piece = next(piece for piece in self.pieces if text.startswith(piece))
            result.append(self.ids[piece])
            text = text[len(piece) :]
        return result

    def decode(self, token_ids):
        return "".join(self.vocab[int(token)] for token in token_ids)

    def apply_chat_template(self, _messages, *, thinking, **_kwargs):
        return self.encode("prompt" + (THINK_OPEN if thinking else RESPONSE_OPEN))


class FrontendStopBoundaryReproTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tokenizer = _Tokenizer()

    def _renderer(self, renderer_type):
        return renderer_type(
            self.tokenizer,
            RendererParams(
                model_type="kimi_k3" if renderer_type is KimiK3Renderer else "test",
                max_seq_len=256,
                eos_token_id=0,
                stop_word_ids_list=[],
            ),
            GenerateEnvConfig(),
            vit_config=VitConfig(),
        )

    async def _render(
        self,
        renderer_type,
        chunks,
        *,
        stop_ids=(),
        stop_strings=(),
        max_new_tokens=128,
        stream=True,
        logprobs=True,
    ):
        renderer = self._renderer(renderer_type)
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="question")],
            stream=stream,
            logprobs=logprobs,
            top_logprobs=0 if logprobs else None,
            enable_thinking=False,
            stop=list(stop_strings),
        )
        config = GenerateConfig(
            max_new_tokens=max_new_tokens,
            is_streaming=True,
            stop_words_list=list(stop_ids)
            + [self.tokenizer.encode(word) for word in stop_strings],
            stop_words_str=list(stop_strings),
            in_think_mode=False,
            thinking_mode=ThinkingMode.DISABLED,
        )

        async def outputs():
            total = 0
            for text in chunks:
                await asyncio.sleep(0)
                tokens = self.tokenizer.encode(text)
                total += len(tokens)
                probabilities = torch.zeros(len(tokens), len(self.tokenizer.vocab))
                probabilities[torch.arange(len(tokens)), tokens] = 1.0
                yield GenerateOutputs(
                    generate_outputs=[
                        GenerateOutput(
                            output_ids=torch.tensor(tokens),
                            all_probs=probabilities if logprobs else None,
                            aux_info=AuxInfo(input_len=4, output_len=total),
                        )
                    ]
                )

        return [
            response
            async for response in renderer.render_response_stream(
                outputs(), request, config
            )
        ]

    @staticmethod
    def _content(responses):
        return "".join(
            choice.delta.content or ""
            for response in responses
            for choice in response.choices
        )

    @staticmethod
    def _probability_text(responses):
        return "".join(
            item.token
            for response in responses
            for choice in response.choices
            if choice.logprobs and choice.logprobs.content
            for item in choice.logprobs.content
        )

    def _assert_finished(self, responses, reason):
        self.assertEqual(responses[-1].choices[0].finish_reason, reason)
        self.assertIsNotNone(responses[-1].usage)

    async def test_id_only_stop_prefix_is_buffered_with_its_probabilities(self):
        for logprobs in (False, True):
            with self.subTest(logprobs=logprobs):
                responses = await self._render(
                    CustomChatRenderer,
                    ["A", "S", "T", "OP"],
                    stop_ids=[self.tokenizer.encode("STOP")],
                    logprobs=logprobs,
                )
                self.assertEqual(self._content(responses), "A")
                if logprobs:
                    self.assertEqual(self._probability_text(responses), "A")
                self._assert_finished(responses, FinisheReason.stop)

    async def test_id_only_stop_in_one_frame_drops_stop_and_trailing_rows(self):
        responses = await self._render(
            CustomChatRenderer,
            ["ASTOPB"],
            stop_ids=[self.tokenizer.encode("STOP")],
        )
        self.assertEqual(self._content(responses), "A")
        self.assertEqual(self._probability_text(responses), "A")
        self._assert_finished(responses, FinisheReason.stop)

    async def test_failed_id_stop_prefix_is_released_with_its_probabilities(self):
        responses = await self._render(
            CustomChatRenderer,
            ["A", "S", "T", "X", "<eos>"],
            stop_ids=[self.tokenizer.encode("STOP")],
        )
        self.assertEqual(self._content(responses), "ASTX")
        self.assertEqual(self._probability_text(responses), "ASTX")
        self._assert_finished(responses, FinisheReason.stop)

    async def test_length_limit_releases_incomplete_id_stop_prefix(self):
        responses = await self._render(
            CustomChatRenderer,
            ["A", "S", "T"],
            stop_ids=[self.tokenizer.encode("STOP")],
            max_new_tokens=3,
        )
        self.assertEqual(self._content(responses), "AST")
        self.assertEqual(self._probability_text(responses), "AST")
        self._assert_finished(responses, FinisheReason.length)

    async def test_k3_string_stop_drops_hidden_probability_rows(self):
        # A context-dependent token can contain the stop string and following
        # text without containing its standalone stop-token ID sequence.
        self.tokenizer = _Tokenizer(extra_pieces=["STOPB"])
        for stream in (False, True):
            with self.subTest(stream=stream):
                responses = await self._render(
                    KimiK3Renderer,
                    ["ASTOPB"],
                    stop_strings=["STOP"],
                    stream=stream,
                )
                self.assertEqual(self._content(responses), "A")
                self.assertEqual(self._probability_text(responses), "A")
                self._assert_finished(responses, FinisheReason.stop)

    async def test_k3_length_flush_reassembles_pending_xtml_before_emitting(self):
        chunks = ["A<|close|>", "response<|sep|>ST"]
        responses = await self._render(
            KimiK3Renderer,
            chunks,
            stop_strings=["STOP"],
            max_new_tokens=sum(len(self.tokenizer.encode(chunk)) for chunk in chunks),
            logprobs=False,
        )
        # The stop-prefix buffer contains the rest of the response closure.
        # Reassemble it before flushing; text after the closure is not content.
        self.assertEqual(self._content(responses), "A")
        self._assert_finished(responses, FinisheReason.length)


class K3PartialPromptReproTest(unittest.TestCase):
    def test_partial_prompt_grammar_continues_the_already_open_channel(self):
        tokenizer = _Tokenizer()
        compiler = xgr.GrammarCompiler(
            xgr.TokenizerInfo(tokenizer.vocab, stop_token_ids=[0]), max_threads=1
        )
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                renderer = KimiK3Renderer(
                    tokenizer,
                    RendererParams("kimi_k3", 256, 0, []),
                    GenerateEnvConfig(),
                    vit_config=VitConfig(),
                )
                endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
                endpoint.chat_renderer = renderer
                endpoint.template_renderer = renderer
                endpoint.tokenizer = tokenizer
                endpoint.model_type = "kimi_k3"
                request = ChatCompletionRequest(
                    messages=[
                        ChatMessage(role="user", content="question"),
                        ChatMessage(
                            role="assistant", content="prefilled ", partial=True
                        ),
                    ],
                    enable_thinking=thinking,
                )
                rendered = endpoint.render_chat(request)
                config = GenerateConfig(
                    max_new_tokens=32,
                    in_think_mode=thinking,
                    thinking_mode=(
                        ThinkingMode.ENABLED if thinking else ThinkingMode.DISABLED
                    ),
                    max_thinking_tokens=16 if thinking else 0,
                )
                plan = renderer.build_chat_completion_response_format_plan(
                    request, config, rendered.input_ids
                )
                plan.apply_to_config(config)
                matcher = xgr.GrammarMatcher(
                    compiler.compile_structural_tag(config.structural_tag)
                )
                self.assertTrue(matcher.accept_token(tokenizer.ids["n"]))


class SpeculativeBeamAdmissionReproTest(unittest.TestCase):
    def setUp(self):
        self.visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        self.visitor.sp_config = SimpleNamespace(model_type="mtp")

    @staticmethod
    def _input(**kwargs):
        return GenerateInput(
            request_id=1,
            token_ids=torch.tensor([1], dtype=torch.int32),
            mm_inputs=[],
            generate_config=GenerateConfig(max_new_tokens=4, **kwargs),
        )

    def test_variable_beams_are_rejected_before_speculative_execution(self):
        request = self._input(num_beams=1, variable_num_beams=[2, 1])
        self.assertTrue(request.generate_config.has_num_beams())
        with self.assertRaises(FtRuntimeException):
            self.visitor.check_sp_supported(request)

    def test_single_width_schedule_and_explicit_non_sp_request_remain_supported(self):
        self.visitor.check_sp_supported(
            self._input(num_beams=1, variable_num_beams=[1, 1])
        )
        self.visitor.check_sp_supported(
            self._input(variable_num_beams=[2, 1], force_disable_sp_run=True)
        )


if __name__ == "__main__":
    unittest.main()
