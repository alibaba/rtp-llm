import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import Mock, patch

from jinja2 import Environment

from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    RoleEnum,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.chatglm47_renderer import ChatGlm47Renderer


def make_request(content: str, partial: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[
            ChatMessage(role=RoleEnum.user, content=content, partial=partial)
        ]
    )


class GlmRendererRequestContextTest(IsolatedAsyncioTestCase):
    @staticmethod
    def _renderer(template: str = "{{ messages[0].content }}"):
        renderer = object.__new__(ChatGlm47Renderer)
        renderer.chat_template = template
        renderer.tokenizer = Mock()
        renderer.tokenizer.encode.side_effect = lambda prompt: [len(prompt)]
        renderer.think_mode = True
        return renderer

    def test_chat_template_is_compiled_once_with_isolated_contexts(self):
        renderer = self._renderer()
        requests = [make_request(f"message-[{index:02d}]") for index in range(16)]
        original_from_string = Environment.from_string

        with patch.object(
            Environment,
            "from_string",
            autospec=True,
            side_effect=original_from_string,
        ) as from_string:
            with ThreadPoolExecutor(max_workers=8) as executor:
                rendered = list(executor.map(renderer.render_chat, requests))

        self.assertEqual(from_string.call_count, 1)
        for index, result in enumerate(rendered):
            self.assertEqual(result.rendered_prompt, f"message-[{index:02d}]")

    async def test_reasoning_uses_each_request_prompt_without_rerendering(self):
        renderer = self._renderer()
        force_request = make_request("first<think>")
        plain_request = make_request("second")

        renderer.render_chat(force_request)
        renderer.render_chat(plain_request)
        force_statuses, plain_statuses = await asyncio.gather(
            renderer._create_status_list(2, force_request),
            renderer._create_status_list(2, plain_request),
        )

        self.assertEqual(renderer.tokenizer.encode.call_count, 2)
        self.assertTrue(
            all(
                status.reasoning_parser.detector._in_reasoning
                for status in force_statuses
            )
        )
        self.assertTrue(
            all(
                not status.reasoning_parser.detector._in_reasoning
                for status in plain_statuses
            )
        )
        force_reasoning, force_content = force_statuses[
            0
        ].reasoning_parser.parse_non_stream("reasoning</think>answer")
        plain_reasoning, plain_content = plain_statuses[
            0
        ].reasoning_parser.parse_non_stream("reasoning</think>answer")
        self.assertEqual((force_reasoning, force_content), ("reasoning", "answer"))
        self.assertEqual(
            (plain_reasoning, plain_content), ("", "reasoning</think>answer")
        )
        self.assertNotIn(
            "_force_reasoning_from_rendered_prompt", force_request.model_dump()
        )

    async def test_direct_status_creation_renders_once_for_all_choices(self):
        renderer = self._renderer()
        request = make_request("direct<think>")

        statuses = await renderer._create_status_list(4, request)

        self.assertEqual(renderer.tokenizer.encode.call_count, 1)
        self.assertTrue(
            all(
                status.reasoning_parser.detector._in_reasoning
                for status in statuses
            )
        )

    async def test_direct_render_failure_is_not_retried_per_choice(self):
        renderer = self._renderer()
        renderer.render_chat = Mock(side_effect=ValueError("invalid template"))
        request = make_request("direct")

        statuses = await renderer._create_status_list(4, request)

        self.assertEqual(renderer.render_chat.call_count, 1)
        self.assertTrue(
            all(
                not status.reasoning_parser.detector._in_reasoning
                for status in statuses
            )
        )

    def test_partial_prepopulate_updates_marker_from_final_prompt(self):
        renderer = self._renderer("base<think>")
        endpoint = object.__new__(OpenaiEndpoint)
        endpoint.chat_renderer = renderer
        endpoint.template_renderer = None
        endpoint.tokenizer = renderer.tokenizer
        request = make_request("prepopulated answer", partial=True)

        rendered = OpenaiEndpoint.render_chat(endpoint, request)

        self.assertEqual(rendered.rendered_prompt, "base<think>prepopulated answer")
        self.assertFalse(request._force_reasoning_from_rendered_prompt)


if __name__ == "__main__":
    main()
