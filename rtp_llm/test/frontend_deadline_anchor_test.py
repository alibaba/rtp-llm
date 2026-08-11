import contextlib
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import ANY, MagicMock, patch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    StreamResponseObject,
)
from rtp_llm.utils.base_model_datatypes import RequestDeadlineAnchor
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)


async def collect_last(responses):
    response = None
    async for response in responses:
        pass
    return response


async def empty_source():
    if False:
        yield None


class GenerateChoiceHarness:
    tokenizer = object()

    def render_response_stream(self, output_generator, request, generate_config):
        async def render():
            async with contextlib.aclosing(output_generator):
                yield StreamResponseObject()

        return render()


class BackendVisitorHarness:
    def __init__(self):
        self.deadline_snapshot = None

    async def enqueue(self, generate_input):
        self.deadline_snapshot = (
            generate_input.request_deadline_monotonic_s,
            generate_input.request_deadline_unix_ms,
            generate_input.ttft_deadline_monotonic_s,
        )
        return empty_source()


class FrontendDeadlineAnchorTest(IsolatedAsyncioTestCase):
    async def test_chat_completion_passes_one_ingress_anchor_to_renderer_chain(self):
        server = object.__new__(FrontendServer)
        server._global_controller = MagicMock()
        server._global_controller.increment.return_value = 1
        server.py_env_configs = SimpleNamespace(
            server_config=SimpleNamespace(ip="127.0.0.1", server_port=1234)
        )
        server.server_id = "0"
        server._openai_endpoint = MagicMock()
        server._openai_endpoint.chat_completion.return_value = (
            CompleteResponseAsyncGenerator(empty_source(), collect_last)
        )

        async def infer_wrap(_request, _raw_request, generate_call):
            return generate_call()

        server._infer_wrap = infer_wrap
        request = SimpleNamespace(model_dump=MagicMock(return_value={}))
        anchor = RequestDeadlineAnchor(monotonic_s=10.0, unix_ms=20_000)

        with patch(
            "rtp_llm.frontend.frontend_server.RequestDeadlineAnchor.now",
            return_value=anchor,
        ) as deadline_now, patch(
            "rtp_llm.frontend.frontend_server.generate_request_id", return_value=17
        ):
            response = await FrontendServer.chat_completion(
                server, request, MagicMock()
            )

        server._openai_endpoint.chat_completion.assert_called_once_with(
            17, request, ANY, anchor
        )
        deadline_now.assert_called_once_with()
        await response.aclose()

    async def test_generate_choice_uses_anchor_before_backend_enqueue(self):
        visitor = BackendVisitorHarness()
        renderer = GenerateChoiceHarness()
        anchor = RequestDeadlineAnchor(monotonic_s=10.0, unix_ms=20_000)
        choice_generator = CustomChatRenderer.generate_choice(
            renderer,
            request_id=1,
            input_ids=[1],
            mm_inputs=[],
            generate_config=GenerateConfig(
                is_streaming=True, timeout_ms=500, ttft_timeout_ms=200
            ),
            backend_rpc_server_visitor=visitor,
            request=object(),
            request_deadline_anchor=anchor,
        )

        await choice_generator.__anext__()
        await choice_generator.aclose()

        self.assertEqual(visitor.deadline_snapshot, (10.5, 20_500, 10.2))


if __name__ == "__main__":
    main()
