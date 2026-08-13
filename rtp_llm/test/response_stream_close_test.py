import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch

from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    StreamResponseObject,
)
from rtp_llm.utils.complete_response_async_generator import (
    CompleteResponseAsyncGenerator,
)


class CloseTrackedAsyncIterator:
    def __init__(self, values):
        self._values = iter(values)
        self._closed = False
        self.close_calls = 0
        self.release_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        try:
            return next(self._values)
        except StopIteration:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def aclose(self):
        self.close_calls += 1
        if not self._closed:
            self._closed = True
            self.release_count += 1


async def collect_last(responses):
    response = None
    async for response in responses:
        pass
    return response


class RendererHarness:
    render_response_stream = CustomChatRenderer.render_response_stream

    async def _create_status_list(self, n, request):
        return [object() for _ in range(n)]

    def in_think_mode(self, request):
        return False

    def should_process_think(self, request):
        return False

    async def _generate_first(self, nums_output):
        return "first"

    async def _update_single_status(self, *args):
        return SimpleNamespace(extra_outputs=object())

    async def _generate_stream_response(self, delta_list, think_status_list):
        return "delta"

    def _check_all_finished(self, status_list):
        return True

    async def _flush_buffer(self, *args):
        return "flush"

    async def _generate_final(self, *args):
        return "final"


class ControllerHarness:
    def __init__(self):
        self.decrement_count = 0

    def decrement(self):
        self.decrement_count += 1


class FrontendHarness:
    def __init__(self):
        self._frontend_worker = object()
        self._access_logger = MagicMock()
        self._global_controller = ControllerHarness()
        self.rank_id = "0"
        self.server_id = "0"

    async def _collect_complete_response_and_record_access_log(self, request, response):
        return None


def make_openai_response(source):
    return OpenaiEndpoint._complete_stream_response(source, debug_info=None)


class ResponseStreamCloseTest(IsolatedAsyncioTestCase):
    async def test_openai_close_reaches_choice_generator_once(self):
        source = CloseTrackedAsyncIterator([StreamResponseObject()])
        response = make_openai_response(source)

        await response.__anext__()
        await response.aclose()
        await response.aclose()

        self.assertEqual(source.close_calls, 1)
        self.assertEqual(source.release_count, 1)

    async def test_renderer_close_reaches_model_output_once(self):
        request = SimpleNamespace(n=1, aux_info=False)
        generate_config = SimpleNamespace(
            stop_words_str=[],
            variable_num_beams=[],
            num_beams=1,
            is_streaming=True,
            max_new_tokens=1,
        )
        output = SimpleNamespace(generate_outputs=[object()])
        source = CloseTrackedAsyncIterator([output])
        rendered = RendererHarness().render_response_stream(
            source, request, generate_config
        )

        self.assertEqual(await rendered.__anext__(), "first")
        await rendered.aclose()

        self.assertEqual(source.close_calls, 1)
        self.assertEqual(source.release_count, 1)

    async def test_frontend_close_reaches_full_openai_chain_once(self):
        source = CloseTrackedAsyncIterator([StreamResponseObject()])
        openai_response = make_openai_response(source)
        server = FrontendHarness()
        with patch("rtp_llm.frontend.frontend_server.kmonitor", new=MagicMock()):
            reported_response = await FrontendServer._call_generate_with_report(
                server, lambda: openai_response
            )
            body = FrontendServer.stream_response(
                server, {"stream": True}, reported_response
            )
            self.assertTrue((await body.__anext__()).startswith("data: "))
            await body.aclose()
            await reported_response.aclose()

        self.assertEqual(source.close_calls, 1)
        self.assertEqual(source.release_count, 1)


if __name__ == "__main__":
    main()
