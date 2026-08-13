import asyncio
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import MagicMock, patch

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.frontend.frontend_server import FrontendServer


class _Chunk:
    def model_dump_json(self, exclude_none=True):
        return '{"id":"chunk"}'


class _Response:
    def __init__(self, values=(), error=None):
        self._values = iter(values)
        self._error = error
        self._error_raised = False
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._values)
        except StopIteration:
            if self._error is not None and not self._error_raised:
                self._error_raised = True
                raise self._error
            raise StopAsyncIteration

    async def aclose(self):
        self.close_calls += 1


class _Controller:
    def __init__(self):
        self.decrement_calls = 0

    def decrement(self):
        self.decrement_calls += 1


class _AccessLogger:
    def __init__(self):
        self.errors = []

    def log_exception_access(self, request, error):
        self.errors.append(error)


class _FrontendHarness:
    def __init__(self, collect_error=None, collect_gate=None):
        self._global_controller = _Controller()
        self._access_logger = _AccessLogger()
        self.rank_id = "0"
        self.server_id = "0"
        self.complete_response_calls = 0
        self.collect_error = collect_error
        self.collect_gate = collect_gate
        self.collect_started = asyncio.Event()

    async def _collect_complete_response_and_record_access_log(
        self, request, response
    ):
        self.complete_response_calls += 1
        self.collect_started.set()
        if self.collect_gate is not None:
            await self.collect_gate.wait()
        if self.collect_error is not None:
            raise self.collect_error


class FrontendSseDoneTest(IsolatedAsyncioTestCase):
    async def _collect(self, request, response, collect_error=None):
        server = _FrontendHarness(collect_error=collect_error)
        body = FrontendServer.stream_response(server, request, response)
        with patch(
            "rtp_llm.frontend.frontend_server.kmonitor",
            new=MagicMock(),
        ):
            chunks = [chunk async for chunk in body]
        return server, chunks

    async def test_openai_natural_completion_emits_standard_done(self):
        response = _Response([_Chunk()])
        server, chunks = await self._collect({"stream": True}, response)

        self.assertEqual(
            chunks,
            [
                'data: {"id":"chunk"}\r\n\r\n',
                "data: [DONE]\r\n\r\n",
            ],
        )
        self.assertEqual(server.complete_response_calls, 1)
        self.assertEqual(response.close_calls, 1)

    async def test_legacy_natural_completion_keeps_existing_done(self):
        response = _Response([_Chunk()])
        server, chunks = await self._collect({"stream": False}, response)

        self.assertEqual(
            chunks,
            [
                'data:{"id":"chunk"}\r\n\r\n',
                "data:[done]\r\n\r\n",
            ],
        )
        self.assertEqual(server.complete_response_calls, 1)
        self.assertEqual(response.close_calls, 1)

    async def test_cancel_does_not_emit_success_done(self):
        response = _Response(error=asyncio.CancelledError())
        server, chunks = await self._collect({"stream": True}, response)

        self.assertEqual(chunks, [])
        self.assertEqual(server.complete_response_calls, 0)
        self.assertEqual(len(server._access_logger.errors), 1)
        self.assertEqual(response.close_calls, 1)

    async def test_error_does_not_emit_success_done(self):
        response = _Response(error=RuntimeError("injected"))
        with patch(
            "rtp_llm.frontend.frontend_server.format_exception",
            return_value={"error_code": 1, "message": "injected"},
        ):
            server, chunks = await self._collect({"stream": True}, response)

        self.assertEqual(len(chunks), 1)
        self.assertNotIn("[DONE]", chunks[0])
        self.assertIn("injected", chunks[0])
        self.assertEqual(server.complete_response_calls, 0)
        self.assertEqual(len(server._access_logger.errors), 1)
        self.assertEqual(response.close_calls, 1)

    async def test_tool_parse_error_after_chunk_emits_606_without_done(self):
        response = _Response(
            [_Chunk()],
            error=FtRuntimeException(
                ExceptionType.EXECUTION_EXCEPTION, "incomplete tool call"
            ),
        )

        server, chunks = await self._collect({"stream": True}, response)

        self.assertEqual(chunks[0], 'data: {"id":"chunk"}\r\n\r\n')
        self.assertIn('"error_code": 606', chunks[1])
        self.assertIn('"error_code_str": "606_EXECUTION_EXCEPTION"', chunks[1])
        self.assertIn('"message": "incomplete tool call"', chunks[1])
        self.assertNotIn("[DONE]", "".join(chunks))
        self.assertNotIn("<tool_call>", "".join(chunks))
        self.assertNotIn("Traceback", chunks[1])
        self.assertEqual(server.complete_response_calls, 0)
        self.assertEqual(len(server._access_logger.errors), 1)
        self.assertEqual(response.close_calls, 1)

    async def test_collect_error_does_not_emit_success_done(self):
        response = _Response([_Chunk()])
        with patch(
            "rtp_llm.frontend.frontend_server.format_exception",
            return_value={"error_code": 1, "message": "collect failed"},
        ):
            server, chunks = await self._collect(
                {"stream": True},
                response,
                collect_error=RuntimeError("collect failed"),
            )

        self.assertEqual(len(chunks), 2)
        self.assertNotIn("[DONE]", "".join(chunks))
        self.assertIn("collect failed", chunks[-1])
        self.assertEqual(server.complete_response_calls, 1)
        self.assertEqual(len(server._access_logger.errors), 1)
        self.assertEqual(response.close_calls, 1)

    async def test_collect_cancel_does_not_emit_success_done(self):
        response = _Response([_Chunk()])
        server, chunks = await self._collect(
            {"stream": True},
            response,
            collect_error=asyncio.CancelledError(),
        )

        self.assertEqual(chunks, ['data: {"id":"chunk"}\r\n\r\n'])
        self.assertEqual(server.complete_response_calls, 1)
        self.assertEqual(len(server._access_logger.errors), 1)
        self.assertEqual(response.close_calls, 1)

    async def test_done_waits_for_successful_collection(self):
        collect_gate = asyncio.Event()
        response = _Response([_Chunk()])
        server = _FrontendHarness(collect_gate=collect_gate)
        body = FrontendServer.stream_response(
            server,
            {"stream": True},
            response,
        )

        self.assertEqual(
            await body.__anext__(),
            'data: {"id":"chunk"}\r\n\r\n',
        )
        pending_done = asyncio.create_task(body.__anext__())
        await server.collect_started.wait()
        self.assertFalse(pending_done.done())

        collect_gate.set()
        self.assertEqual(await pending_done, "data: [DONE]\r\n\r\n")
        with self.assertRaises(StopAsyncIteration):
            await body.__anext__()
        self.assertEqual(response.close_calls, 1)


if __name__ == "__main__":
    main()
