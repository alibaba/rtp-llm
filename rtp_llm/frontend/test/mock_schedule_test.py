import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI, HTTPException
from rtp_llm.frontend.mock_schedule import register_mock_schedule
from rtp_llm.utils.concurrency_controller import ConcurrencyController


class MockScheduleTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.visitor = SimpleNamespace(
            max_seq_len=8192,
            fill_request_info=Mock(),
            get_master_route_addrs=AsyncMock(side_effect=self.accept),
            enqueue=Mock(side_effect=AssertionError("must never fetch engine output")),
        )
        self.frontend = SimpleNamespace(
            _frontend_worker=SimpleNamespace(
                backend_rpc_server_visitor=self.visitor,
                pipeline=SimpleNamespace(encode=Mock(return_value=[1, 2, 3])),
            ),
            py_env_configs=SimpleNamespace(
                server_config=SimpleNamespace(ip="127.0.0.1", server_port=22910)
            ),
            server_id="test",
            _global_controller=ConcurrencyController(1),
        )
        app = FastAPI()

        async def track(call):
            return await call()

        register_mock_schedule(app, self.frontend, track)
        self.schedule = next(
            r.endpoint for r in app.routes if r.path == "/internal/mock/schedule"
        )

    def tearDown(self):
        self.assertEqual(
            self.frontend._global_controller.get_available_concurrency(), 1
        )

    async def accept(self, request):
        self.request = request
        request.enqueued_by_master = True

    async def test_ack_is_not_inference_completion(self):
        result = await self.schedule(
            {"input_ids": [1, 2], "max_new_tokens": 17, "priority": 70}
        )
        self.assertEqual(result.status_code, 202)
        body = json.loads(result.body)
        self.assertFalse(body["inference_completed"])
        self.assertFalse(body["fetch_output_stream"])
        self.assertEqual(self.request.token_ids.tolist(), [1, 2])
        self.assertEqual(self.request.generate_config.max_new_tokens, 17)
        self.assertEqual(self.request.headers["x-dashscope-inner-qos-level"], "70")
        self.visitor.enqueue.assert_not_called()

    async def test_prompt_uses_frontend_tokenizer(self):
        await self.schedule({"prompt": "hello"})
        self.frontend._frontend_worker.pipeline.encode.assert_called_once_with("hello")
        self.assertEqual(self.request.token_ids.tolist(), [1, 2, 3])

    async def test_nonbatch_is_not_accepted(self):
        self.visitor.get_master_route_addrs.side_effect = None
        self.visitor.get_master_route_addrs.return_value = None
        with self.assertRaises(HTTPException) as ctx:
            await self.schedule({"input_ids": [1]})
        self.assertEqual(ctx.exception.status_code, 409)

    async def test_route_failure_is_not_accepted(self):
        self.visitor.get_master_route_addrs.side_effect = None
        self.visitor.get_master_route_addrs.return_value = object()
        with self.assertRaises(HTTPException) as ctx:
            await self.schedule({"input_ids": [1]})
        self.assertEqual(ctx.exception.status_code, 503)

    async def test_invalid_plan_never_reaches_master(self):
        for body in (
            {"input_ids": [True]},
            {"input_ids": []},
            {"input_ids": [1], "prompt": "hello"},
            {"input_ids": [1], "max_new_tokens": 0},
            {"input_ids": [1] * 8192},
            {"prompt": 123},
        ):
            with self.subTest(body_keys=list(body)):
                with self.assertRaises(HTTPException) as ctx:
                    await self.schedule(body)
                self.assertEqual(ctx.exception.status_code, 422)
        self.visitor.get_master_route_addrs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
