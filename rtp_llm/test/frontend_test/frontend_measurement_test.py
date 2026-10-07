import ast
import asyncio
import json
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Load the small measurement helper without importing the model/server stack.
FRONTEND = Path(__file__).parents[2] / "frontend"
sys.path.insert(0, str(FRONTEND))
import frontend_measurement

measure = frontend_measurement


class FrontendMeasurementTest(unittest.TestCase):
    def test_disabled_emits_nothing(self):
        with patch.dict(os.environ, {"RTP_FRONTEND_MEASUREMENT": "0"}), patch.object(
            measure.logging, "info"
        ) as log:
            self.assertIsNone(measure.begin("id", 1, 0, 0))
            measure.serialized(None, 0, 0, "x")
            measure.finish(None, "ok")
            log.assert_not_called()

    def test_correlated_single_completion_record_and_cpu_boundaries(self):
        with patch.dict(os.environ, {"RTP_FRONTEND_MEASUREMENT": "1"}), patch.object(
            measure.time, "time", side_effect=[100, 103, 105]
        ), patch.object(measure.time, "thread_time", return_value=2.5), patch.object(
            measure.logging, "info"
        ) as log:
            m = measure.begin("client", 123, 2, 3)
            measure.serialized(m, 2.0, 102, "中")
            measure.finish(m, "ok")
            self.assertEqual(m["serialization_cpu_s"], 0.5)
            self.assertEqual(m["serialization_wall_s"], 1)
            self.assertEqual(m["sse_json_bytes"], 3)
            self.assertEqual(m["client_request_id"], "client")
            self.assertEqual(
                json.loads(log.call_args.args[1])["server_request_id"], 123
            )


class StreamMeasurementTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tree = ast.parse((FRONTEND / "frontend_server.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef)
            and any(
                isinstance(m, ast.AsyncFunctionDef) and m.name == "stream_response"
                for m in n.body
            )
        )
        method = next(
            m
            for m in cls.body
            if isinstance(m, ast.AsyncFunctionDef) and m.name == "stream_response"
        )
        for arg in method.args.args:
            arg.annotation = None
        method.returns = None
        namespace = dict(
            asyncio=asyncio,
            time=time,
            json=json,
            logging=Mock(),
            frontend_measurement=frontend_measurement,
            CURRENT_TRACE_STATE=SimpleNamespace(get=lambda: None),
            format_exception=lambda exc: {"error": str(exc)},
            kmonitor=Mock(),
            AccMetrics=Mock(),
        )
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                "<actual stream_response>",
                "exec",
            ),
            namespace,
        )
        self.method = namespace["stream_response"]

        async def collect(*args):
            return None

        self.owner = SimpleNamespace(
            _global_controller=Mock(),
            _access_logger=Mock(),
            _collect_complete_response_and_record_access_log=collect,
            rank_id=0,
            server_id=0,
        )

    async def test_success_bytes_and_accounting_unchanged(self):
        async def source():
            for text in ('{"content":"你好"}', '{"content":"world"}'):
                yield SimpleNamespace(model_dump_json=lambda text=text, **kwargs: text)

        for enabled in (False, True):
            row = (
                dict(
                    serialization_cpu_s=0,
                    serialization_wall_s=0,
                    sse_events=0,
                    sse_json_bytes=0,
                )
                if enabled
                else None
            )
            with patch.object(frontend_measurement.logging, "info") as log:
                values = [
                    v
                    async for v in self.method(
                        self.owner, {"stream": True, "_measurement": row}, source()
                    )
                ]
            self.assertEqual(
                values,
                [
                    'data: {"content":"你好"}\r\n\r\n',
                    'data: {"content":"world"}\r\n\r\n',
                ],
            )
            self.assertEqual(log.call_count, int(enabled))
            if enabled:
                self.assertEqual(row["status"], "ok")
                self.assertEqual(row["sse_events"], 2)
                self.assertLessEqual(
                    row["stream_done_wall"], row["handler_finished_wall"]
                )
        self.assertEqual(self.owner._global_controller.decrement.call_count, 2)

    async def test_cancellation_closes_source_and_preserves_cleanup(self):
        closed = []

        async def source():
            try:
                yield SimpleNamespace(model_dump_json=lambda **kwargs: "{}")
                await asyncio.sleep(10)
            finally:
                closed.append(True)

        row = dict(
            serialization_cpu_s=0,
            serialization_wall_s=0,
            sse_events=0,
            sse_json_bytes=0,
        )
        response = self.method(
            self.owner, {"stream": True, "_measurement": row}, source()
        )
        await response.__anext__()
        with patch.object(frontend_measurement.logging, "info"):
            await response.aclose()
        self.assertEqual(closed, [True])
        self.assertEqual(row["status"], "interrupted")
        self.owner._global_controller.decrement.assert_called_once()


if __name__ == "__main__":
    unittest.main()
