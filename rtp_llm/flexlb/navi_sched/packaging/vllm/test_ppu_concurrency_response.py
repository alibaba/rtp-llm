import os

"""Exercise actual API admission and ASGI cleanup without loading a device."""

import ast
import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace

from starlette.responses import JSONResponse, StreamingResponse

ROOT = Path(os.environ.get("VLLM_SOURCE_ROOT", Path.cwd())).resolve()


class Controller:
    def __init__(self):
        self.current = 0
        self.releases = 0

    def increment(self, rank_id):
        self.current += 1

    def decrement(self):
        self.current -= 1
        self.releases += 1


def load_namespace(kind, controller, handler):
    namespace = {
        "StreamingResponse": StreamingResponse,
        "JSONResponse": JSONResponse,
        "controller": controller,
        "current_time_ms": lambda: 0,
        "reportMetricWithRankId": lambda *args: None,
        "GaugeMetrics": SimpleNamespace(
            LANTENCY_METRIC=1,
            RESPONSE_FIRST_TOKEN_RT_METRIC=2,
            RESPONSE_ITER_RT_METRIC=3,
        ),
        "AccMetrics": SimpleNamespace(SUCCESS_QPS_METRIC=1),
        "metrics_header": lambda *args: {},
        "ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL": "endpoint-load-metrics-format",
        "ErrorResponse": type("ErrorResponse", (), {}),
        "ChatCompletionResponse": type("ChatCompletionResponse", (), {}),
        "CompletionResponse": type("CompletionResponse", (), {}),
        "MediaDownloadSizeExceededError": type("MediaError", (Exception,), {}),
        "ConcurrencyException": type("ConcurrencyException", (Exception,), {}),
        "chat": lambda request: handler,
        "completion": lambda request: handler,
    }
    utils = ast.parse((ROOT / "vllm/whale_vllm/vllm_server/utils/utils.py").read_text())
    nodes = [
        n
        for n in utils.body
        if isinstance(n, (ast.ClassDef, ast.AsyncFunctionDef))
        and n.name in ("ConcurrencyStreamingResponse", "controller_decrease_wrapper")
    ]
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "production cleanup", "exec"),
        namespace,
    )
    source = ast.parse(
        (ROOT / f"vllm/entrypoints/openai/{kind}/api_router.py").read_text()
    )
    route = next(
        n
        for n in source.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == f"create_{kind}"
    )
    route.decorator_list = []
    route.returns = None
    for arg in route.args.args:
        arg.annotation = None
    exec(
        compile(
            ast.Module(body=[route], type_ignores=[]), "production admission", "exec"
        ),
        namespace,
    )
    return namespace


class PoolingCapacityTest(unittest.TestCase):
    def test_rejected_third_request_cannot_release_two_inflight_slots(self):
        from threading import Lock
        from unittest.mock import Mock
        import time

        namespace = dict(
            Lock=Lock,
            time=time,
            EngineRequestRejectedError=RuntimeError,
            kmonitor=SimpleNamespace(report=Mock()),
            AccMetrics=SimpleNamespace(
                QPS_METRIC=0, SUCCESS_QPS_METRIC=1, ERROR_QPS_METRIC=2
            ),
            GaugeMetrics=SimpleNamespace(LANTENCY_METRIC=3),
            reportMetricWithRankId=Mock(),
            current_time_ms=lambda: 0,
        )
        source = ast.parse(
            (
                ROOT / "vllm/whale_vllm/vllm_server/utils/concurrency_controller.py"
            ).read_text()
        )
        classes = [
            n
            for n in source.body
            if isinstance(n, ast.ClassDef)
            and n.name in ("ConcurrencyException", "ConcurrencyController")
        ]
        exec(
            compile(
                ast.Module(body=classes, type_ignores=[]), "actual controller", "exec"
            ),
            namespace,
        )
        controller = namespace["ConcurrencyController"](max_concurrency=2)
        namespace.update(
            controller=controller,
            create_concurrency_response=lambda *args: JSONResponse({}, status_code=429),
        )
        source = ast.parse(
            (ROOT / "vllm/whale_vllm/vllm_server/utils/utils.py").read_text()
        )
        serve = next(
            n
            for n in source.body
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "serve_pooling_request"
        )
        exec(
            compile(
                ast.Module(body=[serve], type_ignores=[]),
                "actual pooling entry",
                "exec",
            ),
            namespace,
        )

        async def check():
            gate = asyncio.Event()

            async def handler(*args):
                await gate.wait()
                return JSONResponse({})

            request = SimpleNamespace(update_request=lambda: None)
            raw = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(rank_id=0)))
            first = asyncio.create_task(
                namespace["serve_pooling_request"](request, raw, handler)
            )
            second = asyncio.create_task(
                namespace["serve_pooling_request"](request, raw, handler)
            )
            await asyncio.sleep(0)
            self.assertEqual(controller.current_concurrency, 2)
            rejected = await namespace["serve_pooling_request"](request, raw, handler)
            self.assertEqual(rejected.status_code, 429)
            self.assertEqual(controller.current_concurrency, 2)
            gate.set()
            await asyncio.gather(first, second)
            self.assertEqual(controller.current_concurrency, 0)

        asyncio.run(check())


class ConcurrencyResponseTest(unittest.TestCase):
    def test_every_response_termination_releases_exactly_once(self):
        for kind in ("chat_completion", "completion"):
            for termination in (
                "complete",
                "headers_disconnect",
                "headers_cancel",
                "body_disconnect",
                "body_cancel",
                "body_cleanup_error",
                "generator_error",
            ):
                with self.subTest(kind=kind, termination=termination):
                    asyncio.run(self.run_case(kind, termination))

    async def run_case(self, kind, termination):
        controller = Controller()
        lifecycle = {"started": False, "closed": False}

        async def generate():
            lifecycle["started"] = True
            try:
                yield "first"
                if termination == "generator_error":
                    raise RuntimeError("generation failed")
                yield "second"
            finally:
                lifecycle["closed"] = True
                if termination == "body_cleanup_error":
                    raise RuntimeError("backend cleanup failed")

        backend_generator = generate()

        async def handle(*args):
            return backend_generator

        handler = SimpleNamespace(**{f"create_{kind}": handle})
        ns = load_namespace(kind, controller, handler)
        request = SimpleNamespace(stream=True, update_request=lambda: None)
        raw = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(rank_id=0)), headers={}
        )
        response = await ns[f"create_{kind}"](request, raw)
        self.assertEqual(1, controller.current)
        sent = []

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if (
                message["type"] == "http.response.start"
                and termination.startswith("headers")
            ) or (
                message["type"] == "http.response.body"
                and termination.startswith("body")
            ):
                if termination.endswith("cancel"):
                    raise asyncio.CancelledError()
                raise OSError("connection lost")
            sent.append(message)

        try:
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
            )
        except (Exception, asyncio.CancelledError):
            if termination == "complete":
                raise
        else:
            self.assertEqual("complete", termination)
        self.assertEqual(0, controller.current)
        self.assertEqual(1, controller.releases)
        self.assertEqual(lifecycle["started"], lifecycle["closed"])
        await response.body_iterator.aclose()
        self.assertEqual(0, controller.current)
        self.assertEqual(1, controller.releases)
        if termination == "complete":
            self.assertEqual(
                b"firstsecond",
                b"".join(
                    m.get("body", b"")
                    for m in sent
                    if m["type"] == "http.response.body"
                ),
            )


if __name__ == "__main__":
    unittest.main()
