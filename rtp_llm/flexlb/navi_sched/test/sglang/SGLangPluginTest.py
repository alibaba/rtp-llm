import asyncio
import os
import subprocess
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class SGLangLifespanTest(unittest.IsolatedAsyncioTestCase):
    async def test_http_startup_waits_for_agent_and_cleans_up(self):
        from navi_sched.adapters.sglang.NaviSchedSGLangPlugin import _http_lifespan
        from sglang.srt.entrypoints import http_server

        started = asyncio.Event()
        ready = asyncio.Event()
        serving = asyncio.Event()
        events = []
        state = {"upstream": True}

        async def start():
            started.set()
            await ready.wait()
            events.append("agent_ready")

        runtime = SimpleNamespace(
            start=start, close=lambda: events.append("agent_closed")
        )

        @asynccontextmanager
        async def upstream(app):
            events.append("upstream_started")
            try:
                yield state
            finally:
                events.append("upstream_closed")

        async def serve():
            async with _http_lifespan(upstream, object()) as value:
                self.assertIs(value, state)
                serving.set()
                events.append("serving")

        with patch.object(
            http_server,
            "get_global_state",
            return_value=SimpleNamespace(
                tokenizer_manager=SimpleNamespace(navi_sched=runtime)
            ),
        ):
            task = asyncio.create_task(serve())
            await started.wait()
            self.assertFalse(serving.is_set())
            ready.set()
            await task
        self.assertEqual(
            events,
            [
                "upstream_started",
                "agent_ready",
                "serving",
                "agent_closed",
                "upstream_closed",
            ],
        )

    async def test_agent_start_failure_prevents_http_readiness(self):
        from navi_sched.adapters.sglang.NaviSchedSGLangPlugin import _http_lifespan
        from sglang.srt.entrypoints import http_server

        events = []

        async def start():
            raise RuntimeError("agent failed to initialize")

        runtime = SimpleNamespace(
            start=start, close=lambda: events.append("agent_closed")
        )

        @asynccontextmanager
        async def upstream(app):
            try:
                yield
            finally:
                events.append("upstream_closed")

        with patch.object(
            http_server,
            "get_global_state",
            return_value=SimpleNamespace(
                tokenizer_manager=SimpleNamespace(navi_sched=runtime)
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "agent failed to initialize"):
                async with _http_lifespan(upstream, object()):
                    self.fail("HTTP startup accepted an uninitialized Agent")
        self.assertEqual(events, ["agent_closed", "upstream_closed"])


class SGLangPluginTest(unittest.TestCase):
    def test_dspark_parallel_resolution_keeps_pipeline_parallelism_rejected(self):
        from navi_sched.adapters.sglang.NaviSchedSGLangPlugin import _server_args

        args = SimpleNamespace(
            kv_events_config=None,
            speculative_algorithm="DSPARK",
            disaggregation_mode="null",
            device="cuda",
            attn_dp_size=1,
        )
        resolved = []

        def resolve(value):
            resolved.append(value)
            return "native result"

        with patch(
            "sglang.srt.runtime_context.get_platform",
            return_value=SimpleNamespace(is_cuda=True),
        ):
            self.assertEqual(_server_args(resolve, args), "native result")
            self.assertEqual(resolved, [args])
            args.attn_dp_size = 8
            args.tokenizer_worker_num = 8
            self.assertEqual(_server_args(resolve, args), "native result")
            self.assertEqual(resolved, [args, args])
            args.pp_size = 2
            with self.assertRaisesRegex(ValueError, "requires pp_size=1"):
                _server_args(resolve, args)

    def test_real_hook_registry_preserves_native_destination_entry(self):
        result = self._run_real_hook_registry("0.5.23")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_local_distribution_version_installs_real_hooks(self):
        result = self._run_real_hook_registry("0.5.23+vendor.1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_different_public_versions_are_rejected(self):
        for version in ("0.5.24+vendor.1", "0.5.23rc1", "0.5.23.post1"):
            with self.subTest(version=version):
                result = self._run_real_hook_registry(version)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("requires SGLang 0.5.23", result.stderr)

    def _run_real_hook_registry(self, version):
        adapters = str(Path(__file__).resolve().parents[2] / "adapters")
        script = """
import inspect
import json
import os
import navi_sched
import navi_sched.adapters
navi_sched.__path__.insert(0, NAVI_PACKAGE)
navi_sched.adapters.__path__.insert(0, ADAPTERS)
import sglang.version
sglang.version.__version__ = os.environ["TEST_SGLANG_VERSION"]
from sglang.srt.plugins import load_plugins
load_plugins()
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from navi_sched.adapters.sglang import NaviSchedSGLangPlugin as plugin
from navi_sched.adapters.sglang.SGLangRuntime import SGLangRuntime
from sglang.utils import TypeBasedDispatcher
from types import SimpleNamespace
from unittest.mock import patch
from sglang.srt.runtime_context import get_platform
assert TokenizerManager.create_abort_task is not inspect.unwrap(TokenizerManager.create_abort_task)
manager = SimpleNamespace(_result_dispatcher=TypeBasedDispatcher([]))
args = ServerArgs(model_path="none", device="cuda")
manager.server_args = args
platform = get_platform()
class CudaPlatform:
    is_cuda = True
    def __getattr__(self, name):
        return getattr(platform, name)
with patch("sglang.srt.runtime_context.get_platform", return_value=CudaPlatform()):
    args.resolve_once()
assert json.loads(args.kv_events_config) == {"publisher": "navi_sched", "format": "dynamo"}
plugin._manager_init(lambda manager: None, manager)
try:
    assert manager.navi_sched.original_generate is inspect.unwrap(TokenizerManager.generate_request)
    assert manager.navi_sched.original_generate is not TokenizerManager.generate_request
finally:
    manager.navi_sched.close()
""".replace(
            "ADAPTERS", repr(adapters)
        ).replace(
            "NAVI_PACKAGE", repr(str(Path(adapters).parent / "navi_sched"))
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            env={
                **os.environ,
                "SGLANG_NAVI_SCHED_ENABLED": "1",
                "SGLANG_NAVI_SCHED_TARGET_BIZ": "test_worker",
                "SGLANG_NAVI_SCHED_MASTER_BIZ": "test_master",
                "SGLANG_PLUGINS": "navi_sched",
                "TEST_SGLANG_VERSION": version,
            },
            capture_output=True,
            text=True,
            timeout=60,
        )


if __name__ == "__main__":
    unittest.main()
