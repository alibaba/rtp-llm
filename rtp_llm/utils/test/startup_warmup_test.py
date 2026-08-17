import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

# Load the standalone startup helper without importing rtp_llm.__init__, which
# intentionally pulls in the model runtime and GPU libraries.
MODULE_PATH = Path(__file__).resolve().parents[1] / "startup_warmup.py"
SPEC = importlib.util.spec_from_file_location("startup_warmup_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
STARTUP_WARMUP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STARTUP_WARMUP
SPEC.loader.exec_module(STARTUP_WARMUP)

MONITOR_MODULE_PATH = Path(__file__).resolve().parents[1] / "triton_compile_patch.py"
MONITOR_SPEC = importlib.util.spec_from_file_location(
    "triton_compile_patch_under_test", MONITOR_MODULE_PATH
)
assert MONITOR_SPEC is not None and MONITOR_SPEC.loader is not None
TRITON_COMPILE_PATCH = importlib.util.module_from_spec(MONITOR_SPEC)
MONITOR_SPEC.loader.exec_module(TRITON_COMPILE_PATCH)

CaseResult = STARTUP_WARMUP.CaseResult
PromptBuilder = STARTUP_WARMUP.PromptBuilder
WarmupError = STARTUP_WARMUP.WarmupError
parse_cases = STARTUP_WARMUP.parse_cases
publish_gate = STARTUP_WARMUP.publish_gate
publish_phase = STARTUP_WARMUP.publish_phase
snapshot_jit_artifacts = STARTUP_WARMUP.snapshot_jit_artifacts
compile_event_count = STARTUP_WARMUP.compile_event_count
validate_second_round = STARTUP_WARMUP.validate_second_round
ServingPathWarmup = STARTUP_WARMUP.ServingPathWarmup
WarmupCase = STARTUP_WARMUP.WarmupCase


class FakeTokenizerClient:
    def post_json(self, path, payload):
        self.last_path = path
        content = payload["messages"][0]["content"]
        return {"token_ids": list(range(len(content.split()) + 5))}


class StartupWarmupTest(unittest.TestCase):
    def test_parse_cases_deduplicates_and_validates(self):
        cases = parse_cases("64x1, 256x8,64x1")
        self.assertEqual(
            [(case.target_tokens, case.batch_size) for case in cases],
            [(64, 1), (256, 8)],
        )
        with self.assertRaises(WarmupError):
            parse_cases("64")
        with self.assertRaises(WarmupError):
            parse_cases("64x0")

    def test_prompt_builder_targets_rendered_token_count(self):
        client = FakeTokenizerClient()
        content, count = PromptBuilder(client, "default").closest_content(
            64, family=1, variant=2
        )
        self.assertEqual(client.last_path, "/tokenize")
        self.assertEqual(count, 64)
        self.assertTrue(content.startswith("[startup-warmup-1-2]"))

    def test_second_round_rejects_large_ttft_regression(self):
        first = {"no-prefix:64x1": CaseResult(100.0, (64,))}
        validate_second_round(
            first, {"no-prefix:64x1": CaseResult(140.0, (64,))}, 1.5, 10.0, 1000.0
        )
        with self.assertRaises(WarmupError):
            validate_second_round(
                first, {"no-prefix:64x1": CaseResult(160.0, (64,))}, 1.5, 10.0, 1000.0
            )
        with self.assertRaises(WarmupError):
            validate_second_round(
                first, {"no-prefix:64x1": CaseResult(140.0, (64,))}, 1.5, 10.0, 120.0
            )

    def test_multi_request_warmup_uses_concurrent_serving_calls(self):
        class ConcurrentClient:
            def __init__(self, expected_parallel):
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0
                self.paths = []
                # A barrier is deterministic where a sleep is not: if the calls really do
                # overlap every thread reaches it and the test passes immediately; if they are
                # serialised the barrier times out and fails fast instead of relying on a 20ms
                # window holding on a loaded CI machine.
                self.barrier = threading.Barrier(expected_parallel, timeout=5)

            def post_json(self, path, payload):
                with self.lock:
                    self.paths.append(path)
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                self.barrier.wait()
                with self.lock:
                    self.active -= 1
                return {
                    "choices": [{}],
                    "aux_info": {"first_token_cost_time": 10.0, "input_len": 64},
                }

        client = ConcurrentClient(expected_parallel=2)
        warmup = ServingPathWarmup(client, "default", [WarmupCase(64, 2)], [])
        result = warmup._infer(["first", "second"], reuse_cache=False)
        self.assertEqual(result.input_lengths, (64, 64))
        self.assertEqual(client.paths, ["/v1/chat/completions", "/v1/chat/completions"])
        self.assertEqual(client.max_active, 2)

    def test_gate_publish_is_atomic_and_jit_snapshot_tracks_files(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            cache = root / "cache"
            cache.mkdir()
            artifact = cache / "kernel.so"
            artifact.write_text("binary")
            first_snapshot = snapshot_jit_artifacts([cache])
            self.assertEqual(len(first_snapshot), 1)
            time.sleep(0.001)
            artifact.write_text("recompiled")
            second_snapshot = snapshot_jit_artifacts([cache])
            self.assertEqual(len(second_snapshot - first_snapshot), 1)

            gate = root / "state" / "ready"
            publish_gate(gate, {"ok": True})
            self.assertEqual(gate.read_text(), '{"ok": true}\n')
            self.assertEqual(list(gate.parent.glob(".*.tmp")), [])

            phase = root / "state" / "phase"
            publish_phase(phase, "CANARY")
            self.assertEqual(phase.read_text(), "CANARY\n")

            events = root / "state" / "jit.jsonl"
            events.write_text('{"phase":"WARMUP"}\n\n')
            self.assertEqual(compile_event_count(events), 1)

    def test_triton_compile_monitor_records_the_active_phase(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            phase = root / "phase"
            events = root / "events.jsonl"
            phase.write_text("CANARY\n")
            env = {
                "RTP_LLM_TRITON_COMPILE_PHASE_FILE": str(phase),
                "RTP_LLM_TRITON_COMPILE_EVENT_FILE": str(events),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                monitor = TRITON_COMPILE_PATCH.TritonCompileMonitor()
                wrapped = monitor(lambda src, target=None, options=None: "compiled")
                self.assertEqual(wrapped(object()), "compiled")

            event = json.loads(events.read_text())
            self.assertEqual(event["phase"], "CANARY")
            self.assertEqual(event["kernel_name"], "unknown")

    def test_triton_compile_monitor_is_enabled_only_when_requested(self):
        with mock.patch.object(
            TRITON_COMPILE_PATCH, "enable_compile_monitor"
        ) as enable:
            with mock.patch.dict(
                os.environ, {"RTP_LLM_TRITON_COMPILE_MONITOR": "1"}, clear=False
            ):
                TRITON_COMPILE_PATCH.maybe_enable_compile_monitor()
            enable.assert_called_once_with()

        with mock.patch.object(
            TRITON_COMPILE_PATCH, "enable_compile_monitor"
        ) as enable:
            with mock.patch.dict(
                os.environ, {"RTP_LLM_TRITON_COMPILE_MONITOR": "0"}, clear=False
            ):
                TRITON_COMPILE_PATCH.maybe_enable_compile_monitor()
            enable.assert_not_called()


class HttpJsonClientTest(unittest.TestCase):
    """HttpJsonClient is this module's only production boundary; every other test replaces it
    with a fake, so its five failure branches are covered here against a patched urlopen."""

    def setUp(self):
        self.client = STARTUP_WARMUP.HttpJsonClient("http://localhost:8088", timeout_s=1.0)

    @staticmethod
    def _response(status, body):
        response = mock.MagicMock()
        response.status = status
        response.read.return_value = body
        response.__enter__ = lambda self_: self_
        response.__exit__ = lambda self_, *args: False
        return response

    def test_post_json_returns_parsed_object(self):
        with mock.patch.object(
            STARTUP_WARMUP.urllib.request, "urlopen", return_value=self._response(200, b'{"a": 1}')
        ):
            self.assertEqual(self.client.post_json("/tokenize", {}), {"a": 1})

    def test_post_json_raises_on_http_error(self):
        error = STARTUP_WARMUP.urllib.error.HTTPError(
            url="http://localhost:8088/tokenize", code=503, msg="busy", hdrs=None, fp=None
        )
        with mock.patch.object(STARTUP_WARMUP.urllib.request, "urlopen", side_effect=error):
            with self.assertRaisesRegex(WarmupError, "returned HTTP 503"):
                self.client.post_json("/tokenize", {})

    def test_post_json_raises_on_transport_error(self):
        with mock.patch.object(
            STARTUP_WARMUP.urllib.request,
            "urlopen",
            side_effect=STARTUP_WARMUP.urllib.error.URLError("refused"),
        ):
            with self.assertRaisesRegex(WarmupError, "request failed: URLError"):
                self.client.post_json("/tokenize", {})

    def test_post_json_raises_on_non_200_status(self):
        with mock.patch.object(
            STARTUP_WARMUP.urllib.request, "urlopen", return_value=self._response(204, b"{}")
        ):
            with self.assertRaisesRegex(WarmupError, "returned HTTP 204"):
                self.client.post_json("/tokenize", {})

    def test_post_json_raises_on_invalid_json(self):
        with mock.patch.object(
            STARTUP_WARMUP.urllib.request, "urlopen", return_value=self._response(200, b"not-json")
        ):
            with self.assertRaisesRegex(WarmupError, "returned invalid JSON"):
                self.client.post_json("/tokenize", {})

    def test_post_json_raises_on_non_object_response(self):
        with mock.patch.object(
            STARTUP_WARMUP.urllib.request, "urlopen", return_value=self._response(200, b"[1, 2]")
        ):
            with self.assertRaisesRegex(WarmupError, "non-object response"):
                self.client.post_json("/tokenize", {})

    def test_get_status_maps_errors_to_codes(self):
        error = STARTUP_WARMUP.urllib.error.HTTPError(
            url="http://localhost:8088/", code=503, msg="busy", hdrs=None, fp=None
        )
        with mock.patch.object(STARTUP_WARMUP.urllib.request, "urlopen", side_effect=error):
            self.assertEqual(self.client.get_status("/"), 503)
        with mock.patch.object(
            STARTUP_WARMUP.urllib.request, "urlopen", side_effect=OSError("down")
        ):
            self.assertEqual(self.client.get_status("/"), 0)


class MainEntrypointTest(unittest.TestCase):
    """main() decides whether an instance ever becomes healthy, so its env handling is pinned
    here: it publishes the gate only on success and leaves phase=FAILED behind otherwise."""

    def test_main_is_disabled_without_gate_file(self):
        self.assertEqual(STARTUP_WARMUP.main({}), 0)

    def test_main_rejects_batch_above_concurrency_limit_without_publishing_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = Path(tmp) / "gate"
            phase = Path(tmp) / "phase"
            env = {
                "RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE": str(gate),
                "RTP_LLM_TRITON_COMPILE_PHASE_FILE": str(phase),
                "RTP_LLM_STARTUP_WARMUP_CASES": "64x8",
                "RTP_LLM_STARTUP_WARMUP_PREFIX_CASES": "64x1",
                "CONCURRENCY_LIMIT": "4",
            }
            with mock.patch.object(
                STARTUP_WARMUP.ServingPathWarmup, "wait_until_backend_is_ready"
            ) as wait:
                self.assertEqual(STARTUP_WARMUP.main(env), 1)
                # Rejected on config alone: it must not wait for, or touch, the backend.
                wait.assert_not_called()
            self.assertFalse(gate.exists())
            self.assertIn("FAILED", phase.read_text())

    def test_main_leaves_failed_phase_when_backend_never_becomes_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = Path(tmp) / "gate"
            phase = Path(tmp) / "phase"
            env = {
                "RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE": str(gate),
                "RTP_LLM_TRITON_COMPILE_PHASE_FILE": str(phase),
                "RTP_LLM_STARTUP_WARMUP_CASES": "64x1",
                "RTP_LLM_STARTUP_WARMUP_PREFIX_CASES": "64x1",
            }
            with mock.patch.object(
                STARTUP_WARMUP.ServingPathWarmup,
                "wait_until_backend_is_ready",
                side_effect=WarmupError("backend not ready"),
            ):
                self.assertEqual(STARTUP_WARMUP.main(env), 1)
            # No gate means /health stays 503 -- the instance must not join traffic.
            self.assertFalse(gate.exists())
            self.assertIn("FAILED", phase.read_text())

    def test_readiness_probe_uses_a_route_the_gate_does_not_block(self):
        """Self-lock guard. The gate this script publishes makes /health, /status and
        cm2_status return 503, so the readiness probe must use a route that stays open --
        otherwise the script waits for a gate only it can publish. That coupling is implicit
        in frontend_app.py (check_startup_warmup_ready is wired into health_check and exempts
        /liveness; the / route never calls it), so pin both halves here."""
        probed = []

        class RecordingClient:
            def get_status(self, path, timeout_s=5.0):
                probed.append(path)
                return 200

        warmup = ServingPathWarmup(RecordingClient(), "default", [WarmupCase(64, 1)], [])
        warmup.wait_until_backend_is_ready(1.0)
        self.assertEqual(probed, ["/"])

        frontend_source = (
            Path(__file__).resolve().parents[2] / "frontend" / "frontend_app.py"
        ).read_text()
        root_route = frontend_source.index('@app.get("/")\n')
        next_route = frontend_source.index("@app.get", root_route + 1)
        self.assertNotIn(
            "check_startup_warmup_ready",
            frontend_source[root_route:next_route],
            "the / route must stay outside the startup gate or startup_warmup deadlocks",
        )


if __name__ == "__main__":
    unittest.main()
