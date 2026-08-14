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
            def __init__(self):
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0
                self.paths = []

            def post_json(self, path, payload):
                with self.lock:
                    self.paths.append(path)
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.02)
                with self.lock:
                    self.active -= 1
                return {
                    "choices": [{}],
                    "aux_info": {"first_token_cost_time": 10.0, "input_len": 64},
                }

        client = ConcurrentClient()
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


if __name__ == "__main__":
    unittest.main()
