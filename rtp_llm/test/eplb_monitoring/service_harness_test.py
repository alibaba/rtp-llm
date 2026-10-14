"""CPU control-flow tests for strict output identity and count acceptance."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.eplb_monitoring import service_smoke_test as harness


def response(text="first", length=32):
    return {
        "finished": True,
        "response": text,
        "output_ids": [[1] * length],
        "aux_info": {
            "output_len": length,
            "input_len": 178,
            "first_token_cost_time": 20.0,
            "wait_time": 1.0,
            "cost_time": 180.0,
        },
    }


class ServiceHarnessTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.case = harness.ExpertLoadServiceTest("test_engine")
        self.case.output = Path(folder.name)
        self.case.world = 2
        self.case.checkpoint = "unused"
        self.case.config = {"vocab_size": 1000}
        self.case.prompt = "same input"
        self.case._cycle = 1
        harness.ExpertLoadServiceTest.run_id = 0
        self.manager = SimpleNamespace(
            port=1234, server_pid=42, start_server=lambda *a: True, stop_server=Mock()
        )
        self.factory = self.stack.enter_context(
            patch.object(self.case, "_make_manager", return_value=self.manager)
        )
        self.stack.enter_context(patch.object(self.case, "_artifacts", return_value=[]))
        self.stop = self.manager.stop_server

    def _responses(self, bodies):
        replies = [Mock(json=Mock(return_value=body)) for body in bodies]
        return self.stack.enter_context(
            patch.object(harness.requests, "post", side_effect=replies)
        )

    def _run(self, enabled=False, warmups=2, measures=2):
        return self.case._run(enabled, True, True, 100, measures, False, warmups)

    def test_default_strict_rejects_changed_output(self):
        self._responses([response(), response("changed")])
        with self.assertRaises(AssertionError):
            self._run()
        self.stop.assert_called_once_with()

    def test_retains_token_ids_and_request_journal(self):
        post = self._responses([response() for _ in range(4)])
        result = self._run()
        self.assertEqual(
            [s["output_text"] for s in result["samples"]], ["first", "first"]
        )
        rows = [
            json.loads(row)
            for row in next(self.case.output.glob("*_requests.jsonl"))
            .read_text()
            .splitlines()
        ]
        self.assertEqual(
            [r["phase"] for r in rows], ["warmup", "warmup", "measure", "measure"]
        )
        self.assertEqual([r["request_index"] for r in rows], [0, 1, 2, 3])
        self.assertTrue(all(r["cycle"] == 1 and r["run_id"] == 1 for r in rows))
        self.assertEqual(rows[-1]["response"]["output_ids"], [[1] * 32])
        for call in post.call_args_list:
            config = call.kwargs["json"]["generate_config"]
            self.assertTrue(config["return_output_ids"])
            self.assertNotIn("ebnf", config)
            self.assertNotIn("regex", config)
        self.assertEqual(
            self.factory.call_args.kwargs["env_args"]["RTP_LLM_DROP_BROAD_SYNC"], "0"
        )

    def test_rejects_length_error_and_journals_response(self):
        self._responses([response(), response(length=31)])
        with self.assertRaises(AssertionError):
            self._run()
        rows = next(self.case.output.glob("*_requests.jsonl")).read_text().splitlines()
        self.assertEqual(json.loads(rows[-1])["response"]["aux_info"]["output_len"], 31)

    def test_rejects_count_failure(self):
        self._responses([response() for _ in range(4)])
        with self.assertRaisesRegex(AssertionError, "produced no windows"):
            self._run(enabled=True)
        result = json.loads(next(self.case.output.glob("*_result.json")).read_text())
        self.assertEqual(len(result["samples"]), 2)
        self.stop.assert_called_once_with()

    def test_rejects_changed_input_length(self):
        bad = response()
        bad["aux_info"]["input_len"] = 177
        self._responses([response(), bad])
        with self.assertRaises(AssertionError):
            self._run()

    def test_does_not_swallow_http_failure(self):
        reply = Mock()
        reply.raise_for_status.side_effect = harness.requests.HTTPError("HTTP 500")
        with patch.object(harness.requests, "post", return_value=reply):
            with self.assertRaises(harness.requests.HTTPError):
                self._run()
        self.stop.assert_called_once_with()

    def test_rejects_nonfinite_timing_and_invalid_ids(self):
        for field in ("timing", "ids"):
            with self.subTest(field=field):
                bad = response()
                if field == "timing":
                    bad["aux_info"]["cost_time"] = float("nan")
                else:
                    bad["output_ids"][0][0] = True
                with patch.object(
                    harness.requests,
                    "post",
                    return_value=Mock(json=Mock(return_value=bad)),
                ):
                    with self.assertRaises(AssertionError):
                        self._run(warmups=1, measures=1)

    def test_performance_checks_cross_run_outputs(self):
        with patch.dict(
            harness.os.environ, {"EXPERT_LOAD_TEST_MODE": "performance"}
        ), patch.object(
            self.case,
            "_run",
            side_effect=[
                {"output": "a", "output_ids": [1]},
                {"output": "b", "output_ids": [1]},
            ],
        ):
            with self.assertRaises(AssertionError):
                self.case.test_engine()

    def test_same_text_with_different_ids_fails_within_run(self):
        changed = response()
        changed["output_ids"][0][0] = 2
        self._responses([response(), response(), changed])
        with self.assertRaises(AssertionError):
            self._run()
        self.stop.assert_called_once_with()

    def test_same_text_with_different_ids_fails_across_runs(self):
        for mode in ("functional", "performance"):
            with self.subTest(mode=mode), patch.dict(
                harness.os.environ,
                {"EXPERT_LOAD_TEST_MODE": mode, "EXPERT_LOAD_TEST_CASES": "1:1"},
            ), patch.object(
                self.case,
                "_run",
                side_effect=[
                    {"output": "same", "output_ids": [1]},
                    {"output": "same", "output_ids": [2]},
                ],
            ):
                with self.assertRaises(AssertionError):
                    self.case.test_engine()

    def test_performance_enforces_regression_with_identical_outputs(self):
        for regression in (False, True):
            with self.subTest(regression=regression):
                runs = []
                for i, mode in enumerate(("off", "on", "on", "off") * 3):
                    samples = [
                        {
                            "prefill_ms": 22.0 if regression and mode == "on" else 20.0,
                            "decode_ms_per_token": 5.0,
                            "throughput_tokens_per_s": 160.0,
                            "output_len": 32,
                            "wall_s": 0.2,
                            "output_text": "same",
                            "output_ids": [1] * 32,
                        }
                        for _ in range(64)
                    ]
                    runs.append(
                        {
                            "mode": mode,
                            "run_id": i + 1,
                            "window_count": 21 if mode == "on" else 0,
                            "samples": samples,
                            "output": "same",
                            "output_ids": [1] * 32,
                        }
                    )
                with patch.dict(
                    harness.os.environ,
                    {
                        "EXPERT_LOAD_TEST_MODE": "performance",
                        "EXPERT_LOAD_TEST_CYCLES": "3",
                        "EXPERT_LOAD_TEST_WARMUPS": "8",
                        "EXPERT_LOAD_TEST_SAMPLES": "64",
                        "EXPERT_LOAD_TEST_WINDOW": "100",
                    },
                ), patch.object(self.case, "_run", side_effect=runs) as run, patch(
                    "builtins.print"
                ):
                    if regression:
                        with self.assertRaises(AssertionError):
                            self.case.test_engine()
                    else:
                        self.case.test_engine()
                self.assertEqual(run.call_count, 12)
                raw = json.loads((self.case.output / "abba_raw.json").read_text())
                report = json.loads(
                    (self.case.output / "abba_analysis.json").read_text()
                )
                self.assertEqual(sum(len(r["samples"]) for r in raw), 768)
                self.assertEqual(
                    report["conclusion"], "regression" if regression else "ok"
                )


if __name__ == "__main__":
    unittest.main()
