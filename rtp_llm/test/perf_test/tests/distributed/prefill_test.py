"""Performance framework behavior and failure regressions."""

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.entry import plan_from_source
from rtp_llm.test.perf_test.errors import CaseExecutionError
from rtp_llm.test.perf_test.execution.distributed import prefill as prefill_runtime
from rtp_llm.test.perf_test.execution.distributed.prefill import ExternalPrefillSession
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.prefill import (
    prefill_external_aux,
    prefill_external_CASE_CACHE_LEN,
    prefill_external_CASE_INPUT_LEN,
    prefill_external_FakeModelServer,
    prefill_external_run_through_session,
    prefill_external_Tokenizer,
    prefill_external_TOPOLOGY,
    prefill_external_write_prefill_config,
)


def external_plan(root):
    path = Path(prefill_external_write_prefill_config(root))
    return plan_from_source(
        json.loads(path.read_text()),
        source_dir=path.parent,
        result_directory=str(root / "run"),
    )


class DistributedRuntimeTest(unittest.TestCase):

    def test_external_prefill_uses_same_deadline_for_http_and_dashsc(self):
        from rtp_llm.test.perf_test.execution.runtime import HttpPrefillClient

        for options, seconds in (({}, 7200), ({"timeout_seconds": 1234}, 1234)):
            with self.subTest(options=options):
                post = Mock(
                    return_value=SimpleNamespace(
                        status_code=200, json=lambda: {"aux_info": {}}
                    )
                )

                def http(endpoint, **kwargs):
                    return HttpPrefillClient(endpoint, post=post, **kwargs)

                with patch.object(
                    prefill_runtime, "HttpPrefillClient", side_effect=http
                ), patch.object(
                    importlib.import_module(
                        "rtp_llm.test.perf_test.prefill.dashsc_transport"
                    ),
                    "DashScPrefillClient",
                ) as dashsc:
                    session = prefill_runtime.ExternalPrefillSession(
                        "http://localhost:18000",
                        topology={},
                        transport="http_text",
                        **options
                    )
                    session.client.request(
                        phase="probe",
                        case_id="probe",
                        request_index=0,
                        run_index=0,
                        prompt="hello",
                        input_ids=[1],
                        output_length=1,
                    )
                    self.assertEqual(seconds, post.call_args.kwargs["timeout"])
                    prefill_runtime.ExternalPrefillSession(
                        "http://localhost:18000",
                        topology={},
                        transport="dashsc_input_ids",
                        **options
                    )
                    self.assertEqual(
                        seconds, dashsc.call_args.kwargs["timeout_seconds"]
                    )

    def test_external_prefill_master_completes_against_http_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (plan, source) = external_plan(root)
            with prefill_external_FakeModelServer(
                [
                    prefill_external_aux(16, 0, 5.0),
                    prefill_external_aux(16, 8, 5.0),
                    prefill_external_aux(prefill_external_CASE_INPUT_LEN, 0, 5.0),
                    prefill_external_aux(
                        prefill_external_CASE_INPUT_LEN,
                        prefill_external_CASE_CACHE_LEN,
                        9000.0,
                    ),
                    prefill_external_aux(
                        prefill_external_CASE_INPUT_LEN,
                        prefill_external_CASE_CACHE_LEN,
                        12.5,
                    ),
                ]
            ) as server:
                with patch(
                    "rtp_llm.test.perf_test.test_util._load_tokenizer",
                    return_value=prefill_external_Tokenizer(),
                ):
                    result = prefill_runtime.execute_master_prefill(
                        plan,
                        source_config=source,
                        endpoint=server.endpoint,
                        tokenizer_path="/m",
                        server_log_path=None,
                        profiler_directory=root,
                        run_context={"repository_revisions": {"public": "a" * 40}},
                    )
            manifest = json.loads((root / "run/manifest.json").read_text())
            self.assertEqual("a" * 40, manifest["code_version"])
            self.assertEqual("completed", result.state.value)
            self.assertEqual(
                "completed",
                json.loads((root / "run/results/result.json").read_text())["status"],
            )
            self.assertTrue((root / "run/results/summary.json").is_file())


class ExternalPrefillSessionTest(unittest.TestCase):

    def test_prefill_runner_completes_through_the_external_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (plan, _source) = external_plan(root)
            with prefill_external_FakeModelServer(
                [
                    prefill_external_aux(prefill_external_CASE_INPUT_LEN, 0, 5.0),
                    prefill_external_aux(
                        prefill_external_CASE_INPUT_LEN,
                        prefill_external_CASE_CACHE_LEN,
                        9000.0,
                    ),
                    prefill_external_aux(
                        prefill_external_CASE_INPUT_LEN,
                        prefill_external_CASE_CACHE_LEN,
                        12.5,
                    ),
                ]
            ) as server:
                with ExternalPrefillSession(
                    server.endpoint,
                    topology=prefill_external_TOPOLOGY,
                    transport="http_text",
                ) as session:
                    outcome = prefill_external_run_through_session(
                        plan, session, root / "run"
                    )
            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertIsNone(outcome.failure)
            self.assertEqual(1, len(outcome.case_results))
            result = outcome.case_results[0]
            for field in ("case_id", "identity", "status", "runs", "errors", "metrics"):
                self.assertIn(field, result)
            self.assertEqual("ok", result["status"])
            self.assertEqual(
                prefill_external_CASE_INPUT_LEN, result["metrics"]["input_length"]
            )
            self.assertEqual(
                prefill_external_CASE_CACHE_LEN,
                result["metrics"]["observed_reuse_length"],
            )
            self.assertEqual(12.5, result["metrics"]["latency_ms"])
            self.assertEqual(40.0, result["metrics"]["throughput_tokens_per_second"])
            persisted = json.loads(
                (root / "run" / "results" / "cases.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual("ok", persisted["status"])
            self.assertEqual(5, len(server.requests))
            (configure_seed, seed, configure_measure, warmup, measure) = server.requests
            for update in (configure_seed, configure_measure):
                self.assertEqual("/update_scheduler_info", update["path"])
                self.assertEqual({"batch_size": 1, "mode": "prefill"}, update["body"])
            self.assertEqual("/", seed["path"])
            self.assertEqual("none", seed["headers"]["X-DashScope-LoggingConsent"])
            self.assertEqual(1, seed["body"]["generate_config"]["max_new_tokens"])
            self.assertNotEqual(seed["body"]["prompt"], measure["body"]["prompt"])
            self.assertNotEqual(warmup["body"]["prompt"], measure["body"]["prompt"])

    def test_prefill_runner_flags_invalid_reuse_through_the_external_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (plan, _source) = external_plan(root)
            with prefill_external_FakeModelServer(
                [
                    prefill_external_aux(prefill_external_CASE_INPUT_LEN, 0, 5.0),
                    prefill_external_aux(prefill_external_CASE_INPUT_LEN, 0, 12.5),
                ]
            ) as server:
                session = ExternalPrefillSession(
                    server.endpoint,
                    topology=prefill_external_TOPOLOGY,
                    transport="http_text",
                )
                outcome = prefill_external_run_through_session(
                    plan, session, root / "run"
                )
                session.close()
            self.assertEqual(RunState.PARTIAL, outcome.state)
            self.assertIsInstance(outcome.failure, CaseExecutionError)
            result = outcome.case_results[0]
            self.assertEqual("invalid_reuse", result["status"])
            self.assertIsNone(result["metrics"])
            self.assertEqual("invalid_reuse", result["errors"][0]["code"])
            self.assertIn("expected reuse 16", result["errors"][0]["message"])
            persisted = json.loads(
                (root / "run" / "results" / "cases.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual("invalid_reuse", persisted["status"])


if __name__ == "__main__":
    unittest.main()
