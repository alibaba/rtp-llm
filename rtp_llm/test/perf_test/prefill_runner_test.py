"""Core performance behavior contracts."""

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.perf_test.errors import CaseExecutionError
from rtp_llm.test.perf_test.modes import ModeExecutionContext, PrefillAdapter
from rtp_llm.test.perf_test.multi_node.prefill_runtime import (
    ExternalPrefillSession,
    build_external_prefill_plan,
)
from rtp_llm.test.perf_test.prefill.prompts import PrefixPromptFactory
from rtp_llm.test.perf_test.prefill.runner import PrefillRunner
from rtp_llm.test.perf_test.run_directory import initialize_run_directory
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore
from rtp_llm.test.perf_test.test_support import (
    prefill_external_aux,
    prefill_external_CASE_CACHE_LEN,
    prefill_external_CASE_INPUT_LEN,
    prefill_external_COMMIT_TAIL_TOKENS,
    prefill_external_FakeModelServer,
    prefill_external_Tokenizer,
    prefill_external_TOPOLOGY,
    prefill_external_write_prefill_config,
    prefill_resume_config,
    prefill_resume_plan,
)


def prefill_external_run_through_session(plan, session, run_root):
    """Drive the prefill runner factory exactly like the unified executor."""
    (run_root / "results").mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    adapter = PrefillAdapter(
        lambda context: PrefillRunner(
            context.session.client,
            prefill_external_Tokenizer(),
            commit_tail_tokens=prefill_external_COMMIT_TAIL_TOKENS,
            max_batch_tokens=plan.resolved_config["engine"]["max_batch_tokens"],
            monotonic=iter([0.0, 0.2]).__next__,
        )
    )
    return adapter(
        ModeExecutionContext(plan=plan, session=session, store=RunStore(run_root))
    )


class prefill_runner_Tokenizer:

    def encode(self, text):
        return text.split()


class ShortColdPromptTest(unittest.TestCase):
    def test_short_cold_runs_have_distinct_exact_length_inputs(self):
        tokenizer = SimpleNamespace(encode=list)
        prompts = PrefixPromptFactory(tokenizer).build_case_prompts(
            "prefill-58bcef9044dc7047168737e3_g0_r0", 7, 0, 3
        )
        self.assertEqual([7, 7, 7], [len(ids) for ids in prompts.run_ids])
        self.assertEqual(3, len({tuple(ids) for ids in prompts.run_ids}))


class prefill_runner_Client:

    def __init__(self, responses, batch_size=1):
        self.responses = {}
        seed_index = measure_index = 0
        for response in responses:
            if response["input_len"] <= 12:
                key = ("seed", None, seed_index)
                seed_index += 1
            else:
                key = (
                    "measure",
                    measure_index // batch_size,
                    measure_index % batch_size,
                )
                measure_index += 1
            self.responses[key] = response
        self.requests = []
        self.scheduler_calls = []

    def configure_scheduler(self, batch_size, mode="prefill"):
        assert mode == "prefill"
        self.scheduler_calls.append(batch_size)

    def request(self, **request):
        self.requests.append(request)
        return self.responses[
            request["phase"], request["run_index"], request["request_index"]
        ]


def prefill_runner_response(
    input_len, reuse_len, latency=10.0, success=True, output_len=1
):
    return {
        "success": success,
        "input_len": input_len,
        "output_len": output_len,
        "reuse_len": reuse_len,
        "latency_ms": latency,
    }


def prefill_runner_case(cache_len=8, batch_size=1, measure_runs=2):
    return {
        "case_id": "prefill-case",
        "input_len": 16,
        "cache_len": cache_len,
        "compute_len": 16 - cache_len,
        "batch_size": batch_size,
        "measure_runs": measure_runs,
        "seed": 7,
        "prefix_policy": "shared_seed" if batch_size == 1 else "independent",
    }


def prefill_runner_grouped_case():
    return dict(
        prefill_runner_case(batch_size=3, measure_runs=1),
        input_len=24,
        cache_len=8,
        compute_len=16,
        prefix_policy="independent",
        measurement_contract="concurrent_grouped_batch_v1",
        request_groups=[
            {"count": 1, "input_len": 16, "cache_len": 8},
            {"count": 2, "input_len": 24, "cache_len": 0},
        ],
    )


class SeedBatchClient:
    """Require complete concurrent scheduler groups without a GPU server."""

    def __init__(self, cache_lengths, *, failed_member=None, raised_member=None):
        self.cache_lengths = cache_lengths
        self.failed_member = failed_member
        self.raised_member = raised_member
        self.batches = []
        self.completed_seeds = set()
        self.lock = threading.Lock()

    def configure_scheduler(self, batch_size, mode="prefill"):
        assert mode == "prefill"
        if self.batches:
            assert self.batches[-1]["completed"] == len(self.batches[-1]["requests"])
        self.batches.append(
            {
                "barrier": threading.Barrier(batch_size, timeout=2),
                "requests": [],
                "completed": 0,
            }
        )

    def request(self, **request):
        with self.lock:
            batch = self.batches[-1]
            batch["requests"].append(request)
        try:
            batch["barrier"].wait()
            member = request["request_index"]
            seed = request["phase"] == "seed"
            if not seed:
                assert self.completed_seeds == {
                    index for index, length in enumerate(self.cache_lengths) if length
                }
            if seed and member == self.raised_member:
                raise RuntimeError("seed transport failed")
            return prefill_runner_response(
                len(request["input_ids"]),
                0 if seed else self.cache_lengths[member],
                success=not (seed and member == self.failed_member),
            )
        finally:
            with self.lock:
                batch["completed"] += 1
                if request["phase"] == "seed":
                    self.completed_seeds.add(request["request_index"])


class SeedBatchTest(unittest.TestCase):
    def run_seeds(
        self,
        cache_lengths,
        *,
        max_batch_tokens=4096,
        workspace_tokens=None,
        cache_block_size=4,
        client=None,
    ):
        client = client or SeedBatchClient(cache_lengths)
        case = dict(
            prefill_runner_case(batch_size=len(cache_lengths), measure_runs=1),
            input_len=32,
            cache_len=max(cache_lengths),
            prefix_policy="independent",
            request_groups=[
                {"count": 1, "input_len": 32, "cache_len": length}
                for length in cache_lengths
            ],
        )
        result = PrefillRunner(
            client,
            prefill_runner_Tokenizer(),
            commit_tail_tokens=4,
            cache_block_size=cache_block_size,
            max_batch_tokens=max_batch_tokens,
            workspace_tokens=workspace_tokens,
        ).run_case(case)
        return client, result

    def test_seeds_overlap_in_batches_of_eight_with_short_tail(self):
        client, result = self.run_seeds([8] * 10)
        self.assertEqual("ok", result["status"])
        self.assertEqual([8, 2, 10], [len(b["requests"]) for b in client.batches])
        self.assertEqual(10, len(result["metrics"]["measurements"]))

    def test_token_budget_includes_output_and_skips_cold_members(self):
        for budget, expected in (
            (30, [[0, 2], [3, 4]]),
            (29, [[0], [2, 3], [4]]),
        ):
            with self.subTest(budget=budget):
                client, result = self.run_seeds(
                    [8, 0, 12, 4, 8], max_batch_tokens=budget
                )
                self.assertEqual("ok", result["status"])
                self.assertEqual(
                    expected,
                    [
                        sorted(r["request_index"] for r in b["requests"])
                        for b in client.batches[:-1]
                    ],
                )

    def test_workspace_budget_counts_block_padding(self):
        client, result = self.run_seeds(
            [8, 0, 12, 4, 8], workspace_tokens=32, cache_block_size=8
        )
        self.assertEqual("ok", result["status"])
        self.assertEqual(
            [[0], [2], [3, 4]],
            [
                sorted(r["request_index"] for r in b["requests"])
                for b in client.batches[:-1]
            ],
        )

    def test_oversized_seed_fails_before_any_dispatch(self):
        for limits in ({"max_batch_tokens": 24}, {"workspace_tokens": 27}):
            with self.subTest(limits=limits):
                client, result = self.run_seeds([8, 20], **limits)
                self.assertEqual("request_failed", result["status"])
                self.assertIsNone(result["metrics"])
                self.assertEqual([], client.batches)

    def test_failed_seed_batch_drains_peers_and_stops(self):
        for failure in ("failed_member", "raised_member"):
            with self.subTest(failure=failure):
                client = SeedBatchClient([8] * 10, **{failure: 0})
                _, result = self.run_seeds([8] * 10, client=client)
                self.assertEqual("request_failed", result["status"])
                self.assertIsNone(result["metrics"])
                self.assertEqual(1, len(client.batches))
                self.assertEqual(set(range(8)), client.completed_seeds)

    def test_submission_failure_aborts_before_dispatch(self):
        submitted = []

        class Executor(ThreadPoolExecutor):
            def submit(self, fn, *args, **kwargs):
                if submitted:
                    raise RuntimeError("submission failed")
                future = super().submit(fn, *args, **kwargs)
                submitted.append(future)
                return future

        with patch(
            "rtp_llm.test.perf_test.prefill.runner.ThreadPoolExecutor", Executor
        ):
            client, result = self.run_seeds([8] * 4)
        self.assertEqual("request_failed", result["status"])
        self.assertEqual([], client.batches[0]["requests"])
        self.assertEqual(1, len(submitted))
        self.assertTrue(submitted[0].done())

    def test_scheduler_failure_stops_before_seed_dispatch(self):
        class Client(SeedBatchClient):
            def configure_scheduler(self, batch_size, mode="prefill"):
                raise RuntimeError("scheduler unavailable")

        client = Client([8] * 4)
        _, result = self.run_seeds([8] * 4, client=client)
        self.assertEqual("request_failed", result["status"])
        self.assertEqual([], client.batches)


class PrefillRunnerTest(unittest.TestCase):

    def test_batch_four_measure_requests_overlap(self):
        barrier = threading.Barrier(4, timeout=2)

        class Client(prefill_runner_Client):

            def request(self, **request):
                if request["phase"] == "measure":
                    barrier.wait()
                return super().request(**request)

        client = Client(
            [prefill_runner_response(12, 0)] * 4 + [prefill_runner_response(16, 8)] * 4,
            4,
        )
        result = PrefillRunner(
            client,
            prefill_runner_Tokenizer(),
            commit_tail_tokens=4,
            max_batch_tokens=4096,
        ).run_case(prefill_runner_case(batch_size=4, measure_runs=1))
        self.assertEqual("ok", result["status"])
        self.assertEqual([4, 4], client.scheduler_calls)
        self.assertEqual(["seed"] * 4, [r["phase"] for r in client.requests[:4]])

    def test_seed_failure_stops_before_measurement(self):
        client = prefill_runner_Client(
            [prefill_runner_response(12, 0, success=False)] * 4, 4
        )
        result = PrefillRunner(
            client,
            prefill_runner_Tokenizer(),
            commit_tail_tokens=4,
            max_batch_tokens=4096,
        ).run_case(prefill_runner_case(batch_size=4, measure_runs=1))
        self.assertEqual("request_failed", result["status"])
        self.assertEqual([4], client.scheduler_calls)
        self.assertEqual(["seed"] * 4, [r["phase"] for r in client.requests])

    def test_round_setup_failure_aborts_and_drains_submitted_workers(self):
        for failure in ("submit", "clock"):
            with self.subTest(failure=failure):
                release = threading.Event()
                submitted = []
                results = []
                errors = []

                class Executor(ThreadPoolExecutor):

                    def submit(self, fn, *args, **kwargs):
                        if failure == "submit" and submitted:
                            raise RuntimeError("submission failed")
                        future = super().submit(fn, *args, **kwargs)
                        submitted.append(future)
                        return future

                def monotonic():
                    raise RuntimeError("clock failed")

                client = prefill_runner_Client([prefill_runner_response(16, 0)] * 4, 4)

                def run():
                    try:
                        results.append(
                            PrefillRunner(
                                client,
                                prefill_runner_Tokenizer(),
                                commit_tail_tokens=4,
                                max_batch_tokens=4096,
                                monotonic=monotonic,
                            ).run_case(
                                prefill_runner_case(
                                    cache_len=0, batch_size=4, measure_runs=1
                                )
                            )
                        )
                    except Exception as error:
                        errors.append(error)

                with patch(
                    "rtp_llm.test.perf_test.prefill.runner.ThreadPoolExecutor", Executor
                ), patch(
                    "rtp_llm.test.perf_test.prefill.runner.threading",
                    SimpleNamespace(Event=lambda: release),
                ):
                    worker = threading.Thread(target=run)
                    worker.start()
                    try:
                        worker.join(timeout=2)
                        self.assertFalse(worker.is_alive(), "round setup deadlocked")
                    finally:
                        release.set()
                        worker.join(timeout=5)
                self.assertEqual([], errors)
                self.assertEqual(1 if failure == "submit" else 4, len(submitted))
                self.assertTrue(all((future.done() for future in submitted)))
                self.assertEqual([], client.requests)
                self.assertEqual(
                    "request_failed" if failure == "submit" else "invalid_timing",
                    results[0]["status"],
                )
                self.assertIsNone(results[0]["metrics"])

    def test_throughput_is_mean_of_round_rates(self):
        times = iter([0.0, 0.2, 1.0, 1.5])
        client = prefill_runner_Client(
            [prefill_runner_response(12, 0)] + [prefill_runner_response(16, 8)] * 2
        )
        result = PrefillRunner(
            client,
            prefill_runner_Tokenizer(),
            commit_tail_tokens=4,
            max_batch_tokens=4096,
            monotonic=lambda: next(times),
        ).run_case(prefill_runner_case())
        self.assertEqual(28.0, result["metrics"]["throughput_tokens_per_second"])

    def test_valid_cached_case_seeds_then_measures_exact_reuse(self):
        client = prefill_runner_Client(
            [
                prefill_runner_response(12, 0),
                dict(
                    prefill_runner_response(16, 8, latency=10.0),
                    ttft_ms=10.0,
                    wait_time_ms=1.0,
                ),
                dict(
                    prefill_runner_response(16, 8, latency=14.0),
                    ttft_ms=14.0,
                    wait_time_ms=3.0,
                ),
            ]
        )
        runner = PrefillRunner(
            client,
            prefill_runner_Tokenizer(),
            commit_tail_tokens=4,
            max_batch_tokens=4096,
        )
        result = runner.run_case(prefill_runner_case())
        self.assertEqual("ok", result["status"])
        self.assertEqual(3, len(client.requests))
        self.assertEqual("seed", client.requests[0]["phase"])
        self.assertEqual(
            ["measure", "measure"], [r["phase"] for r in client.requests[1:]]
        )
        self.assertEqual(12.0, result["metrics"]["latency_ms"])
        self.assertEqual(12.0, result["metrics"]["ttft_ms"])
        self.assertEqual(2.0, result["metrics"]["wait_time_ms"])
        self.assertEqual(
            [1.0, 3.0], [m["wait_time_ms"] for m in result["metrics"]["measurements"]]
        )
        self.assertEqual(8, result["metrics"]["observed_reuse_length"])
        self.assertEqual(2, len(result["metrics"]["measurements"]))

    def test_cold_case_skips_seed_and_requires_zero_reuse(self):
        client = prefill_runner_Client(
            [prefill_runner_response(16, 0), prefill_runner_response(16, 0)]
        )
        result = PrefillRunner(
            client,
            prefill_runner_Tokenizer(),
            commit_tail_tokens=4,
            max_batch_tokens=4096,
        ).run_case(prefill_runner_case(cache_len=0))
        self.assertEqual("ok", result["status"])
        self.assertEqual(["measure", "measure"], [r["phase"] for r in client.requests])


class GroupedRunnerTest(unittest.TestCase):

    def run_group(self, bad_reuse=False):
        barrier = threading.Barrier(3, timeout=2)

        class Client(prefill_runner_Client):

            def request(self, **request):
                self.requests.append(request)
                if request["phase"] == "measure":
                    barrier.wait()
                    return prefill_runner_response(
                        len(request["input_ids"]),
                        8 if bad_reuse or request["request_index"] == 0 else 0,
                        latency=10 * (request["request_index"] + 1),
                    )
                return prefill_runner_response(len(request["input_ids"]), 0)

        client = Client([])
        times = iter([0, 0.2])
        result = PrefillRunner(
            client,
            prefill_runner_Tokenizer(),
            commit_tail_tokens=4,
            max_batch_tokens=4096,
            monotonic=lambda: next(times),
        ).run_case(prefill_runner_grouped_case())
        return (client, result)

    def test_grouped_members_use_own_geometry_and_round_wall_time(self):
        (client, result) = self.run_group()
        self.assertEqual("ok", result["status"])
        self.assertEqual([1, 3], client.scheduler_calls)
        self.assertEqual(
            [12], [len(r["input_ids"]) for r in client.requests if r["phase"] == "seed"]
        )
        metrics = result["metrics"]
        self.assertEqual(
            (64, 8, 56),
            (
                metrics["input_length"],
                metrics["expected_reuse_length"],
                metrics["compute_length"],
            ),
        )
        self.assertEqual(280, metrics["throughput_tokens_per_second"])
        self.assertEqual(30, metrics["latency_ms"])
        self.assertEqual(
            [16, 24, 24], [m["input_length"] for m in metrics["measurements"]]
        )
        self.assertEqual(
            [8, 0, 0], [m["expected_reuse_length"] for m in metrics["measurements"]]
        )
        self.assertTrue(
            all((m["round_elapsed_ms"] == 200 for m in metrics["measurements"]))
        )

    def test_grouped_reuse_is_validated_per_member(self):
        (_, result) = self.run_group(bad_reuse=True)
        self.assertEqual("invalid_reuse", result["status"])


class ProbeTest(unittest.TestCase):

    def test_probe_checks_seed_and_hit(self):
        from rtp_llm.test.perf_test.prefill import probe

        for broken_phase in (None, "seed", "measure"):
            requests = []

            class Client:

                def configure_scheduler(self, batch_size, mode):
                    self.batch = batch_size

                def request(self, **request):
                    requests.append(request)
                    reuse = 0 if request["phase"] == "seed" else 8
                    if request["phase"] == broken_phase:
                        reuse += 1
                    return prefill_runner_response(len(request["input_ids"]), reuse)

            client = Client()
            if broken_phase:
                with self.assertRaisesRegex(ValueError, "reuse"):
                    probe.probe_cache_block(client, prefill_runner_Tokenizer(), 8, 4)
            else:
                result = probe.probe_cache_block(
                    client, prefill_runner_Tokenizer(), 8, 4
                )
                self.assertEqual("ok", result["status"])
                self.assertEqual([12, 16], [len(r["input_ids"]) for r in requests])
                self.assertEqual(
                    requests[0]["input_ids"][:8], requests[1]["input_ids"][:8]
                )
                self.assertNotEqual(
                    requests[0]["input_ids"][8], requests[1]["input_ids"][8]
                )
            self.assertEqual(1 if broken_phase == "seed" else 2, len(requests))

    def test_probe_failure_persisted_before_any_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            plan = prefill_resume_plan(root)
            initialize_run_directory(
                root,
                source_config=prefill_resume_config(),
                plan=plan,
                run_id="probe",
                code_version="test",
            )
            calls = []

            class Runner:

                def run_case(self, case):
                    calls.append(case)

            def fail(_context):
                raise ValueError("wrong block reuse")

            adapter = PrefillAdapter(lambda context: Runner(), probe=fail)
            with self.assertRaisesRegex(ValueError, "wrong block reuse"):
                adapter(
                    ModeExecutionContext(
                        plan=plan, session=object(), store=RunStore(root)
                    )
                )
            self.assertEqual([], calls)
            self.assertEqual({}, RunStore(root).read_committed_cases())
            import json

            artifact = json.loads(
                (root / "checkpoints" / "prefill_cache_probe.json").read_text()
            )
            self.assertEqual("failed", artifact["status"])


class ExternalPrefillSessionTest(unittest.TestCase):

    def test_prefill_runner_completes_through_the_external_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = prefill_external_write_prefill_config(root)
            (plan, _source) = build_external_prefill_plan(
                path, str(root / "run"), prefill_external_TOPOLOGY
            )
            with prefill_external_FakeModelServer(
                [
                    prefill_external_aux(prefill_external_CASE_INPUT_LEN, 0, 5.0),
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
            self.assertEqual(4, len(server.requests))
            (configure_seed, seed, configure_measure, measure) = server.requests
            for update in (configure_seed, configure_measure):
                self.assertEqual("/update_scheduler_info", update["path"])
                self.assertEqual({"batch_size": 1, "mode": "prefill"}, update["body"])
            self.assertEqual("/", seed["path"])
            self.assertEqual("none", seed["headers"]["X-DashScope-LoggingConsent"])
            self.assertEqual(1, seed["body"]["generate_config"]["max_new_tokens"])
            self.assertNotEqual(seed["body"]["prompt"], measure["body"]["prompt"])

    def test_prefill_runner_flags_invalid_reuse_through_the_external_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = prefill_external_write_prefill_config(root)
            (plan, _source) = build_external_prefill_plan(
                path, str(root / "run"), prefill_external_TOPOLOGY
            )
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
