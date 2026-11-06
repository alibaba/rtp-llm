"""Core performance behavior contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.errors import PlanningError, ResourceError
from rtp_llm.test.perf_test.modes import ModeExecutionContext, PrefillAdapter
from rtp_llm.test.perf_test.plan import build_execution_plan, thaw
from rtp_llm.test.perf_test.prefill.case_store import (
    append_prefill_case,
    read_prefill_cases,
)
from rtp_llm.test.perf_test.prefill.checkpoint import PrefillResume, resume_fingerprints
from rtp_llm.test.perf_test.run_directory import (
    RunDirectoryLock,
    initialize_run_directory,
    inspect_run_directory,
)
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.storage import RunStore, atomic_write_json
from rtp_llm.test.perf_test.test_support import (
    fixture_engine,
    fixture_execution,
    fixture_model,
    fixture_workload,
    prefill_resume_config,
    prefill_resume_plan,
)


def run_directory_source_config():
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="test-model", type="registered_model", checkpoint="/models/test"
        ),
        "engine": fixture_engine(),
        "workload": fixture_workload(),
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }


def run_directory_plan(result_directory):
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(run_directory_source_config()),
        cases=[{"input_len": 512, "cache_len": 0, "batch_size": 1}],
        result_directory=str(result_directory),
    )


def run_directory_tree_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def prefill_resume_case_result(planned, status):
    return {
        "case_id": planned.case_id,
        "identity": thaw(planned.identity),
        "status": status,
        "runs": 1,
        "errors": [] if status == "ok" else [{"code": status, "message": "old"}],
        "metrics": {"latency_ms": 1.0} if status == "ok" else None,
    }


def prefill_resume_tree(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def prefill_resume_initialized(root):
    run_plan = prefill_resume_plan(root)
    initialize_run_directory(
        root,
        source_config=prefill_resume_config(),
        plan=run_plan,
        run_id="run",
        code_version="rev",
    )
    store = RunStore(root)
    store.write_prefill_case(
        run_plan.cases[0].case_id, prefill_resume_case_result(run_plan.cases[0], "ok")
    )
    store.write_prefill_case(
        run_plan.cases[1].case_id,
        prefill_resume_case_result(run_plan.cases[1], "invalid_reuse"),
    )
    fingerprints = resume_fingerprints(run_plan)
    store.write_checkpoint(
        "prefill_progress",
        {
            "schema_version": 1,
            **fingerprints,
            "completed_case_ids": [case.case_id for case in run_plan.cases],
            "case_statuses": {
                run_plan.cases[0].case_id: "ok",
                run_plan.cases[1].case_id: "invalid_reuse",
            },
        },
    )
    return (run_plan, store)


def prefill_adapter_plan(root):
    config = {
        "schema_version": 1,
        "model": fixture_model(),
        "engine": fixture_engine(max_seq_len=4097, concurrency_limit=1),
        "workload": fixture_workload(compute_steps=[256], cache_lengths=[0]),
        "execution": fixture_execution(),
        "output": {},
        "regression": {},
        "publish": {},
    }
    cases = [
        {
            "input_len": 512,
            "cache_len": 0,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        },
        {
            "input_len": 1024,
            "cache_len": 512,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        },
    ]
    return build_execution_plan(
        mode="prefill",
        action="run",
        resolved=resolve_config(config),
        cases=cases,
        result_directory=str(root),
    )


def prefill_adapter_prepare_store(root):
    (root / "checkpoints").mkdir(parents=True)
    (root / "results").mkdir(parents=True)
    return RunStore(root)


def prefill_adapter_result(case, status):
    return {
        "case_id": case["case_id"],
        "identity": {k: v for (k, v) in case.items() if k != "case_id"},
        "status": status,
        "runs": 1,
        "errors": [] if status == "ok" else [{"code": status, "message": "failed"}],
        "metrics": {"latency_ms": 1.0} if status == "ok" else None,
    }


class PrefillResumeTest(unittest.TestCase):

    def test_resume_preserves_ok_bytes_and_reruns_every_other_case(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            (run_plan, store) = prefill_resume_initialized(root)
            ok_path = root / "results" / "cases.jsonl"
            ok_before = ok_path.read_bytes()
            runner = Mock()
            runner.run_case.side_effect = lambda case: prefill_resume_case_result(
                next(
                    (item for item in run_plan.cases if item.case_id == case["case_id"])
                ),
                "ok",
            )
            resume = PrefillResume(PrefillAdapter(lambda _context: runner))
            outcome = resume(
                ModeExecutionContext(
                    plan=prefill_resume_plan(root, action="resume"),
                    session=object(),
                    store=store,
                )
            )
            self.assertEqual(RunState.COMPLETED, outcome.state)
            runner.run_case.assert_called_once()
            self.assertEqual(
                run_plan.cases[1].case_id, runner.run_case.call_args.args[0]["case_id"]
            )
            self.assertTrue(ok_path.read_bytes().startswith(ok_before))
            self.assertTrue(
                all(
                    (
                        case["status"] == "ok"
                        for case in store.read_prefill_cases(run_plan).values()
                    )
                )
            )

    def test_fingerprint_mismatch_causes_no_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            (_, store) = prefill_resume_initialized(root)
            before = prefill_resume_tree(root)
            resume = PrefillResume(PrefillAdapter(lambda _context: Mock()))
            mismatched = prefill_resume_plan(
                root, action="resume", config=prefill_resume_config({"reuse_cache": 0})
            )
            with self.assertRaisesRegex(PlanningError, "config_fingerprint"):
                resume(
                    ModeExecutionContext(plan=mismatched, session=object(), store=store)
                )
            self.assertEqual(before, prefill_resume_tree(root))

    def test_concurrent_resume_is_rejected_without_runner_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            (_, store) = prefill_resume_initialized(root)
            runner = Mock()
            resume = PrefillResume(PrefillAdapter(lambda _context: runner))
            with RunDirectoryLock(root, action="resume"):
                before = prefill_resume_tree(root)
                with self.assertRaises(ResourceError):
                    resume(
                        ModeExecutionContext(
                            plan=prefill_resume_plan(root, action="resume"),
                            session=object(),
                            store=store,
                        )
                    )
                self.assertEqual(before, prefill_resume_tree(root))
            runner.assert_not_called()


class PrefillAdapterTest(unittest.TestCase):

    def test_failed_case_is_committed_and_independent_case_continues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = prefill_adapter_plan(root)
            runner = Mock()
            runner.run_case.side_effect = lambda case: prefill_adapter_result(
                case, "request_failed" if case["cache_len"] == 0 else "ok"
            )
            adapter = PrefillAdapter(lambda _context: runner)
            outcome = adapter(
                ModeExecutionContext(
                    plan=plan,
                    session=object(),
                    store=prefill_adapter_prepare_store(root),
                )
            )
            self.assertEqual(RunState.PARTIAL, outcome.state)
            self.assertEqual(2, runner.run_case.call_count)
            committed = RunStore(root).read_prefill_cases(plan)
            self.assertEqual([case.case_id for case in plan.cases], list(committed))
            self.assertEqual(
                2, len((root / "results/cases.jsonl").read_text().splitlines())
            )

    def test_interrupt_preserves_only_already_committed_cases_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = prefill_adapter_plan(root)
            runner = Mock()
            calls = 0

            def execute(case):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise KeyboardInterrupt()
                return prefill_adapter_result(case, "ok")

            runner.run_case.side_effect = execute
            adapter = PrefillAdapter(lambda _context: runner)
            store = prefill_adapter_prepare_store(root)
            with self.assertRaises(KeyboardInterrupt):
                adapter(ModeExecutionContext(plan=plan, session=object(), store=store))
            committed = store.read_prefill_cases(plan)
            self.assertEqual([plan.cases[0].case_id], list(committed))


class PrefillCaseStoreTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "results").mkdir()
        self.path = self.root / "results" / "cases.jsonl"
        self.planned = {"one": {"shape": [1, 2]}, "two": {"shape": [3]}}
        self.one = {"case_id": "one", "identity": {"shape": [1, 2]}, "status": "ok"}
        self.two = {"case_id": "two", "identity": {"shape": [3]}, "status": "failed"}

    def test_torn_tail_is_ignored_and_repaired_before_append(self):
        append_prefill_case(self.path, "one", self.one)
        with self.path.open("ab") as stream:
            stream.write(b'{"case_id":"broken')
        self.assertEqual({"one": self.one}, read_prefill_cases(self.path, self.planned))
        append_prefill_case(self.path, "two", self.two)
        self.assertEqual(
            {"one": self.one, "two": self.two},
            read_prefill_cases(self.path, self.planned),
        )
        self.assertEqual(2, len(self.path.read_bytes().splitlines()))


class AtomicStorageTest(unittest.TestCase):

    def test_fault_before_replace_preserves_committed_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            atomic_write_json(path, {"state": "planned"})

            def interrupt(stage, _target, _temporary):
                if stage == "before_replace":
                    raise KeyboardInterrupt("interrupted")

            with self.assertRaises(KeyboardInterrupt):
                atomic_write_json(path, {"state": "running"}, fault_injector=interrupt)
            self.assertEqual({"state": "planned"}, json.loads(path.read_text()))
            self.assertEqual(
                ["manifest.json"], [item.name for item in path.parent.iterdir()]
            )

    def test_run_store_commits_every_mutable_artifact_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "checkpoints").mkdir()
            (root / "results" / "cases").mkdir(parents=True)
            store = RunStore(root)
            store.write_manifest({"state": "running"})
            store.write_checkpoint("progress", {"completed": ["case-a"]})
            store.write_case("case-a", {"case_id": "case-a", "status": "ok"})
            store.write_final_result({"status": "completed"})
            store.write_summary({"successful_cases": 1})
            expected = {
                "manifest.json": {"state": "running"},
                "checkpoints/progress.json": {"completed": ["case-a"]},
                "results/cases/case-a.json": {"case_id": "case-a", "status": "ok"},
                "results/result.json": {"status": "completed"},
                "results/summary.json": {"successful_cases": 1},
            }
            for relative, value in expected.items():
                with self.subTest(relative=relative):
                    self.assertEqual(value, json.loads((root / relative).read_text()))


class RunDirectoryTest(unittest.TestCase):

    def test_second_writer_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run-001"
            initialize_run_directory(
                root,
                source_config=run_directory_source_config(),
                plan=run_directory_plan(root),
                run_id="run-001",
                code_version="deadbeef",
            )
            with RunDirectoryLock(root, action="run"):
                before = run_directory_tree_bytes(root)
                with self.assertRaisesRegex(
                    ResourceError, "already has a writer"
                ) as error:
                    with RunDirectoryLock(root, action="resume"):
                        self.fail("second writer entered lock")
                self.assertEqual(3, error.exception.exit_code)
                self.assertEqual(before, run_directory_tree_bytes(root))
            self.assertFalse((root / ".writer.lock").exists())
            self.assertEqual("run-001", inspect_run_directory(root).manifest["run_id"])


if __name__ == "__main__":
    unittest.main()
