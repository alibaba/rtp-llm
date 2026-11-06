import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.aone.collector import collect_aone_results
from rtp_llm.test.perf_test.aone.config_schema import parse_execution
from rtp_llm.test.perf_test.aone.package import generate_aone_package
from rtp_llm.test.perf_test.aone.plan import plan_resources
from rtp_llm.test.perf_test.aone.sharding import plan_shards
from rtp_llm.test.perf_test.aone.worker import load_remote_inputs
from rtp_llm.test.perf_test.capabilities import validate_aone_capability
from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.execution.aone import AOneExecutor
from rtp_llm.test.perf_test.execution.aone_bazel import (
    AONE_TEST_TIMEOUT_S,
    BazelAOneIntegration,
    create_default_aone_executor,
)
from rtp_llm.test.perf_test.plan import build_execution_plan, thaw
from rtp_llm.test.perf_test.results import parse_result_envelope
from rtp_llm.test.perf_test.run_directory import (
    initialize_run_directory,
    inspect_run_directory,
)
from rtp_llm.test.perf_test.state import RunState


def prefill_source(
    *, model=None, engine=None, workload=None, execution=None, output=None
):
    """Fresh minimal remote prefill config; section overrides are explicit per case."""
    source = {
        "schema_version": 1,
        "model": {"label": "test", "type": "registered", "checkpoint": "/m"},
        "engine": {
            "max_seq_len": 4096,
            "max_batch_tokens": 8192,
            "concurrency_limit": 2,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [256],
            "batch_sizes": [1],
        },
        "execution": {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": 2,
            "shard_count": 2,
        },
        "output": {},
        "regression": {},
        "publish": {},
    }
    for section, overrides in (
        ("model", model),
        ("engine", engine),
        ("workload", workload),
        ("execution", execution),
        ("output", output),
    ):
        if overrides is not None:
            source[section].update(overrides)
    return source


def _source_config(shard_count):
    return prefill_source(
        engine={"max_seq_len": 4097, "concurrency_limit": 1},
        execution={"max_parallel": shard_count, "shard_count": shard_count},
    )


def _setup(root, shard_count):
    source = _source_config(shard_count)
    root.parent.mkdir(parents=True, exist_ok=True)
    config_path = root.parent / "submit-host-only.json"
    config_path.write_text(json.dumps(source))
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
    plan = build_execution_plan(
        resource_planner=plan_resources,
        capability_validator=validate_aone_capability,
        mode="prefill",
        action="run",
        resolved=load_resolved_config(config_path, execution_parser=parse_execution),
        cases=cases,
        result_directory=str(root),
    )
    initialize_run_directory(
        root, source_config=source, plan=plan, run_id="run-aone", code_version="rev"
    )
    config_path.unlink()
    return plan


def _orchestrator_case_result(case):
    identity = thaw(case.identity)
    return {
        "case_id": case.case_id,
        "identity": identity,
        "status": "ok",
        "runs": 1,
        "errors": [],
        "metrics": {
            "input_length": identity["input_len"],
            "expected_reuse_length": identity["cache_len"],
            "observed_reuse_length": identity["cache_len"],
            "compute_length": identity["compute_len"],
            "latency_ms": 10.0,
            "throughput_tokens_per_second": 1000.0,
            "validation": {"shape": True, "reuse": True, "timing": True},
            "seed": 1,
            "prefix_policy": {"kind": "shared_seed"},
            "measurements": [{"run": 0, "latency_ms": 10.0}],
        },
    }


def _remote_results(plan, failed_shard=None):
    shard_plan = plan_shards(plan)
    cases = {case.case_id: case for case in plan.cases}
    payloads = {}
    for shard in shard_plan.shards:
        failed = shard.shard_id == failed_shard
        completed = () if failed else shard.case_ids
        payloads[shard.shard_id] = {
            "schema_version": 1,
            "run_id": "run-aone",
            "shard_id": shard.shard_id,
            "mode": "prefill",
            "status": "failed" if failed else "completed",
            "config_fingerprint": plan.config_fingerprint,
            "workload_fingerprint": plan.workload_fingerprint,
            "owned_case_ids": list(shard.case_ids),
            "completed_case_ids": list(completed),
            "cases": [
                _orchestrator_case_result(cases[case_id]) for case_id in completed
            ],
            "engine_diagnostics": {},
            "failure": {"type": "EngineError", "message": "remote"} if failed else None,
        }
    return payloads


class AOneSingleTest(unittest.TestCase):

    def test_remote_rules_keep_watchdog_at_whole_test_budget(self):
        actions = []
        scope = {"native": SimpleNamespace(py_test=lambda **kw: actions.append(kw))}
        exec((Path(__file__).parent / "aone_defs.bzl").read_text(), scope)
        common = dict(name="test", gpu_type="H20_CU13")
        for rule in ("unified_prefill_ci_test", "unified_decode_remote_test"):
            scope[rule](**common, config="config.json", gpu_count=2)
        scope["perf_test_aone_shard"](
            **common,
            resolved_config="config.json",
            resolved_workload="workload.json",
            shard_plan="shards.json",
            shard_id="0",
            gpu_count=2,
        )
        scope["perf_test_gang_shard"](
            **common,
            resolved_config="config.json",
            resolved_workload="workload.json",
            gang_plan="gang.json",
            gpus_per_host=2,
            machine_count=2,
        )
        for action in actions:
            with self.subTest(main=action["main"]):
                self.assertEqual(
                    AONE_TEST_TIMEOUT_S,
                    int(action.get("env", {}).get("RTP_TEST_WATCHDOG_SECONDS", 1800)),
                )

    def test_normal_submission_collects_portable_shards_and_formal_report(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "runs/normal"
            plan = _setup(root, 2)
            payloads = _remote_results(plan)
            commands = []
            process = Mock()
            process.wait.return_value = 0

            def run(command, **kwargs):
                commands.append(command)
                return SimpleNamespace(returncode=0)

            def popen(command, **kwargs):
                commands.append(command)
                for index, payload in enumerate(payloads.values()):
                    output = (
                        workspace
                        / "bazel-testlogs/runs/normal/workload/aone_package"
                        / f"shard_{index:04d}/test.outputs"
                    )
                    output.mkdir(parents=True)
                    if index == 0:
                        with zipfile.ZipFile(output / "outputs.zip", "w") as archive:
                            archive.writestr(
                                "perf_test_shard-0000/shard_result.json",
                                json.dumps(payload),
                            )
                    else:
                        output = output / "perf_test_shard-0001"
                        output.mkdir()
                        (output / "shard_result.json").write_text(json.dumps(payload))
                return process

            integration = BazelAOneIntegration(
                workspace_root=workspace,
                resource="daily",
                gpu_type="H20",
                max_parallel=2,
                run_command=run,
                popen_command=popen,
            )
            with patch(
                "rtp_llm.test.perf_test.execution.aone_bazel._workspace_root",
                return_value=workspace,
            ), patch(
                "rtp_llm.test.perf_test.execution.aone_bazel.BazelAOneIntegration",
                return_value=integration,
            ):
                outcome = create_default_aone_executor(plan)
            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual(0, outcome.exit_code)
            self.assertEqual("formal", outcome.report["report_kind"])
            self.assertEqual(2, len(outcome.collection.cases))
            self.assertEqual(2, len(commands))
            self.assertIn("build", commands[0])
            for command in commands:
                self.assertIn("//runs/normal/workload/aone_package:shard_0000", command)
                self.assertIn("//runs/normal/workload/aone_package:shard_0001", command)
            self.assertIn("--jobs=2", commands[1])
            self.assertIn("test", commands[1])
            self.assertIn("--strategy=TestRunner=remote", commands[1])
            process.wait.assert_called_once()
            generated = root / "workload/aone_package"
            for index, case in enumerate(plan_shards(plan).shards):
                loaded = load_remote_inputs(
                    resolved_config=generated / "resolved_config.json",
                    resolved_workload=generated / "resolved_workload.json",
                    shard_plan=generated / "shard_plan.json",
                    shard_id=f"shard-{index:04d}",
                )
                self.assertEqual(case.case_ids, loaded.owned_case_ids)
                self.assertEqual(plan.workload_fingerprint, loaded.workload_fingerprint)
            self.assertEqual(
                "aone",
                parse_result_envelope(
                    json.loads((root / "results/result.json").read_text())
                ).backend,
            )
            self.assertTrue(
                all(
                    (
                        stage["status"] == "completed"
                        for stage in inspect_run_directory(root)
                        .manifest["stages"]
                        .values()
                    )
                )
            )

    def test_normal_remote_failure_preserves_valid_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs/partial"
            plan = _setup(root, 2)
            payloads = _remote_results(plan, failed_shard="shard-0001")
            outcome = AOneExecutor(
                workspace_root=Path(directory),
                builder=Mock(),
                submitter=lambda target, shard_id: shard_id,
                result_collector=lambda job, shard_id: payloads[job],
            ).execute(plan)
            self.assertEqual(RunState.PARTIAL, outcome.state)
            self.assertEqual(5, outcome.exit_code)
            self.assertEqual("diagnostic", outcome.report["report_kind"])
            self.assertEqual(1, outcome.report["counts"]["valid"])
            parsed = parse_result_envelope(
                json.loads((root / "results/result.json").read_text())
            )
            self.assertEqual(RunState.PARTIAL, parsed.status)
            self.assertEqual(2, len(parsed.cases))
            self.assertEqual(
                "partial",
                inspect_run_directory(root).manifest["stages"]["collect"]["status"],
            )

    def test_assignment_is_cost_balanced_and_independent_of_case_order(self):
        from dataclasses import replace

        with tempfile.TemporaryDirectory() as directory:
            plan = _setup(Path(directory) / "run", 2)
            assignments = plan_shards(plan)
            # Costs are 1024*512 and 512*512: the larger case goes first.
            by_length = {
                thaw(case.identity)["input_len"]: case.case_id for case in plan.cases
            }
            self.assertEqual(
                ((by_length[1024],), (by_length[512],)), assignments.ownership
            )
            self.assertEqual(
                (524288, 262144), tuple(s.estimated_cost for s in assignments.shards)
            )
            self.assertEqual(
                assignments,
                plan_shards(replace(plan, cases=tuple(reversed(plan.cases)))),
            )

    def test_frozen_runfiles_survive_submit_package_removal(self):
        import shutil

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _setup(root / "run", 2)
            package = generate_aone_package(
                root / "generated",
                plan,
                run_id="remote",
                ownership=plan_shards(plan).ownership,
            )
            remote = root / "remote"
            shutil.copytree(package.root, remote)
            shutil.rmtree(package.root)
            for index in range(2):
                loaded = load_remote_inputs(
                    resolved_config=remote / "resolved_config.json",
                    resolved_workload=remote / "resolved_workload.json",
                    shard_plan=remote / "shard_plan.json",
                    shard_id=f"shard-{index:04d}",
                )
                self.assertEqual("remote", loaded.run_id)
                self.assertEqual(1, len(loaded.owned_case_ids))

            # Evaluate the generated BUILD and its public rule, recording actions.
            declarations = []

            class Native:
                def py_test(self, **kwargs):
                    declarations.append(kwargs)

            def load(label, symbol):
                self.assertEqual("//rtp_llm/test/perf_test:aone_defs.bzl", label)
                self.assertEqual("perf_test_aone_shard", symbol)

            rule_scope = {"native": Native()}
            exec((Path(__file__).parent / "aone_defs.bzl").read_text(), rule_scope)
            exec(
                (remote / "BUILD.bazel").read_text(),
                {
                    "load": load,
                    "test_suite": lambda **kwargs: None,
                    "perf_test_aone_shard": rule_scope["perf_test_aone_shard"],
                },
            )
            self.assertEqual(
                ["shard_0000", "shard_0001"], [d["name"] for d in declarations]
            )
            for declaration in declarations:
                self.assertEqual(
                    "//rtp_llm/test/perf_test:aone/worker.py", declaration["main"]
                )
                self.assertEqual(
                    [
                        "//rtp_llm:sdk",
                        "resolved_config.json",
                        "resolved_workload.json",
                        "shard_plan.json",
                    ],
                    declaration["data"],
                )
                self.assertEqual(
                    {"gpu": "H20", "gpu_count": "1"}, declaration["exec_properties"]
                )
                self.assertEqual(
                    {"WORLD_SIZE": "1", "RTP_TEST_WATCHDOG_SECONDS": "7000"},
                    declaration["env"],
                )

    def test_collector_rejects_untrusted_claims_before_accepting_results(self):
        import copy

        with tempfile.TemporaryDirectory() as directory:
            plan = _setup(Path(directory) / "run", 2)

            def mutate(kind, results):
                first, second = results
                if kind == "duplicate-shard":
                    results.append(copy.deepcopy(first))
                elif kind == "unknown-shard":
                    first["shard_id"] = "unknown"
                elif kind == "missing-case":
                    first.update(completed_case_ids=[], cases=[])
                elif kind == "duplicate-case":
                    first["cases"].append(copy.deepcopy(first["cases"][0]))
                elif kind == "out-of-scope":
                    first["cases"] = second["cases"]
                elif kind == "config-fingerprint":
                    first["config_fingerprint"] = "invalid"
                elif kind == "workload-fingerprint":
                    first["workload_fingerprint"] = "invalid"
                elif kind == "identity":
                    first["cases"][0]["identity"]["input_len"] += 1
                elif kind == "completion":
                    first["completed_case_ids"] = []

            for kind in (
                "duplicate-shard",
                "unknown-shard",
                "missing-case",
                "duplicate-case",
                "out-of-scope",
                "config-fingerprint",
                "workload-fingerprint",
                "identity",
                "completion",
            ):
                with self.subTest(kind=kind):
                    results = list(_remote_results(plan).values())
                    mutate(kind, results)
                    outcome = collect_aone_results(
                        plan,
                        run_id="run-aone",
                        shard_plan=plan_shards(plan),
                        results=results,
                    )
                    self.assertEqual(RunState.FAILED, outcome.state)
                    self.assertEqual((), outcome.cases)
                    self.assertEqual(
                        results[1]["cases"], list(outcome.diagnostic_cases)
                    )
                    self.assertTrue(outcome.rejected_shards)
                    self.assertIsNotNone(outcome.failure)

    def test_missing_shard_retains_valid_results(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = _setup(Path(directory) / "run", 2)
            results = list(_remote_results(plan).values())
            outcome = collect_aone_results(
                plan,
                run_id="run-aone",
                shard_plan=plan_shards(plan),
                results=results[:1],
            )
            self.assertEqual(RunState.PARTIAL, outcome.state)
            self.assertEqual(("shard-0001",), outcome.missing_shard_ids)
            self.assertEqual(
                [results[0]["cases"][0]["case_id"]],
                [case["case_id"] for case in outcome.cases],
            )

    def test_package_rejects_invalid_ownership_before_writing(self):
        from rtp_llm.test.perf_test.errors import PlanningError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = _setup(root / "run", 2)
            first, second = (case.case_id for case in plan.cases)
            invalid = (
                ((first,),),
                ((first,), (first,)),
                ((first,), ()),
                ((first,), ("unknown",)),
                ((first, first), (second,)),
            )
            for ownership in invalid:
                with self.subTest(ownership=ownership):
                    generated = root / "generated"
                    with self.assertRaises(PlanningError):
                        generate_aone_package(
                            generated, plan, run_id="invalid", ownership=ownership
                        )
                    self.assertFalse(generated.exists())

    def test_nonzero_bazel_exit_keeps_valid_case_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "run"
            plan = _setup(root, 2)
            payloads = _remote_results(plan)

            def popen(_command, **_kwargs):
                for index, payload in enumerate(payloads.values()):
                    output = (
                        workspace
                        / "bazel-testlogs/run/workload/aone_package"
                        / f"shard_{index:04d}/test.outputs"
                        / f"perf_test_shard-{index:04d}"
                    )
                    output.mkdir(parents=True)
                    (output / "shard_result.json").write_text(json.dumps(payload))
                return SimpleNamespace(wait=lambda: 3)

            integration = BazelAOneIntegration(
                workspace_root=workspace,
                resource="daily",
                gpu_type="H20",
                max_parallel=2,
                run_command=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
                popen_command=popen,
            )
            outcome = AOneExecutor(
                workspace_root=workspace,
                builder=integration.build,
                submitter=integration.submit,
                result_collector=integration.collect,
            ).execute(plan)
            self.assertEqual(RunState.PARTIAL, outcome.state)
            self.assertEqual(5, outcome.exit_code)
            self.assertEqual(2, len(outcome.collection.cases))
            self.assertTrue(
                all(case["status"] == "ok" for case in outcome.collection.cases)
            )
            manifest = inspect_run_directory(root).manifest
            self.assertIn(
                "exited 3", manifest["stages"]["collect"]["errors"]["shard-0000"]
            )

    def test_integrity_failure_preserves_valid_shards_independent_of_order(self):
        for kind in ("fingerprint", "ownership", "identity"):
            for bad_index in (0, 1):
                with self.subTest(kind=kind, bad_index=bad_index):
                    with tempfile.TemporaryDirectory() as directory:
                        plan = _setup(Path(directory) / "run", 2)
                        results = list(_remote_results(plan).values())
                        rejected = results[bad_index]
                        accepted = results[1 - bad_index]
                        accepted["engine_diagnostics"] = {"log_path": "accepted.log"}
                        if kind == "fingerprint":
                            rejected["config_fingerprint"] = "invalid"
                        elif kind == "ownership":
                            rejected["owned_case_ids"] = []
                        else:
                            rejected["cases"][0]["identity"]["input_len"] += 1
                        outcome = collect_aone_results(
                            plan,
                            run_id="run-aone",
                            shard_plan=plan_shards(plan),
                            results=results,
                        )
                        self.assertEqual(RunState.FAILED, outcome.state)
                        self.assertEqual((), outcome.cases)
                        self.assertEqual(
                            accepted["cases"], list(outcome.diagnostic_cases)
                        )
                        self.assertIn(
                            kind, outcome.rejected_shards[rejected["shard_id"]]
                        )
                        self.assertEqual(
                            {accepted["shard_id"]: {"log_path": "accepted.log"}},
                            outcome.shard_diagnostics,
                        )

    def test_nonstring_status_is_a_controlled_collection_failure(self):
        for status in ([], {}, None, 1):
            with self.subTest(
                status=status
            ), tempfile.TemporaryDirectory() as directory:
                plan = _setup(Path(directory) / "run", 2)
                results = list(_remote_results(plan).values())
                results[0]["status"] = status
                outcome = collect_aone_results(
                    plan,
                    run_id="run-aone",
                    shard_plan=plan_shards(plan),
                    results=results,
                )
                self.assertEqual(RunState.FAILED, outcome.state)
                self.assertEqual((), outcome.cases)
                self.assertEqual(results[1]["cases"], list(outcome.diagnostic_cases))
                self.assertIn("status", outcome.rejected_shards["shard-0000"])

    def test_executor_finalizes_rejected_shards_with_separate_diagnostics(self):
        for invalid in ("fingerprint", [], {}):
            with self.subTest(
                invalid=invalid
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "run"
                plan = _setup(root, 2)
                payloads = _remote_results(plan)
                accepted = payloads["shard-0000"]
                accepted["engine_diagnostics"] = {"log_path": "accepted.log"}
                rejected = payloads["shard-0001"]
                if invalid == "fingerprint":
                    rejected["config_fingerprint"] = "invalid"
                else:
                    rejected["status"] = invalid
                outcome = AOneExecutor(
                    workspace_root=directory,
                    builder=lambda _package: None,
                    submitter=lambda _target, shard_id: shard_id,
                    result_collector=lambda job, _shard_id: payloads[job],
                ).execute(plan)
                self.assertEqual(RunState.FAILED, outcome.state)
                self.assertNotEqual(0, outcome.exit_code)
                self.assertEqual((), outcome.collection.cases)
                diagnostics = json.loads(
                    (root / "results/aone_diagnostics.json").read_text()
                )
                self.assertEqual("run-aone", diagnostics["run_id"])
                self.assertEqual(accepted["cases"], diagnostics["cases"])
                self.assertEqual(
                    {"shard-0000": {"log_path": "accepted.log"}},
                    diagnostics["shard_diagnostics"],
                )
                reason = "fingerprint" if invalid == "fingerprint" else "status"
                self.assertIn(reason, diagnostics["rejected_shards"]["shard-0001"])
                manifest = inspect_run_directory(root).manifest
                self.assertEqual("failed", manifest["state"])
                self.assertEqual("failed", manifest["stages"]["collect"]["status"])
                self.assertIn(
                    reason, manifest["stages"]["collect"]["errors"]["shard-0001"]
                )
                final = parse_result_envelope(
                    json.loads((root / "results/result.json").read_text())
                )
                self.assertEqual(RunState.FAILED, final.status)
                self.assertTrue(all(case.status != "ok" for case in final.cases))

    def test_collector_replaces_nonfinite_engine_diagnostics_with_error(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = _setup(Path(directory) / "run", 2)
            results = list(_remote_results(plan).values())
            results[0]["engine_diagnostics"] = json.loads(
                '{"timing": {"values": [1e999, -1e999]}}'
            )
            results[1]["config_fingerprint"] = "invalid"
            outcome = collect_aone_results(
                plan,
                run_id="run-aone",
                shard_plan=plan_shards(plan),
                results=results,
            )
            self.assertEqual(RunState.FAILED, outcome.state)
            self.assertEqual(results[0]["cases"], list(outcome.diagnostic_cases))
            safe = outcome.shard_diagnostics["shard-0000"]
            self.assertIn("engine_diagnostics", safe["collection_errors"][0])
            json.dumps(outcome.shard_diagnostics, allow_nan=False)

    def test_executor_finalizes_overflowing_remote_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            plan = _setup(root, 2)
            payloads = _remote_results(plan)
            payloads["shard-0000"]["engine_diagnostics"] = json.loads('{"time":1e999}')
            payloads["shard-0001"]["config_fingerprint"] = "invalid"
            outcome = AOneExecutor(
                workspace_root=directory,
                builder=lambda _package: None,
                submitter=lambda _target, shard_id: shard_id,
                result_collector=lambda job, _shard_id: payloads[job],
            ).execute(plan)
            self.assertEqual(RunState.FAILED, outcome.state)
            self.assertEqual("failed", inspect_run_directory(root).manifest["state"])
            artifact = json.loads((root / "results/aone_diagnostics.json").read_text())
            self.assertEqual(payloads["shard-0000"]["cases"], artifact["cases"])
            self.assertIn(
                "collection_errors", artifact["shard_diagnostics"]["shard-0000"]
            )
            json.dumps(artifact, allow_nan=False)
            self.assertEqual(
                "failed",
                json.loads((root / "results/result.json").read_text())["status"],
            )

    def test_diagnostic_write_failure_does_not_skip_failed_terminal_result(self):
        for error in (
            OSError("diagnostics destination denied"),
            ValueError("diagnostic serialization failed"),
        ):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "run"
                plan = _setup(root, 2)
                payloads = _remote_results(plan)
                payloads["shard-0001"]["config_fingerprint"] = "invalid"
                executor = AOneExecutor(
                    workspace_root=directory,
                    builder=lambda _package: None,
                    submitter=lambda _target, shard_id: shard_id,
                    result_collector=lambda job, _shard_id: payloads[job],
                )
                with patch(
                    "rtp_llm.test.perf_test.execution.aone.atomic_write_json",
                    side_effect=error,
                ):
                    outcome = executor.execute(plan)
                self.assertEqual(RunState.FAILED, outcome.state)
                manifest = inspect_run_directory(root).manifest
                self.assertEqual("failed", manifest["state"])
                errors = manifest["stages"]["collect"]["errors"]
                self.assertIn(str(error), errors["diagnostics"])
                self.assertIn("fingerprint", errors["shard-0001"])
                final = json.loads((root / "results/result.json").read_text())
                self.assertEqual("failed", final["status"])
                self.assertEqual(
                    "failed",
                    json.loads((root / "results/summary.json").read_text())["status"],
                )

    def test_diagnostic_write_failure_preserves_interrupted_terminal_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            plan = _setup(root, 2)
            payloads = _remote_results(plan)
            payloads["shard-0000"]["config_fingerprint"] = "invalid"

            def collect(job, _shard_id):
                if job == "shard-0001":
                    raise KeyboardInterrupt()
                return payloads[job]

            executor = AOneExecutor(
                workspace_root=directory,
                builder=lambda _package: None,
                submitter=lambda _target, shard_id: shard_id,
                result_collector=collect,
            )
            with patch(
                "rtp_llm.test.perf_test.execution.aone.atomic_write_json",
                side_effect=OSError("diagnostics denied"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    executor.execute(plan)
            manifest = inspect_run_directory(root).manifest
            self.assertEqual("interrupted", manifest["state"])
            self.assertIn(
                "diagnostics denied",
                manifest["stages"]["collect"]["errors"]["diagnostics"],
            )
            self.assertEqual(
                "interrupted",
                json.loads((root / "results/result.json").read_text())["status"],
            )


if __name__ == "__main__":
    unittest.main()
