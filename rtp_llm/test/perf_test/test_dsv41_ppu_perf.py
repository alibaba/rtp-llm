"""CPU contracts for real grid sizing, stream accounting and sweep results."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from rtp_llm.test.perf_test import dsv41_ppu_perf as perf

HERE = Path(__file__).resolve().parent


def load_source(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def framework_sources():
    """Use real perf sources; replace only installed/compiled dependencies."""
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules))
        sys.modules["prettytable"] = types.SimpleNamespace(PrettyTable=MagicMock())
        dataclass = load_source("_ppu_perf_dataclass_test", "dataclass.py")
        prefix = "rtp_llm.test.perf_test."
        requests = types.SimpleNamespace(post=MagicMock())

        def check(condition, message):
            if not condition:
                raise RuntimeError(message)

        sys.modules.update(
            {
                "requests": requests,
                prefix + "dataclass": dataclass,
                "rtp_llm.utils.util": types.SimpleNamespace(check_with_info=check),
                prefix
                + "dataset": types.SimpleNamespace(
                    KNOWN_DATASETS={}, extract_arg=MagicMock()
                ),
                prefix
                + "hub_download": types.SimpleNamespace(
                    needs_perf_hub_resolve=MagicMock(),
                    resolve_checkpoint_or_tokenizer_for_perf=MagicMock(),
                ),
                prefix
                + "perf_utils": types.SimpleNamespace(
                    auto_generate_bs_list=lambda limit: [1, 8, limit]
                ),
                prefix
                + "sampling": types.SimpleNamespace(
                    prepare_distribution_config=MagicMock()
                ),
            }
        )
        impl = load_source("_ppu_perf_impl_test", "batch_perf_impl.py")
        config = load_source("_ppu_perf_config_test", "perf_config.py")
        yield dataclass, impl, config, requests


def response_evidence(case, owner=0, step_ms=4):
    events = []
    for count in range(1, case["output_tokens_per_request"] + 1):
        aux = {
            "input_len": case["input_len"],
            "output_len": count,
            "reuse_len": 0,
            "wait_time": 2,
            "cost_time": 10 + (count - 1) * step_ms,
            "first_token_cost_time": 10,
        }
        events.append(
            {
                "aux_info": aux,
                "arrival_ns": count * 1000000,
                "finished": count == case["output_tokens_per_request"],
            }
        )
    final = {"aux_info": events[-1]["aux_info"], "finished": True}
    return {
        "success": True,
        "events": events,
        "metric_response": final,
        "final": final,
        "owner_rank": owner,
    }


def write_raw(case, runs=2, step_ms=4):
    dp = len(case["dp_request_counts"])
    data = {
        "global_batch_size": case["batch_size"],
        "dp_size": dp,
        "measurement_runs": [
            {
                "run": i,
                "started_ns": 0,
                "ended_ns": 1000000000,
                "responses": [
                    response_evidence(case, r % dp, step_ms)
                    for r in range(case["batch_size"])
                ],
            }
            for i in range(runs)
        ],
    }
    path = Path(case["raw_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


class TestSweepPlan(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def args(self, *argv):
        return perf.parse_args(
            ["--checkpoint-path", "/models/v41", "--result-dir", "/tmp/ppu-plan", *argv]
        )

    def test_cartesian_grids_and_exact_phase_semantics(self):
        plan = perf.build_plan(
            self.args("all", "--batches", "1,8,16", "--context-lengths", "2k,8k")
        )
        prefill, decode = plan["jobs"]
        self.assertEqual([len(job["cases"]) for job in plan["jobs"]], [6, 6])
        self.assertEqual(prefill["topology"]["attention_cp"], 8)
        self.assertEqual(decode["topology"]["attention_dp"], 8)
        self.assertEqual(prefill["cases"][0]["input_len"], 2048)
        self.assertEqual(decode["cases"][0]["input_len"], 2049)
        self.assertEqual(
            decode["cases"][0]["dp_request_counts"], [1, 0, 0, 0, 0, 0, 0, 0]
        )
        self.assertEqual(decode["cases"][-1]["dp_request_counts"], [2] * 8)
        for job in plan["jobs"]:
            self.assertEqual(job["environment"]["PERF_PROFILE_RUNS"], "0")
            needed = max(
                c["input_len"] + c["output_tokens_per_request"] for c in job["cases"]
            )
            self.assertGreaterEqual(int(job["environment"]["MAX_SEQ_LEN"]), needed)
        self.assertEqual(
            prefill["command"][
                prefill["command"].index("--max_context_batch_size") + 1
            ],
            "16",
        )

    def test_fake_balance_is_explicit_and_recorded_for_both_backends(self):
        for backend in ("grouped", "megamoe"):
            for enabled in (False, True):
                option = (
                    "--fake-balance-expert" if enabled else "--no-fake-balance-expert"
                )
                job = perf.build_plan(
                    self.args("decode", "--moe-backend", backend, option)
                )["jobs"][0]
                self.assertEqual(job["fake_balance_expert"], enabled)
                self.assertTrue(
                    all(case["fake_balance_expert"] == enabled for case in job["cases"])
                )
                self.assertEqual(
                    job["environment"]["FAKE_BALANCE_EXPERT"], str(int(enabled))
                )
                index = job["command"].index("--fake_balance_expert")
                self.assertEqual(job["command"][index + 1], str(int(enabled)))
        with patch.dict(os.environ, {"FAKE_BALANCE_EXPERT": "1"}):
            self.assertTrue(self.args().fake_balance_expert)
            self.assertFalse(self.args("--no-fake-balance-expert").fake_balance_expert)

    def test_old_decode_and_graph_environment_still_work(self):
        with patch.dict(
            os.environ,
            {
                "RTP_V41_PERF_INPUT_LEN": "4097",
                "RTP_V41_PERF_BATCHES": "1,8",
                "RTP_V41_PERF_STEPS": "33",
            },
        ):
            job = perf.build_plan(self.args())["jobs"][0]
        self.assertEqual([c["input_len"] for c in job["cases"]], [4097, 4097])
        self.assertEqual(job["cases"][0]["context_length"], 4096)
        self.assertEqual(job["cases"][0]["output_tokens_per_request"], 33)
        switch = "RTP_V41_PPU_GRAPH_MEGA_MHC"
        self.assertEqual(job["environment"][switch], "0")
        graph = perf.build_plan(self.args("graph"))["jobs"][0]
        self.assertTrue(graph["graph"])
        self.assertEqual(graph["environment"][switch], "1")
        self.assertEqual(graph["cases"][0]["output_tokens_per_request"], 8)
        command = graph["command"]
        self.assertEqual(command[command.index("--decode_capture_config") + 1], "1,2")
        with patch.dict(os.environ, {switch: "0"}):
            graph = perf.build_plan(self.args("graph"))["jobs"][0]
            self.assertEqual(graph["environment"][switch], "0")
            graph = perf.build_plan(self.args("graph", "--graph-mega-mhc"))["jobs"][0]
            self.assertEqual(graph["environment"][switch], "1")
        graph = perf.build_plan(self.args("graph", "--no-graph-mega-mhc"))["jobs"][0]
        self.assertEqual(graph["environment"][switch], "0")
        with patch.dict(os.environ, {"RTP_V41_PPU_MHC_BACKEND": "legacy"}):
            graph = perf.build_plan(self.args("graph"))["jobs"][0]
            self.assertEqual(graph["environment"][switch], "0")

    def test_graph_attention_defaults_follow_dependency_and_overrides(self):
        packed = "RTP_V41_PPU_GRAPH_PACKED_ATTENTION"
        indexer = "RTP_V41_PPU_GRAPH_VISIBLE_INDEXER"

        def switches(*argv):
            job = perf.build_plan(self.args(*argv))["jobs"][0]
            return job["environment"][packed], job["environment"][indexer]

        self.assertEqual(switches("graph"), ("0", "1"))
        with patch.dict(os.environ, {"RTP_V41_PPU_PACKED_DEPENDENCY": "/fixture/packed.json"}):
            self.assertEqual(switches("decode"), ("0", "0"))
            self.assertEqual(switches("graph"), ("1", "1"))
            self.assertEqual(switches("graph", "--no-graph-packed-attention"), ("0", "1"))
            self.assertEqual(switches("graph", "--no-graph-visible-indexer"), ("1", "0"))
            with patch.dict(os.environ, {packed: "0", indexer: "0"}):
                self.assertEqual(switches("graph"), ("0", "0"))
                self.assertEqual(
                    switches("graph", "--graph-packed-attention", "--graph-visible-indexer"),
                    ("1", "1"),
                )

    def test_invalid_grids_fail_before_model_start(self):
        invalid = [
            ["--batches", "0"],
            ["--batches", "8,8"],
            ["--batches", "9"],
            ["--context-lengths", "2047"],
            ["--max-seq-len", "2048"],
            ["--max-seq-len", "0"],
            ["--max-seq-len", "-1"],
            ["--measure-runs", "0"],
            ["--decode-tokens", "1"],
            ["--decode-graph", "--batches", "64", "--capture-batches", "1"],
            ["--decode-graph", "--moe-backend", "grouped"],
            ["--graph-mega-mhc"],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                self.args(*argv)

    def test_long_context_grids_reserve_output_space_without_a_fixed_ceiling(self):
        jobs = perf.build_plan(
            self.args("all", "--batches", "8", "--context-lengths", "32k,64k,128k,256k")
        )["jobs"]
        for job in jobs:
            with self.subTest(phase=job["phase"]):
                self.assertEqual(len(job["cases"]), 4)
                self.assertEqual(int(job["environment"]["MAX_SEQ_LEN"]), 262400)
                for case in job["cases"]:
                    self.assertGreaterEqual(
                        int(job["environment"]["MAX_SEQ_LEN"]),
                        case["input_len"] + case["output_tokens_per_request"],
                    )
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                job = perf.build_plan(
                    self.args(
                        phase, "--context-lengths", "128k", "--max-seq-len", "262144"
                    )
                )["jobs"][0]
                self.assertEqual(job["environment"]["MAX_SEQ_LEN"], "262144")
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.args(
                        phase, "--context-lengths", "128k", "--max-seq-len", "131072"
                    )

    def test_dry_run_needs_no_native_runtime_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-results"
            with redirect_stdout(io.StringIO()) as captured:
                code = perf.main(
                    [
                        "all",
                        "--checkpoint-path",
                        "/models/v41",
                        "--result-dir",
                        str(output),
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(len(json.loads(captured.getvalue())["jobs"]), 2)
            self.assertFalse(output.exists())


class TestResults(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        with patch.dict(os.environ, {}, clear=True):
            self.args = perf.parse_args(
                [
                    "all",
                    "--checkpoint-path",
                    "/models/v41",
                    "--result-dir",
                    self.directory.name,
                    "--batches",
                    "8",
                    "--context-lengths",
                    "2048",
                    "--decode-tokens",
                    "3",
                ]
            )
        self.prefill, self.decode = [
            job["cases"][0] for job in perf.build_plan(self.args)["jobs"]
        ]

    def test_throughput_uses_actual_tokens_and_full_wall_window(self):
        write_raw(self.prefill)
        result = perf.summarize_case(self.prefill, 2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["throughput"]["input_tokens_per_s"], 8 * 2048)
        self.assertEqual(result["throughput"]["input_tokens_per_s_per_device"], 2048)
        self.assertEqual(result["latency_ms"]["ttft"]["mean"], 8)
        self.assertIsNone(result["latency_ms"]["tpot"])
        write_raw(self.decode)
        result = perf.summarize_case(self.decode, 2)
        self.assertEqual(result["throughput"]["output_tokens_per_s"], 24)
        self.assertIsNone(result["throughput"]["input_tokens_per_s"])
        self.assertEqual(result["latency_ms"]["tpot"]["mean"], 4)
        self.assertEqual(result["latency_ms"]["token_interval"]["count"], 32)

    def test_latency_tails_are_retained_without_trimming(self):
        data = write_raw(self.decode)
        data["measurement_runs"][1]["responses"][-1] = response_evidence(
            self.decode, 7, step_ms=100
        )
        Path(self.decode["raw_path"]).write_text(json.dumps(data))
        result = perf.summarize_case(self.decode, 2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["latency_ms"]["token_interval"]["max"], 100)
        self.assertGreater(result["latency_ms"]["token_interval"]["p95"], 4)
        self.assertFalse(result["trimmed"])

    def test_partial_failed_wrong_length_and_wrong_owner_have_no_performance(self):
        for defect in (
            "success",
            "length",
            "owner",
            "missing_event",
            "reuse",
            "run_count",
        ):
            with self.subTest(defect=defect):
                case = self.prefill if defect == "reuse" else self.decode
                data = write_raw(case)
                response = data["measurement_runs"][0]["responses"][0]
                if defect == "success":
                    response["success"] = False
                elif defect == "length":
                    response["metric_response"]["aux_info"]["input_len"] -= 1
                elif defect == "owner":
                    response["owner_rank"] = 9
                elif defect == "missing_event":
                    del response["events"][1]
                elif defect == "reuse":
                    response["metric_response"]["aux_info"]["reuse_len"] = 32
                else:
                    data["measurement_runs"].pop()
                Path(case["raw_path"]).write_text(json.dumps(data))
                result = perf.summarize_case(case, 2)
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["throughput"])
                self.assertTrue(result["errors"])

    def test_missing_and_malformed_evidence_is_a_saved_failure(self):
        for data in (None, "{", "[]", json.dumps({"measurement_runs": [None]})):
            if data is not None:
                Path(self.decode["raw_path"]).parent.mkdir(parents=True, exist_ok=True)
                Path(self.decode["raw_path"]).write_text(data)
            self.assertEqual(perf.summarize_case(self.decode, 2)["status"], "failed")

    def test_measuring_both_phases_finishes_before_separate_profiles(self):
        runtime = Path(self.directory.name) / "runtime"
        runtime.mkdir()
        (runtime / "native-runtime.json").write_text('{"commit":"cpu-fixture"}')
        output = Path(self.directory.name) / "results"
        argv = [
            "all",
            "--checkpoint-path",
            "/models/v41",
            "--runtime-root",
            str(runtime),
            "--result-dir",
            str(output),
            "--batches",
            "8",
            "--context-lengths",
            "2048",
            "--decode-tokens",
            "3",
            "--profile-runs",
            "1",
        ]
        calls = []

        def execute(args, job, directory=None, profile_runs=0):
            calls.append((job["phase"], profile_runs))
            if not profile_runs:
                for case in job["cases"]:
                    write_raw(case, args.measure_runs)

        with patch.dict(os.environ, {}, clear=True), patch.object(
            perf, "execute_phase", side_effect=execute
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(perf.main(argv), 0)
            with self.assertRaises(FileExistsError):
                perf.main(argv)
        self.assertEqual(
            calls, [("prefill", 0), ("decode", 0), ("prefill", 1), ("decode", 1)]
        )
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "ok")
        rows = [
            json.loads(line)
            for line in (output / "results.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(rows), 2)

    def test_runner_failure_still_saves_every_requested_point(self):
        runtime = Path(self.directory.name) / "runtime"
        runtime.mkdir()
        (runtime / "native-runtime.json").write_text("{}")
        output = Path(self.directory.name) / "failed"
        with patch.dict(os.environ, {}, clear=True), patch.object(
            perf, "execute_phase", side_effect=RuntimeError("startup failed")
        ), redirect_stdout(io.StringIO()):
            code = perf.main(
                [
                    "all",
                    "--checkpoint-path",
                    "/models/v41",
                    "--runtime-root",
                    str(runtime),
                    "--result-dir",
                    str(output),
                    "--batches",
                    "8",
                    "--context-lengths",
                    "2048,4096",
                ]
            )
        self.assertEqual(code, 1)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(len(summary["cases"]), 4)
        self.assertTrue(all(row["throughput"] is None for row in summary["cases"]))

    def test_execution_passes_phase_flags_and_isolates_profile_output(self):
        args = self.args
        args.runtime_root = self.directory.name
        job = perf.build_plan(args)["jobs"][0]
        with patch.object(perf, "phase_environment", return_value={}), patch.object(
            perf, "prepare_token_map", return_value="/token-map"
        ), patch.object(perf.subprocess, "run") as run:
            run.return_value.returncode = 0
            perf.execute_phase(args, job)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--partial") + 1], "2")
            self.assertEqual(run.call_args.kwargs["cwd"], args.runtime_root)
            self.assertEqual(
                run.call_args.kwargs["env"]["RTP_V41_TOKEN_MAP_CACHE"], "/token-map"
            )
            profile_dir = Path(self.directory.name) / "profiles"
            perf.execute_phase(args, job, profile_dir, 2)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--profile_runs") + 1], "2")
            self.assertEqual(
                command[command.index("--result_dir") + 1], str(profile_dir)
            )
            self.assertEqual(run.call_args.kwargs["env"]["PERF_MEASURE_RUNS"], "1")
            run.return_value.returncode = 3
            with self.assertRaisesRegex(RuntimeError, "exited 3"):
                perf.execute_phase(
                    args, job, Path(self.directory.name) / "failed-process"
                )

    def test_graph_receipts_require_every_rank_and_native_work(self):
        directory = Path(self.directory.name) / "receipts"
        directory.mkdir()
        job = {"environment": {"RTP_V41_PERF_EVIDENCE_DIR": str(directory)}}
        for rank in range(8):
            report = {
                "rank": rank,
                "world_size": 8,
                "model_layers": [40],
                "decode_graph_replays": 2,
                "mega_mhc": [{"graph_capture_calls": 0, "fallbacks": {}}],
                "engines": [
                    {
                        "layers": 40,
                        "backend": "megamoe",
                        "precision": "w4a4",
                        "grouped_gemm_calls": 0,
                        "mega_moe_calls": 80,
                        "mega_received_assignments": 16,
                    }
                ],
            }
            (directory / f"moe-rank{rank}.json").write_text(json.dumps(report))
        perf.check_graph_evidence(job)
        report["decode_graph_replays"] = 0
        (directory / "moe-rank7.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "rank 7"):
            perf.check_graph_evidence(job)

        job["environment"]["RTP_V41_PPU_GRAPH_MEGA_MHC"] = "1"
        with self.assertRaisesRegex(ValueError, "rank 0: incomplete Mega-mHC"):
            perf.check_graph_evidence(job)
        report["decode_graph_replays"] = 2
        report["mega_mhc"][0]["graph_capture_calls"] = 80
        for rank in range(8):
            report["rank"] = rank
            (directory / f"moe-rank{rank}.json").write_text(json.dumps(report))
        perf.check_graph_evidence(job)


class TestStandardFramework(unittest.TestCase):
    def test_prefill_explicit_batch_sizes_admission_and_global_batch(self):
        with framework_sources() as (_, _, config, _), patch.dict(
            os.environ, {"PERF_GLOBAL_BATCH": "1"}, clear=True
        ):
            args = config.parse_args(
                ["--partial", "2", "--dp_size", "1", "--batch_size", "1,8,16"]
            )[0]
            prepared = config.prepare_config(args, [])
            self.assertEqual(prepared.batch_size_list, [1, 8, 16])
            self.assertEqual(prepared.max_concurrency, 16)
            args = config.parse_args(["--partial", "2"])[0]
            self.assertEqual(config.prepare_config(args, []).batch_size_list, [1])

    def test_native_prefill_stream_requests_exactly_one_token(self):
        with framework_sources() as (_, impl, _, requests), patch.dict(
            os.environ, {"PERF_DETAILED_LATENCIES": "1"}, clear=True
        ):
            response = requests.post.return_value
            response.status_code = 200
            response.iter_lines.return_value = [
                b'data:{"finished":true,"aux_info":[{"input_len":2048,"output_len":1,"cost_time":25}]}',
                b"data:[done]",
            ]
            result = impl._curl_server_single_worker(
                0, 20000, "hello", False, 129, 60, native_dp_size=1
            )
            self.assertTrue(result.success)
            self.assertEqual(result.prefill_time, 25)
            self.assertTrue(requests.post.call_args.kwargs["stream"])
            self.assertEqual(
                requests.post.call_args.kwargs["json"]["generate_config"][
                    "max_new_tokens"
                ],
                1,
            )
            response.close.assert_called_once()

    def test_raw_prefill_rounds_and_client_pool_cleanup_on_failure(self):
        with framework_sources() as (
            dataclass,
            impl,
            _,
            _,
        ), tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {"PERF_DETAILED_LATENCIES": "1", "PERF_RAW_OUTPUT_DIR": temporary},
            clear=True,
        ), patch.object(
            impl, "ProcessPoolExecutor"
        ):
            client = impl.BatchPerfImpl(
                8000,
                1,
                2,
                "hello",
                is_decode=False,
                warmup_runs=0,
                measure_runs=1,
                profile_runs=0,
                profile_trace_name="prefill_test",
            )
            wire = {
                "aux_info": {
                    "input_len": 128,
                    "output_len": 1,
                    "cost_time": 10,
                    "first_token_cost_time": 10,
                }
            }
            response = dataclass.ResponseInfo(wire)
            response.raw_measurement = {"success": True, "metric_response": wire}
            with patch.object(client, "_set_concurrency"), patch.object(
                client, "_curl_server_responses", return_value=[response, response]
            ):
                metric = client.run()
            self.assertEqual(metric.success_requests, 2)
            raw = json.loads((Path(temporary) / "prefill_test.json").read_text())
            self.assertEqual(raw["phase"], "prefill")
            self.assertEqual(len(raw["measurement_runs"][0]["responses"]), 2)
            client.executor.shutdown.assert_called_once_with(wait=True)
            failing = impl.BatchPerfImpl(8000, 1, 1, "hello")
            with patch.object(
                failing, "_set_concurrency", side_effect=RuntimeError("scheduler")
            ), self.assertRaises(RuntimeError):
                failing.run()
            failing.executor.shutdown.assert_called_with(wait=True)


if __name__ == "__main__":
    unittest.main()
