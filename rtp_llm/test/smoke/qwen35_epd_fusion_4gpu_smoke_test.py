"""CPU-only harness tests; no model imports or GPUs."""

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PATH = Path(__file__).with_name("qwen35_epd_fusion_4gpu_smoke.py")
import sys

sys.path.insert(0, str(PATH.parent))
import http.server
import json
import threading
import time
from types import SimpleNamespace

import qwen35_epd_fusion_4gpu_smoke as smoke


class FusionSmokeTest(unittest.TestCase):
    def test_coordinator_graph_oracle_uses_observed_execution_rank(self):
        row = dict(rank=1, prefill=0, batch=63, mode="graph", graph_bs=64)
        self.assertEqual(smoke.validate_coordinator_boundary([row], 63, 1), [row])
        for records in (
            [],
            [dict(row, batch=62)],
            [dict(row, graph_bs=96)],
            [dict(row, mode="eager")],
        ):
            with self.assertRaises(RuntimeError):
                smoke.validate_coordinator_boundary(records, 63, 1)
        eager = dict(row, mode="eager", graph_bs=0)
        self.assertEqual(smoke.validate_coordinator_boundary([eager], 63, 0), [eager])

    def test_steady_completion_windows_exclude_warmup_and_drain(self):
        rows = [
            dict(
                started_s=0,
                completed_s=end,
                ok=ok,
                client_rank=rank,
                usage=dict(completion_tokens=100),
            )
            for end, ok, rank in (
                (9, True, 0),
                (10, True, 0),
                (15, False, 1),
                (19, True, 2),
                (20, True, 3),
                (30, True, 0),
            )
        ]
        result = smoke.steady_summary(rows, 10, 10, 2)
        self.assertEqual(result["measurement"]["successful"], 3)
        self.assertEqual(result["measurement"]["errors"], 1)
        self.assertEqual(result["measurement"]["qps"], 3 / 20)
        self.assertEqual([r["successful"] for r in result["rounds"]], [2, 1])
        self.assertEqual(result["warmup_completions"], 1)
        self.assertEqual(result["drain_completions"], 1)
        self.assertFalse(result["all_valid"])
        self.assertFalse(result["rounds_within_10_percent"])

    def test_steady_windows_count_carry_in_and_track_inflight(self):
        rows = [
            dict(
                started_s=start,
                completed_s=end,
                ok=True,
                client_rank=0,
                usage=dict(completion_tokens=200),
            )
            for start, end in ((0, 11), (11, 21), (21, 31))
        ]
        result = smoke.steady_summary(rows, 10, 10, 2)
        self.assertEqual(result["measurement"]["qps"], 0.1)
        self.assertEqual(result["measurement"]["completed_output_tokens_per_s"], 20)
        self.assertEqual(result["measurement"]["in_flight_at_start"], 1)
        self.assertEqual(result["measurement"]["in_flight_at_end"], 1)
        self.assertTrue(result["rounds_within_10_percent"])

    def test_steady_lanes_refill_until_deadline_and_drain(self):
        import sys
        import time
        from types import SimpleNamespace

        sleep = time.sleep

        def response(url, payload, index, expected):
            sleep(0.015)
            return dict(
                index=index,
                url=url,
                ok=True,
                usage=dict(completion_tokens=2),
                e2e_s=0.015,
                ttft_s=0.01,
            )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.config.py_config_modules": SimpleNamespace(
                        MIN_WORKER_INFO_PORT_NUM=9
                    )
                },
            ):
                with mock.patch.object(
                    smoke, "request", side_effect=response
                ), mock.patch.object(
                    smoke.time, "sleep", side_effect=lambda x: sleep(min(x, 0.02))
                ):
                    rows, result = smoke.run_steady_load(
                        SimpleNamespace(port=20000),
                        {},
                        {},
                        4,
                        0,
                        1,
                        2,
                        Path(directory),
                        "steady_test",
                    )
            self.assertGreater(len(rows), 100)
            self.assertTrue(all(r["started_s"] < 2 for r in rows))
            self.assertGreater(result["drain_completions"], 0)
            self.assertEqual(result["peak_in_flight"], 4)
            self.assertFalse(result["aborted"])
            for lane in range(4):
                local = [r for r in rows if r["client_lane"] == lane]
                self.assertGreater(len(local), 10)
                self.assertEqual(
                    {r["url"] for r in local},
                    {f"http://127.0.0.1:{20000+lane*9}/v1/chat/completions"},
                )

    def test_global_cadence_is_explicit_and_rejects_invalid_arguments(self):
        env, original = smoke.server_config("0,1,2,3", "baseline", "mega_moe_fp8", 1)
        args = smoke.throughput_config(
            env, original, "prefill-first", "16", 1, "g1-run", "cadence"
        )[1]
        self.assertIn("--pdfusion_coord_mode cadence", args)
        self.assertIn("--decode_prefill_ratio 16", args)
        self.assertIn(
            "--pdfusion_coord_mode off",
            smoke.throughput_config(env, original, "prefill-first")[1],
        )
        for ratio, trace, run in (
            ("0", 1, "x"),
            ("1/2", 1, "x"),
            ("16", 0, "x"),
            ("16", 1, "x" * 64),
        ):
            with self.assertRaises(ValueError):
                smoke.throughput_config(
                    env, original, "prefill-first", ratio, trace, run, "cadence"
                )

    def test_schedule_trace_and_ratio_reach_server_argv(self):
        env, original = smoke.server_config("0,1,2,3", "baseline", "mega_moe_fp8", 1)
        _, args = smoke.throughput_config(
            env, original, "prefill-first", "2", 1, "o0-b1"
        )
        self.assertIn("--decode_prefill_ratio 2", args)
        self.assertNotIn("--decode_prefill_ratio 0", args)
        self.assertIn("--pdfusion_schedule_trace 1 --pdfusion_trace_run_id o0-b1", args)
        self.assertNotIn(
            "--pdfusion_schedule_trace",
            smoke.throughput_config(env, original, "prefill-first")[1],
        )
        for policy, ratio, run in (
            ("fifo", "2", "x"),
            ("prefill-first", "-1", "x"),
            ("prefill-first", "2", "unset"),
            ("prefill-first", "2", "x y"),
        ):
            with self.assertRaises(ValueError):
                smoke.throughput_config(env, original, policy, ratio, 1, run)

    def test_throughput_policy_and_balanced_sweep(self):
        env, original = smoke.server_config("0,1,2,3", "baseline", "mega_moe_fp8", 1)
        _, fifo = smoke.throughput_config(env, original, "fifo")
        _, packed = smoke.throughput_config(env, original, "prefill-first")
        self.assertNotIn("--pdfusion_scheduler_mode", fifo)
        self.assertIn(
            "--pdfusion_scheduler_mode ratio --decode_prefill_ratio 0", packed
        )
        for args in (fifo, packed):
            self.assertIn("--concurrency_limit 4", args)
            self.assertIn("--max_context_batch_size 1", args)
            self.assertIn("--decode_capture_config 1,2,4", args)
        plan = smoke.throughput_plan(2)
        self.assertEqual([c for _, c, _ in plan[1:]], [4, 8, 16, 16, 8, 4])
        for _, concurrency, count in plan:
            assignments = [
                [i % 4 for i in range(w, count, concurrency)]
                for w in range(concurrency)
            ]
            self.assertTrue(all(len(set(a)) == 1 for a in assignments))
            self.assertEqual(
                [sum(a.count(r) for a in assignments) for r in range(4)],
                [count // 4] * 4,
            )

    def test_throughput_requests_use_all_four_frontends_and_refill_lanes(self):
        import sys
        from types import SimpleNamespace

        fake_config = SimpleNamespace(MIN_WORKER_INFO_PORT_NUM=9)

        def response(url, payload, index, expected):
            return dict(index=index, url=url, ok=True)

        with mock.patch.dict(
            sys.modules, {"rtp_llm.config.py_config_modules": fake_config}
        ):
            with mock.patch.object(smoke, "request", side_effect=response):
                for concurrency in (4, 8, 16):
                    rows = smoke.run_throughput_batch(
                        SimpleNamespace(port=20000), {}, {}, concurrency, 32
                    )
                    self.assertEqual(sorted(r["index"] for r in rows), list(range(32)))
                    for rank in range(4):
                        local = [r for r in rows if r["client_rank"] == rank]
                        self.assertEqual(len(local), 8)
                        self.assertEqual(
                            {r["url"] for r in local},
                            {
                                f"http://127.0.0.1:{20000 + rank * 9}/v1/chat/completions"
                            },
                        )
                    for lane in range(concurrency):
                        local = [r for r in rows if r["client_lane"] == lane]
                        self.assertEqual(
                            [r["index"] for r in local],
                            list(range(lane, 32, concurrency)),
                        )

    def test_capacity_configuration_keeps_graph_coverage_and_kv_budget(self):
        env, args = smoke.server_config("0,1,2,3", "baseline", "mega_moe_fp8", 1)
        _, configured = smoke.capacity_config(env, args, 24, 16384, "1,2,4,8,16,24", 1)
        self.assertIn("--concurrency_limit 24", configured)
        self.assertIn("--kv_cache_mem_mb 16384", configured)
        self.assertIn("--decode_capture_config 1,2,4,8,16,24", configured)
        self.assertIn("--max_batch_tokens_size 32768", configured)
        for captures in ("", "1,2,4,8,16", "1,2,2,24"):
            with self.assertRaises(ValueError):
                smoke.capacity_config(env, args, 24, 16384, captures, 1)
        plan = smoke.capacity_plan("8,16,24", 2)
        self.assertEqual([c for _, c, _ in plan], [16, 32, 32, 64, 64, 96, 96])

    def test_capacity_auto_kv_preserves_runtime_reserve(self):
        env, args = smoke.server_config("0,1,2,3", "baseline", "mega_moe_fp8", 1)
        _, configured = smoke.capacity_config(
            env, args, 160, 0, "1,2,4,8,16,32,64,96,128,160", 1
        )
        self.assertIn("--kv_cache_mem_mb 0", configured)
        self.assertIn("--reserver_runtime_mem_mb 24576", configured)
        self.assertIn("--concurrency_limit 160", configured)
        with self.assertRaises(ValueError):
            smoke.capacity_config(env, args, 160, -1, "1,160", 1)

    def test_capacity_refinement_uses_weakest_rank_and_tests_next_batch(self):
        summary = dict(
            max_observed_decode_batch_by_rank={"0": 119, "1": 118, "2": 120, "3": 119}
        )
        plan = smoke.capacity_refinement(summary, [32, 64, 96, 128, 160], 2, 160)
        self.assertEqual([n for _, n, _ in plan], [472, 472, 476, 476])
        self.assertEqual(smoke.capacity_refinement(summary, [118, 119], 2, 160), [])
        self.assertEqual(len(smoke.capacity_refinement(summary, [], 2, 118)), 2)

    def test_capacity_evidence_does_not_count_queued_requests_or_graph_padding(self):
        phase = dict(
            phase="capacity_b24_r1",
            concurrency=96,
            count=96,
            wall_s=100,
            output_tokens_per_s=1000,
        )
        report = dict(
            batches=[phase], requests=[dict(phase=phase["phase"], ok=True)] * 96
        )
        records = [dict(rank=r, prefill=0, batch=18, graph_bs=24) for r in range(4)]
        evidence = smoke.capacity_summary(report, dict(records=records))
        self.assertFalse(evidence["levels"]["24"]["all_ranks_reached_requested_batch"])
        self.assertEqual(
            evidence["max_observed_decode_batch_by_rank"],
            {str(r): 18 for r in range(4)},
        )
        records += [dict(rank=r, prefill=0, batch=24, graph_bs=24) for r in range(4)]
        self.assertTrue(
            smoke.capacity_summary(report, dict(records=records))["levels"]["24"][
                "all_ranks_reached_requested_batch"
            ]
        )

    def test_gpu_mapping_and_config(self):
        for value in ("0,1,2", "4,4,6,7", "-1,2,3,4"):
            with self.assertRaises(ValueError):
                smoke.gpu_ids(value)
        env, args = smoke.server_config("4,5,6,7")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
        self.assertEqual(env["ROLE_TYPE"], "PDFUSION")
        self.assertIn("--vit_separation 0", args)
        self.assertIn("--enable_cuda_graph 0", args)

    def test_mega_graph_configuration(self):
        for graph in (0, 1):
            env, args = smoke.server_config(
                "0,1,2,3", "baseline", "mega_moe_fp8", graph
            )
            self.assertEqual(env["MOE_STRATEGY"], "mega_moe_fp8")
            self.assertIn("--use_deepep_moe 0", args)
            self.assertIn(f"--enable_cuda_graph {graph}", args)
            self.assertEqual("--decode_capture_config 1,2,4" in args, bool(graph))
            self.assertEqual(env["RTP_QWEN35_DECODE_FUSION"], "0")
        with self.assertRaises(ValueError):
            smoke.server_config("0,1,2,3", decode_graph=1)

    def test_fp8_switches_control_both_environment_and_server_argument(self):
        for cache, native in ((0, 0), (1, 0), (1, 1)):
            env, args = smoke.server_config(
                "0,1,2,3", "baseline", "mega_moe_fp8", 1, cache, native
            )
            self.assertEqual(env["FP8_KV_CACHE"], str(cache))
            self.assertEqual(env["RTP_QWEN35_NATIVE_FP8_ATTN"], str(native))
            self.assertIn(f"--fp8_kv_cache {cache}", args)
            self.assertIn(f"--seq_size_per_block {4096 if cache else 2048}", args)
            self.assertIn("--kernel_seq_size_per_block 64", args)
            self.assertIn("--act_type BF16", args)
            self.assertIn("--enable_cuda_graph 1", args)
        with self.assertRaises(ValueError):
            smoke.server_config("0,1,2,3", fp8_kv_cache=0, native_fp8_attn=1)

    def test_fp8_shared_pool_rejects_undersized_allocation_block(self):
        with self.assertRaisesRegex(ValueError, "requires seq_size_per_block=4096"):
            smoke.server_config(
                "0,1,2,3", fp8_kv_cache=1, native_fp8_attn=1, seq_size_per_block=2048
            )
        _, args = smoke.server_config("0,1,2,3", seq_size_per_block=4096)
        self.assertIn("--seq_size_per_block 4096", args)
        self.assertIn("--fp8_kv_cache 0", args)

    def test_native_fp8_evidence_requires_dispatched_fp8_tensors_per_phase(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            logs = root / "fusion_logs"
            logs.mkdir()
            lines = []
            for rank in range(4):
                pid = 100 + rank
                lines.append(f"[process-{pid}] [rank: {rank}] initialize process_group")
                lines.append(
                    f"[process-{pid}] FP8_ATTN_EXECUTION phase=prefill native=1 q=torch.float8_e4m3fn kv=torch.float8_e4m3fn out=torch.bfloat16"
                )
                dtype = "bfloat16" if rank == 3 else "float8_e4m3fn"
                lines.append(
                    f"[process-{pid}] FP8_ATTN_EXECUTION phase=decode native=1 q=torch.{dtype} kv=torch.float8_e4m3fn out=torch.bfloat16"
                )
            (logs / "process.log").write_text("\n".join(lines))
            evidence = smoke.execution_evidence(root)["native_fp8"]
            self.assertTrue(all(evidence[str(r)]["prefill"] for r in range(4)))
            self.assertTrue(all(evidence[str(r)]["decode"] for r in range(3)))
            self.assertFalse(evidence["3"]["decode"])

    def test_mega_options_reach_worker_and_separated_pd(self):
        from types import SimpleNamespace

        a = SimpleNamespace(
            gpus="0,1,2,3",
            encoder_gpus="",
            cache_root="/tmp/cache",
            model_dir="/tmp/model",
            data_dir="/tmp/data",
            output="/tmp/out",
            profile="baseline",
            perf_repeats=2,
            bazel_option=[],
            moe_strategy="mega_moe_fp8",
            decode_graph=1,
            graph_edge_cases=1,
            scheduler_policy="prefill-first",
            throughput_sweep=1,
            steady_concurrency="348,512,640",
            steady_warmup_seconds=240,
            steady_window_seconds=300,
            steady_windows=2,
            fp8_kv_cache=1,
            native_fp8_attn=1,
            seq_size_per_block=4096,
        )
        command = smoke.bazel_command(a)
        for option in (
            "moe-strategy=mega_moe_fp8",
            "decode-graph=1",
            "graph-edge-cases=1",
            "scheduler-policy=prefill-first",
            "throughput-sweep=1",
            "steady-concurrency=348,512,640",
            "steady-warmup-seconds=240",
            "steady-window-seconds=300",
            "steady-windows=2",
            "fp8-kv-cache=1",
            "native-fp8-attn=1",
            "seq-size-per-block=4096",
        ):
            self.assertIn("--test_arg=--" + option, command)
        configs = smoke.separated_service_configs(
            "4,5,6,7",
            "0,1",
            "baseline",
            dict(encoder_0=20000, encoder_1=21000, fusion=22000),
            9,
            "mega_moe_fp8",
            1,
            1,
            1,
        )
        self.assertIn("--enable_cuda_graph 1", configs[-1]["args"])
        self.assertIn("--moe_strategy mega_moe_fp8", configs[-1]["args"])
        self.assertIn("--fp8_kv_cache 1", configs[-1]["args"])
        self.assertEqual(configs[-1]["env"]["RTP_QWEN35_NATIVE_FP8_ATTN"], "1")
        for encoder in configs[:2]:
            self.assertIn("--enable_cuda_graph 0", encoder["args"])

    def test_execution_evidence_requires_real_replay_on_every_rank(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            logs = root / "fusion_logs"
            logs.mkdir()
            lines = ["MegaMoE FP8 weights prepared during model construction"]
            for rank in range(4):
                lines += [
                    f"GRAPH_CAPTURE_READY rank={rank}",
                    f"MODEL_EXECUTION rank={rank} mode=eager prefill=1 batch=1 graph_bs=0 count=1",
                    f"MODEL_EXECUTION rank={rank} mode=graph prefill=0 batch=3 graph_bs=4 count=1",
                ]
            (root / "logs").mkdir()
            (logs / "process.log").write_text(
                "\n".join(
                    f"[process-{100 + rank}][rank: {rank}] initialize process_group\n"
                    f"[process-{100 + rank}] MegaMoE FP8 weights prepared during model construction"
                    for rank in range(4)
                )
            )
            path = root / "logs/engine.log"
            path.write_text("\n".join(lines[1:]))
            evidence = smoke.execution_evidence(root)
            self.assertEqual(smoke.validate_execution_evidence(evidence, 1, 1), [])
            prefill_graph = dict(
                evidence, records=evidence["records"] + [dict(mode="graph", prefill=1)]
            )
            self.assertIn(
                "Prefill unexpectedly used graph execution",
                smoke.validate_execution_evidence(prefill_graph, 1, 1),
            )
            with path.open("a") as f:
                f.write(
                    "\nMODEL_EXECUTION rank=0 mode=eager prefill=0 batch=1 graph_bs=0 count=1"
                )
            self.assertIn(
                "Decode silently used eager execution in graph validation",
                smoke.validate_execution_evidence(smoke.execution_evidence(root), 1, 1),
            )
            path.write_text("\n".join(lines[:-1]))
            errors = smoke.validate_execution_evidence(
                smoke.execution_evidence(root), 1, 1
            )
            self.assertIn("rank 3: missing replay evidence", errors)
            # Capture alone never establishes actual request-time graph use.
            path.write_text(
                "\n".join(line for line in lines if "mode=graph" not in line)
            )
            self.assertTrue(
                smoke.validate_execution_evidence(smoke.execution_evidence(root), 1, 1)
            )

    def test_fallback_evidence_excludes_metric_registration_warnings(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            logs = root / "fusion_logs"
            logs.mkdir()
            actual = [
                "decode cuda graph: batch size 5 exceeds max captured 4, fallback to normal run",
                "decode cuda graph: capture_range_ is empty, cannot run",
            ]
            (logs / "process.log").write_text(
                "\n".join(
                    ["no metric named VIT_CUDA_GRAPH_FALLBACK_QPS_METRIC", *actual]
                )
            )
            self.assertEqual(smoke.execution_evidence(root)["fallbacks"], actual)

    def test_response_requires_fusion_and_complete_output(self):
        expected = {"prompt_tokens": 123, "video_tokens": 20240}
        result = dict(
            http_status=200,
            content="video description",
            done=True,
            finish_reason="stop",
            usage=dict(
                prompt_tokens=123, prompt_tokens_details=dict(video_tokens=20240)
            ),
            aux_info=dict(pd_sep=False, reuse_len=0),
        )
        self.assertEqual(smoke.validate_response(result, expected), [])
        result.pop("done")
        self.assertEqual(smoke.validate_response(result, expected), [])
        for patch in (
            dict(finish_reason=None),
            dict(finish_reason="length"),
            dict(content=""),
            dict(aux_info={}),
            dict(aux_info=dict(pd_sep=True, reuse_len=0)),
            dict(usage={}),
        ):
            self.assertTrue(smoke.validate_response(dict(result, **patch), expected))

    def test_multi_choice_stream_requires_each_completion(self):
        import sys
        from types import SimpleNamespace

        expected = dict(prompt_tokens=123, video_tokens=0)

        def run(missing=False, truncated=False):
            rows = []
            for i in range(3 if not missing else 2):
                rows.append(
                    dict(choices=[dict(index=i, delta=dict(content=f"answer-{i}"))])
                )
            for i in range(3 if not missing else 2):
                rows.append(
                    dict(
                        choices=[
                            dict(
                                index=i,
                                delta={},
                                finish_reason=(
                                    "length" if truncated and i == 1 else "stop"
                                ),
                            )
                        ]
                    )
                )
            rows.append(
                dict(
                    choices=[],
                    usage=dict(prompt_tokens=123, completion_tokens=9),
                    aux_info=dict(pd_sep=False, reuse_len=0),
                )
            )
            response = mock.MagicMock(status_code=200)
            response.__enter__.return_value = response
            response.iter_lines.return_value = [
                b"data: " + smoke.json.dumps(row).encode() for row in rows
            ]
            fake = SimpleNamespace(post=mock.Mock(return_value=response))
            with mock.patch.dict(sys.modules, requests=fake):
                return smoke.request("http://test", dict(n=3), 0, expected)

        result = run()
        self.assertTrue(result["ok"], result)
        self.assertEqual(
            [c["content"] for c in result["choices"].values()],
            ["answer-0", "answer-1", "answer-2"],
        )
        self.assertFalse(run(missing=True)["ok"])
        self.assertFalse(run(truncated=True)["ok"])

    def test_selected_gpu_check_ignores_other_cards(self):
        with mock.patch.object(
            smoke.subprocess,
            "check_output",
            side_effect=["4, 1, 0\n5, 1, 0\n6, 1, 0\n7, 1, 0\n", ""],
        ) as call:
            smoke.check_gpu_idle("4,5,6,7")
            for args in call.call_args_list:
                self.assertIn("--id=4,5,6,7", args.args[0])
        with mock.patch.object(
            smoke.subprocess,
            "check_output",
            return_value="4, 2000, 0\n5, 1, 0\n6, 1, 0\n7, 1, 0\n",
        ):
            with self.assertRaises(RuntimeError):
                smoke.check_gpu_idle("4,5,6,7")

    def test_profiles_preserve_fusion_topology(self):
        for profile in ("baseline", "fused", "flashinfer"):
            env, args = smoke.server_config("0,1,2,3", profile)
            self.assertEqual(
                env["RTP_QWEN35_DECODE_FUSION"], "0" if profile == "baseline" else "1"
            )
            self.assertEqual(
                env["RTP_QWEN35_FUSED_CONV_QKV_NORM"],
                "0" if profile == "baseline" else "1",
            )
            self.assertEqual(
                env["RTP_QWEN35_GDN_DECODE_BACKEND"],
                "flashinfer" if profile == "flashinfer" else "native",
            )
            self.assertEqual(env["ROLE_TYPE"], "PDFUSION")
            self.assertIn("--enable_cuda_graph 0", args)
            self.assertIn("--moe_strategy fp8_per_block_ep_normal", args)

    def test_performance_excludes_warmup(self):
        measured = dict(
            phase="single_video_1",
            ok=True,
            ttft_s=2,
            e2e_s=6,
            tpot_ms=1000,
            usage=dict(completion_tokens=5),
        )
        warm = dict(measured, phase="warmup", ttft_s=999)
        report = dict(
            requests=[warm, measured],
            batches=[
                dict(phase="warmup", wall_s=999),
                dict(phase="single_video_1", wall_s=6),
            ],
        )
        summary = smoke.performance_summary(report)["single_video"]
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["ttft_s"]["median"], 2)
        self.assertAlmostEqual(summary["output_tokens_per_s"], 5 / 6)

    def test_e2pd4_gpu_assignment_and_route(self):
        self.assertEqual(smoke.selected_gpus("4,5,6,7", "0,1"), "0,1,4,5,6,7")
        self.assertEqual(smoke.selected_gpus("4,5,6,7", "0"), "0,4,5,6,7")
        for encoder in ("0,1,2", "0,0", "0,4", "-1", "0,"):
            with self.assertRaises(ValueError):
                smoke.selected_gpus("4,5,6,7", encoder)
        ports = dict(encoder_0=20000, encoder_1=21000, fusion=22000)
        configs = smoke.separated_service_configs(
            "4,5,6,7", "0,1", "baseline", ports, 9
        )
        self.assertEqual([c["gpus"] for c in configs], [[0], [1], [4, 5, 6, 7]])
        for encoder in configs[:2]:
            self.assertEqual(encoder["env"]["ROLE_TYPE"], "VIT")
            self.assertEqual(encoder["env"]["WORLD_SIZE"], "1")
            self.assertIn("--vit_separation 1", encoder["args"])
        pd = configs[2]
        self.assertEqual(pd["env"]["ROLE_TYPE"], "PDFUSION")
        self.assertEqual(pd["env"]["VIT_SEPARATION"], "2")
        self.assertEqual(pd["env"]["WORLD_SIZE"], "4")
        self.assertNotIn("REMOTE_SERVER_PORT", pd["env"])
        group = smoke.json.loads(pd["env"]["MODEL_SERVICE_CONFIG"])["role_endpoints"][0]
        self.assertEqual(
            group["vit_endpoint"]["address"], "127.0.0.1:20000,127.0.0.1:21000"
        )
        self.assertEqual(
            group["pd_fusion_endpoint"]["address"],
            "127.0.0.1:22000,127.0.0.1:22009,127.0.0.1:22018,127.0.0.1:22027",
        )
        self.assertNotIn("prefill_endpoint", group)
        self.assertNotIn("decode_endpoint", group)

    def test_e2pd4_bazel_selection(self):
        from types import SimpleNamespace

        a = SimpleNamespace(
            gpus="4,5,6,7",
            encoder_gpus="0,1",
            cache_root="/tmp/bazel",
            model_dir="/tmp/model",
            data_dir="/tmp/data",
            output="/tmp/out",
            profile="baseline",
            perf_repeats=2,
            bazel_option=[],
        )
        command = smoke.bazel_command(a)
        self.assertIn(smoke.ENCODER_TARGET, command)
        self.assertIn("--test_env=GPU_COUNT=6", command)
        self.assertIn("--test_env=CUDA_VISIBLE_DEVICES=0,1,4,5,6,7", command)
        self.assertIn("--test_arg=--gpus=4,5,6,7", command)

    def test_encoder_evidence_requires_success_on_both_services(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i in range(2):
                logs = root / f"encoder_{i}_logs"
                logs.mkdir()
                row = dict(
                    request_id=i,
                    exception=None,
                    response="MMEmbeddingRes(embeddings_shape=[20424,4096])",
                )
                (logs / "mm_access_r0_s0.log").write_text(smoke.json.dumps(row) + "\n")
            self.assertTrue(all(smoke.encoder_log_evidence(root).values()))
            (root / "encoder_1_logs/mm_access_r0_s0.log").write_text(
                '{"exception": "failure", "response": null}\n'
            )
            self.assertFalse(all(smoke.encoder_log_evidence(root).values()))

    def test_encoder_start_failure_cleans_all_owned_services(self):
        import sys
        from types import ModuleType, SimpleNamespace

        module = ModuleType("rtp_llm.test.utils.maga_server_manager")
        config_module = ModuleType("rtp_llm.config.py_config_modules")
        config_module.MIN_WORKER_INFO_PORT_NUM = 9
        managers = [mock.Mock(port=20000 + i * 1000, exit_code=None) for i in range(3)]
        for i, m in enumerate(managers):
            m.start_server.return_value = i != 0
        module.MagaServerManager = mock.Mock(side_effect=managers)
        module.MagaServerManager.get_free_port.side_effect = [20000, 21000, 22000]
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(
                output=d,
                gpus="4,5,6,7",
                encoder_gpus="0,1",
                data_dir=str(PATH.parent / "qwen35_e2p4d2_data"),
                model_dir=d,
            )
            with mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,4,5,6,7"}
            ), mock.patch.dict(
                sys.modules,
                {module.__name__: module, config_module.__name__: config_module},
            ), mock.patch.object(
                smoke, "check_gpu_idle", return_value={}
            ), mock.patch.object(
                smoke, "expected_tokens", return_value={}
            ), mock.patch.object(
                smoke.threading.Thread, "start"
            ), mock.patch.object(
                smoke.threading.Thread, "join"
            ), mock.patch.object(
                smoke.concurrent.futures, "ThreadPoolExecutor"
            ) as pool:
                pool.return_value.__enter__.return_value.map.side_effect = (
                    lambda fn, items: list(map(fn, items))
                )
                self.assertEqual(smoke.run_worker(args), 1)
            for m in managers:
                m.stop_server.assert_called_once()
            result = smoke.json.loads(Path(d, "result.json").read_text())
            self.assertEqual(
                result["readiness"], dict(encoder_0=False, encoder_1=True, fusion=True)
            )

    def test_text_response_has_no_video_usage(self):
        expected = dict(prompt_tokens=24601, video_tokens=0)
        row = dict(
            http_status=200,
            content="report",
            finish_reason="stop",
            usage=dict(prompt_tokens=24601, prompt_tokens_details=None),
            aux_info=dict(pd_sep=False, reuse_len=0),
        )
        self.assertEqual(smoke.validate_response(row, expected), [])
        row["usage"]["prompt_tokens_details"] = dict(video_tokens=1)
        self.assertIn("video token mismatch", smoke.validate_response(row, expected))

    def test_text_bazel_target_preserves_four_gpu_selection(self):
        from types import SimpleNamespace

        a = SimpleNamespace(
            gpus="4,5,6,7",
            encoder_gpus="",
            workload="text",
            cache_root="/tmp/bazel",
            model_dir="/tmp/model",
            data_dir="/tmp/data",
            output="/tmp/out",
            profile="baseline",
            perf_repeats=2,
            bazel_option=[],
        )
        command = smoke.bazel_command(a)
        self.assertIn(smoke.TEXT_TARGET, command)
        self.assertIn("--test_env=CUDA_VISIBLE_DEVICES=4,5,6,7", command)
        self.assertIn("--test_env=GPU_COUNT=4", command)
        self.assertIn("--test_arg=--workload=text", command)

    def test_passive_gpu_lock_never_signals_compute_processes(self):
        spec = importlib.util.spec_from_file_location(
            "device_resource", PATH.parent.parent / "utils/device_resource.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        resource = object.__new__(mod.DeviceResource)
        resource.passive_lock = True
        resource.gpu_ids = ["4", "5", "6", "7"]
        with mock.patch.object(
            mod, "_nvidia_smi", return_value="nvidia-smi"
        ), mock.patch.object(mod.os, "kill") as kill, mock.patch.object(
            resource, "_get_gpu_pids"
        ) as query:
            for pids in ([123], None):
                query.return_value = pids
                self.assertFalse(resource._ensure_gpus_released())
                self.assertTrue(resource._has_zombie_gpu_contexts("4"))
            query.return_value = []
            self.assertTrue(resource._ensure_gpus_released())
            self.assertFalse(resource._has_zombie_gpu_contexts("4"))
            kill.assert_not_called()
        resource.query_timeout = 60
        with mock.patch.object(
            mod.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="")
        ) as run:
            self.assertEqual(resource._get_gpu_pids("4"), [])
            self.assertEqual(run.call_args.kwargs["timeout"], 60)

    def test_fixture_integrity(self):
        data = PATH.parent / "qwen35_e2p4d2_data"
        smoke.validate_fixture(data)
        request = smoke.payload_for(data)
        self.assertEqual(request["max_tokens"], 4096)
        self.assertEqual(
            request["messages"][0]["content"][0]["preprocess_config"]["fps"], 6
        )
        with tempfile.TemporaryDirectory() as d:
            from shutil import copy

            copy(data / "manifest.json", d)
            for name in ("video.mp4", "messages.json"):
                Path(d, name).write_text("bad")
            with self.assertRaises(ValueError):
                smoke.validate_fixture(Path(d))

    def test_foreign_gpu_use_after_lock_stops_before_model(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(output=d, gpus="4,5,6,7", data_dir=d, model_dir=d)
            with mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": args.gpus}
            ), mock.patch.object(
                smoke, "check_gpu_idle", side_effect=RuntimeError("foreign GPU use")
            ), mock.patch.object(
                smoke, "expected_tokens"
            ) as tokens, mock.patch.object(
                smoke.threading.Thread, "start"
            ), mock.patch.object(
                smoke.threading.Thread, "join"
            ):
                self.assertEqual(smoke.run_worker(args), 1)
                tokens.assert_not_called()
            result = smoke.json.loads(Path(d, "result.json").read_text())
            self.assertEqual(result["phase"], "gpu_availability_after_lock")
            self.assertEqual(result["requests"], [])

    def test_worker_failure_is_persisted(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(output=d, gpus="4,5,6,7", data_dir=d, model_dir=d)
            with mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": args.gpus}
            ), mock.patch.object(
                smoke, "check_gpu_idle", return_value={}
            ), mock.patch.object(
                smoke.threading.Thread, "start"
            ), mock.patch.object(
                smoke.threading.Thread, "join"
            ):
                self.assertEqual(smoke.run_worker(args), 1)
            self.assertEqual(
                smoke.json.loads(Path(d, "result.json").read_text())["status"], "FAILED"
            )

    def test_startup_failure_stops_only_owned_manager(self):
        import sys
        from types import ModuleType, SimpleNamespace

        module = ModuleType("rtp_llm.test.utils.maga_server_manager")
        manager = mock.Mock()
        manager.port = 12345
        manager.exit_code = 1
        manager.start_server.return_value = False
        module.MagaServerManager = mock.Mock(return_value=manager)
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(
                output=d,
                gpus="4,5,6,7",
                data_dir=str(PATH.parent / "qwen35_e2p4d2_data"),
                model_dir=d,
            )
            with mock.patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": args.gpus}
            ), mock.patch.object(
                smoke, "check_gpu_idle", return_value={}
            ), mock.patch.dict(
                sys.modules, {module.__name__: module}
            ), mock.patch.object(
                smoke, "expected_tokens", return_value={}
            ), mock.patch.object(
                smoke.threading.Thread, "start"
            ), mock.patch.object(
                smoke.threading.Thread, "join"
            ):
                self.assertEqual(smoke.run_worker(args), 1)
            manager.stop_server.assert_called_once()
            result = smoke.json.loads(Path(d, "result.json").read_text())
            self.assertEqual(result["phase"], "model_initialization")


class MeasurementTest(unittest.TestCase):
    def setUp(self):
        self.options = dict(smoke.CLIENT_OPTIONS)

    def tearDown(self):
        smoke.CLIENT_OPTIONS.clear()
        smoke.CLIENT_OPTIONS.update(self.options)

    def test_fixed_output_rejects_early_stop_and_checks_backend_count(self):
        payload, expected = smoke.fixed_output_payload(
            {}, dict(prompt_tokens=24601, video_tokens=0), 1024
        )
        self.assertEqual(payload["extra_configs"]["min_new_tokens"], 1024)
        response = dict(
            http_status=200,
            content="text",
            finish_reason="length",
            usage=dict(prompt_tokens=24601, completion_tokens=1024),
            aux_info=dict(pd_sep=False, reuse_len=0, output_len=1024),
        )
        self.assertEqual(smoke.validate_response(response, expected), [])
        response["usage"]["completion_tokens"] = 1000
        self.assertIn(
            "fixed output token mismatch", smoke.validate_response(response, expected)
        )
        response["usage"]["completion_tokens"] = 1024
        response["aux_info"]["output_len"] = 1000
        self.assertIn(
            "fixed backend output token mismatch",
            smoke.validate_response(response, expected),
        )
        with self.assertRaises(ValueError):
            smoke.fixed_output_payload({}, expected, 9000)

    def test_measured_lanes_drain_journal_without_losing_rows(self):
        smoke.CLIENT_OPTIONS["mode"] = "measured"

        def response(url, payload, index, expected):
            time.sleep(0.01)
            return dict(
                index=index,
                ok=True,
                e2e_s=0.01,
                ttft_s=0.001,
                usage=dict(completion_tokens=10),
            )

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.config.py_config_modules": SimpleNamespace(
                        MIN_WORKER_INFO_PORT_NUM=9
                    )
                },
            ), mock.patch.object(smoke, "request", side_effect=response):
                rows, summary = smoke.run_steady_load(
                    SimpleNamespace(port=20000), {}, {}, 4, 0, 1, 1, out, "test"
                )
            saved = [
                json.loads(line)
                for line in (out / "test-requests.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), len(saved))
            self.assertEqual({r["index"] for r in rows}, {r["index"] for r in saved})
            self.assertEqual(summary["peak_in_flight"], 4)
            self.assertTrue(all(r["started_s"] < 1 for r in saved))
            self.assertGreater(summary["drain_completions"], 0)
            self.assertTrue(all("journal_queue_s" in r for r in saved))

    def test_journal_failure_is_fatal(self):
        smoke.CLIENT_OPTIONS["mode"] = "measured"
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "test-requests.jsonl").write_text("preserve")
            with mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.config.py_config_modules": SimpleNamespace(
                        MIN_WORKER_INFO_PORT_NUM=9
                    )
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "journal failed"):
                    smoke.run_steady_load(
                        SimpleNamespace(port=1), {}, {}, 4, 0, 1, 1, out, "test"
                    )
            self.assertEqual((out / "test-requests.jsonl").read_text(), "preserve")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    fail = False

    def log_message(self, *args):
        pass

    def do_POST(self):
        json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(0.03)
        event = dict(
            choices=[
                dict(index=0, delta=dict(content="test output"), finish_reason="length")
            ],
            usage=dict(prompt_tokens=24601, completion_tokens=4, total_tokens=24605),
            aux_info=dict(pd_sep=False, reuse_len=0, output_len=4),
        )
        self.send_response(500 if self.fail else 200)
        self.send_header("Connection", "close")
        self.close_connection = True
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        body = ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
        self.wfile.write(f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n")
        self.wfile.flush()


def fail_one_client(port, options, *args):
    if options["index_offset"] == 0:
        raise RuntimeError("injected client startup failure")
    return smoke._client_process_entry(port, options, *args)


class ProcessLoadTest(unittest.TestCase):
    def setUp(self):
        self.options = dict(smoke.CLIENT_OPTIONS)

    def tearDown(self):
        smoke.CLIENT_OPTIONS.clear()
        smoke.CLIENT_OPTIONS.update(self.options)

    @classmethod
    def setUpClass(cls):
        for base in range(18800, 19000, 4):
            servers = []
            try:
                for rank in range(4):
                    servers.append(
                        http.server.ThreadingHTTPServer(
                            ("127.0.0.1", base + rank), Handler
                        )
                    )
            except OSError:
                for server in servers:
                    server.server_close()
                continue
            cls.servers = servers
            cls.port = base
            break
        else:
            raise RuntimeError("no free four-port test range")
        cls.threads = [
            threading.Thread(target=s.serve_forever, daemon=True) for s in cls.servers
        ]
        for thread in cls.threads:
            thread.start()

    @classmethod
    def tearDownClass(cls):
        for server in cls.servers:
            server.shutdown()
            server.server_close()
        for thread in cls.threads:
            thread.join()

    def run_load(self, out):
        smoke.CLIENT_OPTIONS.update(
            mode="measured",
            processes=4,
            worker_stride=1,
            chunk_size=4096,
            profile_requests=0,
            profile_dir=str(out),
        )
        return smoke.run_steady_load(
            SimpleNamespace(port=self.port),
            {},
            dict(prompt_tokens=24601, video_tokens=0, fixed_output_tokens=4),
            8,
            0,
            1,
            1,
            out,
            "steady_c8",
        )

    def test_shared_clock_unique_indices_rank_mapping_and_drain(self):
        Handler.fail = False
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            rows, summary = self.run_load(out)
            self.assertTrue(summary["all_valid"])
            self.assertFalse(summary["aborted"])
            self.assertEqual(summary["client_processes"], 4)
            self.assertGreater(summary["drain_completions"], 0)
            self.assertEqual(len({r["index"] for r in rows}), len(rows))
            self.assertEqual({r["client_lane"] for r in rows}, set(range(8)))
            self.assertTrue(all(r["client_rank"] == r["client_lane"] % 4 for r in rows))
            summaries = [
                json.loads(p.read_text())
                for p in out.glob("steady_c8-processes/*/steady_c8-summary.json")
            ]
            self.assertEqual(
                {s["started_at"] for s in summaries}, {summary["started_at"]}
            )
            self.assertEqual(len(rows), sum(s["total_requests"] for s in summaries))
            saved = [
                json.loads(line)
                for line in (out / "steady_c8-requests.jsonl").read_text().splitlines()
            ]
            self.assertEqual(saved, rows)
            self.assertTrue(all(r["started_s"] < 1 for r in rows))
            for pid in summary["client_pids"]:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_response_error_aborts_all_processes_and_preserves_rows(self):
        Handler.fail = True
        try:
            with tempfile.TemporaryDirectory() as directory:
                rows, summary = self.run_load(Path(directory))
                self.assertTrue(summary["aborted"])
                self.assertFalse(summary["all_valid"])
                self.assertTrue(rows)
                self.assertTrue(all(not r["ok"] for r in rows))
                self.assertLessEqual(len(rows), 8)
        finally:
            Handler.fail = False

    def test_startup_crash_releases_other_processes_from_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            start = time.monotonic()
            with mock.patch.object(smoke, "_client_process_entry", fail_one_client):
                with self.assertRaisesRegex(RuntimeError, "load client process failed"):
                    self.run_load(out)
            self.assertLess(time.monotonic() - start, 10)
            pids = json.loads((out / "steady_c8-processes/processes.json").read_text())[
                "pids"
            ]
            for pid in pids:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)


class StreamingBufferTest(unittest.TestCase):
    def test_actual_parser_preserves_split_utf8_and_parses_before_done(self):
        import requests

        text = "中🧪文" * 3000
        parsed = threading.Event()
        observations = []
        first = (
            "data: "
            + json.dumps(
                dict(
                    choices=[
                        dict(index=0, delta=dict(content=text), finish_reason=None)
                    ]
                ),
                ensure_ascii=False,
            )
            + "\n\n"
        ).encode()
        last = (
            "data: "
            + json.dumps(
                dict(
                    choices=[dict(index=0, delta={}, finish_reason="length")],
                    usage=dict(
                        prompt_tokens=24601, completion_tokens=4, total_tokens=24605
                    ),
                    aux_info=dict(pd_sep=False, reuse_len=0, output_len=4),
                )
            )
            + "\n\ndata: [DONE]\n\n"
        ).encode()
        split_utf8 = first.index("🧪".encode()) + 1

        class StreamingHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()
                try:
                    for begin, end in (
                        (0, split_utf8),
                        (split_utf8, 4094),
                        (4094, len(first) - 1),
                        (len(first) - 1, len(first)),
                    ):
                        piece = first[begin:end]
                        self.wfile.write(
                            f"{len(piece):x}\r\n".encode() + piece + b"\r\n"
                        )
                        self.wfile.flush()
                    # No [DONE] until the actual request parser decoded the
                    # first JSON event. A buffering regression times out.
                    observations.append(parsed.wait(timeout=3))
                    self.wfile.write(
                        f"{len(last):x}\r\n".encode() + last + b"\r\n0\r\n\r\n"
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        original_loads, original_post = json.loads, requests.post

        def observe(value, *args, **kwargs):
            decoded = original_loads(value, *args, **kwargs)
            if decoded.get("choices", [{}])[0].get("delta", {}).get("content") == text:
                parsed.set()
            return decoded

        def bounded_post(*args, **kwargs):
            kwargs["timeout"] = 2
            return original_post(*args, **kwargs)

        options = dict(smoke.CLIENT_OPTIONS)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StreamingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for chunk in (1, 4096):
                parsed.clear()
                smoke.CLIENT_OPTIONS.update(
                    mode="measured", chunk_size=chunk, profile_requests=0
                )
                with mock.patch.object(
                    json, "loads", side_effect=observe
                ), mock.patch.object(requests, "post", side_effect=bounded_post):
                    result = smoke.request(
                        f"http://127.0.0.1:{server.server_port}/",
                        {},
                        0,
                        dict(
                            prompt_tokens=24601, video_tokens=0, fixed_output_tokens=4
                        ),
                    )
                self.assertTrue(result["ok"], result.get("validation_errors"))
                self.assertEqual(result["content"], text)
                self.assertTrue(parsed.is_set())
            self.assertEqual(observations, [True, True])
        finally:
            smoke.CLIENT_OPTIONS.clear()
            smoke.CLIENT_OPTIONS.update(options)
            server.shutdown()
            server.server_close()
            thread.join()


class LongBudgetTest(unittest.TestCase):
    def args(self):
        return [
            "smoke",
            "--output",
            "/tmp/not-launched-budget-test",
            "--workload",
            "text",
            "--gpus",
            "4,5,6,7",
            "--moe-strategy",
            "mega_moe_fp8",
            "--decode-graph",
            "1",
            "--fp8-kv-cache",
            "1",
            "--native-fp8-attn",
            "1",
            "--seq-size-per-block",
            "4096",
            "--rank-concurrency",
            "256",
            "--graph-batches",
            "1,2,4,8,16,32,64,96,99,128,160,192,196,224,256",
            "--steady-concurrency",
            "768,784",
            "--steady-warmup-seconds",
            "240",
            "--steady-window-seconds",
            "900",
            "--steady-windows",
            "4",
            "--client-mode",
            "measured",
            "--client-processes",
            "4",
        ]

    def test_old_budget_rejects_long_plan_before_launch(self):
        with mock.patch.object(sys, "argv", self.args()), mock.patch.object(
            smoke, "run_launcher"
        ) as launch:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(
                SystemExit
            ) as error:
                smoke.main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("exceeds the configured test budget", stderr.getvalue())
            launch.assert_not_called()

    def test_explicit_budget_reaches_bazel_and_worker_with_graph_fp8_preserved(self):
        with mock.patch.object(
            sys, "argv", self.args() + ["--test-timeout-seconds", "14400"]
        ), mock.patch.object(smoke, "run_launcher", side_effect=smoke.bazel_command):
            command = smoke.main()
        self.assertIn("--test_timeout=14400", command)
        self.assertNotIn("--test_timeout=7200", command)
        for name, value in [
            ("test-timeout-seconds", "14400"),
            ("decode-graph", "1"),
            ("fp8-kv-cache", "1"),
            ("native-fp8-attn", "1"),
            ("seq-size-per-block", "4096"),
            ("steady-window-seconds", "900"),
        ]:
            self.assertIn("--test_arg=--" + name + "=" + value, command)


class EncoderTopologyTest(unittest.TestCase):
    def test_bazel_target_resource_declarations_match_worker_args(self):
        targets = []
        scope = dict(
            load=lambda *a: None,
            SMOKE_FRAMEWORK_DEPS=[],
            native=SimpleNamespace(py_test=lambda **kw: targets.append(kw)),
        )
        exec(PATH.with_suffix(".bzl").read_text(), scope)
        scope["qwen35_epd_fusion_4gpu_suite"]()
        for count in (1, 2):
            target = next(
                t for t in targets if t["name"] == f"qwen35_e{count}pd4_smoke"
            )
            self.assertEqual(target["exec_properties"]["gpu_count"], str(count + 4))
            self.assertEqual(target["env"]["GPU_COUNT"], str(count + 4))
            self.assertEqual(target["env"]["WORLD_SIZE"], str(count + 4))
            self.assertIn(
                "--encoder-gpus=" + ("0" if count == 1 else "0,1"), target["args"]
            )

    def test_proxy_configs_and_five_six_gpu_resources(self):
        ports = dict(encoder_0=20000, encoder_1=21000, fusion=22000)
        env, args = smoke.server_config("4,5,6,7", "baseline", "mega_moe_fp8", 1)
        for ids, count in [("0", 1), ("0,1", 2)]:
            configs = smoke.dp2_service_configs(
                "4,5,6,7", ids, env, args, ports, 9, "candidate-a", 137438953472
            )
            enc, pd = configs
            self.assertEqual(enc["name"], f"encoder_dp{count}")
            self.assertEqual(enc["gpus"], list(range(count)))
            for field in ("WORLD_SIZE", "TP_SIZE", "DP_SIZE", "EP_SIZE"):
                self.assertEqual(enc["env"][field], "1")
            for field in ("GPU_COUNT", "VIT_SERVER_COUNT"):
                self.assertEqual(enc["env"][field], str(count))
            self.assertIn(f"--vit_server_count {count}", enc["args"])
            self.assertEqual(
                enc["env"].get("RTP_VIT_SINGLE_WORKER_PROXY"),
                "1" if count == 1 else None,
            )
            self.assertEqual(enc["env"]["CUDA_VISIBLE_DEVICES"], ids)
            self.assertEqual(pd["env"]["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
            route = json.loads(pd["env"]["MODEL_SERVICE_CONFIG"])["role_endpoints"][0]
            self.assertEqual(route["vit_endpoint"]["address"], "127.0.0.1:20000")
            for field in ("MM_CACHE_CPU_MAX_BYTES", "MM_CACHE_GPU_MAX_BYTES"):
                self.assertEqual(enc["env"][field], "0")
            a = SimpleNamespace(
                gpus="4,5,6,7",
                encoder_gpus=ids,
                encoder_dp2=1,
                cache_root="/tmp/cache",
                model_dir="/tmp/model",
                data_dir="/tmp/data",
                output="/tmp/out",
                profile="baseline",
                perf_repeats=0,
                bazel_option=[],
            )
            command = smoke.bazel_command(a)
            self.assertIn(
                smoke.SINGLE_ENCODER_TARGET if count == 1 else smoke.ENCODER_TARGET,
                command,
            )
            self.assertIn(f"--test_env=GPU_COUNT={count+4}", command)
            self.assertIn(f"--test_env=WORLD_SIZE={count+4}", command)
            self.assertIn(f"--test_arg=--encoder-gpus={ids}", command)
            self.assertIn("--run_under=//rtp_llm/test/utils:gpu_lock", command)
        with self.assertRaises(ValueError):
            smoke.encoder_gpu_ids("")

    def test_worker_evidence_does_not_require_absent_worker_or_accept_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            logs = root / "encoder_dp1_logs"
            logs.mkdir()
            access = logs / "mm_access_r0_s0.log"
            access.write_text(
                json.dumps(
                    dict(exception=None, response="MMEmbeddingRes(shape=[20424,4096])")
                )
                + "\n"
            )
            evidence = smoke.encoder_log_evidence(root, 1)
            self.assertEqual(set(evidence), {"encoder_0"})
            self.assertTrue(all(evidence.values()))
            access.write_text(
                json.dumps(dict(exception="failed", response="MMEmbeddingRes")) + "\n"
            )
            self.assertFalse(all(smoke.encoder_log_evidence(root, 1).values()))

    def test_single_proxy_steady_cli_alias(self):
        argv = [
            "smoke",
            "--output",
            "/tmp/not-launched-encoder-test",
            "--gpus",
            "4,5,6,7",
            "--encoder-gpus",
            "0",
            "--encoder-proxy",
            "1",
            "--steady-concurrency",
            "512",
            "--rank-concurrency",
            "256",
            "--client-mode",
            "measured",
            "--client-processes",
            "4",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            smoke, "run_launcher", side_effect=smoke.bazel_command
        ):
            command = smoke.main()
        self.assertIn(smoke.SINGLE_ENCODER_TARGET, command)
        self.assertIn("--test_arg=--encoder-dp2=1", command)

    def test_real_spawn_function_keeps_single_proxy_and_worker_route(self):
        # Execute the actual launcher function with process creation mocked, avoiding
        # model/CUDA imports while checking worker arguments and proxy endpoints.
        import ast
        import logging

        start_path = PATH.parents[2] / "start_server.py"
        tree = ast.parse(start_path.read_text())
        func = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "start_vit_server_impl"
        )
        func.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
        process = mock.Mock()
        scope = dict(
            PyEnvConfigs=object,
            ProcessManager=object,
            logging=logging,
            os=os,
            load_gpu_nic_affinity=mock.Mock(),
            torch=SimpleNamespace(multiprocessing=SimpleNamespace(Process=process)),
        )
        exec(compile(module, str(start_path), "exec"), scope)
        vit = mock.Mock()
        proxy = mock.Mock()
        modules = {
            "rtp_llm.multimodal.vit_start_server": SimpleNamespace(
                vit_start_server=vit
            ),
            "rtp_llm.multimodal.vit_proxy_start_server": SimpleNamespace(
                vit_proxy_start_server=proxy
            ),
        }
        for count, enabled, expected in [(1, "0", 1), (1, "1", 2), (2, "0", 3)]:
            process.reset_mock()
            config = SimpleNamespace(
                server_config=SimpleNamespace(
                    start_port=20000,
                    vit_server_count=count,
                    rpc_server_port=20001,
                    server_port=20000,
                ),
                vit_config=SimpleNamespace(
                    output_transport=SimpleNamespace(rdma=SimpleNamespace(port=22000))
                ),
            )
            with mock.patch.dict(sys.modules, modules), mock.patch.dict(
                os.environ, {"RTP_VIT_SINGLE_WORKER_PROXY": enabled}
            ):
                result = scope["start_vit_server_impl"](config)
            self.assertEqual(len(result), expected)
            self.assertEqual(process.call_count, expected)
            if expected > 1:
                calls = process.call_args_list
                self.assertEqual(
                    [c.kwargs["args"][0] for c in calls[:-1]], list(range(count))
                )
                self.assertTrue(all(c.kwargs["args"][4] is True for c in calls[:-1]))
                self.assertEqual(
                    calls[-1].kwargs["args"][1],
                    [f"127.0.0.1:{20002+2*i}" for i in range(count)],
                )
                self.assertEqual(
                    [c.kwargs["args"][5] for c in calls[:-1]],
                    [22000 + i for i in range(count)],
                )


if __name__ == "__main__":
    unittest.main()
