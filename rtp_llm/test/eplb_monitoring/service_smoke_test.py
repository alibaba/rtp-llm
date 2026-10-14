"""Exercise monitoring through the real engine; save counts and paired timings."""

import argparse
import json
import math
import os
import shlex
import time
import unittest
from collections import Counter
from pathlib import Path

import requests


class ExpertLoadServiceTest(unittest.TestCase):
    @staticmethod
    def _make_manager(**kwargs):
        # Keep harness control-flow tests independent of native model libraries.
        from rtp_llm.test.utils.maga_server_manager import MagaServerManager

        return MagaServerManager(**kwargs)

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = os.environ["EXPERT_LOAD_TEST_CHECKPOINT"]
        cls.config = json.loads((Path(cls.checkpoint) / "config.json").read_text())
        if cls.config["model_type"] != "deepseek_v2":
            raise ValueError("This conservation check requires a DeepSeek-V2 model")
        cls.output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]).resolve()
        cls.world = int(os.environ.get("WORLD_SIZE", "1"))
        topology = argparse.ArgumentParser(add_help=False)
        for name in ("tp_size", "ep_size", "world_size"):
            topology.add_argument("--" + name, type=int, default=1)
        topology.add_argument("--cp_rotate_method")
        sizes, _ = topology.parse_known_args(shlex.split(os.environ["SMOKE_ARGS"]))
        if cls.world not in (1, 2) or any(
            getattr(sizes, name) != cls.world
            for name in ("tp_size", "ep_size", "world_size")
        ):
            raise ValueError("This count oracle requires TP=EP=WORLD of 1 or 2")
        if sizes.cp_rotate_method or os.environ.get("CP_ROTATE_METHOD"):
            raise ValueError("This count oracle requires CP to remain disabled")
        if cls.config.get("moe_layer_freq", 1) != 1:
            raise ValueError("This count oracle requires consecutive MoE layers")
        cls.topk = cls.config["num_experts_per_tok"]
        cls.layer_count = cls.config["num_hidden_layers"]
        cls.dense_layers = cls.config.get("first_k_dense_replace", 0)
        cls.prompt = "Explain why the sum of two even integers is even. " * 16
        cls.run_id = 0

    def _request(self, manager, length):
        generate_config = {
            "max_new_tokens": length,
            "min_new_tokens": length,
            "top_k": 1,
            "random_seed": 17,
            "return_output_ids": True,
        }
        started = time.monotonic()
        response = requests.post(
            f"http://127.0.0.1:{manager.port}/",
            json={
                "prompt": self.prompt,
                "top_k": 1,
                "generate_config": generate_config,
            },
            timeout=600,
        )
        response.raise_for_status()
        body = response.json()
        # End request timing before any diagnostic file I/O, identically for
        # OFF and ON. Preserve even a malformed response before assertions.
        wall_s = time.monotonic() - started
        entry = {
            **self._request_context,
            "request": {"prompt": self.prompt, "generate_config": generate_config},
            "response": body,
            "wall_s": wall_s,
        }
        with self._journal.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        self.assertTrue(body.get("finished"), body)
        aux = body["aux_info"]
        if isinstance(aux, list):
            self.assertEqual(len(aux), 1)
            aux = aux[0]
        self.assertEqual(aux["output_len"], length)
        sample = {
            "prefill_ms": aux["first_token_cost_time"] - aux["wait_time"],
            "decode_ms_per_token": (
                (aux["cost_time"] - aux["first_token_cost_time"]) / (length - 1)
                if length > 1
                else 0.0
            ),
            "throughput_tokens_per_s": length / wall_s,
            "wall_s": wall_s,
            "input_len": aux["input_len"],
            "output_len": aux["output_len"],
        }
        ids = body.get("output_ids")
        self.assertIsInstance(ids, list)
        self.assertEqual(len(ids), 1)
        self.assertIsInstance(ids[0], list)
        self.assertEqual(len(ids[0]), length)
        self.assertTrue(
            all(
                isinstance(token, int)
                and not isinstance(token, bool)
                and 0 <= token < self.config["vocab_size"]
                for token in ids[0]
            ),
            ids,
        )
        self.assertIsInstance(body["response"], str)
        sample.update(
            self._request_context,
            output_text=body["response"],
            output_ids=ids[0],
        )
        for field in (
            "prefill_ms",
            "decode_ms_per_token",
            "throughput_tokens_per_s",
        ):
            self.assertTrue(math.isfinite(sample[field]) and sample[field] > 0, sample)
        return body["response"], sample

    def _artifacts(self, folder):
        artifacts = []
        for path in folder.rglob("*.json"):
            item = json.loads(path.read_text())
            if item.get("schema_version") == 1 and "rank_history" in item:
                artifacts.append(item)
        return sorted(artifacts, key=lambda x: (x["world_rank"], x["step_begin"]))

    def _profile(self, manager, role, graph, asynchronous):
        trace_name = "eplb_smoke_" + role
        response = requests.post(
            f"http://127.0.0.1:{manager.port}/start_profile",
            json={
                "gen_timeline": True,
                "trace_name": trace_name,
                "start_step": 0,
                "num_steps": 6,
                "enable_all_rank": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        self.assertEqual(response.json().get("status"), "ok", response.text)
        self._request_context = {**self._request_context, "phase": "profile"}
        self._request(manager, 16)
        deadline = time.monotonic() + 60
        names = set()
        paths = []
        while time.monotonic() < deadline:
            paths = list(self.output.rglob(trace_name + "*.json"))
            try:
                events = [
                    event
                    for path in paths
                    for event in json.loads(path.read_text())["traceEvents"]
                ]
                names = {event.get("name", "") for event in events}
                if names and len(paths) >= self.world:
                    break
            except (ValueError, KeyError):
                pass
            time.sleep(0.25)
        self.assertTrue(names, "Profiler did not export a complete trace")
        self.assertEqual(len(paths), self.world)
        for path in paths:
            counts = Counter(
                event.get("name", "")
                for event in json.loads(path.read_text())["traceEvents"]
                if event.get("ph") == "X"
            )
            self.assertEqual(counts["py_model.forward"], 6, str(path))
            self.assertEqual(counts["expert_load_monitor.flush"], 3, str(path))
            self.assertEqual(
                counts["expert_load_monitor.copy_to_host"],
                3 if "_wr0_" in path.name else 0,
                str(path),
            )
            self.assertEqual(
                counts["expert_load_monitor.all_reduce"],
                3 if self.world > 1 else 0,
                str(path),
            )
        replay = "cuda_graph.forward(replayDecode)"
        self.assertEqual(replay in names, graph, sorted(names))
        if asynchronous:
            self.assertTrue(
                {
                    "executor.dispatch_output(stream_async)",
                    "executor.dispatch_output_async",
                }
                & names,
                "Requested async execution was absent from the trace",
            )
        return [str(p.relative_to(self.output)) for p in paths]

    def _check_counts(
        self,
        artifacts,
        graph,
        input_len,
        requested_steps,
        request_count,
        last_output_len,
    ):
        from rtp_llm.test.eplb_monitoring.artifact_validator import (
            validate_window_sequence,
        )

        self.assertTrue(artifacts, "Enabled monitor produced no windows")
        by_rank = {}
        for item in artifacts:
            by_rank.setdefault(item["world_rank"], []).append(item)
        for windows in by_rank.values():
            self.assertGreaterEqual(len(windows), 2)
            validation = validate_window_sequence(windows)
            self.assertTrue(validation["ok"], validation)
            self.assertEqual(windows[0]["step_begin"], 1)
            observed_totals = []
            for window in windows:
                self.assertEqual(window["layer_count"], self.layer_count)
                for step in window["rank_history"]:
                    totals = [sum(ranks) for ranks in step]
                    self.assertEqual(
                        totals[: self.dense_layers], [0] * self.dense_layers
                    )
                    routed = totals[self.dense_layers :]
                    observed_totals.append(routed[0])
                    self.assertGreater(routed[0], 0)
                    self.assertEqual(routed, [routed[0]] * len(routed))
                    self.assertEqual(routed[0] % self.topk, 0)
                    # With DP1 and CP disabled, DeepEP partitions the TP
                    # replicas into disjoint source rows. The global count is
                    # T*K in both EP1 and EP2, including graph bucket padding.
                    self.assertIn(
                        routed[0],
                        {input_len * self.topk, (2 if graph else 1) * self.topk},
                    )
            first = windows[0]["rank_history"][0][self.dense_layers]
            self.assertEqual(sum(first), input_len * self.topk)
            # Async scheduling may submit an extra decode before completion
            # is dispatched. At most W-1 forwards remain in the unflushed tail.
            self.assertGreaterEqual(
                len(observed_totals),
                requested_steps - windows[-1]["window_capacity"] + 1,
            )
            self.assertIn((2 if graph else 1) * self.topk, observed_totals)
            if last_output_len >= windows[-1]["window_capacity"]:
                self.assertEqual(
                    observed_totals.count(input_len * self.topk), request_count
                )
        # Some deployments emit only rank 0. When multiple ranks emit the same
        # reduced window, their raw data must agree exactly.
        reference = next(iter(by_rank.values()))
        for windows in by_rank.values():
            self.assertEqual(
                [(x["step_begin"], x["rank_history"]) for x in windows],
                [(x["step_begin"], x["rank_history"]) for x in reference],
            )

    def _run(
        self,
        enabled,
        graph,
        asynchronous,
        window,
        measure_count,
        profile,
        warmup_count=2,
    ):
        self.assertGreaterEqual(warmup_count, 1)
        type(self).run_id += 1
        role = (
            f"monitor_{self.run_id}_g{int(graph)}_a{int(asynchronous)}_{int(enabled)}"
        )
        snapshot = self.output / role / "windows"
        snapshot.mkdir(parents=True)
        traces = self.output / role / "traces"
        traces.mkdir()
        args = shlex.split(os.environ.get("SMOKE_ARGS", "")) + [
            "--tp_size",
            str(self.world),
            "--ep_size",
            str(self.world),
            "--world_size",
            str(self.world),
            "--dp_size",
            "1",
            "--enable_cuda_graph",
            str(int(graph)),
            "--decode_capture_config",
            "2",
            "--eplb_mode",
            "NONE",
            "--eplb_metrics_report_interval",
            str(window if enabled else 0),
            "--eplb_metrics_snapshot_dir",
            str(snapshot),
            "--eplb_metrics_snapshot_max_files",
            "10000",
            "--reuse_cache",
            "0",
            "--torch_cuda_profiler_dir",
            str(traces),
        ]
        manager = self._make_manager(
            role_name=role,
            env_args={
                "RTP_LLM_STREAM_ASYNC": str(int(asynchronous)),
                # Async must enable device inputs even with an explicit opt-out.
                "RTP_LLM_DEVICE_INPUT": "0",
                "RTP_LLM_DROP_BROAD_SYNC": "0",
                # These isolated services have no load-balancer convergence
                # window. Keep orderly shutdown without the production linger.
                "RTP_LLM_BACKEND_POST_FRONTEND_DRAIN_SECONDS": "0",
                "FRONTEND_PRE_STOP_DRAIN_SECONDS": "0",
                "DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "0",
            },
            smoke_args_str=shlex.join(args),
        )
        result = {
            "mode": "on" if enabled else "off",
            "graph": graph,
            "async": asynchronous,
            "window": window,
            "warmup_count": warmup_count,
            "measure_count": measure_count,
            "samples": [],
            "cycle": getattr(self, "_cycle", None),
            "run_id": self.run_id,
            "role": role,
            "drop_broad_sync": 0,
        }
        self._journal = self.output / f"{role}_requests.jsonl"

        def request(phase, index):
            self._request_context = {
                "cycle": result["cycle"],
                "run_id": self.run_id,
                "role": role,
                "mode": result["mode"],
                "phase": phase,
                "phase_index": index,
                "request_index": index + (warmup_count if phase == "measure" else 0),
            }
            return self._request(manager, 32)

        try:
            self.assertTrue(manager.start_server(self.checkpoint, "deepseek2"))
            baseline_output, first_sample = request("warmup", 0)
            result["output"] = baseline_output
            result["output_ids"] = first_sample["output_ids"]
            for index in range(1, warmup_count):
                output, sample = request("warmup", index)
                self.assertEqual(sample["input_len"], first_sample["input_len"])
                self.assertEqual(output, baseline_output)
                self.assertEqual(sample["output_ids"], first_sample["output_ids"])
            before = {
                (x["world_rank"], x["step_begin"]) for x in self._artifacts(snapshot)
            }
            for index in range(measure_count):
                output, sample = request("measure", index)
                self.assertEqual(sample["input_len"], first_sample["input_len"])
                self.assertEqual(output, baseline_output)
                self.assertEqual(sample["output_ids"], first_sample["output_ids"])
                result["samples"].append(sample)
            measured = self._artifacts(snapshot)
            result["window_count"] = sum(
                x["world_rank"] == 0
                and (x["world_rank"], x["step_begin"]) not in before
                for x in measured
            )
            if profile:
                result["traces"] = self._profile(manager, role, graph, asynchronous)
            artifacts = self._artifacts(snapshot)
            if enabled:
                self._check_counts(
                    artifacts,
                    graph,
                    first_sample["input_len"],
                    32 * (warmup_count + measure_count) + (16 if profile else 0),
                    warmup_count + measure_count + int(profile),
                    16 if profile else 32,
                )
            else:
                self.assertEqual(artifacts, [])
            result["output"] = baseline_output
            result["server_pid"] = manager.server_pid
            return result
        finally:
            manager.stop_server()
            (self.output / f"{role}_result.json").write_text(
                json.dumps(result, indent=2)
            )

    def test_engine(self):
        mode = os.environ.get("EXPERT_LOAD_TEST_MODE", "functional")
        if mode == "functional":
            cases = [(False, False), (False, True), (True, False), (True, True)]
            requested = os.environ.get("EXPERT_LOAD_TEST_CASES")
            if requested:
                cases = [
                    tuple(bool(int(x)) for x in case.split(":"))
                    for case in requested.split(",")
                ]
            for graph, asynchronous in cases:
                with self.subTest(graph=graph, asynchronous=asynchronous):
                    off = self._run(False, graph, asynchronous, 2, 2, False)
                    on = self._run(True, graph, asynchronous, 2, 2, True)
                    self.assertEqual(on["output"], off["output"])
                    self.assertEqual(on["output_ids"], off["output_ids"])
        elif mode == "performance":
            from rtp_llm.test.eplb_monitoring.perf_analyzer import analyze_abba

            window = int(os.environ.get("EXPERT_LOAD_TEST_WINDOW", "100"))
            self.assertGreater(window, 0)
            warmups = int(os.environ.get("EXPERT_LOAD_TEST_WARMUPS", "8"))
            samples = int(os.environ.get("EXPERT_LOAD_TEST_SAMPLES", "64"))
            self.assertGreaterEqual(warmups, 8)
            self.assertGreaterEqual(samples, 64)
            # Every measured run covers at least three complete windows.
            samples = max(samples, (3 * window + 31) // 32)
            cycles = int(os.environ.get("EXPERT_LOAD_TEST_CYCLES", "3"))
            self.assertGreaterEqual(cycles, 3)
            graph = os.environ.get("EXPERT_LOAD_TEST_GRAPH", "1") == "1"
            asynchronous = os.environ.get("EXPERT_LOAD_TEST_ASYNC", "1") == "1"
            records = []
            for cycle in range(cycles):
                self._cycle = cycle
                for enabled in (False, True, True, False):
                    record = self._run(
                        enabled,
                        graph,
                        asynchronous,
                        window,
                        samples,
                        False,
                        warmup_count=warmups,
                    )
                    record["cycle"] = cycle
                    records.append(record)
                    self.assertEqual(record["output"], records[0]["output"])
                    self.assertEqual(record["output_ids"], records[0]["output_ids"])
                    (self.output / "abba_raw.json").write_text(
                        json.dumps(records, indent=2)
                    )
            report = analyze_abba(records, regression_threshold=0.03)
            (self.output / "abba_analysis.json").write_text(
                json.dumps(report, indent=2)
            )
            print(json.dumps(report, indent=2))
            self.assertEqual(report["conclusion"], "ok", report)
        else:
            self.fail(f"Unknown EXPERT_LOAD_TEST_MODE: {mode}")


if __name__ == "__main__":
    unittest.main()
