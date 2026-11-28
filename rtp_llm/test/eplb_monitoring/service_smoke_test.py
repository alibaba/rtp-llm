"""Exercise monitoring through the real engine; save counts and outputs."""

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


def _window_key(window):
    return (
        window["model_id"],
        window["model_type"],
        window["world_rank"],
        window["step_begin"],
    )


class ExpertLoadServiceTest(unittest.TestCase):
    @staticmethod
    def _make_manager(**kwargs):
        # Keep smoke setup independent of native model libraries.
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
            if (
                item.get("schema_version") == 4
                and item.get("model_type") == "target"
                and "rank_samples" in item
            ):
                artifacts.append(item)
        return sorted(artifacts, key=_window_key)

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
        self.assertTrue(artifacts, "Enabled monitor produced no windows")
        self.assertTrue(all(item["world_rank"] == 0 for item in artifacts))
        self.assertGreaterEqual(len(artifacts), 2)
        ordered = sorted(artifacts, key=lambda item: item["step_begin"])
        next_step = 1
        observed_totals = []
        for window in ordered:
            self.assertEqual(window["step_begin"], next_step, "missing forwards")
            self.assertEqual(window["sample_count"], window["window_capacity"])
            next_step += window["sample_count"]
            self.assertEqual(window["step_end"], next_step - 1)
            self.assertEqual(window["layer_count"], self.layer_count)
            self.assertEqual(window["world_size"], self.world)
            for index in range(window["sample_count"]):
                step_id = window["step_begin"] + index
                rows = [samples[index] for samples in window["rank_samples"]]
                totals = [
                    sum(row[layer] for row in rows) for layer in range(self.layer_count)
                ]
                self.assertEqual(totals[: self.dense_layers], [0] * self.dense_layers)
                routed = totals[self.dense_layers :]
                self.assertTrue(routed)
                self.assertGreater(routed[0], 0)
                self.assertEqual(routed, [routed[0]] * len(routed))
                self.assertEqual(routed[0] % self.topk, 0)
                # The controlled service runs one request at a time. Count
                # every executed assignment, including graph padding; there
                # is no separate source-token weight in the snapshot.
                self.assertIn(
                    routed[0],
                    {input_len * self.topk, (2 if graph else 1) * self.topk},
                )
                self.assertLessEqual(step_id, requested_steps + request_count)
                observed_totals.append(routed[0])
        capacity = ordered[-1]["window_capacity"]
        self.assertGreaterEqual(len(observed_totals), requested_steps - capacity + 1)
        self.assertIn((2 if graph else 1) * self.topk, observed_totals)
        # Only the final partial window may remain unpublished. A sufficiently
        # short window must include the prefill from every completed request.
        if capacity <= last_output_len:
            self.assertEqual(
                observed_totals.count(input_len * self.topk), request_count
            )

    def _check_phases(self, artifacts, graph, input_len):
        # With one request at a time and cache reuse disabled, the long prompt
        # identifies prefill; decode executes one token (two with graph padding).
        decode_tokens = 2 if graph else 1
        self.assertGreater(input_len, decode_tokens)
        counts = Counter()
        for window in artifacts:
            for index in range(window["sample_count"]):
                total = sum(
                    rank[index][self.dense_layers] for rank in window["rank_samples"]
                )
                if total == input_len * self.topk:
                    counts["prefill_steps"] += 1
                elif total == decode_tokens * self.topk:
                    counts["decode_steps"] += 1
        for phase in ("prefill", "decode"):
            self.assertGreater(
                counts[f"{phase}_steps"], 0, f"No new {phase} EPLB samples"
            )
        return dict(counts)

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
        server_work_dir = Path(os.environ.get("MAGA_SERVER_WORK_DIR", self.output))
        snapshot = server_work_dir / f"{role}_logs" / "eplb_snapshot"
        traces = self.output / role / "traces"
        traces.mkdir(parents=True)
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
                # Exercise host inputs in both sync and async modes.
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
            before = {_window_key(x) for x in self._artifacts(snapshot)}
            for index in range(measure_count):
                output, sample = request("measure", index)
                self.assertEqual(sample["input_len"], first_sample["input_len"])
                self.assertEqual(output, baseline_output)
                self.assertEqual(sample["output_ids"], first_sample["output_ids"])
                result["samples"].append(sample)
            measured = self._artifacts(snapshot)
            new_windows = [
                x
                for x in measured
                if x["world_rank"] == 0 and _window_key(x) not in before
            ]
            result["window_count"] = len(new_windows)
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
                result["phase_coverage"] = self._check_phases(
                    new_windows, graph, first_sample["input_len"]
                )
                if window <= 32:
                    self.assertEqual(
                        result["phase_coverage"]["prefill_steps"], measure_count
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


if __name__ == "__main__":
    unittest.main()
