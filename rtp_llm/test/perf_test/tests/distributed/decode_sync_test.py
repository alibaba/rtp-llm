"""Performance framework behavior and failure regressions."""

import copy
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.config_schema import ConfigError
from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.entry import plan_from_source
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.execution.distributed.decode import execute_master_decode
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.fixtures import (
    fixture_config,
    fixture_execution,
    fixture_model,
)
from rtp_llm.test.perf_test.tests.support.results import default_executor_request_round


def decode_multi_node_decode_source(*, inputs=(128,), outputs=(16, 64), batch=None):
    return fixture_config(
        model=fixture_model(
            label="external-decode", topology={"tp": 1, "dp": 2, "pp": 1, "ep": 2}
        ),
        workload={
            "kind": "grid",
            "input_lengths": list(inputs),
            "output_lengths": list(outputs),
            "batch": batch or {"strategy": "fixed", "sizes": [2, 4]},
            "measure_runs": 1,
            "warmup_runs": 0,
        },
        execution=fixture_execution(devices=[0, 1]),
    )


def decode_multi_node_write_decode_config(root, *, dp=2):
    source = decode_multi_node_decode_source()
    source["model"]["topology"].update(dp=dp, ep=dp)
    source["execution"]["devices"] = list(range(dp))
    source["output"]["result_root"] = str(root / "results")
    path = root / f"decode-dp{dp}.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


decode_multi_node_TOPOLOGY = {"tp": 1, "dp": 2, "pp": 1, "ep": 2}


class DistributedRuntimeTest(unittest.TestCase):

    def test_cluster_reports_ready_worker_failure_without_waiting_for_silent_peer(self):
        from rtp_llm.test.perf_test.execution.distributed.decode_sync import (
            ClusterDecodeTransport,
        )

        cluster = ClusterDecodeTransport(
            object(),
            bind_host="127.0.0.1",
            control_port=0,
            local_world_size=1,
            tp_size=1,
            dp_size=3,
            member_world_ranks=(0, 1, 2),
            timeout_seconds=0.1,
        )
        (silent_master, silent_peer) = socket.socketpair()
        (failed_master, failed_peer) = socket.socketpair()
        for sock in (silent_master, failed_master):
            sock.settimeout(0.1)
        cluster._peers = {
            1: (silent_master, silent_master.makefile("rw", encoding="utf-8")),
            2: (failed_master, failed_master.makefile("rw", encoding="utf-8")),
        }
        cluster._sequence = 1
        failed_peer.sendall(b'{"sequence":1,"status":"error","error":"boom"}\n')
        try:
            with self.assertRaisesRegex(RuntimeError, "worker 2 failed: boom"):
                cluster._responses()
        finally:
            cluster.close()
            silent_peer.close()
            failed_peer.close()

    def test_decode_worker_rejects_rank_outside_dp_layout_before_connecting(self):
        from rtp_llm.test.perf_test.execution.distributed.decode_sync import (
            serve_decode_worker,
        )

        with self.assertRaisesRegex(ValueError, "layout"):
            serve_decode_worker(
                "127.0.0.1",
                1,
                world_rank=3,
                local_world_size=1,
                tp_size=1,
                dp_size=2,
                local_transport=object(),
                timeout_seconds=0,
            )

    def test_cluster_transport_runs_one_local_batch_per_dp_rank(self):
        from concurrent.futures import ThreadPoolExecutor

        from rtp_llm.test.perf_test.execution.distributed.decode_sync import (
            ClusterDecodeTransport,
            serve_decode_worker,
        )

        class LocalTransport:

            def __init__(self):
                self.configured = []
                self.rounds = []

            def configure_scheduler(self, allocations):
                self.configured.append(tuple(allocations))

            def run_round(self, *, input_lengths, output_length, generate_config):
                self.rounds.append(tuple(input_lengths))
                return default_executor_request_round(
                    input_lengths,
                    output_length,
                    latency={
                        "wait": 0.0,
                        "prefill": 1.0,
                        "decode": 4.0,
                        "tpot": 1.0,
                        "total": 5.0,
                    },
                )

        master_local = LocalTransport()
        worker_local = LocalTransport()
        cluster = ClusterDecodeTransport(
            master_local,
            bind_host="127.0.0.1",
            control_port=0,
            local_world_size=1,
            tp_size=1,
            dp_size=2,
            member_world_ranks=(0, 1),
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            worker = pool.submit(
                serve_decode_worker,
                "127.0.0.1",
                cluster.port,
                world_rank=1,
                local_world_size=1,
                tp_size=1,
                dp_size=2,
                local_transport=worker_local,
            )
            try:
                cluster.connect_workers()
                allocations = [
                    {
                        "worker_index": 0,
                        "local_batch_size": 2,
                        "request_indices": [0, 1],
                        "input_lengths": [10, 11],
                    },
                    {
                        "worker_index": 1,
                        "local_batch_size": 2,
                        "request_indices": [2, 3],
                        "input_lengths": [12, 13],
                    },
                ]
                cluster.configure_scheduler(allocations)
                outcomes = cluster.run_round(
                    input_lengths=(10, 11, 12, 13), output_length=8, generate_config={}
                )
                self.assertEqual([0, 1, 2, 3], [r["request_index"] for r in outcomes])
                self.assertEqual([(10, 11)], master_local.rounds)
                self.assertEqual([(12, 13)], worker_local.rounds)
                self.assertEqual(
                    2, len(master_local.configured[0][0]["request_indices"])
                )
                self.assertEqual(
                    2, len(worker_local.configured[0][0]["request_indices"])
                )
                from rtp_llm.test.perf_test.decode.scheduler import (
                    schedule_decode_requests,
                )

                small = schedule_decode_requests((17,), dp_size=2)
                with self.assertRaisesRegex(PlanningError, "positive"):
                    cluster.configure_scheduler(
                        tuple((w.to_dict() for w in small.workers))
                    )
                with self.assertRaisesRegex(PlanningError, "profiling requires DP=1"):
                    cluster.start_profile(
                        trace_name="unused", steps=1, enable_all_ranks=False
                    )
                self.assertEqual([(10, 11)], master_local.rounds)
                self.assertEqual([(12, 13)], worker_local.rounds)
            finally:
                cluster.close()
            worker.result(timeout=5)

    def test_master_executes_grid_and_writes_standard_artifacts_and_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = decode_multi_node_write_decode_config(root)
            source = json.loads(path.read_text())
            source["workload"]["profiling"] = {"enabled": True}
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ConfigError, "profiling requires DP=1"):
                plan_from_source(
                    json.loads(Path(path).read_text()),
                    source_dir=Path(path).parent,
                    result_directory=str(str(root / "run")),
                )
            topology = {"tp": 2, "dp": 1, "pp": 1, "ep": 1}
            source["model"]["topology"] = topology
            path.write_text(json.dumps(source))
            (plan, source) = plan_from_source(
                json.loads(Path(path).read_text()),
                source_dir=Path(path).parent,
                result_directory=str(str(root / "run")),
            )
            log = root / "server.log"
            requests = []

            class Response:
                status_code = 200

                def __init__(self, body):
                    self.body = body

                def json(self):
                    return self.body

            def post(url, *, json, **_kwargs):
                requests.append((url, copy.deepcopy(json)))
                if url.endswith("/start_profile"):
                    for rank in range(topology["tp"]):
                        trace = root / (json["trace_name"] + f"_wr{rank}_1.json")
                        trace.write_text('{"traceEvents": []}')
                        with log.open("a") as stream:
                            stream.write(f"profiler trace saved: {trace}\n")
                    return Response({"status": "ok"})
                if url.endswith("/update_scheduler_info"):
                    return Response({"status": "ok"})
                return Response(
                    {
                        "aux_info": {
                            "input_len": int(json["prompt"].split("-")[1]),
                            "output_len": json["generate_config"]["max_new_tokens"],
                            "wait_time": 1.0,
                            "first_token_cost_time": 3.0,
                            "cost_time": 18.0,
                        }
                    }
                )

            def transport(**kwargs):
                return DecodeHttpTransport(**kwargs, post=post)

            with patch(
                "rtp_llm.test.perf_test.execution.distributed.decode.DecodeHttpTransport",
                side_effect=transport,
            ):
                result = execute_master_decode(
                    plan,
                    source_config=source,
                    endpoint="http://external:12333",
                    prompt_for_length=lambda length: f"prompt-{length}",
                    server_log_path=str(log),
                    profiler_directory=root / "traces",
                )
            run = root / "run"
            self.assertEqual(RunState.COMPLETED, result.state)
            self.assertEqual(
                "completed", json.loads((run / "manifest.json").read_text())["state"]
            )
            envelope = json.loads((run / "results/result.json").read_text())
            self.assertEqual(4, envelope["summary"]["successful_cases"])
            self.assertEqual(4, len(list((run / "results/cases").glob("*.json"))))
            self.assertEqual(4, len(list((run / "profiles").glob("*/manifest.json"))))
            self.assertEqual(
                "formal",
                json.loads((run / "reports/decode_report.json").read_text())[
                    "report_kind"
                ],
            )
            self.assertTrue((run / "source_config.json").is_file())
            self.assertTrue((run / "resolved_config.json").is_file())
            self.assertTrue((run / "execution_plan.json").is_file())
            self.assertTrue((run / "results/summary.json").is_file())
            self.assertTrue(
                all((url.startswith("http://external:12333") for (url, _) in requests))
            )
            measured_outputs = {
                payload["generate_config"]["max_new_tokens"]
                for (url, payload) in requests
                if url == "http://external:12333"
            }
            self.assertEqual({16, 64}, measured_outputs)
            self.assertTrue(
                all(
                    (
                        case["metrics"]["topology"] == topology
                        for case in envelope["cases"]
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
