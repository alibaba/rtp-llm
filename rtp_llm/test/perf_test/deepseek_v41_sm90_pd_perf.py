"""Manual real P/D benchmark: actual KV transfer and speculative output tokens.

Offered concurrency is pinned to decode ranks, not a constant GPU batch size.
Reuse the smoke lifecycle; never enable the synthetic batch decode scheduler.
"""

import concurrent.futures
import json
import logging
import os
import shlex
import statistics
import subprocess
import time
import unittest
from pathlib import Path

import requests

from rtp_llm.test.perf_test.batch_perf_impl import _wait_for_trace_flush
from rtp_llm.test.perf_test.cache_grid_runner import PrefixPromptFactory
from rtp_llm.test.perf_test.dataclass import ResponseInfo
from rtp_llm.test.smoke import deepseek_v41_sm90_pd_contract_test as pd


def ints(name, default):
    return [int(value) for value in os.environ.get(name, default).split(",")]


class DeepSeekV41Sm90PDPerf(pd.DeepSeekV41Sm90PDContractTest):
    def setUp(self):
        from rtp_llm.test.perf_test.deepseek_v41_sm90_role_test import (
            configure_jit_cache,
        )

        configure_jit_cache()
        self.gamma = int(os.environ.get("PD_PERF_GAMMA", "0"))
        self.batches = ints("PD_PERF_BATCHES", "1,4,16")
        self.lengths = ints("PD_PERF_LENGTHS", "1024,32768")
        # Keep deployment sizing fixed when selecting a shorter diagnostic grid.
        self.measure_lengths = ints(
            "PD_PERF_MEASURE_LENGTHS", ",".join(map(str, self.lengths))
        )
        assert set(self.measure_lengths).issubset(self.lengths)
        self.output_length = int(os.environ.get("PD_PERF_OUTPUT", "128"))
        self.rounds = int(os.environ.get("PD_PERF_ROUNDS", "3"))
        assert self.gamma >= 0 and min(self.batches + self.lengths) > 0
        assert self.output_length > 1 and self.rounds >= 1
        # The P frontend receives all DP ranks, so give it global request slots.
        self.prefill_concurrency_limit = max(self.batches) * self.decode_dp_size
        self.decode_capture_config = ",".join(map(str, self.batches))
        # Size each deployment for its actual target verification rows.
        self.extra_server_env = {
            **self.extra_server_env,
            "LOG_LEVEL": "INFO",
            "DSV41_SM90_DECODE_GRAPH_MAX_ROWS": str(
                int(
                    os.environ.get(
                        "PD_PERF_GRAPH_ROWS", str(max(self.batches) * (self.gamma + 1))
                    )
                )
            ),
        }
        assert max(self.batches) * (self.gamma + 1) <= int(
            self.extra_server_env["DSV41_SM90_DECODE_GRAPH_MAX_ROWS"]
        )
        super().setUp()
        for manager in (self.server, self.graph_server):
            response = self.session.post(
                f"http://127.0.0.1:{manager.port}/set_log_level",
                json={"log_level": "INFO"},
                timeout=10,
            )
            response.raise_for_status()

    def server_args(self):
        args = super().server_args()
        args += (
            f" --max_seq_len {max(self.lengths) + self.output_length + 1024}"
            f" --max_context_batch_size {max(self.batches)}"
            f" --concurrency_limit {max(self.batches)}"
            " --use_batch_decode_scheduler 0"
        )
        if self.gamma:
            args += (
                " --sp_type dspark --sp_model_type deepseek_v41_dspark"
                f" --sp_checkpoint_path {shlex.quote(self.model_path)}"
                f" --sp_act_type BF16 --gen_num_per_cycle {self.gamma}"
            )
        return args

    def routes(self, manager, role, expected):
        response = self.session.get(
            f"http://127.0.0.1:{manager.port}/worker_status", timeout=30
        )
        response.raise_for_status()
        workers = response.json()["results"]
        self.assertEqual({int(w["dp_rank"]) for w in workers}, set(range(expected)))
        self.assertTrue(all(w["alive"] for w in workers))
        result = []
        for worker in sorted(workers, key=lambda w: int(w["dp_rank"])):
            ip, port = worker["address"].rsplit(":", 1)
            result.append(
                dict(role=role, ip=ip, grpc_port=int(port), http_port=manager.port)
            )
        return result

    def batch(self, prompt, length, count, prefill, decode):
        def request(index):
            rank = index % len(decode)
            payload = dict(
                prompt=prompt,
                generate_config=dict(
                    max_new_tokens=self.output_length,
                    min_new_tokens=self.output_length,
                    top_k=1,
                    top_p=1.0,
                    random_seed=42,
                    reuse_cache=False,
                    can_use_pd_separation=True,
                    force_sp_accept=False,
                    return_output_ids=True,
                    aux_info=True,
                    role_addrs=[prefill[0], decode[rank]],
                ),
            )
            started = time.time()
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(
                    f"http://127.0.0.1:{self.server.port}/", json=payload, timeout=900
                )
                response.raise_for_status()
                raw = response.json()
            self.assertNotIn("error_code", raw)
            self.assertIs(raw.get("finished"), True)
            aux = raw["aux_info"]
            self.assertIs(aux["pd_sep"], True)
            self.assertTrue(
                any(
                    route.get("role") == "DECODE"
                    and route.get("ip") == decode[rank]["ip"]
                    and int(route.get("grpc_port", -1)) == decode[rank]["grpc_port"]
                    for route in aux.get("role_addrs", [])
                ),
                aux.get("role_addrs"),
            )
            self.assertEqual(aux["input_len"], length)
            self.assertEqual(aux["output_len"], self.output_length)
            self.assertEqual(aux["prefill_total_reuse_len"], 0)
            self.assertEqual(len(raw["output_ids"][0]), aux["output_len"])
            counters = aux.get("speculative_accepted_tokens_per_pos", [])
            rounds = aux.get("speculative_draft_rounds", 0)
            if self.gamma:
                self.assertGreater(rounds, 0)
                self.assertEqual(len(counters), self.gamma)
                self.assertTrue(all(0 <= n <= rounds for n in counters))
            elif rounds:
                self.fail("non-speculative baseline executed draft rounds")
            metrics = vars(ResponseInfo(raw))
            self.assertGreater(metrics["decode_time_per_token"], 0)
            return dict(
                rank=rank,
                started_at=started,
                wall_s=time.time() - started,
                metrics=metrics,
                raw=raw,
            )

        started = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            responses = list(pool.map(request, range(count)))
        wall = time.time() - started
        tokens = sum(r["metrics"]["output_len"] for r in responses)
        return dict(
            started_at=started,
            wall_s=wall,
            output_tokens=tokens,
            http_tokens_per_s=tokens / wall,
            mean_tpot_ms=statistics.mean(
                r["metrics"]["decode_time_per_token"] for r in responses
            ),
            mean_ttft_ms=statistics.mean(
                r["raw"]["aux_info"]["first_token_cost_time"] for r in responses
            ),
            responses=responses,
        )

    def test_graph_engram_and_prefix_reuse(self):
        # The smoke target owns correctness checks; this target measures the
        # same managed P/D deployment without synthetic KV or forced acceptance.
        prefill = self.routes(self.server, "PREFILL", 1)
        decode = self.routes(self.graph_server, "DECODE", self.decode_dp_size)
        report = dict(
            gamma=self.gamma,
            topology="CP4/DP4 EP4",
            layers=json.loads((Path(self.model_path) / "config.json").read_text())[
                "text_config"
            ]["num_hidden_layers"],
            batch_semantics="offered concurrent requests per decode rank",
            configured_lengths=self.lengths,
            measured_lengths=self.measure_lengths,
            prefill_routes=prefill,
            decode_routes=decode,
            rows=[],
            profiles=[],
            source_sha=os.environ.get("PD_PERF_SOURCE_SHA", "unknown"),
            engine_args=self.server_args(),
            env=self.extra_server_env,
            profile_scope="decode rank0 (TP1), not all DP ranks",
            status="RUNNING",
        )
        destination = self.output_dir / "pd_perf.json"

        def save():
            destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))

        save()
        with (self.output_dir / "gpu_telemetry.csv").open("w") as telemetry:
            sampler = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,memory.used,utilization.gpu,clocks.sm,clocks.mem",
                    "--format=csv,noheader,nounits",
                    "--loop-ms=500",
                ],
                stdout=telemetry,
            )
            try:
                factory = PrefixPromptFactory(self.tokenizer)
                for length in self.measure_lengths:
                    prompt, _ = factory._exact_text_and_ids(
                        length, "Continue counting: 1, 2, 3,"
                    )
                    for batch in self.batches:
                        row = dict(
                            input_len=length, offered_per_rank=batch, warmup=[], runs=[]
                        )
                        report["rows"].append(row)
                        for repeat in range(2 + self.rounds):
                            result = self.batch(
                                prompt, length, batch * len(decode), prefill, decode
                            )
                            row["warmup" if repeat < 2 else "runs"].append(result)
                            save()
                # Profiling is separate from all formal wall-clock measurements.
                if os.environ.get("PD_PERF_PROFILE", "1") == "1":
                    length = self.lengths[0]
                    prompt, _ = factory._exact_text_and_ids(
                        length, "Continue counting: 1, 2, 3,"
                    )
                    for batch in self.batches:
                        offset = Path(self.graph_server.log_file_path).stat().st_size
                        response = self.session.post(
                            f"http://127.0.0.1:{self.graph_server.port}/start_profile",
                            json=dict(
                                gen_timeline=True,
                                trace_name=f"pd_g{self.gamma}_b{batch}_n{length}",
                                start_step=3,
                                num_steps=20,
                                enable_all_rank=True,
                            ),
                            timeout=60,
                        )
                        response.raise_for_status()
                        self.assertEqual(response.json().get("status"), "ok")
                        report["profiles"].append(
                            self.batch(
                                prompt, length, batch * len(decode), prefill, decode
                            )
                        )
                        _wait_for_trace_flush(
                            self.graph_server.log_file_path, offset, 120
                        )
                        save()
                report["status"] = "PASS"
            except Exception as error:
                report.update(status="FAIL", error=str(error))
                raise
            finally:
                sampler.terminate()
                sampler.wait(timeout=15)
                save()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
