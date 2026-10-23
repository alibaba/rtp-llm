"""Manual real P/D feature toggle evidence, with no tensor capture in timed runs.

Wall throughput is serial generated tokens / HTTP response wall time, not model
prefill throughput. GPU memory samples are device-wide resident memory including co-tenants,
not peak allocator memory. Copy/save isolated costs require the separate cost evidence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_test import (
    DeepSeekV4TailReuseFourLayerTest,
)
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager

MARKER = "event=prefix_cache_observation payload="


class _TailReuseBenefitTest(unittest.TestCase):
    cp_size = 1
    _extend_prompt = DeepSeekV4TailReuseFourLayerTest._extend_prompt

    def setUp(self):
        self.model_path = Path(os.environ["T07_FOUR_LAYER_CHECKPOINT"])
        config = json.loads((self.model_path / "config.json").read_text())
        self.assertEqual(config["num_hidden_layers"], 4)
        self.assertEqual(config["compress_ratios"], [0, 0, 4, 128])
        index = self.model_path / "model.safetensors.index.json"
        shards = set(json.loads(index.read_text())["weight_map"].values())
        self.assertTrue(all((self.model_path / name).is_file() for name in shards))
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        self.cards = list(get_gpu_ids())[: self.cp_size + 1]
        self.assertEqual(len(self.cards), self.cp_size + 1)
        self._write(
            "checkpoint.json",
            {
                "path": str(self.model_path),
                "config": config,
                "config_sha256": hashlib.sha256(
                    (self.model_path / "config.json").read_bytes()
                ).hexdigest(),
                "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                "shards": sorted(shards),
                "cp_size": self.cp_size,
                "gpus": self.cards,
            },
        )

    def _write(self, name, value):
        (self.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")

    def _start(self, mode, feature, observation, internal=False):
        ports = [MagaServerManager.get_free_port() for _ in range(2)]
        self.profile_dir = self.output_dir / f"{mode}_profiles"
        self.profile_dir.mkdir()
        for role in ("prefill", "decode"):
            (self.profile_dir / role).mkdir()
        self.profile_dir = self.profile_dir / "prefill"
        route = ServiceRoute(
            service_id=f"tail_benefit_{mode}_cp{self.cp_size}",
            use_local=True,
            role_endpoints=[
                GroupEndPoint(
                    group="default",
                    **{
                        f"{role}_endpoint": EndPoint(
                            type="Vipserver",
                            address=f"127.0.0.1:{port}",
                            protocol="http",
                            path="/",
                        )
                        for role, port in zip(("prefill", "decode"), ports)
                    },
                )
            ],
        )
        env = {
            "LOAD_PYTHON_MODEL": "1",
            "LOG_LEVEL": "INFO",
            "FT_SERVER_TEST": "1",
            "DSV4_MOE_STRATEGY": "mega",
            "DSV4_USE_MEGA_MOE_SE": "0",
            "DSV4_USE_MEGA_MOE_FUSED": "0",
            "DSV4_CHUNK_TOKENS": "2048",
            "REMOTE_RPC_SERVER_IP": "localhost",
            "MODEL_SERVICE_CONFIG": route.model_dump_json(),
            "TORCH_CUDA_PROFILER_DIR": str(self.profile_dir),
            "GEN_TIMELINE_SYNC": "0",
            "MOEDBG": "0",
            "RTP_HOT_HOOK": "0",
            "RTP_LLM_PREFIX_CACHE_OBSERVABILITY": "true" if observation else "false",
        }
        common = (
            " --act_type BF16 --fp8_kv_cache 1 --seq_size_per_block 256"
            " --kernel_seq_size_per_block 128 --max_seq_len 2048"
            " --max_context_batch_size 1 --concurrency_limit 1 --reuse_cache 1"
            " --load_method fastsafetensors --kv_cache_mem_mb 4096"
            " --reserver_runtime_mem_mb 12288 --cache_store_rdma_mode 0"
            " --use_local 1 --load_cache_timeout_ms 120000 --warm_up 0"
            " --enable_device_cache 1 --enable_memory_cache 0"
            " --enable_disk_cache 0 --enable_remote_cache 0 --enable_cuda_graph 0"
        )
        if self._testMethodName == "test_internal_fork_benefit":
            common = common.replace("--kv_cache_mem_mb 4096", "--kv_cache_mem_mb 2048")
        if self.cp_size == 2:
            common += " --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2"
        managers = []
        for i, role in enumerate(("prefill", "decode")):
            world = self.cp_size if i == 0 else 1
            args = (
                f"--tp_size {world} --ep_size {world} --dp_size 1 --world_size {world}"
                + common
            )
            args += f" --role_type {role.upper()}"
            if i == 0:
                args += f" --enable_prefill_partial_tail_reuse {int(feature)}"
                if self._testMethodName == "test_internal_fork_benefit":
                    args += f" --enable_prefill_common_prefix_recovery {int(internal)}"
            if self.cp_size == 2:
                args += " --cp_rotate_method " + (
                    "ALL_GATHER" if i == 0 else "PREFILL_CP"
                )
            manager = MagaServerManager(
                env_args={
                    **env,
                    "WORLD_SIZE": str(world),
                    "REMOTE_SERVER_PORT": ports[1 - i],
                    "TORCH_CUDA_PROFILER_DIR": str(self.profile_dir.parent / role),
                },
                device_ids=(
                    self.cards[: self.cp_size] if i == 0 else self.cards[self.cp_size :]
                ),
                port=ports[i],
                role_name=f"benefit_{mode}_{role}",
                smoke_args_str=args,
            )
            self.addCleanup(manager.stop_server)
            managers.append(manager)
        self.prefill, self.decode = managers
        with ThreadPoolExecutor(max_workers=2) as executor:
            starts = [
                executor.submit(
                    manager.start_server,
                    model_path=str(self.model_path),
                    model_type="deepseek_v4",
                    timeout=1800,
                )
                for manager in managers
            ]
            for start in starts:
                self.assertTrue(
                    start.result(), f"{mode} checkpoint service startup failed"
                )

    def _events(self):
        path = Path(self.prefill.log_file_path)
        return [
            json.loads(line.split(MARKER, 1)[1])
            for line in path.read_text(errors="replace").splitlines()
            if MARKER in line
        ]

    def _memory(self):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                ",".join(map(str, self.cards)),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return {"unix_ns": time.time_ns(), "units": "MiB", "csv": result.stdout}

    def _request(self, mode, label, prompt, reuse, observation, profile=False):
        before = len(self._events())
        query = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": 8,
                "ignore_eos": True,
                "top_k": 1,
                "top_p": 1.0,
                "random_seed": 7,
                "reuse_cache": reuse,
                "enable_device_cache": True,
                "enable_memory_cache": False,
                "enable_disk_cache": False,
                "enable_remote_cache": False,
                "can_use_pd_separation": True,
                "return_input_ids": True,
                "return_output_ids": True,
                "aux_info": True,
            },
        }
        if profile:
            # Request-triggered profiling only sees streams on TP rank zero.
            # Pre-arm all prefill ranks after clean measurements; callbacks
            # start the window after nonempty input synchronization.
            arm_query = {
                "trace_name": label,
                "start_step": 0,
                "num_steps": 1,
                "enable_all_rank": True,
            }
            armed = self.session.post(
                f"http://127.0.0.1:{self.prefill.port}/start_profile",
                json=arm_query,
                timeout=60,
            )
            self._write(
                f"{mode}_{label}.profile_arm.json",
                {
                    "request": arm_query,
                    "status_code": armed.status_code,
                    "response": armed.text,
                },
            )
            armed.raise_for_status()
            self.assertEqual(armed.json().get("status"), "ok", armed.text)
            query["generate_config"].update(
                gen_timeline=True, profile_step=1, profile_trace_name=label
            )
        self._write(f"{mode}_{label}.request.json", query)
        memory_before = self._memory()
        start_ns = time.time_ns()
        begin = time.perf_counter()
        response = self.session.post(
            f"http://127.0.0.1:{self.prefill.port}/", json=query, timeout=1200
        )
        elapsed = time.perf_counter() - begin
        end_ns = time.time_ns()
        (self.output_dir / f"{mode}_{label}.http.txt").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        self.assertNotIn("error_code", actual)
        self.assertIs(actual.get("finished"), True)
        self.assertIs(actual["aux_info"]["pd_sep"], True)
        ids = actual["input_ids"][0]
        self.assertEqual(ids, self.tokenizer.encode(prompt))
        self.assertEqual(len(actual["output_ids"][0]), 8)
        # Request summaries can arrive at report/release after HTTP completes.
        deadline = time.monotonic() + 10
        events = self._events()[before:]
        while (
            observation
            and not any(e.get("kind") == "request" for e in events)
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
            events = self._events()[before:]
        summaries = [e for e in events if e.get("kind") == "request"]
        if observation:
            self.assertEqual(len(summaries), 1, events)
            self.assertEqual(summaries[0]["input_tokens"], len(ids))
            self.assertEqual(
                summaries[0]["used_pos"], actual["aux_info"]["prefill_total_reuse_len"]
            )
        else:
            self.assertEqual(events, [])
        row = {
            "profiled": profile,
            "case": label,
            "mode": mode,
            "input_tokens": len(ids),
            "input_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "output_ids": actual["output_ids"][0],
            "aux_info": actual["aux_info"],
            "start_unix_ns": start_ns,
            "end_unix_ns": end_ns,
            "http_wall_s": elapsed,
            "generated_tokens_per_http_second": len(actual["output_ids"][0]) / elapsed,
            "engine_ttft_ms": actual["aux_info"]["first_token_cost_time"],
            "memory_before": memory_before,
            "memory_after": self._memory(),
            "summary": summaries[0] if summaries else None,
        }
        self._write(f"{mode}_{label}.measurement.json", row)
        return row

    def test_fixed_requests(self):
        results = {}
        for mode, feature, observation in (
            ("off", False, True),
            ("on", True, True),
            ("on_no_observation", True, False),
        ):
            self._start(mode, feature, observation)
            try:
                warm = self._extend_prompt("Unrelated warmup calibration:", 1300)
                self._request(mode, "warmup_source", warm, True, observation)
                warm = self._extend_prompt(warm, 2000)
                self._request(mode, "warmup_target", warm, True, observation)
                self._request(mode, "warmup_cold", warm, False, observation)
                rows = []
                prefixes = [self.tokenizer.encode(warm)]
                for branch in ("Amber", "Beryl", "Cobalt"):
                    source = self._extend_prompt(f"{branch} fixed prefix:", 1300)
                    source_ids = self.tokenizer.encode(source)
                    for previous in prefixes:
                        self.assertNotEqual(source_ids[:256], previous[:256])
                    prefixes.append(source_ids)
                    target = self._extend_prompt(source, 2000)
                    a = self._request(
                        mode, f"{branch}_source", source, True, observation
                    )
                    b = self._request(
                        mode, f"{branch}_target", target, True, observation
                    )
                    cold = self._request(
                        mode, f"{branch}_cold", target, False, observation
                    )
                    self.assertEqual(
                        b["output_ids"],
                        cold["output_ids"],
                        f"{mode}/{branch}: warm vs cold tokens",
                    )
                    g = b["aux_info"]["prefill_total_reuse_len"]
                    if feature:
                        self.assertEqual(g, 1300)
                    else:
                        self.assertLess(g, 1300)
                    rows.extend((a, b))
                results[mode] = rows
                # Independent profile requests follow clean measurements. Their
                # events remain in all_events but never enter measured JSONL.
                if observation:
                    profile_source = self._extend_prompt(
                        "Zircon isolated profiler:", 1300
                    )
                    profile_target = self._extend_prompt(profile_source, 2000)
                    for label, prompt in (
                        ("profile_source", profile_source),
                        ("profile_target", profile_target),
                    ):
                        before_profiles = set(self.profile_dir.glob("*.json"))
                        self._request(
                            mode, label, prompt, True, observation, profile=True
                        )
                        deadline = time.monotonic() + 30
                        while (
                            len(set(self.profile_dir.glob("*.json")) - before_profiles)
                            < self.cp_size
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.1)
                        self.assertGreaterEqual(
                            len(set(self.profile_dir.glob("*.json")) - before_profiles),
                            self.cp_size,
                        )
                        captured = (
                            set(self.profile_dir.glob("*.json")) - before_profiles
                        )
                        for rank in range(self.cp_size):
                            self.assertTrue(
                                any(f"_wr{rank}_" in path.name for path in captured),
                                f"missing prefill profile rank {rank}: {captured}",
                            )
                self._write(f"{mode}.all_events.json", self._events())
                (self.output_dir / f"{mode}.requests.jsonl").write_text(
                    "".join(
                        json.dumps(row["summary"]) + "\n"
                        for row in rows
                        if row["summary"] is not None
                    )
                )
            finally:
                self.prefill.stop_server()
                self.decode.stop_server()
        pairs = []
        for off, on, quiet in zip(
            results["off"], results["on"], results["on_no_observation"]
        ):
            self.assertEqual(off["input_sha256"], on["input_sha256"])
            self.assertEqual(on["input_sha256"], quiet["input_sha256"])
            self.assertEqual(off["output_ids"], on["output_ids"])
            self.assertEqual(on["output_ids"], quiet["output_ids"])
            self.assertEqual(
                on["aux_info"]["prefill_total_reuse_len"],
                quiet["aux_info"]["prefill_total_reuse_len"],
            )
            pairs.append(
                {
                    "case": off["case"],
                    **{
                        side: {
                            "instance": row["summary"]["instance"],
                            "request_id": row["summary"]["request_id"],
                            "input_sha256": row["input_sha256"],
                        }
                        for side, row in (("off", off), ("on", on))
                    },
                }
            )
        self._write(
            "benefit_manifest.json",
            {
                "pairs": pairs,
                "measurements": results,
                "provenance": {
                    "cp_size": self.cp_size,
                    "checkpoint": str(self.model_path),
                    "warmup": "unrelated 1300/2000 source/target and cold 2000 per fresh mode; excluded from measurement",
                    "measurement": "serial HTTP, 3 distinct source/target pairs per mode; cold references excluded",
                    "ttft_units": "engine aux first_token_cost_time in ms",
                    "throughput": "generated output tokens divided by full HTTP wall seconds",
                    "memory": "nvidia-smi resident memory sampled before/after each request; not peak",
                    "copy_save_isolated_cost": "separate evidence required; not inferred from wall time",
                    "tensor_capture": False,
                    "profile_selection": "post-measurement profile_source/profile_target are excluded from measured JSONL and pairs by explicit request-ID selection; all events and raw HTTP retained",
                },
            },
        )

    def test_internal_fork_benefit(self):
        self.assertEqual(self.cp_size, 1)
        results = {}
        for mode, tail, internal, observation in (
            ("both_off", False, False, True),
            ("f_off", True, False, True),
            ("f_on", True, True, True),
            ("f_on_quiet", True, True, False),
        ):
            self._start(mode, feature=tail, observation=observation, internal=internal)
            try:
                warm = self._extend_prompt("Unrelated F benefit warmup:", 1300)
                self._request(mode, "warmup_source", warm, True, observation)
                self._request(
                    mode,
                    "warmup_target",
                    self._extend_prompt(warm, 2000),
                    True,
                    observation,
                )
                prefix = self._extend_prompt("Common F benefit source:", 800)
                source_prompt = self._extend_prompt(prefix, 1300)
                source_ids = self.tokenizer.encode(source_prompt)
                branches = []
                for suffix in (" branch", " amber", " violet", " silver", " orange"):
                    candidate = self.tokenizer.encode(prefix + suffix)
                    if (
                        len(candidate) > 800
                        and candidate[:800] == source_ids[:800]
                        and candidate[800] != source_ids[800]
                        and all(candidate[800] != token for token, _ in branches)
                    ):
                        branches.append(
                            (candidate[800], self._extend_prompt(prefix + suffix, 2000))
                        )
                    if len(branches) == 2:
                        break
                self.assertEqual(len(branches), 2)
                source = self._request(
                    mode, "source_1300", source_prompt, True, observation
                )
                producer = self._request(
                    mode, "producer_2000", branches[0][1], True, observation
                )
                beneficiary = self._request(
                    mode, "beneficiary_2000", branches[1][1], True, observation
                )
                cold = self._request(
                    mode, "beneficiary_cold", branches[1][1], False, observation
                )
                self.assertEqual(source["aux_info"]["prefill_total_reuse_len"], 0)
                self.assertEqual(producer["aux_info"]["prefill_total_reuse_len"], 768)
                self.assertEqual(
                    beneficiary["aux_info"]["prefill_total_reuse_len"],
                    800 if internal else 768,
                )
                self.assertEqual(cold["aux_info"]["prefill_total_reuse_len"], 0)
                self.assertEqual(beneficiary["output_ids"], cold["output_ids"])
                results[mode] = {
                    "source": source,
                    "producer": producer,
                    "beneficiary": beneficiary,
                }
            finally:
                self.prefill.stop_server()
                self.decode.stop_server()
        both_off = results["both_off"]
        off = results["f_off"]
        on = results["f_on"]
        quiet = results["f_on_quiet"]
        for label in ("source", "producer", "beneficiary"):
            self.assertEqual(both_off[label]["input_sha256"], off[label]["input_sha256"])
            self.assertEqual(off[label]["input_sha256"], on[label]["input_sha256"])
            self.assertEqual(on[label]["input_sha256"], quiet[label]["input_sha256"])
            self.assertEqual(both_off[label]["output_ids"], off[label]["output_ids"])
            self.assertEqual(off[label]["output_ids"], on[label]["output_ids"])
            self.assertEqual(on[label]["output_ids"], quiet[label]["output_ids"])
        for label in ("producer", "beneficiary"):
            self.assertEqual(
                both_off[label]["aux_info"]["prefill_total_reuse_len"],
                off[label]["aux_info"]["prefill_total_reuse_len"],
            )
        for label in ("producer", "beneficiary"):
            self.assertEqual(
                on[label]["aux_info"]["prefill_total_reuse_len"],
                quiet[label]["aux_info"]["prefill_total_reuse_len"],
            )
        aggregate = {}
        for mode, rows in results.items():
            measured = list(rows.values())
            aggregate[mode] = {
                "sum_used_tokens": sum(
                    row["aux_info"]["prefill_total_reuse_len"] for row in measured
                ),
                "sum_input_tokens": sum(row["input_tokens"] for row in measured),
                "ttft_ms": {
                    label: row["engine_ttft_ms"] for label, row in rows.items()
                },
                "generated_tokens_per_http_second": {
                    label: row["generated_tokens_per_http_second"]
                    for label, row in rows.items()
                },
            }
        self._write(
            "internal_fork_benefit_manifest.json",
            {
                "modes": results,
                "aggregate": aggregate,
                "beneficiary_delta_g": (
                    on["beneficiary"]["aux_info"]["prefill_total_reuse_len"]
                    - off["beneficiary"]["aux_info"]["prefill_total_reuse_len"]
                ),
                "limits": {
                    "throughput": "serial generated tokens divided by HTTP wall time",
                    "memory": "device-wide before/after samples; no peak attribution",
                    "copy_save_isolated_cost": "not measured separately",
                    "shared_gpu": "timings are descriptive, not a speedup claim",
                },
            },
        )


class TailReuseBenefitNonCPTest(_TailReuseBenefitTest):
    cp_size = 1


class TailReuseBenefitCP2Test(_TailReuseBenefitTest):
    cp_size = 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
