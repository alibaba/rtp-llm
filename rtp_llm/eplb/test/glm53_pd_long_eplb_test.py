import concurrent.futures
import hashlib
import json
import os
import re
import time
import unittest
from pathlib import Path

import numpy as np
import requests
from transformers import PreTrainedTokenizerFast

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM
from rtp_llm.eplb.test.glm53_pd_eplb_test import role_environments
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


def make_workload(tokenizer, total=200000, shared=180000):
    passage = (
        "The research station records temperature, rainfall, wind speed and water quality. "
        "Engineers compare daily observations with historical measurements, check sensor "
        "calibration and document uncertainty before drawing conclusions. The team uses "
        "Python to validate tables, detect missing values and summarize seasonal changes. "
        "A reliable report distinguishes measured results from hypotheses and includes "
        "enough information for another team to reproduce the analysis.\n"
    )
    header = "[gMASK]<sop><|system|>Read the reference material and answer the final question.<|user|>Reference material:\n"
    prefix_ids = tokenizer.encode(header + passage * 4000, add_special_tokens=False)[
        :shared
    ]
    prefix = tokenizer.decode(prefix_ids)
    if len(tokenizer.encode(prefix, add_special_tokens=False)) != shared:
        raise ValueError("Prefix does not round-trip to the required token length")
    filler = tokenizer.encode(passage * 1000, add_special_tokens=False)
    tail = "\nExplain how the station should validate its measurements. Give a detailed numbered procedure.<|assistant|><think>"

    def prompt(nonce):
        beginning = prefix + f"\nIndependent appendix {nonce}:\n"
        budget = total - len(
            tokenizer.encode(beginning + tail, add_special_tokens=False)
        )
        for _ in range(8):
            text = beginning + tokenizer.decode(filler[:budget]) + tail
            length = len(tokenizer.encode(text, add_special_tokens=False))
            if length == total:
                return text
            budget += total - length
        raise ValueError(f"Cannot construct exactly {total} input tokens: {length}")

    return prefix, prompt


class Glm53PdLongEplbTest(unittest.TestCase):
    def test_long_cached_workload(self):
        checkpoint = Path(os.environ["EPLB_GLM5_CHECKPOINT"])
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        self.assertEqual(len(devices), 6)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(checkpoint / "tokenizer.json")
        )
        prefix, make_prompt = make_workload(tokenizer)
        concurrencies = [
            int(x) for x in os.environ.get("EPLB_LONG_CONCURRENCIES", "8,16").split(",")
        ]
        profile_concurrency = concurrencies[-1]
        workloads = {
            (phase, concurrency): [
                make_prompt(f"{phase}-{concurrency}-{i:04d}")
                for i in range(concurrency)
            ]
            for phase, concurrency in [
                *(("measure", c) for c in concurrencies),
                ("profile", profile_concurrency),
            ]
        }
        warm_prompt = make_prompt("warmup-independent")
        manifest = {
            "input_tokens": 200000,
            "prefix_tokens": 180000,
            "output_tokens": 128,
            "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
            "concurrencies": concurrencies,
        }
        (output / "long_workload.json").write_text(json.dumps(manifest, indent=2))
        summaries = []
        scope = output / "expert_observer_scope.json"
        for mode in os.environ.get("EPLB_GLM5_MODES", "STATS,EPLB").split(","):
            ports = {
                role: MagaServerManager.get_free_port()
                for role in ("prefill", "decode")
            }
            envs = role_environments(mode, devices, ports["prefill"], ports["decode"])
            for env in envs.values():
                env.update(
                    MAX_SEQ_LEN="262144",
                    MAX_BATCH_TOKENS_SIZE="262144",
                    CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS="5000",
                    EPLB_OBSERVER_SCOPE_FILE=str(scope),
                )
            envs["prefill"].update(
                KV_CACHE_MEM_MB="32768", MEMORY_CACHE_SIZE_MB="16384"
            )
            envs["decode"].update(
                KV_CACHE_MEM_MB="32768", RTP_LLM_DSA_MLA_HOST_CACHE_MB="196608"
            )
            route = json.loads(envs["prefill"]["MODEL_SERVICE_CONFIG"])
            route["role_endpoints"][0]["decode_endpoint"]["address"] = ",".join(
                f"127.0.0.1:{int(ports['decode']) + rank * MIN_WORKER_INFO_PORT_NUM}"
                for rank in range(2)
            )
            envs["prefill"]["MODEL_SERVICE_CONFIG"] = json.dumps(route)
            (output / f"LONG_{mode}_config.json").write_text(json.dumps(envs, indent=2))
            servers = {
                role: MagaServerManager(
                    role_name=f"long_{mode.lower()}_{role}",
                    env_args=env,
                    port=ports[role],
                )
                for role, env in envs.items()
            }

            def request(name, text, new_tokens=128, pd=True):
                start = time.perf_counter()
                response = requests.post(
                    f"http://127.0.0.1:{ports['prefill']}/",
                    json={
                        "prompt": text,
                        "generate_config": {
                            "max_new_tokens": new_tokens,
                            "min_new_tokens": new_tokens,
                            "top_k": 1,
                            "top_p": 1.0,
                            "random_seed": 123,
                            "is_streaming": False,
                            "return_output_ids": True,
                            "can_use_pd_separation": pd,
                            "timeout_ms": 1800000,
                        },
                    },
                    timeout=1800,
                )
                (output / f"{name}.json").write_text(response.text)
                response.raise_for_status()
                result = response.json()
                self.assertTrue(result["finished"], name)
                aux = result["aux_info"]
                self.assertEqual(aux["input_len"], 200000 if pd else 180000, name)
                self.assertEqual(aux["output_len"], new_tokens, name)
                if pd:
                    self.assertTrue(aux["pd_sep"], name)
                    ratio = aux["prefill_total_reuse_len"] / aux["input_len"]
                    self.assertGreaterEqual(ratio, 0.895, (name, aux))
                    self.assertLessEqual(ratio, 0.905, (name, aux))
                return {
                    "artifact": f"{name}.json",
                    "wall_ms": (time.perf_counter() - start) * 1000,
                    "aux_info": aux,
                }

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    started = [
                        pool.submit(
                            server.start_server,
                            model_path=str(checkpoint),
                            model_type="glm_5",
                            timeout=1800,
                        )
                        for server in servers.values()
                    ]
                    for future in started:
                        self.assertTrue(future.result(), f"{mode} startup failed")
                request(f"{mode}_prefix_warmup", prefix, new_tokens=1, pd=False)
                request(f"{mode}_query_warmup", warm_prompt)
                for (phase, concurrency), prompts in workloads.items():
                    if phase == "profile":
                        temporary = scope.with_suffix(".tmp")
                        temporary.write_text(
                            json.dumps(
                                {
                                    "request": f"{mode}_c{concurrency}_profile",
                                    "prompt": "200K input, 90% shared prefix",
                                }
                            )
                        )
                        temporary.replace(scope)
                    start = time.perf_counter()
                    try:
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=concurrency
                        ) as pool:
                            futures = [
                                pool.submit(
                                    request, f"{mode}_{phase}_c{concurrency}_{i}", text
                                )
                                for i, text in enumerate(prompts)
                            ]
                            records = [future.result() for future in futures]
                    finally:
                        scope.unlink(missing_ok=True)
                    elapsed = time.perf_counter() - start
                    latencies = [item["wall_ms"] for item in records]
                    summary = {
                        "mode": mode,
                        "phase": phase,
                        "concurrency": concurrency,
                        "batch_seconds": elapsed,
                        "output_tokens_per_second": concurrency * 128 / elapsed,
                        "median_latency_ms": float(np.median(latencies)),
                        "p95_latency_ms": float(np.percentile(latencies, 95)),
                        "median_first_token_ms": float(
                            np.median(
                                [
                                    item["aux_info"]["first_token_cost_time"]
                                    for item in records
                                ]
                            )
                        ),
                        "reuse_ratios": [
                            item["aux_info"]["prefill_total_reuse_len"] / 200000
                            for item in records
                        ],
                        "records": records,
                    }
                    summaries.append(summary)
                    (output / "long_results.json").write_text(
                        json.dumps(summaries, indent=2)
                    )
            finally:
                scope.unlink(missing_ok=True)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    list(
                        pool.map(lambda server: server.stop_server(), servers.values())
                    )
            for role, size in (("prefill", 4), ("decode", 2)):
                log = (
                    output / f"long_{mode.lower()}_{role}_logs/process.log"
                ).read_text(errors="replace")
                applied = re.findall(
                    r"\[EPLB\] applied .*?changed_slots=\[([^\]]*)\] ep_rank=(\d+)", log
                )
                if mode == "EPLB":
                    changed = {
                        int(rank)
                        for slots, rank in applied
                        if any(int(x.strip()) > 0 for x in slots.split(","))
                    }
                    self.assertEqual(changed, set(range(size)))
                else:
                    self.assertFalse(applied)
                snapshots = list(
                    (output / "expert_hotspots").glob(
                        f"{mode}_c{profile_concurrency}_profile_{role.upper()}_rank*.pt"
                    )
                )
                self.assertEqual(len(snapshots), size)


if __name__ == "__main__":
    unittest.main()
