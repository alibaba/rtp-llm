"""Fixed SM90 P-only CP8 and D-only DP8 execution smoke/performance matrix.

Decode uses the existing synthetic-KV batch scheduler; it does not measure PD
transfer or validate generation quality. Existing golden smokes remain separate.
"""

import argparse
import importlib.util
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

LENGTHS = (32768, 65536)


def configure_jit_cache():
    # Humming hashes absolute header/include paths. Different Bazel targets
    # must import the selected wheel through its same real directory.
    if "humming" in sys.modules:
        raise RuntimeError("normalize Humming path before importing it")
    spec = importlib.util.find_spec("humming")
    if spec is None or spec.origin is None:
        raise RuntimeError("selected Bazel Humming wheel was not found")
    site_packages = str(Path(spec.origin).resolve().parent.parent)
    sys.path.insert(0, site_packages)
    paths = [
        p
        for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if p and p != site_packages
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join([site_packages] + paths)
    # Bazel may omit HOME; PyTorch otherwise disables its disk kernel cache.
    cache = Path(
        os.environ.setdefault(
            "PYTORCH_KERNEL_CACHE_PATH",
            str(Path.home() / ".cache" / "rtp_llm" / "torch_kernels"),
        )
    )
    cache.mkdir(parents=True, exist_ok=True)
    return {
        "humming_site_packages": site_packages,
        "pytorch_kernel_cache_path": str(cache),
    }


def recipe(role, mode, checkpoint, tokenizer):
    decode = role == "decode"
    local_batches = (1, 4, 16) if decode and mode == "perf" else (1,)
    output_len = (512 if mode == "perf" else 32) if decode else 1
    options = {
        "model_type": "deepseek_v41",
        "checkpoint_path": checkpoint,
        "tokenizer_path": tokenizer,
        "role_type": role.upper(),
        "tp_size": 1 if decode else 8,
        "ep_size": 8,
        "world_size": 8,
        "act_type": "BF16",
        "fp8_kv_cache": 1,
        "seq_size_per_block": 128,
        "max_context_batch_size": max(local_batches),
        "enable_cuda_graph": int(decode),
        "decode_capture_config": ",".join(map(str, local_batches)),
        "dsv4_fixed_pool_blocks": 256,
        "reuse_cache": 0,
        "load_method": "scratch",
        "reserver_runtime_mem_mb": 32768,
        "use_deepep_moe": 1,
        "use_all_gather": 0,
        "use_deepep_low_latency": 0,
        "use_deepep_internode": 0,
        "fake_balance_expert": 0,
        "use_batch_decode_scheduler": 1,
    }
    if not decode:
        # Attention CP transport; MoE remains DeepEP V2 (not AG/AR MoE).
        options["cp_rotate_method"] = "ALL_GATHER"
    env = {
        "WORLD_SIZE": "8",
        "LOAD_PYTHON_MODEL": "1",
        "DSV4_MOE_STRATEGY": "sm90_flashinfer",
        "MOE_STRATEGY": "auto",
        "DSV41_SM90_MOE_COMM": "v2",
        "DSV41_SM90_V2_NUM_SMS": "0",
        "DSV4_USE_MEGA_MOE_SE": "0",
        "DSV4_USE_MEGA_MOE_FUSED": "0",
        "DSV41_CED": str(int(decode)),
        "DSV41_SWA_BOUNDED_REPLAY": str(int(decode)),
        "DSV41_SM90_DECODE_GRAPH": str(int(decode)),
        "DSV41_SM90_DECODE_GRAPH_MAX_ROWS": str(max(local_batches)),
        "DSV41_SM90_MULTIQUERY_INDEXER": "1",
        "DSV41_ENGRAM_PREFETCH": os.environ.get("DSV41_ENGRAM_PREFETCH", "1"),
        "DSV41_ENGRAM_REGISTER_CHUNK_MB": "256",
        "DSV4_MOE_CHUNK_PREFILL": "0",
        "ROLE_TYPE": role.upper(),
        "RTP_LLM_PD_ROUTE_RETRY_ON_UNAVAILABLE": "0",
        "FT_SERVER_TEST": "1",
        "LOG_LEVEL": "INFO",
    }
    if env["DSV41_ENGRAM_PREFETCH"] not in ("0", "1"):
        raise ValueError("DSV41_ENGRAM_PREFETCH must be 0 or 1")
    return options, env, local_batches, output_len


def validate_responses(responses, count, input_len, output_len):
    if len(responses) != count:
        raise AssertionError(f"received {len(responses)} responses, expected {count}")
    for index, response in enumerate(responses):
        if not response.success or (response.input_len, response.output_len) != (
            input_len,
            output_len,
        ):
            raise AssertionError(
                f"request {index}: success={response.success}, "
                f"lengths={response.input_len}/{response.output_len}; "
                f"expected {input_len}/{output_len}"
            )
        if any(
            not math.isfinite(v) or v < 0
            for v in (
                response.prefill_time,
                response.total_time,
                response.wait_time,
                response.decode_time_per_token,
            )
        ):
            raise AssertionError(f"request {index}: invalid timing")


def route_for_worker(worker, port, role):
    host, grpc_port = worker["address"].rsplit(":", 1)
    # The normal GenerateStreamCall frontend filters DECODE role_addrs unless
    # decode_entrance is enabled. PDFUSION selects the exact endpoint without
    # initiating PD transfer; the backend remains a DECODE synthetic-KV engine.
    return {
        "role": "PDFUSION" if role == "decode" else "PREFILL",
        "ip": host,
        "http_port": port,
        "grpc_port": int(grpc_port),
    }


def ready_routes(port, role, timeout=300):
    import requests

    expected = 8 if role == "decode" else 1
    deadline = time.monotonic() + timeout
    last = None
    with requests.Session() as session:
        session.trust_env = False
        while time.monotonic() < deadline:
            try:
                response = session.get(
                    f"http://127.0.0.1:{port}/worker_status", timeout=10
                )
                response.raise_for_status()
                last = response.json()
                workers = last.get("results", [])
                if (
                    int(last.get("dp_size", 0)) == expected
                    and len(workers) == expected
                    and {int(w["dp_rank"]) for w in workers} == set(range(expected))
                    and all(w["alive"] is True for w in workers)
                    and len({w["address"] for w in workers}) == expected
                ):
                    routes = [
                        route_for_worker(w, port, role)
                        for w in sorted(workers, key=lambda w: int(w["dp_rank"]))
                    ]
                    return routes, last
            except (requests.RequestException, ValueError, KeyError) as error:
                last = str(error)
            time.sleep(1)
    raise RuntimeError(f"not all {expected} DP workers became ready: {last}")


def validate_startup(text, role):
    expected_dp = 8 if role == "decode" else 1
    expected_cp = 1 if role == "decode" else 8
    ranks = re.findall(
        r"parallelism: world=8 tp=(\d+)/(\d+) ep=(\d+)/8 dp=(\d+)/(\d+)", text
    )
    actual = {
        (int(tp_rank), int(tp), int(ep), int(dp_rank), int(dp))
        for tp_rank, tp, ep, dp_rank, dp in ranks
    }
    expected = {
        (0, 1, rank, rank if role == "decode" else 0, expected_dp) for rank in range(8)
    }
    if actual != expected:
        raise AssertionError(f"unexpected attention TP / EP / DP topology: {actual}")
    # CP uses get_attn_tp_size()==1; the model's real CP-enabled branch logs
    # this capacity reduction for our fixed 66560-token sequence bound.
    capacities = re.findall(r"CP=(\d+): max_tokens_per_rank", text)
    if role == "prefill" and (len(capacities) < 8 or set(capacities) != {"8"}):
        raise AssertionError(f"expected eight CP8 capacity records, got {capacities}")
    if role == "decode" and any(int(size) != 1 for size in capacities):
        raise AssertionError("decode unexpectedly enabled attention CP")
    warmup = re.findall(r"cp_enabled=(True|False) cp_size=(\d+)", text)
    if any(
        int(size) != expected_cp or enabled != str(expected_cp > 1)
        for enabled, size in warmup
    ):
        raise AssertionError(f"unexpected optional CP warmup records: {warmup}")
    if f"SM90 DeepEP V2: role={role}" not in text:
        raise AssertionError("DeepEP V2 was not initialized for the selected role")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("prefill", "decode"))
    parser.add_argument("--mode", required=True, choices=("smoke", "perf"))
    args = parser.parse_args()
    checkpoint = os.environ.get("CHECKPOINT_PATH", "")
    if not checkpoint or not Path(checkpoint, "config.json").is_file():
        raise ValueError(
            "set CHECKPOINT_PATH to the local DeepSeek V4.1 Flash checkpoint"
        )
    tokenizer = os.environ.get("TOKENIZER_PATH", checkpoint)
    options, env, batches, output_len = recipe(
        args.role, args.mode, checkpoint, tokenizer
    )
    os.environ.update(env)
    output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    # Absolute cache usable under Bazel even when HOME is unset.
    os.environ.setdefault(
        "EP_JIT_CACHE_DIR",
        str(Path(os.environ.get("TEST_TMPDIR", "/tmp")) / "deepep_v2"),
    )
    jit_cache = configure_jit_cache()

    import torch

    from rtp_llm.config.log_config import setup_logging
    from rtp_llm.test.perf_test.batch_perf_impl import (
        BatchPerfImpl,
        _curl_server_single_worker,
    )
    from rtp_llm.test.perf_test.server import EngineServer
    from rtp_llm.test.perf_test.test_util import create_query

    setup_logging()
    if (
        torch.cuda.device_count() != 8
        or any(torch.cuda.get_device_capability(i) != (9, 0) for i in range(8))
        or not torch.version.cuda.startswith("13.")
    ):
        raise RuntimeError("requires eight locked SM90 GPUs and CUDA13")
    decode = args.role == "decode"
    dp = 8 if decode else 1
    engine_args = [
        item for key, value in options.items() for item in ("--" + key, str(value))
    ]
    config = argparse.Namespace(
        partial=1 if decode else 2, result_dir=str(output), dp_size=dp
    )
    EngineServer.propagate_engine_env(engine_args)
    server = EngineServer(config, engine_args)
    evidence = {
        "role": args.role,
        "mode": args.mode,
        "engine_args": options,
        "env": env,
        "jit_cache": jit_cache,
        "dp_size": dp,
        "cp_size": 1 if decode else 8,
        "numerical_golden": "separate existing smoke targets",
        "decode_input": "synthetic KV" if decode else "real prefill",
        "rows": [],
        "status": "RUNNING",
    }
    evidence_path = output / "sm90_role_results.json"
    evidence_path.write_text(json.dumps(evidence, indent=2))
    memory_log = (output / "gpu_memory.csv").open("w")
    memory_sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,uuid,memory.used",
            "--format=csv,noheader,nounits",
            "--loop-ms=500",
        ],
        stdout=memory_log,
        stderr=subprocess.STDOUT,
    )
    evidence["gpu_memory"] = "gpu_memory.csv: whole-device MiB sampled every 500ms"
    profiling = os.environ.get("DSV41_PERF_PROFILE", "0") == "1"
    evidence["profile_after_measurements"] = profiling
    if profiling:
        # The control RPC broadcasts within TP, not across independent DP ranks.
        evidence["profile_scope"] = "rank0 (TP1)" if decode else "TP group"
        os.environ["PERF_PREARM_PROFILE"] = "1"
    try:
        queries = create_query(tokenizer, list(LENGTHS))
        server.start(
            max_seq_len=66560,
            max_concurrency=max(batches),
            server_start_timeout=int(
                os.environ.get("DSV41_SERVER_START_TIMEOUT", "1600")
            ),
        )
        routes, status = ready_routes(server.port, args.role)
        evidence["worker_status"] = status
        evidence["request_routes"] = routes
        validate_startup(
            Path(server.log_file_path).read_text(errors="replace"), args.role
        )

        startup = Path(server.log_file_path).read_text(errors="replace")
        enabled = env["DSV41_ENGRAM_PREFETCH"] == "1"
        if ("SM90 Engram UVA prefetch enabled" in startup) != enabled:
            raise AssertionError(
                "Engram prefetch startup does not match requested switch"
            )
        evidence["server_ready_unix_s"] = time.time()

        class CheckedBatch(BatchPerfImpl):
            def _curl_server_responses(self, profile=False):
                configs = [
                    dict(self.generate_config, role_addrs=[routes[i % dp]])
                    for i in range(self.batch_size)
                ]
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
                    futures = [
                        executor.submit(
                            _curl_server_single_worker,
                            i,
                            self.base_port,
                            self.input_queries[i],
                            self.is_decode,
                            self.decode_test_length,
                            self.wait_time,
                            profile,
                            configs[i],
                            self.profile_trace_name,
                        )
                        for i in range(self.batch_size)
                    ]
                    responses = [future.result() for future in futures]
                validate_responses(
                    responses, self.batch_size, self.expected_input, output_len
                )
                if not profile:
                    self.round_seconds.append(time.monotonic() - started)
                    self.round_responses.append([vars(r).copy() for r in responses])
                return responses

        # Finish every formal case before initializing the profiler.
        for profile in (False, True) if profiling else (False,):
            if profile:
                evidence["profile_started_unix_s"] = time.time()
            for local_batch in batches:
                for length in LENGTHS:
                    impl = CheckedBatch(
                        server.port,
                        dp,
                        local_batch * dp,
                        queries[length],
                        decode,
                        wait_time=600,
                        decode_test_length=output_len,
                        profile=profile,
                        generate_config={
                            "reuse_cache": False,
                            "top_k": 1,
                            "top_p": 1.0,
                            "random_seed": 42,
                            "force_sp_accept": False,
                            "can_use_pd_separation": False,
                        },
                        profile_trace_name=f"{args.role}_bs{local_batch}_seq{length}",
                        warmup_runs=0 if profile else 1,
                        measure_runs=1 if profile or args.mode == "smoke" else 5,
                        profile_runs=int(profile),
                        log_path=server.log_file_path,
                        log_flush_offset=Path(server.log_file_path).stat().st_size,
                    )
                    impl.expected_input = length
                    impl.round_seconds = []
                    impl.round_responses = []
                    started_unix_s = time.time()
                    try:
                        metrics = impl.run()
                    finally:
                        impl.executor.shutdown(wait=True)
                    if profile:
                        continue
                    row = {
                        "started_unix_s": started_unix_s,
                        "finished_unix_s": time.time(),
                        "input_len": length,
                        "output_len": output_len,
                        "batch_per_rank": local_batch,
                        "global_batch": local_batch * dp,
                        "warmup_round_seconds": impl.round_seconds[0],
                        "formal_round_seconds": impl.round_seconds[1:],
                        "metrics": asdict(metrics),
                        "formal_responses": impl.round_responses[1:],
                        "profile_runs": 0,
                    }
                    evidence["rows"].append(row)
                    evidence_path.write_text(json.dumps(evidence, indent=2))
                    logging.info(
                        "SM90_ROLE_RESULT %s",
                        json.dumps(
                            {k: v for k, v in row.items() if k != "formal_responses"}
                        ),
                    )
        evidence["status"] = "PASS"
    except BaseException as error:
        evidence["status"] = "FAIL"
        evidence["error"] = repr(error)
        raise
    finally:
        evidence_path.write_text(json.dumps(evidence, indent=2))
        try:
            server.stop()
        finally:
            memory_sampler.terminate()
            memory_sampler.wait(timeout=10)
            memory_log.close()


if __name__ == "__main__":
    main()
