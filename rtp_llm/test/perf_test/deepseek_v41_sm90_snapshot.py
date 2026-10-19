"""Manual full-layer SM90 snapshot; formal measurements precede all profiling."""

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from rtp_llm.test.perf_test.deepseek_v41_sm90_role_test import (
    configure_jit_cache,
    ready_routes,
    recipe,
    validate_startup,
)

LENGTHS = [1024 * n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
RATIOS = [0, 0.6, 0.7, 0.8, 0.9, 0.95]
BATCHES = [1, 4, 8, 16, 32, 64, 128, 256]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["prefill", "decode"], required=True)
    args = parser.parse_args()
    decode = args.role == "decode"
    cp = os.environ["CHECKPOINT_PATH"]
    options, env, _, _ = recipe(args.role, "perf", cp, cp)
    max_local_batch = int(os.environ.get("SNAPSHOT_MAX_LOCAL_BATCH", "256"))
    local_batches = [b for b in BATCHES if b <= max_local_batch]
    options.update(
        reuse_cache=int(not decode),
        max_context_batch_size=max_local_batch if decode else 1,
        decode_capture_config=",".join(map(str, local_batches)),
        dsv4_fixed_pool_blocks=1024 if decode else 2112,
    )
    cache_mb = int(os.environ.get("SNAPSHOT_KV_CACHE_MB", "0"))
    if cache_mb:
        options["kv_cache_mem_mb"] = cache_mb
    # The measured prefill follows the same bounded CED path as P/D deployment.
    env.update(
        DSV41_CED="1",
        DSV41_SWA_BOUNDED_REPLAY="1",
        SP_TYPE="",
        DSV41_SM90_DECODE_GRAPH_MAX_ROWS=str(max_local_batch),
    )
    os.environ.update(env)
    out = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    configure_jit_cache()
    import aiohttp
    import requests
    import torch

    from rtp_llm.config.log_config import setup_logging
    from rtp_llm.test.perf_test.batch_perf_impl import _wait_for_trace_flush
    from rtp_llm.test.perf_test.cache_grid_runner import PrefixPromptFactory
    from rtp_llm.test.perf_test.dataclass import ResponseInfo
    from rtp_llm.test.perf_test.server import EngineServer
    from rtp_llm.test.perf_test.test_util import _load_tokenizer

    setup_logging()
    assert torch.cuda.device_count() == 8
    assert all(torch.cuda.get_device_capability(i) == (9, 0) for i in range(8))
    assert torch.version.cuda.startswith("13.")
    engine_args = [v for k, value in options.items() for v in ("--" + k, str(value))]
    EngineServer.propagate_engine_env(engine_args)
    server = EngineServer(
        argparse.Namespace(
            partial=1 if decode else 2, result_dir=str(out), dp_size=8 if decode else 1
        ),
        engine_args,
    )
    data = dict(
        scheduler="per-rank requested batch; idle ranks threshold 0",
        role=args.role,
        options=options,
        env=env,
        status="STARTING",
        rows=[],
        profiles=[],
        batch_semantics="per DP rank; batch field is global, batch_per_rank is requested",
        requested_batches_per_rank=BATCHES,
        graph_max_rows=max_local_batch,
        speculation="disabled",
        decode_input="synthetic KV; no PD transfer",
        prefix_key_stride=None,
        formal_before_profile=True,
        profile_scope="rank0" if decode else "TP8",
        profile_start_step=8 if decode else 0,
        profile_num_steps=20 if decode else 1,
        source_sha=os.environ.get("SNAPSHOT_SOURCE_SHA", "unknown"),
    )

    def save():
        (out / "snapshot.json").write_text(json.dumps(data, indent=2))

    save()
    memory = (out / "gpu_telemetry.csv").open("w")
    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            "--loop-ms=500",
        ],
        stdout=memory,
    )
    try:
        server.start(
            max_seq_len=33792 if decode else 264192,
            max_concurrency=max_local_batch if decode else 1,
            server_start_timeout=int(
                os.environ.get("DSV41_SERVER_START_TIMEOUT", "1600")
            ),
        )
        routes, status = ready_routes(server.port, args.role)
        startup_log = Path(server.log_file_path).read_text(errors="replace")
        validate_startup(startup_log, args.role)
        physical_blocks = {
            int(n)
            for n in re.findall(
                r"cache config: layer_num=.*?seq_size_per_block=(\d+)", startup_log
            )
        }
        virtual_blocks = {
            int(n)
            for n in re.findall(
                r"CP sharded KV cache enabled:.*?virtual_block_size=(\d+)", startup_log
            )
        }
        assert len(physical_blocks) == 1, physical_blocks
        assert len(virtual_blocks) <= 1, virtual_blocks
        cache_stride = next(iter(virtual_blocks or physical_blocks))
        data.update(
            prefix_key_stride=cache_stride, kv_cache_sharded=bool(virtual_blocks)
        )
        data.update(status="RUNNING", worker_status=status, ready_at=time.time())
        save()
        tokenizer = _load_tokenizer(cp)
        factory = PrefixPromptFactory(tokenizer)
        session = requests.Session()
        session.trust_env = False

        def scheduler(batch):
            if decode:
                import grpc

                from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
                from rtp_llm.cpp.model_rpc.proto import (
                    model_rpc_service_pb2_grpc as rpc,
                )

                # Set each rank's threshold to its actual request count.
                for rank, route in enumerate(routes):
                    count = batch // 8 + int(rank < batch % 8)
                    with grpc.insecure_channel(
                        f"{route['ip']}:{route['grpc_port']}"
                    ) as channel:
                        rpc.RpcServiceStub(channel).UpdateSchedulerInfo(
                            pb.UpdateSchedulerInfoRequestPB(
                                scheduler_info=json.dumps(
                                    dict(batch_size=count, mode="decode")
                                )
                            ),
                            timeout=60,
                        )
                return
            r = session.post(
                f"http://127.0.0.1:{server.port}/update_scheduler_info",
                json={"batch_size": 1, "mode": args.role},
                timeout=60,
            )
            r.raise_for_status()
            assert r.json().get("status") == "ok", r.text

        def payload(prompt, route, reuse, output_len):
            return {
                "prompt": prompt,
                "generate_config": dict(
                    max_new_tokens=output_len,
                    min_new_tokens=output_len,
                    top_k=1,
                    top_p=1.0,
                    random_seed=42,
                    reuse_cache=reuse,
                    can_use_pd_separation=False,
                    force_sp_accept=False,
                    role_addrs=[route],
                ),
            }

        def parse_response(raw, started, output_len):
            if "error_code" in raw:
                raise RuntimeError(str(raw)[:1000])
            result = vars(ResponseInfo(raw)).copy()
            result.update(
                aux_info=raw.get("aux_info", {}),
                client_wall_ms=(time.time() - started) * 1000,
                started_at=started,
                finished_at=time.time(),
            )
            assert result.get("input_len", 0) > 0, result
            assert result.get("output_len") == output_len, result
            return result

        def request(prompt, route, reuse, output_len):
            started = time.time()
            with requests.Session() as client:
                client.trust_env = False
                r = client.post(
                    f"http://127.0.0.1:{server.port}/",
                    json=payload(prompt, route, reuse, output_len),
                    timeout=int(os.environ.get("SNAPSHOT_REQUEST_TIMEOUT", "900")),
                )
            r.raise_for_status()
            return parse_response(r.json(), started, output_len)

        async def decode_batch(prompt, batch):
            # Keep all DP8*batch requests in flight without thousands of OS threads.
            timeout = aiohttp.ClientTimeout(
                total=int(os.environ.get("SNAPSHOT_REQUEST_TIMEOUT", "900"))
            )
            async with aiohttp.ClientSession(
                trust_env=False,
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=0),
            ) as client:

                async def one(i):
                    started = time.time()
                    async with client.post(
                        f"http://127.0.0.1:{server.port}/",
                        json=payload(prompt, routes[i % 8], False, 512),
                    ) as r:
                        r.raise_for_status()
                        return parse_response(await r.json(), started, 512)

                return await asyncio.gather(*(one(i) for i in range(batch)))

        def arm(name):
            offset = Path(server.log_file_path).stat().st_size
            r = session.post(
                f"http://127.0.0.1:{server.port}/start_profile",
                json={
                    "gen_timeline": True,
                    "trace_name": name,
                    "start_step": data["profile_start_step"],
                    "num_steps": data["profile_num_steps"],
                    "enable_all_rank": True,
                },
                timeout=60,
            )
            r.raise_for_status()
            assert r.json().get("status") == "ok", r.text
            return offset

        def flush(offset):
            _wait_for_trace_flush(server.log_file_path, offset, 120)

        prefill_lengths = [
            int(n)
            for n in os.environ.get(
                "SNAPSHOT_PREFILL_LENGTHS", ",".join(map(str, LENGTHS))
            ).split(",")
        ]
        prefill_ratios = [
            float(n)
            for n in os.environ.get(
                "SNAPSHOT_PREFILL_RATIOS", ",".join(map(str, RATIOS))
            ).split(",")
        ]
        cases = (
            [
                dict(batch=b * 8, batch_per_rank=b, length=32768, ratio=0)
                for b in local_batches
            ]
            if decode
            else [
                dict(batch=1, length=n, ratio=r)
                for n in prefill_lengths
                for r in prefill_ratios
            ]
        )
        # Optional bounded pilots use the identical runner and workload semantics.
        if os.environ.get("SNAPSHOT_PROFILE_ONLY", "0") == "1":
            phases = [True]
        else:
            phases = (
                [False, True]
                if os.environ.get("SNAPSHOT_PROFILE", "1") == "1"
                else [False]
            )
        case_limit = int(os.environ.get("SNAPSHOT_CASE_LIMIT", "0"))
        if case_limit:
            cases = cases[:case_limit]
        for profile in phases:
            if profile:
                data["profile_started_at"] = time.time()
                save()
            for case_id, case in enumerate(cases):
                if (
                    profile
                    and not decode
                    and (
                        case["length"] not in (1024, 32768, 262144)
                        or case["ratio"] not in (0, 0.95)
                    )
                ):
                    continue
                batch, length, ratio = case["batch"], case["length"], case["ratio"]
                cache_len = int(length * ratio) // cache_stride * cache_stride
                row = dict(
                    case,
                    cache_len_requested=cache_len,
                    requested_ratio=ratio,
                    started_at=time.time(),
                    warmup=[],
                    runs=[],
                    status="RUNNING",
                )
                if ratio and cache_len == 0:
                    row.update(
                        status="UNREPRESENTABLE",
                        reason=f"Actual cache key stride is {cache_stride} tokens; requested prefix is shorter",
                    )
                    if not profile:
                        data["rows"].append(row)
                        save()
                    continue
                label = f"{args.role}_b{batch}_n{length}_hit{ratio}"
                scheduler(batch)
                try:
                    if decode:
                        prompt, _ = factory._exact_text_and_ids(
                            length, "Decode snapshot:"
                        )
                        for repeat in range(1 if profile else 7):
                            offset = arm(label) if profile else None
                            start = time.time()
                            results = asyncio.run(decode_batch(prompt, batch))
                            assert all(
                                r["input_len"] == length for r in results
                            ), results
                            record = dict(
                                responses=results,
                                wall_s=time.time() - start,
                                started_at=start,
                                finished_at=time.time(),
                            )
                            if profile:
                                flush(offset)
                            (
                                row["warmup"]
                                if not profile and repeat < 2
                                else row["runs"]
                            ).append(record)
                    else:
                        # Different prefixes for every round isolate prior runs; seeds are untimed.
                        for repeat in range(1 if profile else 7):
                            unique = case_id * 100 + repeat + (100000 if profile else 0)
                            prompt, prefix, _ = factory.make_case(
                                unique, length, cache_len
                            )
                            seed_result = None
                            if cache_len:
                                seed = factory.make_seed(
                                    unique, prefix, cache_len, cache_stride
                                )
                                seed_result = request(seed, routes[0], True, 1)
                            offset = arm(label) if profile else None
                            response = request(prompt, routes[0], bool(cache_len), 1)
                            if profile:
                                flush(offset)
                            assert response["input_len"] == length, response
                            observed = response["aux_info"].get("reuse_len", -1)
                            record = dict(
                                responses=[response],
                                seed=seed_result,
                                reuse_len=observed,
                                actual_hit_ratio=observed / length,
                                uncached_tokens=length - observed,
                            )
                            (
                                row["warmup"]
                                if not profile and repeat < 2
                                else row["runs"]
                            ).append(record)
                            if observed != cache_len:
                                raise RuntimeError(
                                    f"cache mismatch: requested {cache_len}, observed {observed}"
                                )
                    row["status"] = "PASS"
                except Exception as error:
                    row.update(status="FAIL", error=repr(error))
                    logging.exception("snapshot case failed: %s", label)
                row["finished_at"] = time.time()
                data["profiles" if profile else "rows"].append(row)
                save()
                logging.info("SNAPSHOT_CASE %s %s", label, row["status"])
                if row["status"] == "FAIL":
                    # Preserve first failure for diagnosis; never queue more work into a broken engine.
                    raise RuntimeError(f'{label}: {row["error"]}')
        data["status"] = "PASS"
    except BaseException as error:
        data.update(status="FAIL", error=repr(error))
        raise
    finally:
        data["finished_at"] = time.time()
        save()
        try:
            server.stop()
        finally:
            sampler.terminate()
            sampler.wait(timeout=10)
            memory.close()


if __name__ == "__main__":
    main()
