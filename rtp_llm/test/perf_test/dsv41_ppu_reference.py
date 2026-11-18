"""Drive the native cached Prefill and DP8 DSpARK 128K step sweeps."""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import select
import shlex
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

from rtp_llm.test.perf_test import dsv41_ppu_reference_results as result

MODULE = "rtp_llm.test.perf_test.dsv41_ppu_reference"
PREFILL_PROFILES = ("cp8-compact140", "cp8-yoco-off", "dp8-compact140")
OWNER_PORT_STRIDE = 9
DP_PREQUEUE_ENV = "RTP_V41_REFERENCE_DP_PREQUEUE_MS"
DP_FIRST_PREQUEUE_ENV = "RTP_V41_REFERENCE_DP_FIRST_PREQUEUE_MS"
SSH = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
WORDS = "hello world apple tree river blue yellow cat dog time year one two city day street house sky water".split()
ASYNC_FLAGS = {
    "RTP_LLM_STREAM_ASYNC": "useStreamAsync",
    "RTP_LLM_MTP_ASYNC_PREPARE": "enabled",
    "RTP_LLM_DEVICE_INPUT": "enabled",
}
HEALTH_PROBE = f"""
import json
import sys
from pathlib import Path
from urllib.request import ProxyHandler, build_opener
port, size = map(int, sys.argv[1:3])
if len(sys.argv) == 4:
    capacity = Path(sys.argv[3]) / 'capacity.json'
    if not capacity.is_file() or not json.loads(capacity.read_text())['admissible']:
        sys.exit(1)
opener = build_opener(ProxyHandler({{}}))
try:
    for rank in range(size):
        with opener.open(f'http://127.0.0.1:{{port + rank * {OWNER_PORT_STRIDE}}}/health', timeout=2) as response:
            if response.status != 200:
                sys.exit(1)
except OSError:
    sys.exit(1)
"""


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def owner_addresses(address, count):
    host, port = address.rsplit(":", 1)
    return [f"{host}:{int(port) + rank * OWNER_PORT_STRIDE}" for rank in range(count)]


def case_config(phase, point, prefill_profile=PREFILL_PROFILES[0], *, prefill_sampling="waves"):
    decode = phase == "decode"
    legacy_prefill = prefill_profile == PREFILL_PROFILES[0]
    dp_prefill = not decode and prefill_profile == "dp8-compact140"
    yoco_off = not decode and prefill_profile == "cp8-yoco-off"
    batch = point if decode else point // result.FRESH * (result.WORLD if yoco_off else 1)
    case = {
        "phase": phase, "point": point, "batch": batch,
        "global_requests": batch * result.WORLD if decode or dp_prefill else batch,
        "context": result.CONTEXT, "cached_prefix": result.PREFIX,
        "prefill_new_per_request": result.FRESH, "max_seq_len": result.MAX_SEQUENCE,
        "warmup_steps": result.WARMUP[phase], "measure_steps": result.MEASURE[phase],
        "dspark_gamma": 3 if decode else (5 if legacy_prefill else 0),
        "prefill_cp": 1 if dp_prefill else 8, "decode_dp": 8, "ep": 8, "enp": 8,
        "yoco_keep_tokens": 0 if yoco_off else 140, "decode_graph": decode,
        "optimization_profile": "dspark-grouped-graph-async" if decode else "compact140-fp4-megamoe",
    }
    if not decode:
        if yoco_off:
            case["optimization_profile"] = "yoco-off-fp4-megamoe"
        case.update(
            prefill_profile=prefill_profile,
            prefill_dp=result.WORLD if dp_prefill else 1,
            fresh_tokens_total=case["global_requests"] * result.FRESH,
            point_unit="fresh_tokens_per_group" if legacy_prefill else "fresh_tokens_per_ppu",
        )
    if prefill_sampling == "sglang-gist":
        if phase != "prefill" or not (
            (dp_prefill and point == 4096) or (legacy_prefill and point == 32768)
        ):
            raise ValueError("SGLang gist sampling requires DP8 B1 or CP8 global B8 cached Prefill")
        case.update(
            prefill_sampling=prefill_sampling, warmup_steps=8, measure_steps=496,
            dspark_gamma=0, fake_balance_expert=True, window_fractions=[0.2, 0.9],
            requests_per_rank=496,
            requests_per_owner=496 * batch,
            sampling_transport="queued RPC batches; native executor owners and all model ranks verified",
        )
    elif prefill_sampling != "waves":
        raise ValueError(f"unknown Prefill sampling: {prefill_sampling}")
    return case


def role_env(config):
    from rtp_llm.tools.v41.launch import preset_env

    case = config["case"]
    role = config["role"]
    decode = role == "decode"
    paired = case["phase"] == "decode"
    batch = case["batch"]
    node = config["node"]
    output = Path(config["output"])
    parent = dict(os.environ)
    for key in (
        "MODEL_SERVICE_CONFIG", "REMOTE_RPC_SERVER_IP", "REMOTE_SERVER_PORT",
        "RTP_V41_SHARED_EVIDENCE_DIR", "RTP_V41_SHARED_VALIDATE_STATE",
        "RTP_V41_DSPARK_OBSERVE_DIR", "RTP_V41_PERF_EVIDENCE_DIR",
        "RTP_V41_REFERENCE_PERF_DIR", "ACCL_USE_NICS",
        "RTP_V41_PPU_YOCO_KEEP_TOKENS",
    ):
        parent.pop(key, None)
    backend = "grouped" if decode else "megamoe"
    precision = "w4a8" if decode else "w4a4"
    parent.update(RTP_V41_PPU_MOE_BACKEND=backend, RTP_V41_PPU_MOE_PRECISION=precision)
    options = {
        "prefill_address": config["prefill_address"],
        "decode_address": config["decode_address"],
        "rdma": True,
    } if paired else {}
    env = preset_env(
        role, config["checkpoint"], parent=parent,
        start_port=int(node["address"].rsplit(":", 1)[1]),
        max_seq_len=case["max_seq_len"],
        prefill_parallel="cp" if case["prefill_cp"] == result.WORLD else "dp",
        ced_mode="bounded" if case["yoco_keep_tokens"] else "off",
        yoco_keep_tokens=case["yoco_keep_tokens"] or None,
        dspark_gamma=case["dspark_gamma"], dspark_confidence="off",
        dspark_verify_mode="dense", fake_balance_expert=case.get("fake_balance_expert", False),
        kv_cache_sharded=case["prefill_cp"] == result.WORLD,
        shared_grouped_graph=decode, shared_graph_async=decode, **options,
    )
    p_batch = min(case["global_requests"], 16) if paired else batch
    kv_budget = node.get("kv_cache_mem_mb_by_batch", {}).get(str(batch), node.get("kv_cache_mem_mb", 8192))
    env.update(
        REUSE_CACHE="1", PERF_TEST="0", PROFILE="0", ENABLE_MODEL_INPUTS_LOG="0",
        LOG_PATH=str(output / "logs"), RTP_V41_TOKEN_MAP_CACHE=str(output / "token-map.json"),
        USE_BATCH_DECODE_SCHEDULER=str(int(decode or not paired)),
        BATCH_DECODE_SCHEDULER_BATCH_SIZE="1",
        KV_CACHE_MEM_MB=str(kv_budget), RESERVE_BLOCK_RATIO="5",
        MAX_CONTEXT_BATCH_SIZE=str(1 if decode else p_batch),
        MAX_BATCH_TOKENS_SIZE=str(result.CONTEXT * (1 if decode else p_batch) + 1),
        CONCURRENCY_LIMIT=str(batch if decode else max(
            16, case["global_requests"], case.get("requests_per_owner", case.get("requests_per_rank", 0))
        )),
        **dict.fromkeys(ASYNC_FLAGS, str(int(decode))),
        RTP_V41_PPU_MHC_BACKEND="mega", RTP_V41_PPU_GRAPH_MEGA_MHC="1",
        RTP_V41_PPU_PREFILL_DENSE_FP4_SCORE="1", RTP_V41_PPU_PREFILL_SPARSE_FP4_SCORE="1",
        RTP_V41_PPU_GRAPH_PACKED_ATTENTION="0", RTP_V41_PPU_GRAPH_VISIBLE_INDEXER=str(int(decode)),
        RTP_TEST_GRACEFUL_STOP_TIMEOUT="180",
        FRONTEND_PRE_STOP_DRAIN_SECONDS="0", DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS="0",
        BACKEND_POST_FRONTEND_DRAIN_SECONDS="0",
    )
    if decode or not paired:
        env["RTP_V41_REFERENCE_PERF_DIR"] = str(output)
    if decode:
        env["DECODE_CAPTURE_CONFIG"] = f"1,{batch}"
    return env


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def record_async_flags(output):
    evidence = {key: {} for key in ASYNC_FLAGS}
    with (output / "logs" / "engine.log").open(errors="replace") as stream:
        for line in stream:
            for key, state in ASYNC_FLAGS.items():
                if f"{key}=1 -> {state}=1" in line and (rank := re.search(r"\[RANK (\d+)\]", line)):
                    evidence[key][rank[1]] = line.strip()
    save(output / "async-flags.json", evidence)


def check_decode_capacity(output, batch):
    """Check the real opaque FULL pool before attempting to fill a fixed batch."""
    log = output / "logs" / "engine.log"
    if not log.exists():
        return False
    text = log.read_text(errors="replace")
    def observed(pattern):
        return {int(rank): int(value) for rank, value in re.findall(r"\[RANK (\d+)\][^\n]*?" + pattern, text)}

    spans = observed(r"cache config: [^\n]*?seq_size_per_block=(\d+)")
    blocks = observed(r"pool_name=hca_kv, [^;\]\n]*?blocks=(\d+)")
    reserves = observed(r"CoordinatorCacheManager set reserve blocks: ratio=(\d+)%")
    ranks = set(range(result.WORLD))
    if any(report.keys() != ranks for report in (spans, blocks, reserves)):
        return False
    span = min(spans.values())
    available = min((blocks[rank] - 1) * (100 - reserves[rank]) // 100 for rank in ranks)
    demand = batch * ((result.MAX_SEQUENCE + span - 1) // span)
    enough = available >= demand
    save(output / "capacity.json", {"available_blocks": available, "required_blocks": demand,
                                    "tokens_per_block": span, "admissible": enough})
    if not enough:
        raise RuntimeError(f"Decode KV capacity: need {demand} blocks per rank, have {available}")
    return True


def serve(config):
    from rtp_llm.models_py.modules.dsv41.ppu.test.pd_rdma_role import stop_shared_owned
    from rtp_llm.test.utils.device_resource import DeviceResource
    from rtp_llm.tools.v41.preflight import collect_report

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    runtime = Path(config["node"]["runtime"])
    snapshot = read(runtime / ".v41_snapshot.json")
    native = read(runtime / "native-runtime.json")
    for key in ("snapshot_id", "source_sha256"):
        if native[key] != snapshot[key]:
            raise ValueError(f"native runtime differs from snapshot: {key}")
    if snapshot["snapshot_id"] != config["snapshot_id"]:
        raise ValueError("incorrect benchmark snapshot")
    if len(native["libraries"]) != 4:
        raise ValueError("native manifest requires all four build outputs")
    for library in native["libraries"].values():
        if sha256(runtime / library["runtime_path"]) != library["sha256"]:
            raise ValueError(f"native library differs from manifest: {library['runtime_path']}")
    checkpoint = Path(config["checkpoint"])
    env = role_env(config)
    decode = config["role"] == "decode"
    if decode:
        log_config = output / "alog.conf"
        template = runtime / "rtp_llm" / "config" / "alog.conf"
        log_config.write_text(template.read_text().replace("fileName=logs/", f"fileName={output / 'logs'}/"))
        env["FT_ALOG_CONF_PATH"] = str(log_config)
    os.environ.update(env)
    save(output / "launch.json", {
        "snapshot": {key: snapshot[key] for key in ("snapshot_id", "source_sha256", "public_commit", "internal_commit")},
        "role": config["role"], "case": config["case"],
        "graph_eligibility": "DSpARK verify uses paged committed Indexer KV and the tentative journal; packed dual-pool attention and Graph Mega-mHC are ordinary Decode paths",
        "checkpoint": str(checkpoint),
        "checkpoint_files": {name: sha256(checkpoint / name) for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")},
        "env": {key: value for key, value in env.items() if key.startswith("RTP_V41_") or key in ASYNC_FLAGS or key in (
            "USE_BATCH_DECODE_SCHEDULER", "REUSE_CACHE", "DECODE_CAPTURE_CONFIG", "CONCURRENCY_LIMIT",
            "KV_CACHE_MEM_MB", "RESERVE_BLOCK_RATIO", "MAX_CONTEXT_BATCH_SIZE", "MAX_BATCH_TOKENS_SIZE",
            "DSV41_CED", "DSV41_SWA_BOUNDED_REPLAY", "DP_SIZE", "EP_SIZE", "PREFILL_CP_SIZE",
            "SP_TYPE", "GEN_NUM_PER_CIRCLE", "FAKE_BALANCE_EXPERT",
            "RTP_LLM_SCHEDULER_POLL_INTERVAL_MS",
            "DSV4_RECORD_FUNCTION_RANGES", "LOG_LEVEL", "ENGINE_CPU_AFFINITY",
        )},
    })
    with DeviceResource(8) as lease:
        if set(map(int, lease.gpu_ids)) != set(range(8)):
            raise RuntimeError(f"unexpected device lease {lease.gpu_ids}")
        inventory = subprocess.check_output(["ppu-smi"], text=True)
        (output / "gpu-before.txt").write_text(inventory)
        if "No running processes found" not in inventory:
            raise RuntimeError("PPUs already have active processes")
        base = int(config["node"]["address"].rsplit(":", 1)[1])
        for port in range(base - 11, base + 72):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("0.0.0.0", port))
        report = collect_report(require_backend=True)
        save(output / "preflight.json", report)
        if not report["prerequisites_ok"]:
            raise RuntimeError(f"PPU runtime preflight failed in {output}")
        from rtp_llm.models_py.modules.dsv41.ppu.engram import load_token_map

        load_token_map(str(checkpoint / "tokenizer.json"), env["RTP_V41_TOKEN_MAP_CACHE"])
        command = [sys.executable, "-B", "-m", "rtp_llm.start_server"]
        with (output / "process.log").open("x") as log:
            child = subprocess.Popen(command, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
            checked_capacity = not decode
            try:
                while child.poll() is None:
                    if not checked_capacity:
                        checked_capacity = check_decode_capacity(output, config["case"]["batch"])
                    if select.select([sys.stdin], [], [], 2)[0]:
                        break
            finally:
                status = stop_shared_owned(child, output)
                (output / "gpu-after.txt").write_text(subprocess.check_output(["ppu-smi"], text=True))
            if status != 0:
                raise RuntimeError(f"owned service exited {status}, see {output / 'process.log'}")
        if decode:
            record_async_flags(output)


@lru_cache(maxsize=1)
def prompt_words():
    randomizer = random.Random(20261108)
    prefix = "hello " + " ".join(randomizer.choices(WORDS, k=result.PREFIX - 1)) + " "
    tail = " ".join(randomizer.choices(WORDS, k=result.FRESH - 14)) + " "
    return prefix, tail


def prompt(index=None):
    prefix, tail = prompt_words()
    if index is None:
        # One lookahead token permits reuse of the last full CP cache page.
        return prefix
    # Place the salt in the first uncached page so no two timed requests can
    # reuse part of one another's suffix. The HTTP token counts are verified.
    salt = "world " + "".join("world " if index & (1 << bit) else "hello " for bit in range(12))
    return prefix + salt + tail


def dp_prefill_prime_steps(tokenizer, prime_tokens):
    words = prompt().split()
    # A full4096 seed adds one lookahead row and permanently grows MegaMoE
    # to8192. Page-aligned3840 increments keep setup below the measured4096
    # capacity, matching the reference's fixed workspace. The final256-row
    # remainder still creates exactly PREFIX cached tokens before timing.
    previous = 0
    for boundary in (*range(3840, result.PREFIX, 3840), result.PREFIX):
        # The trailing space is one lookahead token. It leaves a full cache
        # page at the boundary without matching the next stage's first token.
        text = " ".join(words[:boundary]) + " "
        tokens = tokenizer.encode(text).ids
        if (
            len(tokens) != boundary + 1
            or tokens[:boundary] != prime_tokens[:boundary]
            or (boundary < result.PREFIX and tokens[boundary] == prime_tokens[boundary])
        ):
            raise ValueError(f"DP Prefill seed has invalid token geometry at {boundary} tokens")
        yield text, boundary + 1, previous
        previous = boundary


def generation_config(tokens, *, paired):
    return {
        "max_new_tokens": tokens, "min_new_tokens": tokens, "ignore_eos": True,
        "top_k": 1024, "top_p": 0.95, "temperature": 0.3, "random_seed": 20261108,
        "reuse_cache": True, "force_sp_accept": False, "can_use_pd_separation": paired,
        "aux_info": True,
    }


def validate_http(response, length, output, *, paired, prefix=None):
    if response.get("error_code") or not response.get("finished"):
        raise RuntimeError(f"unfinished or failed native request: {str(response)[:300]}")
    aux = response["aux_info"]
    if aux["input_len"] != length or aux["output_len"] != output:
        raise ValueError(f"unexpected request lengths: {aux['input_len']}, {aux['output_len']}")
    if bool(aux.get("pd_sep")) != paired:
        raise ValueError("request did not use the planned native P/D route")
    if prefix is not None and aux.get("total_reuse_len", aux.get("reuse_len")) != prefix:
        raise ValueError(f"expected {prefix} cached tokens: {str(aux)[:500]}")
    return {key: aux[key] for key in (
        "input_len", "output_len", "total_reuse_len", "reuse_len", "prefill_total_reuse_len",
        "decode_total_reuse_len", "decode_local_reuse_len", "pd_sep",
    ) if key in aux}


def dp_prequeue_ms(case, environment, *, first_wave=False):
    if case.get("prefill_profile") != "dp8-compact140":
        return 0
    milliseconds = int(environment.get(DP_PREQUEUE_ENV, "0"))
    if not 0 <= milliseconds <= 10000:
        raise ValueError(f"{DP_PREQUEUE_ENV} must be between 0 and 10000 milliseconds")
    if first_wave:
        first = int(environment.get(DP_FIRST_PREQUEUE_ENV, str(milliseconds)))
        if not milliseconds <= first <= 10000:
            raise ValueError(f"{DP_FIRST_PREQUEUE_ENV} must be between the normal gate and 10000 milliseconds")
        return first
    return milliseconds


async def load(config):
    import httpx
    from tokenizers import Tokenizer

    case = config["case"]
    paired = case["phase"] == "decode"
    batch, count = case["batch"], case["global_requests"]
    prequeue_ms = dp_prequeue_ms(case, os.environ)
    first_prequeue_ms = dp_prequeue_ms(case, os.environ, first_wave=True)
    experiment = {"dp_prequeue_ms": prequeue_ms} if prequeue_ms else {}
    if prequeue_ms:
        experiment["dp_first_prequeue_ms"] = first_prequeue_ms
    if case.get("prefill_profile") == "dp8-compact140":
        experiment["dp_prime_chunk_tokens"] = 3840
    prefill_owners = ["http://" + address for address in owner_addresses(
        config["prefill_address"], case.get("prefill_dp", 1)
    )]
    prefill = prefill_owners[0]
    decode = f"http://{config['decode_address']}"
    output = Path(config["output"]) / "http.json"
    tokenizer = Tokenizer.from_file(str(Path(config["checkpoint"]) / "tokenizer.json"))
    prime = tokenizer.encode(prompt()).ids
    for index in (0, count - 1, 239):
        target = tokenizer.encode(prompt(index)).ids
        if (len(prime) != result.PREFIX + 1 or len(target) != result.CONTEXT
                or target[:result.PREFIX] != prime[:-1] or target[result.PREFIX] == prime[-1]):
            raise ValueError("benchmark prompts differ from exact checkpoint token geometry")
    timeout = config["timeout"]
    limits = httpx.Limits(max_connections=count + int(bool(prequeue_ms)), max_keepalive_connections=min(count, 32))
    async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=False) as client:
        async def post(url, **kwargs):
            response = await client.post(url, **kwargs)
            if response.status_code != 200:
                raise RuntimeError(f"{url} HTTP {response.status_code}: {response.text[:300]}")
            return response.json()

        async def set_batch(url, size, mode):
            receipt = await post(url + "/update_scheduler_info", json={"batch_size": size, "mode": mode})
            if receipt.get("status", "ok") != "ok" or receipt.get("error_code") or receipt.get("error"):
                raise RuntimeError(f"batch scheduler rejected {mode} {size}: {receipt}")

        async def generate(url, text, tokens, length, *, prefix=None):
            response = await post(url + "/", json={
                "prompt": text, "generate_config": generation_config(tokens, paired=paired),
            })
            return validate_http(response, length, tokens, paired=paired, prefix=prefix)

        if paired:
            priming = [await generate(prefill, prompt(), 2, result.PREFIX + 1) for _ in range(result.WORLD)]
            save(output, {"priming": priming})
            await set_batch(decode, batch, "decode")
            measured = await asyncio.gather(*(
                generate(prefill, prompt(index), result.OUTPUT, result.CONTEXT) for index in range(count)
            ))
        else:
            if case.get("prefill_profile") == "dp8-compact140":
                priming = []
                for text, length, prefix in dp_prefill_prime_steps(tokenizer, prime):
                    receipts = await asyncio.gather(*(
                        generate(url, text, 1, length, prefix=prefix) for url in prefill_owners
                    ))
                    priming.extend(dict(receipt, **(
                        {"frontend_owner": rank} if case.get("prefill_sampling") == "sglang-gist"
                        else {"rank": rank}
                    )) for rank, receipt in enumerate(receipts))
                    save(output, {"priming": priming})
            else:
                priming = [await generate(prefill, prompt(), 1, result.PREFIX + 1)]
            save(output, dict(experiment, priming=priming))
            # This endpoint broadcasts the batch size to every DP owner.
            if not prequeue_ms:
                await set_batch(prefill, batch, "prefill")
            measured = []
            gist_sampling = case.get("prefill_sampling") == "sglang-gist"
            waves = case["warmup_steps"] + (0 if gist_sampling else case["measure_steps"])
            for wave in range(waves):
                # Serialize before dispatch so rank-local request preparation
                # does not stagger the batch across owners in the event loop.
                payloads = []
                for rank in range(len(prefill_owners)):
                    first = wave * count + rank * batch
                    payloads.append(json.dumps({
                        "prompt_batch": [prompt(first + index) for index in range(batch)],
                        "generate_config": generation_config(1, paired=False),
                    }).encode())
                if prequeue_ms:
                    await set_batch(prefill, batch + 1, "prefill")
                requests = [
                    post(url + "/batch_infer", content=payload, headers={"Content-Type": "application/json"})
                    for url, payload in zip(prefill_owners, payloads)
                ]
                if prequeue_ms:
                    pending = [asyncio.create_task(request) for request in requests]
                    try:
                        # Cold batch-infer tokenization can skew the very
                        # first warmup's owner arrival. This gate is outside
                        # native timing; all later waves retain the same delay.
                        delay = first_prequeue_ms if wave == 0 else prequeue_ms
                        await asyncio.sleep(delay / 1000)
                        for task in pending:
                            if task.done():
                                task.result()
                                raise RuntimeError(f"Prefill wave {wave} finished before the DP admission gate opened")
                        await set_batch(prefill, batch, "prefill")
                        responses = await asyncio.gather(*pending)
                    finally:
                        for task in pending:
                            task.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
                else:
                    responses = await asyncio.gather(*requests)
                for rank, response in enumerate(responses):
                    rows = response["response_batch"]
                    if len(rows) != batch:
                        raise ValueError(f"incomplete prefill wave {wave} on rank {rank}")
                    measured.extend(
                        dict(validate_http(row, result.CONTEXT, 1, paired=False, prefix=result.PREFIX),
                             **({"frontend_owner": rank} if gist_sampling else {"rank": rank}), wave=wave)
                        for row in rows
                    )
                save(output, dict(experiment, priming=priming, requests=measured))
            if gist_sampling:
                # Queue whole measurement batches. Frontend owner IDs
                # need not equal backend ranks; native receipts validate all
                # DP owners received the same number of complete steps. CP8 has
                # one executor owner and keeps eight requests per native step.
                # The central window contains no client wave gates.
                # Eight warmup requests + 496 measured requests on eight ranks
                # fit the existing 12-bit suffix salt without any repetition.
                queued = case["requests_per_owner"]
                payloads = [json.dumps({
                    "prompt_batch": [prompt(case["warmup_steps"] * count + rank * queued + index)
                                     for index in range(queued)],
                    "generate_config": generation_config(1, paired=False),
                }).encode() for rank in range(len(prefill_owners))]
                responses = await asyncio.gather(*(
                    post(url + "/batch_infer", content=payload,
                         headers={"Content-Type": "application/json"})
                    for url, payload in zip(prefill_owners, payloads)
                ))
                for rank, response in enumerate(responses):
                    rows = response["response_batch"]
                    if len(rows) != queued:
                        raise ValueError(f"incomplete queued Prefill on rank {rank}: {len(rows)}/{queued}")
                    measured.extend(
                        dict(validate_http(row, result.CONTEXT, 1, paired=False, prefix=result.PREFIX),
                             frontend_owner=rank, wave=case["warmup_steps"] + index)
                        for index, row in enumerate(rows)
                    )
        save(output, dict(experiment, priming=priming, requests=measured))


def prefill_case_result(path, case):
    local_chunk = case["batch"] * result.FRESH
    if case.get("prefill_sampling") == "sglang-gist":
        result.qualify_prefill_optimizations(
            path, local_chunk, yoco=bool(case["yoco_keep_tokens"]),
            expected_steps=case["warmup_steps"] + case["measure_steps"],
        )
        return result.gist_prefill_result(path, local_chunk, case)
    result.qualify_prefill_optimizations(path, local_chunk, yoco=bool(case["yoco_keep_tokens"]))
    metrics = result.prefill_result(path, local_chunk, owners=case["prefill_dp"])
    if case.get("prefill_profile") == "dp8-compact140":
        http = read(path / "http.json")
        prequeue_ms = http.get("dp_prequeue_ms", 0)
        if prequeue_ms:
            for wave, steps in enumerate(zip(*metrics["sample_steps_by_rank"])):
                if len(set(steps)) != 1:
                    raise ValueError(f"prequeued DP Prefill measured wave {wave} has different owner steps: {steps}")
            metrics.update(dp_prequeue_ms=prequeue_ms,
                           dp_first_prequeue_ms=http.get("dp_first_prequeue_ms", prequeue_ms),
                           dp_prequeue_same_native_step=True)
    return metrics


def command(node, action, config):
    runtime = node["runtime"]
    python = node.get("python", "/opt/conda312/bin/python")
    acquire_timeout = node.get("gpu_acquire_timeout_seconds", 180)
    if type(acquire_timeout) is not int or not 1 <= acquire_timeout <= 3600:
        raise ValueError("gpu_acquire_timeout_seconds must be an integer from 1 to 3600")
    environment = {
        **node.get("environment", {}), "RTP_V41_TEST_RUNTIME": runtime,
        "PYTHONPATH": ":".join([runtime, *node.get("pythonpath", [])]),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "GPU_COUNT": "8",
        "PYTHONDONTWRITEBYTECODE": "1", "RTP_GPU_ACQUIRE_TIMEOUT": str(acquire_timeout),
    }
    child = ["env", *(f"{key}={value}" for key, value in environment.items()),
             python, "-B", "-m", MODULE, action, json.dumps(config)]
    script = "source /usr/local/PPU_SDK/envsetup.sh >/dev/null && cd " + shlex.quote(runtime) + " && exec " + shlex.join(child)
    remote = ["pouch", "exec", "-i", "--user", "0", node["container"], "bash", "-c", script]
    return [*SSH, node["ssh"], shlex.join(remote)]


def wait_ready(process, config, timeout):
    node, role = config["node"], config["role"]
    port = node["address"].rsplit(":", 1)[1]
    owners = result.WORLD if role == "decode" else config["case"].get("prefill_dp", 1)
    remote = ["pouch", "exec", "--user", "0", node["container"],
              node.get("python", "/opt/conda312/bin/python"), "-B", "-c", HEALTH_PROBE,
              port, str(owners)]
    if role == "decode":
        remote.append(config["output"])
    probe = [*SSH, node["ssh"], shlex.join(remote)]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{role} launcher exited before readiness ({process.returncode})")
        try:
            if subprocess.run(probe, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=20).returncode == 0:
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(3)
    raise TimeoutError(f"{role} readiness timed out")


def copy_receipts(node, remote, destination):
    destination.mkdir()
    rsync = shlex.join(["pouch", "exec", "-i", "--user", "0", node["container"], "rsync"])
    subprocess.run([
        "rsync", "-rtz", "-e", shlex.join(SSH), "--rsync-path=" + rsync,
        "--exclude", "token-map.json", "--exclude", "logs/", f"{node['ssh']}:{remote}/", str(destination) + "/",
    ], check=True, timeout=180)


def run_case(args, deployment, case, directory):
    print(f"Starting {case['phase']} {case['point']}", flush=True)
    directory.mkdir(exist_ok=False)
    roles = ("decode", "prefill") if case["phase"] == "decode" else ("prefill",)
    port_offset = 192 * result.POINTS[case["phase"]].index(case["point"])
    nodes = {}
    for role in roles:
        node = deployment[role]
        host, base = node["address"].rsplit(":", 1)
        nodes[role] = dict(node, address=f"{host}:{int(base) + port_offset}")
    configs = {}
    for role in roles:
        node = nodes[role]
        output = str(Path(node["run_root"]) / args.result_dir.name / directory.name / role)
        configs[role] = {
            **{key: deployment[key] for key in ("checkpoint", "snapshot_id")},
            "case": case, "node": node, "role": role, "output": output,
            "prefill_address": nodes["prefill"]["address"],
            "decode_address": nodes.get("decode", {}).get("address"), "timeout": args.timeout,
        }
    failures = []
    services = {}
    with ExitStack() as stack:
        try:
            for role in roles:
                config = configs[role]
                log = stack.enter_context((directory / f"{role}-launcher.log").open("x"))
                services[role] = subprocess.Popen(command(config["node"], "serve", config), stdin=subprocess.PIPE,
                                                  stdout=log, stderr=subprocess.STDOUT)
                wait_ready(services[role], config, min(args.timeout, 1800))
            config = dict(configs["prefill"])
            config["prefill_address"] = "127.0.0.1:" + config["prefill_address"].rsplit(":", 1)[1]
            with (directory / "client.log").open("x") as log:
                subprocess.run(command(config["node"], "load", config), stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
        except subprocess.CalledProcessError as error:
            failures.append(f"native workload exited {error.returncode}, see client.log")
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            failures.append(str(error))
        finally:
            for process in services.values():
                process.stdin.close()
            for role, process in services.items():
                try:
                    if process.wait(timeout=320) != 0:
                        failures.append(f"{role} owned launcher exited {process.returncode}")
                except subprocess.TimeoutExpired:
                    process.terminate()
                    failures.append(f"{role} owned launcher stop timed out")
                config = configs[role]
                try:
                    copy_receipts(config["node"], config["output"], directory / role)
                except (OSError, subprocess.SubprocessError) as error:
                    failures.append(f"{role} receipt collection: {error}")
    row = dict(case, status="ok", snapshot_id=deployment["snapshot_id"])
    try:
        if failures:
            raise RuntimeError("; ".join(failures))
        if case["phase"] == "prefill":
            row.update(prefill_case_result(directory / "prefill", case))
        else:
            flags = read(directory / "decode" / "async-flags.json")
            if not all(set(flags[key]) == set(map(str, range(result.WORLD))) for key in ASYNC_FLAGS):
                raise ValueError("native DSpARK Graph async options did not all execute on all ranks")
            row.update(result.decode_result(directory / "decode", case["batch"]))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        evidence = "\n".join(path.read_text(errors="replace")[-200000:] for path in directory.rglob("*.log"))
        lower = (str(error) + "\n" + evidence).lower()
        status = "oom" if "out of memory" in lower or "cudaerroroutofmemory" in lower else (
            "timeout" if "timed out" in lower else "failed"
        )
        capacity = directory / "decode" / "capacity.json"
        if capacity.exists() and not (reported := read(capacity))["admissible"]:
            row.update(status="capacity", error=f"Decode KV needs {reported['required_blocks']} blocks per rank, "
                       f"has {reported['available_blocks']}")
        else:
            row.update(status=status, error=str(error))
    save(directory / "result.json", row)
    print(f"{directory.name}: {row['status']} {row.get('error', '')}", flush=True)
    return row["status"] == "ok"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] in ("serve", "load"):
        config = json.loads(argv[1])
        if argv[0] == "serve":
            serve(config)
        else:
            asyncio.run(load(config))
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prefill", "decode", "all"))
    parser.add_argument("--protocol", choices=("reference-128k",), required=True)
    parser.add_argument("--deployment", type=Path, help="JSON with checked runtimes and SSH endpoints")
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument(
        "--prefill-profile", choices=PREFILL_PROFILES, default=PREFILL_PROFILES[0],
        help="The legacy CP8 profile labels whole-group chunks; comparison profiles use fresh tokens per PPU.",
    )
    parser.add_argument("--points", type=lambda value: tuple(map(int, value.split(","))))
    parser.add_argument("--prefill-sampling", choices=("waves", "sglang-gist"), default="waves")
    parser.add_argument("--timeout", type=int, default=7200)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    phases = tuple(result.POINTS) if args.phase == "all" else (args.phase,)
    if args.points is not None and (args.phase == "all" or len(set(args.points)) != len(args.points)
                                    or not set(args.points) <= set(result.POINTS[args.phase])):
        parser.error("--points selects a unique subset of the chosen single-phase sweep")
    if args.prefill_sampling == "sglang-gist" and (
        args.phase != "prefill" or (
            args.prefill_profile, args.points
        ) not in (("dp8-compact140", (4096,)), ("cp8-compact140", (32768,)))
    ):
        parser.error("--prefill-sampling sglang-gist requires DP8 --points 4096 or CP8 --points 32768")
    cases = [case_config(phase, point, args.prefill_profile, prefill_sampling=args.prefill_sampling)
             for phase in phases for point in result.POINTS[phase]
             if args.points is None or point in args.points]
    if args.dry_run:
        print(json.dumps(cases, indent=2))
        return 0
    if not args.result_dir or args.timeout <= 0:
        parser.error("a --result-dir and positive --timeout are required")
    if args.plot_only:
        result.plot_results(args.result_dir, phases)
        return 0
    if not args.deployment:
        parser.error("--deployment is required for a device run")
    args.result_dir.mkdir(parents=True, exist_ok=False)
    deployment = read(args.deployment)
    save(args.result_dir / "plan.json", {"snapshot_id": deployment["snapshot_id"], "cases": cases})
    outcomes = []
    for case in cases:
        directory = args.result_dir / f"{case['phase']}-{case['point']}"
        outcomes.append(run_case(args, deployment, case, directory))
    result.plot_results(args.result_dir, phases)
    return int(not all(outcomes))


if __name__ == "__main__":
    sys.exit(main())
