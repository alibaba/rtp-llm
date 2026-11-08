"""Drive the native CP8 Prefill and DP8 DSpARK 128K step sweeps."""

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
SSH = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
WORDS = "hello world apple tree river blue yellow cat dog time year one two city day street house sky water".split()
ASYNC_FLAGS = {
    "RTP_LLM_STREAM_ASYNC": "useStreamAsync",
    "RTP_LLM_MTP_ASYNC_PREPARE": "enabled",
    "RTP_LLM_DEVICE_INPUT": "enabled",
}
HEALTH_PROBE = """
import json
import sys
from pathlib import Path
from urllib.request import ProxyHandler, build_opener
port, size = map(int, sys.argv[1:3])
if len(sys.argv) == 4:
    capacity = Path(sys.argv[3]) / 'capacity.json'
    if not capacity.is_file() or not json.loads(capacity.read_text())['admissible']:
        sys.exit(1)
opener = build_opener(ProxyHandler({}))
try:
    for rank in range(size):
        with opener.open(f'http://127.0.0.1:{port + rank * 9}/health', timeout=2) as response:
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


def case_config(phase, point):
    decode = phase == "decode"
    batch = point if decode else point // result.FRESH
    return {
        "phase": phase, "point": point, "batch": batch,
        "global_requests": batch * result.WORLD if decode else batch,
        "context": result.CONTEXT, "cached_prefix": result.PREFIX,
        "prefill_new_per_request": result.FRESH, "max_seq_len": result.MAX_SEQUENCE,
        "warmup_steps": result.WARMUP[phase], "measure_steps": result.MEASURE[phase],
        "dspark_gamma": 3 if decode else 5,
        "prefill_cp": 8, "decode_dp": 8, "ep": 8, "enp": 8,
        "yoco_keep_tokens": 140, "decode_graph": decode,
        "optimization_profile": "dspark-grouped-graph-async" if decode else "compact140-fp4-megamoe",
    }


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
        max_seq_len=case["max_seq_len"], ced_mode="bounded", yoco_keep_tokens=140,
        dspark_gamma=case["dspark_gamma"], dspark_confidence="off",
        dspark_verify_mode="dense", fake_balance_expert=False,
        kv_cache_sharded=True, shared_grouped_graph=decode, shared_graph_async=decode, **options,
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
        CONCURRENCY_LIMIT=str(batch if decode else max(16, case["global_requests"])),
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


async def load(config):
    import httpx
    from tokenizers import Tokenizer

    case = config["case"]
    paired = case["phase"] == "decode"
    batch, count = case["batch"], case["global_requests"]
    prefill = "http://" + config["prefill_address"]
    decode = f"http://{config['decode_address']}"
    output = Path(config["output"]) / "http.json"
    waves = []
    tokenizer = Tokenizer.from_file(str(Path(config["checkpoint"]) / "tokenizer.json"))
    prime = tokenizer.encode(prompt()).ids
    for index in (0, count - 1, 239):
        target = tokenizer.encode(prompt(index)).ids
        if (len(prime) != result.PREFIX + 1 or len(target) != result.CONTEXT
                or target[:result.PREFIX] != prime[:-1] or target[result.PREFIX] == prime[-1]):
            raise ValueError("benchmark prompts differ from exact checkpoint token geometry")
    timeout = config["timeout"]
    limits = httpx.Limits(max_connections=count, max_keepalive_connections=min(count, 32))
    async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=False) as client:
        async def post(url, payload):
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"{url} HTTP {response.status_code}: {response.text[:300]}")
            return response.json()

        async def set_batch(url, size, mode):
            receipt = await post(url + "/update_scheduler_info", {"batch_size": size, "mode": mode})
            if receipt.get("status", "ok") != "ok" or receipt.get("error_code"):
                raise RuntimeError(f"batch scheduler rejected {mode} {size}: {receipt}")

        async def generate(text, tokens, length):
            response = await post(prefill + "/", {
                "prompt": text, "generate_config": generation_config(tokens, paired=paired),
            })
            return validate_http(response, length, tokens, paired=paired)

        for _ in range(result.WORLD if paired else 1):
            waves.append(await generate(prompt(), 2 if paired else 1, result.PREFIX + 1))
        save(output, {"priming": waves})
        if paired:
            await set_batch(decode, batch, "decode")
            measured = await asyncio.gather(*(
                generate(prompt(index), result.OUTPUT, result.CONTEXT) for index in range(count)
            ))
        else:
            await set_batch(prefill, batch, "prefill")
            measured = []
            for wave in range(case["warmup_steps"] + case["measure_steps"]):
                prompts = [prompt(wave * batch + index) for index in range(batch)]
                response = await post(prefill + "/batch_infer", {
                    "prompt_batch": prompts, "generate_config": generation_config(1, paired=False),
                })
                rows = response["response_batch"]
                if len(rows) != batch:
                    raise ValueError(f"incomplete prefill wave {wave}")
                measured.extend(validate_http(row, result.CONTEXT, 1, paired=False, prefix=result.PREFIX) for row in rows)
                save(output, {"priming": waves, "requests": measured})
        save(output, {"priming": waves, "requests": measured})


def command(node, action, config):
    runtime = node["runtime"]
    python = node.get("python", "/opt/conda312/bin/python")
    environment = {
        **node.get("environment", {}), "RTP_V41_TEST_RUNTIME": runtime,
        "PYTHONPATH": ":".join([runtime, *node.get("pythonpath", [])]),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "GPU_COUNT": "8",
        "PYTHONDONTWRITEBYTECODE": "1", "RTP_GPU_ACQUIRE_TIMEOUT": "180",
    }
    child = ["env", *(f"{key}={value}" for key, value in environment.items()),
             python, "-B", "-m", MODULE, action, json.dumps(config)]
    script = "source /usr/local/PPU_SDK/envsetup.sh >/dev/null && cd " + shlex.quote(runtime) + " && exec " + shlex.join(child)
    remote = ["pouch", "exec", "-i", "--user", "0", node["container"], "bash", "-c", script]
    return [*SSH, node["ssh"], shlex.join(remote)]


def wait_ready(process, config, timeout):
    node, role = config["node"], config["role"]
    port = node["address"].rsplit(":", 1)[1]
    remote = ["pouch", "exec", "--user", "0", node["container"],
              node.get("python", "/opt/conda312/bin/python"), "-B", "-c", HEALTH_PROBE,
              port, "8" if role == "decode" else "1"]
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
            result.qualify_prefill_optimizations(directory / "prefill", case["point"])
            row.update(result.prefill_result(directory / "prefill", case["point"]))
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
    parser.add_argument("--points", type=lambda value: tuple(map(int, value.split(","))))
    parser.add_argument("--timeout", type=int, default=7200)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    phases = tuple(result.POINTS) if args.phase == "all" else (args.phase,)
    if args.points is not None and (args.phase == "all" or len(set(args.points)) != len(args.points)
                                    or not set(args.points) <= set(result.POINTS[args.phase])):
        parser.error("--points selects a unique subset of the chosen single-phase sweep")
    cases = [case_config(phase, point) for phase in phases for point in result.POINTS[phase]
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
