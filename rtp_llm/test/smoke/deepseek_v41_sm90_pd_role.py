"""Own one SM90 eight-GPU P/D fixture; readiness is not an accuracy result.

Build this py_binary, then run it inside the existing development container.
It holds the existing canonical GPU locks until a matching run-id done file or
its bounded lifetime ends. There is no SSH, deployment operation or GPU cleanup
outside the single MagaServerManager-owned process tree.
"""

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import ipaddress
import json
import os
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

CANONICAL_GLOBAL = Path("/tmp/rtp_llm/smoke/test/gpu_status_lock")
CANONICAL_GPUS = Path("/tmp/rtp_llm/smoke/test/gpu_status")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_runtime_network_env(environ, host_ip, peer_ip):
    """Keep build-only wheel proxies out of this owned runtime process."""
    removed = []
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if name in environ:
            environ.pop(name)
            removed.append(name)
    bypass = []
    for value in (environ.get("NO_PROXY", ""), environ.get("no_proxy", ""),
                  f"localhost,127.0.0.1,0.0.0.0,{host_ip},{peer_ip}"):
        bypass.extend(item.strip() for item in value.split(",") if item.strip())
    environ["NO_PROXY"] = environ["no_proxy"] = ",".join(dict.fromkeys(bypass))
    return removed


def owned_directory(value):
    path = Path(value)
    require(path.is_absolute() and path.resolve() == path, "Require an absolute non-symlink owned directory")
    require(path.parent.is_dir() and path.parent.stat().st_uid == os.getuid(), "Output/cache parent must be owned by this user")
    path.mkdir(mode=0o700, exist_ok=True)
    require(path.is_dir() and path.stat().st_uid == os.getuid(), "Directory is not owned by this user")
    return path


def read_lock(path):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), "Canonical GPU lock must be an existing regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    actual = os.fstat(descriptor)
    if (actual.st_dev, actual.st_ino) != (before.st_dev, before.st_ino):
        os.close(descriptor)
        raise ValueError("Canonical GPU lock changed while opening")
    return descriptor, dict(path=str(path), dev=actual.st_dev, inode=actual.st_ino,
                            uid=actual.st_uid, mode=stat.S_IMODE(actual.st_mode))


@contextlib.contextmanager
def canonical_lease(gpu_ids, timeout, global_path=CANONICAL_GLOBAL, gpu_root=CANONICAL_GPUS):
    # Linux flock permits EX on O_RDONLY. It is the SAME inode and flock ABI
    # as the official FileLock wrapper, including root-owned0644 lock files.
    # Missing locks fail; this fixture never creates/chmods/replaces them.
    paths = [global_path] + [gpu_root / str(gpu) for gpu in gpu_ids]
    opened = []
    try:
        for path in paths:
            opened.append(read_lock(path))
        deadline = time.monotonic() + timeout
        while True:
            held = []
            try:
                for descriptor, _ in opened:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held.append(descriptor)
                # Keep the per-GPU locks, but release the global coordinator
                # so unrelated official tests can inspect their own groups.
                fcntl.flock(opened[0][0], fcntl.LOCK_UN)
                break
            except BlockingIOError:
                for descriptor in reversed(held):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Canonical GPU locks are busy")
                time.sleep(0.25)
        yield dict(holder_pid=os.getpid(), physical_gpu_ids=gpu_ids,
                   mechanism="O_RDONLY same-inode flock EX", global_coordinator_held_during_acquire_only=True,
                   records=[row for _, row in opened])
    finally:
        for descriptor, _ in reversed(opened):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def recipe(args):
    prefill = args.role == "prefill"
    graph_rows = max(24, args.local_concurrency * (args.gamma + 1)) if args.graph_max_rows is None else args.graph_max_rows
    require(args.gamma >= 0 and args.verify_budget >= 0 and (args.gamma == 0 or 1 <= args.verify_budget <= args.gamma), "Verify budget must be0 for a baseline or1..gamma for speculative decode")
    require(args.local_concurrency > 0 and graph_rows >= args.local_concurrency * (args.gamma + 1), "Graph rows must cover the actual target verification width")
    captures = sorted({1, args.local_concurrency, *(n for n in (2, 4, 16) if n <= args.local_concurrency)})
    options = dict(
        model_type="deepseek_v41", role_type=args.role.upper(),
        world_size=8, tp_size=8 if prefill else 1, dp_size=1 if prefill else 8,
        ep_size=8, act_type="BF16", fp8_kv_cache=1,
        seq_size_per_block=256, kernel_seq_size_per_block=128,
        max_seq_len=args.max_seq_len,
        max_context_batch_size=1 if prefill else args.local_concurrency,
        concurrency_limit=128 if prefill else args.local_concurrency,
        enable_cuda_graph=0 if prefill else 1,
        decode_capture_config=",".join(map(str, captures)),
        reuse_cache=1, load_method=args.load_method,
        reserver_runtime_mem_mb=8192, kv_cache_mem_mb=args.kv_cache_mem_mb,
        warm_up=1, warm_up_with_loss=0,
        dsv4_fixed_pool_blocks=256,
        cache_store_rdma_mode=0, use_local=1,
        load_cache_timeout_ms=120000,
        prefill_cp_kv_cache_sharded=1, prefill_cp_size=8,
        cp_rotate_method="ALL_GATHER" if prefill else "PREFILL_CP",
        use_deepep_moe=1, use_all_gather=0, use_deepep_low_latency=0,
        use_deepep_internode=0, use_batch_decode_scheduler=0,
        # P must declare the same bounded-replay cache tags as speculative D.
        # Its PREFILL role still controls execution; decoder verify knobs below
        # remain decode-only.
        sp_type="dspark" if args.gamma else "none",
        gen_num_per_cycle=max(1, args.gamma),
    )
    if args.gamma:
        options.update(sp_model_type="deepseek_v41_dspark", sp_checkpoint_path=args.checkpoint, sp_act_type="BF16")
    env = dict(
        WORLD_SIZE="8", LOAD_PYTHON_MODEL="1", LOG_LEVEL=args.log_level,
        FT_SERVER_TEST="1", DSV4_MOE_STRATEGY="mega_sm90_se", MOE_STRATEGY="auto",
        DSV41_SM90_MOE_COMM="v2", DSV4_USE_MEGA_MOE_SE="0",
        DSV41_ENGRAM_PREFETCH="1", DSV41_CPU_ENGRAM_LOOKUP="0",
        DSV41_ENGRAM_DEVICE_SCALES="0", DSV41_CED="1",
        DSV41_SWA_BOUNDED_REPLAY="1", DSV4_MOE_CHUNK_TOKENS="4096",
        DSV41_SM90_MULTIQUERY_INDEXER="1", RTP_LLM_DSPARK_FORCE_TOPK="0",
        RTP_LLM_DEVICE_INPUT="1", RTP_LLM_STREAM_ASYNC="1",
        RTP_LLM_CPU_PREFILL_ASYNC_SCHEDULING="0",
        RTP_LLM_SM90_NVIDIA_WHEEL_PATH="1",
        ROLE_TYPE=args.role.upper(), REMOTE_SERVER_PORT=str(args.peer_port),
        REMOTE_RPC_SERVER_IP=args.peer_ip,
    )
    if prefill:
        env["DSV4_PREFILL_CP_OVERLAP"] = "1"
        env["DSV41_SM90_DECODE_GRAPH"] = "0"
    else:
        # Decode already runs stream-async bookkeeping; skipping the front-loaded
        # broad sync removes the step-boundary bubble (D1), and a 1 ms idle poll
        # prevents DP8 low-BS fake-stream ranks from draining the GPU queue
        # before Draft/Target L0 MegaMoE barriers (D5).
        env.update(
            DSV41_SM90_DECODE_GRAPH="1",
            DSV41_SM90_DECODE_GRAPH_MAX_ROWS=str(graph_rows),
            RTP_LLM_DROP_BROAD_SYNC="1",
            RTP_LLM_SCHEDULER_POLL_INTERVAL_MS="1",
        )
        if args.gamma:
            env.update(SP_DSPARK_CONFIDENCE_MODE="bounded", SP_DSPARK_VERIFY_MODE="packed",
                       SP_DSPARK_VERIFY_BUDGET_MODE="fixed", SP_DSPARK_VERIFY_BUDGET=str(args.verify_budget))
    return options, env


def port_precheck(port):
    require(1024 <= port <= 65435, "Port must leave room for the eight-rank100-port block")
    with contextlib.ExitStack() as stack:
        for number in range(port, port + 100):
            sock = stack.enter_context(socket.socket())
            sock.bind(("0.0.0.0", number))


def get_status(port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as response:
        require(response.status == 200, "Health is not200")
    with opener.open(f"http://127.0.0.1:{port}/worker_status", timeout=5) as response:
        return json.loads(response.read())


def done_requested(path, run_id):
    if not path.exists():
        return False
    require(path.is_file() and not path.is_symlink() and path.stat().st_uid == os.getuid(), "Done file must be an owned regular file")
    value = json.loads(path.read_text())
    require(value == {"run_id": run_id, "action": "stop"}, "Done file has wrong run-id/action")
    return True


def process_identity(pid):
    if pid is None:
        return dict(pid=None, exists=False)
    base = Path(f"/proc/{pid}")
    try:
        fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, exists=True, uid=base.stat().st_uid,
                    state=fields[0], starttime=fields[19])
    except FileNotFoundError:
        return dict(pid=pid, exists=False)
    except OSError as error:
        return dict(pid=pid, exists=None, observation_error=str(error))


def wait_fixture_ready(manager, args, deadline, validator):
    """Consume owned startup logs in bounded chunks; retain only markers."""
    selected, pending, offset = [], "", 0
    pattern = re.compile(r"parallelism:|CP=|cp_enabled=|allocated symm buffer|fused shared expert enabled|SM90 Engram UVA prefetch enabled|Engram shared CPU lookup enabled|Backend server initialized successfully|capture success for batch size")
    last = "awaiting startup"
    while time.monotonic() < deadline:
        process = manager._server_process
        require(process is not None and process.poll() is None, "Owned server exited during startup")
        with Path(manager.log_file_path).open("rb") as reader:
            reader.seek(offset)
            while True:
                chunk = reader.read(4 * 1024 * 1024)
                if not chunk:
                    break
                offset = reader.tell()
                lines = (pending + chunk.decode(errors="replace")).split("\n")
                pending = lines.pop()
                selected.extend(line for line in lines if pattern.search(line))
        text = "\n".join(selected)
        try:
            status = get_status(args.port)
            workers = status["results"]
            expected = 1 if args.role == "prefill" else 8
            require(len(workers) == expected and {int(w["dp_rank"]) for w in workers} == set(range(expected)) and all(w["alive"] for w in workers), "Not all actual DP workers are alive")
            require(all(w["address"].rsplit(":", 1)[0] == args.host_ip for w in workers), "Worker advertises a different container IP")
            validator(text, args.role)
            require(text.count("Backend server initialized successfully, sending ready status") >= 8, "Awaiting all eight backend initializations")
            require("SM90 Engram UVA prefetch enabled" in text, "Awaiting Engram UVA/prefetch initialization")
            require("Engram shared CPU lookup enabled" not in text, "Unexpected CPU Engram path")
            if args.role == "decode":
                require("capture success for batch size" in text, "Awaiting target graph capture")
            return status, text
        except (AssertionError, ValueError, OSError, KeyError) as error:
            last = str(error)
        time.sleep(1)
    raise TimeoutError(f"Fixture readiness deadline exceeded: {last}")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=("prefill", "decode"))
    parser.add_argument("--host-ip", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--peer-ip", required=True)
    parser.add_argument("--peer-port", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--done-file", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--tmp-dir")
    parser.add_argument("--gamma", type=int, default=0)
    parser.add_argument("--verify-budget", type=int)
    parser.add_argument("--local-concurrency", type=int, default=4)
    parser.add_argument("--graph-max-rows", type=int)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--kv-cache-mem-mb", type=int, default=1024)
    parser.add_argument("--load-method", choices=("scratch", "fastsafetensors"), default="scratch")
    parser.add_argument("--log-level", choices=("INFO", "DEBUG"), default="INFO")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--budget-seconds", type=int, default=3600)
    parser.add_argument("--lock-timeout", type=int, default=30)
    args = parser.parse_args()
    if args.verify_budget is None:
        args.verify_budget = 0 if args.gamma == 0 else 3
    return args


def main():
    args = arguments()
    require(os.getuid() == os.geteuid() and os.getuid() > 0, "Run inside the existing container as a non-root user")
    for value in (args.host_ip, args.peer_ip):
        require(ipaddress.ip_address(value).version == 4, "Use the verified container IPv4 addresses")
    removed_proxy_names = prepare_runtime_network_env(os.environ, args.host_ip, args.peer_ip)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,80}", args.run_id), "Invalid run-id")
    require(args.kv_cache_mem_mb > 0 and args.max_seq_len > 0 and args.budget_seconds > args.startup_timeout > 0, "Invalid memory/sequence/lifetime budget")
    output = owned_directory(args.output_dir)
    done = Path(args.done_file)
    require(done.is_absolute() and done.parent == output and not done.exists(), "Use an absent done file directly inside this run's output directory")
    require(not (output / "fixture.json").exists(), "Do not overwrite another fixture receipt")
    cache = owned_directory(args.cache_dir) if args.cache_dir else owned_directory(str(output / "cache"))
    temporary = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix=f"sm90-pd-{os.getuid()}-", dir="/tmp"))
    require(temporary.is_absolute() and temporary.resolve() == temporary and temporary.is_dir()
            and temporary.stat().st_uid == os.getuid() and len(os.fsencode(temporary)) <= 64,
            "Use an owned short TMPDIR for CPU-TP Unix sockets")
    config_bytes = (Path(args.checkpoint) / "config.json").read_bytes()
    raw_config = json.loads(config_bytes)
    config = raw_config.get("text_config", raw_config)
    require(config["num_hidden_layers"] == 40, "The fixture requires the original full40 checkpoint")
    source_bytes = Path(args.source_manifest).read_bytes()
    source = json.loads(source_bytes)
    for name in ("public_sha", "internal_sha"):
        require(re.fullmatch(r"[0-9a-f]{40}", source[name]), "Source manifest must carry full public/internal SHA")
    os.environ.update(RTP_LLM_SM90_NVIDIA_WHEEL_PATH="1", TEST_UNDECLARED_OUTPUTS_DIR=str(output), MAGA_SERVER_WORK_DIR=str(output), TMPDIR=str(temporary))
    # Prevent a controller's previous D environment from enabling packed
    # verification on P (or from leaking into a gamma0 ordinary baseline).
    for name in tuple(os.environ):
        if name.startswith("SP_DSPARK_"):
            os.environ.pop(name)
    for name, subdir in (("FLASHINFER_WORKSPACE_BASE", "flashinfer"), ("TRITON_CACHE_DIR", "triton"), ("DG_JIT_CACHE_DIR", "deep_gemm"), ("PYTORCH_KERNEL_CACHE_PATH", "torch")):
        path = cache / subdir
        path.mkdir(exist_ok=True)
        os.environ[name] = str(path)
    # The controller sets these loader paths BEFORE starting this binary.
    # Python's helper then preserves the same selected closure for children.
    # Do not call setup_jit_cache: its wrapper rewrite interprets argv[1] as a
    # test binary, whereas this fixture is a directly invoked py_binary.
    inherited_libraries = {os.path.realpath(p) for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p}
    runfiles = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
    require(runfiles, "Run the built Bazel py_binary with its runfiles")
    libraries = list(Path(runfiles).glob("pip_gpu_cuda13_sm90_torch_torch/site-packages/torch/lib"))
    libraries += list(Path(runfiles).glob("pip_gpu_cuda13_sm90_torch_nvidia_*/site-packages/nvidia/*/lib"))
    require(libraries and all(str(p.resolve()) in inherited_libraries for p in libraries),
            "Set selected runfiles Torch/NVIDIA LD_LIBRARY_PATH before launching the fixture")
    from rtp_llm.test.utils.jit_sys_path_setup import setup_nvidia_wheel_library_path
    setup_nvidia_wheel_library_path()
    # Preserve the actual Bazel ordered dependency roots for the owned server.
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(str(Path(p).absolute()) for p in sys.path if p))
    from rtp_llm.test.perf_test.deepseek_v41_sm90_role_test import configure_jit_cache, validate_startup
    jit = configure_jit_cache()
    deep_gemm_version = importlib.metadata.version("deep-gemm")
    require(deep_gemm_version.startswith("2.8.2+8692811"), "Select the separate SM90 DeepGEMM wheel")
    require(importlib.metadata.version("nvidia-cutlass-dsl") == "4.8.1", "Select the separate SM90 DSL wheel")
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader,nounits"], text=True).splitlines()
    devices = {int(row.split(",")[0]): row.split(",")[1].strip() for row in gpu_rows}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = [int(item) for item in visible.split(",")] if visible else sorted(devices)
    require(len(gpu_ids) == len(set(gpu_ids)) == 8 and all(devices.get(gpu) == "9.0" for gpu in gpu_ids), "Require exactly eight actual SM90 GPUs")
    options, env = recipe(args)
    p_ip, p_port, d_ip, d_port = ((args.host_ip, args.port, args.peer_ip, args.peer_port) if args.role == "prefill" else (args.peer_ip, args.peer_port, args.host_ip, args.port))
    env["MODEL_SERVICE_CONFIG"] = json.dumps(dict(service_id=args.run_id, use_local=True, role_endpoints=[dict(group="default", prefill_endpoint=dict(type="Vipserver", address=f"{p_ip}:{p_port}", protocol="http", path="/"), decode_endpoint=dict(type="Vipserver", address=f"{d_ip}:{d_port}", protocol="http", path="/"))]))
    report = dict(status="STARTING", run_id=args.run_id, role=args.role, source={key: source[key] for key in ("public_sha", "internal_sha")}, source_manifest_sha256=hashlib.sha256(source_bytes).hexdigest(), checkpoint=dict(path=args.checkpoint, layers=40, config_sha256=hashlib.sha256(config_bytes).hexdigest(), vocab_size=config["vocab_size"]), options=options, env=env, jit=jit, deep_gemm_version=deep_gemm_version, tmp_dir=str(temporary), ordered_python_path=os.environ["PYTHONPATH"].split(os.pathsep), selected_library_dirs=[str(p) for p in libraries], transport="TCP cache_store_rdma_mode=0; not RDMA", expected_cache_contract=dict(source="SM90 recipe, not a runtime pool observation", architecture="SM90", main_entry_bytes=584, seq_size_per_block=256, kernel_seq_size_per_block=128), readiness_is_accuracy_result=False)
    report["runtime_network_policy"] = dict(removed_proxy_variable_names=removed_proxy_names, local_peer_bypass=[args.host_ip, args.peer_ip])
    destination = output / "fixture.json"
    def save():
        destination.write_text(json.dumps(report, indent=2))
    save()
    started = time.monotonic()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Fixture interrupted by signal{signum}")
    signal.signal(signal.SIGTERM, interrupted)
    with canonical_lease(gpu_ids, args.lock_timeout) as lease:
        report["gpu_lease"] = lease
        gpu_selector = "--id=" + ",".join(map(str, gpu_ids))
        active = subprocess.check_output(["nvidia-smi", gpu_selector, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        require(not active, "Existing GPU compute processes: do not clean or stop them")
        port_precheck(args.port)
        from rtp_llm.test.utils.maga_server_manager import MagaServerManager
        manager = MagaServerManager(env_args=env, device_ids=gpu_ids, port=str(args.port), role_name=args.role, smoke_args_str=shlex.join([word for key, value in options.items() for word in ("--" + key, str(value))]))
        try:
            require(manager.start_server(model_path=args.checkpoint, tokenizer_path=args.checkpoint, model_type="deepseek_v41", timeout=args.startup_timeout), "Owned role failed startup; preserve its logs")
            report.update(server_pid=manager.server_pid, log_path=manager.log_file_path)
            old = os.environ.get("SM90_PERF_MOE_STRATEGY")
            os.environ["SM90_PERF_MOE_STRATEGY"] = "mega_sm90_se"
            try:
                status, startup = wait_fixture_ready(manager, args, started + args.startup_timeout, validate_startup)
            finally:
                if old is None:
                    os.environ.pop("SM90_PERF_MOE_STRATEGY", None)
                else:
                    os.environ["SM90_PERF_MOE_STRATEGY"] = old
            (output / "startup-markers.log").write_text(startup)
            report.update(status="FIXTURE_READY", worker_status=status, actual_argv=Path(f"/proc/{manager.server_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(), health="ok")
            report["server_identity"] = dict(
                pid=manager.server_pid, uid=Path(f"/proc/{manager.server_pid}").stat().st_uid,
                starttime=Path(f"/proc/{manager.server_pid}/stat").read_text().rsplit(")", 1)[1].split()[19],
                cwd=os.readlink(f"/proc/{manager.server_pid}/cwd"),
            )
            save()
            while not done_requested(done, args.run_id):
                if time.monotonic() - started >= args.budget_seconds:
                    raise TimeoutError("Fixture budget expired without a matching controller done file")
                process = manager._server_process
                require(process is not None and process.poll() is None, "Owned server exited while waiting for controller")
                time.sleep(1)
            report["status"] = "CONTROLLER_DONE"
        except BaseException as error:
            report.update(status="FIXTURE_FAILED", error=str(error))
            raise
        finally:
            # Never let a recycled, already reaped PID become a cleanup target.
            process = manager._server_process
            before_cleanup = process_identity(None if process is None else process.pid)
            if process is not None and process.poll() is not None:
                with manager._state_lock:
                    manager._server_process = None
            manager.stop_server()
            report["owned_cleanup_invoked"] = True
            after_cleanup = process_identity(None if process is None else process.pid)
            report["cleanup_server_identity"] = dict(before=before_cleanup, after=after_cleanup)
            report["owned_server_stopped"] = (
                after_cleanup["exists"] is False or after_cleanup.get("state") in ("Z", "X")
                or (before_cleanup.get("starttime") is not None and after_cleanup.get("starttime") is not None
                    and before_cleanup["starttime"] != after_cleanup["starttime"])
            )
            report["subprocess_poll_after_cleanup"] = None if process is None else process.poll()
            observations = {}
            for label, fields in (("compute_pids", "pid"), ("memory", "index,memory.used")):
                try:
                    result = subprocess.run(
                        ["nvidia-smi", gpu_selector, "--query-compute-apps=" + fields if label == "compute_pids" else "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                        text=True, capture_output=True, timeout=15,
                    )
                    observations[label] = dict(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
                except (OSError, subprocess.TimeoutExpired) as error:
                    observations[label] = dict(observation_error=str(error))
            report["post_cleanup_gpu_observation"] = observations
            save()


if __name__ == "__main__":
    main()
