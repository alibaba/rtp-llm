#!/usr/bin/env python3
"""Start independently built PD peers over SSH, without changing smoke cases.

The controller starts both roles before checking readiness. Optional request
argv run in Prefill's server runfiles; all case definitions stay in their
existing runners. READY alone is never reported as a smoke PASS.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

def decode_dp_size(config):
    return config.get("decode_dp", 2)

ROLES = ("prefill", "decode")
SCRIPT = "rtp_llm/test/smoke/kimi_k3/two_host_launcher.py"


def validate(config):
    if type(decode_dp_size(config)) is not int or decode_dp_size(config) not in (1, 2):
        raise ValueError("The smoke supports Decode DP1 or DP2")
    for key in ("fp8_gemm", "fp8_kv_cache"):
        if key in config and type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", config["run_id"]):
        raise ValueError("run_id must contain only letters, digits, _ and -")
    user = config.get("container_user", "luohaocheng.lhc")
    if not user or user.split(":")[0] in ("root", "0") or user.startswith("-"):
        raise ValueError("A non-root container user is required")
    for role in ROLES:
        host = config[role]
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host["ssh_target"]) or host["ssh_target"].startswith("-"):
            raise ValueError("Invalid SSH target")
        if host.get("runtime", "docker") not in ("docker", "pouch"):
            raise ValueError("Unsupported container runtime")
        if host.get("runtime_group") not in (None, "pouch") or (
                host.get("runtime_group") and host.get("runtime", "docker") != "pouch"):
            raise ValueError("runtime_group supports only existing Pouch membership")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", host["container"]) or host["container"].startswith("-"):
            raise ValueError("Invalid container")
        ipaddress.IPv4Address(host["ip"])
        for key in ("repo_root", "artifact_root", "checkpoint", "draft_checkpoint"):
            if not Path(host[key]).is_absolute():
                raise ValueError(f"{role}.{key} must be absolute")
        if type(host["port"]) is not int or not 1024 <= host["port"] <= 65535 - 72:
            raise ValueError("Invalid eight-rank port block")
    if config["prefill"]["ip"] == config["decode"]["ip"]:
        raise ValueError("PD peers must use distinct machines")
    if config["prefill"]["cluster"] != config["decode"]["cluster"]:
        raise ValueError("PD peers must belong to the same data-plane cluster")
    requests = config.get("request_argv")
    if requests is not None and (not isinstance(requests, list) or not requests
                                or any(not isinstance(x, str) for x in requests)):
        raise ValueError("request_argv must be a nonempty argv list")
    return config


def run_dir(config, role):
    return Path(config[role]["artifact_root"]) / config["run_id"] / role


def ssh(config, role, argv, **kwargs):
    options = ["-F", config["ssh_config"]] if config.get("ssh_config") else []
    return subprocess.run(["ssh", *options, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                           config[role]["ssh_target"], shlex.join(argv)],
                          text=True, capture_output=True, check=True, timeout=kwargs.pop("timeout", 60), **kwargs)


def runtime_command(host, arguments):
    command = [host.get("runtime", "docker"), *arguments]
    # An existing group membership can outlive a cached SSH session's groups.
    # sg grants no root privilege; use only the explicitly selected Pouch group.
    if host.get("runtime_group") == "pouch":
        return ["sg", "pouch", "-c", shlex.join(command)]
    return command


def remote_command(config, role, operation, detached=False):
    host = config[role]
    return runtime_command(host, ["exec", "-d" if detached else "-i", "-u",
            config.get("container_user", "luohaocheng.lhc"), host["container"],
            "python3", str(Path(host["repo_root"]) / SCRIPT), operation,
            "--role", role, "--config", str(run_dir(config, role) / "control.json")])


def remote(config, role, operation):
    return json.loads(ssh(config, role, remote_command(config, role, operation),
                           timeout=(config.get("request_timeout_s", 7200) if operation=="requests"
                                    else 90 if operation=="stop-role" else 60)).stdout)


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def start_ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def owned_data(path):
    path = path.resolve(strict=True)
    if not re.match(r"^/(?:data[0-9]*|ssd)/", str(path)) or path.stat().st_uid != os.getuid():
        raise ValueError("Use an owned directory on the personal data disk")
    filesystem = subprocess.check_output(["findmnt", "-T", str(path), "-n", "-o", "FSTYPE"], text=True)
    if set(filesystem.split()) - {"ext4", "xfs", "btrfs"}:
        raise ValueError("Artifacts and source must use a local data filesystem")
    return path


def runtime_environment(source, binary):
    """Use the locally built CUDA13 closure, never inherited pip libraries."""
    runfiles = Path(str(binary) + ".runfiles")
    torch = runfiles / "pip_gpu_cuda13_torch_torch/site-packages/torch/lib"
    if not torch.is_dir():
        raise ValueError("Missing independently built CUDA13 server runfiles")
    libraries = sorted(runfiles.glob("pip_gpu_cuda13_torch_nvidia_*/site-packages/nvidia/*/lib"))
    # setproctitle 1.3.6 walks backwards from the first environment string.
    # glibc may relocate GLIBC_TUNABLES to the heap; keep a stable first entry.
    env = {"K31_PD_SMOKE_ENV_ANCHOR": "1", **os.environ}
    for key in list(env):
        if key.startswith(("RTP_", "KIMI_", "SP_")) or key in (
                "CP_ROTATE_METHOD", "QUANTIZATION", "FP8_MLA", "FP8_KV_CACHE",
                "BATCH_PERF_TEST", "ENABLE_BATCH_PERF_TEST", "LD_PRELOAD", "PYTHONPATH"):
            env.pop(key)
    env["LD_LIBRARY_PATH"] = ":".join(map(str, libraries + [torch])) + ":" + ":".join((
        "/opt/rh/gcc-toolset-12/root/usr/lib64", "/usr/local/cuda/lib64",
        "/opt/conda310/lib", "/usr/local/nvidia/lib64", "/usr/lib64"))
    env["SPT_NOENV"] = "1"  # Preserve the process ownership tag in /proc/PID/environ.
    return env


def prepare(config, role):
    if os.getuid() == 0:
        raise ValueError("Run inside the selected container as its personal user")
    root = owned_data(Path(config[role]["artifact_root"]))
    directory = root / config["run_id"] / role
    directory.mkdir(parents=True, exist_ok=False)
    atomic(directory / "control.json", config)
    return {"prepared": True, "uid": os.getuid(), "hostname": socket.gethostname()}


def usable_rdma_hcas(names):
    """Require a GID and, for Ethernet HCAs, a corresponding netdev."""
    usable = []
    for name in names:
        ports = Path("/sys/class/infiniband") / name / "ports"
        for gid in ports.glob("*/gids/*"):
            try:
                address = ipaddress.IPv6Address(gid.read_text().strip())
                if address.is_unspecified or address == ipaddress.IPv6Address("fe80::"):
                    continue
                port = gid.parent.parent
                if (port / "link_layer").read_text().strip() == "InfiniBand":
                    usable.append(name); break
                netdev = (port / "gid_attrs/ndevs" / gid.name).read_text().strip()
                if netdev and (Path("/sys/class/net") / netdev).exists():
                    usable.append(name); break
            except (OSError, ValueError):
                continue
    return usable


def serve(config, role):
    source = owned_data(Path(config[role]["repo_root"]))
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / Path(SCRIPT).parent))
    from checkpoint_guard import preflight
    from server_launcher import launch_config, require_gpu_capacity, validate_rdma_hcas
    from k31_chunk_fixture import fixture_environment
    host, peer = config[role], config["decode" if role == "prefill" else "prefill"]
    directory = owned_data(run_dir(config, role))
    if (directory / "instance.json").exists():
        raise ValueError("Role already launched; use a new run_id")
    actual_hashes = {}
    for name, expected in config.get("source_files", {}).items():
        path = (source / name).resolve(strict=True)
        if not path.is_relative_to(source):
            raise ValueError("Source manifest escapes repo")
        actual_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hashes[name] != expected:
            raise ValueError(f"Source changed: {name}")
    locks = []
    lock_root = Path("/tmp/rtp_llm/smoke/test/gpu_status")
    lock_root.mkdir(parents=True, exist_ok=True)
    for index in range(8):
        lock = (lock_root / str(index)).open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    require_gpu_capacity(directory, host.get("allow_shared_accuracy", False), host.get("min_free_gib", 250), 8)
    for name in ("checkpoint", "draft_checkpoint"):
        storage = host.get(name + "_storage_policy", host.get("storage_policy", "3fs"))
        if storage not in ("3fs", "ssd", "published"):
            raise ValueError("storage_policy must be 3fs, ssd or published")
        storage_root = host.get(name + "_root", host["checkpoint_root"])
        if storage == "published":
            # Reuse an explicitly selected published mount without staging it.
            # Pin both its exact root and filesystem instead of inferring policy
            # from a path name or permitting an arbitrary parent directory.
            checkpoint = Path(host[name]).resolve(strict=True)
            if checkpoint != Path(storage_root).resolve(strict=True):
                raise ValueError("Published checkpoint root must equal the selected checkpoint")
            expected_fs = host.get(name + "_fstype")
            actual_fs = subprocess.check_output(
                ["findmnt", "-T", str(checkpoint), "-n", "-o", "FSTYPE"], text=True).split()
            if not expected_fs or set(actual_fs) != {expected_fs}:
                raise ValueError(f"Published checkpoint filesystem mismatch: {name}")
            policy = {"allow_checkpoint_root": storage_root}
        else:
            policy = {"allow_hf3fs_root" if storage == "3fs" else "local_data_root": storage_root}
        report = preflight(host[name], **policy,
                           load_method="fastsafetensors", check_layout=True)
        atomic(directory / (name + "-preflight.json"), report)
        if not report["passed"]:
            raise ValueError(f"{name} preflight failed")
    sockets = []
    try:
        for port in range(host["port"], host["port"] + 72):
            sock = socket.socket(); sockets.append(sock); sock.bind(("0.0.0.0", port))
    finally:
        for sock in sockets: sock.close()
    links = subprocess.check_output(["rdma", "link", "show"], text=True)
    devices = subprocess.check_output(["ibv_devices"], text=True)
    hcas = host.get("rdma_hcas")
    candidates = list(dict.fromkeys(word.split("/")[0] for line in links.splitlines()
                      if "ACTIVE" in line and "LINK_UP" in line for word in line.split()
                      if word.startswith("mlx5_") and "/" in word))
    usable = usable_rdma_hcas(candidates)
    if hcas and set(hcas.split(",")) - set(usable):
        raise ValueError("Selected HCA lacks a usable GID/netdev in this container")
    hcas = hcas or ",".join(usable)
    layers = {(port / "link_layer").read_text().strip()
              for name in hcas.split(",") if name
              for port in (Path("/sys/class/infiniband") / name / "ports").glob("*")}
    if len(layers) != 1:
        raise ValueError("Choose rdma_hcas from one link layer; do not mix InfiniBand and Ethernet")
    hcas = validate_rdma_hcas(hcas, devices, links)
    atomic(directory / "rdma-preflight.json", dict(active_hcas=candidates,
        usable_hcas=usable, selected_hcas=hcas.split(","), link_layer=next(iter(layers))))
    binary = (source / host.get("server", "bazel-bin/rtp_llm/rtp_llm_server")).resolve(strict=True)
    dp = 1 if role == "prefill" else decode_dp_size(config)
    args = argparse.Namespace(role=role.upper(), server=str(binary), checkpoint=host["checkpoint"],
        draft_checkpoint=host["draft_checkpoint"], model_type="kimi_k31", expected_target_layers=93,
        tp_size=8 // dp, dp_size=dp, ep_size=8,
        fp8_gemm=config.get("fp8_gemm"), fp8_kv_cache=config.get("fp8_kv_cache"),
        prefill_source_tp_size=8, peer_ip=peer["ip"], peer_port=peer["port"], start_port=host["port"],
        orthogonal_smoke=True, warm_up=False, model_warm_up=False,
        execution_evidence=True, reserve_runtime_mem_mb=host.get("reserve_runtime_mem_mb",14336),
        kv_cache_mem_mb=host.get("kv_cache_mem_mb"), memory_cache_size_mb=host.get("memory_cache_size_mb",32768),
        test_block_num=host.get("test_block_num"),
        prefill_token_capacity=host.get("prefill_token_capacity"),
        max_seq_len=host.get("max_seq_len"), request_capacity=host.get("request_capacity"),
        decode_capture_batches=host.get("decode_capture_batches"))
    explicit, command = launch_config(args)
    service = json.loads(explicit["MODEL_SERVICE_CONFIG"])
    service["service_id"] = config["run_id"]
    service["role_endpoints"][0]["group"] = config["run_id"]
    explicit["MODEL_SERVICE_CONFIG"] = json.dumps(service)
    uds = tempfile.TemporaryDirectory(prefix=f"k3pd-{os.getuid()}-", dir="/tmp")
    temporary = tempfile.TemporaryDirectory(prefix=f"k3tmp-{os.getuid()}-", dir="/tmp")
    cache = Path(host["artifact_root"]) / "jit"
    cache.mkdir(exist_ok=True)
    explicit.update(RTP_LLM_CPU_TP_BROADCASTER_DIR=uds.name,
        RTP_LLM_CPU_TP_BROADCASTER_ID=config["run_id"], TMPDIR=temporary.name,
        TMP=temporary.name, TEMP=temporary.name, ACCL_USE_NICS=hcas,
        K31_PD_SMOKE_RUN_ID=config["run_id"], HF_HUB_OFFLINE="1")
    for key, leaf in {"TRITON_CACHE_DIR":"triton", "DG_JIT_CACHE_DIR":"deep-gemm",
                      "FLASHINFER_WORKSPACE_BASE":"flashinfer", "HF_MODULES_CACHE":"hf-modules",
                      "TORCH_EXTENSIONS_DIR":"torch-extensions"}.items():
        (cache / leaf).mkdir(exist_ok=True); explicit[key] = str(cache / leaf)
    (directory / "logs").mkdir()
    explicit["LOG_PATH"] = str(directory / "logs")
    env = runtime_environment(source, binary); env.update(explicit)
    if explicit["QUANTIZATION"] == "FP8_PER_BLOCK" and role == "prefill":
        fixture = fixture_environment(directory, env)
        env.update(fixture); explicit.update(fixture)
    stop = False
    def on_signal(_signum, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGINT, on_signal)
    with (directory / "server.log").open("x") as log:
        process = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        identity = dict(role=role, uid=os.getuid(), hostname=socket.gethostname(),
            manager_pid=os.getpid(), manager_start_ticks=start_ticks(os.getpid()),
            pid_namespace=os.readlink("/proc/self/ns/pid"), server_pid=process.pid,
            server_start_ticks=start_ticks(process.pid), pgid=process.pid,
            command=command, environment={**explicit, "SPT_NOENV": env["SPT_NOENV"],
                "K31_PD_SMOKE_ENV_ANCHOR": env["K31_PD_SMOKE_ENV_ANCHOR"]}, run_id=config["run_id"],
            source_files=actual_hashes, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            performance_validated=False, smoke_passed=None)
        atomic(directory / "instance.json", identity)
        try:
            while process.poll() is None and not stop: time.sleep(1)
        finally:
            stop_owned_group(identity)
            process.wait(timeout=30)
            atomic(directory / "exit.json", dict(exit_code=process.returncode, stop_requested=stop))
    uds.cleanup(); temporary.cleanup()
    return {"exit_code": process.returncode, "stop_requested": stop}


def stop_owned_group(identity):
    """Stop the recorded session after validating UID, namespace and start ticks."""
    if identity["uid"] != os.getuid() or os.readlink("/proc/self/ns/pid") != identity["pid_namespace"]:
        raise ValueError("Service receipt belongs to another UID or PID namespace")
    root_verified = False
    root = Path(f"/proc/{identity['server_pid']}")
    try:
        if (root.stat().st_uid != os.getuid()
                or start_ticks(identity["server_pid"]) != identity["server_start_ticks"]
                or os.getsid(identity["server_pid"]) != identity["pgid"]):
            raise ValueError("Refusing to signal a replaced service leader")
        root_verified = True
    except (FileNotFoundError, ProcessLookupError):
        pass
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit(): continue
        try:
            if os.getpgid(int(entry.name)) != identity["pgid"]: continue
            if entry.stat().st_uid != os.getuid(): raise ValueError("Foreign process in service group")
            if os.getsid(int(entry.name)) != identity["pgid"]:
                raise ValueError("Process no longer belongs to the service session")
            # start_new_session plus the verified leader's start ticks establish
            # ownership even when a child makes /proc/PID/environ unreadable.
            if not root_verified and ("K31_PD_SMOKE_RUN_ID=" + identity["run_id"]).encode() not in (entry/"environ").read_bytes().split(b"\0"):
                raise ValueError("Process group no longer belongs to this run")
            members.append((int(entry.name), start_ticks(int(entry.name))))
        except (FileNotFoundError, ProcessLookupError): pass
    if members:
        os.killpg(identity["pgid"], signal.SIGTERM)
        deadline = time.monotonic() + 55
        while time.monotonic() < deadline:
            alive = False
            for pid, ticks in members:
                try:
                    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                    alive |= fields[0] != "Z" and int(fields[19]) == ticks
                except (FileNotFoundError, ProcessLookupError): pass
            if not alive: return
            time.sleep(.2)
        # Validate ownership again before escalation, including PID reuse.
        for pid, ticks in members:
            path = Path(f"/proc/{pid}")
            try:
                if os.getpgid(pid) != identity["pgid"]: continue
                if (path.stat().st_uid != os.getuid() or start_ticks(pid) != ticks
                        or os.getsid(pid) != identity["pgid"]):
                    raise ValueError("Refusing to signal a replaced PID")
                os.kill(pid, signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError): pass


def status(config, role):
    sys.path.insert(0, str(Path(config[role]["repo_root"])))
    sys.path.insert(0, str(Path(config[role]["repo_root"]) / Path(SCRIPT).parent))
    from worker import pd_endpoints, listening_tcp_ports
    directory = run_dir(config, role)
    if (directory/"exit.json").exists(): return {"healthy":False, "exited":json.loads((directory/"exit.json").read_text())}
    if not (directory/"instance.json").exists(): return {"healthy":False, "starting":True}
    identity=json.loads((directory/"instance.json").read_text())
    try:
        if start_ticks(identity["manager_pid"]) != identity["manager_start_ticks"]:
            return {"healthy":False,"error":"manager PID changed"}
    except FileNotFoundError: return {"healthy":False,"error":"manager exited"}
    endpoints=pd_endpoints(config[role]["port"],1 if role=="prefill" else decode_dp_size(config))
    listeners=listening_tcp_ports(config[role]["ip"])
    missing=[x for x in endpoints if x["port"] not in listeners]
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));unhealthy=[]
    for item in endpoints:
        if item["kind"]!="http" or item["port"] not in listeners: continue
        try:
            with opener.open(f"http://127.0.0.1:{item['port']}/health",timeout=3) as response:
                if response.status!=200: unhealthy.append(item)
        except OSError: unhealthy.append(item)
    result=dict(healthy=not missing and not unhealthy,missing_listeners=missing,unhealthy_http=unhealthy,
                rdma_transfer_verified=False,smoke_passed=None)
    atomic(directory/"readiness.json",result);return result


def stop_role(config, role):
    path=run_dir(config,role)/"instance.json"
    if not path.exists(): return {"stopped":False,"reason":"no process receipt"}
    identity=json.loads(path.read_text());pid=identity["manager_pid"]
    try:
        proc=Path(f"/proc/{pid}")
        if proc.stat().st_uid!=os.getuid() or start_ticks(pid)!=identity["manager_start_ticks"]:
            raise ValueError("Refusing to stop a replaced/foreign manager")
        cmd=(proc/"cmdline").read_bytes().split(b"\0")
        if str(run_dir(config,role)/"control.json").encode() not in cmd:
            raise ValueError("Manager command does not match this run")
        os.kill(pid,signal.SIGTERM)
    except (FileNotFoundError,ProcessLookupError):
        # A manager's exit does not prove that its server session exited.
        stop_owned_group(identity)
        return {"stopped":True,"manager_already_exited":True}
    return {"stop_requested":True,"pid":pid}


def controller(args, config):
    if args.dry_run:
        for role in ROLES: print(role,shlex.join(remote_command(config,role,"serve")))
        return 0
    if args.operation=="stop":
        for role in ROLES: print(role,remote(config,role,"stop-role"))
        return 0
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    atomic(root/"config.json",config);started=[]
    try:
        images=[]
        for role in ROLES:
            host=config[role]
            images.append(ssh(config,role,runtime_command(host,["inspect","--format","{{.Config.Image}}",host["container"]])).stdout.strip())
        if images[0]!=images[1]: raise ValueError("PD containers use different images")
        for role in ROLES:
            command=remote_command(config,role,"prepare")
            ssh(config,role,command,input=json.dumps(config)+"\n")
        # Both peers must exist before waiting for either cache-store endpoint.
        for role in ("decode","prefill"):
            command=remote_command(config,role,"serve",detached=True)
            ssh(config,role,command);started.append(role)
        deadline=time.monotonic()+args.startup_timeout
        while True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                states=dict(zip(ROLES,pool.map(lambda role:remote(config,role,"status"),ROLES)))
            atomic(root/"readiness.json",states)
            if any("error" in s or "exited" in s for s in states.values()):
                raise RuntimeError(f"Role failed: {states}")
            if all(s["healthy"] for s in states.values()): break
            if time.monotonic()>=deadline: raise TimeoutError("PD startup exceeded deadline")
            time.sleep(5)
        for role in ROLES: print(f"{role}: READY",flush=True)
        # The launcher never invents or expands a case list.
        passed=None
        if config.get("request_argv"):
            result=remote(config,"prefill","requests");atomic(root/"requests.json",result)
            passed=result["exit_code"]==0
            if not passed: raise RuntimeError("Existing request runner failed; inspect requests.log")
        atomic(root/"summary.json",dict(ready=True,request_runner_passed=passed,
               complete_smoke_passed=None,performance_validated=False,services_retained=args.keep_services))
        print("READY; no requests run" if passed is None else "Request runner passed; artifacts: "+str(root))
        return 0
    finally:
        if not args.keep_services:
            for role in started: remote(config,role,"stop-role")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("launch","stop","prepare","serve","status","stop-role","requests"))
    parser.add_argument("--config",required=True,type=Path)
    parser.add_argument("--role",choices=ROLES)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--startup-timeout",type=int,default=14400)
    parser.add_argument("--keep-services",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    config=validate(json.loads(sys.stdin.readline() if args.operation=="prepare" else args.config.read_text()))
    if args.operation in ("launch","stop"):
        if args.operation=="launch" and not args.output and not args.dry_run: parser.error("launch requires --output")
        return controller(args,config)
    if not args.role or os.getuid()==0: parser.error("Role operation requires --role and a non-root user")
    if args.operation=="requests":
        source=Path(config[args.role]["repo_root"]);binary=(source/"bazel-bin/rtp_llm/rtp_llm_server").resolve(strict=True)
        env=runtime_environment(source,binary);env["CUDA_VISIBLE_DEVICES"]=""
        runfiles=Path(str(binary)+".runfiles")
        torch_site=runfiles/"pip_gpu_cuda13_torch_torch/site-packages"
        sites=[torch_site]+[p for p in sorted(runfiles.glob("pip*/site-packages")) if p!=torch_site and "_torch_torch" not in str(p)]
        # The runfiles package contains generated protobufs absent from source.
        env["PYTHONPATH"]=":".join(map(str,[runfiles/"rtp_llm",source,run_dir(config,args.role),*sites]))
        env["HF_HUB_OFFLINE"]="1"
        env["HF_MODULES_CACHE"]=str(Path(config[args.role]["artifact_root"])/"jit/hf-modules")
        env["NO_PROXY"]=env["no_proxy"]="localhost,127.0.0.1,"+",".join(config[r]["ip"] for r in ROLES)
        with (run_dir(config,args.role)/"requests.log").open("x") as log:
            code=subprocess.run(config["request_argv"],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        result={"exit_code":code}
    elif args.operation=="serve":
        with (run_dir(config,args.role)/"launcher.log").open("x") as log:
            sys.stdout=log;sys.stderr=log
            try: result=serve(config,args.role)
            except BaseException as exc:
                atomic(run_dir(config,args.role)/"exit.json",dict(exit_code=1,error=f"{type(exc).__name__}: {exc}"))
                raise
            finally: sys.stdout=sys.__stdout__;sys.stderr=sys.__stderr__
    else: result={"prepare":prepare,"status":status,"stop-role":stop_role}[args.operation](config,args.role)
    print(json.dumps(result))
    return (0 if result["stop_requested"] else result["exit_code"]) if args.operation=="serve" else 0


if __name__=="__main__":
    raise SystemExit(main())
