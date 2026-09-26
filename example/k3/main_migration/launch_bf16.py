"""Launch one BF16-compute PD endpoint; FP8 operands are explicit options.

Run as luohaocheng.lhc in a verified, same-image RDMA runtime container after
a fresh same-cluster fleet selection. Compile independently inside lhc_GPU.
This launcher does not select hosts or certify runtime precision/Graph/RDMA.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
import sys


def launch_config(args):
    fp8_gemm = bool(getattr(args, "fp8_gemm", False))
    fp8_kv_cache = bool(getattr(args, "fp8_kv_cache", False))
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    draft = Path(args.draft_checkpoint).resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    text = config.get("text_config", config)
    debug_four_layer = getattr(args, "debug_four_layer", False)
    expected_layers = 4 if debug_four_layer else 93
    if text.get("num_hidden_layers") != expected_layers:
        raise ValueError(f"Selected BF16 profile requires {expected_layers} target layers")
    if debug_four_layer:
        linear = text.get("linear_attn_config", {})
        if linear.get("kda_layers") != [1, 2, 3] or linear.get("full_attn_layers") != [4]:
            raise ValueError("Four-layer debug profile requires the original first three KDA layers and fourth MLA layer")
        if text.get("attn_res_block_size") != 12:
            raise ValueError("Four-layer debug profile must preserve the original AttnRes block size")
    for port in (args.start_port, args.peer_port):
        if port < 1024 or port + 8 * 9 > 65535:
            raise ValueError("Invalid eight-rank service port range")
    reserve_runtime_mem_mb = getattr(args, "reserve_runtime_mem_mb", 14336)
    if reserve_runtime_mem_mb <= 0:
        raise ValueError("Runtime memory reserve must be positive")
    socket.inet_aton(args.peer_ip)
    environment = {
        "MODEL_TYPE": "kimi_k3",
        "CHECKPOINT_PATH": str(checkpoint),
        "TOKENIZER_PATH": str(checkpoint),
        "LOAD_METHOD": "fastsafetensors",
        "LOAD_PYTHON_MODEL": "1",
        "ACT_TYPE": "BF16",
        "SP_TYPE": "mtp",
        "SP_MODEL_TYPE": "kimi_k3_mtp",
        "SP_CHECKPOINT_PATH": str(draft),
        "SP_ACT_TYPE": "BF16",
        "FT_DISABLE_CUSTOM_AR": "1",
        "GEN_NUM_PER_CIRCLE": "3",
        "KIMI_K3_PREFILL_CHUNK_TOKENS": "65536",
        "QUANTIZATION": "FP8_PER_BLOCK" if fp8_gemm else "",
        "SP_QUANTIZATION": "",
        "FP8_KV_CACHE": str(int(fp8_kv_cache)),
        "START_PORT": str(args.start_port),
        "LOCAL_WORLD_SIZE": "8",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "FRONTEND_SERVER_COUNT": "1",
        "THINK_START_TAG": "<|open|>think<|sep|>",
        "THINK_END_TAG": "<|close|>think<|sep|><|open|>response<|sep|>",
        "REMOTE_RPC_SERVER_IP": f"{args.peer_ip}:{args.peer_port + 1}",
        "NO_PROXY": f"localhost,127.0.0.1,{args.peer_ip}",
        "no_proxy": f"localhost,127.0.0.1,{args.peer_ip}",
    }
    options = {
        "role_type": args.role,
        "tp_size": 8,
        "ep_size": 8,
        "dp_size": 1,
        "ffn_sp_size": 8,
        "world_size": 8,
        "local_world_size": 8,
        "world_rank": 0,
        "prefill_cp_kv_cache_sharded": 0,
        "prefill_cp_size": 1,
        "remote_server_port": args.peer_port,
        "use_local": 1,
        "max_seq_len": 262144,
        "max_context_batch_size": 16,
        "max_batch_tokens_size": 65536,
        "concurrency_limit": 16,
        "seq_size_per_block": 4096,
        "kernel_seq_size_per_block": 128 if fp8_kv_cache else 64,
        "linear_step": 1,
        "ssm_state_dtype": "fp32",
        "fp8_kv_cache": int(fp8_kv_cache),
        "reuse_cache": 1,
        "enable_device_cache": 1,
        "moe_strategy": "mega_moe",
        "enable_cuda_graph": int(args.role == "DECODE"),
        "cache_store_rdma_mode": 1,
        "cache_store_rdma_connect_timeout_ms": 30000,
        # Formal smoke must expose the first failure, including PD retries.
        "prefill_retry_times": 0,
        "decode_retry_times": 0,
        "load_cache_timeout_ms": 7200000,
        "load_method": "fastsafetensors",
        "warm_up": 0,
        "reserver_runtime_mem_mb": reserve_runtime_mem_mb,
    }
    if args.role == "DECODE":
        options["decode_capture_config"] = "1,2,3,4,7,8,9,16"
    command = [str(Path(args.server).resolve(strict=True))]
    for key, value in options.items():
        command.extend(["--" + key, str(value)])
    return environment, command


def require_local(path):
    resolved = Path(path).resolve(strict=True)
    if not re.match(r"^/(?:ssd|data[0-9]*)/", str(resolved)):
        raise ValueError(f"Not a local data destination: {resolved}")
    fs = subprocess.check_output(
        ["findmnt", "-T", str(resolved), "-n", "-o", "FSTYPE"], text=True
    ).strip()
    if fs not in {"ext4", "xfs", "btrfs"}:
        raise ValueError(f"Unsupported local filesystem: {resolved}: {fs}")


def cpu_tp_socket_environment(run):
    directory = Path(run) / "uds"
    # Match main's socket suffix. DP is fixed to one in this profile.
    socket_path = directory / "rtp_llm_tp_k3_dp0_0.sock"
    if len(os.fsencode(socket_path)) >= 108:
        raise ValueError("Run directory is too long for the CPU TP Unix socket")
    return {
        "RTP_LLM_CPU_TP_BROADCASTER_DIR": str(directory),
        "RTP_LLM_CPU_TP_BROADCASTER_ID": "k3",
    }


def require_rdma_device(run):
    """Reject missing container devices before loading the full checkpoint.

    This proves local device visibility only; real PD transfer is a separate gate.
    """
    evidence = Path(run) / "rdma-preflight.txt"
    try:
        result = subprocess.run(
            ["ibv_devinfo"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        evidence.write_text(f"RDMA device probe failed: {exc}\n")
        raise RuntimeError("Cannot verify RDMA devices in this runtime container") from exc
    evidence.write_text(result.stdout)
    if result.returncode or not re.search(r"\bstate:\s+PORT_ACTIVE\s*\(4\)", result.stdout):
        raise RuntimeError("No active RDMA port in this runtime container")


def validate_rdma_hcas(value, devices, links):
    names = value.split(",")
    if not names or len(set(names)) != len(names) or any(
        not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names
    ):
        raise ValueError("RDMA HCAs must be unique, non-empty device names")
    available = {line.split()[0] for line in devices.splitlines() if line.split()}
    for name in names:
        if name not in available:
            raise ValueError(f"RDMA HCA absent from ibv_devices: {name}")
        pattern = rf"(?<![A-Za-z0-9_.-]){re.escape(name)}/"
        if not any(
            re.search(pattern, line)
            and re.search(r"\bstate\s+ACTIVE\b", line)
            and re.search(r"\bphysical_state\s+LINK_UP\b", line)
            for line in links.splitlines()
        ):
            raise ValueError(f"RDMA HCA is not ACTIVE/LINK_UP: {name}")
    return ",".join(names)


def rdma_hca_environment(run, value):
    if value is None:
        return {}
    devices = subprocess.check_output(["ibv_devices"], text=True, timeout=30)
    links = subprocess.check_output(["rdma", "link", "show"], text=True, timeout=30)
    selected = validate_rdma_hcas(value, devices, links)
    (Path(run) / "rdma-hcas.json").write_text(json.dumps({
        "devices": devices, "links": links, "ACCL_USE_NICS": selected,
        "note": "Local HCA validation only; real RDMA transfer must still pass."
    }, indent=2))
    return {"ACCL_USE_NICS": selected}


def require_gpu_capacity(run, allow_shared_accuracy=False, min_free_gib=250):
    """Record a final TP8 capacity snapshot; shared runs prove correctness only.

    Host-side fleet selection must also record owners: container PID namespaces
    can hide the owners of nvidia-smi's host PIDs.
    """
    if not math.isfinite(min_free_gib) or min_free_gib <= 0:
        raise ValueError("GPU free-memory requirement must be finite and positive")
    def query(fields):
        return subprocess.check_output(
            ["nvidia-smi", fields, "--format=csv,noheader,nounits"],
            text=True, timeout=30,
        )
    gpu_text = query("--query-gpu=index,uuid,memory.free")
    process_text = query("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory")
    evidence = {
        "allow_shared_accuracy": allow_shared_accuracy,
        "performance_validated": False,
        "min_free_gib": min_free_gib,
        "gpus": gpu_text,
        "compute_processes": process_text,
        "owner_evidence": "See host-side fleet selection and process monitor; host PIDs may be hidden here.",
    }
    (Path(run) / "gpu-preflight.json").write_text(json.dumps(evidence, indent=2) + "\n")
    rows = list(csv.reader(gpu_text.splitlines(), skipinitialspace=True))
    selected = {int(row[0]): float(row[2]) for row in rows if len(row) == 3}
    if any(i not in selected or not math.isfinite(selected[i]) or
           selected[i] < min_free_gib * 1024 for i in range(8)):
        raise RuntimeError("Insufficient free GPU memory for the complete TP8 profile; reselect hosts")
    selected_uuids = {row[1].strip() for row in rows if int(row[0]) in range(8)}
    occupied = [row for row in csv.reader(process_text.splitlines(), skipinitialspace=True)
                if row and row[0].strip() in selected_uuids]
    if occupied and not allow_shared_accuracy:
        raise RuntimeError("GPUs occupied; reselect hosts or explicitly allow shared accuracy validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=["PREFILL", "DECODE"])
    parser.add_argument("--debug-four-layer", action="store_true",
                        help="Use a four-layer diagnostic checkpoint; never counts as full-model acceptance")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--draft-checkpoint", required=True)
    parser.add_argument("--peer-ip", required=True)
    parser.add_argument("--start-port", required=True, type=int)
    parser.add_argument("--peer-port", required=True, type=int)
    parser.add_argument("--server", required=True)
    parser.add_argument("--guard", required=True, help="weight_loader_guard.py")
    parser.add_argument("--rdma-hcas", help="Explicit comma-separated Barex HCA allowlist")
    parser.add_argument("--reserve-runtime-mem-mb", type=int, default=14336,
                        help="Per-rank runtime reserve; validated PD427 used 14336 MiB")
    parser.add_argument(
        "--run-dir", required=True, help="New directory on a local data disk"
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--fp8-gemm", action="store_true", help="Enable FP8 projection GEMM")
    parser.add_argument("--fp8-kv-cache", action="store_true", help="Enable ordinary E4M3 MLA operands and KV cache via the existing FP8_KV_CACHE setting")
    parser.add_argument("--allow-shared-accuracy", action="store_true",
                        help="Allow correctness-only coexistence after host-side isolation checks; never for performance")
    parser.add_argument("--min-free-gib", type=float, default=250,
                        help="Required free memory on each of GPUs 0-7 (default: 250 GiB)")
    args = parser.parse_args()
    environment, command = launch_config(args)
    if args.print_config:
        print(json.dumps({"environment": environment, "command": command}, indent=2))
        return
    if pwd.getpwuid(os.getuid()).pw_name != "luohaocheng.lhc":
        raise ValueError("Must run as luohaocheng.lhc inside the verified runtime container")
    run = Path(args.run_dir).resolve()
    require_local(run.parent)
    environment.update(cpu_tp_socket_environment(run))
    run.mkdir(exist_ok=False)
    (run / "uds").mkdir(mode=0o700)
    require_rdma_device(run)
    environment.update(rdma_hca_environment(run, args.rdma_hcas))
    # Reject potentially inherited experimental settings instead of trusting
    # a shell left over from a DCP/FP8/synthetic-acceptance experiment.
    forbidden = ("CP_ROTATE_METHOD", "QUANTIZATION", "SP_QUANTIZATION")
    for name in forbidden:
        if os.environ.get(name):
            raise ValueError(f"Unset inherited {name} before precision validation")
    inherited = os.environ.copy()
    inherited.update(environment)
    for key, subdir in {
        "TMPDIR": "tmp",
        "LOG_PATH": "logs",
        "TRITON_CACHE_DIR": "triton",
        "DG_JIT_CACHE_DIR": "deep-gemm",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    }.items():
        directory = run / subdir
        directory.mkdir()
        inherited[key] = str(directory)
        environment[key] = str(directory)
    for label, checkpoint in (
        ("target", args.checkpoint),
        ("draft", args.draft_checkpoint),
    ):
        require_local(checkpoint)
        with (run / f"{label}-preflight.txt").open("w") as output:
            subprocess.run(
                [
                    sys.executable,
                    args.guard,
                    "preflight",
                    "--checkpoint",
                    checkpoint,
                    "--local-data-root",
                    str(Path(checkpoint).resolve().parents[0]),
                ],
                env=inherited,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
    # This is an additional final check, not a replacement for fleet selection.
    require_gpu_capacity(run, args.allow_shared_accuracy, args.min_free_gib)
    ports = []
    try:
        for port in range(args.start_port, args.start_port + 8 * 9):
            sock = socket.socket()
            ports.append(sock)
            sock.bind(("0.0.0.0", port))
    finally:
        for sock in ports:
            sock.close()
    (run / "launch.json").write_text(
        json.dumps(
            {
                "profile": (
                    f"{'fp8' if args.fp8_kv_cache or args.fp8_gemm else 'bf16'}-"
                    f"{'debug4' if args.debug_four_layer else 'full93'}-tp8-ep8-sp-mtp3-rdma"
                ),
                "full_model_acceptance_eligible": not args.debug_four_layer,
                "environment": environment,
                "command": command,
                "pid": os.getpid(),
                "allow_shared_accuracy": args.allow_shared_accuracy,
                "performance_validated": False,
            },
            indent=2,
        )
        + "\n"
    )
    os.execve(command[0], command, inherited)


if __name__ == "__main__":
    main()
