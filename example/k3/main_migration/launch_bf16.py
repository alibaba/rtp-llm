"""Launch one full-model BF16 PD endpoint after local preflight.

Run inside lhc_GPU as luohaocheng.lhc after a fresh same-cluster fleet selection.
This launcher does not select hosts or certify runtime precision/Graph/RDMA.
"""

import argparse
import json
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
import sys


def launch_config(args):
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    draft = Path(args.draft_checkpoint).resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    text = config.get("text_config", config)
    if text.get("num_hidden_layers") != 93:
        raise ValueError("Formal BF16 profile requires all 93 target layers")
    for port in (args.start_port, args.peer_port):
        if port < 1024 or port + 8 * 9 > 65535:
            raise ValueError("Invalid eight-rank service port range")
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
        "GEN_NUM_PER_CIRCLE": "3",
        "KIMI_K3_PREFILL_CHUNK_TOKENS": "65536",
        "FP8_GEMM": "0",
        "FP8_MLA": "0",
        "FP8_KV_CACHE": "0",
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
        "kernel_seq_size_per_block": 128,
        "linear_step": 1,
        "ssm_state_dtype": "fp32",
        "int8_kv_cache": 0,
        "fp8_kv_cache": 0,
        "reuse_cache": 1,
        "enable_device_cache": 1,
        "moe_strategy": "mega_moe",
        "enable_cuda_graph": int(args.role == "DECODE"),
        "cache_store_rdma_mode": 1,
        "cache_store_rdma_connect_timeout_ms": 30000,
        "load_cache_timeout_ms": 7200000,
        "load_method": "fastsafetensors",
        "warm_up": 0,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=["PREFILL", "DECODE"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--draft-checkpoint", required=True)
    parser.add_argument("--peer-ip", required=True)
    parser.add_argument("--start-port", required=True, type=int)
    parser.add_argument("--peer-port", required=True, type=int)
    parser.add_argument("--server", required=True)
    parser.add_argument("--guard", required=True, help="weight_loader_guard.py")
    parser.add_argument(
        "--run-dir", required=True, help="New directory on a local data disk"
    )
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    environment, command = launch_config(args)
    if args.print_config:
        print(json.dumps({"environment": environment, "command": command}, indent=2))
        return
    if pwd.getpwuid(os.getuid()).pw_name != "luohaocheng.lhc":
        raise ValueError("Must run as luohaocheng.lhc inside lhc_GPU")
    run = Path(args.run_dir).resolve()
    require_local(run.parent)
    run.mkdir(exist_ok=False)
    # Reject potentially inherited experimental settings instead of trusting
    # a shell left over from a DCP/FP8/synthetic-acceptance experiment.
    forbidden = ("CP_ROTATE_METHOD", "QUANTIZATION", "SP_QUANTIZATION")
    for name in forbidden:
        if os.environ.get(name):
            raise ValueError(f"Unset inherited {name} before BF16 validation")
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
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if processes:
        raise RuntimeError(f"GPUs occupied; reselect hosts. Compute PIDs: {processes}")
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
                "profile": "bf16-full93-tp8-ep8-sp-mtp3-rdma",
                "environment": environment,
                "command": command,
                "pid": os.getpid(),
            },
            indent=2,
        )
        + "\n"
    )
    os.execve(command[0], command, inherited)


if __name__ == "__main__":
    main()
