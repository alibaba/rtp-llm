"""Launch reproducible dual-host CP8 -> DP8 internal TCP bring-up.

This uses the verified standalone compute drivers. It is not the production
HTTP service or native CacheStore, and cannot perform an RDMA verification.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("prefill", "decode"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--peer", default="127.0.0.1", help="D host for P transfers")
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--base-port", type=int, default=19400)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(os.environ.get("RTP_V41_RUN_ROOT", "/tmp/rtp-llm-v41-runs")),
    )
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument(
        "--tokens", type=int, default=129, help="input tokens per request"
    )
    parser.add_argument(
        "--steps", type=int, default=2, help="D steps after P's first token"
    )
    parser.add_argument("--fixture-steps", type=int, default=1)
    parser.add_argument("--owner", type=int, choices=range(8))
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--rdma", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_id):
        parser.error("--run-id must be a unique simple directory name")
    if args.rdma:
        parser.error(
            "this internal fixture transport is TCP only; native CacheStore RDMA remains unverified"
        )
    if not args.synthetic and args.checkpoint_path is None:
        parser.error("--checkpoint-path or --synthetic is required")
    if not 0 < args.tokens or not 0 < args.requests or not 0 < args.steps:
        parser.error("tokens, requests and steps must be positive")
    if args.tokens + args.steps + 1 > 32768:
        parser.error("input + generated output must fit 32K")
    if not 0 <= args.fixture_steps <= args.steps:
        parser.error("fixture steps must be in [0, steps]")
    if not 1024 <= args.base_port <= 65528:
        parser.error("eight internal TCP ports must fit in [1024, 65535]")

    from rtp_llm.models_py.modules.dsv41.ppu.pd_handoff import runtime_identity

    identity = runtime_identity()
    root = Path(__file__).resolve().parents[3]
    run = args.run_root / identity["snapshot_id"] / args.run_id / args.role
    if run.exists():
        parser.error(f"run already exists; choose a fresh run id: {run}")
    if args.role == "decode":
        # Hold all sockets during the check, so duplicates also fail here.
        sockets = []
        try:
            for port in range(args.base_port, args.base_port + 8):
                sock = socket.socket()
                sockets.append(sock)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((args.bind_host, port))
        finally:
            for sock in sockets:
                sock.close()
    report = {
        "identity": identity,
        "role": args.role,
        "transport": "internal-fixture-tcp",
        "native_cache_store": False,
        "rdma_validated": False,
        "attention_tp": 1,
        "attention_cp": 8 if args.role == "prefill" else 1,
        "attention_dp": 1 if args.role == "prefill" else 8,
        "ep": 8,
        "enp": 8,
        "run_id": args.run_id,
        "run_dir": str(run),
        "input_tokens_per_request": args.tokens,
        "requests": args.requests,
        "generated_tokens_per_request": args.steps + 1,
        "source_root": str(root),
        "python": sys.executable,
    }
    if not args.synthetic:
        from rtp_llm.tools.v41.preflight import checkpoint_metadata

        report["checkpoint"] = checkpoint_metadata(args.checkpoint_path)
    print(json.dumps(report, indent=2), flush=True)
    if args.check_only:
        return 0
    run.mkdir(parents=True, exist_ok=False)
    (run / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
    os.environ["RTP_V41_SNAPSHOT_ID"] = identity["snapshot_id"]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    driver_dir = root / "rtp_llm/models_py/modules/dsv41/ppu/tools"
    common = [
        "--device",
        args.device,
        "--nproc",
        "8",
        "--timeout",
        str(args.timeout),
        "--pd-timeout",
        str(args.timeout),
        "--pd-run-id",
        args.run_id,
        "--pd-base-port",
        str(args.base_port),
        "--output",
        str(run / "summary.json"),
    ]
    if args.synthetic:
        common.append("--synthetic")
    else:
        common += [
            "--config-path",
            str(args.checkpoint_path / "config.json"),
            "--checkpoint-path",
            str(args.checkpoint_path),
        ]
    if args.role == "prefill":
        command = [
            sys.executable,
            "-m",
            "rtp_llm.models_py.modules.dsv41.ppu.tools.driver_prefill_cp8",
            *common,
            "--tokens",
            str(args.tokens * args.requests),
            "--requests",
            str(args.requests),
            "--pd-peer",
            args.peer,
            "--fixture-dir",
            str(run / "fixture"),
            "--token-map-cache",
            str(run / "token_map.json"),
        ]
        if args.owner is not None:
            command += ["--pd-owner", str(args.owner)]
    else:
        command = [
            sys.executable,
            str(driver_dir / "driver_decode_dp8.py"),
            *common,
            "--pd-receive",
            "--pd-bind-host",
            args.bind_host,
            "--tokens",
            str(args.tokens),
            "--steps",
            str(args.steps),
            "--fixture-steps",
            str(args.fixture_steps),
        ]
    os.chdir(root)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    raise SystemExit(main())
