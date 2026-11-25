#!/usr/bin/env python3

import argparse
import ctypes
import fcntl
import json
import os
import re
import sys
from pathlib import Path


def check_rdma(device, port):
    verbs = Path("/dev/infiniband")
    hca = Path("/sys/class/infiniband") / device
    userspace = [
        entry
        for entry in Path("/sys/class/infiniband_verbs").glob("uverbs*")
        if (entry / "ibdev").is_file() and (entry / "ibdev").read_text().strip() == device
    ]
    if not any(os.access(verbs / entry.name, os.R_OK | os.W_OK) for entry in userspace):
        raise RuntimeError(f"{device}: no accessible RDMA verbs device under {verbs}")
    state = hca / "ports" / str(port) / "state"
    if not state.is_file() or state.read_text().split(":", 1)[0] != "4":
        raise RuntimeError(f"{device} port {port} is not ACTIVE")


def pick_rdma(port):
    for entry in sorted(Path("/sys/class/infiniband_verbs").glob("uverbs*")):
        try:
            device = (entry / "ibdev").read_text().strip()
            check_rdma(device, port)
            os.close(os.open(Path("/dev/infiniband") / entry.name, os.O_RDWR))
        except (OSError, RuntimeError):
            continue
        return device
    print(f"no openable RDMA device with port {port} ACTIVE", file=sys.stderr)
    sys.exit(77)


def check_gdr():
    cuda = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGet(ctypes.byref(device), 0) != 0:
        raise RuntimeError("cannot initialize CUDA on the allocated GPU")
    for name, attribute in (("GPUDirect RDMA", 116), ("DMA-BUF", 124)):
        supported = ctypes.c_int()
        if cuda.cuDeviceGetAttribute(ctypes.byref(supported), attribute, device) != 0:
            raise RuntimeError(f"cannot query CUDA {name} support")
        if not supported.value:
            print(f"allocated GPU does not expose {name}", file=sys.stderr)
            sys.exit(77)


def start(args, parser):
    if args.role == "check-rdma":
        if not args.rdma_device:
            parser.error("check-rdma requires --rdma-device")
        if args.rdma_device == "auto":
            device = pick_rdma(args.rdma_port)
            if args.transport in ("aft", "aft-gdr"):
                check_gdr()
            print(device)
            return
        check_rdma(args.rdma_device, args.rdma_port)
        print(f"{args.rdma_device} port {args.rdma_port} is accessible and ACTIVE")
        return
    if args.bin_dir is None or args.work_dir is None or args.port_base is None:
        parser.error("starting a service requires --bin-dir, --work-dir, and --port-base")
    if not 1024 <= args.port_base <= 65532:
        parser.error("--port-base must be between 1024 and 65532")
    data_role = args.role in ("consumer", "provider")
    aft = args.transport in ("aft", "aft-rdma", "aft-gdr")
    if data_role:
        if not all((args.meta_url, args.interface)):
            parser.error("a data service requires --meta-url and --interface")
        if args.role == "consumer" and not args.provider_status_url:
            parser.error("consumer requires --provider-status-url for the smoke fixture")
        if args.transport != "tcp":
            if not args.rdma_device:
                parser.error("RDMA requires --rdma-device")
            check_rdma(args.rdma_device, args.rdma_port)
        lock = Path("/dev/shm/rtp_llm_pace_test.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((Path("/dev/shm") / name).exists() for name in ("shm_mps", "shm_mps_meta")):
            raise RuntimeError("PACE shared memory already exists. Give each data service its own /dev/shm.")
        os.set_inheritable(lock.fileno(), True)
    work_dir = args.work_dir.resolve()
    bin_dir = args.bin_dir.resolve()
    binary = bin_dir / ("tair_mempool_server" if data_role else "tair_mempool_meta_service")
    if not os.access(binary, os.X_OK):
        raise RuntimeError(f"{binary} is not executable")
    if data_role and len(os.fsencode(work_dir / "mp_data.sock")) >= 108:
        raise RuntimeError("use a shorter --work-dir for the Unix sockets")
    work_dir.mkdir(parents=True)
    env = os.environ.copy()
    runtime_root = bin_dir.parent if args.role == "consumer" and bin_dir.name == "consumer" else bin_dir
    libraries = [str(bin_dir)]
    if (runtime_root / "runtime-libs").is_dir():
        libraries.append(str(runtime_root / "runtime-libs"))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (*libraries, env.get("LD_LIBRARY_PATH"))))
    env["ENABLE_TAIR_MEMPOOL_HUGEPAGES"] = "OFF"
    if not data_role:
        command = [
            str(binary), "-b", "memory", "-l", "no", "-p", str(args.port_base),
            "-M", str(args.port_base + 1), "-P", "2",
        ]
        if aft:
            command += ["-t", "2", "-i", "1"]
            env["TAIR_MEMPOOL_META_CONN_SUSPECT_MAX_FAIL_CNT"] = "2"
            env["TAIR_MEMPOOL_META_HEALTH_PROBE_BUDGET_MS"] = "500"
    else:
        env.setdefault("TAIR_MEMPOOL_PROVIDER_HTTP_THREADS", "4")
        zero_copy = "0" if args.transport in ("tcp", "aft-rdma") else "1"
        client_env = {
            "TAIR_MEMPOOL_SOCK": str(work_dir / "mp.sock"),
            "TAIR_MEMPOOL_DATA_SOCK": str(work_dir / "mp_data.sock"),
            "TAIR_MEMPOOL_ENABLE_TENT": "0" if aft else "1",
            "TAIR_MEMPOOL_ENABLE_EXTERNAL_ZERO_COPY": zero_copy,
            "TAIR_MEMPOOL_ENABLE_GDR_ZERO_COPY": zero_copy,
            "TAIR_MEMPOOL_ENABLE_GDR_READ_ZERO_COPY": zero_copy,
            "TAIR_MEMPOOL_ENABLE_GDR_WRITE_ZERO_COPY": "0" if aft else zero_copy,
            "TAIR_MEMPOOL_GDR_TRANSFER_MODE": "direct",
        }
        if aft:
            client_env.update({
                "TAIR_MEMPOOL_PIPELINE_WRITE": "0",
                "TAIR_MEMPOOL_VCNS_IO_TIMEOUT_MS": "400",
                "TAIR_MEMPOOL_PROMOTE_TIMEOUT_MS": "700",
                "TAIR_MEMPOOL_RESOLVER_TIMEOUT_MS": "1000",
                "TAIR_MEMPOOL_TIERED_ACQUIRE_TIMEOUT_MS": "1000",
                "TAIR_MEMPOOL_CLIENT_IO_TIMEOUT_MS": "3000",
                "TAIR_MEMPOOL_SYNC_TIMEOUT_MS": "8000",
            })
            env["TAIR_MEMPOOL_AFT_RDMA_ADMIN_SUPPORTED_NODE_NUM"] = "8"
            env["TAIR_MEMPOOL_AFT_QP_SOFT_INFLIGHT_LIMIT"] = "64"
            role_args = ["--tiering_mode", "transparent"]
            if args.role == "provider":
                role_args.append("--push_only_mode")
            else:
                role_args.append("--clientregister")
                env["TAIR_MEMPOOL_EXPECTED_EXTERNAL_CLIENTS"] = zero_copy
        else:
            role_args = ["--use_tent", "--tiering_mode", "off"]
            tent_transport = "tcp" if args.transport == "tcp" else "rdma"
            tent_config = work_dir / "tent.json"
            transports = {
                name: {"enable": name == tent_transport}
                for name in ("rdma", "aft", "barex", "tcp", "uring", "shm", "gds", "mnnvl")
            }
            profile = {
                "transports": transports,
                "policy": [{"name": tent_transport, "segment_type": "memory", "transports": [tent_transport]}],
            }
            if tent_transport == "rdma":
                transports["rdma"]["device"] = {"port": args.rdma_port, "gid_index": args.gid_index}
                profile["topology"] = {"rdma_whitelist": [args.rdma_device]}
            tent_config.write_text(json.dumps(profile, indent=2) + "\n")
            client_env["MC_TENT_CONF"] = str(tent_config)
        env.update(client_env)
        if args.role == "consumer":
            fixture = {
                "domain": args.meta_url,
                "transport": args.transport,
                "provider_status_urls": [args.provider_status_url],
                "consumer_metrics_url": f"http://127.0.0.1:{args.port_base + 3}/pace/metrics",
                "client_env": client_env,
            }
            if args.fault_socket:
                fixture["fault_socket"] = args.fault_socket
            if args.provider_offline_dir:
                fixture["provider_offline_dir"] = args.provider_offline_dir
            if args.provider_metrics_url:
                fixture["provider_metrics_url"] = args.provider_metrics_url
            (work_dir / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
        cache_args = ["-R", "10"] if args.role == "consumer" else ["-R", "0", "-c"]
        command = [
            str(binary), "--role", args.role, *role_args,
            "--network_interface", args.interface, "--metaservice", args.meta_url,
            "--metric_type", "prometheus", "-s", "2048", "--shm_meta_size_mb", "1024",
            *cache_args, "-p", "2", "--ctrl_threads", "2", "-a", str(args.port_base),
            "-r", str(args.port_base + 1), "-t", str(args.port_base + 2), "-m", str(args.port_base + 3),
        ]
    os.chdir(work_dir)
    os.execve(str(binary), command, env)


def main():
    parser = argparse.ArgumentParser(description="Start one PACE DRAM role for AFT or TENT smoke.")
    parser.add_argument("role", choices=("check-rdma", "meta", "provider", "consumer"))
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--port-base", type=int)
    parser.add_argument("--meta-url")
    parser.add_argument("--provider-status-url")
    parser.add_argument("--provider-metrics-url")
    parser.add_argument("--interface")
    parser.add_argument(
        "--transport", choices=("aft", "aft-rdma", "aft-gdr", "tcp", "rdma", "gdr"), default="rdma"
    )
    parser.add_argument("--rdma-device")
    parser.add_argument("--rdma-port", type=int, default=1)
    parser.add_argument("--gid-index", type=int, default=0)
    parser.add_argument("--fault-socket")
    parser.add_argument("--provider-offline-dir")
    args = parser.parse_args()
    if args.rdma_device and not re.fullmatch(r"[a-zA-Z0-9_.-]+", args.rdma_device):
        parser.error("invalid --rdma-device")
    if args.rdma_port < 1 or args.gid_index < 0:
        parser.error("RDMA port must be positive and GID index must be nonnegative")
    start(args, parser)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(f"PACE smoke startup failed: {error}", file=sys.stderr)
        sys.exit(1)
