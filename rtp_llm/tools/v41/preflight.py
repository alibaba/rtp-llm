"""Read-only V4.1 PPU preflight. Failures are data and a nonzero exit code.

This checks prerequisites, not EP/ENP numerical correctness or RDMA transfers.
No full table is loaded or pinned. Small allocation probes are explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from rtp_llm.config.v41_runtime import CONFIG_SHA256, V41RuntimeConfig, env_bool
from rtp_llm.models_py.modules.dsv41.ppu.backend import KERNEL_SYMBOLS
from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import native_cache_protocol

GIB = 1024**3
SDK_VERSION = "2.2.0-663c22"
WHEELS = {
    "torch": "2.11.0+v0.2.0.ppu2.2.0",
    "triton": "3.7.0+v0.1.0.ppu2.2.0",
    "deep-gemm": "1.0.0+v0.4.0.ppu2.2.0",
    "deep-ep": "2.2.1+v0.1.0.ppu2.2.0",
    "flash-mla": "2.0.0+v0.2.0.ppu2.2.0",
    "flash-attn": "2.8.2+v0.2.0.ppu2.2.0",
    "flash-attn-3": "2.8.2+v0.2.0.ppu2.2.0",
    "flashinfer-python": "0.6.8.post1+v0.2.0.ppu2.2.0",
    "fast-hadamard-transform": "1.1.0.post2+v0.1.0.ppu2.2.0",
    "safetensors": "0.8.0",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_metadata(path, expected=CONFIG_SHA256):
    path = Path(path)
    config_hash = sha256(path / "config.json")
    if config_hash != expected:
        raise ValueError(f"config SHA256 mismatch: {config_hash} != {expected}")
    raw = json.loads((path / "config.json").read_text())
    text = raw["text_config"]
    if (text["n_routed_experts"], text["num_experts_per_tok"]) != (384, 6):
        raise ValueError("release requires 384 routed experts and model top-k=6")
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (path / name).is_file()]
    if missing:
        raise ValueError(f"checkpoint shards missing: {missing}")
    # This is explicitly a metadata identity, not a hash of 510 GB of weights.
    # A6 generates and reuses the complete weight signature separately.
    tokenizer = {p.name: sha256(p) for p in path.glob("tokenizer*") if p.is_file()}
    if "tokenizer.json" not in tokenizer:
        raise ValueError("tokenizer.json missing")
    dim = int(text["engram_head_dim"])
    table_bytes = sum(text["engram_num_embeddings"]) * (dim + dim // 32)
    return {
        "config_sha256": config_hash,
        "index_sha256": sha256(index_path),
        "tokenizer_sha256": tokenizer,
        "shard_count": len(shards),
        "declared_weight_bytes": index["metadata"]["total_size"],
        "full_weight_signature": "not_measured",
        "engram_table_bytes": table_bytes,
        "engram_shard_estimate_bytes": (table_bytes + 7) // 8,
    }


def memory_limits(cgroup_root=Path("/sys/fs/cgroup")):
    """Resolve this process's v1/v2 cgroup, including nested container paths."""
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
    limits = []
    roots = {Path(cgroup_root), Path(cgroup_root) / "memory"}
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if not controllers or "memory" in controllers.split(","):
            relative = relative.lstrip("/")
            roots.add(Path(cgroup_root) / relative)
            roots.add(Path(cgroup_root) / "memory" / relative)
    for root in roots:
        # Include ancestor constraints: a child may say "max" under a cap.
        while root == Path(cgroup_root) or Path(cgroup_root) in root.parents:
            for limit_name, used_name in (
                ("memory.max", "memory.current"),
                ("memory.limit_in_bytes", "memory.usage_in_bytes"),
            ):
                limit_path, used_path = root / limit_name, root / used_name
                if limit_path.is_file() and used_path.is_file():
                    value = limit_path.read_text().strip()
                    if value != "max":
                        cap = int(value)
                        if cap < 2**60:
                            limits.append(max(0, cap - int(used_path.read_text())))
            if root == Path(cgroup_root):
                break
            root = root.parent
    candidates = limits + ([] if available is None else [available])
    if not candidates:
        raise ValueError("cannot determine available host/cgroup memory")
    return {
        "host_available_bytes": available,
        "cgroup_remaining_bytes": limits,
        "effective_available_bytes": min(candidates),
    }


def service_ports(start_port, world_size=8, dp_size=1, stride=9, frontend_count=4):
    """Match ServerConfig rank base + offsets and frontend parallel slots.

    One service per host; these probes are advisory until the actual bind.
    """
    ports = set(range(start_port, start_port + world_size * stride))
    ports.update(start_port + 5 + i for i in range(frontend_count))
    # DistributedServer: TCPStore=-1, torch init=-11, dp_tp=-10;
    # each DP rank gets tp/ffn rendezvous at base - dp_rank*11 - 2/-5.
    ports.update((start_port - 1, start_port - 10, start_port - 11))
    for dp_rank in range(dp_size):
        ports.update((start_port - dp_rank * 11 - 2, start_port - dp_rank * 11 - 5))
    if min(ports) < 1024 or max(ports) > 65535:
        raise ValueError("service port range must fit [1024, 65535]")
    return sorted(ports)


def check_ports(start_port, dp_size=1):
    ports = service_ports(start_port, dp_size=dp_size)
    busy = []
    for port in sorted(ports):
        with socket.socket() as probe:
            try:
                probe.bind(("0.0.0.0", port))
            except OSError as error:
                busy.append({"port": port, "error": str(error)})
    if busy:
        raise ValueError(f"ports unavailable: {busy}")
    return {"ports": ports, "rank_stride": 9, "atomic_reservation": False}


def rdma_inventory(sys_class=None, device_root=None):
    sys_class = Path("/sys/class/infiniband") if sys_class is None else Path(sys_class)
    device_root = Path("/dev/infiniband") if device_root is None else Path(device_root)
    devices = {}
    for device in sorted(sys_class.glob("*")):
        devices[device.name] = {
            str(port.name): {
                name: (port / name).read_text().strip()
                for name in ("state", "phys_state", "rate")
                if (port / name).is_file()
            }
            for port in (device / "ports").glob("*")
        }
    nodes = sorted(str(p) for p in device_root.glob("*"))
    accessible, open_errors = [], {}
    for node in nodes:
        if not Path(node).name.startswith("uverbs"):
            continue
        try:
            fd = os.open(node, os.O_RDWR)
            os.close(fd)
            accessible.append(node)
        except OSError as error:
            open_errors[node] = str(error)
    active = any(
        "ACTIVE" in port.get("state", "")
        for ports in devices.values()
        for port in ports.values()
    )
    return {
        "devices": devices,
        "device_nodes": nodes,
        "accessible_uverbs": accessible,
        "device_open_errors": open_errors,
        "active_port": active,
        "transfer_validation": "not_run",
    }


def collect_report(env=None, *, probe_allocations=False, require_backend=False):
    env = dict(os.environ if env is None else env)
    rdma_required = env_bool(env, "CACHE_STORE_RDMA_MODE")
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python": sys.version,
        "uid": os.getuid(),
        "cache_store_mode": "rdma" if rdma_required else "tcp",
        "tool_source_sha256": sha256(__file__),
        "runtime_config_source_sha256": sha256(
            Path(__file__).parents[2] / "config/v41_runtime.py"
        ),
        "checks": [],
        "acceptance": {
            name: "not_run"
            for name in (
                "control_collective_8rank",
                "ep_sample",
                "enp_small_table",
                "tcp_pd",
                "rdma_pd",
            )
        },
    }

    def check(name, action, required=True):
        try:
            result = action()
            report["checks"].append({"name": name, "status": "pass", "details": result})
            return result
        except Exception as error:
            report["checks"].append(
                {
                    "name": name,
                    "status": "fail" if required else "blocked",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            return None

    def require(condition, message, value=None):
        if not condition:
            raise ValueError(message)
        return value

    runtime = check("topology", lambda: V41RuntimeConfig.from_env(env).summary())
    check("native_cache_protocol", native_cache_protocol)
    check(
        "python_abi",
        lambda: require(
            sys.version_info[:2] == (3, 12) and platform.machine() == "x86_64",
            "requires Python 3.12 x86_64",
            platform.platform(),
        ),
    )
    sdk = Path(env.get("PPU_HOME", "/usr/local/PPU_SDK"))
    check(
        "sdk",
        lambda: require(
            (sdk / "release.yaml").read_text().strip() == f"version: {SDK_VERSION}",
            f"requires SDK {SDK_VERSION}",
            SDK_VERSION,
        ),
    )
    check(
        "compiler",
        lambda: require(
            shutil.which("ppu-smi") is not None,
            "ppu-smi must be on PATH for Triton backend selection",
            shutil.which("ppu-smi"),
        ),
    )
    check(
        "bazel",
        lambda: require(
            bool(shutil.which("bazel") or shutil.which("bazelisk")),
            "Bazel/Bazelisk is missing",
        ),
    )

    def native_libraries():
        root = Path(__file__).parents[3]
        directories = [
            root / "rtp_llm/libs",
            root / "bazel-bin",
            root / "bazel-bin/rtp_llm/libs",
        ]
        found = {}
        for name in (
            "libth_transformer_config.so",
            "librtp_compute_ops.so",
            "libth_transformer.so",
        ):
            paths = [
                str(directory / name)
                for directory in directories
                if (directory / name).is_file()
            ]
            require(bool(paths), f"native build product missing: {name}")
            found[name] = paths[0]
        return found

    check("native_libraries", native_libraries, required=require_backend)
    for name, version in WHEELS.items():

        def wheel(name=name, version=version):
            dist = importlib.metadata.distribution(name)
            require(dist.version == version, f"{name}: {dist.version} != {version}")
            return {
                "version": dist.version,
                "record_sha256": hashlib.sha256(
                    (dist.read_text("RECORD") or "").encode()
                ).hexdigest(),
            }

        check(f"wheel:{name}", wheel)
    metadata = check(
        "checkpoint_metadata", lambda: checkpoint_metadata(env["CHECKPOINT_PATH"])
    )
    host = check("host_memory", memory_limits)
    # Whole role owns one table (~189 GiB), not eight replicated tables.
    # Reserve a second shard per worker for loading plus 64 GiB control/runtime.
    required_host = (
        (metadata["engram_table_bytes"] * 2 + 64 * GIB) if metadata else None
    )
    check(
        "host_budget",
        lambda: require(
            host is not None
            and required_host is not None
            and host["effective_available_bytes"] >= required_host,
            f"host/cgroup available memory below ENP+load staging budget {required_host}",
            {
                "required_bytes": required_host,
                "basis": "2 * table + 64 GiB; conservative admission budget",
            },
        ),
    )

    def ipc():
        usage = shutil.disk_usage("/dev/shm")
        require(
            usage.free >= GIB, "prepare at least 1 GiB free /dev/shm for control/CP IPC"
        )
        return dict(zip(("total", "used", "free"), usage))

    check("ipc_shm", ipc)
    pinned_budget = (
        ((metadata["engram_table_bytes"] + 7) // 8 + GIB) if metadata else GIB
    )
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    check(
        "memlock",
        lambda: require(
            soft == resource.RLIM_INFINITY or soft >= pinned_budget,
            f"memlock must cover local shard + 1 GiB ({pinned_budget} bytes)",
            {"soft": soft, "hard": hard, "required_per_rank_bytes": pinned_budget},
        ),
    )
    check(
        "ports",
        lambda: check_ports(
            int(env.get("START_PORT", "8090")), dp_size=int(env.get("DP_SIZE", "1"))
        ),
    )
    rdma = check("rdma_inventory", rdma_inventory)
    check(
        "rdma_device_access",
        lambda: require(
            bool(rdma and rdma["active_port"] and rdma["accessible_uverbs"]),
            "RDMA active ports and openable uverbs required",
            rdma,
        ),
        required=rdma_required,
    )
    try:
        torch = importlib.import_module("torch")
    except Exception as error:
        check("ppu_devices", lambda: require(False, str(error)))
    else:

        def devices():
            require(
                torch.cuda.device_count() == 8, "requires exactly 8 visible PPU devices"
            )
            result = []
            for rank in range(8):
                p = torch.cuda.get_device_properties(rank)
                require(
                    p.name == "ZW-M890P" and p.total_memory >= 144 * GIB,
                    f"unexpected device {rank}: {p}",
                )
                free, total = torch.cuda.mem_get_info(rank)
                result.append(
                    {
                        "rank": rank,
                        "name": p.name,
                        "total_bytes": total,
                        "free_bytes": free,
                    }
                )
            return result

        check("ppu_devices", devices)
        for module_name, functions in KERNEL_SYMBOLS:

            def capabilities(module_name=module_name, functions=functions):
                module = importlib.import_module(module_name)
                present = {
                    name: callable(getattr(module, name, None)) for name in functions
                }
                require(
                    all(present.values()),
                    f"{module_name} missing required symbols: {present}",
                )
                return present

            check(f"symbols:{module_name}", capabilities)
        if probe_allocations:

            def pinned_probe():
                x = torch.empty(1024**2, dtype=torch.uint8, pin_memory=True)
                require(x.is_pinned(), "1 MiB pinned allocation failed")
                return {"bytes": x.numel(), "full_shard_allocation": "not_tested"}

            check("pinned_allocation_probe", pinned_probe)
        else:
            report["checks"].append(
                {"name": "pinned_allocation_probe", "status": "not_run"}
            )

    def provider():
        from rtp_llm.models_py.modules.dsv41.backend import get_v41_backend

        backend = get_v41_backend("ppu")
        if runtime:
            backend.check_runtime(V41RuntimeConfig.from_env(env))
        return {
            "contract_version": backend.contract_version,
            "strategy": backend.strategy,
        }

    check("model_backend", provider, required=require_backend)
    report["prerequisites_ok"] = not any(
        c["status"] == "fail" for c in report["checks"]
    )
    passed = {c["name"] for c in report["checks"] if c["status"] == "pass"}
    report["rdma_ready"] = "rdma_device_access" in passed
    report["model_ready"] = report["prerequisites_ok"] and {
        "native_libraries",
        "model_backend",
    }.issubset(passed)
    report["m0_complete"] = (
        False  # A1/A2/A4 collective/numerical samples are separate gates.
    )
    return report


def compare_reports(left, right):
    """Compare two captured environments, without inferring weight-byte equality."""
    identity_names = ["sdk", "checkpoint_metadata", "native_cache_protocol"] + [
        f"wheel:{name}" for name in WHEELS
    ]
    by_name = [{c["name"]: c for c in report["checks"]} for report in (left, right)]
    differences = []
    for name in identity_names:
        entries = [checks.get(name, {}) for checks in by_name]
        if any(e.get("status") != "pass" for e in entries):
            differences.append(
                {"name": name, "reason": "identity unavailable on one or both hosts"}
            )
            continue
        values = [e["details"] for e in entries]
        if name.startswith("wheel:"):
            values = [v["version"] for v in values]
        if values[0] != values[1]:
            differences.append({"name": name, "reason": "identity mismatch"})
    roles = {c.get("topology", {}).get("details", {}).get("role") for c in by_name}
    if roles != {"PREFILL", "DECODE"}:
        differences.append(
            {"name": "topology", "reason": "requires one PREFILL and one DECODE report"}
        )
    return {
        "schema_version": 1,
        "identities_match": not differences,
        "both_prerequisites_ok": all(r["prerequisites_ok"] for r in (left, right)),
        "differences": differences,
        "full_weight_signature_verified": False,
        "m0_complete": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-allocations", action="store_true")
    parser.add_argument("--require-backend", action="store_true")
    parser.add_argument(
        "--compare", type=Path, nargs=2, metavar=("P_REPORT", "D_REPORT")
    )
    args = parser.parse_args()
    if args.compare:
        report = compare_reports(
            *(json.loads(path.read_text()) for path in args.compare)
        )
        ok = report["identities_match"] and report["both_prerequisites_ok"]
    else:
        report = collect_report(
            probe_allocations=args.probe_allocations,
            require_backend=args.require_backend,
        )
        ok = report["prerequisites_ok"]
    value = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value + "\n")
    print(value)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
