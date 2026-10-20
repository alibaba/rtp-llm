"""Bounded KVCM metadata capacity experiment; deliberately excludes inference traffic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import threading
import time
import urllib.request

FLEXLB = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(FLEXLB / "tools/online_eval"), str(FLEXLB / "tools/online_eval/src")]
from runtime.process import ProcessOps

KEY_BASE = 1_000_000_000_000_000_000
KEY_STRIDE = 10_000_000_000


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def memory(pid):
    fields = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "RssAnon:", "VmSize:", "VmHWM:")):
            name, value, _ = line.split()
            fields[name[:-1]] = int(value) * 1024
    return fields


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kvcm-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port-base", type=int, required=True)
    parser.add_argument("--reporters", type=int, default=32)
    parser.add_argument("--backend-capacity-mib", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--memory-budget-gib", type=int, default=512)
    parser.add_argument("--rss-stop-gib", type=int, default=450)
    parser.add_argument("--checkpoints", default="1000000,10000000,50000000,100000000,200000000,500000000,1000000000,2000000000")
    parser.add_argument("--metric-settle-seconds", type=int, default=60)
    parser.add_argument("--max-seconds", type=int, default=5400)
    args = parser.parse_args()
    checkpoints = [int(x) for x in args.checkpoints.split(",")]
    if not 1 <= args.reporters <= 254 or any(x % args.reporters for x in checkpoints):
        raise ValueError("checkpoints must be divisible by reporters")
    args.out.mkdir(parents=True, exist_ok=False)
    root = args.kvcm_dir.resolve()
    args.out = args.out.resolve()
    base = args.port_base
    endpoint = f"http://127.0.0.1:{base + 1}"
    instance = "capacity_model"
    server = loader = None
    stop = threading.Event()
    errors = []
    began = time.monotonic()
    result = {"status": "RUNNING", "target_keys": checkpoints[-1], "stages": [],
              "reporters": args.reporters, "positions_per_key": 1,
              "scope": "real KVCM; protocol-level synthetic reporters; no Master or inference JVM", "parameters": vars(args).copy()}
    result["parameters"] = {k: str(v) if isinstance(v, Path) else v for k, v in result["parameters"].items()}

    def api(port, route, body, timeout=30):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/{route}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            value = json.loads(response.read())
        if value.get("header", {}).get("status", {}).get("code") not in ("OK", "1", 1):
            raise RuntimeError(f"{route}: {value}")
        return value

    def host(i):
        return f"127.2.0.{i + 1}:{30000 + i}"

    def report(i, event):
        response = api(base + 1, "reportEvent", {"trace_id": f"capacity-control-{i}", "instance_id": instance,
            "host_ip_port": host(i), "storage_type": "ST_EVENT_REPORT_L1P5", "events": [event]})
        if response.get("item_results") and any(x not in ("OK", "1", 1) for x in response["item_results"]):
            raise RuntimeError(f"item failure: {response}")
        return response

    def heartbeat():
        while not stop.wait(5):
            try:
                for i in range(args.reporters):
                    report(i, {"event_type": "EVENT_HEARTBEAT", "heartbeat": {"system_status": {"engine": "rtp-llm"}}})
            except Exception as exc:
                errors.append(repr(exc))
                stop.set()

    def measure():
        if not server.alive():
            raise RuntimeError("KVCM exited: " + server.tail_log())
        value = dict(epoch=time.time(), **memory(server.pid), host_available_bytes=available_memory())
        if value["VmRSS"] > args.rss_stop_gib * 2**30:
            raise MemoryError("RSS resource budget reached")
        if value["host_available_bytes"] < 128 * 2**30:
            raise MemoryError("host MemAvailable reserve reached")
        if shutil.disk_usage(args.out).free < 20 * 2**30:
            raise MemoryError("disk reserve reached")
        if time.monotonic() - began > args.max_seconds:
            raise TimeoutError("experiment time budget reached")
        if errors:
            raise RuntimeError("heartbeat failure: " + errors[0])
        return value

    def query(per_reporter):
        rng = random.Random(per_reporter)
        keys = []
        for i in range(args.reporters):
            offsets = {0, per_reporter - 1, per_reporter // 2}
            offsets.update(rng.randrange(per_reporter) for _ in range(29))
            keys.extend(KEY_BASE + i * KEY_STRIDE + off for off in sorted(offsets))
        start = time.monotonic()
        response = api(base + 1, "getCacheLocation", {"trace_id": "capacity-sample", "instance_id": instance,
            "query_type": "QT_BATCH_GET", "block_keys": [str(x) for x in keys], "location_spec_names": ["mock_hbm"]})
        locations = response.get("locations", [])
        hits = sum(any(s.get("uri", "").split("?", 1)[0] == "rtp-llm://" + host((key - KEY_BASE) // KEY_STRIDE) + "/hbm"
                       for s in item.get("location_specs", [])) for key, item in zip(keys, locations))
        if len(locations) != len(keys) or hits != len(keys):
            save(args.out / "failed-query.json", response)
            raise AssertionError(f"sample query {hits}/{len(keys)}")
        return {"keys": len(keys), "hits": hits, "latency_s": time.monotonic() - start}

    try:
        cpus = sorted(os.sched_getaffinity(0))[:args.cpus]
        server_cpus = cpus[:max(1, len(cpus) * 3 // 4)]
        client_cpus = cpus[len(server_cpus):] or cpus
        result["cpu_affinity"] = {"server": server_cpus, "client": client_cpus}
        for port in range(base, base + 4):
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
        logger = (root / "etc/default_logger_config.conf").read_text()
        logger = logger.replace("alog.rootLogger=INFO", "alog.rootLogger=WARN").replace("alog.logger.access=INFO", "alog.logger.access=WARN")
        (args.out / "logger.conf").write_text(logger)
        argv = ["taskset", "-c", ",".join(map(str, server_cpus)), "prlimit", f"--as={args.memory_budget_gib * 2**30}",
                str(root / "bin/kv_cache_manager_bin"), "-c", str(root / "etc/default_server_config.conf"), "-l", str(args.out / "logger.conf")]
        for name, value in {"rpc_port": base, "http_port": base + 1, "admin_rpc_port": base + 2,
                            "admin_http_port": base + 3, "io_thread_num": 16}.items():
            argv.extend(["--env", f"kvcm.service.{name}={value}"])
        argv.extend(["--env", "kvcm.metrics.report_interval_ms=1000"])
        server = ProcessOps.start(argv, dict(os.environ), args.out / "server.log", cwd=root)
        save(args.out / "server-identity.json", {"pid": server.pid, "argv": argv,
            "binary_sha256": hashlib.sha256((root / "bin/kv_cache_manager_bin").read_bytes()).hexdigest()})
        for _ in range(100):
            if not server.alive():
                raise RuntimeError(server.tail_log())
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{base + 3}/api/healthy", timeout=1):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            raise TimeoutError("KVCM startup")
        api(base + 3, "addStorage", {"trace_id": "capacity", "storage": {"global_unique_name": "capacity_events",
            "storage_type": "ST_EVENT_REPORT_L1P5", "event_report": {"heartbeat_timeout_ms": 3600000,
            "cleanup_grace_ms": 1000, "liveness_check_interval_ms": 1000, "snapshot_min_interval_ms": 1}}})
        api(base + 3, "addStorage", {"trace_id": "capacity", "storage": {"global_unique_name": "capacity_nfs",
            "nfs": {"root_path": str(args.out / "blocks"), "key_count_per_file": 1}, "check_storage_available_when_open": True}})
        group = json.loads((root / "etc/default_startup_config.json").read_text())["instance_group"]
        group.update(name="capacity", storage_candidates=["capacity_nfs"], event_report_storage_candidates=["capacity_events"])
        group["quota"]["capacity"] = 100_000_000_000_000
        group["quota"]["quota_config"] = [{"storage_type": "ST_NFS", "capacity": 100_000_000_000_000}]
        group["cache_config"].pop("cache_prefer_strategy", None)
        group["cache_config"].update(data_storage_strategy="CPS_PREFER_3FS")
        group["cache_config"]["reclaim_strategy"]["storage_unique_name"] = "capacity_nfs"
        if args.backend_capacity_mib:
            group["cache_config"]["meta_indexer_config"]["meta_storage_backend_config"]["storage_uri"] = (
                f"local://capacity?capacity={args.backend_capacity_mib}&num_shard_bits=10&sample_times=10")
        group["cache_config"]["meta_indexer_config"]["max_key_count"] = 2_100_000_000
        group["cache_config"]["meta_indexer_config"]["mutex_shard_num"] = 1024
        save(args.out / "instance-group.json", group)
        api(base + 3, "createInstanceGroup", {"trace_id": "capacity", "instance_group": group})
        api(base + 1, "registerInstance", {"trace_id": "capacity", "instance_group": "capacity", "instance_id": instance,
            "block_size": 1024, "model_deployment": {"model_name": "capacity-model", "dtype": "BF16", "use_mla": False,
                "tp_size": 1, "dp_size": 1, "pp_size": 1}, "location_spec_infos": [{"name": "mock_hbm", "size": 4096}],
            "location_spec_groups": [{"name": "default", "spec_names": ["mock_hbm"]}]})
        save(args.out / "effective-instance-group.json", api(base + 3, "getInstanceGroup", {"trace_id": "capacity", "name": "capacity"}))
        for i in range(args.reporters):
            report(i, {"event_type": "EVENT_NODE_REGISTER", "node_register": {"mediums": ["hbm"]}})
            response = report(i, {"event_type": "EVENT_BLOCK_SNAPSHOT", "block_snapshot": {"blocks": []}})
            if response.get("snapshot_required"):
                raise AssertionError("initial snapshot rejected")
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        baseline = measure()
        result["baseline"] = baseline
        previous = 0
        java = str(Path(os.environ["JAVA_HOME"]) / "bin/java")
        for target in checkpoints:
            if result["stages"]:
                last = result["stages"][-1]
                bpk = (last["memory"]["VmRSS"] - baseline["VmRSS"]) / last["resident_keys"]
                if baseline["VmRSS"] + bpk * target > args.rss_stop_gib * 2**30:
                    result.update(status="RESOURCE_BOUND", stop_reason="next checkpoint projected beyond RSS budget")
                    break
            prefix = args.out / f"load-{target}"
            command = ["taskset", "-c", ",".join(map(str, client_cpus)), java, "-Xmx2g", "-XX:ActiveProcessorCount=8",
                "-cp", str(Path(__file__).parent), "KvcmCapacityLoad", endpoint, instance, str(args.reporters),
                str(previous // args.reporters), str(target // args.reporters), str(args.batch), str(prefix.with_suffix(".json")), str(args.concurrency)]
            loader = ProcessOps.start(command, dict(os.environ), prefix.with_suffix(".log"))
            while loader.alive():
                sample = measure()
                progress_file = prefix.with_suffix(".json")
                progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
                sample.update(stage_target=target, prior_keys=previous, loader=progress)
                with (args.out / "memory.jsonl").open("a") as output:
                    output.write(json.dumps(sample) + "\n")
                save(args.out / "progress.json", sample)
                time.sleep(2)
            if loader.proc.returncode:
                raise RuntimeError("loader failed: " + loader.tail_log())
            progress = json.loads(prefix.with_suffix(".json").read_text())
            if progress["accepted_keys"] != target - previous:
                raise AssertionError("incomplete acknowledged count")
            time.sleep(3)
            stage = {"resident_keys": target, "positions": target, "loader": progress, "memory": measure(),
                     "sample_query": query(target // args.reporters)}
            metric_deadline = time.monotonic() + args.metric_settle_seconds
            observations = []
            while True:
                metrics = api(base + 3, "getMetrics", {"trace_id": "capacity"})
                native = [m for m in metrics.get("metrics", []) if m["metric_name"] == "cache_manager_instance.key_count"
                          and any(t["tag_key"] == "instance_id" and t["tag_value"] == instance for t in m.get("metric_tags", []))]
                count = native[0]["metric_value"].get("float_value") if len(native) == 1 else None
                observations.append({"epoch": time.time(), "keys": count})
                save(args.out / f"metric-settle-{target}.json", observations)
                if count == target:
                    break
                if time.monotonic() >= metric_deadline:
                    save(args.out / f"metrics-{target}.json", metrics)
                    raise AssertionError(f"server resident count differs from target after settling: {native}")
                measure()
                time.sleep(2)
            save(args.out / f"metrics-{target}.json", metrics)
            stage["metric_settle_observations"] = observations
            stage["server_key_count"] = native[0]["metric_value"]["float_value"]
            stage["metrics_file"] = f"metrics-{target}.json"
            stage["incremental_bytes_per_key"] = (stage["memory"]["VmRSS"] - baseline["VmRSS"]) / target
            result["stages"].append(stage)
            previous = target
            save(args.out / "result.json", result)
            print(json.dumps(stage), flush=True)
        else:
            result["status"] = "TARGET_REACHED"
        if result["stages"]:
            last = result["stages"][-1]
            result["projection_2b_rss_bytes"] = baseline["VmRSS"] + last["incremental_bytes_per_key"] * 2_000_000_000
    except (MemoryError, TimeoutError) as exc:
        result.update(status="RESOURCE_BOUND", error=repr(exc))
    except Exception as exc:
        result.update(status="ERROR", error=repr(exc))
        raise
    finally:
        stop.set()
        if loader is not None and loader.alive():
            loader.terminate()
        if server is not None:
            if server.alive():
                try:
                    save(args.out / "metrics-final.json", api(base + 3, "getMetrics", {"trace_id": "capacity-final"}))
                except Exception as exc:
                    result["final_metrics_error"] = repr(exc)
            result["final_memory"] = memory(server.pid) if server.alive() else None
            server.terminate(timeout_s=30)
        result["elapsed_s"] = time.monotonic() - began
        save(args.out / "result.json", result)


if __name__ == "__main__":
    main()
