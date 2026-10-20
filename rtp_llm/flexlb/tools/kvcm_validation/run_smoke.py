"""Remote integration experiment using online_eval owners and real KVCM HTTP APIs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import socket
import time
import urllib.request

FLEXLB = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(FLEXLB / "tools/online_eval"), str(FLEXLB / "tools/online_eval/src")]
from adapters.environment import EnvManager
from adapters.environment_config import EnvSpec
from adapters.engine_ops import EngineOps
from runtime.process import ProcessOps
from runtime.network import http_post_json


def wait_for(probe, seconds=30):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        value = probe()
        if value:
            return value
        time.sleep(0.1)
    raise TimeoutError("condition did not converge")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kvcm-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port-base", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    transcript = []
    result = {"status": "ERROR", "requests": [], "checks": {}, "boundary": "CPU metadata events only; no actual KV payload transfer"}
    manager = EnvManager(args.out / "environment")
    ops = None
    server = None
    server_epoch = 0
    base = args.port_base
    os.environ.update(MOCK_KVCM_ENDPOINT=f"http://127.0.0.1:{base + 1}",
                      MOCK_KVCM_INSTANCE_GROUP="mock_events", MOCK_KVCM_INSTANCE_PREFIX="integration",
                      MOCK_KVCM_STORAGE_TYPE="ST_EVENT_REPORT_L1P5")

    def api(port, route, body):
        request = urllib.request.Request(f"http://127.0.0.1:{port}/api/{route}",
                                         json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.loads(response.read())
        transcript.append({"epoch": time.time(), "route": route, "request": body, "response": data})
        code = data.get("header", {}).get("status", {}).get("code")
        if code not in ("OK", "1", 1):
            raise RuntimeError(f"{route}: {data}")
        return data

    def start_server():
        nonlocal server, server_epoch
        server_epoch += 1
        for port in range(base, base + 4):
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
        root = args.kvcm_dir.resolve()
        argv = [str(root / "bin/kv_cache_manager_bin"), "-c", str(root / "etc/default_server_config.conf"),
                "-l", str(root / "etc/default_logger_config.conf")]
        for key, value in {"rpc_port": base, "http_port": base + 1, "admin_rpc_port": base + 2,
                           "admin_http_port": base + 3, "io_thread_num": 4}.items():
            argv.extend(["--env", f"kvcm.service.{key}={value}"])
        server = ProcessOps.start(argv, dict(os.environ), args.out / f"kvcm-{server_epoch}.log", cwd=root)
        def healthy():
            if not server.alive():
                raise RuntimeError(server.tail_log())
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{base + 3}/api/healthy", timeout=1) as r:
                    return r.status == 200
            except OSError:
                return False
        wait_for(healthy)
        api(base + 3, "addStorage", {"trace_id": "setup", "storage": {"global_unique_name": "event_meta",
            "storage_type": "ST_EVENT_REPORT_L1P5", "event_report": {"heartbeat_timeout_ms": 30000,
            "cleanup_grace_ms": 1000, "liveness_check_interval_ms": 500, "snapshot_min_interval_ms": 1}}})
        api(base + 3, "addStorage", {"trace_id": "setup", "storage": {"global_unique_name": "isolated_nfs",
            "nfs": {"root_path": str(args.out.resolve() / "blocks"), "key_count_per_file": 1},
            "check_storage_available_when_open": True}})
        group = json.loads((root / "etc/default_startup_config.json").read_text())["instance_group"]
        group.update(name="mock_events", storage_candidates=["isolated_nfs"], event_report_storage_candidates=["event_meta"])
        group["cache_config"].pop("cache_prefer_strategy", None)
        group["cache_config"]["data_storage_strategy"] = "CPS_PREFER_3FS"
        group["cache_config"]["reclaim_strategy"]["storage_unique_name"] = "isolated_nfs"
        for item in group["quota"]["quota_config"]:
            item["storage_type"] = {"file": "ST_NFS", "hf3fs": "ST_3FS", "pace": "ST_TAIR_MEMPOOL"}[item["storage_type"]]
        api(base + 3, "createInstanceGroup", {"trace_id": "setup", "instance_group": group})

    def engines():
        return ops.snapshot()["engines"]

    def indexed(engine, keys):
        response = api(base + 1, "getCacheLocation", {"trace_id": "verify", "instance_id": engine["kvcm"]["instance_id"],
            "query_type": "QT_BATCH_GET", "block_keys": [str(k) for k in keys], "location_spec_names": ["mock_hbm"]})
        return [any(s.get("uri") for s in location.get("location_specs", []))
                for location in response.get("locations", [])]

    def fire(rid, keys):
        response = ops.schedule(rid, input_len=256, output_len=2, block_keys=keys)
        stream = ops.start_stream(response, rid, ops.build_generate_input(rid, input_len=256, output_len=2, block_keys=keys))
        if not stream.wait_end(20) or not stream.snap.completed or stream.snap.error or stream.snap.stream_error_code:
            raise RuntimeError(f"request {rid} failed: {stream.snap}")
        result["requests"].append({"request_id": rid, "completed": True, "keys": keys})

    try:
        start_server()
        spec = EnvSpec(label="kvcm-integration", n_prefill=1, n_decode=1, master_profile="single-nonbatch",
                       prefill_cache_blocks=64, decode_cache_blocks=64, master_jvm_heap="1g", mock_heap="1g")
        env = manager.ensure(spec)
        ops = EngineOps("127.0.0.1", env.master_http_port, env.mock_http_port, env.master_management_port)
        wait_for(lambda: all(e.get("kvcm", {}).get("snapshots", 0) >= 1 for e in engines()))
        result["checks"]["registered_and_snapshot"] = True
        fire(100001, [101, 102])
        wait_for(lambda: all(e["cache_key_set"] for e in engines()))
        before = engines()
        (args.out / "after-request.json").write_text(json.dumps(before, indent=2))
        for engine in before:
            wait_for(lambda e=engine: all(indexed(e, e["cache_key_set"])))
        result["checks"]["request_keys_in_kvcm"] = True
        for engine in before:
            keys = engine["cache_key_set"]
            status, body = http_post_json(f"http://127.0.0.1:{env.mock_http_port}/cache_evict", {"engine": engine["name"], "keys": keys})
            assert status == 200, body
            wait_for(lambda e=engine, ks=keys: not any(indexed(e, ks)))
        result["checks"]["evicted_keys_absent"] = True
        server.terminate()
        wait_for(lambda: any(e["kvcm"]["failures"] > 0 for e in engines()), seconds=15)
        fire(100002, [201, 202])
        result["checks"]["request_completes_without_kvcm"] = True
        start_server()
        wait_for(lambda: all(e["kvcm"]["snapshots"] >= 2 and not e["kvcm"]["last_error"] for e in engines()))
        recovered = engines()
        if not all(e["cache_key_set"] and e["kvcm"]["adds"] >= 2 and e["kvcm"]["deletes"] >= 2
                   and e["kvcm"]["heartbeats"] >= 1 for e in recovered):
            raise AssertionError("missing mutation, heartbeat or recovered key evidence")
        for engine in recovered:
            wait_for(lambda e=engine: all(indexed(e, e["cache_key_set"])))
        result["checks"]["restart_snapshot_rebuild"] = True
        result["engines"] = engines()
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = repr(exc)
        raise
    finally:
        try:
            manager.teardown()
            result["checks"]["mock_master_cleanup"] = True
            if result["status"] == "PASS":
                for engine in result["engines"]:
                    wait_for(lambda e=engine: not any(indexed(e, e["cache_key_set"])))
                result["checks"]["host_down_removes_locations"] = True
        except Exception as exc:
            result["status"] = "ERROR"
            result["cleanup_error"] = repr(exc)
            raise
        finally:
            if ops is not None:
                ops.close()
            if server is not None:
                server.terminate()
            result["kvcm_sha256"] = hashlib.sha256((args.kvcm_dir / "bin/kv_cache_manager_bin").read_bytes()).hexdigest()
            (args.out / "protocol.json").write_text(json.dumps(transcript, indent=2))
            (args.out / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
