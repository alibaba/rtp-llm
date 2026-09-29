"""Real SDK and PACE contracts without importing the model/GPU smoke runner."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from rtp_llm.test.smoke.pace_fixture import require_binary_path, require_ok, runfile
from rtp_llm.test.smoke.remote_kvcm_server import RemoteKVCMServer


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def event_contract(server, instance_id, keys):
    host = f"127.0.0.1:{server._rpc_port}"
    base = {"instance_id": instance_id, "host_ip_port": host, "storage_type": 7}

    def report(event_type, params, check=True):
        field = {1: "node_register", 2: "block_add", 3: "block_delete", 6: "block_snapshot"}[event_type]
        return server.post_json("reportEvent", {
            **base, "trace_id": "pace-events-smoke",
            "events": [{"event_type": event_type, field: params}],
        }, check_status=check)

    registered = report(1, {"mediums": ["hbm"]})
    if not registered.get("snapshot_required", False):
        raise AssertionError("Fresh reporter must request a snapshot")
    specs = [{"name": "full", "uri": f"event_report://{host}/hbm"}]
    block = {"block_key": str(keys[0]), "medium": "hbm", "specs": specs}
    report(2, block)
    first = report(6, {"blocks": [block]})
    if not first.get("committed_snapshot_version"):
        raise AssertionError("Snapshot did not commit a reconciliation generation")
    throttled = report(6, {"blocks": [block]}, check=False)
    if int(throttled.get("retry_after_ms", 0)) <= 0:
        raise AssertionError("Repeated snapshot must return retry_after_ms")
    # Respect the server's retry hint before resending a complete snapshot.
    time.sleep(int(throttled["retry_after_ms"]) / 1000.0 + 0.05)
    require_ok(report(6, {"blocks": [block]}))
    request = {
        "instance_id": instance_id, "query_type": 2,
        "block_cache_keys": [str(keys[0])], "medium": ["hbm"], "p2p_host_count": 0,
    }
    hosts = server.post_json("getHostCacheState", request).get("hosts", [])
    matches = [item for item in hosts if item.get("host_ip_port") == host]
    if len(matches) != 1 or int(matches[0].get("local", 0)) != 1:
        raise AssertionError(f"ReportEvent host state did not become visible: {hosts}")
    report(3, {"block_key": str(keys[0]), "medium": "hbm", "spec_names": ["full"]})
    remaining = server.post_json("getHostCacheState", request).get("hosts", [])
    if any(item.get("host_ip_port") == host and int(item.get("local", 0)) > 0 for item in remaining):
        raise AssertionError("Deleted event block is still visible")


def publisher_contract(server, executable, output):
    instance_id = f"pace_publisher_{server._rpc_port}"
    host = f"127.0.0.1:{server._rpc_port}"
    first = time.time_ns()
    second = first + 1
    log_path = output / "pace_publisher.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([
            executable, f"http://127.0.0.1:{server._http_port}",
            server.pace_fixture.instance_group, instance_id, host, str(first), str(second),
        ], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            def wait_for(keys, count):
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise AssertionError(f"Publisher exited early; see {log_path}")
                    response = server.post_json("getHostCacheState", {
                        "instance_id": instance_id, "query_type": 2,
                        "block_cache_keys": [str(key) for key in keys],
                        "medium": ["hbm"], "p2p_host_count": 0,
                    }, check_status=False)
                    code = response.get("header", {}).get("status", {}).get("code")
                    if code in (1, "1", "OK"):
                        matches = [item for item in response.get("hosts", []) if item.get("host_ip_port") == host]
                        actual = int(matches[0].get("local", 0)) if matches else 0
                        if actual == count:
                            return
                    time.sleep(0.05)
                raise AssertionError(f"RTP Publisher host state did not reach {count}; see {log_path}")

            wait_for([first], 1)
            process.stdin.write("add\n")
            process.stdin.flush()
            wait_for([first, second], 2)
            process.stdin.write("delete\n")
            process.stdin.flush()
            wait_for([first], 0)
            wait_for([second], 1)
            process.stdin.write("stop\n")
            process.stdin.flush()
            process.stdin.close()
            if process.wait(timeout=15) != 0:
                raise AssertionError(f"RTP Publisher helper failed; see {log_path}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", required=True)
    parser.add_argument("--publisher", required=True)
    parser.add_argument("--backend", choices=("pace", "pace_ssd"), default="pace")
    args = parser.parse_args()
    require_binary_path(args.sdk)
    require_binary_path(args.publisher)
    server_path = runfile("remote_kv_cache_manager_server", "bin/kv_cache_manager_bin").parent.parent
    output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()))
    with tempfile.TemporaryDirectory(prefix="pace-smoke-", dir=os.environ.get("TEST_TMPDIR")) as directory:
        server = RemoteKVCMServer(str(server_path), {
            "PACE_REQUIRED": "true", "PACE_BACKEND": args.backend,
        }, str(Path(directory) / "logs"), str(output / "kvcm_logs"))
        try:
            if not server.start_server():
                raise RuntimeError("KVCM manager did not start")
            env = os.environ.copy()
            env.update(server.client_env())
            summaries = []
            for query_type in (1, 2, 3, 4):
                instance_id = f"pace_contract_{server._rpc_port}_{query_type}"
                config = server.pace_fixture.sdk_config(server, instance_id, query_type)
                config_path = Path(directory) / f"client_{query_type}.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                result_path = Path(directory) / f"result_{query_type}.json"
                subprocess.run([args.sdk, str(config_path), str(server.pace_fixture.storage_type),
                                str(result_path), str(query_type)], env=env, check=True, timeout=120)
                result = json.loads(result_path.read_text())
                assert_equal(result["byte_mismatches"], 0, "Byte comparison")
                info = server.post_json("getInstanceInfo", {"instance_id": instance_id})
                expected_query = {
                    1: "QT_BATCH_GET", 2: "QT_PREFIX_MATCH",
                    3: "QT_REVERSE_ROLL_SW_MATCH", 4: "QT_PREFIX_MATCH_WITH_MAMBA",
                }[query_type]
                assert_equal(info["instance_info"]["default_query_type"], expected_query, "Registered default query")
                summaries.append({"instance_id": instance_id, "default_query_type": query_type,
                                  "byte_mismatches": 0})
            # Event requests use an explicit query type, independent of the instance default.
            event_contract(server, instance_id, result["keys"])
            publisher_contract(server, args.publisher, output)
            (output / "pace_contract_result.json").write_text(json.dumps({
                "source_id": os.environ["KVCM_EXPECTED_SOURCE_ID"],
                "backend": args.backend, "cases": summaries,
                "event_protocol": "passed", "rtp_publisher": "passed",
                "artifact_sha256": {
                    repo: runfile(repo, "KVCM_ARTIFACT_SHA256").read_text().strip()
                    for repo in ("remote_kv_cache_manager_client_rpm", "remote_kv_cache_manager_server")
                },
            }, indent=2) + "\n", encoding="utf-8")
        finally:
            server.stop_server()
            server.copy_logs()


if __name__ == "__main__":
    main()
