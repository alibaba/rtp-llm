"""Real SDK and PACE contracts without importing the model/GPU smoke runner."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from rtp_llm.test.smoke.pace_fixture import runfile
from rtp_llm.test.smoke.remote_kvcm_server import RemoteKVCMServer


def publisher_contract(server, executable, output):
    instance_id = f"pace_publisher_{server.rpc_port}"
    host = f"127.0.0.1:{server.rpc_port}"
    first = time.time_ns()
    second = first + 1
    log_path = output / "pace_publisher.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([
            executable, f"http://127.0.0.1:{server.http_port}",
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
    args = parser.parse_args()
    server_path = runfile(
        os.environ["TEST_WORKSPACE"],
        "3rdparty/remote_kv_cache_manager/bin/kv_cache_manager_bin",
    ).parent.parent
    output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()))
    with tempfile.TemporaryDirectory(prefix="pace-smoke-", dir=os.environ.get("TEST_TMPDIR")) as directory:
        server = RemoteKVCMServer(str(server_path), {
            "PACE_REQUIRED": "true",
        }, str(Path(directory) / "logs"), str(output / "kvcm_logs"))
        try:
            if not server.start_server():
                raise RuntimeError("KVCM manager did not start")
            env = os.environ.copy()
            env.update(server.client_env())
            instance_id = f"pace_contract_{server.rpc_port}"
            config = server.pace_fixture.sdk_config(server, instance_id)
            config_path = Path(directory) / "client.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            subprocess.run([args.sdk, str(config_path), str(server.pace_fixture.storage_type)],
                           env=env, check=True, timeout=120)
            publisher_contract(server, args.publisher, output)
        finally:
            server.stop_server()
            server.copy_logs()


if __name__ == "__main__":
    main()
