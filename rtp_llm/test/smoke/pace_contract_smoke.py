"""Real SDK and PACE contracts without importing the model/GPU smoke runner."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from rtp_llm.test.smoke.pace_fixture import runfile
from rtp_llm.test.smoke.remote_kvcm_server import RemoteKVCMServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", required=True)
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
        finally:
            server.stop_server()
            server.copy_logs()


if __name__ == "__main__":
    main()
