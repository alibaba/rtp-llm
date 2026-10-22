"""Run the packaged Master with its native VipServer adapter and lifecycle hooks."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess

from master import online


def command(env):
    if env.get("RTP_LLM_MOCK_BUNDLE") != "1":
        raise ValueError("RTP_LLM_MOCK_BUNDLE=1 is required")
    if "discovery_file" in json.loads(env["MODEL_SERVICE_CONFIG"]):
        raise ValueError("independent Master must discover actual VIP members")
    root = Path(env.get("MOCK_BUNDLE_ROOT", "/opt/flexlb-mock"))
    manifest = json.loads((root / "build-manifest.json").read_text())
    if not manifest["components"]["master"].get("vipserver_packaged"):
        raise ValueError("Master image lacks verified VipServer adapter")
    jar = Path(env.get("MOCK_BUNDLE_JAR_DIR", str(root / "tools/mock_bundle/jars"))) / "master.jar"
    port = int(env.get("START_PORT", "28080"))
    java = str(Path(env.get("JAVA_HOME", "/opt/taobao/java")) / "bin/java")
    return [java, *shlex.split(env.get("MOCK_MASTER_JAVA_OPTS", "-Xms512m -Xmx4g")),
            "-jar", str(jar), "--server.port=" + str(port),
            "--management.server.port=" + str(port + 1)]


def main():
    child = subprocess.Popen(command(os.environ))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: child.send_signal(signum))
    try:
        online(child, int(os.environ.get("START_PORT", "28080")))
        raise SystemExit(child.wait())
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
