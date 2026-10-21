"""Run the packaged Master against engines discovered outside this Pod."""
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.request

from standalone_mock import integer, json_object


ROOT = Path(__file__).resolve().parent


def run():
    env = os.environ.copy()
    if env.get("RTP_LLM_MOCK_BUNDLE") != "1":
        raise ValueError("RTP_LLM_MOCK_BUNDLE=1 is required for test-only Master")
    json_object(env["MODEL_SERVICE_CONFIG"], "MODEL_SERVICE_CONFIG")
    port = integer(env, "START_PORT", 7001, maximum=65533)
    runtime = Path(env.get("MOCK_BUNDLE_RUN_DIR", "/home/admin/ai-whale/master"))
    runtime.mkdir(parents=True, exist_ok=True)
    java = str(Path(env.get("JAVA_HOME", "/opt/taobao/java")) / "bin/java")
    jars = Path(env.get("MOCK_BUNDLE_JAR_DIR", str(ROOT / "jars")))
    command = [java, *shlex.split(env.get("MOCK_MASTER_JAVA_OPTS", "-Xms512m -Xmx2g")),
               "-Dreactor.schedulers.defaultBoundedElasticSize=64", "-jar", str(jars / "master.jar"),
               "--server.port=" + str(port), "--management.server.port=" + str(port + 1),
               "--flexlb.log.path=" + str(runtime / "master-logs")]
    child = subprocess.Popen(command, env=env)
    stopping = False

    def forward(signum, _frame):
        nonlocal stopping
        stopping = True
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        deadline = time.monotonic() + integer(env, "MOCK_MASTER_STARTUP_TIMEOUT_S", 120)
        while not stopping:
            if child.poll() is not None:
                raise RuntimeError("Master exited before application readiness")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/hook/process_ok", timeout=2
                ) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Master application readiness deadline exceeded")
            time.sleep(0.2)
        if stopping:
            return
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/hook/after_start", timeout=120
        ) as response:
            response.read()
        raise SystemExit(child.wait())
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    run()
