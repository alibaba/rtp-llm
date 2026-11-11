"""Run the existing reference protocol inside its selected PPU container.

Only command transport and receipt copying change. The reference driver owns
model lifecycle, the eight-device lease, HTTP load, timing and qualification.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from rtp_llm.test.perf_test import dsv41_ppu_reference as reference


def main():
    ssh_command = reference.command

    def local_command(node, action, config):
        if node.get("transport") != "local-container":
            raise ValueError("local reference entry requires local-container transport")
        remote = shlex.split(ssh_command(node, action, config)[-1])
        if remote[:6] != ["pouch", "exec", "-i", "--user", "0", node["container"]]:
            raise ValueError("unexpected reference command transport")
        if remote[6:8] != ["bash", "-c"] or len(remote) != 9:
            raise ValueError("unexpected reference container command")
        return remote[6:]

    def local_ready(process, config, timeout):
        node, role = config["node"], config["role"]
        port = node["address"].rsplit(":", 1)[1]
        owners = reference.result.WORLD if role == "decode" else 1
        probe = [node.get("python", "/opt/conda312/bin/python"), "-B", "-c",
                 reference.HEALTH_PROBE, port, str(owners)]
        if role == "decode":
            probe.append(config["output"])
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{role} launcher exited before readiness ({process.returncode})")
            try:
                if subprocess.run(probe, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=20).returncode == 0:
                    return
            except subprocess.TimeoutExpired:
                pass
            time.sleep(3)
        raise TimeoutError(f"{role} readiness timed out")

    def local_receipts(node, remote, destination):
        if node.get("transport") != "local-container":
            raise ValueError("cannot copy another node's receipts locally")
        if Path(remote).resolve() == Path(destination).resolve():
            raise ValueError("receipt destination must be separate from live output")
        shutil.copytree(remote, destination, ignore=shutil.ignore_patterns("token-map.json", "logs"))

    if os.getuid() != 0:
        raise ValueError("local entry must run as the explicitly selected container root")
    reference.command = local_command
    reference.wait_ready = local_ready
    reference.copy_receipts = local_receipts
    return reference.main()


if __name__ == "__main__":
    raise SystemExit(main())
