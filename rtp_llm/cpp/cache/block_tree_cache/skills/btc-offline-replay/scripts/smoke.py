"""Check the native backend contract using a tiny synthetic trace and an in-process pause boundary."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from tempfile import TemporaryDirectory

SCRIPTS = Path(__file__).resolve().parent


def call(script, *args):
    return subprocess.check_output([sys.executable, str(SCRIPTS / script)] + [str(a) for a in args])


def smoke(backend, base=256, per_file=1000000):
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "sample.jsonl"
        with source.open("w") as f:
            for i, length in enumerate([2176, 1152, 2176, 3200, 1152, 2176]):
                hashes = [f"{j+1:016x}" for j in range(length // base)]
                row = dict(block_tokens=base, request_id=str(i), timestamp=i+1, input_length=length, input_block_hash_ids=hashes)
                f.write(json.dumps(row) + "\n")
        data = root / "data"
        call("prepare.py", source, "--format", "bailian", "--hash-mode", "raw", "--requests-per-file", per_file,
             "--domain", "synthetic", "--out", data)
        config = root / "config.json"
        config.write_text(json.dumps({"tiers": {k: {"full": 256, "swa": 256} for k in ("device", "host", "disk")}}))
        args = ["--backend", backend, "--config", config, "--data", data, "--warmup", 2,
                "--measure", 4, "--revision", "synthetic-contract-test"]
        resumed, single = (root / k for k in ("paused", "single"))
        cmd = [sys.executable, str(SCRIPTS / "replay.py"), *map(str, args),
               "--pause-after", "2", "--out", str(resumed)]
        process = subprocess.Popen(cmd)
        try:
            deadline = time.monotonic() + 60
            while not (resumed / "native/PAUSED").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise ValueError("backend failed to pause")
                time.sleep(.05)
            if (resumed / "native/match-lengths.u64").stat().st_size != 32:
                raise ValueError("pause cursor differs")
            (resumed / "native/CONTINUE").touch()
            if process.wait(timeout=60):
                raise ValueError("continued replay failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        call("replay.py", *args, "--out", single)
        if (resumed / "native/match-lengths.u64").read_bytes() != (single / "native/match-lengths.u64").read_bytes():
            raise ValueError("pause changed per-request matching")
        if (resumed / "native/checkpoint").exists():
            raise ValueError("unexpected checkpoint")
        a, b = json.loads(call("analyze.py", single, resumed))
        if not (a["joint_rate"] == b["joint_rate"] > 0 and a["attainment"] == b["attainment"] == 1):
            raise ValueError("unexpected hit rate or comparison")
        print(f"PASS: {a['requests']} measured requests, {a['input_tokens']} tokens, joint rate {a['joint_rate']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", type=Path)
    backend = parser.parse_args().backend.resolve(strict=True)
    smoke(backend)
    probe = subprocess.run([str(backend), "--btcpfx-reader-version"], capture_output=True, text=True)
    if probe.returncode == 0 and probe.stdout.strip() == "2":
        # Four processed records at the pause: exact boundary and inside file two.
        smoke(backend, base=128, per_file=4)
        smoke(backend, base=256, per_file=3)
