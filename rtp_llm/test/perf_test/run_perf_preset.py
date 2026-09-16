"""Run a mainline benchmark preset with ordinary Python and optional extra arguments."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    presets = json.loads((root / "perf_presets.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=sorted(presets))
    options, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    preset = presets[options.preset]
    env = dict(os.environ, **preset["env"])
    env["PERF_TEST_NAME"] = options.preset
    module = "rtp_llm.test.perf_test." + Path(preset["main"]).stem
    return subprocess.run(
        [sys.executable, "-m", module, *preset["args"], *extra], env=env, check=False
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
