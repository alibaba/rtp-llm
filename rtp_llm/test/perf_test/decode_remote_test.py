"""Run a complete local decode workload inside one ordinary Bazel test action."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from rtp_llm.test.perf_test import entry


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        config_path = Path(args.config).absolute()
        source = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise ValueError("config must contain an object")
        execution = source.get("execution")
        if not isinstance(execution, dict) or execution.get("backend") != "local":
            raise ValueError("ordinary Bazel decode requires execution.backend=local")
        output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if not output_dir:
            raise ValueError(
                "TEST_UNDECLARED_OUTPUTS_DIR must name a writable directory"
            )
        root = Path(output_dir).resolve()
        output = source.setdefault("output", {})
        if not isinstance(output, dict):
            raise ValueError("output must contain an object")
        output["result_root"] = str(root)
        # The writable config copy has a different parent than its runfile.
        # Keep external distribution inputs anchored to the declared config.
        workload = source.get("workload")
        if isinstance(workload, dict) and "distribution_file" in workload:
            distribution = Path(workload["distribution_file"])
            if not distribution.is_absolute():
                workload["distribution_file"] = str(config_path.parent / distribution)
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="decode-config-", dir=root
        ) as directory:
            prepared = Path(directory) / "config.json"
            prepared.write_text(json.dumps(source), encoding="utf-8")
            return entry.main(
                [
                    "--mode=decode",
                    "--action=run",
                    f"--config={prepared}",
                ]
            )
    except (OSError, ValueError, TypeError) as error:
        sys.stderr.write(f"decode_remote_test: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
