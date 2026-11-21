"""Compare two complete, matched V4.1 model-side ARM CI artifacts; stdlib only."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args()
    # Support running this directly from any working directory in a source checkout.
    from importlib.util import module_from_spec, spec_from_file_location

    path = (
        Path(__file__).resolve().parents[4]
        / "rtp_llm/models_py/modules/v41_cache_model_trace.py"
    )
    spec = spec_from_file_location("v41_cache_model_trace_protocol", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    compare = module.compare

    try:
        report = compare(
            json.loads(args.old.read_text()), json.loads(args.new.read_text())
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(
            json.dumps({"status": "invalid_artifact", "error_type": type(exc).__name__})
        )
        return 2
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
