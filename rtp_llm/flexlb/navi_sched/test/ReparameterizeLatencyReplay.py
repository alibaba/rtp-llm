#!/usr/bin/env python3
"""Convert replay anchors between equivalent latency-model coordinates."""

import argparse
import json
import tempfile
from pathlib import Path


MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_FILES = 64
PARAMETER_COUNT = 19


def convert(source, destination, old_scales, new_scales):
    paths = sorted(source.glob("*.jsonl"))
    if not paths or len(paths) > MAX_FILES:
        raise ValueError("unexpected replay file count")
    if destination.exists():
        raise FileExistsError(destination)
    if sum(path.stat().st_size for path in paths) > MAX_INPUT_BYTES:
        raise ValueError("replay exceeds input size limit")
    if (len(old_scales) != PARAMETER_COUNT or
            len(new_scales) != PARAMETER_COUNT or
            any(value <= 0 for value in old_scales + new_scales)):
        raise ValueError("invalid coordinate scales")
    old_to_new = [old / new for old, new in zip(old_scales, new_scales)]
    new_to_old = [new / old for old, new in zip(old_scales, new_scales)]
    with tempfile.TemporaryDirectory(prefix="latency-coordinates-",
                                     dir=destination.parent) as temporary:
        root = Path(temporary)
        for path in paths:
            with path.open() as source_stream, (root / path.name).open("w") as output:
                for line in source_stream:
                    event = json.loads(line)
                    if event["type"] == "update":
                        for suffix in ("before", "after"):
                            for prefix, factors in (
                                    ("parameters", old_to_new),
                                    ("adam_m1", new_to_old),
                                    ("adam_m2", [value * value for value in new_to_old])):
                                key = f"{prefix}_{suffix}"
                                if len(event[key]) != PARAMETER_COUNT:
                                    raise ValueError(f"invalid {key} in {path}")
                                event[key] = [value * factor for value, factor
                                              in zip(event[key], factors)]
                    output.write(json.dumps(event, separators=(",", ":")) + "\n")
        root.rename(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("standard", choices=("prediction", "loss", "log_abs",
                                             "mse_loss"))
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    convert(args.source, args.destination, calibration["current_scales"],
            calibration[f"{args.standard}_scales"])


if __name__ == "__main__":
    main()
