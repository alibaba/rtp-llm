"""Freeze unified random prefill workloads as explicit request batches."""

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from rtp_llm.test.perf_test.config_schema import PrefillWorkloadConfig, parse_config
from rtp_llm.test.perf_test.errors import ConfigurationError, PerfTestError
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases
from rtp_llm.test.perf_test.prefill.random_batch import sample_random_request_batches


def _freeze_workload(raw, workload: PrefillWorkloadConfig, engine) -> bool:
    if workload.grids:
        changed = False
        for item, grid in zip(raw["grids"], workload.grids):
            changed = _freeze_workload(item, grid.workload, engine) or changed
        return changed
    if workload.random_request_batches is None:
        return False
    batches = sample_random_request_batches(workload, engine)
    raw.pop("random_request_batches")
    raw["request_batches"] = [
        {"request_groups": [asdict(group) for group in batch.request_groups]}
        for batch in batches
    ]
    # Explicit groups otherwise default to seed=1, changing grouped case identity.
    raw["seed"] = workload.seed
    return True


def _freeze_config(source):
    frozen = copy.deepcopy(source)
    config = parse_config(source)
    if not isinstance(config.workload, PrefillWorkloadConfig) or not _freeze_workload(
        frozen["workload"], config.workload, config.engine
    ):
        raise ConfigurationError("export requires a random_request_batches workload")
    parsed = parse_config(frozen)
    # The ordinary grouped planner remains the final admission authority.
    plan_prefill_cases(parsed.workload, parsed.engine, dp_size=parsed.model.topology.dp)
    return frozen


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    destinations = parser.add_mutually_exclusive_group(required=True)
    destinations.add_argument("--output", type=Path)
    destinations.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        source = json.loads(args.config.read_text(encoding="utf-8"))
        if args.output is not None:
            outputs = [(args.output, _freeze_config(source))]
        else:
            config = parse_config(source)
            workload = config.workload
            if (
                not isinstance(workload, PrefillWorkloadConfig)
                or workload.grids
                or workload.random_request_batches is None
            ):
                raise ConfigurationError(
                    "--output-dir requires a single top-level random_request_batches workload"
                )
            outputs = []
            for size in workload.random_request_batches.batch_sizes:
                per_batch = copy.deepcopy(source)
                random = per_batch["workload"]["random_request_batches"]
                random["batch_sizes"] = [size]
                if "batch_input_limits" in random:
                    random["batch_input_limits"] = {
                        key: value
                        for key, value in random["batch_input_limits"].items()
                        if key == str(size)
                    }
                per_batch["workload"]["seed"] = workload.seed + size
                outputs.append(
                    (args.output_dir / f"batch_{size}.json", _freeze_config(per_batch))
                )
        # Preflight the complete set before creating directories or writing a file.
        for path, _ in outputs:
            if os.path.lexists(path):
                raise FileExistsError(
                    f"refusing to overwrite existing destination: {path}"
                )
        for path, frozen in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                json.dump(frozen, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
    except (PerfTestError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
