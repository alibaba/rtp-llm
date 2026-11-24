"""Export generated prefill requests into explicit request batches."""

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.config_schema import PrefillWorkloadConfig, parse_config
from rtp_llm.test.perf_test.errors import ConfigurationError, PerfTestError
from rtp_llm.test.perf_test.prefill.logarithmic_grid import (
    generate_logarithmic_request_batches,
)
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases
from rtp_llm.test.perf_test.prefill.random_batch import sample_random_request_batches
from rtp_llm.test.perf_test.prefill.traffic_batch import materialize_traffic_source


def _freeze_workload(raw, workload: PrefillWorkloadConfig, engine) -> bool:
    if workload.grids:
        changed = False
        for item, grid in zip(raw["grids"], workload.grids):
            changed = _freeze_workload(item, grid.workload, engine) or changed
        return changed
    if workload.logarithmic_request_batches is not None:
        batches = generate_logarithmic_request_batches(workload, engine)
        raw.pop("logarithmic_request_batches")
    elif workload.random_request_batches is not None:
        batches = sample_random_request_batches(workload, engine)
        raw.pop("random_request_batches")
    else:
        return False
    raw["request_batches"] = [
        {"request_groups": [asdict(group) for group in batch.request_groups]}
        for batch in batches
    ]
    # Explicit groups otherwise default to seed=1, changing grouped case identity.
    raw["seed"] = workload.seed
    return True


def _freeze_config(source, *, source_dir):
    config = parse_config(source)
    if not isinstance(config.workload, PrefillWorkloadConfig):
        raise ConfigurationError("export requires a prefill workload")
    frozen = copy.deepcopy(source)
    changed = _freeze_workload(frozen["workload"], config.workload, config.engine)
    raw = source["workload"]
    if any("traffic_request_batches" in item for item in [raw, *raw.get("grids", ())]):
        resolved = resolve_config(source, source_dir=source_dir)
        frozen = materialize_traffic_source(
            frozen, resolved.resolved_config["workload"]
        )
        changed = True
    if not changed:
        raise ConfigurationError(
            "export requires random_request_batches, traffic_request_batches "
            "or logarithmic_request_batches"
        )
    parsed = parse_config(frozen)
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
            outputs = [
                (args.output, _freeze_config(source, source_dir=args.config.parent))
            ]
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
                    (
                        args.output_dir / f"batch_{size}.json",
                        _freeze_config(per_batch, source_dir=args.config.parent),
                    )
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
