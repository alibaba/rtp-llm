"""Freeze traffic-sampled prefill workloads as explicit request batches."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.config_schema import PrefillWorkloadConfig, parse_config
from rtp_llm.test.perf_test.errors import ConfigurationError, PerfTestError
from rtp_llm.test.perf_test.prefill.planner import plan_prefill_cases
from rtp_llm.test.perf_test.prefill.traffic_batch import materialize_traffic_source


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        source = json.loads(args.config.read_text(encoding="utf-8"))
        resolved = load_resolved_config(args.config)
        frozen = materialize_traffic_source(
            source, resolved.resolved_config["workload"]
        )
        if frozen == source:
            raise ConfigurationError(
                "export requires a traffic_request_batches workload"
            )
        config = parse_config(frozen)
        if not isinstance(config.workload, PrefillWorkloadConfig):
            raise ConfigurationError("export requires a prefill workload")
        plan_prefill_cases(
            config.workload, config.engine, dp_size=config.model.topology.dp
        )
        if os.path.lexists(args.output):
            raise FileExistsError(
                f"refusing to overwrite existing destination: {args.output}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(frozen, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (PerfTestError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
