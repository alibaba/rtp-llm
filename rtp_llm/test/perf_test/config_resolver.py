"""Canonical config resolution and fingerprints for unified perf tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from rtp_llm.test.perf_test.config_schema import (
    ConfigError,
    ExecutionParser,
    parse_config,
)

DEFAULT_MAX_EXTERNAL_BYTES = 1024 * 1024


class ConfigResolutionError(ConfigError):
    """Raised when source config or external workload materialization fails."""


@dataclass(frozen=True)
class ResolvedConfig:
    resolved_config: Dict[str, Any]
    config_fingerprint: str
    workload_fingerprint: str


def canonical_json(value: Any) -> str:
    """Return the single canonical JSON representation used for fingerprints."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ConfigResolutionError(f"config is not canonical JSON: {error}") from error


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _load_external_json(path: Path, max_bytes: int) -> Any:
    field = "workload.distribution_file"
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ConfigResolutionError("max_external_bytes must be a positive integer")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ConfigResolutionError(f"{field}: cannot read {path}: {error}") from error
    if size > max_bytes:
        raise ConfigResolutionError(
            f"{field}: {path} is too large ({size} bytes; limit {max_bytes})"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigResolutionError(f"{field}: {path} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise ConfigResolutionError(
            f"{field}: {path} is not valid JSON: {error.msg}"
        ) from error
    except OSError as error:
        raise ConfigResolutionError(f"{field}: cannot read {path}: {error}") from error


def _materialize_workload(
    source: Mapping[str, Any],
    normalized: Dict[str, Any],
    source_dir: Path,
    max_external_bytes: int,
) -> None:
    source_workload = source["workload"]
    if source_workload.get("kind") != "distribution":
        return
    source_path = Path(source_workload["distribution_file"])
    if not source_path.is_absolute():
        source_path = source_dir / source_path
    normalized_workload = normalized["workload"]
    normalized_workload.pop("distribution_file", None)
    normalized_workload["distribution"] = _load_external_json(
        source_path, max_external_bytes
    )


def _materialize_traffic(parsed, normalized, source_dir, max_external_bytes):
    from rtp_llm.test.perf_test.prefill.traffic_batch import sample_traffic_batches
    from rtp_llm.test.perf_test.prefill.traffic_distribution import (
        parse_length_distribution,
    )

    def materialize(workload, target):
        spec = workload.traffic_request_batches
        if spec is None:
            return
        path = Path(spec.distribution_file)
        if not path.is_absolute():
            path = source_dir / path
        payload = _load_external_json(path, max_external_bytes)
        try:
            distribution = parse_length_distribution(payload)
        except ValueError as error:
            raise ConfigResolutionError(f"traffic distribution: {error}") from error
        sample = sample_traffic_batches(
            workload, parsed.engine, distribution, dp_size=parsed.model.topology.dp
        )
        target["request_batches"] = [asdict(batch) for batch in sample.batches]
        target["traffic_sampling"] = {
            "distribution_sha256": _fingerprint(payload),
            "parameters": {
                key: value
                for key, value in asdict(spec).items()
                if key != "distribution_file"
            },
            "summary": sample.summary,
        }
        target["max_input_len"] = max(
            group.input_len
            for batch in sample.batches
            for group in batch.request_groups
        )

    workload = parsed.workload
    if workload.kind != "cache_grid":
        return
    materialize(workload, normalized["workload"])
    for grid, target in zip(workload.grids, normalized["workload"]["grids"]):
        materialize(grid.workload, target["workload"])
    if workload.grids:
        # Named grids inherit the first grid's derived fields in parse_config.
        # Traffic sampling can change that grid's maximum after parsing.
        normalized["workload"]["max_input_len"] = normalized["workload"]["grids"][0][
            "workload"
        ]["max_input_len"]


def resolve_config(
    value: Any,
    *,
    source_dir: Optional[Union[str, Path]] = None,
    max_external_bytes: int = DEFAULT_MAX_EXTERNAL_BYTES,
    execution_parser: Optional[ExecutionParser] = None,
) -> ResolvedConfig:
    """Validate, normalize, materialize, and fingerprint a decoded config."""
    parsed = parse_config(value, execution_parser=execution_parser)
    normalized = asdict(parsed)
    workload = normalized["workload"]
    if workload.get("kind") == "cache_grid":
        for child in [workload] + [grid["workload"] for grid in workload["grids"]]:
            if child.get("random_request_batches") is None:
                child.pop("random_request_batches", None)
            child.pop("traffic_request_batches", None)
            for batch in child["request_batches"]:
                if batch["sample_index"] is None:
                    batch.pop("sample_index")
    base_dir = Path.cwd() if source_dir is None else Path(source_dir)
    _materialize_workload(value, normalized, base_dir, max_external_bytes)
    _materialize_traffic(parsed, normalized, base_dir, max_external_bytes)

    config_fields = {
        name: section for name, section in normalized.items() if name != "workload"
    }
    return ResolvedConfig(
        resolved_config=normalized,
        config_fingerprint=_fingerprint(config_fields),
        workload_fingerprint=_fingerprint(normalized["workload"]),
    )


def load_resolved_config(
    path: Union[str, Path],
    *,
    max_external_bytes: int = DEFAULT_MAX_EXTERNAL_BYTES,
    execution_parser: Optional[ExecutionParser] = None,
) -> ResolvedConfig:
    """Load a source JSON config and resolve relative workload inputs."""
    config_path = Path(path)
    try:
        source = json.loads(config_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigResolutionError(f"config {config_path} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise ConfigResolutionError(
            f"config {config_path} is not valid JSON: {error.msg}"
        ) from error
    except OSError as error:
        raise ConfigResolutionError(
            f"cannot read config {config_path}: {error}"
        ) from error
    return resolve_config(
        source,
        source_dir=config_path.parent,
        max_external_bytes=max_external_bytes,
        execution_parser=execution_parser,
    )
