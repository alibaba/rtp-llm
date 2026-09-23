"""Discover real captures from files and describe their observed traffic.

Data admission is a user decision, not a registry. Codec and checksums belong
in the sidecar; defaults identify only the inputs pinned by existing tests.
"""
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from traffic.capture_contract import BLOCK_SIZE as BLOCK
from traffic.codecs import decode

DATA = Path(__file__).resolve().parents[2] / "data"
DEFAULT_TRACE = "glm-5.3_20260921_1400_15m"
DEFAULT_PROFILE = DEFAULT_TRACE


def trace_models():
    return {path.stem: path for path in sorted((DATA / "traffic_models").glob("*.xz"))}


def model_path(name=DEFAULT_TRACE):
    try:
        return trace_models()[name]
    except KeyError:
        raise ValueError(f"unknown traffic model {name!r}") from None


def profile_path(name=DEFAULT_PROFILE):
    for path in (DATA / "calibration").glob("*.profile.json"):
        if path.name == f"{name}.profile.json":
            return path
    raise ValueError(f"unknown calibration profile {name!r}")


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return dict(count=len(ordered), min=ordered[0], max=ordered[-1],
                mean=sum(ordered) / len(ordered),
                **{f"p{p}": ordered[max(0, math.ceil(len(ordered) * p / 100) - 1)]
                   for p in (50, 90, 95, 99)})


def statistics(events, *, version=2):
    """Nearest-rank percentiles; sharing describes the DAG, not cache hits."""
    duration = (events[-1][0] - events[0][0]) / 1000
    counts = [0] * (int(duration) + 1)
    for event in events:
        counts[(event[0] - events[0][0]) // 1000] += 1
    resolution = 1 if version == 3 else BLOCK
    inputs = [event[1] * resolution for event in events]
    shared = [event[3] * BLOCK for event in events]
    return dict(
        scope="captured_requests_only_no_missing_pod_extrapolation",
        request_count=len(events), percentile_method="nearest_rank",
        arrival=dict(event_span_s=duration,
                     mean_qps=len(events) / duration if duration else None,
                     requests_per_second=distribution(counts),
                     bucket_origin="first_event", bucket_ms=1000,
                     final_bucket="included_even_if_partial", idle_buckets_included=True),
        input_tokens=dict(**distribution(inputs), total=sum(inputs), resolution_tokens=resolution),
        prefix_structure=dict(shared_tokens=distribution(shared),
                              token_weighted_shared_fraction=sum(shared) / sum(inputs),
                              requests_with_shared_prefix=sum(value > 0 for value in shared),
                              parentless_requests=sum(event[2] == -1 for event in events),
                              interpretation="structural_sharing_not_observed_cache_hit_rate"),
        output_tokens=None, outcomes=None)


def build_manifest(path, source=None):
    path = Path(path)
    raw = path.read_bytes()
    metadata, events = decode(raw)
    provenance = metadata["provenance"]
    local = timezone(timedelta(hours=8))
    window = {key: datetime.fromtimestamp(provenance[field] / 1000, local).isoformat()
              if provenance.get(field) is not None else None
              for key, field in (("start", "source_start"), ("end", "source_end"))}
    return dict(metadata, manifest_schema_version=2, bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                codec=dict(name="prefix_lineage", version=metadata["version"]),
                data_kind="real", source=source or dict(
                    capture=dict(layer="frontend", deployment=None),
                    spectrum=dict(identity=None, status="unconfirmed", evidence=None),
                    model=dict(name=None, status="unconfirmed", evidence=None),
                    enrichment="manual_spectrum_lookup_not_automated"),
                capture_window=dict(window, timezone="Asia/Shanghai"),
                statistics=statistics(events, version=metadata["version"]),
                limitations=[("Input lengths are exact; original text and tokens are absent."
                              if metadata["version"] == 3 else
                              "Input lengths are block aligned; original text and tokens are absent."),
                             "Output lengths and outcomes are not retained in this codec.",
                             "Prefix sharing is theoretical structure, not measured cache reuse.",
                             "Spectrum origin and model need independently confirmed manual metadata."])


def read_manifest(path):
    path = Path(path)
    sidecar = path.with_suffix(".manifest.json")
    if not sidecar.is_file():
        raise ValueError(f"missing manifest: {sidecar}")
    manifest = json.loads(sidecar.read_text())
    raw = path.read_bytes()
    if manifest.get("bytes") != len(raw) or manifest.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"{path}: model and manifest bytes/SHA256 disagree")
    try:
        metadata, events = decode(raw, manifest)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{path}: manifest codec / model mismatch: {exc}") from None
    for field, value in metadata.items():
        if manifest.get(field) != value:
            raise ValueError(f"{path}: manifest {field} disagrees with model contents")
    if manifest["count"] != len(events):
        raise ValueError(f"{path}: event count disagrees with manifest")
    return manifest
