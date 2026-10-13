"""Generate and verify self-describing capture manifests against their payload."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from traffic.statistics import statistics
from traffic.codecs import decode

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
                    capture=dict(layer="frontend"),
                    attribution=dict(identity=None, status="unconfirmed", evidence=None),
                    model=dict(name=None, status="unconfirmed", evidence=None)),
                capture_window=dict(window, timezone="Asia/Shanghai"),
                statistics=statistics(events, version=metadata["version"], block_size=metadata["block_size"]),
                limitations=[("Input lengths are exact; original text and tokens are absent."
                              if metadata["version"] == 3 else
                              "Input lengths are block aligned; original text and tokens are absent."),
                             "Output lengths and outcomes are not retained in this codec.",
                             "Prefix sharing is theoretical structure, not measured cache reuse.",
                             "Source attribution and model require independent confirmation."])


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
