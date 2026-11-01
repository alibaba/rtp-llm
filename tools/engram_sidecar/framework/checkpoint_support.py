#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared checkpoint metadata and sample utilities for distributed storage.

These helpers do not start a storage process or allocate complete tables.
Private capabilities must never be copied into public evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def emit(status, **fields):
    try:
        print(json.dumps(dict(status=status, **fields), sort_keys=True), flush=True)
    except (OSError, ValueError):
        pass


def private_write(path, payload):
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def failure_fields(error):
    """Expose bounded native CUDA diagnostics without arbitrary exception text.

    Paths, payloads and capabilities must not reach publication logs through an
    exception's message. The native bridge's fixed stage/name/numeric code are
    enough to distinguish allocation pressure from import/access failures.
    """
    fields = dict(error_type=type(error).__name__, code=getattr(error, "code", None))
    if isinstance(error, RuntimeError):
        match = re.match(r"\A([a-z_]{1,64}): (CUDA_[A-Z_]{1,80}) \(code=([0-9]{1,6})\):", str(error))
        if match:
            fields.update(cuda_stage=match[1], cuda_error=match[2], cuda_code=int(match[3]))
    return fields


@dataclass(frozen=True)
class CheckpointSource:
    checkpoint: Path
    config: dict
    model_revision: str
    expected_hash_abi: str | None
    config_sha256: str
    index_sha256: str
    component_fixture: bool


def read_source(*, fixture_root=None, checkpoint=None, model_revision=None):
    """Resolve a trusted local input without importing Torch/native code.

    Revision is an operator-supplied checkpoint identity, not a claim that the
    weights match a public release. The manifest later binds table byte hashes;
    config/index hashes are emitted separately for experiment provenance.
    """
    if (fixture_root is None) == (checkpoint is None):
        raise ValueError("select exactly one of fixture_root and checkpoint")
    expected_hash = None
    if fixture_root is not None:
        fixture_root = Path(fixture_root)
        if not fixture_root.is_absolute() or model_revision is not None:
            raise ValueError("fixture root must be absolute; fixture owns its revision")
        fixture = json.loads((fixture_root / "fixture.json").read_text())
        checkpoint = fixture_root / "owner_checkpoint"
        model_revision, expected_hash = fixture["model_revision"], fixture["hash_abi"]
    checkpoint = Path(checkpoint)
    if (not checkpoint.is_absolute() or not isinstance(model_revision, str)
            or not model_revision.strip() or len(model_revision) > 512
            or any(ord(char) < 32 for char in model_revision)):
        raise ValueError("absolute checkpoint and explicit nonempty model revision required")
    config_bytes = (checkpoint / "config.json").read_bytes()
    raw = json.loads(config_bytes)
    if not isinstance(raw, dict) or not isinstance(raw.get("text_config", {}), dict):
        raise ValueError("invalid checkpoint config")
    config = {**raw, **raw.get("text_config", {})}
    config.pop("text_config", None)
    config["checkpoint_path"] = str(checkpoint)
    if fixture_root is None and raw.get("fixture_component_only"):
        raise ValueError("component fixture requires --fixture-root")
    return CheckpointSource(checkpoint, config, model_revision, expected_hash,
                            digest(config_bytes),
                            digest((checkpoint / "model.safetensors.index.json").read_bytes()),
                            fixture_root is not None)


def selected_samples(checkpoint, manifest):
    """Bind small expected reads to checkpoint bytes, not to imported GPU data."""
    import torch
    from safetensors import safe_open

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    samples = {}
    for tensor in manifest.tensors:
        rows = sorted({0, tensor.shape[0] // 2, tensor.shape[0] - 1} |
                      {row for part in tensor.shards for row in (part.row_begin, part.row_end - 1)})
        relative = Path(index["weight_map"][tensor.name])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe checkpoint relative path")
        with safe_open(checkpoint / relative, framework="pt", device="cpu") as source:
            sliced = source.get_slice(tensor.name)
            # Slice the mapped file a row at a time; never duplicate a full table.
            payload = b"".join(sliced[row:row + 1].contiguous().view(torch.uint8).numpy().tobytes()
                               for row in rows)
        samples[tensor.name] = {"row_ids": rows, "sha256": digest(payload)}
    return samples
