"""Pinned upstream artifact loader used only by qualification tests."""

import hashlib
import json
import os
import re
import unittest
from pathlib import Path

from rtp_llm.test.utils.hf_checkpoint import resolve_hf_checkpoint

# Vendored byte-identical copy of transformers' modeling_qwen4_exp.py plus its
# content-addressed manifest, so qualification needs no external fixture path.
# QWEN4_EXP_UPSTREAM_SRC / QWEN4_EXP_UPSTREAM_MANIFEST still override both.
_VENDORED_UPSTREAM_DIR = Path(__file__).resolve().parent / "reference" / "upstream"
_VENDORED_UPSTREAM_SOURCE = _VENDORED_UPSTREAM_DIR / "modeling_qwen4_exp.py"
_VENDORED_UPSTREAM_MANIFEST = _VENDORED_UPSTREAM_DIR / "manifest.json"


def uncovered_fixture(reason: str):
    if os.environ.get("QWEN4_EXP_REQUIRE_PARITY", "0") == "1":
        raise RuntimeError(f"Qwen4 parity qualification is uncovered: {reason}")
    raise unittest.SkipTest(reason)


def upstream_source_path() -> str:
    source = os.environ.get("QWEN4_EXP_UPSTREAM_SRC", "")
    manifest_path = os.environ.get("QWEN4_EXP_UPSTREAM_MANIFEST", "")
    required = os.environ.get("QWEN4_EXP_REQUIRE_PARITY", "0") == "1"
    if required:
        source = source or str(_VENDORED_UPSTREAM_SOURCE)
        manifest_path = manifest_path or str(_VENDORED_UPSTREAM_MANIFEST)
    if required and (not source or not Path(source).is_file() or not manifest_path):
        raise RuntimeError(
            "Qwen4 parity qualification is uncovered: fixed source/manifest missing"
        )
    checkpoint = os.environ.get("QWEN4_EXP_CKPT_DIR", "")
    if required and checkpoint.startswith("hf://"):
        # These are the real tensors consumed by the qualification below;
        # the rest of the 360-GB BF16 checkpoint is not needed for parity.
        checkpoint = resolve_hf_checkpoint(
            checkpoint,
            (
                "model.language_model.layers.0.attn_hyper_connection.*",
                "*.self_attn.q_norm.weight",
                "*.self_attn.k_norm.weight",
            ),
        )
        os.environ["QWEN4_EXP_CKPT_DIR"] = checkpoint
    if required and (not checkpoint or not Path(checkpoint).is_dir()):
        raise RuntimeError("Qwen4 real-weight parity checkpoint fixture is missing")
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
        kind = manifest.get("source_kind", "git")
        if not re.fullmatch(r"[0-9a-f]{64}", manifest.get("sha256", "")):
            raise RuntimeError("Qwen4 upstream manifest needs a source SHA256")
        if kind == "git":
            valid_revision = re.fullmatch(r"[0-9a-f]{40}", manifest.get("commit") or "")
        elif kind == "content_addressed":
            valid_revision = (
                manifest.get("commit") is None
                and manifest.get("commit_status") == "unverified"
            )
        else:
            valid_revision = False
        if not valid_revision:
            raise RuntimeError(
                "Qwen4 upstream manifest needs a commit or explicit content-addressed provenance"
            )
        if not source or not Path(source).is_file():
            raise RuntimeError("Qwen4 upstream source fixture is missing")
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != manifest["sha256"]:
            raise RuntimeError(
                "Qwen4 upstream source SHA256 differs from the pinned manifest"
            )
    return source
