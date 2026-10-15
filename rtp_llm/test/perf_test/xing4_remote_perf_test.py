"""Xing4.0 perf entry for remote pool workers without the CI-NAS FUSE mount.

The H20 pool workers (133/134) cannot see /mnt/fuse/smoke-test, so perf
targets cannot reference the staged checkpoint there. This main stages
the pinned checkpoint into the worker's ~/.cache once (resumable curl
per file from the whale HF mirror, exact tree-size verification — the
same pattern as rtp_llm/test/smoke/xing4_remote_smoke_test.py), rewrites
the --checkpoint_path/--tokenizer_path argv entries to the local
directory, and delegates to batch_decode_test.main() unchanged. The
runner, modes and measurements are the stock perf framework; only the
checkpoint location differs per worker.
"""

import json
import logging
import os
import subprocess
import sys
import urllib.request

REPO_ID = "XingChen-AGI/Xing4.0-29B-A4B-FP8"
REVISION = "5080375347768be8f18304df141a1aea5b8416c9"
MIRROR = "https://whale-hf-mirror.alibaba-inc.com"
SHARD_COUNT = 41

_STAGED = None


def _local_model_dir() -> str:
    return os.path.join(
        os.path.expanduser("~/.cache/xing4_models"),
        f"{REPO_ID.replace('/', '--')}--{REVISION}",
    )


def _curl(url: str, dest: str) -> None:
    for attempt in range(5):
        proc = subprocess.run(
            ["curl", "-sL", "--retry", "3", "-C", "-", "-o", dest, url],
            capture_output=True,
        )
        if proc.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            return
        logging.warning("curl attempt %d for %s rc=%d", attempt, url, proc.returncode)
    raise RuntimeError(f"download failed: {url}")


def _tree_sizes() -> dict:
    url = f"{MIRROR}/api/models/{REPO_ID}/tree/{REVISION}?recursive=true"
    with urllib.request.urlopen(url, timeout=120) as r:
        entries = json.loads(r.read().decode())
    sizes = {}
    for e in entries:
        if e.get("type") == "file" or "size" in e:
            sizes[e["path"]] = int(e.get("size", 0))
    if not sizes:
        raise RuntimeError(f"empty tree listing from {url}")
    return sizes


def _ensure_model(local_dir: str) -> None:
    marker = os.path.join(local_dir, ".xing4_stage_complete")
    if os.path.exists(marker) and os.path.getsize(marker) == len(REVISION):
        return
    os.makedirs(local_dir, exist_ok=True)
    base = f"{MIRROR}/{REPO_ID}/resolve/{REVISION}"
    sizes = _tree_sizes()
    expected_shards = sorted(
        n for n in sizes if n.startswith("model-") and n.endswith(".safetensors")
    )
    if len(expected_shards) != SHARD_COUNT:
        raise RuntimeError(
            f"tree lists {len(expected_shards)} shards, expected {SHARD_COUNT}"
        )
    for name in sorted(sizes):
        dest = os.path.join(local_dir, name)
        want = sizes[name]
        if os.path.exists(dest) and os.path.getsize(dest) == want:
            continue
        if os.path.exists(dest) and os.path.getsize(dest) > want:
            # A mismatched oversized file (e.g. a resumed download that
            # appended a full response onto a partial) cannot be resumed
            # into the right size; start over.
            os.remove(dest)
        _curl(f"{base}/{name}", dest)
        if os.path.getsize(dest) != want:
            raise RuntimeError(
                f"{name}: got {os.path.getsize(dest)} bytes, expected {want}"
            )
    with open(marker, "w") as f:
        f.write(REVISION)
    logging.info("staged %s locally at %s", REPO_ID, local_dir)


def _stage_once() -> str:
    global _STAGED
    if _STAGED is None:
        model_dir = _local_model_dir()
        _ensure_model(model_dir)
        _STAGED = model_dir
    return _STAGED


def _rewrite_checkpoint_args(argv: list) -> None:
    local = _stage_once()
    out = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--checkpoint_path", "--tokenizer_path") and i + 1 < len(argv):
            out.append(arg)
            out.append(local)
            i += 2
            continue
        if arg.startswith("--checkpoint_path=") or arg.startswith("--tokenizer_path="):
            key = arg.split("=", 1)[0]
            out.append(f"{key}={local}")
            i += 1
            continue
        out.append(arg)
        i += 1
    argv[:] = out


def main() -> None:
    _rewrite_checkpoint_args(sys.argv)
    from rtp_llm.test.perf_test import batch_decode_test

    batch_decode_test.main()


if __name__ == "__main__":
    main()
