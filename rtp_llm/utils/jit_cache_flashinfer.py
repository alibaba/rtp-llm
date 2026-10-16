"""Validate persistent FlashInfer build paths without importing GPU packages."""

import logging
import os
import re
import shutil
from pathlib import Path

from filelock import FileLock, Timeout


def _flashinfer_op_is_stale(op_dir: Path) -> bool:
    rules_path = op_dir / "build.ninja"
    try:
        rules = rules_path.read_text(errors="ignore")
    except OSError:
        return False
    # FlashInfer embeds the absolute include paths of the package instance that
    # generated the rules (both -I and -isystem forms). A bazel action's runfiles
    # tree only lives for that action, so rules anchored there dangle for every
    # later consumer of the shared workspace -- whether or not the tree still
    # exists at eviction time -- and only rules anchored to stable locations
    # (package-copy cache, toolkit dirs) are safe to keep.
    include_paths = re.findall(r"(?:-I|-isystem\s+)(/\S+)", rules)
    if any(".runfiles" in path for path in include_paths):
        return True
    if any(not Path(path).exists() for path in include_paths):
        return True
    src_link = op_dir / "src"
    return src_link.is_symlink() and not src_link.exists()


def evict_dangling_flashinfer_ops(workspace: Path) -> None:
    """Drop cached FlashInfer ops whose build rules anchor to include paths that
    are not stable for later consumers of the shared workspace.

    FlashInfer's generated FMHA ninja rules embed the absolute include paths of the
    flashinfer package instance that generated them. A bazel test action's runfiles
    sandbox is per-action, so rules generated from a runfiles import dangle for every
    later action that reuses the shared workspace and fail with missing-header
    errors -- including when the generating action is still alive at eviction time,
    because its sandbox disappears when the action completes. Evicting the affected
    ops makes FlashInfer regenerate them against the current stable package copy.

    FlashInfer serializes each op's build and load with a FileLock at
    ``cached_ops/tmp/<op_name>.lock``; eviction takes that same lock and re-judges
    staleness only inside it, so it cannot delete an op directory that a concurrent
    compile or load is using or has just refreshed.
    """
    stale_candidates = []
    for root, _dirs, files in os.walk(workspace, followlinks=False):
        op_dir = Path(root)
        # FlashInfer layout only: cached_ops/<op_name>/build.ninja (never trust an
        # unrelated build.ninja deeper in the tree).
        if op_dir.parent.name != "cached_ops" or "build.ninja" not in files:
            continue
        if _flashinfer_op_is_stale(op_dir):
            stale_candidates.append(op_dir)
    for op_dir in sorted(stale_candidates):
        lock_dir = op_dir.parent / "tmp"
        try:
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock = FileLock(str(lock_dir / f"{op_dir.name}.lock"), timeout=300)
            with lock:
                # Re-read the rules under the op's own lock: a compile that ran
                # since the first pass may have replaced the stale rules already.
                if _flashinfer_op_is_stale(op_dir):
                    shutil.rmtree(op_dir, ignore_errors=True)
                    logging.info(
                        f"[Package Setup] Evicted stale FlashInfer cache op: {op_dir}"
                    )
        except Timeout:
            # Somebody else owns the op (likely recompiling it); leave it alone.
            logging.info(f"[Package Setup] Skipped eviction of {op_dir}: lock busy")
