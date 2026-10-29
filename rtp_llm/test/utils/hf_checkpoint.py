"""Materialize revision-pinned Hugging Face fixtures on the test executor."""

import fnmatch
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Tuple


@lru_cache(maxsize=None)
def resolve_hf_checkpoint(path: str, weight_patterns: Tuple[str, ...] = ()) -> str:
    if not path.startswith("hf://"):
        return path
    match = re.fullmatch(r"hf://([^/@]+/[^/@]+)@([0-9a-f]{40})", path)
    if match is None:
        raise ValueError(
            "HF test checkpoints require hf://owner/repo@<40-character commit>"
        )

    from huggingface_hub import hf_hub_download, snapshot_download

    repo, revision = match.groups()
    kwargs = {
        "repo_id": repo,
        "revision": revision,
        "cache_dir": os.environ.get("HF_HUB_CACHE"),
        "token": False,
    }
    patterns = None
    if weight_patterns:
        index_path = hf_hub_download(filename="model.safetensors.index.json", **kwargs)
        weight_map = json.loads(Path(index_path).read_text())["weight_map"]
        shards = set()
        for pattern in weight_patterns:
            selected = {
                v for k, v in weight_map.items() if fnmatch.fnmatchcase(k, pattern)
            }
            if not selected:
                raise RuntimeError(
                    f"HF checkpoint {path} has no tensors matching {pattern}"
                )
            shards.update(selected)
        patterns = ["*.json", *sorted(shards)]

    logging.info(
        "Downloading test checkpoint %s (weight filters: %s)", path, weight_patterns
    )
    local_path = snapshot_download(allow_patterns=patterns, max_workers=4, **kwargs)
    logging.info("Test checkpoint %s is available at %s", path, local_path)
    return local_path
