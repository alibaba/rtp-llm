#!/usr/bin/env python3
"""Make a small local checkpoint view whose seven weight shards stay on 3FS."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


SOURCE_ROOT = Path("/mnt/hf3fs/3fs/models/kimi/kimi-k3")
EXPECTED_SOURCE_CONFIG = "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
EXPECTED_SOURCE_INDEX = "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
EXPECTED_VIEW_CONFIG = "72e146f1be7061dc281ab86ad9a1544c8f5dcdbfbe510c92b85a0d943b58fe54"
EXPECTED_VIEW_INDEX = "dc08028ec4f45b41dfe51d5edff3168cc088404a92cf3b72dd9daaa10e29b358"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve(strict=True)
    original = base / "models/kimi-k3-4layers-ms-20260929"
    destination = base / "models/kimi-k3-4layers-ms-3fs-view-20260929"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    for root, expected_config, expected_index in (
        (SOURCE_ROOT, EXPECTED_SOURCE_CONFIG, EXPECTED_SOURCE_INDEX),
        (original, EXPECTED_VIEW_CONFIG, EXPECTED_VIEW_INDEX),
    ):
        assert digest(root / "config.json") == expected_config, root
        assert digest(root / "model.safetensors.index.json") == expected_index, root
    manifest = json.loads((original / "extraction_manifest.json").read_text())
    mapping = manifest["source_shards"]
    index = json.loads((original / "model.safetensors.index.json").read_text())
    assert len(mapping) == 7
    assert set(index["weight_map"].values()) == set(mapping)
    for view_name, source_name in mapping.items():
        assert Path(view_name).name == view_name
        assert Path(source_name).name == source_name
        assert (original / view_name).stat().st_size == (SOURCE_ROOT / source_name).stat().st_size
    tmp = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
    try:
        for item in original.iterdir():
            if item.is_file() and not item.name.endswith(".safetensors"):
                shutil.copy2(item, tmp / item.name)
        for view_name, source_name in mapping.items():
            (tmp / view_name).symlink_to(SOURCE_ROOT / source_name)
        os.rename(tmp, destination)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    print(json.dumps({"view": str(destination), "shards": len(mapping),
                      "physical_weight_root": str(SOURCE_ROOT),
                      "source_config_sha256": EXPECTED_SOURCE_CONFIG,
                      "source_index_sha256": EXPECTED_SOURCE_INDEX,
                      "view_config_sha256": EXPECTED_VIEW_CONFIG,
                      "view_index_sha256": EXPECTED_VIEW_INDEX}, sort_keys=True))


if __name__ == "__main__":
    main()
