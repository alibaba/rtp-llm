"""Write real checkpoint files with deterministic late metadata on staged FST."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def save_with_late_shard(directory, required, late):
    directory = Path(directory)
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    save_file(required, str(directory / first))
    # A physical tail not listed by the HF index makes FST use its independent
    # range reader for shard two, after finishing collective reads for shard one.
    save_file({**late, "unindexed": torch.zeros(1)}, str(directory / second))
    weight_map = {**dict.fromkeys(required, first), **dict.fromkeys(late, second)}
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
