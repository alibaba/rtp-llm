"""Persist inputs and Decode history so verification never regenerates them."""

from pathlib import Path

import torch

from .block import CacheSnapshot, seed_decode_cache
from .golden import file_sha256, save_exclusive


def prepare_fixture(model, cache, inputs, phase, seed, directory, rank, write):
    path = Path(directory) / f"inputs_rank{rank}.pt"
    contract = dict(
        schema=1,
        phase=phase,
        seed=seed,
        history=(
            "all cache regions initially zero"
            if phase == "prefill"
            else "seeded synthetic BF16 KV and FP32 recurrent state; persisted verbatim"
        ),
    )
    if write:
        if phase == "decode":
            seed_decode_cache(model, seed + 1000)
        snapshot = CacheSnapshot(cache)
        tensors = {"input_ids": inputs.input_ids.cpu()}
        if phase == "decode":
            tensors.update(
                {f"cache.{i}": x.cpu() for i, x in enumerate(snapshot.saved)}
            )
        save_exclusive(path, tensors, contract)
    else:
        fixture = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if fixture["metadata"] != contract:
            raise AssertionError("fixture configuration changed")
        expected = fixture["tensors"]
        if expected["input_ids"].shape != inputs.input_ids.shape:
            raise AssertionError("fixture token shape changed")
        inputs.input_ids.copy_(expected["input_ids"])
        snapshot = CacheSnapshot(cache)
        if phase == "decode":
            if len(expected) != len(snapshot.live) + 1:
                raise AssertionError("fixture cache region count changed")
            for i, (live, saved) in enumerate(zip(snapshot.live, snapshot.saved)):
                value = expected[f"cache.{i}"]
                if value.shape != live.shape or value.dtype != live.dtype:
                    raise AssertionError(f"fixture cache region {i} changed")
                live.copy_(value)
                saved.copy_(live)
    return snapshot, file_sha256(path)
