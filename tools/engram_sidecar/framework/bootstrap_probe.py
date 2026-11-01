#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real file bootstrap/tokenizer/hash/lookup probe, without the engine or WKV.

Run before the heavier model-hook test. Never treat this as full-model evidence.
Only selected expected table rows are generated locally for comparison.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from checkpoint_fixture import LAYERS, table_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--shard-json", type=Path, action="append", required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file
    from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, HostEngramEmbedding, NgramHashState
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _load_native, attach_from_package
    from rtp_llm.models_py.modules.dsv41.engram_startup import package_from_descriptors

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    checkpoint = args.fixture_root / "consumer_checkpoint"
    config = json.loads((checkpoint / "config.json").read_text())
    layout = EngramLayout(config)
    hash_state = NgramHashState(layout, str(checkpoint))
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    if any(".embed." in key for key in index["weight_map"]):
        raise AssertionError("consumer checkpoint must have no embedding table")
    native = _load_native()
    package = package_from_descriptors([str(path) for path in args.shard_json], str(checkpoint))
    before = torch.cuda.memory_allocated(device)
    storage = attach_from_package(package, layout=layout, hash_state=hash_state, device=device)
    after = torch.cuda.memory_allocated(device)
    embeddings = []
    results = []
    try:
        for layer_id, _ in LAYERS:
            embeddings.append(HostEngramEmbedding.from_storage(storage, layer_id))
        inputs = load_file(args.fixture_root / "oracle_inputs.safetensors", device="cpu")
        fixture = json.loads((args.fixture_root / "fixture.json").read_text())
        with torch.inference_mode():
            for case in fixture["cases"]:
                label = case["name"]
                windows = inputs[label + ".engram_token_windows"].to(device)
                hashed = hash_state.hash_token_windows(windows, dead_mask=windows == config["image_token_id"])
                torch.testing.assert_close(hashed.cpu(), inputs[label + ".hash_ids"], rtol=0, atol=0)
                for hash_index, (layer_id, _) in enumerate(LAYERS):
                    ids = hashed[:, hash_index]
                    selected = ids.cpu().long()
                    weight, scale = table_rows(layer_id, selected.clamp_min(0))
                    expected = (weight.float() * torch.exp2(scale.view(torch.uint8).float() - 127)
                                .repeat_interleave(32, -1)).to(torch.bfloat16)
                    expected.masked_fill_(selected[..., None] < 0, 0)
                    actual = embeddings[hash_index](ids, device)
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                    results.append(dict(case=label, layer=layer_id, rows=case["batch_size"], exact=True))
            torch.cuda.synchronize(device)
        info = storage.info()
        print(json.dumps(dict(status="PASS", scope="bootstrap_hash_lookup_only", cases=results,
                              dataset_id=info["dataset_id"], generation=info["generation"],
                              allocation_ids=sorted(info["allocations"]),
                              physical_bytes=info["physical_bytes"],
                              torch_attach_allocated_bytes=after - before,
                              no_consumer_checkpoint_tables=True), sort_keys=True), flush=True)
    finally:
        torch.cuda.synchronize(device)
        for embedding in embeddings:
            embedding.release_external_views()
        embeddings.clear()
        gc.collect()
        storage.close()
        counters = native.resource_counters()
        if any(counters.values()):
            raise AssertionError("native consumer resources remain")
        print(json.dumps(dict(status="CLOSED", counters=counters), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
