"""Real checkpoint loader exercised with deterministic standard safetensors.

This fixture has two complete Engram tables, split across four files. Only this
owner has their checkpoint files. Existing consumer.py independently generates
sample-row expectations; no complete table or checkpoint crosses to consumers.
Owner teardown follows owner_fixture's explicit consumer-exit / STOP protocol.
This is a component integration test, not an RTP server deployment.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

import fixture
import owner_fixture


def write_fixture(root):
    import torch
    from safetensors.torch import save_file

    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    weight_map, hashes = {}, {}
    for layer, rows in fixture.LAYERS:
        weight, scale = fixture.make_table(rows)
        for role, tensor in (("weight", weight.view(torch.float8_e4m3fn)),
                             ("scale", scale)):
            name = f"layers.{layer}.engram.embed.{role}"
            filename = f"layer-{layer}-{role}.safetensors"
            save_file({name: tensor}, str(root / filename))
            weight_map[name] = filename
            hashes[name] = "sha256:" + hashlib.sha256(
                tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return hashes


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=4096)
    args, remainder = parser.parse_known_args()
    if not args.checkpoint.is_absolute():
        parser.error("fixture checkpoint must use an absolute fresh path")
    sys.argv[1:] = remainder
    trusted_hashes = write_fixture(args.checkpoint)

    def create(native, owners, device, numa):
        from rtp_llm.models_py.modules.dsv41.engram_checkpoint import (
            CheckpointError, inspect_checkpoint, load_checkpoint)
        from rtp_llm.models_py.modules.dsv41.engram_storage import CleanupPendingError

        granularity = native.host_numa_granularity(numa, device)
        assert not any(native.resource_counters().values())
        try:
            inspect_checkpoint(args.checkpoint, fixture.expected_layout(),
                               allocation_granularity=granularity, budget_bytes=1)
        except CheckpointError as error:
            assert error.code == "BUDGET_EXCEEDED"
        else:
            raise AssertionError("undersized budget was accepted")
        assert not any(native.resource_counters().values())
        plan = inspect_checkpoint(args.checkpoint, fixture.expected_layout(),
                                  allocation_granularity=granularity, budget_bytes=16 << 20)
        try:
            generation = load_checkpoint(plan, native=native, device=device, numa=numa,
                                         chunk_bytes=args.chunk_bytes)
        except CleanupPendingError as error:
            owners.append(error.resource)  # retain/retry failed rollback at teardown
            raise
        owners.append(generation)
        document = json.loads(generation.manifest_bytes)
        assert {tensor["name"]: tensor["shards"][0]["sha256"]
                for tensor in document["tensors"]} == trusted_hashes
        capabilities = {allocation: base64.b64encode(handle).decode("ascii")
                        for allocation, handle in generation.private_handles.items()}
        descriptions = [dict(allocation_id=allocation,
                             allocation_bytes=layer.allocation_bytes,
                             logical_bytes=layer.logical_bytes)
                        for allocation, layer in zip(capabilities, plan.layers)]
        owner_fixture.emit("CHECKPOINT_LOADED", files=len(plan.files),
                           tensors=len(document["tensors"]), chunk_bytes=args.chunk_bytes,
                           physical_bytes=plan.physical_bytes, full_tensor_hashes_match=True,
                           insufficient_budget_rejected_before_allocation=True)
        return document, capabilities, descriptions

    return owner_fixture.main(create)


if __name__ == "__main__":
    raise SystemExit(main())
