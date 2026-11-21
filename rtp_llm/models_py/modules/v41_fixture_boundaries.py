"""Eager-only, matched Old/New observation of the fixed V4.1 prefill fixture."""

import hashlib
import json
import os
import re
from contextlib import ExitStack
from pathlib import Path
from threading import get_ident
from unittest.mock import patch

import torch


def observe_fixture_layer(call, layer, cp, h, *args, **kwargs):
    """Run the real block once. Only row digests and their layout leave the rank."""
    valid = cp.local_is_real.reshape(-1).bool()
    positions = cp.global_positions.reshape(-1)
    if positions.shape != valid.shape or h.shape[0] != valid.numel():
        raise ValueError("V4.1 fixture CP positions do not match the local hidden")
    positions = positions[valid].tolist()
    mask = layer.engram_token_mask
    if mask is None or mask.shape != valid.shape or mask.dtype != torch.bool:
        raise ValueError("Missing V4.1 fixture Engram token mask")
    if not mask[valid].all().item():
        raise ValueError("The all-text fixed fixture has a masked real token")
    result, metadata, gate = {}, {}, {}

    def save(stage, tensor):
        if stage in result or not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            raise ValueError(f"Invalid or repeated V4.1 fixture stage: {stage}")
        keep = valid
        if stage in ("wkv_in", "wkv_out") and tensor.size(0) != valid.numel():
            # A mask-compacted projection may have dropped padding. Translate
            # the original CP ownership mask to the actual WKV row layout.
            keep = valid[mask]
        if keep.numel() != tensor.size(0):
            raise ValueError(f"Unmappable V4.1 fixture rows: {stage}")
        live = tensor.detach()[keep].contiguous()
        rows = live.view(torch.uint8).reshape(len(positions), -1).cpu().numpy()
        result[stage] = [hashlib.sha256(row).hexdigest() for row in rows]
        metadata[stage] = [str(tensor.dtype), list(tensor.shape[1:])]

    save("incoming", h)
    save("previous_mix", layer.attn_hc._previous_ref().pre_mix_out)
    save("hash_ids", layer.engram_hashes)
    save("token_mask", mask)
    # Snapshot inputs before running each module (Engram/norm may work in
    # place). This fixture is shorter than the norm-fusion/chunking thresholds.
    with ExitStack() as stack:
        if h.is_cuda and hasattr(layer.engram, "q_weight"):
            from rtp_llm.models_py.modules.dsv41 import engram as gate_module
            from rtp_llm.models_py.modules.v41_causal import _gate_witness

            original = gate_module.gated_engram_residual
            owner_thread = get_ident()

            def inspect_gate(*a, **kw):
                if (
                    get_ident() != owner_thread
                    or len(a) != 6
                    or a[2] is not layer.engram.q_weight
                    or a[3] is not layer.engram.k_weight
                ):
                    return original(*a, **kw)
                if gate:
                    raise ValueError("Repeated Engram gate in the cold layer 1 fixture")
                return _gate_witness(original, a, kw, valid, gate)

            stack.enter_context(
                patch.object(gate_module, "gated_engram_residual", inspect_gate)
            )
        for name, module in (
            ("wkv", layer.engram.wkv),
            ("engram", layer.engram),
            ("attn_norm", layer.attn_norm),
            ("attention", layer.attn),
            ("moe", layer.ffn),
        ):

            def before(_module, inputs, key=name + "_in"):
                save(key, inputs[0])

            def after(_module, _inputs, output, key=name + "_out"):
                save(key, output)

            stack.callback(module.register_forward_pre_hook(before).remove)
            stack.callback(module.register_forward_hook(after).remove)
        value = call(h, *args, **kwargs)
    save("outgoing", value)
    save("next_mix", layer.ffn_hc.pre_mix_out)
    rank = int(cp.cp_rank)
    path = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    with (path / f"v41_layer1_boundaries_rank{rank}.json").open("x") as sink:
        json.dump(
            dict(
                schema=1,
                rank=rank,
                positions=positions,
                stages=result,
                metadata=metadata,
                **gate,
            ),
            sink,
            separators=(",", ":"),
        )
    return value


def validate_fixture_boundaries(root, *, layout=(4, 5120, 24, 6144)):
    """Fail closed on skipped hooks, mistyped/mis-sized rows or partial CP coverage."""
    hc, dim, nhash, ein = layout
    bf = "torch.bfloat16"
    expected = {
        **{
            k: [bf, [hc, dim]]
            for k in ("incoming", "engram_in", "engram_out", "outgoing")
        },
        **{
            k: [bf, [dim]]
            for k in (
                "attn_norm_in",
                "attn_norm_out",
                "attention_in",
                "attention_out",
                "moe_in",
                "moe_out",
            )
        },
        **{k: ["torch.float32", [hc]] for k in ("previous_mix", "next_mix")},
        "token_mask": ["torch.bool", []],
        "hash_ids": ["torch.int64", [nhash]],
        "wkv_in": [bf, [ein]],
        "wkv_out": [bf, [(hc + 1) * dim]],
    }
    seen = []
    for rank in range(2):
        record = json.loads(
            (root / f"v41_layer1_boundaries_rank{rank}.json").read_text()
        )
        pos = record["positions"]
        if (
            record["schema"] != 1
            or record["rank"] != rank
            or not pos
            or any(type(i) is not int or not 0 <= i < 783 for i in pos)
            or record["metadata"] != expected
            or record["stages"].keys() != expected.keys()
        ):
            raise ValueError(f"Incomplete V4.1 fixture rank {rank} metadata")
        for name, rows in record["stages"].items():
            if len(rows) != len(pos) or not all(
                re.fullmatch("[0-9a-f]{64}", row) for row in rows
            ):
                raise ValueError(f"Incomplete V4.1 fixture rank {rank} stage {name}")
        seen.extend(pos)
    if sorted(seen) != list(range(783)):
        raise ValueError("Both CP ranks must own every fixture position exactly once")
