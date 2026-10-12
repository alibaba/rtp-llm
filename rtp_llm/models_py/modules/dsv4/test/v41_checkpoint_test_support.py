"""Common named checkpoint controls for native PP and eager DP workers."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.test import (
    test_v41_pp_checkpoint_engine_sm120 as producer,
)
from rtp_llm.utils.model_weight import W


def emit(marker, **values):
    print(marker + " " + json.dumps(values), flush=True)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def byte_hash(value):
    return producer.tensor_hash(value.detach().cpu())


def save_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def loader_oracles(args, weights):
    """Bind actual loader bytes to independent synthetic checkpoint slices."""
    result = {"source_loader": {}, "destination_loader": {}}
    for role in result:
        for rank in range(8):
            pp = role == "source_loader"
            experts = 4 if pp else 2
            expert_begin = (rank % 4 if pp else rank) * experts
            owned = (
                set(range(21) if rank < 4 else range(21, 40)) if pp else set(range(40))
            )
            layers = []
            for layer_id, mapping in enumerate(weights.weights):
                selected = {}
                if layer_id in owned:
                    for name, value in mapping.items():
                        if (
                            name == W.v4_compressor_wgate
                            and args.compress_ratios[layer_id] != 2
                        ):
                            continue
                        if name.startswith("v4.routed."):
                            dtype = (
                                torch.int8
                                if name.endswith("weight")
                                else torch.float8_e8m0fnu
                            )
                            value = value[expert_begin : expert_begin + experts].view(
                                dtype
                            )
                        if name in (W.v4_hc_attn_scale, W.v4_hc_ffn_scale):
                            value = value.reshape(3, 1)
                        selected[name] = value
                layers.append(selected)
            globals_owned = (
                {W.embedding} if pp and rank < 4 else {W.final_ln_gamma, W.lm_head}
            )
            if not pp:
                globals_owned.add(W.embedding)
            global_weights = {
                name: value
                for name, value in weights.global_weights.items()
                if name in globals_owned
            }
            if not pp or rank >= 4:
                global_weights[W.final_ln_beta] = torch.zeros_like(
                    weights.global_weights[W.final_ln_gamma]
                )
            if pp and rank >= 4:
                width = args.vocab_size // 4
                global_weights[W.lm_head] = global_weights[W.lm_head][
                    rank % 4 * width : (rank % 4 + 1) * width
                ]
            result[role][str(rank)] = producer.loader_signature(
                SimpleNamespace(global_weights=global_weights, weights=layers)
            )
    return result
