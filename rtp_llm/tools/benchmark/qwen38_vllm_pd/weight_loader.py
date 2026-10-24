"""Official block-FP8 source to matched online FP8/NVFP4 serving weights."""

import json
import os
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open

from .fp8_weights import quantize_compute_views
from .nvfp4_reference import quantize_expert_to_nvfp4


def decode_block_fp8(weight, scale):
    expected = tuple((size + 127) // 128 for size in weight.shape)
    if tuple(scale.shape) != expected:
        raise ValueError("FP8 block128 weight/scale geometry differs")
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return (weight.float() * expanded[: weight.shape[0], : weight.shape[1]]).to(
        torch.bfloat16
    )


def expert_inventory(index):
    pattern = re.compile(
        r"((?:model\.language_model|mtp)\.layers\.\d+\.mlp\.experts)\."
        r"(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    groups = {}
    for name in index:
        match = pattern.fullmatch(name)
        if match is None:
            continue
        prefix, expert, projection = match.groups()
        if name + "_scale_inv" not in index:
            raise ValueError(f"Missing source block scale: {name}")
        groups.setdefault((prefix, int(expert)), {})[projection] = name
    layers = {prefix for prefix, _ in groups}
    if len(layers) != 49 or len(groups) != 49 * 512:
        raise ValueError(
            f"Incomplete official FP8 expert inventory: {len(layers)} layers, {len(groups)} experts"
        )
    for prefix in layers:
        if {expert for layer, expert in groups if layer == prefix} != set(range(512)):
            raise ValueError(f"Incomplete expert IDs: {prefix}")
    for key, group in groups.items():
        if set(group) != {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"Incomplete expert projections: {key}")
    return groups


def expert_weights(source, index, local_expert_ids, only_mtp=None):
    groups = expert_inventory(index)
    if only_mtp is not None:
        groups = {
            key: value
            for key, value in groups.items()
            if key[0].startswith("mtp.") == only_mtp
        }
    device = torch.device("cuda", torch.cuda.current_device())

    def read(name):
        with safe_open(str(source / index[name]), framework="pt", device="cpu") as file:
            weight = file.get_tensor(name).to(device)
        scale_name = name + "_scale_inv"
        with safe_open(
            str(source / index[scale_name]), framework="pt", device="cpu"
        ) as file:
            scale = file.get_tensor(scale_name).to(device)
        return decode_block_fp8(weight, scale)

    loaded = 0
    for (prefix, expert), group in sorted(groups.items()):
        for part in ("gate_proj", "up_proj", "down_proj"):
            yield f"{prefix}.{expert}.{part}.input_scale", torch.tensor(
                [32 / 2688], dtype=torch.float32
            )
        if local_expert_ids is not None and expert not in local_expert_ids:
            continue
        # RTP quantizes fused up+gate with one global scale; retain that exact
        # NVFP4 convention instead of separately scaling the two projections.
        fused = torch.cat([read(group["up_proj"]), read(group["gate_proj"])], dim=0)
        packed, block_scale, global_scale = quantize_expert_to_nvfp4(fused)
        for part, payload, block in zip(
            ("up_proj", "gate_proj"), packed.chunk(2, 0), block_scale.chunk(2, 0)
        ):
            base = f"{prefix}.{expert}.{part}"
            yield base + ".weight", payload.cpu()
            yield base + ".weight_scale", block.cpu()
            yield base + ".weight_scale_2", global_scale.reshape(1).cpu()
        del fused, packed, block_scale, global_scale
        packed, block_scale, global_scale = quantize_expert_to_nvfp4(
            read(group["down_proj"])
        )
        base = f"{prefix}.{expert}.down_proj"
        yield base + ".weight", packed.cpu()
        yield base + ".weight_scale", block_scale.cpu()
        yield base + ".weight_scale_2", global_scale.reshape(1).cpu()
        loaded += 1
    layer_count = 49 if only_mtp is None else (1 if only_mtp else 48)
    expected = layer_count * (
        len(local_expert_ids) if local_expert_ids is not None else 512
    )
    if loaded != expected:
        raise ValueError(f"Expert conversion incomplete: {loaded}/{expected}")
    print("QWEN38_NVFP4_EXPERTS_LOADED", loaded, flush=True)


def register():
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader("qwen38_fp8_nvfp4_pd")
    class FullStateModelLoader(DefaultModelLoader):
        def load_model(self, vllm_config, model_config, prefix=""):
            model = super().load_model(vllm_config, model_config, prefix)
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                VocabParallelEmbedding,
            )

            tables = []
            for name, module in model.named_modules():
                if not isinstance(module, VocabParallelEmbedding):
                    continue
                weight = module.weight
                is_ple = "ngram_embedding" in name
                if weight.dtype != torch.float8_e4m3fn:
                    raise ValueError(f"Vocabulary table is not FP8: {name}")
                if is_ple:
                    if weight.is_cuda or not weight.is_pinned():
                        raise ValueError(f"PLE big table must be pinned CPU: {name}")
                elif not weight.is_cuda:
                    raise ValueError(f"Main vocabulary/LM head must be on GPU: {name}")
                tables.append(
                    dict(
                        name=name,
                        shape=list(weight.shape),
                        dtype=str(weight.dtype),
                        device=str(weight.device),
                        pinned=weight.is_pinned(),
                        is_ple=is_ple,
                    )
                )
            if not any(not row["is_ple"] for row in tables):
                raise ValueError("Main vocabulary inventory is missing")
            inventory = dict(
                model=type(model).__name__,
                tables=tables,
                scope="actual post-load FP8 table placement;not PD/Graph acceptance",
            )
            if directory := os.environ.get("QWEN38_PD_INVENTORY_DIR"):
                from torch.distributed import get_rank

                target = Path(directory)
                target.mkdir(parents=True, exist_ok=True)
                path = (
                    target
                    / f"{type(model).__name__}-rank{get_rank()}-pid{os.getpid()}.json"
                )
                with path.open("x") as output:
                    json.dump(inventory, output, indent=2)
            print("QWEN38_TABLE_PLACEMENT_PASS", json.dumps(inventory), flush=True)
            return model

        def download_model(self, model_config):
            source = Path(os.environ["QWEN38_PD_SOURCE"])
            config = json.loads((source / "config.json").read_text())
            if config["quantization_config"].get("weight_block_size") != [128, 128]:
                raise ValueError("Official block128 FP8 source is required")
            index = json.loads((source / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            expert_inventory(index)

        def get_all_weights(self, model_config, model):
            self.counter_before_loading_weights = time.perf_counter()
            source = Path(os.environ["QWEN38_PD_SOURCE"])
            index = json.loads((source / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]
            by_shard = {}
            for name, shard in index.items():
                if ".mlp.experts." in name or name.startswith(
                    ("visual.", "model.visual.")
                ):
                    continue
                by_shard.setdefault(shard, []).append(name)
            for shard, names in sorted(by_shard.items()):
                with safe_open(
                    str(source / shard), framework="pt", device="cpu"
                ) as file:
                    for name in sorted(names):
                        yield name, file.get_tensor(name)
            yield from expert_weights(
                source,
                index,
                self.local_expert_ids,
                only_mtp="MTP" in type(model).__name__,
            )

        def load_weights(self, model, model_config):
            super().load_weights(model, model_config)
            quantize_compute_views(model)
