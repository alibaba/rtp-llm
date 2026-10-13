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


def expert_weights(source, index, local_expert_ids):
    pattern = re.compile(
        r"((?:model\.language_model|mtp)\.layers\.\d+\.mlp\.experts)\."
        r"(gate_up_proj|down_proj)$"
    )
    device = torch.device("cuda", torch.cuda.current_device())
    for name, shard in sorted(index.items()):
        match = pattern.fullmatch(name)
        if match is None:
            continue
        scale_name = name + "_scale_inv"
        if scale_name not in index:
            raise ValueError(f"Expert block scale is missing: {name}")
        prefix, projection = match.groups()
        with (
            safe_open(str(source / shard), framework="pt", device="cpu") as weights,
            safe_open(
                str(source / index[scale_name]), framework="pt", device="cpu"
            ) as scales,
        ):
            weight_slice = weights.get_slice(name)
            scale_slice = scales.get_slice(scale_name)
            if weight_slice.get_shape()[0] != 512:
                raise ValueError(f"Unexpected expert count: {name}")
            parts = (
                ("up_proj", "gate_proj")
                if projection == "gate_up_proj"
                else ("down_proj",)
            )
            for expert in range(512):
                for part in parts:
                    yield f"{prefix}.{expert}.{part}.input_scale", torch.tensor(
                        [32 / 2688], dtype=torch.float32
                    )
                if local_expert_ids is not None and expert not in local_expert_ids:
                    continue
                weight = weight_slice[expert].to(device)
                scale = scale_slice[expert].to(device)
                decoded = decode_block_fp8(weight, scale)
                packed, block_scale, global_scale = quantize_expert_to_nvfp4(decoded)
                packed_parts = packed.chunk(len(parts), 0)
                scale_parts = block_scale.chunk(len(parts), 0)
                for part, payload, block in zip(parts, packed_parts, scale_parts):
                    base = f"{prefix}.{expert}.{part}"
                    yield base + ".weight", payload.cpu()
                    yield base + ".weight_scale", block.cpu()
                    yield base + ".weight_scale_2", global_scale.reshape(1).cpu()


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
            yield from expert_weights(source, index, self.local_expert_ids)

        def load_weights(self, model, model_config):
            super().load_weights(model, model_config)
            quantize_compute_views(model)
