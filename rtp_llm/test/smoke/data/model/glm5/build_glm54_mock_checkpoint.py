"""Assemble the GLM-5.4 mock checkpoint without copying donor tensor data."""

import argparse
import filecmp
import json
import math
import os
from pathlib import Path
import struct
import tempfile


INDEX_NAME = "model.safetensors.index.json"
VISION_LINK = "vision-glm53-flash.safetensors"
VISION_PREFIX = "model.visual."
MEDIA_TOKENS = {
    "image_start_token_id": (154830, "<|begin_of_image|>"),
    "image_end_token_id": (154831, "<|end_of_image|>"),
    "video_start_token_id": (154832, "<|begin_of_video|>"),
    "video_end_token_id": (154833, "<|end_of_video|>"),
    "image_token_id": (154854, "<|image|>"),
    "video_token_id": (154855, "<|video|>"),
}


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, contents):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(contents, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def safetensors_header(path):
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"Truncated safetensors header in {path}")
        (header_size,) = struct.unpack("<Q", raw_size)
        if not 0 < header_size <= 100_000_000:
            raise ValueError(f"Invalid safetensors header size in {path}: {header_size}")
        return json.loads(handle.read(header_size)), path.stat().st_size - 8 - header_size


def inspect_donors(text_donor, vision_donor):
    text_config = read_json(text_donor / "config.json")
    flash_config = read_json(vision_donor / "config.json")
    if (
        text_config["architectures"] != ["GlmMoeDsaForCausalLM"]
        or text_config["num_hidden_layers"] != 78
        or text_config["num_nextn_predict_layers"] != 1
        or text_config["hidden_size"] != 6144
        or text_config["quantization_config"]["expert_dtype"] != "fp4"
        or text_config["quantization_config"]["scale_fmt"] != "ue8m0"
    ):
        raise ValueError("Unexpected GLM-5.3 FP4Moe-FP8ue8m0 text donor")
    vision_config = flash_config["vision_config"]
    if (
        flash_config["architectures"] != ["Glm5NextForConditionalGeneration"]
        or vision_config["out_hidden_size"] != 4096
        or vision_config["depth"] != 24
    ):
        raise ValueError("Unexpected GLM-5.3-Flash ViT donor")

    tokenizer_path = text_donor / "tokenizer.json"
    if not filecmp.cmp(tokenizer_path, vision_donor / "tokenizer.json", shallow=False):
        raise ValueError("Donor tokenizers differ")
    tokenizer_config = text_donor / "tokenizer_config.json"
    if not filecmp.cmp(
        tokenizer_config, vision_donor / "tokenizer_config.json", shallow=False
    ):
        raise ValueError("Donor tokenizer configurations differ")
    token_text = {
        token["id"]: token["content"] for token in read_json(tokenizer_path)["added_tokens"]
    }
    template = (vision_donor / "chat_template.jinja").read_text(encoding="utf-8")
    for key, (token_id, expected_token) in MEDIA_TOKENS.items():
        if (
            flash_config[key] != token_id
            or token_text.get(token_id) != expected_token
            or expected_token not in template
        ):
            raise ValueError(f"Donor media token mismatch at {key}")

    text_index = read_json(text_donor / INDEX_NAME)
    text_map = text_index["weight_map"]
    if not {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.77.input_layernorm.weight",
        "model.layers.78.input_layernorm.weight",
    }.issubset(text_map):
        raise ValueError("Text donor lacks target or MTP weights")
    text_shards = sorted(set(text_map.values()))
    if len(text_shards) != 282 or any(key.startswith(VISION_PREFIX) for key in text_map):
        raise ValueError("Unexpected text donor shard layout")

    flash_map = read_json(vision_donor / INDEX_NAME)["weight_map"]
    vision_keys = {key for key in flash_map if key.startswith(VISION_PREFIX)}
    vision_shards = {flash_map[key] for key in vision_keys}
    if len(vision_keys) != 347 or len(vision_shards) != 1:
        raise ValueError("Expected 347 ViT tensors in one Flash shard")
    (vision_shard,) = vision_shards
    if Path(vision_shard).name != vision_shard:
        raise ValueError("Expected a local Flash ViT shard name")
    vision_path = vision_donor / vision_shard
    header, payload_size = safetensors_header(vision_path)
    stored_vision_keys = {key for key in header if key.startswith(VISION_PREFIX)}
    if stored_vision_keys != vision_keys:
        raise ValueError("Flash ViT index does not match the physical shard")
    foreign_keys = set(header) - vision_keys - {"__metadata__"}
    if foreign_keys.intersection(text_map):
        raise ValueError("Unindexed Flash text keys collide with the text checkpoint")
    vision_bytes = 0
    for key in vision_keys:
        entry = header[key]
        start, end = entry["data_offsets"]
        if (
            entry["dtype"] != "BF16"
            or not 0 <= start < end <= payload_size
            or end - start != math.prod(entry["shape"]) * 2
        ):
            raise ValueError(f"Invalid BF16 ViT tensor descriptor: {key}")
        vision_bytes += end - start

    assets = {name: text_donor / name for name in text_shards}
    assets.update(
        {
            VISION_LINK: vision_path,
            "tokenizer.json": tokenizer_path,
            "tokenizer_config.json": tokenizer_config,
            "processor_config.json": vision_donor / "processor_config.json",
            "chat_template.jinja": vision_donor / "chat_template.jinja",
        }
    )
    for name, source in assets.items():
        if Path(name).name != name or not source.is_file():
            raise ValueError(f"Invalid checkpoint asset: {source}")

    config = {
        **text_config,
        "glm54_mock_vit": True,
        "glm5_loop_layer_range": [26, 54],
        "vision_config": vision_config,
    }
    config.update({key: flash_config[key] for key in MEDIA_TOKENS})
    index = {
        **text_index,
        "metadata": {
            **text_index["metadata"],
            "total_size": text_index["metadata"]["total_size"] + vision_bytes,
        },
        "weight_map": {
            **text_map,
            **dict.fromkeys(sorted(vision_keys), VISION_LINK),
        },
    }
    return config, index, assets


def verify(checkpoint, config, index, assets):
    if read_json(checkpoint / "config.json") != config:
        raise ValueError(f"Unexpected config in {checkpoint}")
    if read_json(checkpoint / INDEX_NAME) != index:
        raise ValueError(f"Unexpected weight index in {checkpoint}")
    for name, source in assets.items():
        link = checkpoint / name
        if not link.is_symlink() or link.resolve(strict=True) != source.resolve(strict=True):
            raise ValueError(f"Invalid checkpoint symlink: {link}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--text-donor", type=Path, required=True)
    parser.add_argument("--vision-donor", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.verify_only and args.plan:
        parser.error("--verify-only and --plan cannot be used together")
    destination = args.destination.absolute()
    config, index, assets = inspect_donors(
        args.text_donor.resolve(strict=True),
        args.vision_donor.resolve(strict=True),
    )
    if args.verify_only:
        verify(destination, config, index, assets)
        status = "Verified checkpoint"
    elif args.plan:
        status = "Planned checkpoint"
    else:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Checkpoint destination already exists: {destination}")
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.", dir=destination.parent
        ) as staging_root:
            staged = Path(staging_root) / destination.name
            staged.mkdir()
            for name, source in assets.items():
                (staged / name).symlink_to(source)
            write_json(staged / "config.json", config)
            write_json(staged / INDEX_NAME, index)
            verify(staged, config, index, assets)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Checkpoint destination already exists: {destination}")
            os.rename(staged, destination)
        status = "Published checkpoint"

    print(f"{status}: {destination}")
    print(f"Tensor keys: {len(index['weight_map'])}, total indexed bytes: {index['metadata']['total_size']}")
    print(f"Donor links: {len(assets)}, 78 text layers + one MTP layer, 347 BF16 ViT tensors")


if __name__ == "__main__":
    main()
