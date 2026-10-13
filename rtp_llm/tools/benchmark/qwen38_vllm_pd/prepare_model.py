"""Create metadata for an isolated, source FP8, NVFP4 MoE baseline."""

import argparse
import json
from pathlib import Path


def prepare_model(source: Path, destination: Path) -> dict:
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen4_exp":
        raise ValueError("This benchmark only supports qwen4_exp")
    if config["quantization_config"].get("weight_block_size") != [128, 128]:
        raise ValueError("Official block128 FP8 source is required")
    if destination.exists():
        raise FileExistsError(destination)
    text = config.get("text_config", config)
    quantized = {}
    for prefix, count in (
        ("model.language_model", text["num_hidden_layers"]),
        ("mtp", 1),
    ):
        for index in range(count):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                quantized[f"{prefix}.layers.{index}.mlp.experts.{projection}"] = dict(
                    quant_algo="NVFP4", group_size=16
                )
    # PLE uses its native serialized FP8 method and the released global scale.
    text["ple_embedding_dtype"] = "float8_e4m3fn"
    config["quantization_config"] = dict(
        quant_method="modelopt",
        quant_algo="MIXED_PRECISION",
        group_size=16,
        kv_cache_scheme=dict(type="float", num_bits=8),
        quantized_layers=quantized,
        ignore=[],
    )
    destination.mkdir(parents=True)
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    for path in source.iterdir():
        if (
            path.is_file()
            and not path.name.endswith(".safetensors")
            and path.name
            not in (
                "config.json",
                "hf_quant_config.json",
                "model.safetensors.index.json",
            )
        ):
            (destination / path.name).symlink_to(path.resolve())
    return dict(
        source=str(source),
        model=str(destination),
        expert_format="NVFP4 block16 E4M3 scales",
        dense_format="online FP8",
        main_vocab_device="cuda",
        lm_head_device="cuda",
        ple_format="serialized FP8 pinned CPU/UVA",
        kv_dtype="fp8_e4m3",
        scope="metadata only; use qwen38_fp8_nvfp4_pd loader and isolated plugin",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_model(args.source, args.destination), indent=2))
