"""Export the pinned Kev-4B adapter as a text CausalLM plus an fp32 pointer head.

Run this file directly; it does not import the RTP runtime. Suggested environment:
torch>=2.6,<2.9, transformers==5.17.0, peft==0.21.0, safetensors, huggingface_hub.
A local --base must contain the base revision named below. Its weights are not
authenticated by this tool; the Kev adapter and head always are.
"""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

SOURCE_REPO = "jaredpalmer/kev-4b"
SOURCE_REVISION = "6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
SOURCE_SHA256 = {
    "head.pt": "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c",
    "adapter_config.json": "8a05dfd6c5e7f61a62e6db5d8de8093a8dbbd105aa7ee7b5e7b73f6249f45617",
    "adapter_model.safetensors": "90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2",
}
SPECIAL_TOKENS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
SPECIAL_TOKEN_IDS = [248060, 248061, 248049, 248050, 248062]
HEAD_SHAPES = {
    "q.weight": (256, 2560),
    "q.bias": (256,),
    "k.weight": (256, 2560),
    "k.bias": (256,),
}
HEAD_FIELDS = {
    "args",
    "suite_sha256",
    "init_source",
    "temperature_fit",
    "base",
    "head",
    "base_revision",
    "lora",
    "head_dim",
    "option_isolation",
    "special_embeddings",
    "weights_dtype",
    "temperature",
    "holdout",
    "lora_placement",
    "weights",
}


def _dependencies():
    try:
        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Export requires torch>=2.6,<2.9, transformers==5.17.0, "
            "peft==0.21.0, safetensors and huggingface_hub"
        ) from exc
    return SimpleNamespace(
        torch=torch,
        snapshot_download=snapshot_download,
        PeftModel=PeftModel,
        save_file=save_file,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
    )


def _checkpoint_path(checkpoint, deps):
    local = Path(checkpoint).expanduser()
    if local.is_dir():
        return local.resolve()
    repo, separator, revision = str(checkpoint).partition("@")
    if repo.lower() != SOURCE_REPO or (separator and revision != SOURCE_REVISION):
        raise ValueError(f"Only {SOURCE_REPO}@{SOURCE_REVISION} is supported")
    return Path(
        deps.snapshot_download(
            SOURCE_REPO,
            revision=SOURCE_REVISION,
            allow_patterns=list(SOURCE_SHA256),
        )
    )


def _verified_head_bytes(checkpoint):
    # Verify before torch.load, including locally supplied/cached pickle files.
    head_bytes = None
    for name, expected in SOURCE_SHA256.items():
        path = checkpoint / name
        if name == "head.pt":
            head_bytes = path.read_bytes()
            actual = hashlib.sha256(head_bytes).hexdigest()
        else:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"{name} does not match the pinned Kev-4B SHA256")
    return head_bytes


def _validate_head(meta, torch):
    if not isinstance(meta, dict) or set(meta) - HEAD_FIELDS:
        raise ValueError("Unsupported head metadata fields")
    expected = {
        "base": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "lora": 16,
        "head_dim": 256,
        "option_isolation": False,
        "special_embeddings": False,
        "weights_dtype": "fp32",
        "lora_placement": "full",
    }
    for key, value in expected.items():
        if type(meta.get(key)) is not type(value) or meta[key] != value:
            raise ValueError(f"Unsupported head option {key}: {meta.get(key)!r}")
    if meta.get("weights", "lora") != "lora":
        raise ValueError("Only a LoRA source checkpoint is supported")
    temperature = meta.get("temperature")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and strictly positive")
    head = meta.get("head")
    if not isinstance(head, dict) or set(head) != set(HEAD_SHAPES):
        raise ValueError("head must contain exactly q/k weight and bias")
    for name, shape in HEAD_SHAPES.items():
        tensor = head[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"Invalid fp32 head tensor {name}; expected {shape}")


def _validate_backbone(model):
    expected = {
        "model_type": "qwen3_5_text",
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 32,
        "vocab_size": 248320,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "tie_word_embeddings": True,
        "rms_norm_eps": 1e-6,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    }
    expected["layer_types"] *= 8
    for key, value in expected.items():
        if getattr(model.config, key, None) != value:
            raise ValueError(f"Base model has an unsupported {key}")


def _merge_backbone(checkpoint, base, deps):
    load = {"dtype": deps.torch.float32, "attn_implementation": "eager"}
    if base is None:
        base_source = BASE_MODEL
        load["revision"] = BASE_REVISION
    else:
        base_source = str(Path(base).expanduser().resolve(strict=True))
        if not Path(base_source).is_dir():
            raise ValueError("--base must be a local directory at the pinned revision")
        load["local_files_only"] = True
    model = deps.AutoModelForCausalLM.from_pretrained(base_source, **load)
    _validate_backbone(model)
    adapted = deps.PeftModel.from_pretrained(
        model.model, str(checkpoint), is_trainable=False
    )
    lora = [
        parameter
        for name, parameter in adapted.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    ]
    if len(lora) != 496 or any(p.dtype != deps.torch.float32 for p in lora):
        raise ValueError("Expected all 496 pinned fp32 LoRA tensors to be loaded")
    # Do not cast the adapter to bf16: W + alpha/r * B@A must be merged in fp32.
    with deps.torch.no_grad():
        model.model = adapted.merge_and_unload(safe_merge=True)
    model.eval()
    model.to(deps.torch.bfloat16)
    model.config.use_cache = False
    return model, base_source


def convert(checkpoint, output, base=None):
    """Leave a failed output for diagnosis, without its enabling kev_config.json."""
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must not exist: {output}")
    deps = _dependencies()
    checkpoint = _checkpoint_path(checkpoint, deps)
    head_bytes = _verified_head_bytes(checkpoint)
    meta = deps.torch.load(
        io.BytesIO(head_bytes), map_location="cpu", weights_only=True
    )
    _validate_head(meta, deps.torch)
    output.mkdir(parents=True, exist_ok=False)

    model, base_source = _merge_backbone(checkpoint, base, deps)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    # Match Kev's loader, not a possibly different tokenizer copied by a caller.
    tokenizer_load = (
        {"revision": BASE_REVISION} if base is None else {"local_files_only": True}
    )
    tokenizer = deps.AutoTokenizer.from_pretrained(base_source, **tokenizer_load)
    ids = [tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS]
    if ids != SPECIAL_TOKEN_IDS:
        raise ValueError("Base tokenizer does not have the pinned delimiter IDs")
    tokenizer.save_pretrained(output)
    head = {
        name: tensor.detach().cpu().float().contiguous()
        for name, tensor in meta["head"].items()
    }
    deps.save_file(
        head, str(output / "kev_head.safetensors"), metadata={"format": "pt"}
    )
    manifest = {
        "format_version": 1,
        "head_dim": 256,
        "hidden_size": 2560,
        "temperature": meta["temperature"],
        "special_token_ids": ids,
        "max_options": 32,
        "max_state_tokens": 65536,
        "max_row_tokens": 73728,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "source_revision": SOURCE_REVISION,
        "source_repo": SOURCE_REPO,
        "source_sha256": SOURCE_SHA256,
        "base_source": base_source,
        "base_is_local_override": base is not None,
        "backbone_dtype": "bfloat16",
        "head_dtype": "float32",
        "add_special_tokens": False,
    }
    # The runtime uses the manifest as the opt-in marker. Publish it last.
    partial = output / "kev_config.json.partial"
    partial.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, output / "kev_config.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=SOURCE_REPO,
        help="Pinned official HF repo or a local directory with its exact files",
    )
    parser.add_argument(
        "--base", help="Local base snapshot at the pinned base revision"
    )
    parser.add_argument("--output", required=True, help="New output directory")
    args = parser.parse_args()
    convert(args.checkpoint, args.output, args.base)
    print(f"Exported Kev-4B to {args.output}")


if __name__ == "__main__":
    main()
