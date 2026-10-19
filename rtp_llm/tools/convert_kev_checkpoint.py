"""Export a compatible Kev-4B adapter as a text CausalLM plus an fp32 pointer head.

Run this file directly; it does not import the RTP runtime. Suggested environment:
torch>=2.6,<2.9, transformers==5.17.0, peft==0.21.0, safetensors, huggingface_hub.
The source metadata selects the base revision. Hashes record the files actually
used; they do not authenticate a release. A local --base overrides resolution.
"""

import argparse
import hashlib
import io
import json
import logging
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

SOURCE_REPO = "jaredpalmer/kev-4b"
SOURCE_FILES = ("head.pt", "adapter_config.json", "adapter_model.safetensors")
BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
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
HEAD_DEFAULTS = {
    "weights_dtype": "fp32",
    "lora_placement": "full",
    "head_dim": 256,
    "option_isolation": False,
    "special_embeddings": False,
    "temperature": 1.0,
    "base_revision": None,
    "weights": "lora",
}
logger = logging.getLogger(__name__)


def _dependencies():
    try:
        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from safetensors import safe_open
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
        safe_open=safe_open,
        save_file=save_file,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
    )


def _parse_source(checkpoint):
    """Recognize the supported Hub repo without restricting its revision."""
    source = str(checkpoint)
    if "://" in source:
        url = urlsplit(source)
        if url.scheme != "https" or url.netloc != "huggingface.co":
            return None
        path = unquote(url.path).strip("/")
        if path.lower() != SOURCE_REPO and not path.lower().startswith(
            SOURCE_REPO + "/"
        ):
            return None
        if url.query or url.fragment:
            raise ValueError("Kev URL must not contain a query or fragment")
        if path.lower() == SOURCE_REPO:
            return SOURCE_REPO, None
        prefix = SOURCE_REPO + "/tree/"
        if not path.lower().startswith(prefix):
            raise ValueError("Kev URL must name the repo or /tree/<revision>")
        revision = path[len(prefix) :]
    else:
        repo, separator, revision = source.partition("@")
        if repo.lower() != SOURCE_REPO:
            return None
        if not separator:
            return SOURCE_REPO, None
    if (
        not revision
        or any(char.isspace() for char in revision)
        or any(char in revision for char in ("@", "?", "#", "\\"))
        or any(part in ("", ".", "..") for part in revision.split("/"))
    ):
        raise ValueError("Kev source revision must be a nonempty Hub revision")
    return SOURCE_REPO, revision


def _checkpoint_path(checkpoint, deps):
    local = Path(checkpoint).expanduser()
    if local.is_dir():
        return local.resolve()
    source = _parse_source(checkpoint)
    if source is None:
        raise ValueError(f"Expected a local Kev directory or {SOURCE_REPO}[@revision]")
    repo, revision = source
    return Path(
        deps.snapshot_download(
            repo, revision=revision, allow_patterns=list(SOURCE_FILES)
        )
    )


def _source_revision(path):
    """Return a commit only when the path is an HF model cache snapshot."""
    path = Path(path).expanduser().resolve()
    repo_parts = path.parent.parent.name.split("--")
    if (
        path.is_dir()
        and path.parent.name == "snapshots"
        and len(repo_parts) == 3
        and repo_parts[0] == "models"
        and all(repo_parts[1:])
        and re.fullmatch(r"[0-9a-f]{40}", path.name)
    ):
        return path.name
    return None


def _source_manifest(source, hashes, meta, resolved_base_revision=None):
    source = Path(source).expanduser().resolve()
    revision = _source_revision(source)
    repo = "/".join(source.parent.parent.name.split("--")[1:]) if revision else None
    return {
        "base_model": meta["base"],
        "base_revision": (
            resolved_base_revision
            if resolved_base_revision is not None
            else meta["base_revision"]
        ),
        "source_revision": revision,
        "source_repo": repo,
        "source_sha256": dict(hashes),
    }


def _read_source(checkpoint):
    """Read the head and record actual hashes; malformed files still fail closed."""
    head_bytes = None
    hashes = {}
    for name in SOURCE_FILES:
        path = Path(checkpoint) / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Incomplete Kev checkpoint: missing or empty {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            prefix = stream.read(256)
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Kev checkpoint contains a Git LFS pointer: {name}")
            stream.seek(0)
            if name == "head.pt":
                head_bytes = stream.read()
                digest.update(head_bytes)
            else:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        hashes[name] = digest.hexdigest()
    return head_bytes, hashes


def _normalize_head(meta):
    """Apply the official loader's legacy defaults without losing extra fields."""
    if not isinstance(meta, dict):
        raise ValueError("head metadata must be a dictionary")
    normalized = dict(meta)
    defaults = {key: value for key, value in HEAD_DEFAULTS.items() if key not in meta}
    normalized.update(defaults)
    if defaults:
        logger.info("Applied legacy Kev head defaults: %s", defaults)
    return normalized


def _validate_head(meta, torch):
    if not isinstance(meta, dict):
        raise ValueError("head metadata must be a dictionary")
    expected = {
        "base": BASE_MODEL,
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
    revision = meta.get("base_revision")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("base_revision must be a nonempty revision string or None")
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


def _validate_adapter_config(checkpoint):
    config = json.loads(
        (checkpoint / "adapter_config.json").read_text(encoding="utf-8")
    )
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must contain an object")
    required = {
        "base_model_name_or_path": BASE_MODEL,
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
    }
    for key, value in required.items():
        if type(config.get(key)) is not type(value) or config[key] != value:
            raise ValueError(f"Unsupported adapter option {key}: {config.get(key)!r}")
    defaults = {
        "bias": "none",
        "fan_in_fan_out": False,
        "use_dora": False,
        "use_rslora": False,
        "lora_bias": False,
        "use_qalora": False,
        "rank_pattern": {},
        "alpha_pattern": {},
        "modules_to_save": None,
        "alora_invocation_tokens": None,
        "layer_replication": None,
        "trainable_token_indices": None,
        "target_parameters": None,
    }
    for key, value in defaults.items():
        actual = config.get(key, value)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"Unsupported adapter option {key}: {actual!r}")


def _resolve_base(meta, base, deps):
    if base is None:
        # snapshot_download resolves a mutable ref once. Both loaders below use
        # this exact local snapshot, so model and tokenizer cannot resolve apart.
        source = Path(
            deps.snapshot_download(meta["base"], revision=meta["base_revision"])
        ).resolve()
        revision = _source_revision(source)
        if revision is None:
            raise ValueError("Hub base did not resolve to a commit snapshot")
    else:
        source = Path(base).expanduser().resolve(strict=True)
        if not source.is_dir():
            raise ValueError("--base must be a local directory")
        # A local override is not authenticated as the head's declared revision.
        revision = None
    return str(source), revision


def _validate_adapter_tensors(checkpoint, backbone, deps):
    """Check saved tensors before PEFT can initialize missing adapter weights."""
    parameters = dict(backbone.named_parameters())
    expected = {}
    for layer in range(32):
        projections = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        if layer % 4 == 3:
            projections += [
                f"self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            ]
        else:
            projections += [
                f"linear_attn.{name}"
                for name in (
                    "in_proj_a",
                    "in_proj_b",
                    "in_proj_qkv",
                    "in_proj_z",
                    "out_proj",
                )
            ]
        for projection in projections:
            target = f"layers.{layer}.{projection}"
            weight = parameters.get(target + ".weight")
            if weight is None or len(weight.shape) != 2:
                raise ValueError(
                    f"Base model is missing a supported projection: {target}"
                )
            output_size, input_size = weight.shape
            prefix = f"base_model.model.{target}"
            expected[prefix + ".lora_A.weight"] = (16, input_size)
            expected[prefix + ".lora_B.weight"] = (output_size, 16)
    with deps.safe_open(
        str(checkpoint / "adapter_model.safetensors"), framework="pt", device="cpu"
    ) as adapter:
        actual = set(adapter.keys())
        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            unexpected = sorted(actual - set(expected))
            raise ValueError(
                f"Source adapter tensor keys differ: missing={missing}, unexpected={unexpected}"
            )
        for name, shape in expected.items():
            tensor = adapter.get_tensor(name)
            if tuple(tensor.shape) != shape or tensor.dtype != deps.torch.float32:
                raise ValueError(
                    f"Invalid source fp32 LoRA tensor {name}; expected {shape}"
                )
    return expected


def _merge_backbone(checkpoint, base_source, deps):
    model = deps.AutoModelForCausalLM.from_pretrained(
        base_source,
        dtype=deps.torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    )
    _validate_backbone(model)
    expected = _validate_adapter_tensors(checkpoint, model.model, deps)
    adapted = deps.PeftModel.from_pretrained(
        model.model, str(checkpoint), is_trainable=False
    )
    lora = {
        name.replace(".lora_A.default.", ".lora_A.").replace(
            ".lora_B.default.", ".lora_B."
        ): parameter
        for name, parameter in adapted.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    if set(lora) != set(expected) or any(
        p.dtype != deps.torch.float32 or tuple(p.shape) != expected[name]
        for name, p in lora.items()
    ):
        raise ValueError(
            "Expected all 496 fp32 LoRA tensors to match the source targets"
        )
    # Do not cast the adapter to bf16: W + alpha/r * B@A must be merged in fp32.
    with deps.torch.no_grad():
        model.model = adapted.merge_and_unload(safe_merge=True)
    model.eval()
    model.to(deps.torch.bfloat16)
    model.config.use_cache = False
    return model


def convert(checkpoint, output, base=None):
    """Leave a failed output for diagnosis, without its enabling kev_config.json."""
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must not exist: {output}")
    deps = _dependencies()
    checkpoint = _checkpoint_path(checkpoint, deps)
    head_bytes, hashes = _read_source(checkpoint)
    meta = _normalize_head(
        deps.torch.load(io.BytesIO(head_bytes), map_location="cpu", weights_only=True)
    )
    _validate_head(meta, deps.torch)
    _validate_adapter_config(checkpoint)
    base_source, base_revision = _resolve_base(meta, base, deps)
    output.mkdir(parents=True, exist_ok=False)

    model = _merge_backbone(checkpoint, base_source, deps)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    # Match Kev's loader, not a possibly different tokenizer copied by a caller.
    tokenizer = deps.AutoTokenizer.from_pretrained(base_source, local_files_only=True)
    ids = [tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS]
    if ids != SPECIAL_TOKEN_IDS:
        raise ValueError("Base tokenizer does not have the expected delimiter IDs")
    tokenizer.save_pretrained(output)
    head = {
        name: tensor.detach().cpu().float().contiguous()
        for name, tensor in meta["head"].items()
    }
    deps.save_file(
        head, str(output / "kev_head.safetensors"), metadata={"format": "pt"}
    )
    if _read_source(checkpoint)[1] != hashes:
        raise ValueError(
            "Kev source files changed during conversion; retry with stable files"
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
        **_source_manifest(checkpoint, hashes, meta, base_revision),
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
        help="Official HF repo, optionally @revision or /tree/revision, or a local Kev directory",
    )
    parser.add_argument(
        "--base", help="Local base directory overriding the source metadata's revision"
    )
    parser.add_argument("--output", required=True, help="New output directory")
    args = parser.parse_args()
    convert(args.checkpoint, args.output, args.base)
    print(f"Exported Kev-4B to {args.output}")


if __name__ == "__main__":
    main()
