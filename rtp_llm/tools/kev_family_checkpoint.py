"""Export any checkpoint of the `jaredpalmer/kev-*` family as an RTP text model.

The family shares one serving contract: a causal text backbone without an LM
head, an fp32 pointer head (``q``/``k`` over ``hidden_size -> head_dim``), five
delimiter tokens, and a checkpoint temperature. Members differ in backbone
(Qwen2.5 / Qwen3 / Qwen3.5 / Qwen3.8), adapter versus full weights, hidden size
and context limit, so this module derives every shape and limit from the source
metadata and the base config instead of hard-coding one model.

Only ``torch``, ``safetensors``, ``tokenizers``, ``huggingface_hub`` and
``filelock`` are required: no Transformers classes and no PEFT. LoRA adapters are
merged tensor by tensor in fp32 and rounded once to bf16; full-weight checkpoints
are re-linked or copied unchanged.

Run directly for an offline export::

    python rtp_llm/tools/kev_family_checkpoint.py \
        --checkpoint jaredpalmer/kev-9b --output /path/to/export
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

FAMILY_AUTHOR = "jaredpalmer"
FAMILY_PREFIX = "jaredpalmer/kev-"
SPECIAL_TOKENS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
HEAD_NAMES = ("q.weight", "q.bias", "k.weight", "k.bias")
LORA_FILES = ("head.pt", "adapter_config.json", "adapter_model.safetensors")
FULL_AUX_FILES = ("head.pt", "config.json", "model.safetensors.index.json")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)
BASE_PATTERNS = (
    "config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    *TOKENIZER_FILES,
)
SOURCE_PATTERNS = (
    "head.pt",
    "adapter_config.json",
    "adapter_model.safetensors",
    "config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    *TOKENIZER_FILES,
)
EXCLUDED_PREFIXES = ("visual.", "vision.", "mtp.", "model.visual.", "model.vision.")
EXCLUDED_NAMES = {"lm_head.weight", "model.lm_head.weight"}
SHARD_BYTES = 4 * 1024**3
PREPARER_VERSION = 1
INTEGRITY_FILE = "kev_family_prepare.json"
ARCHITECTURES = {
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "qwen3_5": "Qwen3_5ForCausalLM",
    "qwen3_5_text": "Qwen3_5ForCausalLM",
    "qwen3_5_moe": "Qwen3_5MoeForCausalLM",
    "qwen3_5_moe_text": "Qwen3_5MoeForCausalLM",
}
HEAD_DEFAULTS = {
    "base_revision": None,
    "head_dim": 256,
    "lora": 16,
    "lora_placement": "full",
    "option_isolation": False,
    "special_embeddings": False,
    "temperature": 1.0,
    "weights": "lora",
    "weights_dtype": "fp32",
}
MAX_OPTIONS = 255
MAX_STATE_TOKENS = 65536
MAX_ROW_TOKENS = 73728
MIN_BRANCH_TOKENS = 8192


def _dependencies():
    import torch
    from filelock import FileLock
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tokenizers import Tokenizer

    return SimpleNamespace(
        torch=torch,
        FileLock=FileLock,
        snapshot_download=snapshot_download,
        safe_open=safe_open,
        save_file=save_file,
        Tokenizer=Tokenizer,
    )


def is_family_repo(repo: str) -> bool:
    return repo.lower().startswith(FAMILY_PREFIX) and len(repo) > len(FAMILY_PREFIX)


def parse_source(checkpoint) -> tuple[str, str | None] | None:
    """Recognize an HF family repo (optionally pinned) or return None."""
    source = str(checkpoint).strip()
    if "://" in source:
        url = urlsplit(source)
        if url.scheme != "https" or url.netloc != "huggingface.co":
            return None
        parts = [unquote(part) for part in url.path.split("/") if part]
        if len(parts) < 2:
            return None
        repo = "/".join(parts[:2])
        revision = None
        if len(parts) == 4 and parts[2] == "tree":
            revision = parts[3]
        elif len(parts) != 2:
            return None
    elif source.count("/") == 1:
        repo, _, revision = source.partition("@")
        revision = revision or None
    else:
        return None
    if not is_family_repo(repo):
        return None
    if revision is not None and (
        not revision
        or any(char.isspace() for char in revision)
        or any(char in revision for char in ("@", "?", "#", "\\"))
        or any(part in ("", ".", "..") for part in revision.split("/"))
    ):
        raise ValueError("Kev source revision must be a nonempty Hub revision")
    return repo, revision


def source_revision(path: Path) -> str | None:
    """Return a commit only when the path is an HF model cache snapshot."""
    path = Path(path).expanduser().resolve()
    parts = path.parent.parent.name.split("--")
    if (
        path.is_dir()
        and path.parent.name == "snapshots"
        and len(parts) == 3
        and parts[0] == "models"
        and all(parts[1:])
        and re.fullmatch(r"[0-9a-f]{40}", path.name)
    ):
        return path.name
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _shard_files(path: Path) -> list[Path]:
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = _json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid safetensors index: {index}")
        names = sorted(set(weight_map.values()))
    else:
        names = sorted(p.name for p in path.glob("*.safetensors"))
    files = [path / name for name in names]
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise ValueError(f"Checkpoint is missing shards: {missing}")
    if not files:
        raise ValueError(f"No safetensors shards under {path}")
    return files


def _inventory(path: Path, deps) -> tuple[dict, dict]:
    """Map every tensor name to its shard, rejecting duplicate names."""
    locations: dict[str, str] = {}
    for shard in _shard_files(path):
        with deps.safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in locations:
                    raise ValueError(f"Duplicate tensor {key} in {shard}")
                locations[key] = shard.name
    return locations


def _read_meta(source: Path, deps) -> dict:
    head = source / "head.pt"
    if not head.is_file() or head.stat().st_size == 0:
        raise ValueError("Incomplete Kev checkpoint: missing or empty head.pt")
    with head.open("rb") as stream:
        prefix = stream.read(256)
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Kev checkpoint contains a Git LFS pointer: head.pt")
        stream.seek(0)
        meta = deps.torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(meta, dict):
        raise ValueError("head metadata must be a dictionary")
    normalized = dict(meta)
    defaults = {key: value for key, value in HEAD_DEFAULTS.items() if key not in meta}
    normalized.update(defaults)
    return normalized


def _validate_meta(meta: dict, hidden_size: int | None, deps) -> None:
    base = meta.get("base")
    if not isinstance(base, str) or not base:
        raise ValueError("head metadata needs a nonempty base")
    revision = meta.get("base_revision")
    if revision is not None and (
        not isinstance(revision, str) or not revision.strip()
    ):
        raise ValueError("base_revision must be a nonempty revision string or None")
    if meta["option_isolation"] is not False:
        raise ValueError("Kev row execution does not support option_isolation")
    if meta["special_embeddings"] is not False:
        raise ValueError("Kev row execution does not support special embeddings")
    if meta.get("lora_placement", "full") != "full":
        raise ValueError("Only full-row LoRA placement is supported")
    if meta["weights"] not in ("lora", "full"):
        raise ValueError(f"Unsupported Kev weights: {meta['weights']!r}")
    if meta["weights_dtype"] not in ("fp32", "bf16"):
        raise ValueError(f"Unsupported Kev weights_dtype: {meta['weights_dtype']!r}")
    head_dim = meta.get("head_dim")
    if type(head_dim) is not int or head_dim <= 0:
        raise ValueError("head_dim must be a positive integer")
    rank = meta.get("lora")
    if type(rank) is not int or rank < 0:
        raise ValueError("lora rank must be a nonnegative integer")
    if meta["weights"] == "lora" and rank <= 0:
        raise ValueError("A LoRA checkpoint needs a positive rank")
    temperature = meta.get("temperature")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and strictly positive")
    head = meta.get("head")
    if not isinstance(head, dict) or set(head) != set(HEAD_NAMES):
        raise ValueError("head must contain exactly q/k weight and bias")
    if hidden_size is None:
        return
    expected = {
        "q.weight": (head_dim, hidden_size),
        "q.bias": (head_dim,),
        "k.weight": (head_dim, hidden_size),
        "k.bias": (head_dim,),
    }
    for name, shape in expected.items():
        tensor = head[name]
        if (
            not isinstance(tensor, deps.torch.Tensor)
            or tuple(tensor.shape) != shape
            or tensor.dtype != deps.torch.float32
            or not deps.torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"Invalid fp32 head tensor {name}; expected {shape}")


def _validate_adapter_config(source: Path, meta: dict) -> None:
    config = _json(source / "adapter_config.json")
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must contain an object")
    if config.get("peft_type") != "LORA":
        raise ValueError("Only a LoRA adapter is supported")
    if config.get("base_model_name_or_path") != meta["base"]:
        raise ValueError("Adapter base differs from the head metadata")
    if type(config.get("r")) is not int or config["r"] != meta["lora"]:
        raise ValueError("Adapter rank differs from the head metadata")
    if type(config.get("lora_alpha")) is not int or config["lora_alpha"] <= 0:
        raise ValueError("Adapter lora_alpha must be a positive integer")
    unsupported = {
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
    for key, value in unsupported.items():
        actual = config.get(key, value)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"Unsupported adapter option {key}: {actual!r}")


def _adapter_scale(source: Path, meta: dict) -> float:
    config = _json(source / "adapter_config.json")
    return float(config["lora_alpha"]) / float(config["r"])


def _text_prefix(locations: dict) -> str:
    keys = [
        key
        for key in locations
        if key not in EXCLUDED_NAMES and not key.startswith(EXCLUDED_PREFIXES)
    ]
    if not keys:
        raise ValueError("Checkpoint has no text tensors")
    if all(key.startswith("model.language_model.") for key in keys):
        return "model.language_model."
    if all(key.startswith("model.") for key in keys):
        return "model."
    raise ValueError("Unsupported backbone tensor prefix")


def _wanted_keys(locations: dict, prefix: str) -> dict[str, str]:
    wanted = {}
    for key, shard in locations.items():
        if key in EXCLUDED_NAMES or key.startswith(EXCLUDED_PREFIXES):
            continue
        if not key.startswith(prefix):
            continue
        if key.endswith("rotary_emb.inv_freq"):
            continue
        wanted[key] = shard
    if prefix + "embed_tokens.weight" not in wanted:
        raise ValueError("Backbone is missing embed_tokens")
    if prefix + "norm.weight" not in wanted:
        raise ValueError("Backbone is missing the final norm")
    return wanted


ADAPTER_PREFIX = "base_model.model."


def _merge_tensor(weight, lora_a, lora_b, rank: int, scale: float, deps):
    if (
        weight.ndim != 2
        or lora_a.ndim != 2
        or lora_b.ndim != 2
        or lora_a.dtype != deps.torch.float32
        or lora_b.dtype != deps.torch.float32
        or tuple(lora_a.shape)[0] != rank
        or tuple(lora_b.shape)[1] != rank
        or tuple(lora_a.shape)[1] != weight.shape[1]
        or tuple(lora_b.shape)[0] != weight.shape[0]
    ):
        raise ValueError("Invalid LoRA merge operands")
    with deps.torch.no_grad():
        merged = weight.float() + (lora_b.float() @ lora_a.float()) * scale
        if not deps.torch.isfinite(merged).all().item():
            raise ValueError("LoRA merge produced nonfinite values")
        result = merged.to(deps.torch.bfloat16).contiguous()
        if not deps.torch.isfinite(result).all().item():
            raise ValueError("LoRA merge overflows BF16")
        return result


def _export_lora(source: Path, base: Path, output: Path, meta: dict, deps) -> dict:
    from contextlib import ExitStack

    _validate_adapter_config(source, meta)
    base_config = _json(base / "config.json")
    text_config = _text_config(base_config)
    hidden = text_config["hidden_size"]
    _validate_meta(meta, hidden, deps)
    locations = _inventory(base, deps)
    prefix = _text_prefix(locations)
    wanted = _wanted_keys(locations, prefix)
    scale = _adapter_scale(source, meta)
    rank = meta["lora"]

    with ExitStack() as stack:
        adapter = stack.enter_context(
            deps.safe_open(
                str(source / "adapter_model.safetensors"),
                framework="pt",
                device="cpu",
            )
        )
        targets: dict[str, tuple[str, str]] = {}
        for key in adapter.keys():
            if not key.startswith(ADAPTER_PREFIX) or ".lora_" not in key:
                raise ValueError(f"Unexpected adapter tensor {key}")
            target = key[len(ADAPTER_PREFIX) :].rsplit(".lora_", 1)[0]
            pair = list(targets.get(target, ("", "")))
            pair[0 if key.endswith(".lora_A.weight") else 1] = key
            targets[target] = (pair[0], pair[1])
        missing = [target for target, pair in targets.items() if not all(pair)]
        if missing:
            raise ValueError(f"Incomplete LoRA pairs: {missing}")

        handles: dict[str, object] = {}
        pending, pending_bytes, total, weight_map = {}, 0, 0, {}
        counter = {"n": 0}

        def flush():
            nonlocal pending, pending_bytes
            if not pending:
                return
            counter["n"] += 1
            name = f"model-{counter['n']:05d}.safetensors"
            deps.save_file(pending, str(output / name), metadata={"format": "pt"})
            weight_map.update({key: name for key in pending})
            pending, pending_bytes = {}, 0

        used = set()
        for key in sorted(wanted):
            shard_name = wanted[key]
            handle = handles.get(shard_name)
            if handle is None:
                handle = stack.enter_context(
                    deps.safe_open(
                        str(base / shard_name), framework="pt", device="cpu"
                    )
                )
                handles[shard_name] = handle
            weight = handle.get_tensor(key)
            target = key[len(prefix) :].removesuffix(".weight")
            pair = targets.get(target)
            if pair is not None:
                a_key, b_key = pair
                weight = _merge_tensor(
                    weight,
                    adapter.get_tensor(a_key),
                    adapter.get_tensor(b_key),
                    rank,
                    scale,
                    deps,
                )
                used.add(target)
            else:
                weight = weight.to(deps.torch.bfloat16).contiguous()
            size = weight.numel() * weight.element_size()
            if pending_bytes + size > SHARD_BYTES:
                flush()
            pending[key] = weight
            pending_bytes += size
            total += size
            if pending_bytes >= SHARD_BYTES:
                flush()
        flush()

    unused = sorted(set(targets) - used)
    if unused:
        raise ValueError(f"Adapter targets are missing from the backbone: {unused}")
    _write_json(
        output / "model.safetensors.index.json",
        {"metadata": {"total_size": total}, "weight_map": weight_map},
    )
    _write_json(output / "config.json", text_config)
    _write_tokenizer(base, output, meta, deps)
    _write_head(output, meta, deps)
    return {
        "hidden_size": hidden,
        "max_position_embeddings": text_config.get("max_position_embeddings"),
    }


def _text_config(config: dict) -> dict:
    text = dict(config.get("text_config", config))
    for name in ("text_config", "vision_config", "quantization_config"):
        text.pop(name, None)
    model_type = text.get("model_type")
    architecture = ARCHITECTURES.get(model_type)
    if architecture is not None:
        text["architectures"] = [architecture]
    else:
        current = text.get("architectures")
        if not (
            isinstance(current, list)
            and current
            and str(current[0]).endswith("ForCausalLM")
        ):
            raise ValueError(f"Unsupported backbone model_type: {model_type!r}")
    text["dtype"] = "bfloat16"
    text["use_cache"] = False
    text.setdefault("bos_token_id", None)
    text.setdefault("pad_token_id", None)
    if "rope_parameters" in text:
        text["partial_rotary_factor"] = text["rope_parameters"].get(
            "partial_rotary_factor", text.get("partial_rotary_factor", 0.25)
        )
    text.pop("_name_or_path", None)
    return text


def _write_head(output: Path, meta: dict, deps) -> None:
    head = {
        name: tensor.detach().cpu().float().contiguous()
        for name, tensor in meta["head"].items()
    }
    deps.save_file(head, str(output / "kev_head.safetensors"), metadata={"format": "pt"})


def _tokenizer_ids(path: Path, deps) -> list[int]:
    tokenizer = deps.Tokenizer.from_file(str(path / "tokenizer.json"))
    ids = [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS]
    if any(value is None for value in ids) or len(set(ids)) != len(SPECIAL_TOKENS):
        raise ValueError("Tokenizer does not contain the five Kev delimiters")
    return list(ids)


def _write_tokenizer(base: Path, output: Path, meta: dict, deps) -> None:
    config = _json(base / "tokenizer_config.json")
    tokenizer = deps.Tokenizer.from_file(str(base / "tokenizer.json"))
    regex = config.get("pretokenize_regex")
    if isinstance(regex, str) and regex:
        from tokenizers import Regex, decoders, pre_tokenizers

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(regex), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
    added = config.get("added_tokens_decoder")
    if isinstance(added, dict):
        from tokenizers import AddedToken

        for token_id, attributes in sorted(added.items(), key=lambda p: int(p[0])):
            tokenizer.add_tokens([AddedToken(**attributes)])
            if tokenizer.token_to_id(attributes["content"]) != int(token_id):
                raise ValueError("Added token ID differs from the tokenizer config")
    tokenizer.save(str(output / "tokenizer.json"))
    for filename in TOKENIZER_FILES:
        if filename in ("tokenizer.json",) or not (base / filename).is_file():
            continue
        shutil.copyfile(base / filename, output / filename)
    config.pop("fix_mistral_regex", None)
    _write_json(output / "tokenizer_config.json", config)
    _tokenizer_ids(output, deps)


def _export_full(source: Path, output: Path, meta: dict, deps) -> dict:
    config = _json(source / "config.json")
    if "text_config" in config:
        raise ValueError("Composite full-weight checkpoints are not supported")
    text_config = _text_config(config)
    hidden = text_config["hidden_size"]
    _validate_meta(meta, hidden, deps)
    locations = _inventory(source, deps)
    prefix = _text_prefix(locations)
    wanted = _wanted_keys(locations, prefix)
    if set(wanted) != set(locations):
        raise ValueError("Full-weight checkpoint contains non-text tensors")
    index = _json(source / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Invalid safetensors index")
    if set(weight_map) != set(wanted):
        raise ValueError("Safetensors index does not match the text tensors")
    _write_json(output / "config.json", text_config)
    for shard in _shard_files(source):
        destination = output / shard.name
        try:
            os.link(shard, destination)
        except OSError:
            shutil.copy2(shard, destination)
    shutil.copyfile(
        source / "model.safetensors.index.json",
        output / "model.safetensors.index.json",
    )
    if not (source / "tokenizer.json").is_file():
        raise ValueError("Full-weight checkpoint is missing tokenizer.json")
    _write_tokenizer(source, output, meta, deps)
    _write_head(output, meta, deps)
    return {
        "hidden_size": hidden,
        "max_position_embeddings": text_config.get("max_position_embeddings"),
    }


def validate_export(path: Path) -> None:
    """Validate a prepared family export without loading model weights."""
    manifest = _json(path / "kev_config.json")
    required = {
        "format_version": 1,
        "backbone_dtype": "bfloat16",
        "head_dtype": "float32",
        "add_special_tokens": False,
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value for key, value in required.items()
    ):
        raise ValueError("Exported Kev manifest differs from the supported contract")
    for key in ("hidden_size", "head_dim", "temperature"):
        value = manifest.get(key)
        if value is None or (isinstance(value, bool)) or not isinstance(
            value, (int, float)
        ):
            raise ValueError(f"Invalid Kev manifest {key}")
    if not isinstance(manifest.get("base_model"), str) or not manifest["base_model"]:
        raise ValueError("Kev manifest needs base_model")
    if not (path / "config.json").is_file():
        raise ValueError("Exported checkpoint is missing config.json")
    config = _json(path / "config.json")
    if config.get("hidden_size") != manifest["hidden_size"]:
        raise ValueError("Kev manifest hidden_size differs from the config")
    if not (path / "kev_head.safetensors").is_file():
        raise ValueError("Exported checkpoint is missing kev_head.safetensors")
    ids = _tokenizer_ids(path, _dependencies())
    if ids != manifest.get("special_token_ids"):
        raise ValueError("Kev delimiter IDs differ from the manifest")
    locations = _inventory(path, _dependencies())
    if not locations:
        raise ValueError("Exported checkpoint has no tensors")


def _limits(info: dict) -> tuple[int, int]:
    max_positions = info.get("max_position_embeddings")
    row = MAX_ROW_TOKENS
    if type(max_positions) is int and max_positions > 0:
        row = min(row, max_positions)
    state = min(MAX_STATE_TOKENS, max(1, row - MIN_BRANCH_TOKENS))
    return state, row


def _manifest(info: dict, ids: list[int], meta: dict) -> dict:
    state, row = _limits(info)
    return {
        "format_version": 1,
        "hidden_size": info["hidden_size"],
        "head_dim": meta["head_dim"],
        "temperature": float(meta["temperature"]),
        "special_token_ids": ids,
        "max_options": MAX_OPTIONS,
        "max_state_tokens": state,
        "max_row_tokens": row,
        "base_model": meta["base"],
        "base_revision": meta.get("base_revision"),
        "option_isolation": False,
        "backbone_dtype": "bfloat16",
        "head_dtype": "float32",
        "add_special_tokens": False,
    }


def _source_files(source: Path, meta: dict) -> list[Path]:
    if meta["weights"] == "lora":
        return [source / name for name in LORA_FILES]
    return [source / name for name in FULL_AUX_FILES] + _shard_files(source)


def _source_identity(source: Path, meta: dict, repo: str | None, revision: str | None) -> dict:
    if repo is not None:
        return {"repo": repo, "revision": revision}
    identity = {
        "path": str(source.resolve()),
        "head_sha256": _sha256(source / "head.pt"),
    }
    if meta["weights"] == "lora":
        identity["adapter_sha256"] = _sha256(source / "adapter_model.safetensors")
    else:
        identity["config_sha256"] = _sha256(source / "config.json")
        identity["index_sha256"] = _sha256(source / "model.safetensors.index.json")
        identity["shards"] = {
            shard.name: shard.stat().st_size for shard in _shard_files(source)
        }
    return identity


def _resolve_base(meta: dict, base, deps) -> tuple[Path, str | None, dict]:
    if base is not None:
        path = Path(base).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError("base override must be a local directory")
        identity = {
            "base": meta["base"],
            "local_override": True,
            "path": str(path),
        }
        return path, meta.get("base_revision"), identity
    path = Path(
        deps.snapshot_download(
            meta["base"],
            revision=meta.get("base_revision"),
            allow_patterns=list(BASE_PATTERNS),
        )
    )
    revision = source_revision(path)
    identity = {
        "base": meta["base"],
        "revision": revision or meta.get("base_revision"),
    }
    return path, revision, identity


def export(checkpoint, output, base=None) -> dict:
    """Export one family checkpoint. The output directory must not exist."""
    deps = _dependencies()
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must not exist: {output}")
    local = Path(str(checkpoint)).expanduser()
    repo = revision = None
    resolved_revision = None
    if local.is_dir():
        source = local.resolve()
    else:
        parsed = parse_source(checkpoint)
        if parsed is None:
            raise ValueError("Expected a local Kev directory or jaredpalmer/kev-* repo")
        repo, revision = parsed
        source = Path(
            deps.snapshot_download(
                repo, revision=revision, allow_patterns=list(SOURCE_PATTERNS)
            )
        )
        resolved_revision = source_revision(source)
    meta = _read_meta(source, deps)
    _validate_meta(meta, None, deps)
    output.mkdir(parents=True, exist_ok=False)
    temporary = output
    try:
        if meta["weights"] == "lora":
            base_path, base_revision, _ = _resolve_base(meta, base, deps)
            info = _export_lora(source, base_path, temporary, meta, deps)
        else:
            base_path, base_revision = source, meta.get("base_revision")
            info = _export_full(source, temporary, meta, deps)
        ids = _tokenizer_ids(temporary, deps)
        manifest = {
            **_manifest(info, ids, meta),
            "source_repo": repo,
            "source_revision": resolved_revision or revision,
            "source_sha256": {
                file.name: _sha256(file)
                for file in _source_files(source, meta)
                if file.name == "head.pt"
            },
            "base_source": str(base_path),
            "base_is_local_override": base is not None,
        }
        if base_revision is not None:
            manifest["base_revision"] = base_revision
        _write_json(temporary / "kev_config.json", manifest)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _cache_root(explicit) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    if os.environ.get("KEV_CACHE_DIR"):
        return Path(os.environ["KEV_CACHE_DIR"]).expanduser()
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser() / "rtp-llm-kev-family"
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub" / "rtp-llm-kev-family"


def _cache_key(source: dict, base, meta: dict) -> str:
    identity = {
        "preparer_version": PREPARER_VERSION,
        "source": source,
        "base": None if base is None else str(Path(base).expanduser().resolve()),
        "meta": {
            key: meta.get(key)
            for key in ("base", "base_revision", "lora", "head_dim", "weights", "temperature")
        },
    }
    encoded = json.dumps(identity, sort_keys=True).encode()
    return "kev-family-" + hashlib.sha256(encoded).hexdigest()[:32]


def _source_path(checkpoint, deps):
    local = Path(str(checkpoint)).expanduser()
    if local.is_dir():
        source = local.resolve()
        return source, _read_meta(source, deps), None, None
    parsed = parse_source(checkpoint)
    if parsed is None:
        raise ValueError("Expected a local Kev directory or jaredpalmer/kev-* repo")
    repo, revision = parsed
    source = Path(
        deps.snapshot_download(
            repo,
            revision=revision,
            allow_patterns=["head.pt", "adapter_config.json", "config.json"],
        )
    )
    return source, _read_meta(source, deps), repo, source_revision(source) or revision


def prepare_cached(checkpoint, *, cache_root=None, base=None) -> str:
    """Prepare (once) and return a verified text checkpoint directory."""
    deps = _dependencies()
    source, meta, repo, revision = _source_path(checkpoint, deps)
    _validate_meta(meta, None, deps)
    key = _cache_key(_source_identity(source, meta, repo, revision), base, meta)
    root = _cache_root(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / key
    with deps.FileLock(str(root / (key + ".lock"))):
        if destination.is_dir() and (destination / "kev_config.json").is_file():
            return str(destination)
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Invalid Kev cache at {destination}")
        temporary = root / f".{key}-{uuid.uuid4().hex}"
        try:
            export(checkpoint, temporary, base=base)
            with (temporary / "kev_config.json").open("rb") as stream:
                digest = hashlib.sha256(stream.read()).hexdigest()
            _write_json(
                temporary / INTEGRITY_FILE,
                {"version": PREPARER_VERSION, "key": key, "manifest_sha256": digest},
            )
            os.replace(temporary, destination)
            return str(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local directory or jaredpalmer/kev-* repo")
    parser.add_argument("--base", help="Local base directory overriding the head metadata")
    parser.add_argument("--output", required=True, help="New output directory")
    args = parser.parse_args()
    manifest = export(args.checkpoint, args.output, args.base)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
