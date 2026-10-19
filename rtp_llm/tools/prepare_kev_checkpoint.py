"""Prepare compatible Kev-4B checkpoints without PEFT or HF model classes.

Cache identities use actual source bytes and resolved base revisions. A local
``base`` override is isolated by a fingerprint of its complete file contents.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace

from rtp_llm.tools import convert_kev_checkpoint as reference

PREPARER_VERSION = 4
TEXT_PREFIX = "model.language_model."
ADAPTER_PREFIX = "base_model.model."
SHARD_BYTES = 1024**3
INTEGRITY_FILE = "kev_prepare.json"
# Qwen2Tokenizer in the reference Transformers 5.17 export rebuilds this
# pretokenizer, rather than preserving the base JSON's different \p{M} regex.
KEV_PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)
BASE_FILES = (
    "config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    *TOKENIZER_FILES,
)


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


def _recognize(checkpoint: str):
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        if (path / "kev_config.json").is_file():
            return "export", path.resolve()
        if (path / "head.pt").is_file():
            if not all((path / name).is_file() for name in reference.SOURCE_FILES):
                raise ValueError("Incomplete raw Kev checkpoint")
            return "raw", path.resolve()
        return None
    parsed = reference._parse_source(checkpoint)
    return ("remote", parsed) if parsed is not None else None


def _load_source(source: Path, deps) -> tuple[dict, dict]:
    head_bytes, hashes = reference._read_source(source)
    metadata = reference._normalize_head(
        deps.torch.load(io.BytesIO(head_bytes), map_location="cpu", weights_only=True)
    )
    reference._validate_head(metadata, deps.torch)
    return metadata, hashes


def _assert_source_unchanged(source: Path, expected: dict) -> None:
    if reference._read_source(source)[1] != expected:
        raise ValueError(
            "Kev source changed during preparation; retry with stable files"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _safe_filename(name: str) -> str:
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Checkpoint index contains an invalid filename")
    return name


def _tensor_header(path: Path) -> dict:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors file: {path.name}")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 2 <= length <= 128 * 1024**2:
            raise ValueError(f"Invalid safetensors header: {path.name}")
        header = json.loads(stream.read(length))
    payload_size = path.stat().st_size - 8 - length
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    position = 0
    for name, info in sorted(tensors.items(), key=lambda item: item[1]["data_offsets"]):
        dtype_size = {"BF16": 2, "F16": 2, "F32": 4}.get(info.get("dtype"))
        shape = info.get("shape")
        if (
            dtype_size is None
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim < 0 for dim in shape)
        ):
            raise ValueError(f"Invalid tensor metadata: {name}")
        end = position + math.prod(shape) * dtype_size
        if info.get("data_offsets") != [position, end]:
            raise ValueError(f"Invalid tensor offsets: {name}")
        position = end
    if not tensors or position != payload_size:
        raise ValueError(f"Truncated or oversized safetensors payload: {path.name}")
    return tensors


def _inventory(path: Path) -> tuple[dict, dict]:
    index_path = path / "model.safetensors.index.json"
    index = _json(index_path)["weight_map"] if index_path.is_file() else None
    if index is not None:
        if not isinstance(index, dict) or not index:
            raise ValueError("Invalid model weight index")
        filenames = sorted({_safe_filename(name) for name in index.values()})
    else:
        filenames = ["model.safetensors"]
    tensors, locations = {}, {}
    for filename in filenames:
        for key, info in _tensor_header(path / filename).items():
            if key in tensors:
                raise ValueError(f"Duplicate checkpoint tensor: {key}")
            tensors[key], locations[key] = info, filename
    if index is not None and index != locations:
        raise ValueError("Weight index does not match safetensors contents")
    return tensors, locations


def _expected_shapes() -> tuple[dict, set]:
    shapes = {"embed_tokens.weight": (248320, 2560), "norm.weight": (2560,)}
    adapted = set()
    for layer in range(32):
        prefix = f"layers.{layer}."
        shapes[prefix + "input_layernorm.weight"] = (2560,)
        shapes[prefix + "post_attention_layernorm.weight"] = (2560,)
        projections = {
            "mlp.gate_proj": (9216, 2560),
            "mlp.up_proj": (9216, 2560),
            "mlp.down_proj": (2560, 9216),
        }
        if layer % 4 == 3:
            projections.update(
                {
                    "self_attn.q_proj": (8192, 2560),
                    "self_attn.k_proj": (1024, 2560),
                    "self_attn.v_proj": (1024, 2560),
                    "self_attn.o_proj": (2560, 4096),
                }
            )
            for name in ("q_norm", "k_norm"):
                shapes[prefix + f"self_attn.{name}.weight"] = (256,)
        else:
            projections.update(
                {
                    "linear_attn.in_proj_a": (32, 2560),
                    "linear_attn.in_proj_b": (32, 2560),
                    "linear_attn.in_proj_qkv": (8192, 2560),
                    "linear_attn.in_proj_z": (4096, 2560),
                    "linear_attn.out_proj": (2560, 4096),
                }
            )
            for name, shape in {
                "A_log": (32,),
                "dt_bias": (32,),
                "conv1d.weight": (8192, 1, 4),
                "norm.weight": (128,),
            }.items():
                shapes[prefix + "linear_attn." + name] = shape
        for name, shape in projections.items():
            shapes[prefix + name + ".weight"] = shape
            adapted.add(prefix + name)
    return shapes, adapted


def _text_config(base: Path) -> dict:
    config = _json(base / "config.json")
    text = dict(config.get("text_config", config))
    reference._validate_backbone(SimpleNamespace(config=SimpleNamespace(**text)))
    expected = {
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "attn_output_gate": True,
        "attention_bias": False,
    }
    if any(text.get(name) != value for name, value in expected.items()):
        raise ValueError("Base linear/full attention configuration differs from Kev-4B")
    for name in ("text_config", "vision_config", "quantization_config"):
        if name in text:
            raise ValueError(f"Unsupported text backbone field: {name}")
    text.update(architectures=["Qwen3_5ForCausalLM"], dtype="bfloat16", use_cache=False)
    text.setdefault("bos_token_id", None)
    text.setdefault("pad_token_id", None)
    # No persisted rotary buffers are created: the runtime derives them in FP32.
    text["partial_rotary_factor"] = text.get("rope_parameters", {}).get(
        "partial_rotary_factor", 0.25
    )
    return text


def _validate_tensor_set(header: dict, shapes: dict, dtype: str | dict) -> None:
    if set(header) != set(shapes):
        raise ValueError(
            "Checkpoint tensor names do not match the supported architecture"
        )
    for name, shape in shapes.items():
        expected_dtype = dtype[name] if isinstance(dtype, dict) else dtype
        if (
            header[name]["dtype"] != expected_dtype
            or tuple(header[name]["shape"]) != shape
        ):
            raise ValueError(f"Unexpected {expected_dtype} tensor shape: {name}")


def _base_dtypes(shapes: dict) -> dict:
    # Qwen3.5 preserves these two GDN parameters in FP32 (48 tensors).
    # Everything else, including dt_bias, is BF16. Exports are uniformly BF16.
    return {
        name: (
            "F32"
            if name.endswith((".linear_attn.A_log", ".linear_attn.norm.weight"))
            else "BF16"
        )
        for name in shapes
    }


def _validate_adapter(source: Path, shapes: dict, adapted: set) -> set:
    reference._validate_adapter_config(source)
    expected_tensors = {}
    for target in adapted:
        output_size, input_size = shapes[target + ".weight"]
        expected_tensors[ADAPTER_PREFIX + target + ".lora_A.weight"] = (16, input_size)
        expected_tensors[ADAPTER_PREFIX + target + ".lora_B.weight"] = (output_size, 16)
    if len(expected_tensors) != 496:
        raise ValueError("Expected exactly 496 LoRA tensors")
    _validate_tensor_set(
        _tensor_header(source / "adapter_model.safetensors"), expected_tensors, "F32"
    )
    return set(expected_tensors)


def _merge_tensor(weight, lora_a, lora_b, torch):
    """Match CPU PEFT merge order: W_fp32 + ((B_fp32 @ A_fp32) * 2)."""
    if (
        weight.device.type != "cpu"
        or lora_a.device.type != "cpu"
        or lora_b.device.type != "cpu"
        or lora_a.dtype != torch.float32
        or lora_b.dtype != torch.float32
        or weight.ndim != 2
        or lora_a.ndim != 2
        or lora_b.ndim != 2
        or tuple(lora_a.shape) != (16, weight.shape[1])
        or tuple(lora_b.shape) != (weight.shape[0], 16)
    ):
        raise ValueError("Invalid CPU fp32 LoRA merge operands")
    with torch.no_grad():
        merged = weight.float() + (lora_b @ lora_a) * (32 / 16)
        if not torch.isfinite(merged).all().item():
            raise ValueError("LoRA merge produced nonfinite values")
        result = merged.to(torch.bfloat16).contiguous()
        if not torch.isfinite(result).all().item():
            raise ValueError("LoRA merge overflows BF16")
        return result


def _tokenizer_ids(path: Path, deps) -> list[int]:
    tokenizer = deps.Tokenizer.from_file(str(path / "tokenizer.json"))
    ids = [tokenizer.token_to_id(token) for token in reference.SPECIAL_TOKENS]
    if ids != reference.SPECIAL_TOKEN_IDS:
        raise ValueError("Base tokenizer does not have the supported delimiter IDs")
    return ids


def _normalize_tokenizer(tokenizer, added_tokens: dict) -> None:
    """Reproduce the reference tokenizer backend without Transformers classes."""
    from tokenizers import AddedToken, Regex, decoders, pre_tokenizers

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(KEV_PRETOKENIZE_REGEX), behavior="isolated", invert=False
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    # Base tokenizer.json contains 22 entries, while tokenizer_config contains
    # all 33. The reference loader imports the extra 11 before save_pretrained.
    for token_id, attributes in sorted(
        added_tokens.items(), key=lambda pair: int(pair[0])
    ):
        tokenizer.add_tokens([AddedToken(**attributes)])
        if tokenizer.token_to_id(attributes["content"]) != int(token_id):
            raise ValueError("Added token ID differs from the tokenizer config")


def _prepare_tokenizer(base: Path, output: Path, deps) -> None:
    config = _json(base / "tokenizer_config.json")
    added = config.get("added_tokens_decoder")
    if (
        config.get("tokenizer_class") != "Qwen2Tokenizer"
        or not isinstance(added, dict)
        or set(added) != {str(i) for i in range(248044, 248077)}
    ):
        raise ValueError(
            "Base tokenizer config differs from the supported Qwen2 tokenizer"
        )
    tokenizer = deps.Tokenizer.from_file(str(base / "tokenizer.json"))
    _normalize_tokenizer(tokenizer, added)
    tokenizer.save(str(output / "tokenizer.json"))
    for filename in TOKENIZER_FILES:
        if (
            filename not in ("tokenizer.json", "tokenizer_config.json")
            and (base / filename).is_file()
        ):
            shutil.copyfile(base / filename, output / filename)
    config["pretokenize_regex"] = KEV_PRETOKENIZE_REGEX
    # Transformers 5.2 forwards this option both explicitly and through kwargs,
    # causing a duplicate-keyword exception for vocabularies larger than 100k.
    # Leave its default unchanged; the canonical backend already has our regex.
    config.pop("fix_mistral_regex", None)
    _write_json(output / "tokenizer_config.json", config)
    _tokenizer_ids(output, deps)


def _stream_weights(source: Path, base: Path, output: Path, deps) -> None:
    shapes, adapted = _expected_shapes()
    all_keys = _validate_adapter(source, shapes, adapted)
    header, locations = _inventory(base)
    text_header = {
        key: value for key, value in header.items() if key.startswith(TEXT_PREFIX)
    }
    full_shapes = {TEXT_PREFIX + k: v for k, v in shapes.items()}
    _validate_tensor_set(text_header, full_shapes, _base_dtypes(full_shapes))
    used, pending, weight_map = set(), {}, {}
    pending_bytes, total_size, shard_index = 0, 0, 0

    def flush():
        nonlocal pending, pending_bytes, shard_index
        if not pending:
            return
        shard_index += 1
        name = f"model-{shard_index:05d}.safetensors"
        deps.save_file(pending, str(output / name), metadata={"format": "pt"})
        weight_map.update({key: name for key in pending})
        pending, pending_bytes = {}, 0

    with deps.safe_open(
        str(source / "adapter_model.safetensors"), framework="pt", device="cpu"
    ) as adapter:
        for filename in sorted(set(locations[key] for key in text_header)):
            with deps.safe_open(
                str(base / filename), framework="pt", device="cpu"
            ) as shard:
                for key in sorted(k for k in text_header if locations[k] == filename):
                    name = key[len(TEXT_PREFIX) :]
                    weight = shard.get_tensor(key)
                    target = name.removesuffix(".weight")
                    if target in adapted:
                        a_key = ADAPTER_PREFIX + target + ".lora_A.weight"
                        b_key = ADAPTER_PREFIX + target + ".lora_B.weight"
                        weight = _merge_tensor(
                            weight,
                            adapter.get_tensor(a_key),
                            adapter.get_tensor(b_key),
                            deps.torch,
                        )
                        used.update((a_key, b_key))
                    elif not deps.torch.isfinite(weight).all().item():
                        raise ValueError(f"Nonfinite base tensor: {key}")
                    weight = weight.to(deps.torch.bfloat16).contiguous()
                    size = weight.numel() * weight.element_size()
                    if pending_bytes + size > SHARD_BYTES:
                        flush()
                    pending[key] = weight
                    pending_bytes += size
                    total_size += size
                    if pending_bytes >= SHARD_BYTES:
                        flush()
        flush()
    if used != all_keys:
        raise ValueError("Not all 496 LoRA keys were applied exactly once")
    _write_json(
        output / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )


def _convert(
    source: Path,
    base: Path,
    output: Path,
    deps,
    local_override: bool,
    *,
    metadata: dict,
    source_hashes: dict,
    resolved_base_revision: str | None,
) -> None:
    config = _text_config(base)
    ids = _tokenizer_ids(base, deps)
    _write_json(output / "config.json", config)
    _stream_weights(source, base, output, deps)
    _prepare_tokenizer(base, output, deps)
    deps.save_file(
        {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in metadata["head"].items()
        },
        str(output / "kev_head.safetensors"),
        metadata={"format": "pt"},
    )
    _write_json(
        output / "kev_config.json",
        {
            "format_version": 1,
            "head_dim": 256,
            "hidden_size": 2560,
            "temperature": metadata["temperature"],
            "special_token_ids": ids,
            "max_options": 32,
            "max_state_tokens": 65536,
            "max_row_tokens": 73728,
            **reference._source_manifest(
                source,
                source_hashes,
                metadata,
                resolved_base_revision=resolved_base_revision,
            ),
            "base_source": str(base),
            "base_is_local_override": local_override,
            "backbone_dtype": "bfloat16",
            "head_dtype": "float32",
            "add_special_tokens": False,
        },
    )


def _validate_export(path: Path, deps) -> None:
    manifest = _json(path / "kev_config.json")
    expected = {
        "format_version": 1,
        "head_dim": 256,
        "hidden_size": 2560,
        "special_token_ids": reference.SPECIAL_TOKEN_IDS,
        "base_model": reference.BASE_MODEL,
        "backbone_dtype": "bfloat16",
        "head_dtype": "float32",
        "add_special_tokens": False,
    }
    if not isinstance(manifest, dict) or any(
        manifest.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("Exported Kev manifest differs from the supported contract")
    for name in ("source_revision", "base_revision", "source_repo"):
        if name not in manifest:
            raise ValueError(f"Missing Kev manifest {name}")
        value = manifest.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Invalid Kev manifest {name}")
    hashes = manifest.get("source_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(reference.SOURCE_FILES)
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            for digest in hashes.values()
        )
    ):
        raise ValueError("Invalid Kev source content fingerprints")
    temperature = manifest.get("temperature")
    if (
        type(temperature) not in (float, int)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Invalid Kev temperature")
    for name in ("max_options", "max_state_tokens", "max_row_tokens"):
        if type(manifest.get(name)) is not int or manifest[name] <= 0:
            raise ValueError(f"Invalid Kev manifest {name}")
    _text_config(path)
    shapes, _ = _expected_shapes()
    header, _ = _inventory(path)
    _validate_tensor_set(
        header, {TEXT_PREFIX + k: v for k, v in shapes.items()}, "BF16"
    )
    _validate_tensor_set(
        _tensor_header(path / "kev_head.safetensors"), reference.HEAD_SHAPES, "F32"
    )
    _tokenizer_ids(path, deps)
    if not (path / "tokenizer_config.json").is_file():
        raise ValueError("Exported checkpoint is missing tokenizer_config.json")


def _cache_root(explicit) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    if os.environ.get("KEV_CACHE_DIR"):
        return Path(os.environ["KEV_CACHE_DIR"]).expanduser()
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser() / "rtp-llm-kev"
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub" / "rtp-llm-kev"


def _directory_fingerprint(path: Path) -> dict:
    # Content, rather than path/stat metadata, detects in-place replacement even
    # when file sizes and modification times are preserved by the downloader.
    return {
        str(file.relative_to(path)): _sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _assert_base_unchanged(path: Path, identity: dict) -> None:
    if "files" in identity and _directory_fingerprint(path) != identity["files"]:
        raise ValueError(
            "Base content changed during preparation; retry with stable files"
        )


def _cache_key(source_hashes: dict, base_identity: dict) -> str:
    identity = {
        "preparer_version": PREPARER_VERSION,
        "source_sha256": source_hashes,
        "base": base_identity,
    }
    encoded = json.dumps(identity, sort_keys=True).encode()
    return "kev-4b-" + hashlib.sha256(encoded).hexdigest()


def _resolve_base(base, metadata: dict, deps) -> tuple[Path, str | None, dict]:
    if base is not None:
        path = Path(base).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError("base override must be a local directory")
        identity = {
            "model": metadata["base"],
            "local_override": True,
            "path": str(path),
            "files": _directory_fingerprint(path),
        }
        return path, metadata["base_revision"], identity
    path = Path(
        deps.snapshot_download(
            metadata["base"],
            revision=metadata["base_revision"],
            allow_patterns=["config.json"],
        )
    )
    revision = reference._source_revision(path)
    if revision is None:
        # Nonstandard download providers may not return an HF snapshot path.
        # Fetch the complete base before fingerprinting, including on cache hits.
        path = Path(
            deps.snapshot_download(
                metadata["base"],
                revision=metadata["base_revision"],
                allow_patterns=list(BASE_FILES),
            )
        )
        revision = reference._source_revision(path)
    identity = {"model": metadata["base"], "resolved_revision": revision}
    if revision is None:
        identity.update(
            requested_revision=metadata["base_revision"],
            files=_directory_fingerprint(path),
        )
    return path, revision, identity


def _record_integrity(path: Path, key: str) -> None:
    files = {
        p.name: {"size": p.stat().st_size, "sha256": _sha256(p)}
        for p in sorted(path.iterdir())
        if p.is_file() and p.name != INTEGRITY_FILE
    }
    _write_json(
        path / INTEGRITY_FILE, {"version": PREPARER_VERSION, "key": key, "files": files}
    )


def _valid_cache(path: Path, key: str, deps) -> bool:
    try:
        integrity = _json(path / INTEGRITY_FILE)
        if integrity["version"] != PREPARER_VERSION or integrity["key"] != key:
            return False
        files = integrity["files"]
        actual = {p.name for p in path.iterdir() if p.name != INTEGRITY_FILE}
        if not isinstance(files, dict) or not files or set(files) != actual:
            return False
        for name, info in files.items():
            file = path / _safe_filename(name)
            if (
                file.is_symlink()
                or file.stat().st_size != info["size"]
                or _sha256(file) != info["sha256"]
            ):
                return False
        _validate_export(path, deps)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_kev_checkpoint(
    checkpoint: str, *, cache_root=None, base=None
) -> str | None:
    """Return a verified text checkpoint path, or None for ordinary models.

    FileLock serializes workers. Successful results are published by atomic
    directory rename; failures remove only this invocation's temporary output.
    """
    recognized = _recognize(checkpoint)
    if recognized is None:
        return None
    kind, location = recognized
    deps = _dependencies()
    if kind == "export":
        # Legacy converter outputs have no integrity sidecar; validate their
        # manifest, all tensor headers/payload sizes, and tokenizer directly.
        _validate_export(location, deps)
        return checkpoint
    if kind == "remote":
        repo, revision = location
        source = Path(
            deps.snapshot_download(
                repo,
                revision=revision,
                allow_patterns=list(reference.SOURCE_FILES),
            )
        )
    else:
        source = location
    metadata, source_hashes = _load_source(source, deps)
    resolved_base, base_revision, base_identity = _resolve_base(base, metadata, deps)
    key = _cache_key(source_hashes, base_identity)
    root = _cache_root(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / key
    with deps.FileLock(str(root / (key + ".lock"))):
        if _valid_cache(destination, key, deps):
            _assert_source_unchanged(source, source_hashes)
            _assert_base_unchanged(resolved_base, base_identity)
            return str(destination)
        if destination.exists() or destination.is_symlink():
            raise ValueError(
                f"Invalid Kev cache at {destination}; remove this cache entry and retry"
            )
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=root))
        try:
            if base is None and "files" not in base_identity:
                resolved_base = Path(
                    deps.snapshot_download(
                        metadata["base"],
                        # Resolve mutable refs once, then fetch weights at that SHA.
                        revision=base_revision or metadata["base_revision"],
                        allow_patterns=list(BASE_FILES),
                    )
                )
                actual_revision = reference._source_revision(resolved_base)
                if base_revision is not None and actual_revision != base_revision:
                    raise ValueError("Base snapshot changed while downloading weights")
            _convert(
                source,
                resolved_base,
                temporary,
                deps,
                base is not None,
                metadata=metadata,
                source_hashes=source_hashes,
                resolved_base_revision=base_revision,
            )
            _assert_source_unchanged(source, source_hashes)
            _assert_base_unchanged(resolved_base, base_identity)
            _validate_export(temporary, deps)
            _record_integrity(temporary, key)
            os.replace(temporary, destination)
            return str(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
