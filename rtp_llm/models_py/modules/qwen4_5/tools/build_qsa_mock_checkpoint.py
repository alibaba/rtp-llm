"""Build QSA-mocked Qwen 4.5-Flash checkpoints (compact 8L synthetic or 40L 1109 overlay).

Modes:
  - ``synthetic-8l``: Self-contained 8-layer Qwen 4.5-Flash (MXFP4 + FP8 PLE + YoCo + QSA)
    checkpoint matching ``qwen4_5_pd_smoke_test._build_synthetic_8layer_checkpoint`` plus
    QSA indexer config/weights on full-attention layers ``3`` (source) and ``7`` (YoCo consumer).
  - ``overlay-40l``: Read-only symlink overlay onto the 40-layer 1109 checkpoint adding
    ``model-qsa-mock-00001-of-00001.safetensors`` (30 BF16 QSA indexer tensors across all 10
    non-GDN layers ``3, 7, 11, 15, 19, 23, 27, 31, 35, 39``), updated ``config.json``,
    updated ``model.safetensors.index.json``, and ``mock_qsa_manifest.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from rtp_llm.config.qwen4_5_config import (
    parse_qwen4_5_config_json,
    parse_qwen4_5_quant_config,
)
from rtp_llm.models_py.model_desc.qwen4_5 import Qwen4_5ForCausalLM
from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
    SafetensorsShardReader,
    arch_config_to_hf_dict,
    save_qwen4_5_safetensors_checkpoint,
    write_safetensors_file,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig
from rtp_llm.models_py.modules.qwen4_5.tokenizer import (
    _BYTE_TO_UNICODE,
    Qwen2BPETokenizer,
)

DEFAULT_1109_BASE_DIR = "/mnt/hf3fs/3fs/models/Qwen4.5-Flash-1109-bf16-re-mxfp4-ple-fp8"
DEFAULT_8L_OUT_DIR = "/data0/luohaocheng.lhc/models/Qwen4.5-Flash-8L-qsa-mock"
DEFAULT_40L_OUT_DIR = "/data0/luohaocheng.lhc/models/Qwen4.5-Flash-1109-qsa-mock"
QSA_SHARD_NAME = "model-qsa-mock-00001-of-00001.safetensors"
FORBIDDEN_STORAGE_ROOT = Path("/3fs-data/3fs/mtp_test")

# Lossless single-letter ASCII tokens (A,D,G,J,M,P,S,V) matching
# qwen4_5_pd_smoke_test._build_synthetic_8layer_checkpoint.
SYNTHETIC_ACTIVE_IDS = (37, 40, 43, 46, 49, 52, 55, 58)
TOKENIZER_FILES = (
    "chat_template.jinja",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
)


@dataclass(frozen=True)
class QSAConfigParams:
    """QSA indexer configuration matching Qwen 4.5-Flash / ``qwen4_exp`` schema."""

    indexer_head_dim: int = 128
    indexer_n_heads: int = 8
    indexer_kv_heads: int = 1
    indexer_compress_ratio: int = 4
    indexer_budget: int = 2048

    def __post_init__(self) -> None:
        for name in (
            "indexer_head_dim",
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_compress_ratio",
            "indexer_budget",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.indexer_kv_heads != 1:
            raise ValueError(f"QSA indexer requires indexer_kv_heads=1, got {self.indexer_kv_heads}")
        if self.indexer_budget % self.indexer_compress_ratio != 0:
            raise ValueError(
                f"indexer_budget ({self.indexer_budget}) must be divisible by "
                f"indexer_compress_ratio ({self.indexer_compress_ratio})"
            )

    def to_config_fields(self) -> Dict[str, int]:
        return {
            "indexer_head_dim": int(self.indexer_head_dim),
            "indexer_n_heads": int(self.indexer_n_heads),
            "indexer_kv_heads": int(self.indexer_kv_heads),
            "indexer_compress_ratio": int(self.indexer_compress_ratio),
            "indexer_budget": int(self.indexer_budget),
        }


def _assert_safe_output_dir(output_dir: Path, base_dir: Optional[Path] = None) -> Path:
    resolved_out = output_dir.resolve()
    forbidden = Path(os.path.abspath(str(FORBIDDEN_STORAGE_ROOT)))
    if resolved_out == forbidden or forbidden in resolved_out.parents:
        raise ValueError(f"Refusing to write inside protected path: {resolved_out}")
    if base_dir is not None and resolved_out == base_dir.resolve():
        raise ValueError(f"output_dir must differ from read-only base_dir: {resolved_out}")
    return resolved_out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_tensor(t: torch.Tensor) -> str:
    cpu = t.detach().cpu().contiguous()
    raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes() if cpu.numel() > 0 else b""
    return hashlib.sha256(raw).hexdigest()


def _full_attention_layers(config_json: Mapping[str, Any]) -> List[int]:
    layer_types = config_json.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        return [idx for idx, ltype in enumerate(layer_types) if ltype == "full_attention"]
    num_layers = int(config_json["num_hidden_layers"])
    interval = int(config_json["full_attention_interval"])
    return [idx for idx in range(num_layers) if (idx + 1) % interval == 0]


def _build_qsa_tensors(
    *,
    full_attn_layers: Sequence[int],
    hidden_size: int,
    qsa: QSAConfigParams,
    seed: int,
    prefix: str = "model.",
) -> Dict[str, torch.Tensor]:
    """Create deterministic BF16 QSA indexer weights for all full-attention layers.

    Uses ``qwen4_exp`` exact checkpoint key names and Gemma ``+1`` zero-centered
    RMSNorm convention (raw ``q_layernorm.weight`` / ``k_layernorm.weight`` initialized
    to zeros because ``exact_head_rms_norm`` applies ``1.0 + gamma.float()``).
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    qk_out_dim = (qsa.indexer_n_heads + qsa.indexer_kv_heads) * qsa.indexer_head_dim
    scale = 1.0 / math.sqrt(float(hidden_size))
    tensors: Dict[str, torch.Tensor] = {}

    for layer_idx in sorted(int(i) for i in full_attn_layers):
        idx_prefix = f"{prefix}layers.{layer_idx}.self_attn.indexer."
        qk_w = (
            torch.randn((qk_out_dim, hidden_size), generator=gen, dtype=torch.float32) * scale
        ).to(torch.bfloat16)
        tensors[f"{idx_prefix}index_qk_proj.weight"] = qk_w
        tensors[f"{idx_prefix}q_layernorm.weight"] = torch.zeros(
            (qsa.indexer_head_dim,), dtype=torch.bfloat16
        )
        tensors[f"{idx_prefix}k_layernorm.weight"] = torch.zeros(
            (qsa.indexer_head_dim,), dtype=torch.bfloat16
        )
    return tensors


def _tensor_manifest_record(name: str, shard: str, t: torch.Tensor) -> Dict[str, Any]:
    tf = t.detach().cpu().float()
    return {
        "name": name,
        "shard": shard,
        "dtype": str(t.dtype).replace("torch.", ""),
        "shape": list(t.shape),
        "numel": int(t.numel()),
        "nbytes": int(t.numel() * t.element_size()),
        "sha256": _sha256_tensor(t),
        "mean": float(tf.mean().item()) if t.numel() else 0.0,
        "std": float(tf.std(unbiased=False).item()) if t.numel() else 0.0,
        "min": float(tf.min().item()) if t.numel() else 0.0,
        "max": float(tf.max().item()) if t.numel() else 0.0,
    }


def _write_synthetic_tokenizer(ckpt_dir: Path, vocab_size: int, model_max_length: int) -> None:
    """Write HF Qwen2Tokenizer + ``tokenizer.json`` with lossless ASCII & byte coverage.

    Keeps printable ASCII 32..126 at IDs 4..98 so ``SYNTHETIC_ACTIVE_IDS``
    (37, 40, 43, 46, 49, 52, 55, 58) map 1-to-1 to (A,D,G,J,M,P,S,V).
    """
    special_tokens = [
        (0, "<|pad|>"),
        (1, "<|im_start|>"),
        (2, "<|endoftext|>"),
        (3, "<|im_end|>"),
    ]
    vocab: Dict[str, int] = {tok: tid for tid, tok in special_tokens}
    next_id = len(special_tokens)

    # IDs 4..98: printable ASCII 32..126 (  -> u0120)
    for cp in range(32, 127):
        tok = _BYTE_TO_UNICODE[cp]
        assert tok not in vocab
        vocab[tok] = next_id
        next_id += 1

    # ID 99: newline byte (u010a), then remaining raw UTF-8 bytes 0..255 for lossless fallback
    for cp in [10] + [b for b in range(256) if b != 10 and not (32 <= b < 127)]:
        tok = _BYTE_TO_UNICODE[cp]
        if tok not in vocab and next_id < vocab_size:
            vocab[tok] = next_id
            next_id += 1

    while next_id < vocab_size:
        vocab[f"tok_{next_id}"] = next_id
        next_id += 1

    (ckpt_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )
    (ckpt_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")

    added_decoder = {
        str(tid): {
            "content": tok,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
        for tid, tok in special_tokens
    }
    (ckpt_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "add_prefix_space": False,
                "bos_token": "<|im_start|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|endoftext|>",
                "model_max_length": int(model_max_length),
                "clean_up_tokenization_spaces": False,
                "added_tokens_decoder": added_decoder,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (ckpt_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<|im_start|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|endoftext|>",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model_vocab = {tok: tid for tok, tid in vocab.items() if tid >= len(special_tokens)}
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": tid,
                "content": tok,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for tid, tok in special_tokens
        ],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {
                        "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    },
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": False,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": model_vocab,
            "merges": [],
        },
    }
    (ckpt_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer_json, ensure_ascii=False), encoding="utf-8"
    )


def _load_safetensors_shard(shard_path: Path) -> Dict[str, torch.Tensor]:
    """Read all tensors from a single safetensors file via ``SafetensorsShardReader``."""
    reader = SafetensorsShardReader(shard_path.parent, prefer_safetensors=False)
    try:
        header, _ = reader._get_header(shard_path.name)
        return {
            k: reader.get_tensor(k)
            for k in header
            if k != "__metadata__"
        }
    finally:
        reader.close()


def _write_manifest(
    *,
    output_dir: Path,
    mode: str,
    base_dir: Optional[Path],
    config_json: Mapping[str, Any],
    qsa: QSAConfigParams,
    qsa_seed: int,
    qsa_tensors: Mapping[str, torch.Tensor],
    qsa_shard_map: Mapping[str, str],
    extra: Mapping[str, Any],
) -> Path:
    full_attn_layers = _full_attention_layers(config_json)
    share_map_parsed = Qwen4_5ArchConfig.from_dict(config_json).get_attention_sharing()
    full_attn_consumers = sorted(l for l in full_attn_layers if l in share_map_parsed)
    full_attn_sources = sorted({share_map_parsed[l] for l in full_attn_consumers})

    files_sha256: Dict[str, Dict[str, Any]] = {}
    for fname in sorted(os.listdir(output_dir)):
        fpath = output_dir / fname
        if fpath.is_symlink() or not fpath.is_file():
            continue
        if fname == "mock_qsa_manifest.json":
            continue
        files_sha256[fname] = {
            "size_bytes": fpath.stat().st_size,
            "sha256": _sha256_file(fpath),
        }

    manifest = {
        "schema_version": 1,
        "mode": mode,
        "base_dir": str(base_dir) if base_dir is not None else None,
        "qsa_seed": int(qsa_seed),
        "qsa_config": qsa.to_config_fields(),
        "qsa_convention": {
            "tensor_key_pattern": [
                "model.layers.{i}.self_attn.indexer.index_qk_proj.weight",
                "model.layers.{i}.self_attn.indexer.q_layernorm.weight",
                "model.layers.{i}.self_attn.indexer.k_layernorm.weight",
            ],
            "rmsnorm_mode": "gemma_plus_one",
            "rmsnorm_note": (
                "q_layernorm.weight and k_layernorm.weight are raw zero-initialized BF16 "
                "checkpoint tensors; Qwen4ExpQSAIndexer / exact_head_rms_norm applies (1.0 + gamma.float())."
            ),
            "yoco_consumers_have_full_indexer_weights": True,
        },
        "architecture_summary": {
            "hidden_size": int(config_json["hidden_size"]),
            "num_hidden_layers": int(config_json["num_hidden_layers"]),
            "full_attention_interval": int(config_json["full_attention_interval"]),
            "head_dim": int(config_json["head_dim"]),
            "partial_rotary_factor": float(config_json["partial_rotary_factor"]),
            "rms_norm_eps": float(config_json["rms_norm_eps"]),
            "full_attention_layers": full_attn_layers,
            "yoco_full_attn_sources": full_attn_sources,
            "yoco_full_attn_consumers": full_attn_consumers,
            "share_map": config_json.get("share_map"),
        },
        "qsa_tensor_count": len(qsa_tensors),
        "qsa_total_bytes": sum(int(t.numel() * t.element_size()) for t in qsa_tensors.values()),
        "qsa_tensors": [
            _tensor_manifest_record(name, qsa_shard_map[name], qsa_tensors[name])
            for name in sorted(qsa_tensors.keys())
        ],
        "generated_files": files_sha256,
        **extra,
    }
    manifest_path = output_dir / "mock_qsa_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def validate_checkpoint(
    output_dir: Path,
    *,
    qsa: QSAConfigParams,
    expected_qsa_tensors: Optional[Mapping[str, torch.Tensor]] = None,
    check_all_shard_headers: bool = True,
) -> Dict[str, Any]:
    """Validate config.json, index coverage, symlinks, safetensors headers, and QSA weights."""
    config_path = output_dir / "config.json"
    index_path = output_dir / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {output_dir}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing model.safetensors.index.json in {output_dir}")

    config_json = json.loads(config_path.read_text(encoding="utf-8"))
    arch_cfg = parse_qwen4_5_config_json(config_json, strict=True)
    quant_cfg = parse_qwen4_5_quant_config(config_json)
    for k, expected_val in qsa.to_config_fields().items():
        if int(config_json.get(k, -1)) != expected_val:
            raise AssertionError(f"config.json[{k!r}]={config_json.get(k)!r} != {expected_val}")

    index_json = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: Dict[str, str] = index_json["weight_map"]
    shard_to_keys: Dict[str, List[str]] = {}
    for tname, sname in weight_map.items():
        shard_to_keys.setdefault(sname, []).append(tname)

    total_tensor_bytes = 0
    dtype_bytes = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "F64": 8,
        "I8": 1,
        "U8": 1,
        "I16": 2,
        "I32": 4,
        "I64": 8,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
    }
    for sname, expected_keys in sorted(shard_to_keys.items()):
        spath = output_dir / sname
        if spath.is_symlink() and not spath.exists():
            raise FileNotFoundError(f"Broken symlink shard: {spath} -> {os.readlink(spath)}")
        if not spath.is_file():
            raise FileNotFoundError(f"Missing shard file: {spath}")
        if not check_all_shard_headers and sname != QSA_SHARD_NAME:
            continue
        with spath.open("rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                raise ValueError(f"Truncated safetensors header in {spath}")
            header_len = struct.unpack("<Q", raw_len)[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
        file_size = spath.stat().st_size
        data_base = 8 + header_len
        for key in expected_keys:
            if key not in header:
                raise KeyError(f"Tensor {key!r} mapped to {sname} but absent from shard header")
            info = header[key]
            off0, off1 = info["data_offsets"]
            expected_bytes = math.prod(info["shape"]) * dtype_bytes[info["dtype"]]
            if (off1 - off0) != expected_bytes or data_base + off1 > file_size:
                raise ValueError(
                    f"Invalid offsets/bounds for {key!r} in {sname}: offsets={info[data_offsets]}, "
                    f"expected_bytes={expected_bytes}, file_size={file_size}"
                )
            total_tensor_bytes += expected_bytes

    if check_all_shard_headers:
        meta_total = int(index_json.get("metadata", {}).get("total_size", -1))
        if meta_total != total_tensor_bytes:
            raise AssertionError(
                f"index metadata.total_size={meta_total} != actual header bytes={total_tensor_bytes}"
            )

    full_layers = _full_attention_layers(config_json)
    expected_qk_shape = [
        (qsa.indexer_n_heads + qsa.indexer_kv_heads) * qsa.indexer_head_dim,
        arch_cfg.hidden_size,
    ]
    reader = SafetensorsShardReader(output_dir, prefer_safetensors=False)
    try:
        for lidx in full_layers:
            pfx = f"model.layers.{lidx}.self_attn.indexer."
            for suffix, shape in (
                ("index_qk_proj.weight", expected_qk_shape),
                ("q_layernorm.weight", [qsa.indexer_head_dim]),
                ("k_layernorm.weight", [qsa.indexer_head_dim]),
            ):
                k = pfx + suffix
                if k not in weight_map:
                    raise KeyError(f"Missing QSA tensor in index: {k}")
                dt, sh = reader.get_tensor_spec(k)
                if dt != torch.bfloat16 or list(sh) != shape:
                    raise AssertionError(f"QSA tensor {k} got {(dt, list(sh))}, expected {(torch.bfloat16, shape)}")
                if expected_qsa_tensors is not None:
                    loaded = reader.get_tensor(k)
                    if not torch.equal(loaded, expected_qsa_tensors[k]):
                        raise AssertionError(f"QSA tensor {k} content mismatch against expected")
    finally:
        reader.close()

    if (output_dir / "tokenizer.json").exists():
        tok = Qwen2BPETokenizer(output_dir)
        sample = "ADGJMPSV\nHello 123"
        decoded = tok.decode(tok.encode(sample), skip_special_tokens=False)
        if decoded != sample:
            raise AssertionError(f"Tokenizer round-trip mismatch: {decoded!r} != {sample!r}")

    return {
        "shard_count": len(shard_to_keys),
        "tensor_count": len(weight_map),
        "total_size": total_tensor_bytes,
        "full_attention_layers": full_layers,
        "quant_method": quant_cfg.quant_method,
        "ple_quant_scheme": quant_cfg.ple_quant_scheme,
    }


def build_synthetic_8l(
    *,
    output_dir: Path,
    qsa: QSAConfigParams,
    qsa_seed: int,
    base_seed: int,
    hidden_size: int,
    max_position_embeddings: int,
) -> Dict[str, Any]:
    """Build self-contained 8-layer Qwen 4.5-Flash + QSA mock checkpoint."""
    out_dir = _assert_safe_output_dir(output_dir)
    if hidden_size == 512:
        inter_size = 1024
        num_heads = 8
        lin_k_heads, lin_v_heads = 4, 16
        hc_lowrank = 32
        routed_h, moe_i, shared_i = 256, 256, 128
        ple_dim = 512
    else:
        if hidden_size % 128 != 0:
            raise ValueError(f"hidden_size must be divisible by 128, got {hidden_size}")
        scale_f = hidden_size // 512
        inter_size = 1024 * scale_f
        num_heads = 8 * scale_f
        lin_k_heads, lin_v_heads = 4 * scale_f, 16 * scale_f
        hc_lowrank = 32 * scale_f
        routed_h, moe_i, shared_i = 256 * scale_f, 256 * scale_f, 128 * scale_f
        ple_dim = hidden_size

    cfg = Qwen4_5ArchConfig(
        vocab_size=2048,
        hidden_size=hidden_size,
        num_hidden_layers=8,
        intermediate_size=inter_size,
        first_k_dense_replace=1,
        rms_norm_eps=1e-6,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        max_position_embeddings=max_position_embeddings,
        full_attention_interval=4,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        head_dim=256,
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        rope_theta=10000.0,
        use_full_attn_kv_sconv=True,
        full_attn_kv_sconv_residual=True,
        full_attn_kv_sconv_use_activation=False,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=lin_k_heads,
        linear_key_head_dim=64,
        linear_num_value_heads=lin_v_heads,
        linear_value_head_dim=64,
        gdn_conv_residual=False,
        gdn_no_q_conv=False,
        norm_before_gate=True,
        output_gate_type="sigmoid",
        use_linear_attn_output_sconv=True,
        use_full_attn_output_sconv=True,
        use_ffn_output_sconv=True,
        output_sconv_use_activation=False,
        hc_count=4,
        hc_lowrank=hc_lowrank,
        hc_prev_mix=True,
        hc_per_branch_norm=True,
        hc_rmsnorm_bias=True,
        num_experts=32,
        num_experts_per_tok=4,
        routed_expert_hidden_size=routed_h,
        moe_intermediate_size=moe_i,
        shared_expert_intermediate_size=shared_i,
        shared_expert_gate=True,
        router_precision="fp32_logits",
        router_score_function="sigmoid",
        moe_router_pre_softmax=True,
        router_normalize_topk_scores=True,
        moe_router_enable_expert_bias=True,
        hidden_act="situ_glu",
        hidden_act_gate_beta=4.0,
        hidden_act_up_beta=25.0,
        clip_mlp_output=1000.0,
        ple_layer_ids=(2,),
        ple_embed_dim=ple_dim,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=4,
        ngram_vocab_size_base=512,
        make_ngram_vocab_size_divisible_by=64,
        split_ngram_parts=8,
        ple_quant_scheme="fp8_pertensor",
        share_map="4->8,5->6,5->7",
        enable_yoco_decode=True,
        quant_method="mxfp4",
    )

    with torch.no_grad():
        model = Qwen4_5ForCausalLM(
            cfg,
            use_triton_mxfp4=False,
            compute_mode="bf16",
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        model.init_synthetic_weights(seed=base_seed, ple_mode="gpu_full")
        for layer in model.model.layers:
            if layer.is_moe and layer.mlp.expert_bias is not None:
                layer.mlp.expert_bias.zero_()
                n_exp = layer.mlp.num_experts
                shard = n_exp // 4
                for r in range(4):
                    e_id = r * shard + (layer.layer_idx % shard)
                    layer.mlp.expert_bias[e_id] = 4.0
        gen = torch.Generator().manual_seed(2028)
        q_mat, _ = torch.linalg.qr(
            torch.randn((cfg.hidden_size, 8), generator=gen, dtype=torch.float32)
        )
        active_ids = list(SYNTHETIC_ACTIVE_IDS)
        model.lm_head.weight.mul_(0.02)
        model.lm_head.weight[:4].mul_(0.05)
        model.lm_head.weight[active_ids] = (q_mat.T * 2.5).to(torch.bfloat16)

    save_qwen4_5_safetensors_checkpoint(model, out_dir)
    _write_synthetic_tokenizer(out_dir, cfg.vocab_size, model_max_length=max_position_embeddings)

    shard_name = "model-00001-of-00001.safetensors"
    shard_path = out_dir / shard_name
    base_tensors = _load_safetensors_shard(shard_path)

    cfg_dict = arch_config_to_hf_dict(cfg)
    cfg_dict["dtype"] = "bfloat16"
    cfg_dict["rope_parameters"] = {
        "partial_rotary_factor": cfg.partial_rotary_factor,
        "rope_theta": cfg.rope_theta,
        "rope_type": "default",
    }
    if "quantization_config" in cfg_dict:
        cfg_dict["quantization_config"]["ple_quant_scheme"] = cfg.ple_quant_scheme
        cfg_dict["quantization_config"]["modules_to_convert"] = [
            "ple.ple_embedding.ngram_embedding"
        ]
    cfg_dict.update(qsa.to_config_fields())
    (out_dir / "config.json").write_text(
        json.dumps(cfg_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    full_layers = _full_attention_layers(cfg_dict)
    qsa_tensors = _build_qsa_tensors(
        full_attn_layers=full_layers,
        hidden_size=cfg.hidden_size,
        qsa=qsa,
        seed=qsa_seed,
    )
    merged_tensors = dict(base_tensors)
    merged_tensors.update(qsa_tensors)
    write_safetensors_file(merged_tensors, shard_path)

    weight_map = {k: shard_name for k in sorted(merged_tensors.keys())}
    total_size = sum(int(t.numel() * t.element_size()) for t in merged_tensors.values())
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    qsa_shard_map = {k: shard_name for k in qsa_tensors}
    _write_manifest(
        output_dir=out_dir,
        mode="synthetic-8l",
        base_dir=None,
        config_json=cfg_dict,
        qsa=qsa,
        qsa_seed=qsa_seed,
        qsa_tensors=qsa_tensors,
        qsa_shard_map=qsa_shard_map,
        extra={
            "base_seed": int(base_seed),
            "active_ids": list(SYNTHETIC_ACTIVE_IDS),
            "active_tokens": [chr(32 + tid - 4) for tid in SYNTHETIC_ACTIVE_IDS],
            "total_tensors": len(weight_map),
            "total_size_bytes": total_size,
        },
    )
    val_summary = validate_checkpoint(
        out_dir, qsa=qsa, expected_qsa_tensors=qsa_tensors, check_all_shard_headers=True
    )
    return {"output_dir": str(out_dir), **val_summary}


def _link_file_from_base(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def build_overlay_40l(
    *,
    base_dir: Path,
    output_dir: Path,
    qsa: QSAConfigParams,
    qsa_seed: int,
) -> Dict[str, Any]:
    """Create read-only symlink overlay onto 40L 1109 checkpoint with QSA mock shard."""
    resolved_base = base_dir.resolve()
    if not resolved_base.is_dir():
        raise FileNotFoundError(f"Base checkpoint directory not found: {resolved_base}")
    out_dir = _assert_safe_output_dir(output_dir, base_dir=resolved_base)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = resolved_base / "config.json"
    base_idx_path = resolved_base / "model.safetensors.index.json"
    base_cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    base_idx = json.loads(base_idx_path.read_text(encoding="utf-8"))

    hidden_size = int(base_cfg["hidden_size"])
    full_layers = _full_attention_layers(base_cfg)

    # Symlink all original .safetensors shards + tokenizer files from read-only base_dir
    shard_names = sorted(
        f for f in os.listdir(resolved_base)
        if f.endswith(".safetensors") and f != QSA_SHARD_NAME
    )
    for fname in shard_names:
        _link_file_from_base(resolved_base / fname, out_dir / fname)
    for tok_name in TOKENIZER_FILES:
        src_tok = resolved_base / tok_name
        if src_tok.exists():
            _link_file_from_base(src_tok, out_dir / tok_name)

    qsa_tensors = _build_qsa_tensors(
        full_attn_layers=full_layers,
        hidden_size=hidden_size,
        qsa=qsa,
        seed=qsa_seed,
    )
    qsa_shard_path = out_dir / QSA_SHARD_NAME
    write_safetensors_file(qsa_tensors, qsa_shard_path)
    qsa_bytes = sum(int(t.numel() * t.element_size()) for t in qsa_tensors.values())

    updated_cfg = dict(base_cfg)
    updated_cfg.update(qsa.to_config_fields())
    (out_dir / "config.json").write_text(
        json.dumps(updated_cfg, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    base_meta = dict(base_idx.get("metadata") or {})
    base_total_size = int(base_meta.get("total_size", 0))
    updated_meta = dict(base_meta)
    updated_meta["total_size"] = base_total_size + qsa_bytes

    updated_weight_map = dict(base_idx["weight_map"])
    for key in sorted(qsa_tensors.keys()):
        updated_weight_map[key] = QSA_SHARD_NAME

    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": updated_meta,
                "weight_map": updated_weight_map,
            },
            indent=4,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    qsa_shard_map = {k: QSA_SHARD_NAME for k in qsa_tensors}
    _write_manifest(
        output_dir=out_dir,
        mode="overlay-40l",
        base_dir=resolved_base,
        config_json=updated_cfg,
        qsa=qsa,
        qsa_seed=qsa_seed,
        qsa_tensors=qsa_tensors,
        qsa_shard_map=qsa_shard_map,
        extra={
            "base_shard_count": len(shard_names),
            "base_tensor_count": len(base_idx["weight_map"]),
            "base_total_size_bytes": base_total_size,
            "qsa_shard": QSA_SHARD_NAME,
            "total_tensors": len(updated_weight_map),
            "total_size_bytes": updated_meta["total_size"],
            "quantization_config_preserved": True,
        },
    )
    val_summary = validate_checkpoint(
        out_dir, qsa=qsa, expected_qsa_tensors=qsa_tensors, check_all_shard_headers=True
    )
    return {"output_dir": str(out_dir), "symlinked_shards": len(shard_names), **val_summary}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build QSA-mocked Qwen 4.5-Flash checkpoints (synthetic-8l or overlay-40l)."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("synthetic-8l", "overlay-40l"),
        help="Checkpoint generation mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory (defaults by mode).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(DEFAULT_1109_BASE_DIR),
        help="Read-only source 1109 checkpoint directory (overlay-40l only).",
    )
    parser.add_argument("--seed", type=int, default=20261124, help="Seed for QSA mock tensors.")
    parser.add_argument("--base-seed", type=int, default=2027, help="Base seed for synthetic-8l.")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden size for synthetic-8l.")
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=8192,
        help="Max position embeddings for synthetic-8l.",
    )
    parser.add_argument("--indexer-head-dim", type=int, default=128)
    parser.add_argument("--indexer-n-heads", type=int, default=8)
    parser.add_argument("--indexer-kv-heads", type=int, default=1)
    parser.add_argument("--indexer-compress-ratio", type=int, default=4)
    parser.add_argument("--indexer-budget", type=int, default=2048)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    qsa = QSAConfigParams(
        indexer_head_dim=args.indexer_head_dim,
        indexer_n_heads=args.indexer_n_heads,
        indexer_kv_heads=args.indexer_kv_heads,
        indexer_compress_ratio=args.indexer_compress_ratio,
        indexer_budget=args.indexer_budget,
    )
    if args.mode == "synthetic-8l":
        out_dir = args.output_dir or Path(DEFAULT_8L_OUT_DIR)
        summary = build_synthetic_8l(
            output_dir=out_dir,
            qsa=qsa,
            qsa_seed=args.seed,
            base_seed=args.base_seed,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
        )
    else:
        out_dir = args.output_dir or Path(DEFAULT_40L_OUT_DIR)
        summary = build_overlay_40l(
            base_dir=args.base_dir,
            output_dir=out_dir,
            qsa=qsa,
            qsa_seed=args.seed,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
