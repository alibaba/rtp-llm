"""Build an exact 5-layer full-width representative Qwen4.5-Flash Target + DFlash2 checkpoint
from a real 40L 1109 checkpoint (zero-copy symlinks for the 128 FP8 PLE shards) or
deterministically synthesize a bit-identical full-width 5L checkpoint (--synthetic-fullwidth).

Layer mapping (0-based in 5L <- 0-based in 40L 1109):
  - L0 <- L0  : linear_attention (GDN Normal, head) + Dense MLP (12800) + HC + ShortConv
  - L1 <- L20.linear_attn (GDN YoCo Source) + L1.(ple + mlp + hc + sconv) (PLE + LatentMoE MXFP4)
  - L2 <- L19 : full_attention (QSA YoCo Source, H=8) + LatentMoE MXFP4 + HC + ShortConv
  - L3 <- L21 : linear_attention (GDN YoCo Consumer 2->4) + LatentMoE MXFP4 + HC + ShortConv
  - L4 <- L39 : full_attention (QSA YoCo Consumer 3->5, final-layer HC) + LatentMoE MXFP4
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.config.qwen4_5_config import parse_qwen4_5_config_json
from rtp_llm.models.qwen4_5.qwen4_5_weight import (
    DTYPE_NUM_BYTES,
    DTYPE_TO_TORCH,
    expected_checkpoint_tensor_specs,
    validate_qwen4_5_weight_map,
)
from rtp_llm.models_py.modules.qwen4_5.ple import (
    DEFAULT_PLE_LAYER_MULTIPLIERS,
    DEFAULT_PLE_NGRAM_VOCAB_SIZES,
    DEFAULT_PLE_WEIGHT_SCALE,
    compute_ngram_vocab_sizes,
)
from rtp_llm.models_py.modules.qwen4_5.tools.build_dflash_mock_checkpoint import (
    DEFAULT_CONFIGS,
    build_dflash2_config,
    _shard_plans,
    _tensor_specs,
)
from rtp_llm.models_py.modules.qwen4_5.tools.build_qsa_mock_checkpoint import (
    QSAConfigParams,
    QSA_SHARD_NAME,
    TOKENIZER_FILES,
    _build_qsa_tensors,
    _link_file_from_base,
)

_DEFAULT_OUT_CHUNK_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB per packed slice shard
_PLE_SHARD_RE = re.compile(
    r"^model\.layers\.1\.ple\.ple_embedding\.(?:ngram_embedding\.shard_(\d+)\.weight|ngram_embedding\.weight_scale)$"
)
_PLE_WEIGHT_ONLY_RE = re.compile(
    r"^model\.layers\.1\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
)


def _rewrite_modules_to_not_convert(items: Sequence[str]) -> List[str]:
    """Keep matching global patterns and remap copied layer indices to 5L indices."""
    layer_remap = {0: 0, 1: 1, 20: 1, 19: 2, 21: 3, 39: 4}
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        if not item.startswith("model.layers."):
            if item not in seen:
                seen.add(item)
                out.append(item)
            continue
        parts = item.split(".", 3)
        if len(parts) < 4 or not parts[2].isdigit():
            continue
        old_lid = int(parts[2])
        if old_lid not in layer_remap:
            continue
        new_lid = layer_remap[old_lid]
        # Only take linear_attn from layer 20 when mapping into L1
        if old_lid == 20 and not parts[3].startswith("linear_attn."):
            continue
        if old_lid == 1 and parts[3].startswith("linear_attn."):
            continue
        mapped = f"model.layers.{new_lid}.{parts[3]}"
        if mapped not in seen:
            seen.add(mapped)
            out.append(mapped)
    return out


def _map_1109_key_to_5l(key: str) -> Optional[str]:
    if not key.startswith("model.layers."):
        return key
    parts = key.split(".", 3)
    if len(parts) < 4 or not parts[2].isdigit():
        return None
    lid = int(parts[2])
    rest = parts[3]
    if "indexer." in rest:
        return None
    if lid == 0:
        return key
    if lid == 20 and rest.startswith("linear_attn."):
        return f"model.layers.1.{rest}"
    if lid == 1 and not rest.startswith("linear_attn."):
        return key
    if lid == 19:
        return f"model.layers.2.{rest}"
    if lid == 21:
        return f"model.layers.3.{rest}"
    if lid == 39:
        return f"model.layers.4.{rest}"
    return None


def _build_5l_target_config(base_cfg: Mapping[str, Any], qsa: QSAConfigParams) -> Dict[str, Any]:
    """Construct exact 5L full-width Qwen4.5-Flash Target config from 40L base config."""
    cfg5 = dict(base_cfg)
    cfg5["num_hidden_layers"] = 5
    cfg5["layer_types"] = [
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
    ]
    cfg5["first_k_dense_replace"] = 1
    cfg5["ple_layer_ids"] = [2]
    cfg5["share_map"] = "2->4,3->5"
    cfg5.update(qsa.to_config_fields())
    qcfg = dict(cfg5.get("quantization_config") or {})
    if "modules_to_not_convert" in qcfg and isinstance(qcfg["modules_to_not_convert"], list):
        qcfg["modules_to_not_convert"] = _rewrite_modules_to_not_convert(
            qcfg["modules_to_not_convert"]
        )
    cfg5["quantization_config"] = qcfg
    return cfg5


def _clean_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_f in os.listdir(out_dir):
        if old_f.endswith(".safetensors") or old_f in ("config.json", "model.safetensors.index.json"):
            (out_dir / old_f).unlink()


def _link_tokenizer_files(base_dir: Path, out_dir: Path) -> None:
    for fname in (*TOKENIZER_FILES, "added_tokens.json"):
        src = base_dir / fname
        if src.exists():
            _link_file_from_base(src, out_dir / fname)


def build_target_5l_slice(
    *,
    base_dir: Path,
    output_dir: Path,
    qsa_seed: int = 20261124,
    max_shard_bytes: int = _DEFAULT_OUT_CHUNK_BYTES,
) -> Dict[str, Any]:
    resolved_base = base_dir.resolve()
    if not resolved_base.is_dir():
        raise FileNotFoundError(f"Base 1109 checkpoint directory not found: {resolved_base}")
    out_dir = output_dir.resolve() if output_dir.exists() else output_dir
    if out_dir == resolved_base:
        raise ValueError("Refusing to write 5L slice into base directory")
    if "/3fs-data/3fs/mtp_test" in str(out_dir):
        raise ValueError("Forbidden output path under /3fs-data/3fs/mtp_test")

    _clean_output_dir(out_dir)

    base_cfg = json.loads((resolved_base / "config.json").read_text(encoding="utf-8"))
    base_idx = json.loads((resolved_base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    base_wm: Dict[str, str] = dict(base_idx["weight_map"])

    _link_tokenizer_files(resolved_base, out_dir)

    # Identify pure/reusable PLE shard files to symlink zero-copy
    weight_map: Dict[str, str] = {}
    symlinked_ple_shards: set[str] = set()
    copy_by_src_shard: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
    for src_key, shard_name in base_wm.items():
        dst_key = _map_1109_key_to_5l(src_key)
        if dst_key is None:
            continue
        if _PLE_SHARD_RE.match(dst_key):
            weight_map[dst_key] = shard_name
            symlinked_ple_shards.add(shard_name)
        else:
            copy_by_src_shard[shard_name].append((src_key, dst_key))

    for shard_name in sorted(symlinked_ple_shards):
        _link_file_from_base(resolved_base / shard_name, out_dir / shard_name)

    # Stream copy selected non-PLE tensors into compact slice-*.safetensors shards
    written_shards: List[str] = []
    cur_tensors: Dict[str, torch.Tensor] = {}
    cur_bytes = 0
    total_copied_bytes = 0
    slice_shard_idx = 0

    def _flush_slice_shard() -> None:
        nonlocal cur_tensors, cur_bytes, slice_shard_idx
        if not cur_tensors:
            return
        slice_shard_idx += 1
        sname = f"slice-{slice_shard_idx:05d}.safetensors"
        save_file(cur_tensors, str(out_dir / sname), metadata={"format": "pt"})
        for k in cur_tensors:
            weight_map[k] = sname
        written_shards.append(sname)
        cur_tensors.clear()
        cur_bytes = 0

    for src_shard in sorted(copy_by_src_shard.keys()):
        pairs = copy_by_src_shard[src_shard]
        with safe_open(str(resolved_base / src_shard), framework="pt", device="cpu") as sf:
            for src_key, dst_key in pairs:
                t = sf.get_tensor(src_key)
                nbytes = int(t.numel() * t.element_size())
                if cur_tensors and cur_bytes + nbytes > max_shard_bytes:
                    _flush_slice_shard()
                cur_tensors[dst_key] = t
                cur_bytes += nbytes
                total_copied_bytes += nbytes
    _flush_slice_shard()

    qsa = QSAConfigParams(
        indexer_head_dim=128,
        indexer_n_heads=8,
        indexer_kv_heads=1,
        indexer_compress_ratio=4,
        indexer_budget=2048,
    )
    qsa_tensors = _build_qsa_tensors(
        full_attn_layers=[2, 4],
        hidden_size=int(base_cfg["hidden_size"]),
        qsa=qsa,
        seed=qsa_seed,
    )
    save_file(qsa_tensors, str(out_dir / QSA_SHARD_NAME), metadata={"format": "pt"})
    for k in qsa_tensors:
        weight_map[k] = QSA_SHARD_NAME
    qsa_bytes = sum(int(t.numel() * t.element_size()) for t in qsa_tensors.values())

    ple_shard_bytes = sum(
        (resolved_base / s).stat().st_size for s in symlinked_ple_shards
    )
    total_size = total_copied_bytes + qsa_bytes + ple_shard_bytes

    cfg5 = _build_5l_target_config(base_cfg, qsa)
    (out_dir / "config.json").write_text(
        json.dumps(cfg5, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(out_dir),
        "symlinked_ple_shards": len(symlinked_ple_shards),
        "written_slice_shards": len(written_shards),
        "qsa_shard": QSA_SHARD_NAME,
        "copied_gb": round(total_copied_bytes / 1e9, 3),
        "num_tensors": len(weight_map),
    }


def _tensor_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF


def _make_synthetic_target_tensor(
    key: str,
    dtype_str: str,
    shape: Tuple[int, ...],
    *,
    base_seed: int,
    dev: torch.device,
    ple_meta: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Deterministically generate one Target checkpoint tensor (bit-identical across nodes)."""
    if key in ple_meta:
        return ple_meta[key].clone()

    gen = torch.Generator(device=dev)
    gen.manual_seed(_tensor_seed(base_seed, key))

    if dtype_str == "F8_E4M3":
        raw = torch.randint(0, 120, shape, generator=gen, dtype=torch.uint8, device=dev)
        sign = torch.randint(0, 2, shape, generator=gen, dtype=torch.uint8, device=dev) << 7
        raw.bitwise_or_(sign)
        return raw.view(torch.float8_e4m3fn).cpu()

    if dtype_str == "U8":
        if key.endswith(".weight_scale"):
            t = torch.randint(121, 124, shape, generator=gen, dtype=torch.uint8, device=dev)
        else:
            t = torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8, device=dev)
        return t.cpu()

    td = DTYPE_TO_TORCH[dtype_str]
    if key.endswith("linear_attn.A_log"):
        t = torch.empty(shape, dtype=torch.float32, device=dev).uniform_(-2.0, -0.5, generator=gen)
        return t.to(td).cpu()
    if key.endswith("linear_attn.dt_bias"):
        t = torch.empty(shape, dtype=torch.float32, device=dev).uniform_(-1.0, 0.5, generator=gen)
        return t.to(td).cpu()
    if key.endswith("mlp.expert_bias"):
        t = torch.empty(shape, dtype=torch.float32, device=dev).uniform_(-0.05, 0.05, generator=gen)
        return t.to(td).cpu()
    if len(shape) == 1:
        if "linear_attn." in key and key.endswith("norm.weight"):
            return torch.ones(shape, dtype=td)
        return torch.zeros(shape, dtype=td)
    if len(shape) == 3:
        t = torch.empty(shape, dtype=torch.float32, device=dev).normal_(0.0, 0.05, generator=gen)
        return t.to(td).cpu()

    if key.endswith("mlp.gate.weight"):
        std = 2.0 / math.sqrt(max(shape[-1], 1))
    elif key.endswith("embed_tokens.weight") or key == "lm_head.weight":
        std = 0.02
    else:
        std = min(0.02, 1.0 / math.sqrt(max(shape[-1], 1)))
    t = torch.empty(shape, dtype=torch.float32, device=dev).normal_(0.0, std, generator=gen)
    return t.to(td).cpu()


def build_synthetic_target_5l(
    *,
    base_dir: Path,
    output_dir: Path,
    seed: int = 42,
    qsa_seed: int = 20261124,
    max_shard_bytes: int = _DEFAULT_OUT_CHUNK_BYTES,
    use_gpu: bool = True,
    workers: int = 4,
) -> Dict[str, Any]:
    """Synthesize a bit-identical full-width 5L Qwen4.5-Flash QSA checkpoint without 40L shards."""
    resolved_base = base_dir.resolve()
    if not (resolved_base / "config.json").is_file():
        raise FileNotFoundError(f"Base config.json not found in: {resolved_base}")
    out_dir = output_dir.resolve() if output_dir.exists() else output_dir
    if out_dir == resolved_base:
        raise ValueError("Refusing to write 5L slice into base directory")
    if "/3fs-data/3fs/mtp_test" in str(out_dir):
        raise ValueError("Forbidden output path under /3fs-data/3fs/mtp_test")

    _clean_output_dir(out_dir)
    _link_tokenizer_files(resolved_base, out_dir)

    base_cfg = json.loads((resolved_base / "config.json").read_text(encoding="utf-8"))
    qsa = QSAConfigParams(
        indexer_head_dim=128,
        indexer_n_heads=8,
        indexer_kv_heads=1,
        indexer_compress_ratio=4,
        indexer_budget=2048,
    )
    cfg5 = _build_5l_target_config(base_cfg, qsa)
    arch_cfg = parse_qwen4_5_config_json(cfg5, strict=True)
    specs = expected_checkpoint_tensor_specs(arch_cfg, prefix="model.")

    n_heads = (arch_cfg.ngram_size - 1) * arch_cfg.heads_per_ngram
    if arch_cfg.ngram_vocab_size_base == 35000000 and n_heads == len(DEFAULT_PLE_NGRAM_VOCAB_SIZES):
        vocab_values = DEFAULT_PLE_NGRAM_VOCAB_SIZES
    else:
        vocab_values = compute_ngram_vocab_sizes(int(arch_cfg.ngram_vocab_size_base), n_heads)
    offsets_list: List[int] = []
    running = 0
    for sz in vocab_values:
        offsets_list.append(running)
        running += int(sz)

    ple_prefix = "model.layers.1.ple.ple_embedding."
    ple_meta: Dict[str, torch.Tensor] = {
        f"{ple_prefix}ngram_embedding.weight_scale": torch.tensor(
            [DEFAULT_PLE_WEIGHT_SCALE], dtype=torch.bfloat16
        ),
        f"{ple_prefix}layer_multipliers": torch.tensor(
            DEFAULT_PLE_LAYER_MULTIPLIERS[: arch_cfg.ngram_size], dtype=torch.int64
        ),
        f"{ple_prefix}ngram_heads_vocab_sizes": torch.tensor(vocab_values, dtype=torch.int64),
        f"{ple_prefix}ngram_heads_offsets": torch.tensor(offsets_list, dtype=torch.int64),
    }

    dev = torch.device("cuda:0") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
    weight_map: Dict[str, str] = {}
    total_tensor_bytes = 0
    total_disk_bytes = 0

    ple_keys: List[Tuple[int, str, str, Tuple[int, ...]]] = []
    non_ple_keys: List[Tuple[str, str, Tuple[int, ...]]] = []
    for k in sorted(specs.keys()):
        dt, shp = specs[k]
        m = _PLE_WEIGHT_ONLY_RE.match(k)
        if m is not None:
            ple_keys.append((int(m.group(1)), k, dt, shp))
        elif "indexer." in k:
            continue
        else:
            non_ple_keys.append((k, dt, shp))
    ple_keys.sort(key=lambda x: x[0])

    # 1. Write QSA indexer shard identically to build_target_5l_slice
    qsa_tensors = _build_qsa_tensors(
        full_attn_layers=[2, 4],
        hidden_size=int(cfg5["hidden_size"]),
        qsa=qsa,
        seed=qsa_seed,
    )
    qsa_path = out_dir / QSA_SHARD_NAME
    save_file(qsa_tensors, str(qsa_path), metadata={"format": "pt"})
    for k, t in qsa_tensors.items():
        weight_map[k] = QSA_SHARD_NAME
        total_tensor_bytes += int(t.numel() * t.element_size())
    total_disk_bytes += qsa_path.stat().st_size

    # 2. Stream non-PLE tensors into <= 4 GiB slice shards
    cur_tensors: Dict[str, torch.Tensor] = {}
    cur_bytes = 0
    slice_idx = 0
    written_slice_shards = 0

    def _flush_non_ple() -> None:
        nonlocal cur_bytes, slice_idx, written_slice_shards, total_disk_bytes
        if not cur_tensors:
            return
        slice_idx += 1
        sname = f"slice-{slice_idx:05d}.safetensors"
        spath = out_dir / sname
        save_file(cur_tensors, str(spath), metadata={"format": "pt"})
        for tk in cur_tensors:
            weight_map[tk] = sname
        total_disk_bytes += spath.stat().st_size
        written_slice_shards += 1
        cur_tensors.clear()
        cur_bytes = 0

    for k, dt, shp in non_ple_keys:
        nbytes = int(math.prod(shp) * DTYPE_NUM_BYTES[dt])
        if cur_tensors and cur_bytes + nbytes > max_shard_bytes:
            _flush_non_ple()
        t = _make_synthetic_target_tensor(
            k, dt, shp, base_seed=seed, dev=dev, ple_meta=ple_meta
        )
        cur_tensors[k] = t
        cur_bytes += nbytes
        total_tensor_bytes += nbytes
    _flush_non_ple()

    # 3. Generate 128 PLE FP8 shards deterministically on GPU/CPU and overlap disk saves
    num_ple = len(ple_keys)
    io_workers = max(1, min(workers, 4))
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as io_ex:
        pending: List[concurrent.futures.Future[Tuple[str, str, int, int]]] = []

        def _save_one(tk: str, sname: str, tensor: torch.Tensor) -> Tuple[str, str, int, int]:
            sp = out_dir / sname
            save_file({tk: tensor}, str(sp), metadata={"format": "pt"})
            return tk, sname, int(tensor.numel() * tensor.element_size()), sp.stat().st_size

        for s_idx, k, dt, shp in ple_keys:
            sname = f"ple-shard-{s_idx:05d}-of-{num_ple:05d}.safetensors"
            t = _make_synthetic_target_tensor(
                k, dt, shp, base_seed=seed, dev=dev, ple_meta=ple_meta
            )
            pending.append(io_ex.submit(_save_one, k, sname, t))
            if len(pending) >= io_workers * 2:
                done, not_done = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                pending = list(not_done)
                for fut in done:
                    tk, sn, tb, db = fut.result()
                    weight_map[tk] = sn
                    total_tensor_bytes += tb
                    total_disk_bytes += db
        for fut in concurrent.futures.as_completed(pending):
            tk, sn, tb, db = fut.result()
            weight_map[tk] = sn
            total_tensor_bytes += tb
            total_disk_bytes += db

    val = validate_qwen4_5_weight_map(arch_cfg, weight_map)
    if val["matched_count"] != val["expected_count"]:
        raise RuntimeError(f"Synthetic 5L weight map validation failed: {val}")

    (out_dir / "config.json").write_text(
        json.dumps(cfg5, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_tensor_bytes}, "weight_map": dict(sorted(weight_map.items()))},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(out_dir),
        "mode": "synthetic_fullwidth",
        "seed": seed,
        "ple_shards": num_ple,
        "written_slice_shards": written_slice_shards,
        "qsa_shard": QSA_SHARD_NAME,
        "total_gb": round(total_disk_bytes / 1e9, 3),
        "num_tensors": len(weight_map),
        "validated": val["matched_count"] == val["expected_count"],
    }


def _make_dflash_shard_fast(
    idx: int,
    num_shards: int,
    shard_items: Sequence[Tuple[str, Tuple[int, ...], str, str]],
    output_dir: Path,
    base_seed: int,
    use_gpu: bool,
) -> Tuple[int, List[Tuple[str, str]], int, int]:
    """Deterministic per-shard generation (can run across CPU threads or GPU)."""
    dev = torch.device("cuda:0") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
    gen = torch.Generator(device=dev)
    gen.manual_seed(base_seed + idx * 1009)

    shard_name = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
    tensors: Dict[str, torch.Tensor] = {}
    tensor_bytes = 0
    for name, shape, dtype_tag, init_kind in shard_items:
        if dtype_tag == "u8":
            if init_kind == "u8_scale":
                t = torch.randint(120, 129, shape, generator=gen, dtype=torch.uint8, device=dev)
            else:
                t = torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8, device=dev)
            t = t.cpu()
        else:
            td = torch.float32 if dtype_tag == "fp32" else torch.bfloat16
            if init_kind == "zeros":
                t = torch.zeros(shape, dtype=td)
            elif init_kind == "ones":
                t = torch.ones(shape, dtype=td)
            elif init_kind == "conv_identity":
                t = torch.zeros(shape, dtype=td)
                t[:, 0, :] = 1.0
            else:
                t = (torch.randn(shape, generator=gen, dtype=torch.float32, device=dev) * 0.02).to(td).cpu()
        tensors[name] = t
        tensor_bytes += int(t.numel() * t.element_size())

    shard_path = output_dir / shard_name
    save_file(tensors, str(shard_path), metadata={"format": "pt"})
    return idx, [(name, shard_name) for name in tensors], tensor_bytes, shard_path.stat().st_size


def build_dflash2_5l_mock(
    *,
    target_dir: Path,
    output_dir: Path,
    seed: int = 42,
    max_shard_bytes: int = 2_000_000_000,
    workers: int = 8,
    use_gpu: bool = True,
) -> Dict[str, Any]:
    out_dir = output_dir
    if "/3fs-data/3fs/mtp_test" in str(out_dir):
        raise ValueError("Forbidden output path under /3fs-data/3fs/mtp_test")
    _clean_output_dir(out_dir)

    base_cfg = json.loads((target_dir / "config.json").read_text(encoding="utf-8"))
    profile = dict(DEFAULT_CONFIGS["40l"])
    profile["target_layer_ids"] = [0, 1, 2, 3, 4]
    cfg = build_dflash2_config(base_cfg, profile, seed=seed)
    cfg["num_target_layers"] = 5
    cfg["dflash_config"]["num_target_layers"] = 5
    cfg["dflash_config"]["target_layer_ids"] = [0, 1, 2, 3, 4]

    (out_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _link_tokenizer_files(target_dir, out_dir)

    specs = _tensor_specs(cfg)
    shard_plans = _shard_plans(specs, max_shard_bytes)
    num_shards = len(shard_plans)
    weight_map: Dict[str, str] = {}
    total_tensor_bytes = 0
    total_disk_bytes = 0

    if use_gpu and torch.cuda.is_available():
        for idx, items in enumerate(shard_plans, start=1):
            print(f"[dflash_5l] shard {idx}/{num_shards} ({len(items)} tensors)...", flush=True)
            _, pairs, tbytes, dbytes = _make_dflash_shard_fast(
                idx, num_shards, items, out_dir, seed, use_gpu=True
            )
            for k, s in pairs:
                weight_map[k] = s
            total_tensor_bytes += tbytes
            total_disk_bytes += dbytes
            torch.cuda.empty_cache()
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_make_dflash_shard_fast, idx, num_shards, items, out_dir, seed, False)
                for idx, items in enumerate(shard_plans, start=1)
            ]
            for fut in concurrent.futures.as_completed(futs):
                _, pairs, tbytes, dbytes = fut.result()
                for k, s in pairs:
                    weight_map[k] = s
                total_tensor_bytes += tbytes
                total_disk_bytes += dbytes

    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_tensor_bytes}, "weight_map": dict(sorted(weight_map.items()))},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(out_dir),
        "num_shards": num_shards,
        "num_tensors": len(weight_map),
        "total_gb": round(total_disk_bytes / 1e9, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 5L full-width Qwen4.5-Flash + DFlash2 checkpoints.")
    parser.add_argument("--base_1109", "--src", dest="base_1109", required=True, help="Path to 40L 1109 base checkpoint or meta dir")
    parser.add_argument("--target_out", required=True, help="Output directory for 5L full-width QSA Target")
    parser.add_argument("--dflash_out", default="", help="Optional output directory for 5L full-width DFlash2")
    parser.add_argument("--synthetic-fullwidth", action="store_true", help="Synthesize bit-identical 5L fullwidth weights without 40L shards")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for --synthetic-fullwidth")
    parser.add_argument("--qsa_seed", type=int, default=20261124)
    parser.add_argument("--dflash_seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--dflash_only", "--dflash-only", dest="dflash_only", action="store_true", help="Skip Target build when target_out already exists")
    args = parser.parse_args()

    if args.dflash_only:
        idx_path = Path(args.target_out) / "model.safetensors.index.json"
        if not idx_path.is_file():
            raise FileNotFoundError(f"--dflash_only requires existing target index: {idx_path}")
        t_res = {"output_dir": str(args.target_out), "reused": True}
        print(f"TARGET_5L_REUSED: {args.target_out}", flush=True)
    elif args.synthetic_fullwidth:
        t_res = build_synthetic_target_5l(
            base_dir=Path(args.base_1109),
            output_dir=Path(args.target_out),
            seed=args.seed,
            qsa_seed=args.qsa_seed,
            use_gpu=not args.no_gpu,
            workers=args.workers,
        )
    else:
        t_res = build_target_5l_slice(
            base_dir=Path(args.base_1109),
            output_dir=Path(args.target_out),
            qsa_seed=args.qsa_seed,
        )
    print("TARGET_5L:", json.dumps(t_res))
    if args.dflash_out:
        d_res = build_dflash2_5l_mock(
            target_dir=Path(args.target_out),
            output_dir=Path(args.dflash_out),
            seed=args.dflash_seed,
            workers=args.workers,
            use_gpu=not args.no_gpu,
        )
        print("DFLASH_5L:", json.dumps(d_res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
