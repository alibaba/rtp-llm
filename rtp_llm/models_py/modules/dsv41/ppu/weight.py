"""V4.1 geometry, RoPE tables and checkpoint readers for PPU Engram."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch

from rtp_llm.models_py.modules.dsv41.ppu.engram import EngramShardLayout, HostShard
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUResidentBF16Linear
from rtp_llm.models_py.modules.dsv41.ppu.mhc import MHCConfig
from rtp_llm.models_py.modules.dsv41.ppu.moe import (
    DEFAULT_TOP_K,
    ROUTED_SCALING_FACTOR,
    SWIGLU_LIMIT,
)


def _cfg(config: dict, *names, default=None, required=False):
    for name in names:
        if name in config:
            return config[name]
    if required and default is None:
        raise KeyError(f"V4.1 config is missing any of {list(names)}")
    return default


def _cfg_int(config: dict, *names, default=None, required=False) -> int:
    value = _cfg(config, *names, default=default, required=required)
    return None if value is None else int(value)


@dataclass(frozen=True)
class V41Geometry:
    """Full model geometry + layer roles parsed from the V4.1 text config."""

    hidden_size: int
    num_layers: int
    vocab_size: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    window: int
    compress_ratios: Tuple[int, ...]
    kv_source_layer_ids: Tuple[int, ...]
    index_source_layer_ids: Tuple[int, ...]
    engram_layer_ids: Tuple[int, ...]
    engram_num_embeddings: Tuple[int, ...]
    engram_vocab_size: int  # bucket base, independent of tokenizer vocab_size
    engram_max_ngram_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_compressed_vocab_size: int
    engram_pad_token_id: int
    engram_token_map: Tuple[int, ...]
    image_token_id: int
    # indexer
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    # moe
    n_routed_experts: int
    n_activated_experts: int
    n_shared_experts: int
    moe_inter_dim: int
    routed_scaling_factor: float
    swiglu_limit: float
    score_func: str
    # mhc
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    norm_eps: float
    # rope
    rope_theta: float
    compress_rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    original_seq_len: int
    max_seq_len: int

    # -- roles ---------------------------------------------------------------

    def ratio(self, layer_id: int) -> int:
        if 0 <= layer_id < len(self.compress_ratios):
            return int(self.compress_ratios[layer_id])
        return 0

    def kv_source_for(self, layer_id: int) -> Optional[int]:
        sources = [s for s in self.kv_source_layer_ids if s <= layer_id]
        return max(sources) if sources else None

    def index_source_for(self, layer_id: int) -> Optional[int]:
        sources = [s for s in self.index_source_layer_ids if s <= layer_id]
        return max(sources) if sources else None

    def sm_scale(self) -> float:
        return float(self.head_dim) ** -0.5

    def mhc_config(self) -> MHCConfig:
        return MHCConfig(
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps,
            rms_eps=self.norm_eps,
        )

    def engram_enabled(self) -> bool:
        return bool(self.engram_layer_ids)

    def num_local_experts(self, ep_size: int) -> int:
        if self.n_routed_experts % ep_size:
            raise ValueError(
                f"n_routed_experts {self.n_routed_experts} not divisible by "
                f"ep_size {ep_size}"
            )
        return self.n_routed_experts // ep_size

    def validate(self) -> "V41Geometry":
        if len(self.compress_ratios) < self.num_layers:
            raise ValueError(
                f"compress_ratios has {len(self.compress_ratios)} entries, "
                f"need {self.num_layers}"
            )
        bad = [r for r in self.compress_ratios[: self.num_layers] if r not in (0, 1, 2)]
        if bad:
            raise ValueError(f"V4.1 compress ratios must be 0/1/2, got {bad}")
        for source in self.kv_source_layer_ids:
            if not 0 <= source < self.num_layers:
                raise ValueError(f"kv source layer {source} out of range")
            if self.ratio(source) not in (1, 2):
                raise ValueError(
                    f"kv source layer {source} must have ratio 1 or 2, got "
                    f"{self.ratio(source)}"
                )
        for source in self.index_source_layer_ids:
            if not 0 <= source < self.num_layers:
                raise ValueError(f"index source layer {source} out of range")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("engram_layer_ids and engram_num_embeddings must match")
        if self.head_dim != 512:
            # The 288B/528B pool codecs are fixed to 512-dim KV latents.
            raise ValueError(f"V4.1 KV head_dim must be 512, got {self.head_dim}")
        if self.index_head_dim != 128:
            raise ValueError(
                f"V4.1 indexer head dim must be 128, got {self.index_head_dim}"
            )
        if self.n_heads % self.o_groups:
            raise ValueError("n_heads must be divisible by o_groups")
        if self.n_activated_experts > self.n_routed_experts:
            raise ValueError("top-k experts exceeds the routed expert count")
        return self

    @classmethod
    def from_config(cls, config: dict) -> "V41Geometry":
        config = dict(config or {})
        rope_scaling = dict(config.get("rope_scaling") or {})
        engram_layers = tuple(
            int(x) for x in _cfg(config, "engram_layer_ids", default=[]) or []
        )
        engram_counts = tuple(
            int(x)
            for x in _cfg(config, "engram_num_embeddings", default=[] or []) or []
        )
        vocab = _cfg_int(config, "vocab_size", required=True)
        token_map = _cfg(config, "engram_token_map", default=None)
        if token_map is None:
            token_map = list(range(vocab))
        token_map = tuple(int(x) for x in token_map)
        compressed = _cfg_int(
            config, "engram_compressed_vocab_size", default=len(token_map)
        )
        num_layers = _cfg_int(config, "num_hidden_layers", "num_layers", required=True)
        # The real checkpoint's compress_ratios carries trailing entries past
        # num_hidden_layers (multimodal composite config). Truncate to the
        # text-model layer count so V41Geometry matches CacheGeometry and the
        # exported fixture manifest (layers beyond num_layers are dead data).
        ratios = tuple(int(x) for x in _cfg(config, "compress_ratios", required=True))[
            :num_layers
        ]
        geometry = cls(
            hidden_size=_cfg_int(config, "hidden_size", required=True),
            num_layers=num_layers,
            vocab_size=vocab,
            n_heads=_cfg_int(config, "num_attention_heads", "n_heads", required=True),
            head_dim=_cfg_int(config, "head_dim", "size_per_head", default=512),
            rope_head_dim=_cfg_int(config, "rope_head_dim", default=64),
            q_lora_rank=_cfg_int(config, "q_lora_rank", required=True),
            o_lora_rank=_cfg_int(config, "o_lora_rank", required=True),
            o_groups=_cfg_int(config, "o_groups", default=8),
            window=_cfg_int(config, "sliding_window", "window_size", default=128),
            compress_ratios=ratios,
            kv_source_layer_ids=tuple(
                int(x) for x in _cfg(config, "kv_source_layer_ids", default=[]) or []
            ),
            index_source_layer_ids=tuple(
                int(x) for x in _cfg(config, "index_source_layer_ids", default=[]) or []
            ),
            engram_layer_ids=engram_layers,
            engram_num_embeddings=engram_counts,
            engram_vocab_size=_cfg_int(config, "engram_vocab_size", default=vocab),
            engram_max_ngram_size=_cfg_int(config, "engram_max_ngram_size", default=4),
            engram_n_heads=_cfg_int(config, "engram_n_heads", default=8),
            engram_head_dim=_cfg_int(config, "engram_head_dim", default=256),
            engram_compressed_vocab_size=compressed,
            engram_pad_token_id=_cfg_int(
                config, "engram_pad_token_id", default=token_map[0]
            ),
            engram_token_map=token_map,
            image_token_id=_cfg_int(config, "image_token_id", default=129264),
            index_n_heads=_cfg_int(
                config,
                "indexer_head_num",
                "index_head_num",
                "index_n_heads",
                default=64,
            ),
            index_head_dim=_cfg_int(
                config, "indexer_head_dim", "index_head_dim", default=128
            ),
            index_topk=_cfg_int(config, "indexer_topk", "index_topk", default=512),
            n_routed_experts=_cfg_int(
                config, "n_routed_experts", "num_routed_experts", required=True
            ),
            n_activated_experts=_cfg_int(
                config,
                "num_experts_per_tok",
                "n_activated_experts",
                default=DEFAULT_TOP_K,
            ),
            n_shared_experts=_cfg_int(config, "n_shared_experts", required=True),
            moe_inter_dim=_cfg_int(
                config,
                "moe_intermediate_size",
                "moe_inter_dim",
                "inter_dim",
                required=True,
            ),
            routed_scaling_factor=float(
                _cfg(config, "routed_scaling_factor", default=ROUTED_SCALING_FACTOR)
            ),
            swiglu_limit=float(_cfg(config, "swiglu_limit", default=SWIGLU_LIMIT)),
            score_func=str(_cfg(config, "scoring_func", default="sqrtsoftplus")),
            hc_mult=_cfg_int(config, "hc_mult", "n_hc", default=4),
            hc_sinkhorn_iters=_cfg_int(config, "hc_sinkhorn_iters", default=20),
            hc_eps=float(_cfg(config, "hc_eps", default=1e-6)),
            norm_eps=float(_cfg(config, "rms_norm_eps", default=1e-6)),
            rope_theta=float(_cfg(config, "rope_theta", default=10000.0)),
            compress_rope_theta=float(
                _cfg(config, "compress_rope_theta", default=160000.0)
            ),
            rope_factor=float(
                _cfg(config, "rope_factor", default=rope_scaling.get("factor", 16.0))
            ),
            beta_fast=int(
                _cfg(config, "beta_fast", default=rope_scaling.get("beta_fast", 32))
            ),
            beta_slow=int(
                _cfg(config, "beta_slow", default=rope_scaling.get("beta_slow", 1))
            ),
            original_seq_len=int(
                _cfg(
                    config,
                    "original_seq_len",
                    default=rope_scaling.get("original_max_position_embeddings", 0),
                )
            ),
            max_seq_len=int(_cfg(config, "max_position_embeddings", default=4096)),
        )
        if (
            geometry.engram_enabled()
            and len(token_map)
            and max(token_map) + 1 != compressed
        ):
            raise ValueError(
                "engram token map does not cover the compressed vocab "
                f"({max(token_map) + 1} != {compressed}); supply the "
                "tokenizer-derived 'engram_token_map' in the config"
            )
        return geometry.validate()


_DTYPES_IN = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}


def read_safetensors_file(path, wanted: Optional[Sequence[str]] = None):
    """Read tensors from one safetensors file (real package when available)."""
    try:
        from safetensors.torch import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = (
                handle.keys()
                if wanted is None
                else [key for key in wanted if key in handle.keys()]
            )
            return {key: handle.get_tensor(key) for key in keys}
    except ImportError:
        pass
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        data_start = 8 + header_len
        out = {}
        for name, entry in header.items():
            if wanted is not None and name not in wanted:
                continue
            begin, end = entry["data_offsets"]
            handle.seek(data_start + begin)
            raw = handle.read(end - begin)
            dtype = _DTYPES_IN[entry["dtype"]]
            count = 1
            for size in entry["shape"]:
                count *= int(size)
            if entry["dtype"] in ("F8_E4M3", "F8_E8M0"):
                base = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                tensor = base.view(dtype) if entry["dtype"] == "F8_E4M3" else base
            else:
                tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
            out[name] = tensor.reshape(list(entry["shape"])).clone()
        return out


class _CheckpointSource:
    """Uniform ``get(name)`` over a checkpoint dir or an in-memory dict."""

    def __init__(self, checkpoint: Union[str, Path, dict, "_CheckpointSource"]):
        self.tensors: Optional[Dict[str, torch.Tensor]] = None
        self.files: Optional[Dict[str, Path]] = None
        self.root: Optional[Path] = None
        if isinstance(checkpoint, _CheckpointSource):
            self.tensors = checkpoint.tensors
            self.files = checkpoint.files
            self.root = checkpoint.root
        elif isinstance(checkpoint, (str, Path)):
            self.root = Path(checkpoint)
            index = self.root / "model.safetensors.index.json"
            self.files = {}
            if index.is_file():
                weight_map = json.loads(index.read_text())["weight_map"]
                for name, file_name in weight_map.items():
                    self.files[name] = self.root / file_name
            else:
                for file_name in sorted(self.root.glob("*.safetensors")):
                    for name in read_safetensors_file(file_name):
                        self.files[name] = file_name
            if not self.files:
                raise ValueError(
                    f"checkpoint {self.root} has no safetensors weights "
                    "(expected model.safetensors.index.json or *.safetensors)"
                )
        elif isinstance(checkpoint, dict):
            self.tensors = dict(checkpoint)
        else:
            raise TypeError(
                "checkpoint must be a directory path or a {name: tensor} dict"
            )

    def has(self, name: str) -> bool:
        if self.tensors is not None:
            return name in self.tensors
        return name in self.files  # type: ignore[operator]

    def get(self, name: str) -> Optional[torch.Tensor]:
        if self.tensors is not None:
            return self.tensors.get(name)
        path = self.files.get(name)  # type: ignore[union-attr]
        if path is None:
            return None
        tensor = read_safetensors_file(path, wanted=[name]).get(name)
        if tensor is None:
            raise KeyError(f"checkpoint declared {name} but the shard lacks it")
        return tensor

    def get_lazy(self, name: str) -> Optional[Callable[[], torch.Tensor]]:
        """Zero-arg callable opening the tensor lazily (large Engram tables)."""
        if self.tensors is not None:
            tensor = self.tensors.get(name)
            return None if tensor is None else (lambda: tensor)
        path = self.files.get(name)  # type: ignore[union-attr]
        if path is None:
            return None
        return lambda: read_safetensors_file(path, wanted=[name])[name]


def _as_scale(scale: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if scale is None:
        return None
    if scale.dtype in (torch.float8_e8m0fnu, torch.uint8, torch.int8, torch.float32):
        return scale
    if scale.element_size() == 1:
        return scale.view(torch.uint8)
    return scale.float()


def _load_dense(
    source: _CheckpointSource,
    prefix: str,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> PPUResidentBF16Linear:
    """One dense projection: ``{prefix}.weight`` (+ ``.scale``) -> BF16 resident."""
    values = source.get(f"{prefix}.weight")
    if values is None:
        raise KeyError(f"checkpoint is missing {prefix}.weight")
    scale = _as_scale(source.get(f"{prefix}.scale"))
    from rtp_llm.models_py.modules.dsv41.ppu.quant_mode import MXFP4_FP8_PTPC, quant_mode

    if prefix.endswith(".engram.wkv") and quant_mode() == MXFP4_FP8_PTPC:
        from rtp_llm.models_py.modules.dsv41.ppu.quant_conversion import convert_engram_wkv

        weight = convert_engram_wkv(values, scale)
        from rtp_llm.models_py.modules.dsv41.ppu.quant_conversion import audit_loaded_tensor

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        audit_loaded_tensor(prefix + ".weight", weight, rank)
        return PPUResidentBF16Linear(weight, output_dtype=output_dtype)
    if scale is not None and values.dtype in (torch.uint8, torch.int8):
        values = values.view(torch.float8_e4m3fn)
    if scale is None and values.dtype == torch.float8_e4m3fn:
        raise KeyError(f"{prefix}.weight is FP8 but {prefix}.scale is missing")
    if scale is not None:
        return PPUResidentBF16Linear.from_quantized(
            values, scale, "fp8_32x32_ue8m0", output_dtype=output_dtype
        )
    return PPUResidentBF16Linear.from_quantized(
        values, None, "bf16", output_dtype=output_dtype
    )


def _load_engram_shard(
    source: _CheckpointSource,
    layer_id: int,
    geometry: V41Geometry,
    enp_rank: int,
    enp_size: int,
) -> Tuple[HostShard, EngramShardLayout]:
    hash_index = geometry.engram_layer_ids.index(layer_id)
    rows = geometry.engram_num_embeddings[hash_index]
    dim = geometry.engram_head_dim
    layout = EngramShardLayout.from_table(rows, enp_size)
    payload = source.get_lazy(f"layers.{layer_id}.engram.embed.weight")
    scale = source.get_lazy(f"layers.{layer_id}.engram.embed.scale")
    if payload is None or scale is None:
        raise KeyError(
            f"layer {layer_id} is an Engram layer but the checkpoint lacks "
            f"layers.{layer_id}.engram.embed.weight/.scale"
        )
    # load_owned_rows expects the rank's OWNED row range (see its contract),
    # not the full table: slice [row_start, row_start + valid_rows).
    start = layout.row_start(enp_rank)
    count = layout.valid_rows(enp_rank)
    shard = HostShard.load_owned_rows(
        lambda: payload()[start : start + count],
        lambda: scale()[start : start + count],
        layout,
        enp_rank,
    )
    if shard.dim != dim:
        raise ValueError(
            f"layer {layer_id} engram table dim {shard.dim} != config {dim}"
        )
    return shard, layout


def _build_freqs_cis(geometry: V41Geometry, base: float, device) -> torch.Tensor:
    from rtp_llm.models_py.modules.dsv41.rope import precompute_freqs_cis

    return precompute_freqs_cis(
        geometry.rope_head_dim,
        max(geometry.max_seq_len, 16),
        geometry.original_seq_len,
        base,
        geometry.rope_factor,
        geometry.beta_fast,
        geometry.beta_slow,
        device=device,
    )


def synthetic_config(
    *,
    num_layers: int = 8,
    hidden_size: int = 64,
    vocab_size: int = 97,
    n_heads: int = 2,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    q_lora_rank: int = 32,
    o_lora_rank: int = 32,
    o_groups: int = 1,
    window: int = 8,
    index_n_heads: int = 2,
    index_topk: int = 4,
    n_routed_experts: int = 8,
    n_activated_experts: int = 6,
    n_shared_experts: int = 1,
    moe_inter_dim: int = 32,
    hc_mult: int = 4,
    max_position_embeddings: int = 256,
    kv_sources=(2, 6),
    index_sources=(2, 6),
    engram_layers=(1,),
    engram_num_embeddings=(2048,),
    engram_head_dim: int = 64,
    engram_n_heads: int = 2,
    engram_max_ngram_size: int = 4,
    kv_block_entries: int = 8,
    swa_block_entries: int = 8,
) -> dict:
    """A tiny but structurally complete V4.1 config covering every layer role.

    Default roles (8 layers): 0 SWA-only, 1 SWA+Engram, 2 ratio-2 kv/index
    producer, 3/4/5 ratio-2 consumers, 6 ratio-1 producer, 7 ratio-1
    consumer.
    """
    sources = sorted(int(s) for s in kv_sources)
    ratios = [0] * num_layers
    # The newest producer set carries ratio 1; the rest ratio 2 (mirrors the
    # real model: {2, 8, 14} ratio 2, {20} ratio 1).
    for position, source in enumerate(sources):
        ratios[source] = 1 if position == len(sources) - 1 else 2
    # Consumers adopt the newest source at or below them.
    for layer in range(num_layers):
        if ratios[layer]:
            continue
        below = [s for s in sources if s < layer]
        if below:
            ratios[layer] = ratios[below[-1]]
    compressed = vocab_size
    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "vocab_size": vocab_size,
        "num_attention_heads": n_heads,
        "head_dim": head_dim,
        "rope_head_dim": rope_head_dim,
        "q_lora_rank": q_lora_rank,
        "o_lora_rank": o_lora_rank,
        "o_groups": o_groups,
        "sliding_window": window,
        "compress_ratios": ratios,
        "kv_source_layer_ids": list(kv_sources),
        "index_source_layer_ids": list(index_sources),
        "engram_layer_ids": list(engram_layers),
        "engram_num_embeddings": list(engram_num_embeddings),
        "engram_vocab_size": vocab_size,
        "engram_max_ngram_size": engram_max_ngram_size,
        "engram_n_heads": engram_n_heads,
        "engram_head_dim": engram_head_dim,
        "engram_compressed_vocab_size": compressed,
        "engram_pad_token_id": 0,
        "indexer_head_num": index_n_heads,
        "indexer_head_dim": 128,
        "indexer_topk": index_topk,
        "n_routed_experts": n_routed_experts,
        "num_experts_per_tok": n_activated_experts,
        "n_shared_experts": n_shared_experts,
        "moe_intermediate_size": moe_inter_dim,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "scoring_func": "sqrtsoftplus",
        "hc_mult": hc_mult,
        "hc_sinkhorn_iters": 4,
        "hc_eps": 1e-6,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
        "max_position_embeddings": max_position_embeddings,
        "kv_block_entries": kv_block_entries,
        "swa_block_entries": swa_block_entries,
    }
