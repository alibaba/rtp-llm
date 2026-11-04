"""PPU weight loading for DeepSeek V4.1 Flash.

Two layers live here:

1. A STANDALONE core (no RTP framework imports needed):
   :class:`V41Geometry` parses the V4.1 text config dict (checkpoint
   ``config.json`` merged with ``text_config``), :func:`load_ppu_weights`
   builds a :class:`PPUWeightBundle` from a checkpoint directory (real
   safetensors layout, read with the ``safetensors`` package when present and
   a minimal built-in reader otherwise) or from an in-memory ``{name:
   tensor}`` dict, and :func:`write_synthetic_checkpoint` materializes small
   synthetic checkpoints for offline drivers and tests. The offline drivers
   call this core directly.

2. A framework-facing descriptor class :class:`DeepSeekV41PPUWeight` for the
   A0 backend route (``get_v41_backend("ppu").weight_cls``). When the RTP
   framework chain is importable it subclasses the CUDA-side
   ``DeepSeekV41Weight`` descriptor (identical checkpoint keys — the PPU
   provider only changes how tensors are consumed), and additionally offers
   :meth:`DeepSeekV41PPUWeight.from_model_weights` to build the bundle from a
   populated framework ``ModelWeights``. On hosts without the framework chain
   the descriptor still exists (so ``V41Backend.validate`` passes) but its
   descriptor methods fail loudly.

Weight conventions (strategy ``bf16_dense_w4a16_moe_sparse_mla_host_enp_v1``):

* dense / attention / shared-expert / router / Engram-WKV FP8 32x32 UE8M0
  weights are dequantized ONCE at load into resident BF16
  (:func:`PPUResidentBF16Linear.from_quantized`);
* routed experts stay native packed FP4 e2m1 + UE8M0 group-32 scales, sliced
  to this EP rank's local experts and handed to ``PPUOps.bind_expert_weights``
  (stacked ``[E, 2I, K/2]`` / ``[E, D, I/2]`` with ``w13 = [w1; w3]`` rows);
* Engram tables are NOT copied per rank: only this ENP rank's owned rows are
  materialized into an L1 ``HostShard`` (layout from
  ``EngramShardLayout.from_table``), plus the FP8 WKV projection, q/k gate
  weights and hash parameters;
* mHC coefficients load as FP32 and are bound via ``bind_mhc_config``;
* RoPE tables (main + compressed) are precomputed with the reference
  ``dsv41.rope`` math.

Checkpoint key names mirror the CUDA descriptor graph in
``rtp_llm/models/deepseek_v4.py`` / ``deepseek_v41.py`` (``layers.{i}.attn.
wq_a.weight`` + ``.scale``, ``ffn.experts.{id}.w1.weight`` + ``.scale``,
``layers.{i}.engram.*``, ``embed.weight``, ``norm.weight``, ``head.weight``).
"""

from __future__ import annotations

import json
import importlib.util
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from rtp_llm.models_py.modules.dsv41.ppu.engram import (
    EngramHashParams,
    EngramShardLayout,
    HostShard,
)
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUResidentBF16Linear
from rtp_llm.models_py.modules.dsv41.ppu.mhc import MHCConfig
from rtp_llm.models_py.modules.dsv41.ppu.moe import (
    DEFAULT_TOP_K,
    ROUTED_SCALING_FACTOR,
    SWIGLU_LIMIT,
)

__all__ = [
    "V41Geometry",
    "PPUWeightBundle",
    "load_ppu_weights",
    "write_synthetic_checkpoint",
    "DeepSeekV41PPUWeight",
]


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


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Minimal safetensors IO (used when the safetensors package is unavailable)
# ---------------------------------------------------------------------------

_DTYPES_OUT = {
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int64: "I64",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e8m0fnu: "U8",
}
_DTYPES_IN = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.contiguous()
    if tensor.dtype == torch.bfloat16:
        # numpy has no bf16; move the raw little-endian bytes instead.
        return tensor.view(torch.uint16).numpy().tobytes()
    if tensor.dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn):
        return tensor.view(torch.uint8).numpy().tobytes()
    return tensor.numpy().tobytes()


def write_safetensors(path, tensors: Dict[str, torch.Tensor]) -> None:
    """Write one safetensors file (real package when available, else builtin)."""
    try:
        from safetensors.torch import save_file

        save_file(
            {
                name: (
                    tensor
                    if tensor.dtype != torch.float8_e8m0fnu
                    else tensor.view(torch.uint8)
                )
                for name, tensor in tensors.items()
            },
            str(path),
        )
        return
    except Exception:
        pass
    header = {}
    blob = bytearray()
    for name, tensor in tensors.items():
        raw = _tensor_bytes(tensor)
        header[str(name)] = {
            "dtype": _DTYPES_OUT[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [len(blob), len(blob) + len(raw)],
        }
        blob.extend(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(bytes(blob))


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


# ---------------------------------------------------------------------------
# Checkpoint sources
# ---------------------------------------------------------------------------


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


def _load_dense(source: _CheckpointSource, prefix: str) -> PPUResidentBF16Linear:
    """One dense projection: ``{prefix}.weight`` (+ ``.scale``) -> BF16 resident."""
    values = source.get(f"{prefix}.weight")
    if values is None:
        raise KeyError(f"checkpoint is missing {prefix}.weight")
    scale = _as_scale(source.get(f"{prefix}.scale"))
    if scale is not None and values.dtype in (torch.uint8, torch.int8):
        values = values.view(torch.float8_e4m3fn)
    if scale is None and values.dtype == torch.float8_e4m3fn:
        raise KeyError(f"{prefix}.weight is FP8 but {prefix}.scale is missing")
    if scale is not None:
        return PPUResidentBF16Linear.from_quantized(values, scale, "fp8_32x32_ue8m0")
    return PPUResidentBF16Linear.from_quantized(values, None, "bf16")


def _load_vector(
    source: _CheckpointSource, name: str, dtype=torch.bfloat16
) -> Optional[torch.Tensor]:
    tensor = source.get(name)
    if tensor is None:
        return None
    return tensor.to(dtype)


# ---------------------------------------------------------------------------
# Per-layer weight record
# ---------------------------------------------------------------------------


@dataclass
class PPULayerWeights:
    """Everything one decoder layer needs, post-load."""

    layer_id: int
    ratio: int
    is_kv_source: bool
    is_index_source: bool
    # norms
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    q_norm: Optional[torch.Tensor]
    kv_norm: Optional[torch.Tensor]
    attn_sink: Optional[torch.Tensor]
    # attention projections (BF16 resident)
    wq_a: PPUResidentBF16Linear
    wq_b: PPUResidentBF16Linear
    wkv: PPUResidentBF16Linear
    wo_a: PPUResidentBF16Linear
    wo_b: PPUResidentBF16Linear
    # kv-source (compressor + indexer-K) weights
    compressor_wkv: Optional[PPUResidentBF16Linear] = None
    compressor_wgate: Optional[PPUResidentBF16Linear] = None
    compressor_norm: Optional[torch.Tensor] = None
    index_wk: Optional[PPUResidentBF16Linear] = None
    index_k_norm: Optional[torch.Tensor] = None
    # index-source (query side) weights
    index_wq: Optional[PPUResidentBF16Linear] = None
    index_weights_proj: Optional[torch.Tensor] = None
    # mHC coefficients (FP32)
    hc_attn_fn: torch.Tensor = None
    hc_attn_base: torch.Tensor = None
    hc_attn_scale: torch.Tensor = None
    hc_ffn_fn: torch.Tensor = None
    hc_ffn_base: torch.Tensor = None
    hc_ffn_scale: torch.Tensor = None
    # MoE
    router_weight: torch.Tensor = None
    router_bias: Optional[torch.Tensor] = None
    shared_w13: Optional[PPUResidentBF16Linear] = None
    shared_w2: Optional[PPUResidentBF16Linear] = None
    routed_w13_weight: Optional[torch.Tensor] = None  # [E, 2I, K/2] int8/uint8
    routed_w13_scale: Optional[torch.Tensor] = None  # [E, 2I, K/32]
    routed_w2_weight: Optional[torch.Tensor] = None  # [E, D, I/2]
    routed_w2_scale: Optional[torch.Tensor] = None  # [E, D, I/32]
    # Engram (table itself lives in the bundle's shards)
    engram_wkv: Optional[PPUResidentBF16Linear] = None
    engram_q_weight: Optional[torch.Tensor] = None
    engram_k_weight: Optional[torch.Tensor] = None

    def mhc_parameters(self, which: str) -> Dict[str, torch.Tensor]:
        return {
            "fn": getattr(self, f"hc_{which}_fn"),
            "base": getattr(self, f"hc_{which}_base"),
            "scale": getattr(self, f"hc_{which}_scale"),
        }


# ---------------------------------------------------------------------------
# The weight bundle
# ---------------------------------------------------------------------------


class PPUWeightBundle:
    """All weights of one EP/ENP rank, ready to bind into ``PPUOps``."""

    def __init__(
        self,
        geometry: V41Geometry,
        *,
        embedding: torch.Tensor,
        final_norm: torch.Tensor,
        lm_head: Optional[torch.Tensor],
        layers: Dict[int, PPULayerWeights],
        engram_shards: Dict[int, Tuple[HostShard, EngramShardLayout]],
        engram_hash_params: Optional[EngramHashParams],
        freqs_cis_main: torch.Tensor,
        freqs_cis_compressed: torch.Tensor,
    ):
        self.geometry = geometry
        self.embedding = embedding
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.layers = layers
        self.engram_shards = engram_shards
        self.engram_hash_params = engram_hash_params
        self.freqs_cis_main = freqs_cis_main
        self.freqs_cis_compressed = freqs_cis_compressed

    def freqs_for(self, layer_id: int) -> torch.Tensor:
        """The layer's RoPE table (compressed base for ratio 1/2 layers)."""
        return (
            self.freqs_cis_compressed
            if self.geometry.ratio(layer_id)
            else self.freqs_cis_main
        )

    def bind(self, ops, *, enpm_group: Optional[object] = None) -> None:
        """Inject every binding the PPUOps contract needs.

        * ``bind_mhc_config`` with the model's MHCConfig;
        * ``bind_expert_weights(layer_id, (w13_w, w13_s), (w2_w, w2_s))`` per
          MoE layer (native FP4 shards, never BF16-expanded);
        * ``bind_engram_shard(layer_id, shard, layout, group)`` per Engram
          layer.
        """
        ops.bind_mhc_config(self.geometry.mhc_config())
        for layer_id, layer in self.layers.items():
            if layer.routed_w13_weight is None:
                raise RuntimeError(
                    f"layer {layer_id} has no routed expert weights; the PPU "
                    "MoE path refuses to run without them"
                )
            ops.bind_expert_weights(
                layer_id,
                (layer.routed_w13_weight, layer.routed_w13_scale),
                (layer.routed_w2_weight, layer.routed_w2_scale),
            )
        for layer_id, (shard, layout) in self.engram_shards.items():
            ops.bind_engram_shard(layer_id, shard, layout, group=enpm_group)

    def to_device(self, device) -> "PPUWeightBundle":
        """Move the resident tensors to ``device`` (host shards stay on CPU)."""
        device = torch.device(device)

        def move(value):
            if value is None or not isinstance(value, torch.Tensor):
                return value
            return value.to(device)

        def move_linear(linear):
            if linear is None:
                return None
            return PPUResidentBF16Linear(move(linear.weight), move(linear.bias))

        layers = {}
        for layer_id, layer in self.layers.items():
            layers[layer_id] = PPULayerWeights(
                **{
                    key: (
                        move_linear(value)
                        if isinstance(value, PPUResidentBF16Linear)
                        else move(value)
                    )
                    for key, value in layer.__dict__.items()
                }
            )
        return PPUWeightBundle(
            self.geometry,
            embedding=move(self.embedding),
            final_norm=move(self.final_norm),
            lm_head=move(self.lm_head),
            layers=layers,
            engram_shards=self.engram_shards,
            engram_hash_params=self.engram_hash_params,
            freqs_cis_main=move(self.freqs_cis_main),
            freqs_cis_compressed=move(self.freqs_cis_compressed),
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _require(tensor: Optional[torch.Tensor], name: str) -> torch.Tensor:
    if tensor is None:
        raise KeyError(f"checkpoint is missing {name}")
    return tensor


def _load_routed_experts(
    source: _CheckpointSource,
    layer_id: int,
    geometry: V41Geometry,
    ep_rank: int,
    ep_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack this rank's local experts: w13=[w1;w3] rows, w2, with scales."""
    local = geometry.num_local_experts(ep_size)
    start, end = ep_rank * local, (ep_rank + 1) * local
    inter, hidden = geometry.moe_inter_dim, geometry.hidden_size
    prefix = f"layers.{layer_id}.ffn.experts"
    stacked = source.get(f"{prefix}.w13.weight")
    if stacked is not None:
        w13_w = _require(stacked, f"{prefix}.w13.weight")
        w13_s = _require(
            _as_scale(source.get(f"{prefix}.w13.scale")), f"{prefix}.w13.scale"
        )
        w2_w = _require(source.get(f"{prefix}.w2.weight"), f"{prefix}.w2.weight")
        w2_s = _require(
            _as_scale(source.get(f"{prefix}.w2.scale")), f"{prefix}.w2.scale"
        )
        if w13_w.dtype in (torch.int8, torch.uint8):
            if tuple(w13_w.shape) != (local, 2 * inter, hidden // 2):
                raise ValueError(
                    f"layer {layer_id} stacked w13 shape {tuple(w13_w.shape)} "
                    f"!= {(local, 2 * inter, hidden // 2)}"
                )
            return w13_w, w13_s, w2_w, w2_s
        raise TypeError("stacked expert weights must be packed int8/uint8 FP4")
    w13_rows, w13_scales, w2_rows, w2_scales = [], [], [], []
    for expert in range(start, end):
        pieces_w, pieces_s = [], []
        for sub in ("w1", "w3"):
            name = f"{prefix}.{expert}.{sub}.weight"
            weight = _require(source.get(name), name)
            scale_name = f"{prefix}.{expert}.{sub}.scale"
            scale = _as_scale(_require(source.get(scale_name), scale_name))
            if weight.dtype not in (torch.int8, torch.uint8):
                raise TypeError(f"{name} must be packed int8/uint8 FP4")
            expected = (inter, hidden // 2)
            if tuple(weight.shape) != expected:
                raise ValueError(f"{name} shape {tuple(weight.shape)} != {expected}")
            pieces_w.append(weight)
            pieces_s.append(scale)
        w13_rows.append(torch.cat(pieces_w, dim=0))
        w13_scales.append(torch.cat(pieces_s, dim=0))
        name = f"{prefix}.{expert}.w2.weight"
        weight = _require(source.get(name), name)
        scale = _as_scale(
            _require(
                source.get(f"{prefix}.{expert}.w2.scale"), f"{prefix}.{expert}.w2.scale"
            )
        )
        if tuple(weight.shape) != (hidden, inter // 2):
            raise ValueError(
                f"{name} shape {tuple(weight.shape)} != {(hidden, inter // 2)}"
            )
        w2_rows.append(weight)
        w2_scales.append(scale)
    return (
        torch.stack(w13_rows),
        torch.stack(w13_scales),
        torch.stack(w2_rows),
        torch.stack(w2_scales),
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


def load_ppu_weights(
    config: dict,
    checkpoint: Union[str, Path, dict],
    *,
    ep_rank: int = 0,
    ep_size: int = 1,
    enp_size: Optional[int] = None,
    enp_rank: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> PPUWeightBundle:
    """Build the rank's :class:`PPUWeightBundle` from a config + checkpoint.

    ``config`` is the V4.1 text config dict (``config.json`` merged with its
    ``text_config``). ``checkpoint`` is either a checkpoint directory (real or
    synthetic safetensors; sharded via ``model.safetensors.index.json``) or an
    in-memory ``{checkpoint_name: tensor}`` dict. Engram tables are sliced to
    the ENP rank's owned rows only. ``enp_size`` defaults to ``ep_size``
    (single-process runs use 1).
    """
    geometry = V41Geometry.from_config(config)
    enp_size = ep_size if enp_size is None else int(enp_size)
    enp_rank = ep_rank if enp_rank is None else int(enp_rank)
    source = _CheckpointSource(checkpoint)

    embedding = _require(source.get("embed.weight"), "embed.weight").to(torch.bfloat16)
    final_norm = _require(source.get("norm.weight"), "norm.weight").to(torch.bfloat16)
    lm_head = source.get("head.weight")
    if lm_head is not None:
        lm_head = lm_head.to(torch.bfloat16)

    mix_width = geometry.hc_mult * (geometry.hc_mult + 2)
    hc_dim = geometry.hc_mult * geometry.hidden_size
    layers: Dict[int, PPULayerWeights] = {}
    engram_shards: Dict[int, Tuple[HostShard, EngramShardLayout]] = {}
    for layer_id in range(geometry.num_layers):
        prefix = f"layers.{layer_id}"
        ratio = geometry.ratio(layer_id)
        is_kv_source = layer_id in geometry.kv_source_layer_ids
        is_index_source = layer_id in geometry.index_source_layer_ids
        layer = PPULayerWeights(
            layer_id=layer_id,
            ratio=ratio,
            is_kv_source=is_kv_source,
            is_index_source=is_index_source,
            attn_norm=_require(
                _load_vector(source, f"{prefix}.attn_norm.weight"),
                f"{prefix}.attn_norm",
            ),
            ffn_norm=_require(
                _load_vector(source, f"{prefix}.ffn_norm.weight"), f"{prefix}.ffn_norm"
            ),
            q_norm=_load_vector(source, f"{prefix}.attn.q_norm.weight"),
            kv_norm=_load_vector(source, f"{prefix}.attn.kv_norm.weight"),
            attn_sink=_load_vector(
                source, f"{prefix}.attn.attn_sink", dtype=torch.float32
            ),
            wq_a=_load_dense(source, f"{prefix}.attn.wq_a"),
            wq_b=_load_dense(source, f"{prefix}.attn.wq_b"),
            wkv=_load_dense(source, f"{prefix}.attn.wkv"),
            wo_a=_load_dense(source, f"{prefix}.attn.wo_a"),
            wo_b=_load_dense(source, f"{prefix}.attn.wo_b"),
        )
        if is_kv_source:
            layer.compressor_wkv = _load_dense(source, f"{prefix}.attn.compressor.wkv")
            layer.compressor_norm = _load_vector(
                source, f"{prefix}.attn.compressor.norm.weight"
            )
            if ratio == 2:
                layer.compressor_wgate = _load_dense(
                    source, f"{prefix}.attn.compressor.wgate"
                )
            layer.index_wk = _load_dense(source, f"{prefix}.attn.indexer.wk")
            layer.index_k_norm = _load_vector(
                source, f"{prefix}.attn.indexer.k_norm.weight"
            )
        if is_index_source:
            layer.index_wq = _load_dense(source, f"{prefix}.attn.indexer.wq_b")
            layer.index_weights_proj = _load_vector(
                source, f"{prefix}.attn.indexer.weights_proj.weight"
            )
        for which in ("attn", "ffn"):
            setattr(
                layer,
                f"hc_{which}_fn",
                _require(
                    source.get(f"{prefix}.hc_{which}_fn"), f"{prefix}.hc_{which}_fn"
                ).to(torch.float32),
            )
            setattr(
                layer,
                f"hc_{which}_base",
                _require(
                    source.get(f"{prefix}.hc_{which}_base"),
                    f"{prefix}.hc_{which}_base",
                ).to(torch.float32),
            )
            setattr(
                layer,
                f"hc_{which}_scale",
                _require(
                    source.get(f"{prefix}.hc_{which}_scale"),
                    f"{prefix}.hc_{which}_scale",
                ).to(torch.float32),
            )
        layer.router_weight = _require(
            source.get(f"{prefix}.ffn.gate.weight"), f"{prefix}.ffn.gate.weight"
        ).to(torch.bfloat16)
        layer.router_bias = _load_vector(
            source, f"{prefix}.ffn.gate.bias", dtype=torch.float32
        )
        if geometry.n_shared_experts:
            w1 = _load_dense(source, f"{prefix}.ffn.shared_experts.w1")
            w3 = _load_dense(source, f"{prefix}.ffn.shared_experts.w3")
            layer.shared_w13 = PPUResidentBF16Linear(
                torch.cat((w1.weight, w3.weight), dim=0)
            )
            layer.shared_w2 = _load_dense(source, f"{prefix}.ffn.shared_experts.w2")
        (
            layer.routed_w13_weight,
            layer.routed_w13_scale,
            layer.routed_w2_weight,
            layer.routed_w2_scale,
        ) = _load_routed_experts(source, layer_id, geometry, ep_rank, ep_size)
        if layer_id in geometry.engram_layer_ids:
            layer.engram_wkv = _load_dense(source, f"{prefix}.engram.wkv")
            layer.engram_q_weight = _require(
                source.get(f"{prefix}.engram.q_weight"), f"{prefix}.engram.q_weight"
            ).to(torch.bfloat16)
            layer.engram_k_weight = _require(
                source.get(f"{prefix}.engram.k_weight"), f"{prefix}.engram.k_weight"
            ).to(torch.bfloat16)
            engram_shards[layer_id] = _load_engram_shard(
                source, layer_id, geometry, enp_rank, enp_size
            )
        # Shape validation of the mHC coefficients.
        for which in ("attn", "ffn"):
            fn = getattr(layer, f"hc_{which}_fn")
            if tuple(fn.shape) != (mix_width, hc_dim):
                raise ValueError(
                    f"layer {layer_id} hc_{which}_fn shape {tuple(fn.shape)} "
                    f"!= {(mix_width, hc_dim)}"
                )
        layers[layer_id] = layer

    hash_params = None
    if geometry.engram_enabled():
        hash_params = EngramHashParams.build(
            layer_ids=geometry.engram_layer_ids,
            max_ngram_size=geometry.engram_max_ngram_size,
            n_heads=geometry.engram_n_heads,
            vocab_size=geometry.vocab_size,
            compressed_vocab_size=geometry.engram_compressed_vocab_size,
            num_embeddings=geometry.engram_num_embeddings,
            token_map=list(geometry.engram_token_map),
            pad_token_id=geometry.engram_pad_token_id,
        )

    device = torch.device(device) if device is not None else embedding.device
    bundle = PPUWeightBundle(
        geometry,
        embedding=embedding,
        final_norm=final_norm,
        lm_head=lm_head,
        layers=layers,
        engram_shards=engram_shards,
        engram_hash_params=hash_params,
        freqs_cis_main=_build_freqs_cis(geometry, geometry.rope_theta, device),
        freqs_cis_compressed=_build_freqs_cis(
            geometry, geometry.compress_rope_theta, device
        ),
    )
    # Move EVERY resident (per-layer linears, norms, mHC coefficients, router
    # weights) to the target device — only embedding/norm/head/freqs moved
    # before, so device runs crashed on CPU layer weights (the first mHC
    # Triton call rejected the CPU ``fn`` tensor). Host Engram shards stay on
    # the CPU by design; hash params stay host-side (CPU window math).
    return bundle.to_device(device)


# ---------------------------------------------------------------------------
# Synthetic checkpoints (offline drivers / tests)
# ---------------------------------------------------------------------------


def _rand_like(shape, dtype, generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).to(dtype)


def _quantize_fp8_block32(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of L1's dequant: BF16 [N, K] -> FP8 payload + UE8M0 scales."""
    n, k = weight.shape
    blocks = weight.float().reshape(n // 32, 32, k // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3))
    exponent = torch.round(torch.log2(amax.clamp_min(1e-30) / 448.0)) + 127
    exponent = exponent.clamp(1, 254).to(torch.uint8)
    scale = torch.exp2(exponent.float() - 127.0)
    codes = (blocks / scale[:, None, :, None]).clamp(-448, 448)
    payload = codes.reshape(n, k).to(torch.float8_e4m3fn)
    return payload, exponent


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


def write_synthetic_checkpoint(
    config: dict,
    path,
    *,
    seed: int = 0,
    with_lm_head: bool = True,
    hc_fn_scale: float = 1.0,
) -> Path:
    """Materialize a synthetic checkpoint directory for ``config``.

    ``hc_fn_scale`` shrinks the mHC mixing matrices: the default random
    coefficients grow the mHC residual trunk to ~1e5 while the final hidden
    stays O(1) — near-cancellation amplifies a single bf16 ULP into large
    output differences, which makes batch-shape-dependent GEMM rounding
    (CP8 chunks vs a full CP1 sequence) look like a functional error. A
    scale < 1 keeps the trunk O(1) so cross-layout comparisons are stable.
    """
    geometry = V41Geometry.from_config(config)
    generator = torch.Generator().manual_seed(seed)
    tensors: Dict[str, torch.Tensor] = {}

    def add_dense(name: str, out_features: int, in_features: int):
        weight = _rand_like((out_features, in_features), torch.bfloat16, generator)
        payload, scale = _quantize_fp8_block32(weight.float())
        tensors[f"{name}.weight"] = payload
        tensors[f"{name}.scale"] = scale

    def add_bf16(name: str, shape):
        tensors[name] = _rand_like(shape, torch.bfloat16, generator)

    def add_fp32(name: str, shape):
        tensors[name] = _rand_like(shape, torch.float32, generator)

    def add_hc_fn(name: str, shape):
        tensors[name] = _rand_like(shape, torch.float32, generator) * float(hc_fn_scale)

    add_bf16("embed.weight", (geometry.vocab_size, geometry.hidden_size))
    add_bf16("norm.weight", (geometry.hidden_size,))
    if with_lm_head:
        add_bf16("head.weight", (geometry.vocab_size, geometry.hidden_size))
    mix_width = geometry.hc_mult * (geometry.hc_mult + 2)
    hc_dim = geometry.hc_mult * geometry.hidden_size
    group_heads = geometry.n_heads // geometry.o_groups
    for layer_id in range(geometry.num_layers):
        prefix = f"layers.{layer_id}"
        add_bf16(f"{prefix}.attn_norm.weight", (geometry.hidden_size,))
        add_bf16(f"{prefix}.ffn_norm.weight", (geometry.hidden_size,))
        add_bf16(f"{prefix}.attn.q_norm.weight", (geometry.q_lora_rank,))
        add_bf16(f"{prefix}.attn.kv_norm.weight", (geometry.head_dim,))
        add_fp32(f"{prefix}.attn.attn_sink", (geometry.n_heads,))
        add_dense(f"{prefix}.attn.wq_a", geometry.q_lora_rank, geometry.hidden_size)
        add_dense(
            f"{prefix}.attn.wq_b",
            geometry.n_heads * geometry.head_dim,
            geometry.q_lora_rank,
        )
        add_dense(f"{prefix}.attn.wkv", geometry.head_dim, geometry.hidden_size)
        add_dense(
            f"{prefix}.attn.wo_a",
            geometry.o_groups * geometry.o_lora_rank,
            group_heads * geometry.head_dim,
        )
        add_dense(
            f"{prefix}.attn.wo_b",
            geometry.hidden_size,
            geometry.o_groups * geometry.o_lora_rank,
        )
        if layer_id in geometry.kv_source_layer_ids:
            add_dense(
                f"{prefix}.attn.compressor.wkv", geometry.head_dim, geometry.hidden_size
            )
            add_bf16(f"{prefix}.attn.compressor.norm.weight", (geometry.head_dim,))
            if geometry.ratio(layer_id) == 2:
                add_dense(
                    f"{prefix}.attn.compressor.wgate",
                    geometry.head_dim,
                    geometry.hidden_size,
                )
            add_dense(
                f"{prefix}.attn.indexer.wk", geometry.index_head_dim, geometry.head_dim
            )
            add_bf16(f"{prefix}.attn.indexer.k_norm.weight", (geometry.index_head_dim,))
        if layer_id in geometry.index_source_layer_ids:
            add_dense(
                f"{prefix}.attn.indexer.wq_b",
                geometry.index_n_heads * geometry.index_head_dim,
                geometry.q_lora_rank,
            )
            add_bf16(
                f"{prefix}.attn.indexer.weights_proj.weight",
                (geometry.index_n_heads, geometry.hidden_size),
            )
        for which in ("attn", "ffn"):
            add_hc_fn(f"{prefix}.hc_{which}_fn", (mix_width, hc_dim))
            add_fp32(f"{prefix}.hc_{which}_base", (mix_width,))
            add_fp32(f"{prefix}.hc_{which}_scale", (3,))
        add_bf16(
            f"{prefix}.ffn.gate.weight",
            (geometry.n_routed_experts, geometry.hidden_size),
        )
        add_fp32(f"{prefix}.ffn.gate.bias", (geometry.n_routed_experts,))
        if geometry.n_shared_experts:
            add_dense(
                f"{prefix}.ffn.shared_experts.w1",
                geometry.moe_inter_dim,
                geometry.hidden_size,
            )
            add_dense(
                f"{prefix}.ffn.shared_experts.w3",
                geometry.moe_inter_dim,
                geometry.hidden_size,
            )
            add_dense(
                f"{prefix}.ffn.shared_experts.w2",
                geometry.hidden_size,
                geometry.moe_inter_dim,
            )
        for expert in range(geometry.n_routed_experts):
            for sub, out_features, in_features in (
                ("w1", geometry.moe_inter_dim, geometry.hidden_size),
                ("w2", geometry.hidden_size, geometry.moe_inter_dim),
                ("w3", geometry.moe_inter_dim, geometry.hidden_size),
            ):
                name = f"{prefix}.ffn.experts.{expert}.{sub}"
                packed = torch.randint(
                    -128, 128, (out_features, in_features // 2), generator=generator
                ).to(torch.int8)
                tensors[f"{name}.weight"] = packed
                tensors[f"{name}.scale"] = torch.randint(
                    118,
                    136,
                    (out_features, in_features // 32),
                    generator=generator,
                ).to(torch.uint8)
        if layer_id in geometry.engram_layer_ids:
            hash_index = geometry.engram_layer_ids.index(layer_id)
            rows = geometry.engram_num_embeddings[hash_index]
            dim = geometry.engram_head_dim
            payload = _rand_like((rows, dim), torch.bfloat16, generator).to(
                torch.float8_e4m3fn
            )
            tensors[f"{prefix}.engram.embed.weight"] = payload
            tensors[f"{prefix}.engram.embed.scale"] = torch.randint(
                120, 136, (rows, dim // 32), generator=generator
            ).to(torch.uint8)
            n_hash_cols = (geometry.engram_max_ngram_size - 1) * geometry.engram_n_heads
            add_dense(
                f"{prefix}.engram.wkv",
                (geometry.hc_mult + 1) * geometry.hidden_size,
                n_hash_cols * dim,
            )
            add_bf16(
                f"{prefix}.engram.q_weight",
                (geometry.hc_mult, geometry.hidden_size),
            )
            add_bf16(
                f"{prefix}.engram.k_weight",
                (geometry.hc_mult, geometry.hidden_size),
            )
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    write_safetensors(root / "weights.safetensors", tensors)
    (root / "config.json").write_text(json.dumps(config, indent=2))
    return root


# ---------------------------------------------------------------------------
# Framework-facing descriptor (A0 weight_cls route)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - depends on the full framework chain
    # Native server startup makes this top-level extension importable before
    # resolving weight_cls. Standalone drivers do not need it. Importing the
    # framework without an extension triggers ops.find_upper_so, which walks
    # every ancestor's entire tree on a CPU checkout (including network mounts).
    native_libs = Path(__file__).resolve().parents[5] / "rtp_llm" / "libs"
    if importlib.util.find_spec("libth_transformer_config") is None and not any(
        native_libs.glob("libth_transformer_config*.so")
    ):
        raise ImportError("native RTP config extension is unavailable")
    from rtp_llm.models.deepseek_v41 import DeepSeekV41Weight as _FrameworkBase

    _FRAMEWORK_IMPORT_ERROR: Optional[Exception] = None
except Exception as error:  # noqa: BLE001 - report any import failure
    _FrameworkBase = object  # type: ignore[assignment,misc]
    _FRAMEWORK_IMPORT_ERROR = error


class DeepSeekV41PPUWeight(_FrameworkBase):  # type: ignore[misc,valid-type]
    """``weight_cls`` for ``get_v41_backend("ppu")``.

    On a host with the full RTP framework chain this subclasses the CUDA-side
    ``DeepSeekV41Weight`` (the checkpoint descriptor graph is identical — the
    PPU provider only consumes the loaded tensors differently: FP8 dequant to
    resident BF16 instead of MXFP8 linears). On hosts without the chain the
    class still exists so ``V41Backend.validate`` passes, but descriptor
    methods raise instead of silently serving a partial graph.

    Use :meth:`from_model_weights` (framework ``ModelWeights`` in hand) or the
    standalone :func:`load_ppu_weights` (offline) to build the actual
    :class:`PPUWeightBundle`.
    """

    @classmethod
    def framework_base_available(cls) -> bool:
        return _FrameworkBase is not object

    def _require_framework(self):
        if _FrameworkBase is object:
            raise RuntimeError(
                "DeepSeekV41PPUWeight framework descriptors require the RTP "
                f"framework weight chain, which failed to import here: "
                f"{_FRAMEWORK_IMPORT_ERROR!r}. Use load_ppu_weights() for "
                "offline/diagnostic weight loading."
            )

    def _get_weight_info(self):  # pragma: no cover - framework hosts only
        self._require_framework()
        info = super()._get_weight_info()
        # C++ owns the final vocabulary projection and its physical TP split.
        # The PPU core loads its EP/ENP shards through load_ppu_weights, which
        # deliberately keeps Attention TP1 under P's physical CP/TP8 group.
        # Do not let the generic loader TP-split these private projections or
        # run NVIDIA FP4/MegaMoE preprocessing on a duplicate model.
        info.layer_weights = [[] for _ in range(self._num_layers)]
        return info

    @classmethod
    def from_model_weights(
        cls,
        weights,
        config: dict,
        *,
        ep_rank: int = 0,
        ep_size: int = 1,
        enp_size: Optional[int] = None,
        device=None,
        checkpoint_path: Optional[str] = None,
        token_map: Optional[Sequence[int]] = None,
    ) -> PPUWeightBundle:
        """Build the bundle from a populated framework ``ModelWeights``.

        ``weights`` follows the framework contract: ``weights.global_weights``
        keyed by the ``W`` enum, ``weights.weights[layer_id]`` per-layer dicts
        keyed by ``W`` (values and FP8 scales under separate ``W.v4_*_w`` /
        ``W.v4_*_s`` names). Engram tables are loaded from
        ``checkpoint_path`` (the framework loader deliberately skips them);
        ``token_map`` is the tokenizer-derived compressed-vocab map.
        """
        from rtp_llm.utils.model_weight import W

        geometry = V41Geometry.from_config(
            dict(config, **({"engram_token_map": list(token_map)} if token_map else {}))
        )
        tensors: Dict[str, torch.Tensor] = {}

        def put(name: str, tensor):
            if tensor is not None:
                tensors[name] = tensor

        global_weights = weights.global_weights
        put("embed.weight", global_weights.get(W.embedding))
        put("norm.weight", global_weights.get(W.final_ln_gamma))
        put("head.weight", global_weights.get(W.lm_head))
        for layer_id in range(geometry.num_layers):
            layer = weights.weights[layer_id]
            prefix = f"layers.{layer_id}"

            def dense(attr, name):
                weight = layer.get(getattr(W, attr))
                if weight is None:
                    raise KeyError(
                        f"framework weights lack {attr} for layer {layer_id}"
                    )
                scale = layer.get(getattr(W, attr[:-2] + "_s"))
                if weight.dtype in (torch.uint8, torch.int8) and scale is not None:
                    weight = weight.view(torch.float8_e4m3fn)
                if scale is not None:
                    tensors[f"{name}.weight"] = weight
                    tensors[f"{name}.scale"] = _as_scale(scale)
                else:
                    tensors[f"{name}.weight"] = weight

            dense("v4_attn_wq_a_w", f"{prefix}.attn.wq_a")
            dense("v4_attn_wq_b_w", f"{prefix}.attn.wq_b")
            dense("v4_attn_wkv_w", f"{prefix}.attn.wkv")
            dense("v4_attn_wo_a_w", f"{prefix}.attn.wo_a")
            dense("v4_attn_wo_b_w", f"{prefix}.attn.wo_b")
            for name, key in (
                (f"{prefix}.attn_norm.weight", W.v4_attn_norm),
                (f"{prefix}.ffn_norm.weight", W.v4_ffn_norm),
                (f"{prefix}.attn.q_norm.weight", W.v4_attn_q_norm),
                (f"{prefix}.attn.kv_norm.weight", W.v4_attn_kv_norm),
            ):
                put(name, layer.get(key))
            put(f"{prefix}.attn.attn_sink", layer.get(W.v4_attn_sink))
            if layer_id in geometry.kv_source_layer_ids:
                put(
                    f"{prefix}.attn.compressor.wkv.weight",
                    layer.get(W.v4_compressor_wkv),
                )
                put(
                    f"{prefix}.attn.compressor.norm.weight",
                    layer.get(W.v4_compressor_norm),
                )
                if geometry.ratio(layer_id) == 2:
                    put(
                        f"{prefix}.attn.compressor.wgate.weight",
                        layer.get(W.v4_compressor_wgate),
                    )
                put(f"{prefix}.attn.indexer.wk.weight", layer.get(W.v41_indexer_wk))
                put(
                    f"{prefix}.attn.indexer.k_norm.weight",
                    layer.get(W.v41_indexer_k_norm),
                )
            if layer_id in geometry.index_source_layer_ids:
                weight = layer.get(W.v4_indexer_wq_b_w)
                scale = layer.get(W.v4_indexer_wq_b_s)
                if weight is None or scale is None:
                    raise KeyError(
                        f"framework weights lack the indexer wq_b projection "
                        f"for layer {layer_id}"
                    )
                if weight.dtype in (torch.uint8, torch.int8):
                    weight = weight.view(torch.float8_e4m3fn)
                tensors[f"{prefix}.attn.indexer.wq_b.weight"] = weight
                tensors[f"{prefix}.attn.indexer.wq_b.scale"] = _as_scale(scale)
                put(
                    f"{prefix}.attn.indexer.weights_proj.weight",
                    layer.get(W.v4_indexer_weights_proj_w),
                )
            for which in ("attn", "ffn"):
                for sub in ("fn", "base", "scale"):
                    put(
                        f"{prefix}.hc_{which}_{sub}",
                        layer.get(getattr(W, f"v4_hc_{which}_{sub}")),
                    )
            put(f"{prefix}.ffn.gate.weight", layer.get(W.v4_router_w))
            put(f"{prefix}.ffn.gate.bias", layer.get(W.v4_router_bias))
            if geometry.n_shared_experts:
                for sub in ("w1", "w3", "w2"):
                    weight = layer.get(getattr(W, f"v4_shared_{sub}_w"))
                    scale = layer.get(getattr(W, f"v4_shared_{sub}_s"))
                    if weight is None:
                        raise KeyError(
                            f"framework weights lack shared expert {sub} for "
                            f"layer {layer_id}"
                        )
                    if weight.dtype in (torch.uint8, torch.int8):
                        if scale is None:
                            raise KeyError(
                                f"shared expert {sub} is FP8 but its scale is "
                                f"missing on layer {layer_id}"
                            )
                        weight = weight.view(torch.float8_e4m3fn)
                    tensors[f"{prefix}.ffn.shared_experts.{sub}.weight"] = weight
                    if scale is not None:
                        tensors[f"{prefix}.ffn.shared_experts.{sub}.scale"] = _as_scale(
                            scale
                        )
            # Routed experts: stacked [E, *, K/2] under the W names, sliced to
            # the local shard by _load_routed_experts through per-expert keys.
            local = geometry.num_local_experts(ep_size)
            for sub, w_name, s_name in (
                ("w1", W.v4_routed_w1_w, W.v4_routed_w1_s),
                ("w2", W.v4_routed_w2_w, W.v4_routed_w2_s),
                ("w3", W.v4_routed_w3_w, W.v4_routed_w3_s),
            ):
                stacked_w = layer.get(w_name)
                stacked_s = layer.get(s_name)
                if stacked_w is None or stacked_s is None:
                    raise KeyError(
                        f"framework weights lack routed {sub} for layer {layer_id}"
                    )
                start = ep_rank * local
                for expert in range(start, start + local):
                    row = expert - start
                    tensors[f"{prefix}.ffn.experts.{expert}.{sub}.weight"] = stacked_w[
                        expert
                    ]
                    tensors[f"{prefix}.ffn.experts.{expert}.{sub}.scale"] = stacked_s[
                        expert
                    ]
        if geometry.engram_enabled() and checkpoint_path is None:
            raise ValueError(
                "Engram layers require checkpoint_path (the framework loader "
                "does not materialize the host tables)"
            )
        source = _CheckpointSource(tensors)
        if geometry.engram_enabled():
            engram_source = _CheckpointSource(checkpoint_path)
            for layer_id in geometry.engram_layer_ids:
                prefix = f"layers.{layer_id}"
                for suffix in ("embed.weight", "embed.scale", "q_weight", "k_weight"):
                    name = f"{prefix}.engram.{suffix}"
                    tensor = engram_source.get(name)
                    if tensor is None:
                        raise KeyError(f"checkpoint is missing {name}")
                    tensors[name] = tensor
                weight = engram_source.get(f"{prefix}.engram.wkv.weight")
                scale = engram_source.get(f"{prefix}.engram.wkv.scale")
                if weight is None or scale is None:
                    raise KeyError(f"checkpoint is missing {prefix}.engram.wkv.*")
                tensors[f"{prefix}.engram.wkv.weight"] = weight
                tensors[f"{prefix}.engram.wkv.scale"] = scale
            source = _CheckpointSource(tensors)
        return load_ppu_weights(
            config,
            source,
            ep_rank=ep_rank,
            ep_size=ep_size,
            enp_size=enp_size,
            device=device,
        )
