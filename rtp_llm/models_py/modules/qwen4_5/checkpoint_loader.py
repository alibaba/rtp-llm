"""Zero-dependency HuggingFace 1109 Safetensors reader/writer & RTP ModelWeights binder.

Compatible with:
  - Production environments with ``fastsafetensors`` / ``safetensors`` + ``FastSafetensors`` loader
  - Pure-PyTorch containers (zero external pip dependencies via mmap + ``torch.frombuffer``)
  - Multi-GPU Expert Parallelism (EP) for routed MXFP4 experts and 128-shard FP8 PLE tables
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import mmap
import os
import struct
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist


from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig
from rtp_llm.models_py.modules.qwen4_5.moe import Qwen4_5DenseMLP, Qwen4_5LatentMoE
from rtp_llm.models_py.modules.qwen4_5.ple import DEFAULT_PLE_WEIGHT_SCALE
from rtp_llm.models_py.modules.qwen4_5.tokenizer import Qwen2BPETokenizer

_SAFETENSORS_TO_TORCH: Dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}

_TORCH_TO_SAFETENSORS: Dict[torch.dtype, str] = {
    v: k for k, v in _SAFETENSORS_TO_TORCH.items()
}


def arch_config_to_hf_dict(cfg: Qwen4_5ArchConfig) -> Dict[str, Any]:
    """Serialize ``Qwen4_5ArchConfig`` to HuggingFace 1109 ``config.json`` schema."""
    data: Dict[str, Any] = {
        "architectures": ["Qwen4_5ForCausalLM"],
        "model_type": "qwen4_5",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "intermediate_size": cfg.intermediate_size,
        "first_k_dense_replace": cfg.first_k_dense_replace,
        "rms_norm_eps": cfg.rms_norm_eps,
        "bos_token_id": cfg.bos_token_id,
        "eos_token_id": cfg.eos_token_id,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "max_position_embeddings": cfg.max_position_embeddings,
        "full_attention_interval": cfg.full_attention_interval,
        "layer_types": list(cfg.layer_types),
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "attention_bias": cfg.attention_bias,
        "attn_output_gate": cfg.attn_output_gate,
        "partial_rotary_factor": cfg.partial_rotary_factor,
        "rope_theta": cfg.rope_theta,
        "share_kv_projection_with_rope": cfg.share_kv_projection_with_rope,
        "use_full_attn_kv_sconv": cfg.use_full_attn_kv_sconv,
        "full_attn_kv_sconv_residual": cfg.full_attn_kv_sconv_residual,
        "full_attn_kv_sconv_use_activation": cfg.full_attn_kv_sconv_use_activation,
        "linear_conv_kernel_dim": cfg.linear_conv_kernel_dim,
        "linear_num_key_heads": cfg.linear_num_key_heads,
        "linear_key_head_dim": cfg.linear_key_head_dim,
        "linear_num_value_heads": cfg.linear_num_value_heads,
        "linear_value_head_dim": cfg.linear_value_head_dim,
        "gdn_conv_residual": cfg.gdn_conv_residual,
        "gdn_no_q_conv": cfg.gdn_no_q_conv,
        "norm_before_gate": cfg.norm_before_gate,
        "output_gate_type": cfg.output_gate_type,
        "use_linear_attn_output_sconv": cfg.use_linear_attn_output_sconv,
        "use_full_attn_output_sconv": cfg.use_full_attn_output_sconv,
        "use_ffn_output_sconv": cfg.use_ffn_output_sconv,
        "output_sconv_use_activation": cfg.output_sconv_use_activation,
        "hybrid_sharing_consumer_layer_no_output_sconv": cfg.hybrid_sharing_consumer_layer_no_output_sconv,
        "hc_count": cfg.hc_count,
        "hc_lowrank": cfg.hc_lowrank,
        "hc_prev_mix": cfg.hc_prev_mix,
        "hc_prev_combine": cfg.hc_prev_combine,
        "hc_per_branch_norm": cfg.hc_per_branch_norm,
        "hc_rmsnorm_bias": cfg.hc_rmsnorm_bias,
        "num_experts": cfg.num_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "routed_expert_hidden_size": cfg.routed_expert_hidden_size,
        "moe_intermediate_size": cfg.moe_intermediate_size,
        "shared_expert_intermediate_size": cfg.shared_expert_intermediate_size,
        "shared_expert_gate": cfg.shared_expert_gate,
        "router_precision": cfg.router_precision,
        "router_score_function": cfg.router_score_function,
        "moe_router_pre_softmax": cfg.moe_router_pre_softmax,
        "router_normalize_topk_scores": cfg.router_normalize_topk_scores,
        "moe_router_enable_expert_bias": cfg.moe_router_enable_expert_bias,
        "hidden_act": cfg.hidden_act,
        "hidden_act_gate_beta": cfg.hidden_act_gate_beta,
        "hidden_act_up_beta": cfg.hidden_act_up_beta,
        "clip_mlp_output": cfg.clip_mlp_output,
        "clip_attn_output": cfg.clip_attn_output,
        "ple_layer_ids": list(cfg.ple_layer_ids),
        "ple_embed_dim": cfg.ple_embed_dim,
        "ple_conv_kernel_size": cfg.ple_conv_kernel_size,
        "ngram_size": cfg.ngram_size,
        "heads_per_ngram": cfg.heads_per_ngram,
        "ngram_vocab_size_base": cfg.ngram_vocab_size_base,
        "make_ngram_vocab_size_divisible_by": cfg.make_ngram_vocab_size_divisible_by,
        "split_ngram_parts": cfg.split_ngram_parts,
        "ple_quant_scheme": cfg.ple_quant_scheme,
        "share_map": cfg.share_map,
        "enable_yoco_decode": cfg.enable_yoco_decode,
    }
    if cfg.has_qsa:
        data["indexer_head_dim"] = int(cfg.indexer_head_dim)
        data["indexer_n_heads"] = int(cfg.indexer_n_heads)
        data["indexer_kv_heads"] = int(cfg.indexer_kv_heads)
        data["indexer_compress_ratio"] = int(cfg.indexer_compress_ratio)
        data["indexer_budget"] = int(cfg.indexer_budget)
    if cfg.quant_method == "mxfp4":
        data["quantization_config"] = {
            "quant_method": "mxfp4",
            "modules_to_not_convert": ["mlp.gate", "mlp.shared_expert"],
        }
    return data


def _tensor_raw_bytes(t: torch.Tensor) -> bytes:
    """Extract raw contiguous CPU bytes from any PyTorch tensor including FP8/BF16."""
    cpu_t = t.detach().cpu().contiguous()
    if cpu_t.numel() == 0:
        return b""
    return cpu_t.view(torch.uint8).numpy().tobytes()


def write_safetensors_file(tensors: Mapping[str, torch.Tensor], path: Union[str, Path]) -> int:
    """Write tensors in standard HuggingFace ``.safetensors`` binary format."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header: Dict[str, Any] = {"__metadata__": {"format": "pt"}}
    raw_chunks: List[bytes] = []
    offset = 0
    for key in sorted(tensors.keys()):
        t = tensors[key].detach().cpu().contiguous()
        st_dtype = _TORCH_TO_SAFETENSORS.get(t.dtype)
        if st_dtype is None:
            raise ValueError(f"Unsupported tensor dtype {t.dtype} for {key!r}")
        raw = t.view(torch.uint8).reshape(-1).numpy().tobytes() if t.numel() > 0 else b""
        nbytes = len(raw)
        header[key] = {
            "dtype": st_dtype,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        raw_chunks.append(raw)
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    # Safetensors spec aligns header with trailing spaces to 8 bytes (optional, cleaner)
    pad = (-len(header_bytes)) % 8
    if pad:
        header_bytes += b" " * pad
    prefix = struct.pack("<Q", len(header_bytes))

    with out_path.open("wb") as f:
        f.write(prefix)
        f.write(header_bytes)
        for chunk in raw_chunks:
            if chunk:
                f.write(chunk)
    return len(prefix) + len(header_bytes) + offset


try:
    from safetensors import safe_open as _safetensors_safe_open
except Exception:
    _safetensors_safe_open = None


class SafetensorsShardReader:
    """On-demand Safetensors shard reader (uses fastsafetensors/safetensors if installed, else zero-dep mmap)."""

    def __init__(self, ckpt_dir: Union[str, Path], *, prefer_safetensors: bool = True) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.prefer_safetensors = bool(prefer_safetensors and _safetensors_safe_open is not None)
        index_path = self.ckpt_dir / "model.safetensors.index.json"
        if index_path.exists():
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map: Dict[str, str] = dict(index_data["weight_map"])
        else:
            shards = sorted(self.ckpt_dir.glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError(
                    f"No model.safetensors.index.json or *.safetensors found in {self.ckpt_dir}"
                )
            self.weight_map = {}
            for sp in shards:
                meta, _ = self._read_header(sp)
                for k in meta:
                    if k != "__metadata__":
                        self.weight_map[k] = sp.name

        self._handles: Dict[str, Tuple[Any, mmap.mmap, int, Dict[str, Any]]] = {}
        self._st_handles: Dict[str, Any] = {}
        self._header_cache: Dict[str, Tuple[Dict[str, Any], int]] = {}

    @staticmethod
    def _read_header(path: Path) -> Tuple[Dict[str, Any], int]:
        with path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_json = json.loads(f.read(header_len).decode("utf-8"))
            data_base = 8 + header_len
        return header_json, data_base

    def _get_header(self, shard_name: str) -> Tuple[Dict[str, Any], int]:
        cached = self._header_cache.get(shard_name)
        if cached is None:
            cached = self._read_header(self.ckpt_dir / shard_name)
            self._header_cache[shard_name] = cached
        return cached

    def resolve_key(self, *candidates: str) -> str:
        for k in candidates:
            if k in self.weight_map:
                return k
        raise KeyError(f"None of candidate keys {candidates!r} present in {self.ckpt_dir}")

    def get_tensor_spec(self, key: str) -> Tuple[torch.dtype, Tuple[int, ...]]:
        if key not in self.weight_map:
            raise KeyError(f"Tensor {key!r} not present in {self.ckpt_dir}")
        header, _ = self._get_header(self.weight_map[key])
        info = header[key]
        return _SAFETENSORS_TO_TORCH[info["dtype"]], tuple(int(d) for d in info["shape"])

    def _open_shard(self, shard_name: str) -> Tuple[Any, mmap.mmap, int, Dict[str, Any]]:
        if shard_name not in self._handles:
            sp = self.ckpt_dir / shard_name
            f = sp.open("rb")
            try:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except Exception:
                f.close()
                raise
            header_len = struct.unpack_from("<Q", mm, 0)[0]
            header = json.loads(mm[8 : 8 + header_len].decode("utf-8"))
            self._handles[shard_name] = (f, mm, 8 + header_len, header)
        return self._handles[shard_name]

    def close(self) -> None:
        for ctx in self._st_handles.values():
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        self._st_handles.clear()
        for f, mm, _, _ in self._handles.values():
            try:
                mm.close()
            finally:
                f.close()
        self._handles.clear()

    def __enter__(self) -> "SafetensorsShardReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def has_tensor(self, key: str) -> bool:
        return key in self.weight_map

    def get_tensor(self, key: str, *, device: Optional[torch.device] = None) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"Tensor {key!r} not present in {self.ckpt_dir}")
        shard_name = self.weight_map[key]
        if self.prefer_safetensors and _safetensors_safe_open is not None:
            if shard_name not in self._st_handles:
                dev_str = str(device) if device is not None else "cpu"
                ctx = _safetensors_safe_open(str(self.ckpt_dir / shard_name), framework="pt", device=dev_str)
                self._st_handles[shard_name] = ctx.__enter__()
            t = self._st_handles[shard_name].get_tensor(key)
            if device is not None and t.device != device:
                t = t.to(device)
            return t
        _, mm, data_base, header = self._open_shard(shard_name)
        info = header[key]
        dtype = _SAFETENSORS_TO_TORCH[info["dtype"]]
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]
        nbytes = end - start
        if nbytes == 0:
            t = torch.empty(shape, dtype=dtype, device="cpu")
        else:
            raw_bytes = bytearray(mm[data_base + start : data_base + end])
            t = torch.frombuffer(raw_bytes, dtype=torch.uint8).view(dtype).reshape(shape)
        if device is not None and device.type != "cpu":
            t = t.to(device)
        return t

    @staticmethod
    def _pread_exact(fd: int, nbytes: int, offset: int) -> bytearray:
        buf = bytearray(nbytes)
        pos = 0
        while pos < nbytes:
            chunk = os.pread(fd, nbytes - pos, offset + pos)
            if not chunk:
                raise EOFError(f"Unexpected EOF during pread at offset={offset + pos}, need={nbytes}")
            clen = len(chunk)
            buf[pos : pos + clen] = chunk
            pos += clen
        return buf

    def stream_copy_targets(
        self,
        tasks: Sequence[Tuple[str, torch.Tensor, Optional[torch.dtype]]],
        *,
        num_workers: int = 8,
        chunk_bytes: int = 64 * 1024 * 1024,
    ) -> int:
        """Load many checkpoint tensors in parallel, sorted by (shard, file offset).

        Copies each tensor slice directly into its pre-allocated target GPU/CPU tensor
        with bounded host memory (huge 2D tensors >= ``chunk_bytes`` stream in row chunks).
        Returns total bytes copied.
        """
        if not tasks:
            return 0

        shard_plans: Dict[str, List[Tuple[int, int, str, torch.dtype, Tuple[int, ...], torch.Tensor, Optional[torch.dtype]]]] = {}
        # Pre-populate header cache sequentially for all referenced shards (very fast, small JSON)
        seen_shards: set[str] = set()
        for key, dst, expected_dtype in tasks:
            if key not in self.weight_map:
                raise KeyError(f"Tensor {key!r} not present in {self.ckpt_dir}")
            sname = self.weight_map[key]
            seen_shards.add(sname)
            header, data_base = self._get_header(sname)
            info = header[key]
            start, end = info["data_offsets"]
            src_dt = _SAFETENSORS_TO_TORCH[info["dtype"]]
            shape = tuple(int(d) for d in info["shape"])
            shard_plans.setdefault(sname, []).append(
                (data_base + int(start), data_base + int(end), key, src_dt, shape, dst, expected_dtype)
            )

        for sname in shard_plans:
            shard_plans[sname].sort(key=lambda item: item[0])

        total_bytes_read = 0

        def _worker(shard_name: str, items: List[Tuple[int, int, str, torch.dtype, Tuple[int, ...], torch.Tensor, Optional[torch.dtype]]]) -> int:
            local_bytes = 0
            fd = os.open(str(self.ckpt_dir / shard_name), os.O_RDONLY)
            try:
                for abs_start, abs_end, key, src_dt, shape, dst, expected_dtype in items:
                    nbytes = abs_end - abs_start
                    local_bytes += nbytes
                    target_dt = expected_dtype if expected_dtype is not None else dst.dtype
                    if dst.device.type == "cuda":
                        torch.cuda.set_device(dst.device)
                    if nbytes == 0:
                        continue
                    if tuple(dst.shape) != shape:
                        raise ValueError(
                            f"Shape mismatch loading {key!r}: target={tuple(dst.shape)} vs ckpt={shape}"
                        )
                    # Stream very large row-major tensors in row slices to keep CPU RAM bounded
                    if nbytes > chunk_bytes and len(shape) >= 2 and math.prod(shape[1:]) > 0:
                        elem_bytes = max(1, src_dt.itemsize)
                        row_elems = math.prod(shape[1:])
                        row_bytes = row_elems * elem_bytes
                        rows_per_chunk = max(1, chunk_bytes // row_bytes)
                        total_rows = shape[0]
                        for r0 in range(0, total_rows, rows_per_chunk):
                            r1 = min(total_rows, r0 + rows_per_chunk)
                            c_off = abs_start + r0 * row_bytes
                            c_len = (r1 - r0) * row_bytes
                            raw = self._pread_exact(fd, c_len, c_off)
                            t_cpu = (
                                torch.frombuffer(raw, dtype=torch.uint8)
                                .view(src_dt)
                                .reshape((r1 - r0, *shape[1:]))
                            )
                            if t_cpu.dtype != target_dt:
                                t_cpu = t_cpu.to(dtype=target_dt)
                            dst.data[r0:r1].copy_(t_cpu, non_blocking=False)
                    else:
                        raw = self._pread_exact(fd, nbytes, abs_start)
                        t_cpu = (
                            torch.frombuffer(raw, dtype=torch.uint8)
                            .view(src_dt)
                            .reshape(shape)
                        )
                        if t_cpu.dtype != target_dt:
                            t_cpu = t_cpu.to(dtype=target_dt)
                        dst.data.copy_(t_cpu, non_blocking=False)
            finally:
                os.close(fd)
            return local_bytes

        prev_dev = torch.cuda.current_device() if torch.cuda.is_available() else None
        eff_workers = max(1, min(int(num_workers), len(shard_plans)))
        try:
            if eff_workers == 1:
                for sname, items in shard_plans.items():
                    total_bytes_read += _worker(sname, items)
            else:
                with ThreadPoolExecutor(max_workers=eff_workers) as pool:
                    futs = [pool.submit(_worker, sname, items) for sname, items in shard_plans.items()]
                    for fut in futs:
                        total_bytes_read += fut.result()
        finally:
            if prev_dev is not None:
                torch.cuda.set_device(prev_dev)
        return total_bytes_read


def _export_hc_tensors(
    hc: Any,
    hc_prefix: str,
    out: Dict[str, torch.Tensor],
    *,
    emits_latent: bool = True,
) -> None:
    out[f"{hc_prefix}hc_norm.weight"] = hc.hc_norm_weight
    out[f"{hc_prefix}hc_norm.bias"] = hc.hc_norm_bias
    if getattr(hc, "block_inject_weight", None) is not None:
        out[f"{hc_prefix}block_inject_weight.weight"] = hc.block_inject_weight
    if emits_latent and getattr(hc, "input_mix_weight_down", None) is not None:
        out[f"{hc_prefix}input_mix_weight_down.weight"] = hc.input_mix_weight_down
    out[f"{hc_prefix}input_mix_weight_up.weight"] = hc.input_mix_weight_up


@torch.no_grad()
def save_qwen4_5_safetensors_checkpoint(
    causal_lm: Any,
    out_dir: Union[str, Path],
    *,
    prefix: str = "model.",
) -> Dict[str, Any]:
    """Export an in-memory ``Qwen4_5ForCausalLM`` (or ``Qwen4_5Model``) to exact HF 1109 schema.

    Raises ``RuntimeError`` if ``validate_qwen4_5_weight_map`` reports any missing
    or unexpected tensor key.
    """
    backbone = causal_lm.model if hasattr(causal_lm, "model") else causal_lm
    lm_head = getattr(causal_lm, "lm_head", None)
    cfg: Qwen4_5ArchConfig = backbone.config
    if getattr(backbone, "ep_size", 1) != 1:
        raise ValueError("save_qwen4_5_safetensors_checkpoint expects an unsharded ep_size=1 model")

    tensors: Dict[str, torch.Tensor] = {}
    tensors[f"{prefix}embed_tokens.weight"] = backbone.embed_tokens.weight
    if not cfg.tie_word_embeddings:
        assert lm_head is not None, "lm_head is required when tie_word_embeddings=False"
        tensors["lm_head.weight"] = lm_head.weight

    _export_hc_tensors(
        backbone.hyper_connection_mixer,
        f"{prefix}hyper_connection_mixer.",
        tensors,
        emits_latent=True,
    )

    for layer in backbone.layers:
        lid = layer.layer_idx
        p = f"{prefix}layers.{lid}."
        _export_hc_tensors(layer.attn_hyper_connection, f"{p}attn_hyper_connection.", tensors)
        _export_hc_tensors(
            layer.mlp_hyper_connection,
            f"{p}mlp_hyper_connection.",
            tensors,
            emits_latent=(lid < cfg.num_hidden_layers - 1),
        )

        # PLE (Layer 1)
        if layer.ple is not None:
            ple = layer.ple
            ple_p = f"{p}ple."
            assert ple._ngram_shards is not None, "PLE shards must be initialized before export"
            for s_idx, shard in enumerate(ple._ngram_shards):
                tensors[f"{ple_p}ple_embedding.ngram_embedding.shard_{s_idx}.weight"] = shard
            if cfg.ple_quant_scheme == "fp8_pertensor":
                tensors[f"{ple_p}ple_embedding.ngram_embedding.weight_scale"] = (
                    ple.weight_scale.reshape(1).to(torch.bfloat16)
                )
            tensors[f"{ple_p}ple_embedding.layer_multipliers"] = ple.layer_multipliers.to(torch.int64)
            tensors[f"{ple_p}ple_embedding.ngram_heads_offsets"] = ple.ngram_heads_offsets.to(torch.int64)
            tensors[f"{ple_p}ple_embedding.ngram_heads_vocab_sizes"] = ple.ngram_heads_vocab_sizes.to(
                torch.int64
            )
            tensors[f"{ple_p}key_proj.weight"] = ple.key_proj.weight
            tensors[f"{ple_p}value_proj.weight"] = ple.value_proj.weight
            tensors[f"{ple_p}conv1d.weight"] = ple.conv1d.weight
            tensors[f"{ple_p}norm_key.weight"] = ple.norm_key
            tensors[f"{ple_p}norm_query.weight"] = ple.norm_query
            tensors[f"{ple_p}norm_conv.weight"] = ple.norm_conv

        # Attention
        if layer.linear_attn is not None:
            la_mod = layer.linear_attn
            la = f"{p}linear_attn."
            tensors[f"{la}in_proj_qkvz.weight"] = la_mod.in_proj_qkvz.weight
            if not layer.is_consumer:
                assert la_mod.in_proj_ba is not None
                tensors[f"{la}in_proj_ba.weight"] = la_mod.in_proj_ba.weight
                tensors[f"{la}A_log"] = la_mod.A_log.to(torch.bfloat16)
                tensors[f"{la}dt_bias"] = la_mod.dt_bias
            if la_mod.conv1d is not None:
                tensors[f"{la}conv1d.weight"] = la_mod.conv1d.weight
            tensors[f"{la}norm.weight"] = la_mod.norm.weight
            tensors[f"{la}out_proj.weight"] = la_mod.out_proj.weight
            if layer.attn_output_sconv is not None:
                tensors[f"{p}linear_attn_output_sconv.weight"] = layer.attn_output_sconv.weight
        else:
            sa_mod = layer.self_attn
            assert sa_mod is not None
            sa = f"{p}self_attn."
            tensors[f"{sa}q_proj.weight"] = sa_mod.q_proj.weight
            tensors[f"{sa}q_norm.weight"] = sa_mod.q_norm.weight
            tensors[f"{sa}o_proj.weight"] = sa_mod.o_proj.weight
            if not layer.is_consumer:
                assert sa_mod.k_proj is not None and sa_mod.v_proj is not None and sa_mod.k_norm is not None
                tensors[f"{sa}k_proj.weight"] = sa_mod.k_proj.weight
                tensors[f"{sa}v_proj.weight"] = sa_mod.v_proj.weight
                tensors[f"{sa}k_norm.weight"] = sa_mod.k_norm.weight
                if sa_mod.k_sconv is not None:
                    tensors[f"{sa}k_sconv.weight"] = sa_mod.k_sconv.weight
                if sa_mod.v_sconv is not None:
                    tensors[f"{sa}v_sconv.weight"] = sa_mod.v_sconv.weight
            if layer.attn_output_sconv is not None:
                tensors[f"{p}full_attn_output_sconv.weight"] = layer.attn_output_sconv.weight
            if sa_mod.indexer is not None:
                idx = sa_mod.indexer
                tensors[f"{sa}indexer.index_qk_proj.weight"] = idx.index_qk_proj.weight
                tensors[f"{sa}indexer.q_layernorm.weight"] = idx.q_layernorm.weight
                tensors[f"{sa}indexer.k_layernorm.weight"] = idx.k_layernorm.weight

        # FFN
        if isinstance(layer.mlp, Qwen4_5DenseMLP):
            mlp = layer.mlp
            tensors[f"{p}mlp.gate_proj.weight"] = mlp.gate_proj.weight
            tensors[f"{p}mlp.up_proj.weight"] = mlp.up_proj.weight
            tensors[f"{p}mlp.down_proj.weight"] = mlp.down_proj.weight
        else:
            assert isinstance(layer.mlp, Qwen4_5LatentMoE)
            moe = layer.mlp
            tensors[f"{p}mlp.latent_down_proj"] = moe.latent_down_proj.weight
            tensors[f"{p}mlp.latent_gate_proj"] = moe.latent_gate_proj.weight
            tensors[f"{p}mlp.latent_up_proj"] = moe.latent_up_proj.weight
            tensors[f"{p}mlp.latent_scale_gate"] = moe.latent_scale_gate.weight
            tensors[f"{p}mlp.pre_moe_latent_layernorm.weight"] = moe.pre_moe_latent_layernorm.weight
            tensors[f"{p}mlp.gate.weight"] = moe.gate.weight
            if moe.expert_bias is not None:
                tensors[f"{p}mlp.expert_bias"] = moe.expert_bias.to(torch.float32)
            tensors[f"{p}mlp.shared_expert.gate_proj.weight"] = moe.shared_expert.gate_proj.weight
            tensors[f"{p}mlp.shared_expert.up_proj.weight"] = moe.shared_expert.up_proj.weight
            tensors[f"{p}mlp.shared_expert.down_proj.weight"] = moe.shared_expert.down_proj.weight
            if moe.shared_expert_gate is not None:
                tensors[f"{p}mlp.shared_expert_gate.weight"] = moe.shared_expert_gate.weight

            assert (
                moe.w13_u8 is not None
                and moe.w13_scale is not None
                and moe.w2_u8 is not None
                and moe.w2_scale is not None
            )
            w13_nk = moe.w13_u8.transpose(1, 2) if moe.weights_transposed else moe.w13_u8
            w2_nk = moe.w2_u8.transpose(1, 2) if moe.weights_transposed else moe.w2_u8
            m_i = cfg.moe_intermediate_size
            for e in range(cfg.num_experts):
                ep = f"{p}mlp.experts.{e}."
                tensors[f"{ep}gate_proj.weight"] = w13_nk[e, :m_i]
                tensors[f"{ep}gate_proj.weight_scale"] = moe.w13_scale[e, :m_i]
                tensors[f"{ep}up_proj.weight"] = w13_nk[e, m_i:]
                tensors[f"{ep}up_proj.weight_scale"] = moe.w13_scale[e, m_i:]
                tensors[f"{ep}down_proj.weight"] = w2_nk[e]
                tensors[f"{ep}down_proj.weight_scale"] = moe.w2_scale[e]

        if layer.ffn_output_sconv is not None:
            tensors[f"{p}ffn_output_sconv.weight"] = layer.ffn_output_sconv.weight

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "config.json").write_text(
        json.dumps(arch_config_to_hf_dict(cfg), indent=2), encoding="utf-8"
    )
    shard_name = "model-00001-of-00001.safetensors"
    total_file_bytes = write_safetensors_file(tensors, out_path / shard_name)
    weight_map = {k: shard_name for k in sorted(tensors.keys())}
    index_obj = {
        "metadata": {
            "total_size": sum(int(t.numel() * t.element_size()) for t in tensors.values())
        },
        "weight_map": weight_map,
    }
    (out_path / "model.safetensors.index.json").write_text(
        json.dumps(index_obj, indent=2), encoding="utf-8"
    )

    from rtp_llm.models.qwen4_5.qwen4_5_weight import validate_qwen4_5_weight_map

    specs_map = {
        k: (_TORCH_TO_SAFETENSORS[t.dtype], tuple(t.shape))
        for k, t in tensors.items()
    }
    validation = validate_qwen4_5_weight_map(cfg, specs_map, prefix=prefix)
    return {
        "missing": [],
        "unexpected": [],
        "num_expected": validation["expected_count"],
        "num_matched": validation["matched_count"],
        "extra_ignored_count": validation["extra_ignored_count"],
        "total_file_bytes": total_file_bytes,
    }


def _copy_param(dst: torch.Tensor, src: torch.Tensor) -> None:
    dst.data.copy_(src.to(device=dst.device, dtype=dst.dtype))


def _enqueue_hc_copy_ops(
    hc: Any,
    reader: SafetensorsShardReader,
    hc_prefix: str,
    ops: List[Tuple[str, torch.Tensor, Optional[torch.dtype]]],
    *,
    emits_latent: bool = True,
) -> None:
    ops.append((f"{hc_prefix}hc_norm.weight", hc.hc_norm_weight, None))
    ops.append((f"{hc_prefix}hc_norm.bias", hc.hc_norm_bias, None))
    if getattr(hc, "block_inject_weight", None) is not None:
        ops.append((f"{hc_prefix}block_inject_weight.weight", hc.block_inject_weight, None))
    if emits_latent and getattr(hc, "input_mix_weight_down", None) is not None:
        ops.append((f"{hc_prefix}input_mix_weight_down.weight", hc.input_mix_weight_down, None))
    ops.append((f"{hc_prefix}input_mix_weight_up.weight", hc.input_mix_weight_up, None))


@torch.no_grad()
def load_qwen4_5_from_checkpoint(
    causal_lm: Any,
    ckpt_path: Union[str, Path],
    *,
    prefix: str = "model.",
    ple_mode: str = "gpu_sharded",
    num_workers: int = 8,
    verbose: bool = False,
) -> None:
    """Load HF 1109 checkpoint (single-GPU or EP-sharded) via fast parallel streaming."""
    t0 = time.perf_counter()
    backbone = causal_lm.model if hasattr(causal_lm, "model") else causal_lm
    lm_head = getattr(causal_lm, "lm_head", None)
    cfg: Qwen4_5ArchConfig = backbone.config
    ep_size = getattr(backbone, "ep_size", 1)
    ep_rank = getattr(backbone, "ep_rank", 0)
    ep_group = getattr(backbone, "ep_group", None)

    device = backbone.embed_tokens.weight.device
    ops: List[Tuple[str, torch.Tensor, Optional[torch.dtype]]] = []
    ple_plans: List[Tuple[Any, List[torch.Tensor], Optional[torch.Tensor], int, int]] = []

    with SafetensorsShardReader(ckpt_path) as reader:
        ops.append((f"{prefix}embed_tokens.weight", backbone.embed_tokens.weight, None))
        if lm_head is not None and not cfg.tie_word_embeddings:
            ops.append(("lm_head.weight", lm_head.weight, None))
        elif lm_head is not None and cfg.tie_word_embeddings:
            lm_head.weight = backbone.embed_tokens.weight

        _enqueue_hc_copy_ops(
            backbone.hyper_connection_mixer,
            reader,
            f"{prefix}hyper_connection_mixer.",
            ops,
            emits_latent=True,
        )

        for layer in backbone.layers:
            lid = layer.layer_idx
            p = f"{prefix}layers.{lid}."
            _enqueue_hc_copy_ops(
                layer.attn_hyper_connection, reader, f"{p}attn_hyper_connection.", ops
            )
            _enqueue_hc_copy_ops(
                layer.mlp_hyper_connection,
                reader,
                f"{p}mlp_hyper_connection.",
                ops,
                emits_latent=(lid < cfg.num_hidden_layers - 1),
            )

            # PLE (Layer 1)
            if layer.ple is not None:
                ple = layer.ple
                ple_p = f"{p}ple."
                ops.append((f"{ple_p}key_proj.weight", ple.key_proj.weight, None))
                ops.append((f"{ple_p}value_proj.weight", ple.value_proj.weight, None))
                ops.append((f"{ple_p}conv1d.weight", ple.conv1d.weight, None))
                ops.append((f"{ple_p}norm_key.weight", ple.norm_key, None))
                ops.append((f"{ple_p}norm_query.weight", ple.norm_query, None))
                ops.append((f"{ple_p}norm_conv.weight", ple.norm_conv, None))
                ops.append((f"{ple_p}ple_embedding.layer_multipliers", ple.layer_multipliers, torch.int64))
                ops.append((f"{ple_p}ple_embedding.ngram_heads_offsets", ple.ngram_heads_offsets, torch.int64))
                ops.append((f"{ple_p}ple_embedding.ngram_heads_vocab_sizes", ple.ngram_heads_vocab_sizes, torch.int64))

                scale_key = f"{ple_p}ple_embedding.ngram_embedding.weight_scale"
                w_scale_tensor: Optional[torch.Tensor] = None
                if reader.has_tensor(scale_key):
                    w_scale_tensor = torch.empty((1,), dtype=torch.bfloat16, device=device)
                    ops.append((scale_key, w_scale_tensor, torch.bfloat16))

                total_shards = cfg.split_ngram_parts
                if ple_mode == "gpu_sharded" and ep_size > 1:
                    local_shards = total_shards // ep_size
                    shard_offset = ep_rank * local_shards
                    shard_dev = device
                elif ple_mode == "cpu_uva":
                    local_shards = total_shards
                    shard_offset = 0
                    shard_dev = torch.device("cpu")
                else:
                    local_shards = total_shards
                    shard_offset = 0
                    shard_dev = device

                shard_bufs: List[torch.Tensor] = []
                for s in range(shard_offset, shard_offset + local_shards):
                    skey = f"{ple_p}ple_embedding.ngram_embedding.shard_{s}.weight"
                    s_dt, s_sh = reader.get_tensor_spec(skey)
                    sbuf = torch.empty(s_sh, dtype=s_dt, device=shard_dev)
                    shard_bufs.append(sbuf)
                    ops.append((skey, sbuf, s_dt))
                ple_plans.append((ple, shard_bufs, w_scale_tensor, shard_offset, total_shards))

            # Attention
            if layer.linear_attn is not None:
                la_mod = layer.linear_attn
                la = f"{p}linear_attn."
                ops.append((f"{la}in_proj_qkvz.weight", la_mod.in_proj_qkvz.weight, None))
                if not layer.is_consumer:
                    assert la_mod.in_proj_ba is not None
                    ops.append((f"{la}in_proj_ba.weight", la_mod.in_proj_ba.weight, None))
                    ops.append((f"{la}A_log", la_mod.A_log, torch.float32))
                    ops.append((f"{la}dt_bias", la_mod.dt_bias, None))
                if la_mod.conv1d is not None:
                    ops.append((f"{la}conv1d.weight", la_mod.conv1d.weight, None))
                ops.append((f"{la}norm.weight", la_mod.norm.weight, None))
                ops.append((f"{la}out_proj.weight", la_mod.out_proj.weight, None))
                if layer.attn_output_sconv is not None:
                    ops.append((f"{p}linear_attn_output_sconv.weight", layer.attn_output_sconv.weight, None))
            else:
                sa_mod = layer.self_attn
                assert sa_mod is not None
                sa = f"{p}self_attn."
                ops.append((f"{sa}q_proj.weight", sa_mod.q_proj.weight, None))
                ops.append((f"{sa}q_norm.weight", sa_mod.q_norm.weight, None))
                ops.append((f"{sa}o_proj.weight", sa_mod.o_proj.weight, None))
                if not layer.is_consumer:
                    assert sa_mod.k_proj is not None and sa_mod.v_proj is not None and sa_mod.k_norm is not None
                    ops.append((f"{sa}k_proj.weight", sa_mod.k_proj.weight, None))
                    ops.append((f"{sa}v_proj.weight", sa_mod.v_proj.weight, None))
                    ops.append((f"{sa}k_norm.weight", sa_mod.k_norm.weight, None))
                    if sa_mod.k_sconv is not None:
                        ops.append((f"{sa}k_sconv.weight", sa_mod.k_sconv.weight, None))
                    if sa_mod.v_sconv is not None:
                        ops.append((f"{sa}v_sconv.weight", sa_mod.v_sconv.weight, None))
                if layer.attn_output_sconv is not None:
                    ops.append((f"{p}full_attn_output_sconv.weight", layer.attn_output_sconv.weight, None))
                if sa_mod.indexer is not None:
                    idx = sa_mod.indexer
                    qsa_specs = (
                        (f"{sa}indexer.index_qk_proj.weight", idx.index_qk_proj.weight),
                        (f"{sa}indexer.q_layernorm.weight", idx.q_layernorm.weight),
                        (f"{sa}indexer.k_layernorm.weight", idx.k_layernorm.weight),
                    )
                    for qsa_k, qsa_param in qsa_specs:
                        s_dt, s_sh = reader.get_tensor_spec(qsa_k)
                        if s_dt not in (torch.bfloat16, qsa_param.dtype) or tuple(s_sh) != tuple(qsa_param.shape):
                            raise ValueError(
                                f"QSA tensor {qsa_k!r} spec mismatch: got dtype={s_dt}, shape={tuple(s_sh)}; "
                                f"expected dtype=torch.bfloat16, shape={tuple(qsa_param.shape)}"
                            )
                        ops.append((qsa_k, qsa_param, None))

            # FFN
            if isinstance(layer.mlp, Qwen4_5DenseMLP):
                mlp = layer.mlp
                ops.append((f"{p}mlp.gate_proj.weight", mlp.gate_proj.weight, None))
                ops.append((f"{p}mlp.up_proj.weight", mlp.up_proj.weight, None))
                ops.append((f"{p}mlp.down_proj.weight", mlp.down_proj.weight, None))
            else:
                assert isinstance(layer.mlp, Qwen4_5LatentMoE)
                moe = layer.mlp
                ops.append((reader.resolve_key(f"{p}mlp.latent_down_proj", f"{p}mlp.latent_down_proj.weight"), moe.latent_down_proj.weight, None))
                ops.append((reader.resolve_key(f"{p}mlp.latent_gate_proj", f"{p}mlp.latent_gate_proj.weight"), moe.latent_gate_proj.weight, None))
                ops.append((reader.resolve_key(f"{p}mlp.latent_up_proj", f"{p}mlp.latent_up_proj.weight"), moe.latent_up_proj.weight, None))
                ops.append((reader.resolve_key(f"{p}mlp.latent_scale_gate", f"{p}mlp.latent_scale_gate.weight"), moe.latent_scale_gate.weight, None))
                ops.append((f"{p}mlp.pre_moe_latent_layernorm.weight", moe.pre_moe_latent_layernorm.weight, None))
                ops.append((f"{p}mlp.gate.weight", moe.gate.weight, None))
                if moe.expert_bias is not None:
                    ops.append((f"{p}mlp.expert_bias", moe.expert_bias, torch.float32))
                ops.append((f"{p}mlp.shared_expert.gate_proj.weight", moe.shared_expert.gate_proj.weight, None))
                ops.append((f"{p}mlp.shared_expert.up_proj.weight", moe.shared_expert.up_proj.weight, None))
                ops.append((f"{p}mlp.shared_expert.down_proj.weight", moe.shared_expert.down_proj.weight, None))
                if moe.shared_expert_gate is not None:
                    ops.append((f"{p}mlp.shared_expert_gate.weight", moe.shared_expert_gate.weight, None))

                # Ensure local MXFP4 buffers are allocated in checkpoint [E_local, N, K//2] layout
                m_i = moe.moe_intermediate_size
                r_h = moe.latent_dim
                if moe.w13_u8 is None or moe.w13_scale is None or moe.w2_u8 is None or moe.w2_scale is None or moe.weights_transposed:
                    moe.register_buffer("w13_u8", torch.empty((moe.local_num_experts, 2 * m_i, r_h // 2), dtype=torch.uint8, device=device))
                    moe.register_buffer("w13_scale", torch.empty((moe.local_num_experts, 2 * m_i, r_h // 32), dtype=torch.uint8, device=device))
                    moe.register_buffer("w2_u8", torch.empty((moe.local_num_experts, r_h, m_i // 2), dtype=torch.uint8, device=device))
                    moe.register_buffer("w2_scale", torch.empty((moe.local_num_experts, r_h, m_i // 32), dtype=torch.uint8, device=device))
                    moe.weights_transposed = False

                assert moe.w13_u8 is not None and moe.w13_scale is not None
                assert moe.w2_u8 is not None and moe.w2_scale is not None
                for local_idx, e in enumerate(range(moe.ep_start, moe.ep_end)):
                    ep = f"{p}mlp.experts.{e}."
                    ops.append((f"{ep}gate_proj.weight", moe.w13_u8[local_idx, :m_i], torch.uint8))
                    ops.append((f"{ep}gate_proj.weight_scale", moe.w13_scale[local_idx, :m_i], torch.uint8))
                    ops.append((f"{ep}up_proj.weight", moe.w13_u8[local_idx, m_i:], torch.uint8))
                    ops.append((f"{ep}up_proj.weight_scale", moe.w13_scale[local_idx, m_i:], torch.uint8))
                    ops.append((f"{ep}down_proj.weight", moe.w2_u8[local_idx], torch.uint8))
                    ops.append((f"{ep}down_proj.weight_scale", moe.w2_scale[local_idx], torch.uint8))

            if layer.ffn_output_sconv is not None:
                ops.append((f"{p}ffn_output_sconv.weight", layer.ffn_output_sconv.weight, None))

        total_bytes = reader.stream_copy_targets(ops, num_workers=num_workers)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for ple, shard_bufs, w_scale_tensor, shard_offset, total_shards in ple_plans:
            w_scale = w_scale_tensor if w_scale_tensor is not None else DEFAULT_PLE_WEIGHT_SCALE
            if ple_mode == "gpu_sharded" and ep_size > 1:
                ple.set_ngram_table(
                    shard_bufs,
                    weight_scale=w_scale,
                    shard_offset=shard_offset,
                    total_shards=total_shards,
                    ep_group=ep_group,
                )
            elif ple_mode == "cpu_uva":
                ple.set_ngram_table(
                    shard_bufs,
                    weight_scale=w_scale,
                    pin_cpu=True,
                    enable_cpu_uva=True,
                )
            else:
                ple.set_ngram_table(shard_bufs, weight_scale=w_scale)

    causal_lm.prepare_weights()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if verbose:
        dt = time.perf_counter() - t0
        gib = total_bytes / (1024 ** 3)
        print(
            f"[EP rank {ep_rank}/{ep_size}] Loaded {len(ops)} tensors ({gib:.2f} GiB) from {ckpt_path} in {dt:.2f}s ({gib / max(dt, 1e-6):.2f} GiB/s)",
            flush=True,
        )


@torch.no_grad()
def bind_qwen4_5_model_weights(
    backbone: Any,
    model_weights: Any,
    arch_cfg: Qwen4_5ArchConfig,
    *,
    lm_head: Optional[Any] = None,
    ep_size: int = 1,
    ep_rank: int = 0,
    ep_group: Any = None,
) -> None:
    """Bind RTP-LLM ``ModelWeights`` (keyed by ``QWEN4_5_W.*``) onto ``Qwen4_5Model``.

    Enables seamless startup through RTP-LLM native FastSafetensors ``ModelLoader``.
    """
    device = backbone.embed_tokens.weight.device
    _copy_param(backbone.embed_tokens.weight, model_weights.get_global_weight(QWEN4_5_W.embedding))
    if lm_head is not None:
        if arch_cfg.tie_word_embeddings:
            lm_head.weight = backbone.embed_tokens.weight
        else:
            _copy_param(lm_head.weight, model_weights.get_global_weight(QWEN4_5_W.lm_head))

    mixer = backbone.hyper_connection_mixer
    _copy_param(mixer.hc_norm_weight, model_weights.get_global_weight(QWEN4_5_W.hc_mixer_norm_w))
    _copy_param(mixer.hc_norm_bias, model_weights.get_global_weight(QWEN4_5_W.hc_mixer_norm_b))
    _copy_param(mixer.input_mix_weight_down, model_weights.get_global_weight(QWEN4_5_W.hc_mixer_mix_down))
    _copy_param(mixer.input_mix_weight_up, model_weights.get_global_weight(QWEN4_5_W.hc_mixer_mix_up))

    for layer in backbone.layers:
        lid = layer.layer_idx
        lw = model_weights.weights[lid]

        # HC blocks
        _copy_param(layer.attn_hyper_connection.hc_norm_weight, lw[QWEN4_5_W.hc_attn_norm_w])
        _copy_param(layer.attn_hyper_connection.hc_norm_bias, lw[QWEN4_5_W.hc_attn_norm_b])
        _copy_param(layer.attn_hyper_connection.block_inject_weight, lw[QWEN4_5_W.hc_attn_inject])
        _copy_param(layer.attn_hyper_connection.input_mix_weight_down, lw[QWEN4_5_W.hc_attn_mix_down])
        _copy_param(layer.attn_hyper_connection.input_mix_weight_up, lw[QWEN4_5_W.hc_attn_mix_up])

        _copy_param(layer.mlp_hyper_connection.hc_norm_weight, lw[QWEN4_5_W.hc_mlp_norm_w])
        _copy_param(layer.mlp_hyper_connection.hc_norm_bias, lw[QWEN4_5_W.hc_mlp_norm_b])
        _copy_param(layer.mlp_hyper_connection.block_inject_weight, lw[QWEN4_5_W.hc_mlp_inject])
        if layer.mlp_hyper_connection.input_mix_weight_down is not None:
            _copy_param(layer.mlp_hyper_connection.input_mix_weight_down, lw[QWEN4_5_W.hc_mlp_mix_down])
        _copy_param(layer.mlp_hyper_connection.input_mix_weight_up, lw[QWEN4_5_W.hc_mlp_mix_up])

        # PLE
        if layer.ple is not None:
            ple = layer.ple
            _copy_param(ple.key_proj.weight, lw[QWEN4_5_W.ple_key_proj_w])
            _copy_param(ple.value_proj.weight, lw[QWEN4_5_W.ple_value_proj_w])
            _copy_param(ple.conv1d.weight, lw[QWEN4_5_W.ple_conv_w])
            _copy_param(ple.norm_key, lw[QWEN4_5_W.ple_norm_key_w])
            _copy_param(ple.norm_query, lw[QWEN4_5_W.ple_norm_query_w])
            _copy_param(ple.norm_conv, lw[QWEN4_5_W.ple_norm_conv_w])
            ple.layer_multipliers.copy_(lw[QWEN4_5_W.ple_multipliers].to(device=device, dtype=torch.int64))
            ple.ngram_heads_offsets.copy_(lw[QWEN4_5_W.ple_ngram_offsets].to(device=device, dtype=torch.int64))
            ple.ngram_heads_vocab_sizes.copy_(
                lw[QWEN4_5_W.ple_ngram_vocab_sizes].to(device=device, dtype=torch.int64)
            )
            shards_raw = lw[QWEN4_5_W.ple_ngram_shards]
            w_scale = lw.get(QWEN4_5_W.ple_ngram_scale, DEFAULT_PLE_WEIGHT_SCALE)
            if isinstance(shards_raw, (list, tuple)):
                total_shards = arch_cfg.split_ngram_parts
                local_shards = len(shards_raw)
                shard_offset = ep_rank * local_shards if local_shards < total_shards else 0
                ple.set_ngram_table(
                    list(shards_raw),
                    weight_scale=w_scale,
                    shard_offset=shard_offset,
                    total_shards=total_shards,
                    ep_group=ep_group if local_shards < total_shards else None,
                    enable_cpu_uva=True,
                )
            else:
                ple.set_ngram_table(shards_raw, weight_scale=w_scale, enable_cpu_uva=True)

        # Attention
        if layer.linear_attn is not None:
            la = layer.linear_attn
            _copy_param(la.in_proj_qkvz.weight, lw[QWEN4_5_W.linear_in_proj_qkvz])
            if not layer.is_consumer:
                assert la.in_proj_ba is not None
                _copy_param(la.in_proj_ba.weight, lw[QWEN4_5_W.linear_in_proj_ba])
                la.A_log.data.copy_(lw[QWEN4_5_W.linear_a_log].to(device=device, dtype=torch.float32))
                _copy_param(la.dt_bias, lw[QWEN4_5_W.linear_dt_bias])
            if la.conv1d is not None:
                _copy_param(la.conv1d.weight, lw[QWEN4_5_W.linear_conv1d])
            _copy_param(la.norm.weight, lw[QWEN4_5_W.linear_norm_w])
            _copy_param(la.out_proj.weight, lw[QWEN4_5_W.linear_out_proj])
            if layer.attn_output_sconv is not None:
                _copy_param(layer.attn_output_sconv.weight, lw[QWEN4_5_W.linear_attn_output_sconv])
        else:
            sa = layer.self_attn
            assert sa is not None
            _copy_param(sa.q_proj.weight, lw[QWEN4_5_W.full_q_proj])
            _copy_param(sa.q_norm.weight, lw[QWEN4_5_W.full_q_norm_w])
            _copy_param(sa.o_proj.weight, lw[QWEN4_5_W.full_o_proj])
            if not layer.is_consumer:
                assert sa.k_proj is not None and sa.v_proj is not None and sa.k_norm is not None
                _copy_param(sa.k_proj.weight, lw[QWEN4_5_W.full_k_proj])
                _copy_param(sa.v_proj.weight, lw[QWEN4_5_W.full_v_proj])
                _copy_param(sa.k_norm.weight, lw[QWEN4_5_W.full_k_norm_w])
                if sa.k_sconv is not None:
                    _copy_param(sa.k_sconv.weight, lw[QWEN4_5_W.full_k_sconv])
                if sa.v_sconv is not None:
                    _copy_param(sa.v_sconv.weight, lw[QWEN4_5_W.full_v_sconv])
            if layer.attn_output_sconv is not None:
                _copy_param(layer.attn_output_sconv.weight, lw[QWEN4_5_W.full_attn_output_sconv])
            if sa.indexer is not None:
                idx = sa.indexer
                qsa_bind = (
                    (QWEN4_5_W.full_indexer_qk_proj, idx.index_qk_proj.weight),
                    (QWEN4_5_W.full_indexer_q_norm_w, idx.q_layernorm.weight),
                    (QWEN4_5_W.full_indexer_k_norm_w, idx.k_layernorm.weight),
                )
                for w_name, qsa_param in qsa_bind:
                    w_src = lw[w_name]
                    if w_src.dtype != torch.bfloat16 or tuple(w_src.shape) != tuple(qsa_param.shape):
                        raise ValueError(
                            f"QSA ModelWeights {w_name!r} (layer {lid}) mismatch: "
                            f"got dtype={w_src.dtype}, shape={tuple(w_src.shape)}; "
                            f"expected dtype=torch.bfloat16, shape={tuple(qsa_param.shape)}"
                        )
                    _copy_param(qsa_param, w_src)

        # FFN
        if isinstance(layer.mlp, Qwen4_5DenseMLP):
            _copy_param(layer.mlp.gate_proj.weight, lw[QWEN4_5_W.dense_gate_proj])
            _copy_param(layer.mlp.up_proj.weight, lw[QWEN4_5_W.dense_up_proj])
            _copy_param(layer.mlp.down_proj.weight, lw[QWEN4_5_W.dense_down_proj])
        else:
            assert isinstance(layer.mlp, Qwen4_5LatentMoE)
            moe = layer.mlp
            _copy_param(moe.latent_down_proj.weight, lw[QWEN4_5_W.moe_latent_down_proj])
            _copy_param(moe.latent_gate_proj.weight, lw[QWEN4_5_W.moe_latent_gate_proj])
            _copy_param(moe.latent_up_proj.weight, lw[QWEN4_5_W.moe_latent_up_proj])
            _copy_param(moe.latent_scale_gate.weight, lw[QWEN4_5_W.moe_latent_scale_gate])
            _copy_param(moe.pre_moe_latent_layernorm.weight, lw[QWEN4_5_W.moe_pre_latent_ln_w])
            _copy_param(moe.gate.weight, lw[QWEN4_5_W.moe_router_gate])
            if moe.expert_bias is not None:
                moe.expert_bias.data.copy_(lw[QWEN4_5_W.moe_expert_bias].to(device=device, dtype=torch.float32))
            _copy_param(moe.shared_expert.gate_proj.weight, lw[QWEN4_5_W.moe_shared_gate_proj])
            _copy_param(moe.shared_expert.up_proj.weight, lw[QWEN4_5_W.moe_shared_up_proj])
            _copy_param(moe.shared_expert.down_proj.weight, lw[QWEN4_5_W.moe_shared_down_proj])
            if moe.shared_expert_gate is not None:
                _copy_param(moe.shared_expert_gate.weight, lw[QWEN4_5_W.moe_shared_expert_gate])

            wg = torch.stack(
                [lw[QWEN4_5_W.moe_expert_gate_w.format(e=e)] for e in range(moe.ep_start, moe.ep_end)],
                dim=0,
            ).to(device)
            wgs = torch.stack(
                [lw[QWEN4_5_W.moe_expert_gate_scale.format(e=e)] for e in range(moe.ep_start, moe.ep_end)],
                dim=0,
            ).to(device)
            wu = torch.stack(
                [lw[QWEN4_5_W.moe_expert_up_w.format(e=e)] for e in range(moe.ep_start, moe.ep_end)],
                dim=0,
            ).to(device)
            wus = torch.stack(
                [lw[QWEN4_5_W.moe_expert_up_scale.format(e=e)] for e in range(moe.ep_start, moe.ep_end)],
                dim=0,
            ).to(device)
            wd = torch.stack(
                [lw[QWEN4_5_W.moe_expert_down_w.format(e=e)] for e in range(moe.ep_start, moe.ep_end)],
                dim=0,
            ).to(device)
            wds = torch.stack(
                [lw[QWEN4_5_W.moe_expert_down_scale.format(e=e)] for e in range(moe.ep_start, moe.ep_end)],
                dim=0,
            ).to(device)
            moe.load_mxfp4_weights(
                w_gate_u8=wg,
                w_gate_scale=wgs,
                w_up_u8=wu,
                w_up_scale=wus,
                w2_u8=wd,
                w2_scale=wds,
                transpose_weights=False,
            )

        if layer.ffn_output_sconv is not None:
            _copy_param(layer.ffn_output_sconv.weight, lw[QWEN4_5_W.ffn_output_sconv])

    backbone.prepare_weights()


__all__ = [
    "Qwen2BPETokenizer",
    "SafetensorsShardReader",
    "arch_config_to_hf_dict",
    "bind_qwen4_5_model_weights",
    "load_qwen4_5_from_checkpoint",
    "save_qwen4_5_safetensors_checkpoint",
    "write_safetensors_file",
]
