"""Qwen 4.5-Flash (``Qwen4_5ForCausalLM``) RTP-LLM model registration wrapper.

Heavy C++ dependencies (``rtp_llm.models.base_model``, ``rtp_llm.ops``) are
imported lazily so configuration parsing, weight validation, and KV cache
planning work cleanly in pure-Python test environments as well as in compiled
Bazel/RTP-LLM binaries.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Collection, Dict, Mapping, Optional, Sequence, Type, Union

from rtp_llm.config.qwen4_5_config import (
    EXPECTED_MODEL_TYPE,
    Qwen4_5ModelConfig,
    Qwen4_5ParamAndStateEstimate,
    estimate_qwen4_5_params_and_state,
    parse_qwen4_5_config_json,
    parse_qwen4_5_quant_config,
)
from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import (
    Qwen4_5KVCachePlan,
    build_qwen4_5_kv_cache_plan,
    build_qwen4_5_kv_cache_spec_descs,
)
from rtp_llm.models.qwen4_5.qwen4_5_weight import (
    Qwen4_5Weight,
    validate_qwen4_5_weight_map,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig

try:
    from rtp_llm.models.base_model import BaseModel as _BaseModel

    _HAS_RTP_BASE_MODEL = True
except Exception:  # pragma: no cover - pure-Python fallback without C++ ops
    _BaseModel = object  # type: ignore[assignment,misc]
    _HAS_RTP_BASE_MODEL = False


class Qwen4_5(_BaseModel):
    """RTP-LLM model wrapper for Qwen 4.5-Flash (``Qwen4_5ForCausalLM``).

    Architecture summary (1109-bf16-re-mxfp4-ple-fp8):
      - 40 hybrid layers (30 Gated DeltaNet linear + 10 MQA-1 full attention)
      - YoCo cross-layer attention sharing (layer 19 -> 5 full consumers;
        layer 20 -> 14 linear consumers)
      - 4-stream Hyper-Connections with ``hc_prev_mix`` and global mixer norm
      - Layer 1 FP8 per-tensor 128-shard PLE n-gram table
      - Layer 0 dense FFN; Layers 1..39 Latent MoE (640 MXFP4 routed experts,
        top-10, SiTU-GLU) + BF16 shared expert
      - ShortConv on FullAttn K/V, attention outputs, and FFN outputs
    """

    MODEL_TYPE: str = EXPECTED_MODEL_TYPE
    ARCHITECTURES = ("Qwen4_5ForCausalLM",)
    _model_config_cls = Qwen4_5ModelConfig

    @staticmethod
    def get_weight_cls() -> Type[Qwen4_5Weight]:
        return Qwen4_5Weight

    def support_pp(self) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        return True

    def _py_model_class(self) -> Any:
        from rtp_llm.models_py.model_desc.qwen4_5_runtime import Qwen4_5RTPModel

        return Qwen4_5RTPModel

    @classmethod
    def _create_config(cls, ckpt_path: str) -> Qwen4_5ModelConfig:
        return cls.create_config(ckpt_path)

    @classmethod
    def create_config(
        cls,
        ckpt_path: Union[str, Path],
        *,
        strict: bool = True,
        **kwargs: Any,
    ) -> Qwen4_5ModelConfig:
        """Create a ``Qwen4_5ModelConfig`` from a checkpoint directory or ``config.json``."""
        p = Path(ckpt_path)
        config_path = p / "config.json" if p.is_dir() else p
        config_json = json.loads(config_path.read_text(encoding="utf-8"))
        return cls._create_config_from_json(
            str(config_path.parent), config_json, strict=strict
        )

    @classmethod
    def _create_config_from_json(
        cls,
        ckpt_path: str,
        config_json: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> Qwen4_5ModelConfig:
        arch_cfg = parse_qwen4_5_config_json(config_json, strict=strict)
        quant_cfg = parse_qwen4_5_quant_config(config_json)
        config = cls._model_config_cls(
            arch_config=arch_cfg,
            quant_config=quant_cfg,
            ckpt_path=ckpt_path,
            raw_config=config_json,
        )
        cls._populate_model_config(config, arch_cfg)
        return config

    @classmethod
    def _populate_model_config(
        cls,
        config: Qwen4_5ModelConfig,
        arch_cfg: Qwen4_5ArchConfig,
    ) -> None:
        config.model_type = cls.MODEL_TYPE
        config.vocab_size = arch_cfg.vocab_size
        config.hidden_size = arch_cfg.hidden_size
        config.num_layers = arch_cfg.num_hidden_layers
        config.hc_mult = arch_cfg.hc_count
        config.expert_num = arch_cfg.num_experts
        config.moe_k = arch_cfg.num_experts_per_tok
        config.tie_word_embeddings = arch_cfg.tie_word_embeddings
        cls._populate_qsa_fields(config, arch_cfg)

        # Attach pure-Python KV cache plan and exact sizing summary
        config.qwen4_5_kv_cache_plan = build_qwen4_5_kv_cache_plan(arch_cfg)
        config.qwen4_5_param_and_state_estimate = estimate_qwen4_5_params_and_state(
            arch_cfg
        )

        # Populate compiled RTP-LLM ModelConfig sub-structures when ops are available
        if _HAS_RTP_BASE_MODEL:
            try:
                config.max_seq_len = int(arch_cfg.max_position_embeddings)
                config.config_dtype = "bfloat16"
                config.has_lm_head = True
                config.enable_fp32_lm_head = False
                config.has_post_decoder_layernorm = False
                config.has_pre_decoder_layernorm = False
                config.inter_size = arch_cfg.intermediate_size
                config.moe_inter_size = arch_cfg.moe_intermediate_size
                config.moe_layer_index = list(
                    range(arch_cfg.first_k_dense_replace, arch_cfg.num_hidden_layers)
                )
                if hasattr(config, "special_tokens"):
                    config.special_tokens.bos_token_id = int(arch_cfg.bos_token_id)
                    config.special_tokens.eos_token_id = int(arch_cfg.eos_token_id)
                if hasattr(config, "attn_config"):
                    cls._apply_qsa_attn_config(config.attn_config, arch_cfg)
                if hasattr(config, "hybrid_attention_config"):
                    # GDN uses OPAQUE_STATE cache descriptors rather than KVCacheSpecType.LINEAR;
                    # CacheConfigCreator requires hybrid_attention_types empty and linear_num_value_heads == 0.
                    config.hybrid_attention_config.enable_hybrid_attention = False
                    config.hybrid_attention_config.hybrid_attention_types = []
                if hasattr(config, "linear_attention_config"):
                    config.linear_attention_config.linear_num_value_heads = 0
                config.kv_cache_spec_descs = build_qwen4_5_kv_cache_spec_descs(
                    config, arch_cfg
                )
            except Exception:
                pass

    @staticmethod
    def _populate_qsa_fields(config: Any, arch_cfg: Qwen4_5ArchConfig) -> None:
        config.qwen4_5_has_qsa = bool(arch_cfg.has_qsa)
        config._qwen4_indexer_head_dim = int(arch_cfg.indexer_head_dim)
        config._qwen4_indexer_n_heads = int(arch_cfg.indexer_n_heads)
        config._qwen4_indexer_kv_heads = int(arch_cfg.indexer_kv_heads)
        config._qwen4_indexer_compress_ratio = int(arch_cfg.indexer_compress_ratio)
        config._qwen4_indexer_budget = int(arch_cfg.indexer_budget)

    @staticmethod
    def _apply_qsa_attn_config(attn: Any, arch_cfg: Qwen4_5ArchConfig) -> None:
        attn.head_num = arch_cfg.num_attention_heads
        attn.kv_head_num = arch_cfg.num_key_value_heads
        attn.size_per_head = arch_cfg.head_dim
        if arch_cfg.has_qsa:
            attn.is_sparse = True
            if hasattr(attn, "use_sparse_gqa_fmha"):
                attn.use_sparse_gqa_fmha = True
            attn.indexer_head_dim = int(arch_cfg.indexer_head_dim)
            attn.indexer_head_num = int(arch_cfg.indexer_n_heads)
            attn.indexer_topk = int(
                arch_cfg.indexer_budget // arch_cfg.indexer_compress_ratio
            )

    @classmethod
    def _apply_kv_cache_config(
        cls, model_config: Any, kv_cache_config: Any
    ) -> None:
        attn = getattr(model_config, "attn_config", None)
        if attn is not None and kv_cache_config is not None:
            seq_size = int(getattr(kv_cache_config, "seq_size_per_block", 0) or 0)
            if seq_size > 0 and getattr(attn, "tokens_per_block", 0) in (0, None):
                attn.tokens_per_block = seq_size
            kernel_seq_size = int(
                getattr(kv_cache_config, "kernel_seq_size_per_block", 0) or 0
            )
            if kernel_seq_size > 0 and getattr(attn, "kernel_tokens_per_block", 0) in (
                0,
                None,
            ):
                attn.kernel_tokens_per_block = kernel_seq_size
        cls._post_build_model_config(model_config)

    @classmethod
    def _post_build_model_config(cls, model_config: Any) -> None:
        """Finalize RTP-LLM runtime config, hybrid KV cache specs, and PD cache tags."""
        arch_cfg = getattr(model_config, "qwen4_5_arch_config", None) or getattr(
            model_config, "arch_config", None
        )
        if arch_cfg is None:
            raw_cfg = getattr(model_config, "raw_config", None)
            if isinstance(raw_cfg, Mapping):
                arch_cfg = parse_qwen4_5_config_json(raw_cfg, strict=False)
                model_config.qwen4_5_arch_config = arch_cfg
        if arch_cfg is None:
            return

        attn = getattr(model_config, "attn_config", None)
        if attn is not None:
            if getattr(attn, "tokens_per_block", 0) in (0, None):
                attn.tokens_per_block = 64
            if getattr(attn, "kernel_tokens_per_block", 0) in (0, None):
                attn.kernel_tokens_per_block = attn.tokens_per_block
            cls._apply_qsa_attn_config(attn, arch_cfg)
            if getattr(attn, "dtype", None) is None:
                import torch

                attn.dtype = torch.bfloat16

        cls._populate_qsa_fields(model_config, arch_cfg)
        model_config.hc_mult = arch_cfg.hc_count
        model_config.has_lm_head = True
        model_config.has_post_decoder_layernorm = False
        model_config.has_pre_decoder_layernorm = False
        model_config.qwen4_5_kv_cache_plan = cls.build_kv_cache_plan(arch_cfg)
        try:
            if hasattr(model_config, "hybrid_attention_config"):
                model_config.hybrid_attention_config.enable_hybrid_attention = False
                model_config.hybrid_attention_config.hybrid_attention_types = []
            if hasattr(model_config, "linear_attention_config"):
                model_config.linear_attention_config.linear_num_value_heads = 0
            model_config.kv_cache_spec_descs = build_qwen4_5_kv_cache_spec_descs(
                model_config,
                arch_cfg,
                allow_small_block_explicit=True,
            )
        except Exception:
            # Pure-Python environment without compiled rtp_llm.ops
            pass

    def load(self, skip_python_model: bool = False) -> None:
        if (
            self.parallelism_config.pp_size > 1
            and not self.model_config.is_mtp
            and not self.support_pp()
        ):
            raise RuntimeError("Qwen4_5 does not support pipeline parallelism")
        if (
            self.hw_kernel_config.enable_cuda_graph
            and self.support_cuda_graph() is False
        ):
            raise RuntimeError("Qwen4_5 does not support CUDA graph in py model mode")

        self.custom_module = None
        self.model_weights_loader = None
        self.py_eplb = None
        self.weight_manager = None
        device_str = self._get_device_str()
        self._load(device_str)
        if skip_python_model:
            return
        logging.info(
            f"Creating Qwen4_5 python model for {self.model_config.ckpt_path} on {device_str}"
        )
        self._create_python_model()

    @staticmethod
    def _resolve_load_parallel_topology(parallelism_config: Any, *, dist_module: Any):
        """Resolve runtime TP/DP/EP/SP topology for `_load()` with opt-in env rollout."""
        from rtp_llm.models_py.model_desc.qwen4_5 import (
            _SUPPORTED_PARALLEL_MODES,
            resolve_qwen4_5_parallel_topology,
        )

        mode_aliases = {
            "tpsp4": "sp4_tp4_ep4",
            "tpsp4_ep4": "sp4_tp4_ep4",
        }

        def _parse_mode(raw: Any, source: str) -> Optional[str]:
            if raw is None:
                return None
            mode_s = str(raw).strip().lower()
            if not mode_s:
                return None
            mode_s = mode_aliases.get(mode_s, mode_s)
            if mode_s not in _SUPPORTED_PARALLEL_MODES:
                raise ValueError(
                    f"Unsupported {source}={raw!r}; expected one of {_SUPPORTED_PARALLEL_MODES}"
                )
            return mode_s

        def _parse_sp_size(raw: Any, source: str) -> Optional[int]:
            if raw is None:
                return None
            if isinstance(raw, str):
                s_val = raw.strip()
                if not s_val:
                    return None
                try:
                    sp_i = int(s_val)
                except ValueError as exc:
                    raise ValueError(f"Invalid integer for {source}={raw!r}") from exc
            else:
                sp_i = int(raw)
            if sp_i < 0:
                raise ValueError(f"{source} must be non-negative, got {raw!r}")
            return None if sp_i == 0 else sp_i

        tp_size = int(parallelism_config.tp_size)
        tp_rank = int(parallelism_config.tp_rank)
        dp_size = int(parallelism_config.dp_size)
        dp_rank = int(getattr(parallelism_config, "dp_rank", 0))
        ep_size_cfg = int(getattr(parallelism_config, "ep_size", 1))
        ep_rank_cfg = int(getattr(parallelism_config, "ep_rank", 0))

        cfg_mode = _parse_mode(
            getattr(parallelism_config, "parallel_mode", None),
            "parallelism_config.parallel_mode",
        )
        raw_parallel_mode = (
            cfg_mode
            if cfg_mode is not None
            else _parse_mode(os.environ.get("QWEN4_5_PARALLEL_MODE"), "QWEN4_5_PARALLEL_MODE")
        )

        cfg_sp_size = _parse_sp_size(
            getattr(parallelism_config, "sp_size", None),
            "parallelism_config.sp_size",
        )
        # Explicit parallelism_config.parallel_mode controls topology without mixing stale env overrides.
        raw_sp_size = (
            cfg_sp_size
            if (cfg_sp_size is not None or cfg_mode is not None)
            else _parse_sp_size(os.environ.get("QWEN4_5_SP_SIZE"), "QWEN4_5_SP_SIZE")
        )

        raw_sp_rank_attr = getattr(parallelism_config, "sp_rank", None)
        raw_sp_rank = int(raw_sp_rank_attr) if raw_sp_rank_attr is not None else None
        raw_dp_attn = getattr(parallelism_config, "dp_attn", None)

        if raw_parallel_mode == "sp4_tp4_ep4" and dp_size != 1:
            raise ValueError(
                f"parallel_mode='sp4_tp4_ep4' requires dp_size=1, got dp_size={dp_size} (tp_size={tp_size})."
            )
        if raw_parallel_mode == "dp2_tpsp2_ep4" and dp_size <= 1:
            raise ValueError(
                f"parallel_mode='dp2_tpsp2_ep4' requires dp_size > 1, got dp_size={dp_size}."
            )
        if raw_parallel_mode in ("sp4_tp4_ep4", "dp2_tpsp2_ep4"):
            if tp_size <= 1:
                raise ValueError(
                    f"parallel_mode={raw_parallel_mode!r} requires tp_size > 1, got tp_size={tp_size}."
                )
            if raw_sp_size == 1:
                raise ValueError(
                    f"parallel_mode={raw_parallel_mode!r} conflicts with sp_size=1."
                )
        if raw_parallel_mode in ("tp4_ep4", "dp4_ep4", "dp2_tp2_ep4") and (
            raw_sp_size is not None and raw_sp_size > 1
        ):
            raise ValueError(
                f"parallel_mode={raw_parallel_mode!r} conflicts with sp_size={raw_sp_size}."
            )
        if raw_sp_size is not None and raw_sp_size > 1 and (tp_size <= 1 or raw_sp_size != tp_size):
            raise ValueError(
                f"SP requires tp_size == sp_size > 1, got tp_size={tp_size}, sp_size={raw_sp_size}."
            )

        tp_group = getattr(parallelism_config, "tp_group", None)
        dp_group = getattr(parallelism_config, "dp_group", None)
        sp_group = getattr(parallelism_config, "sp_group", None)
        ep_group = getattr(parallelism_config, "ep_group", None)

        dist_init = bool(dist_module.is_available() and dist_module.is_initialized())
        collective_get_group = None
        collective_group_cls = None
        if dist_init:
            try:
                from rtp_llm.models_py.distributed import collective_torch

                if getattr(collective_torch, "_initialized", False):
                    from rtp_llm.models_py.distributed.collective_torch import _get_group
                    from rtp_llm.models_py.distributed.rank_layout import Group

                    collective_get_group = _get_group
                    collective_group_cls = Group
            except Exception:
                pass

        if tp_size > 1 and tp_group is None and collective_get_group is not None:
            try:
                tp_group = collective_get_group(collective_group_cls.TP)
            except Exception:
                pass

        eff_ep_size = 1
        eff_ep_rank = 0
        if tp_size > 1 and dp_size == 1:
            eff_ep_size = tp_size
            eff_ep_rank = tp_rank
            if ep_group is None:
                ep_group = tp_group
        elif ep_size_cfg > 1:
            eff_ep_size = ep_size_cfg
            eff_ep_rank = ep_rank_cfg
            if ep_group is None and collective_get_group is not None:
                try:
                    ep_group = collective_get_group(collective_group_cls.WORLD)
                except Exception:
                    pass

        # Always resolve SP/DP/TP topology even when collective_torch already populated tp_group & ep_group,
        # so sp_size == tp_size inherits sp_group = tp_group, sp_rank = tp_rank, dp_attn = False cleanly.
        topo = resolve_qwen4_5_parallel_topology(
            parallel_mode=raw_parallel_mode,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            ep_size=eff_ep_size,
            ep_rank=eff_ep_rank,
            ep_group=ep_group,
            dp_size=dp_size,
            dp_rank=dp_rank,
            dp_group=dp_group,
            sp_size=raw_sp_size,
            sp_rank=raw_sp_rank,
            sp_group=sp_group,
            dp_attn=raw_dp_attn,
        )
        sp_or_mode_explicit = (raw_parallel_mode is not None) or (
            raw_sp_size is not None and raw_sp_size > 1
        )
        resolved_sp_group = topo.sp_group
        if (
            resolved_sp_group is None
            and dist_init
            and int(topo.sp_size) > 1
            and int(topo.sp_size) == int(topo.tp_size)
            and topo.tp_group is not None
        ):
            resolved_sp_group = topo.tp_group
        eff_parallel_mode = topo.parallel_mode if sp_or_mode_explicit else None
        eff_dp_attn = topo.dp_attn if sp_or_mode_explicit else raw_dp_attn
        return {
            "parallel_mode": eff_parallel_mode,
            "tp_size": int(topo.tp_size),
            "tp_rank": int(topo.tp_rank),
            "tp_group": topo.tp_group,
            "dp_size": int(topo.dp_size),
            "dp_rank": int(topo.dp_rank),
            "dp_group": topo.dp_group,
            "ep_size": int(topo.ep_size),
            "ep_rank": int(topo.ep_rank),
            "ep_group": topo.ep_group,
            "sp_size": int(topo.sp_size),
            "sp_rank": int(topo.sp_rank),
            "sp_group": resolved_sp_group,
            "dp_attn": eff_dp_attn,
        }

    def _load(self, device: str) -> None:
        import torch
        import torch.distributed as dist

        from rtp_llm.model_loader.model_weight_info import ModelWeights
        from rtp_llm.models_py.model_desc.qwen4_5 import Qwen4_5ForCausalLM
        from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
            load_qwen4_5_from_checkpoint,
        )
        from rtp_llm.models_py.modules.qwen4_5.fast_comm import (
            get_qwen4_5_fast_all_reduce,
        )
        from rtp_llm.utils.model_weight import W

        self.device = device
        arch_cfg = self.model_config.arch_config
        kv_dtype = str(getattr(self.model_config, "kv_dtype", "bf16") or "bf16")
        topo_cfg = self._resolve_load_parallel_topology(
            self.parallelism_config, dist_module=dist
        )
        parallel_mode = topo_cfg["parallel_mode"]
        tp_size = topo_cfg["tp_size"]
        tp_rank = topo_cfg["tp_rank"]
        tp_group = topo_cfg["tp_group"]
        dp_size = topo_cfg["dp_size"]
        dp_rank = topo_cfg["dp_rank"]
        dp_group = topo_cfg["dp_group"]
        eff_ep_size = topo_cfg["ep_size"]
        eff_ep_rank = topo_cfg["ep_rank"]
        ep_group = topo_cfg["ep_group"]
        sp_size = topo_cfg["sp_size"]
        sp_rank = topo_cfg["sp_rank"]
        sp_group = topo_cfg["sp_group"]
        dp_attn = topo_cfg["dp_attn"]
        if sp_size > 1 or parallel_mode is not None:
            logging.info(
                "Qwen4_5._load topology: parallel_mode=%s tp=%d/%d dp=%d/%d ep=%d/%d sp=%d/%d dp_attn=%s",
                parallel_mode,
                tp_rank,
                tp_size,
                dp_rank,
                dp_size,
                eff_ep_rank,
                eff_ep_size,
                sp_rank,
                sp_size,
                dp_attn,
            )

        # Refresh TP-aware KV/GDN cache plan and KVCacheSpecDescs before engine pool creation.
        if tp_size > 1:
            self.model_config.qwen4_5_kv_cache_plan = self.build_kv_cache_plan(
                arch_cfg, tp_size=tp_size, parallel_mode=parallel_mode
            )
            try:
                self.model_config.kv_cache_spec_descs = build_qwen4_5_kv_cache_spec_descs(
                    self.model_config,
                    arch_cfg,
                    tp_size=tp_size,
                    parallel_mode=parallel_mode,
                    allow_small_block_explicit=True,
                )
            except Exception:
                pass

        torch_dev = torch.device(device)
        if torch_dev.type == "cuda":
            torch.cuda.set_device(torch_dev)

        fast_all_reduce = None
        if (
            tp_size > 1
            and tp_group is not None
            and torch_dev.type == "cuda"
            and dist.is_available()
            and dist.is_initialized()
        ):
            fast_all_reduce = get_qwen4_5_fast_all_reduce(
                tp_group,
                device=torch_dev,
                dtype=torch.bfloat16,
                hidden_dims=(int(arch_cfg.hidden_size), int(arch_cfg.routed_expert_hidden_size)),
            )

        causal_lm = Qwen4_5ForCausalLM(
            arch_cfg,
            use_triton_mxfp4=True,
            compute_mode="bf16",
            dtype=torch.bfloat16,
            device=torch_dev,
            init_mxfp4_weights=False,
            parallel_mode=parallel_mode,
            ep_size=eff_ep_size,
            ep_rank=eff_ep_rank,
            ep_group=ep_group,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            dp_size=dp_size,
            dp_rank=dp_rank,
            dp_group=dp_group,
            sp_size=sp_size,
            sp_rank=sp_rank,
            sp_group=sp_group,
            dp_attn=dp_attn,
            kv_dtype=kv_dtype,
            dense_linear_mode=getattr(self.model_config, "dense_linear_mode", None),
            fast_all_reduce=fast_all_reduce,
        )
        num_workers = int(os.environ.get("QWEN4_5_LOAD_WORKERS", "8"))
        load_qwen4_5_from_checkpoint(
            causal_lm,
            self.model_config.ckpt_path,
            ple_mode="gpu_sharded",
            num_workers=num_workers,
            verbose=True,
        )
        self._qwen4_5_causal_lm = causal_lm

        weights = ModelWeights(
            num_layers=int(self.model_config.num_layers),
            device=device,
            dtype=torch.bfloat16,
        )
        weights.global_weights[W.embedding] = (
            causal_lm.model.embed_tokens.weight.detach().contiguous()
        )
        lm_w = causal_lm.lm_head.weight.detach()
        if tp_size > 1 and int(lm_w.shape[0]) == int(arch_cfg.vocab_size):
            vocab_size = int(lm_w.shape[0])
            if vocab_size % tp_size != 0:
                raise ValueError(
                    f"Qwen4_5 vocab_size ({vocab_size}) must be divisible by tp_size ({tp_size})"
                )
            vocab_per_rank = vocab_size // tp_size
            start = tp_rank * vocab_per_rank
            lm_head_slice = lm_w[start : start + vocab_per_rank].contiguous()
        else:
            lm_head_slice = lm_w.contiguous()
        # Do NOT populate W.final_ln_gamma; Qwen4_5Model outputs already post-normed hidden states.
        weights.global_weights[W.lm_head] = lm_head_slice
        self.weight = weights

    def _create_python_model(self) -> None:
        from rtp_llm.models_py.model_desc.qwen4_5_runtime import Qwen4_5RTPModel

        self.py_model = Qwen4_5RTPModel.from_causal_lm(
            self._qwen4_5_causal_lm,
            model_config=self.model_config,
            parallelism_config=self.parallelism_config,
            weight=self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
        )

    @classmethod
    def build_kv_cache_plan(
        cls,
        arch_cfg: Qwen4_5ArchConfig,
        *,
        tp_size: int = 1,
        parallel_mode: Optional[str] = None,
        enable_yoco_decode: Optional[bool] = None,
        kv_dtype: str = "BF16",
    ) -> Qwen4_5KVCachePlan:
        return build_qwen4_5_kv_cache_plan(
            arch_cfg,
            tp_size=tp_size,
            parallel_mode=parallel_mode,
            enable_yoco_decode=enable_yoco_decode,
            kv_dtype=kv_dtype,
        )

    @classmethod
    def estimate_params_and_state(
        cls,
        arch_cfg: Qwen4_5ArchConfig,
        *,
        enable_yoco_decode: Optional[bool] = None,
        kv_dtype_bytes: int = 2,
    ) -> Qwen4_5ParamAndStateEstimate:
        return estimate_qwen4_5_params_and_state(
            arch_cfg,
            enable_yoco_decode=enable_yoco_decode,
            kv_dtype_bytes=kv_dtype_bytes,
        )

    @classmethod
    def validate_checkpoint_index(
        cls,
        arch_cfg: Qwen4_5ArchConfig,
        weight_map: Union[Mapping[str, Any], Collection[str]],
        *,
        prefix: str = "model.",
        allow_extra_prefixes: Sequence[str] = (),
    ) -> Dict[str, Any]:
        return validate_qwen4_5_weight_map(
            arch_cfg,
            weight_map,
            prefix=prefix,
            allow_extra_prefixes=allow_extra_prefixes,
        )


def register_qwen4_5_model() -> None:
    """Register ``qwen4_5`` / ``Qwen4_5ForCausalLM`` with ``model_factory_register``."""
    from rtp_llm.model_factory_register import register_model

    register_model(Qwen4_5.MODEL_TYPE, Qwen4_5, list(Qwen4_5.ARCHITECTURES))


try:
    register_qwen4_5_model()
except Exception:  # pragma: no cover - safe if registry is unavailable
    pass
