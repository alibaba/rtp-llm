"""Opt-in, per-model batch invariant execution. CUDA imports stay lazy."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchInvariantPolicy:
    enabled: bool = False
    profile: str = "dense-gqa-v1"

    @contextmanager
    def construction(self):
        token = _construction_policy.set(self)
        try:
            yield
        finally:
            _construction_policy.reset(token)

    def validate_model(self, model):
        if not self.enabled:
            return
        import torch

        from rtp_llm.ops import ActivationType, RopeStyle

        if getattr(type(model), "batch_invariant_capability", None) != self.profile:
            raise ValueError(
                f"batch invariant: {type(model).__name__} has no audited {self.profile} execution contract"
            )
        pc = model.parallelism_config
        if (
            any(
                getattr(pc, key) != 1
                for key in ("tp_size", "dp_size", "ep_size", "pp_size", "ffn_sp_size")
            )
            or pc.get_attn_tp_size() != 1
            or pc.get_ffn_tp_size() != 1
        ):
            raise ValueError(
                "batch invariant dense-gqa-v1 currently requires single-rank execution"
            )
        if (
            pc.prefill_cp_config.is_enabled()
            or pc.ffn_disaggregate_config.enable_ffn_disaggregate
        ):
            raise ValueError(
                "batch invariant dense-gqa-v1 does not support CP or disaggregated FFN"
            )
        cfg = model.config
        attn = cfg.getAttentionConfigs(1)
        if attn.dtype != torch.bfloat16 or attn.use_mla or not attn.is_causal:
            raise ValueError(
                "batch invariant dense-gqa-v1 requires BF16 causal MHA/GQA"
            )
        if attn.rope_config.style != RopeStyle.Base or attn.size_per_head not in (
            64,
            128,
            256,
        ):
            raise ValueError(
                "batch invariant dense-gqa-v1 requires base RoPE and head_dim 64/128/256"
            )
        if cfg.activation_type != ActivationType.Swiglu:
            raise ValueError("batch invariant dense-gqa-v1 requires SwiGLU")
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            attn_kv_dtype,
        )

        if attn_kv_dtype(attn) != torch.bfloat16:
            raise ValueError("batch invariant dense-gqa-v1 requires BF16 KV cache")
        if torch.cuda.get_device_capability()[0] in (9, 10):
            from rtp_llm.ops.compute_ops import rtp_llm_ops

            if not hasattr(rtp_llm_ops, "batch_invariant_cublaslt_gemm"):
                raise RuntimeError(
                    "batch invariant requires rebuilt native libraries with the scoped cuBLASLt operation"
                )


_construction_policy = ContextVar(
    "rtp_batch_invariant_construction", default=BatchInvariantPolicy()
)


def construction_policy():
    """Only consulted during construction; modules retain an immutable snapshot."""
    return _construction_policy.get()
