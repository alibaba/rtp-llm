"""Replicated QSA checkpoint owner, using the existing selection runtime."""

import torch

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp.hyper_connection import (
    Qwen4ProjectionWeight,
    Qwen4TensorWeight,
)


class Qwen4QSAIndexer(RtpModule):
    def __init__(
        self,
        hidden_size,
        *,
        n_heads,
        head_dim,
        token_budget,
        compress_ratio,
        kv_heads=1,
        norm_eps=1e-6,
        dtype=torch.bfloat16,
        fp8_storage=False,
        rope_config=None,
    ):
        super().__init__()
        if any(
            type(x) is not int or x <= 0
            for x in (hidden_size, n_heads, head_dim, token_budget, compress_ratio)
        ):
            raise ValueError(
                "Qwen4 QSA dimensions and budget must be positive integers"
            )
        if kv_heads != 1 or token_budget % compress_ratio:
            raise ValueError(
                "Qwen4 QSA requires MQA and a budget divisible by compress_ratio"
            )
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.token_budget = token_budget
        self.compress_ratio = compress_ratio
        self.norm_eps = norm_eps
        self.rope_config = rope_config
        self.index_qk_proj = Qwen4ProjectionWeight(
            ((n_heads + kv_heads) * head_dim, hidden_size),
            dtype=dtype,
            fp8_storage=fp8_storage,
        )
        self.q_layernorm = Qwen4TensorWeight((head_dim,), dtype=dtype)
        self.k_layernorm = Qwen4TensorWeight((head_dim,), dtype=dtype)
        object.__setattr__(self, "_runtime", None)

    def process_weights_after_loading(self):
        if self._runtime is not None:
            return
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import Qwen4ExpQSAIndexer

        self.index_qk_proj.process_weights_after_loading()
        runtime = Qwen4ExpQSAIndexer(
            self.index_qk_proj.weight,
            self.q_layernorm.weight,
            self.k_layernorm.weight,
            n_heads=self.n_heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            token_budget=self.token_budget,
            compress_ratio=self.compress_ratio,
            norm_eps=self.norm_eps,
            projection_scale=self.index_qk_proj.storage_scale,
            rope_config=self.rope_config,
        )
        for parameter in self.parameters():
            parameter._rtp_requires_restart_for_update = True
        object.__setattr__(self, "_runtime", runtime)

    @property
    def runtime(self):
        if self._runtime is None:
            raise RuntimeError("Qwen4 QSA weights have not been finalized")
        return self._runtime

    def forward(self, *args, **kwargs):
        return self.runtime.forward_causal(*args, **kwargs)
