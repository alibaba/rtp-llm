"""Qwen4 PLE checkpoint subtree bound to the existing PLE runtime."""

import torch

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen4_exp.hyper_connection import (
    Qwen4ProjectionWeight,
    Qwen4TensorWeight,
)
from rtp_llm.models_py.new_models.qwen4_exp.ple_weights import Qwen4PLEEmbeddingWeights


class Qwen4PLELayer(RtpModule):
    def __init__(
        self,
        *,
        hidden_size,
        hc_mult,
        ngram_size,
        heads_per_ngram,
        shard_count,
        embedding_dim,
        conv_kernel_size,
        eos_token_id,
        tp_size=1,
        tp_rank=0,
        norm_eps=1e-6,
        prefetch_budget_mb=256,
        dtype=torch.bfloat16,
        source_fp8=False,
        fp8_storage=False,
        preserve_fp8_tables=None,
    ):
        super().__init__()
        if any(
            type(x) is not int or x <= 0
            for x in (hidden_size, hc_mult, conv_kernel_size)
        ):
            raise ValueError("Qwen4 PLE dimensions must be positive integers")
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.conv_kernel_size = conv_kernel_size
        self.eos_token_id = eos_token_id
        self.norm_eps = norm_eps
        self.prefetch_budget_mb = prefetch_budget_mb
        width = hidden_size * hc_mult
        input_width = (ngram_size - 1) * heads_per_ngram * embedding_dim
        self.ple_embedding = Qwen4PLEEmbeddingWeights(
            ngram_size=ngram_size,
            heads_per_ngram=heads_per_ngram,
            shard_count=shard_count,
            embedding_dim=embedding_dim,
            tp_size=tp_size,
            tp_rank=tp_rank,
            dtype=dtype,
            source_fp8=source_fp8,
            preserve_fp8=(fp8_storage and source_fp8) if preserve_fp8_tables is None else preserve_fp8_tables,
        )
        self.key_proj = Qwen4ProjectionWeight(
            (width, input_width), dtype=dtype, fp8_storage=fp8_storage
        )
        self.value_proj = Qwen4ProjectionWeight(
            (hidden_size, input_width), dtype=dtype, fp8_storage=fp8_storage
        )
        self.conv1d = Qwen4TensorWeight((width, 1, conv_kernel_size), dtype=dtype)
        self.norm_key = Qwen4TensorWeight((width,), dtype=dtype)
        self.norm_query = Qwen4TensorWeight((width,), dtype=dtype)
        self.norm_conv = Qwen4TensorWeight((width,), dtype=dtype)
        object.__setattr__(self, "_runtime", None)

    def checkpoint_name_filter(self, prefix):
        return self.ple_embedding.checkpoint_name_filter(prefix + "ple_embedding.")

    def process_weights_after_loading(self):
        if self._runtime is not None:
            return
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple import (
            Qwen4ExpNGramEmbedding,
            Qwen4ExpPLELayer,
        )

        table = self.ple_embedding.ngram_embedding
        table.process_weights_after_loading()
        self.key_proj.process_weights_after_loading()
        self.value_proj.process_weights_after_loading()
        embedding = Qwen4ExpNGramEmbedding(
            **self.ple_embedding.runtime_arguments(
                eos_token_id=self.eos_token_id,
                prefetch_budget_mb=self.prefetch_budget_mb,
            )
        )
        try:
            runtime = Qwen4ExpPLELayer(
                embedding,
                self.key_proj.weight,
                self.value_proj.weight,
                self.conv1d.weight,
                self.norm_key.weight,
                self.norm_query.weight,
                self.norm_conv.weight,
                hc_mult=self.hc_mult,
                hidden_size=self.hidden_size,
                conv_kernel_size=self.conv_kernel_size,
                norm_eps=self.norm_eps,
                projection_scales=(
                    self.key_proj.storage_scale,
                    self.value_proj.storage_scale,
                ),
            )
            # UVA preparation may replace pageable source storage. The leaf
            # remains the sole owner of its TP-local CPU shards after adoption.
            table.adopt_runtime_shards(embedding.shards)
        except BaseException:
            embedding.close()
            raise
        for parameter in self.parameters():
            parameter._rtp_requires_restart_for_update = True
        for buffer in self.buffers():
            # Hash metadata and raw FP8 table scales participate in cached
            # state/UVA consumers too; updating only these tensors is not safe.
            buffer._rtp_requires_restart_for_update = True
        object.__setattr__(self, "_runtime", runtime)

    @property
    def runtime(self):
        if self._runtime is None:
            raise RuntimeError("Qwen4 PLE weights have not been finalized")
        return self._runtime

    def forward(self, *args, **kwargs):
        return self.runtime(*args, **kwargs)

    def close(self):
        if self._runtime is not None:
            self._runtime.ple_embedding.close()
