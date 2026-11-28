"""Direct parameters of the DeepSeek-V3.2 / GLM-5 sparse-attention indexer."""
import torch
from torch import nn

from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.module_base import RtpModule


class _IndexerGateLinear(ColumnParallelLinear):
    """FP32 gate: the fused CUDA small-token kernel streams complete [N,K] rows."""

    fused_gate = False

    def process_weights_after_loading(self):
        if not self.fused_gate or torch.version.hip is not None:
            return super().process_weights_after_loading()
        # This ordinary FP32 projection falls back to F.linear on CUDA. Keep
        # one canonical Parameter for both paths, including weight updates.
        self.weight.data = self.weight.detach().contiguous()
        self._post_load_done = True


class SparseIndexerParameters(RtpModule):
    def __init__(
        self, index_n_heads, index_head_dim, hidden_size, q_lora_rank,
        layernorm_eps, quant_config, params_dtype, prefix="self_attn.indexer",
    ):
        super().__init__()
        # wq_b: q_lora_rank → index_n_heads * index_head_dim
        self.wq_b = ColumnParallelLinear(
            input_size=q_lora_rank,
            output_size=index_n_heads * index_head_dim,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            bias=False,
            params_dtype=params_dtype,
        )
        # wk: hidden_size → index_head_dim. The HF Indexer is MQA-style for
        # the key path: a single shared k vector of size `index_head_dim`,
        # not n_heads independent keys. ckpt shape is [head_dim, hidden].
        self.wk = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=index_head_dim,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
            bias=False,
            params_dtype=params_dtype,
        )
        # HF Indexer applies k_norm per-head over `index_head_dim`, not over
        # the full `index_n_heads * index_head_dim` flat feature. The ckpt
        # bias/weight tensors are sized `index_head_dim`.
        self.k_norm = nn.LayerNorm(
            index_head_dim,
            eps=layernorm_eps,
            elementwise_affine=True,
            bias=True,
            dtype=params_dtype,
        ).requires_grad_(False)
        # weights_proj: hidden_size → index_n_heads
        self.weights_proj = _IndexerGateLinear(
            input_size=hidden_size,
            output_size=index_n_heads,
            tp_size=1,
            tp_rank=0,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
            bias=False,
            params_dtype=torch.float32,
        )

    def _apply(self, fn, recurse=True):
        # Main's Indexer._get_logits_head_gate explicitly uses x.float().
        # Casting this projection to BF16 causes an input/weight mismatch, or
        # irreversibly rounds a checkpoint's FP32 router weights.
        source = self.weights_proj.weight.detach()
        result = super()._apply(fn, recurse)
        target = self.weights_proj.weight
        if target.dtype != torch.float32:
            with torch.no_grad():
                target.data = source.to(device=target.device, dtype=torch.float32)
        return result
