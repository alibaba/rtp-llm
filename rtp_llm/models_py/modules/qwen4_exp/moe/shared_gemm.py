"""Prepare the canonical BF16 Linear layout for Qwen4 shared projections."""

import torch

from rtp_llm.utils.model_weight import W


@torch.inference_mode()
def prepare_bf16_shared_weights(source_layers, config=None, parallelism=None):
    """Normalize TP shards, including replicated weights from older rank bundles."""
    for source in source_layers:
        sharded = (
            config is not None
            and getattr(config, "_qwen4_dense_storage", None) == "source_bf16"
            and parallelism is not None
        )
        if sharded:
            up, down = source.get(W.ffn_w13), source.get(W.ffn_w2)
            if up is None or down is None:
                continue
            hidden, intermediate = config.hidden_size, config.inter_size
            tp, rank = parallelism.get_attn_tp_size(), parallelism.tp_rank
            if intermediate % tp:
                raise ValueError("Shared expert width must be divisible by TP")
            local = intermediate // tp
            if up.shape == (hidden, 2 * intermediate) and tp > 1:
                gate, value = up.chunk(2, dim=1)
                up.set_(
                    torch.cat(
                        (
                            gate[:, rank * local : (rank + 1) * local],
                            value[:, rank * local : (rank + 1) * local],
                        ),
                        dim=1,
                    )
                )
            if down.shape == (intermediate, hidden) and tp > 1:
                down.set_(down[rank * local : (rank + 1) * local].clone())
            if up.shape != (hidden, 2 * local) or down.shape != (local, hidden):
                raise ValueError("Shared expert TP shard does not match model geometry")
        for name, scale_name in ((W.ffn_w13, W.ffn_s13), (W.ffn_w2, W.ffn_s2)):
            weight = source.get(name)
            if (
                weight is None
                or weight.ndim != 2
                or weight.dtype != torch.bfloat16
                or not weight.is_cuda
                or source.get(scale_name) is not None
                or weight.t().is_contiguous()
                or torch.cuda.get_device_capability(weight.device) != (12, 0)
            ):
                continue

            # Run before the Linear modules take their transpose views. Keep
            # the same tensor owner so model updates still copy into this storage.
            weight.set_(weight.t().contiguous().t())
