"""HY V4 MTP model using the generic single-residual draft flow."""

from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel
from rtp_llm.utils.model_weight import W


class Hy4MtpModel(GenericMoeMTPModel):
    """One HY V4 draft block.

    MTP intentionally disables iHC, while its ``MlaAttention`` still consumes
    the HY-specific elementwise gate and per-head attention sink weights.
    """

    def __init__(self, model_config, parallelism_config, weights, *args, **kwargs):
        if model_config.enable_ihc:
            raise ValueError("HY V4 MTP checkpoint does not contain iHC weights")
        if not model_config.gated_mla or not model_config.learnable_sink:
            raise ValueError("HY V4 MTP requires gated MLA and attention sinks")
        for layer_idx, layer_weights in enumerate(
            weights.weights[: model_config.num_layers]
        ):
            missing = [
                key
                for key in (W.attn_gate_w, W.hy4_attn_sink)
                if key not in layer_weights
            ]
            if missing:
                raise KeyError(
                    f"HY V4 MTP layer {layer_idx} is missing required MLA "
                    f"weights: {missing}"
                )
        super().__init__(
            model_config, parallelism_config, weights, *args, **kwargs
        )


__all__ = ["Hy4MtpModel"]
