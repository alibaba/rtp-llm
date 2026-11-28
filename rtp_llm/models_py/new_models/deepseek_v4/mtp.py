"""Direct parameters for V4's single next-token MTP layer, not DSpARK."""

import torch

from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.model_desc.deepseek_v4_mtp_model import DeepSeekV4MtpModel
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.dsv4.fp8.linear_owner import Dsv4FP8Linear
from rtp_llm.models_py.new_models.deepseek_v4.backbone import OwnedV4Transformer
from rtp_llm.models_py.new_models.deepseek_v4.language import (
    OwnedV4ModelMixin,
    _BackboneMapper,
)


class _MtpMapper(_BackboneMapper):
    fusion_leaves = frozenset(
        (
            "enorm.weight",
            "hnorm.weight",
            "e_proj.weight",
            "e_proj.scale",
            "h_proj.weight",
            "h_proj.scale",
        )
    )

    def map_name(self, name):
        local = name.removeprefix("mtp.0.")
        if local != name and local in self.fusion_leaves:
            return local
        return super().map_name(name)


class DeepSeekV4MtpForCausalLM(OwnedV4ModelMixin, DeepSeekV4MtpModel, RtpModule):
    """Reuse main's MTP hidden-state fusion, shared buffers and graph contract."""

    def create_backbone(self, args, load_config):
        if (
            args.n_layers != 1
            or list(args.compress_ratios) != [0]
            or args.n_hash_layers
        ):
            raise ValueError("V4 MTP requires exactly one non-hash SWA layer")
        return OwnedV4Transformer(
            args,
            params_dtype=load_config.compute_dtype,
            # Main's V4 MTP descriptor always loads its own BF16 output head.
            head_dtype=torch.bfloat16,
            layer_prefix="mtp",
            global_prefix="mtp.0.",
        )

    def __init__(self, model_config, load_config):
        super().__init__(model_config, load_config)
        args = self._v4_args
        # Main calls these four modules directly; no after-load reconstruction
        # from global weights, no second set of Parameter/storage references.
        self.enorm = RMSNorm(args.dim, args.norm_eps, load_config.compute_dtype)
        self.hnorm = RMSNorm(args.dim, args.norm_eps, load_config.compute_dtype)
        self.e_proj = Dsv4FP8Linear(args.dim, args.dim)
        self.h_proj = Dsv4FP8Linear(args.dim, args.dim)
        self.WEIGHTS_MAPPER = _MtpMapper(self.v4)

    def checkpoint_weight_name_filter(self):
        # Shared by file path only: unlike DSpARK, main's ordinary MTP service
        # does not declare target parameter loans. Select the root vocabulary
        # and only stage 0; ignore all target layers and any sibling stages.
        return lambda name: name in ("embed.weight", "head.weight") or (
            name.startswith("mtp.0.")
            and name not in ("mtp.0.embed.weight", "mtp.0.head.weight")
        )
