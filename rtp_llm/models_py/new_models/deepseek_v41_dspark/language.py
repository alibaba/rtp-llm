"""V4.1 DSpARK direct ownership on main's actual draft/commit/graph runtime."""

import torch

from rtp_llm.models_py.model_desc.deepseek_v41_dspark_model import (
    DeepSeekV41DSparkModel,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v41.backbone import runtime_backbone_class
from rtp_llm.models_py.new_models.deepseek_v41.owner import OwnedV41Mixin
from rtp_llm.models_py.new_models.deepseek_v41_dspark.checkpoint import (
    V41DraftGlobalsMixin,
    V41DraftParameterMixin,
)


class DeepSeekV41DSparkForCausalLM(
    V41DraftParameterMixin, OwnedV41Mixin, DeepSeekV41DSparkModel, RtpModule
):
    _captures_aux_hidden = False
    preserved_ft_parameter_loans = {
        "v4.embed.weight": "global.embedding",
        "v4.head.weight": "global.lm_head",
    }

    @property
    def eplb_attention_only(self):
        return self._commit_only_prefill

    def __init__(self, model_config, load_config):
        super().__init__(model_config, load_config)
        self._install_draft_auxiliary(
            self._v4_args.dim,
            self._dspark_target_layer_ids,
            self._v4_args.vocab_size,
            self._dspark_markov_rank,
            self._v4_args.n_layers - 1,
            device=load_config.device,
        )

    def create_backbone(self, args, load_config):
        args.n_mtp_layers = args.n_hash_layers = 0
        args.commit_only = self._commit_only_prefill
        if any(args.compress_ratios[: args.n_layers]) or any(
            args.v41_config.get(name, ())
            for name in (
                "kv_source_layer_ids",
                "index_source_layer_ids",
                "engram_layer_ids",
            )
        ):
            raise ValueError(
                "V4.1 DSpARK requires independent SWA draft layers without target sources/Engram"
            )

        class DraftTransformer(V41DraftGlobalsMixin, runtime_backbone_class()):
            pass

        return DraftTransformer(
            args,
            checkpoint_path=self._owned_checkpoint_path,
            layer_prefix="mtp",
            global_prefix=f"mtp.{args.n_layers - 1}.",
            head_dtype=(
                torch.float32 if self.config.enable_fp32_lm_head else torch.bfloat16
            ),
        )
