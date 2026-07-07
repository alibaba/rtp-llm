"""MiniMax-M3 top-level container for the new loader.

Mirrors ``new_models/qwen3_vl_moe/model.py``.  The VL model composes four
sub-modules whose attribute names are chosen so that HF checkpoint keys,
after ``WEIGHTS_MAPPER`` prefix rewriting, land on the correct child:

  vision_tower          -> MiniMaxVisionTransformer  (CLIP ViT-H/14, 32 layers)
  multi_modal_projector -> MiniMaxProjector           (2-layer GELU MLP)
  patch_merge_mlp       -> MiniMaxPatchMergeMLP       (2-layer GELU MLP)
  language_model        -> MiniMaxM3ForCausalLM       (MoE LLM backbone)
"""

import logging
from typing import Any

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.minimax_m3.language import MiniMaxM3ForCausalLM
from rtp_llm.models_py.new_models.minimax_m3.vision import (
    MiniMaxPatchMergeMLP,
    MiniMaxProjector,
    MiniMaxVisionTransformer,
)
from rtp_llm.models_py.weight_mapper import WeightsMapper

logger = logging.getLogger(__name__)


class MiniMaxM3VLForConditionalGeneration(RtpModule):

    # HF MiniMax-M3-VL checkpoint key layout:
    #   language_model.model.*    -> language_model.*   (strip extra "model.")
    #   language_model.lm_head.*  -> language_model.lm_head.*  (pass-through)
    #   vision_tower.*            -> vision_tower.*      (pass-through)
    #   multi_modal_projector.*   -> multi_modal_projector.*  (pass-through)
    #   patch_merge_mlp.*         -> patch_merge_mlp.*   (pass-through)
    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={
            "language_model.model.": "language_model.",
            "language_model.lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__()
        self.model_config = model_config
        self.load_config = load_config

        self.vision_tower = MiniMaxVisionTransformer(
            model_config=model_config, load_config=load_config
        )
        self.multi_modal_projector = MiniMaxProjector(
            model_config=model_config, load_config=load_config
        )
        self.patch_merge_mlp = MiniMaxPatchMergeMLP(
            model_config=model_config, load_config=load_config
        )
        self.language_model = MiniMaxM3ForCausalLM(
            model_config=model_config, load_config=load_config
        )

    def initialize(self, init_resource) -> bool:
        return self.language_model.initialize(init_resource)

    def prepare_fmha_impl(self, inputs, is_cuda_graph: bool = False):
        return self.language_model.prepare_fmha_impl(inputs, is_cuda_graph)

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights

        mapped_iter = self.WEIGHTS_MAPPER.apply(weights_iter)
        dropped = []
        count = 0
        for full_name, tensor in mapped_iter:
            count += 1
            if "." not in full_name:
                dropped.append(full_name)
                continue
            prefix, rest = full_name.split(".", 1)
            child = self._get_child_module(prefix)
            if child is not None and hasattr(child, "load_weights"):
                child.load_weights({rest: tensor})
            else:
                dropped.append(full_name)

        logger.info(
            "[MiniMaxM3VLForConditionalGeneration] streamed %d mapped weights",
            count,
        )
        if dropped:
            sample = dropped[:10]
            more = (
                f" (+{len(dropped) - len(sample)} more)"
                if len(dropped) > len(sample)
                else ""
            )
            logger.warning(
                "[MiniMaxM3VLForConditionalGeneration] dropped %d weights: %s%s",
                len(dropped),
                sample,
                more,
            )

    def forward(self, inputs, fmha_impl: Any = None):
        return self.language_model.forward(inputs, fmha_impl=fmha_impl)
