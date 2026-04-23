"""Weight info for Kimi-K2.5 (text core only).

The text path reuses DeepSeekV2Weight but Kimi-K2.5 ships its text model
under the `language_model.model.layers.{i}.*` HF prefix (with the
`language_model.` wrapper). We rewrite the checkpoint names produced by
DeepSeekV2Weight to add this prefix.

The ViT / patchmerger weight info has moved out — it is now declared by
``KimiK25VitWeight`` in :mod:`rtp_llm.multimodal.multimodal_mixins.kimi_k25.kimi_k25_mixin`
and loaded through the unified multimodal-mixin factory path.
"""

from rtp_llm.model_loader.weight_module import CompositeWeight, WeightModule
from rtp_llm.models.deepseek_v2 import DeepSeekV2Weight

_LANG_PREFIX = "language_model."


def _rewrite_ckpt_names(weight: WeightModule) -> None:
    """Recursively prefix every CkptWeightInfo.name in `weight` with
    `language_model.` so DeepSeekV2Weight's hard-coded `model.*` /
    `lm_head.*` paths match Kimi-K2.5's `language_model.model.*` /
    `language_model.lm_head.*` checkpoint layout."""
    weights = getattr(weight, "weights", None)
    if weights:
        for ckpt in weights:
            if not ckpt.name.startswith(_LANG_PREFIX):
                ckpt.name = _LANG_PREFIX + ckpt.name
    if isinstance(weight, CompositeWeight):
        for sub in weight.sub_weights.values():
            _rewrite_ckpt_names(sub)


class KimiK25Weight(DeepSeekV2Weight):
    def _process_meta(self, meta_dict, weight_keys):
        # Kimi-K2.5 keys are prefixed with `language_model.`; strip it
        # before delegating to DeepSeekV2's HF-style probes.
        stripped_keys = {
            (k[len(_LANG_PREFIX) :] if k.startswith(_LANG_PREFIX) else k)
            for k in weight_keys
        }
        super()._process_meta(meta_dict, stripped_keys)

    def _get_weight_info(self):
        info = super()._get_weight_info()
        for w in info.weights:
            _rewrite_ckpt_names(w)
        for layer in info.layer_weights:
            if isinstance(layer, list):
                for w in layer:
                    _rewrite_ckpt_names(w)
            else:
                _rewrite_ckpt_names(layer)
        return info
