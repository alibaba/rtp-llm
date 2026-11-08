"""Framework DSpark descriptors; no private checkpoint reload or allocator."""

from rtp_llm.models.deepseek_v4 import DeepSeekV4Weight
from rtp_llm.models.deepseek_v41 import DeepSeekV41DSparkWeight
from rtp_llm.models_py.modules.dsv41.ppu.model_weights import _ppu_descriptors
from rtp_llm.utils.model_weight import W


class DeepSeekV41PPUDSparkWeight(DeepSeekV41DSparkWeight):
    def _get_hf_layer_weight_info(self, layer_id):
        return _ppu_descriptors(super()._get_hf_layer_weight_info(layer_id))

    def _get_weight_info(self):
        info = super()._get_weight_info()
        info.weights = _ppu_descriptors(info.weights)
        return info

    def get_weight_info(self):
        # Use the same framework normalization/quantization pipeline. The
        # public commit-only filter names composite FP8 weights; our raw
        # PPU descriptors have an explicit scale sibling which must survive.
        info = DeepSeekV4Weight.get_weight_info(self)
        if self.prefill_commit_only:
            layers = {W.v4_attn_wkv_w, W.v4_attn_wkv_s, W.v4_attn_kv_norm}
            globals_ = {
                W.embedding, W.lm_head, W.v4_dspark_main_norm,
                W.v4_dspark_main_proj_w, W.v4_dspark_main_proj_s,
            }
            info.layer_weights = [[w for w in row if w.name in layers] for row in info.layer_weights]
            info.weights = [w for w in info.weights if w.name in globals_]
        return info
