from rtp_llm.utils.import_util import has_internal_source

from .base_multimodal_mixin import (
    BaseMultiModalDeployWeightInfo,
    BaseMultiModalMixin,
    BaseVitWeights,
)
from .chatglm4v.chatglm4v_mixin import ChatGlmV4VisionMixin
from .deepseek_vl2.deepseek_vl2_mixin import DeepSeekVLV2Mixin
from .kimi_k25.kimi_k25_mixin import KimiK25Mixin
from .llava.llava_mixin import LlavaMixin
from .qwen2_5_vl.qwen2_5_vl_mixin import Qwen2_5_VLMixin
from .qwen2_audio.qwen2_audio_mixin import Qwen2_AudioMixin
from .qwen2_vl.qwen2_vl_mixin import Qwen2_VLMixin
# Qwen3.5's Vision implementation requires a newer Transformers API than the
# CUDA13 serving image ships.  Registering that unrelated multimodal model must
# not stop a text-only model from starting: without this guard every K3 rank
# dies at import with "No module named 'transformers.modeling_layers'".
# Deployments that do have the newer dependency keep the registration.
try:
    from .qwen3_5_moe.qwen3_5_moe_mixin import Qwen3_5MoeMixin
except ImportError as exc:
    if exc.name not in {"transformers", "transformers.modeling_layers"}:
        raise
try:
    from .qwen3_vl_mixin import Qwen3_VLMixin
except ImportError as exc:
    if exc.name != "transformers":
        raise
from .qwen_vl.qwen_vl_mixin import QwenVLMixin

if has_internal_source():
    import internal_source.rtp_llm.multimodal_mixins.internal_init
