from rtp_llm.ops import DeviceExporter, DeviceType, get_device

device_type = get_device().get_device_type()

if device_type == DeviceType.ROCm:
    from rtp_llm.models_py.modules.rocm.norm import RMSNorm
else:
    from rtp_llm.models_py.modules.norm import RMSNorm

from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.models_py.modules.mlp import DenseMLP
from rtp_llm.models_py.modules.norm import AddBiasResLayerNorm, LayerNorm, RMSNormTorch
