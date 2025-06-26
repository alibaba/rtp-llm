import torch
import torch.nn as nn
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.norm import RMSNorm
from torch import dtype as _dtype
from typing import Optional, Dict, TypedDict, Tuple, Any
from typing_extensions import Unpack
from rtp_llm.utils.model_weight import W
from rtp_llm.models_py.modules.linear import Linear

class AttentionKwargs(TypedDict, total=False):
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    attn_params: Any
