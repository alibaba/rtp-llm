"""Bind K3.1's selected NVFP4 GEMMs; keep ordinary K3 FP8/BF16 dispatch."""

import os

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig, ModelOptFp4Config
from rtp_llm.models.kimi_k3.nvfp4_config import uses_fp8_output
from rtp_llm.models_py.modules.factory.linear import LinearFactory


def create_kimi_projection(config, weights, name, scale_key=None, hardware=None):
    weight = weights[name]
    # Import the NVFP4 loader only for K3.1; ordinary K3 and draft retain their
    # existing dependency and dtype policy.
    modules = getattr(config, "k31_nvfp4_modules", ())
    selected = False
    if modules:
        from rtp_llm.models.kimi_k3.nvfp4_weight import (
            ATTENTION_PROJECTION_MODULES,
            nvfp4_scale_keys,
        )

        module = ATTENTION_PROJECTION_MODULES.get(name)
        if uses_fp8_output(module, modules) and weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"{name}: attention output requires grouped FP8 weights")
        selected = module in modules
    if selected:
        if getattr(config, "is_mtp", False):
            raise ValueError("NVFP4 cannot be used by the K3.1 MTP draft")
        if weight.dtype != torch.uint8:
            raise ValueError(
                f"{name}: selected NVFP4 projection requires packed E2M1 weight codes"
            )
        if not weight.is_cuda or torch.cuda.get_device_capability(
            weight.device
        ) not in ((10, 0), (10, 3)):
            raise ValueError("K3.1 NVFP4 requires CUDA SM100/SM103 hardware")
        if torch.version.cuda is None or int(torch.version.cuda.split(".")[0]) < 13:
            raise ValueError("K3.1 NVFP4 requires the repository's CUDA13 dependencies")
        if os.environ.get("RTP_LLM_FP4_GEMM_BACKEND", "cutlass") != "cutlass":
            raise ValueError("K3.1 NVFP4 requires RTP_LLM_FP4_GEMM_BACKEND=cutlass")
        local, outer = nvfp4_scale_keys(name)
        if any(key not in weights for key in (local, outer)):
            raise ValueError(f"{name}: missing NVFP4 local/global scales")
        logical_n_key = name + ".nvfp4.logical_n"
        if logical_n_key not in weights:
            raise ValueError(f"{name}: missing NVFP4 logical output size")
        projection = LinearFactory.create_linear_from_weights(
            weights,
            name,
            scale_key=local,
            weight_scale_2_key=outer,
            quant_config=ModelOptFp4Config(
                bits=4, group_size=16, is_quanted=False, activation_scheme="dynamic"
            ),
            hw_kernel_config=hardware,
        )
        projection.logical_output_size = int(weights[logical_n_key].item())
        return projection
    if weight.dtype == torch.uint8:
        raise ValueError(
            f"{name}: packed NVFP4 projection requires explicit module quantization"
        )
    if scale_key is None:
        return LinearFactory.create_linear_from_weights(
            weights, name, hw_kernel_config=hardware
        )
    fp8 = weight.dtype == torch.float8_e4m3fn
    return LinearFactory.create_linear(
        weight,
        bias=None,
        weight_scales=weights[scale_key] if fp8 else None,
        quant_config=Fp8BlockWiseQuantConfig() if fp8 else None,
        hw_kernel_config=hardware,
    )
