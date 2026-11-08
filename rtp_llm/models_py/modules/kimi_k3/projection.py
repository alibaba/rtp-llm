"""Bind K3.1's selected NVFP4 GEMMs; keep ordinary K3 FP8/BF16 dispatch."""

import os
import json
import logging

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
            f"{name}: packed NVFP4 projection requires KDA_NVFP4/DSA_NVFP4"
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


def bind_projection_evidence(attention, config, parallelism, layer):
    """Record actual instances and first completed forwards, without tensor reads."""
    if os.environ.get("KIMI_K3_SMOKE_EVIDENCE") != "1":
        return
    kind = "kda" if hasattr(attention, "f_b") else "dsa"
    names = (
        (("input", "input"), ("forget", "f_b"), ("output", "output"))
        if kind == "kda"
        else (("input", "input"), ("query", "q_b"), ("output", "output"))
    )
    for label, attribute in names:
        projection = getattr(attention, attribute)
        linear = type(projection).__name__
        weight_dtype = str(projection.weight.dtype)
        operand_precision = {
            ("CudaFp4GEMMLinear", "torch.uint8"): "NVFP4",
            ("CudaFp8GEMMLinear", "torch.float8_e4m3fn"): "FP8",
            ("CudaF16Linear", "torch.bfloat16"): "BF16",
        }.get((linear, weight_dtype), "unknown")
        sources = [("fused", projection)]
        fragments = ["fused"]
        if (
            kind == "dsa"
            and label == "input"
            and getattr(projection, "dynamic_input_scale", False)
            and hasattr(attention, "natural_q_gate_input")
        ):
            fragments = ["q_gate", "kv"]
            sources += [
                ("q_gate", attention.natural_q_gate_input.nvfp4),
                ("kv", attention.natural_kv_input),
            ]
        row = dict(
            model="draft" if config.is_mtp else "target",
            rank=parallelism.local_rank,
            layer=layer,
            module=f"{kind}.{label}",
            linear=linear,
            weight_dtype=weight_dtype,
            weight_precision=operand_precision,
            activation_precision=operand_precision,
            kda_nvfp4=bool(getattr(config, "kda_nvfp4", False)),
            dsa_nvfp4=bool(getattr(config, "dsa_nvfp4", False)),
            execution_fragments=fragments,
            activation_scheme=(
                "dynamic"
                if getattr(projection, "dynamic_input_scale", False)
                else "native"
            ),
        )
        logging.info(
            "[K31_PROJECTION_EVENT] %s", json.dumps(dict(row, event="initialized"))
        )
        if (config.is_mtp and layer == 0 and label == "output"
                and getattr(projection.weight, "is_cuda", False)):
            device = projection.weight.device
            free, total = torch.cuda.mem_get_info(device)
            stats = torch.cuda.memory_stats(device)
            logging.info(
                "[K31_CUDA_MEMORY] %s",
                json.dumps(dict(
                    phase="before_engine_cache", rank=parallelism.local_rank,
                    allocated=stats["allocated_bytes.all.current"],
                    reserved=stats["reserved_bytes.all.current"],
                    inactive_split=stats["inactive_split_bytes.all.current"],
                    device_free=free, device_total=total,
                )),
            )
        for fragment, operator in sources:
            if not getattr(operator, "dynamic_input_scale", False):
                continue
            from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import Fp4Activation

            def completed(
                module, args, output, *, receipt=row, part=fragment, seen=set()
            ):
                value = args[0]
                packed = isinstance(value, Fp4Activation)
                source = value.values if packed else value
                if seen or source.shape[0] == 0:
                    return
                seen.add(True)
                logging.info(
                    "[K31_PROJECTION_EVENT] %s",
                    json.dumps(
                        dict(
                            receipt,
                            event="executed",
                            fragment=part,
                            linear=type(module).__name__,
                            weight_dtype=str(module.weight.dtype),
                            input_dtype=str(source.dtype),
                            prequantized_input=packed,
                            output_dtype=str(output.dtype),
                            shape=list(output.shape),
                            cuda_graph_capture=torch.cuda.is_current_stream_capturing(),
                        )
                    ),
                )

            operator.register_forward_hook(completed)
