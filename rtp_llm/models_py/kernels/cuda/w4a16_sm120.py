"""Tensor API for the SM120 INT4/group8 GEMM."""

import torch

_HADAMARD_BLOCK = 128


def support(target, k: int = None) -> bool:
    """W4A16 capability entry.

    support(n, k): weight shape; returns False for shapes that must fall back
    to the default path. n % 256: the GEMM tiles N by 64/128/256 and weight
    loads are not N-masked. k % 128: Hadamard rotate block.
    support(device) / support(config): hard requirements; raise ValueError
    when unsatisfied.
    """
    if k is not None:
        return target % 256 == 0 and k % _HADAMARD_BLOCK == 0
    if isinstance(target, torch.device):
        if target.type != "cuda" or torch.version.hip is not None:
            raise ValueError("SM120 W4A16 requires an NVIDIA CUDA device")
        if not torch.cuda.is_available():
            raise ValueError("SM120 W4A16 requires an available CUDA device")
        capability = torch.cuda.get_device_capability(target)
        if capability != (12, 0):
            raise ValueError(
                f"SM120 W4A16 requires compute capability (12, 0), "
                f"got {capability} on {target}"
            )
        return True
    if target.moe_style != 0 or target.expert_num > 0:
        raise ValueError("SM120 W4A16 dense FFN does not support MoE models")
    if target.quant_config is not None:
        if target.quant_config.is_quanted():
            raise ValueError(
                "SM120 W4A16 dense FFN requires unquantized checkpoint weights; "
                "pre-quantized checkpoints are not supported"
            )
        if target.quant_config.get_method() not in {
            "FP8",
            "FP8_DYNAMIC_PER_TENSOR",
            "FP8_PER_BLOCK",
            "FP8_PER_CHANNEL_COMPRESSED",
            "FP8_PER_CHANNEL_QUARK",
        }:
            raise ValueError(
                "SM120 W4A16 dense FFN only supports online FP8 quantization"
            )
    if target.lora_infos:
        raise ValueError("SM120 W4A16 dense FFN does not support LoRA")
    if target.compute_dtype != torch.bfloat16:
        raise ValueError("SM120 W4A16 dense FFN requires BF16 compute dtype")
    return True


def transform(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    output_size, input_size = weight.shape
    packed = torch.empty(
        (input_size // 16, output_size // 2, 4),
        dtype=torch.int32,
        device=weight.device,
    )
    scales = torch.empty(
        (input_size // 32, output_size, 4), dtype=torch.uint8, device=weight.device
    )
    scratch = torch.empty(weight.shape, dtype=torch.uint8, device=weight.device)
    rtp_llm_ops.w4a16_sm120_transform(weight, packed, scales, scratch)
    return packed, scales


def gemm(
    inputs: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    split_k: int = 0,
) -> torch.Tensor:
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    input_size, output_size = packed.shape[0] * 16, packed.shape[1] * 2
    if out is None:
        out = torch.empty(
            (inputs.shape[0], output_size), dtype=inputs.dtype, device=inputs.device
        )
    rtp_llm_ops.w4a16_sm120_gemm(
        inputs, packed, scales, out, output_size, input_size, split_k
    )
    return out


def rotate(inputs: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    signed = inputs * signs
    return rtp_llm_ops.w4a16_sm120_hadamard(
        signed.reshape(-1, _HADAMARD_BLOCK), _HADAMARD_BLOCK**-0.5
    ).reshape(inputs.shape)


@torch.inference_mode()
def quantize_weight(weight: torch.Tensor, seed: int):
    output_size, input_size = weight.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (input_size,), generator=generator)
    signs = (signs * 2 - 1).to(device=weight.device, dtype=weight.dtype)
    packed = torch.empty(
        (input_size // 16, output_size // 2, 4),
        dtype=torch.int32,
        device=weight.device,
    )
    scales = torch.empty(
        (input_size // 32, output_size, 4),
        dtype=torch.uint8,
        device=weight.device,
    )
    for start in range(0, output_size, 256):
        end = min(start + 256, output_size)
        rotated = rotate(weight[start:end].float(), signs).to(weight.dtype)
        chunk_packed, chunk_scales = transform(rotated.contiguous())
        packed[:, start // 2 : end // 2].copy_(chunk_packed)
        scales[:, start:end].copy_(chunk_scales)
    return packed, scales, signs
