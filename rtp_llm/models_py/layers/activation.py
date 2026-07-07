"""共享激活算子。

新 loader 的原则:推理 forward 与旧 loader 用同一套 rtp 融合 kernel(保证效率 + 数值
一致),只在加载权重的方式上不同。这里把激活算子默认接到 rtp 融合 kernel,非 CUDA /
kernel 不可用时回退到 fp32 eager(可移植兜底)。
"""

import torch

_SILU_FALLBACK_WARNED = False


def swigluoai_and_mul(
    gate_up: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """OpenAI SwiGLU variant used by MiniMax-M3.

    ``gate_up`` is laid out as [gate, up]. The activation is
    ``gate * sigmoid(alpha * gate) * (up + 1)`` with the same clipping used by
    the GPT-OSS/MiniMax config family.
    """
    d = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :d].float(), gate_up[..., d:].float()
    gate = torch.clamp(gate, max=limit)
    up = torch.clamp(up, min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + 1.0)).to(gate_up.dtype)


def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """SiLU(gate) * up。

    Args:
        gate_up: [..., 2d] 融合张量(前半为 gate,后半为 up)。
    Returns:
        [..., d] 张量。
    CUDA/ROCm 走 rtp 融合 ``silu_and_mul``(内部 fp32,与旧 loader 一致);
    其它后端 eager 但上采 fp32,贴近融合 kernel 精度。
    """
    if gate_up.is_cuda:
        try:
            d = gate_up.shape[-1] // 2
            out = torch.empty(
                gate_up.shape[:-1] + (d,),
                dtype=gate_up.dtype,
                device=gate_up.device,
            )
            if getattr(torch.version, "hip", None) is not None:
                import aiter

                aiter.silu_and_mul(out, gate_up)
                return out

            from rtp_llm.ops.compute_ops import rtp_llm_ops

            stream_id = torch.cuda.current_stream().cuda_stream
            rtp_llm_ops.silu_and_mul(out, gate_up, stream_id)
            return out
        except Exception as e:
            global _SILU_FALLBACK_WARNED
            if not _SILU_FALLBACK_WARNED:
                _SILU_FALLBACK_WARNED = True
                import logging

                logging.getLogger(__name__).warning(
                    "[silu_and_mul] 融合 kernel 不可用,回退 fp32 eager: %s", e
                )
    d = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :d], gate_up[..., d:]
    return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate_up.dtype)
