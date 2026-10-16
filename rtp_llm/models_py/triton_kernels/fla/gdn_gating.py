import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.utils import is_amd
from rtp_llm.models_py.utils.prefill_input_log import trace_triton

# adpated from vllm: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    stride_ab: tl.constexpr,  # stride for a and b batch dimension
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
    STABLE_SOFTPLUS: tl.constexpr = False,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)

    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    ba_off = i_b * seq_len * stride_ab + i_s * stride_ab + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + ba_off, mask=mask)
    blk_b = tl.load(b + ba_off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    if STABLE_SOFTPLUS and beta == 1.0:
        # Align FP32 softplus with vLLM's FlashInfer prefill operation order.
        # Decode/native callers keep stable_softplus=False.
        softplus_x = tl.where(
            x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x))
        )
        softplus_x = tl.where(x <= threshold, softplus_x, x)
    else:
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    *,
    beta_fp32: bool = False,
    stable_softplus: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused log-g and sigmoid beta for Gated Delta Net.

    g is always FP32. FlashInfer prefill requests FP32 beta directly;
    other callers retain the existing CUDA b.dtype / ROCm FP32 default.
    """
    batch, num_heads = a.shape
    seq_len = 1

    # Get strides for a and b to support non-contiguous tensors
    # Expected shape: [batch, num_heads]
    stride_ab = a.stride(0)  # batch dimension stride
    stride_ah = a.stride(1)  # heads dimension stride
    stride_bb = b.stride(0)
    stride_bh = b.stride(1)

    assert (
        stride_ah == 1 and stride_bh == 1
    ), f"stride_ah must be 1 and stride_bh must be 1"
    assert stride_ab == stride_bb, f"stride_ab must be equal to stride_bb"

    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    # Align FlashInfer prefill beta with vLLM: store the sigmoid result
    # directly in FP32 without an intermediate BF16 rounding. Native CUDA
    # and ROCm defaults keep their existing dtype contract.
    beta_dtype = torch.float32 if is_amd or beta_fp32 else b.dtype
    beta_output = torch.empty(1, batch, num_heads, dtype=beta_dtype, device=b.device)
    if batch == 0:
        return g, beta_output
    trace_triton(
        "gdn_gating:fused_gdn_gating_kernel[grid]",
        fused_gdn_gating_kernel,
        grid,
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        stride_ab,
        beta,
        threshold,
        8,
        stable_softplus,
        num_warps=1,
    )
    return g, beta_output
