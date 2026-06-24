#pragma once

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "rtp_llm/models_py/bindings/cuda/kernels/fused_layernorm_fp8_group_quant.h"

namespace rtp_llm {

// Fused AddBiasRes + LayerNorm + FP8 Per-Group Quantization.
//
// Inputs (in-place semantics matching fused_add_layernorm):
//   input:    [tokens, hidden_dim] BF16/FP16 -- hidden states
//   residual: [tokens, hidden_dim] BF16/FP16 -- residual for skip connection
//   bias:     [hidden_dim] or empty tensor
//   weight:   [hidden_dim] -- LayerNorm gamma
//   beta:     [hidden_dim] -- LayerNorm beta (or empty)
//   eps:      float
//   group_size: int (e.g. 128)
//
// After call:
//   residual is overwritten with (input + residual + bias) -- pre-norm value
//   Returns (fp8_output, group_scales):
//     fp8_output:    [tokens, hidden_dim] float8_e4m3fn
//     group_scales:  column-major [tokens, hidden_dim/group_size] float32
//                    with stride (1, tokens) matching CUTLASS SM120 expectation
inline std::tuple<torch::Tensor, torch::Tensor>
fused_add_layernorm_fp8_group_quant(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& bias,
    torch::Tensor& weight,
    torch::Tensor& beta,
    float          eps,
    int            group_size) {

    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(residual.dim() == 2, "residual must be 2D");
    TORCH_CHECK(input.dtype() == residual.dtype(), "input and residual must have same dtype");
    TORCH_CHECK(input.dtype() == torch::kBFloat16 || input.dtype() == torch::kFloat16,
                "input must be bf16 or fp16");

    const int tokens = input.size(0);
    const int hidden_dim = input.size(1);
    TORCH_CHECK(hidden_dim % group_size == 0,
                "hidden_dim must be divisible by group_size");

    const int num_groups = hidden_dim / group_size;

    // Allocate FP8 output
    auto fp8_output = torch::empty(
        {tokens, hidden_dim},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(input.device()));

    // Allocate column-major scale output: physical shape (num_groups, tokens),
    // permuted to logical (tokens, num_groups) with stride (1, tokens).
    auto group_scales = torch::empty(
        {num_groups, tokens},
        torch::TensorOptions().dtype(torch::kFloat32).device(input.device())
    ).permute({1, 0});  // logical: (tokens, num_groups), stride: (1, tokens)

    const void* bias_ptr = (bias.numel() > 0) ? bias.data_ptr() : nullptr;
    const void* beta_ptr = (beta.numel() > 0) ? beta.data_ptr() : nullptr;

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    int dtype_flag = (input.scalar_type() == at::ScalarType::BFloat16) ? 0 : 1;

    // Matching fused_add_layernorm semantics:
    //   input  (hidden_states) -> overwritten with normed output (BF16)
    //   residual               -> overwritten with pre-norm value (input + residual + bias)
    invokeAddBiasResLayerNormFP8GroupQuant(
        input.data_ptr(),         // normed_output: write normed BF16 here (in-place to input)
        residual.data_ptr(),      // residual_output: write pre-norm value (in-place)
        fp8_output.data_ptr(),
        static_cast<float*>(group_scales.data_ptr()),
        input.data_ptr(),
        residual.data_ptr(),
        bias_ptr,
        weight.data_ptr(),
        beta_ptr,
        eps,
        tokens,
        hidden_dim,
        group_size,
        tokens,                   // scale_stride = tokens for column-major
        dtype_flag,
        stream);

    return std::make_tuple(fp8_output, group_scales);
}

}  // namespace rtp_llm
