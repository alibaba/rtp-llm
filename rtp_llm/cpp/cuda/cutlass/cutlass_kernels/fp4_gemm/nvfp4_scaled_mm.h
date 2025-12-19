#pragma once

#include <torch/all.h>

void cutlass_scaled_fp4_mm_sm100a_sm120a(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha);

void scaled_fp4_quant_sm100a_sm120a(
    torch::Tensor const& output,
    torch::Tensor const& input,
    torch::Tensor const& output_sf,
    torch::Tensor const& input_sf);
