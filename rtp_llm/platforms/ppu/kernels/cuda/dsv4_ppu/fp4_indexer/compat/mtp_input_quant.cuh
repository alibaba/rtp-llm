// Copyright (c) 2024 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Adapt the RMSNorm reduction from flashinfer/norm.cuh to two MTP inputs.
// Mask + two independent RMSNorm/FP8 producers for the Flash MTP input.
// Keep the installed FlashInfer 8-element/thread, 16-warp RMS reduction,
// BF16 norm boundary, and compute-ops v2 block-128 quantization arithmetic.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

namespace {
template<typename Position>
__global__ void mtp_input_quant_kernel(const bf16_t*   embedding,
                                       const bf16_t*   hidden,
                                       const Position* positions,
                                       const bf16_t*   e_weight,
                                       const bf16_t*   h_weight,
                                       fp8_e4m3_t*     e_payload,
                                       float*          e_scales,
                                       fp8_e4m3_t*     h_payload,
                                       float*          h_scales,
                                       int             tokens,
                                       float           e_eps,
                                       float           h_eps) {
    const bool  is_embedding = blockIdx.x < tokens;
    const int   row          = is_embedding ? blockIdx.x : blockIdx.x - tokens;
    const auto* input        = (is_embedding ? embedding : hidden) + int64_t{row} * 4096;
    const auto* weight       = is_embedding ? e_weight : h_weight;
    auto*       payload      = (is_embedding ? e_payload : h_payload) + int64_t{row} * 4096;
    auto*       scales       = is_embedding ? e_scales : h_scales;
    const int   rows         = is_embedding ? tokens : tokens * 4;
    const int   tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const bool  masked = is_embedding && positions[row] == 0;
    float       values[8];
    float       squares = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        values[j] = masked ? 0.f : float(input[tid * 8 + j]);
        squares   = __fmaf_rn(values[j], values[j], squares);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        squares += __shfl_xor_sync(0xffffffff, squares, offset);
    }
    __shared__ float warp_squares[16];
    if (lane == 0)
        warp_squares[warp] = squares;
    __syncthreads();
    if (warp == 0) {
        squares = lane < 16 ? warp_squares[lane] : 0.f;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            squares += __shfl_xor_sync(0xffffffff, squares, offset);
        }
        if (lane == 0)
            warp_squares[0] = squares;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(warp_squares[0] / 4096.f + (is_embedding ? e_eps : h_eps));
    float       absmax  = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        values[j] = float(bf16_t(values[j] * inv_rms * float(weight[tid * 8 + j])));
        absmax    = fmaxf(absmax, fabsf(values[j]));
    }
#pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) {
        absmax = fmaxf(absmax, __shfl_xor_sync(0xffffffff, absmax, offset, 16));
    }
    absmax                = fmaxf(absmax, 1e-4f);
    const float scale     = __fmul_rn(absmax, 1.f / 448.f);
    const float inv_scale = __fdiv_rn(448.f, absmax);
    if (tid % 16 == 0)
        scales[(tid / 16) * rows + row] = scale;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        payload[tid * 8 + j] = fp8_e4m3_t(fminf(fmaxf(values[j] * inv_scale, -448.f), 448.f));
    }
}

template<typename Position>
struct MtpInputQuant {
    static void forward(tvm::ffi::TensorView embedding,
                        tvm::ffi::TensorView hidden,
                        tvm::ffi::TensorView positions,
                        tvm::ffi::TensorView e_weight,
                        tvm::ffi::TensorView h_weight,
                        tvm::ffi::TensorView e_payload,
                        tvm::ffi::TensorView e_scales,
                        tvm::ffi::TensorView h_payload,
                        tvm::ffi::TensorView h_scales,
                        float                e_eps,
                        float                h_eps) {
        using namespace host;
        auto tokens = SymbolicSize{"tokens"};
        auto device = SymbolicDevice{};
        device.set_options<kDLCUDA>();
        TensorMatcher({tokens, 4096}).with_dtype<bf16_t>().with_device(device).verify(embedding);
        TensorMatcher({tokens, 4, 4096}).with_dtype<bf16_t>().with_device(device).verify(hidden);
        TensorMatcher({tokens}).with_dtype<Position>().with_device(device).verify(positions);
        TensorMatcher({4096}).with_dtype<bf16_t>().with_device(device).verify(e_weight);
        TensorMatcher({4096}).with_dtype<bf16_t>().with_device(device).verify(h_weight);
        auto hidden_rows = SymbolicSize{"hidden_rows"};
        hidden_rows.set_value(tokens.unwrap() * 4);
        TensorMatcher({tokens, 4096}).with_dtype<fp8_e4m3_t>().with_device(device).verify(e_payload);
        TensorMatcher({32, tokens}).with_dtype<float>().with_device(device).verify(e_scales);
        TensorMatcher({hidden_rows, 4096}).with_dtype<fp8_e4m3_t>().with_device(device).verify(h_payload);
        TensorMatcher({32, hidden_rows}).with_dtype<float>().with_device(device).verify(h_scales);
        const int n = tokens.unwrap();
        if (!n)
            return;
        LaunchKernel(dim3(n * 5), dim3(512), device.unwrap())(mtp_input_quant_kernel<Position>,
                                                              static_cast<const bf16_t*>(embedding.data_ptr()),
                                                              static_cast<const bf16_t*>(hidden.data_ptr()),
                                                              static_cast<const Position*>(positions.data_ptr()),
                                                              static_cast<const bf16_t*>(e_weight.data_ptr()),
                                                              static_cast<const bf16_t*>(h_weight.data_ptr()),
                                                              static_cast<fp8_e4m3_t*>(e_payload.data_ptr()),
                                                              static_cast<float*>(e_scales.data_ptr()),
                                                              static_cast<fp8_e4m3_t*>(h_payload.data_ptr()),
                                                              static_cast<float*>(h_scales.data_ptr()),
                                                              n,
                                                              e_eps,
                                                              h_eps);
    }
};
}  // namespace
