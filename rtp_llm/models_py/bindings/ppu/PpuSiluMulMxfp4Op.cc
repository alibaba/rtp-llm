// Copyright 2026 Alibaba Group Holding Limited.
// SPDX-License-Identifier: Apache-2.0
#include "rtp_llm/models_py/bindings/ppu/PpuSiluMulMxfp4Op.h"
#ifdef USE_PPU
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/ppu/kernels/ppu_silu_mul_mxfp4.h"
#include <limits>
namespace rtp_llm {
std::tuple<torch::Tensor, torch::Tensor> PpuSiluAndMulPostQuantMxfp4(
    torch::Tensor gate_up, double swiglu_limit, bool apply_swiglu_limit) {
    TORCH_CHECK(gate_up.is_cuda(), "gate_up must be a PPU tensor");
    TORCH_CHECK(gate_up.scalar_type() == torch::kBFloat16, "gate_up must be BF16");
    TORCH_CHECK(gate_up.dim() == 2, "gate_up must have shape [N, 2H]");
    TORCH_CHECK(gate_up.is_contiguous(), "gate_up must be contiguous");
    const int64_t num_tokens = gate_up.size(0);
    const int64_t two_hidden = gate_up.size(1);
    // Each complete 16-element group writes one aligned uint2 to the packed
    // output. Keep every row aligned, including in the scalar-tail kernel.
    TORCH_CHECK(two_hidden > 0 && two_hidden % 32 == 0,
                "2H must be positive and H must be a multiple of 16");
    const int64_t hidden = two_hidden / 2;
    TORCH_CHECK(hidden <= std::numeric_limits<int>::max() &&
                    num_tokens <= std::numeric_limits<int>::max(),
                "shape exceeds PPU launcher limits");
    const int block_n = hidden % 512 == 0 ? 512 : (hidden % 256 == 0 ? 256 : 128);
    const int64_t hidden_padded = (hidden + block_n - 1) / block_n * block_n;
    const int64_t scale_alloc = hidden_padded / 64;
    const int64_t scale_valid = (hidden + 63) / 64;
    auto packed = torch::empty({num_tokens, hidden / 2},
                               gate_up.options().dtype(torch::kUInt8));
    auto scale_storage = torch::empty({scale_alloc, num_tokens},
                                      gate_up.options().dtype(torch::kUInt16));
    if (num_tokens != 0) {
        StreamType stream = GET_CURRENT_STREAM();
        invokePpuSiluMulMxfp4(gate_up.data_ptr(),
                              packed.data_ptr<uint8_t>(),
                              reinterpret_cast<uint8_t*>(scale_storage.data_ptr<uint16_t>()),
                              gate_up.stride(0),
                              packed.stride(0) * packed.element_size(),
                              scale_storage.stride(0) * scale_storage.element_size(),
                              scale_storage.stride(1) * scale_storage.element_size(),
                              static_cast<int>(hidden),
                              static_cast<int>(num_tokens),
                              apply_swiglu_limit,
                              static_cast<float>(swiglu_limit),
                              stream);
    }
    auto scale = scale_storage.slice(0, 0, scale_valid).transpose(0, 1);
    return {packed, scale};
}
}  // namespace rtp_llm
#endif
