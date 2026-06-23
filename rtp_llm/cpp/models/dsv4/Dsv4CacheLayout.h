#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>

namespace rtp_llm {

// FP8 KV slot byte sizes mirror the canonical fp8_model1_mla layout written
// by Python (rtp_llm/models_py/modules/dsv4/fp8/_compressor_vllm_triton.py).
// 584B = 448 fp8 NoPE + 64 bf16 RoPE + 8 UE8M0 scales; 132B = 128 fp8 + 4 fp32
// scale. FlashMLA SM100 sparse_attn (head64 instantiations) hard-requires
// ``k_cache.stride(0) % TMA_K_STRIDE == 0``; TMA_K_STRIDE for the FP8
// path is 576 bytes (= 9 cachelines x 64B). Natural 256 * 584 = 149504 fails
// (149504 % 576 = 320), so block_size_bytes() returns the physical padded stride.
inline constexpr uint32_t DSV4_FP8_KV_ENTRY_BYTES            = 584;
inline constexpr uint32_t DSV4_FP8_INDEXER_ENTRY_BYTES       = 132;
inline constexpr size_t   DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES = 576;
inline constexpr uint32_t DSV4_SWA_WINDOW_ENTRIES            = 128;

inline size_t alignDsv4Fp8KvBlockBytes(size_t natural, size_t extra_multiple = 1) {
    const size_t align = std::lcm(DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES, std::max<size_t>(extra_multiple, 1));
    return ((natural + align - 1) / align) * align;
}

}  // namespace rtp_llm
