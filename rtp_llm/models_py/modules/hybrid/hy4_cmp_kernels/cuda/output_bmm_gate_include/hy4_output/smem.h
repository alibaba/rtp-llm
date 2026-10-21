#pragma once

#include <cstdint>

namespace hy4_output {

// Shared-memory layout for the GLM5 swap-AB BF16 core: M16 store tiles,
// then `stages` pairs of A/B load tiles, followed by pipeline metadata.
constexpr int bf16_gemm_smem_bytes(int block_m, int block_n, int block_k,
                                   int stages, int cd_element_bytes = 2,
                                   int multicast = 2,
                                   bool multicast_on_a = true,
                                   int store_stages = 2) {
    constexpr int kStoreBlockM = 16;
    constexpr int kPipelineMetadataBytes =
        32 * 8 * 3 + 2 * 8 * 2 + 8 + sizeof(std::uint32_t);
    const int cd_bytes =
        store_stages * kStoreBlockM * block_n * cd_element_bytes;
    const int load_block_m = block_m / (multicast_on_a ? multicast : 1);
    const int load_block_n = block_n / (multicast_on_a ? 1 : multicast);
    const int stage_bytes =
        (load_block_m + load_block_n) * block_k * sizeof(std::uint16_t);
    return cd_bytes + stages * stage_bytes + kPipelineMetadataBytes;
}

}  // namespace hy4_output
