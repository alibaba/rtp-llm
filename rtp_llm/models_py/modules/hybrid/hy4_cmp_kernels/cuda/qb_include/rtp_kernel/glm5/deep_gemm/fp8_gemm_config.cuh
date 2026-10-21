#pragma once

#include <cstdint>

namespace rtp_kernel::glm5::deep_gemm_config {

constexpr int kSm100SharedCapacityBytes = 232448;

constexpr int align_up(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

constexpr int fixed_shared_bytes(
        int block_m, int block_n, int swizzle_cd, int output_element_bytes,
        bool swap_ab, int tma_store_stages = 2) {
    constexpr int kEpilogueStages = 2;
    const int store_m = swap_ab ? 16 : (block_m < 128 ? block_m : 128);
    const int store_n =
        swap_ab ? block_n : swizzle_cd / output_element_bytes;
    const int cd =
        tma_store_stages * store_m * store_n * output_element_bytes;
    const int epilogue_barriers =
        2 * kEpilogueStages * sizeof(std::uint64_t);
    return cd + epilogue_barriers + sizeof(std::uint32_t);
}

constexpr int shared_bytes_per_stage(
        int block_m, int block_n, int load_block_m, int load_block_n) {
    constexpr int kBlockK = 128;
    constexpr int kScaleTile = 128;
    constexpr int kPipelineBarriers = 3;
    const int ab = (load_block_m + load_block_n) * kBlockK;
    const int sf =
        (align_up(block_m, kScaleTile) + align_up(block_n, kScaleTile)) *
        sizeof(std::uint32_t);
    return ab + sf + kPipelineBarriers * sizeof(std::uint64_t);
}

constexpr int shared_bytes(
        int block_m, int block_n, int load_block_m, int load_block_n,
        int stages, int swizzle_cd,
        int output_element_bytes, bool swap_ab,
        int tma_store_stages = 2) {
    return align_up(
        fixed_shared_bytes(
            block_m, block_n, swizzle_cd, output_element_bytes, swap_ab,
            tma_store_stages) +
            stages * shared_bytes_per_stage(
                block_m, block_n, load_block_m, load_block_n),
        16);
}

constexpr int max_stages(
        int block_m, int block_n, int load_block_m, int load_block_n,
        int swizzle_cd, int output_element_bytes,
        bool swap_ab,
        int extra_shared_bytes, int shared_budget) {
    for (int stages = 32; stages > 0; --stages) {
        if (shared_bytes(
                block_m, block_n, load_block_m, load_block_n,
                stages, swizzle_cd,
                output_element_bytes, swap_ab) + extra_shared_bytes <=
            shared_budget) {
            return stages;
        }
    }
    return 0;
}

}  // namespace rtp_kernel::glm5::deep_gemm_config
