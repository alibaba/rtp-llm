#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"

namespace rtp_llm {
namespace test {

namespace {

MLAKVCacheSpec makeMLASpec(DataType dtype, bool use_aiter_fp8_layout) {
    MLAKVCacheSpec spec;
    spec.type                 = KVCacheSpecType::MultiHeadLatentAttention;
    spec.dtype                = dtype;
    spec.layer_num            = 1;
    spec.local_head_num_kv    = 1;
    spec.seq_size_per_block   = 16;
    spec.kv_lora_rank         = 512;
    spec.rope_head_dim        = 64;
    spec.use_aiter_fp8_layout = use_aiter_fp8_layout;
    return spec;
}

}  // namespace

TEST(MLAKVCacheSpecTest, ComputesBF16BlockSize) {
    auto spec = makeMLASpec(DataType::TYPE_BF16, /*use_aiter_fp8_layout=*/false);

    const size_t expected_block_size = (512 + 64) * 16;
    EXPECT_EQ(spec.block_size(), expected_block_size);
    EXPECT_EQ(spec.block_size_bytes(), expected_block_size * 2);
}

TEST(MLAKVCacheSpecTest, ComputesNativeFP8BlockSize) {
    auto spec = makeMLASpec(DataType::TYPE_FP8_E4M3, /*use_aiter_fp8_layout=*/false);

    // Native RTP FP8 MLA layout stores 512 fp8 NoPE bytes, 4 fp32 scales,
    // and 64 bf16 RoPE values per token.
    const size_t expected_block_size = (512 + 4 * 4 + 64 * 2) * 16;
    EXPECT_EQ(spec.block_size(), expected_block_size);
    EXPECT_EQ(spec.block_size_bytes(), expected_block_size);
}

TEST(MLAKVCacheSpecTest, ComputesAiterFP8BlockSize) {
    auto spec = makeMLASpec(DataType::TYPE_FP8_E4M3, /*use_aiter_fp8_layout=*/true);

    // AITER FP8 MLA layout keeps only the packed 512 fp8 NoPE + 64 fp8 RoPE
    // bytes in KV cache; scales are passed separately.
    const size_t expected_block_size = (512 + 64) * 16;
    EXPECT_EQ(spec.block_size(), expected_block_size);
    EXPECT_EQ(spec.block_size_bytes(), expected_block_size);
}

}  // namespace test
}  // namespace rtp_llm
