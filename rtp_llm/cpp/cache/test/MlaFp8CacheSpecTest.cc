#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"

namespace rtp_llm {
namespace {

TEST(MlaFp8CacheSpecTest, OrdinaryE4m3Has576BytesPerToken) {
    AttentionConfigs attention{};
    attention.kv_lora_rank  = 512;
    attention.rope_head_dim = 64;
    SpecBuildContext context{};
    context.attn_config        = &attention;
    context.seq_size_per_block = 128;

    KVCacheSpecDesc ordinary{};
    ordinary.tag            = "full";
    ordinary.cache_type     = KVCacheSpecType::MultiHeadLatentAttention;
    ordinary.dtype          = DataType::TYPE_FP8_E4M3;
    ordinary.mla_fp8_e4m3   = true;
    auto fp8 = MLAKVCacheSpec::build(ordinary, context);
    EXPECT_EQ(fp8->block_size_bytes(), 128u * 576u);

    ordinary.mla_fp8_e4m3 = false;
    auto mixed = MLAKVCacheSpec::build(ordinary, context);
    EXPECT_EQ(mixed->block_size_bytes(), 128u * 656u);

    ordinary.dtype = DataType::TYPE_BF16;
    auto bf16 = MLAKVCacheSpec::build(ordinary, context);
    EXPECT_EQ(bf16->block_size_bytes(), 128u * 576u * 2u);
}

TEST(MlaFp8CacheSpecTest, OrdinaryE4m3RequiresFp8Storage) {
    AttentionConfigs attention{};
    attention.kv_lora_rank  = 512;
    attention.rope_head_dim = 64;
    SpecBuildContext context{};
    context.attn_config        = &attention;
    context.seq_size_per_block = 128;
    KVCacheSpecDesc desc{};
    desc.tag            = "full";
    desc.cache_type     = KVCacheSpecType::MultiHeadLatentAttention;
    desc.dtype          = DataType::TYPE_BF16;
    desc.mla_fp8_e4m3   = true;
    EXPECT_ANY_THROW(MLAKVCacheSpec::build(desc, context));
}

}  // namespace
}  // namespace rtp_llm
