#include <gtest/gtest.h>
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"

namespace rtp_llm {
TEST(MLAFp8CacheSpecTest, OrdinaryAndMixedFormatsHaveDifferentPhysicalSizes) {
    AttentionConfigs attn;
    attn.tokens_per_block = 128;
    attn.kv_lora_rank = 512;
    attn.rope_head_dim = 64;
    attn.kv_cache_dtype = KvCacheDataType::FP8;
    ParallelismConfig pc;
    MLAKVCacheSpec mixed(attn, pc);
    mixed.dtype = TYPE_FP8_E4M3;
    EXPECT_EQ(mixed.block_size_bytes(), 128u * 656u);
    attn.mla_fp8_compute = true;
    MLAKVCacheSpec plain(attn, pc);
    plain.dtype = TYPE_FP8_E4M3;
    EXPECT_EQ(plain.block_size_bytes(), 128u * 576u);
    EXPECT_EQ(plain.k_block_size_bytes() + plain.v_block_size_bytes(), plain.block_size_bytes());
    attn.mla_fp8_compute = false;
    attn.kv_cache_dtype = KvCacheDataType::BASE;
    MLAKVCacheSpec bf16(attn, pc);
    bf16.dtype = TYPE_BF16;
    EXPECT_EQ(bf16.block_size_bytes(), 128u * 576u * 2u);
}

} // namespace rtp_llm
