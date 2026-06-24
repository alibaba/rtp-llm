#include "rtp_llm/cpp/config/ModelConfig.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(ModelConfigTest, SkipsRopeKvCacheForEmbeddingAttention) {
    ModelConfig model_config;
    model_config.use_kvcache                  = false;
    model_config.attn_config.rope_config.style = RopeStyle::No;

    const auto attention_config = model_config.getAttentionConfigs(1);

    EXPECT_FALSE(attention_config.fuse_qkv_add_bias);
    EXPECT_FALSE(attention_config.need_rope_kv_cache);
}

TEST(ModelConfigTest, KeepsRopeKvCacheWhenKvCacheIsEnabled) {
    ModelConfig model_config;
    model_config.use_kvcache                  = true;
    model_config.attn_config.rope_config.style = RopeStyle::No;

    const auto attention_config = model_config.getAttentionConfigs(1);

    EXPECT_TRUE(attention_config.fuse_qkv_add_bias);
    EXPECT_TRUE(attention_config.need_rope_kv_cache);
}

TEST(ModelConfigTest, KeepsRopeKvCacheWhenRopeIsEnabled) {
    ModelConfig model_config;
    model_config.use_kvcache                  = false;
    model_config.attn_config.rope_config.style = RopeStyle::Base;

    const auto attention_config = model_config.getAttentionConfigs(1);

    EXPECT_TRUE(attention_config.fuse_qkv_add_bias);
    EXPECT_TRUE(attention_config.need_rope_kv_cache);
}

}  // namespace
}  // namespace rtp_llm
