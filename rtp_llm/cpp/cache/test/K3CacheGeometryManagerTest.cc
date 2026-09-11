#include <gtest/gtest.h>
#include <stdexcept>

#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {
namespace {

ModelConfig makeK3ModelConfig(int physical_page_tokens, bool draft) {
    ModelConfig model;
    model.num_layers                                                = draft ? 2 : 4;
    model.max_seq_len                                               = physical_page_tokens;
    model.data_type                                                 = DataType::TYPE_BF16;
    model.attn_config.use_mla                                       = true;
    model.attn_config.kv_cache_dtype                                = KvCacheDataType::BASE;
    model.attn_config.tokens_per_block                              = physical_page_tokens;
    model.attn_config.kv_lora_rank                                  = 512;
    model.attn_config.rope_head_dim                                 = 64;
    model.attn_config.head_num                                      = 96;
    model.attn_config.kv_head_num                                   = 1;
    model.attn_config.size_per_head                                 = 192;
    model.hybrid_attention_config.enable_hybrid_attention           = true;
    model.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    // Layer zero deliberately belongs to group 1, not the FULL group 0.
    model.hybrid_attention_config.hybrid_attention_types =
        draft ? std::vector<HybridAttentionType>{HybridAttentionType::LINEAR, HybridAttentionType::NONE} :
                std::vector<HybridAttentionType>{HybridAttentionType::LINEAR,
                                                 HybridAttentionType::NONE,
                                                 HybridAttentionType::LINEAR,
                                                 HybridAttentionType::NONE};
    model.linear_attention_config.linear_num_key_heads   = 96;
    model.linear_attention_config.linear_num_value_heads = 96;
    model.linear_attention_config.linear_key_head_dim    = 128;
    model.linear_attention_config.linear_value_head_dim  = 128;
    model.linear_attention_config.linear_conv_kernel_dim = 2;
    model.linear_attention_config.ssm_state_dtype        = DataType::TYPE_FP32;
    return model;
}

class K3CacheGeometryManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        initLogger();
        createDevice();
    }
};

TEST_F(K3CacheGeometryManagerTest, TargetFp8AndMtpBf16UseSeparatePhysicalPools) {
    for (const int tp : {1, 2, 4, 8}) {
        for (const int candidates : {1, 3}) {
            SCOPED_TRACE(::testing::Message() << "TP=" << tp << " candidates=" << candidates);
            ParallelismConfig parallelism;
            parallelism.tp_size = parallelism.ep_size = tp;
            auto target = makeK3ModelConfig(128, false);
            target.attn_config.mla_fp8_compute = true;
            target.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
            auto draft = makeK3ModelConfig(128, true);
            draft.num_layers = 1;
            draft.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::NONE};
            KVCacheConfig kv;
            kv.test_block_num = 2;
            kv.seq_size_per_block = kv.kernel_seq_size_per_block = 128;
            SpeculativeExecutionConfig sp;
            sp.type = SP_TYPE_MTP;
            sp.model_type = "kimi_k3_mtp";
            sp.gen_num_per_cycle = candidates;
            auto target_config = CacheConfigCreator::createBasicConfig(target, parallelism, kv, false, candidates);
            auto draft_config = CacheConfigCreator::createBasicConfig(draft, parallelism, kv, true, candidates);
            auto config = CacheConfigCreator::createSpConfig(
                target, draft, parallelism, RuntimeConfig{}, kv, sp, std::nullopt, true, false);
            ASSERT_EQ(config.cache_specs.size(), 3);
            ASSERT_EQ(config.mtp_sub_configs.size(), 1);
            EXPECT_EQ(config.group_types, (std::vector<CacheGroupType>{
                CacheGroupType::FULL, CacheGroupType::LINEAR, CacheGroupType::FULL}));
            EXPECT_EQ(config.cache_specs[0]->dtype, TYPE_FP8_E4M3);
            EXPECT_EQ(config.cache_specs[2]->dtype, TYPE_BF16);
            EXPECT_EQ(config.cache_specs[0]->k_block_size_bytes() + config.cache_specs[0]->v_block_size_bytes(),
                      128u * 576u);
            EXPECT_EQ(config.cache_specs[2]->block_size_bytes(), 128u * 576u * 2u);
            EXPECT_EQ(config.block_size_bytes, target_config.block_size_bytes + draft_config.block_size_bytes);
            EXPECT_EQ(config.layer_to_group_id, (std::vector<int>{1, 0, 1, 0, 2}));
            EXPECT_EQ(config.layer_to_block_stride_bytes[1], 128u * 576u);
            EXPECT_EQ(config.layer_to_block_stride_bytes[4], 128u * 576u * 2u);
            KVCacheManager manager(config, true, nullptr, kv, parallelism);
            ASSERT_TRUE(manager.init());
            const auto main = manager.getMainModelCacheLayerLayout();
            const auto mtp = manager.getMTPModuleCacheLayerLayout(0);
            const auto all = manager.allLayerCacheBase();
            ASSERT_EQ(all.layers_to_kv_buffer_ptrs.size(), 5);
            ASSERT_EQ(mtp.layers_to_kv_buffer_ptrs.size(), 1);
            const auto& mtp_view = mtp.layers_to_kv_buffer_ptrs[0];
            EXPECT_EQ(mtp_view.data_ptr(), all.layers_to_kv_buffer_ptrs[4].data_ptr());
            EXPECT_EQ(mtp_view.stride(0) * mtp_view.element_size(), 128u * 576u * 2u);
            EXPECT_EQ(main.layers_to_kv_buffer_ptrs[1].stride(0)
                          * main.layers_to_kv_buffer_ptrs[1].element_size(), 128u * 576u);
            EXPECT_EQ(mtp_view.data_ptr(), manager.convertIndexToAddr(0, 4).kv_addr);
            EXPECT_NE(mtp_view.data_ptr(), manager.convertIndexToAddr(0, 1).kv_addr);
            EXPECT_NE(mtp_view.data_ptr(), manager.convertIndexToAddr(0, 0).kv_addr);
            // The test write helper must also use the draft's BF16 byte
            // offsets, rather than borrowing the first (FP8 target) spec.
            auto k = torch::full({128 * 512 * 2}, 41, torch::kUInt8);
            auto v = torch::full({128 * 64 * 2}, 29, torch::kUInt8);
            ASSERT_TRUE(manager.setKVBlockValue(0, 4, k, v));
            auto bytes = torch::from_blob(mtp_view.data_ptr(), {128 * 576 * 2},
                                          torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA)).cpu();
            EXPECT_TRUE(torch::equal(bytes.slice(0, 0, k.numel()), k));
            EXPECT_TRUE(torch::equal(bytes.slice(0, k.numel()), v));
        }
    }
}

}  // namespace
}  // namespace test
}  // namespace rtp_llm
