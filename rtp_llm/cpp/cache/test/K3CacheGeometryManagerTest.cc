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
    // A synthetic LINEAR draft exercises physical-group isolation from target KDA.
    model.model_type                                               = draft ? "test_k3_linear_mtp" : "kimi_k3";
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

void expectGeometry(
    const CacheLayerLayout& layout, const CacheConfig& physical_config, int page_tokens, int shards, bool decode) {
    EXPECT_EQ(layout.local_shard_count, decode ? 1 : shards);
    EXPECT_EQ(layout.linear_step, physical_config.linear_step);
    EXPECT_EQ(layout.group_types, physical_config.group_types);
    EXPECT_EQ(layout.group_region_names, physical_config.group_region_names);
    EXPECT_EQ(layout.group_seq_size_per_block, physical_config.group_seq_size_per_block);
    ASSERT_EQ(layout.group_seq_size_per_block.size(), layout.group_types.size());
    for (size_t group = 0; group < layout.group_types.size(); ++group) {
        SCOPED_TRACE(group);
        const auto kind = layout.group_types[group];
        ASSERT_TRUE(kind == CacheGroupType::FULL || kind == CacheGroupType::LINEAR || kind == CacheGroupType::SWA);
        EXPECT_EQ(layout.group_seq_size_per_block[group],
                  static_cast<size_t>(kind == CacheGroupType::LINEAR ? page_tokens * shards : page_tokens));
    }
    ASSERT_EQ(layout.layer_group_types.size(), layout.layer_to_groups.size());
    for (size_t layer = 0; layer < layout.layer_to_groups.size(); ++layer) {
        const int group = layout.layer_to_groups[layer];
        ASSERT_GE(group, 0);
        ASSERT_LT(static_cast<size_t>(group), layout.group_types.size());
        EXPECT_EQ(layout.layer_group_types[layer], layout.group_types[group]);
        EXPECT_EQ(layout.resolvePhysicalGroupId(layer, KVCacheRegionName::DEFAULT), std::optional<int>(group));
    }
}

void expectReplayMetadata(const CacheLayerLayout& layout, int64_t slots, int64_t steps, int64_t heads) {
    ASSERT_TRUE(layout.linear_replay.has_value());
    const auto& replay = *layout.linear_replay;
    for (const auto* tensors : {&replay.keys, &replay.updates, &replay.log_gates, &replay.conv_inputs}) {
        ASSERT_EQ(tensors->size(), layout.layer_to_groups.size());
        for (size_t layer = 0; layer < tensors->size(); ++layer) {
            const auto& tensor = (*tensors)[layer];
            const bool linear = layout.layer_group_types[layer] == CacheGroupType::LINEAR;
            EXPECT_EQ(tensor.defined(), linear);
            if (!linear) {
                continue;
            }
            ASSERT_TRUE(tensor.defined());
            EXPECT_TRUE(tensor.is_cuda());
            EXPECT_EQ(tensor.device(), layout.layers_to_kv_buffer_ptrs[layer].device());
            const bool conv = tensors == &replay.conv_inputs;
            EXPECT_EQ(tensor.scalar_type(), conv ? torch::kBFloat16 : torch::kFloat32);
            EXPECT_EQ(tensor.sizes().vec(),
                      conv ? std::vector<int64_t>({slots, steps, 3 * heads * 128}) :
                             std::vector<int64_t>({slots, steps, heads, 128}));
        }
    }
    for (const auto* header : {&replay.slot_generations, &replay.log_epochs, &replay.valid_counts, &replay.error_flags}) {
        ASSERT_TRUE(header->defined());
        EXPECT_TRUE(header->is_cuda());
        EXPECT_EQ(header->sizes().vec(), std::vector<int64_t>({slots}));
        const bool wide = header == &replay.slot_generations || header == &replay.log_epochs;
        EXPECT_EQ(header->scalar_type(), wide ? torch::kInt64 : torch::kInt32);
    }
}

class K3CacheGeometryManagerTest: public ::testing::Test {
protected:
    void SetUp() override {
        initLogger();
        createDevice();
    }
};

TEST_F(K3CacheGeometryManagerTest, ExportRejectsInconsistentPhysicalSpecs) {
    ParallelismConfig parallelism;
    parallelism.tp_size                            = 8;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    for (int invalid = 0; invalid < 7; ++invalid) {
        SCOPED_TRACE(invalid);
        auto config = CacheConfigCreator::createBasicConfig(
            makeK3ModelConfig(128, false), parallelism, KVCacheConfig{}, false, 1);
        ASSERT_EQ(config.cache_specs.size(), 2);
        switch (invalid) {
            case 0:
                config.cache_specs.clear();
                break;
            case 1:
                config.group_seq_size_per_block.pop_back();
                break;
            case 2:
                config.cache_specs[0].reset();
                break;
            case 3:
                config.cache_specs[1]->seq_size_per_block = 128;
                break;
            case 4:
                config.cache_specs[0]->type = KVCacheSpecType::LinearAttention;
                break;
            case 5:
                config.cache_specs[0]                     = std::make_shared<LinearKVCacheSpec>();
                config.cache_specs[0]->type               = KVCacheSpecType::MultiHeadLatentAttention;
                config.cache_specs[0]->seq_size_per_block = 128;
                break;
            case 6:
                config.cache_specs[1]                     = std::make_shared<MLAKVCacheSpec>();
                config.cache_specs[1]->type               = KVCacheSpecType::LinearAttention;
                config.cache_specs[1]->seq_size_per_block = 1024;
                break;
        }
        KVCacheManager manager(config, true, nullptr, KVCacheConfig{}, parallelism);
        EXPECT_THROW(manager.getMainModelCacheLayerLayout(), std::invalid_argument);
    }
}

TEST_F(K3CacheGeometryManagerTest, AllocatedMainAndMtpLayoutsRetainLocalAndUpstreamGeometry) {
    for (const int page_tokens : {128, 256}) {
        for (const int shards : {2, 4, 8}) {
            for (const bool decode : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "B=" << page_tokens << " D=" << shards << " decode=" << decode);
                ParallelismConfig parallelism;
                parallelism.role_type                          = decode ? RoleType::DECODE : RoleType::PREFILL;
                parallelism.tp_size                            = shards;
                parallelism.tp_rank                            = shards - 1;
                parallelism.prefill_cp_config.kv_cache_sharded = !decode;
                parallelism.prefill_cp_config.prefill_cp_size  = shards;
                KVCacheConfig kv_config;
                kv_config.test_block_num            = 2;
                kv_config.seq_size_per_block        = page_tokens;
                kv_config.kernel_seq_size_per_block = decode ? 128 : page_tokens;
                kv_config.linear_step               = 1;
                SpeculativeExecutionConfig speculative;
                speculative.type              = SP_TYPE_MTP;
                speculative.gen_num_per_cycle = 2;
                RuntimeConfig runtime;
                runtime.max_generate_batch_size = 2;
                auto config = CacheConfigCreator::createSpConfig(makeK3ModelConfig(page_tokens, false),
                                                                 makeK3ModelConfig(page_tokens, true),
                                                                 parallelism,
                                                                 runtime,
                                                                 kv_config,
                                                                 speculative,
                                                                 std::nullopt,
                                                                 true,
                                                                 false);
                ASSERT_EQ(config.mtp_sub_configs.size(), 2);
                // Warmup bypasses inter-rank capacity synchronization only. init()
                // still executes the production allocator and allocates CUDA pools.
                KVCacheManager manager(config, true, nullptr, kv_config, parallelism);
                ASSERT_TRUE(manager.init());
                const auto main = manager.getMainModelCacheLayerLayout();
                expectGeometry(main, config, page_tokens, shards, decode);
                expectReplayMetadata(main, 2, 3, 96 / shards);
                EXPECT_EQ(main.layer_to_groups, (std::vector<int>{1, 0, 1, 0}));
                ASSERT_EQ(config.linear_replay_group_ids, std::vector<int>({main.layer_to_groups[0]}));
                EXPECT_TRUE(config.linear_replay_channelwise_gate);
                ASSERT_EQ(main.layers_to_kv_buffer_ptrs.size(), 4);
                const auto all = manager.allLayerCacheBase();
                ASSERT_EQ(all.layers_to_kv_buffer_ptrs.size(), 8);
                for (size_t layer = 0; layer < 8; ++layer) {
                    const auto& buffer = all.layers_to_kv_buffer_ptrs[layer];
                    // Physical bytes: BF16 MLA latent+RoPE, or FP32 KDA
                    // state plus one BF16 convolution-history row.
                    const size_t expected_stride = layer % 2 == 0 ?
                                                       (96 / shards) * (128 * 128 * sizeof(float) + 3 * 128 * 2) :
                                                       page_tokens * (512 + 64) * 2;
                    EXPECT_EQ(buffer.stride(0) * buffer.element_size(), expected_stride);
                }
                for (size_t layer = 0; layer < 4; ++layer) {
                    ASSERT_TRUE(main.layers_to_kv_buffer_ptrs[layer].is_cuda());
                    EXPECT_EQ(main.layers_to_kv_buffer_ptrs[layer].data_ptr(),
                              manager.convertIndexToAddr(0, layer).kv_addr);
                }
                for (int module = 0; module < 2; ++module) {
                    const auto mtp = manager.getMTPModuleCacheLayerLayout(module);
                    expectGeometry(mtp, config, page_tokens, shards, decode);
                    EXPECT_FALSE(mtp.linear_replay.has_value());
                    ASSERT_EQ(mtp.layer_to_groups.size(), 2);
                    ASSERT_EQ(mtp.layer_to_group_ids.size(), 2);
                    ASSERT_EQ(mtp.layer_region_to_group_id.size(), 2);
                    EXPECT_NE(mtp.layer_to_groups[0], main.layer_to_groups[0]);
                    EXPECT_FALSE(config.isLinearReplayGroup(mtp.layer_to_groups[0]));
                    ASSERT_EQ(mtp.layers_to_kv_buffer_ptrs.size(), 2);
                    for (int local_layer = 0; local_layer < 2; ++local_layer) {
                        const int global_layer = 4 + 2 * module + local_layer;
                        const int physical_group = config.layer_to_group_id[global_layer];
                        EXPECT_EQ(mtp.layer_to_groups[local_layer], physical_group);
                        EXPECT_EQ(mtp.layer_to_group_ids[local_layer], std::vector<int>({physical_group}));
                        ASSERT_GT(mtp.layer_region_to_group_id[local_layer].size(),
                                  static_cast<size_t>(KVCacheRegionName::DEFAULT));
                        EXPECT_EQ(mtp.layer_region_to_group_id[local_layer][static_cast<size_t>(KVCacheRegionName::DEFAULT)],
                                  physical_group);
                        EXPECT_EQ(mtp.layer_group_types[local_layer],
                                  local_layer == 0 ? CacheGroupType::LINEAR : CacheGroupType::FULL);
                        ASSERT_TRUE(mtp.layers_to_kv_buffer_ptrs[local_layer].is_cuda());
                        EXPECT_EQ(mtp.layers_to_kv_buffer_ptrs[local_layer].data_ptr(),
                                  all.layers_to_kv_buffer_ptrs[global_layer].data_ptr());
                        EXPECT_EQ(mtp.layers_to_kv_buffer_ptrs[local_layer].data_ptr(),
                                  manager.convertIndexToAddr(0, global_layer).kv_addr);
                        EXPECT_NE(mtp.layers_to_kv_buffer_ptrs[local_layer].data_ptr(),
                                  main.layers_to_kv_buffer_ptrs[local_layer].data_ptr());
                    }
                }
            }
        }
    }
}

TEST_F(K3CacheGeometryManagerTest, Eagle3SwaDraftRetainsPhysicalGeometryAndTargetReplay) {
    constexpr int page_tokens = 128;
    constexpr int shards      = 8;
    for (const bool decode : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "decode=" << decode);
        ParallelismConfig parallelism;
        parallelism.role_type                         = decode ? RoleType::DECODE : RoleType::PREFILL;
        parallelism.tp_size                           = shards;
        parallelism.tp_rank                           = shards - 1;
        parallelism.prefill_cp_config.kv_cache_sharded = !decode;
        parallelism.prefill_cp_config.prefill_cp_size  = shards;
        auto draft = makeK3ModelConfig(page_tokens, true);
        draft.model_type = "kimi_k3_mla_swa_eagle3";
        draft.num_layers = 1;
        draft.attn_config.sliding_window = 4096;
        draft.hybrid_attention_config.hybrid_attention_types = {HybridAttentionType::SLIDING_WINDOW};
        KVCacheConfig kv_config;
        kv_config.test_block_num            = 2;
        kv_config.seq_size_per_block        = page_tokens;
        kv_config.kernel_seq_size_per_block = page_tokens;
        kv_config.linear_step               = 1;
        RuntimeConfig runtime;
        runtime.max_generate_batch_size = 2;
        SpeculativeExecutionConfig speculative;
        speculative.type              = SP_TYPE_EAGLE3;
        speculative.model_type        = draft.model_type;
        speculative.gen_num_per_cycle = 2;
        const auto config = CacheConfigCreator::createSpConfig(makeK3ModelConfig(page_tokens, false),
                                                               draft,
                                                               parallelism,
                                                               runtime,
                                                               kv_config,
                                                               speculative,
                                                               std::nullopt,
                                                               true,
                                                               true);
        ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
        ASSERT_EQ(config.group_types,
                  (std::vector<CacheGroupType>{CacheGroupType::FULL, CacheGroupType::LINEAR, CacheGroupType::SWA}));
        ASSERT_EQ(config.cache_specs.size(), 3u);
        ASSERT_NE(dynamic_cast<const MLAKVCacheSpec*>(config.cache_specs[2].get()), nullptr);
        EXPECT_EQ(config.cache_specs[2]->type, KVCacheSpecType::MultiHeadLatentAttention);
        EXPECT_EQ(config.cache_specs[2]->seq_size_per_block, page_tokens);
        EXPECT_TRUE(config.mtp_sub_configs[0]->linear_replay_group_ids.empty());

        // Allowing an MLA-backed SWA group must still reject bad spans and forged spec types.
        for (int invalid = 0; invalid < 3; ++invalid) {
            SCOPED_TRACE(invalid);
            auto malformed = config;
            if (invalid == 0) {
                malformed.group_seq_size_per_block[2] = page_tokens * shards;
            } else {
                auto wrong_spec = std::make_shared<LinearKVCacheSpec>();
                wrong_spec->seq_size_per_block = page_tokens;
                wrong_spec->type = invalid == 1 ? KVCacheSpecType::LinearAttention :
                                                  KVCacheSpecType::MultiHeadLatentAttention;
                malformed.cache_specs[2] = wrong_spec;
            }
            KVCacheManager invalid_manager(malformed, true, nullptr, kv_config, parallelism);
            EXPECT_THROW(invalid_manager.getMainModelCacheLayerLayout(), std::invalid_argument);
        }

        KVCacheManager manager(config, true, nullptr, kv_config, parallelism);
        ASSERT_TRUE(manager.init());
        const auto main = manager.getMainModelCacheLayerLayout();
        expectGeometry(main, config, page_tokens, shards, decode);
        expectReplayMetadata(main, 2, 3, 96 / shards);
        ASSERT_EQ(main.layer_to_groups, (std::vector<int>{1, 0, 1, 0}));
        EXPECT_EQ(config.linear_replay_group_ids, std::vector<int>({1}));
        const auto mtp = manager.getMTPModuleCacheLayerLayout(0);
        expectGeometry(mtp, config, page_tokens, shards, decode);
        EXPECT_FALSE(mtp.linear_replay.has_value());
        ASSERT_EQ(mtp.layer_to_groups, std::vector<int>({2}));
        EXPECT_EQ(mtp.layer_group_types, std::vector<CacheGroupType>({CacheGroupType::SWA}));
        EXPECT_EQ(mtp.layer_to_group_ids, std::vector<std::vector<int>>({{2}}));
        ASSERT_EQ(mtp.layer_region_to_group_id.size(), 1u);
        ASSERT_GT(mtp.layer_region_to_group_id[0].size(), static_cast<size_t>(KVCacheRegionName::DEFAULT));
        EXPECT_EQ(mtp.layer_region_to_group_id[0][static_cast<size_t>(KVCacheRegionName::DEFAULT)], 2);
        EXPECT_FALSE(config.isLinearReplayGroup(2));
        ASSERT_EQ(mtp.layers_to_kv_buffer_ptrs.size(), 1u);
        const auto& buffer = mtp.layers_to_kv_buffer_ptrs[0];
        ASSERT_TRUE(buffer.is_cuda());
        EXPECT_EQ(buffer.stride(0) * buffer.element_size(), page_tokens * (512 + 64) * sizeof(uint16_t));
        EXPECT_EQ(buffer.data_ptr(), manager.convertIndexToAddr(0, 4).kv_addr);
        EXPECT_EQ(buffer.data_ptr(), manager.allLayerCacheBase().layers_to_kv_buffer_ptrs[4].data_ptr());
        for (const auto& target_buffer : main.layers_to_kv_buffer_ptrs) {
            EXPECT_NE(buffer.data_ptr(), target_buffer.data_ptr());
        }
    }
}

}  // namespace
}  // namespace test
}  // namespace rtp_llm
