#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm::test {

namespace {

ModelConfig makeHybridModelConfig(int64_t num_layers) {
    ModelConfig config;
    config.num_layers                   = num_layers;
    config.max_seq_len                  = 128;
    config.hidden_size                  = 64;
    config.vocab_size                   = 1024;
    config.data_type                    = DataType::TYPE_FP16;
    config.attn_config.head_num         = 2;
    config.attn_config.kv_head_num      = 2;
    config.attn_config.size_per_head    = 16;
    config.attn_config.tokens_per_block = 4;
    config.attn_config.kv_cache_dtype   = KvCacheDataType::BASE;

    config.linear_attention_config.linear_conv_kernel_dim = 2;
    config.linear_attention_config.linear_key_head_dim    = 8;
    config.linear_attention_config.linear_value_head_dim  = 8;
    config.linear_attention_config.linear_num_key_heads   = 2;
    config.linear_attention_config.linear_num_value_heads = 2;

    config.hybrid_attention_config.enable_hybrid_attention = true;
    config.hybrid_attention_config.hybrid_attention_types.resize(static_cast<size_t>(num_layers));
    config.kv_cache_spec_descs.assign(static_cast<size_t>(num_layers), {});
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        const bool linear = layer % 4 != 3;
        config.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(layer)] =
            linear ? HybridAttentionType::LINEAR : HybridAttentionType::NONE;
        config.kv_cache_spec_descs[static_cast<size_t>(layer)].push_back(
            linear ? KVCacheSpecDesc{"linear", KVCacheSpecType::LinearAttention} :
                     KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention});
    }
    return config;
}

ModelConfig makeSingleModelConfig(int64_t num_layers) {
    ModelConfig config;
    config.num_layers                   = num_layers;
    config.max_seq_len                  = 128;
    config.hidden_size                  = 64;
    config.vocab_size                   = 1024;
    config.data_type                    = DataType::TYPE_FP16;
    config.attn_config.head_num         = 2;
    config.attn_config.kv_head_num      = 2;
    config.attn_config.size_per_head    = 16;
    config.attn_config.tokens_per_block = 4;
    config.attn_config.kv_cache_dtype   = KvCacheDataType::BASE;
    config.kv_cache_spec_descs.assign(static_cast<size_t>(num_layers),
                                      {KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention}});
    return config;
}

ParallelismConfig makePpConfig(int64_t num_layers, int64_t pp_size, int64_t pp_rank) {
    ParallelismConfig config;
    config.pp_size          = pp_size;
    config.pp_rank          = pp_rank;
    const int64_t base      = num_layers / pp_size;
    const int64_t remainder = num_layers % pp_size;
    for (int64_t stage = 0; stage < pp_size; ++stage) {
        config.pp_stage_layer_counts.push_back(base + (stage < remainder ? 1 : 0));
    }
    return config;
}

size_t countLayersOfType(const CacheConfig& config, CacheGroupType type) {
    size_t count = 0;
    for (const auto& group : config.groups()) {
        if (group.policy.group_type == type) {
            count += config.layerIdsForGroup(group.tag).size();
        }
    }
    return count;
}

KVCacheSpecDesc makeOpaqueStateDesc(const std::string& tag) {
    KVCacheSpecDesc desc;
    desc.tag                  = tag;
    desc.cache_type           = KVCacheSpecType::OpaqueState;
    desc.entry_elems          = 16;
    desc.entry_dtype          = DataType::TYPE_FP32;
    desc.explicit_entry_count = 4;
    return desc;
}

}  // namespace

TEST(ConfigAggregateCompatibilityTest, LegacyAggregateConfigRetainsTokenBudgets) {
    FIFOSchedulerConfig config{4, 8192, "ratio", "2", false, 8, 2048, 128};
    EXPECT_EQ(config.max_inited_kv_cache_streams, 8);
    EXPECT_EQ(config.max_batch_tokens_without_cache, 2048);
    EXPECT_EQ(config.prefill_chunk_size, 128);
    EXPECT_FALSE(config.async_cache_prepare);
    KVCacheConfig cache_config{true};
    EXPECT_TRUE(cache_config.reuse_cache);
    EXPECT_EQ(cache_config.shared_hybrid_cache, -1);
}

TEST(PPStageCacheConfigTest, PpOneKeepsWholeModel) {
    const auto model  = makeHybridModelConfig(8);
    const auto config = CacheConfigCreator::createConfig(model, ParallelismConfig{}, KVCacheConfig{});
    EXPECT_EQ(config.layer_num, 8u);
    EXPECT_EQ(config.global_layer_begin, 0u);
    EXPECT_EQ(config.layerIdsForGroup("linear"), (std::vector<int>{0, 1, 2, 4, 5, 6}));
    EXPECT_EQ(config.layerIdsForGroup("full"), (std::vector<int>{3, 7}));
}

TEST(PPStageCacheConfigTest, StageScopeUsesConfiguredPartition) {
    const auto                                     model    = makeHybridModelConfig(65);
    const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 17}, {17, 33}, {33, 49}, {49, 65}};
    for (int64_t rank = 0; rank < 4; ++rank) {
        const auto config       = CacheConfigCreator::createConfig(model, makePpConfig(65, 4, rank), KVCacheConfig{});
        const auto [begin, end] = expected[static_cast<size_t>(rank)];
        EXPECT_EQ(config.global_layer_begin, static_cast<uint32_t>(begin));
        EXPECT_EQ(config.layer_num, static_cast<uint32_t>(end - begin));
        ASSERT_EQ(config.topology().layers().size(), static_cast<size_t>(end - begin));
        for (int64_t local = 0; local < end - begin; ++local) {
            const auto& tag = model.kv_cache_spec_descs[static_cast<size_t>(begin + local)][0].tag;
            EXPECT_EQ(config.topology().layer(static_cast<int>(local)).group_tags, (std::vector<std::string>{tag}));
        }
    }
}

TEST(PPStageCacheConfigTest, IndependentPoolsUseStageLocalGeometry) {
    const auto model = makeHybridModelConfig(8);
    for (int64_t rank = 0; rank < 2; ++rank) {
        const auto config = CacheConfigCreator::createConfig(model, makePpConfig(8, 2, rank), KVCacheConfig{});
        EXPECT_EQ(config.layer_num, 4u);
        EXPECT_EQ(config.layer_all_num(), 4u);
        EXPECT_EQ(config.global_layer_begin, static_cast<uint32_t>(rank) * 4u);
        EXPECT_EQ(config.topology().layers().size(), 4u);
        EXPECT_EQ(countLayersOfType(config, CacheGroupType::LINEAR), 3u);
        EXPECT_EQ(countLayersOfType(config, CacheGroupType::FULL), 1u);
    }
}

TEST(PPStageCacheConfigTest, CreateConfigScopesBeforeBuildingTopology) {
    const auto    model = makeSingleModelConfig(8);
    KVCacheConfig options;
    const auto    config = CacheConfigCreator::createConfig(model, makePpConfig(8, 2, 1), options);
    EXPECT_EQ(config.layer_num, 4u);
    EXPECT_EQ(config.global_layer_begin, 4u);
    ASSERT_EQ(config.groupNums(), 1);
    EXPECT_EQ(config.layerIdsForGroup("full"), (std::vector<int>{0, 1, 2, 3}));
}

TEST(PPStageCacheConfigTest, StagePartitionPreservesCacheReplayAndHashPolicy) {
    auto model                    = makeSingleModelConfig(8);
    model.cache_min_replay_tokens = 4;
    model.cache_key_hash_seed     = 123;
    for (int64_t rank = 0; rank < 2; ++rank) {
        SCOPED_TRACE(rank);
        const auto layout = makePpConfig(8, 2, rank);
        for (const auto& config : {CacheConfigCreator::createConfig(model, layout, KVCacheConfig{}),
                                   CacheConfigCreator::createWarmupConfig(model, layout, KVCacheConfig{}, 0)}) {
            EXPECT_EQ(config.layer_num, 4u);
            EXPECT_EQ(config.global_layer_begin, static_cast<uint32_t>(rank) * 4u);
            EXPECT_EQ(config.cache_min_replay_tokens, 4);
            EXPECT_EQ(config.cache_key_hash_seed, 123);
            EXPECT_EQ(config.maxPrefixReuseTokens(16), 12);
        }
    }
}

TEST(PPStageCacheConfigTest, WarmupUsesStageLocalTopologyAndSmallFinalizedPools) {
    const auto    model = makeHybridModelConfig(8);
    KVCacheConfig kv_cache_config;
    kv_cache_config.reuse_cache_tail_blocks = 4;
    kv_cache_config.linear_pool_blocks      = 16;
    const auto config = CacheConfigCreator::createWarmupConfig(model, makePpConfig(8, 2, 1), kv_cache_config, 0);
    EXPECT_EQ(config.layer_num, 4u);
    EXPECT_EQ(config.global_layer_begin, 4u);
    EXPECT_EQ(config.reuse_cache_tail_blocks, 4);
    for (const auto& group : config.groups()) {
        EXPECT_EQ(group.block_num, 2u);
    }
}

TEST(PPStageCacheConfigTest, PpAllowsOpaqueStateAndSwaGroups) {
    auto model                       = makeSingleModelConfig(8);
    model.attn_config.sliding_window = 32;
    model.kv_cache_spec_descs[0]     = {makeOpaqueStateDesc("state")};

    const auto config = CacheConfigCreator::createConfig(model, makePpConfig(8, 2, 0), KVCacheConfig{});
    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_TRUE(config.use_opaque_kv_cache_store);
    EXPECT_EQ(config.group("state").policy.group_type, CacheGroupType::SWA);
    EXPECT_EQ(config.group("state").policy.sliding_window_size, 32);

    /** Both layouts contain 64 FP32 elements per block. Layer 0 has 4 entries, layer 4 has 8. */
    auto conflicting_state                 = model.kv_cache_spec_descs[0].front();
    conflicting_state.entry_elems          = 8;
    conflicting_state.explicit_entry_count = 8;
    model.kv_cache_spec_descs[4]           = {conflicting_state};
    for (const auto& parallelism : {makePpConfig(8, 1, 0), makePpConfig(8, 2, 0), makePpConfig(8, 2, 1)}) {
        SCOPED_TRACE(::testing::Message() << "pp_size=" << parallelism.pp_size << ", pp_rank=" << parallelism.pp_rank);
        try {
            CacheConfigCreator::createConfig(model, parallelism, KVCacheConfig{});
            ADD_FAILURE() << "expected conflicting opaque layouts for tag=state to be rejected";
        } catch (const std::exception& e) {
            EXPECT_NE(std::string(e.what()).find("cache tag=state has multiple physical prototypes"), std::string::npos)
                << e.what();
        }
    }
}

TEST(PPStageCacheConfigTest, TransferPolicySurvivesSingletonStagesAndWarmup) {
    for (bool hybrid : {false, true}) {
        const auto model = hybrid ? makeHybridModelConfig(8) : makeSingleModelConfig(8);
        for (int64_t rank : {0, 1, 2}) {
            auto layout                  = makePpConfig(8, 3, rank);
            layout.pp_stage_layer_counts = {4, 3, 1};
            const auto config            = CacheConfigCreator::createConfig(model, layout, KVCacheConfig{});
            const auto warmup            = CacheConfigCreator::createWarmupConfig(model, layout, KVCacheConfig{}, 0);
            EXPECT_EQ(config.groupNums(), hybrid ? 2 : 1);
            EXPECT_EQ(warmup.groupTags(), config.groupTags());
            if (hybrid) {
                EXPECT_EQ(config.layerIdsForGroup("linear").empty(), rank == 2);
                EXPECT_EQ(config.layerIdsForGroup("full").empty(), rank == 1);
                EXPECT_EQ(warmup.layerIdsForGroup("linear"), config.layerIdsForGroup("linear"));
                EXPECT_EQ(warmup.layerIdsForGroup("full"), config.layerIdsForGroup("full"));
            }
            EXPECT_EQ(config.usesGroupCacheTransferPolicy(), hybrid);
            EXPECT_EQ(warmup.usesGroupCacheTransferPolicy(), hybrid);
        }
    }
}

TEST(PPStageCacheConfigTest, PoolBlockWeightsAcceptTagsOwnedByOtherStages) {
    const auto model = makeHybridModelConfig(8);
    KVCacheConfig kv_cache_config;
    kv_cache_config.kv_cache_pool_block_weights = R"({"full":8,"linear":1})";
    // {4,3,1}: stage 1 owns only LINEAR layers, stage 2 only a FULL layer.
    // The weights JSON is broadcast verbatim to every rank, so each stage must
    // accept the tags owned by the other stages instead of failing on
    // "unknown pool", and the weights stay readable on every stage.
    for (int64_t rank = 0; rank < 3; ++rank) {
        SCOPED_TRACE(rank);
        auto layout                  = makePpConfig(8, 3, rank);
        layout.pp_stage_layer_counts = {4, 3, 1};
        const auto config            = CacheConfigCreator::createConfig(model, layout, kv_cache_config);
        EXPECT_EQ(config.blockWeight("full"), 8u);
        EXPECT_EQ(config.blockWeight("linear"), 1u);
        const uint32_t expected_limit = std::numeric_limits<int32_t>::max() / (rank == 1 ? 1u : 8u);
        EXPECT_EQ(config.maxBaselineBlockNum(), expected_limit);
        EXPECT_NO_THROW(CacheConfigCreator::sizingSignature(config, expected_limit));
    }
}

TEST(PPStageCacheConfigTest, PoolBlockWeightsRejectUnknownTagOnSingleStage) {
    const auto  model = makeHybridModelConfig(8);
    KVCacheConfig kv_cache_config;
    kv_cache_config.kv_cache_pool_block_weights = R"({"full":8,"swa":1})";
    // Without PP slicing the local topology is the whole model, so a tag no
    // stage owns stays a hard configuration error.
    EXPECT_THROW(CacheConfigCreator::createConfig(model, ParallelismConfig{}, kv_cache_config), std::exception);
}

TEST(PPStageCacheConfigTest, DraftInheritsWholeModelTransferPolicyOnLastStage) {
    auto layout                      = makePpConfig(8, 2, 1);
    layout.pp_stage_layer_counts     = {7, 1};
    const auto                 draft = makeSingleModelConfig(1);
    SpeculativeExecutionConfig speculative;
    speculative.type              = SP_TYPE_MTP;
    speculative.gen_num_per_cycle = 2;
    for (bool hybrid : {false, true}) {
        const auto target = hybrid ? makeHybridModelConfig(8) : makeSingleModelConfig(8);
        const auto config =
            CacheConfigCreator::createConfig(target, layout, KVCacheConfig{}, speculative, &draft, true, false);
        ASSERT_EQ(config.groupNums(), hybrid ? 2 : 1);
        EXPECT_EQ(config.layerIdsForGroup("full"), (std::vector<int>{0, 1, 2}));
        ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
        for (const auto& sub : config.mtp_sub_configs) {
            ASSERT_EQ(sub->groupTags(), config.groupTags());
            EXPECT_EQ(sub->layerIdsForGroup("full"), (std::vector<int>{0}));
            EXPECT_EQ(sub->usesGroupCacheTransferPolicy(), hybrid);
        }
    }

    auto target = CacheConfigCreator::createConfig(makeHybridModelConfig(8), ParallelismConfig{}, KVCacheConfig{});
    target.model_has_multiple_cache_groups = false;
    ASSERT_EQ(target.groupNums(), 2);
    const auto draft_config = CacheConfigCreator::createConfig(draft, ParallelismConfig{}, KVCacheConfig{});
    const auto sub          = target.mergeMTPModule(draft_config, 0, target.layer_num);
    EXPECT_TRUE(sub->model_has_multiple_cache_groups);
}

TEST(PPStageCacheConfigTest, PerTagOverridesReplaceDerivedCounts) {
    const auto model  = makeHybridModelConfig(8);
    auto       config = CacheConfigCreator::createConfig(model, makePpConfig(8, 2, 0), KVCacheConfig{});

    PPBlockNumOverrides overrides{{"full", 60u}, {"linear", 70u}};
    config.finalizeBlockNums(60, RuntimeConfig{}, &overrides);
    EXPECT_EQ(config.group("full").block_num, 60u);
    EXPECT_EQ(config.group("linear").block_num, 70u);
}

TEST(PPStageCacheConfigTest, DraftCacheExistsOnlyOnLastStageAndIsNotSliced) {
    const auto                 target = makeSingleModelConfig(7);
    const auto                 draft  = makeSingleModelConfig(2);
    SpeculativeExecutionConfig speculative;
    speculative.type              = SP_TYPE_MTP;
    speculative.gen_num_per_cycle = 2;

    EXPECT_THROW(CacheConfigCreator::createConfig(
                     target, makePpConfig(7, 3, 1), KVCacheConfig{}, speculative, &draft, true, false),
                 std::exception);

    auto last_layout                  = makePpConfig(7, 3, 2);
    last_layout.pp_stage_layer_counts = {2, 3, 2};
    auto config =
        CacheConfigCreator::createConfig(target, last_layout, KVCacheConfig{}, speculative, &draft, true, false);
    EXPECT_EQ(config.layer_num, 2u);
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    for (const auto& sub_config : config.mtp_sub_configs) {
        ASSERT_TRUE(sub_config);
        EXPECT_EQ(sub_config->layer_num, 2u);
        EXPECT_EQ(sub_config->global_layer_begin, 0u);
    }
}

TEST(PPStageCacheConfigTest, PerTagOverridesReachEveryDraftModule) {
    const auto                 target = makeSingleModelConfig(4);
    const auto                 draft  = makeSingleModelConfig(2);
    SpeculativeExecutionConfig speculative;
    speculative.type              = SP_TYPE_MTP;
    speculative.gen_num_per_cycle = 2;

    auto config = CacheConfigCreator::createConfig(
        target, makePpConfig(target.num_layers, 2, 1), KVCacheConfig{}, speculative, &draft, true, false);
    PPBlockNumOverrides overrides{{"full", 32u}};
    config.finalizeBlockNums(/*global_block_num=*/24, RuntimeConfig{}, &overrides);

    EXPECT_EQ(config.group("full").block_num, 32u);
    ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
    for (const auto& sub_config : config.mtp_sub_configs) {
        ASSERT_NE(sub_config, nullptr);
        EXPECT_EQ(sub_config->group("full").block_num, 32u);
    }
}

TEST(PPStageCacheConfigTest, MissingPerTagOverrideFails) {
    const auto          model  = makeHybridModelConfig(8);
    auto                config = CacheConfigCreator::createConfig(model, makePpConfig(8, 2, 0), KVCacheConfig{});
    PPBlockNumOverrides incomplete{{"full", 60u}};
    EXPECT_THROW(config.finalizeBlockNums(60, RuntimeConfig{}, &incomplete), std::exception);
}

TEST(SharedCacheConfigTest, Glm53PrefillPolicyKeepsFullExpertParallelism) {
    auto model                 = makeHybridModelConfig(8);
    model.model_type           = "glm5_3_flash";
    model.attn_config.head_num = model.attn_config.kv_head_num = 8;
    model.linear_attention_config.linear_num_key_heads = model.linear_attention_config.linear_num_value_heads = 8;
    ParallelismConfig parallel;
    parallel.role_type = RoleType::PREFILL;
    parallel.tp_size = parallel.ep_size = parallel.world_size = parallel.local_world_size = 8;
    const KVCacheConfig options;
    const auto          config = CacheConfigCreator::createConfig(model, parallel, options);
    EXPECT_TRUE(CacheConfigCreator::useSharedBlockPool(config, model, parallel, options));

    parallel.role_type = RoleType::DECODE;
    EXPECT_FALSE(CacheConfigCreator::useSharedBlockPool(config, model, parallel, options));
    parallel.role_type = RoleType::PREFILL;
    model.model_type   = "another_hybrid_model";
    EXPECT_FALSE(CacheConfigCreator::useSharedBlockPool(config, model, parallel, options));
}

TEST(SharedCacheConfigTest, UnsupportedExplicitPoliciesFailBeforeAllocation) {
    auto model       = makeHybridModelConfig(8);
    model.model_type = "glm5_3_flash";
    ParallelismConfig parallel;
    const auto        config = CacheConfigCreator::createConfig(model, parallel, KVCacheConfig{});
    KVCacheConfig     options;
    options.shared_hybrid_cache = 1;
    EXPECT_TRUE(CacheConfigCreator::useSharedBlockPool(config, model, parallel, options));
    for (const auto method : {CPRotateMethod::ALL_GATHER, CPRotateMethod::PREFILL_CP}) {
        parallel.prefill_cp_config.method = method;
        EXPECT_THROW(CacheConfigCreator::useSharedBlockPool(config, model, parallel, options), std::exception);
        options.shared_hybrid_cache = -1;
        EXPECT_FALSE(CacheConfigCreator::useSharedBlockPool(config, model, parallel, options));
        options.shared_hybrid_cache = 1;
    }
    parallel.prefill_cp_config.method = CPRotateMethod::DISABLED;
    parallel.pp_size                  = 2;
    EXPECT_THROW(CacheConfigCreator::useSharedBlockPool(config, model, parallel, options), std::exception);
    parallel.pp_size           = 1;
    options.linear_pool_blocks = 32;
    const auto independent     = CacheConfigCreator::createConfig(model, parallel, options);
    EXPECT_THROW(CacheConfigCreator::useSharedBlockPool(independent, model, parallel, options), std::exception);
    options.shared_hybrid_cache = 0;
    EXPECT_FALSE(CacheConfigCreator::useSharedBlockPool(independent, model, parallel, options));
}

TEST(SharedCacheConfigTest, ModelManagedPrefillKvShardingRetainsSharedHybridPolicy) {
    auto model                 = makeHybridModelConfig(8);
    model.model_type           = "glm5_3_flash";
    model.attn_config.head_num = model.attn_config.kv_head_num = 8;
    model.linear_attention_config.linear_num_key_heads = model.linear_attention_config.linear_num_value_heads = 8;
    ParallelismConfig parallel;
    parallel.role_type = RoleType::PREFILL;
    parallel.tp_size = parallel.ep_size = parallel.world_size = parallel.local_world_size = 8;
    parallel.prefill_cp_config.kv_cache_sharded                                           = true;
    ASSERT_FALSE(parallel.prefill_cp_config.is_enabled());
    KVCacheConfig options;
    const auto    config = CacheConfigCreator::createConfig(model, parallel, options);
    EXPECT_TRUE(CacheConfigCreator::useSharedBlockPool(config, model, parallel, options));
    options.shared_hybrid_cache = 1;
    EXPECT_TRUE(CacheConfigCreator::useSharedBlockPool(config, model, parallel, options));
    parallel.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    EXPECT_THROW(CacheConfigCreator::useSharedBlockPool(config, model, parallel, options), std::exception);
    options.shared_hybrid_cache = -1;
    EXPECT_FALSE(CacheConfigCreator::useSharedBlockPool(config, model, parallel, options));
}

TEST(SharedCacheConfigTest, ByteBudgetAgreementRetainsMoreThanInt32Bytes) {
    constexpr int64_t gib          = int64_t{1} << 30;
    const int64_t     candidates[] = {80 * gib + 256, 52 * gib, 64 * gib};
    EXPECT_EQ(CacheConfigCreator::selectConfirmedCacheBytes(candidates, 3), 52 * gib);
    EXPECT_EQ(CacheConfigCreator::synchronizeCacheBytes(52 * gib, ParallelismConfig{}), 52 * gib);
    const int64_t invalid[] = {52 * gib, 0};
    EXPECT_THROW(CacheConfigCreator::selectConfirmedCacheBytes(invalid, 2), std::exception);
    EXPECT_THROW(CacheConfigCreator::selectConfirmedCacheBytes(nullptr, 0), std::exception);
    EXPECT_THROW(CacheConfigCreator::synchronizeCacheBytes(uint64_t{1} << 63, ParallelismConfig{}), std::exception);
}

TEST(SharedCacheConfigTest, TestBlockBudgetMatchesExistingPhysicalOwnerLedger) {
    const auto model                = makeHybridModelConfig(8);
    auto       draft                = makeSingleModelConfig(2);
    draft.attn_config.size_per_head = 32;
    SpeculativeExecutionConfig speculative;
    speculative.type              = SP_TYPE_MTP;
    speculative.gen_num_per_cycle = 2;
    KVCacheConfig options;
    options.test_block_num = 19;
    auto config =
        CacheConfigCreator::createConfig(model, ParallelismConfig{}, options, speculative, &draft, true, false);
    config.finalizeBlockNums(19, RuntimeConfig{});
    uint64_t expected_bytes = 0;
    for (const auto& group : config.groups()) {
        expected_bytes += config.blockSizeBytesForGroup(group.tag) * group.block_num;
    }
    EXPECT_EQ(CacheConfigCreator::computeLocalCacheBytes(config, model, {}, options, {}, std::nullopt, speculative),
              expected_bytes);
}

}  // namespace rtp_llm::test
