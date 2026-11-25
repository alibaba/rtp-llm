#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"
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

TEST(PPStageCacheConfigTest, PpOneKeepsWholeModel) {
    const auto model  = makeHybridModelConfig(8);
    const auto config = CacheConfigCreator::createConfig(model, ParallelismConfig{}, KVCacheConfig{});
    EXPECT_EQ(config.layer_num(), 8u);
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
        EXPECT_EQ(config.layer_num(), static_cast<uint32_t>(end - begin));
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
        EXPECT_EQ(config.layer_num(), 4u);
        EXPECT_EQ(config.topology().layers().size(), 4u);
        EXPECT_EQ(config.global_layer_begin, static_cast<uint32_t>(rank * 4));
        EXPECT_EQ(countLayersOfType(config, CacheGroupType::LINEAR), 3u);
        EXPECT_EQ(countLayersOfType(config, CacheGroupType::FULL), 1u);
    }
}

TEST(PPStageCacheConfigTest, CreateConfigScopesBeforeBuildingTopology) {
    const auto    model = makeSingleModelConfig(8);
    KVCacheConfig options;
    const auto    config = CacheConfigCreator::createConfig(model, makePpConfig(8, 2, 1), options);
    EXPECT_EQ(config.layer_num(), 4u);
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
            EXPECT_EQ(config.layer_num(), 4u);
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
    EXPECT_EQ(config.layer_num(), 4u);
    EXPECT_EQ(config.reuse_cache_tail_blocks(), 4);
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
    EXPECT_TRUE(config.use_opaque_kv_cache_store());
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

TEST(PPStageCacheConfigTest, MiddleMainStateRingUsesTheSpeculativeRoundSize) {
    auto target                               = makeSingleModelConfig(3);
    auto ring                                 = makeOpaqueStateDesc("main_state");
    ring.entry_count_mode                     = OpaqueBlockEntryCountMode::STATE_RING;
    ring.compression_ratio                    = 4;
    ring.state_ring_include_gen_num_per_cycle = true;
    for (auto& layer : target.kv_cache_spec_descs) {
        layer.push_back(ring);
    }

    auto draft                          = makeSingleModelConfig(1);
    draft.kv_cache_spec_descs[0][0].tag = "mtp_full";
    SpeculativeExecutionConfig speculative;
    speculative.type              = SP_TYPE_MTP;
    speculative.gen_num_per_cycle = 3;
    const auto first = CacheConfigCreator::createConfig(
        target, makePpConfig(3, 3, 0), KVCacheConfig{}, speculative, &draft, true, false);
    const auto middle = CacheConfigCreator::createConfig(
        target, makePpConfig(3, 3, 1), KVCacheConfig{}, speculative);
    const auto last = CacheConfigCreator::createConfig(
        target, makePpConfig(3, 3, 2), KVCacheConfig{}, speculative, &draft, true, false);
    const auto no_speculation = CacheConfigCreator::createConfig(target, makePpConfig(3, 3, 1), KVCacheConfig{});

    const auto bytes = middle.group("main_state").spec->block_size_bytes();
    EXPECT_EQ(bytes, first.group("main_state").spec->block_size_bytes());
    EXPECT_EQ(bytes, last.group("main_state").spec->block_size_bytes());
    EXPECT_GT(bytes, no_speculation.group("main_state").spec->block_size_bytes());
}

TEST(PPStageCacheConfigTest, DraftKeepsItsOwnTransferPolicyOnLastStage) {
    auto layout                         = makePpConfig(8, 2, 1);
    layout.pp_stage_layer_counts        = {7, 1};
    auto draft                          = makeSingleModelConfig(1);
    draft.kv_cache_spec_descs[0][0].tag = "mtp_full";
    SpeculativeExecutionConfig speculative;
    speculative.type              = SP_TYPE_MTP;
    speculative.gen_num_per_cycle = 2;
    for (bool hybrid : {false, true}) {
        const auto target = hybrid ? makeHybridModelConfig(8) : makeSingleModelConfig(8);
        const auto config =
            CacheConfigCreator::createConfig(target, layout, KVCacheConfig{}, speculative, &draft, true, false);
        ASSERT_EQ(config.groupNums(), hybrid ? 2 : 1);
        EXPECT_EQ(config.layerIdsForGroup("full"), (std::vector<int>{0}));
        if (hybrid) {
            EXPECT_TRUE(config.layerIdsForGroup("linear").empty());
        }
        EXPECT_EQ(config.usesGroupCacheTransferPolicy(), hybrid);
        ASSERT_EQ(config.moduleCount(), 2u);
        {
            ASSERT_EQ(config.topology(1).groups().size(), 1u);
            EXPECT_FALSE(config.usesGroupCacheTransferPolicy(1));
        }
    }
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
    const auto target = makeSingleModelConfig(7);
    auto       draft  = makeSingleModelConfig(3);
    for (auto& layer : draft.kv_cache_spec_descs) {
        layer[0].tag = "mtp_full";
    }
    SpeculativeExecutionConfig speculative;
    speculative.type              = SP_TYPE_MTP;
    speculative.gen_num_per_cycle = 2;

    const std::vector<uint32_t> stage_begins{0, 2, 5};
    for (size_t rank = 0; rank < stage_begins.size(); ++rank) {
        SCOPED_TRACE(rank);
        auto layout                  = makePpConfig(7, 3, rank);
        layout.pp_stage_layer_counts = {2, 3, 2};
        const auto config =
            CacheConfigCreator::createConfig(target, layout, KVCacheConfig{}, speculative, &draft, true, false);
        EXPECT_EQ(config.global_layer_begin, stage_begins[rank]);
        EXPECT_EQ(config.layer_num(), static_cast<uint32_t>(layout.pp_stage_layer_counts[rank]));
        ASSERT_EQ(config.moduleCount(), 2u);
        EXPECT_EQ(config.allGroupNums(), 2);
        const auto& sub = config.topology(1);
        ASSERT_EQ(sub.groupTags(), (std::vector<std::string>{"mtp_full"}));
        const auto snapshot = StageCacheSnapshot::captureLocalStage(config, 42, {});
        ASSERT_EQ(snapshot.groups.size(), 2u);
        const auto& draft_group = snapshot.groups.back();
        EXPECT_EQ(draft_group.tag, "mtp_full");
        EXPECT_TRUE(draft_group.is_draft);
        EXPECT_EQ(draft_group.has_local_backing, rank == 2);
        EXPECT_EQ(draft_group.block_num, rank == 2 ? 42u : 0u);
        if (rank == 2) {
            EXPECT_EQ(config.topology(1).layers().size(), 3u);
            EXPECT_EQ(sub.layerIdsForGroup("mtp_full"), (std::vector<int>{0, 1, 2}));
            EXPECT_GT(sub.blockSizeBytesForGroup("mtp_full"), 0u);
        } else {
            EXPECT_EQ(config.topology(1).layers().size(), 0u);
            EXPECT_TRUE(sub.layerIdsForGroup("mtp_full").empty());
            EXPECT_EQ(sub.blockSizeBytesForGroup("mtp_full"), 0u);
        }
    }
}

TEST(PPStageCacheConfigTest, PerTagOverridesApplyToMainAndDraftIndependently) {
    const auto target = makeSingleModelConfig(4);
    auto       draft  = makeSingleModelConfig(2);
    for (auto& layer : draft.kv_cache_spec_descs) {
        layer[0].tag = "mtp_full";
    }
    SpeculativeExecutionConfig speculative;
    speculative.type              = SP_TYPE_MTP;
    speculative.gen_num_per_cycle = 2;

    auto config = CacheConfigCreator::createConfig(
        target, makePpConfig(target.num_layers, 2, 1), KVCacheConfig{}, speculative, &draft, true, false);
    PPBlockNumOverrides overrides{{"full", 32u}, {"mtp_full", 16u}};
    config.finalizeBlockNums(/*global_block_num=*/24, RuntimeConfig{}, &overrides);

    EXPECT_EQ(config.group("full").block_num, 32u);
    ASSERT_EQ(config.moduleCount(), 2u);
    {
        EXPECT_EQ(config.topology(1).group("mtp_full").block_num, 16u);
    }
}

TEST(PPStageCacheConfigTest, MissingPerTagOverrideFails) {
    const auto          model  = makeHybridModelConfig(8);
    auto                config = CacheConfigCreator::createConfig(model, makePpConfig(8, 2, 0), KVCacheConfig{});
    PPBlockNumOverrides incomplete{{"full", 60u}};
    EXPECT_THROW(config.finalizeBlockNums(60, RuntimeConfig{}, &incomplete), std::exception);
}

}  // namespace rtp_llm::test
