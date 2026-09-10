#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include <algorithm>
#include <map>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

void registerExecCtxOps(pybind11::module& m);

namespace test {

static ModelConfig makeHybridModelConfig(int64_t num_layers) {
    ModelConfig cfg;
    cfg.num_layers                   = num_layers;
    cfg.max_seq_len                  = 128;
    cfg.hidden_size                  = 64;
    cfg.vocab_size                   = 1024;
    cfg.data_type                    = DataType::TYPE_FP16;
    cfg.attn_config.head_num         = 2;
    cfg.attn_config.kv_head_num      = 2;
    cfg.attn_config.size_per_head    = 16;
    cfg.attn_config.tokens_per_block = 4;
    cfg.attn_config.use_mla          = false;
    cfg.attn_config.kv_cache_dtype   = KvCacheDataType::BASE;

    cfg.linear_attention_config.linear_conv_kernel_dim = 2;
    cfg.linear_attention_config.linear_key_head_dim    = 8;
    cfg.linear_attention_config.linear_value_head_dim  = 8;
    cfg.linear_attention_config.linear_num_key_heads   = 2;
    cfg.linear_attention_config.linear_num_value_heads = 2;

    cfg.hybrid_attention_config.enable_hybrid_attention = true;
    cfg.hybrid_attention_config.hybrid_attention_types.resize(static_cast<size_t>(num_layers));
    cfg.kv_cache_spec_descs.assign(static_cast<size_t>(num_layers), {});
    for (int64_t i = 0; i < num_layers; ++i) {
        const bool linear = (i % 4) != 3;
        cfg.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)] =
            linear ? HybridAttentionType::LINEAR : HybridAttentionType::NONE;
        cfg.kv_cache_spec_descs[static_cast<size_t>(i)].push_back(
            linear ? KVCacheSpecDesc{"linear", KVCacheSpecType::LinearAttention} :
                     KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention});
    }
    return cfg;
}

static ModelConfig makeIndependentPoolModelConfig(int64_t num_layers) {
    ModelConfig cfg                                               = makeHybridModelConfig(num_layers);
    cfg.hybrid_attention_config.enable_independent_kv_cache_pools = true;
    return cfg;
}

static ModelConfig makeSingleModelConfig(int64_t num_layers) {
    ModelConfig cfg;
    cfg.num_layers                   = num_layers;
    cfg.max_seq_len                  = 128;
    cfg.hidden_size                  = 64;
    cfg.vocab_size                   = 1024;
    cfg.data_type                    = DataType::TYPE_FP16;
    cfg.attn_config.head_num         = 2;
    cfg.attn_config.kv_head_num      = 2;
    cfg.attn_config.size_per_head    = 16;
    cfg.attn_config.tokens_per_block = 4;
    cfg.attn_config.use_mla          = false;
    cfg.attn_config.kv_cache_dtype   = KvCacheDataType::BASE;
    cfg.kv_cache_spec_descs.assign(static_cast<size_t>(num_layers), {});
    for (int64_t i = 0; i < num_layers; ++i) {
        cfg.kv_cache_spec_descs[static_cast<size_t>(i)].push_back(
            KVCacheSpecDesc{"full", KVCacheSpecType::MultiHeadAttention});
    }
    return cfg;
}

static ParallelismConfig makePpConfig(int64_t num_layers, int64_t pp_size, int64_t pp_rank) {
    ParallelismConfig pc;
    pc.pp_size         = pp_size;
    pc.pp_rank         = pp_rank;
    const int64_t base = num_layers / pp_size;
    const int64_t rem  = num_layers % pp_size;
    for (int64_t stage = 0; stage < pp_size; ++stage) {
        pc.pp_stage_layer_counts.push_back(base + (stage < rem ? 1 : 0));
    }
    return pc;
}

static size_t countLayersOfType(const CacheConfig& config, CacheGroupType type) {
    size_t count = 0;
    for (const auto& group : config.topology().groups()) {
        if (group.policy.group_type == type) {
            count += group.layer_ids.size();
        }
    }
    return count;
}

TEST(PPStageCacheConfig, stageScopedPp1IsIdentity) {
    const auto mc     = makeHybridModelConfig(8);
    const auto staged = CacheConfigCreator::stageScopedModelConfig(mc, ParallelismConfig{});
    EXPECT_EQ(staged.num_layers, 8);
    EXPECT_EQ(staged.kv_cache_spec_descs.size(), 8u);
    EXPECT_EQ(staged.hybrid_attention_config.hybrid_attention_types.size(), 8u);
}

TEST(PPStageCacheConfig, stageScopedMatchesLayerPartition) {
    const auto                                     mc       = makeHybridModelConfig(65);
    const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 17}, {17, 33}, {33, 49}, {49, 65}};
    for (int64_t rank = 0; rank < 4; ++rank) {
        const auto staged       = CacheConfigCreator::stageScopedModelConfig(mc, makePpConfig(65, 4, rank));
        const auto [begin, end] = expected[static_cast<size_t>(rank)];
        EXPECT_EQ(staged.num_layers, end - begin) << "rank=" << rank;
        ASSERT_EQ(staged.kv_cache_spec_descs.size(), static_cast<size_t>(end - begin)) << "rank=" << rank;
        ASSERT_EQ(staged.hybrid_attention_config.hybrid_attention_types.size(), static_cast<size_t>(end - begin))
            << "rank=" << rank;
        for (int64_t l = 0; l < end - begin; ++l) {
            EXPECT_EQ(staged.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(l)],
                      mc.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(begin + l)])
                << "rank=" << rank << " local=" << l;
            ASSERT_EQ(staged.kv_cache_spec_descs[static_cast<size_t>(l)].size(), 1u);
            ASSERT_EQ(mc.kv_cache_spec_descs[static_cast<size_t>(begin + l)].size(), 1u);
            EXPECT_EQ(staged.kv_cache_spec_descs[static_cast<size_t>(l)][0].cache_type,
                      mc.kv_cache_spec_descs[static_cast<size_t>(begin + l)][0].cache_type)
                << "rank=" << rank << " local=" << l;
            EXPECT_EQ(staged.kv_cache_spec_descs[static_cast<size_t>(l)][0].tag,
                      mc.kv_cache_spec_descs[static_cast<size_t>(begin + l)][0].tag)
                << "rank=" << rank << " local=" << l;
        }
    }
}

TEST(PPStageCacheConfig, stageScopedRejectsInvalidRankAndEmptyStage) {
    const auto mc = makeHybridModelConfig(8);
    EXPECT_THROW(CacheConfigCreator::stageScopedModelConfig(mc, makePpConfig(8, 2, 2)), std::exception);
    EXPECT_THROW(CacheConfigCreator::stageScopedModelConfig(mc, makePpConfig(8, 2, -1)), std::exception);
    const auto tiny = makeHybridModelConfig(2);
    EXPECT_THROW(CacheConfigCreator::stageScopedModelConfig(tiny, makePpConfig(2, 4, 2)), std::exception);
}

TEST(PPStageCacheConfig, hybridPp1Baseline) {
    const auto mc     = makeHybridModelConfig(8);
    const auto config = CacheConfigCreator::createBasicConfig(mc, ParallelismConfig{}, false, 0);
    EXPECT_EQ(config.layer_num, 8u);
    EXPECT_EQ(config.layer_all_num, 8u);
    EXPECT_EQ(config.topology().layers().size(), 8u);
    EXPECT_EQ(countLayersOfType(config, CacheGroupType::LINEAR), 6u);
    EXPECT_EQ(countLayersOfType(config, CacheGroupType::FULL), 2u);
}

TEST(PPStageCacheConfig, independentPoolPp2SlicesGeometry) {
    const auto mc = makeIndependentPoolModelConfig(8);
    for (int64_t rank = 0; rank < 2; ++rank) {
        const auto stage = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, rank), false, 0);
        EXPECT_EQ(stage.layer_num, 4u) << "rank=" << rank;
        EXPECT_EQ(stage.layer_all_num, 4u) << "rank=" << rank;
        ASSERT_EQ(stage.topology().layers().size(), 4u) << "rank=" << rank;
        for (size_t l = 0; l < 4; ++l) {
            EXPECT_EQ(stage.topology().layers()[l].layer_id, static_cast<int>(l)) << "rank=" << rank;
        }
        EXPECT_EQ(countLayersOfType(stage, CacheGroupType::LINEAR), 3u) << "rank=" << rank;
        EXPECT_EQ(countLayersOfType(stage, CacheGroupType::FULL), 1u) << "rank=" << rank;
        ASSERT_EQ(stage.groupNums(), 2) << "rank=" << rank;
        std::vector<std::string> tags;
        for (const auto& group : stage.topology().groups()) {
            tags.push_back(group.tag);
        }
        std::sort(tags.begin(), tags.end());
        EXPECT_EQ(tags, (std::vector<std::string>{"full", "linear"})) << "rank=" << rank;
        EXPECT_TRUE(stage.use_independent_block_pools) << "rank=" << rank;
    }
}

TEST(PPStageCacheConfig, independentPoolMhaStrideCountsPhysicalBlockOnce) {
    const auto mc = makeIndependentPoolModelConfig(8);

    KVCacheConfig kv_cache_config;
    kv_cache_config.seq_size_per_block        = 8;
    kv_cache_config.kernel_seq_size_per_block = 2;
    const auto config = HybridPoolConfigCreator::createConfig(mc, makePpConfig(8, 2, 0), kv_cache_config, false, 0);

    const auto  full_gid = static_cast<size_t>(config.groupIdForTag("full"));
    const auto& spec     = config.specForGroup(full_gid);
    ASSERT_EQ(spec->type, KVCacheSpecType::MultiHeadAttention);
    ASSERT_EQ(config.kernelBlocksPerKvBlockForGroup(full_gid), 4u);
    EXPECT_EQ(config.kvBlockStrideBytesForGroup(full_gid), spec->block_size_bytes());
    EXPECT_EQ(config.kvScaleStrideBytesForGroup(full_gid), spec->scale_block_size_bytes());
}

TEST(PPStageCacheConfig, independentPoolPp2UnevenSplit) {
    const auto mc = makeIndependentPoolModelConfig(9);

    const auto stage0 = CacheConfigCreator::createBasicConfig(mc, makePpConfig(9, 2, 0), false, 0);
    EXPECT_EQ(stage0.layer_num, 5u);
    EXPECT_EQ(countLayersOfType(stage0, CacheGroupType::LINEAR), 4u);
    EXPECT_EQ(countLayersOfType(stage0, CacheGroupType::FULL), 1u);

    const auto stage1 = CacheConfigCreator::createBasicConfig(mc, makePpConfig(9, 2, 1), false, 0);
    EXPECT_EQ(stage1.layer_num, 4u);
    EXPECT_EQ(countLayersOfType(stage1, CacheGroupType::LINEAR), 3u);
    EXPECT_EQ(countLayersOfType(stage1, CacheGroupType::FULL), 1u);
}

TEST(PPStageCacheConfig, singlePp2SlicesGeometry) {
    const auto mc    = makeSingleModelConfig(8);
    const auto whole = CacheConfigCreator::createBasicConfig(mc, ParallelismConfig{}, false, 0);
    ASSERT_EQ(whole.layer_num, 8u);

    for (int64_t rank = 0; rank < 2; ++rank) {
        const auto stage = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, rank), false, 0);
        EXPECT_EQ(stage.layer_num, 4u) << "rank=" << rank;
        ASSERT_EQ(stage.groupNums(), 1) << "rank=" << rank;
        const auto& group = stage.topology().groups()[0];
        ASSERT_EQ(group.layer_ids.size(), 4u) << "rank=" << rank;
        for (size_t l = 0; l < 4; ++l) {
            EXPECT_EQ(group.layer_ids[l], static_cast<int>(l)) << "rank=" << rank;
        }
        EXPECT_EQ(stage.block_size_bytes, whole.block_size_bytes / 2) << "rank=" << rank;
        EXPECT_EQ(stage.layer_to_block_stride_bytes.size(), 4u) << "rank=" << rank;
    }
}

TEST(PPStageCacheConfig, independentPoolPp2TagSubsets) {
    auto    mc          = makeIndependentPoolModelConfig(32);
    int64_t linear_seen = 0;
    for (int64_t i = 0; i < 32; ++i) {
        if (mc.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)] == HybridAttentionType::LINEAR) {
            mc.kv_cache_spec_descs[static_cast<size_t>(i)][0].tag = "linear" + std::to_string(linear_seen / 8);
            ++linear_seen;
        }
    }

    const std::vector<std::vector<std::string>> expected_tags = {
        {"full", "linear0", "linear1"},
        {"full", "linear1", "linear2"},
    };
    for (int64_t rank = 0; rank < 2; ++rank) {
        const auto stage = CacheConfigCreator::createBasicConfig(mc, makePpConfig(32, 2, rank), false, 0);
        ASSERT_EQ(stage.groupNums(), 3) << "rank=" << rank;
        std::vector<std::string> tags;
        for (const auto& group : stage.topology().groups()) {
            tags.push_back(group.tag);
        }
        std::sort(tags.begin(), tags.end());
        auto expected = expected_tags[static_cast<size_t>(rank)];
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(tags, expected) << "rank=" << rank;
        EXPECT_EQ(countLayersOfType(stage, CacheGroupType::FULL), 4u) << "rank=" << rank;
        EXPECT_EQ(countLayersOfType(stage, CacheGroupType::LINEAR), 12u) << "rank=" << rank;
    }

    auto stage0_cfg = CacheConfigCreator::createBasicConfig(mc, makePpConfig(32, 2, 0), false, 0);
    auto stage1_cfg = CacheConfigCreator::createBasicConfig(mc, makePpConfig(32, 2, 1), false, 0);
    stage0_cfg.finalizeBlockNums(100, RuntimeConfig{});
    stage1_cfg.finalizeBlockNums(100, RuntimeConfig{});
    const auto validation =
        validatePPTopology({StageCacheSnapshot::fromConfig(stage0_cfg), StageCacheSnapshot::fromConfig(stage1_cfg)});
    ASSERT_FALSE(validation.ok);
    EXPECT_NE(validation.error.find("absent from stage 0"), std::string::npos);
}

TEST(PPStageCacheConfig, canonicalIndicesFilledFromValidation) {
    auto    mc          = makeIndependentPoolModelConfig(32);
    int64_t linear_seen = 0;
    for (int64_t i = 0; i < 32; ++i) {
        if (mc.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)] == HybridAttentionType::LINEAR) {
            mc.kv_cache_spec_descs[static_cast<size_t>(i)][0].tag = "linear" + std::to_string(linear_seen / 24);
            ++linear_seen;
        }
    }
    auto stage0_cfg = CacheConfigCreator::createBasicConfig(mc, makePpConfig(32, 2, 0), false, 0);
    auto stage1_cfg = CacheConfigCreator::createBasicConfig(mc, makePpConfig(32, 2, 1), false, 0);
    stage0_cfg.finalizeBlockNums(100, RuntimeConfig{});
    stage1_cfg.finalizeBlockNums(100, RuntimeConfig{});

    const std::vector<StageCacheSnapshot> stages     = {StageCacheSnapshot::fromConfig(stage0_cfg),
                                                        StageCacheSnapshot::fromConfig(stage1_cfg)};
    const auto                            validation = validatePPTopology(stages);
    ASSERT_TRUE(validation.ok) << validation.error;

    ASSERT_EQ(validation.canonical_groups.size(), 2u);
    EXPECT_EQ(validation.canonical_groups[0].tag, "linear0");
    EXPECT_EQ(validation.canonical_groups[1].tag, "full");

    applyPPCanonicalIndices(stage0_cfg, validation);
    EXPECT_EQ(stage0_cfg.topology().canonicalIndicesSnapshot(), (std::vector<size_t>{0, 1}));

    applyPPCanonicalIndices(stage1_cfg, validation);
    EXPECT_EQ(stage1_cfg.topology().canonicalIndicesSnapshot(), (std::vector<size_t>{0, 1}));
}

TEST(PPStageCacheConfig, canonicalIndicesRejectUnknownTag) {
    const auto mc     = makeIndependentPoolModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0);
    config.finalizeBlockNums(100, RuntimeConfig{});

    PPValidationResult validation;
    validation.ok = true;
    CanonicalGroupEntry entry;
    entry.tag               = "full";
    entry.logical_block_num = 100;
    validation.canonical_groups.push_back(entry);
    EXPECT_THROW(applyPPCanonicalIndices(config, validation), std::exception);
}

TEST(PPStageCacheConfig, ppRejectsHybridPositionalGrouping) {
    const auto mc = makeHybridModelConfig(8);
    EXPECT_THROW(CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0), std::exception);
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    EXPECT_THROW(CacheConfigCreator::createConfig(mc, makePpConfig(8, 2, 0), runtime_config, kv_cache_config),
                 std::exception);
    EXPECT_NO_THROW(CacheConfigCreator::createBasicConfig(mc, ParallelismConfig{}, false, 0));
}

TEST(PPStageCacheConfig, ppAllowsIndependentPools) {
    const auto mc = makeIndependentPoolModelConfig(8);
    EXPECT_NO_THROW(CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0));
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = 8;
    EXPECT_NO_THROW(CacheConfigCreator::createConfig(mc, makePpConfig(8, 2, 0), runtime_config, kv_cache_config));
}

TEST(PPStageCacheConfig, ppRejectsOpaquePools) {
    auto            mc = makeIndependentPoolModelConfig(8);
    KVCacheSpecDesc state_desc{"state", KVCacheSpecType::OpaqueState};
    state_desc.entry_elems = 16;
    state_desc.entry_dtype = DataType::TYPE_FP32;
    mc.kv_cache_spec_descs[2].push_back(state_desc);
    EXPECT_THROW(CacheConfigCreator::stageScopedModelConfig(mc, makePpConfig(8, 2, 0)), std::exception);
    EXPECT_NO_THROW(CacheConfigCreator::stageScopedModelConfig(mc, ParallelismConfig{}));
}

TEST(PPStageCacheConfig, heterogeneousLinearTagsLegalUnderPp) {
    auto    mc         = makeIndependentPoolModelConfig(32);
    int64_t linear_idx = 0;
    for (int64_t i = 0; i < 32; ++i) {
        if (mc.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)] == HybridAttentionType::LINEAR) {
            mc.kv_cache_spec_descs[static_cast<size_t>(i)][0].tag         = "linear_" + std::to_string(linear_idx);
            mc.kv_cache_spec_descs[static_cast<size_t>(i)][0].entry_elems = (linear_idx % 2 == 0) ? 64u : 128u;
            ++linear_idx;
        }
    }
    EXPECT_NO_THROW(CacheConfigCreator::createBasicConfig(mc, ParallelismConfig{}, false, 0));
    EXPECT_NO_THROW(CacheConfigCreator::createBasicConfig(mc, makePpConfig(32, 2, 0), false, 0));
}

TEST(PPStageCacheConfig, arbitraryRetainedTagNamesPassThrough) {
    auto mc = makeIndependentPoolModelConfig(8);
    for (int64_t i = 0; i < 8; ++i) {
        if (mc.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)] != HybridAttentionType::LINEAR) {
            mc.kv_cache_spec_descs[static_cast<size_t>(i)][0].tag = "linear0";
        }
    }
    EXPECT_NO_THROW(CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0));
}

TEST(PPStageCacheConfig, mtpOnlyLastStageOwnsCompleteDraftCache) {
    const auto    score   = makeSingleModelConfig(7);
    const auto    propose = makeSingleModelConfig(2);
    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num            = 32;
    kv_cache_config.kernel_seq_size_per_block = 2;
    SpeculativeExecutionConfig sp_config;
    sp_config.type = SP_TYPE_MTP;

    for (int tp_size : {1, 2}) {
        for (int steps : {1, 2, 4}) {
            sp_config.gen_num_per_cycle = steps;
            for (int rank = 0; rank < 3; ++rank) {
                auto pc                      = makePpConfig(7, 3, rank);
                pc.pp_stage_layer_counts     = {2, 3, 2};
                pc.tp_size                   = tp_size;
                const auto     config        = rank == 2 ?
                                                   CacheConfigCreator::createSpConfig(score,
                                                                           propose,
                                                                           pc,
                                                                           RuntimeConfig{},
                                                                           kv_cache_config,
                                                                           sp_config,
                                                                           std::nullopt,
                                                                           true,
                                                                           false) :
                                                   CacheConfigCreator::createConfig(
                                            score, pc, RuntimeConfig{}, kv_cache_config, std::nullopt, sp_config);
                const uint32_t target_layers = pc.pp_stage_layer_counts[rank];
                const uint32_t draft_layers  = rank == 2 ? 2 : 0;
                const uint32_t total_layers  = target_layers + draft_layers;
                EXPECT_EQ(config.layer_num, target_layers);
                EXPECT_EQ(config.layer_all_num, total_layers);
                ASSERT_EQ(config.topology().layers().size(), total_layers);
                ASSERT_EQ(config.groupNums(), 1);
                EXPECT_EQ(config.layerIdsForGroup(0).size(), total_layers);
                EXPECT_EQ(config.block_num, 32);
                // K/V * two KV heads / TP * head dim * four tokens * FP16 bytes.
                EXPECT_EQ(config.block_size_bytes, total_layers * (512u / tp_size));
                EXPECT_EQ(config.kernelBlocksPerKvBlockForGroup(0), 2u);
                ASSERT_EQ(config.mtp_sub_configs.size(), rank == 2 ? 1 : 0);
                for (const auto& sub : config.mtp_sub_configs) {
                    ASSERT_TRUE(sub);
                    EXPECT_EQ(sub->layer_num, 2u);
                    EXPECT_EQ(sub->layer_all_num, 2u);
                    EXPECT_EQ(sub->block_num, config.block_num);
                    EXPECT_EQ(sub->tagForGroup(0), config.tagForGroup(0));
                    EXPECT_EQ(sub->layerIdsForGroup(0), (std::vector<int>{0, 1}));
                    EXPECT_EQ(sub->kernelBlocksPerKvBlockForGroup(0), 2u);
                }
                EXPECT_EQ(pc.pp_size, 3);
                EXPECT_EQ(pc.pp_rank, rank);
                EXPECT_EQ(pc.pp_stage_layer_counts, (std::vector<int64_t>{2, 3, 2}));
            }
        }
    }
}

TEST(PPStageCacheConfig, mtpJointBudgetUsesLocalTargetAndDraftLayers) {
    for (bool independent_pools : {false, true}) {
        auto score                                                        = makeSingleModelConfig(8);
        auto propose                                                      = makeSingleModelConfig(2);
        score.hybrid_attention_config.enable_independent_kv_cache_pools   = independent_pools;
        propose.hybrid_attention_config.enable_independent_kv_cache_pools = independent_pools;
        KVCacheConfig kv_cache_config;
        kv_cache_config.kv_cache_mem_mb = 1;
        SpeculativeExecutionConfig sp_config;
        sp_config.type              = SP_TYPE_MTP;
        sp_config.gen_num_per_cycle = 2;
        std::vector<CacheConfig>        configs;
        std::vector<StageCacheSnapshot> snapshots;
        for (int rank = 0; rank < 2; ++rank) {
            auto pc                  = makePpConfig(8, 2, rank);
            pc.pp_stage_layer_counts = {5, 3};
            const auto config =
                rank == 1 ?
                    CacheConfigCreator::createSpConfig(
                        score, propose, pc, RuntimeConfig{}, kv_cache_config, sp_config, std::nullopt, true, false) :
                    CacheConfigCreator::createConfig(
                        score, pc, RuntimeConfig{}, kv_cache_config, std::nullopt, sp_config);
            const size_t bytes_per_block = 5u * 512u;
            const auto   expected_blocks = (1024u * 1024u) / bytes_per_block;
            EXPECT_EQ(config.block_size_bytes, bytes_per_block);
            EXPECT_EQ(config.block_num, expected_blocks);
            EXPECT_EQ(config.blockNumForGroup(0), expected_blocks);
            for (const auto& sub : config.mtp_sub_configs) {
                EXPECT_EQ(sub->block_num, expected_blocks);
                EXPECT_EQ(sub->blockNumForGroup(0), expected_blocks);
            }
            configs.push_back(config);
            snapshots.push_back(StageCacheSnapshot::fromConfig(config));
        }
        const auto validation = validatePPTopology(snapshots);
        ASSERT_TRUE(validation.ok) << validation.error;
        ASSERT_EQ(validation.canonical_groups.size(), 1u);
        EXPECT_EQ(validation.canonical_groups[0].logical_block_num, 409u);
        for (auto& config : configs) {
            config.finalizeBlockNums(
                validation.agreed.paged_block_num, RuntimeConfig{}, &validation.agreed.block_num_overrides);
            EXPECT_NO_THROW(validatePPComposedBlockNums(config, validation.agreed));
            applyPPCanonicalIndices(config, validation);
            EXPECT_EQ(config.block_num, 409u);
            EXPECT_EQ(config.blockNumForGroup(0), 409u);
            for (const auto& sub : config.mtp_sub_configs) {
                EXPECT_EQ(sub->block_num, 409u);
                EXPECT_EQ(sub->blockNumForGroup(0), 409u);
            }
        }
    }
}

TEST(PPStageCacheConfig, negotiatedCapacityUpdatesMtpSubConfigsAndCanonicalGroups) {
    for (bool independent_pools : {false, true}) {
        for (int steps : {1, 2, 4}) {
            auto score                                                        = makeSingleModelConfig(4);
            auto propose                                                      = makeSingleModelConfig(2);
            score.hybrid_attention_config.enable_independent_kv_cache_pools   = independent_pools;
            propose.hybrid_attention_config.enable_independent_kv_cache_pools = independent_pools;
            if (independent_pools) {
                // Stage 0 owns [first_only, full]; the last stage and draft only own [full].
                score.kv_cache_spec_descs[0][0].tag = "first_only";
            }
            KVCacheConfig kv_cache_config;
            kv_cache_config.test_block_num            = 24;
            kv_cache_config.kernel_seq_size_per_block = 2;
            SpeculativeExecutionConfig sp_config;
            sp_config.type              = SP_TYPE_MTP;
            sp_config.gen_num_per_cycle = steps;
            const auto first_config     = CacheConfigCreator::createConfig(
                score, makePpConfig(4, 2, 0), RuntimeConfig{}, kv_cache_config, std::nullopt, sp_config);
            kv_cache_config.test_block_num = 32;
            const auto last_pc             = makePpConfig(4, 2, 1);
            const auto config              = CacheConfigCreator::createSpConfig(
                score, propose, last_pc, RuntimeConfig{}, kv_cache_config, sp_config, std::nullopt, true, false);
            ASSERT_EQ(config.groupNums(), 1);
            ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
            const auto draft_topology = config.mtp_sub_configs[0]->topologyPtr();
            const auto draft_strides  = config.mtp_sub_configs[0]->layer_to_block_stride_bytes;
            ASSERT_EQ(draft_topology->groupById(0).block_num, 32u);
            ASSERT_EQ(draft_topology->groupById(0).canonical_idx, 0u);
            const size_t canonical_idx = independent_pools ? 1u : 0u;
            ASSERT_EQ(first_config.groupIdForTag(config.tagForGroup(0)), static_cast<int>(canonical_idx));
            const auto validation = validatePPTopology(
                {StageCacheSnapshot::fromConfig(first_config), StageCacheSnapshot::fromConfig(config)});
            ASSERT_TRUE(validation.ok) << validation.error;

            auto capped = config;
            capped.finalizeBlockNums(
                validation.agreed.paged_block_num, RuntimeConfig{}, &validation.agreed.block_num_overrides);
            EXPECT_NO_THROW(validatePPComposedBlockNums(capped, validation.agreed));
            applyPPCanonicalIndices(capped, validation);
            EXPECT_EQ(capped.block_num, 24u);
            EXPECT_EQ(capped.blockNumForGroup(0), 24u);
            EXPECT_EQ(capped.topology().groupById(0).canonical_idx, canonical_idx);
            EXPECT_EQ(capped.layer_to_block_stride_bytes, config.layer_to_block_stride_bytes);
            EXPECT_EQ(capped.layerIdsForGroup(0), config.layerIdsForGroup(0));
            for (const auto& sub_config : capped.mtp_sub_configs) {
                const auto& sub = *sub_config;
                ASSERT_EQ(sub.groupNums(), 1);
                EXPECT_EQ(sub.block_num, 24u);
                EXPECT_EQ(sub.blockNumForGroup(0), 24u);
                EXPECT_EQ(sub.topology().groupById(0).canonical_idx, canonical_idx);
                EXPECT_EQ(sub.layer_num, 2u);
                EXPECT_EQ(sub.layer_all_num, 2u);
                EXPECT_EQ(sub.layerIdsForGroup(0), (std::vector<int>{0, 1}));
                EXPECT_EQ(sub.layer_to_block_stride_bytes, draft_strides);
                EXPECT_EQ(sub.kvBlockStrideBytesForGroup(0), draft_topology->groupById(0).kv_block_stride_bytes);
                EXPECT_EQ(sub.kvScaleStrideBytesForGroup(0), draft_topology->groupById(0).kv_scale_stride_bytes);
                EXPECT_EQ(sub.kernelBlocksPerKvBlockForGroup(0), 2u);
            }
        }
    }
}

class PPCacheNegotiationTest: public ::testing::Test {
protected:
    void SetUp() override {
        ops_ = pybind11::module_::import("types").attr("ModuleType")("pp_cache_test").cast<pybind11::module_>();
        registerExecCtxOps(ops_);
        const auto unused_p2p = pybind11::cpp_function([]() { ADD_FAILURE() << "unexpected P2P call"; });
        ops_.attr("register_pp_ops")(
            unused_p2p, unused_p2p, pybind11::cpp_function([this](const pybind11::bytes& payload) {
                local_snapshot_ = StageCacheSnapshot::deserialize(static_cast<std::string>(payload));
                pybind11::list snapshots;
                snapshots.append(pybind11::bytes(first_snapshot_.serialize()));
                snapshots.append(payload);
                return snapshots;
            }));
    }

    void TearDown() override {
        ops_.attr("clear_pp_ops")();
    }

    pybind11::scoped_interpreter interpreter_;
    pybind11::module_            ops_;
    StageCacheSnapshot           first_snapshot_;
    StageCacheSnapshot           local_snapshot_;
};

TEST_F(PPCacheNegotiationTest, snapshotSizingDoesNotMutateMtpConfig) {
    for (bool independent_pools : {false, true}) {
        for (uint32_t first_blocks : {24u, 8u}) {
            SCOPED_TRACE(::testing::Message() << "independent_pools=" << independent_pools
                                             << ", first_blocks=" << first_blocks);
            auto score                                                        = makeSingleModelConfig(4);
            auto propose                                                      = makeSingleModelConfig(2);
            score.hybrid_attention_config.enable_independent_kv_cache_pools   = independent_pools;
            propose.hybrid_attention_config.enable_independent_kv_cache_pools = independent_pools;
            KVCacheConfig kv_cache_config;
            kv_cache_config.test_block_num = first_blocks;
            SpeculativeExecutionConfig sp_config;
            sp_config.type              = SP_TYPE_MTP;
            sp_config.gen_num_per_cycle = 2;
            const auto first_config = CacheConfigCreator::createConfig(
                score, makePpConfig(4, 2, 0), RuntimeConfig{}, kv_cache_config, std::nullopt, sp_config);
            first_snapshot_                = StageCacheSnapshot::fromConfig(first_config);
            local_snapshot_                = {};
            kv_cache_config.test_block_num = 32;
            const auto config = CacheConfigCreator::createSpConfig(score,
                                                                   propose,
                                                                   makePpConfig(4, 2, 1),
                                                                   RuntimeConfig{},
                                                                   kv_cache_config,
                                                                   sp_config,
                                                                   std::nullopt,
                                                                   true,
                                                                   false);
            ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
            const auto topology       = config.topologyPtr();
            const auto draft          = config.mtp_sub_configs[0];
            const auto draft_topology = draft->topologyPtr();
            PPCacheCapacityNegotiator negotiator;
            if (first_blocks == 24u) {
                const auto validation = negotiator.negotiate(config, 28, RuntimeConfig{});
                ASSERT_TRUE(validation.ok) << validation.error;
                EXPECT_EQ(validation.agreed.paged_block_num, 24u);
            } else {
                try {
                    negotiator.negotiate(config, 28, RuntimeConfig{});
                    FAIL() << "expected PP capacity skew rejection";
                } catch (const std::exception& e) {
                    EXPECT_NE(std::string(e.what()).find("capacity skew"), std::string::npos);
                }
            }
            EXPECT_EQ(local_snapshot_.block_nums, (std::vector<uint32_t>{28u}));
            EXPECT_EQ(local_snapshot_.group_tags, StageCacheSnapshot::fromConfig(config).group_tags);
            EXPECT_EQ(config.block_num, 32u);
            EXPECT_EQ(config.topologyPtr(), topology);
            ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
            EXPECT_EQ(config.mtp_sub_configs[0], draft);
            EXPECT_EQ(draft->block_num, 32u);
            EXPECT_EQ(draft->topologyPtr(), draft_topology);
        }
    }
}

TEST(PPStageCacheConfig, composedCapacityRejectsMtpSubConfigMismatch) {
    for (bool independent_pools : {false, true}) {
        auto score                                                        = makeSingleModelConfig(4);
        auto propose                                                      = makeSingleModelConfig(2);
        score.hybrid_attention_config.enable_independent_kv_cache_pools   = independent_pools;
        propose.hybrid_attention_config.enable_independent_kv_cache_pools = independent_pools;
        KVCacheConfig kv_cache_config;
        kv_cache_config.test_block_num = 24;
        SpeculativeExecutionConfig sp_config;
        sp_config.type              = SP_TYPE_MTP;
        sp_config.gen_num_per_cycle = 2;
        auto config = CacheConfigCreator::createSpConfig(score,
                                                         propose,
                                                         makePpConfig(4, 2, 1),
                                                         RuntimeConfig{},
                                                         kv_cache_config,
                                                         sp_config,
                                                         std::nullopt,
                                                         true,
                                                         false);
        ASSERT_EQ(config.mtp_sub_configs.size(), 1u);
        const auto validation = validatePPTopology({StageCacheSnapshot::fromConfig(config)});
        ASSERT_TRUE(validation.ok) << validation.error;
        const auto topology = config.topologyPtr();
        auto&      draft    = *config.mtp_sub_configs[0];
        config.mtp_sub_configs.push_back(nullptr);
        for (uint32_t draft_blocks : {23u, 25u}) {
            draft.finalizeBlockNums(draft_blocks, RuntimeConfig{});
            const auto draft_topology = draft.topologyPtr();
            EXPECT_THROW(validatePPComposedBlockNums(config, validation.agreed), std::exception);
            EXPECT_EQ(config.block_num, 24u);
            EXPECT_EQ(config.topologyPtr(), topology);
            EXPECT_EQ(draft.block_num, draft_blocks);
            EXPECT_EQ(draft.topologyPtr(), draft_topology);
        }
        draft.finalizeBlockNums(24, RuntimeConfig{});
        EXPECT_NO_THROW(validatePPComposedBlockNums(config, validation.agreed));
    }
}

TEST(PPStageCacheConfig, mtpPp1KeepsAllTargetAndDraftLayers) {
    for (bool independent_pools : {false, true}) {
        auto score                                                        = makeSingleModelConfig(7);
        auto propose                                                      = makeSingleModelConfig(2);
        score.hybrid_attention_config.enable_independent_kv_cache_pools   = independent_pools;
        propose.hybrid_attention_config.enable_independent_kv_cache_pools = independent_pools;
        KVCacheConfig kv_cache_config;
        kv_cache_config.test_block_num = 32;
        SpeculativeExecutionConfig sp_config;
        sp_config.type              = SP_TYPE_MTP;
        sp_config.gen_num_per_cycle = 2;
        const auto config           = CacheConfigCreator::createSpConfig(score,
                                                               propose,
                                                               ParallelismConfig{},
                                                               RuntimeConfig{},
                                                               kv_cache_config,
                                                               sp_config,
                                                               std::nullopt,
                                                               true,
                                                               false);
        EXPECT_EQ(config.layer_num, 7u);
        EXPECT_EQ(config.layer_all_num, 11u);
        EXPECT_EQ(config.block_size_bytes, 11u * 512u);
        ASSERT_EQ(config.mtp_sub_configs.size(), 2u);
        for (const auto& sub : config.mtp_sub_configs) {
            EXPECT_EQ(sub->layer_num, 2u);
            EXPECT_EQ(sub->block_num, 32u);
        }
    }
}

TEST(PPStageCacheConfig, speculativeGateRejectsMtpOnNonLastStages) {
    const auto    score   = makeSingleModelConfig(7);
    const auto    propose = makeSingleModelConfig(2);
    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = 32;
    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 1;
    for (int rank : {0, 1}) {
        EXPECT_THROW(CacheConfigCreator::createSpConfig(score,
                                                        propose,
                                                        makePpConfig(7, 3, rank),
                                                        RuntimeConfig{},
                                                        kv_cache_config,
                                                        sp_config,
                                                        std::nullopt,
                                                        true,
                                                        false),
                     std::exception);
    }
}

TEST(PPStageCacheConfig, speculativeGateRejectsOtherDraftTypes) {
    const auto    score   = makeSingleModelConfig(4);
    const auto    propose = makeSingleModelConfig(1);
    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = 32;
    SpeculativeExecutionConfig sp_config;
    sp_config.gen_num_per_cycle = 1;
    for (auto type : {SP_TYPE_VANILLA, SP_TYPE_EAGLE3, SP_TYPE_DSPARK}) {
        sp_config.type = type;
        EXPECT_THROW(CacheConfigCreator::createSpConfig(score,
                                                        propose,
                                                        makePpConfig(4, 2, 1),
                                                        RuntimeConfig{},
                                                        kv_cache_config,
                                                        sp_config,
                                                        std::nullopt,
                                                        true,
                                                        false),
                     std::exception);
    }
}

TEST(PPStageCacheConfig, overrideTableAppliesPerTagCounts) {
    // Different counts per group: the case no single global count can express,
    // which is why the agreement lands as per-tag overrides.
    const auto mc     = makeIndependentPoolModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0);
    config.finalizeBlockNums(100, RuntimeConfig{});
    const auto group_num = static_cast<size_t>(config.groupNums());
    ASSERT_GT(group_num, 1u);

    PPBlockNumOverrides             overrides;
    std::map<std::string, uint32_t> expected_by_tag;
    for (size_t gid = 0; gid < group_num; ++gid) {
        const auto tag       = config.tagForGroup(gid);
        expected_by_tag[tag] = gid == 0 ? 60u : 70u;
        overrides.emplace(tag, gid == 0 ? 60u : 70u);
    }
    config.finalizeBlockNums(60, RuntimeConfig{}, &overrides);

    for (size_t gid = 0; gid < group_num; ++gid) {
        EXPECT_EQ(config.blockNumForGroup(gid), expected_by_tag[config.tagForGroup(gid)]) << "gid=" << gid;
    }
    EXPECT_EQ(config.block_num, 60);
}

TEST(PPStageCacheConfig, overrideCapsSingleGroup) {
    const auto mc     = makeSingleModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 1), false, 0);
    config.finalizeBlockNums(50, RuntimeConfig{});
    ASSERT_EQ(config.groupNums(), 1);

    PPBlockNumOverrides overrides{{config.tagForGroup(0), 30u}};
    config.finalizeBlockNums(30, RuntimeConfig{}, &overrides);
    EXPECT_EQ(config.blockNumForGroup(0), 30u);
    EXPECT_EQ(config.block_num, 30);
}

TEST(PPStageCacheConfig, fuseRejectsComposedBelowAgreed) {
    /* The agreement covers this stage's own snapshot, so a composed count off
       the agreed value means capacity moved after the negotiation; must abort. */
    const auto mc     = makeSingleModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 1), false, 0);
    config.finalizeBlockNums(50, RuntimeConfig{});

    NegotiatedCapacity agreed;
    agreed.paged_block_num                            = 90;
    agreed.block_num_overrides[config.tagForGroup(0)] = 90;
    EXPECT_THROW(validatePPComposedBlockNums(config, agreed), std::exception);
}

TEST(PPStageCacheConfig, fuseRejectsTagMissingFromAgreement) {
    const auto mc     = makeSingleModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 1), false, 0);
    config.finalizeBlockNums(50, RuntimeConfig{});

    NegotiatedCapacity agreed;
    agreed.paged_block_num                       = 30;
    agreed.block_num_overrides["some-other-tag"] = 30;
    EXPECT_THROW(validatePPComposedBlockNums(config, agreed), std::exception);
}

TEST(PPStageCacheConfig, explicitPoolKeepsPinnedCountUnderOverride) {
    const auto mc     = makeIndependentPoolModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0);
    config.finalizeBlockNums(100, RuntimeConfig{});
    ASSERT_EQ(config.groupNums(), 2);
    const auto linear_gid = config.tagForGroup(0) == "linear" ? 0 : 1;
    const auto full_gid   = 1 - linear_gid;

    PPBlockNumOverrides overrides{{"full", 100u}, {"linear", 50u}};
    config.finalizeBlockNums(100, RuntimeConfig{}, &overrides);
    EXPECT_EQ(config.blockNumForGroup(static_cast<size_t>(full_gid)), 100u);
    EXPECT_EQ(config.blockNumForGroup(static_cast<size_t>(linear_gid)), 50u);
    EXPECT_EQ(config.block_num, 100);
}

TEST(PPStageCacheConfig, managerConstructorRequiresNegotiatorUnderPp) {
    const auto mc     = makeIndependentPoolModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0);
    config.finalizeBlockNums(100, RuntimeConfig{});

    EXPECT_THROW(
        KVCacheManager(config, false, nullptr, KVCacheConfig{}, makePpConfig(8, 2, 0), RuntimeConfig{}), std::exception);
}

TEST(PPStageCacheConfig, managerWarmupSkipsCapacityCommunicationUnderPp) {
    const auto mc = makeIndependentPoolModelConfig(8);
    auto       pc = makePpConfig(8, 2, 0);
    pc.tp_size    = 2;
    pc.dp_size    = 2;
    pc.world_size = 8;
    auto config   = CacheConfigCreator::createBasicConfig(mc, pc, false, 0);
    config.finalizeBlockNums(100, RuntimeConfig{});

    KVCacheManager manager(config, true, nullptr, KVCacheConfig{}, pc, RuntimeConfig{});
    EXPECT_EQ(manager.cacheConfig().block_num, 1u);
    for (const auto& group : manager.cacheConfig().topology().groups()) {
        EXPECT_EQ(group.block_num, 1u);
    }
}

TEST(PPStageCacheConfig, managerAppliesNegotiatedCountsByTag) {
    const auto mc     = makeIndependentPoolModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0);
    config.finalizeBlockNums(100, RuntimeConfig{});
    ASSERT_EQ(config.groupNums(), 2);
    auto       first_snapshot = StageCacheSnapshot::fromConfig(config);
    auto       last_snapshot  = first_snapshot;
    const auto group_num      = static_cast<size_t>(config.groupNums());
    for (size_t gid = 0; gid < group_num; ++gid) {
        const bool is_full             = config.tagForGroup(gid) == "full";
        first_snapshot.block_nums[gid] = is_full ? 90u : 80u;
        last_snapshot.block_nums[gid]  = is_full ? 70u : 90u;
    }
    const auto validation = validatePPTopology({first_snapshot, last_snapshot});
    ASSERT_TRUE(validation.ok) << validation.error;
    EXPECT_EQ(validation.agreed.paged_block_num, 70u);
    EXPECT_EQ(validation.agreed.block_num_overrides.at("full"), 70u);
    EXPECT_EQ(validation.agreed.block_num_overrides.at("linear"), 80u);

    class FixedNegotiator: public CacheCapacityNegotiator {
    public:
        explicit FixedNegotiator(PPValidationResult validation): validation_(std::move(validation)) {}

        PPValidationResult
        negotiate(const CacheConfig& topology, uint32_t local_block_num, const RuntimeConfig& runtime_config) override {
            return validation_;
        }

        void validateComposed(const CacheConfig& composed, const NegotiatedCapacity& agreed) override {
            validatePPComposedBlockNums(composed, agreed);
        }

    private:
        PPValidationResult validation_;
    };

    KVCacheManager manager(config,
                           false,
                           nullptr,
                           KVCacheConfig{},
                           makePpConfig(8, 2, 0),
                           RuntimeConfig{},
                           SpeculativeExecutionConfig{},
                           PDSepConfig{},
                           CacheStoreConfig{},
                           false,
                           std::make_shared<FixedNegotiator>(validation));

    const auto& capped = manager.cacheConfig();
    for (size_t gid = 0; gid < group_num; ++gid) {
        EXPECT_EQ(capped.blockNumForGroup(gid), capped.tagForGroup(gid) == "full" ? 70u : 80u) << "gid=" << gid;
    }
    EXPECT_EQ(capped.block_num, 70u);
}

}  // namespace test
}  // namespace rtp_llm
