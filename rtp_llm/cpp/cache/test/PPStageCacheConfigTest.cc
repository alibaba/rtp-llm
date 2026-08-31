// Stage-scoped CacheConfig construction.
//
// Pins three behaviors:
//  1. pp_size=1 equivalence: stage scoping is a no-op, configs are identical
//     to whole-model construction.
//  2. pp_size>1: every stage builds cache geometry only for its own layer
//     range: local layer ids, per-stage layer counts, per-stage block bytes.
//  3. Hybrid cache tags stay global (assigned at config-build time) and pass
//     through the stage slice unchanged; irregular hybrid patterns and
//     pp_size > full-layer-count are gated.
// Also covers the startup gates for paths that are not stage-scoped yet
// (hybrid-pool / independent KV pools, speculative execution).

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {
namespace test {

// 3 linear + 1 full attention unit (qwen3_next style).
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

// Independent-pool variant (qwen3 future path): type-tagged pools
// ("full"/"linear"), built through HybridPoolConfigCreator.
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
    pc.pp_size = pp_size;
    pc.pp_rank = pp_rank;
    // Materialized partition, mirroring what the production Python decision
    // point (resolve_pp_partition) writes: even split, remainder to earlier
    // stages. Test-only zero counts are allowed to probe empty-stage guards.
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

// ---------------------------------------------------------------------------
// stageScopedModelConfig: the slicing primitive
// ---------------------------------------------------------------------------

TEST(PPStageCacheConfig, stageScopedPp1IsIdentity) {
    const auto mc     = makeHybridModelConfig(8);
    const auto staged = CacheConfigCreator::stageScopedModelConfig(mc, ParallelismConfig{});
    EXPECT_EQ(staged.num_layers, 8);
    EXPECT_EQ(staged.kv_cache_spec_descs.size(), 8u);
    EXPECT_EQ(staged.hybrid_attention_config.hybrid_attention_types.size(), 8u);
}

TEST(PPStageCacheConfig, stageScopedMatchesLayerPartition) {
    // 65 layers, pp=4 -> 17/16/16/16, remainder to earlier stages
    // (golden values of the Python default partition, even_split_counts).
    const auto                                     mc       = makeHybridModelConfig(65);
    const std::vector<std::pair<int64_t, int64_t>> expected = {{0, 17}, {17, 33}, {33, 49}, {49, 65}};
    for (int64_t rank = 0; rank < 4; ++rank) {
        const auto staged       = CacheConfigCreator::stageScopedModelConfig(mc, makePpConfig(65, 4, rank));
        const auto [begin, end] = expected[static_cast<size_t>(rank)];
        EXPECT_EQ(staged.num_layers, end - begin) << "rank=" << rank;
        ASSERT_EQ(staged.kv_cache_spec_descs.size(), static_cast<size_t>(end - begin)) << "rank=" << rank;
        ASSERT_EQ(staged.hybrid_attention_config.hybrid_attention_types.size(), static_cast<size_t>(end - begin))
            << "rank=" << rank;
        // Sliced content must equal the corresponding global slice.
        for (int64_t l = 0; l < end - begin; ++l) {
            EXPECT_EQ(staged.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(l)],
                      mc.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(begin + l)])
                << "rank=" << rank << " local=" << l;
            ASSERT_EQ(staged.kv_cache_spec_descs[static_cast<size_t>(l)].size(), 1u);
            ASSERT_EQ(mc.kv_cache_spec_descs[static_cast<size_t>(begin + l)].size(), 1u);
            // Cache type and tag are preserved verbatim by the slice: tags
            // stay global and are never renamed per stage.
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
    // 2 layers over 4 stages: ranks 2/3 own zero layers.
    const auto tiny = makeHybridModelConfig(2);
    EXPECT_THROW(CacheConfigCreator::stageScopedModelConfig(tiny, makePpConfig(2, 4, 2)), std::exception);
}

// ---------------------------------------------------------------------------
// Hybrid creator (qwen3.8-27B path)
// ---------------------------------------------------------------------------

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
        // 8 layers / 2 stages -> 4 local layers, ids renumbered from 0.
        EXPECT_EQ(stage.layer_num, 4u) << "rank=" << rank;
        EXPECT_EQ(stage.layer_all_num, 4u) << "rank=" << rank;
        ASSERT_EQ(stage.topology().layers().size(), 4u) << "rank=" << rank;
        for (size_t l = 0; l < 4; ++l) {
            EXPECT_EQ(stage.topology().layers()[l].layer_id, static_cast<int>(l)) << "rank=" << rank;
        }
        // 3 linear + 1 full per stage (unit-aligned split of the 3L+1F
        // cycle), merged into one type-tagged pool each.
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

TEST(PPStageCacheConfig, independentPoolPp2UnevenSplit) {
    // 9 layers of the regular 3L+1F period (trailing partial period), even
    // split 5/4: stage0 [0,5) = 4L+1F, stage1 [5,9) = 3L+1F. Pool geometry
    // differs across stages; tags stay identical.
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

// ---------------------------------------------------------------------------
// Single creator (dense models)
// ---------------------------------------------------------------------------

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
        // Per-block bytes scale with the stage layer count.
        EXPECT_EQ(stage.block_size_bytes, whole.block_size_bytes / 2) << "rank=" << rank;
        EXPECT_EQ(stage.layer_to_block_stride_bytes.size(), 4u) << "rank=" << rank;
    }
}

// ---------------------------------------------------------------------------
// Global tag pass-through and gates
// ---------------------------------------------------------------------------

TEST(PPStageCacheConfig, independentPoolPp2TagSubsets) {
    // Mimics per-position linear tags on a 32-layer 3L+1F model, then
    // verifies pp=2 keeps the tags: stages own tag SUBSETS (linear0/linear1
    // vs linear1/linear2) with the shared tag geometry intact — the pairing
    // case the canonical table reconciles.
    auto    mc          = makeIndependentPoolModelConfig(32);
    int64_t linear_seen = 0;
    for (int64_t i = 0; i < 32; ++i) {
        if (mc.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)] == HybridAttentionType::LINEAR) {
            mc.kv_cache_spec_descs[static_cast<size_t>(i)][0].tag = "linear" + std::to_string(linear_seen / 8);
            ++linear_seen;
        }
    }

    const std::vector<std::vector<std::string>> expected_tags = {
        {"full", "linear0", "linear1"},  // stage 0: global linear layers 0-11
        {"full", "linear1", "linear2"},  // stage 1: global linear layers 12-23
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

    // The stage-0-superset gate rejects this topology at validation time:
    // stage 1 owns linear2 which stage 0 lacks (bookkeeping-only allocation
    // is not supported).
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
    // Full startup pipeline on the pairing case: snapshot exchange ->
    // validation -> per-stage canonical_idx backfill. Positional tags with a
    // 24-linear width keep both stages inside the stage-0-superset rule
    // (24 linear layers total < width -> a single shared "linear0" group).
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

    // Under the superset rule the canonical table equals stage 0's own
    // group list (stage-0 first-seen order).
    ASSERT_EQ(validation.canonical_groups.size(), 2u);
    EXPECT_EQ(validation.canonical_groups[0].tag, "linear0");
    EXPECT_EQ(validation.canonical_groups[1].tag, "full");

    applyPPCanonicalIndices(stage0_cfg, validation);
    EXPECT_EQ(stage0_cfg.topology().canonicalIndicesSnapshot(), (std::vector<size_t>{0, 1}));

    applyPPCanonicalIndices(stage1_cfg, validation);
    // stage1 local [linear0, full] maps to canonical [0, 1] by tag, not by
    // position.
    EXPECT_EQ(stage1_cfg.topology().canonicalIndicesSnapshot(), (std::vector<size_t>{0, 1}));
}

TEST(PPStageCacheConfig, canonicalIndicesRejectUnknownTag) {
    const auto mc     = makeIndependentPoolModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0);
    config.finalizeBlockNums(100, RuntimeConfig{});

    // Canonical table that lacks the local "linear" tag.
    PPValidationResult validation;
    validation.ok = true;
    CanonicalGroupEntry entry;
    entry.tag               = "full";
    entry.logical_block_num = 100;
    validation.canonical_groups.push_back(entry);
    EXPECT_THROW(applyPPCanonicalIndices(config, validation), std::exception);
}

TEST(PPStageCacheConfig, ppRejectsHybridPositionalGrouping) {
    // The legacy positional grouping (enable_hybrid_attention without
    // independent pools) is retired for PP; pp=1 stays allowed.
    const auto mc = makeHybridModelConfig(8);
    EXPECT_THROW(CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0), std::exception);
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    EXPECT_THROW(CacheConfigCreator::createConfig(mc, makePpConfig(8, 2, 0), runtime_config, kv_cache_config),
                 std::exception);
    EXPECT_NO_THROW(CacheConfigCreator::createBasicConfig(mc, ParallelismConfig{}, false, 0));
}

TEST(PPStageCacheConfig, ppAllowsIndependentPools) {
    // Independent-pool grouping is the PP path: both createBasicConfig and
    // createConfig accept pp_size>1 (the old ban is lifted).
    const auto mc = makeIndependentPoolModelConfig(8);
    EXPECT_NO_THROW(CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0));
    RuntimeConfig runtime_config;
    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = 8;
    EXPECT_NO_THROW(CacheConfigCreator::createConfig(mc, makePpConfig(8, 2, 0), runtime_config, kv_cache_config));
}

TEST(PPStageCacheConfig, ppRejectsOpaquePools) {
    // Opaque (state / byte-addressed KV) pools are out of PP v1 scope; the
    // gate fires at the stage slice on the local descs.
    auto            mc = makeIndependentPoolModelConfig(8);
    KVCacheSpecDesc state_desc{"state", KVCacheSpecType::OpaqueState};
    state_desc.entry_elems = 16;
    state_desc.entry_dtype = DataType::TYPE_FP32;
    mc.kv_cache_spec_descs[2].push_back(state_desc);
    EXPECT_THROW(CacheConfigCreator::stageScopedModelConfig(mc, makePpConfig(8, 2, 0)), std::exception);
    // pp=1: the slicing primitive returns early, so the gate never runs.
    EXPECT_NO_THROW(CacheConfigCreator::stageScopedModelConfig(mc, ParallelismConfig{}));
}

TEST(PPStageCacheConfig, heterogeneousLinearTagsLegalUnderPp) {
    // 32 layers; linear layers alternate two different cache layouts and are
    // tagged individually. Each tag keeps its own pool, and pp=2 stays legal
    // (cross-stage geometry is reconciled by the canonical table at
    // startup).
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
    // A full-layer pool may carry any tag name (here "linear0"); tags are
    // never generated per stage, so no collision semantics apply.
    auto mc = makeIndependentPoolModelConfig(8);
    for (int64_t i = 0; i < 8; ++i) {
        if (mc.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)] != HybridAttentionType::LINEAR) {
            mc.kv_cache_spec_descs[static_cast<size_t>(i)][0].tag = "linear0";
        }
    }
    EXPECT_NO_THROW(CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0));
}

// ---------------------------------------------------------------------------
// Gates for paths that are not stage-scoped yet
// ---------------------------------------------------------------------------

TEST(PPStageCacheConfig, speculativeGate) {
    const auto score   = makeSingleModelConfig(4);
    const auto propose = makeSingleModelConfig(1);

    RuntimeConfig              runtime_config;
    KVCacheConfig              kv_cache_config;
    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_MTP;
    sp_config.gen_num_per_cycle = 1;

    EXPECT_THROW(CacheConfigCreator::createSpConfig(score,
                                                    propose,
                                                    makePpConfig(4, 2, 0),
                                                    runtime_config,
                                                    kv_cache_config,
                                                    sp_config,
                                                    std::nullopt,
                                                    /*is_mtp=*/true,
                                                    /*is_eagle=*/false),
                 std::exception);
}

// ---------------------------------------------------------------------------
// applyPPLogicalBlockNums: per-tag min consumption
// ---------------------------------------------------------------------------

TEST(PPStageCacheConfig, applyLogicalBlockNumsCapsByTagNotGid) {
    const auto mc     = makeIndependentPoolModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0);
    config.finalizeBlockNums(100, RuntimeConfig{});
    ASSERT_EQ(config.block_num, 100);
    const auto group_num = static_cast<size_t>(config.groupNums());
    ASSERT_GT(group_num, 1u);

    // Simulated cross-stage result: some groups have lower capacity elsewhere.
    // Reverse the canonical row order to prove pairing is by tag name, not
    // table position.
    PPValidationResult validation;
    validation.ok = true;
    for (size_t gid = 0; gid < group_num; ++gid) {
        CanonicalGroupEntry entry;
        entry.tag               = config.tagForGroup(gid);
        entry.logical_block_num = gid == 0 ? 60u : 70u;
        validation.canonical_groups.push_back(std::move(entry));
    }
    std::reverse(validation.canonical_groups.begin(), validation.canonical_groups.end());

    std::map<std::string, uint32_t> expected_by_tag;
    for (const auto& entry : validation.canonical_groups) {
        expected_by_tag[entry.tag] = entry.logical_block_num;
    }
    std::vector<size_t> kv_strides_before;
    std::vector<size_t> scale_strides_before;
    for (size_t gid = 0; gid < group_num; ++gid) {
        kv_strides_before.push_back(config.kvBlockStrideBytesForGroup(gid));
        scale_strides_before.push_back(config.kvScaleStrideBytesForGroup(gid));
    }

    applyPPLogicalBlockNums(config, validation);

    for (size_t gid = 0; gid < group_num; ++gid) {
        EXPECT_EQ(config.blockNumForGroup(gid), expected_by_tag[config.tagForGroup(gid)]) << "gid=" << gid;
        EXPECT_EQ(config.kvBlockStrideBytesForGroup(gid), kv_strides_before[gid]) << "gid=" << gid;
        EXPECT_EQ(config.kvScaleStrideBytesForGroup(gid), scale_strides_before[gid]) << "gid=" << gid;
    }
    // Top-level block count follows the cross-stage minimum.
    EXPECT_EQ(config.block_num, 60);
}

TEST(PPStageCacheConfig, applyLogicalBlockNumsCapsAtCanonicalMin) {
    const auto mc     = makeSingleModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 1), false, 0);
    config.finalizeBlockNums(50, RuntimeConfig{});
    ASSERT_EQ(config.groupNums(), 1);
    const auto tag = config.tagForGroup(0);

    // A SMALLER canonical count (a poorer owner elsewhere) caps this stage.
    PPValidationResult validation;
    validation.ok = true;
    CanonicalGroupEntry entry;
    entry.tag               = tag;
    entry.logical_block_num = 30;
    validation.canonical_groups.push_back(entry);
    applyPPLogicalBlockNums(config, validation);
    EXPECT_EQ(config.blockNumForGroup(0), 30u);
    EXPECT_EQ(config.block_num, 30);
}

TEST(PPStageCacheConfig, applyLogicalBlockNumsRejectsLocalBelowCanonicalMin) {
    // The canonical min includes this stage's own snapshot value, so a local
    // count below it means the local capacity changed after the startup
    // snapshot exchange. That must abort startup instead of silently
    // building a pool smaller than the id space the leading stage may issue.
    const auto mc     = makeSingleModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 1), false, 0);
    config.finalizeBlockNums(50, RuntimeConfig{});
    ASSERT_EQ(config.groupNums(), 1);

    PPValidationResult validation;
    validation.ok = true;
    CanonicalGroupEntry entry;
    entry.tag               = config.tagForGroup(0);
    entry.logical_block_num = 90;  // larger than the local 50 -> invariant broken
    validation.canonical_groups.push_back(entry);
    EXPECT_THROW(applyPPLogicalBlockNums(config, validation), std::exception);
}

TEST(PPStageCacheConfig, applyLogicalBlockNumsRejectsTagMissingFromCanonicalTable) {
    const auto mc     = makeSingleModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 1), false, 0);
    config.finalizeBlockNums(50, RuntimeConfig{});

    // Canonical table without the local tag: the union invariant is broken.
    PPValidationResult validation;
    validation.ok = true;
    CanonicalGroupEntry entry;
    entry.tag               = "some-other-tag";
    entry.logical_block_num = 30;
    validation.canonical_groups.push_back(entry);
    EXPECT_THROW(applyPPLogicalBlockNums(config, validation), std::exception);
}

TEST(PPStageCacheConfig, applyLogicalBlockNumsExplicitPoolDoesNotCapTopLevel) {
    // The top-level block_num tracks the budget-following paged pools only:
    // an explicitly sized pool must not drag it down.
    const auto mc     = makeIndependentPoolModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0);
    config.finalizeBlockNums(100, RuntimeConfig{});
    ASSERT_EQ(config.groupNums(), 2);
    const auto linear_gid = config.tagForGroup(0) == "linear" ? 0 : 1;
    const auto full_gid   = 1 - linear_gid;

    PPValidationResult validation;
    validation.ok = true;
    CanonicalGroupEntry full_entry;
    full_entry.tag               = "full";
    full_entry.type              = CacheGroupType::FULL;
    full_entry.logical_block_num = 100;
    validation.canonical_groups.push_back(full_entry);
    CanonicalGroupEntry linear_entry;
    linear_entry.tag                = "linear";
    linear_entry.type               = CacheGroupType::LINEAR;
    linear_entry.logical_block_num  = 50;
    linear_entry.explicit_block_num = 50;  // explicitly sized -> decoupled
    validation.canonical_groups.push_back(linear_entry);

    applyPPLogicalBlockNums(config, validation);
    EXPECT_EQ(config.blockNumForGroup(static_cast<size_t>(full_gid)), 100u);
    EXPECT_EQ(config.blockNumForGroup(static_cast<size_t>(linear_gid)), 50u);
    // Top-level follows the paged pool, not the explicit one.
    EXPECT_EQ(config.block_num, 100);
}

// ---------------------------------------------------------------------------
// KVCacheManager constructor wiring (B-lite): the cap is applied internally,
// ordered after allocateAndSync's finalizeBlockNums by construction.
// ---------------------------------------------------------------------------

TEST(PPStageCacheConfig, managerConstructorRequiresCapacityUnderPp) {
    const auto mc     = makeIndependentPoolModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0);
    config.finalizeBlockNums(100, RuntimeConfig{});

    // pp_size>1 without the validator result -> fail fast at construction.
    EXPECT_THROW(KVCacheManager(config, false, nullptr, KVCacheConfig{}, makePpConfig(8, 2, 0), RuntimeConfig{}),
                 std::exception);
}

TEST(PPStageCacheConfig, managerConstructorCapsGroupsByTag) {
    const auto mc     = makeIndependentPoolModelConfig(8);
    auto       config = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), false, 0);
    config.finalizeBlockNums(100, RuntimeConfig{});
    const auto group_num = static_cast<size_t>(config.groupNums());
    ASSERT_GT(group_num, 1u);

    PPValidationResult validation;
    validation.ok = true;
    std::map<std::string, uint32_t> expected_by_tag;
    for (size_t gid = 0; gid < group_num; ++gid) {
        const auto tag       = config.tagForGroup(gid);
        expected_by_tag[tag] = gid == 0 ? 60u : 70u;
        CanonicalGroupEntry entry;
        entry.tag               = tag;
        entry.logical_block_num = gid == 0 ? 60u : 70u;
        validation.canonical_groups.push_back(std::move(entry));
    }

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
                           validation);

    const auto& capped = manager.cacheConfig();
    for (size_t gid = 0; gid < group_num; ++gid) {
        EXPECT_EQ(capped.blockNumForGroup(gid), expected_by_tag[capped.tagForGroup(gid)]) << "gid=" << gid;
    }
    EXPECT_EQ(capped.block_num, 60);
}

}  // namespace test
}  // namespace rtp_llm
