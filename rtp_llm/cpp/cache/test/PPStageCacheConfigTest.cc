// Stage-scoped CacheConfig construction and the per-tag capacity agreement.
//
// Pins: pp_size=1 equivalence (scoping is a no-op); pp_size>1 per-stage
// geometry (local layer ids, layer counts, block bytes); tags staying global
// across the slice; and how the manager resolves counts from a negotiated
// agreement. The agreement is consulted in allocateAndSync, so the agreement
// cases drive a KVCacheManager construction with canned negotiator hooks.

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/CacheCapacityNegotiator.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {
namespace test {

namespace {

KVCacheConfig defaultKvConfig() {
    return KVCacheConfig{};
}

// Pins the block count for cases that reach the budget solve: an unpinned
// createConfig solves it from live free memory, and two independent solves
// drift apart on a shared host.
KVCacheConfig pinnedKvConfig(int block_num = 1000) {
    KVCacheConfig config;
    config.test_block_num = block_num;
    return config;
}

}  // namespace

// 3 linear + 1 full attention unit (qwen3_next style). Under the tag-routed
// architecture this builds exactly two type-tagged pools ("full"/"linear").
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
    for (const auto& group : config.groups()) {
        if (group.policy.group_type == type) {
            count += config.groupLayerIds(group.tag).size();
        }
    }
    return count;
}

static std::vector<std::string> sortedGroupTags(const CacheConfig& config) {
    std::vector<std::string> tags;
    tags.reserve(config.groups().size());
    for (const auto& group : config.groups()) {
        tags.push_back(group.tag);
    }
    std::sort(tags.begin(), tags.end());
    return tags;
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
// Stage-local geometry through the single creator entry
// ---------------------------------------------------------------------------

TEST(PPStageCacheConfig, hybridPp1Baseline) {
    const auto mc     = makeHybridModelConfig(8);
    const auto config = CacheConfigCreator::createBasicConfig(mc, ParallelismConfig{}, defaultKvConfig(), 0);
    EXPECT_EQ(config.layer_num, 8u);
    EXPECT_EQ(config.layer_all_num, 8u);
    EXPECT_EQ(config.layers().size(), 8u);
    EXPECT_EQ(countLayersOfType(config, CacheGroupType::LINEAR), 6u);
    EXPECT_EQ(countLayersOfType(config, CacheGroupType::FULL), 2u);
    // Tag-routed grouping: one type-tagged pool per attention kind.
    EXPECT_EQ(sortedGroupTags(config), (std::vector<std::string>{"full", "linear"}));
}

TEST(PPStageCacheConfig, hybridPp2SlicesGeometry) {
    const auto mc = makeHybridModelConfig(8);
    for (int64_t rank = 0; rank < 2; ++rank) {
        const auto stage = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, rank), defaultKvConfig(), 0);
        // 8 layers / 2 stages -> 4 local layers, ids renumbered from 0.
        EXPECT_EQ(stage.layer_num, 4u) << "rank=" << rank;
        EXPECT_EQ(stage.layer_all_num, 4u) << "rank=" << rank;
        ASSERT_EQ(stage.layers().size(), 4u) << "rank=" << rank;
        // 3 linear + 1 full per stage (unit-aligned split of the 3L+1F
        // cycle), merged into one type-tagged pool each.
        EXPECT_EQ(countLayersOfType(stage, CacheGroupType::LINEAR), 3u) << "rank=" << rank;
        EXPECT_EQ(countLayersOfType(stage, CacheGroupType::FULL), 1u) << "rank=" << rank;
        ASSERT_EQ(stage.groupNums(), 2) << "rank=" << rank;
        EXPECT_EQ(sortedGroupTags(stage), (std::vector<std::string>{"full", "linear"})) << "rank=" << rank;
        // Local layer ids are renumbered from 0 within the stage.
        const auto& linear_ids = stage.groupLayerIds("linear");
        ASSERT_EQ(linear_ids.size(), 3u) << "rank=" << rank;
        for (int local : linear_ids) {
            EXPECT_GE(local, 0) << "rank=" << rank;
            EXPECT_LT(local, 4) << "rank=" << rank;
        }
    }
}

TEST(PPStageCacheConfig, hybridPp2UnevenSplit) {
    // 9 layers of the regular 3L+1F period (trailing partial period), even
    // split 5/4: stage0 [0,5) = 4L+1F, stage1 [5,9) = 3L+1F. Pool geometry
    // differs across stages; tags stay identical.
    const auto mc = makeHybridModelConfig(9);

    const auto stage0 = CacheConfigCreator::createBasicConfig(mc, makePpConfig(9, 2, 0), defaultKvConfig(), 0);
    EXPECT_EQ(stage0.layer_num, 5u);
    EXPECT_EQ(countLayersOfType(stage0, CacheGroupType::LINEAR), 4u);
    EXPECT_EQ(countLayersOfType(stage0, CacheGroupType::FULL), 1u);

    const auto stage1 = CacheConfigCreator::createBasicConfig(mc, makePpConfig(9, 2, 1), defaultKvConfig(), 0);
    EXPECT_EQ(stage1.layer_num, 4u);
    EXPECT_EQ(countLayersOfType(stage1, CacheGroupType::LINEAR), 3u);
    EXPECT_EQ(countLayersOfType(stage1, CacheGroupType::FULL), 1u);
}

TEST(PPStageCacheConfig, singlePp2SlicesGeometry) {
    const auto mc    = makeSingleModelConfig(8);
    const auto whole = CacheConfigCreator::createBasicConfig(mc, ParallelismConfig{}, defaultKvConfig(), 0);
    ASSERT_EQ(whole.layer_num, 8u);
    ASSERT_EQ(whole.groupNums(), 1);
    const std::string tag = whole.groups().front().tag;

    for (int64_t rank = 0; rank < 2; ++rank) {
        const auto stage = CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, rank), defaultKvConfig(), 0);
        EXPECT_EQ(stage.layer_num, 4u) << "rank=" << rank;
        ASSERT_EQ(stage.groupNums(), 1) << "rank=" << rank;
        const auto& layer_ids = stage.groupLayerIds(tag);
        ASSERT_EQ(layer_ids.size(), 4u) << "rank=" << rank;
        for (size_t l = 0; l < 4; ++l) {
            EXPECT_EQ(layer_ids[l], static_cast<int>(l)) << "rank=" << rank;
        }
        // Per-block bytes scale with the stage layer count (block bytes =
        // layer_count * per-layer stride for the single group).
        EXPECT_EQ(stage.blockSizeBytes(tag), whole.blockSizeBytes(tag) / 2) << "rank=" << rank;
    }
}

// ---------------------------------------------------------------------------
// Global tag pass-through and gates
// ---------------------------------------------------------------------------

TEST(PPStageCacheConfig, pp2TagSubsetsRejectedAtValidation) {
    // Per-position linear tags make stages own tag SUBSETS; local construction
    // still works, but the allocation-authority gate rejects the topology.
    auto    mc          = makeHybridModelConfig(32);
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
        const auto stage = CacheConfigCreator::createBasicConfig(mc, makePpConfig(32, 2, rank), defaultKvConfig(), 0);
        ASSERT_EQ(stage.groupNums(), 3) << "rank=" << rank;
        EXPECT_EQ(sortedGroupTags(stage), expected_tags[static_cast<size_t>(rank)]) << "rank=" << rank;
        EXPECT_EQ(countLayersOfType(stage, CacheGroupType::FULL), 4u) << "rank=" << rank;
        EXPECT_EQ(countLayersOfType(stage, CacheGroupType::LINEAR), 12u) << "rank=" << rank;
    }

    // Allocation authority: the union has linear2 but stage 0 owns no such pool,
    // so it could not issue block ids for it. Not a capacity problem.
    const auto stage0_cfg = CacheConfigCreator::createBasicConfig(mc, makePpConfig(32, 2, 0), defaultKvConfig(), 0);
    const auto stage1_cfg = CacheConfigCreator::createBasicConfig(mc, makePpConfig(32, 2, 1), defaultKvConfig(), 0);
    const auto validation = validatePPTopology(
        {StageCacheSnapshot::fromTopologyAndCapacity(stage0_cfg, /*local_capacity=*/100, RuntimeConfig{}),
         StageCacheSnapshot::fromTopologyAndCapacity(stage1_cfg, /*local_capacity=*/100, RuntimeConfig{})});
    ASSERT_FALSE(validation.ok);
    EXPECT_NE(validation.error.find("linear2"), std::string::npos);
    EXPECT_NE(validation.error.find("which stage 0 does not"), std::string::npos);
}

TEST(PPStageCacheConfig, ppAcceptsPpSizeAboveOne) {
    // Independent-pool grouping is the only path: both createBasicConfig and
    // createConfig accept pp_size>1 directly.
    const auto mc = makeHybridModelConfig(8);
    EXPECT_NO_THROW(CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), defaultKvConfig(), 0));
    KVCacheConfig kv_cache_config;
    kv_cache_config.test_block_num = 8;
    EXPECT_NO_THROW(CacheConfigCreator::createConfig(
        mc, makePpConfig(8, 2, 0), RuntimeConfig{}, kv_cache_config, std::nullopt, std::nullopt));
}

TEST(PPStageCacheConfig, ppRejectsOpaquePools) {
    // Opaque (state / byte-addressed KV) pools are out of PP phase-1 scope;
    // the gate fires at the stage slice on the local descs.
    auto            mc = makeHybridModelConfig(8);
    KVCacheSpecDesc state_desc{"state", KVCacheSpecType::OpaqueState};
    state_desc.entry_elems = 16;
    state_desc.entry_dtype = DataType::TYPE_FP32;
    mc.kv_cache_spec_descs[2].push_back(state_desc);
    EXPECT_THROW(CacheConfigCreator::stageScopedModelConfig(mc, makePpConfig(8, 2, 0)), std::exception);
    // pp=1: the slicing primitive returns early, so the gate never runs.
    EXPECT_NO_THROW(CacheConfigCreator::stageScopedModelConfig(mc, ParallelismConfig{}));
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

TEST(PPStageCacheConfig, heterogeneousLinearTagsLegalUnderPp) {
    // Two linear layouts tagged individually: each tag keeps its own pool and
    // pp=2 stays legal, cross-stage geometry being reconciled at startup.
    auto    mc         = makeHybridModelConfig(32);
    int64_t linear_idx = 0;
    for (int64_t i = 0; i < 32; ++i) {
        if (mc.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)] == HybridAttentionType::LINEAR) {
            mc.kv_cache_spec_descs[static_cast<size_t>(i)][0].tag         = "linear_" + std::to_string(linear_idx);
            mc.kv_cache_spec_descs[static_cast<size_t>(i)][0].entry_elems = (linear_idx % 2 == 0) ? 64u : 128u;
            ++linear_idx;
        }
    }
    EXPECT_NO_THROW(CacheConfigCreator::createBasicConfig(mc, ParallelismConfig{}, defaultKvConfig(), 0));
    EXPECT_NO_THROW(CacheConfigCreator::createBasicConfig(mc, makePpConfig(32, 2, 0), defaultKvConfig(), 0));
}

TEST(PPStageCacheConfig, arbitraryRetainedTagNamesPassThrough) {
    // A full-layer pool may carry any tag name (here "linear0"); tags are
    // never generated per stage, so no collision semantics apply.
    auto mc = makeHybridModelConfig(8);
    for (int64_t i = 0; i < 8; ++i) {
        if (mc.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)] != HybridAttentionType::LINEAR) {
            mc.kv_cache_spec_descs[static_cast<size_t>(i)][0].tag = "linear0";
        }
    }
    EXPECT_NO_THROW(CacheConfigCreator::createBasicConfig(mc, makePpConfig(8, 2, 0), defaultKvConfig(), 0));
}

// ---------------------------------------------------------------------------
// Capacity agreement cases: the negotiation runs inside allocateAndSync, so a
// canned agreement is injected via the manager's negotiator hook.
// ---------------------------------------------------------------------------

namespace {

// Returns a fixed agreement. validateComposed is deliberately a no-op: this
// file pins the resolution of an agreement, including the silent fallback
// that a real PP hook fuses against (see validatePPComposedBlockNums).
class FixedNegotiator: public CacheCapacityNegotiator {
public:
    FixedNegotiator(uint32_t paged_block_num, PPBlockNumOverrides overrides):
        agreed{paged_block_num, std::move(overrides)} {}

    NegotiatedCapacity
    negotiate(const CacheConfig& topology, uint32_t local_block_num, const RuntimeConfig& runtime_config) override {
        return agreed;
    }

    void validateComposed(const CacheConfig& composed, const NegotiatedCapacity& agreed_capacity) override {}

    NegotiatedCapacity agreed;
};

// Locally sized config plus a manager constructed with the hook; tp_size=1
// keeps allocateAndSync free of real collectives. Spelled out once so the
// cases stay readable.
CacheConfig createWithHook(const ModelConfig&                              mc,
                           const ParallelismConfig&                        pc,
                           const std::shared_ptr<CacheCapacityNegotiator>& negotiator) {
    auto           config = CacheConfigCreator::createConfig(mc, pc, RuntimeConfig{}, pinnedKvConfig());
    KVCacheManager manager(std::move(config),
                           /*warmup=*/false,
                           /*metrics_reporter=*/nullptr,
                           pinnedKvConfig(),
                           pc,
                           RuntimeConfig{},
                           SpeculativeExecutionConfig{},
                           PDSepConfig{},
                           CacheStoreConfig{},
                           /*use_cuda_malloc_block_pool=*/false,
                           negotiator);
    return manager.cacheConfig();
}

}  // namespace

TEST(PPStageCacheConfig, groupAbsentFromOverrideFallsBackToDerivation) {
    // A tag missing from the agreement is derived from the paged yardstick as
    // usual. That silent fallback is exactly what validatePPComposedBlockNums
    // fuses against, so the hook here deliberately does not run the fuse.
    const auto mc         = makeHybridModelConfig(8);
    auto       negotiator = std::make_shared<FixedNegotiator>(/*paged_block_num=*/70,
                                                        PPBlockNumOverrides{{"full", 25}});  // "linear" absent

    const auto config = createWithHook(mc, makePpConfig(8, 2, 0), negotiator);

    EXPECT_EQ(config.group("full").block_num, 25u);
    EXPECT_EQ(config.group("linear").block_num, 70u);
}

TEST(PPStageCacheConfig, explicitPoolKeepsPinnedCountUnderOverride) {
    // An explicit group keeps its pinned count (owners must agree on it), so
    // the pool stays decoupled from the paged yardstick.
    auto mc                             = makeHybridModelConfig(8);
    auto linear                         = KVCacheSpecDesc{"linear", KVCacheSpecType::LinearAttention};
    linear.capacity                     = CacheCapacityPolicyDesc{};
    linear.capacity->explicit_block_num = 50;
    for (int64_t i = 0; i < 8; ++i) {
        if (mc.hybrid_attention_config.hybrid_attention_types[static_cast<size_t>(i)] == HybridAttentionType::LINEAR) {
            mc.kv_cache_spec_descs[static_cast<size_t>(i)][0] = linear;
        }
    }

    auto negotiator =
        std::make_shared<FixedNegotiator>(/*paged_block_num=*/100, PPBlockNumOverrides{{"full", 100}, {"linear", 50}});
    const auto config = createWithHook(mc, makePpConfig(8, 2, 0), negotiator);

    ASSERT_EQ(config.groupNums(), 2);
    EXPECT_EQ(config.group("full").block_num, 100u);
    EXPECT_EQ(config.group("linear").block_num, 50u);
    EXPECT_EQ(config.block_num, 100);
}

}  // namespace test
}  // namespace rtp_llm
