#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"

namespace rtp_llm {

namespace {

std::string policyFingerprint(CacheGroupType type) {
    return defaultCacheGroupPolicy(type).policyFingerprint();
}

StageCacheGroupSnapshot makeGroupSnapshot(std::string    tag,
                                          CacheGroupType type,
                                          uint32_t       blocks,
                                          size_t         seq               = 4,
                                          size_t         kernel            = 4,
                                          bool           block_tree_member = true) {
    StageCacheGroupSnapshot group;
    group.tag                       = std::move(tag);
    group.type                      = type;
    group.block_tree_group_member   = block_tree_member;
    group.seq_size_per_block        = seq;
    group.kernel_seq_size_per_block = kernel;
    group.cache_key_token_stride    = seq;
    group.block_num                 = blocks;
    group.policy_fingerprint        = policyFingerprint(type);
    return group;
}

StageCacheSnapshot makeFullSnapshot(uint32_t blocks, size_t seq = 4, size_t kernel = 4, size_t combined_stride = 4096) {
    StageCacheSnapshot s;
    s.groups                           = {makeGroupSnapshot("full", CacheGroupType::FULL, blocks, seq, kernel)};
    s.block_tree_combined_stride_bytes = combined_stride;
    return s;
}

StageCacheSnapshot makeHybridSnapshot(uint32_t full_blocks, uint32_t linear_blocks, size_t combined_stride = 8192) {
    StageCacheSnapshot s;
    s.groups = {
        makeGroupSnapshot("full", CacheGroupType::FULL, full_blocks, 4, 2),
        makeGroupSnapshot("linear", CacheGroupType::LINEAR, linear_blocks),
    };
    s.block_tree_combined_stride_bytes = combined_stride;
    return s;
}

StageCacheSnapshot makeSameTypeSnapshot(std::vector<std::string> tags, uint32_t blocks) {
    StageCacheSnapshot snapshot;
    snapshot.groups.reserve(tags.size());
    for (auto& tag : tags) {
        snapshot.groups.push_back(makeGroupSnapshot(std::move(tag), CacheGroupType::FULL, blocks));
    }
    snapshot.block_tree_combined_stride_bytes = 4096;
    return snapshot;
}

std::vector<uint32_t> canonicalBlockNums(const PPValidationResult& result) {
    std::vector<uint32_t> nums;
    nums.reserve(result.canonical_groups.size());
    for (const auto& entry : result.canonical_groups) {
        nums.push_back(entry.logical_block_num);
    }
    return nums;
}

}  // namespace

TEST(PPTopologyValidatorTest, SingleStageTriviallyPasses) {
    auto result = validatePPTopology({makeFullSnapshot(42)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{42}));
    EXPECT_EQ(result.agreed.block_tree_combined_stride_bytes, 4096u);
}

TEST(PPTopologyValidatorTest, EmptyStagesTriviallyPasses) {
    auto result = validatePPTopology({});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.canonical_groups.empty());
}

TEST(PPTopologyValidatorTest, LeadingStageRequiresAtLeastOneCacheGroup) {
    auto result = validatePPTopology({StageCacheSnapshot{}});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("stage 0 cache snapshot has no cache groups"), std::string::npos);
}

TEST(PPTopologyValidatorTest, SameTypePartialOverlapFailsUntilPerRankProjectionIsImplemented) {
    auto result = validatePPTopology({makeSameTypeSnapshot({"csa_kv", "indexer_kv"}, 100),
                                      makeSameTypeSnapshot({"csa_kv"}, 90),
                                      makeSameTypeSnapshot({"indexer_kv"}, 80),
                                      makeSameTypeSnapshot({"indexer_kv", "csa_kv"}, 70)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("has 1 of 2 group-set members"), std::string::npos);
    EXPECT_NE(result.error.find("partial group-set overlap is not supported yet"), std::string::npos);
}

TEST(PPTopologyValidatorTest, SameTypeFullOverlapIgnoresLocalTagOrder) {
    auto result = validatePPTopology(
        {makeSameTypeSnapshot({"csa_kv", "indexer_kv"}, 100), makeSameTypeSnapshot({"indexer_kv", "csa_kv"}, 90)});
    ASSERT_TRUE(result.ok) << result.error;
}

TEST(PPTopologyValidatorTest, MissingGroupTypePassesFullOrEmptyGate) {
    auto result = validatePPTopology({makeHybridSnapshot(100, 50), makeFullSnapshot(90, 4, 2)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{90, 50}));
}

TEST(PPTopologyValidatorTest, ReorderedTagsAndSubsetKeepStage0Order) {
    auto reordered = makeHybridSnapshot(80, 30);
    std::swap(reordered.groups[0], reordered.groups[1]);

    auto result = validatePPTopology({makeHybridSnapshot(100, 50), reordered, makeFullSnapshot(70, 4, 2)});
    ASSERT_TRUE(result.ok) << result.error;
    ASSERT_EQ(result.canonical_groups.size(), 2u);
    EXPECT_EQ(result.canonical_groups[0].tag, "full");
    EXPECT_EQ(result.canonical_groups[1].tag, "linear");
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{70, 30}));
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 70u);
    EXPECT_EQ(result.agreed.block_num_overrides.at("linear"), 30u);
}

TEST(PPTopologyValidatorTest, NonStage0OwnedTagFails) {
    StageCacheSnapshot other = makeFullSnapshot(80);
    other.groups[0].tag      = "swa";
    auto result              = validatePPTopology({makeHybridSnapshot(100, 50), other});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("[swa]"), std::string::npos);
    EXPECT_NE(result.error.find("absent from stage 0"), std::string::npos);
    EXPECT_NE(result.error.find("must own every cache group"), std::string::npos);
}

TEST(PPTopologyValidatorTest, LinearWithoutFullPasses) {
    StageCacheSnapshot linear_only;
    linear_only.groups                           = {makeGroupSnapshot("linear", CacheGroupType::LINEAR, 50)};
    linear_only.block_tree_combined_stride_bytes = 4096;
    auto result                                  = validatePPTopology({makeHybridSnapshot(100, 50), linear_only});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{100, 50}));
}

TEST(PPTopologyValidatorTest, SharedTagTypeMismatchFails) {
    StageCacheSnapshot bad = makeHybridSnapshot(100, 50);
    bad.groups[0].type     = CacheGroupType::LINEAR;
    bad.groups[1].type     = CacheGroupType::FULL;

    auto result = validatePPTopology({makeHybridSnapshot(100, 50), bad});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("group [full] type differs"), std::string::npos);
}

TEST(PPTopologyValidatorTest, SeqSizeMismatchFails) {
    auto result = validatePPTopology({makeFullSnapshot(100, /*seq=*/4), makeFullSnapshot(100, /*seq=*/8)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("seq_size_per_block"), std::string::npos);
}

TEST(PPTopologyValidatorTest, KernelSeqSizeMismatchFails) {
    auto result = validatePPTopology({makeFullSnapshot(100, 4, /*kernel=*/4), makeFullSnapshot(100, 4, /*kernel=*/2)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("kernel_seq_size_per_block"), std::string::npos);
}

TEST(PPTopologyValidatorTest, CacheKeyTokenStrideMismatchFails) {
    auto bad                             = makeFullSnapshot(100);
    bad.groups[0].cache_key_token_stride = 8;
    auto result                          = validatePPTopology({makeFullSnapshot(100), bad});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("cache_key_token_stride"), std::string::npos);
}

TEST(PPTopologyValidatorTest, ZeroBlockNumFails) {
    auto result = validatePPTopology({makeFullSnapshot(100), makeFullSnapshot(0)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("0 KV blocks"), std::string::npos);
}

TEST(PPTopologyValidatorTest, DuplicateTagFails) {
    auto duplicate = makeSameTypeSnapshot({"full", "full"}, 100);
    auto result    = validatePPTopology({duplicate});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("duplicate cache group tag"), std::string::npos);
}

TEST(PPTopologyValidatorTest, PrefixReuseMismatchFails) {
    auto stage0                              = makeFullSnapshot(100);
    auto stage1                              = makeFullSnapshot(90);
    stage1.groups[0].block_tree_group_member = false;
    auto policy                              = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.enable_prefix_reuse               = false;
    stage1.groups[0].policy_fingerprint      = policy.policyFingerprint();

    auto result = validatePPTopology({stage0, stage1});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("policy"), std::string::npos);
}

TEST(PPTopologyValidatorTest, CaptureLocalStageBuildsSnapshot) {
    CacheConfig      cache_config;
    auto             spec = std::make_shared<MHAKVCacheSpec>("default", 4, 2, 1);
    std::vector<int> layer_ids(2);
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    cache_config.layer_num          = 2;
    cache_config.seq_size_per_block = 4;
    cache_config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});
    cache_config.finalizeBlockNums(42, RuntimeConfig{});

    auto snapshot = StageCacheSnapshot::captureLocalStage(cache_config, 4096);
    ASSERT_EQ(snapshot.groups.size(), 1u);
    EXPECT_EQ(snapshot.groups[0].tag, "default");
    EXPECT_EQ(snapshot.groups[0].type, CacheGroupType::FULL);
    EXPECT_TRUE(snapshot.groups[0].block_tree_group_member);
    EXPECT_EQ(snapshot.groups[0].cache_key_token_stride, 4u);
    EXPECT_EQ(snapshot.groups[0].block_num, 42u);
    EXPECT_EQ(snapshot.block_tree_combined_stride_bytes, 4096u);
}

class MockStageSnapshotCollector: public StageSnapshotCollector {
public:
    explicit MockStageSnapshotCollector(std::vector<StageCacheSnapshot> snapshots): snapshots_(std::move(snapshots)) {}

    std::vector<StageCacheSnapshot> collect() override {
        return snapshots_;
    }

private:
    std::vector<StageCacheSnapshot> snapshots_;
};

TEST(PPTopologyValidatorTest, InitGeometryMultiStageMockComputesMin) {
    MockStageSnapshotCollector collector({makeHybridSnapshot(100, 50), makeHybridSnapshot(90, 48)});
    auto                       result = initPPCacheGeometry(collector);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{90, 48}));
}

TEST(PPTopologyValidatorTest, CanonicalGroupsIdenticalStages) {
    auto result = validatePPTopology({makeHybridSnapshot(100, 50), makeHybridSnapshot(90, 55)});
    ASSERT_TRUE(result.ok) << result.error;
    ASSERT_EQ(result.canonical_groups.size(), 2u);
    EXPECT_EQ(result.canonical_groups[0].tag, "full");
    EXPECT_EQ(result.canonical_groups[0].logical_block_num, 90u);
    EXPECT_EQ(result.canonical_groups[0].type, CacheGroupType::FULL);
    EXPECT_EQ(result.canonical_groups[1].tag, "linear");
    EXPECT_EQ(result.canonical_groups[1].logical_block_num, 50u);
    EXPECT_EQ(result.canonical_groups[1].type, CacheGroupType::LINEAR);
    EXPECT_EQ(result.canonical_groups[1].seq_size_per_block, 4u);
    EXPECT_EQ(result.canonical_groups[1].kernel_seq_size_per_block, 4u);
    EXPECT_EQ(result.canonical_groups[1].cache_key_token_stride, 4u);
}

TEST(PPTopologyValidatorTest, SnapshotSerializeRoundTrip) {
    const auto original = makeHybridSnapshot(120, 45);
    const auto payload  = original.serialize();
    EXPECT_EQ(payload.compare(0, 3, "v1|"), 0);
    const auto decoded = StageCacheSnapshot::deserialize(payload);
    ASSERT_EQ(decoded.groups.size(), original.groups.size());
    for (size_t i = 0; i < original.groups.size(); ++i) {
        EXPECT_EQ(decoded.groups[i].tag, original.groups[i].tag);
        EXPECT_EQ(decoded.groups[i].type, original.groups[i].type);
        EXPECT_EQ(decoded.groups[i].block_tree_group_member, original.groups[i].block_tree_group_member);
        EXPECT_EQ(decoded.groups[i].seq_size_per_block, original.groups[i].seq_size_per_block);
        EXPECT_EQ(decoded.groups[i].kernel_seq_size_per_block, original.groups[i].kernel_seq_size_per_block);
        EXPECT_EQ(decoded.groups[i].cache_key_token_stride, original.groups[i].cache_key_token_stride);
        EXPECT_EQ(decoded.groups[i].block_num, original.groups[i].block_num);
        EXPECT_EQ(decoded.groups[i].explicit_block_num, original.groups[i].explicit_block_num);
        EXPECT_EQ(decoded.groups[i].policy_fingerprint, original.groups[i].policy_fingerprint);
    }
    EXPECT_EQ(decoded.block_tree_combined_stride_bytes, original.block_tree_combined_stride_bytes);

    EXPECT_THROW(StageCacheSnapshot::deserialize(""), std::exception);
    EXPECT_THROW(StageCacheSnapshot::deserialize("v2|a|1|4|4|4|1|0"), std::exception);
    EXPECT_THROW(StageCacheSnapshot::deserialize("v9|full|1|1|4|2|4|1|0|4096|p"), std::exception);
    EXPECT_THROW(StageCacheSnapshot::deserialize("v1|full|1|1|4|2|4|1,2|0|4096|p"), std::exception);
}

TEST(PPTopologyValidatorTest, SwaGroupPasses) {
    auto make = [](uint32_t full_blocks, uint32_t swa_blocks) {
        StageCacheSnapshot snapshot = makeFullSnapshot(full_blocks);
        snapshot.groups.push_back(makeGroupSnapshot("swa", CacheGroupType::SWA, swa_blocks, 4, 4, false));
        return snapshot;
    };

    auto result = validatePPTopology({make(50, 25), make(40, 20)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{40, 20}));
}

TEST(PPTopologyValidatorTest, ExplicitBlockNumMismatchFails) {
    StageCacheSnapshot a           = makeFullSnapshot(100);
    a.groups[0].explicit_block_num = 256;
    StageCacheSnapshot b           = makeFullSnapshot(100);
    b.groups[0].explicit_block_num = 512;

    auto result = validatePPTopology({a, b});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("explicit_block_num"), std::string::npos);
}

TEST(PPTopologyValidatorTest, PolicyFingerprintCoversSamePolicyFields) {
    CacheGroupPolicy base;
    const auto       base_fp = base.policyFingerprint();
    EXPECT_EQ(base_fp, base.policyFingerprint());
    auto expect_different = [&](const CacheGroupPolicy& changed) {
        EXPECT_FALSE(CacheConfig::samePolicy(base, changed));
        EXPECT_NE(base_fp, changed.policyFingerprint());
    };

    CacheGroupPolicy changed = base;
    changed.group_type       = CacheGroupType::SWA;
    expect_different(changed);
    changed                     = base;
    changed.enable_prefix_reuse = false;
    expect_different(changed);
    changed            = base;
    changed.reservable = false;
    expect_different(changed);
    changed                    = base;
    changed.explicit_block_num = 7;
    expect_different(changed);
    changed                        = base;
    changed.charge_to_paged_budget = true;
    expect_different(changed);
    changed                  = base;
    changed.memory_placement = CacheMemoryPlacement::HOST_PINNED;
    expect_different(changed);
    changed                    = base;
    changed.active_tail_blocks = 3;
    expect_different(changed);
    changed                      = base;
    changed.validate_tail_blocks = false;
    expect_different(changed);
    changed            = base;
    changed.cp_mapping = CpBlockMappingMode::COMPACT_LAST_RANK;
    expect_different(changed);
    changed          = base;
    changed.cp_slice = CpBlockSliceMode::EQUAL_BYTES;
    expect_different(changed);
    changed                     = base;
    changed.sliding_window_size = 32;
    expect_different(changed);
}

TEST(PPTopologyValidatorTest, AgreedInputsReducedFromCanonicalTable) {
    auto result = validatePPTopology({makeHybridSnapshot(100, 50, 8192), makeHybridSnapshot(90, 55, 12288)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 90u);
    EXPECT_EQ(result.agreed.block_num_overrides.at("linear"), 50u);
    EXPECT_EQ(result.agreed.paged_block_num, 50u);
    EXPECT_EQ(result.agreed.block_tree_combined_stride_bytes, 12288u);
}

TEST(PPTopologyValidatorTest, AgreedInputsDecoupleExplicitPools) {
    auto make = [](uint32_t full_blocks) {
        StageCacheSnapshot s           = makeHybridSnapshot(full_blocks, 50);
        s.groups[1].explicit_block_num = 50;
        auto policy                    = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
        policy.explicit_block_num      = 50;
        s.groups[1].policy_fingerprint = policy.policyFingerprint();
        return s;
    };
    auto result = validatePPTopology({make(100), make(80)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 80u);
    EXPECT_EQ(result.agreed.block_num_overrides.at("linear"), 50u);
    EXPECT_EQ(result.agreed.paged_block_num, 80u);
}

}  // namespace rtp_llm
