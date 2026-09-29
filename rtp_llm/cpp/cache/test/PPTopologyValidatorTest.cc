#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"

using namespace std;

namespace rtp_llm {

namespace {

std::string policyFingerprint(CacheGroupType type) {
    return cacheGroupPolicyFingerprint(defaultCacheGroupPolicy(type));
}

StageCacheSnapshot makeFullSnapshot(uint32_t blocks, size_t seq = 4, size_t kernel = 4) {
    StageCacheSnapshot s;
    s.group_tags                = {"full"};
    s.group_types               = {CacheGroupType::FULL};
    s.seq_size_per_block        = {seq};
    s.kernel_seq_size_per_block = {kernel};
    s.cache_key_token_strides   = {seq};
    s.block_nums                = {blocks};
    s.explicit_block_nums       = {0};
    s.policy_fingerprints       = {policyFingerprint(CacheGroupType::FULL)};
    return s;
}

StageCacheSnapshot makeHybridSnapshot(uint32_t full_blocks, uint32_t linear_blocks) {
    StageCacheSnapshot s;
    s.group_tags                = {"full", "linear"};
    s.group_types               = {CacheGroupType::FULL, CacheGroupType::LINEAR};
    s.seq_size_per_block        = {4, 4};
    s.kernel_seq_size_per_block = {2, 4};
    s.cache_key_token_strides   = {4, 4};
    s.block_nums                = {full_blocks, linear_blocks};
    s.explicit_block_nums       = {0, 0};
    s.policy_fingerprints       = {policyFingerprint(CacheGroupType::FULL),
                                   policyFingerprint(CacheGroupType::LINEAR)};
    return s;
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

TEST(PPTopologyValidatorTest, AllIdenticalPasses) {
    auto stages = {makeHybridSnapshot(100, 50), makeHybridSnapshot(90, 55), makeHybridSnapshot(95, 50)};
    auto result = validatePPTopology(stages);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{90, 50}));
}

TEST(PPTopologyValidatorTest, SingleStageTriviallyPasses) {
    auto result = validatePPTopology({makeFullSnapshot(42)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{42}));
}

TEST(PPTopologyValidatorTest, EmptyStagesTriviallyPasses) {
    auto result = validatePPTopology({});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.canonical_groups.empty());
}

TEST(PPTopologyValidatorTest, TagSubsetPassesUnderPairing) {
    StageCacheSnapshot subset = makeFullSnapshot(70, /*seq=*/4, /*kernel=*/2);  /** only the full group */
    auto               result = validatePPTopology({makeHybridSnapshot(100, 50), subset});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{70, 50}));
}

TEST(PPTopologyValidatorTest, ReorderedTagsAndSubsetKeepStage0Order) {
    auto reordered = makeHybridSnapshot(80, 30);
    std::swap(reordered.group_tags[0], reordered.group_tags[1]);
    std::swap(reordered.group_types[0], reordered.group_types[1]);
    std::swap(reordered.seq_size_per_block[0], reordered.seq_size_per_block[1]);
    std::swap(reordered.kernel_seq_size_per_block[0], reordered.kernel_seq_size_per_block[1]);
    std::swap(reordered.cache_key_token_strides[0], reordered.cache_key_token_strides[1]);
    std::swap(reordered.block_nums[0], reordered.block_nums[1]);
    std::swap(reordered.explicit_block_nums[0], reordered.explicit_block_nums[1]);
    std::swap(reordered.policy_fingerprints[0], reordered.policy_fingerprints[1]);

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
    other.group_tags         = {"swa"};
    auto result              = validatePPTopology({makeHybridSnapshot(100, 50), other});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("absent from stage 0"), std::string::npos);
    EXPECT_NE(result.error.find("must own every cache group"), std::string::npos);
}

TEST(PPTopologyValidatorTest, LinearWithoutFullPasses) {
    StageCacheSnapshot linear_only;
    linear_only.group_tags                = {"linear"};
    linear_only.group_types               = {CacheGroupType::LINEAR};
    linear_only.seq_size_per_block        = {4};
    linear_only.kernel_seq_size_per_block = {4};
    linear_only.cache_key_token_strides   = {4};
    linear_only.block_nums                = {50};
    linear_only.explicit_block_nums       = {0};
    linear_only.policy_fingerprints       = {policyFingerprint(CacheGroupType::LINEAR)};
    auto result                           = validatePPTopology({makeHybridSnapshot(100, 50), linear_only});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{100, 50}));
}

TEST(PPTopologyValidatorTest, PairingStillChecksSharedGeometry) {
    StageCacheSnapshot subset = makeFullSnapshot(80, /*seq=*/8);  /** seq differs */
    auto               result = validatePPTopology({makeHybridSnapshot(100, 50), subset});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("seq_size_per_block"), std::string::npos);
}

TEST(PPTopologyValidatorTest, SharedTagTypeMismatchFails) {
    StageCacheSnapshot bad = makeHybridSnapshot(100, 50);
    bad.group_types        = {CacheGroupType::LINEAR, CacheGroupType::FULL};  /** swapped */

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
    auto bad                       = makeFullSnapshot(100);
    bad.cache_key_token_strides[0] = 8;
    auto result                    = validatePPTopology({makeFullSnapshot(100), bad});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("cache_key_token_stride"), std::string::npos);
}

TEST(PPTopologyValidatorTest, CapacitySkewUsesOwnerMinimum) {
    auto result = validatePPTopology({makeFullSnapshot(100), makeFullSnapshot(10)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{10}));
}

TEST(PPTopologyValidatorTest, ZeroBlockNumFails) {
    auto result = validatePPTopology({makeFullSnapshot(100), makeFullSnapshot(0)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("0 KV blocks"), std::string::npos);
}

TEST(PPTopologyValidatorTest, InternallyInconsistentFails) {
    StageCacheSnapshot bad = makeFullSnapshot(100);
    bad.block_nums         = {100, 90};  /** one extra element */
    auto result            = validatePPTopology({makeFullSnapshot(100), bad});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("internally inconsistent"), std::string::npos);
}

TEST(PPTopologyValidatorTest, FromConfigBuildsSnapshot) {
    CacheConfig cache_config;
    auto        spec = std::make_shared<MHAKVCacheSpec>("default", 4, 2, 1);
    std::vector<int> layer_ids(2);
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    cache_config.layer_num          = 2;
    cache_config.seq_size_per_block = 4;
    cache_config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});
    cache_config.finalizeBlockNums(42, RuntimeConfig{});

    auto snapshot = StageCacheSnapshot::fromConfig(cache_config);
    ASSERT_TRUE(snapshot.internallyConsistent());
    EXPECT_EQ(snapshot.group_tags, (std::vector<std::string>{"default"}));
    EXPECT_EQ(snapshot.group_types, (std::vector<CacheGroupType>{CacheGroupType::FULL}));
    EXPECT_EQ(snapshot.cache_key_token_strides, (std::vector<size_t>{4}));
    EXPECT_EQ(snapshot.block_nums, (std::vector<uint32_t>{42}));
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

TEST(PPTopologyValidatorTest, InitGeometryLocalCollectorPasses) {
    LocalStageSnapshotCollector collector(makeFullSnapshot(64));
    auto                        result = initPPCacheGeometry(collector);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{64}));
}

TEST(PPTopologyValidatorTest, InitGeometryMultiStageMockComputesMin) {
    MockStageSnapshotCollector collector({makeHybridSnapshot(100, 50), makeHybridSnapshot(90, 48)});
    auto                       result = initPPCacheGeometry(collector);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{90, 48}));
}

TEST(PPTopologyValidatorTest, InitGeometryAcceptsCapacitySkew) {
    MockStageSnapshotCollector collector({makeFullSnapshot(100), makeFullSnapshot(10)});
    auto                       result = initPPCacheGeometry(collector);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{10}));
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

TEST(PPTopologyValidatorTest, LaterStageOnlyTagFails) {
    auto stage1 = makeHybridSnapshot(90, 55);
    stage1.group_tags.push_back("linear1");
    stage1.group_types.push_back(CacheGroupType::LINEAR);
    stage1.seq_size_per_block.push_back(4);
    stage1.kernel_seq_size_per_block.push_back(4);
    stage1.cache_key_token_strides.push_back(4);
    stage1.block_nums.push_back(30);
    stage1.explicit_block_nums.push_back(0);
    stage1.policy_fingerprints.push_back(policyFingerprint(CacheGroupType::LINEAR));

    auto result = validatePPTopology({makeHybridSnapshot(100, 50), stage1});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("[linear1]"), std::string::npos);
    EXPECT_NE(result.error.find("absent from stage 0"), std::string::npos);
}

TEST(PPTopologyValidatorTest, CanonicalGroupsRejectNonStage0TypeConflict) {
    auto stage0 = makeFullSnapshot(100);
    stage0.group_tags.push_back("x");
    stage0.group_types.push_back(CacheGroupType::FULL);
    stage0.seq_size_per_block.push_back(4);
    stage0.kernel_seq_size_per_block.push_back(4);
    stage0.cache_key_token_strides.push_back(4);
    stage0.block_nums.push_back(40);
    stage0.explicit_block_nums.push_back(0);
    stage0.policy_fingerprints.push_back(policyFingerprint(CacheGroupType::FULL));

    auto stage1 = makeFullSnapshot(90);
    stage1.group_tags.push_back("x");
    stage1.group_types.push_back(CacheGroupType::FULL);
    stage1.seq_size_per_block.push_back(4);
    stage1.kernel_seq_size_per_block.push_back(4);
    stage1.cache_key_token_strides.push_back(4);
    stage1.block_nums.push_back(40);
    stage1.explicit_block_nums.push_back(0);
    stage1.policy_fingerprints.push_back(policyFingerprint(CacheGroupType::FULL));

    auto stage2 = makeFullSnapshot(95);
    stage2.group_tags.push_back("x");
    stage2.group_types.push_back(CacheGroupType::LINEAR);
    stage2.seq_size_per_block.push_back(4);
    stage2.kernel_seq_size_per_block.push_back(4);
    stage2.cache_key_token_strides.push_back(4);
    stage2.block_nums.push_back(40);
    stage2.explicit_block_nums.push_back(0);
    stage2.policy_fingerprints.push_back(policyFingerprint(CacheGroupType::LINEAR));

    auto result = validatePPTopology({stage0, stage1, stage2});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("group [x] type differs"), std::string::npos);
}

TEST(PPTopologyValidatorTest, CanonicalGroupsAcceptNonStage0Skew) {
    auto stage0 = makeHybridSnapshot(100, 100);
    auto stage1 = makeHybridSnapshot(100, 100);
    auto stage2 = makeHybridSnapshot(90, 10);

    auto result = validatePPTopology({stage0, stage1, stage2});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{90, 10}));
}

TEST(PPTopologyValidatorTest, CanonicalGroupsSingleStage) {
    auto result = validatePPTopology({makeHybridSnapshot(70, 30)});
    ASSERT_TRUE(result.ok) << result.error;
    ASSERT_EQ(result.canonical_groups.size(), 2u);
    EXPECT_EQ(result.canonical_groups[0].tag, "full");
    EXPECT_EQ(result.canonical_groups[0].logical_block_num, 70u);
    EXPECT_EQ(result.canonical_groups[1].tag, "linear");
    EXPECT_EQ(result.canonical_groups[1].logical_block_num, 30u);
}

TEST(PPTopologyValidatorTest, SnapshotSerializeRoundTrip) {
    const auto original = makeHybridSnapshot(120, 45);
    const auto payload  = original.serialize();
    const auto decoded  = StageCacheSnapshot::deserialize(payload);
    EXPECT_EQ(decoded.group_tags, original.group_tags);
    EXPECT_EQ(decoded.group_types, original.group_types);
    EXPECT_EQ(decoded.seq_size_per_block, original.seq_size_per_block);
    EXPECT_EQ(decoded.kernel_seq_size_per_block, original.kernel_seq_size_per_block);
    EXPECT_EQ(decoded.cache_key_token_strides, original.cache_key_token_strides);
    EXPECT_EQ(decoded.block_nums, original.block_nums);
    EXPECT_EQ(decoded.explicit_block_nums, original.explicit_block_nums);
    EXPECT_EQ(decoded.policy_fingerprints, original.policy_fingerprints);

    EXPECT_THROW(StageCacheSnapshot::deserialize(""), std::exception);
    EXPECT_THROW(StageCacheSnapshot::deserialize("v2|a|1|4|4|4|1|0"), std::exception);
    EXPECT_THROW(StageCacheSnapshot::deserialize("v9|full|1|4|2|4|1|0|p"), std::exception);
    EXPECT_THROW(StageCacheSnapshot::deserialize("v2|full|1|4|2|4|1,2|0|p"), std::exception);
}

TEST(PPTopologyValidatorTest, SwaGroupPasses) {
    auto make = [](uint32_t full_blocks, uint32_t swa_blocks) {
        StageCacheSnapshot snapshot = makeFullSnapshot(full_blocks);
        snapshot.group_tags.push_back("swa");
        snapshot.group_types.push_back(CacheGroupType::SWA);
        snapshot.seq_size_per_block.push_back(4);
        snapshot.kernel_seq_size_per_block.push_back(4);
        snapshot.cache_key_token_strides.push_back(4);
        snapshot.block_nums.push_back(swa_blocks);
        snapshot.explicit_block_nums.push_back(0);
        snapshot.policy_fingerprints.push_back(policyFingerprint(CacheGroupType::SWA));
        return snapshot;
    };

    auto result = validatePPTopology({make(50, 25), make(40, 20)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{40, 20}));
}

TEST(PPTopologyValidatorTest, ExplicitBlockNumMismatchFails) {
    StageCacheSnapshot a  = makeFullSnapshot(100);
    a.explicit_block_nums = {256};
    StageCacheSnapshot b  = makeFullSnapshot(100);
    b.explicit_block_nums = {512};

    auto result = validatePPTopology({a, b});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("explicit_block_num"), std::string::npos);
}

TEST(PPTopologyValidatorTest, ExplicitBlockNumAgreementPasses) {
    StageCacheSnapshot a  = makeFullSnapshot(256);
    a.explicit_block_nums = {256};
    StageCacheSnapshot b  = makeFullSnapshot(256);
    b.explicit_block_nums = {256};

    auto result = validatePPTopology({a, b});
    ASSERT_TRUE(result.ok) << result.error;
    ASSERT_EQ(result.canonical_groups.size(), 1u);
    EXPECT_EQ(result.canonical_groups[0].explicit_block_num, 256u);
    EXPECT_EQ(result.canonical_groups[0].logical_block_num, 256u);
}

TEST(PPTopologyValidatorTest, PolicyFingerprintCoversSamePolicyFields) {
    CacheGroupPolicy base;
    const auto       base_fp = cacheGroupPolicyFingerprint(base);
    EXPECT_EQ(base_fp, cacheGroupPolicyFingerprint(base));

    CacheGroupPolicy changed = base;
    changed.memory_placement = CacheMemoryPlacement::HOST_PINNED;
    EXPECT_NE(base_fp, cacheGroupPolicyFingerprint(changed));
    changed                     = base;
    changed.enable_prefix_reuse = false;
    EXPECT_NE(base_fp, cacheGroupPolicyFingerprint(changed));
    changed                    = base;
    changed.active_tail_blocks = 3;
    EXPECT_NE(base_fp, cacheGroupPolicyFingerprint(changed));
    changed                     = base;
    changed.sliding_window_size = 32;
    EXPECT_NE(base_fp, cacheGroupPolicyFingerprint(changed));
}

TEST(PPTopologyValidatorTest, PolicyMismatchFails) {
    StageCacheSnapshot a  = makeFullSnapshot(100);
    StageCacheSnapshot b  = makeFullSnapshot(100);
    b.policy_fingerprints = {"different-policy"};

    auto result = validatePPTopology({a, b});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("policy"), std::string::npos);
}

TEST(PPTopologyValidatorTest, AgreedInputsReducedFromCanonicalTable) {
    auto result = validatePPTopology({makeHybridSnapshot(100, 50), makeHybridSnapshot(90, 55)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 90u);
    EXPECT_EQ(result.agreed.block_num_overrides.at("linear"), 50u);
    EXPECT_EQ(result.agreed.paged_block_num, 50u);
}

TEST(PPTopologyValidatorTest, AgreedInputsDecoupleExplicitPools) {
    auto make = [](uint32_t full_blocks) {
        StageCacheSnapshot s     = makeHybridSnapshot(full_blocks, 50);
        s.explicit_block_nums[1] = 50;
        auto policy               = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
        policy.explicit_block_num = 50;
        s.policy_fingerprints[1]  = cacheGroupPolicyFingerprint(policy);
        return s;
    };
    auto result = validatePPTopology({make(100), make(80)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 80u);
    EXPECT_EQ(result.agreed.block_num_overrides.at("linear"), 50u);
    EXPECT_EQ(result.agreed.paged_block_num, 80u);
}

}  // namespace rtp_llm
