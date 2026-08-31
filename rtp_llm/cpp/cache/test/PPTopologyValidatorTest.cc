// Unit tests for validatePPTopology — startup-time cache geometry validation
// across PP stages (rejects bad partitions, computes per-group logical block
// count as the min across stages).

#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"

using namespace std;

namespace rtp_llm {

namespace {

// Single-FULL-group snapshot; tweak fields per test case.
StageCacheSnapshot makeFullSnapshot(uint32_t blocks, size_t seq = 4, size_t kernel = 4) {
    StageCacheSnapshot s;
    s.group_tags                = {"full"};
    s.group_types               = {CacheGroupType::FULL};
    s.seq_size_per_block        = {seq};
    s.kernel_seq_size_per_block = {kernel};
    s.block_nums                = {blocks};
    s.explicit_block_nums       = {0};
    s.policy_fingerprints       = {"t0:r1:e0:v1:x0:c0:p0:a0:w1:m0:s0"};
    return s;
}

// Two-group (FULL + LINEAR) snapshot, Kimi3-style hybrid.
StageCacheSnapshot makeHybridSnapshot(uint32_t full_blocks, uint32_t linear_blocks) {
    StageCacheSnapshot s;
    s.group_tags                = {"full", "linear"};
    s.group_types               = {CacheGroupType::FULL, CacheGroupType::LINEAR};
    s.seq_size_per_block        = {4, 4};
    s.kernel_seq_size_per_block = {2, 4};
    s.block_nums                = {full_blocks, linear_blocks};
    s.explicit_block_nums       = {0, 0};
    s.policy_fingerprints       = {"t0:r1:e0:v1:x0:c0:p0:a0:w1:m0:s0", "t1:r1:e0:v1:x0:c0:p0:a0:w1:m0:s0"};
    return s;
}

// Logical block counts in canonical table order.
std::vector<uint32_t> canonicalBlockNums(const PPValidationResult& result) {
    std::vector<uint32_t> nums;
    nums.reserve(result.canonical_groups.size());
    for (const auto& entry : result.canonical_groups) {
        nums.push_back(entry.logical_block_num);
    }
    return nums;
}

}  // namespace

// Case 1: all stages identical -> pass, per-group min computed.
TEST(PPTopologyValidatorTest, AllIdenticalPasses) {
    auto stages = {makeHybridSnapshot(100, 50), makeHybridSnapshot(90, 55), makeHybridSnapshot(95, 50)};
    auto result = validatePPTopology(stages);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{90, 50}));
}

// Case 2: single stage (pp_size=1) -> trivially pass, degenerates to today.
TEST(PPTopologyValidatorTest, SingleStageTriviallyPasses) {
    auto result = validatePPTopology({makeFullSnapshot(42)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{42}));
}

// Case 3: empty input -> trivially pass with empty result.
TEST(PPTopologyValidatorTest, EmptyStagesTriviallyPasses) {
    auto result = validatePPTopology({});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.canonical_groups.empty());
}

// Case 4: a stage holding a tag subset is legitimate under pairing semantics
// (stage-scoped topologies own tag subsets) -> passes; min is taken only
// over stages owning each tag.
TEST(PPTopologyValidatorTest, TagSubsetPassesUnderPairing) {
    StageCacheSnapshot subset = makeFullSnapshot(70, /*seq=*/4, /*kernel=*/2);  // only the full group
    auto               result = validatePPTopology({makeHybridSnapshot(100, 50), subset});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{70, 50}));
}

// Case 4b: stage-0-superset rule: any stage owning a group that stage 0
// does not own is rejected (the leading stage must physically own every
// cache group, since it issues all block ids from its own pools).
TEST(PPTopologyValidatorTest, NonStage0OwnedTagFails) {
    StageCacheSnapshot other = makeFullSnapshot(80);
    other.group_tags         = {"swa"};
    auto result              = validatePPTopology({makeHybridSnapshot(100, 50), other});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("absent from stage 0"), std::string::npos);
    EXPECT_NE(result.error.find("must own every cache group"), std::string::npos);
}

// Case 4c: hybrid stage (LINEAR groups) without any FULL group -> reject.
TEST(PPTopologyValidatorTest, LinearWithoutFullFails) {
    StageCacheSnapshot linear_only;
    linear_only.group_tags                = {"linear"};
    linear_only.group_types               = {CacheGroupType::LINEAR};
    linear_only.seq_size_per_block        = {4};
    linear_only.kernel_seq_size_per_block = {4};
    linear_only.block_nums                = {50};
    linear_only.explicit_block_nums       = {0};
    linear_only.policy_fingerprints       = {"t1:r1:e0:v1:x0:c0:p0:a0:w1:m0:s0"};
    auto result                           = validatePPTopology({makeHybridSnapshot(100, 50), linear_only});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("no FULL group"), std::string::npos);
}

// Case 4d: the pairing path still compares geometry of shared tags.
TEST(PPTopologyValidatorTest, PairingStillChecksSharedGeometry) {
    StageCacheSnapshot subset = makeFullSnapshot(80, /*seq=*/8);  // seq differs
    auto               result = validatePPTopology({makeHybridSnapshot(100, 50), subset});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("seq_size_per_block"), std::string::npos);
}

// Case 5: same tags but swapped type sequence -> reject.
TEST(PPTopologyValidatorTest, TypeSequenceMismatchFails) {
    StageCacheSnapshot bad = makeHybridSnapshot(100, 50);
    bad.group_types        = {CacheGroupType::LINEAR, CacheGroupType::FULL};  // swapped

    auto result = validatePPTopology({makeHybridSnapshot(100, 50), bad});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("type sequence"), std::string::npos);
}

// Case 6: seq_size_per_block differs -> reject.
TEST(PPTopologyValidatorTest, SeqSizeMismatchFails) {
    auto result = validatePPTopology({makeFullSnapshot(100, /*seq=*/4), makeFullSnapshot(100, /*seq=*/8)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("seq_size_per_block"), std::string::npos);
}

// Case 7: kernel_seq_size_per_block differs -> reject.
TEST(PPTopologyValidatorTest, KernelSeqSizeMismatchFails) {
    auto result = validatePPTopology({makeFullSnapshot(100, 4, /*kernel=*/4), makeFullSnapshot(100, 4, /*kernel=*/2)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("kernel_seq_size_per_block"), std::string::npos);
}

// Case 8: capacity skew above threshold -> reject (bad layer split).
TEST(PPTopologyValidatorTest, CapacitySkewTooLargeFails) {
    // 100 / 60 = 1.67 > 1.5
    auto result = validatePPTopology({makeFullSnapshot(100), makeFullSnapshot(60)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("skew too large"), std::string::npos);
}

// Case 9: capacity skew within threshold -> pass with min.
TEST(PPTopologyValidatorTest, CapacitySkewWithinThresholdPasses) {
    // 100 / 80 = 1.25 <= 1.5
    auto result = validatePPTopology({makeFullSnapshot(100), makeFullSnapshot(80)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{80}));
}

// Case 10: some stage has 0 blocks for a group -> reject (cannot allocate).
TEST(PPTopologyValidatorTest, ZeroBlockNumFails) {
    auto result = validatePPTopology({makeFullSnapshot(100), makeFullSnapshot(0)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("0 KV blocks"), std::string::npos);
}

// Case 11: internally inconsistent snapshot -> reject.
TEST(PPTopologyValidatorTest, InternallyInconsistentFails) {
    StageCacheSnapshot bad = makeFullSnapshot(100);
    bad.block_nums         = {100, 90};  // one extra element
    auto result            = validatePPTopology({makeFullSnapshot(100), bad});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("internally inconsistent"), std::string::npos);
}

// Case 12: fromConfig builds the snapshot from a real CacheConfig.
TEST(PPTopologyValidatorTest, FromConfigBuildsSnapshot) {
    CacheConfig cache_config;
    cache_config.block_num = 42;
    auto spec              = std::make_shared<MHAKVCacheSpec>();
    spec->tag              = "default";
    std::vector<int> layer_ids(2);
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    cache_config.layer_num     = 2;
    cache_config.layer_all_num = 2;
    cache_config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});

    auto snapshot = StageCacheSnapshot::fromConfig(cache_config);
    ASSERT_TRUE(snapshot.internallyConsistent());
    EXPECT_EQ(snapshot.group_tags, (std::vector<std::string>{"default"}));
    EXPECT_EQ(snapshot.group_types, (std::vector<CacheGroupType>{CacheGroupType::FULL}));
    ASSERT_EQ(snapshot.block_nums.size(), 1u);
    EXPECT_EQ(snapshot.block_nums[0], 42u);
}

// Mock collector returning a fixed list of snapshots (stands in for the
// real startup exchange).
class MockStageSnapshotCollector: public StageSnapshotCollector {
public:
    explicit MockStageSnapshotCollector(std::vector<StageCacheSnapshot> snapshots): snapshots_(std::move(snapshots)) {}

    std::vector<StageCacheSnapshot> collect() override {
        return snapshots_;
    }

private:
    std::vector<StageCacheSnapshot> snapshots_;
};

// Case 13: initPPCacheGeometry with the local single-stage collector
// (pp_size=1 path) -> trivially passes with the stage's own block count.
TEST(PPTopologyValidatorTest, InitGeometryLocalCollectorPasses) {
    LocalStageSnapshotCollector collector(makeFullSnapshot(64));
    auto                        result = initPPCacheGeometry(collector);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{64}));
}

// Case 14: initPPCacheGeometry with a multi-stage mock collector -> per-group min.
TEST(PPTopologyValidatorTest, InitGeometryMultiStageMockComputesMin) {
    MockStageSnapshotCollector collector({makeHybridSnapshot(100, 50), makeHybridSnapshot(90, 48)});
    auto                       result = initPPCacheGeometry(collector);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{90, 48}));
}

// Case 15: initPPCacheGeometry propagates validation failure from the collector data.
TEST(PPTopologyValidatorTest, InitGeometryFailsOnMismatch) {
    MockStageSnapshotCollector collector({makeFullSnapshot(100), makeFullSnapshot(10)});  // skew 10x
    auto                       result = initPPCacheGeometry(collector);
    ASSERT_FALSE(result.ok);
    EXPECT_TRUE(result.canonical_groups.empty());
}

// Case 17: canonical group table for identical stages follows stage-0 order
// and carries the per-tag logical minimum.
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
}

// Case 18: a tag owned only by a later stage is rejected by the stage-0
// superset rule (bookkeeping-only allocation is not supported).
TEST(PPTopologyValidatorTest, LaterStageOnlyTagFails) {
    auto stage1 = makeHybridSnapshot(90, 55);
    stage1.group_tags.push_back("linear1");
    stage1.group_types.push_back(CacheGroupType::LINEAR);
    stage1.seq_size_per_block.push_back(4);
    stage1.kernel_seq_size_per_block.push_back(4);
    stage1.block_nums.push_back(30);
    stage1.explicit_block_nums.push_back(0);
    stage1.policy_fingerprints.push_back("t0:r1:e0:v1:x0:c0:p0:a0:w1:m0:s0");

    auto result = validatePPTopology({makeHybridSnapshot(100, 50), stage1});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("[linear1]"), std::string::npos);
    EXPECT_NE(result.error.find("absent from stage 0"), std::string::npos);
}

// Case 19: same tag reported with different types by two non-stage-0 stages
// is rejected by the canonical consistency check.
TEST(PPTopologyValidatorTest, CanonicalGroupsRejectNonStage0TypeConflict) {
    auto stage0 = makeFullSnapshot(100);
    stage0.group_tags.push_back("x");
    stage0.group_types.push_back(CacheGroupType::FULL);
    stage0.seq_size_per_block.push_back(4);
    stage0.kernel_seq_size_per_block.push_back(4);
    stage0.block_nums.push_back(40);
    stage0.explicit_block_nums.push_back(0);
    stage0.policy_fingerprints.push_back("t0:r1:e0:v1:x0:c0:p0:a0:w1:m0:s0");

    auto stage1 = makeFullSnapshot(90);
    stage1.group_tags.push_back("x");
    stage1.group_types.push_back(CacheGroupType::FULL);
    stage1.seq_size_per_block.push_back(4);
    stage1.kernel_seq_size_per_block.push_back(4);
    stage1.block_nums.push_back(40);
    stage1.explicit_block_nums.push_back(0);
    stage1.policy_fingerprints.push_back("t0:r1:e0:v1:x0:c0:p0:a0:w1:m0:s0");

    auto stage2 = makeFullSnapshot(95);
    stage2.group_tags.push_back("x");
    stage2.group_types.push_back(CacheGroupType::LINEAR);
    stage2.seq_size_per_block.push_back(4);
    stage2.kernel_seq_size_per_block.push_back(4);
    stage2.block_nums.push_back(40);
    stage2.explicit_block_nums.push_back(0);
    stage2.policy_fingerprints.push_back("t0:r1:e0:v1:x0:c0:p0:a0:w1:m0:s0");

    auto result = validatePPTopology({stage0, stage1, stage2});
    ASSERT_FALSE(result.ok);
    // Identical tag sets elevate to the strict path, where the divergent
    // type sequence is caught first.
    EXPECT_NE(result.error.find("type sequence differs from stage 0"), std::string::npos);
}

// Case 20: capacity skew on a tag owned by all stages is caught at startup
// instead of overrunning the smaller owner's pool at runtime.
TEST(PPTopologyValidatorTest, CanonicalGroupsRejectNonStage0Skew) {
    auto stage0 = makeFullSnapshot(100);
    stage0.group_tags.push_back("y");
    stage0.group_types.push_back(CacheGroupType::LINEAR);
    stage0.seq_size_per_block.push_back(4);
    stage0.kernel_seq_size_per_block.push_back(4);
    stage0.block_nums.push_back(100);
    stage0.explicit_block_nums.push_back(0);
    stage0.policy_fingerprints.push_back("t0:r1:e0:v1:x0:c0:p0:a0:w1:m0:s0");

    auto stage1 = makeFullSnapshot(100);
    stage1.group_tags.push_back("y");
    stage1.group_types.push_back(CacheGroupType::LINEAR);
    stage1.seq_size_per_block.push_back(4);
    stage1.kernel_seq_size_per_block.push_back(4);
    stage1.block_nums.push_back(100);
    stage1.explicit_block_nums.push_back(0);
    stage1.policy_fingerprints.push_back("t0:r1:e0:v1:x0:c0:p0:a0:w1:m0:s0");

    auto stage2 = makeFullSnapshot(90);
    stage2.group_tags.push_back("y");
    stage2.group_types.push_back(CacheGroupType::LINEAR);
    stage2.seq_size_per_block.push_back(4);
    stage2.kernel_seq_size_per_block.push_back(4);
    stage2.block_nums.push_back(10);
    stage2.explicit_block_nums.push_back(0);
    stage2.policy_fingerprints.push_back("t0:r1:e0:v1:x0:c0:p0:a0:w1:m0:s0");

    auto result = validatePPTopology({stage0, stage1, stage2});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("skew too large"), std::string::npos);
    EXPECT_NE(result.error.find("[y]"), std::string::npos);
}

// Case 21: pp_size=1 canonical table equals the stage's own groups.
TEST(PPTopologyValidatorTest, CanonicalGroupsSingleStage) {
    auto result = validatePPTopology({makeHybridSnapshot(70, 30)});
    ASSERT_TRUE(result.ok) << result.error;
    ASSERT_EQ(result.canonical_groups.size(), 2u);
    EXPECT_EQ(result.canonical_groups[0].tag, "full");
    EXPECT_EQ(result.canonical_groups[0].logical_block_num, 70u);
    EXPECT_EQ(result.canonical_groups[1].tag, "linear");
    EXPECT_EQ(result.canonical_groups[1].logical_block_num, 30u);
}

// Case 16 (wire format): serialize/deserialize round trip keeps every
// field, and malformed payloads are rejected.
TEST(PPTopologyValidatorTest, SnapshotSerializeRoundTrip) {
    const auto original = makeHybridSnapshot(120, 45);
    const auto payload  = original.serialize();
    const auto decoded  = StageCacheSnapshot::deserialize(payload);
    EXPECT_EQ(decoded.group_tags, original.group_tags);
    EXPECT_EQ(decoded.group_types, original.group_types);
    EXPECT_EQ(decoded.seq_size_per_block, original.seq_size_per_block);
    EXPECT_EQ(decoded.kernel_seq_size_per_block, original.kernel_seq_size_per_block);
    EXPECT_EQ(decoded.block_nums, original.block_nums);
    EXPECT_EQ(decoded.explicit_block_nums, original.explicit_block_nums);
    EXPECT_EQ(decoded.policy_fingerprints, original.policy_fingerprints);

    EXPECT_THROW(StageCacheSnapshot::deserialize(""), std::exception);
    // Truncated field list (7 of 8 fields).
    EXPECT_THROW(StageCacheSnapshot::deserialize("v1|a|0|1|1|1|0"), std::exception);
    // Unknown versions are rejected outright (all stages run one binary).
    EXPECT_THROW(StageCacheSnapshot::deserialize("v9|full|0|4|2|1|0|p"), std::exception);
    // Entry count mismatch between tags and block_nums.
    EXPECT_THROW(StageCacheSnapshot::deserialize("v1|full|0|4|2|1,2|0|p"), std::exception);
}

// Case 20: a stage holding an SWA group is rejected (v1 scope: paged pools
// only under PP).
TEST(PPTopologyValidatorTest, SwaGroupFails) {
    StageCacheSnapshot with_swa = makeFullSnapshot(50);
    with_swa.group_tags.push_back("swa");
    with_swa.group_types.push_back(CacheGroupType::SWA);
    with_swa.seq_size_per_block.push_back(4);
    with_swa.kernel_seq_size_per_block.push_back(4);
    with_swa.block_nums.push_back(25);
    with_swa.explicit_block_nums.push_back(0);
    with_swa.policy_fingerprints.push_back("t2:r1:e0:v1:x0:c0:p0:a0:w1:m0:s0");

    auto result = validatePPTopology({makeFullSnapshot(50), with_swa});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("SWA"), std::string::npos);
}

// Case 21: same-tag owners must agree on explicit pool sizing.
TEST(PPTopologyValidatorTest, ExplicitBlockNumMismatchFails) {
    StageCacheSnapshot a  = makeFullSnapshot(100);
    a.explicit_block_nums = {256};
    StageCacheSnapshot b  = makeFullSnapshot(100);
    b.explicit_block_nums = {512};

    auto result = validatePPTopology({a, b});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("explicit_block_num"), std::string::npos);
}

// Case 22: identical explicit sizing passes; the canonical entry carries it.
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

// Case 23 (policy fingerprint): the digest reflects every samePolicy()
// field; identical policies digest identically.
TEST(PPTopologyValidatorTest, PolicyFingerprintCoversSamePolicyFields) {
    CacheGroupPolicy base;
    const auto       base_fp = cacheGroupPolicyFingerprint(base);
    EXPECT_EQ(base_fp, cacheGroupPolicyFingerprint(base));

    // Flipping any samePolicy()-compared field changes the digest.
    CacheGroupPolicy changed = base;
    changed.memory_placement = CacheMemoryPlacement::HOST_PINNED;
    EXPECT_NE(base_fp, cacheGroupPolicyFingerprint(changed));
    changed                     = base;
    changed.enable_prefix_reuse = false;
    EXPECT_NE(base_fp, cacheGroupPolicyFingerprint(changed));
    changed                    = base;
    changed.active_tail_blocks = 3;
    EXPECT_NE(base_fp, cacheGroupPolicyFingerprint(changed));
}

// Case 24: same-tag owners with diverging pool policies are rejected.
TEST(PPTopologyValidatorTest, PolicyMismatchFails) {
    StageCacheSnapshot a  = makeFullSnapshot(100);
    StageCacheSnapshot b  = makeFullSnapshot(100);
    b.policy_fingerprints = {"t0:r0:e0:v1:x0:c0:p0:a0:w1:m0:s0"};  // reuse off

    auto result = validatePPTopology({a, b});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("policy"), std::string::npos);
}

}  // namespace rtp_llm
