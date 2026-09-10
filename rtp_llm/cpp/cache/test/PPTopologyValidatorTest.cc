// Unit tests for validatePPTopology, the startup cache geometry check.
//
// Pinned: canonical table = cross-stage UNION in tag-sorted order, each entry
// capacity = min over that tag OWNERS only (single-owner tags never capped);
// the stage 0 superset rule; same-tag geometry and policy agreement; wire v1.

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/CacheCapacityNegotiator.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"

using namespace std;

namespace rtp_llm {

namespace {

// Geometry of makeHybridSnapshot's "full" group, so a single-group snapshot can
// stand in as a subset stage without tripping the geometry checks.
constexpr size_t kHybridFullSeq    = 4;
constexpr size_t kHybridFullKernel = 2;

const char* const kFullFingerprint   = "t0:r1:e0:v1:x0:a0:w1:m0:s0";
const char* const kLinearFingerprint = "t1:r1:e0:v1:x0:a0:w1:m0:s0";

// Single-FULL-group snapshot ("full"); also used as a subset stage of a hybrid
// stage0 when built with the hybrid full-group geometry.
StageCacheSnapshot makeFullSnapshot(uint32_t blocks, size_t seq = 4, size_t kernel = 4) {
    StageCacheSnapshot s;
    s.group_tags                = {"full"};
    s.group_types               = {CacheGroupType::FULL};
    s.seq_size_per_block        = {seq};
    s.kernel_seq_size_per_block = {kernel};
    s.block_nums                = {blocks};
    s.explicit_block_nums       = {0};
    s.policy_fingerprints       = {kFullFingerprint};
    return s;
}

// Subset of makeHybridSnapshot: only the "full" group, same geometry.
StageCacheSnapshot makeHybridFullSubset(uint32_t full_blocks) {
    return makeFullSnapshot(full_blocks, kHybridFullSeq, kHybridFullKernel);
}

// Two-group (FULL + LINEAR) snapshot, qwen3-next-style hybrid.
StageCacheSnapshot makeHybridSnapshot(uint32_t full_blocks, uint32_t linear_blocks) {
    StageCacheSnapshot s;
    s.group_tags                = {"full", "linear"};
    s.group_types               = {CacheGroupType::FULL, CacheGroupType::LINEAR};
    s.seq_size_per_block        = {kHybridFullSeq, 4};
    s.kernel_seq_size_per_block = {kHybridFullKernel, 4};
    s.block_nums                = {full_blocks, linear_blocks};
    s.explicit_block_nums       = {0, 0};
    s.policy_fingerprints       = {kFullFingerprint, kLinearFingerprint};
    return s;
}

// Appends one more group to a snapshot (tag/type/geometry/counts).
void appendGroup(StageCacheSnapshot& s,
                 const std::string&  tag,
                 CacheGroupType      type,
                 uint32_t            blocks,
                 uint32_t            explicit_blocks = 0,
                 const std::string&  fingerprint     = kFullFingerprint) {
    s.group_tags.push_back(tag);
    s.group_types.push_back(type);
    s.seq_size_per_block.push_back(4);
    s.kernel_seq_size_per_block.push_back(4);
    s.block_nums.push_back(blocks);
    s.explicit_block_nums.push_back(explicit_blocks);
    s.policy_fingerprints.push_back(fingerprint);
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

std::vector<std::string> canonicalTags(const PPValidationResult& result) {
    std::vector<std::string> tags;
    tags.reserve(result.canonical_groups.size());
    for (const auto& entry : result.canonical_groups) {
        tags.push_back(entry.tag);
    }
    return tags;
}

// Single-group ModelConfig with the given tag on every layer.
ModelConfig makeSingleGroupModel(const std::string& tag, int64_t layer_num) {
    ModelConfig config;
    config.num_layers                   = layer_num;
    config.data_type                    = DataType::TYPE_FP16;
    config.attn_config.head_num         = 4;
    config.attn_config.kv_head_num      = 2;
    config.attn_config.size_per_head    = 8;
    config.attn_config.tokens_per_block = 4;
    config.attn_config.kv_cache_dtype   = KvCacheDataType::BASE;
    KVCacheSpecDesc desc;
    desc.tag        = tag;
    desc.cache_type = KVCacheSpecType::MultiHeadAttention;
    config.kv_cache_spec_descs.assign(static_cast<size_t>(layer_num), {desc});
    return config;
}

// Unsized topology skeleton through the official creator path.
CacheConfig makeTopology(const ModelConfig& model) {
    return CacheConfigCreator::createBasicConfig(model, ParallelismConfig{}, KVCacheConfig{}, 0);
}

// Sized single-group CacheConfig; test_block_num pins the count deterministically.
CacheConfig makeSingleGroupConfig(const std::string& tag, uint32_t blocks, int64_t layer_num) {
    KVCacheConfig kv_config;
    kv_config.test_block_num = blocks;
    return CacheConfigCreator::createConfig(
        makeSingleGroupModel(tag, layer_num), ParallelismConfig{}, RuntimeConfig{}, kv_config);
}

}  // namespace

// ---------------------------------------------------------------------------
// Happy paths and the per-tag minimum semantics
// ---------------------------------------------------------------------------

// Case 1: all stages own the same groups -> per-tag min over owners, and the
// top-level yardstick is the min over the paged entries.
TEST(PPTopologyValidatorTest, AllIdenticalPasses) {
    auto result =
        validatePPTopology({makeHybridSnapshot(100, 50), makeHybridSnapshot(90, 55), makeHybridSnapshot(95, 50)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalTags(result), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{90, 50}));
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 90u);
    EXPECT_EQ(result.agreed.block_num_overrides.at("linear"), 50u);
    EXPECT_EQ(result.agreed.paged_block_num, 50u);  // min over both paged entries
}

// Case 2 (the key behavior): a tag owned by one stage keeps that capacity.
// Only shared tags are capped, since single-owner ids never cross a boundary.
TEST(PPTopologyValidatorTest, SingleOwnerTagIsNotCapped) {
    // stage0 (superset): full=100, linear=100. stage1 (subset): full=80 only.
    auto result = validatePPTopology({makeHybridSnapshot(100, 100), makeHybridFullSubset(80)});
    ASSERT_TRUE(result.ok) << result.error;

    ASSERT_EQ(result.canonical_groups.size(), 2u);
    EXPECT_EQ(canonicalTags(result), (std::vector<std::string>{"full", "linear"}));
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 80u);     // shared -> min(100, 80)
    EXPECT_EQ(result.agreed.block_num_overrides.at("linear"), 100u);  // single owner -> unchanged
    EXPECT_EQ(result.agreed.paged_block_num, 80u);
}

// Case 2b: the subset stage's own view — it receives an override table covering
// the whole union, including tags it does not own. finalizeBlockNums matches by
// tag, so the extra entries are inert rather than an error.
TEST(PPTopologyValidatorTest, OverrideTableCoversWholeUnion) {
    auto result = validatePPTopology({makeHybridSnapshot(100, 100), makeHybridFullSubset(80)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.agreed.block_num_overrides.size(), 2u);
    EXPECT_NE(result.agreed.block_num_overrides.find("linear"), result.agreed.block_num_overrides.end());
}

// Case 3: single stage (pp_size=1) -> trivially pass, degenerates to today.
TEST(PPTopologyValidatorTest, SingleStageTriviallyPasses) {
    auto result = validatePPTopology({makeFullSnapshot(42)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{42}));
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 42u);
    EXPECT_EQ(result.agreed.paged_block_num, 42u);
}

// Case 4: empty input -> trivially pass with no canonical rows and no inputs.
TEST(PPTopologyValidatorTest, EmptyStagesTriviallyPasses) {
    auto result = validatePPTopology({});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_TRUE(result.canonical_groups.empty());
    EXPECT_TRUE(result.agreed.block_num_overrides.empty());
    EXPECT_EQ(result.agreed.paged_block_num, 0u);
}

// Case 5: the canonical table is published in tag-lexicographic order
// regardless of declaration order within a stage or stage arrival order.
TEST(PPTopologyValidatorTest, CanonicalTableIsTagSorted) {
    // stage0 declares {zeta, alpha, full} in that (unsorted) first-encounter
    // order and is the superset; stage1 owns only {full}.
    StageCacheSnapshot stage0;
    stage0.group_tags                = {"zeta", "alpha", "full"};
    stage0.group_types               = {CacheGroupType::FULL, CacheGroupType::FULL, CacheGroupType::FULL};
    stage0.seq_size_per_block        = {4, 4, 4};
    stage0.kernel_seq_size_per_block = {4, 4, 4};
    stage0.block_nums                = {100, 100, 100};
    stage0.explicit_block_nums       = {0, 0, 0};
    stage0.policy_fingerprints       = {kFullFingerprint, kFullFingerprint, kFullFingerprint};

    auto result = validatePPTopology({stage0, makeFullSnapshot(90)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalTags(result), (std::vector<std::string>{"alpha", "full", "zeta"}));
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{100, 90, 100}));
    EXPECT_EQ(result.agreed.block_num_overrides.at("alpha"), 100u);  // single owner
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 90u);    // shared -> min
    EXPECT_EQ(result.agreed.block_num_overrides.at("zeta"), 100u);   // single owner
    EXPECT_EQ(result.agreed.paged_block_num, 90u);
}

// ---------------------------------------------------------------------------
// Allocation-authority gate (stage 0 must own the union)
// ---------------------------------------------------------------------------

// Case 6: a downstream stage owning a tag stage 0 lacks is rejected — the
// leading stage issues every block id, so it must own a pool for each tag.
TEST(PPTopologyValidatorTest, NonStage0OwnedTagFails) {
    auto stage1 = makeHybridSnapshot(90, 55);
    appendGroup(stage1, "extra", CacheGroupType::FULL, 40);

    auto result = validatePPTopology({makeHybridSnapshot(100, 50), stage1});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("extra"), std::string::npos);
    EXPECT_NE(result.error.find("which stage 0 does not"), std::string::npos);
}

// Case 6b: same rule with three stages — the offender need not be stage 1.
TEST(PPTopologyValidatorTest, LaterStageOnlyTagFails) {
    auto stage2 = makeHybridSnapshot(95, 50);
    appendGroup(stage2, "linear1", CacheGroupType::LINEAR, 30, 0, kLinearFingerprint);

    auto result = validatePPTopology({makeHybridSnapshot(100, 50), makeHybridSnapshot(90, 50), stage2});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("linear1"), std::string::npos);
}

// Case 6c: a snapshot listing the same tag twice is rejected rather than
// silently counted as a second owner (which would skew the minimum).
TEST(PPTopologyValidatorTest, DuplicateTagInSnapshotFails) {
    StageCacheSnapshot dup = makeFullSnapshot(100);
    appendGroup(dup, "full", CacheGroupType::FULL, 100);

    auto result = validatePPTopology({dup, makeFullSnapshot(100)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("twice"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Per-stage structural invariants
// ---------------------------------------------------------------------------

// Case 7: hybrid stage (LINEAR groups) without any FULL group -> reject.
TEST(PPTopologyValidatorTest, LinearWithoutFullFails) {
    StageCacheSnapshot linear_only;
    linear_only.group_tags                = {"linear"};
    linear_only.group_types               = {CacheGroupType::LINEAR};
    linear_only.seq_size_per_block        = {4};
    linear_only.kernel_seq_size_per_block = {4};
    linear_only.block_nums                = {50};
    linear_only.explicit_block_nums       = {0};
    linear_only.policy_fingerprints       = {kLinearFingerprint};

    auto result = validatePPTopology({makeHybridSnapshot(100, 50), linear_only});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("no FULL group"), std::string::npos);
}

// Case 8: a stage holding an SWA group is rejected (paged pools only under PP).
TEST(PPTopologyValidatorTest, SwaGroupFails) {
    auto with_swa = makeFullSnapshot(50);
    appendGroup(with_swa, "swa", CacheGroupType::SWA, 25, 0, "t2:r1:e0:v1:x0:a0:w1:m0:s0");

    auto result = validatePPTopology({makeFullSnapshot(50), with_swa});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("SWA"), std::string::npos);
}

// Case 9: internally inconsistent snapshot -> reject.
TEST(PPTopologyValidatorTest, InternallyInconsistentFails) {
    auto bad       = makeFullSnapshot(100);
    bad.block_nums = {100, 90};  // one extra element

    auto result = validatePPTopology({makeFullSnapshot(100), bad});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("internally inconsistent"), std::string::npos);
}

// Case 10: any group anywhere with zero blocks -> reject.
TEST(PPTopologyValidatorTest, ZeroBlockNumFails) {
    auto result = validatePPTopology({makeFullSnapshot(100), makeFullSnapshot(0)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("0 KV blocks"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Same-tag owner agreement (now by tag pairing, not by positional equality)
// ---------------------------------------------------------------------------

// Case 11: same tag reported with different types -> reject via the canonical
// pairing check.
TEST(PPTopologyValidatorTest, TypeMismatchFails) {
    auto bad        = makeHybridSnapshot(100, 50);
    bad.group_types = {CacheGroupType::LINEAR, CacheGroupType::FULL};  // swapped

    auto result = validatePPTopology({makeHybridSnapshot(100, 50), bad});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("type differs from the canonical entry"), std::string::npos);
}

// Case 11b: a type conflict introduced by a third stage is caught the same way.
TEST(PPTopologyValidatorTest, ThirdStageTypeConflictFails) {
    auto stage0 = makeFullSnapshot(100);
    appendGroup(stage0, "x", CacheGroupType::FULL, 40);
    auto stage1 = makeFullSnapshot(90);
    appendGroup(stage1, "x", CacheGroupType::FULL, 40);
    auto stage2 = makeFullSnapshot(95);
    appendGroup(stage2, "x", CacheGroupType::LINEAR, 40, 0, kLinearFingerprint);

    auto result = validatePPTopology({stage0, stage1, stage2});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("type differs from the canonical entry"), std::string::npos);
}

// Case 12: seq_size_per_block differs -> reject.
TEST(PPTopologyValidatorTest, SeqSizeMismatchFails) {
    auto result = validatePPTopology({makeFullSnapshot(100, /*seq=*/4), makeFullSnapshot(100, /*seq=*/8)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("seq_size_per_block"), std::string::npos);
}

// Case 12b: a legal SUBSET stage still has to agree on the shared tag's
// geometry — being a subset relaxes ownership, not geometry.
TEST(PPTopologyValidatorTest, SubsetGeometryMismatchFails) {
    auto subset = makeFullSnapshot(80, /*seq=*/8, kHybridFullKernel);  // seq differs from stage0's full

    auto result = validatePPTopology({makeHybridSnapshot(100, 50), subset});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("seq_size_per_block"), std::string::npos);
}

// Case 13: kernel_seq_size_per_block differs -> reject.
TEST(PPTopologyValidatorTest, KernelSeqSizeMismatchFails) {
    auto result = validatePPTopology({makeFullSnapshot(100, 4, /*kernel=*/4), makeFullSnapshot(100, 4, /*kernel=*/2)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("kernel_seq_size_per_block"), std::string::npos);
}

// Case 14: same-tag owners must agree on explicit pool sizing.
TEST(PPTopologyValidatorTest, ExplicitBlockNumMismatchFails) {
    auto a                = makeFullSnapshot(100);
    a.explicit_block_nums = {256};
    auto b                = makeFullSnapshot(100);
    b.explicit_block_nums = {512};

    auto result = validatePPTopology({a, b});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("explicit_block_num"), std::string::npos);
}

// Case 15: identical explicit sizing passes and the canonical entry carries it.
TEST(PPTopologyValidatorTest, ExplicitBlockNumAgreementPasses) {
    auto a                = makeFullSnapshot(256);
    a.explicit_block_nums = {256};
    auto b                = makeFullSnapshot(256);
    b.explicit_block_nums = {256};

    auto result = validatePPTopology({a, b});
    ASSERT_TRUE(result.ok) << result.error;
    ASSERT_EQ(result.canonical_groups.size(), 1u);
    EXPECT_EQ(result.canonical_groups[0].explicit_block_num, 256u);
    EXPECT_EQ(result.canonical_groups[0].logical_block_num, 256u);
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 256u);
}

// Case 16: same-tag owners with diverging pool policies are rejected.
TEST(PPTopologyValidatorTest, PolicyMismatchFails) {
    auto a                = makeFullSnapshot(100);
    auto b                = makeFullSnapshot(100);
    b.policy_fingerprints = {"t0:r0:e0:v1:x0:a0:w1:m0:s0"};  // reuse off

    auto result = validatePPTopology({a, b});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("policy"), std::string::npos);
}

// Case 17 (policy fingerprint): the digest reflects every samePolicy() field.
TEST(PPTopologyValidatorTest, PolicyFingerprintCoversSamePolicyFields) {
    CacheGroupPolicy base;
    const auto       base_fp = cacheGroupPolicyFingerprint(base);
    EXPECT_EQ(base_fp, cacheGroupPolicyFingerprint(base));

    CacheGroupPolicy changed    = base;
    changed.enable_prefix_reuse = false;
    EXPECT_NE(base_fp, cacheGroupPolicyFingerprint(changed));
    changed                    = base;
    changed.active_tail_blocks = 3;
    EXPECT_NE(base_fp, cacheGroupPolicyFingerprint(changed));
    changed                      = base;
    changed.validate_tail_blocks = false;
    EXPECT_NE(base_fp, cacheGroupPolicyFingerprint(changed));
    changed            = base;
    changed.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    EXPECT_NE(base_fp, cacheGroupPolicyFingerprint(changed));
}

// ---------------------------------------------------------------------------
// Top-level yardstick (paged_block_num)
// ---------------------------------------------------------------------------

// Case 18: an explicitly sized pool is decoupled from the global budget and
// must not drag the top-level count down.
TEST(PPTopologyValidatorTest, ExplicitPoolExcludedFromPagedYardstick) {
    auto a = makeFullSnapshot(100);
    appendGroup(a, "small", CacheGroupType::FULL, 16, /*explicit=*/16, "t0:r1:e0:v1:x16:a0:w1:m0:s0");

    auto result = validatePPTopology({a});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.agreed.paged_block_num, 100u);  // not 16
    EXPECT_EQ(result.agreed.block_num_overrides.at("small"), 16u);
}

// Case 19: the yardstick is the min over ALL paged entries, so a smaller shared
// paged group does pull it down (that is its job).
TEST(PPTopologyValidatorTest, PagedYardstickIsMinOverPagedEntries) {
    // full=100 shared, linear=48 shared -> yardstick 48
    auto result = validatePPTopology({makeHybridSnapshot(100, 50), makeHybridSnapshot(100, 48)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 100u);
    EXPECT_EQ(result.agreed.block_num_overrides.at("linear"), 48u);
    EXPECT_EQ(result.agreed.paged_block_num, 48u);
}

// ---------------------------------------------------------------------------
// Capacity skew (deployment-quality guard, not a correctness one)
// ---------------------------------------------------------------------------

// Case 20: skew above threshold -> reject.
TEST(PPTopologyValidatorTest, CapacitySkewTooLargeFails) {
    // 100 / 60 = 1.67 > 1.5
    auto result = validatePPTopology({makeFullSnapshot(100), makeFullSnapshot(60)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("skew too large"), std::string::npos);
}

// Case 21: skew within threshold -> pass with the min.
TEST(PPTopologyValidatorTest, CapacitySkewWithinThresholdPasses) {
    // 100 / 80 = 1.25 <= 1.5
    auto result = validatePPTopology({makeFullSnapshot(100), makeFullSnapshot(80)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{80}));
    EXPECT_EQ(result.agreed.paged_block_num, 80u);
}

// Case 22: skew on a tag owned by all stages is attributed to that tag.
TEST(PPTopologyValidatorTest, SkewAttributedToOffendingTag) {
    auto stage0 = makeFullSnapshot(100);
    appendGroup(stage0, "y", CacheGroupType::LINEAR, 100, 0, kLinearFingerprint);
    auto stage1 = makeFullSnapshot(100);
    appendGroup(stage1, "y", CacheGroupType::LINEAR, 100, 0, kLinearFingerprint);
    auto stage2 = makeFullSnapshot(90);
    appendGroup(stage2, "y", CacheGroupType::LINEAR, 10, 0, kLinearFingerprint);

    auto result = validatePPTopology({stage0, stage1, stage2});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("skew too large"), std::string::npos);
    EXPECT_NE(result.error.find("[y]"), std::string::npos);
}

// Case 23: a single-owner tag is exempt from skew by construction (max == min).
TEST(PPTopologyValidatorTest, SingleOwnerTagExemptFromSkew) {
    // linear is owned by stage0 alone with a count far above stage1's full
    // capacity; that spread is legitimate and must not be rejected.
    auto result = validatePPTopology({makeHybridSnapshot(100, 100), makeHybridFullSubset(70)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.agreed.block_num_overrides.at("linear"), 100u);
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 70u);
}

// ---------------------------------------------------------------------------
// Snapshot capture from a real topology skeleton
// ---------------------------------------------------------------------------

// Case 24: capture derives per-group counts from the topology plus the measured
// local capacity, through the same rule composition uses.
TEST(PPTopologyValidatorTest, FromTopologyAndCapacityDerivesCounts) {
    const CacheConfig topology = makeTopology(makeSingleGroupModel("default", /*layer_num=*/2));
    EXPECT_EQ(topology.block_num, 0u);  // skeleton is unsized
    EXPECT_TRUE(topology.groups()[0].block_num == 0u);

    const auto snapshot = StageCacheSnapshot::fromTopologyAndCapacity(topology, /*local_capacity=*/42, RuntimeConfig{});
    ASSERT_TRUE(snapshot.internallyConsistent());
    EXPECT_EQ(snapshot.group_tags, (std::vector<std::string>{"default"}));
    EXPECT_EQ(snapshot.group_types, (std::vector<CacheGroupType>{CacheGroupType::FULL}));
    EXPECT_EQ(snapshot.block_nums, (std::vector<uint32_t>{42}));
    EXPECT_EQ(snapshot.explicit_block_nums, (std::vector<uint32_t>{0}));
}

// Case 25: capture normalizes the local record order to tag-lexicographic, so
// identical topologies serialize identically on every stage.
TEST(PPTopologyValidatorTest, FromTopologySortsGroupsByTag) {
    // First-encounter order is [zeta, alpha] (layer 0 declares zeta).
    ModelConfig model                   = makeSingleGroupModel("zeta", /*layer_num=*/2);
    model.kv_cache_spec_descs[1][0].tag = "alpha";
    const CacheConfig topology          = makeTopology(model);
    ASSERT_EQ(topology.groupNums(), 2);

    const auto snapshot = StageCacheSnapshot::fromTopologyAndCapacity(topology, /*local_capacity=*/32, RuntimeConfig{});
    ASSERT_TRUE(snapshot.internallyConsistent());
    EXPECT_EQ(snapshot.group_tags, (std::vector<std::string>{"alpha", "zeta"}));
    EXPECT_EQ(snapshot.block_nums, (std::vector<uint32_t>{32, 32}));
}

// Case 26: capture does not mutate the topology it was given.
TEST(PPTopologyValidatorTest, FromTopologyLeavesSkeletonUnsized) {
    CacheConfig topology = makeTopology(makeSingleGroupModel("default", /*layer_num=*/2));
    (void)StageCacheSnapshot::fromTopologyAndCapacity(topology, /*local_capacity=*/64, RuntimeConfig{});
    EXPECT_EQ(topology.block_num, 0u);
    EXPECT_EQ(topology.groups()[0].block_num, 0u);
}

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

// Case 27: v1 round trip keeps every field; malformed payloads are rejected.
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
    EXPECT_EQ(payload.rfind("v1|", 0), 0u);

    EXPECT_THROW(StageCacheSnapshot::deserialize(""), std::exception);
    // Truncated field list (7 of 8 fields).
    EXPECT_THROW(StageCacheSnapshot::deserialize("v1|a|0|1|1|1|0"), std::exception);
    // Unknown versions are rejected outright (all stages run one binary).
    EXPECT_THROW(StageCacheSnapshot::deserialize("v9|full|0|4|2|1|0|p"), std::exception);
    // An extra trailing field (9 of 8) is rejected on field count.
    EXPECT_THROW(StageCacheSnapshot::deserialize("v1|full|0|4|2|1|0|p|42"), std::exception);
    // Entry count mismatch between tags and block_nums.
    EXPECT_THROW(StageCacheSnapshot::deserialize("v1|full|0|4|2|1,2|0|p"), std::exception);
}

// ---------------------------------------------------------------------------
// Collector entry points
// ---------------------------------------------------------------------------

// Mock collector returning a fixed list of snapshots (stands in for the real
// startup exchange).
class MockStageSnapshotCollector: public StageSnapshotCollector {
public:
    explicit MockStageSnapshotCollector(std::vector<StageCacheSnapshot> snapshots): snapshots_(std::move(snapshots)) {}

    std::vector<StageCacheSnapshot> collect() override {
        return snapshots_;
    }

private:
    std::vector<StageCacheSnapshot> snapshots_;
};

// Case 28: initPPCacheGeometry with the local single-stage collector.
TEST(PPTopologyValidatorTest, InitGeometryLocalCollectorPasses) {
    LocalStageSnapshotCollector collector(makeFullSnapshot(64));
    auto                        result = initPPCacheGeometry(collector);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{64}));
    EXPECT_EQ(result.agreed.paged_block_num, 64u);
}

// Case 29: initPPCacheGeometry with a multi-stage mock collector.
TEST(PPTopologyValidatorTest, InitGeometryMultiStageMockComputesMin) {
    MockStageSnapshotCollector collector({makeHybridSnapshot(100, 50), makeHybridSnapshot(90, 48)});
    auto                       result = initPPCacheGeometry(collector);
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{90, 48}));
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 90u);
    EXPECT_EQ(result.agreed.block_num_overrides.at("linear"), 48u);
}

// Case 30: initPPCacheGeometry propagates validation failure.
TEST(PPTopologyValidatorTest, InitGeometryFailsOnMismatch) {
    MockStageSnapshotCollector collector({makeFullSnapshot(100), makeFullSnapshot(10)});  // skew 10x
    auto                       result = initPPCacheGeometry(collector);
    ASSERT_FALSE(result.ok);
    EXPECT_TRUE(result.canonical_groups.empty());
    EXPECT_TRUE(result.agreed.block_num_overrides.empty());
}

// ---------------------------------------------------------------------------
// Post-composition fuse
// ---------------------------------------------------------------------------

// Case 31: validatePPComposedBlockNums requires exact equality with the
// canonical minima — composition writes those values straight from the
// override table, so anything else is an implementation defect.
TEST(PPTopologyValidatorTest, ComposedBlockNumsMustEqualCanonical) {
    auto result = validatePPTopology({makeFullSnapshot(100), makeFullSnapshot(80)});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 80u);

    CacheConfig exact = makeSingleGroupConfig("full", /*blocks=*/80, /*layer_num=*/2);
    EXPECT_NO_THROW(validatePPComposedBlockNums(exact, result.agreed));

    // Below the agreed count: stages would end up with divergent pools.
    CacheConfig shrunk = makeSingleGroupConfig("full", /*blocks=*/40, /*layer_num=*/2);
    EXPECT_THROW(validatePPComposedBlockNums(shrunk, result.agreed), std::exception);

    // Above it is equally a defect (equality, not a lower bound).
    CacheConfig grown = makeSingleGroupConfig("full", /*blocks=*/100, /*layer_num=*/2);
    EXPECT_THROW(validatePPComposedBlockNums(grown, result.agreed), std::exception);

    // A local tag missing from the canonical table fails fast.
    CacheConfig unknown_tag = makeSingleGroupConfig("other", /*blocks=*/80, /*layer_num=*/2);
    EXPECT_THROW(validatePPComposedBlockNums(unknown_tag, result.agreed), std::exception);
}

// Case 32: end-to-end shape of the flow — local sizing, then negotiation in
// the manager. The composed config's per-group counts are exactly the
// canonical minima, and a single-owner tag survives at the richer stage's own
// capacity.
TEST(PPTopologyValidatorTest, MeasureNegotiateComposeEndToEnd) {
    // stage0 topology: two groups ("default" tag), unsized skeleton.
    ModelConfig model                   = makeSingleGroupModel("zeta", /*layer_num=*/2);
    model.kv_cache_spec_descs[0][0].tag = "alpha";
    const CacheConfig topology          = makeTopology(model);
    ASSERT_EQ(topology.groupNums(), 2);

    // stage0 measures 100; a subset stage1 reports only "alpha" at 80.
    const auto stage0_snap =
        StageCacheSnapshot::fromTopologyAndCapacity(topology, /*local_capacity=*/100, RuntimeConfig{});
    ASSERT_EQ(stage0_snap.group_tags, (std::vector<std::string>{"alpha", "zeta"}));
    EXPECT_EQ(stage0_snap.block_nums, (std::vector<uint32_t>{100, 100}));

    const auto stage1_snap = StageCacheSnapshot::fromTopologyAndCapacity(
        makeTopology(makeSingleGroupModel("alpha", /*layer_num=*/1)), /*local_capacity=*/80, RuntimeConfig{});

    const auto validation = validatePPTopology({stage0_snap, stage1_snap});
    ASSERT_TRUE(validation.ok) << validation.error;
    EXPECT_EQ(validation.agreed.block_num_overrides.at("alpha"), 80u);  // shared -> capped
    EXPECT_EQ(validation.agreed.block_num_overrides.at("zeta"), 100u);  // single owner -> kept
    EXPECT_EQ(validation.agreed.paged_block_num, 80u);

    // Feed the agreement through the manager's negotiation point, with the
    // hook running the real fuse: validator -> manager sizing -> fuse.
    class AgreementHook: public CacheCapacityNegotiator {
    public:
        explicit AgreementHook(const NegotiatedCapacity& agreed): agreed_(agreed) {}

        NegotiatedCapacity negotiate(const CacheConfig&   hook_topology,
                                     uint32_t             local_block_num,
                                     const RuntimeConfig& runtime_config) override {
            EXPECT_EQ(hook_topology.groupNums(), 2);
            EXPECT_GT(local_block_num, 0u);
            return agreed_;
        }

        void validateComposed(const CacheConfig& composed, const NegotiatedCapacity& agreed) override {
            validatePPComposedBlockNums(composed, agreed);
        }

    private:
        NegotiatedCapacity agreed_;
    };

    // Pinned so the case never solves against live free memory; the agreement
    // overrides the measured count anyway.
    KVCacheConfig kv_config;
    kv_config.test_block_num = 1000;

    auto        hook     = std::make_shared<AgreementHook>(validation.agreed);
    CacheConfig composed = CacheConfigCreator::createConfig(model, ParallelismConfig{}, RuntimeConfig{}, kv_config);

    // pp_size>1 selects the negotiation path; tp_size=1 keeps it free of real
    // collectives. The config stays whole-model: only the agreement flow is
    // under test here, not layer scoping.
    ParallelismConfig pp_parallelism;
    pp_parallelism.pp_size = 2;
    pp_parallelism.pp_rank = 0;
    KVCacheManager manager(std::move(composed),
                           /*warmup=*/false,
                           /*metrics_reporter=*/nullptr,
                           kv_config,
                           pp_parallelism,
                           RuntimeConfig{},
                           SpeculativeExecutionConfig{},
                           PDSepConfig{},
                           CacheStoreConfig{},
                           /*use_cuda_malloc_block_pool=*/false,
                           hook);
    composed = manager.cacheConfig();
    EXPECT_EQ(composed.block_num, 80u);
    EXPECT_EQ(composed.group("alpha").block_num, 80u);
    EXPECT_EQ(composed.group("zeta").block_num, 100u);
}

}  // namespace rtp_llm
