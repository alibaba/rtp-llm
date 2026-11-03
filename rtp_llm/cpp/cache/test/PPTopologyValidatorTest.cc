#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/PPTopologyValidator.h"

namespace rtp_llm {

namespace {

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
    group.policy_fingerprint        = defaultCacheGroupPolicy(type).policyFingerprint();
    return group;
}

StageCacheSnapshot makeFullSnapshot(uint32_t blocks, size_t seq = 4, size_t kernel = 4, size_t pool_stride = 4096) {
    StageCacheSnapshot s;
    s.groups                = {makeGroupSnapshot("full", CacheGroupType::FULL, blocks, seq, kernel)};
    s.block_tree.group_sets = {{"full", pool_stride}};
    return s;
}

StageCacheSnapshot makeHybridSnapshot(uint32_t full_blocks, uint32_t linear_blocks) {
    StageCacheSnapshot s;
    s.groups = {
        makeGroupSnapshot("full", CacheGroupType::FULL, full_blocks, 4, 2),
        makeGroupSnapshot("linear", CacheGroupType::LINEAR, linear_blocks),
    };
    s.block_tree.group_sets = {{"full", 4096}, {"linear", 4096}};
    return s;
}

StageCacheSnapshot makeTierSnapshot(uint32_t full_blocks, uint32_t swa_blocks) {
    auto snapshot = makeFullSnapshot(full_blocks, 4, 2);
    snapshot.groups.push_back(makeGroupSnapshot("swa", CacheGroupType::SWA, swa_blocks));
    auto policy                           = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.enable_prefix_reuse            = true;
    snapshot.groups[1].policy_fingerprint = policy.policyFingerprint();
    snapshot.block_tree.group_sets.push_back({"swa", 4096});
    return snapshot;
}

StageCacheSnapshot makeSameTypeSnapshot(std::vector<std::string> tags, uint32_t blocks) {
    StageCacheSnapshot snapshot;
    snapshot.groups.reserve(tags.size());
    for (auto& tag : tags) {
        snapshot.groups.push_back(makeGroupSnapshot(std::move(tag), CacheGroupType::FULL, blocks));
    }
    snapshot.block_tree.group_sets = {{snapshot.groups.front().tag, 4096}};
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
    EXPECT_TRUE(result.agreed.block_tree.host.empty());
    EXPECT_TRUE(result.agreed.block_tree.disk.empty());
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
    auto result =
        validatePPTopology({makeSameTypeSnapshot({"csa_kv", "indexer_kv"}, 100), makeSameTypeSnapshot({"csa_kv"}, 90)});
    ASSERT_FALSE(result.ok);
    EXPECT_NE(result.error.find("has 1 of 2 group-set members"), std::string::npos);
    EXPECT_NE(result.error.find("partial group-set overlap is not supported yet"), std::string::npos);
}

TEST(PPTopologyValidatorTest, SameTypeFullOverlapIgnoresLocalTagOrder) {
    auto result = validatePPTopology(
        {makeSameTypeSnapshot({"csa_kv", "indexer_kv"}, 100), makeSameTypeSnapshot({"indexer_kv", "csa_kv"}, 90)});
    ASSERT_TRUE(result.ok) << result.error;
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
    linear_only.groups                = {makeGroupSnapshot("linear", CacheGroupType::LINEAR, 50)};
    linear_only.block_tree.group_sets = {{"linear", 4096}};
    auto result                       = validatePPTopology({makeHybridSnapshot(100, 50), linear_only});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{100, 50}));
}

TEST(PPTopologyValidatorTest, SharedTagTypeMismatchFails) {
    StageCacheSnapshot bad = makeHybridSnapshot(100, 50);
    bad.groups[0].type     = CacheGroupType::LINEAR;

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
    CacheConfig cache_config;
    auto        spec                = std::make_shared<MHAKVCacheSpec>("default", 4, 2, 1);
    cache_config.layer_num          = 2;
    cache_config.seq_size_per_block = 4;
    cache_config.fromGroupedSpecs({spec}, {{0, 1}}, {CacheGroupType::FULL}, {"default"});
    cache_config.finalizeBlockNums(42, RuntimeConfig{});

    auto snapshot = StageCacheSnapshot::captureLocalStage(cache_config, {{{"default", 4096}}, 1024 * 1024, 0});
    ASSERT_EQ(snapshot.groups.size(), 1u);
    EXPECT_EQ(snapshot.groups[0].tag, "default");
    EXPECT_EQ(snapshot.groups[0].type, CacheGroupType::FULL);
    EXPECT_TRUE(snapshot.groups[0].block_tree_group_member);
    EXPECT_EQ(snapshot.groups[0].cache_key_token_stride, 4u);
    EXPECT_EQ(snapshot.groups[0].block_num, 42u);
    ASSERT_EQ(snapshot.block_tree.group_sets.size(), 1u);
    EXPECT_EQ(snapshot.block_tree.group_sets[0].first_tag, "default");
    EXPECT_EQ(snapshot.block_tree.group_sets[0].pool_stride, 4096u);
    EXPECT_EQ(snapshot.block_tree.host_budget_bytes, 1024u * 1024u);
}

TEST(PPTopologyValidatorTest, CanonicalGroupsAndAgreedInputsUsePerTagMinima) {
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
    EXPECT_EQ(result.agreed.block_num_overrides.at("full"), 90u);
    EXPECT_EQ(result.agreed.block_num_overrides.at("linear"), 50u);
    EXPECT_EQ(result.agreed.paged_block_num, 50u);
}

TEST(PPTopologyValidatorTest, SnapshotSerializeRoundTrip) {
    auto original                         = makeHybridSnapshot(120, 45);
    original.groups[1].explicit_block_num = 45;
    original.block_tree.host_budget_bytes = 1024 * 1024;
    original.block_tree.disk_budget_bytes = 2 * 1024 * 1024;
    const auto payload                    = original.serialize();
    EXPECT_EQ(payload.substr(0, payload.find('|')), "full\x1flinear");
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
    ASSERT_EQ(decoded.block_tree.group_sets.size(), original.block_tree.group_sets.size());
    for (size_t i = 0; i < original.block_tree.group_sets.size(); ++i) {
        EXPECT_EQ(decoded.block_tree.group_sets[i].first_tag, original.block_tree.group_sets[i].first_tag);
        EXPECT_EQ(decoded.block_tree.group_sets[i].pool_stride, original.block_tree.group_sets[i].pool_stride);
    }
    EXPECT_EQ(decoded.block_tree.host_budget_bytes, original.block_tree.host_budget_bytes);
    EXPECT_EQ(decoded.block_tree.disk_budget_bytes, original.block_tree.disk_budget_bytes);

    const auto empty = StageCacheSnapshot::deserialize(StageCacheSnapshot{}.serialize());
    EXPECT_TRUE(empty.groups.empty());
    EXPECT_TRUE(empty.block_tree.group_sets.empty());

    EXPECT_THROW(StageCacheSnapshot::deserialize(""), std::exception);
    EXPECT_THROW(StageCacheSnapshot::deserialize("a|1|4|4|4|1|0"), std::exception);
    EXPECT_THROW(StageCacheSnapshot::deserialize("full|1|1|4|2|4|1,2|0|p|full|4096|1024|0"), std::exception);
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

TEST(PPTopologyValidatorTest, TierCountsUseAgreedDeviceBlocksAndSeparateBudgets) {
    auto stage0                         = makeTierSnapshot(10, 10);
    auto stage1                         = makeTierSnapshot(20, 10);
    stage0.block_tree.host_budget_bytes = stage1.block_tree.host_budget_bytes = 120 * 4096;
    stage0.block_tree.disk_budget_bytes                                       = 240 * 4096;
    stage1.block_tree.disk_budget_bytes                                       = 60 * 4096;

    auto result = validatePPTopology({stage0, stage1});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(canonicalBlockNums(result), (std::vector<uint32_t>{10, 10}));
    for (const auto type : {CacheGroupType::FULL, CacheGroupType::SWA}) {
        EXPECT_EQ(result.agreed.block_tree.host.at(type), 60u);
        EXPECT_EQ(result.agreed.block_tree.disk.at(type), 30u);
    }
}

TEST(PPTopologyValidatorTest, TierCountsRespectStageStridesAndMissingGroupSets) {
    auto stage0                                 = makeTierSnapshot(40, 10);
    auto stage1                                 = makeFullSnapshot(30, 4, 2, 8192);
    stage0.block_tree.group_sets[1].pool_stride = 8192;
    stage0.block_tree.host_budget_bytes = stage1.block_tree.host_budget_bytes = 120 * 4096;

    auto result = validatePPTopology({stage0, stage1});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.agreed.block_tree.host.at(CacheGroupType::FULL), 60u);
    EXPECT_EQ(result.agreed.block_tree.host.at(CacheGroupType::SWA), 24u);
    EXPECT_TRUE(result.agreed.block_tree.disk.empty());
}

TEST(PPTopologyValidatorTest, TierWeightsUseEachStageFirstMemberWithUnequalPools) {
    auto stage0 = makeTierSnapshot(40, 20);
    stage0.groups.push_back(makeGroupSnapshot("full_b", CacheGroupType::FULL, 10, 4, 2));
    auto stage1                = stage0;
    stage1.groups[0].block_num = 30;
    stage1.groups[2].block_num = 15;
    std::swap(stage1.groups[0], stage1.groups[2]);
    stage1.block_tree.group_sets[0].first_tag = "full_b";
    stage0.block_tree.host_budget_bytes = stage1.block_tree.host_budget_bytes = 100 * 4096;

    auto result = validatePPTopology({stage0, stage1});
    ASSERT_TRUE(result.ok) << result.error;
    EXPECT_EQ(result.agreed.block_tree.host.at(CacheGroupType::FULL), 33u);
    EXPECT_EQ(result.agreed.block_tree.host.at(CacheGroupType::SWA), 40u);
}

TEST(PPTopologyValidatorTest, TierSizingRejectsInconsistentEnablementAndMissingGroupSets) {
    auto stage0                         = makeTierSnapshot(40, 10);
    auto stage1                         = stage0;
    stage0.block_tree.host_budget_bytes = 120 * 4096;
    auto result                         = validatePPTopology({stage0, stage1});
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("enablement"), std::string::npos);

    stage1.block_tree.host_budget_bytes = stage0.block_tree.host_budget_bytes;
    stage1.block_tree.group_sets.pop_back();
    result = validatePPTopology({stage0, stage1});
    EXPECT_FALSE(result.ok);
    EXPECT_NE(result.error.find("missing BlockTreeCache group-set sizing"), std::string::npos);
}

}  // namespace rtp_llm
