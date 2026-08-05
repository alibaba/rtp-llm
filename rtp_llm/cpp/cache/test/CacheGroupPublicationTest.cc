#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm::test {

// Pins the publication completeness-set selection used by
// KVCacheAllocator::reuseParticipatingGroupIds() and its Hybrid override,
// which both delegate to reuseParticipatingGroupIdsFromPolicies().

namespace {

CacheGroupPolicy policyOf(CacheGroupType type, bool enable_prefix_reuse) {
    auto policy                = defaultCacheGroupPolicy(type);
    policy.enable_prefix_reuse = enable_prefix_reuse;
    return policy;
}

}  // namespace

TEST(CacheGroupPublicationTest, PredicateAcceptsOnlyDenselyMaterializedReuseGroups) {
    EXPECT_TRUE(cacheGroupPublishesPrefixChain(defaultCacheGroupPolicy(CacheGroupType::FULL)));
    // LINEAR enables prefix reuse but only materializes tail blocks.
    EXPECT_FALSE(cacheGroupPublishesPrefixChain(defaultCacheGroupPolicy(CacheGroupType::LINEAR)));
    // SWA neither reuses prefixes nor materializes densely.
    EXPECT_FALSE(cacheGroupPublishesPrefixChain(defaultCacheGroupPolicy(CacheGroupType::SWA)));

    // A dense group with reuse disabled must also be excluded.
    EXPECT_FALSE(cacheGroupPublishesPrefixChain(policyOf(CacheGroupType::FULL, /*enable_prefix_reuse=*/false)));
}

TEST(CacheGroupPublicationTest, SingleFullGroupIsRequired) {
    const auto ids = reuseParticipatingGroupIdsFromPolicies({defaultCacheGroupPolicy(CacheGroupType::FULL)});
    EXPECT_EQ((std::vector<int>{0}), ids);
}

TEST(CacheGroupPublicationTest, FullPlusSwaKeepsOnlyFullEvenWhenSwaReuseIsForcedOn) {
    const auto ids = reuseParticipatingGroupIdsFromPolicies({
        defaultCacheGroupPolicy(CacheGroupType::FULL),
        policyOf(CacheGroupType::SWA, /*enable_prefix_reuse=*/true),
    });
    EXPECT_EQ((std::vector<int>{0}), ids);
}

TEST(CacheGroupPublicationTest, FullPlusLinearKeepsOnlyFull) {
    const auto ids = reuseParticipatingGroupIdsFromPolicies({
        defaultCacheGroupPolicy(CacheGroupType::FULL),
        defaultCacheGroupPolicy(CacheGroupType::LINEAR),
    });
    EXPECT_EQ((std::vector<int>{0}), ids);
}

TEST(CacheGroupPublicationTest, FullPlusReuseDisabledGroupKeepsOnlyFull) {
    const auto ids = reuseParticipatingGroupIdsFromPolicies({
        defaultCacheGroupPolicy(CacheGroupType::FULL),
        policyOf(CacheGroupType::FULL, /*enable_prefix_reuse=*/false),
    });
    EXPECT_EQ((std::vector<int>{0}), ids);
}

TEST(CacheGroupPublicationTest, NoEligibleGroupYieldsEmptySetForNullFallback) {
    const auto ids = reuseParticipatingGroupIdsFromPolicies({
        policyOf(CacheGroupType::FULL, /*enable_prefix_reuse=*/false),
        defaultCacheGroupPolicy(CacheGroupType::LINEAR),
        defaultCacheGroupPolicy(CacheGroupType::SWA),
    });
    EXPECT_TRUE(ids.empty());
    EXPECT_TRUE(reuseParticipatingGroupIdsFromPolicies({}).empty());
}

TEST(CacheGroupPublicationTest, MultipleFullGroupsAreAllRequired) {
    const auto ids = reuseParticipatingGroupIdsFromPolicies({
        defaultCacheGroupPolicy(CacheGroupType::FULL),
        defaultCacheGroupPolicy(CacheGroupType::SWA),
        defaultCacheGroupPolicy(CacheGroupType::FULL),
    });
    EXPECT_EQ((std::vector<int>{0, 2}), ids);
}

}  // namespace rtp_llm::test
