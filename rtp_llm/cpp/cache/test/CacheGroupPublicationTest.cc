#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm::test {
namespace {

CacheGroupPolicy policyOf(CacheGroupType type, bool reuse = true) {
    auto policy                = defaultCacheGroupPolicy(type);
    policy.enable_prefix_reuse = reuse;
    return policy;
}

}  // namespace

TEST(CacheGroupPublicationTest, PublishesOnlyDenseReusableGroups) {
    EXPECT_TRUE(cacheGroupPublishesPrefixChain(policyOf(CacheGroupType::FULL)));
    EXPECT_FALSE(cacheGroupPublishesPrefixChain(policyOf(CacheGroupType::FULL, false)));
    EXPECT_FALSE(cacheGroupPublishesPrefixChain(policyOf(CacheGroupType::LINEAR)));
    EXPECT_FALSE(cacheGroupPublishesPrefixChain(policyOf(CacheGroupType::SWA)));
}

TEST(CacheGroupPublicationTest, SelectsEveryEligibleGroupInOrder) {
    const auto ids = reuseParticipatingGroupIdsFromPolicies({
        policyOf(CacheGroupType::FULL),
        policyOf(CacheGroupType::SWA),
        policyOf(CacheGroupType::FULL, false),
        policyOf(CacheGroupType::FULL),
        policyOf(CacheGroupType::LINEAR),
    });
    EXPECT_EQ((std::vector<int>{0, 3}), ids);
    EXPECT_TRUE(reuseParticipatingGroupIdsFromPolicies({}).empty());
}

}  // namespace rtp_llm::test
