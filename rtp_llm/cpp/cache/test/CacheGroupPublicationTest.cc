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
    const std::vector<CacheGroupPolicy> policies({
        policyOf(CacheGroupType::FULL),
        policyOf(CacheGroupType::SWA),
        policyOf(CacheGroupType::FULL, false),
        policyOf(CacheGroupType::FULL),
        policyOf(CacheGroupType::LINEAR),
    });
    std::vector<size_t>                 selected;
    for (size_t gid = 0; gid < policies.size(); ++gid) {
        if (cacheGroupPublishesPrefixChain(policies[gid])) {
            selected.push_back(gid);
        }
    }
    EXPECT_EQ((std::vector<size_t>{0, 3}), selected);
}

}  // namespace rtp_llm::test
