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

TEST(CacheGroupPublicationTest, PublishesReusableFullAndSwaGroups) {
    EXPECT_TRUE(cacheGroupPublishesEvents(policyOf(CacheGroupType::FULL)));
    EXPECT_FALSE(cacheGroupPublishesEvents(policyOf(CacheGroupType::FULL, false)));
    EXPECT_FALSE(cacheGroupPublishesEvents(policyOf(CacheGroupType::LINEAR)));
    EXPECT_TRUE(cacheGroupPublishesEvents(policyOf(CacheGroupType::SWA)));
    EXPECT_FALSE(cacheGroupPublishesEvents(policyOf(CacheGroupType::LINEAR, false)));
    EXPECT_FALSE(cacheGroupPublishesEvents(policyOf(CacheGroupType::SWA, false)));
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
        if (cacheGroupPublishesEvents(policies[gid])) {
            selected.push_back(gid);
        }
    }
    EXPECT_EQ((std::vector<size_t>{0, 1, 3}), selected);
}

}  // namespace rtp_llm::test
