#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {
namespace test {

class LocalCacheKeysTest: public ::testing::Test {
protected:
    KVCacheResource make(const CacheKeysType& keys) {
        KVCacheResource r;
        r.cacheKeys() = keys;
        return r;
    }
};

TEST_F(LocalCacheKeysTest, CpSize1Passthrough) {
    auto r   = make({10, 20, 30, 40});
    auto out = r.localCacheKeys(0, 1);
    ASSERT_EQ(out.size(), 4u);
    EXPECT_EQ(out[0], 10);
    EXPECT_EQ(out[1], 20);
    EXPECT_EQ(out[2], 30);
    EXPECT_EQ(out[3], 40);
}

TEST_F(LocalCacheKeysTest, CpSize2EvenLengthLastRank) {
    auto r = make({100, 101, 200, 201, 300, 301, 400, 401});
    // last-rank stride: rank=1, size=2 → idx 1,3,5,7
    auto out = r.localCacheKeys(1, 2);
    ASSERT_EQ(out.size(), 4u);
    EXPECT_EQ(out[0], 101);
    EXPECT_EQ(out[1], 201);
    EXPECT_EQ(out[2], 301);
    EXPECT_EQ(out[3], 401);
}

TEST_F(LocalCacheKeysTest, CpSize2Rank0) {
    auto r   = make({100, 101, 200, 201});
    auto out = r.localCacheKeys(0, 2);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 100);
    EXPECT_EQ(out[1], 200);
}

TEST_F(LocalCacheKeysTest, CpSize4NonDivisibleLastRankShorter) {
    // 10 keys, cp_size=4 → last-rank (3) takes idx 3, 7 → length 2 (vs blocks=ceil(10/4)=3)
    auto r   = make({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto out = r.localCacheKeys(3, 4);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 3);
    EXPECT_EQ(out[1], 7);
}

TEST_F(LocalCacheKeysTest, EmptyKeys) {
    auto r   = make({});
    auto out = r.localCacheKeys(0, 4);
    EXPECT_TRUE(out.empty());
}

TEST_F(LocalCacheKeysTest, KeysShorterThanCpSizeReturnsEmptyForLastRank) {
    auto r   = make({42});
    auto out = r.localCacheKeys(3, 4);  // last-rank starts at idx 3, but only 1 key
    EXPECT_TRUE(out.empty());
}

TEST_F(LocalCacheKeysTest, KeysShorterThanCpSizeRank0HasOne) {
    auto r   = make({42});
    auto out = r.localCacheKeys(0, 4);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0], 42);
}

}  // namespace test
}  // namespace rtp_llm
