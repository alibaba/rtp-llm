#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(RemoteCacheConfigTest, ParsesBinaryFlag) {
    EXPECT_EQ(parseRemoteCacheGdrEnabled("1"), std::optional<bool>(true));
    EXPECT_EQ(parseRemoteCacheGdrEnabled("0"), std::optional<bool>(false));
    EXPECT_EQ(parseRemoteCacheGdrEnabled(""), std::nullopt);
    EXPECT_EQ(parseRemoteCacheGdrEnabled("true"), std::nullopt);
    EXPECT_EQ(parseRemoteCacheGdrEnabled(" 1 "), std::nullopt);
    EXPECT_EQ(parseRemoteCacheGdrEnabled("unexpected"), std::nullopt);
    EXPECT_EQ(parseRemoteCacheGdrEnabled("2"), std::nullopt);
}

}  // namespace
}  // namespace rtp_llm
