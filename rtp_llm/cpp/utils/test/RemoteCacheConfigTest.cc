#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(RemoteCacheConfigTest, ParsesBooleanSpellingsWithoutProcessEnvironment) {
    for (const auto value : {"1", "true", "TRUE", "yes", "on", "enable", "enabled", "  On  "}) {
        ASSERT_EQ(parseRemoteCacheGdrEnabled(value), std::optional<bool>(true)) << value;
    }
    for (const auto value : {"0", "false", "FALSE", "no", "off", "disable", "disabled", "  Off  "}) {
        ASSERT_EQ(parseRemoteCacheGdrEnabled(value), std::optional<bool>(false)) << value;
    }
    EXPECT_EQ(parseRemoteCacheGdrEnabled(""), std::nullopt);
    EXPECT_EQ(parseRemoteCacheGdrEnabled("unexpected"), std::nullopt);
    EXPECT_EQ(parseRemoteCacheGdrEnabled("2"), std::nullopt);
}

}  // namespace
}  // namespace rtp_llm
