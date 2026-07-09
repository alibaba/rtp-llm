#include "rtp_llm/cpp/normal_engine/DecodeTokenTraceLogger.h"

#include "gtest/gtest.h"

namespace rtp_llm {

TEST(DecodeTokenTraceLoggerTest, defaultConfigIsDisabled) {
    auto config = DecodeTokenTraceConfig::fromValues(false, "", "", true, 16);

    EXPECT_FALSE(config.enabled);
    EXPECT_FALSE(config.matches("trace_test3_bad_0055"));
}

TEST(DecodeTokenTraceLoggerTest, commaSeparatedFiltersMatchBySubstring) {
    auto config = DecodeTokenTraceConfig::fromValues(true, "test3_bad_0040, test3_bad_0055", "", true, 16);

    EXPECT_TRUE(config.matches("prefix_seq000375_test3_bad_0055_src"));
    EXPECT_TRUE(config.matches("prefix_seq001576_test3_bad_0040_src"));
    EXPECT_FALSE(config.matches("prefix_seq000001_test3_bad_0001_src"));
}

TEST(DecodeTokenTraceLoggerTest, emptyFilterMatchesAllWhenEnabled) {
    auto config = DecodeTokenTraceConfig::fromValues(true, "", "", true, 16);

    EXPECT_TRUE(config.matches("any_trace"));
    EXPECT_TRUE(config.matches(""));
}

TEST(DecodeTokenTraceLoggerTest, jsonEscapeHandlesControlCharacters) {
    EXPECT_EQ("a\\\\b\\\"c\\nd", DecodeTokenTraceLogger::jsonEscape("a\\b\"c\nd"));
}

}  // namespace rtp_llm
