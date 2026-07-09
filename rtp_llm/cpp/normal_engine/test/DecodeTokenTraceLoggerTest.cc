#include "rtp_llm/cpp/normal_engine/DecodeTokenTraceLogger.h"

#include "gtest/gtest.h"

namespace rtp_llm {

TEST(DecodeTokenTraceLoggerTest, defaultConfigIsDisabled) {
    auto config = DecodeTokenTraceConfig::fromValues(false, "", "", true, 16, false, "", 128, 4, 128);

    EXPECT_FALSE(config.enabled);
    EXPECT_FALSE(config.bad_watch_enabled);
    EXPECT_FALSE(config.matches("trace_test3_bad_0055"));
}

TEST(DecodeTokenTraceLoggerTest, commaSeparatedFiltersMatchBySubstring) {
    auto config =
        DecodeTokenTraceConfig::fromValues(true, "test3_bad_0040, test3_bad_0055", "", true, 16, false, "", 128, 4, 128);

    EXPECT_TRUE(config.matches("prefix_seq000375_test3_bad_0055_src"));
    EXPECT_TRUE(config.matches("prefix_seq001576_test3_bad_0040_src"));
    EXPECT_FALSE(config.matches("prefix_seq000001_test3_bad_0001_src"));
}

TEST(DecodeTokenTraceLoggerTest, emptyFilterMatchesAllWhenEnabled) {
    auto config = DecodeTokenTraceConfig::fromValues(true, "", "", true, 16, false, "", 128, 4, 128);

    EXPECT_TRUE(config.matches("any_trace"));
    EXPECT_TRUE(config.matches(""));
}

TEST(DecodeTokenTraceLoggerTest, badWatchConfigClampsMinimums) {
    auto config = DecodeTokenTraceConfig::fromValues(false, "", "", true, 16, true, "/tmp/watch.jsonl", 1, 1, -1);

    EXPECT_TRUE(config.bad_watch_enabled);
    EXPECT_EQ("/tmp/watch.jsonl", config.bad_watch_output_path);
    EXPECT_EQ(16, config.bad_watch_tail_size);
    EXPECT_EQ(2, config.bad_watch_min_cf);
    EXPECT_EQ(0, config.bad_watch_history_size);
}

TEST(DecodeTokenTraceLoggerTest, jsonEscapeHandlesControlCharacters) {
    EXPECT_EQ("a\\\\b\\\"c\\nd", DecodeTokenTraceLogger::jsonEscape("a\\b\"c\nd"));
}

TEST(DecodeTokenTraceLoggerTest, repeatedSuffixDetectorReportsAlternatingPattern) {
    const std::vector<int> tokens = {42, 7, 59140, 220, 59140, 220, 59140, 220, 59140, 220};

    auto info = DecodeTokenTraceLogger::debugFindRepeatedSuffixForTest(tokens, 8, 4);

    ASSERT_TRUE(info.matched);
    EXPECT_EQ(2, info.pattern_size);
    EXPECT_EQ(4, info.repeat_count);
    EXPECT_EQ((std::vector<int>{59140, 220}), info.pattern);
}

TEST(DecodeTokenTraceLoggerTest, repeatedSuffixDetectorIgnoresNonSuffixRepeats) {
    const std::vector<int> tokens = {59140, 220, 59140, 220, 1, 2, 3, 4};

    auto info = DecodeTokenTraceLogger::debugFindRepeatedSuffixForTest(tokens, 8, 4);

    EXPECT_FALSE(info.matched);
}

TEST(DecodeTokenTraceLoggerTest, blockTraceSeparatesModelStepFromNextStep) {
    auto at_boundary = DecodeTokenTraceLogger::debugComputeBlockTraceForTest(10688, 64);
    EXPECT_EQ(166, at_boundary.model_read_logical_block);
    EXPECT_EQ(166, at_boundary.model_write_logical_block);
    EXPECT_EQ(166, at_boundary.next_read_logical_block);
    EXPECT_EQ(167, at_boundary.next_write_logical_block);

    auto after_boundary = DecodeTokenTraceLogger::debugComputeBlockTraceForTest(10689, 64);
    EXPECT_EQ(166, after_boundary.model_read_logical_block);
    EXPECT_EQ(167, after_boundary.model_write_logical_block);
    EXPECT_EQ(167, after_boundary.next_read_logical_block);
    EXPECT_EQ(167, after_boundary.next_write_logical_block);
}

}  // namespace rtp_llm
