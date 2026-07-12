#include "rtp_llm/cpp/normal_engine/DecodeTokenTraceLogger.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include <sys/mman.h>
#include <unistd.h>

#include "rtp_llm/cpp/utils/DecodeProbeTrigger.h"
#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

bool unlinkSharedMemoryIfPresent(const std::string& name) {
    return ::shm_unlink(name.c_str()) == 0 || errno == ENOENT;
}

class EnvironmentVariableGuard {
public:
    explicit EnvironmentVariableGuard(const char* name): name_(name) {}

    void set(const char* value) {
        const char* previous = std::getenv(name_.c_str());
        was_set_             = previous != nullptr;
        if (was_set_) {
            previous_value_ = previous;
        }
        ASSERT_EQ(0, setenv(name_.c_str(), value, 1));
    }

    void restore() const {
        if (was_set_) {
            EXPECT_EQ(0, setenv(name_.c_str(), previous_value_.c_str(), 1));
        } else {
            EXPECT_EQ(0, unsetenv(name_.c_str()));
        }
    }

private:
    std::string name_;
    std::string previous_value_;
    bool        was_set_{false};
};

class RetrospectiveProbeTestEnvironment : public ::testing::Environment {
public:
    RetrospectiveProbeTestEnvironment(): debug_("RTPLLM_RETROSPECTIVE_PROBE_DEBUG"),
                                         shm_name_env_("RTPLLM_RETROSPECTIVE_PROBE_SHM_NAME"),
                                         world_rank_("WORLD_RANK"),
                                         world_size_("WORLD_SIZE") {}

    void SetUp() override {
        shm_name_ = "/rtpllm_decode_token_trace_logger_test_" + std::to_string(::getpid());
        ASSERT_TRUE(unlinkSharedMemoryIfPresent(shm_name_)) << std::strerror(errno);
        debug_.set("1");
        shm_name_env_.set(shm_name_.c_str());
        world_rank_.set("0");
        world_size_.set("1");
    }

    void TearDown() override {
        EXPECT_TRUE(unlinkSharedMemoryIfPresent(shm_name_)) << std::strerror(errno);
        world_size_.restore();
        world_rank_.restore();
        shm_name_env_.restore();
        debug_.restore();
    }

private:
    EnvironmentVariableGuard debug_;
    EnvironmentVariableGuard shm_name_env_;
    EnvironmentVariableGuard world_rank_;
    EnvironmentVariableGuard world_size_;
    std::string              shm_name_;
};

[[maybe_unused]] ::testing::Environment* const retrospective_probe_test_environment =
    ::testing::AddGlobalTestEnvironment(new RetrospectiveProbeTestEnvironment);

uint64_t expectAndAcknowledgeEvent(const std::string& trace_id, const std::string& reason, int sequence_length) {
    DecodeProbeTriggerEvent event;
    if (!DecodeProbeTrigger::peek(event)) {
        ADD_FAILURE() << "expected a retrospective trigger event";
        return 0;
    }
    EXPECT_EQ(trace_id, event.trace_id);
    EXPECT_EQ(reason, event.reason);
    EXPECT_EQ(sequence_length, event.observed_sequence_length);
    EXPECT_TRUE(DecodeProbeTrigger::acknowledge(event.generation));
    return event.generation;
}

uint64_t currentTriggerGeneration() {
    DecodeProbeTriggerEvent event;
    return DecodeProbeTrigger::peek(event) ? event.generation : 0;
}

}  // namespace

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

TEST(DecodeTokenTraceLoggerTest, CfPatternPublishesOneRetrospectiveEvent) {
    const std::string trace_id = "cf_probe_trace";
    const std::vector<int> tokens = {
        27, 9500, 1419, 9500, 29, 27, 9500, 1419, 9500, 29,
        27, 9500, 1419, 9500, 29, 27, 9500, 1419, 9500, 29,
    };

    ASSERT_TRUE(DecodeTokenTraceLogger::debugFeedBadWatchTokensForTest(trace_id, tokens, 123));
    const auto generation = expectAndAcknowledgeEvent(trace_id, "cf_tail_repeat", 123);
    EXPECT_FALSE(DecodeTokenTraceLogger::debugFeedBadWatchTokensForTest(trace_id, tokens, 143));
    EXPECT_EQ(generation, currentTriggerGeneration());
}

TEST(DecodeTokenTraceLoggerTest, GenericRepeatedSuffixPublishesOneEvent) {
    const std::string trace_id = "repeated_suffix_probe_trace";
    const std::vector<int> tokens(8, 59140);

    ASSERT_TRUE(DecodeTokenTraceLogger::debugFeedBadWatchTokensForTest(trace_id, tokens, 456));
    const auto generation = expectAndAcknowledgeEvent(trace_id, "repeated_suffix", 456);
    EXPECT_FALSE(DecodeTokenTraceLogger::debugFeedBadWatchTokensForTest(trace_id, tokens, 464));
    EXPECT_EQ(generation, currentTriggerGeneration());
}

TEST(DecodeTokenTraceLoggerTest, RepeatedKpOpenPublishesBeforeGenericRepeatThreshold) {
    const std::string trace_id = "repeated_kp_open_probe_trace";
    // Qwen3.6 tokenizes "<kp><kp><kp" as the sequence below. This is already
    // an invalid nested protocol prefix, but contains no pattern repeated eight times.
    const std::vector<int> tokens = {27, 46880, 1721, 46880, 1721, 46880};

    ASSERT_TRUE(DecodeTokenTraceLogger::debugFeedBadWatchTokensForTest(trace_id, tokens, 789));
    const auto generation = expectAndAcknowledgeEvent(trace_id, "repeated_kp_open", 789);
    EXPECT_FALSE(DecodeTokenTraceLogger::debugFeedBadWatchTokensForTest(trace_id, {1721, 46880}, 791));
    EXPECT_EQ(generation, currentTriggerGeneration());
}

TEST(DecodeTokenTraceLoggerTest, LongAdjacentSpanDoesNotPublishAfterTwoCopies) {
    std::vector<int>  pattern;
    for (int token = 100; token < 120; ++token) {
        pattern.push_back(token);
    }
    std::vector<int> tokens = {7, 8, 9};
    tokens.insert(tokens.end(), pattern.begin(), pattern.end());
    tokens.insert(tokens.end(), pattern.begin(), pattern.end());
    const auto generation_before = currentTriggerGeneration();

    EXPECT_FALSE(DecodeTokenTraceLogger::debugFeedBadWatchTokensForTest("long_span_two_copy_trace", tokens, 900));
    EXPECT_EQ(generation_before, currentTriggerGeneration());
}

TEST(DecodeTokenTraceLoggerTest, LongAdjacentSpanPublishesAfterThreeCopies) {
    const std::string trace_id = "long_span_three_copy_trace";
    std::vector<int>  pattern;
    for (int token = 100; token < 120; ++token) {
        pattern.push_back(token);
    }
    std::vector<int> tokens = {7, 8, 9};
    tokens.insert(tokens.end(), pattern.begin(), pattern.end());
    tokens.insert(tokens.end(), pattern.begin(), pattern.end());
    tokens.insert(tokens.end(), pattern.begin(), pattern.end());

    ASSERT_TRUE(DecodeTokenTraceLogger::debugFeedBadWatchTokensForTest(trace_id, tokens, 920));
    const auto generation = expectAndAcknowledgeEvent(trace_id, "repeated_long_span", 920);
    EXPECT_FALSE(DecodeTokenTraceLogger::debugFeedBadWatchTokensForTest(trace_id, {121}, 921));
    EXPECT_EQ(generation, currentTriggerGeneration());
}

TEST(DecodeTokenTraceLoggerTest, ShortAdjacentSpanDoesNotPublish) {
    const std::vector<int> pattern = {200, 201, 202, 203, 204, 205, 206, 207};
    std::vector<int>       tokens  = pattern;
    tokens.insert(tokens.end(), pattern.begin(), pattern.end());
    const auto generation_before = currentTriggerGeneration();

    EXPECT_FALSE(DecodeTokenTraceLogger::debugFeedBadWatchTokensForTest("short_span_trace", tokens, 32));
    EXPECT_EQ(generation_before, currentTriggerGeneration());
}

TEST(DecodeTokenTraceLoggerTest, NormalFormatTagsDoNotPublishEvent) {
    const std::vector<int> tokens = {
        27, 9500, 1419, 9500, 29, 42, 7, 59140, 220,
        27, 46880, 29, 42, 510, 46880, 29,
        27, 46880, 29, 43, 510, 46880, 29,
    };
    const auto generation_before = currentTriggerGeneration();

    EXPECT_FALSE(DecodeTokenTraceLogger::debugFeedBadWatchTokensForTest("normal_format_trace", tokens, 64));
    EXPECT_EQ(generation_before, currentTriggerGeneration());
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

TEST(DecodeTokenTraceLoggerTest, reuseTailUsesContainingLogicalBlock) {
    EXPECT_EQ(-1, DecodeTokenTraceLogger::debugComputeReuseTailLogicalBlockForTest(0, 64));
    EXPECT_EQ(-1, DecodeTokenTraceLogger::debugComputeReuseTailLogicalBlockForTest(64, 0));
    EXPECT_EQ(0, DecodeTokenTraceLogger::debugComputeReuseTailLogicalBlockForTest(1, 64));
    EXPECT_EQ(0, DecodeTokenTraceLogger::debugComputeReuseTailLogicalBlockForTest(64, 64));
    EXPECT_EQ(1, DecodeTokenTraceLogger::debugComputeReuseTailLogicalBlockForTest(65, 64));
    EXPECT_EQ(106, DecodeTokenTraceLogger::debugComputeReuseTailLogicalBlockForTest(6848, 64));
}

TEST(DecodeTokenTraceLoggerTest, blockWindowClampsWithoutIntegerOverflow) {
    EXPECT_EQ((std::pair<int, int>{104, 109}), DecodeTokenTraceLogger::debugComputeBlockWindowForTest(106, 200));
    EXPECT_EQ((std::pair<int, int>{128, 128}),
              DecodeTokenTraceLogger::debugComputeBlockWindowForTest(std::numeric_limits<int>::max(), 128));
    EXPECT_EQ((std::pair<int, int>{0, 0}), DecodeTokenTraceLogger::debugComputeBlockWindowForTest(-1, 128));
}

}  // namespace rtp_llm
