#include "rtp_llm/cpp/repetition/TokenToolCallLoopGuard.h"

#include "gtest/gtest.h"

#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace rtp_llm {
namespace {

std::vector<int> toolA() {
    return {1, 2, 10, 3, 4};
}

std::vector<int> toolB() {
    return {1, 2, 20, 3, 4};
}

std::vector<int> withFillers(std::initializer_list<std::vector<int>> spans) {
    std::vector<int> ids;
    int              filler = 1000;
    for (const auto& span : spans) {
        ids.push_back(filler++);
        ids.insert(ids.end(), span.begin(), span.end());
    }
    ids.push_back(filler++);
    return ids;
}

TEST(TokenToolCallLoopGuardTest, InputFourPlusOutputHitsThresholdFive) {
    const auto result = checkToolCallLoop(
        withFillers({toolA(), toolA(), toolA(), toolA()}), toolA(), {{1, 2}}, {{3, 4}}, 5, 16);

    EXPECT_TRUE(result.hit);
    EXPECT_EQ(result.repeat_count, 5);
    EXPECT_EQ(result.current_span_tokens, 5);
}

TEST(TokenToolCallLoopGuardTest, InputThreePlusOutputDoesNotHitThresholdFive) {
    const auto result =
        checkToolCallLoop(withFillers({toolA(), toolA(), toolA()}), toolA(), {{1, 2}}, {{3, 4}}, 5, 16);

    EXPECT_FALSE(result.hit);
    EXPECT_EQ(result.repeat_count, 4);
}

TEST(TokenToolCallLoopGuardTest, BrokenHistoryTailDoesNotAttachEarlierMatches) {
    const auto result = checkToolCallLoop(
        withFillers({toolA(), toolA(), toolA(), toolA(), toolB()}), toolA(), {{1, 2}}, {{3, 4}}, 5, 16);

    EXPECT_FALSE(result.hit);
    EXPECT_EQ(result.repeat_count, 1);
}

TEST(TokenToolCallLoopGuardTest, OutputSameSpanContinuesHistoryChain) {
    const auto result = checkToolCallLoop(
        withFillers({toolA(), toolA(), toolA(), toolA()}),
        withFillers({toolA(), toolA()}),
        {{1, 2}},
        {{3, 4}},
        5,
        16);

    EXPECT_TRUE(result.hit);
    EXPECT_EQ(result.repeat_count, 6);
}

TEST(TokenToolCallLoopGuardTest, OutputDifferentSpanBreaksHistoryChain) {
    const auto result = checkToolCallLoop(
        withFillers({toolA(), toolA(), toolA(), toolA()}),
        withFillers({toolB(), toolA()}),
        {{1, 2}},
        {{3, 4}},
        5,
        16);

    EXPECT_FALSE(result.hit);
    EXPECT_EQ(result.repeat_count, 1);
}

TEST(TokenToolCallLoopGuardTest, OverflowSpanBreaksHistoryChain) {
    std::vector<int> output = {1, 2, 10, 11, 12, 1000};
    const auto       tail   = toolA();
    output.insert(output.end(), tail.begin(), tail.end());

    const auto result =
        checkToolCallLoop(withFillers({toolA(), toolA(), toolA(), toolA()}), output, {{1, 2}}, {{3, 4}}, 5, 5);

    EXPECT_FALSE(result.hit);
    EXPECT_EQ(result.repeat_count, 1);
}

TEST(TokenToolCallLoopGuardTest, NoCompletedOutputSpanDoesNotHit) {
    const auto result = checkToolCallLoop(
        withFillers({toolA(), toolA(), toolA(), toolA()}), {1, 2, 10}, {{1, 2}}, {{3, 4}}, 5, 16);

    EXPECT_FALSE(result.hit);
}

TEST(TokenToolCallLoopGuardTest, EmptyOutputIdsDoesNotHit) {
    const auto result =
        checkToolCallLoop(withFillers({toolA(), toolA(), toolA(), toolA()}), {}, {{1, 2}}, {{3, 4}}, 5, 16);

    EXPECT_FALSE(result.hit);
    EXPECT_EQ(result.repeat_count, 0);
    EXPECT_EQ(result.current_span_tokens, 0);
    EXPECT_EQ(result.marker_index, -1);
}

TEST(TokenToolCallLoopGuardTest, SharedPrefixBeginMarkersPreferLongestAndReportMarkerIndex) {
    const std::vector<int> long_tool = {1, 2, 9, 20, 5, 6};
    const auto result = checkToolCallLoop(
        withFillers({long_tool, long_tool, long_tool, long_tool}),
        long_tool,
        {{1, 2}, {1, 2, 9}},
        {{3, 4}, {5, 6}},
        5,
        16);

    EXPECT_TRUE(result.hit);
    EXPECT_EQ(result.repeat_count, 5);
    EXPECT_EQ(result.current_span_tokens, 6);
    EXPECT_EQ(result.marker_index, 1);
}

TEST(TokenToolCallLoopGuardTest, ThrowsOnMarkerListSizeMismatch) {
    EXPECT_THROW(checkToolCallLoop({}, {}, {{1, 2}}, {}, 5, 16), std::invalid_argument);
}

TEST(TokenToolCallLoopGuardTest, ThrowsOnEmptyBeginMarker) {
    EXPECT_THROW(checkToolCallLoop({}, {}, {{}}, {{3, 4}}, 5, 16), std::invalid_argument);
}

TEST(TokenToolCallLoopGuardTest, ThrowsOnEmptyEndMarker) {
    EXPECT_THROW(checkToolCallLoop({}, {}, {{1, 2}}, {{}}, 5, 16), std::invalid_argument);
}

}  // namespace
}  // namespace rtp_llm
