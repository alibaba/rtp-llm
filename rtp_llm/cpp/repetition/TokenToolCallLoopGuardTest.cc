#include "rtp_llm/cpp/repetition/TokenToolCallLoopGuard.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#define RTP_EXPECT(expr)                                                                            \
    do {                                                                                            \
        if (!(expr)) {                                                                              \
            std::cerr << __FILE__ << ":" << __LINE__ << " check failed: " << #expr << std::endl;   \
            std::abort();                                                                           \
        }                                                                                           \
    } while (false)

namespace rtp_llm {
namespace {

std::vector<ToolCallMarkerIds> testMarkers() {
    ToolCallMarkerIds marker;
    marker.begin_ids = {1, 2};
    marker.end_ids = {3, 4};
    marker.name = "test_invoke";
    return {marker};
}

std::vector<int> toolA() {
    return {1, 2, 10, 3, 4};
}

std::vector<int> toolB() {
    return {1, 2, 20, 3, 4};
}

std::vector<int> withFillers(std::initializer_list<std::vector<int>> spans) {
    std::vector<int> ids;
    int filler = 1000;
    for (const auto& span : spans) {
        ids.push_back(filler++);
        ids.insert(ids.end(), span.begin(), span.end());
    }
    ids.push_back(filler++);
    return ids;
}

void testRecorderMatchesAcrossChunks() {
    ToolCallSpanRecorder recorder(testMarkers(), 16);
    auto spans = recorder.updateMany({9, 1});
    RTP_EXPECT(spans.empty());
    spans = recorder.updateMany({2, 10, 3});
    RTP_EXPECT(spans.empty());
    spans = recorder.updateMany({4, 8});
    RTP_EXPECT(spans.size() == 1);
    RTP_EXPECT(spans[0].marker_index == 0);
    RTP_EXPECT(spans[0].token_ids == toolA());
    RTP_EXPECT(!spans[0].overflow);
    RTP_EXPECT(!recorder.insideSpan());
}

void testInputFourPlusCurrentHitsThresholdFive() {
    const auto markers = testMarkers();
    const auto input = withFillers({toolA(), toolA(), toolA(), toolA()});
    TokenToolCallLoopGuard guard(markers, 5, 16);

    const auto result = guard.checkCompletedSpan(input, toolA(), 0);

    RTP_EXPECT(result.hit);
    RTP_EXPECT(result.repeat_count == 5);
    RTP_EXPECT(result.history_suffix_count == 4);
    RTP_EXPECT(result.current_suffix_count == 1);
    RTP_EXPECT(result.current_span_tokens == 5);
}

void testInputThreePlusCurrentDoesNotHitThresholdFive() {
    const auto markers = testMarkers();
    const auto input = withFillers({toolA(), toolA(), toolA()});
    TokenToolCallLoopGuard guard(markers, 5, 16);

    const auto result = guard.checkCompletedSpan(input, toolA(), 0);

    RTP_EXPECT(!result.hit);
    RTP_EXPECT(result.repeat_count == 4);
    RTP_EXPECT(result.history_suffix_count == 3);
}

void testBrokenHistoryTailDoesNotAttachEarlierMatches() {
    const auto markers = testMarkers();
    const auto input = withFillers({toolA(), toolA(), toolA(), toolA(), toolB()});
    TokenToolCallLoopGuard guard(markers, 5, 16);

    const auto result = guard.checkCompletedSpan(input, toolA(), 0);

    RTP_EXPECT(!result.hit);
    RTP_EXPECT(result.repeat_count == 1);
    RTP_EXPECT(result.history_suffix_count == 0);
}

void testCurrentSameSpanContinuesHistoryChain() {
    const auto markers = testMarkers();
    const auto input = withFillers({toolA(), toolA(), toolA(), toolA()});
    TokenToolCallLoopGuard guard(markers, 5, 16);

    const auto first = guard.checkCompletedSpan(input, toolA(), 0);
    const auto second = guard.checkCompletedSpan(input, toolA(), 0);

    RTP_EXPECT(first.hit);
    RTP_EXPECT(first.repeat_count == 5);
    RTP_EXPECT(second.hit);
    RTP_EXPECT(second.repeat_count == 6);
    RTP_EXPECT(second.history_suffix_count == 4);
    RTP_EXPECT(second.current_suffix_count == 2);
}

void testCurrentDifferentSpanBreaksHistoryChain() {
    const auto markers = testMarkers();
    const auto input = withFillers({toolA(), toolA(), toolA(), toolA()});
    TokenToolCallLoopGuard guard(markers, 5, 16);

    const auto first = guard.checkCompletedSpan(input, toolB(), 0);
    const auto second = guard.checkCompletedSpan(input, toolA(), 0);

    RTP_EXPECT(!first.hit);
    RTP_EXPECT(first.repeat_count == 1);
    RTP_EXPECT(!second.hit);
    RTP_EXPECT(second.repeat_count == 1);
    RTP_EXPECT(second.history_suffix_count == 0);
}

void testOverflowSpanDoesNotHitAndResetsCurrentChain() {
    const auto markers = testMarkers();
    const auto input = withFillers({toolA(), toolA(), toolA(), toolA()});
    TokenToolCallLoopGuard guard(markers, 5, 16);

    const auto first = guard.checkCompletedSpan(input, toolA(), 0);
    CompletedToolCallSpan overflow;
    overflow.marker_index = 0;
    overflow.overflow = true;
    const auto overflow_result = guard.checkCompletedSpan(input, overflow);
    const auto after = guard.checkCompletedSpan(input, toolA(), 0);

    RTP_EXPECT(first.hit);
    RTP_EXPECT(!overflow_result.hit);
    RTP_EXPECT(overflow_result.span_overflow);
    RTP_EXPECT(after.hit);
    RTP_EXPECT(after.repeat_count == 5);
    RTP_EXPECT(after.current_suffix_count == 1);
}

void testRecorderOverflowEmitsOverflowSpan() {
    ToolCallSpanRecorder recorder(testMarkers(), 4);
    const auto spans = recorder.updateMany({1, 2, 10, 11, 12});

    RTP_EXPECT(spans.size() == 1);
    RTP_EXPECT(spans[0].overflow);
    RTP_EXPECT(spans[0].marker_index == 0);
    RTP_EXPECT(spans[0].token_ids.empty());
    RTP_EXPECT(!recorder.insideSpan());
}

}  // namespace
}  // namespace rtp_llm

int main() {
    rtp_llm::testRecorderMatchesAcrossChunks();
    rtp_llm::testInputFourPlusCurrentHitsThresholdFive();
    rtp_llm::testInputThreePlusCurrentDoesNotHitThresholdFive();
    rtp_llm::testBrokenHistoryTailDoesNotAttachEarlierMatches();
    rtp_llm::testCurrentSameSpanContinuesHistoryChain();
    rtp_llm::testCurrentDifferentSpanBreaksHistoryChain();
    rtp_llm::testOverflowSpanDoesNotHitAndResetsCurrentChain();
    rtp_llm::testRecorderOverflowEmitsOverflowSpan();
    std::cout << "TokenToolCallLoopGuard tests passed\n";
    return 0;
}
