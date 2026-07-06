#include "rtp_llm/cpp/repetition/TokenToolCallLoopGuard.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#define RTP_EXPECT(expr)                                                                                               \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            std::cerr << __FILE__ << ":" << __LINE__ << " check failed: " << #expr << std::endl;                       \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (false)

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

void testInputFourPlusOutputHitsThresholdFive() {
    const auto result =
        checkToolCallLoop(withFillers({toolA(), toolA(), toolA(), toolA()}), toolA(), {{1, 2}}, {{3, 4}}, 5, 16);

    RTP_EXPECT(result.hit);
    RTP_EXPECT(result.repeat_count == 5);
    RTP_EXPECT(result.current_span_tokens == 5);
}

void testInputThreePlusOutputDoesNotHitThresholdFive() {
    const auto result = checkToolCallLoop(withFillers({toolA(), toolA(), toolA()}), toolA(), {{1, 2}}, {{3, 4}}, 5, 16);

    RTP_EXPECT(!result.hit);
    RTP_EXPECT(result.repeat_count == 4);
}

void testBrokenHistoryTailDoesNotAttachEarlierMatches() {
    const auto result = checkToolCallLoop(
        withFillers({toolA(), toolA(), toolA(), toolA(), toolB()}), toolA(), {{1, 2}}, {{3, 4}}, 5, 16);

    RTP_EXPECT(!result.hit);
    RTP_EXPECT(result.repeat_count == 1);
}

void testOutputSameSpanContinuesHistoryChain() {
    const auto result = checkToolCallLoop(
        withFillers({toolA(), toolA(), toolA(), toolA()}), withFillers({toolA(), toolA()}), {{1, 2}}, {{3, 4}}, 5, 16);

    RTP_EXPECT(result.hit);
    RTP_EXPECT(result.repeat_count == 6);
}

void testOutputDifferentSpanBreaksHistoryChain() {
    const auto result = checkToolCallLoop(
        withFillers({toolA(), toolA(), toolA(), toolA()}), withFillers({toolB(), toolA()}), {{1, 2}}, {{3, 4}}, 5, 16);

    RTP_EXPECT(!result.hit);
    RTP_EXPECT(result.repeat_count == 1);
}

void testOverflowSpanBreaksHistoryChain() {
    std::vector<int> output = {1, 2, 10, 11, 12, 1000};
    const auto       tail   = toolA();
    output.insert(output.end(), tail.begin(), tail.end());

    const auto result =
        checkToolCallLoop(withFillers({toolA(), toolA(), toolA(), toolA()}), output, {{1, 2}}, {{3, 4}}, 5, 5);

    RTP_EXPECT(!result.hit);
    RTP_EXPECT(result.repeat_count == 1);
}

void testNoCompletedOutputSpanDoesNotHit() {
    const auto result =
        checkToolCallLoop(withFillers({toolA(), toolA(), toolA(), toolA()}), {1, 2, 10}, {{1, 2}}, {{3, 4}}, 5, 16);

    RTP_EXPECT(!result.hit);
}

}  // namespace
}  // namespace rtp_llm

int main() {
    rtp_llm::testInputFourPlusOutputHitsThresholdFive();
    rtp_llm::testInputThreePlusOutputDoesNotHitThresholdFive();
    rtp_llm::testBrokenHistoryTailDoesNotAttachEarlierMatches();
    rtp_llm::testOutputSameSpanContinuesHistoryChain();
    rtp_llm::testOutputDifferentSpanBreaksHistoryChain();
    rtp_llm::testOverflowSpanBreaksHistoryChain();
    rtp_llm::testNoCompletedOutputSpanDoesNotHit();
    std::cout << "TokenToolCallLoopGuard tests passed\n";
    return 0;
}
