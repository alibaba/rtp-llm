// Focused getMultimodalTokenSpans regression: the d3 nested-aware 3-separator
// descriptors (start/end/payload) against the official prompt shape, plus the
// unchanged flat 1/2-separator behavior. Stdlib-only parser, no engine behind it.
#include <stdexcept>
#include <utility>
#include <vector>
#include "gtest/gtest.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTokenUtils.h"

using namespace std;

namespace rtp_llm {

namespace {
// d3-shaped token ids: ordinary text 1..9, video pad 80, image pad 81,
// vision markers 90 (start) / 91 (end).
constexpr int32_t kVideoPad    = 80;
constexpr int32_t kImagePad    = 81;
constexpr int32_t kVisionStart = 90;
constexpr int32_t kVisionEnd   = 91;

vector<pair<int32_t, int32_t>>
spansOf(const vector<int32_t>& tokens, const vector<vector<int64_t>>& separators, bool include_separators = false) {
    return getMultimodalTokenSpans(tokens, separators, include_separators);
}
}  // namespace

TEST(MultimodalTokenSpansTest, d3SingleVideoReturnsOneLeafPerTwoFrameGroup) {
    // Official shape: the template's outer pair wraps the per-group
    // "<timestamp> <|vision_start|> pads <|vision_end|>" pairs.
    const vector<int32_t> tokens = {1,
                                    kVisionStart,
                                    7,
                                    kVisionStart,
                                    kVideoPad,
                                    kVideoPad,
                                    kVideoPad,
                                    kVideoPad,
                                    kVisionEnd,
                                    8,
                                    kVisionStart,
                                    kVideoPad,
                                    kVideoPad,
                                    kVideoPad,
                                    kVideoPad,
                                    kVisionEnd,
                                    kVisionEnd,
                                    9};
    const auto            spans  = spansOf(tokens, {{kVisionStart, kVisionEnd, kVideoPad}});
    // Only the two pad runs are spans; the outer pair, the timestamps and all
    // markers stay text.
    ASSERT_EQ(spans.size(), 2U);
    EXPECT_EQ(spans[0], make_pair(4, 8));
    EXPECT_EQ(spans[1], make_pair(11, 15));
}

TEST(MultimodalTokenSpansTest, d3ImageAndVideoMixtureReturnsLeavesInPromptOrder) {
    // The image placeholder's single pad is a leaf of the image descriptor; the
    // video descriptor only takes the group pad runs; both share the markers.
    const vector<int32_t> tokens = {1,
                                    kVisionStart,
                                    kImagePad,
                                    kVisionEnd,
                                    5,
                                    kVisionStart,
                                    7,
                                    kVisionStart,
                                    kVideoPad,
                                    kVideoPad,
                                    kVisionEnd,
                                    8,
                                    kVisionStart,
                                    kVideoPad,
                                    kVideoPad,
                                    kVisionEnd,
                                    kVisionEnd,
                                    9};
    const auto spans = spansOf(tokens, {{kVisionStart, kVisionEnd, kImagePad}, {kVisionStart, kVisionEnd, kVideoPad}});
    // One span per actual feature group (image + 2 groups), in prompt order.
    ASSERT_EQ(spans.size(), 3U);
    EXPECT_EQ(spans[0], make_pair(2, 3));
    EXPECT_EQ(spans[1], make_pair(8, 10));
    EXPECT_EQ(spans[2], make_pair(13, 15));
}

TEST(MultimodalTokenSpansTest, d3MultipleVideosKeepEveryVideoIndependent) {
    // Two videos, 1-token groups (the expansion of a 1-token group is itself
    // placeholder-shaped): every group's own pad run is its own leaf span.
    const vector<int32_t> tokens = {1,
                                    kVisionStart,
                                    7,
                                    kVisionStart,
                                    kVideoPad,
                                    kVisionEnd,
                                    8,
                                    kVisionStart,
                                    kVideoPad,
                                    kVisionEnd,
                                    kVisionEnd,
                                    4,
                                    kVisionStart,
                                    6,
                                    kVisionStart,
                                    kVideoPad,
                                    kVideoPad,
                                    kVisionEnd,
                                    kVisionEnd,
                                    9};
    const auto            spans  = spansOf(tokens, {{kVisionStart, kVisionEnd, kVideoPad}});
    ASSERT_EQ(spans.size(), 3U);
    EXPECT_EQ(spans[0], make_pair(4, 5));
    EXPECT_EQ(spans[1], make_pair(8, 9));
    EXPECT_EQ(spans[2], make_pair(15, 17));
}

TEST(MultimodalTokenSpansTest, d3IncludeSeparatorsKeepsTheLeafMarkers) {
    // include_separators=true takes the whole leaf pair (markers + payload),
    // the same rule the 2-separator form applies; wrappers never become spans.
    const vector<int32_t> tokens = {1, kVisionStart, 7, kVisionStart, kVideoPad, kVideoPad, kVisionEnd, kVisionEnd, 9};
    const auto            spans  = spansOf(tokens, {{kVisionStart, kVisionEnd, kVideoPad}}, true);
    ASSERT_EQ(spans.size(), 1U);
    EXPECT_EQ(spans[0], make_pair(3, 7));
}

TEST(MultimodalTokenSpansTest, d3MalformedNestingIsRejected) {
    // A payload token directly inside a pair that also carries text.
    EXPECT_THROW(spansOf({kVisionStart, 7, kVideoPad, kVisionEnd}, {{kVisionStart, kVisionEnd, kVideoPad}}),
                 std::invalid_argument);
    // A payload run wrapping a nested pair: the outer "leaf" contains a child.
    EXPECT_THROW(spansOf({kVisionStart, kVideoPad, kVisionStart, kVideoPad, kVisionEnd, kVisionEnd},
                         {{kVisionStart, kVisionEnd, kVideoPad}}),
                 std::invalid_argument);
    // Image and video payload tokens inside ONE leaf pair: neither descriptor
    // may claim a mixed leaf.
    EXPECT_THROW(spansOf({kVisionStart, kImagePad, kVideoPad, kVisionEnd},
                         {{kVisionStart, kVisionEnd, kImagePad}, {kVisionStart, kVisionEnd, kVideoPad}}),
                 std::invalid_argument);
    // A payload token outside every pair.
    EXPECT_THROW(spansOf({1, kVideoPad, 2}, {{kVisionStart, kVisionEnd, kVideoPad}}), std::invalid_argument);
    // An end marker without a start.
    EXPECT_THROW(spansOf({kVisionEnd}, {{kVisionStart, kVisionEnd, kVideoPad}}), std::invalid_argument);
    // An unclosed start.
    EXPECT_THROW(spansOf({kVisionStart, kVideoPad}, {{kVisionStart, kVisionEnd, kVideoPad}}), std::invalid_argument);
}

TEST(MultimodalTokenSpansTest, d3WrapperPairsWithoutPayloadStayText) {
    // A pair whose direct content is only text (no payload, no nested pair) is
    // a wrapper, never a span: nothing feature-bearing to replace.
    const auto spans = spansOf({1, kVisionStart, 7, kVisionEnd, 2}, {{kVisionStart, kVisionEnd, kVideoPad}});
    EXPECT_TRUE(spans.empty());
}

TEST(MultimodalTokenSpansTest, flatSeparatorsStayStrict) {
    // 1-separator: every occurrence is its own span.
    const auto single = spansOf({1, kVideoPad, 2, kVideoPad, 3}, {{kVideoPad}});
    ASSERT_EQ(single.size(), 2U);
    EXPECT_EQ(single[0], make_pair(1, 2));
    EXPECT_EQ(single[1], make_pair(3, 4));
    // 2-separator: nested markers are still rejected outright.
    EXPECT_THROW(spansOf({kVisionStart, kVisionStart, kVisionEnd, kVisionEnd}, {{kVisionStart, kVisionEnd}}),
                 std::invalid_argument);
    EXPECT_THROW(spansOf({kVisionEnd}, {{kVisionStart, kVisionEnd}}), std::invalid_argument);
    EXPECT_THROW(spansOf({kVisionStart, 5}, {{kVisionStart, kVisionEnd}}), std::invalid_argument);
    // 2-separator flat pair keeps the interior span.
    const auto flat = spansOf({1, kVisionStart, 5, kVisionEnd, 2}, {{kVisionStart, kVisionEnd}});
    ASSERT_EQ(flat.size(), 1U);
    EXPECT_EQ(flat[0], make_pair(2, 3));
}

TEST(MultimodalTokenSpansTest, invalidSeparatorSizesAreRejected) {
    EXPECT_THROW(spansOf({1, 2}, {{}}), std::invalid_argument);
    EXPECT_THROW(spansOf({1, 2}, {{2, 3, 4, 5}}), std::invalid_argument);
    // The 3-separator tokens must be distinct.
    EXPECT_THROW(spansOf({1, 2}, {{kVisionStart, kVisionStart, kVideoPad}}), std::invalid_argument);
    EXPECT_THROW(spansOf({1, 2}, {{kVisionStart, kVisionEnd, kVisionStart}}), std::invalid_argument);
}

TEST(MultimodalTokenSpansTest, overlappingSpansAcrossSeparatorsAreRejected) {
    // Two descriptors with different markers that interleave: the shared
    // overlap check must still reject the result, 3-separator or not.
    EXPECT_THROW(spansOf({kVisionStart, 92, 5, 93, kVisionEnd}, {{kVisionStart, kVisionEnd}, {92, 93}}),
                 std::invalid_argument);
}

}  // namespace rtp_llm
