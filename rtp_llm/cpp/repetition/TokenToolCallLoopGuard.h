#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace rtp_llm {

struct ToolCallMarkerIds {
    std::vector<int> begin_ids;
    std::vector<int> end_ids;
    std::string name;
};

struct CompletedToolCallSpan {
    int marker_index = -1;
    std::vector<int> token_ids;
    bool overflow = false;
};

struct ToolCallLoopResult {
    bool hit = false;
    int repeat_count = 0;
    int threshold = 0;
    int current_span_tokens = 0;
    int marker_index = -1;
    int history_suffix_count = 0;
    int current_suffix_count = 0;
    bool span_overflow = false;
};

class ToolCallSpanRecorder {
public:
    explicit ToolCallSpanRecorder(std::vector<ToolCallMarkerIds> markers,
                                  int max_span_tokens = 16384);

    void reset();
    std::vector<CompletedToolCallSpan> updateMany(const std::vector<int>& token_ids);
    bool insideSpan() const {
        return inside_span_;
    }

private:
    struct Matcher {
        std::vector<int> pattern;
        std::vector<int> lps;
        std::size_t state = 0;

        Matcher() = default;
        explicit Matcher(std::vector<int> ids);

        void reset();
        bool feed(int token_id);
    };

    std::vector<ToolCallMarkerIds> markers_;
    std::vector<Matcher> begin_matchers_;
    std::vector<Matcher> end_matchers_;
    int max_span_tokens_ = 16384;
    bool inside_span_ = false;
    int current_marker_index_ = -1;
    std::vector<int> current_span_ids_;

    void resetMatchers();
    bool beginMatched(int token_id, int* marker_index);
    bool endMatched(int token_id);
    CompletedToolCallSpan makeOverflowSpan();
};

class TokenToolCallLoopGuard {
public:
    explicit TokenToolCallLoopGuard(std::vector<ToolCallMarkerIds> markers,
                                    int repeat_threshold = 5,
                                    int max_span_tokens = 16384);

    void reset();

    ToolCallLoopResult checkCompletedSpan(const std::vector<int>& input_ids,
                                          const CompletedToolCallSpan& current_span);

    ToolCallLoopResult checkCompletedSpan(const std::vector<int>& input_ids,
                                          const std::vector<int>& current_span_ids,
                                          int marker_index,
                                          bool span_overflow = false);

private:
    std::vector<ToolCallMarkerIds> markers_;
    int repeat_threshold_ = 5;
    int max_span_tokens_ = 16384;
    bool history_scanned_ = false;
    std::vector<CompletedToolCallSpan> history_spans_;
    bool seen_current_span_ = false;
    int previous_marker_index_ = -1;
    std::vector<int> previous_current_span_ids_;
    int previous_repeat_count_ = 0;
    int previous_history_suffix_count_ = 0;
    int previous_current_suffix_count_ = 0;

    void ensureHistoryScanned(const std::vector<int>& input_ids);
    int historySuffixCountFor(const CompletedToolCallSpan& current_span) const;
    void resetCurrentChain();
};

std::vector<CompletedToolCallSpan> scanToolCallSpansOnce(
    const std::vector<int>& token_ids,
    const std::vector<ToolCallMarkerIds>& markers,
    int max_span_tokens = 16384);

ToolCallLoopResult checkToolCallLoopOnce(
    const std::vector<int>& input_ids,
    const std::vector<int>& current_span_ids,
    int marker_index,
    const std::vector<ToolCallMarkerIds>& markers,
    int repeat_threshold = 5,
    int max_span_tokens = 16384);

}  // namespace rtp_llm
