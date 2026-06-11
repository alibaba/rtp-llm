#include "rtp_llm/cpp/repetition/TokenToolCallLoopGuard.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace rtp_llm {
namespace {

int normalizePositive(int value, int fallback) {
    return value > 0 ? value : fallback;
}

std::vector<int> buildLps(const std::vector<int>& pattern) {
    std::vector<int> lps(pattern.size(), 0);
    std::size_t len = 0;
    for (std::size_t i = 1; i < pattern.size();) {
        if (pattern[i] == pattern[len]) {
            lps[i++] = static_cast<int>(++len);
        } else if (len > 0) {
            len = static_cast<std::size_t>(lps[len - 1]);
        } else {
            lps[i++] = 0;
        }
    }
    return lps;
}

std::vector<ToolCallMarkerIds> normalizeMarkers(std::vector<ToolCallMarkerIds> markers) {
    for (const auto& marker : markers) {
        if (marker.begin_ids.empty() || marker.end_ids.empty()) {
            throw std::invalid_argument("tool_call marker begin_ids/end_ids must be non-empty");
        }
    }
    return markers;
}

bool sameSpan(const CompletedToolCallSpan& lhs, const CompletedToolCallSpan& rhs) {
    return !lhs.overflow && !rhs.overflow && lhs.marker_index == rhs.marker_index &&
           lhs.token_ids == rhs.token_ids;
}

bool sameSpanIds(const CompletedToolCallSpan& span,
                 int marker_index,
                 const std::vector<int>& token_ids) {
    return !span.overflow && span.marker_index == marker_index &&
           span.token_ids == token_ids;
}

}  // namespace

ToolCallSpanRecorder::Matcher::Matcher(std::vector<int> ids):
    pattern(std::move(ids)), lps(buildLps(pattern)) {}

void ToolCallSpanRecorder::Matcher::reset() {
    state = 0;
}

bool ToolCallSpanRecorder::Matcher::feed(int token_id) {
    if (pattern.empty()) {
        return false;
    }
    while (state > 0 && token_id != pattern[state]) {
        state = static_cast<std::size_t>(lps[state - 1]);
    }
    if (token_id == pattern[state]) {
        ++state;
    }
    if (state != pattern.size()) {
        return false;
    }
    state = static_cast<std::size_t>(lps[state - 1]);
    return true;
}

ToolCallSpanRecorder::ToolCallSpanRecorder(std::vector<ToolCallMarkerIds> markers,
                                           int max_span_tokens):
    markers_(normalizeMarkers(std::move(markers))),
    max_span_tokens_(normalizePositive(max_span_tokens, 16384)) {
    begin_matchers_.reserve(markers_.size());
    end_matchers_.reserve(markers_.size());
    for (const auto& marker : markers_) {
        begin_matchers_.emplace_back(marker.begin_ids);
        end_matchers_.emplace_back(marker.end_ids);
    }
}

void ToolCallSpanRecorder::resetMatchers() {
    for (auto& matcher : begin_matchers_) {
        matcher.reset();
    }
    for (auto& matcher : end_matchers_) {
        matcher.reset();
    }
}

void ToolCallSpanRecorder::reset() {
    inside_span_ = false;
    current_marker_index_ = -1;
    current_span_ids_.clear();
    resetMatchers();
}

bool ToolCallSpanRecorder::beginMatched(int token_id, int* marker_index) {
    int matched_index = -1;
    std::size_t matched_size = 0;
    for (std::size_t i = 0; i < begin_matchers_.size(); ++i) {
        if (!begin_matchers_[i].feed(token_id)) {
            continue;
        }
        const std::size_t size = markers_[i].begin_ids.size();
        if (matched_index < 0 || size > matched_size) {
            matched_index = static_cast<int>(i);
            matched_size = size;
        }
    }
    if (matched_index < 0) {
        return false;
    }
    *marker_index = matched_index;
    return true;
}

bool ToolCallSpanRecorder::endMatched(int token_id) {
    if (current_marker_index_ < 0 ||
        current_marker_index_ >= static_cast<int>(end_matchers_.size())) {
        return false;
    }
    return end_matchers_[current_marker_index_].feed(token_id);
}

CompletedToolCallSpan ToolCallSpanRecorder::makeOverflowSpan() {
    CompletedToolCallSpan span;
    span.marker_index = current_marker_index_;
    span.overflow = true;
    return span;
}

std::vector<CompletedToolCallSpan> ToolCallSpanRecorder::updateMany(
    const std::vector<int>& token_ids) {
    std::vector<CompletedToolCallSpan> completed;
    for (int token_id : token_ids) {
        if (!inside_span_) {
            int marker_index = -1;
            if (!beginMatched(token_id, &marker_index)) {
                continue;
            }
            inside_span_ = true;
            current_marker_index_ = marker_index;
            current_span_ids_ = markers_[marker_index].begin_ids;
            for (auto& matcher : begin_matchers_) {
                matcher.reset();
            }
            end_matchers_[marker_index].reset();
            continue;
        }

        current_span_ids_.push_back(token_id);
        if (endMatched(token_id)) {
            CompletedToolCallSpan span;
            span.marker_index = current_marker_index_;
            span.token_ids = current_span_ids_;
            completed.push_back(std::move(span));
            inside_span_ = false;
            current_marker_index_ = -1;
            current_span_ids_.clear();
            resetMatchers();
            continue;
        }

        if (static_cast<int>(current_span_ids_.size()) > max_span_tokens_) {
            completed.push_back(makeOverflowSpan());
            inside_span_ = false;
            current_marker_index_ = -1;
            current_span_ids_.clear();
            resetMatchers();
        }
    }
    return completed;
}

TokenToolCallLoopGuard::TokenToolCallLoopGuard(std::vector<ToolCallMarkerIds> markers,
                                               int repeat_threshold,
                                               int max_span_tokens):
    markers_(normalizeMarkers(std::move(markers))),
    repeat_threshold_(std::max(2, repeat_threshold)),
    max_span_tokens_(normalizePositive(max_span_tokens, 16384)) {}

void TokenToolCallLoopGuard::resetCurrentChain() {
    seen_current_span_ = false;
    previous_marker_index_ = -1;
    previous_current_span_ids_.clear();
    previous_repeat_count_ = 0;
    previous_history_suffix_count_ = 0;
    previous_current_suffix_count_ = 0;
}

void TokenToolCallLoopGuard::reset() {
    history_scanned_ = false;
    history_spans_.clear();
    resetCurrentChain();
}

void TokenToolCallLoopGuard::ensureHistoryScanned(const std::vector<int>& input_ids) {
    if (history_scanned_) {
        return;
    }
    history_spans_ = scanToolCallSpansOnce(input_ids, markers_, max_span_tokens_);
    history_scanned_ = true;
}

int TokenToolCallLoopGuard::historySuffixCountFor(
    const CompletedToolCallSpan& current_span) const {
    int suffix_count = 0;
    for (const auto& history_span : history_spans_) {
        if (sameSpan(history_span, current_span)) {
            ++suffix_count;
        } else {
            suffix_count = 0;
        }
    }
    return suffix_count;
}

ToolCallLoopResult TokenToolCallLoopGuard::checkCompletedSpan(
    const std::vector<int>& input_ids,
    const CompletedToolCallSpan& current_span) {
    ToolCallLoopResult result;
    result.threshold = repeat_threshold_;
    result.current_span_tokens = static_cast<int>(current_span.token_ids.size());
    result.marker_index = current_span.marker_index;
    result.span_overflow = current_span.overflow ||
                           result.current_span_tokens > max_span_tokens_;

    if (result.span_overflow || current_span.token_ids.empty() ||
        current_span.marker_index < 0 ||
        current_span.marker_index >= static_cast<int>(markers_.size())) {
        resetCurrentChain();
        return result;
    }

    ensureHistoryScanned(input_ids);

    int history_suffix_count = 0;
    int current_suffix_count = 1;
    int repeat_count = 1;
    if (!seen_current_span_) {
        history_suffix_count = historySuffixCountFor(current_span);
        repeat_count = history_suffix_count + 1;
    } else if (sameSpanIds(current_span,
                           previous_marker_index_,
                           previous_current_span_ids_)) {
        history_suffix_count = previous_history_suffix_count_;
        current_suffix_count = previous_current_suffix_count_ + 1;
        repeat_count = previous_repeat_count_ + 1;
    }

    seen_current_span_ = true;
    previous_marker_index_ = current_span.marker_index;
    previous_current_span_ids_ = current_span.token_ids;
    previous_repeat_count_ = repeat_count;
    previous_history_suffix_count_ = history_suffix_count;
    previous_current_suffix_count_ = current_suffix_count;

    result.repeat_count = repeat_count;
    result.history_suffix_count = history_suffix_count;
    result.current_suffix_count = current_suffix_count;
    result.hit = repeat_count >= repeat_threshold_;
    return result;
}

ToolCallLoopResult TokenToolCallLoopGuard::checkCompletedSpan(
    const std::vector<int>& input_ids,
    const std::vector<int>& current_span_ids,
    int marker_index,
    bool span_overflow) {
    CompletedToolCallSpan span;
    span.marker_index = marker_index;
    span.token_ids = current_span_ids;
    span.overflow = span_overflow;
    return checkCompletedSpan(input_ids, span);
}

std::vector<CompletedToolCallSpan> scanToolCallSpansOnce(
    const std::vector<int>& token_ids,
    const std::vector<ToolCallMarkerIds>& markers,
    int max_span_tokens) {
    ToolCallSpanRecorder recorder(markers, max_span_tokens);
    return recorder.updateMany(token_ids);
}

ToolCallLoopResult checkToolCallLoopOnce(
    const std::vector<int>& input_ids,
    const std::vector<int>& current_span_ids,
    int marker_index,
    const std::vector<ToolCallMarkerIds>& markers,
    int repeat_threshold,
    int max_span_tokens) {
    TokenToolCallLoopGuard guard(markers, repeat_threshold, max_span_tokens);
    return guard.checkCompletedSpan(input_ids, current_span_ids, marker_index);
}

}  // namespace rtp_llm
