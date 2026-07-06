#include "rtp_llm/cpp/repetition/TokenToolCallLoopGuard.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace rtp_llm {
namespace {

struct Marker {
    std::vector<int> begin_ids;
    std::vector<int> end_ids;
};

// One detected tool call: the token window from a begin marker through the
// matching end marker (inclusive). ``overflow`` flags a window that grew past
// the size cap without closing — it acts as a barrier and is never equal to a
// real span.
struct Span {
    int              marker_index = -1;
    std::vector<int> token_ids;
    bool             overflow = false;
};

std::vector<Marker> buildMarkers(const std::vector<std::vector<int>>& begin_ids,
                                 const std::vector<std::vector<int>>& end_ids) {
    if (begin_ids.size() != end_ids.size()) {
        throw std::invalid_argument("tool_call marker begin/end list size mismatch");
    }
    std::vector<Marker> markers;
    markers.reserve(begin_ids.size());
    for (std::size_t i = 0; i < begin_ids.size(); ++i) {
        if (begin_ids[i].empty() || end_ids[i].empty()) {
            throw std::invalid_argument("tool_call marker begin_ids/end_ids must be non-empty");
        }
        markers.push_back({begin_ids[i], end_ids[i]});
    }
    return markers;
}

// True when ``pattern`` equals the tokens ending exactly at index ``end``.
bool endsWith(const std::vector<int>& tokens, int end, const std::vector<int>& pattern) {
    const int len = static_cast<int>(pattern.size());
    if (end + 1 < len) {
        return false;
    }
    const int start = end - len + 1;
    for (int k = 0; k < len; ++k) {
        if (tokens[start + k] != pattern[k]) {
            return false;
        }
    }
    return true;
}

// Index of the marker whose begin sequence ends at ``end``, preferring the
// longest match when several markers share a prefix. -1 when none match.
int beginMarkerEndingAt(const std::vector<int>& tokens, int end, const std::vector<Marker>& markers) {
    int         best     = -1;
    std::size_t best_len = 0;
    for (std::size_t i = 0; i < markers.size(); ++i) {
        const auto& begin = markers[i].begin_ids;
        if (endsWith(tokens, end, begin) && (best < 0 || begin.size() > best_len)) {
            best     = static_cast<int>(i);
            best_len = begin.size();
        }
    }
    return best;
}

// Cut ``tokens`` into the list of completed tool-call spans, left to right.
//
// Each span is found in two phases, which is why there are two loops:
//   1. outer loop = find a span's BEGIN. Step one token at a time until some
//      begin marker finishes exactly at ``i``.
//   2. inner loop = find THAT span's END. From the next token on, scan forward
//      for the same marker's end sequence (or for the size cap).
// Once the end is found, the outer loop resumes right after it (``i = j + 1``),
// so spans never overlap and no token is scanned twice — the whole thing is
// O(n), not O(n^2), despite the nesting.
//
// Three ways the inner loop ends, per span:
//   - end marker found  -> push the [begin .. end] token window as a real span.
//   - size cap exceeded -> push an ``overflow`` barrier and move past it.
//   - input runs out    -> span never completed; drop it and stop scanning.
// The end is only matched against tokens after the begin finished
// (``end_start >= content_start``), so a begin and its end can never overlap.
std::vector<Span> scanSpans(const std::vector<int>& tokens, const std::vector<Marker>& markers, int max_span_tokens) {
    std::vector<Span> spans;
    const int         n = static_cast<int>(tokens.size());
    int               i = 0;
    while (i < n) {
        // Phase 1: is a begin marker completed at ``i``? If not, keep walking.
        const int marker_index = beginMarkerEndingAt(tokens, i, markers);
        if (marker_index < 0) {
            ++i;
            continue;
        }
        const auto& marker        = markers[marker_index];
        const int   span_start    = i - static_cast<int>(marker.begin_ids.size()) + 1;
        const int   content_start = i + 1;

        // Phase 2: walk the content looking for this marker's matching end.
        int  j      = content_start;
        bool closed = false;
        for (; j < n; ++j) {
            const int end_start = j - static_cast<int>(marker.end_ids.size()) + 1;
            if (end_start >= content_start && endsWith(tokens, j, marker.end_ids)) {
                Span span;
                span.marker_index = marker_index;
                span.token_ids.assign(tokens.begin() + span_start, tokens.begin() + j + 1);
                spans.push_back(std::move(span));
                closed = true;
                break;
            }
            if (j - span_start + 1 > max_span_tokens) {
                Span overflow;
                overflow.marker_index = marker_index;
                overflow.overflow     = true;
                spans.push_back(std::move(overflow));
                closed = true;
                break;
            }
        }
        // Reached end of input without closing: not a completed call — drop it.
        if (!closed) {
            break;
        }
        // Resume past the span we just consumed; spans don't overlap.
        i = j + 1;
    }
    return spans;
}

bool sameSpan(const Span& lhs, const Span& rhs) {
    return !lhs.overflow && !rhs.overflow && lhs.marker_index == rhs.marker_index && lhs.token_ids == rhs.token_ids;
}

}  // namespace

ToolCallLoopCheckResult checkToolCallLoop(const std::vector<int>&              input_ids,
                                          const std::vector<int>&              output_ids,
                                          const std::vector<std::vector<int>>& marker_begin_ids,
                                          const std::vector<std::vector<int>>& marker_end_ids,
                                          int                                  repeat_threshold,
                                          int                                  max_span_tokens) {
    const int  threshold = std::max(2, repeat_threshold);
    const int  span_cap  = max_span_tokens > 0 ? max_span_tokens : 16384;
    const auto markers   = buildMarkers(marker_begin_ids, marker_end_ids);

    const auto output_spans = scanSpans(output_ids, markers, span_cap);
    if (output_spans.empty()) {
        return {};
    }
    const auto history_spans = scanSpans(input_ids, markers, span_cap);

    // Count the longest run of consecutive identical spans across the combined
    // [history, output] timeline, then report the strongest run that ends on an
    // output span — the model must still be repeating now, not only in history.
    // An overflow span resets the run.
    ToolCallLoopCheckResult best;
    int                     run  = 0;
    const Span*             prev = nullptr;

    auto advance = [&](const Span& span) {
        if (span.overflow) {
            run  = 0;
            prev = nullptr;
        } else {
            run  = (prev != nullptr && sameSpan(*prev, span)) ? run + 1 : 1;
            prev = &span;
        }
    };

    for (const auto& span : history_spans) {
        advance(span);
    }
    for (const auto& span : output_spans) {
        advance(span);
        const int  repeat_count = span.overflow ? 0 : run;
        const bool hit          = !span.overflow && repeat_count >= threshold;
        if (std::make_pair(hit, repeat_count) > std::make_pair(best.hit, best.repeat_count)) {
            best.hit                 = hit;
            best.repeat_count        = repeat_count;
            best.current_span_tokens = static_cast<int>(span.token_ids.size());
            best.marker_index        = span.marker_index;
        }
    }
    return best;
}

}  // namespace rtp_llm
