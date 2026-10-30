#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagTokenLayout.h"

namespace rtp_llm {

inline constexpr char kMixedForceBatchGroupError[] =
    "force batch group must not mix max_new_tokens == 0 and max_new_tokens != 0 streams";
inline constexpr char kMixedExecutionModeBatchError[] =
    "model execution batch must not mix max_new_tokens == 0 and max_new_tokens != 0 streams";
inline constexpr char kDecodeRolePrefillOnlyError[] = "DECODE role does not support max_new_tokens == 0";

inline bool isPrefillOnly(const GenerateStreamPtr& stream) {
    return stream->generateConfig()->isPrefillOnly();
}

struct ModelExecutionCapacity {
    // Physical Prefill rows and query-CP layout of the independently allocated draft.
    int64_t max_prefill_draft_tokens = 0;
    size_t  draft_prefill_cp_size    = 1;
    bool    target_only_runs_draft   = false;
};

inline bool reservesExecution(const GenerateStreamPtr& stream) {
    return !stream->isFakeStream() && !stream->hasError() && !stream->isFinished()
           && !stream->hasEvent(StreamEvents::GenerateDone);
}

inline size_t paddedPrefillTokens(int length, size_t prefill_cp_size) {
    auto token_count = static_cast<size_t>(std::max(length, 0));
    if (prefill_cp_size > 1) {
        token_count = makeZigzagTokenLayout(token_count, prefill_cp_size).padded_token_count;
    }
    return token_count;
}

inline size_t prefillTokenCostWithoutCache(const GenerateStreamPtr& stream, size_t prefill_cp_size) {
    // Pad each expanded sequence independently, before multiplying by fanout.
    return paddedPrefillTokens(stream->contextLength(), prefill_cp_size)
           * static_cast<size_t>(stream->currentBatchSize());
}

// KV admission includes reused prefix tokens in logical sequence length.
inline size_t prefillSeqLenWithCache(const GenerateStreamPtr& stream) {
    const auto input_length  = static_cast<size_t>(std::max(stream->contextLength(), 0));
    const auto prefix_length = static_cast<size_t>(std::max(stream->prefixLength(), 0));
    return input_length + prefix_length;
}

inline size_t prefillTokenCostWithCache(const GenerateStreamPtr& stream) {
    return prefillSeqLenWithCache(stream) * static_cast<size_t>(stream->currentBatchSize());
}

inline int64_t prefillReservationTokens(const GenerateStreamPtr& stream, size_t cp_size) {
    if (!reservesExecution(stream) || !stream->isContextStream() || stream->chunkedPrefillEnabled()) {
        return 0;
    }
    // An unfinished allocator load may lose its tentative prefix on failure.
    // Keep a no-cache upper bound until RUNNING publishes finalized reuse.
    const auto length    = stream->getStatus() == StreamState::RUNNING ? stream->contextLength() : stream->seqLength();
    const auto tokens    = paddedPrefillTokens(length, cp_size);
    const auto sequences = static_cast<size_t>(std::max(stream->currentBatchSize(), 0));
    if (sequences && tokens > static_cast<size_t>(std::numeric_limits<int64_t>::max()) / sequences) {
        return std::numeric_limits<int64_t>::max();
    }
    return static_cast<int64_t>(tokens * sequences);
}

template<typename StreamRange>
bool hasMixedExecutionModes(const StreamRange& streams) {
    bool has_execution_mode = false;
    bool prefill_only       = false;
    for (const auto& stream : streams) {
        const bool stream_prefill_only = isPrefillOnly(stream);
        if (!has_execution_mode) {
            has_execution_mode = true;
            prefill_only       = stream_prefill_only;
        } else if (prefill_only != stream_prefill_only) {
            return true;
        }
    }
    return false;
}

}  // namespace rtp_llm
