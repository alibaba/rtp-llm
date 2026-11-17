#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
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
    int64_t max_execution_tokens = 0;
    int64_t max_decode_sequences = 0;
    // Natural (CP-restored) Prefill rows of the independently allocated draft.
    int64_t max_prefill_draft_tokens = 0;
    bool    target_only_runs_draft   = false;
};

struct ExecutionCost {
    int64_t prefill_tokens       = 0;
    int64_t prefill_draft_tokens = 0;
    int64_t decode_sequences     = 0;
};

struct ExecutionCapacityResult {
    bool prefill_fits = true;
    bool draft_fits   = true;
    bool decode_fits  = true;

    bool fits() const {
        return prefill_fits && draft_fits && decode_fits;
    }
};

// Account in physical target rows, natural draft rows and Decode sequences.
// Callers retain their own stream selection and overflow handling policies.
// execution_tp_size must already be normalized to a positive value.
class ExecutionCapacityLedger {
public:
    ExecutionCapacityLedger(ExecutionCost capacity, int64_t max_execution_tokens, int64_t execution_tp_size):
        capacity_(capacity) {
        if (max_execution_tokens > 0) {
            const auto aligned = max_execution_tokens / execution_tp_size * execution_tp_size;
            capacity_.prefill_tokens =
                capacity_.prefill_tokens > 0 ? std::min(capacity_.prefill_tokens, aligned) : aligned;
        }
        remaining_ = capacity_;
    }

    bool enabled() const {
        return capacity_.prefill_tokens > 0 || capacity_.prefill_draft_tokens > 0 || capacity_.decode_sequences > 0;
    }

    const ExecutionCost& capacity() const {
        return capacity_;
    }

    ExecutionCapacityResult consume(const ExecutionCost& cost) {
        return {consumeDimension(cost.prefill_tokens, capacity_.prefill_tokens, remaining_.prefill_tokens),
                consumeDimension(
                    cost.prefill_draft_tokens, capacity_.prefill_draft_tokens, remaining_.prefill_draft_tokens),
                consumeDimension(cost.decode_sequences, capacity_.decode_sequences, remaining_.decode_sequences)};
    }

private:
    static bool consumeDimension(int64_t cost, int64_t capacity, int64_t& remaining) {
        if (capacity <= 0) {
            return true;
        }
        const bool fits = cost <= remaining;
        remaining       = fits ? remaining - cost : 0;
        return fits;
    }

    ExecutionCost capacity_;
    ExecutionCost remaining_;
};

// Convert a model's physical rows into a count of complete, SP-padded requests.
// Zero is an absent contract, not a request-count tuning limit.
inline int64_t decodeSequenceCapacity(int64_t tokens, int64_t tp, const std::vector<int64_t>& query_widths) {
    if (tokens == 0) {
        return 0;
    }
    if (tokens < 0 || tp <= 0) {
        throw std::invalid_argument("execution capacity and TP geometry must be positive");
    }
    int64_t capacity = std::numeric_limits<int64_t>::max();
    for (const auto width : query_widths) {
        if (width <= 0) {
            throw std::invalid_argument("execution query width must be positive");
        }
        const int64_t alignment = tp / std::gcd(tp, width);
        // Divide before multiplying, including for extreme configured widths.
        const int64_t requests = tokens / width / alignment * alignment;
        if (requests == 0) {
            throw std::invalid_argument("initialized execution workspace cannot hold one padded request");
        }
        capacity = std::min(capacity, requests);
    }
    return capacity;
}

inline bool reservesExecution(const GenerateStreamPtr& stream) {
    return !stream->isFakeStream() && !stream->hasError() && !stream->isFinished()
           && !stream->hasEvent(StreamEvents::GenerateDone);
}

inline int64_t decodeSequenceReservation(const GenerateStreamPtr& stream, RoleType role) {
    // Configuration-defined maximum also covers later beam expansion and avoids
    // consulting worker-owned output-token history during async Decode.
    const bool offloaded_prefill   = role == RoleType::PREFILL && stream->queryPdSep() && stream->isContextStream();
    const bool finishes_in_prefill = stream->isContextStream() && stream->generateConfig()->max_new_tokens <= 1;
    return reservesExecution(stream) && !finishes_in_prefill && !offloaded_prefill ? stream->maxBatchSize() : 0;
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
