#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <optional>
#include <vector>

namespace rtp_llm {

enum class DecodeExecutionMode : uint8_t {
    EAGER,
    FIXED,
    GRAPH,
};

enum class DecodeExecutionPhase : uint8_t {
    TARGET,
    TARGET_VERIFY,
    MTP_DRAFT,
    MTP_COMMIT,
    DSPARK_PROPOSAL,
    DSPARK_COMMIT,
};

struct DecodeExecution {
    DecodeExecutionMode  mode;
    DecodeExecutionPhase phase;
    int                  bucket;
    int                  token_width;
};

// The prepare callback returns the staged graph key only after device work
// completes, or zero if the runner cannot prepare this invocation.
class AgreedDecodeGraphPreparation {
public:
    template<typename Prepare>
    bool prepare(Prepare prepare_graph) {
        discard();
        const int key = prepare_graph();
        key_.store(key, std::memory_order_release);
        return key > 0;
    }

    void discard() {
        key_.store(0, std::memory_order_release);
    }

    bool consume(const std::optional<DecodeExecutionPhase>& phase,
                 const std::optional<DecodeExecution>&      execution,
                 int                                        token_width,
                 int                                        bucket,
                 int                                        graph_key) {
        const int prepared_key = key_.exchange(0, std::memory_order_acq_rel);
        return phase && execution && execution->mode == DecodeExecutionMode::GRAPH && execution->phase == *phase
               && execution->token_width == token_width && execution->bucket == bucket && prepared_key > 0
               && prepared_key == graph_key;
    }

private:
    std::atomic<int> key_{0};
};

// A decode runner owns one phase and token width. Its graph instances are
// individually keyed by bucket. Legacy runners infer a key only when the
// caller did not supply an execution decision.
inline std::optional<int> selectDecodeGraphBucket(const std::vector<int>&               buckets,
                                                  std::optional<DecodeExecutionPhase>   phase,
                                                  int                                   token_width,
                                                  int64_t                               local_requests,
                                                  const std::optional<DecodeExecution>& execution) {
    if (phase.has_value() != execution.has_value()) {
        return std::nullopt;
    }
    if (execution) {
        if (execution->mode != DecodeExecutionMode::GRAPH || execution->phase != *phase
            || execution->token_width != token_width || local_requests <= 0 || execution->bucket < local_requests
            || !std::binary_search(buckets.begin(), buckets.end(), execution->bucket)) {
            return std::nullopt;
        }
        return execution->bucket;
    }
    const auto it = std::lower_bound(buckets.begin(), buckets.end(), local_requests);
    return it != buckets.end() ? std::optional<int>(*it) : std::nullopt;
}

}  // namespace rtp_llm
