#pragma once

#include <vector>
#include <utility>
#include <algorithm>

#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

// Keep request-owned copy sources and save destinations alive from the first
// GPU use through the final local drain. A cancellation may call
// GenerateStream::releaseResource concurrently with executor forward.
class PrefillCacheWorkGuard {
public:
    PrefillCacheWorkGuard(const StreamGroups& groups, bool enabled) {
        if (!enabled) {
            return;
        }
        for (const auto& stream : groups.contextStreams()) {
            if (stream->streamCacheResource().prefixCacheRegistration().has_value()) {
                RTP_LLM_CHECK_WITH_INFO(stream->prefixLength() >= 0,
                                        "prefill cache work prefix length must be non-negative");
                RTP_LLM_CHECK_WITH_INFO(stream->currentBatchSize() == 1,
                                        "registered prefill cache work requires one sequence");
                const auto& registration = *stream->streamCacheResource().prefixCacheRegistration();
                // contextTokens(0, prefix, contextLength) is the original global
                // suffix before Python CP sharding; never count beam rows or
                // generated decode tokens as completed input endpoints.
                const size_t executed_end = std::min(registration.original_input_tokens,
                    static_cast<size_t>(stream->prefixLength()) + stream->currentExecuteTokens(0).size());
                stream->incPendingAsyncBookkeeping();
                streams_.push_back(stream);
                completion_endpoints_.emplace_back(stream, executed_end);
            }
        }
    }

    PrefillCacheWorkGuard(const PrefillCacheWorkGuard&) = delete;
    PrefillCacheWorkGuard& operator=(const PrefillCacheWorkGuard&) = delete;

    void markGpuWorkSubmitted() {
        gpu_work_submitted_ = true;
    }

    void markComplete() {
        gpu_work_submitted_ = false;
    }

    const std::vector<std::pair<GenerateStreamPtr, size_t>>& completionEndpoints() const {
        return completion_endpoints_;
    }

    ~PrefillCacheWorkGuard() {
        if (gpu_work_submitted_) {
            // Preserve the existing fatal CUDA exception path. Best-effort
            // local drain prevents early block reuse on ordinary unwinding.
            try {
                cudaSyncAndCheck();
            } catch (...) {
            }
        }
        for (const auto& stream : streams_) {
            stream->decPendingAsyncBookkeepingAndMaybeRelease();
        }
    }

private:
    std::vector<GenerateStreamPtr> streams_;
    std::vector<std::pair<GenerateStreamPtr, size_t>> completion_endpoints_;
    bool                           gpu_work_submitted_{false};
};

}  // namespace rtp_llm
