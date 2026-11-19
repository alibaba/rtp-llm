#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
#include "rtp_llm/cpp/cuda_graph/decode_execution.h"

namespace rtp_llm {

enum class DecodePreparationStatus : int64_t {
    READY,
    GRAPH_UNAVAILABLE,
    INVALID,
};

struct AgreedDecodePlan {
    DecodeExecution execution;
    bool            valid_local_input;
};

// The engine supplies these callbacks for models that must agree on execution
// before entering model collectives.
class AgreedDecodeExecution {
public:
    virtual ~AgreedDecodeExecution() = default;

    virtual void verifyModelInitialization(bool local_ok)                                                        = 0;
    virtual void verifyCapture(DecodeExecutionPhase phase, int token_width, const std::vector<int>& buckets)     = 0;
    virtual void finishCapture(DecodeExecutionPhase phase)                                                       = 0;
    virtual void verifyCaptureReadiness(DecodeExecutionPhase phase, int key, bool before_capture, bool local_ok) = 0;
    virtual AgreedDecodePlan
                 plan(DecodeExecutionPhase phase, int token_width, int64_t local_requests, bool allow_graph)  = 0;
    virtual void rejectSpeculativeInput(DecodeExecutionPhase phase)                                           = 0;
    virtual DecodeExecution finishPreparation(const DecodeExecution& planned, DecodePreparationStatus status) = 0;
    virtual bool            agreeSpeculativeStatus(bool local_ok)                                             = 0;
};

inline DecodeExecutionPhase firstSpeculativeDecodePhase(bool dspark, int64_t proposal_width) {
    if (dspark) {
        return DecodeExecutionPhase::DSPARK_PROPOSAL;
    }
    return proposal_width > 1 ? DecodeExecutionPhase::MTP_DRAFT : DecodeExecutionPhase::TARGET_VERIFY;
}

inline void reportInvalidSpeculativeInput(AgreedDecodeExecution* agreement, DecodeExecutionPhase phase) noexcept {
    if (agreement != nullptr) {
        try {
            agreement->rejectSpeculativeInput(phase);
        } catch (...) {
            // The rejection exchange also fails on this rank. Keep the
            // original local error or status for the caller.
        }
    }
}

template<typename Prepare>
decltype(auto)
prepareSpeculativeInput(AgreedDecodeExecution* agreement, DecodeExecutionPhase phase, Prepare&& prepare) {
    // Only wrap local preparation. Once forward enters PLAN, its own error
    // handling owns the protocol and a second rejection would desynchronize it.
    try {
        return std::forward<Prepare>(prepare)();
    } catch (...) {
        reportInvalidSpeculativeInput(agreement, phase);
        throw;
    }
}

inline DecodeExecution finishPendingDecodePreparation(AgreedDecodeExecution&          agreement,
                                                      std::optional<DecodeExecution>& pending,
                                                      DecodePreparationStatus         status) {
    // An exchange failure must not cause a second vote in the error handler.
    const auto planned = std::exchange(pending, std::nullopt);
    return agreement.finishPreparation(planned.value(), status);
}

template<typename PrepareGraph, typename DiscardGraph, typename PrepareEager>
DecodeExecution prepareAgreedDecodeExecution(AgreedDecodeExecution&          agreement,
                                             std::optional<DecodeExecution>& pending,
                                             PrepareGraph                    prepare_graph,
                                             DiscardGraph                    discard_graph,
                                             PrepareEager                    prepare_eager) {
    if (pending.value().mode == DecodeExecutionMode::GRAPH) {
        const auto status = prepare_graph(pending.value()) ? DecodePreparationStatus::READY :
                                                             DecodePreparationStatus::GRAPH_UNAVAILABLE;
        const auto agreed = finishPendingDecodePreparation(agreement, pending, status);
        if (agreed.mode == DecodeExecutionMode::GRAPH) {
            return agreed;
        }
        pending = agreed;
        discard_graph();
    }
    prepare_eager(pending.value());
    return finishPendingDecodePreparation(agreement, pending, DecodePreparationStatus::READY);
}

}  // namespace rtp_llm
