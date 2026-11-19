#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include "rtp_llm/cpp/models/AgreedDecodeExecution.h"

namespace rtp_llm {

struct EngineInitParams;
struct WarmUpResult;

enum class DecodeReplicaRound {
    IDLE,
    EXECUTE,
    PAUSED,
    STOP,
    INVALID,
};

enum class DecodeReplicaStartupStage {
    WARMUP_INPUTS,
    CACHE_MANAGER,
    ENGINE_READY,
};

struct DecodeReplicaRequest {
    bool fake;
    bool context;
    bool cache_ready;
    int  beams;
    int  sequences;
    bool force_disable_speculation;
};

// The constructor thread owns startup. The ordered engine thread owns round
// agreement and passes the model-phase interface to its executors.
class DecodeReplicaCoordinator: public AgreedDecodeExecution {
public:
    virtual const char*        admissionError(const DecodeReplicaRequest& request) const                           = 0;
    virtual DecodeReplicaRound beginRound(bool paused, bool stopping)                                              = 0;
    virtual DecodeReplicaRound agreeScheduledRequests(int64_t local_requests)                                      = 0;
    virtual void               rejectUnpreparedRound()                                                             = 0;
    virtual void               rejectUnpreparedTarget()                                                            = 0;
    virtual void     reserveWorkspace(const EngineInitParams& params, std::optional<WarmUpResult>& warm_up_result) = 0;
    virtual uint32_t agreeCacheCapacity(uint32_t local_blocks)                                                     = 0;
    virtual void     verifyStartup(DecodeReplicaStartupStage stage, bool local_ok)                                 = 0;
};

using DecodeReplicaCoordinatorFactory = std::unique_ptr<DecodeReplicaCoordinator> (*)(const EngineInitParams&,
                                                                                      std::string_view,
                                                                                      bool has_draft);

template<typename Notify>
void notifyDecodeReplicaFailure(Notify&& notify) noexcept {
    try {
        std::forward<Notify>(notify)();
    } catch (...) {
        // The rejection also fails on this rank. Preserve the local cause.
    }
}

template<typename Prepare>
void prepareDecodeReplicaStartup(DecodeReplicaCoordinator* coordinator,
                                 DecodeReplicaStartupStage stage,
                                 Prepare&&                 prepare) {
    try {
        std::forward<Prepare>(prepare)();
    } catch (...) {
        if (coordinator) {
            notifyDecodeReplicaFailure([&] { coordinator->verifyStartup(stage, false); });
        }
        throw;
    }
    if (coordinator) {
        coordinator->verifyStartup(stage, true);
    }
}

template<typename Prepare>
uint32_t prepareDecodeReplicaCacheCapacity(DecodeReplicaCoordinator* coordinator, Prepare&& prepare) {
    uint32_t local_blocks;
    try {
        local_blocks = std::forward<Prepare>(prepare)();
    } catch (...) {
        if (coordinator) {
            notifyDecodeReplicaFailure([&] { coordinator->agreeCacheCapacity(0); });
        }
        throw;
    }
    return coordinator ? coordinator->agreeCacheCapacity(local_blocks) : local_blocks;
}

template<typename Streams>
int64_t countScheduledDecodeRequests(Streams& streams) {
    int64_t requests = 0;
    for (const auto& stream : streams) {
        if (stream->isFakeStream()) {
            continue;
        }
        if (stream->isContextStream() || stream->nextBatchSize() != 1) {
            return -1;
        }
        ++requests;
    }
    if (requests > 0) {
        streams.remove_if([](const auto& stream) { return stream->isFakeStream(); });
    }
    return requests;
}

template<typename Schedule, typename AgreeCounts>
DecodeReplicaRound scheduleDecodeReplicaRound(DecodeReplicaRound control, Schedule schedule, AgreeCounts agree_counts) {
    if (control != DecodeReplicaRound::EXECUTE) {
        return control;
    }
    return agree_counts(schedule());
}

}  // namespace rtp_llm
