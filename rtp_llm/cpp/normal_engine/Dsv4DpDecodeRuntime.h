#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <vector>
#include "rtp_llm/cpp/normal_engine/Dsv4DpDecodePolicy.h"

namespace rtp_llm {

// Owned by NormalEngine. Only construction and the ordered engine thread
// exchange records; the attention and dispatch workers never call this object.
class Dsv4DpDecodeRuntime final: public DecodeReplicaCoordinator {
public:
    using WorkspaceReservation = void (*)(const EngineInitParams&, std::optional<WarmUpResult>&);
    explicit Dsv4DpDecodeRuntime(int world_size, WorkspaceReservation reserve_workspace = nullptr);
    ~Dsv4DpDecodeRuntime() override;

    const char*        admissionError(const DecodeReplicaRequest& request) const override;
    DecodeReplicaRound beginRound(bool paused, bool stopping) override;
    DecodeReplicaRound agreeScheduledRequests(int64_t local_requests) override;
    void               rejectUnpreparedRound() override;
    void               rejectUnpreparedTarget() override;
    void               rejectSpeculativeInput(DecodeExecutionPhase phase) override;
    bool               agreeSpeculativeStatus(bool local_ok) override;
    void     reserveWorkspace(const EngineInitParams& params, std::optional<WarmUpResult>& warm_up_result) override;
    uint32_t agreeCacheCapacity(uint32_t local_blocks) override;
    void     verifyStartup(DecodeReplicaStartupStage stage, bool local_ok) override;
    void     verifyModelInitialization(bool local_ok) override;
    void     verifyCapture(DecodeExecutionPhase phase, int token_width, const std::vector<int>& buckets) override;
    void     finishCapture(DecodeExecutionPhase phase) override;
    void     verifyCaptureReadiness(DecodeExecutionPhase phase, int key, bool before_capture, bool local_ok) override;
    AgreedDecodePlan
    plan(DecodeExecutionPhase phase, int token_width, int64_t local_requests, bool allow_graph) override;
    DecodeExecution finishPreparation(const DecodeExecution& planned, DecodePreparationStatus status) override;

private:
    std::vector<int64_t> exchange(Dsv4DecodeControl       control,
                                  DecodeExecutionPhase    phase,
                                  int64_t                 value,
                                  int                     bucket          = 0,
                                  int                     width           = 0,
                                  const std::vector<int>& capture_buckets = {});

    struct ControlGroup;
    struct CaptureCatalog {
        int              token_width;
        std::vector<int> buckets;
        uint64_t         fallbacks = 0;
    };
    struct ScheduledRequests {
        int64_t local;
        int64_t max;
    };

    std::unique_ptr<ControlGroup>                  control_group_;
    const int                                      world_size_;
    const WorkspaceReservation                     reserve_workspace_;
    int64_t                                        sequence_                 = 0;
    bool                                           target_preparation_voted_ = false;
    std::map<DecodeExecutionPhase, CaptureCatalog> capture_catalogs_;
    std::optional<ScheduledRequests>               scheduled_requests_;
};

}  // namespace rtp_llm
