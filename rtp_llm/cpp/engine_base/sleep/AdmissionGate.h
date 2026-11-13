#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "grpc++/grpc++.h"

#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"

namespace rtp_llm {

// Structured admission result serialized as ErrorDetailsPB on RPC rejection.
struct AdmissionCheckResult {
    bool        admitted   = true;
    int64_t     error_code = 0;  // ErrorCode::ENGINE_UNAVAILABLE (8600) when denied
    std::string error_code_str;  // "ENGINE_UNAVAILABLE"
    std::string message;
    std::string instance_id;
    int64_t     sleep_epoch = 0;
    std::string state;  // RUNNING|DRAINING|SUSPENDING|SLEEPING|WAKING_UP|ERROR
};

struct AdmissionAcquireResult {
    AdmissionCheckResult  detail;
    std::function<void()> complete;
};

// Protocol adapter for scheduler-owned admission. Inference entries notify
// complete() from their final cleanup; no lease leaves the scheduler.
// health/status paths may use check() or checkDetail(). Any state other than
// RUNNING is rejected with a retryable ENGINE_UNAVAILABLE carrying
// instance_id / sleep_epoch / state.
class AdmissionGate {
public:
    // Optional sleep status view is not owned and must outlive this adapter.
    explicit AdmissionGate(SleepLifecycleController* controller, std::string instance_id = ""):
        admission_(controller ? controller->admission() : nullptr),
        controller_(controller),
        instance_id_(std::move(instance_id)) {}
    // Linearizable admission. A successful result contains an idempotent
    // completion notification for the final cleanup owner.
    AdmissionAcquireResult acquire() const;
    // Only for internal KV continuations of admitted work, never new roots.
    AdmissionAcquireResult acquireCacheTransfer() const;

    // Open scheduler admission -> OK. Otherwise UNAVAILABLE with the
    // error body serialized into grpc error_details as ErrorDetailsPB.
    grpc::Status check() const;

    // Structured variant for RPC error handling.
    AdmissionCheckResult checkDetail() const;

    static grpc::Status toGrpcStatus(const AdmissionCheckResult& result);

private:
    AdmissionAcquireResult              acquireImpl(bool continuation) const;
    std::shared_ptr<SchedulerAdmission> admission_;
    SleepLifecycleController* controller_;  // not owned
    std::string               instance_id_;
};

}  // namespace rtp_llm
