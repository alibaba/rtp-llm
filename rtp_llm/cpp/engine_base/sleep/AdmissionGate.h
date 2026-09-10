#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "grpc++/grpc++.h"

#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"

namespace rtp_llm {

// Structured admission result. When denied, all fields are populated so RPC
// callers can serialize ErrorDetailsPB and HTTP callers can build a JSON body
// with the same schema:
//   {error_code, error_code_str, message, instance_id, sleep_epoch, state}
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
    AdmissionCheckResult detail;
    AdmissionLease       lease;
};

// Unified admission gate. Inference entries use acquire() and retain its lease;
// health/status paths may use check() or checkDetail(). Any state other than
// RUNNING is rejected with a retryable ENGINE_UNAVAILABLE carrying
// instance_id / sleep_epoch / state.
class AdmissionGate {
public:
    // controller is not owned and must outlive the gate (it lives in
    // EngineBase, which is never destructed).
    explicit AdmissionGate(SleepLifecycleController* controller, std::string instance_id = ""):
        controller_(controller), instance_id_(std::move(instance_id)) {}

    // Linearizable admission check. A successful result carries a move-only
    // lease that must remain alive for the full inference request.
    AdmissionAcquireResult acquire() const;

    // RUNNING (or no controller wired) -> OK. Otherwise UNAVAILABLE with the
    // error body serialized into grpc error_details as ErrorDetailsPB.
    grpc::Status check() const;

    // Structured variant for the HTTP layer (and tests).
    AdmissionCheckResult checkDetail() const;

    // JSON body for HTTP responses, same schema as the gRPC error details.
    static std::string  toJson(const AdmissionCheckResult& result);
    static grpc::Status toGrpcStatus(const AdmissionCheckResult& result);

private:
    SleepLifecycleController* controller_;  // not owned
    std::string               instance_id_;
};

}  // namespace rtp_llm
