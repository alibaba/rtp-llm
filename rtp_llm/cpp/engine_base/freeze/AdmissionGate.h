#pragma once

#include <cstdint>
#include <string>

#include "grpc++/grpc++.h"

#include "rtp_llm/cpp/engine_base/freeze/FreezeLifecycleController.h"

namespace rtp_llm {

// Structured admission result (design doc M4 error body). When denied, all
// fields are populated so RPC callers can serialize ErrorDetailsPB and HTTP
// callers can build a JSON body with the same schema:
//   {error_code, error_code_str, message, instance_id, freeze_epoch, state}
struct AdmissionCheckResult {
    bool        admitted   = true;
    int64_t     error_code = 0;  // ErrorCode::ENGINE_UNAVAILABLE (8600) when denied
    std::string error_code_str;  // "ENGINE_UNAVAILABLE"
    std::string message;
    std::string instance_id;
    int64_t     freeze_epoch = 0;
    std::string state;  // RUNNING|DRAINING|FREEZING|FROZEN|RESUMING|ERROR
};

// Unified admission gate (design doc M4, constraint C5). Single check() called
// at every inference entry (gRPC + HTTP + embedding); any state other than
// RUNNING is rejected with a retryable ENGINE_UNAVAILABLE error carrying
// instance_id / freeze_epoch / state.
class AdmissionGate {
public:
    // controller is not owned and must outlive the gate (it lives in
    // EngineBase, which is never destructed per constraint C1).
    explicit AdmissionGate(const FreezeLifecycleController* controller, std::string instance_id = ""):
        controller_(controller), instance_id_(std::move(instance_id)) {}

    // RUNNING (or no controller wired) -> OK. Otherwise UNAVAILABLE with the
    // M4 error body serialized into grpc error_details as ErrorDetailsPB.
    grpc::Status check() const;

    // Structured variant for the HTTP layer (and tests).
    AdmissionCheckResult checkDetail() const;

    // JSON body for HTTP responses, same schema as the gRPC error details.
    static std::string toJson(const AdmissionCheckResult& result);

    const std::string& instanceId() const {
        return instance_id_;
    }

private:
    const FreezeLifecycleController* controller_;  // not owned
    std::string                      instance_id_;
};

}  // namespace rtp_llm
