#include "rtp_llm/cpp/engine_base/freeze/AdmissionGate.h"

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

namespace {

std::string jsonEscape(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (const char c : input) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += c;
        }
    }
    return out;
}

}  // namespace

AdmissionCheckResult AdmissionGate::checkDetail() const {
    AdmissionCheckResult result;
    result.instance_id = instance_id_;
    if (controller_ == nullptr) {
        result.state = freezeStateToString(FreezeState::RUNNING);
        return result;
    }
    const FreezeState state = controller_->state();
    result.freeze_epoch     = controller_->freezeEpoch();
    result.state            = freezeStateToString(state);
    if (state == FreezeState::RUNNING) {
        return result;
    }
    result.admitted       = false;
    result.error_code     = static_cast<int64_t>(ErrorCode::ENGINE_UNAVAILABLE);
    result.error_code_str = ErrorCodeToString(ErrorCode::ENGINE_UNAVAILABLE);
    result.message        = "engine unavailable: instance [" + instance_id_ + "] is " + result.state
                     + " (freeze_epoch=" + std::to_string(result.freeze_epoch) + "), request can be retried elsewhere";
    return result;
}

grpc::Status AdmissionGate::check() const {
    const auto result = checkDetail();
    if (result.admitted) {
        return grpc::Status::OK;
    }
    ErrorDetailsPB details;
    details.set_error_code(result.error_code);
    details.set_error_message(result.message);
    details.set_error_code_str(result.error_code_str);
    details.set_instance_id(result.instance_id);
    details.set_freeze_epoch(result.freeze_epoch);
    details.set_state(result.state);
    std::string serialized;
    if (details.SerializeToString(&serialized)) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, result.message, serialized);
    }
    return grpc::Status(grpc::StatusCode::UNAVAILABLE, result.message);
}

std::string AdmissionGate::toJson(const AdmissionCheckResult& result) {
    return "{\"error_code\":" + std::to_string(result.error_code) + ",\"error_code_str\":\""
           + jsonEscape(result.error_code_str) + "\",\"message\":\"" + jsonEscape(result.message)
           + "\",\"instance_id\":\"" + jsonEscape(result.instance_id) + "\",\"freeze_epoch\":"
           + std::to_string(result.freeze_epoch) + ",\"state\":\"" + jsonEscape(result.state) + "\"}";
}

}  // namespace rtp_llm
