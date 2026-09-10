#include "rtp_llm/cpp/model_rpc/SleepRpcUtils.h"

namespace rtp_llm::sleep_rpc {

grpc::Status resultToGrpcStatus(const SleepResult& result) {
    if (result.ok) {
        return grpc::Status::OK;
    }
    switch (result.code) {
        case SleepResult::Code::DISABLED:
        case SleepResult::Code::UNIMPLEMENTED:
            return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, result.message);
        case SleepResult::Code::INVALID_ARGUMENT:
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, result.message);
        case SleepResult::Code::FAILED_PRECONDITION:
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, result.message);
        case SleepResult::Code::OK:
            return grpc::Status::OK;
    }
    return grpc::Status(grpc::StatusCode::UNKNOWN, result.message);
}

void fillStatusProto(const SleepStatus& status, SleepStatusResponsePB* response) {
    response->set_state(sleepStateToString(status.state));
    response->set_sleep_epoch(status.sleep_epoch);
    response->set_kv_memory_state(status.kv_memory_state);
    response->set_device_kv_cache_valid(status.device_kv_cache_valid);
    response->set_active_request_count(status.active_request_count);
    response->set_active_cache_transfer_count(status.active_cache_transfer_count);
    response->set_gpu_resource_state(status.gpu_resource_state);
    response->set_last_error(status.last_error);
    response->set_sleep_mode_enabled(status.sleep_mode_enabled);
    response->set_effective(status.effective);
    response->set_disabled_reason(status.disabled_reason);
    for (const auto level : status.supported_levels) {
        response->add_supported_levels(level);
    }
    for (const auto& mode : status.supported_modes) {
        response->add_supported_modes(mode);
    }
}

}  // namespace rtp_llm::sleep_rpc
