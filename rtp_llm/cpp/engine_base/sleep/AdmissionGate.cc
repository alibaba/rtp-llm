#include "rtp_llm/cpp/engine_base/sleep/AdmissionGate.h"

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

namespace {

AdmissionCheckResult makeCheckResult(const std::string& instance_id, SleepState state, int64_t sleep_epoch) {
    AdmissionCheckResult result;
    result.instance_id = instance_id;
    result.sleep_epoch = sleep_epoch;
    result.state       = sleepStateToString(state);
    if (state == SleepState::RUNNING) {
        return result;
    }
    result.admitted       = false;
    result.error_code     = static_cast<int64_t>(ErrorCode::ENGINE_UNAVAILABLE);
    result.error_code_str = ErrorCodeToString(ErrorCode::ENGINE_UNAVAILABLE);
    result.message = "engine unavailable: " + result.state + " (sleep_epoch=" + std::to_string(result.sleep_epoch)
                     + "), request can be retried elsewhere";
    return result;
}

}  // namespace

AdmissionCheckResult AdmissionGate::checkDetail() const {
    auto state = controller_ ? controller_->state() : SleepState::RUNNING;
    if (!admission_) {
        state = SleepState::ERROR;
    } else if (!admission_->rootsOpen() && state == SleepState::RUNNING) {
        state = SleepState::DRAINING;
    }
    return makeCheckResult(instance_id_, state, controller_ ? controller_->sleepEpoch() : 0);
}

AdmissionAcquireResult AdmissionGate::acquire() const {
    return acquireImpl(false);
}

grpc::Status AdmissionGate::check() const {
    return toGrpcStatus(checkDetail());
}

AdmissionAcquireResult AdmissionGate::acquireCacheTransfer() const {
    return acquireImpl(true);
}

AdmissionAcquireResult AdmissionGate::acquireImpl(bool continuation) const {
    AdmissionAcquireResult result;
    if (!admission_) {
        result.detail = makeCheckResult(instance_id_, SleepState::ERROR, 0);
        return result;
    }
    auto acquired = admission_->admit(continuation);
    if (!acquired.accepted) {
        auto state = controller_ ? controller_->state() : SleepState::DRAINING;
        if (state == SleepState::RUNNING) {
            state = SleepState::DRAINING;
        }
        result.detail = makeCheckResult(instance_id_, state, controller_ ? controller_->sleepEpoch() : 0);
        if (continuation && state == SleepState::DRAINING) {
            // Unlike a new root, a continuation can enter DRAINING until the
            // freeze barrier closes its gate. Describe that distinct phase
            // without changing the retryable domain code or wire schema.
            result.detail.message =
                "engine unavailable: cache-transfer continuation admission is frozen in DRAINING (sleep_epoch="
                + std::to_string(result.detail.sleep_epoch)
                + "), retry the inference request after wake or on another engine";
        }
    }
    result.complete = std::move(acquired.complete);
    return result;
}

grpc::Status AdmissionGate::toGrpcStatus(const AdmissionCheckResult& result) {
    if (result.admitted) {
        return grpc::Status::OK;
    }
    ErrorDetailsPB details;
    details.set_error_code(result.error_code);
    details.set_error_message(result.message);
    details.set_error_code_str(result.error_code_str);
    details.set_instance_id(result.instance_id);
    details.set_sleep_epoch(result.sleep_epoch);
    details.set_state(result.state);
    std::string serialized;
    if (details.SerializeToString(&serialized)) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, result.message, serialized);
    }
    return grpc::Status(grpc::StatusCode::UNAVAILABLE, result.message);
}

}  // namespace rtp_llm
