#pragma once

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm::sleep_rpc {

grpc::Status resultToGrpcStatus(const SleepResult& result);
void         fillStatusProto(const SleepStatus& status, SleepStatusResponsePB* response);

}  // namespace rtp_llm::sleep_rpc
