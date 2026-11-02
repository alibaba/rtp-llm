#pragma once

#include "grpcpp/support/status.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

ErrorInfo parseMultimodalGrpcError(const grpc::Status& status);

}  // namespace rtp_llm
