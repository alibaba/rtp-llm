#include "rtp_llm/cpp/multimodal_processor/transport/grpc/MMGrpcError.h"

#include <limits>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"

namespace rtp_llm {

ErrorInfo parseMultimodalGrpcError(const grpc::Status& status) {
    ErrorDetailsPB details;
    if (!status.error_details().empty() && details.ParseFromString(status.error_details()) && details.error_code() > 0
        && details.error_code() <= std::numeric_limits<int>::max()) {
        // The worker owns business error classification. Preserve numeric codes,
        // including Python-only/new codes, instead of imposing a ViT whitelist.
        return ErrorInfo(static_cast<ErrorCode>(details.error_code()),
                         details.error_message().empty() ? status.error_message() : details.error_message());
    }

    // Older workers may only send a named error prefix. Keep the existing
    // transport fallback when structured details are absent or malformed.
    if (auto error = parseMultimodalErrorMessage(status.error_message())) {
        return *error;
    }
    switch (status.error_code()) {
        case grpc::StatusCode::CANCELLED:
            return ErrorInfo(ErrorCode::CANCELLED, status.error_message());
        case grpc::StatusCode::RESOURCE_EXHAUSTED:
            return ErrorInfo(ErrorCode::CONCURRENCY_LIMIT_ERROR, status.error_message());
        case grpc::StatusCode::UNAVAILABLE:
        case grpc::StatusCode::DEADLINE_EXCEEDED:
            return ErrorInfo(ErrorCode::MM_REMOTE_RPC_FAILED, status.error_message());
        default:
            return ErrorInfo(ErrorCode::UNKNOWN_ERROR, status.error_message());
    }
}

}  // namespace rtp_llm
