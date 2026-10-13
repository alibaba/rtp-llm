#pragma once

#include <limits>
#include <string>

#include "grpc++/support/status.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorMessage.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

// Preserve the caller's stage error code; this is diagnostic context only.
// Put machine-readable cause/state before potentially long free-form messages
// so an outer status truncation cannot discard the useful downstream cause.
inline std::string formatGrpcErrorStatus(const grpc::Status& status) {
    std::string    result = "grpc_code=" + std::to_string(static_cast<int>(status.error_code()));
    ErrorDetailsPB details;
    if (!status.error_details().empty() && details.ParseFromString(status.error_details())
        && details.error_code() != 0) {
        auto symbol = details.error_code_str();
        if (symbol.empty()) {
            symbol = details.error_code() >= std::numeric_limits<int>::min()
                             && details.error_code() <= std::numeric_limits<int>::max() ?
                         ErrorCodeToString(static_cast<ErrorCode>(details.error_code())) :
                         "UNRECOGNIZED";
        }
        result +=
            ", business_code=" + std::to_string(details.error_code()) + ", symbol=" + safeRpcErrorMessage(symbol, 64);
        if (!details.state().empty() || details.sleep_epoch() != 0) {
            result += ", state=" + safeRpcErrorMessage(details.state(), 32)
                      + ", sleep_epoch=" + std::to_string(details.sleep_epoch());
        }
        result += ", business_message=" + safeRpcErrorMessage(details.error_message(), 256);
        if (status.error_message() != details.error_message()) {
            result += ", grpc_message=" + safeRpcErrorMessage(status.error_message(), 256);
        }
    } else {
        // Empty/unknown-only protobufs and arbitrary bytes have no business
        // code. Keep them diagnosable without pretending they are ErrorDetailsPB.
        result += ", grpc_details_hex=" + rpcErrorDetailsHex(status.error_details())
                  + ", grpc_message=" + safeRpcErrorMessage(status.error_message(), 256);
    }
    return result;
}

}  // namespace rtp_llm
