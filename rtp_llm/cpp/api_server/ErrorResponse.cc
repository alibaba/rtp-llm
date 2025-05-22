#include "rtp_llm/cpp/api_server/ErrorResponse.h"

namespace rtp_llm {

void ErrorResponse::Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) {
    json.Jsonize("error_code", error_code, error_code);
    json.Jsonize("message", error_msg, error_msg);
}

std::string ErrorResponse::CreateErrorResponseJsonString(int error_code, const std::string& error_msg) {
    ErrorResponse response;
    response.error_code = error_code;
    response.error_msg  = error_msg;
    return ToJsonString(response, /*isCompact=*/true);
}

}  // namespace rtp_llm