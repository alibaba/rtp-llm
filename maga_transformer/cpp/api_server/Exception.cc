#include "maga_transformer/cpp/api_server/Exception.h"

#include "maga_transformer/cpp/api_server/ErrorResponse.h"

namespace rtp_llm {

std::string HttpApiServerException::formatException(const std::exception& e) {
    std::string res;
    if (const auto he = dynamic_cast<const HttpApiServerException*>(&e); he) {
        res = ErrorResponse::CreateErrorResponseJsonString(he->getType(), he->getMessage());
    } else {
        res = ErrorResponse::CreateErrorResponseJsonString(
                HttpApiServerException::UNKNOWN_ERROR,
                std::string("inference failed, exception occurred: ") + e.what());
    }
    return res;
}

}  // namespace rtp_llm
