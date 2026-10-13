#pragma once

#include <optional>
#include <string>

#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

std::optional<ErrorCode> parseMultimodalErrorCode(int error_code);
std::optional<ErrorInfo> parseMultimodalErrorMessage(const std::string& error_message);
bool                     isRetryableMultimodalError(ErrorCode error_code);

}  // namespace rtp_llm
