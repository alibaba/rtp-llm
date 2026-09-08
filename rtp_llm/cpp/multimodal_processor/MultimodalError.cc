#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"

#include <array>
#include <string_view>

namespace rtp_llm {
namespace {

constexpr std::array<ErrorCode, 7> kMultimodalErrorCodes = {{
    ErrorCode::MM_LONG_PROMPT_ERROR,
    ErrorCode::MM_WRONG_FORMAT_ERROR,
    ErrorCode::MM_PROCESS_ERROR,
    ErrorCode::MM_EMPTY_ENGINE_ERROR,
    ErrorCode::MM_NOT_SUPPORTED_ERROR,
    ErrorCode::MM_DOWNLOAD_FAILED,
    ErrorCode::MM_REMOTE_RPC_FAILED,
}};

}  // namespace

std::optional<ErrorCode> parseMultimodalErrorCode(int error_code) {
    for (const auto code : kMultimodalErrorCodes) {
        if (static_cast<int>(code) == error_code) {
            return code;
        }
    }
    return std::nullopt;
}

std::optional<ErrorInfo> parseMultimodalErrorMessage(const std::string& error_message) {
    if (error_message.empty() || error_message.front() != '[') {
        return std::nullopt;
    }

    const auto closing_bracket = error_message.find(']');
    if (closing_bracket == std::string::npos) {
        return std::nullopt;
    }

    const std::string_view error_name(error_message.data() + 1, closing_bracket - 1);
    for (const auto code : kMultimodalErrorCodes) {
        if (error_name == ErrorCodeToString(code)) {
            auto message_begin = closing_bracket + 1;
            if (message_begin < error_message.size() && error_message[message_begin] == ' ') {
                ++message_begin;
            }
            return ErrorInfo(code, error_message.substr(message_begin));
        }
    }
    return std::nullopt;
}

bool isRetryableMultimodalError(ErrorCode error_code) {
    switch (error_code) {
        case ErrorCode::MM_PROCESS_ERROR:
        case ErrorCode::MM_EMPTY_ENGINE_ERROR:
        case ErrorCode::MM_DOWNLOAD_FAILED:
        case ErrorCode::MM_REMOTE_RPC_FAILED:
            return true;
        default:
            return false;
    }
}

}  // namespace rtp_llm
