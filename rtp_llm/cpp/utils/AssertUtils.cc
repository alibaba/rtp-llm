#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

[[noreturn]] void throwRuntimeError(const char* const file, int const line, std::string const& info) {
    auto error_msg = std::string("[ERROR] ") + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n";
    fflush(stdout);
    fflush(stderr);
    throw RTP_EXCEPTION(error_msg);
}

[[noreturn]] void myAssert(const char* const file, int const line, std::string const& info) {
    auto error_msg = std::string("[ERROR] ") + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n";
    RTP_LLM_LOG_ERROR("FATAIL ERROR!!! %s", error_msg.c_str());
    throwRuntimeError(file, line, info);
}

}  // namespace rtp_llm
