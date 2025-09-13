#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {

[[noreturn]] void throwRuntimeError(const char* const file, int const line, std::string const& info) {
    auto error_msg =
        std::string("[FT][ERROR] ") + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n";
    fflush(stdout);
    fflush(stderr);
    throw FT_EXCEPTION(error_msg);
}

[[noreturn]] void myAssert(const char* const file, int const line, std::string const& info) {
    auto error_msg =
        std::string("[FT][ERROR] ") + info + " Assertion fail: " + file + ":" + std::to_string(line) + " \n";
    RTP_LLM_LOG_ERROR("FATAIL ERROR!!! %s", error_msg.c_str());
    if (StaticConfig::user_ft_core_dump_on_exception) {
        fflush(stdout);
        fflush(stderr);
        abort();
    }
    throwRuntimeError(file, line, info);
}

}  // namespace rtp_llm
