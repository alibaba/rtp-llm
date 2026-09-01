#include "rtp_llm/cpp/core/OpStatus.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/StackTrace.h"

namespace rtp_llm {

OpException::OpException(const OpStatus& status): status_(status) {
    std::stringstream ss;
    ss << "OpException[" << static_cast<int32_t>(status_.error_type) << "]: " << status_.error_message << std::endl;
    RTP_LLM_LOG_INFO("%s", ss.str().c_str());
    const auto stack = getStackTrace();
    RTP_LLM_STACKTRACE_LOG_INFO("%s", stack.c_str());
    ss << stack;
    detail_str_ = ss.str();
    if (StaticConfig::user_ft_core_dump_on_exception) {
        std::fflush(stdout);
        std::fflush(stderr);
        std::abort();
    }
}

}  // namespace rtp_llm
