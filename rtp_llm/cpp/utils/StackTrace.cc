#include "absl/debugging/symbolize.h"
#include "absl/debugging/stacktrace.h"

#include "rtp_llm/cpp/utils/StackTrace.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <execinfo.h>
#include <unistd.h>
#include <sstream>

namespace rtp_llm {

static constexpr int kMaxStackDepth = 64;

std::string getStackTrace() {
    std::stringstream stack_ss;
    void*             addrs[kMaxStackDepth];

    int stack_depth = backtrace(addrs, kMaxStackDepth);
    for (int i = 2; i < stack_depth; ++i) {
        char line[2048];
        char buf[1024];
        if (absl::Symbolize(addrs[i], buf, sizeof(buf))) {
            snprintf(line, 2048, "@  %16p  %s\n", addrs[i], buf);
        } else {
            snprintf(line, 2048, "@  %16p  (unknown)\n", addrs[i]);
        }
        stack_ss << std::string(line);
    }
    return stack_ss.str();
}

void printStackTrace() {
    sleep(10);
    RTP_LLM_STACKTRACE_LOG_INFO("%s", getStackTrace().c_str());
    fflush(stdout);
    fflush(stderr);
    sleep(10);
}

}  // namespace rtp_llm
