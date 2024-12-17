#include "absl/debugging/symbolize.h"
#include "absl/debugging/stacktrace.h"

#include "maga_transformer/cpp/utils/StackTrace.h"
#include "maga_transformer/cpp/utils/Logger.h"

#include <execinfo.h>
#include <unistd.h>
#include <sstream>


namespace fastertransformer {

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
    FT_STACKTRACE_LOG_INFO("%s", getStackTrace().c_str());
    fflush(stdout);
    fflush(stderr);
}

}
