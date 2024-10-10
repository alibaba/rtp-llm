#include <csignal>
#include <sstream>
#include <unistd.h>
#include <iostream>

#include "maga_transformer/cpp/utils/SignalUtils.h"
#include "src/fastertransformer/utils/logger.h"

namespace rtp_llm {

static constexpr int kMaxStackDepth = 64;

void printStackTrace(int signum, siginfo_t* siginfo, void* ucontext) {
    void*             addrs[kMaxStackDepth];
    int               stack_depth = absl::GetStackTrace(addrs, kMaxStackDepth, 2);
    std::stringstream stack_ss;
    time_t            current_time = time(nullptr);
    stack_ss << std::endl << "*** Aborted at " << current_time << " (unix time) try \"date -d @\"" << current_time
             << "if you are using GNU date***" << std::endl;

    switch (signum) {
        case SIGSEGV:
            stack_ss << "*** SIGSEGV (@0x" << std::hex << reinterpret_cast<uintptr_t>(siginfo->si_addr) << std::dec
                     << ") received by PID " << getpid() << " (TID " << gettid() << "); stack trace: ***" << std::endl;
            break;
        case SIGFPE:
            stack_ss << "*** SIGFPE (@0x" << std::hex << reinterpret_cast<uintptr_t>(siginfo->si_addr) << std::dec
                     << ") received by PID " << getpid() << " (TID " << gettid() << "); stack trace: ***" << std::endl;
            break;
        case SIGILL:
            stack_ss << "*** SIGILL (@0x" << std::hex << reinterpret_cast<uintptr_t>(siginfo->si_addr) << std::dec
                     << ") received by PID " << getpid() << " (TID " << gettid() << "); stack trace: ***" << std::endl;
            break;
        case SIGABRT:
            stack_ss << "*** SIGABRT (@0x" << std::hex << reinterpret_cast<uintptr_t>(siginfo->si_addr) << std::dec
                     << ") received by PID " << getpid() << " (TID " << gettid() << "); stack trace: ***" << std::endl;
            break;
        case SIGBUS:
            stack_ss << "*** SIGBUS (@0x" << std::hex << reinterpret_cast<uintptr_t>(siginfo->si_addr) << std::dec
                     << ") received by PID " << getpid() << " (TID " << gettid() << "); stack trace: ***" << std::endl;
            break;
        default:
            stack_ss << "*** Unknown signal (" << signum << ") received by PID " << getpid() << " (TID " << gettid()
                     << "); stack trace: ***" << std::endl;
            break;
    }

    for (int i = 0; i < stack_depth; ++i) {
        char line[2048];
        char buf[1024];
        if (absl::Symbolize(addrs[i], buf, sizeof(buf))) {
            snprintf(line, 2048, "@  %16p  %s\n", addrs[i], buf);
        } else {
            snprintf(line, 2048, "@  %16p  (unknown)\n", addrs[i]);
        }
        stack_ss << std::string(line);
    }
    FT_STACKTRACE_LOG_INFO("%s", stack_ss.str().c_str());
}

void flushLog() {
    fastertransformer::Logger::getEngineLogger().flush();
    fastertransformer::Logger::getStackTraceLogger().flush();
    fastertransformer::Logger::getAccessLogger().flush();
}

void getSighandler(int signum, siginfo_t* siginfo, void* ucontext) {
    printStackTrace(signum, siginfo, ucontext);
    flushLog();
    signal(signum, SIG_DFL);
    kill(getpid(), signum);
}

bool installSighandler() {
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_sigaction = getSighandler;
    action.sa_flags     = SA_SIGINFO;
    sigfillset(&action.sa_mask);

    if (sigaction(SIGSEGV, &action, nullptr) != 0) return false;
    if (sigaction(SIGFPE, &action, nullptr) != 0) return false;
    if (sigaction(SIGILL, &action, nullptr) != 0) return false;
    if (sigaction(SIGABRT, &action, nullptr) != 0) return false;
    if (sigaction(SIGBUS, &action, nullptr) != 0) return false;

    return true;
}

};  // namespace rtp_llm