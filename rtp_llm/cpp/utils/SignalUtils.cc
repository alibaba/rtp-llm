#include <csignal>
#include <sstream>
#include <unistd.h>
#include <iostream>
#include <execinfo.h>
#include <dlfcn.h>
#include <sstream>

#include "rtp_llm/cpp/utils/SignalUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/StackTrace.h"

namespace rtp_llm {

void printSignalStackTrace(int signum, siginfo_t* siginfo, void* ucontext) {
    std::stringstream stack_ss;
    time_t            current_time = time(nullptr);
    stack_ss << std::endl
             << "*** Aborted at " << current_time << " (unix time) try \"date -d @" << current_time
             << "\" if you are using GNU date***" << std::endl;

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
    RTP_LLM_STACKTRACE_LOG_INFO("%s", stack_ss.str().c_str());

    rtp_llm::printStackTrace();
}

void flushLog() {
    Logger::getEngineLogger().flush();
    Logger::getStackTraceLogger().flush();
    Logger::getAccessLogger().flush();
}

void getSighandler(int signum, siginfo_t* siginfo, void* ucontext) {
    printSignalStackTrace(signum, siginfo, ucontext);
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

    if (sigaction(SIGSEGV, &action, nullptr) != 0)
        return false;
    if (sigaction(SIGFPE, &action, nullptr) != 0)
        return false;
    if (sigaction(SIGILL, &action, nullptr) != 0)
        return false;
    if (sigaction(SIGABRT, &action, nullptr) != 0)
        return false;
    if (sigaction(SIGBUS, &action, nullptr) != 0)
        return false;

    return true;
}

};  // namespace rtp_llm