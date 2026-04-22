#include <csignal>
#include <cstdio>
#include <cerrno>
#include <ctime>
#include <cstring>
#include <unistd.h>

#include <execinfo.h>

#include "rtp_llm/cpp/utils/SignalUtils.h"

namespace rtp_llm {

namespace {

constexpr int kStderrFd  = 2;
constexpr int kMaxFrames = 64;

void writeAll(int fd, const char* data, size_t len) {
    while (len > 0) {
        ssize_t n = write(fd, data, len);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        data += static_cast<size_t>(n);
        len -= static_cast<size_t>(n);
    }
}

// fdatasync/fsync 非 POSIX 异步信号安全函数；此处仍调用，仅为尽量把崩溃输出刷向存储端。
// 失败（如对 tty/pipe 的 EINVAL）直接忽略。
void syncFdDataAndFullBestEffort(int fd) {
    if (fd < 0) {
        return;
    }
    int r;
    do {
        r = fdatasync(fd);
    } while (r < 0 && errno == EINTR);
    do {
        r = fsync(fd);
    } while (r < 0 && errno == EINTR);
}

}  // namespace

void printSignalStackTrace(int signum, siginfo_t* siginfo, void* /*ucontext*/) {
    char        line[768];
    int         len          = 0;
    time_t      current_time = time(nullptr);
    const void* fault_addr   = (siginfo != nullptr) ? siginfo->si_addr : nullptr;

    len = snprintf(line, sizeof(line), "\n*** Aborted at %ld (unix time) ***\n", static_cast<long>(current_time));
    if (len > 0 && static_cast<size_t>(len) < sizeof(line)) {
        writeAll(kStderrFd, line, static_cast<size_t>(len));
    }

    switch (signum) {
        case SIGSEGV:
            len = snprintf(line,
                           sizeof(line),
                           "*** SIGSEGV (%p) received by PID %d (TID %d); stack trace: ***\n",
                           fault_addr,
                           static_cast<int>(getpid()),
                           static_cast<int>(gettid()));
            break;
        case SIGFPE:
            len = snprintf(line,
                           sizeof(line),
                           "*** SIGFPE (%p) received by PID %d (TID %d); stack trace: ***\n",
                           fault_addr,
                           static_cast<int>(getpid()),
                           static_cast<int>(gettid()));
            break;
        case SIGILL:
            len = snprintf(line,
                           sizeof(line),
                           "*** SIGILL (%p) received by PID %d (TID %d); stack trace: ***\n",
                           fault_addr,
                           static_cast<int>(getpid()),
                           static_cast<int>(gettid()));
            break;
        case SIGABRT:
            len = snprintf(line,
                           sizeof(line),
                           "*** SIGABRT (%p) received by PID %d (TID %d); stack trace: ***\n",
                           fault_addr,
                           static_cast<int>(getpid()),
                           static_cast<int>(gettid()));
            break;
        case SIGBUS:
            len = snprintf(line,
                           sizeof(line),
                           "*** SIGBUS (%p) received by PID %d (TID %d); stack trace: ***\n",
                           fault_addr,
                           static_cast<int>(getpid()),
                           static_cast<int>(gettid()));
            break;
        default:
            len = snprintf(line,
                           sizeof(line),
                           "*** Unknown signal (%d) received by PID %d (TID %d); stack trace: ***\n",
                           signum,
                           static_cast<int>(getpid()),
                           static_cast<int>(gettid()));
            break;
    }
    if (len > 0 && static_cast<size_t>(len) < sizeof(line)) {
        writeAll(kStderrFd, line, static_cast<size_t>(len));
    }

    void* frames[kMaxFrames];
    int   depth = backtrace(frames, kMaxFrames);
    if (depth > 0) {
        backtrace_symbols_fd(frames, depth, kStderrFd);
    }
    writeAll(kStderrFd, "\n", 1);

    // 同上：非信号安全，仅作尽力刷盘。
    syncFdDataAndFullBestEffort(kStderrFd);
}

void getSighandler(int signum, siginfo_t* siginfo, void* ucontext) {
    printSignalStackTrace(signum, siginfo, ucontext);
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
