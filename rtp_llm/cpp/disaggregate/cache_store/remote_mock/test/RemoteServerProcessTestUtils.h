#pragma once
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <poll.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
namespace rtp_llm::remote_mock_test {
inline bool waitReady(int descriptor, std::chrono::steady_clock::time_point deadline, std::string& output) {
    pollfd fd{descriptor, POLLIN, 0};
    for (;;) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline)
            return false;
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        const int  result    = poll(&fd, 1, static_cast<int>(std::clamp<int64_t>(remaining, 1, 1000)));
        if (result == 0 || (result < 0 && errno == EINTR))
            continue;
        if (result < 0 || (fd.revents & (POLLERR | POLLNVAL)))
            return false;
        if (!(fd.revents & (POLLIN | POLLHUP)))
            continue;
        char       buffer[1024];
        const auto count = ::read(descriptor, buffer, sizeof(buffer));
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            return false;
        output.append(buffer, count);
        if (output.find("REMOTE_MOCK_READY") != std::string::npos)
            return true;
    }
}
inline pid_t waitChild(pid_t pid, int* status) {
    pid_t result;
    do {
        result = waitpid(pid, status, 0);
    } while (result < 0 && errno == EINTR);
    return result;
}
}  // namespace rtp_llm::remote_mock_test
