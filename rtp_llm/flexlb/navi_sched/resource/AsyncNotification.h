#pragma once

#include <cerrno>
#include <cstdint>
#include <mutex>
#include <system_error>
#include <sys/eventfd.h>
#include <unistd.h>

namespace navi_sched {

// The fd only signals readiness; data stays in the existing native queues.
class AsyncNotification {
public:
    ~AsyncNotification()
    {
        if (_fd >= 0) {
            close(_fd);
        }
    }

    int fd()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_fd < 0) {
            _fd = eventfd(1, EFD_NONBLOCK | EFD_CLOEXEC);
            if (_fd < 0) {
                throw std::system_error(errno, std::generic_category(), "eventfd");
            }
        }
        return _fd;
    }

    void notify()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_fd >= 0) {
            const uint64_t value = 1;
            ssize_t result;
            do {
                result = write(_fd, &value, sizeof(value));
            } while (result < 0 && errno == EINTR);
            // EAGAIN means the fd is already readable, so no wakeup is lost.
        }
    }

private:
    std::mutex _mutex;
    int _fd = -1;
};

}
