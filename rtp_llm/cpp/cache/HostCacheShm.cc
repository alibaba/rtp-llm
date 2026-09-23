#include "rtp_llm/cpp/cache/HostCacheShm.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fcntl.h>
#include <limits>
#include <linux/magic.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <system_error>
#include <unistd.h>

namespace rtp_llm {
namespace {

bool enabled(const char* key) {
    const char* raw = std::getenv(key);
    if (raw == nullptr) {
        return false;
    }
    std::string value(raw);
    const auto  begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return false;
    }
    value = value.substr(begin, value.find_last_not_of(" \t\r\n") - begin + 1);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

[[noreturn]] void fail(const char* operation, const std::string& name) {
    throw std::system_error(errno, std::generic_category(), std::string(operation) + " " + name);
}

}  // namespace

bool useScrHostCacheShm() {
    return enabled("RTPLLM_ENABLE_SCR") && enabled("RTP_LLM_SCR_HOST_CACHE_SHM");
}

HostCacheShm::HostCacheShm(size_t size_bytes): size_bytes_(size_bytes) {
    if (size_bytes == 0 || size_bytes > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
        throw std::invalid_argument("host cache shm size must be positive and fit in off_t");
    }
    static std::atomic<unsigned long long> sequence{0};
    const auto                             epoch = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto prefix = "/rtpllm-scr-host-kv-" + std::to_string(geteuid()) + "-" + std::to_string(getpid()) + "-"
                        + std::to_string(epoch) + "-";
    int fd = -1;
    for (int attempt = 0; attempt < 128; ++attempt) {
        name_ = prefix + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
        fd    = shm_open(name_.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd >= 0) {
            break;
        }
        if (errno != EEXIST) {
            fail("shm_open", name_);
        }
    }
    if (fd < 0) {
        fail("shm_open: unique name exhausted", name_);
    }
    try {
        struct statfs fs{};
        if (fstatfs(fd, &fs) != 0) {
            fail("fstatfs", name_);
        }
        if (fs.f_type != TMPFS_MAGIC) {
            throw std::runtime_error("SCR host cache requires /dev/shm to be tmpfs");
        }
        struct statvfs space{};
        if (fstatvfs(fd, &space) != 0) {
            fail("fstatvfs", name_);
        }
        // This is only a fail-fast check, not a reservation against other ranks.
        // Do not fallocate here: it would populate pages before the NUMA policy.
        if (space.f_frsize == 0 || size_bytes / space.f_frsize + (size_bytes % space.f_frsize != 0) > space.f_bavail) {
            throw std::runtime_error("not enough /dev/shm space for SCR host cache: " + std::to_string(size_bytes));
        }
        if (ftruncate(fd, static_cast<off_t>(size_bytes)) != 0) {
            fail("ftruncate", name_);
        }
        data_ = mmap(nullptr, size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            fail("shared mmap", name_);
        }
    } catch (...) {
        (void)close(fd);
        (void)shm_unlink(name_.c_str());
        throw;
    }
    // The mapping and linked name keep the object alive; no extra FD needs C/R.
    (void)close(fd);
}

HostCacheShm::~HostCacheShm() {
    if (data_ != nullptr) {
        (void)munmap(data_, size_bytes_);
        (void)shm_unlink(name_.c_str());
    }
}

}  // namespace rtp_llm
