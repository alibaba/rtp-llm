#include "rtp_llm/cpp/distribute/CpuBroadcaster.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace rtp_llm {

namespace {

constexpr int      kInitTimeoutMs             = 120 * 1000;
constexpr int      kDefaultBroadcastTimeoutMs = 0;  // Match NCCL: idle TP workers may wait indefinitely.
constexpr int      kBroadcastStallWarnMs      = 30 * 1000;
constexpr char     kLinkProbeToken            = 0x5a;
constexpr uint64_t kBroadcastFrameMagic       = 0x5254504c4c4d5450ULL;

struct BroadcastFrameHeader {
    uint64_t magic;
    uint64_t nbytes;
};

int broadcastTimeoutMs() {
    const char* value = std::getenv("RTP_LLM_CPU_TP_BROADCASTER_BROADCAST_TIMEOUT_MS");
    if (value == nullptr || *value == '\0') {
        return kDefaultBroadcastTimeoutMs;
    }
    char* end    = nullptr;
    long  parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > 24L * 60 * 60 * 1000) {
        return kDefaultBroadcastTimeoutMs;
    }
    return static_cast<int>(parsed);
}

std::string makeUdsPath(const std::string& base, int rank) {
    return base + "_" + std::to_string(rank) + ".sock";
}

void closeFd(int& fd) {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

void shutdownAndCloseFd(int& fd) {
    if (fd >= 0) {
        ::shutdown(fd, SHUT_RDWR);
        closeFd(fd);
    }
}

void cleanupRank0State(std::vector<int>& peer_fds, int& listen_fd, std::string& uds_path) {
    for (int& fd : peer_fds) {
        closeFd(fd);
    }
    closeFd(listen_fd);
    if (!uds_path.empty()) {
        ::unlink(uds_path.c_str());
        uds_path.clear();
    }
}

bool isRetryableConnectError(int err) {
    return err == EINTR || err == ENOENT || err == ECONNREFUSED;
}

bool waitFdUntil(int fd, short events, std::chrono::steady_clock::time_point deadline) {
    const bool no_timeout = deadline == std::chrono::steady_clock::time_point::max();
    const auto start      = std::chrono::steady_clock::now();
    while (true) {
        int remaining_ms = kBroadcastStallWarnMs;
        if (!no_timeout) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                errno = ETIMEDOUT;
                return false;
            }
            remaining_ms =
                static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
        }

        struct pollfd pfd {};
        pfd.fd     = fd;
        pfd.events = events;
        int rc     = ::poll(&pfd, 1, remaining_ms);
        if (rc > 0) {
            if (pfd.revents & (POLLERR | POLLNVAL)) {
                errno = EIO;
                return false;
            }
            if (pfd.revents & events) {
                return true;
            }
            if ((events & POLLIN) && (pfd.revents & POLLHUP)) {
                return true;
            }
            if (pfd.revents & POLLHUP) {
                errno = EPIPE;
                return false;
            }
            continue;
        }
        if (rc == 0) {
            if (no_timeout) {
                const auto waited_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
                        .count();
                RTP_LLM_LOG_WARNING("CpuBroadcaster broadcast I/O still waiting after %lld ms "
                                    "(fd=%d events=0x%x timeout disabled)",
                                    static_cast<long long>(waited_ms),
                                    fd,
                                    static_cast<unsigned int>(events));
                continue;
            }
            errno = ETIMEDOUT;
            return false;
        }
        if (errno == EINTR) {
            continue;
        }
        return false;
    }
}

bool waitReadableUntil(int fd, std::chrono::steady_clock::time_point deadline) {
    return waitFdUntil(fd, POLLIN, deadline);
}

bool waitWritableUntil(int fd, std::chrono::steady_clock::time_point deadline) {
    return waitFdUntil(fd, POLLOUT, deadline);
}

int acceptWithTimeout(int listen_fd, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (waitReadableUntil(listen_fd, deadline)) {
        int fd = ::accept(listen_fd, nullptr, nullptr);
        if (fd >= 0) {
            return fd;
        }
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
            continue;
        }
        return -1;
    }
    return -1;
}

ssize_t sendNoSignal(int fd, const void* buf, std::size_t nbytes, int extra_flags = 0) {
#ifdef MSG_NOSIGNAL
    return ::send(fd, buf, nbytes, MSG_NOSIGNAL | extra_flags);
#else
    return ::send(fd, buf, nbytes, extra_flags);
#endif
}

// Loop until `nbytes` written or fatal error. Returns -1 on error.
ssize_t writeAll(int fd, const void* buf, std::size_t nbytes) {
    const char* p    = static_cast<const char*>(buf);
    std::size_t left = nbytes;
    while (left > 0) {
        ssize_t n = sendNoSignal(fd, p, left);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (n == 0) {
            return -1;
        }
        p += n;
        left -= n;
    }
    return static_cast<ssize_t>(nbytes);
}

ssize_t writeAllWithTimeout(int fd, const void* buf, std::size_t nbytes, int timeout_ms) {
    const char* p        = static_cast<const char*>(buf);
    std::size_t left     = nbytes;
    const auto  deadline = timeout_ms <= 0 ? std::chrono::steady_clock::time_point::max() :
                                             std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (left > 0) {
        if (!waitWritableUntil(fd, deadline)) {
            return -1;
        }
        ssize_t n = sendNoSignal(fd, p, left, MSG_DONTWAIT);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            return -1;
        }
        if (n == 0) {
            errno = EPIPE;
            return -1;
        }
        p += n;
        left -= n;
    }
    return static_cast<ssize_t>(nbytes);
}

ssize_t readAllWithTimeout(int fd, void* buf, std::size_t nbytes, int timeout_ms) {
    char*       p        = static_cast<char*>(buf);
    std::size_t left     = nbytes;
    const auto  deadline = timeout_ms <= 0 ? std::chrono::steady_clock::time_point::max() :
                                             std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (left > 0) {
        if (!waitReadableUntil(fd, deadline)) {
            return -1;
        }
        ssize_t n = ::recv(fd, p, left, MSG_DONTWAIT);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            return -1;
        }
        if (n == 0) {
            errno = ECONNRESET;
            return -1;
        }
        p += n;
        left -= n;
    }
    return static_cast<ssize_t>(nbytes);
}

}  // namespace

CpuBroadcaster& CpuBroadcaster::instance() {
    static CpuBroadcaster i;
    return i;
}

CpuBroadcaster::~CpuBroadcaster() {
    std::lock_guard<std::mutex> lock(mu_);
    cleanupStateLocked();
}

void CpuBroadcaster::cleanupStateLocked() {
    for (int& fd : peer_fds_) {
        closeFd(fd);
    }
    peer_fds_.clear();
    closeFd(listen_fd_);
    if (!my_uds_path_.empty()) {
        ::unlink(my_uds_path_.c_str());
        my_uds_path_.clear();
    }
    base_path_.clear();
    rank_                  = 0;
    world_size_            = 1;
    broadcast_timeout_ms_  = kDefaultBroadcastTimeoutMs;
    broadcast_in_progress_ = false;
    failed_                = false;
    initialized_.store(false, std::memory_order_release);
}

void CpuBroadcaster::markBroadcastFailedLocked() {
    failed_                = true;
    broadcast_in_progress_ = false;
    for (int& fd : peer_fds_) {
        shutdownAndCloseFd(fd);
    }
    closeFd(listen_fd_);
    if (!my_uds_path_.empty()) {
        ::unlink(my_uds_path_.c_str());
        my_uds_path_.clear();
    }
}

void CpuBroadcaster::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    if (initialized_.load(std::memory_order_acquire)) {
        RTP_LLM_CHECK_WITH_INFO(!broadcast_in_progress_,
                                "CpuBroadcaster::reset called while broadcastCPU is in progress; "
                                "concurrent broadcastCPU/reset is unsupported");
    }
    cleanupStateLocked();
}

void CpuBroadcaster::initialize(int rank, int world_size, const std::string& base_path) {
    std::lock_guard<std::mutex> lock(mu_);

    if (initialized_.load(std::memory_order_acquire)) {
        if (rank_ == rank && world_size_ == world_size && base_path_ == base_path) {
            return;
        }
        RTP_LLM_FAIL("CpuBroadcaster re-init mismatch: was rank=%d size=%d path=%s, "
                     "now rank=%d size=%d path=%s",
                     rank_,
                     world_size_,
                     base_path_.c_str(),
                     rank,
                     world_size,
                     base_path.c_str());
    }

    broadcast_in_progress_ = false;
    failed_                = false;
    if (world_size <= 1) {
        // Single-rank no-op; broadcast() short-circuits.
        rank_       = rank;
        world_size_ = world_size;
        base_path_  = base_path;
        initialized_.store(true, std::memory_order_release);
        return;
    }

    rank_       = rank;
    world_size_ = world_size;
    base_path_  = base_path;
    peer_fds_.assign(world_size, -1);
    broadcast_timeout_ms_ = broadcastTimeoutMs();

    if (rank == 0) {
        try {
            const std::string path = makeUdsPath(base_path, 0);
            // Remove any stale socket left by a previous crashed run.
            ::unlink(path.c_str());

            listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
            RTP_LLM_CHECK_WITH_INFO(listen_fd_ >= 0, "CpuBroadcaster socket: %s", std::strerror(errno));

            struct sockaddr_un addr {};
            addr.sun_family = AF_UNIX;
            RTP_LLM_CHECK_WITH_INFO(
                path.size() < sizeof(addr.sun_path), "CpuBroadcaster UDS path too long: %s", path.c_str());
            std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

            int rc = ::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
            RTP_LLM_CHECK_WITH_INFO(rc == 0, "CpuBroadcaster bind(%s): %s", path.c_str(), std::strerror(errno));
            my_uds_path_ = path;

            rc = ::listen(listen_fd_, world_size - 1);
            RTP_LLM_CHECK_WITH_INFO(rc == 0, "CpuBroadcaster listen: %s", std::strerror(errno));

            // Accept world_size-1 peers during bootstrap only. Request-time
            // broadcast I/O uses a configurable timeout.
            for (int i = 1; i < world_size; ++i) {
                int fd    = acceptWithTimeout(listen_fd_, kInitTimeoutMs);
                int saved = errno;
                RTP_LLM_CHECK_WITH_INFO(fd >= 0,
                                        "CpuBroadcaster accept timed out after %d ms on %s while waiting for "
                                        "peer %d/%d: %s",
                                        kInitTimeoutMs,
                                        path.c_str(),
                                        i,
                                        world_size - 1,
                                        std::strerror(saved));

                bool close_accepted = true;
                try {
                    int     peer_rank = -1;
                    ssize_t n         = readAllWithTimeout(fd, &peer_rank, sizeof(peer_rank), kInitTimeoutMs);
                    saved             = errno;
                    RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(peer_rank)),
                                            "CpuBroadcaster handshake read failed after %d ms: %s",
                                            kInitTimeoutMs,
                                            std::strerror(saved));
                    RTP_LLM_CHECK_WITH_INFO(peer_rank > 0 && peer_rank < world_size,
                                            "CpuBroadcaster bad peer_rank: %d (world_size=%d)",
                                            peer_rank,
                                            world_size);
                    RTP_LLM_CHECK_WITH_INFO(
                        peer_fds_[peer_rank] < 0, "CpuBroadcaster duplicate peer rank: %d", peer_rank);
                    peer_fds_[peer_rank] = fd;
                    close_accepted       = false;
                } catch (...) {
                    if (close_accepted) {
                        closeFd(fd);
                    }
                    throw;
                }
            }
            for (int peer_rank = 1; peer_rank < world_size; ++peer_rank) {
                ssize_t n = writeAllWithTimeout(
                    peer_fds_[peer_rank], &kLinkProbeToken, sizeof(kLinkProbeToken), kInitTimeoutMs);
                int saved = errno;
                RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(kLinkProbeToken)),
                                        "CpuBroadcaster link probe write to rank %d failed after %d ms: %s",
                                        peer_rank,
                                        kInitTimeoutMs,
                                        std::strerror(saved));
            }
            RTP_LLM_LOG_INFO("CpuBroadcaster rank 0: accepted %d peer(s) on %s", world_size - 1, path.c_str());
        } catch (...) {
            cleanupRank0State(peer_fds_, listen_fd_, my_uds_path_);
            throw;
        }
    } else {
        const std::string  path = makeUdsPath(base_path, 0);
        struct sockaddr_un addr {};
        addr.sun_family = AF_UNIX;
        RTP_LLM_CHECK_WITH_INFO(
            path.size() < sizeof(addr.sun_path), "CpuBroadcaster UDS path too long: %s", path.c_str());
        std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

        // Retry connect only during bootstrap to tolerate small rank scheduling
        // skew; request-time broadcast I/O uses a configurable timeout.
        constexpr int kSleepMs     = 50;
        constexpr int kMaxAttempts = kInitTimeoutMs / kSleepMs;
        int           fd           = -1;
        for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
            fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
            RTP_LLM_CHECK_WITH_INFO(fd >= 0, "CpuBroadcaster socket: %s", std::strerror(errno));
            int rc = ::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
            if (rc == 0) {
                break;
            }
            int saved = errno;
            ::close(fd);
            fd = -1;
            if (!isRetryableConnectError(saved)) {
                RTP_LLM_FAIL("CpuBroadcaster connect(%s) failed: %s", path.c_str(), std::strerror(saved));
            }
            if (attempt + 1 == kMaxAttempts) {
                RTP_LLM_FAIL("CpuBroadcaster connect(%s) failed after %d attempts: %s",
                             path.c_str(),
                             kMaxAttempts,
                             std::strerror(saved));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kSleepMs));
        }

        try {
            // Send our rank so the server can index peer_fds_ correctly.
            int     my_rank = rank;
            ssize_t n       = writeAll(fd, &my_rank, sizeof(my_rank));
            RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(my_rank)),
                                    "CpuBroadcaster handshake write failed");

            char probe = 0;
            n          = readAllWithTimeout(fd, &probe, sizeof(probe), kInitTimeoutMs);
            int saved  = errno;
            RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(probe)) && probe == kLinkProbeToken,
                                    "CpuBroadcaster link probe read failed after %d ms: %s",
                                    kInitTimeoutMs,
                                    std::strerror(saved));
        } catch (...) {
            closeFd(fd);
            throw;
        }

        peer_fds_[0] = fd;
        RTP_LLM_LOG_INFO("CpuBroadcaster rank %d: connected to rank 0 at %s", rank, path.c_str());
    }

    initialized_.store(true, std::memory_order_release);
}

void CpuBroadcaster::broadcast(void* buf, std::size_t nbytes, int root) {
    int              rank       = 0;
    int              world_size = 1;
    int              timeout_ms = 0;
    std::vector<int> peer_fds;

    {
        std::lock_guard<std::mutex> lock(mu_);
        RTP_LLM_CHECK_WITH_INFO(initialized_.load(std::memory_order_acquire),
                                "CpuBroadcaster::broadcast called before initialize");
        if (world_size_ <= 1 || nbytes == 0) {
            return;
        }
        RTP_LLM_CHECK_WITH_INFO(!failed_,
                                "CpuBroadcaster::broadcast called after a failed broadcast; "
                                "reset and reinitialize before reuse");
        RTP_LLM_CHECK_WITH_INFO(root == 0, "CpuBroadcaster supports only root=0 (star topology); got %d", root);
        RTP_LLM_CHECK_WITH_INFO(!broadcast_in_progress_,
                                "CpuBroadcaster::broadcast does not support concurrent or re-entrant "
                                "broadcastCPU calls");
        RTP_LLM_CHECK_WITH_INFO(static_cast<int>(peer_fds_.size()) == world_size_,
                                "CpuBroadcaster invalid peer fd state: size=%zu world_size=%d",
                                peer_fds_.size(),
                                world_size_);
        broadcast_in_progress_ = true;
        rank                   = rank_;
        world_size             = world_size_;
        timeout_ms             = broadcast_timeout_ms_;
        peer_fds               = peer_fds_;
    }

    auto finish_broadcast = [this](bool failed) {
        std::lock_guard<std::mutex> lock(mu_);
        failed_                = failed_ || failed;
        broadcast_in_progress_ = false;
    };
    auto fail_broadcast = [this]() {
        std::lock_guard<std::mutex> lock(mu_);
        markBroadcastFailedLocked();
    };

    try {
        if (rank == 0) {
            const BroadcastFrameHeader header{kBroadcastFrameMagic, static_cast<uint64_t>(nbytes)};
            for (int k = 1; k < world_size; ++k) {
                ssize_t n = writeAllWithTimeout(peer_fds[k], &header, sizeof(header), timeout_ms);
                RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(header)),
                                        "CpuBroadcaster frame header write to rank %d failed after %d ms: %s",
                                        k,
                                        timeout_ms,
                                        std::strerror(errno));
                n = writeAllWithTimeout(peer_fds[k], buf, nbytes, timeout_ms);
                RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(nbytes),
                                        "CpuBroadcaster write to rank %d (%zu bytes) failed after %d ms: %s",
                                        k,
                                        nbytes,
                                        timeout_ms,
                                        std::strerror(errno));
            }
        } else {
            BroadcastFrameHeader header{};
            ssize_t              n = readAllWithTimeout(peer_fds[0], &header, sizeof(header), timeout_ms);
            RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(header)),
                                    "CpuBroadcaster frame header read from rank 0 failed after %d ms: %s",
                                    timeout_ms,
                                    std::strerror(errno));
            RTP_LLM_CHECK_WITH_INFO(header.magic == kBroadcastFrameMagic && header.nbytes == nbytes,
                                    "CpuBroadcaster frame header mismatch: magic=%llu nbytes=%llu "
                                    "expected_magic=%llu expected_nbytes=%zu",
                                    static_cast<unsigned long long>(header.magic),
                                    static_cast<unsigned long long>(header.nbytes),
                                    static_cast<unsigned long long>(kBroadcastFrameMagic),
                                    nbytes);
            n = readAllWithTimeout(peer_fds[0], buf, nbytes, timeout_ms);
            RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(nbytes),
                                    "CpuBroadcaster read from rank 0 (%zu bytes) failed after %d ms: %s",
                                    nbytes,
                                    timeout_ms,
                                    std::strerror(errno));
        }
    } catch (...) {
        fail_broadcast();
        throw;
    }
    finish_broadcast(false);
}

}  // namespace rtp_llm
