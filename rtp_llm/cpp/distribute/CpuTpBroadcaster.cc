#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"

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
constexpr int      kDefaultBroadcastTimeoutMs = 120 * 1000;
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
    if (end == value || *end != '\0' || parsed <= 0 || parsed > 24L * 60 * 60 * 1000) {
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
    while (true) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            errno = ETIMEDOUT;
            return false;
        }

        const auto remaining_ms =
            static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
        struct pollfd pfd{};
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
    const auto  deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
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
    const auto  deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
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

CpuTpBroadcaster& CpuTpBroadcaster::instance() {
    static CpuTpBroadcaster i;
    return i;
}

CpuTpBroadcaster::~CpuTpBroadcaster() {
    std::lock_guard<std::mutex> lock(mu_);
    cleanupStateLocked();
}

void CpuTpBroadcaster::cleanupStateLocked() {
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
    tp_rank_               = 0;
    tp_size_               = 1;
    broadcast_in_progress_ = false;
    failed_                = false;
    initialized_.store(false, std::memory_order_release);
}

void CpuTpBroadcaster::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    if (initialized_.load(std::memory_order_acquire)) {
        RTP_LLM_CHECK_WITH_INFO(!broadcast_in_progress_,
                                "CpuTpBroadcaster::reset called while broadcastCPU is in progress; "
                                "concurrent broadcastCPU/reset is unsupported");
    }
    cleanupStateLocked();
}

void CpuTpBroadcaster::initialize(int tp_rank, int tp_size, const std::string& base_path) {
    std::lock_guard<std::mutex> lock(mu_);

    if (initialized_.load(std::memory_order_acquire)) {
        if (tp_rank_ == tp_rank && tp_size_ == tp_size && base_path_ == base_path) {
            return;
        }
        RTP_LLM_FAIL("CpuTpBroadcaster re-init mismatch: was rank=%d size=%d path=%s, "
                     "now rank=%d size=%d path=%s",
                     tp_rank_,
                     tp_size_,
                     base_path_.c_str(),
                     tp_rank,
                     tp_size,
                     base_path.c_str());
    }

    broadcast_in_progress_ = false;
    failed_                = false;
    if (tp_size <= 1) {
        // Single-rank no-op; broadcast() short-circuits.
        tp_rank_   = tp_rank;
        tp_size_   = tp_size;
        base_path_ = base_path;
        initialized_.store(true, std::memory_order_release);
        return;
    }

    tp_rank_   = tp_rank;
    tp_size_   = tp_size;
    base_path_ = base_path;
    peer_fds_.assign(tp_size, -1);

    if (tp_rank == 0) {
        try {
            const std::string path = makeUdsPath(base_path, 0);
            // Remove any stale socket left by a previous crashed run.
            ::unlink(path.c_str());

            listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
            RTP_LLM_CHECK_WITH_INFO(listen_fd_ >= 0, "CpuTpBroadcaster socket: %s", std::strerror(errno));

            struct sockaddr_un addr{};
            addr.sun_family = AF_UNIX;
            RTP_LLM_CHECK_WITH_INFO(
                path.size() < sizeof(addr.sun_path), "CpuTpBroadcaster UDS path too long: %s", path.c_str());
            std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

            int rc = ::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
            RTP_LLM_CHECK_WITH_INFO(rc == 0, "CpuTpBroadcaster bind(%s): %s", path.c_str(), std::strerror(errno));
            my_uds_path_ = path;

            rc = ::listen(listen_fd_, tp_size - 1);
            RTP_LLM_CHECK_WITH_INFO(rc == 0, "CpuTpBroadcaster listen: %s", std::strerror(errno));

            // Accept tp_size-1 peers during bootstrap only. Request-time
            // broadcast I/O uses a configurable timeout.
            for (int i = 1; i < tp_size; ++i) {
                int fd    = acceptWithTimeout(listen_fd_, kInitTimeoutMs);
                int saved = errno;
                RTP_LLM_CHECK_WITH_INFO(fd >= 0,
                                        "CpuTpBroadcaster accept timed out after %d ms on %s while waiting for "
                                        "peer %d/%d: %s",
                                        kInitTimeoutMs,
                                        path.c_str(),
                                        i,
                                        tp_size - 1,
                                        std::strerror(saved));

                bool close_accepted = true;
                try {
                    int     peer_rank = -1;
                    ssize_t n         = readAllWithTimeout(fd, &peer_rank, sizeof(peer_rank), kInitTimeoutMs);
                    saved             = errno;
                    RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(peer_rank)),
                                            "CpuTpBroadcaster handshake read failed after %d ms: %s",
                                            kInitTimeoutMs,
                                            std::strerror(saved));
                    RTP_LLM_CHECK_WITH_INFO(peer_rank > 0 && peer_rank < tp_size,
                                            "CpuTpBroadcaster bad peer_rank: %d (tp_size=%d)",
                                            peer_rank,
                                            tp_size);
                    RTP_LLM_CHECK_WITH_INFO(
                        peer_fds_[peer_rank] < 0, "CpuTpBroadcaster duplicate peer rank: %d", peer_rank);
                    peer_fds_[peer_rank] = fd;
                    close_accepted       = false;
                } catch (...) {
                    if (close_accepted) {
                        closeFd(fd);
                    }
                    throw;
                }
            }
            for (int peer_rank = 1; peer_rank < tp_size; ++peer_rank) {
                ssize_t n = writeAllWithTimeout(
                    peer_fds_[peer_rank], &kLinkProbeToken, sizeof(kLinkProbeToken), kInitTimeoutMs);
                int saved = errno;
                RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(kLinkProbeToken)),
                                        "CpuTpBroadcaster link probe write to rank %d failed after %d ms: %s",
                                        peer_rank,
                                        kInitTimeoutMs,
                                        std::strerror(saved));
            }
            RTP_LLM_LOG_INFO("CpuTpBroadcaster rank 0: accepted %d peer(s) on %s", tp_size - 1, path.c_str());
        } catch (...) {
            cleanupRank0State(peer_fds_, listen_fd_, my_uds_path_);
            throw;
        }
    } else {
        const std::string  path = makeUdsPath(base_path, 0);
        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        RTP_LLM_CHECK_WITH_INFO(
            path.size() < sizeof(addr.sun_path), "CpuTpBroadcaster UDS path too long: %s", path.c_str());
        std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

        // Retry connect only during bootstrap to tolerate small rank scheduling
        // skew; request-time broadcast I/O uses a configurable timeout.
        constexpr int kSleepMs     = 50;
        constexpr int kMaxAttempts = kInitTimeoutMs / kSleepMs;
        int           fd           = -1;
        for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
            fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
            RTP_LLM_CHECK_WITH_INFO(fd >= 0, "CpuTpBroadcaster socket: %s", std::strerror(errno));
            int rc = ::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));
            if (rc == 0) {
                break;
            }
            int saved = errno;
            ::close(fd);
            fd = -1;
            if (!isRetryableConnectError(saved)) {
                RTP_LLM_FAIL("CpuTpBroadcaster connect(%s) failed: %s", path.c_str(), std::strerror(saved));
            }
            if (attempt + 1 == kMaxAttempts) {
                RTP_LLM_FAIL("CpuTpBroadcaster connect(%s) failed after %d attempts: %s",
                             path.c_str(),
                             kMaxAttempts,
                             std::strerror(saved));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kSleepMs));
        }

        try {
            // Send our rank so the server can index peer_fds_ correctly.
            int     my_rank = tp_rank;
            ssize_t n       = writeAll(fd, &my_rank, sizeof(my_rank));
            RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(my_rank)),
                                    "CpuTpBroadcaster handshake write failed");

            char probe = 0;
            n          = readAllWithTimeout(fd, &probe, sizeof(probe), kInitTimeoutMs);
            int saved  = errno;
            RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(probe)) && probe == kLinkProbeToken,
                                    "CpuTpBroadcaster link probe read failed after %d ms: %s",
                                    kInitTimeoutMs,
                                    std::strerror(saved));
        } catch (...) {
            closeFd(fd);
            throw;
        }

        peer_fds_[0] = fd;
        RTP_LLM_LOG_INFO("CpuTpBroadcaster rank %d: connected to rank 0 at %s", tp_rank, path.c_str());
    }

    initialized_.store(true, std::memory_order_release);
}

void CpuTpBroadcaster::broadcast(void* buf, std::size_t nbytes, int root) {
    int              tp_rank = 0;
    int              tp_size = 1;
    std::vector<int> peer_fds;

    {
        std::lock_guard<std::mutex> lock(mu_);
        RTP_LLM_CHECK_WITH_INFO(initialized_.load(std::memory_order_acquire),
                                "CpuTpBroadcaster::broadcast called before initialize");
        if (tp_size_ <= 1 || nbytes == 0) {
            return;
        }
        RTP_LLM_CHECK_WITH_INFO(!failed_,
                                "CpuTpBroadcaster::broadcast called after a failed broadcast; "
                                "reset and reinitialize before reuse");
        RTP_LLM_CHECK_WITH_INFO(root == 0, "CpuTpBroadcaster supports only root=0 (star topology); got %d", root);
        RTP_LLM_CHECK_WITH_INFO(!broadcast_in_progress_,
                                "CpuTpBroadcaster::broadcast does not support concurrent or re-entrant "
                                "broadcastCPU calls");
        RTP_LLM_CHECK_WITH_INFO(static_cast<int>(peer_fds_.size()) == tp_size_,
                                "CpuTpBroadcaster invalid peer fd state: size=%zu tp_size=%d",
                                peer_fds_.size(),
                                tp_size_);
        broadcast_in_progress_ = true;
        tp_rank                = tp_rank_;
        tp_size                = tp_size_;
        peer_fds               = peer_fds_;
    }

    auto finish_broadcast = [this](bool failed) {
        std::lock_guard<std::mutex> lock(mu_);
        failed_                = failed_ || failed;
        broadcast_in_progress_ = false;
    };

    try {
        const int timeout_ms = broadcastTimeoutMs();
        if (tp_rank == 0) {
            const BroadcastFrameHeader header{kBroadcastFrameMagic, static_cast<uint64_t>(nbytes)};
            for (int k = 1; k < tp_size; ++k) {
                ssize_t n = writeAllWithTimeout(peer_fds[k], &header, sizeof(header), timeout_ms);
                RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(header)),
                                        "CpuTpBroadcaster frame header write to rank %d failed after %d ms: %s",
                                        k,
                                        timeout_ms,
                                        std::strerror(errno));
                n = writeAllWithTimeout(peer_fds[k], buf, nbytes, timeout_ms);
                RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(nbytes),
                                        "CpuTpBroadcaster write to rank %d (%zu bytes) failed after %d ms: %s",
                                        k,
                                        nbytes,
                                        timeout_ms,
                                        std::strerror(errno));
            }
        } else {
            BroadcastFrameHeader header{};
            ssize_t              n = readAllWithTimeout(peer_fds[0], &header, sizeof(header), timeout_ms);
            RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(header)),
                                    "CpuTpBroadcaster frame header read from rank 0 failed after %d ms: %s",
                                    timeout_ms,
                                    std::strerror(errno));
            RTP_LLM_CHECK_WITH_INFO(header.magic == kBroadcastFrameMagic && header.nbytes == nbytes,
                                    "CpuTpBroadcaster frame header mismatch: magic=%llu nbytes=%llu "
                                    "expected_magic=%llu expected_nbytes=%zu",
                                    static_cast<unsigned long long>(header.magic),
                                    static_cast<unsigned long long>(header.nbytes),
                                    static_cast<unsigned long long>(kBroadcastFrameMagic),
                                    nbytes);
            n = readAllWithTimeout(peer_fds[0], buf, nbytes, timeout_ms);
            RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(nbytes),
                                    "CpuTpBroadcaster read from rank 0 (%zu bytes) failed after %d ms: %s",
                                    nbytes,
                                    timeout_ms,
                                    std::strerror(errno));
        }
    } catch (...) {
        finish_broadcast(true);
        throw;
    }
    finish_broadcast(false);
}

}  // namespace rtp_llm
