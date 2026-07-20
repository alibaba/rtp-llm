#include "rtp_llm/cpp/distribute/CpuBroadcaster.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <errno.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

namespace rtp_llm {

struct CpuBroadcastSharedState {
    // Consecutive generations use alternating slots. A root cannot finish the
    // next generation without every peer, so two slots preserve the committed
    // decision for any peer that is still returning from the previous one.
    alignas(uint32_t) uint32_t decisions[2] = {0, 0};
};

static_assert(__atomic_always_lock_free(sizeof(uint32_t), nullptr),
              "CpuBroadcaster shared decision requires lock-free 32-bit atomics");

namespace {

constexpr int      kInitTimeoutMs             = 120 * 1000;
constexpr int      kDefaultBroadcastTimeoutMs = 0;  // Match NCCL: idle TP workers may wait indefinitely.
constexpr int      kBroadcastStallWarnMs      = 30 * 1000;
constexpr int      kRootHealthPollMs          = 100;
constexpr char     kLinkProbeToken            = 0x5a;
constexpr char     kSharedStateReadyToken     = 0x6a;
constexpr char     kBroadcastReadyToken       = 0x6b;
constexpr uint32_t kBroadcastFailedMask       = uint32_t{1} << 31;
constexpr uint64_t kBroadcastFrameMagic       = 0x5254504c4c4d5450ULL;

struct BroadcastFrameHeader {
    uint64_t magic;
    uint64_t nbytes;
};

int broadcastTimeoutMs() {
    const char* value = std::getenv("RTP_LLM_CPU_TP_BROADCASTER_BROADCAST_TIMEOUT_MS");
    if (value == nullptr) {
        return kDefaultBroadcastTimeoutMs;
    }
    errno        = 0;
    char* end    = nullptr;
    long  parsed = std::strtol(value, &end, 10);
    RTP_LLM_CHECK_WITH_INFO(errno != ERANGE && end != value && *end == '\0' && parsed >= 0
                                && parsed <= 24L * 60 * 60 * 1000,
                            "CpuBroadcaster invalid RTP_LLM_CPU_TP_BROADCASTER_BROADCAST_TIMEOUT_MS='%s': "
                            "expected an integer in [0, 86400000]",
                            value);
    return static_cast<int>(parsed);
}

std::string makeUdsPath(const std::string& base, int rank) {
    return base + "_" + std::to_string(rank) + ".sock";
}

std::string makeSharedStatePath(const std::string& base) {
    return base + ".state";
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

CpuBroadcastSharedState* mapSharedState(int fd) {
    void* addr = ::mmap(nullptr, sizeof(CpuBroadcastSharedState), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    RTP_LLM_CHECK_WITH_INFO(addr != MAP_FAILED, "CpuBroadcaster mmap shared state failed: %s", std::strerror(errno));
    return static_cast<CpuBroadcastSharedState*>(addr);
}

void validateSharedStateFd(int fd, const std::string& path) {
    struct stat st{};
    int         rc = ::fstat(fd, &st);
    RTP_LLM_CHECK_WITH_INFO(
        rc == 0, "CpuBroadcaster stat shared state %s failed: %s", path.c_str(), std::strerror(errno));
    RTP_LLM_CHECK_WITH_INFO(S_ISREG(st.st_mode) && st.st_uid == ::geteuid(),
                            "CpuBroadcaster shared state %s is not a regular file owned by uid %u",
                            path.c_str(),
                            static_cast<unsigned int>(::geteuid()));
    RTP_LLM_CHECK_WITH_INFO(st.st_size == static_cast<off_t>(sizeof(CpuBroadcastSharedState)),
                            "CpuBroadcaster shared state %s has invalid size %lld",
                            path.c_str(),
                            static_cast<long long>(st.st_size));
}

CpuBroadcastSharedState* createSharedState(const std::string& path) {
    ::unlink(path.c_str());
    int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    RTP_LLM_CHECK_WITH_INFO(
        fd >= 0, "CpuBroadcaster open shared state %s failed: %s", path.c_str(), std::strerror(errno));
    try {
        int rc = ::ftruncate(fd, sizeof(CpuBroadcastSharedState));
        RTP_LLM_CHECK_WITH_INFO(
            rc == 0, "CpuBroadcaster resize shared state %s failed: %s", path.c_str(), std::strerror(errno));
        validateSharedStateFd(fd, path);
        CpuBroadcastSharedState* state = mapSharedState(fd);
        closeFd(fd);
        for (uint32_t& decision : state->decisions) {
            __atomic_store_n(&decision, uint32_t{0}, __ATOMIC_RELEASE);
        }
        return state;
    } catch (...) {
        closeFd(fd);
        ::unlink(path.c_str());
        throw;
    }
}

CpuBroadcastSharedState* openSharedState(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    RTP_LLM_CHECK_WITH_INFO(
        fd >= 0, "CpuBroadcaster open shared state %s failed: %s", path.c_str(), std::strerror(errno));
    try {
        validateSharedStateFd(fd, path);
        CpuBroadcastSharedState* state = mapSharedState(fd);
        closeFd(fd);
        return state;
    } catch (...) {
        closeFd(fd);
        throw;
    }
}

void unmapSharedState(CpuBroadcastSharedState*& state) {
    if (state != nullptr) {
        ::munmap(state, sizeof(CpuBroadcastSharedState));
        state = nullptr;
    }
}

uint32_t* decisionSlot(CpuBroadcastSharedState* state, uint32_t generation) {
    return &state->decisions[generation & 1U];
}

uint32_t loadDecision(const uint32_t* decision) {
    return __atomic_load_n(decision, __ATOMIC_ACQUIRE);
}

void wakeDecisionWaiters(uint32_t* decision) {
    ::syscall(SYS_futex, decision, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
}

bool publishDecision(uint32_t* decision, uint32_t expected, uint32_t desired) {
    if (__atomic_compare_exchange_n(decision, &expected, desired, false, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)) {
        wakeDecisionWaiters(decision);
        return true;
    }
    return false;
}

bool rootConnectionClosed(int root_fd) {
    struct pollfd pfd{};
    pfd.fd     = root_fd;
    pfd.events = POLLIN;
    int rc     = ::poll(&pfd, 1, 0);
    if (rc < 0) {
        if (errno == EINTR) {
            return false;
        }
        return true;
    }
    if (rc == 0) {
        return false;
    }
    if (pfd.revents & POLLNVAL) {
        errno = EBADF;
        return true;
    }
    if (pfd.revents & (POLLERR | POLLHUP)) {
        errno = ECONNRESET;
        return true;
    }
    if (pfd.revents & POLLIN) {
        // A following broadcast can already be buffered after root commits.
        // Only EOF is a failure; leave any next-frame data untouched.
        char    byte = 0;
        ssize_t n    = ::recv(root_fd, &byte, sizeof(byte), MSG_PEEK | MSG_DONTWAIT);
        if (n == 0) {
            errno = ECONNRESET;
            return true;
        }
        if (n < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
            return true;
        }
    }
    return false;
}

bool waitForDecision(uint32_t* decision_slot,
                     uint32_t  previous_decision,
                     uint32_t  next_generation,
                     int       root_fd,
                     int       timeout_ms,
                     uint32_t& decision) {
    const bool no_timeout = timeout_ms <= 0;
    const auto start      = std::chrono::steady_clock::now();
    const auto deadline =
        no_timeout ? std::chrono::steady_clock::time_point::max() : start + std::chrono::milliseconds(timeout_ms);
    auto next_warning = start + std::chrono::milliseconds(kBroadcastStallWarnMs);

    while ((decision = loadDecision(decision_slot)) == previous_decision) {
        if (rootConnectionClosed(root_fd)) {
            return false;
        }

        const auto now = std::chrono::steady_clock::now();
        if (!no_timeout) {
            if (now >= deadline) {
                errno = ETIMEDOUT;
                return false;
            }
        }
        auto wait_deadline = now + std::chrono::milliseconds(kRootHealthPollMs);
        if (!no_timeout && deadline < wait_deadline) {
            wait_deadline = deadline;
        }
        int wait_ms =
            static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(wait_deadline - now).count());
        if (wait_ms <= 0) {
            wait_ms = 1;
        }
        struct timespec wait_time{wait_ms / 1000, static_cast<long>(wait_ms % 1000) * 1000 * 1000};
        int             rc = static_cast<int>(
            ::syscall(SYS_futex, decision_slot, FUTEX_WAIT, previous_decision, &wait_time, nullptr, 0));
        if (rc == 0 || errno == EINTR || errno == EAGAIN) {
            continue;
        }
        if (errno == ETIMEDOUT) {
            const auto after_wait = std::chrono::steady_clock::now();
            if (!no_timeout || after_wait < next_warning) {
                continue;
            }
            const auto waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(after_wait - start).count();
            RTP_LLM_LOG_WARNING("CpuBroadcaster commit decision still waiting after %lld ms "
                                "(generation=%u timeout disabled)",
                                static_cast<long long>(waited_ms),
                                next_generation);
            next_warning = after_wait + std::chrono::milliseconds(kBroadcastStallWarnMs);
            continue;
        }
        return false;
    }
    return true;
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
        int fd = ::accept4(listen_fd, nullptr, nullptr, SOCK_CLOEXEC);
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

uint32_t cpu_broadcast_detail::nextGeneration(uint32_t previous_generation) {
    RTP_LLM_CHECK_WITH_INFO((previous_generation & kBroadcastFailedMask) == 0,
                            "CpuBroadcaster invalid committed generation 0x%x",
                            previous_generation);
    return (previous_generation + 1) & ~kBroadcastFailedMask;
}

cpu_broadcast_detail::AbortDecision
cpu_broadcast_detail::abortOrObserveCommit(uint32_t* decision, uint32_t previous_generation, uint32_t next_generation) {
    uint32_t observed = previous_generation;
    if (__atomic_compare_exchange_n(
            decision, &observed, next_generation | kBroadcastFailedMask, false, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)) {
        ::syscall(SYS_futex, decision, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
        return AbortDecision::kAborted;
    }
    RTP_LLM_CHECK_WITH_INFO(observed == next_generation || observed == (next_generation | kBroadcastFailedMask),
                            "CpuBroadcaster abort for generation %u observed invalid decision 0x%x",
                            next_generation,
                            observed);
    return observed == next_generation ? AbortDecision::kCommitted : AbortDecision::kAborted;
}

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
    unmapSharedState(shared_state_);
    if (!shared_state_path_.empty()) {
        ::unlink(shared_state_path_.c_str());
        shared_state_path_.clear();
    }
    if (!my_uds_path_.empty()) {
        ::unlink(my_uds_path_.c_str());
        my_uds_path_.clear();
    }
    base_path_.clear();
    rank_                  = 0;
    world_size_            = 1;
    broadcast_timeout_ms_  = kDefaultBroadcastTimeoutMs;
    broadcast_generation_  = 0;
    broadcast_in_progress_ = false;
    failed_                = false;
    initialized_.store(false, std::memory_order_release);
}

void CpuBroadcaster::markBroadcastFailedLocked(uint32_t failed_generation) {
    RTP_LLM_LOG_WARNING("CpuBroadcaster rank %d generation %u entered a terminal failed state; "
                        "automatic c10d/NCCL fallback is disabled to prevent TP rank divergence. "
                        "A coordinated reset/re-init or process restart is required",
                        rank_,
                        failed_generation);
    failed_                = true;
    broadcast_in_progress_ = false;
    for (int& fd : peer_fds_) {
        shutdownAndCloseFd(fd);
    }
    closeFd(listen_fd_);
    unmapSharedState(shared_state_);
    if (!shared_state_path_.empty()) {
        ::unlink(shared_state_path_.c_str());
        shared_state_path_.clear();
    }
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
    broadcast_generation_ = 0;

    if (rank == 0) {
        try {
            const std::string path = makeUdsPath(base_path, 0);
            shared_state_path_     = makeSharedStatePath(base_path);
            shared_state_          = createSharedState(shared_state_path_);
            // Remove any stale socket left by a previous crashed run.
            ::unlink(path.c_str());

            listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
            RTP_LLM_CHECK_WITH_INFO(listen_fd_ >= 0, "CpuBroadcaster socket: %s", std::strerror(errno));

            struct sockaddr_un addr{};
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
            // Bootstrap is complete once every rank has connected. Drop the
            // listener and its pathname before request-time broadcasts so no
            // later fork/exec can keep the rendezvous endpoint alive.
            closeFd(listen_fd_);
            ::unlink(my_uds_path_.c_str());
            my_uds_path_.clear();
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
            for (int peer_rank = 1; peer_rank < world_size; ++peer_rank) {
                char    ready = 0;
                ssize_t n     = readAllWithTimeout(peer_fds_[peer_rank], &ready, sizeof(ready), kInitTimeoutMs);
                int     saved = errno;
                RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(ready)) && ready == kSharedStateReadyToken,
                                        "CpuBroadcaster shared state ready read from rank %d failed after %d ms: %s",
                                        peer_rank,
                                        kInitTimeoutMs,
                                        std::strerror(saved));
            }
            // Every peer now owns a mapping; remove the pathname so stale files
            // cannot collide with a later session.
            ::unlink(shared_state_path_.c_str());
            shared_state_path_.clear();
            RTP_LLM_LOG_INFO("CpuBroadcaster rank 0: accepted %d peer(s) on %s", world_size - 1, path.c_str());
        } catch (...) {
            cleanupStateLocked();
            throw;
        }
    } else {
        const std::string  path = makeUdsPath(base_path, 0);
        struct sockaddr_un addr{};
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
            fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
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

            shared_state_ = openSharedState(makeSharedStatePath(base_path));
            n     = writeAllWithTimeout(fd, &kSharedStateReadyToken, sizeof(kSharedStateReadyToken), kInitTimeoutMs);
            saved = errno;
            RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(kSharedStateReadyToken)),
                                    "CpuBroadcaster shared state ready write failed after %d ms: %s",
                                    kInitTimeoutMs,
                                    std::strerror(saved));
        } catch (...) {
            closeFd(fd);
            unmapSharedState(shared_state_);
            throw;
        }

        peer_fds_[0] = fd;
        RTP_LLM_LOG_INFO("CpuBroadcaster rank %d: connected to rank 0 at %s", rank, path.c_str());
    }

    initialized_.store(true, std::memory_order_release);
}

void CpuBroadcaster::broadcast(void* buf, std::size_t nbytes, int root) {
    int                      rank                = 0;
    int                      world_size          = 1;
    int                      timeout_ms          = 0;
    uint32_t                 previous_generation = 0;
    uint32_t                 next_generation     = 0;
    CpuBroadcastSharedState* shared_state        = nullptr;
    uint32_t*                decision_slot       = nullptr;
    uint32_t                 previous_decision   = 0;
    std::vector<int>         peer_fds;

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
        RTP_LLM_CHECK_WITH_INFO(shared_state_ != nullptr, "CpuBroadcaster shared decision state is not mapped");
        rank                = rank_;
        world_size          = world_size_;
        timeout_ms          = broadcast_timeout_ms_;
        previous_generation = broadcast_generation_;
        next_generation     = cpu_broadcast_detail::nextGeneration(previous_generation);
        shared_state        = shared_state_;
        decision_slot       = decisionSlot(shared_state, next_generation);
        previous_decision   = loadDecision(decision_slot);
        RTP_LLM_CHECK_WITH_INFO((previous_decision & kBroadcastFailedMask) == 0,
                                "CpuBroadcaster generation %u found stale abort decision 0x%x",
                                next_generation,
                                previous_decision);
        peer_fds               = peer_fds_;
        broadcast_in_progress_ = true;
    }

    auto finish_broadcast = [this, next_generation](bool failed) {
        std::lock_guard<std::mutex> lock(mu_);
        failed_ = failed_ || failed;
        if (!failed) {
            broadcast_generation_ = next_generation;
        }
        broadcast_in_progress_ = false;
    };
    auto fail_broadcast = [this, decision_slot, previous_decision, next_generation]() {
        publishDecision(decision_slot, previous_decision, next_generation | kBroadcastFailedMask);
        std::lock_guard<std::mutex> lock(mu_);
        markBroadcastFailedLocked(next_generation);
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
            for (int k = 1; k < world_size; ++k) {
                char    ready = 0;
                ssize_t n     = readAllWithTimeout(peer_fds[k], &ready, sizeof(ready), timeout_ms);
                RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(ready)) && ready == kBroadcastReadyToken,
                                        "CpuBroadcaster ready token read from rank %d failed after %d ms: %s",
                                        k,
                                        timeout_ms,
                                        std::strerror(errno));
            }
            // This single shared-memory CAS is the commit point. Unlike a
            // per-peer success-token loop, it cannot let an earlier peer
            // return successfully before a later peer's notification fails.
            RTP_LLM_CHECK_WITH_INFO(publishDecision(decision_slot, previous_decision, next_generation),
                                    "CpuBroadcaster commit for generation %u rejected by peer failure (decision=0x%x)",
                                    next_generation,
                                    loadDecision(decision_slot));
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
            n = writeAllWithTimeout(peer_fds[0], &kBroadcastReadyToken, sizeof(kBroadcastReadyToken), timeout_ms);
            RTP_LLM_CHECK_WITH_INFO(n == static_cast<ssize_t>(sizeof(kBroadcastReadyToken)),
                                    "CpuBroadcaster ready token write to rank 0 failed after %d ms: %s",
                                    timeout_ms,
                                    std::strerror(errno));
            uint32_t decision = previous_decision;
            if (!waitForDecision(
                    decision_slot, previous_decision, next_generation, peer_fds[0], timeout_ms, decision)) {
                const int  saved_errno = errno;
                const auto timeout_result =
                    cpu_broadcast_detail::abortOrObserveCommit(decision_slot, previous_decision, next_generation);
                if (timeout_result == cpu_broadcast_detail::AbortDecision::kCommitted) {
                    // Root's commit won the race with this peer's wait failure. The
                    // shared decision is authoritative, so this rank succeeds
                    // with every other rank instead of closing its transport.
                    decision = next_generation;
                } else {
                    RTP_LLM_FAIL("CpuBroadcaster commit decision wait for generation %u failed after %d ms "
                                 "or root disconnect: %s",
                                 next_generation,
                                 timeout_ms,
                                 std::strerror(saved_errno));
                }
            }
            RTP_LLM_CHECK_WITH_INFO(decision == next_generation,
                                    "CpuBroadcaster generation %u aborted by a peer failure (decision=0x%x)",
                                    next_generation,
                                    decision);
        }
    } catch (...) {
        fail_broadcast();
        throw;
    }
    finish_broadcast(false);
}

}  // namespace rtp_llm
