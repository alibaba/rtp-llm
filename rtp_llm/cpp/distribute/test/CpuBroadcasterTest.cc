#include "rtp_llm/cpp/distribute/CpuBroadcaster.h"

#include "gtest/gtest.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rtp_llm {
namespace {

constexpr char     kExpectedLinkProbeToken      = 0x5a;
constexpr char     kExpectedBroadcastSuccess    = 0x6b;
constexpr uint64_t kExpectedBroadcastFrameMagic = 0x5254504c4c4d5450ULL;

struct BroadcastFrameHeader {
    uint64_t magic;
    uint64_t nbytes;
};

std::string makeTempBase() {
    std::string       pattern = "/tmp/cpu_broadcaster_test.XXXXXX";
    std::vector<char> buf(pattern.begin(), pattern.end());
    buf.push_back('\0');
    char* dir = ::mkdtemp(buf.data());
    if (dir == nullptr) {
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    return std::string(dir) + "/bcast";
}

std::string socketPath(const std::string& base) {
    return base + "_0.sock";
}

void cleanupTempBase(const std::string& base, int max_rank = 4) {
    for (int rank = 0; rank <= max_rank; ++rank) {
        ::unlink((base + "_" + std::to_string(rank) + ".sock").c_str());
    }
    const auto slash = base.rfind('/');
    if (slash != std::string::npos) {
        ::rmdir(base.substr(0, slash).c_str());
    }
}

ssize_t writeAllRaw(int fd, const void* buf, size_t nbytes) {
    const char* p    = static_cast<const char*>(buf);
    size_t      left = nbytes;
    while (left > 0) {
#ifdef MSG_NOSIGNAL
        ssize_t n = ::send(fd, p, left, MSG_NOSIGNAL);
#else
        ssize_t n = ::send(fd, p, left, 0);
#endif
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

ssize_t readAllRaw(int fd, void* buf, size_t nbytes) {
    char*  p    = static_cast<char*>(buf);
    size_t left = nbytes;
    while (left > 0) {
        ssize_t n = ::read(fd, p, left);
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

int connectWithRetry(const std::string& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) {
            return -1;
        }

        struct sockaddr_un addr {};
        addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
        if (::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0) {
            return fd;
        }
        ::close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return -1;
}

int fakePeerSendRank(const std::string& base, int peer_rank) {
    int fd = connectWithRetry(socketPath(base));
    if (fd < 0) {
        std::fprintf(stderr, "fake peer connect failed: %s\n", std::strerror(errno));
        return 1;
    }
    int rc = writeAllRaw(fd, &peer_rank, sizeof(peer_rank)) == static_cast<ssize_t>(sizeof(peer_rank)) ? 0 : 1;
    ::close(fd);
    return rc;
}

int connectPeerAndReadProbe(const std::string& base, int peer_rank, std::string& error) {
    int fd = connectWithRetry(socketPath(base));
    if (fd < 0) {
        error = std::string("fake peer connect failed: ") + std::strerror(errno);
        return -1;
    }
    if (writeAllRaw(fd, &peer_rank, sizeof(peer_rank)) != static_cast<ssize_t>(sizeof(peer_rank))) {
        error = "fake peer handshake write failed";
        ::close(fd);
        return -1;
    }
    char probe = 0;
    if (readAllRaw(fd, &probe, sizeof(probe)) != static_cast<ssize_t>(sizeof(probe))
        || probe != kExpectedLinkProbeToken) {
        error = "fake peer link probe read failed";
        ::close(fd);
        return -1;
    }
    return fd;
}

void peerReadOneInt(const std::string& base, int& observed, std::string& error) {
    int fd = connectPeerAndReadProbe(base, 1, error);
    if (fd < 0) {
        return;
    }
    BroadcastFrameHeader header{};
    if (readAllRaw(fd, &header, sizeof(header)) != static_cast<ssize_t>(sizeof(header))
        || header.magic != kExpectedBroadcastFrameMagic || header.nbytes != sizeof(observed)) {
        error = "fake peer frame header read failed";
        ::close(fd);
        return;
    }
    if (readAllRaw(fd, &observed, sizeof(observed)) != static_cast<ssize_t>(sizeof(observed))) {
        error = "fake peer payload read failed";
        ::close(fd);
        return;
    }
    char success_token = 0;
    if (readAllRaw(fd, &success_token, sizeof(success_token)) != static_cast<ssize_t>(sizeof(success_token))
        || success_token != kExpectedBroadcastSuccess) {
        error = "fake peer success token read failed";
    }
    ::close(fd);
}

void peerHoldUnreadPayload(const std::string& base,
                           std::atomic<bool>& payload_ready,
                           std::atomic<bool>& release_peer,
                           std::string&       error) {
    int fd = connectPeerAndReadProbe(base, 1, error);
    if (fd < 0) {
        return;
    }
    while (!release_peer.load(std::memory_order_acquire)) {
        struct pollfd pfd {};
        pfd.fd     = fd;
        pfd.events = POLLIN;
        int rc     = ::poll(&pfd, 1, 10);
        if (rc > 0 && (pfd.revents & POLLIN)) {
            payload_ready.store(true, std::memory_order_release);
            break;
        }
        if (rc < 0 && errno != EINTR) {
            error = std::string("fake peer poll failed: ") + std::strerror(errno);
            ::close(fd);
            return;
        }
    }
    while (!release_peer.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ::close(fd);
}

int fakeServerWrongProbe(const std::string& base) {
    const std::string path = socketPath(base);
    ::unlink(path.c_str());

    int listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::fprintf(stderr, "fake server socket failed: %s\n", std::strerror(errno));
        return 1;
    }

    struct sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(listen_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::fprintf(stderr, "fake server bind failed: %s\n", std::strerror(errno));
        ::close(listen_fd);
        return 1;
    }
    if (::listen(listen_fd, 1) != 0) {
        std::fprintf(stderr, "fake server listen failed: %s\n", std::strerror(errno));
        ::close(listen_fd);
        ::unlink(path.c_str());
        return 1;
    }

    int fd = ::accept(listen_fd, nullptr, nullptr);
    if (fd < 0) {
        std::fprintf(stderr, "fake server accept failed: %s\n", std::strerror(errno));
        ::close(listen_fd);
        ::unlink(path.c_str());
        return 1;
    }
    int peer_rank = -1;
    int rc        = 0;
    if (readAllRaw(fd, &peer_rank, sizeof(peer_rank)) != static_cast<ssize_t>(sizeof(peer_rank))) {
        rc = 1;
    }
    const char wrong_probe = 0x13;
    if (writeAllRaw(fd, &wrong_probe, sizeof(wrong_probe)) != static_cast<ssize_t>(sizeof(wrong_probe))) {
        rc = 1;
    }
    ::close(fd);
    ::close(listen_fd);
    ::unlink(path.c_str());
    return rc;
}

int fakeRootSendMismatchedFrame(const std::string& base) {
    const std::string path = socketPath(base);
    ::unlink(path.c_str());

    int listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::fprintf(stderr, "fake root socket failed: %s\n", std::strerror(errno));
        return 1;
    }

    struct sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(listen_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::fprintf(stderr, "fake root bind failed: %s\n", std::strerror(errno));
        ::close(listen_fd);
        return 1;
    }
    if (::listen(listen_fd, 1) != 0) {
        std::fprintf(stderr, "fake root listen failed: %s\n", std::strerror(errno));
        ::close(listen_fd);
        ::unlink(path.c_str());
        return 1;
    }

    int fd = ::accept(listen_fd, nullptr, nullptr);
    if (fd < 0) {
        std::fprintf(stderr, "fake root accept failed: %s\n", std::strerror(errno));
        ::close(listen_fd);
        ::unlink(path.c_str());
        return 1;
    }

    int peer_rank = -1;
    int rc        = 0;
    if (readAllRaw(fd, &peer_rank, sizeof(peer_rank)) != static_cast<ssize_t>(sizeof(peer_rank)) || peer_rank != 1) {
        rc = 1;
    }
    if (writeAllRaw(fd, &kExpectedLinkProbeToken, sizeof(kExpectedLinkProbeToken))
        != static_cast<ssize_t>(sizeof(kExpectedLinkProbeToken))) {
        rc = 1;
    }
    const BroadcastFrameHeader bad_header{kExpectedBroadcastFrameMagic, sizeof(int) + 1};
    if (writeAllRaw(fd, &bad_header, sizeof(bad_header)) != static_cast<ssize_t>(sizeof(bad_header))) {
        rc = 1;
    }

    ::close(fd);
    ::close(listen_fd);
    ::unlink(path.c_str());
    return rc;
}

int expectThrowContains(const std::function<void()>& fn, const std::string& needle) {
    try {
        fn();
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(needle) != std::string::npos) {
            return 0;
        }
        std::fprintf(stderr, "unexpected exception: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "expected exception containing '%s'\n", needle.c_str());
    return 1;
}

pid_t spawnChild(const std::function<int()>& fn) {
    pid_t pid = ::fork();
    if (pid == 0) {
        ::alarm(30);
        try {
            int rc = fn();
            std::fflush(stdout);
            std::fflush(stderr);
            ::_exit(rc);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "child exception: %s\n", e.what());
            std::fflush(stderr);
            ::_exit(1);
        } catch (...) {
            std::fprintf(stderr, "child unknown exception\n");
            std::fflush(stderr);
            ::_exit(1);
        }
    }
    return pid;
}

void expectChildrenOk(const std::vector<pid_t>& pids) {
    for (pid_t pid : pids) {
        EXPECT_GT(pid, 0);
        if (pid <= 0) {
            continue;
        }
        int   status = 0;
        pid_t waited = ::waitpid(pid, &status, 0);
        EXPECT_EQ(waited, pid);
        if (waited != pid) {
            continue;
        }
        if (!WIFEXITED(status)) {
            ADD_FAILURE() << "child " << pid << " terminated by signal";
            continue;
        }
        EXPECT_EQ(WEXITSTATUS(status), 0) << "child " << pid << " failed";
    }
}

int happyPathChild(int rank, int tp_size, const std::string& base) {
    auto& bcast = CpuBroadcaster::instance();
    bcast.reset();
    bcast.initialize(rank, tp_size, base);

    int value = rank == 0 ? (0x123400 + tp_size) : 0;
    bcast.broadcast(&value, sizeof(value), 0);
    if (value != 0x123400 + tp_size) {
        std::fprintf(stderr, "rank %d got value %d\n", rank, value);
        return 1;
    }

    std::array<char, 16> payload{};
    if (rank == 0) {
        std::memcpy(payload.data(), "cpu-broadcast-ok", payload.size());
    }
    bcast.broadcast(payload.data(), payload.size(), 0);
    if (std::memcmp(payload.data(), "cpu-broadcast-ok", payload.size()) != 0) {
        std::fprintf(stderr, "rank %d got bad payload\n", rank);
        return 1;
    }

    int empty = 7;
    bcast.broadcast(&empty, 0, 0);
    if (empty != 7) {
        std::fprintf(stderr, "rank %d zero-byte broadcast mutated data\n", rank);
        return 1;
    }

    bcast.reset();
    return 0;
}

void runHappyPath(int tp_size) {
    const std::string  base = makeTempBase();
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] { return happyPathChild(0, tp_size, base); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    for (int rank = 1; rank < tp_size; ++rank) {
        pids.push_back(spawnChild([=] { return happyPathChild(rank, tp_size, base); }));
    }
    expectChildrenOk(pids);
    cleanupTempBase(base, tp_size);
}

TEST(CpuBroadcasterTest, BroadcastHappyPathTp2) {
    runHappyPath(2);
}

TEST(CpuBroadcasterTest, BroadcastHappyPathTp4) {
    runHappyPath(4);
}

TEST(CpuBroadcasterTest, Rank0RejectsBadPeerRank) {
    const std::string  base = makeTempBase();
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] {
        auto& bcast = CpuBroadcaster::instance();
        bcast.reset();
        return expectThrowContains([&] { bcast.initialize(0, 2, base); }, "bad peer_rank");
    }));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pids.push_back(spawnChild([=] { return fakePeerSendRank(base, 9); }));
    expectChildrenOk(pids);
    cleanupTempBase(base);
}

TEST(CpuBroadcasterTest, Rank0RejectsDuplicatePeerRank) {
    const std::string  base = makeTempBase();
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] {
        auto& bcast = CpuBroadcaster::instance();
        bcast.reset();
        return expectThrowContains([&] { bcast.initialize(0, 3, base); }, "duplicate peer rank");
    }));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pids.push_back(spawnChild([=] { return fakePeerSendRank(base, 1); }));
    pids.push_back(spawnChild([=] { return fakePeerSendRank(base, 1); }));
    expectChildrenOk(pids);
    cleanupTempBase(base);
}

TEST(CpuBroadcasterTest, NonRootRejectsBadLinkProbe) {
    const std::string  base = makeTempBase();
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] { return fakeServerWrongProbe(base); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pids.push_back(spawnChild([=] {
        auto& bcast = CpuBroadcaster::instance();
        bcast.reset();
        return expectThrowContains([&] { bcast.initialize(1, 2, base); }, "link probe read failed");
    }));
    expectChildrenOk(pids);
    cleanupTempBase(base);
}

TEST(CpuBroadcasterTest, NonRootRejectsMismatchedFrameHeader) {
    const std::string  base = makeTempBase();
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] { return fakeRootSendMismatchedFrame(base); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pids.push_back(spawnChild([=] {
        auto& bcast = CpuBroadcaster::instance();
        bcast.reset();
        bcast.initialize(1, 2, base);

        int value = 0;
        int rc    = expectThrowContains([&] { bcast.broadcast(&value, sizeof(value), 0); }, "frame header mismatch");
        rc |= expectThrowContains([&] { bcast.broadcast(&value, sizeof(value), 0); },
                                  "reset and reinitialize before reuse");
        bcast.reset();
        return rc;
    }));
    expectChildrenOk(pids);
    cleanupTempBase(base);
}

TEST(CpuBroadcasterTest, RootWriteFailureClosesOtherPeers) {
    auto& bcast = CpuBroadcaster::instance();
    bcast.reset();

    const std::string base = makeTempBase();

    std::atomic<bool> rank1_closed{false};
    std::atomic<bool> rank2_ready{false};
    std::atomic<bool> rank2_done{false};
    std::atomic<bool> rank2_saw_close{false};
    std::atomic<int>  rank2_fd{-1};
    std::string       rank1_error;
    std::string       rank2_error;

    std::thread rank1_thread([&] {
        int fd = connectPeerAndReadProbe(base, 1, rank1_error);
        if (fd < 0) {
            return;
        }
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
        rank1_closed.store(true, std::memory_order_release);
    });

    std::thread rank2_thread([&] {
        int fd = connectPeerAndReadProbe(base, 2, rank2_error);
        if (fd < 0) {
            rank2_done.store(true, std::memory_order_release);
            return;
        }
        rank2_fd.store(fd, std::memory_order_release);
        rank2_ready.store(true, std::memory_order_release);

        BroadcastFrameHeader header{};
        ssize_t              n = readAllRaw(fd, &header, sizeof(header));
        if (n < 0) {
            rank2_saw_close.store(true, std::memory_order_release);
        } else {
            rank2_error = "rank2 unexpectedly received a frame from failed root";
        }
        ::close(fd);
        rank2_done.store(true, std::memory_order_release);
    });

    bcast.initialize(0, 3, base);

    const auto ready_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((!rank1_closed.load(std::memory_order_acquire) || !rank2_ready.load(std::memory_order_acquire))
           && std::chrono::steady_clock::now() < ready_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (!rank1_closed.load(std::memory_order_acquire) || !rank2_ready.load(std::memory_order_acquire)) {
        int fd = rank2_fd.load(std::memory_order_acquire);
        if (fd >= 0) {
            ::shutdown(fd, SHUT_RDWR);
        }
        rank1_thread.join();
        rank2_thread.join();
        bcast.reset();
        cleanupTempBase(base);
        FAIL() << "fake peers were not ready; rank1_error=" << rank1_error << " rank2_error=" << rank2_error;
    }

    std::string broadcast_error;
    try {
        std::vector<char> payload(1024 * 1024, 0x55);
        bcast.broadcast(payload.data(), payload.size(), 0);
    } catch (const std::exception& e) { broadcast_error = e.what(); }

    const auto close_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!rank2_done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < close_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!rank2_done.load(std::memory_order_acquire)) {
        int fd = rank2_fd.load(std::memory_order_acquire);
        if (fd >= 0) {
            ::shutdown(fd, SHUT_RDWR);
        }
    }

    rank1_thread.join();
    rank2_thread.join();

    EXPECT_TRUE(rank1_error.empty()) << rank1_error;
    EXPECT_TRUE(rank2_error.empty()) << rank2_error;
    EXPECT_FALSE(broadcast_error.empty());
    EXPECT_TRUE(rank2_done.load(std::memory_order_acquire));
    EXPECT_TRUE(rank2_saw_close.load(std::memory_order_acquire));

    std::string retry_error;
    try {
        int value = 1;
        bcast.broadcast(&value, sizeof(value), 0);
    } catch (const std::exception& e) { retry_error = e.what(); }
    EXPECT_NE(retry_error.find("reset and reinitialize before reuse"), std::string::npos) << retry_error;

    bcast.reset();
    cleanupTempBase(base);
}

TEST(CpuBroadcasterTest, PayloadFailureDoesNotLetEarlierPeerReturn) {
    auto& bcast = CpuBroadcaster::instance();
    bcast.reset();

    const std::string base = makeTempBase();

    std::atomic<bool> rank1_ready{false};
    std::atomic<bool> rank1_done{false};
    std::atomic<bool> rank1_saw_close{false};
    std::atomic<bool> rank2_closed{false};
    std::atomic<int>  rank1_fd{-1};
    std::string       rank1_error;
    std::string       rank2_error;

    std::thread rank1_thread([&] {
        int fd = connectPeerAndReadProbe(base, 1, rank1_error);
        if (fd < 0) {
            rank1_done.store(true, std::memory_order_release);
            return;
        }
        rank1_fd.store(fd, std::memory_order_release);
        rank1_ready.store(true, std::memory_order_release);

        BroadcastFrameHeader header{};
        if (readAllRaw(fd, &header, sizeof(header)) != static_cast<ssize_t>(sizeof(header))
            || header.magic != kExpectedBroadcastFrameMagic || header.nbytes != sizeof(int)) {
            rank1_error = "rank1 frame header read failed";
            ::close(fd);
            rank1_done.store(true, std::memory_order_release);
            return;
        }
        int observed = 0;
        if (readAllRaw(fd, &observed, sizeof(observed)) != static_cast<ssize_t>(sizeof(observed)) || observed != 17) {
            rank1_error = "rank1 payload read failed";
            ::close(fd);
            rank1_done.store(true, std::memory_order_release);
            return;
        }
        char    success_token = 0;
        ssize_t n             = readAllRaw(fd, &success_token, sizeof(success_token));
        if (n < 0) {
            rank1_saw_close.store(true, std::memory_order_release);
        } else {
            rank1_error = "rank1 unexpectedly received success token";
        }
        ::close(fd);
        rank1_done.store(true, std::memory_order_release);
    });

    std::thread rank2_thread([&] {
        int fd = connectPeerAndReadProbe(base, 2, rank2_error);
        if (fd < 0) {
            return;
        }
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
        rank2_closed.store(true, std::memory_order_release);
    });

    bcast.initialize(0, 3, base);

    const auto ready_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((!rank1_ready.load(std::memory_order_acquire) || !rank2_closed.load(std::memory_order_acquire))
           && std::chrono::steady_clock::now() < ready_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (!rank1_ready.load(std::memory_order_acquire) || !rank2_closed.load(std::memory_order_acquire)) {
        int fd = rank1_fd.load(std::memory_order_acquire);
        if (fd >= 0) {
            ::shutdown(fd, SHUT_RDWR);
        }
        rank1_thread.join();
        rank2_thread.join();
        bcast.reset();
        cleanupTempBase(base);
        FAIL() << "fake peers were not ready; rank1_error=" << rank1_error << " rank2_error=" << rank2_error;
    }

    std::string broadcast_error;
    try {
        int value = 17;
        bcast.broadcast(&value, sizeof(value), 0);
    } catch (const std::exception& e) { broadcast_error = e.what(); }

    const auto close_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!rank1_done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < close_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!rank1_done.load(std::memory_order_acquire)) {
        int fd = rank1_fd.load(std::memory_order_acquire);
        if (fd >= 0) {
            ::shutdown(fd, SHUT_RDWR);
        }
    }

    rank1_thread.join();
    rank2_thread.join();

    EXPECT_TRUE(rank1_error.empty()) << rank1_error;
    EXPECT_TRUE(rank2_error.empty()) << rank2_error;
    EXPECT_FALSE(broadcast_error.empty());
    EXPECT_TRUE(rank1_done.load(std::memory_order_acquire));
    EXPECT_TRUE(rank1_saw_close.load(std::memory_order_acquire));

    bcast.reset();
    cleanupTempBase(base);
}

TEST(CpuBroadcasterTest, NonRootFailsFastOnNonRetryableConnectError) {
    const std::string base = makeTempBase();
    ASSERT_EQ(::symlink((base + "_0.sock").c_str(), socketPath(base).c_str()), 0);

    auto& bcast = CpuBroadcaster::instance();
    bcast.reset();
    const auto start = std::chrono::steady_clock::now();
    try {
        bcast.initialize(1, 2, base);
        ADD_FAILURE() << "expected CpuBroadcaster connect failure";
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("CpuBroadcaster connect("), std::string::npos) << msg;
        EXPECT_NE(msg.find("failed:"), std::string::npos) << msg;
        EXPECT_EQ(msg.find("failed after"), std::string::npos) << msg;
    }
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    EXPECT_LT(elapsed_ms, 1000);
    bcast.reset();

    cleanupTempBase(base);
}

TEST(CpuBroadcasterTest, ResetAllowsNewBasePath) {
    auto& bcast = CpuBroadcaster::instance();
    bcast.reset();

    const std::string base1 = makeTempBase();
    const std::string base2 = makeTempBase();
    bcast.initialize(0, 1, base1);
    ASSERT_TRUE(bcast.isInitialized());
    bcast.reset();
    ASSERT_FALSE(bcast.isInitialized());

    bcast.initialize(0, 1, base2);
    ASSERT_TRUE(bcast.isInitialized());
    bcast.reset();

    cleanupTempBase(base1);
    cleanupTempBase(base2);
}

TEST(CpuBroadcasterTest, AllowsCrossThreadBroadcastAndResetWhenIdle) {
    auto& bcast = CpuBroadcaster::instance();
    bcast.reset();

    const std::string base     = makeTempBase();
    int               observed = 0;
    std::string       peer_error;
    std::thread       peer_thread([&] { peerReadOneInt(base, observed, peer_error); });
    bcast.initialize(0, 2, base);

    std::string broadcast_error;
    std::thread broadcast_thread([&] {
        try {
            int value = 7;
            bcast.broadcast(&value, sizeof(value), 0);
        } catch (const std::exception& e) { broadcast_error = e.what(); }
    });
    broadcast_thread.join();
    peer_thread.join();
    EXPECT_TRUE(broadcast_error.empty()) << broadcast_error;
    EXPECT_TRUE(peer_error.empty()) << peer_error;
    EXPECT_EQ(observed, 7);

    std::string reset_error;
    std::thread reset_thread([&] {
        try {
            bcast.reset();
        } catch (const std::exception& e) { reset_error = e.what(); }
    });
    reset_thread.join();
    EXPECT_TRUE(reset_error.empty()) << reset_error;

    bcast.reset();
    cleanupTempBase(base);
}

TEST(CpuBroadcasterTest, ResetRejectsInFlightBroadcast) {
    auto& bcast = CpuBroadcaster::instance();
    bcast.reset();

    const std::string base = makeTempBase();
    std::atomic<bool> payload_ready{false};
    std::atomic<bool> release_peer{false};
    std::string       peer_error;
    std::thread       peer_thread([&] { peerHoldUnreadPayload(base, payload_ready, release_peer, peer_error); });
    bcast.initialize(0, 2, base);

    std::string broadcast_error;
    std::thread broadcast_thread([&] {
        try {
            std::vector<char> payload(64 * 1024 * 1024, 0x7f);
            bcast.broadcast(payload.data(), payload.size(), 0);
        } catch (const std::exception& e) { broadcast_error = e.what(); }
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!payload_ready.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    std::string reset_error;
    if (payload_ready.load(std::memory_order_acquire)) {
        try {
            bcast.reset();
        } catch (const std::exception& e) { reset_error = e.what(); }
    }

    release_peer.store(true, std::memory_order_release);
    peer_thread.join();
    broadcast_thread.join();

    EXPECT_TRUE(peer_error.empty()) << peer_error;
    EXPECT_TRUE(payload_ready.load(std::memory_order_acquire)) << "fake peer did not observe broadcast payload";
    EXPECT_NE(reset_error.find("reset called while broadcastCPU is in progress"), std::string::npos) << reset_error;
    EXPECT_FALSE(broadcast_error.empty());

    bcast.reset();
    cleanupTempBase(base);
}

TEST(CpuBroadcasterTest, BroadcastTimeoutRequiresResetBeforeReuse) {
    auto& bcast = CpuBroadcaster::instance();
    bcast.reset();
    ::setenv("RTP_LLM_CPU_TP_BROADCASTER_BROADCAST_TIMEOUT_MS", "50", 1);

    const std::string base = makeTempBase();
    std::atomic<bool> payload_ready{false};
    std::atomic<bool> release_peer{false};
    std::string       peer_error;
    std::thread       peer_thread([&] { peerHoldUnreadPayload(base, payload_ready, release_peer, peer_error); });
    bcast.initialize(0, 2, base);

    std::string broadcast_error;
    try {
        std::vector<char> payload(64 * 1024 * 1024, 0x7f);
        bcast.broadcast(payload.data(), payload.size(), 0);
    } catch (const std::exception& e) { broadcast_error = e.what(); }

    std::string retry_error;
    try {
        int value = 1;
        bcast.broadcast(&value, sizeof(value), 0);
    } catch (const std::exception& e) { retry_error = e.what(); }

    release_peer.store(true, std::memory_order_release);
    peer_thread.join();
    ::unsetenv("RTP_LLM_CPU_TP_BROADCASTER_BROADCAST_TIMEOUT_MS");

    EXPECT_TRUE(peer_error.empty()) << peer_error;
    EXPECT_NE(broadcast_error.find("failed after 50 ms"), std::string::npos) << broadcast_error;
    EXPECT_NE(retry_error.find("reset and reinitialize before reuse"), std::string::npos) << retry_error;

    bcast.reset();
    cleanupTempBase(base);
}

}  // namespace
}  // namespace rtp_llm
