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
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rtp_llm {
namespace {

constexpr char     kExpectedLinkProbeToken      = 0x5a;
constexpr char     kExpectedSharedStateReady    = 0x6a;
constexpr char     kExpectedBroadcastReady      = 0x6b;
constexpr uint32_t kExpectedBroadcastFailedMask = uint32_t{1} << 31;
constexpr uint64_t kExpectedBroadcastFrameMagic = 0x5254504c4c4d5450ULL;

struct BroadcastFrameHeader {
    uint64_t magic;
    uint64_t nbytes;
};

struct TestSharedBroadcastState {
    alignas(uint32_t) uint32_t decision;
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

std::string sharedStatePath(const std::string& base) {
    return base + ".state";
}

void cleanupTempBase(const std::string& base, int max_rank = 4) {
    for (int rank = 0; rank <= max_rank; ++rank) {
        ::unlink((base + "_" + std::to_string(rank) + ".sock").c_str());
    }
    ::unlink(sharedStatePath(base).c_str());
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

TestSharedBroadcastState* mapTestSharedState(int fd) {
    void* addr = ::mmap(nullptr, sizeof(TestSharedBroadcastState), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    return addr == MAP_FAILED ? nullptr : static_cast<TestSharedBroadcastState*>(addr);
}

TestSharedBroadcastState* openTestSharedState(const std::string& base, std::string& error) {
    int fd = ::open(sharedStatePath(base).c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        error = std::string("fake peer open shared state failed: ") + std::strerror(errno);
        return nullptr;
    }
    TestSharedBroadcastState* state = mapTestSharedState(fd);
    ::close(fd);
    if (state == nullptr) {
        error = std::string("fake peer mmap shared state failed: ") + std::strerror(errno);
    }
    return state;
}

TestSharedBroadcastState* createTestSharedState(const std::string& base, std::string& error) {
    const std::string path = sharedStatePath(base);
    ::unlink(path.c_str());
    int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        error = std::string("fake root open shared state failed: ") + std::strerror(errno);
        return nullptr;
    }
    if (::ftruncate(fd, sizeof(TestSharedBroadcastState)) != 0) {
        error = std::string("fake root resize shared state failed: ") + std::strerror(errno);
        ::close(fd);
        ::unlink(path.c_str());
        return nullptr;
    }
    TestSharedBroadcastState* state = mapTestSharedState(fd);
    ::close(fd);
    if (state == nullptr) {
        error = std::string("fake root mmap shared state failed: ") + std::strerror(errno);
        ::unlink(path.c_str());
        return nullptr;
    }
    __atomic_store_n(&state->decision, uint32_t{0}, __ATOMIC_RELEASE);
    return state;
}

void unmapTestSharedState(TestSharedBroadcastState*& state) {
    if (state != nullptr) {
        ::munmap(state, sizeof(TestSharedBroadcastState));
        state = nullptr;
    }
}

bool waitForTestDecision(TestSharedBroadcastState* state, uint32_t expected, uint32_t& decision) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((decision = __atomic_load_n(&state->decision, __ATOMIC_ACQUIRE)) == expected
           && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return decision != expected;
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

int connectPeerAndReadProbe(const std::string&         base,
                            int                        peer_rank,
                            std::string&               error,
                            TestSharedBroadcastState** shared_state = nullptr) {
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
    TestSharedBroadcastState* state = openTestSharedState(base, error);
    if (state == nullptr) {
        ::close(fd);
        return -1;
    }
    if (writeAllRaw(fd, &kExpectedSharedStateReady, sizeof(kExpectedSharedStateReady))
        != static_cast<ssize_t>(sizeof(kExpectedSharedStateReady))) {
        error = "fake peer shared state ready write failed";
        unmapTestSharedState(state);
        ::close(fd);
        return -1;
    }
    if (shared_state != nullptr) {
        *shared_state = state;
    } else {
        unmapTestSharedState(state);
    }
    return fd;
}

void peerReadOneInt(const std::string& base, int& observed, std::string& error) {
    TestSharedBroadcastState* state = nullptr;
    int                       fd    = connectPeerAndReadProbe(base, 1, error, &state);
    if (fd < 0) {
        return;
    }
    BroadcastFrameHeader header{};
    if (readAllRaw(fd, &header, sizeof(header)) != static_cast<ssize_t>(sizeof(header))
        || header.magic != kExpectedBroadcastFrameMagic || header.nbytes != sizeof(observed)) {
        error = "fake peer frame header read failed";
        unmapTestSharedState(state);
        ::close(fd);
        return;
    }
    if (readAllRaw(fd, &observed, sizeof(observed)) != static_cast<ssize_t>(sizeof(observed))) {
        error = "fake peer payload read failed";
        unmapTestSharedState(state);
        ::close(fd);
        return;
    }
    if (writeAllRaw(fd, &kExpectedBroadcastReady, sizeof(kExpectedBroadcastReady))
        != static_cast<ssize_t>(sizeof(kExpectedBroadcastReady))) {
        error = "fake peer broadcast ready write failed";
    } else {
        uint32_t decision = 0;
        if (!waitForTestDecision(state, 0, decision) || decision != 1) {
            error = "fake peer did not observe committed generation";
        }
    }
    unmapTestSharedState(state);
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

    std::string               state_error;
    TestSharedBroadcastState* state = createTestSharedState(base, state_error);
    if (state == nullptr) {
        std::fprintf(stderr, "%s\n", state_error.c_str());
        return 1;
    }
    auto cleanup_state = [&] {
        unmapTestSharedState(state);
        ::unlink(sharedStatePath(base).c_str());
    };

    int listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::fprintf(stderr, "fake root socket failed: %s\n", std::strerror(errno));
        cleanup_state();
        return 1;
    }

    struct sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(listen_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::fprintf(stderr, "fake root bind failed: %s\n", std::strerror(errno));
        ::close(listen_fd);
        cleanup_state();
        return 1;
    }
    if (::listen(listen_fd, 1) != 0) {
        std::fprintf(stderr, "fake root listen failed: %s\n", std::strerror(errno));
        ::close(listen_fd);
        ::unlink(path.c_str());
        cleanup_state();
        return 1;
    }

    int fd = ::accept(listen_fd, nullptr, nullptr);
    if (fd < 0) {
        std::fprintf(stderr, "fake root accept failed: %s\n", std::strerror(errno));
        ::close(listen_fd);
        ::unlink(path.c_str());
        cleanup_state();
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
    char state_ready = 0;
    if (readAllRaw(fd, &state_ready, sizeof(state_ready)) != static_cast<ssize_t>(sizeof(state_ready))
        || state_ready != kExpectedSharedStateReady) {
        rc = 1;
    }
    const BroadcastFrameHeader bad_header{kExpectedBroadcastFrameMagic, sizeof(int) + 1};
    if (writeAllRaw(fd, &bad_header, sizeof(bad_header)) != static_cast<ssize_t>(sizeof(bad_header))) {
        rc = 1;
    }

    ::close(fd);
    ::close(listen_fd);
    ::unlink(path.c_str());
    cleanup_state();
    return rc;
}

int fakeRootWaitAfterAllPeersReady(const std::string& base, int ready_fd) {
    constexpr int             world_size = 3;
    constexpr int             value      = 0x1234;
    const std::string         path       = socketPath(base);
    std::string               state_error;
    TestSharedBroadcastState* state = createTestSharedState(base, state_error);
    if (state == nullptr) {
        std::fprintf(stderr, "%s\n", state_error.c_str());
        return 1;
    }

    int              listen_fd = -1;
    std::vector<int> peer_fds(world_size, -1);
    auto             cleanup = [&] {
        for (int fd : peer_fds) {
            if (fd >= 0) {
                ::close(fd);
            }
        }
        if (listen_fd >= 0) {
            ::close(listen_fd);
        }
        if (ready_fd >= 0) {
            ::close(ready_fd);
        }
        ::unlink(path.c_str());
        unmapTestSharedState(state);
        ::unlink(sharedStatePath(base).c_str());
    };
    auto fail = [&](const char* message) {
        std::fprintf(stderr, "fake root %s: %s\n", message, std::strerror(errno));
        cleanup();
        return 1;
    };

    ::unlink(path.c_str());
    listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        return fail("socket failed");
    }

    struct sockaddr_un addr {};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(listen_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
        return fail("bind failed");
    }
    if (::listen(listen_fd, world_size - 1) != 0) {
        return fail("listen failed");
    }

    for (int connected = 1; connected < world_size; ++connected) {
        int fd = ::accept(listen_fd, nullptr, nullptr);
        if (fd < 0) {
            return fail("accept failed");
        }
        int peer_rank = -1;
        if (readAllRaw(fd, &peer_rank, sizeof(peer_rank)) != static_cast<ssize_t>(sizeof(peer_rank)) || peer_rank <= 0
            || peer_rank >= world_size || peer_fds[peer_rank] >= 0) {
            ::close(fd);
            errno = EPROTO;
            return fail("peer handshake failed");
        }
        peer_fds[peer_rank] = fd;
        if (writeAllRaw(fd, &kExpectedLinkProbeToken, sizeof(kExpectedLinkProbeToken))
            != static_cast<ssize_t>(sizeof(kExpectedLinkProbeToken))) {
            return fail("link probe write failed");
        }
        char state_ready = 0;
        if (readAllRaw(fd, &state_ready, sizeof(state_ready)) != static_cast<ssize_t>(sizeof(state_ready))
            || state_ready != kExpectedSharedStateReady) {
            errno = EPROTO;
            return fail("shared state ready read failed");
        }
    }

    const BroadcastFrameHeader header{kExpectedBroadcastFrameMagic, sizeof(value)};
    for (int rank = 1; rank < world_size; ++rank) {
        if (writeAllRaw(peer_fds[rank], &header, sizeof(header)) != static_cast<ssize_t>(sizeof(header))
            || writeAllRaw(peer_fds[rank], &value, sizeof(value)) != static_cast<ssize_t>(sizeof(value))) {
            return fail("payload write failed");
        }
    }
    for (int rank = 1; rank < world_size; ++rank) {
        char ready = 0;
        if (readAllRaw(peer_fds[rank], &ready, sizeof(ready)) != static_cast<ssize_t>(sizeof(ready))
            || ready != kExpectedBroadcastReady) {
            errno = EPROTO;
            return fail("broadcast ready read failed");
        }
    }

    const char all_ready = 1;
    if (::write(ready_fd, &all_ready, sizeof(all_ready)) != static_cast<ssize_t>(sizeof(all_ready))) {
        return fail("parent notification failed");
    }
    ::close(ready_fd);
    ready_fd = -1;

    // Deliberately never publish commit or abort. The parent terminates this
    // process after every peer has sent ready, reproducing root loss in the
    // exact pre-decision window.
    while (true) {
        ::pause();
    }
}

int productionRootExecChildAndWait(const std::string& base, int ready_fd) {
    auto& bcast = CpuBroadcaster::instance();
    bcast.reset();
    bcast.initialize(0, 3, base);

    int exec_pipe[2] = {-1, -1};
    if (::pipe2(exec_pipe, O_CLOEXEC) != 0) {
        std::fprintf(stderr, "root exec status pipe failed: %s\n", std::strerror(errno));
        return 1;
    }

    pid_t exec_pid = ::fork();
    if (exec_pid < 0) {
        std::fprintf(stderr, "root fork for exec child failed: %s\n", std::strerror(errno));
        ::close(exec_pipe[0]);
        ::close(exec_pipe[1]);
        return 1;
    }
    if (exec_pid == 0) {
        ::close(exec_pipe[0]);
        ::close(ready_fd);
        ::execl("/bin/sleep", "sleep", "30", static_cast<char*>(nullptr));
        const int     saved   = errno;
        const ssize_t ignored = ::write(exec_pipe[1], &saved, sizeof(saved));
        (void)ignored;
        ::_exit(127);
    }

    ::close(exec_pipe[1]);
    int     exec_error = 0;
    ssize_t n;
    do {
        n = ::read(exec_pipe[0], &exec_error, sizeof(exec_error));
    } while (n < 0 && errno == EINTR);
    ::close(exec_pipe[0]);
    if (n != 0) {
        std::fprintf(stderr,
                     "root exec child failed before close-on-exec: %s\n",
                     n == static_cast<ssize_t>(sizeof(exec_error)) ? std::strerror(exec_error) : "short status read");
        ::kill(exec_pid, SIGKILL);
        ::waitpid(exec_pid, nullptr, 0);
        return 1;
    }

    if (::write(ready_fd, &exec_pid, sizeof(exec_pid)) != static_cast<ssize_t>(sizeof(exec_pid))) {
        std::fprintf(stderr, "root exec child notification failed: %s\n", std::strerror(errno));
        ::kill(exec_pid, SIGKILL);
        ::waitpid(exec_pid, nullptr, 0);
        return 1;
    }
    ::close(ready_fd);

    // The parent kills this root after /bin/sleep has successfully exec'd. If
    // a production UDS fd lacks close-on-exec, sleep keeps the peer connection
    // alive and the surviving ranks below remain blocked with timeout disabled.
    while (true) {
        ::pause();
    }
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

bool waitChildWithTimeout(pid_t pid, std::chrono::milliseconds timeout, int& status) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        pid_t waited = ::waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            return true;
        }
        if (waited < 0 && errno != EINTR) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
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

TEST(CpuBroadcasterTest, WaitFailureAndCommitUseOneLinearizationPoint) {
    constexpr uint32_t previous_generation = 0;
    constexpr uint32_t next_generation     = 1;

    // Wait failure wins: it publishes abort, so root's later commit CAS must lose.
    uint32_t decision = previous_generation;
    EXPECT_EQ(cpu_broadcast_detail::abortOrObserveCommit(&decision, previous_generation, next_generation),
              cpu_broadcast_detail::AbortDecision::kAborted);
    EXPECT_EQ(__atomic_load_n(&decision, __ATOMIC_ACQUIRE), next_generation | kExpectedBroadcastFailedMask);
    uint32_t expected = previous_generation;
    EXPECT_FALSE(
        __atomic_compare_exchange_n(&decision, &expected, next_generation, false, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE));
    EXPECT_EQ(expected, next_generation | kExpectedBroadcastFailedMask);

    // Root wins: the peer's abort CAS must lose and be interpreted as the
    // same successful commit already observed by every other rank.
    __atomic_store_n(&decision, previous_generation, __ATOMIC_RELEASE);
    expected = previous_generation;
    ASSERT_TRUE(
        __atomic_compare_exchange_n(&decision, &expected, next_generation, false, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE));
    EXPECT_EQ(cpu_broadcast_detail::abortOrObserveCommit(&decision, previous_generation, next_generation),
              cpu_broadcast_detail::AbortDecision::kCommitted);
    EXPECT_EQ(__atomic_load_n(&decision, __ATOMIC_ACQUIRE), next_generation);
}

TEST(CpuBroadcasterTest, RootExitBeforeDecisionAbortsAllSurvivingRanks) {
    ::unsetenv("RTP_LLM_CPU_TP_BROADCASTER_BROADCAST_TIMEOUT_MS");
    const std::string base = makeTempBase();

    int ready_pipe[2] = {-1, -1};
    ASSERT_EQ(::pipe(ready_pipe), 0);
    const int ready_write_fd = ready_pipe[1];
    pid_t     root_pid       = spawnChild([base, ready_write_fd, ready_read_fd = ready_pipe[0]] {
        ::close(ready_read_fd);
        return fakeRootWaitAfterAllPeersReady(base, ready_write_fd);
    });
    ASSERT_GT(root_pid, 0);
    ::close(ready_pipe[1]);
    ready_pipe[1] = -1;

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::vector<pid_t> peer_pids;
    for (int rank = 1; rank < 3; ++rank) {
        peer_pids.push_back(spawnChild([base, rank, ready_read_fd = ready_pipe[0]] {
            ::close(ready_read_fd);
            auto& bcast = CpuBroadcaster::instance();
            bcast.reset();
            bcast.initialize(rank, 3, base);

            int value = 0;
            int rc    = 1;
            try {
                bcast.broadcast(&value, sizeof(value), 0);
                std::fprintf(stderr, "rank %d unexpectedly committed after root exit\n", rank);
            } catch (const std::exception& e) {
                const std::string message = e.what();
                if (message.find("commit decision wait") != std::string::npos
                    || message.find("aborted by a peer failure") != std::string::npos) {
                    rc = 0;
                } else {
                    std::fprintf(stderr, "rank %d got unexpected exception: %s\n", rank, e.what());
                }
            }
            if (value != 0x1234) {
                std::fprintf(stderr, "rank %d did not receive payload before root exit\n", rank);
                rc = 1;
            }
            bcast.reset();
            return rc;
        }));
        ASSERT_GT(peer_pids.back(), 0);
    }

    struct pollfd pfd {};
    pfd.fd     = ready_pipe[0];
    pfd.events = POLLIN;
    int ready_rc;
    do {
        ready_rc = ::poll(&pfd, 1, 5000);
    } while (ready_rc < 0 && errno == EINTR);
    char all_ready = 0;
    if (ready_rc > 0 && (pfd.revents & POLLIN)) {
        EXPECT_EQ(::read(ready_pipe[0], &all_ready, sizeof(all_ready)), static_cast<ssize_t>(sizeof(all_ready)));
    }
    ::close(ready_pipe[0]);
    ready_pipe[0] = -1;

    EXPECT_EQ(ready_rc, 1) << "fake root did not observe every ready token";
    EXPECT_EQ(all_ready, 1);
    EXPECT_EQ(::kill(root_pid, SIGKILL), 0);

    int root_status = 0;
    EXPECT_EQ(::waitpid(root_pid, &root_status, 0), root_pid);
    EXPECT_TRUE(WIFSIGNALED(root_status));
    EXPECT_EQ(WTERMSIG(root_status), SIGKILL);

    for (pid_t peer_pid : peer_pids) {
        int  peer_status = 0;
        bool exited      = waitChildWithTimeout(peer_pid, std::chrono::seconds(5), peer_status);
        if (!exited) {
            ::kill(peer_pid, SIGKILL);
            ::waitpid(peer_pid, &peer_status, 0);
        }
        EXPECT_TRUE(exited) << "surviving rank remained stuck after root exit";
        if (exited) {
            EXPECT_TRUE(WIFEXITED(peer_status));
            EXPECT_EQ(WEXITSTATUS(peer_status), 0);
        }
    }

    cleanupTempBase(base, 3);
}

TEST(CpuBroadcasterTest, ExecChildDoesNotKeepRootConnectionsAlive) {
    ::unsetenv("RTP_LLM_CPU_TP_BROADCASTER_BROADCAST_TIMEOUT_MS");
    const std::string base = makeTempBase();

    int ready_pipe[2] = {-1, -1};
    ASSERT_EQ(::pipe(ready_pipe), 0);
    const int ready_write_fd = ready_pipe[1];
    pid_t     root_pid       = spawnChild([base, ready_write_fd, ready_read_fd = ready_pipe[0]] {
        ::close(ready_read_fd);
        return productionRootExecChildAndWait(base, ready_write_fd);
    });
    ASSERT_GT(root_pid, 0);
    ::close(ready_pipe[1]);
    ready_pipe[1] = -1;

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::vector<pid_t> peer_pids;
    for (int rank = 1; rank < 3; ++rank) {
        peer_pids.push_back(spawnChild([base, rank, ready_read_fd = ready_pipe[0]] {
            ::close(ready_read_fd);
            auto& bcast = CpuBroadcaster::instance();
            bcast.reset();
            bcast.initialize(rank, 3, base);
            int value = 0;
            int rc    = expectThrowContains([&] { bcast.broadcast(&value, sizeof(value), 0); },
                                         "frame header read from rank 0 failed");
            bcast.reset();
            return rc;
        }));
        ASSERT_GT(peer_pids.back(), 0);
    }

    pid_t   exec_pid = -1;
    ssize_t n        = readAllRaw(ready_pipe[0], &exec_pid, sizeof(exec_pid));
    ::close(ready_pipe[0]);
    ready_pipe[0] = -1;
    ASSERT_EQ(n, static_cast<ssize_t>(sizeof(exec_pid)));
    ASSERT_GT(exec_pid, 0);
    EXPECT_EQ(::access(socketPath(base).c_str(), F_OK), -1);
    EXPECT_EQ(errno, ENOENT) << "bootstrap listener pathname remained after initialization";

    EXPECT_EQ(::kill(root_pid, SIGKILL), 0);
    int root_status = 0;
    EXPECT_EQ(::waitpid(root_pid, &root_status, 0), root_pid);
    EXPECT_TRUE(WIFSIGNALED(root_status));
    EXPECT_EQ(WTERMSIG(root_status), SIGKILL);

    for (pid_t peer_pid : peer_pids) {
        int  peer_status = 0;
        bool exited      = waitChildWithTimeout(peer_pid, std::chrono::seconds(5), peer_status);
        if (!exited) {
            ::kill(peer_pid, SIGKILL);
            ::waitpid(peer_pid, &peer_status, 0);
        }
        EXPECT_TRUE(exited) << "exec child inherited a root UDS connection";
        if (exited) {
            EXPECT_TRUE(WIFEXITED(peer_status));
            EXPECT_EQ(WEXITSTATUS(peer_status), 0);
        }
    }

    EXPECT_EQ(::kill(exec_pid, SIGKILL), 0);
    cleanupTempBase(base, 3);
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

TEST(CpuBroadcasterTest, ReadyPeerObservesAbortWhenLaterPeerFails) {
    auto& bcast = CpuBroadcaster::instance();
    bcast.reset();

    const std::string base = makeTempBase();

    std::atomic<bool> rank1_ready{false};
    std::atomic<bool> rank1_done{false};
    std::atomic<bool> rank1_saw_abort{false};
    std::atomic<bool> rank2_ready{false};
    std::atomic<bool> rank2_closed{false};
    std::atomic<int>  rank1_fd{-1};
    std::atomic<int>  rank2_fd{-1};
    std::string       rank1_error;
    std::string       rank2_error;

    std::thread rank1_thread([&] {
        TestSharedBroadcastState* state = nullptr;
        int                       fd    = connectPeerAndReadProbe(base, 1, rank1_error, &state);
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
            unmapTestSharedState(state);
            ::close(fd);
            rank1_done.store(true, std::memory_order_release);
            return;
        }
        int observed = 0;
        if (readAllRaw(fd, &observed, sizeof(observed)) != static_cast<ssize_t>(sizeof(observed)) || observed != 17) {
            rank1_error = "rank1 payload read failed";
            unmapTestSharedState(state);
            ::close(fd);
            rank1_done.store(true, std::memory_order_release);
            return;
        }
        if (writeAllRaw(fd, &kExpectedBroadcastReady, sizeof(kExpectedBroadcastReady))
            != static_cast<ssize_t>(sizeof(kExpectedBroadcastReady))) {
            rank1_error = "rank1 broadcast ready write failed";
        } else {
            uint32_t decision = 0;
            if (!waitForTestDecision(state, 0, decision)) {
                rank1_error = "rank1 timed out waiting for shared abort decision";
            } else if (decision == (kExpectedBroadcastFailedMask | uint32_t{1})) {
                rank1_saw_abort.store(true, std::memory_order_release);
            } else {
                rank1_error = "rank1 unexpectedly observed committed generation";
            }
        }
        unmapTestSharedState(state);
        ::close(fd);
        rank1_done.store(true, std::memory_order_release);
    });

    std::thread rank2_thread([&] {
        int fd = connectPeerAndReadProbe(base, 2, rank2_error);
        if (fd < 0) {
            return;
        }
        rank2_fd.store(fd, std::memory_order_release);
        rank2_ready.store(true, std::memory_order_release);
        BroadcastFrameHeader header{};
        int                  observed = 0;
        if (readAllRaw(fd, &header, sizeof(header)) != static_cast<ssize_t>(sizeof(header))
            || header.magic != kExpectedBroadcastFrameMagic || header.nbytes != sizeof(observed)
            || readAllRaw(fd, &observed, sizeof(observed)) != static_cast<ssize_t>(sizeof(observed))
            || observed != 17) {
            rank2_error = "rank2 payload read failed";
        }
        // Drop the connection after receiving the payload but before sending
        // the ready token. Rank 1 is already waiting on the shared decision,
        // so root must publish one abort visible to every surviving rank.
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
        rank2_closed.store(true, std::memory_order_release);
    });

    bcast.initialize(0, 3, base);

    const auto ready_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((!rank1_ready.load(std::memory_order_acquire) || !rank2_ready.load(std::memory_order_acquire))
           && std::chrono::steady_clock::now() < ready_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (!rank1_ready.load(std::memory_order_acquire) || !rank2_ready.load(std::memory_order_acquire)) {
        int fd = rank1_fd.load(std::memory_order_acquire);
        if (fd >= 0) {
            ::shutdown(fd, SHUT_RDWR);
        }
        fd = rank2_fd.load(std::memory_order_acquire);
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
    EXPECT_TRUE(rank1_saw_abort.load(std::memory_order_acquire));

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
