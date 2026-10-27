#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"

#include "gtest/gtest.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>
#include <signal.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rtp_llm {
namespace {

std::string makeTempBase() {
    std::string       pattern = "/tmp/cpu_tp_broadcaster_test.XXXXXX";
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

        struct sockaddr_un addr{};
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

int fakeServerWrongProbe(const std::string& base) {
    const std::string path = socketPath(base);
    ::unlink(path.c_str());

    int listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::fprintf(stderr, "fake server socket failed: %s\n", std::strerror(errno));
        return 1;
    }

    struct sockaddr_un addr{};
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
    auto& bcast = CpuTpBroadcaster::instance();
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

int mismatchedPayloadSizeChild(int rank, const std::string& base) {
    auto& bcast = CpuTpBroadcaster::instance();
    bcast.reset();
    bcast.initialize(rank, 2, base);

    if (rank == 0) {
        uint32_t value = 7;
        bcast.broadcast(&value, sizeof(value), 0);
        bcast.reset();
        return 0;
    }

    uint64_t  value = 0;
    const int rc    = expectThrowContains([&] { bcast.broadcast(&value, sizeof(value), 0); }, "payload size mismatch");
    bcast.reset();
    return rc;
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

TEST(CpuTpBroadcasterTest, BroadcastHappyPathTp2) {
    runHappyPath(2);
}

TEST(CpuTpBroadcasterTest, BroadcastHappyPathTp4) {
    runHappyPath(4);
}

TEST(CpuTpBroadcasterTest, RejectsPayloadSizeMismatchBeforeReadingPayload) {
    const std::string  base = makeTempBase();
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] { return mismatchedPayloadSizeChild(0, base); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pids.push_back(spawnChild([=] { return mismatchedPayloadSizeChild(1, base); }));
    expectChildrenOk(pids);
    cleanupTempBase(base, 2);
}

TEST(CpuTpBroadcasterTest, Rank0RejectsBadPeerRank) {
    const std::string  base = makeTempBase();
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] {
        auto& bcast = CpuTpBroadcaster::instance();
        bcast.reset();
        return expectThrowContains([&] { bcast.initialize(0, 2, base); }, "bad peer_rank");
    }));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pids.push_back(spawnChild([=] { return fakePeerSendRank(base, 9); }));
    expectChildrenOk(pids);
    cleanupTempBase(base);
}

TEST(CpuTpBroadcasterTest, Rank0RejectsDuplicatePeerRank) {
    const std::string  base = makeTempBase();
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] {
        auto& bcast = CpuTpBroadcaster::instance();
        bcast.reset();
        return expectThrowContains([&] { bcast.initialize(0, 3, base); }, "duplicate peer rank");
    }));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pids.push_back(spawnChild([=] { return fakePeerSendRank(base, 1); }));
    pids.push_back(spawnChild([=] { return fakePeerSendRank(base, 1); }));
    expectChildrenOk(pids);
    cleanupTempBase(base);
}

TEST(CpuTpBroadcasterTest, NonRootRejectsBadLinkProbe) {
    const std::string  base = makeTempBase();
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] { return fakeServerWrongProbe(base); }));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    pids.push_back(spawnChild([=] {
        auto& bcast = CpuTpBroadcaster::instance();
        bcast.reset();
        return expectThrowContains([&] { bcast.initialize(1, 2, base); }, "link probe read failed");
    }));
    expectChildrenOk(pids);
    cleanupTempBase(base);
}

TEST(CpuTpBroadcasterTest, ResetAllowsNewBasePath) {
    auto& bcast = CpuTpBroadcaster::instance();
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

// Candidate additions for CpuTpBroadcasterTest.cc. Not applied or executed.
// Wire-only helpers deliberately do not include the implementation header:
// malformed frames must exercise the real public receiver validation path.
struct TestBroadcastFrame {
    uint32_t magic         = 0x52545042;
    uint16_t version       = 1;
    uint16_t header_bytes  = 24;
    uint64_t sequence      = 0;
    uint64_t payload_bytes = 0;
};
static_assert(sizeof(TestBroadcastFrame) == 24, "test must match framed UDS ABI");

// Socket send() cannot write a pipe (ENOTSOCK); use write() for test rendezvous.
ssize_t writePipeByte(int fd, char value) {
    ssize_t count;
    do {
        count = ::write(fd, &value, 1);
    } while (count < 0 && errno == EINTR);
    return count;
}

int expectTypedShutdown(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const CpuTpShutdown&) {
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "expected CpuTpShutdown, got: %s\n", error.what());
        return 1;
    }
    std::fprintf(stderr, "expected CpuTpShutdown, operation returned normally\n");
    return 1;
}

int expectOrdinaryFailure(const std::function<void()>& fn, const std::string& needle = "") {
    try {
        fn();
    } catch (const CpuTpShutdown&) {
        std::fprintf(stderr, "protocol failure must not become graceful shutdown\n");
        return 1;
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(needle) != std::string::npos) {
            return 0;
        }
        std::fprintf(stderr, "ordinary error did not contain %s: %s\n", needle.c_str(), error.what());
        return 1;
    }
    std::fprintf(stderr, "expected ordinary protocol error, operation returned normally\n");
    return 1;
}

int rawServer(const std::string& base, const std::function<int(int)>& exchange) {
    const auto path = socketPath(base);
    ::unlink(path.c_str());
    int listener = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (listener < 0) {
        return 1;
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);
    if (::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 || ::listen(listener, 1) != 0) {
        ::close(listener);
        return 1;
    }
    const int peer = ::accept(listener, nullptr, nullptr);
    ::close(listener);
    if (peer < 0) {
        return 1;
    }
    int        rank  = -1;
    const char probe = 0x5a;
    int        rc    = 1;
    if (readAllRaw(peer, &rank, sizeof(rank)) == static_cast<ssize_t>(sizeof(rank)) && rank == 1
        && writeAllRaw(peer, &probe, sizeof(probe)) == static_cast<ssize_t>(sizeof(probe))) {
        rc = exchange(peer);
    }
    ::close(peer);
    ::unlink(path.c_str());
    return rc;
}

void runTerminalRanks(int tp_size, bool reset_session = false, bool claims = false) {
    const std::string  first  = makeTempBase();
    const std::string  second = reset_session ? makeTempBase() : "";
    std::vector<pid_t> pids;
    for (int rank = 0; rank < tp_size; ++rank) {
        pids.push_back(spawnChild([=] {
            auto& bcast = CpuTpBroadcaster::instance();
            bcast.reset();
            const int sessions = reset_session ? 2 : 1;
            for (int session = 0; session < sessions; ++session) {
                const auto& base = session == 0 ? first : second;
                bcast.initialize(rank, tp_size, base);
                if (claims) {
                    if (expectOrdinaryFailure([&] { bcast.claimEngineSession((rank + 1) % tp_size, tp_size); })
                        || expectOrdinaryFailure([&] { bcast.claimEngineSession(rank, tp_size + 1); })) {
                        return 1;
                    }
                    bcast.claimEngineSession(rank, tp_size);
                    if (expectOrdinaryFailure([&] { bcast.claimEngineSession(rank, tp_size); })) {
                        return 1;
                    }
                }
                uint32_t value = rank == 0 ? 0x1234u + session : 0;
                bcast.broadcast(&value, sizeof(value), 0);
                if (value != 0x1234u + session) {
                    return 1;
                }
                char empty_sentinel = 0x17;
                bcast.broadcast(&empty_sentinel, 0, 0);
                if (empty_sentinel != 0x17) {
                    return 1;
                }
                std::array<unsigned char, 16> bytes{};
                if (rank == 0) {
                    bytes.fill(static_cast<unsigned char>(session + 7));
                }
                bcast.broadcast(bytes.data(), bytes.size(), 0);
                for (const auto byte : bytes) {
                    if (byte != session + 7) {
                        return 1;
                    }
                }
                if (rank == 0) {
                    bcast.sendShutdown();
                    bcast.sendShutdown();  // Public operation is idempotent.
                } else {
                    const auto before = bytes;
                    if (expectTypedShutdown([&] { bcast.broadcast(bytes.data(), bytes.size(), 0); })
                        || bytes != before) {
                        return 1;
                    }
                }
                // A terminated same-path initialize must not silently revive it.
                if (expectOrdinaryFailure([&] { bcast.initialize(rank, tp_size, base); })) {
                    return 1;
                }
                bcast.reset();
                if (bcast.isInitialized()) {
                    return 1;
                }
            }
            return 0;
        }));
    }
    expectChildrenOk(pids);
    cleanupTempBase(first, tp_size);
    if (reset_session) {
        cleanupTempBase(second, tp_size);
    }
}

TEST(CpuTpBroadcasterTest, DataAndZeroLengthFramesPrecedeShutdownTp2) {
    runTerminalRanks(2);
}

TEST(CpuTpBroadcasterTest, DataAndZeroLengthFramesPrecedeShutdownTp4) {
    runTerminalRanks(4);
}

TEST(CpuTpBroadcasterTest, ShutdownThenResetStartsFreshSequenceAndSessionTp2) {
    runTerminalRanks(2, true);
}

TEST(CpuTpBroadcasterTest, EngineClaimRejectsMismatchAndDuplicateThenResetClearsClaim) {
    runTerminalRanks(2, true, true);
}

TEST(CpuTpBroadcasterTest, ShutdownWakesBlockedReceiveWithoutModifyingOutput) {
    const auto         base = makeTempBase();
    std::array<int, 2> ready{};
    ASSERT_EQ(::pipe(ready.data()), 0);
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] {
        ::close(ready[1]);
        auto& bcast = CpuTpBroadcaster::instance();
        bcast.reset();
        bcast.initialize(0, 2, base);
        char       entered = 0;
        const auto count   = readAllRaw(ready[0], &entered, 1);
        ::close(ready[0]);
        if (count != 1 || entered != 'B') {
            return 1;
        }
        bcast.sendShutdown();
        bcast.reset();
        return 0;
    }));
    pids.push_back(spawnChild([=] {
        ::close(ready[0]);
        auto& bcast = CpuTpBroadcaster::instance();
        bcast.reset();
        bcast.initialize(1, 2, base);
        uint64_t           value = 0xdeadbeef;
        std::promise<void> entered;
        auto               entering = entered.get_future();
        auto               result   = std::async(std::launch::async, [&] {
            entered.set_value();
            return expectTypedShutdown([&] { bcast.broadcast(&value, sizeof(value), 0); });
        });
        entering.wait();
        if (result.wait_for(std::chrono::milliseconds(100)) != std::future_status::timeout) {
            return 1;
        }
        const char blocked = 'B';
        if (writePipeByte(ready[1], blocked) != 1) {
            return 1;
        }
        ::close(ready[1]);
        if (result.wait_for(std::chrono::seconds(5)) != std::future_status::ready || result.get() != 0
            || value != 0xdeadbeef) {
            return 1;
        }
        bcast.reset();
        return 0;
    }));
    ::close(ready[0]);
    ::close(ready[1]);
    expectChildrenOk(pids);
    cleanupTempBase(base);
}

TEST(CpuTpBroadcasterTest, RepeatedShutdownWritesExactlyOneOrderedTerminalFrame) {
    const auto         base = makeTempBase();
    std::array<int, 2> checked{};
    ASSERT_EQ(::pipe(checked.data()), 0);
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] {
        ::close(checked[1]);
        auto& bcast = CpuTpBroadcaster::instance();
        bcast.reset();
        bcast.initialize(0, 2, base);
        bcast.broadcast(nullptr, 0, 0);
        bcast.sendShutdown();
        bcast.sendShutdown();
        char       done  = 0;
        const auto count = readAllRaw(checked[0], &done, 1);
        ::close(checked[0]);
        bcast.reset();
        return count == 1 && done == 'D' ? 0 : 1;
    }));
    pids.push_back(spawnChild([=] {
        ::close(checked[0]);
        const int fd = connectWithRetry(socketPath(base));
        if (fd < 0) {
            return 1;
        }
        const int          rank  = 1;
        char               probe = 0;
        TestBroadcastFrame normal{}, terminal{};
        if (writeAllRaw(fd, &rank, sizeof(rank)) != static_cast<ssize_t>(sizeof(rank)) || readAllRaw(fd, &probe, 1) != 1
            || probe != 0x5a || readAllRaw(fd, &normal, sizeof(normal)) != static_cast<ssize_t>(sizeof(normal))
            || normal.magic != 0x52545042 || normal.version != 1 || normal.header_bytes != sizeof(normal)
            || normal.sequence != 0 || normal.payload_bytes != 0
            || readAllRaw(fd, &terminal, sizeof(terminal)) != static_cast<ssize_t>(sizeof(terminal))
            || terminal.magic != 0x52545042 || terminal.version != 1 || terminal.header_bytes != sizeof(terminal)
            || terminal.sequence != 1 || terminal.payload_bytes != std::numeric_limits<uint64_t>::max()) {
            return 1;
        }
        pollfd pending{};
        pending.fd     = fd;
        pending.events = POLLIN;
        int available;
        do {
            available = ::poll(&pending, 1, 100);
        } while (available < 0 && errno == EINTR);
        if (available != 0) {
            return 1;  // Even an extra terminal byte or early EOF is a failure.
        }
        const char done = 'D';
        const int  rc   = writePipeByte(checked[1], done) == 1 ? 0 : 1;
        ::close(checked[1]);
        ::close(fd);
        return rc;
    }));
    ::close(checked[0]);
    ::close(checked[1]);
    expectChildrenOk(pids);
    cleanupTempBase(base);
}

void runForgedTerminal(TestBroadcastFrame terminal,
                       const std::string& error,
                       bool               preceding_zero = false,
                       std::size_t        wire_bytes     = sizeof(TestBroadcastFrame)) {
    const auto         base = makeTempBase();
    std::vector<pid_t> pids;
    pids.push_back(spawnChild([=] {
        return rawServer(base, [&](int fd) {
            if (preceding_zero) {
                const TestBroadcastFrame first{};
                if (writeAllRaw(fd, &first, sizeof(first)) != static_cast<ssize_t>(sizeof(first))) {
                    return 1;
                }
            }
            return writeAllRaw(fd, &terminal, wire_bytes) == static_cast<ssize_t>(wire_bytes) ? 0 : 1;
        });
    }));
    pids.push_back(spawnChild([=] {
        auto& bcast = CpuTpBroadcaster::instance();
        bcast.reset();
        bcast.initialize(1, 2, base);
        if (preceding_zero) {
            bcast.broadcast(nullptr, 0, 0);
        }
        uint64_t  value = 0xabcdef;
        const int rc    = expectOrdinaryFailure([&] { bcast.broadcast(&value, sizeof(value), 0); }, error);
        bcast.reset();
        return rc == 0 && value == 0xabcdef ? 0 : 1;
    }));
    expectChildrenOk(pids);
    cleanupTempBase(base);
}

TEST(CpuTpBroadcasterTest, OrdinaryEofIsNotShutdown) {
    TestBroadcastFrame frame{};
    runForgedTerminal(frame, "frame-header read", false, 0);
}

TEST(CpuTpBroadcasterTest, PartialTerminalHeaderIsNotShutdown) {
    TestBroadcastFrame frame{};
    frame.payload_bytes = std::numeric_limits<uint64_t>::max();
    runForgedTerminal(frame, "frame-header read", false, sizeof(frame) - 1);
}

TEST(CpuTpBroadcasterTest, TerminalStillValidatesMagic) {
    TestBroadcastFrame frame{};
    frame.magic ^= 1;
    frame.payload_bytes = std::numeric_limits<uint64_t>::max();
    runForgedTerminal(frame, "invalid frame header");
}

TEST(CpuTpBroadcasterTest, TerminalStillValidatesVersion) {
    TestBroadcastFrame frame{};
    ++frame.version;
    frame.payload_bytes = std::numeric_limits<uint64_t>::max();
    runForgedTerminal(frame, "invalid frame header");
}

TEST(CpuTpBroadcasterTest, TerminalStillValidatesHeaderSize) {
    TestBroadcastFrame frame{};
    --frame.header_bytes;
    frame.payload_bytes = std::numeric_limits<uint64_t>::max();
    runForgedTerminal(frame, "invalid frame header");
}

TEST(CpuTpBroadcasterTest, TerminalStillValidatesExpectedSequence) {
    TestBroadcastFrame frame{};
    frame.sequence      = 1;
    frame.payload_bytes = std::numeric_limits<uint64_t>::max();
    runForgedTerminal(frame, "sequence mismatch");
}

TEST(CpuTpBroadcasterTest, TerminalCannotReplayEarlierSequence) {
    TestBroadcastFrame frame{};
    frame.sequence      = 0;
    frame.payload_bytes = std::numeric_limits<uint64_t>::max();
    runForgedTerminal(frame, "sequence mismatch", true);
}

TEST(CpuTpBroadcasterTest, ShutdownIsRootOnlyAndSentinelCannotBeData) {
    static_assert(sizeof(std::size_t) == sizeof(uint64_t), "target requires 64-bit size_t");
    const auto         base = makeTempBase();
    std::vector<pid_t> pids;
    for (int rank = 0; rank < 2; ++rank) {
        pids.push_back(spawnChild([=] {
            auto& bcast = CpuTpBroadcaster::instance();
            bcast.reset();
            bcast.initialize(rank, 2, base);
            if (rank == 0) {
                if (expectOrdinaryFailure(
                        [&] { bcast.broadcast(nullptr, std::numeric_limits<std::size_t>::max(), 0); })) {
                    return 1;
                }
            } else if (expectOrdinaryFailure([&] { bcast.sendShutdown(); })) {
                return 1;
            }
            uint32_t value = rank == 0 ? 27 : 0;
            bcast.broadcast(&value, sizeof(value), 0);
            if (value != 27) {
                return 1;
            }
            int rc = 0;
            if (rank == 0) {
                bcast.sendShutdown();
            } else {
                rc = expectTypedShutdown([&] { bcast.broadcast(&value, sizeof(value), 0); });
            }
            bcast.reset();
            return rc;
        }));
    }
    expectChildrenOk(pids);
    cleanupTempBase(base);
}

}  // namespace
}  // namespace rtp_llm
