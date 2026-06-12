#include "rtp_llm/cpp/distribute/CpuTpBroadcaster.h"

#include "gtest/gtest.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rtp_llm {
namespace {

std::string makeTempBase() {
    std::string pattern = "/tmp/cpu_tp_broadcaster_test.XXXXXX";
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
        int status = 0;
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

    std::array<char, 16> payload {};
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
    const std::string base = makeTempBase();
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

TEST(CpuTpBroadcasterTest, Rank0RejectsBadPeerRank) {
    const std::string base = makeTempBase();
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
    const std::string base = makeTempBase();
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
    const std::string base = makeTempBase();
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

}  // namespace
}  // namespace rtp_llm
