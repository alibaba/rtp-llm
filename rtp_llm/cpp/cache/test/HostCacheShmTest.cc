#include "rtp_llm/cpp/cache/HostCacheShm.h"

#include <gtest/gtest.h>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rtp_llm {
namespace {

class EnvGuard {
public:
    EnvGuard(const char* name, const char* value): name_(name), present_(std::getenv(name) != nullptr) {
        if (present_) {
            previous_ = std::getenv(name);
        }
        setenv(name, value, 1);
    }
    ~EnvGuard() {
        if (present_) {
            setenv(name_.c_str(), previous_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

private:
    std::string name_;
    bool        present_;
    std::string previous_;
};

TEST(HostCacheShmTest, RequiresBothScrAndExplicitOptIn) {
    for (const auto* scr : {"0", "1"}) {
        for (const auto* shm : {"0", "1"}) {
            EnvGuard scr_guard("RTPLLM_ENABLE_SCR", scr);
            EnvGuard shm_guard("RTP_LLM_SCR_HOST_CACHE_SHM", shm);
            EXPECT_EQ(useScrHostCacheShm(), std::string(scr) == "1" && std::string(shm) == "1");
        }
    }
    EnvGuard scr_guard("RTPLLM_ENABLE_SCR", " TRUE ");
    EnvGuard shm_guard("RTP_LLM_SCR_HOST_CACHE_SHM", " on ");
    EXPECT_TRUE(useScrHostCacheShm());
}

TEST(HostCacheShmTest, NamedZeroFilledSharedMappingAndCleanup) {
    std::string      path;
    constexpr size_t size = 64 * 1024;
    {
        HostCacheShm arena(size);
        path = arena.path();
        struct stat st{};
        ASSERT_EQ(stat(path.c_str(), &st), 0);
        EXPECT_EQ(st.st_size, static_cast<off_t>(size));
        EXPECT_EQ(st.st_mode & 0777, static_cast<mode_t>(0600));
        EXPECT_EQ(st.st_nlink, static_cast<nlink_t>(1));
        auto* data = static_cast<unsigned char*>(arena.data());
        for (size_t i = 0; i < size; ++i) {
            ASSERT_EQ(data[i], 0) << "offset=" << i;
        }
        int fd = shm_open(arena.name().c_str(), O_RDWR, 0);
        ASSERT_GE(fd, 0);
        auto* other = static_cast<unsigned char*>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        close(fd);
        ASSERT_NE(other, MAP_FAILED);
        data[0] = 0x5a;
        EXPECT_EQ(other[0], 0x5a);
        other[size - 1] = 0x3c;
        EXPECT_EQ(data[size - 1], 0x3c);
        EXPECT_EQ(munmap(other, size), 0);
        // No early shm_unlink: CRIU must be able to reopen the original name.
        ASSERT_EQ(stat(path.c_str(), &st), 0);
        EXPECT_EQ(st.st_nlink, static_cast<nlink_t>(1));
    }
    EXPECT_EQ(access(path.c_str(), F_OK), -1);
    EXPECT_EQ(errno, ENOENT);
}

TEST(HostCacheShmTest, IndependentArenasAndSharedOwnership) {
    auto         first    = std::make_shared<HostCacheShm>(4096);
    auto         retained = first;
    HostCacheShm second(4096);
    EXPECT_NE(first->name(), second.name());
    std::memset(first->data(), 0x5a, 4096);
    EXPECT_EQ(static_cast<unsigned char*>(second.data())[0], 0);
    const auto path = first->path();
    first.reset();
    EXPECT_EQ(access(path.c_str(), F_OK), 0);
    retained.reset();
    EXPECT_EQ(access(path.c_str(), F_OK), -1);
}

TEST(HostCacheShmTest, InvalidSizeAndAllocationFailureDoNotLeakFiles) {
    EXPECT_THROW(HostCacheShm arena(0), std::invalid_argument);
    EXPECT_THROW(HostCacheShm arena(std::numeric_limits<size_t>::max()), std::invalid_argument);
    auto count = []() {
        const auto prefix = "rtpllm-scr-host-kv-" + std::to_string(geteuid()) + "-" + std::to_string(getpid()) + "-";
        auto*      dir    = opendir("/dev/shm");
        if (dir == nullptr) {
            throw std::runtime_error("opendir /dev/shm failed");
        }
        size_t result = 0;
        while (auto* entry = readdir(dir)) {
            result += std::string(entry->d_name).find(prefix) == 0;
        }
        closedir(dir);
        return result;
    };
    const auto before = count();
    // Valid off_t, but impossible tmpfs capacity: fails before populating pages.
    EXPECT_THROW(HostCacheShm arena(static_cast<size_t>(std::numeric_limits<off_t>::max())), std::runtime_error);
    EXPECT_EQ(count(), before);
}

}  // namespace
}  // namespace rtp_llm
