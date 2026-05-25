#include "gtest/gtest.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/connector/memory/DiskBlockPool.h"

namespace rtp_llm::test {
namespace {

class FakeDiskBlockIO: public IDiskBlockIO {
public:
    bool openAndPreallocate(const std::string& file_path, size_t bytes, bool buffered_io) override {
        file_path_   = file_path;
        bytes_       = bytes;
        buffered_io_ = buffered_io;
        opened_      = true;
        return true;
    }

    bool read(uint64_t offset, void* dst, size_t bytes) override {
        read_calls++;
        if (fail_read || !opened_ || dst == nullptr || offset + bytes > bytes_) {
            return false;
        }
        return true;
    }

    bool write(uint64_t offset, const void* src, size_t bytes) override {
        write_calls++;
        if (fail_write || !opened_ || src == nullptr || offset + bytes > bytes_) {
            return false;
        }
        return true;
    }

    void close() override {
        opened_ = false;
    }

    std::string debugString() const override {
        return "FakeDiskBlockIO";
    }

    bool fail_read{false};
    bool fail_write{false};
    int  read_calls{0};
    int  write_calls{0};

private:
    std::string file_path_;
    size_t      bytes_{0};
    bool        buffered_io_{true};
    bool        opened_{false};
};

class TempDir {
public:
    TempDir() {
        char tmpl[] = "/tmp/rtp_disk_pool_test_XXXXXX";
        auto path   = ::mkdtemp(tmpl);
        EXPECT_NE(path, nullptr);
        if (path != nullptr) {
            path_ = path;
        }
    }
    ~TempDir() {
        if (path_.empty()) {
            return;
        }
        const auto work_dir = path_ + "/rtp_llm_disk_kv";
        if (auto* dir = ::opendir(work_dir.c_str())) {
            while (auto* entry = ::readdir(dir)) {
                const std::string name(entry->d_name);
                if (name == "." || name == "..") {
                    continue;
                }
                ::unlink((work_dir + "/" + name).c_str());
            }
            ::closedir(dir);
        }
        ::rmdir(work_dir.c_str());
        ::rmdir(path_.c_str());
    }
    const std::string& path() const {
        return path_;
    }

private:
    std::string path_;
};

DiskBlockPoolConfig makeConfig(const std::string& path, size_t disk_size_bytes = 3 * 4096) {
    DiskBlockPoolConfig config;
    config.work_dir         = path;
    config.local_rank       = 0;
    config.world_rank       = 0;
    config.disk_size_bytes  = disk_size_bytes;
    config.block_size_bytes = 1024;
    config.buffered_io      = true;
    config.pool_kind        = CacheBlockKind::COMPLETE;
    return config;
}

}  // namespace

TEST(DiskBlockPoolTest, InitPreallocatesFileAndCleansStaleFiles) {
    TempDir temp_dir;
    ASSERT_FALSE(temp_dir.path().empty());

    const auto work_dir = temp_dir.path() + "/rtp_llm_disk_kv";
    ASSERT_EQ(::mkdir(work_dir.c_str(), 0755), 0);
    const auto stale = work_dir + "/rank_stale.kv";
    int        fd    = ::open(stale.c_str(), O_CREAT | O_WRONLY, 0600);
    ASSERT_GE(fd, 0);
    ::close(fd);
    ASSERT_EQ(::access(stale.c_str(), F_OK), 0);

    DiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));

    DiskBlockPool pool(makeConfig(guard.workDir()));
    ASSERT_TRUE(pool.init());
    EXPECT_EQ(::access(stale.c_str(), F_OK), -1);
    EXPECT_EQ(::access(pool.filePath().c_str(), F_OK), 0);
    EXPECT_NE(pool.filePath().find("rank_0_world_0_complete.kv"), std::string::npos);
    EXPECT_EQ(pool.totalSlots(), 3u);
    EXPECT_EQ(pool.freeSlots(), 3u);
}

TEST(DiskBlockPoolTest, InitFailsWhenMountPathDoesNotExist) {
    TempDir temp_dir;
    ASSERT_FALSE(temp_dir.path().empty());

    DiskMountGuard guard;
    EXPECT_FALSE(guard.init(temp_dir.path() + "/missing_mount"));
}

TEST(DiskBlockPoolTest, MountGuardAllowsTwoPoolsOnSameMountWithoutDeletingFirst) {
    TempDir temp_dir;
    ASSERT_FALSE(temp_dir.path().empty());

    DiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));

    DiskBlockPool complete_pool(makeConfig(guard.workDir()));
    ASSERT_TRUE(complete_pool.init());
    ASSERT_EQ(::access(complete_pool.filePath().c_str(), F_OK), 0);

    auto incomplete_cfg       = makeConfig(guard.workDir(), 6 * 4096);
    incomplete_cfg.pool_kind  = CacheBlockKind::INCOMPLETE;
    incomplete_cfg.local_rank = 0;
    incomplete_cfg.world_rank = 0;
    DiskBlockPool incomplete_pool(incomplete_cfg);
    ASSERT_TRUE(incomplete_pool.init());

    EXPECT_EQ(::access(complete_pool.filePath().c_str(), F_OK), 0);
    EXPECT_EQ(::access(incomplete_pool.filePath().c_str(), F_OK), 0);
    EXPECT_NE(complete_pool.filePath(), incomplete_pool.filePath());
}

TEST(DiskBlockPoolTest, ReserveCommitAbortAndFreeSlots) {
    TempDir       temp_dir;
    DiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));
    DiskBlockPool pool(makeConfig(guard.workDir()));
    ASSERT_TRUE(pool.init());

    auto slot = pool.malloc();
    ASSERT_TRUE(slot.has_value());
    EXPECT_EQ(pool.freeSlots(), 2u);

    pool.blockCacheReference(*slot);
    pool.requestFree(*slot);
    EXPECT_EQ(pool.freeSlots(), 2u);
    EXPECT_EQ(pool.availableSlots(), 3u);

    pool.blockCacheFree(*slot);
    EXPECT_EQ(pool.freeSlots(), 3u);
}

TEST(DiskBlockPoolTest, RequestRefPreventsReuseUntilReleased) {
    TempDir       temp_dir;
    DiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));
    DiskBlockPool pool(makeConfig(guard.workDir()));
    ASSERT_TRUE(pool.init());

    auto slot = pool.malloc();
    ASSERT_TRUE(slot.has_value());
    pool.blockCacheReference(*slot);
    pool.requestReference(*slot);

    pool.blockCacheFree(*slot);
    pool.requestFree(*slot);
    EXPECT_EQ(pool.freeSlots(), 2u);

    pool.requestFree(*slot);
    EXPECT_EQ(pool.freeSlots(), 3u);
}

TEST(DiskBlockPoolTest, ReadWriteFullSlot) {
    TempDir       temp_dir;
    DiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));
    DiskBlockPool pool(makeConfig(guard.workDir()));
    ASSERT_TRUE(pool.init());

    auto slot = pool.malloc();
    ASSERT_TRUE(slot.has_value());
    std::vector<unsigned char> write_buf(pool.slotStrideBytes(), 0x5a);
    std::vector<unsigned char> read_buf(pool.slotStrideBytes(), 0);

    ASSERT_TRUE(pool.write(*slot, write_buf.data(), write_buf.size()));
    ASSERT_TRUE(pool.read(*slot, read_buf.data(), read_buf.size()));
    EXPECT_EQ(read_buf, write_buf);
    EXPECT_EQ(pool.writeBytes(), write_buf.size());
    EXPECT_EQ(pool.readBytes(), read_buf.size());
}

TEST(DiskBlockPoolTest, InjectedIOFailureDoesNotChangeRefsOrBytes) {
    TempDir       temp_dir;
    DiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));

    auto* fake_io = new FakeDiskBlockIO();
    DiskBlockPool pool(makeConfig(guard.workDir(), 2 * 4096), std::unique_ptr<IDiskBlockIO>(fake_io));
    ASSERT_TRUE(pool.init());

    auto slot = pool.malloc();
    ASSERT_TRUE(slot.has_value());
    EXPECT_EQ(pool.freeSlots(), 1u);

    std::vector<unsigned char> buf(pool.slotStrideBytes(), 0x5a);
    fake_io->fail_write = true;
    EXPECT_FALSE(pool.write(*slot, buf.data(), buf.size()));
    EXPECT_EQ(pool.writeBytes(), 0u);
    EXPECT_EQ(pool.freeSlots(), 1u);

    fake_io->fail_read = true;
    EXPECT_FALSE(pool.read(*slot, buf.data(), buf.size()));
    EXPECT_EQ(pool.readBytes(), 0u);
    EXPECT_EQ(pool.freeSlots(), 1u);

    pool.requestFree(*slot);
    EXPECT_EQ(pool.freeSlots(), 2u);
    EXPECT_EQ(fake_io->write_calls, 1);
    EXPECT_EQ(fake_io->read_calls, 1);
}

TEST(DiskBlockPoolTest, InjectedIOFailurePreservesCacheRefUntilExplicitFree) {
    TempDir       temp_dir;
    DiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));

    auto* fake_io = new FakeDiskBlockIO();
    DiskBlockPool pool(makeConfig(guard.workDir(), 2 * 4096), std::unique_ptr<IDiskBlockIO>(fake_io));
    ASSERT_TRUE(pool.init());

    auto slot = pool.malloc();
    ASSERT_TRUE(slot.has_value());
    pool.blockCacheReference(*slot);
    pool.requestFree(*slot);
    EXPECT_EQ(pool.freeSlots(), 1u);

    std::vector<unsigned char> buf(pool.slotStrideBytes(), 0x6b);
    fake_io->fail_write = true;
    EXPECT_FALSE(pool.write(*slot, buf.data(), buf.size()));
    EXPECT_EQ(pool.writeBytes(), 0u);
    EXPECT_EQ(pool.freeSlots(), 1u);

    pool.blockCacheFree(*slot);
    EXPECT_EQ(pool.freeSlots(), 2u);
}

TEST(DiskBlockPoolTest, DISABLED_PerfWriteReadBandwidth) {
    const char* mount_path = std::getenv("DISK_BLOCK_POOL_PERF_PATH");
    if (mount_path == nullptr || mount_path[0] == '\0') {
        GTEST_SKIP() << "set DISK_BLOCK_POOL_PERF_PATH to run disk block pool perf test";
    }
    const size_t perf_mb =
        std::getenv("DISK_BLOCK_POOL_PERF_MB") ? std::strtoull(std::getenv("DISK_BLOCK_POOL_PERF_MB"), nullptr, 10)
                                               : 256;
    const size_t block_mb =
        std::getenv("DISK_BLOCK_POOL_PERF_BLOCK_MB") ?
            std::strtoull(std::getenv("DISK_BLOCK_POOL_PERF_BLOCK_MB"), nullptr, 10) :
            4;
    ASSERT_GT(perf_mb, 0u);
    ASSERT_GT(block_mb, 0u);

    DiskMountGuard guard;
    ASSERT_TRUE(guard.init(mount_path));
    auto config              = makeConfig(guard.workDir(), perf_mb * 1024ULL * 1024ULL);
    config.block_size_bytes  = block_mb * 1024ULL * 1024ULL;
    config.buffered_io       = true;
    DiskBlockPool pool(config);
    ASSERT_TRUE(pool.init());

    std::vector<int32_t> slots;
    while (auto slot = pool.malloc()) {
        slots.push_back(*slot);
    }
    ASSERT_FALSE(slots.empty());
    std::vector<unsigned char> write_buf(pool.slotStrideBytes(), 0x5a);
    std::vector<unsigned char> read_buf(pool.slotStrideBytes(), 0);

    const auto write_start = std::chrono::steady_clock::now();
    for (const auto slot : slots) {
        ASSERT_TRUE(pool.write(slot, write_buf.data(), write_buf.size()));
    }
    const auto write_end = std::chrono::steady_clock::now();
    const auto read_start = std::chrono::steady_clock::now();
    for (const auto slot : slots) {
        ASSERT_TRUE(pool.read(slot, read_buf.data(), read_buf.size()));
    }
    const auto read_end = std::chrono::steady_clock::now();

    const auto write_us = std::chrono::duration_cast<std::chrono::microseconds>(write_end - write_start).count();
    const auto read_us  = std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count();
    const double write_gib = static_cast<double>(pool.writeBytes()) / 1024.0 / 1024.0 / 1024.0;
    const double read_gib  = static_cast<double>(pool.readBytes()) / 1024.0 / 1024.0 / 1024.0;
    fprintf(stderr,
            "DISK_BLOCK_POOL_PERF slots=%zu slot_stride=%zu write_g=%.3f write_sec=%.3f write_gibps=%.3f "
            "read_g=%.3f read_sec=%.3f read_gibps=%.3f\n",
            slots.size(),
            pool.slotStrideBytes(),
            write_gib,
            static_cast<double>(write_us) / 1000000.0,
            write_gib / (static_cast<double>(write_us) / 1000000.0),
            read_gib,
            static_cast<double>(read_us) / 1000000.0,
            read_gib / (static_cast<double>(read_us) / 1000000.0));

    for (const auto slot : slots) {
        pool.requestFree(slot);
    }
}

TEST(DiskBlockPoolTest, FullPoolReturnsNullopt) {
    TempDir       temp_dir;
    DiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));
    DiskBlockPool pool(makeConfig(guard.workDir(), 2 * 4096));
    ASSERT_TRUE(pool.init());
    ASSERT_TRUE(pool.malloc().has_value());
    ASSERT_TRUE(pool.malloc().has_value());
    EXPECT_FALSE(pool.malloc().has_value());
}

}  // namespace rtp_llm::test
