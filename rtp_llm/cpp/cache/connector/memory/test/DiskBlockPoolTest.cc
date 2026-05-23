#include "gtest/gtest.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/connector/memory/DiskBlockPool.h"

namespace rtp_llm::test {
namespace {

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
        ::unlink((work_dir + "/rank_0_world_0.kv").c_str());
        ::unlink((work_dir + "/rank_stale.kv").c_str());
        ::unlink((work_dir + "/.lock").c_str());
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
    config.mount_path       = path;
    config.local_rank       = 0;
    config.world_rank       = 0;
    config.disk_size_bytes  = disk_size_bytes;
    config.block_size_bytes = 1024;
    config.buffered_io      = true;
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

    DiskBlockPool pool(makeConfig(temp_dir.path()));
    ASSERT_TRUE(pool.init());
    EXPECT_EQ(::access(stale.c_str(), F_OK), -1);
    EXPECT_EQ(::access(pool.filePath().c_str(), F_OK), 0);
    EXPECT_EQ(pool.totalSlots(), 3u);
    EXPECT_EQ(pool.freeSlots(), 3u);
}

TEST(DiskBlockPoolTest, InitFailsWhenMountPathDoesNotExist) {
    TempDir temp_dir;
    ASSERT_FALSE(temp_dir.path().empty());

    DiskBlockPool pool(makeConfig(temp_dir.path() + "/missing_mount"));
    EXPECT_FALSE(pool.init());
}

TEST(DiskBlockPoolTest, ReserveCommitAbortAndFreeSlots) {
    TempDir       temp_dir;
    DiskBlockPool pool(makeConfig(temp_dir.path()));
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
    DiskBlockPool pool(makeConfig(temp_dir.path()));
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
    DiskBlockPool pool(makeConfig(temp_dir.path()));
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

TEST(DiskBlockPoolTest, FullPoolReturnsNullopt) {
    TempDir       temp_dir;
    DiskBlockPool pool(makeConfig(temp_dir.path(), 2 * 4096));
    ASSERT_TRUE(pool.init());
    ASSERT_TRUE(pool.malloc().has_value());
    ASSERT_TRUE(pool.malloc().has_value());
    EXPECT_FALSE(pool.malloc().has_value());
}

}  // namespace rtp_llm::test
