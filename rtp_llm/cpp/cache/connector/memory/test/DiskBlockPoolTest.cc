#include "gtest/gtest.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/connector/memory/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryDiskBlockCache.h"

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

TEST(DiskBlockPoolTest, ConcurrentReadCompletionCannotRecycleSlotBeforeSecondReaderPinsIt) {
    TempDir        temp_dir;
    DiskMountGuard guard;
    ASSERT_TRUE(guard.init(temp_dir.path()));
    DiskBlockPool pool(makeConfig(guard.workDir(), /*disk_size_bytes=*/4096));
    ASSERT_TRUE(pool.init());
    ASSERT_EQ(pool.totalSlots(), 1u);
    MemoryDiskBlockCache            cache;
    std::array<unsigned char, 1024> original;
    std::array<unsigned char, 1024> actual;
    original.fill(0x11);
    auto slot = pool.malloc();
    ASSERT_TRUE(slot.has_value());
    ASSERT_TRUE(pool.write(*slot, original.data(), original.size()));

    MemoryDiskBlockCache::CacheItem item;
    item.cache_key    = 101;
    item.backing_type = CacheBackingType::DISK;
    item.disk_slot    = *slot;
    item.block_size   = original.size();
    pool.blockCacheReference(*slot);
    pool.requestFree(*slot);
    ASSERT_TRUE(cache.putCommitted(item).first);

    auto reader_a = cache.matchAndMarkInFlight(item.cache_key);
    pool.requestReference(reader_a.disk_slot);
    // Pause B at the exact gap between matchAndMarkInFlight and requestReference.
    auto reader_b = cache.matchAndMarkInFlight(item.cache_key);
    auto removed  = cache.removeIfMatch(item.cache_key, item.backing_type, NULL_BLOCK_IDX, *slot, reader_a.generation);
    if (removed.has_value()) {
        pool.blockCacheFree(removed->disk_slot);
    }
    pool.requestFree(reader_a.disk_slot);
    auto retired = cache.releaseInFlight(item.cache_key, item.backing_type, NULL_BLOCK_IDX, *slot, reader_a.generation);
    if (retired.has_value()) {
        pool.blockCacheFree(retired->disk_slot);
    }
    EXPECT_FALSE(cache.contains(item.cache_key));
    EXPECT_EQ(pool.freeSlots(), 0u);
    // Before the fix a writer could allocate this slot and overwrite B's source.
    EXPECT_FALSE(pool.malloc().has_value());

    pool.requestReference(reader_b.disk_slot);
    ASSERT_TRUE(pool.read(reader_b.disk_slot, actual.data(), actual.size()));
    EXPECT_EQ(actual, original);
    pool.requestFree(reader_b.disk_slot);
    retired = cache.releaseInFlight(item.cache_key, item.backing_type, NULL_BLOCK_IDX, *slot, reader_b.generation);
    ASSERT_TRUE(retired.has_value());
    pool.blockCacheFree(retired->disk_slot);
    EXPECT_EQ(pool.freeSlots(), 1u);
    // Reclamation is delayed, not leaked: a new writer can now use the slot.
    auto replacement = pool.malloc();
    ASSERT_TRUE(replacement.has_value());
    EXPECT_EQ(*replacement, *slot);
    pool.requestFree(*replacement);
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
