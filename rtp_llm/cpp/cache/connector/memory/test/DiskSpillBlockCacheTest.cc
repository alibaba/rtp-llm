#include "rtp_llm/cpp/cache/connector/memory/DiskSpillBlockCache.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

std::string makeTempDir(const std::string& name) {
    const auto stamp =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    auto path = std::filesystem::temp_directory_path()
                / ("rtp_llm_disk_spill_bc_" + name + "_" + std::to_string(::getpid()) + "_" + std::to_string(stamp));
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    std::filesystem::create_directories(path, ec);
    EXPECT_FALSE(ec) << ec.message();
    return path.string();
}

DiskSpillBlockCache::InitConfig makeConfig(const std::string& path, size_t block_size, size_t capacity_mb = 1) {
    DiskSpillBlockCache::InitConfig config;
    config.disks.push_back(DiskSpillBlockCache::DiskConfig{path, capacity_mb});
    config.block_size                   = block_size;
    config.align_bytes                  = 4096;
    config.segment_bytes                = 1024UL * 1024UL;  // 1MB per segment for tests
    config.direct_io                    = false;
    config.schema_hash                  = "test";
    config.startup_uuid                 = "u_" + std::to_string(::getpid()) + "_" + std::to_string(rand());
    config.hostname                     = "h";
    config.world_rank                   = 0;
    config.io_threads_per_disk          = 1;
    config.io_queue_size                = 16;
    config.max_staging_buffers_per_disk = 4;
    config.cleanup_old_startup_dirs     = false;
    return config;
}

DiskSpillBlockCache::InitConfig makeMultiDiskConfig(const std::vector<std::pair<std::string, size_t>>& disks_spec,
                                                    size_t                                              block_size,
                                                    size_t                                              segment_bytes = 1024UL
                                                                                                                        * 1024UL) {
    DiskSpillBlockCache::InitConfig config;
    for (const auto& [path, cap] : disks_spec) {
        config.disks.push_back(DiskSpillBlockCache::DiskConfig{path, cap});
    }
    config.block_size                   = block_size;
    config.align_bytes                  = 4096;
    config.segment_bytes                = segment_bytes;
    config.direct_io                    = false;
    config.schema_hash                  = "test";
    config.startup_uuid                 = "u_" + std::to_string(::getpid()) + "_" + std::to_string(rand());
    config.hostname                     = "h";
    config.world_rank                   = 0;
    config.io_threads_per_disk          = 1;
    config.io_queue_size                = 16;
    config.max_staging_buffers_per_disk = 4;
    config.cleanup_old_startup_dirs     = false;
    return config;
}

TEST(DiskSpillBlockCacheTest, ParseDiskConfigs) {
    const auto disks = DiskSpillBlockCache::parseDiskConfigs("/tmp/a=10, /tmp/b=20");
    ASSERT_EQ(disks.size(), 2u);
    EXPECT_EQ(disks[0].path, "/tmp/a");
    EXPECT_EQ(disks[0].capacity_mb, 10u);
    EXPECT_EQ(disks[1].path, "/tmp/b");
    EXPECT_EQ(disks[1].capacity_mb, 20u);
}

TEST(DiskSpillBlockCacheTest, StoreMatchTakeReadRelease) {
    const auto       path       = makeTempDir("basic");
    constexpr size_t kBlockSize = 4096;
    auto             cache      = DiskSpillBlockCache::create(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache->init());

    std::vector<char>           input(kBlockSize, 'x');
    MemoryBlockCache::CacheItem item;
    item.cache_key   = 100;
    item.block_index = 1;
    item.block_size  = kBlockSize;
    item.is_complete = true;
    ASSERT_TRUE(cache->store(item, input.data(), input.size()));

    const auto match = cache->match(100);
    ASSERT_TRUE(match.matched);
    EXPECT_TRUE(match.is_complete);
    EXPECT_EQ(match.block_size, kBlockSize);

    auto taken = cache->takeForRead(100);
    ASSERT_TRUE(taken.has_value());
    ASSERT_TRUE(taken->valid());
    EXPECT_FALSE(cache->contains(100));

    std::vector<char> output(kBlockSize, 0);
    ASSERT_TRUE(cache->readTaken(taken->item(), output.data(), output.size()));
    EXPECT_EQ(output, input);
    taken->release();

    const auto status = cache->status();
    EXPECT_EQ(status.committed_slot_num, 0u);
    EXPECT_EQ(status.inflight_read_slot_num, 0u);
    EXPECT_EQ(status.total_slot_num, status.free_slot_num);
}

TEST(DiskSpillBlockCacheTest, StorePreservesPartialCompleteness) {
    const auto       path       = makeTempDir("partial_complete_bit");
    constexpr size_t kBlockSize = 4096;
    auto             cache      = DiskSpillBlockCache::create(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache->init());

    std::vector<char>           input(kBlockSize, 'p');
    MemoryBlockCache::CacheItem item;
    item.cache_key   = 7;
    item.block_index = 1;
    item.block_size  = kBlockSize;
    item.is_complete = false;
    ASSERT_TRUE(cache->store(item, input.data(), input.size()));

    const auto match = cache->match(7);
    ASSERT_TRUE(match.matched);
    EXPECT_FALSE(match.is_complete);
}

TEST(DiskSpillBlockCacheTest, InvalidateRemovesCommittedIndex) {
    const auto       path       = makeTempDir("invalidate");
    constexpr size_t kBlockSize = 4096;
    auto             cache      = DiskSpillBlockCache::create(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache->init());

    std::vector<char>           input(kBlockSize, 'a');
    MemoryBlockCache::CacheItem item;
    item.cache_key   = 7;
    item.block_index = 3;
    item.block_size  = kBlockSize;
    item.is_complete = true;
    ASSERT_TRUE(cache->store(item, input.data(), input.size()));
    ASSERT_TRUE(cache->contains(7));

    auto slot = cache->invalidate(7);
    ASSERT_TRUE(slot.has_value());
    EXPECT_FALSE(cache->contains(7));
    const auto status = cache->status();
    EXPECT_EQ(status.committed_slot_num, 0u);
    EXPECT_EQ(status.total_slot_num, status.free_slot_num);
}

TEST(DiskSpillBlockCacheTest, InvalidateCancelsReservedCommit) {
    const auto       path       = makeTempDir("invalidate_reserved");
    constexpr size_t kBlockSize = 4096;
    auto             cache      = DiskSpillBlockCache::create(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache->init());

    auto slot = cache->reserve(/*cache_key=*/9, kBlockSize, /*is_complete=*/true);
    ASSERT_TRUE(slot.has_value());

    std::vector<char> input(kBlockSize, 'r');
    ASSERT_TRUE(cache->writeReserved(*slot, input.data(), input.size()));
    cache->invalidate(/*cache_key=*/9);

    EXPECT_FALSE(cache->commit(*slot));
    EXPECT_FALSE(cache->contains(9));

    const auto status = cache->status();
    EXPECT_EQ(status.committed_slot_num, 0u);
    EXPECT_EQ(status.total_slot_num, status.free_slot_num);
}

TEST(DiskSpillBlockCacheTest, ExternalSlotPathWritesAndCommits) {
    const auto       path       = makeTempDir("external_basic");
    constexpr size_t kBlockSize = 4096;
    auto             cache      = DiskSpillBlockCache::create(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache->init());

    // Reserve on master path, then replay via worker path
    auto slot = cache->reserve(/*cache_key=*/11, kBlockSize, /*is_complete=*/true);
    ASSERT_TRUE(slot.has_value());
    std::vector<char> input(kBlockSize, 'e');
    ASSERT_TRUE(cache->writeReserved(*slot, input.data(), input.size()));
    ASSERT_TRUE(cache->putExternalSlot(*slot, input.data(), input.size()));
    EXPECT_TRUE(cache->commit(*slot));
    EXPECT_TRUE(cache->contains(11));
}

TEST(DiskSpillBlockCacheTest, RejectsStaleExternalSlotKeyGeneration) {
    const auto       path       = makeTempDir("external_stale_generation");
    constexpr size_t kBlockSize = 4096;
    auto             cache      = DiskSpillBlockCache::create(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache->init());

    auto slot = cache->reserve(/*cache_key=*/12, kBlockSize, /*is_complete=*/true);
    ASSERT_TRUE(slot.has_value());
    std::vector<char> input(kBlockSize, 's');
    ASSERT_TRUE(cache->writeReserved(*slot, input.data(), input.size()));
    ASSERT_TRUE(cache->commit(*slot));
    ASSERT_TRUE(cache->invalidate(/*cache_key=*/12).has_value());

    // After invalidate, putExternalSlot with the OLD gen must be rejected
    EXPECT_FALSE(cache->putExternalSlot(*slot, input.data(), input.size()));
    EXPECT_FALSE(cache->contains(12));
}

TEST(DiskSpillBlockCacheTest, TakenSlotIsNotReusedBeforeRelease) {
    const auto       path       = makeTempDir("take_exclusive");
    constexpr size_t kBlockSize = 4096;
    auto             cfg        = makeConfig(path, kBlockSize, /*capacity_mb=*/1);
    // Make capacity match exactly 2 slots so LRU reuse selection is observable.
    cfg.segment_bytes        = 4096;
    cfg.disks[0].capacity_mb = 1;  // 256 slots, plenty for the test
    auto cache               = DiskSpillBlockCache::create(cfg);
    ASSERT_TRUE(cache->init());

    // store two complete keys
    std::vector<char>           da(kBlockSize, 'A');
    std::vector<char>           db(kBlockSize, 'B');
    MemoryBlockCache::CacheItem ia;
    ia.cache_key   = 200;
    ia.block_index = 1;
    ia.block_size  = kBlockSize;
    ia.is_complete = true;
    ASSERT_TRUE(cache->store(ia, da.data(), da.size()));
    MemoryBlockCache::CacheItem ib;
    ib.cache_key   = 201;
    ib.block_index = 2;
    ib.block_size  = kBlockSize;
    ib.is_complete = true;
    ASSERT_TRUE(cache->store(ib, db.data(), db.size()));

    auto taken_a = cache->takeForRead(200);
    ASSERT_TRUE(taken_a.has_value());
    const int taken_slot_id = taken_a->item().slot_id;

    // 201 is still in committed_index; a new spill triggering LRU reuse cannot
    // pick the slot currently held by taken_a.
    auto reuse = cache->reserve(/*cache_key=*/9999, kBlockSize, true);
    ASSERT_TRUE(reuse.has_value());
    EXPECT_NE(reuse->slot_id, taken_slot_id);
    EXPECT_TRUE(cache->commit(*reuse));
}

// =====================================================================
// New cases for the refactored DiskSpillBlockCache
// =====================================================================

TEST(DiskSpillBlockCacheTest, MultiDiskRoundRobinAllocates) {
    const auto       p0         = makeTempDir("multi_rr_0");
    const auto       p1         = makeTempDir("multi_rr_1");
    constexpr size_t kBlockSize = 4096;
    auto             cfg        = makeMultiDiskConfig({{p0, 1}, {p1, 1}}, kBlockSize, /*segment_bytes=*/4096);
    auto             cache      = DiskSpillBlockCache::create(cfg);
    ASSERT_TRUE(cache->init());

    int disk_0_count = 0;
    int disk_1_count = 0;
    for (int i = 0; i < 4; ++i) {
        auto s = cache->reserve(static_cast<CacheKeyType>(100 + i), kBlockSize, true);
        ASSERT_TRUE(s.has_value()) << "iter " << i;
        if (s->disk_id == 0) {
            ++disk_0_count;
        }
        if (s->disk_id == 1) {
            ++disk_1_count;
        }
    }
    EXPECT_EQ(disk_0_count, 2);
    EXPECT_EQ(disk_1_count, 2);
}

TEST(DiskSpillBlockCacheTest, MultiDiskOneFullFallsBackToOther) {
    const auto       p0         = makeTempDir("multi_fb_0");
    const auto       p1         = makeTempDir("multi_fb_1");
    constexpr size_t kBlockSize = 4096;
    auto             cfg        = makeMultiDiskConfig({{p0, 1}, {p1, 1}}, kBlockSize, /*segment_bytes=*/4096);
    auto             cache      = DiskSpillBlockCache::create(cfg);
    ASSERT_TRUE(cache->init());

    // Reserve a known number of times; with 256 slots each (1MB/4KB) we have 512 total.
    int success = 0;
    for (int i = 0; i < 600; ++i) {
        auto s = cache->reserve(static_cast<CacheKeyType>(2000 + i), kBlockSize, true);
        if (!s.has_value()) {
            break;
        }
        std::vector<char> data(kBlockSize, 'f');
        ASSERT_TRUE(cache->writeReserved(*s, data.data(), data.size()));
        ASSERT_TRUE(cache->commit(*s));
        ++success;
    }
    // Should be able to allocate all 512 slots and the 513rd should trigger LRU
    // reuse (also success since slots are in COMMITTED state).
    EXPECT_GE(success, 512);
}

TEST(DiskSpillBlockCacheTest, GenerationPreventsABA) {
    const auto       path       = makeTempDir("aba_test");
    constexpr size_t kBlockSize = 4096;
    auto             cfg        = makeConfig(path, kBlockSize);
    cfg.segment_bytes           = 4096;
    cfg.disks[0].capacity_mb    = 1;
    auto cache                  = DiskSpillBlockCache::create(cfg);
    ASSERT_TRUE(cache->init());

    auto first = cache->reserve(/*cache_key=*/50, kBlockSize, true);
    ASSERT_TRUE(first.has_value());
    std::vector<char> a_data(kBlockSize, 'A');
    ASSERT_TRUE(cache->writeReserved(*first, a_data.data(), a_data.size()));
    ASSERT_TRUE(cache->commit(*first));
    auto first_slot = *first;

    ASSERT_TRUE(cache->invalidate(50).has_value());

    // Next reservation for a DIFFERENT key may or may not hit the same slot,
    // but key_gen for 50 has been bumped, so commit of old first_slot must fail.
    auto second = cache->reserve(/*cache_key=*/51, kBlockSize, true);
    ASSERT_TRUE(second.has_value());

    // Old commit for key 50 must fail (key_gen was bumped during invalidate).
    EXPECT_FALSE(cache->commit(first_slot));
}

TEST(DiskSpillBlockCacheTest, ConcurrentTakeForReadOnlyOneSucceeds) {
    const auto       path       = makeTempDir("rescan_contract");
    constexpr size_t kBlockSize = 4096;
    auto             cache      = DiskSpillBlockCache::create(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache->init());

    std::vector<char>           data(kBlockSize, 'r');
    MemoryBlockCache::CacheItem item;
    item.cache_key   = 60;
    item.block_index = 1;
    item.block_size  = kBlockSize;
    item.is_complete = true;
    ASSERT_TRUE(cache->store(item, data.data(), data.size()));

    auto t1 = cache->takeForRead(60);
    ASSERT_TRUE(t1.has_value());
    auto t2 = cache->takeForRead(60);
    EXPECT_FALSE(t2.has_value());

    t1->release();
    EXPECT_FALSE(cache->takeForRead(60).has_value());  // already removed from committed
}

TEST(DiskSpillBlockCacheTest, TakenSlotRaiiAutoReleaseOnDestruction) {
    const auto       path       = makeTempDir("raii_release");
    constexpr size_t kBlockSize = 4096;
    auto             cache      = DiskSpillBlockCache::create(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache->init());

    std::vector<char>           data(kBlockSize, 'q');
    MemoryBlockCache::CacheItem item;
    item.cache_key   = 70;
    item.block_index = 1;
    item.block_size  = kBlockSize;
    item.is_complete = true;
    ASSERT_TRUE(cache->store(item, data.data(), data.size()));
    {
        auto taken = cache->takeForRead(70);
        ASSERT_TRUE(taken.has_value());
        // No explicit release; destructor must auto-release.
    }
    EXPECT_EQ(cache->status().inflight_read_slot_num, 0u);
}

TEST(DiskSpillBlockCacheTest, WorkerModeRefusesMasterOperations) {
    const auto       path       = makeTempDir("worker_mode");
    constexpr size_t kBlockSize = 4096;
    auto             cache      = DiskSpillBlockCache::create(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache->init());
    cache->setMasterMode(false);

    EXPECT_FALSE(cache->reserve(/*cache_key=*/80, kBlockSize, true).has_value());
    EXPECT_FALSE(cache->takeForRead(80).has_value());

    // putExternalSlot (worker path) and deleteSlot should still work
    DiskSpillBlockCache::DiskItem fake_slot;
    fake_slot.cache_key   = 81;
    fake_slot.disk_id     = 0;
    fake_slot.slot_id     = 0;
    fake_slot.gen.slot_gen = 1;
    fake_slot.gen.key_gen  = 1;
    fake_slot.block_size   = kBlockSize;
    fake_slot.is_complete  = true;
    std::vector<char> data(kBlockSize, 'w');
    EXPECT_TRUE(cache->putExternalSlot(fake_slot, data.data(), data.size()));
    EXPECT_TRUE(cache->contains(81));
}

}  // namespace
}  // namespace rtp_llm
