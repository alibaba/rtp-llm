#include "rtp_llm/cpp/cache/connector/memory/DiskSpillBlockCache.h"

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <unistd.h>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

std::string makeTempDir(const std::string& name) {
    auto path =
        std::filesystem::temp_directory_path() / ("rtp_llm_disk_spill_" + name + "_" + std::to_string(::getpid()));
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    std::filesystem::create_directories(path, ec);
    EXPECT_FALSE(ec) << ec.message();
    return path.string();
}

DiskSpillBlockCache::InitConfig makeConfig(const std::string& path, size_t block_size, size_t capacity_mb = 1) {
    DiskSpillBlockCache::InitConfig config;
    config.disks.push_back(DiskSpillBlockCache::DiskConfig{path, capacity_mb});
    config.block_size = block_size;
    return config;
}

}  // namespace

TEST(DiskSpillBlockCacheTest, ParseDiskConfigs) {
    const auto disks = DiskSpillBlockCache::parseDiskConfigs("/tmp/a=10, /tmp/b=20");
    ASSERT_EQ(disks.size(), 2u);
    EXPECT_EQ(disks[0].path, "/tmp/a");
    EXPECT_EQ(disks[0].capacity_mb, 10u);
    EXPECT_EQ(disks[1].path, "/tmp/b");
    EXPECT_EQ(disks[1].capacity_mb, 20u);
}

TEST(DiskSpillBlockCacheTest, StoreMatchTakeReadRelease) {
    const auto          path       = makeTempDir("basic");
    constexpr size_t    kBlockSize = 1024;
    DiskSpillBlockCache cache(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache.init());

    std::vector<char>           input(kBlockSize, 'x');
    MemoryBlockCache::CacheItem item;
    item.cache_key   = 100;
    item.block_index = 1;
    item.block_size  = kBlockSize;
    item.is_complete = true;
    ASSERT_TRUE(cache.store(item, input.data(), input.size()));

    const auto match = cache.match(100);
    ASSERT_TRUE(match.matched);
    EXPECT_TRUE(match.is_complete);
    EXPECT_EQ(match.block_size, kBlockSize);

    auto taken = cache.takeForRead(100);
    ASSERT_TRUE(taken.has_value());
    EXPECT_FALSE(cache.contains(100));

    std::vector<char> output(kBlockSize, 0);
    ASSERT_TRUE(cache.readTaken(*taken, output.data(), output.size()));
    EXPECT_EQ(output, input);
    EXPECT_TRUE(cache.releaseReadSlot(*taken));

    const auto status = cache.status();
    EXPECT_EQ(status.item_num, 0u);
    EXPECT_EQ(status.total_slot_num, status.free_slot_num);
}

TEST(DiskSpillBlockCacheTest, StorePreservesPartialCompleteness) {
    const auto          path       = makeTempDir("partial_complete_bit");
    constexpr size_t    kBlockSize = 512;
    DiskSpillBlockCache cache(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache.init());

    std::vector<char>           input(kBlockSize, 'p');
    MemoryBlockCache::CacheItem item;
    item.cache_key   = 7;
    item.block_index = 1;
    item.block_size  = kBlockSize;
    item.is_complete = false;
    ASSERT_TRUE(cache.store(item, input.data(), input.size()));

    const auto match = cache.match(7);
    ASSERT_TRUE(match.matched);
    EXPECT_FALSE(match.is_complete);
}

TEST(DiskSpillBlockCacheTest, InvalidateRemovesCommittedIndex) {
    const auto          path       = makeTempDir("invalidate");
    constexpr size_t    kBlockSize = 512;
    DiskSpillBlockCache cache(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache.init());

    std::vector<char>           input(kBlockSize, 'a');
    MemoryBlockCache::CacheItem item;
    item.cache_key   = 7;
    item.block_index = 3;
    item.block_size  = kBlockSize;
    item.is_complete = true;
    ASSERT_TRUE(cache.store(item, input.data(), input.size()));
    ASSERT_TRUE(cache.contains(7));

    auto slot = cache.invalidate(7);
    ASSERT_TRUE(slot.has_value());
    EXPECT_FALSE(cache.contains(7));
    const auto status = cache.status();
    EXPECT_EQ(status.item_num, 0u);
    EXPECT_EQ(status.total_slot_num, status.free_slot_num);
}

TEST(DiskSpillBlockCacheTest, InvalidateCancelsReservedCommit) {
    const auto          path       = makeTempDir("invalidate_reserved");
    constexpr size_t    kBlockSize = 512;
    DiskSpillBlockCache cache(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache.init());

    auto slot = cache.reserve(/*cache_key=*/9, kBlockSize, /*is_complete=*/true);
    ASSERT_TRUE(slot.has_value());

    std::vector<char> input(kBlockSize, 'r');
    ASSERT_TRUE(cache.writeReserved(*slot, input.data(), input.size()));
    EXPECT_FALSE(cache.invalidate(/*cache_key=*/9).has_value());

    EXPECT_FALSE(cache.commit(*slot));
    EXPECT_FALSE(cache.contains(9));
    EXPECT_TRUE(cache.abort(*slot));

    const auto status = cache.status();
    EXPECT_EQ(status.item_num, 0u);
    EXPECT_EQ(status.total_slot_num, status.free_slot_num);
}

TEST(DiskSpillBlockCacheTest, ExternalSlotWithoutKeyGenerationKeepsExistingReservation) {
    const auto          path       = makeTempDir("external_preserve_generation");
    constexpr size_t    kBlockSize = 512;
    DiskSpillBlockCache cache(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache.init());

    auto slot = cache.reserve(/*cache_key=*/11, kBlockSize, /*is_complete=*/true);
    ASSERT_TRUE(slot.has_value());

    std::vector<char> input(kBlockSize, 'e');
    ASSERT_TRUE(cache.writeReserved(*slot, input.data(), input.size()));

    auto external_slot           = *slot;
    external_slot.key_generation = 0;
    ASSERT_TRUE(cache.putExternalSlot(external_slot, input.data(), input.size()));
    EXPECT_TRUE(cache.commit(*slot));
    EXPECT_TRUE(cache.contains(11));
}

TEST(DiskSpillBlockCacheTest, RejectsStaleExternalSlotKeyGeneration) {
    const auto          path       = makeTempDir("external_stale_generation");
    constexpr size_t    kBlockSize = 512;
    DiskSpillBlockCache cache(makeConfig(path, kBlockSize));
    ASSERT_TRUE(cache.init());

    auto slot = cache.reserve(/*cache_key=*/12, kBlockSize, /*is_complete=*/true);
    ASSERT_TRUE(slot.has_value());

    std::vector<char> input(kBlockSize, 's');
    ASSERT_TRUE(cache.writeReserved(*slot, input.data(), input.size()));
    ASSERT_TRUE(cache.commit(*slot));
    ASSERT_TRUE(cache.invalidate(/*cache_key=*/12).has_value());

    EXPECT_FALSE(cache.putExternalSlot(*slot, input.data(), input.size()));
    EXPECT_FALSE(cache.contains(12));
}

TEST(DiskSpillBlockCacheTest, TakenSlotIsNotReusedBeforeRelease) {
    const auto          path       = makeTempDir("take_exclusive");
    constexpr size_t    kBlockSize = 1024 * 1024;
    DiskSpillBlockCache cache(makeConfig(path, kBlockSize, /*capacity_mb=*/1));
    ASSERT_TRUE(cache.init());

    std::vector<char>           input(kBlockSize, 'z');
    MemoryBlockCache::CacheItem item;
    item.cache_key   = 1;
    item.block_index = 1;
    item.block_size  = kBlockSize;
    item.is_complete = true;
    ASSERT_TRUE(cache.store(item, input.data(), input.size()));

    auto taken = cache.takeForRead(1);
    ASSERT_TRUE(taken.has_value());
    auto reserved = cache.reserve(/*cache_key=*/2, kBlockSize, /*is_complete=*/true);
    EXPECT_FALSE(reserved.has_value());

    ASSERT_TRUE(cache.releaseReadSlot(*taken));
    reserved = cache.reserve(/*cache_key=*/2, kBlockSize, /*is_complete=*/true);
    EXPECT_TRUE(reserved.has_value());
}

}  // namespace rtp_llm
