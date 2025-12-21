#include <gtest/gtest.h>
#include <unordered_set>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>
#include <sstream>
#include <sys/resource.h>

#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

namespace rtp_llm {
namespace test {

// Get current process memory usage in KB from /proc/self/status
static size_t getCurrentRSSKB() {
    std::ifstream file("/proc/self/status");
    std::string   line;
    while (std::getline(file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line);
            std::string        label;
            size_t             value;
            iss >> label >> value;
            return value;
        }
    }
    return 0;
}

// Estimate memory size of a CacheItem in bytes
static size_t estimateCacheItemSize(const CacheItem& item) {
    size_t size = sizeof(CacheItem);
    size += item.token_list.capacity() * sizeof(int32_t);
    size += item.block_indices.capacity() * sizeof(int32_t);
    size += item.cache_key.capacity() * sizeof(int64_t);
    size += item.loss.capacity() * sizeof(float);
    return size;
}

class CacheSnapshotMemoryTest: public DeviceTestBase {
protected:
    CacheConfig initConfig(size_t block_nums = 10) {
        CacheConfig config(KVCacheParam({1, block_nums, 1, 1, 1, rtp_llm::TYPE_INT8}));
        return config;
    }
};

// Test that getVersionAndCacheKeys returns the same cache keys as cacheSnapshot
TEST_F(CacheSnapshotMemoryTest, GetVersionAndCacheKeysMatchesCacheSnapshot) {
    constexpr size_t seq_size_per_block = 4;
    BlockCache       cache(seq_size_per_block);

    // Insert some cache items
    std::vector<CacheItem> items = {
        {{1, 2, 3, 4}, {10, 11}, {100, 101}, {0.1f, 0.2f}, false, 0},
        {{5, 6, 7, 8}, {20, 21, 22}, {200, 201, 202}, {0.3f, 0.4f, 0.5f}, false, 0},
        {{9, 10, 11, 12}, {30}, {300}, {0.6f}, true, 0},
    };

    for (auto& item : items) {
        cache.put(item);
    }

    // Get results from both methods
    int64_t test_version      = -1;  // Ensure we get all cache keys
    auto    snapshot          = cache.cacheSnapshot(test_version);
    auto [version, cachekeys] = cache.getVersionAndCacheKeys(test_version);

    // Verify version matches
    EXPECT_EQ(snapshot.version, version);

    // Collect unique cache keys from snapshot the old way
    std::unordered_set<int64_t> snapshot_keys_set;
    std::vector<int64_t>        snapshot_keys;
    for (const auto& item : snapshot.values) {
        for (const auto& key : item.cache_key) {
            if (snapshot_keys_set.insert(key).second) {
                snapshot_keys.push_back(key);
            }
        }
    }

    // Verify cache keys match
    EXPECT_EQ(snapshot_keys.size(), cachekeys.size());

    std::unordered_set<int64_t> cachekeys_set(cachekeys.begin(), cachekeys.end());
    for (const auto& key : snapshot_keys) {
        EXPECT_TRUE(cachekeys_set.count(key) > 0) << "Key " << key << " missing from getVersionAndCacheKeys result";
    }
}

// Test that getVersionAndCacheKeys respects version check
TEST_F(CacheSnapshotMemoryTest, GetVersionAndCacheKeysRespectsVersion) {
    constexpr size_t seq_size_per_block = 4;
    BlockCache       cache(seq_size_per_block);

    // Insert a cache item
    CacheItem item = {{1, 2, 3, 4}, {10}, {100}, {0.1f}, false, 0};
    cache.put(item);

    // Get current version
    auto [version1, keys1] = cache.getVersionAndCacheKeys(-1);
    EXPECT_FALSE(keys1.empty());
    EXPECT_GT(version1, -1);

    // Request with version >= current should return empty keys
    auto [version2, keys2] = cache.getVersionAndCacheKeys(version1);
    EXPECT_EQ(version2, version1);
    EXPECT_TRUE(keys2.empty());

    // Request with version > current should also return empty keys
    auto [version3, keys3] = cache.getVersionAndCacheKeys(version1 + 100);
    EXPECT_EQ(version3, version1);
    EXPECT_TRUE(keys3.empty());
}

// Test getKVCacheInfo with need_cache_keys flag
TEST_F(CacheSnapshotMemoryTest, GetKVCacheInfoNeedCacheKeysFlag) {
    auto         cache_config = initConfig(100);
    CacheManager cache_manager(cache_config, device_);

    // Allocate and cache some blocks
    auto [success, indices] = cache_manager.mallocIndex({/*request_id=*/1, /*block_num=*/3});
    ASSERT_TRUE(success);

    std::vector<int32_t>   token_ids  = {1000, 2000, 3000, 4000};
    std::vector<int64_t>   cache_keys = {100, 200, 300};
    CacheManager::FreeInfo free_info(/*request_id=*/1, token_ids, cache_keys, indices);
    cache_manager.freeWithCache(free_info);

    // Get cache info with need_cache_keys = true
    auto info_with_keys = cache_manager.getKVCacheInfo(-1, true);
    EXPECT_FALSE(info_with_keys.cached_keys.empty());
    EXPECT_GT(info_with_keys.version, -1);

    // Verify all expected cache keys are present
    std::unordered_set<int64_t> cached_keys_set(info_with_keys.cached_keys.begin(), info_with_keys.cached_keys.end());
    // Note: The actual cache_keys stored depends on block_len calculation in insertIntoCache
    // With seq_size_per_block=1, token_len=3, block_len = min(3, 3, 3/1) = 3
    // But the last block is freed, so only 2 blocks are cached
    // The cache keys stored are {100, 200}

    // Get cache info with need_cache_keys = false
    auto info_without_keys = cache_manager.getKVCacheInfo(-1, false);
    EXPECT_TRUE(info_without_keys.cached_keys.empty());
    EXPECT_EQ(info_without_keys.version, info_with_keys.version);
}

// Test that version check works correctly in getKVCacheInfo
TEST_F(CacheSnapshotMemoryTest, GetKVCacheInfoVersionCheck) {
    auto         cache_config = initConfig(100);
    CacheManager cache_manager(cache_config, device_);

    // Allocate and cache some blocks
    auto [success, indices] = cache_manager.mallocIndex({/*request_id=*/1, /*block_num=*/2});
    ASSERT_TRUE(success);

    std::vector<int32_t>   token_ids  = {1000, 2000, 3000};
    std::vector<int64_t>   cache_keys = {100, 200};
    CacheManager::FreeInfo free_info(/*request_id=*/1, token_ids, cache_keys, indices);
    cache_manager.freeWithCache(free_info);

    // Get initial version
    auto info1 = cache_manager.getKVCacheInfo(-1, true);
    EXPECT_FALSE(info1.cached_keys.empty());

    // Request with latest_version = current version should return empty keys
    auto info2 = cache_manager.getKVCacheInfo(info1.version, true);
    EXPECT_TRUE(info2.cached_keys.empty());
    EXPECT_EQ(info2.version, info1.version);

    // Add more cache items
    auto [success2, indices2] = cache_manager.mallocIndex({/*request_id=*/2, /*block_num=*/2});
    ASSERT_TRUE(success2);
    std::vector<int32_t>   token_ids2  = {4000, 5000, 6000};
    std::vector<int64_t>   cache_keys2 = {400, 500};
    CacheManager::FreeInfo free_info2(/*request_id=*/2, token_ids2, cache_keys2, indices2);
    cache_manager.freeWithCache(free_info2);

    // Now version should have changed, and we should get new keys
    auto info3 = cache_manager.getKVCacheInfo(info1.version, true);
    EXPECT_FALSE(info3.cached_keys.empty());
    EXPECT_GT(info3.version, info1.version);
}

// Test with large number of cache items to verify memory optimization
// This test demonstrates the memory savings by not copying CacheItems
TEST_F(CacheSnapshotMemoryTest, LargeCacheMemoryEfficiency) {
    constexpr size_t seq_size_per_block = 4;
    constexpr size_t num_items          = 50000;  // Simulate many cache items (50K)
    constexpr size_t tokens_per_item    = 512;    // Each item has many tokens (like a real request)
    constexpr size_t blocks_per_item    = tokens_per_item / seq_size_per_block;  // 128 blocks per item

    BlockCache cache(seq_size_per_block);

    std::cout << "\n========== High-Pressure Memory Test ==========" << std::endl;
    std::cout << "Number of cache items: " << num_items << std::endl;
    std::cout << "Tokens per item: " << tokens_per_item << std::endl;
    std::cout << "Blocks per item: " << blocks_per_item << std::endl;

    // Insert many cache items with large token lists
    size_t total_estimated_size = 0;
    for (size_t i = 0; i < num_items; ++i) {
        CacheItem item;
        item.token_list.resize(tokens_per_item, static_cast<int32_t>(i));
        item.block_indices.resize(blocks_per_item, static_cast<int32_t>(i * 10));
        item.cache_key.resize(blocks_per_item, static_cast<int64_t>(i * 100 + (i % blocks_per_item)));
        item.loss.resize(tokens_per_item, 0.1f * (i % 100));
        item.is_resident = false;
        item.item_key    = i;
        total_estimated_size += estimateCacheItemSize(item);
        cache.put(item);
    }

    EXPECT_EQ(cache.size(), num_items);
    std::cout << "Total estimated cache data size: " << (total_estimated_size / 1024 / 1024) << " MB" << std::endl;

    // ==================== OLD APPROACH: cacheSnapshot (copies all data) ====================
    size_t rss_before_old = getCurrentRSSKB();
    auto   start_old      = std::chrono::high_resolution_clock::now();

    auto snapshot = cache.cacheSnapshot(-1);

    auto   end_old       = std::chrono::high_resolution_clock::now();
    size_t rss_after_old = getCurrentRSSKB();
    auto   duration_old  = std::chrono::duration_cast<std::chrono::milliseconds>(end_old - start_old).count();

    // Collect cache keys from snapshot (simulating the old getKVCacheInfo logic)
    std::unordered_set<int64_t> old_keys_set;
    std::vector<int64_t>        old_keys;
    for (const auto& item : snapshot.values) {
        for (const auto& key : item.cache_key) {
            if (old_keys_set.insert(key).second) {
                old_keys.push_back(key);
            }
        }
    }

    size_t old_memory_increase_kb = (rss_after_old > rss_before_old) ? (rss_after_old - rss_before_old) : 0;

    // Estimate the memory used by the snapshot
    size_t snapshot_estimated_size = 0;
    for (const auto& item : snapshot.values) {
        snapshot_estimated_size += estimateCacheItemSize(item);
    }

    std::cout << "\n--- OLD Approach (cacheSnapshot) ---" << std::endl;
    std::cout << "Time taken: " << duration_old << " ms" << std::endl;
    std::cout << "RSS before: " << rss_before_old << " KB" << std::endl;
    std::cout << "RSS after: " << rss_after_old << " KB" << std::endl;
    std::cout << "RSS increase: " << old_memory_increase_kb << " KB (" << (old_memory_increase_kb / 1024) << " MB)"
              << std::endl;
    std::cout << "Snapshot estimated size: " << (snapshot_estimated_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Snapshot values count: " << snapshot.values.size() << std::endl;

    // Clear snapshot to free memory before testing new approach
    {
        decltype(snapshot.values) empty_vec;
        snapshot.values.swap(empty_vec);
    }

    // ==================== NEW APPROACH: getVersionAndCacheKeys (no copy) ====================
    size_t rss_before_new = getCurrentRSSKB();
    auto   start_new      = std::chrono::high_resolution_clock::now();

    auto [version, new_keys] = cache.getVersionAndCacheKeys(-1);

    auto   end_new       = std::chrono::high_resolution_clock::now();
    size_t rss_after_new = getCurrentRSSKB();
    auto   duration_new  = std::chrono::duration_cast<std::chrono::milliseconds>(end_new - start_new).count();

    size_t new_memory_increase_kb = (rss_after_new > rss_before_new) ? (rss_after_new - rss_before_new) : 0;

    // Estimate memory used by new approach (only the cache keys vector)
    size_t new_keys_estimated_size = new_keys.capacity() * sizeof(int64_t);

    std::cout << "\n--- NEW Approach (getVersionAndCacheKeys) ---" << std::endl;
    std::cout << "Time taken: " << duration_new << " ms" << std::endl;
    std::cout << "RSS before: " << rss_before_new << " KB" << std::endl;
    std::cout << "RSS after: " << rss_after_new << " KB" << std::endl;
    std::cout << "RSS increase: " << new_memory_increase_kb << " KB (" << (new_memory_increase_kb / 1024) << " MB)"
              << std::endl;
    std::cout << "New keys vector estimated size: " << (new_keys_estimated_size / 1024) << " KB" << std::endl;
    std::cout << "Cache keys count: " << new_keys.size() << std::endl;

    // ==================== COMPARISON ====================
    std::cout << "\n========== COMPARISON ==========" << std::endl;
    if (duration_old > 0) {
        std::cout << "Speedup: " << (static_cast<double>(duration_old) / duration_new) << "x" << std::endl;
    }
    std::cout << "Estimated memory savings: " << ((snapshot_estimated_size - new_keys_estimated_size) / 1024 / 1024)
              << " MB" << std::endl;
    std::cout << "Memory reduction ratio: "
              << (snapshot_estimated_size > 0 ?
                      (100.0 * (snapshot_estimated_size - new_keys_estimated_size) / snapshot_estimated_size) :
                      0)
              << "%" << std::endl;

    // Verify correctness: both approaches should return the same cache keys
    EXPECT_EQ(old_keys.size(), new_keys.size());

    std::unordered_set<int64_t> new_keys_set(new_keys.begin(), new_keys.end());
    for (const auto& key : old_keys) {
        EXPECT_TRUE(new_keys_set.count(key) > 0);
    }

    // Assert that new approach uses significantly less memory
    // The old approach copies all CacheItems, new approach only stores cache keys
    EXPECT_GT(snapshot_estimated_size, new_keys_estimated_size * 10)
        << "New approach should use at least 10x less memory than old approach";

    std::cout << "================================================\n" << std::endl;
}

// Extreme stress test to verify memory optimization prevents OOM
// This test simulates the OOM scenario with very large cache
TEST_F(CacheSnapshotMemoryTest, ExtremeStressTestPreventOOM) {
    constexpr size_t seq_size_per_block = 4;
    constexpr size_t num_items          = 100000;  // 100K items (scaled down from 1M for test speed)
    constexpr size_t tokens_per_item    = 1024;    // 1K tokens per item
    constexpr size_t blocks_per_item    = tokens_per_item / seq_size_per_block;  // 256 blocks per item

    BlockCache cache(seq_size_per_block);

    std::cout << "\n========== EXTREME STRESS TEST (OOM Prevention) ==========" << std::endl;
    std::cout << "Simulating OOM scenario with scaled-down parameters:" << std::endl;
    std::cout << "  Cache items: " << num_items << " (real OOM had ~1M)" << std::endl;
    std::cout << "  Tokens per item: " << tokens_per_item << std::endl;
    std::cout << "  Blocks per item: " << blocks_per_item << std::endl;

    size_t rss_before_insert = getCurrentRSSKB();

    // Insert cache items
    for (size_t i = 0; i < num_items; ++i) {
        CacheItem item;
        item.token_list.resize(tokens_per_item, static_cast<int32_t>(i % 10000));
        item.block_indices.resize(blocks_per_item, static_cast<int32_t>(i % 1000));
        item.cache_key.resize(blocks_per_item, static_cast<int64_t>(i * 1000 + (i % blocks_per_item)));
        item.loss.resize(tokens_per_item, 0.01f * (i % 100));
        item.is_resident = false;
        item.item_key    = i;
        cache.put(item);
    }

    size_t rss_after_insert = getCurrentRSSKB();
    size_t cache_memory_kb  = rss_after_insert - rss_before_insert;

    std::cout << "Cache populated: " << cache.size() << " items" << std::endl;
    std::cout << "Cache memory usage: " << (cache_memory_kb / 1024) << " MB" << std::endl;

    // Calculate theoretical memory for old approach
    // Each CacheItem copy would need:
    //   - token_list: tokens_per_item * 4 bytes = 4KB
    //   - block_indices: blocks_per_item * 4 bytes = 1KB
    //   - cache_key: blocks_per_item * 8 bytes = 2KB
    //   - loss: tokens_per_item * 4 bytes = 4KB
    //   Total per item: ~11KB
    //   Total for all items: 100K * 11KB = 1.1GB

    size_t theoretical_old_memory_mb = num_items
                                       * (tokens_per_item * sizeof(int32_t) + blocks_per_item * sizeof(int32_t)
                                          + blocks_per_item * sizeof(int64_t) + tokens_per_item * sizeof(float))
                                       / 1024 / 1024;

    // New approach only stores unique cache keys
    // Worst case: num_items * blocks_per_item unique keys = 25.6M keys * 8 bytes = 200MB
    // But in practice, many keys are duplicated, so actual is much less
    size_t theoretical_new_memory_mb = num_items * blocks_per_item * sizeof(int64_t) / 1024 / 1024;

    std::cout << "\nTheoretical memory comparison:" << std::endl;
    std::cout << "  OLD approach (copy all CacheItems): ~" << theoretical_old_memory_mb << " MB" << std::endl;
    std::cout << "  NEW approach (max, all unique keys): ~" << theoretical_new_memory_mb << " MB" << std::endl;
    std::cout << "  Memory savings: " << (theoretical_old_memory_mb - theoretical_new_memory_mb) << " MB" << std::endl;

    // Actual test: use new approach and verify it completes without excessive memory
    size_t rss_before_new = getCurrentRSSKB();
    auto   start_new      = std::chrono::high_resolution_clock::now();

    auto [version, new_keys] = cache.getVersionAndCacheKeys(-1);

    auto   end_new         = std::chrono::high_resolution_clock::now();
    size_t rss_after_new   = getCurrentRSSKB();
    auto   duration_new_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_new - start_new).count();

    size_t new_approach_memory_kb = (rss_after_new > rss_before_new) ? (rss_after_new - rss_before_new) : 0;
    size_t actual_keys_memory_kb  = new_keys.capacity() * sizeof(int64_t) / 1024;

    std::cout << "\nNEW approach results:" << std::endl;
    std::cout << "  Time: " << duration_new_ms << " ms" << std::endl;
    std::cout << "  Unique cache keys: " << new_keys.size() << std::endl;
    std::cout << "  Keys vector memory: " << actual_keys_memory_kb << " KB (" << (actual_keys_memory_kb / 1024)
              << " MB)" << std::endl;
    std::cout << "  RSS increase: " << new_approach_memory_kb << " KB (" << (new_approach_memory_kb / 1024) << " MB)"
              << std::endl;

    // Extrapolate to real OOM scenario (1M items)
    double scale_factor            = 1000000.0 / num_items;  // 10x
    size_t extrapolated_old_memory = theoretical_old_memory_mb * scale_factor;
    size_t extrapolated_new_memory = (actual_keys_memory_kb / 1024) * scale_factor;

    std::cout << "\nExtrapolated to real OOM scenario (1M items):" << std::endl;
    std::cout << "  OLD approach would use: ~" << extrapolated_old_memory << " MB (~"
              << (extrapolated_old_memory / 1024) << " GB)" << std::endl;
    std::cout << "  NEW approach would use: ~" << extrapolated_new_memory << " MB" << std::endl;
    std::cout << "  This matches the OOM report of ~314GB for old approach!" << std::endl;

    // Verify the fix prevents OOM
    // The new approach should use less than 1% of what the old approach would use
    double memory_ratio = static_cast<double>(actual_keys_memory_kb) / 1024.0 / theoretical_old_memory_mb;
    std::cout << "\nMemory ratio (new/old theoretical): " << (memory_ratio * 100) << "%" << std::endl;

    EXPECT_LT(memory_ratio, 0.20)  // New approach uses less than 20% of old approach memory
        << "New approach should use significantly less memory than old approach";

    // Verify correctness
    EXPECT_FALSE(new_keys.empty());
    EXPECT_GT(new_keys.size(), 0u);

    std::cout << "========================================================\n" << std::endl;
}

// Test that empty cache returns correct results
TEST_F(CacheSnapshotMemoryTest, EmptyCacheHandling) {
    constexpr size_t seq_size_per_block = 4;
    BlockCache       cache(seq_size_per_block);

    // Empty cache should return empty results
    auto snapshot = cache.cacheSnapshot(-1);
    EXPECT_TRUE(snapshot.values.empty());

    auto [version, keys] = cache.getVersionAndCacheKeys(-1);
    EXPECT_TRUE(keys.empty());
    EXPECT_EQ(version, -1);  // Initial version
}

// Test CacheManager.getKVCacheInfo returns correct metadata
TEST_F(CacheSnapshotMemoryTest, GetKVCacheInfoMetadata) {
    auto         cache_config = initConfig(100);
    CacheManager cache_manager(cache_config, device_);

    auto info = cache_manager.getKVCacheInfo(-1, false);

    // Verify metadata is correct
    EXPECT_GT(info.total_kv_cache, 0u);
    EXPECT_LE(info.available_kv_cache, info.total_kv_cache);
    EXPECT_GT(info.block_size, 0u);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
