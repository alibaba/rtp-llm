#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/cache/test/mock/MockDistKvCache.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;
using namespace ::testing;

namespace rtp_llm {

class CacheManagerTest: public DeviceTestBase {
protected:
    CacheConfig initConfig() {
        // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
        CacheConfig config(KVCacheParam({1, 4, 1, 1, 1, rtp_llm::TYPE_INT8}));
        return config;
    }

    std::vector<int64_t> constructCacheKey(CacheManager& cache_manager, const vector<int>& token_ids) {
        auto            seq_size_per_block = cache_manager.config_.seq_size_per_block;
        auto            total_blocks       = token_ids.size() / seq_size_per_block;
        vector<int64_t> cache_keys;
        int64_t         hash = 0;
        for (int index = 0; index < total_blocks; index++) {
            auto start_pos = token_ids.begin() + index * seq_size_per_block;
            hash           = std::accumulate(start_pos, start_pos + seq_size_per_block, hash, std::plus<int>());
            cache_keys.push_back(hash);
        }
        return cache_keys;
    }

    CacheManager::MatchInfo mallocWithCache(CacheManager&              cache_manager,
                                            const vector<int>&         token_ids,
                                            const vector<vector<int>>& mm_bounds      = {},
                                            bool                       need_loss      = false,
                                            int                        need_block_num = -1) {
        if (need_block_num == -1) {
            need_block_num = token_ids.size();
        }
        auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
        CacheManager::AdvancedMallocInfo malloc_info(request_id, token_ids, cache_keys, mm_bounds, need_loss);
        auto                             match_info = cache_manager.mallocWithCache(malloc_info);
        if (match_info.cache_blocks.size() < need_block_num) {
            auto [success, index] =
                cache_manager.mallocIndex({request_id, uint32_t(need_block_num - match_info.cache_blocks.size())});
            if (success) {
                match_info.cache_blocks.insert(match_info.cache_blocks.end(), index.begin(), index.end());
            } else {
                cache_manager.free(match_info.cache_blocks);
                return {0, {}, {}};
            }
        }
        return match_info;
    }

    void freeWithCache(CacheManager&           cache_manager,
                       const std::vector<int>& block_indices,
                       const vector<int>&      token_ids,
                       const vector<float>&    loss = {}) {
        auto                   cache_keys = constructCacheKey(cache_manager, token_ids);
        CacheManager::FreeInfo free_info(request_id, token_ids, cache_keys, block_indices, loss);
        cache_manager.freeWithCache(free_info);
    }

    void insertResidentCache(CacheManager&           cache_manager,
                             const std::vector<int>& block_indices,
                             const vector<int>&      token_ids) {
        auto                   cache_keys = constructCacheKey(cache_manager, token_ids);
        CacheManager::FreeInfo free_info(request_id, token_ids, cache_keys, block_indices);
        cache_manager.insertResidentCache(free_info);
    }

protected:
    int64_t request_id = 0;
};

TEST_F(CacheManagerTest, testSimple) {
    auto         cache_config = initConfig();
    CacheManager cache_manager(cache_config, device_);
    auto         allocator = cache_manager.kvCacheAllocator();

    ASSERT_EQ(cache_manager.freeBlockNums(), 3);

    auto [success1, index1] = cache_manager.mallocIndex({request_id, 1});
    ASSERT_TRUE(success1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);

    auto [success2, index2] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_TRUE(success2);
    auto [success3, _] = cache_manager.mallocIndex({request_id, 1});
    ASSERT_FALSE(success3);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 1);

    cache_manager.free(index1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 0);

    cache_manager.free(index2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 0);

    ASSERT_EQ(cache_manager.freeBlockNums(), 3);
}

TEST_F(CacheManagerTest, testAllocateWithFreeCache) {
    auto         cache_config = initConfig();
    CacheManager cache_manager(cache_config, device_);
    auto         allocator = cache_manager.kvCacheAllocator();

    auto match_info = mallocWithCache(cache_manager, {1000, 2000, 3000});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
    ASSERT_FALSE(match_info.reuse_length);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 1);

    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 2000, 3000});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 0);

    auto [success2, index2] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_EQ(index2, std::vector<int>({1, 2}));
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.freeBlockNums(), 1);
}

TEST_F(CacheManagerTest, testLossCache) {
    auto         cache_config = initConfig();
    CacheManager cache_manager(cache_config, device_);

    // test no loss malloc
    auto match_info = mallocWithCache(cache_manager, {1000, 2000, 3000});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 2000, 3000});
    match_info = mallocWithCache(cache_manager, {1000, 2000, 3000}, {}, true, 0);
    ASSERT_EQ(match_info.reuse_length, 0);
    ASSERT_EQ(match_info.cache_blocks.size(), 0);
    ASSERT_EQ(match_info.loss.size(), 0);

    // test malloc with loss
    // pop cache item
    auto [success, index] = cache_manager.mallocIndex({request_id, 3});
    ASSERT_EQ(index, std::vector<int>({1, 2, 3}));
    freeWithCache(cache_manager, index, {1000, 2000, 3000}, {0.1, 0.2});
    ASSERT_EQ(cache_manager.freeBlockNums(), 1);
    match_info = mallocWithCache(cache_manager, {1000, 2000, 3000}, {}, true, 0);
    ASSERT_EQ(match_info.reuse_length, 1);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1}));
    ASSERT_EQ(match_info.loss, std::vector<float>({0.1}));
}

TEST_F(CacheManagerTest, testAllocateWithReuse) {
    auto         cache_config = initConfig();
    CacheManager cache_manager(cache_config, device_);
    auto         allocator = cache_manager.kvCacheAllocator();

    ASSERT_EQ(cache_manager.freeBlockNums(), 3);
    auto [success1, index1] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_EQ(index1, std::vector<int>({1, 2}));
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 0);

    freeWithCache(cache_manager, index1, {1000, 1002});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 1);

    auto match_info = mallocWithCache(cache_manager, {1000, 1002});
    ASSERT_EQ(cache_manager.freeBlockNums(), 1);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2}));
    ASSERT_EQ(match_info.reuse_length, 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 0);

    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 0);

    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(match_info.reuse_length, 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.cacheItemNum(), 1);

    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002, 1003});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 2);
}

TEST_F(CacheManagerTest, testAllocateWithMultimodalReuse) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 2;
    CacheManager cache_manager(cache_config, device_);
    auto         allocator = cache_manager.kvCacheAllocator();

    ASSERT_EQ(cache_manager.freeBlockNums(), 9);
    auto [success1, index1] = cache_manager.mallocIndex({request_id, 4});
    ASSERT_EQ(index1, std::vector<int>({1, 2, 3, 4}));
    ASSERT_EQ(cache_manager.cacheItemNum(), 0);

    freeWithCache(cache_manager, index1, {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(4), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 1);

    auto match_info = mallocWithCache(cache_manager, {1000, 1001, 1002, 1003, 1004, 1005, 1006}, {{5, 2}}, false, 4);
    ASSERT_EQ(cache_manager.freeBlockNums(), 4);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 4, 5}));
    ASSERT_EQ(match_info.reuse_length, 4);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(5), 1);

    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1001, 1002, 1003, 1004, 1005, 1006});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(4), 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 1);

    match_info = mallocWithCache(cache_manager, {1000, 1001, 1002, 1003, 1004, 1015}, {{1, 2}, {3, 2}}, false, 3);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({4, 5, 6}));
    ASSERT_EQ(match_info.reuse_length, 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(5), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(6), 1);
    ASSERT_EQ(cache_manager.cacheItemNum(), 1);

    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1001, 1002, 1003, 1004, 1015});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(5), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(6), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 2);
}

TEST_F(CacheManagerTest, testMatchMaxLen) {
    auto cache_config       = initConfig();
    cache_config.block_nums = 100;
    CacheManager cache_manager(cache_config, device_);
    auto         allocator = cache_manager.kvCacheAllocator();
    ASSERT_EQ(cache_manager.freeBlockNums(), 99);

    // malloc cache item 1
    auto match_info = mallocWithCache(cache_manager, {1000, 1002});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2}));
    ASSERT_EQ(match_info.reuse_length, 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);

    // insert cache item 1
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 0);

    // malloc cache item 2
    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(match_info.reuse_length, 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(4), 0);

    // insert cache item 2
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002, 1003});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(4), 0);

    // malloc cache item 3
    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003, 1004});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3, 4}));
    ASSERT_EQ(match_info.reuse_length, 2);  // Assuming 2 blocks were reused, replace with actual logic
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 3);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(6), 0);

    // insert cache item 3
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002, 1003, 1004});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 3);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(4), 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(6), 0);

    // trigger match max len cache item
    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003, 1004});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3, 4}));
    ASSERT_EQ(match_info.reuse_length, 3);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 4);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 3);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(6), 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(7), 0);
}

TEST_F(CacheManagerTest, testPopNoResidentCacheItem) {
    auto         cache_config = initConfig();
    CacheManager cache_manager(cache_config, device_);
    auto         allocator = cache_manager.kvCacheAllocator();
    ASSERT_EQ(cache_manager.freeBlockNums(), 3);

    // malloc cache item 1
    auto [success1, index1] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_TRUE(success1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);

    // insert cache item 1
    freeWithCache(cache_manager, index1, {1000, 1002});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 0);

    // trigger reuse cache, pop cache item 1, malloc from free failed
    auto match_info = mallocWithCache(cache_manager, {1000, 1002, 1003, 1004});
    ASSERT_EQ(match_info.reuse_length, 0);
    ASSERT_EQ(cache_manager.freeBlockNums(), 3);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 0);

    // trigger malloc block from free failed
    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003, 1004});
    ASSERT_FALSE(match_info.reuse_length);

    // insert cache item 2
    match_info = mallocWithCache(cache_manager, {100, 1002});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2}));
    ASSERT_EQ(match_info.reuse_length, 0);
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002});
    ASSERT_EQ(cache_manager.freeBlockNums(), 2);

    // trigger pop cache item 2 from cache, malloc success
    match_info = mallocWithCache(cache_manager, {2000, 2002, 2003});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(match_info.reuse_length, 0);
    ASSERT_EQ(cache_manager.freeBlockNums(), 0);
}

TEST_F(CacheManagerTest, testPopTwoCache) {
    auto cache_config       = initConfig();
    cache_config.block_nums = 7;
    CacheManager cache_manager(cache_config, device_);
    auto         allocator = cache_manager.kvCacheAllocator();
    ASSERT_EQ(cache_manager.freeBlockNums(), 6);

    // insert cache item 1
    auto match_info = mallocWithCache(cache_manager, {1000, 1002});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2}));
    ASSERT_EQ(match_info.reuse_length, 0);
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002});
    ASSERT_EQ(cache_manager.freeBlockNums(), 5);
    ASSERT_TRUE(cache_manager.blockCache().hasKey({1000}));

    // insert cache item 2
    match_info = mallocWithCache(cache_manager, {2000, 2002, 2003});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({2, 3, 4}));
    ASSERT_EQ(match_info.reuse_length, 0);
    freeWithCache(cache_manager, match_info.cache_blocks, {2000, 2002, 2003});
    ASSERT_EQ(cache_manager.freeBlockNums(), 3);
    ASSERT_TRUE(cache_manager.blockCache().hasKey({2000, 2002}));

    // malloc cache item 3 lead to pop cache item 2
    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003, 1004, 1005});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3, 4, 5}));
    ASSERT_EQ(match_info.reuse_length, 1);

    // cache item 1 is in cache
    ASSERT_TRUE(cache_manager.blockCache().hasKey({1000}));

    // cache item 2 is not in cache
    ASSERT_FALSE(cache_manager.blockCache().hasKey({2000, 2002}));
}

TEST_F(CacheManagerTest, testPopWithResident) {
    auto cache_config       = initConfig();
    cache_config.block_nums = 6;
    CacheManager cache_manager(cache_config, device_);
    auto         allocator = cache_manager.kvCacheAllocator();
    ASSERT_EQ(cache_manager.freeBlockNums(), 5);

    // Insert resident cache item
    auto [success1, index1] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_EQ(index1, std::vector<int>({1, 2}));
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);

    insertResidentCache(cache_manager, index1, {1000, 1002});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 0);
    ASSERT_TRUE(cache_manager.blockCache().isResident({1000}));

    // Insert cache item 2
    auto match_info = mallocWithCache(cache_manager, {2000, 2002, 2003});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({2, 3, 4}));
    ASSERT_EQ(match_info.reuse_length, 0);
    freeWithCache(cache_manager, match_info.cache_blocks, {2000, 2002, 2003});
    ASSERT_EQ(cache_manager.freeBlockNums(), 2);
    ASSERT_TRUE(cache_manager.blockCache().hasKey({2000, 2002}));
    ASSERT_TRUE(cache_manager.blockCache().isResident({1000}));

    // Malloc cache item 3 lead to pop cache item 2
    match_info = mallocWithCache(cache_manager, {2000, 2002, 2003, 2004, 2005});
    ASSERT_FALSE(match_info.reuse_length);
    // Cache item 1 is in cache
    ASSERT_TRUE(cache_manager.blockCache().hasKey({1000}));
    ASSERT_TRUE(cache_manager.blockCache().isResident({1000}));
    // Cache item 2 is not in cache
    ASSERT_FALSE(cache_manager.blockCache().hasKey({2000, 2002}));
}

TEST_F(CacheManagerTest, testResident) {
    auto cache_config       = initConfig();
    cache_config.block_nums = 100;
    CacheManager cache_manager(cache_config, device_);
    auto         allocator = cache_manager.kvCacheAllocator();
    ASSERT_EQ(cache_manager.freeBlockNums(), 99);

    // Malloc for resident block
    auto [success1, index1] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_EQ(index1, std::vector<int>({1, 2}));
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);

    // Insert resident cache item
    insertResidentCache(cache_manager, index1, {1000, 1002});
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 0);
    ASSERT_TRUE(cache_manager.blockCache().isResident({1000}));

    // Put not pop resident cache item
    auto match_info = mallocWithCache(cache_manager, {1000, 1002});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2}));
    ASSERT_EQ(match_info.reuse_length, 1);
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002});
    ASSERT_TRUE(cache_manager.blockCache().isResident({1000}));

    // Match resident cache item
    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(match_info.reuse_length, 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);

    // Put not pop resident cache item
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002, 1003});
    ASSERT_TRUE(cache_manager.blockCache().isResident({1000}));

    // Not match
    match_info = mallocWithCache(cache_manager, {2000, 2002, 2003});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({3, 4, 5}));
    ASSERT_EQ(match_info.reuse_length, 0);
}

TEST_F(CacheManagerTest, testSeqSizePerBlock) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 100;
    cache_config.seq_size_per_block = 2;
    CacheManager cache_manager(cache_config, device_);
    auto         allocator = cache_manager.kvCacheAllocator();
    ASSERT_EQ(cache_manager.freeBlockNums(), 99);

    // Malloc cache item 1
    auto match_info = mallocWithCache(cache_manager, {1000, 1002}, {}, false, 1);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1}));
    ASSERT_EQ(match_info.reuse_length, 0);
    // Insert cache item 1
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002});
    ASSERT_FALSE(cache_manager.blockCache().hasKey({1000, 1002}));

    // Malloc cache item 2
    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003}, {}, false, 2);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2}));
    ASSERT_EQ(match_info.reuse_length, 0);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 0);
    // Free cache item 2
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002, 1003});
    ASSERT_TRUE(cache_manager.blockCache().hasKey({1000, 1002}));

    // Malloc cache item 3
    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003, 1004, 1005}, {}, false, 3);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(match_info.reuse_length, 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(allocator->blockRefCounter().getRefCounter(3), 1);
    // Free cache item 3
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002, 1003, 1004, 1005});

    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003, 1004, 1005}, {}, false, 3);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(match_info.reuse_length, 4);
}

TEST_F(CacheManagerTest, testSetBlockValue) {
    // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
    CacheConfig  cache_config(KVCacheParam({2, 4, 1, 1, 2, rtp_llm::TYPE_INT8}));
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.freeBlockNums(), 3);

    vector<int8_t> k_vec(cache_config.kv_block_size, 1);
    vector<int8_t> v_vec(cache_config.kv_block_size, 1);
    auto           k_buffer = rtp_llm::vector2Buffer(k_vec);
    auto           v_buffer = rtp_llm::vector2Buffer(v_vec);
    cache_manager.setKVBlockValue(1, *k_buffer, *v_buffer);

    auto testFunc = [&](int block_index, int block_value) {
        auto [kbuffer, vbuffer] = cache_manager.getKVBlockValue(block_index);
        auto host_kbuffer       = device_->clone({*kbuffer, AllocationType::HOST});
        // auto host_vbuffer       = device_->clone({*vbuffer, AllocationType::HOST});
        ASSERT_EQ(cache_config.kv_block_size, host_kbuffer->size());
        // ASSERT_EQ(cache_config.kv_block_size, host_vbuffer->size());
        for (size_t i = 0; i < host_kbuffer->size(); i++) {
            ASSERT_EQ(block_value, host_kbuffer->data<int8_t>()[i]);
            // ASSERT_EQ(block_value, host_vbuffer->data<int8_t>()[i]);
        }

        for (size_t layer_id = 0; layer_id < cache_config.layer_num; layer_id++) {
            auto [kbuffer, vbuffer] = cache_manager.getKVBlockValue(block_index, layer_id);
            auto host_kbuffer       = device_->clone({*kbuffer, AllocationType::HOST});
            // auto host_vbuffer       = device_->clone({*vbuffer, AllocationType::HOST});
            ASSERT_EQ(cache_config.k_block_stride, host_kbuffer->size());
            // ASSERT_EQ(cache_config.k_block_stride, host_vbuffer->size());
            for (size_t i = 0; i < host_kbuffer->size(); i++) {
                ASSERT_EQ(block_value, host_kbuffer->data<int8_t>()[i]);
                // ASSERT_EQ(block_value, host_vbuffer->data<int8_t>()[i]);
            }
        }
    };
    testFunc(1, 1);

    cache_manager.blockCopy(1, 3);
    testFunc(3, 1);
}

TEST_F(CacheManagerTest, blockBatchCopy) {
    uint src_blocks_num = 4;
    uint dst_blocks_num = 9;
    uint blocks_num     = src_blocks_num + dst_blocks_num;
    // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
    CacheConfig  cache_config(KVCacheParam({2, 1 + blocks_num, 7, 128, 16, rtp_llm::TYPE_INT8}));
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.freeBlockNums(), blocks_num);

    for (int i = 1; i <= src_blocks_num; ++i) {
        vector<int8_t> k_vec(cache_config.kv_block_size, i);
        vector<int8_t> v_vec(cache_config.kv_block_size, i);
        auto           k_buffer = rtp_llm::vector2Buffer(k_vec);
        auto           v_buffer = rtp_llm::vector2Buffer(v_vec);
        cache_manager.setKVBlockValue(i, *k_buffer, *v_buffer);
    }

    auto testFunc = [&](int block_index, int block_value) {
        auto [kbuffer, vbuffer] = cache_manager.getKVBlockValue(block_index);
        auto host_kbuffer       = device_->clone({*kbuffer, AllocationType::HOST});
        // auto host_vbuffer       = device_->clone({*vbuffer, AllocationType::HOST});
        ASSERT_EQ(cache_config.kv_block_size, host_kbuffer->size());
        // ASSERT_EQ(cache_config.kv_block_size, host_vbuffer->size());
        for (size_t i = 0; i < host_kbuffer->size(); i++) {
            ASSERT_EQ(block_value, host_kbuffer->data<int8_t>()[i]);
            // ASSERT_EQ(block_value, host_vbuffer->data<int8_t>()[i]);
        }

        for (size_t layer_id = 0; layer_id < cache_config.layer_num; layer_id++) {
            auto [kbuffer, vbuffer] = cache_manager.getKVBlockValue(block_index, layer_id);
            auto host_kbuffer       = device_->clone({*kbuffer, AllocationType::HOST});
            // auto host_vbuffer       = device_->clone({*vbuffer, AllocationType::HOST});
            ASSERT_EQ(cache_config.k_block_stride, host_kbuffer->size());
            // ASSERT_EQ(cache_config.k_block_stride, host_vbuffer->size());
            for (size_t i = 0; i < host_kbuffer->size(); i++) {
                ASSERT_EQ(block_value, host_kbuffer->data<int8_t>()[i]);
                // ASSERT_EQ(block_value, host_vbuffer->data<int8_t>()[i]);
            }
        }
    };

    // check cache before copy
    for (int i = 1; i <= src_blocks_num; ++i) {
        SCOPED_TRACE("src before");
        testFunc(i, i);
    }
    for (int i = src_blocks_num + 1; i < blocks_num; ++i) {
        SCOPED_TRACE("dst before");
        testFunc(i, 0);
    }

    // do copy
    std::vector<BlockIdPair> copy_mapping;
    for (int i = src_blocks_num + 1; i < blocks_num; ++i) {
        int src_block = 1 + (i - src_blocks_num - 1) % src_blocks_num;
        copy_mapping.push_back({src_block, i});
    }
    cache_manager.blockBatchCopy(copy_mapping);

    // check cache after copy
    for (int i = 1; i <= src_blocks_num; ++i) {
        SCOPED_TRACE("src after");
        testFunc(i, i);
    }
    for (int i = src_blocks_num + 1; i < blocks_num; ++i) {
        SCOPED_TRACE("dst after");
        int expected_value = 1 + (i - src_blocks_num - 1) % src_blocks_num;
        testFunc(i, expected_value);
    }
}

TEST_F(CacheManagerTest, testBlockCacheHoldBlockNums) {
    // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
    CacheConfig  cache_config(KVCacheParam({2, 10, 1, 1, 1, rtp_llm::TYPE_INT8}));
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.block_cache_.holdBlockNums(), 0);
    ASSERT_EQ(cache_manager.availableBlockNums(), 9);

    auto match_info1 = mallocWithCache(cache_manager, {1000, 2000, 3000});
    ASSERT_EQ(match_info1.cache_blocks, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(cache_manager.availableBlockNums(), 6);

    freeWithCache(cache_manager, match_info1.cache_blocks, {1000, 2000, 3000});
    ASSERT_EQ(cache_manager.block_cache_.holdBlockNums(), 2);
    ASSERT_EQ(cache_manager.availableBlockNums(), 9);

    auto match_info2 = mallocWithCache(cache_manager, {1000, 1002, 1003});
    ASSERT_EQ(match_info2.cache_blocks, std::vector<int>({1, 3, 4}));
    ASSERT_EQ(match_info2.reuse_length, 1);
    ASSERT_EQ(cache_manager.availableBlockNums(), 6);

    auto match_info3 = mallocWithCache(cache_manager, {1000, 1004, 1005});
    ASSERT_EQ(match_info3.cache_blocks, std::vector<int>({1, 5, 6}));
    ASSERT_EQ(match_info3.reuse_length, 1);
    ASSERT_EQ(cache_manager.availableBlockNums(), 4);

    freeWithCache(cache_manager, match_info2.cache_blocks, {1000, 1002, 1003});
    ASSERT_EQ(cache_manager.block_cache_.holdBlockNums(), 3);
    ASSERT_EQ(cache_manager.availableBlockNums(), 6);

    freeWithCache(cache_manager, match_info3.cache_blocks, {1000, 1004, 1005});
    ASSERT_EQ(cache_manager.block_cache_.holdBlockNums(), 4);
    ASSERT_EQ(cache_manager.availableBlockNums(), 9);

    ASSERT_EQ(cache_manager.block_cache_.size(), 3);
    cache_manager.maybeFreeBlockFromCache(9);
    ASSERT_EQ(cache_manager.block_cache_.size(), 0);
    ASSERT_EQ(cache_manager.availableBlockNums(), 9);
    ASSERT_EQ(cache_manager.freeBlockNums(), 9);

    auto [success4, blocks4] = cache_manager.malloc({request_id, 3});
    ASSERT_EQ(blocks4.block_id, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(cache_manager.availableBlockNums(), 6);
    cache_manager.free(blocks4.block_id);
    ASSERT_EQ(cache_manager.availableBlockNums(), 9);
}

// engine flag disabled, use local cache, not match in dist kvcache
TEST_F(CacheManagerTest, testMatchImpl_SkipsDistKvCache_WhenEngineFlagDisabled) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    std::vector<int>                 token_ids  = {10, 20, 30, 40};
    auto                             cache_keys = std::vector<int64_t>{10, 20, 30, 40};
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/12,
                                                 token_ids,
                                                 cache_keys,
                                                 /*mm_bounds=*/{},
                                                 /*need_loss=*/false,
                                                 /*verbose=*/false,
                                                 /*adapter_name=*/"",
                                                 /*enable_3fs=*/true);

    auto mock_ptr                      = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_        = mock_ptr;
    cache_manager.enable_dist_kvcache_ = false;  // engine flag off

    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mock_ptr, getForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    auto match_info = cache_manager.mallocWithCache(malloc_info);
    ASSERT_GE(match_info.reuse_length, 0u);
}

// query-level enable_3fs=false, use local cache, not match in dist kvcache
TEST_F(CacheManagerTest, testMatchImpl_SkipsDistKvCache_When3FSDisabled) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    std::vector<int>                 token_ids  = {10, 20, 30, 40};
    auto                             cache_keys = std::vector<int64_t>{10, 20, 30, 40};
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/13,
                                                 token_ids,
                                                 cache_keys,
                                                 /*mm_bounds=*/{},
                                                 /*need_loss=*/false,
                                                 /*verbose=*/false,
                                                 /*adapter_name=*/"",
                                                 /*enable_3fs=*/false);

    auto mock_ptr                      = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_        = mock_ptr;
    cache_manager.enable_dist_kvcache_ = true;

    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mock_ptr, getForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    auto match_info = cache_manager.mallocWithCache(malloc_info);
    ASSERT_GE(match_info.reuse_length, 0u);
}

// need_loss = true, use local cache, not match in dist kvcache
TEST_F(CacheManagerTest, testMatchImpl_SkipsDistKvCache_WhenNeedLossTrue) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    std::vector<int>                 token_ids  = {10, 20, 30, 40};
    auto                             cache_keys = std::vector<int64_t>{10, 20, 30, 40};
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/11,
                                                 token_ids,
                                                 cache_keys,
                                                 /*mm_bounds=*/{},
                                                 /*need_loss=*/true,
                                                 /*verbose=*/false,
                                                 /*adapter_name=*/"",
                                                 /*enable_3fs=*/true);

    auto mock_ptr                      = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_        = mock_ptr;
    cache_manager.enable_dist_kvcache_ = true;

    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mock_ptr, getForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    auto match_info = cache_manager.mallocWithCache(malloc_info);
    ASSERT_GE(match_info.reuse_length, 0u);
}

// local matched length already complete, use local cache, not match in dist kvcache
TEST_F(CacheManagerTest, testMatchImpl_SkipsDistKvCache_WhenLocalMatchedAll) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    // Prepare 3 local cached blocks for keys derived from {100,200,300,400}
    auto [ok, idx] = cache_manager.mallocIndex({request_id, 3});
    ASSERT_TRUE(ok);
    freeWithCache(cache_manager, idx, {100, 200, 300, 400});

    // Now request only first 3 tokens -> cache_keys size = 3, local matched length = 3
    std::vector<int>                 token_ids  = {100, 200, 300};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/14,
                                                 token_ids,
                                                 cache_keys,
                                                 /*mm_bounds=*/{},
                                                 /*need_loss=*/false,
                                                 /*verbose=*/false,
                                                 /*adapter_name=*/"",
                                                 /*enable_3fs=*/true);

    auto mock_ptr                      = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_        = mock_ptr;
    cache_manager.enable_dist_kvcache_ = true;

    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_)).Times(0);
    EXPECT_CALL(*mock_ptr, getForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    auto match_info = cache_manager.mallocWithCache(malloc_info);
    // token_ids=3, seq_size_per_block=1 => floor((3-1)/1)=2 blocks reusable
    ASSERT_EQ(match_info.reuse_length, 2u);
    ASSERT_EQ(match_info.local_reuse_length, 2u);
    ASSERT_EQ(match_info.remote_reuse_length, 0u);
    // cache_blocks length equals reuse_block_num
    ASSERT_EQ(match_info.cache_blocks.size(), 2u);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({idx[0], idx[1]}));
}

// local completely not matched, remote matched partially
TEST_F(CacheManagerTest, testMatchImpl_LocalMatchedNone_RemoteMatchedPartially) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    // No local cache prepared to ensure local match = 0
    // Request 6 tokens -> (6-1)/1 = 5 max reusable by length
    std::vector<int>                 token_ids  = {100, 200, 300, 400, 500, 600};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/3,
                                                 token_ids,
                                                 cache_keys,
                                                 /*mm_bounds=*/{},
                                                 /*need_loss=*/false,
                                                 /*verbose=*/false,
                                                 /*adapter_name=*/"",
                                                 /*enable_3fs=*/true);

    auto mock_ptr                      = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_        = mock_ptr;
    cache_manager.enable_dist_kvcache_ = true;

    // local matched len is 0, remote total matched len returns 3; get succeeds
    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(3));
    EXPECT_CALL(*mock_ptr, getForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(true));

    auto match_info = cache_manager.mallocWithCache(malloc_info);
    // Reuse length should be 3; local part = 0, remote part = 3
    ASSERT_EQ(match_info.reuse_length, 3u);
    ASSERT_EQ(match_info.local_reuse_length, 0u);
    ASSERT_EQ(match_info.remote_reuse_length, 3u);
    // cache_blocks size equals reused block num
    ASSERT_EQ(match_info.cache_blocks.size(), 3u);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
}

// local none, remote matches all possible keys
TEST_F(CacheManagerTest, testMatchImpl_LocalMatchedNone_RemoteMatchedAll) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    // No local cache prepared (local matched = 0)
    // token_ids size 5 -> allowed reuse blocks = 4
    std::vector<int>                 token_ids  = {10, 20, 30, 40, 50};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/21,
                                                 token_ids,
                                                 cache_keys,
                                                 /*mm_bounds=*/{},
                                                 /*need_loss=*/false,
                                                 /*verbose=*/false,
                                                 /*adapter_name=*/"",
                                                 /*enable_3fs=*/true);

    auto mock_ptr                      = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_        = mock_ptr;
    cache_manager.enable_dist_kvcache_ = true;

    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(5));  // remote matched full length
    EXPECT_CALL(*mock_ptr, getForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(Invoke([=](const std::vector<int64_t>&        actual_cache_keys,
                             const std::vector<int32_t>&        block_indices,
                             size_t                             ignore_block_num,
                             int64_t                            request_id,
                             std::map<std::string, std::string> extra_metas) -> bool {
            EXPECT_EQ(actual_cache_keys.size(), 5u);
            EXPECT_EQ(actual_cache_keys, (std::vector<int64_t>(cache_keys.begin(), cache_keys.begin() + 5)));
            // need_block_num = 5
            EXPECT_EQ(block_indices.size(), 5u);
            EXPECT_EQ(block_indices, (std::vector<int32_t>{1, 2, 3, 4, 5}));
            EXPECT_EQ(ignore_block_num, 0u);
            EXPECT_EQ(request_id, malloc_info.request_id);
            return true;
        }));

    auto match_info = cache_manager.mallocWithCache(malloc_info);
    // Cropped by token limit: reuse blocks = 4
    ASSERT_EQ(match_info.reuse_length, 4u);
    ASSERT_EQ(match_info.local_reuse_length, 0u);
    ASSERT_EQ(match_info.remote_reuse_length, 4u);
    ASSERT_EQ(match_info.cache_blocks.size(), 4u);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3, 4}));
}

// local partially matched, remote matched partially
TEST_F(CacheManagerTest, testMatchImpl_LocalMatchedPartially_RemoteMatchedPartially) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    // Prepare 1 local cached block with id 1
    {
        auto [ok, idx] = cache_manager.mallocIndex({request_id, 1});
        ASSERT_TRUE(ok);
        freeWithCache(cache_manager, idx, {100, 101});
    }

    // token_ids size 5 -> allowed reuse blocks = 4; remote total match returns 3
    std::vector<int>                 token_ids  = {100, 200, 300, 400, 500};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/22,
                                                 token_ids,
                                                 cache_keys,
                                                 /*mm_bounds=*/{},
                                                 /*need_loss=*/false,
                                                 /*verbose=*/false,
                                                 /*adapter_name=*/"",
                                                 /*enable_3fs=*/true);

    auto mock_ptr                      = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_        = mock_ptr;
    cache_manager.enable_dist_kvcache_ = true;

    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(3));  // total matched 3 (local 1 + remote 2)
    EXPECT_CALL(*mock_ptr, getForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(Invoke([=](const std::vector<int64_t>&        actual_cache_keys,
                             const std::vector<int32_t>&        block_indices,
                             size_t                             ignore_block_num,
                             int64_t                            request_id,
                             std::map<std::string, std::string> extra_metas) -> bool {
            EXPECT_EQ(actual_cache_keys.size(), 3u);
            EXPECT_EQ(block_indices.size(), 2u);  // need_block_num = 2
            // With block 1 already held, remote allocates {2,3}
            EXPECT_EQ(block_indices, (std::vector<int32_t>{2, 3}));
            EXPECT_EQ(ignore_block_num, 1u);
            EXPECT_EQ(request_id, malloc_info.request_id);
            return true;
        }));

    auto match_info = cache_manager.mallocWithCache(malloc_info);
    ASSERT_EQ(match_info.reuse_length, 3u);
    ASSERT_EQ(match_info.local_reuse_length, 1u);
    ASSERT_EQ(match_info.remote_reuse_length, 2u);
    ASSERT_EQ(match_info.cache_blocks.size(), 3u);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
}

// local partially matched, remote matched all possible keys
TEST_F(CacheManagerTest, testMatchImpl_LocalMatchedPartially_RemoteMatchedAll) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    // Prepare 2 local cached blocks: {1,2}
    std::vector<int> local_idx;
    {
        auto [ok, idx] = cache_manager.mallocIndex({request_id, 2});
        ASSERT_TRUE(ok);
        local_idx = idx;
        freeWithCache(cache_manager, idx, {200, 201, 202});
    }

    // token_ids size 6 -> allowed reuse blocks = 5; remote total match returns 6 (full)
    std::vector<int>                 token_ids  = {200, 300, 400, 500, 600, 700};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/23,
                                                 token_ids,
                                                 cache_keys,
                                                 /*mm_bounds=*/{},
                                                 /*need_loss=*/false,
                                                 /*verbose=*/false,
                                                 /*adapter_name=*/"",
                                                 /*enable_3fs=*/true);

    auto mock_ptr                      = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_        = mock_ptr;
    cache_manager.enable_dist_kvcache_ = true;

    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(6));  // total matched = 6
    EXPECT_CALL(*mock_ptr, getForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(Invoke([=](const std::vector<int64_t>&        actual_cache_keys,
                             const std::vector<int32_t>&        block_indices,
                             size_t                             ignore_block_num,
                             int64_t                            request_id,
                             std::map<std::string, std::string> extra_metas) -> bool {
            EXPECT_EQ(actual_cache_keys.size(), 6u);
            EXPECT_EQ(block_indices.size(), 5u);  // need_block_num = 6 - local(1) = 5
            // With blocks {1,2} held, remote allocates {3,4,5,6,7}
            EXPECT_EQ(block_indices, (std::vector<int32_t>{3, 4, 5, 6, 7}));
            EXPECT_EQ(ignore_block_num, 1u);
            EXPECT_EQ(request_id, malloc_info.request_id);
            return true;
        }));

    auto match_info = cache_manager.mallocWithCache(malloc_info);
    // Cropped by token limit: 6 blocks total
    ASSERT_EQ(match_info.reuse_length, 5u);
    ASSERT_EQ(match_info.local_reuse_length, 1u);
    ASSERT_EQ(match_info.remote_reuse_length, 4u);
    ASSERT_EQ(match_info.cache_blocks.size(), 5u);
    // local hold {1}, remote hold {3, 4, 5, 6}
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 3, 4, 5, 6}));
}

// Loss present: reuse_block_num should decrement by 1
TEST_F(CacheManagerTest, testMatchImpl_HasLoss_DecrementsReuseBlockNum_LocalMatchedAll) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    // Insert cache with loss for 3 tokens
    auto [ok, idx] = cache_manager.mallocIndex({request_id, 3});
    ASSERT_TRUE(ok);
    // To cache 3 blocks locally, provide 4 tokens (token_len = 3) and 3 loss values
    freeWithCache(cache_manager, idx, {10, 20, 30, 40}, /*loss=*/{0.1f, 0.2f, 0.3f});

    // Request length 5 -> (5-1)=4 allowed; local matched length is 3; due to loss, reuse_block_num becomes 2
    auto match_info = mallocWithCache(cache_manager,
                                      {10, 20, 30, 40, 50},
                                      /*mm_bounds=*/{},
                                      /*need_loss=*/true,
                                      /*need_block_num=*/0);
    ASSERT_EQ(match_info.reuse_length, 2u);
    // With loss present, reuse_block_num reduced by 1, all from local
    ASSERT_EQ(match_info.local_reuse_length, 2u);
    ASSERT_EQ(match_info.remote_reuse_length, 0u);
    // No top-up here, should return exactly 2 reused blocks (first two cached indices)
    ASSERT_EQ(match_info.cache_blocks.size(), 2u);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({idx[0], idx[1]}));
    ASSERT_EQ(match_info.loss.size(), 2u);
    ASSERT_EQ(match_info.loss, std::vector<float>({0.1f, 0.2f}));
}

// When matched length > reuse_block_num due to mm_bounds cropping,
// the tail of match_result.block_indices should be freed (not returned).
TEST_F(CacheManagerTest, testMatchImpl_ReuseBlockNumLessThanMatchBlockNum_LocalMatchedAll) {
    auto         cache_config = initConfig();
    CacheManager cache_manager(cache_config, device_);

    // Prepare 3 local cached blocks {1,2,3}
    auto [ok, idx] = cache_manager.mallocIndex({request_id, 3});
    ASSERT_TRUE(ok);
    freeWithCache(cache_manager, idx, {100, 200, 300, 400});

    // Request same 4 tokens; local matched length = 3. Crop reuse_length to 1 via mm_bounds.
    // With seq_size_per_block=1, mm_bounds {1,100} will crop any reuse_length in (1,101) to 1.
    std::vector<int> token_ids = {100, 200, 300, 400};
    auto             mm_bounds = std::vector<std::vector<int>>{{1, 100}};

    auto match_info = mallocWithCache(cache_manager, token_ids, mm_bounds, /*need_loss=*/false, /*need_block_num=*/1);

    ASSERT_EQ(match_info.reuse_length, 1);
    ASSERT_EQ(match_info.local_reuse_length, 1u);
    ASSERT_EQ(match_info.remote_reuse_length, 0u);
    ASSERT_EQ(match_info.cache_blocks.size(), 1u);
    // Only the first matched block should remain
    ASSERT_EQ(match_info.cache_blocks[0], idx[0]);
}

// token_ids.size() <= 1 -> freeImpl all, no cache entry
TEST_F(CacheManagerTest, testInsertIntoCache_TokenLenLessThan1_FreeAllBlocks) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    size_t free0   = cache_manager.freeBlockNums();
    auto [ok, idx] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_TRUE(ok);
    ASSERT_EQ(cache_manager.freeBlockNums(), free0 - 2);

    std::vector<int>       token_ids  = {1000};  // size <= 1 triggers full free
    auto                   cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::FreeInfo free_info(request_id, token_ids, cache_keys, idx);
    cache_manager.insertIntoCache(free_info);

    ASSERT_EQ(cache_manager.freeBlockNums(), free0);
    ASSERT_FALSE(cache_manager.blockCache().hasKey({1000}));
}

// put to block cache
TEST_F(CacheManagerTest, testInsertIntoCache_PutToBlockCache) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    auto [ok, idx] = cache_manager.mallocIndex({request_id, 3});
    ASSERT_TRUE(ok);
    size_t free_after_malloc = cache_manager.freeBlockNums();

    // token_ids size = 3 -> token_len = 2 -> block_len = 2
    std::vector<int>       token_ids  = {200, 201, 202};
    auto                   cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::FreeInfo free_info(request_id, token_ids, cache_keys, idx);
    cache_manager.insertIntoCache(free_info);

    // One tail block freed back
    ASSERT_EQ(cache_manager.freeBlockNums(), free_after_malloc + 1);
    // Cache holds the 2-block prefix key {200,201}
    ASSERT_TRUE(cache_manager.blockCache().hasKey({200, 201}));
}

// put to block cache twice
TEST_F(CacheManagerTest, testInsertIntoCache_PutToBlockCacheTwice) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);
    auto         allocator = cache_manager.kvCacheAllocator();

    auto [ok, idx] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_TRUE(ok);
    cache_manager.incrBlockRefCounter(idx);
    ASSERT_TRUE(allocator->blockRefCounter().getRefCounter(idx[0]) == 2);
    ASSERT_TRUE(allocator->blockRefCounter().getRefCounter(idx[1]) == 2);

    std::vector<int>       token_ids  = {200, 201, 202};
    auto                   cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::FreeInfo free_info(request_id, token_ids, cache_keys, idx);
    cache_manager.insertIntoCache(free_info);

    ASSERT_TRUE(cache_manager.blockCache().hasKey({200, 201}));
    ASSERT_TRUE(allocator->blockRefCounter().getRefCounter(idx[0]) == 2);
    ASSERT_TRUE(allocator->blockRefCounter().getRefCounter(idx[1]) == 2);

    cache_manager.insertIntoCache(free_info);
    ASSERT_TRUE(cache_manager.blockCache().hasKey({200, 201}));
    // ref count should be decremented
    ASSERT_TRUE(allocator->blockRefCounter().getRefCounter(idx[0]) == 1);
    ASSERT_TRUE(allocator->blockRefCounter().getRefCounter(idx[1]) == 1);
}

// loss present -> do NOT put to dist
TEST_F(CacheManagerTest, testInsertIntoCache_LossNotEmpty_PutToBlockCache_NotPutToDistKvCache) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    auto mock_ptr                      = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_        = mock_ptr;
    cache_manager.enable_dist_kvcache_ = true;

    auto [ok, idx] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_TRUE(ok);
    std::vector<int>   token_ids  = {400, 401, 402};  // block_len = 2
    auto               cache_keys = constructCacheKey(cache_manager, token_ids);
    std::vector<float> loss       = {0.1f, 0.2f};

    EXPECT_CALL(*mock_ptr, putForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    CacheManager::FreeInfo free_info(
        request_id, token_ids, cache_keys, idx, loss, /*adapter_name=*/"test_adapter_name", /*enable_3fs=*/true);
    cache_manager.insertResidentCache(free_info);
}

// loss empty and dist enabled -> put to dist called with prefix block_len
TEST_F(CacheManagerTest, testInsertIntoCache_PutToDistKvCache) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    auto mock_ptr                      = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_        = mock_ptr;
    cache_manager.enable_dist_kvcache_ = true;

    auto [ok, idx] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_TRUE(ok);

    std::vector<int> token_ids  = {300, 301, 302};  // block_len = 2
    auto             cache_keys = constructCacheKey(cache_manager, token_ids);

    std::string adapter_name                   = "test_adapter_name";
    cache_manager.lora_info_map_[adapter_name] = "test_ckpt_path";

    EXPECT_CALL(*mock_ptr, putForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(Invoke([&](const std::vector<int64_t>& actual_cache_keys,
                             const std::vector<int32_t>& actual_block_indices,
                             size_t                      ignore_block_num,
                             int64_t                     actual_request_id,
                             MapStrStr                   extra_metas) -> bool {
            EXPECT_EQ(actual_cache_keys.size(), 2u);
            EXPECT_EQ(actual_block_indices.size(), 2u);
            EXPECT_EQ(ignore_block_num, 0u);
            EXPECT_EQ(actual_request_id, request_id);
            EXPECT_TRUE(extra_metas.at("LORA_CKPT_PATH") == std::to_string(std::hash<std::string>()("test_ckpt_path")));
            return true;
        }));

    CacheManager::FreeInfo free_info(
        request_id, token_ids, cache_keys, idx, /*loss=*/{}, adapter_name, /*enable_3fs=*/true);
    cache_manager.insertResidentCache(free_info);
}

// local already complete => return
TEST_F(CacheManagerTest, testMatchInDistKvCache_LocalMatchedAll) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    std::vector<int>                 token_ids  = {10, 20, 30};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/1, token_ids, cache_keys, /*mm_bounds=*/{});
    BlockCache::MatchResult          match_result;
    match_result.block_indices = {1, 2, 3};
    // Simulate real path: local matched blocks are already held by this query
    cache_manager.incrRefCounter(match_result.block_indices);

    cache_manager.matchInDistKvCache(malloc_info, match_result);
    ASSERT_EQ(match_result.block_indices.size(), 3u);
    ASSERT_EQ(match_result.block_indices, std::vector<int>({1, 2, 3}));
}

// dist_kvcache_ == nullptr => early return
TEST_F(CacheManagerTest, testMatchInDistKvCache_NoDistKvCache_DistKvCacheIsNull) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    std::vector<int>                 token_ids  = {10, 20, 30};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/1, token_ids, cache_keys, /*mm_bounds=*/{});
    BlockCache::MatchResult          match_result;
    match_result.block_indices = {5};
    cache_manager.incrRefCounter(match_result.block_indices);

    cache_manager.matchInDistKvCache(malloc_info, match_result);
    ASSERT_EQ(match_result.block_indices.size(), 1u);
    ASSERT_EQ(match_result.block_indices, std::vector<int>({5}));
}

// matched length <= 0 => early return
TEST_F(CacheManagerTest, testMatchInDistKvCache_MatchedLenLessThanZero) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    auto mock_ptr               = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_ = mock_ptr;

    std::vector<int>                 token_ids  = {10, 20, 30};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/2, token_ids, cache_keys, /*mm_bounds=*/{});
    BlockCache::MatchResult          match_result;
    match_result.block_indices = {5};
    cache_manager.incrRefCounter(match_result.block_indices);

    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(0));

    cache_manager.matchInDistKvCache(malloc_info, match_result);
    ASSERT_EQ(match_result.block_indices.size(), 1u);
    ASSERT_EQ(match_result.block_indices, std::vector<int>({5}));
}

// need_block_num <= 0 (remote total matched length == local matched length) => return
TEST_F(CacheManagerTest, testMatchInDistKvCache_NoAdditionalNeed) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    auto mock_ptr               = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_ = mock_ptr;

    std::vector<int>                 token_ids  = {10, 20, 30};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/3, token_ids, cache_keys, /*mm_bounds=*/{});
    BlockCache::MatchResult          match_result;
    match_result.block_indices = {1, 2};
    cache_manager.incrRefCounter(match_result.block_indices);

    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(2));  // equal to local

    cache_manager.matchInDistKvCache(malloc_info, match_result);
    ASSERT_EQ(match_result.block_indices.size(), 2u);
    ASSERT_EQ(match_result.block_indices, std::vector<int>({1, 2}));
}

// malloc failed => return (no change)
TEST_F(CacheManagerTest, testMatchInDistKvCache_MallocFailed) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 2;  // free blocks will be 1
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    auto mock_ptr               = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_ = mock_ptr;

    std::vector<int>                 token_ids  = {10, 20, 30, 40};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/4, token_ids, cache_keys, /*mm_bounds=*/{});
    BlockCache::MatchResult          match_result;
    match_result.block_indices = {};

    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(3));  // need_block_num = 3 > free (=1)

    cache_manager.matchInDistKvCache(malloc_info, match_result);
    ASSERT_EQ(match_result.block_indices.size(), 0u);
}

// getForAllRank failed => free allocated blocks and return
TEST_F(CacheManagerTest, testMatchInDistKvCache_GetFailed) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    auto mock_ptr               = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_ = mock_ptr;

    std::vector<int>                 token_ids  = {10, 20, 30, 40};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/5, token_ids, cache_keys, /*mm_bounds=*/{});
    BlockCache::MatchResult          match_result;
    match_result.block_indices = {7};
    cache_manager.incrRefCounter(match_result.block_indices);

    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(3));  // need_block_num = 2
    EXPECT_CALL(*mock_ptr, getForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(::testing::Return(false));

    auto free_before = cache_manager.freeBlockNums();
    cache_manager.matchInDistKvCache(malloc_info, match_result);
    auto free_after = cache_manager.freeBlockNums();

    ASSERT_EQ(match_result.block_indices.size(), 1u);
    ASSERT_EQ(free_before, free_after);  // allocated blocks were freed
    ASSERT_EQ(match_result.block_indices, std::vector<int>({7}));
}

// success path => update matched length and append blocks
TEST_F(CacheManagerTest, testMatchInDistKvCache_Success) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    auto mock_ptr               = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_ = mock_ptr;

    std::vector<int>                 token_ids  = {10, 20, 30, 40, 50};
    auto                             cache_keys = constructCacheKey(cache_manager, token_ids);
    CacheManager::AdvancedMallocInfo malloc_info(/*request_id=*/6, token_ids, cache_keys, /*mm_bounds=*/{});

    BlockCache::MatchResult match_result;
    // Allocate the two locally matched blocks for this same request to remove them from the free list
    KVCacheAllocator::SimpleMallocInfo block_malloc_info(/*request_id=*/6, 2);
    auto [success, resource] = cache_manager.malloc(block_malloc_info);
    ASSERT_TRUE(success);
    ASSERT_EQ(resource.block_id.size(), 2u);
    match_result.block_indices = resource.block_id;

    EXPECT_CALL(*mock_ptr, matchForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(Invoke([=](const std::vector<int64_t>&        actual_cache_keys,
                             size_t                             ignore_block_num,
                             int64_t                            request_id,
                             std::map<std::string, std::string> extra_metas) -> int {
            EXPECT_EQ(actual_cache_keys, cache_keys);
            EXPECT_EQ(ignore_block_num, match_result.block_indices.size());
            EXPECT_EQ(request_id, malloc_info.request_id);
            return 4;  // need_block_num = 2
        }));
    EXPECT_CALL(*mock_ptr, getForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(Invoke([=](const std::vector<int64_t>&        actual_cache_keys,
                             const std::vector<int32_t>&        block_indices,
                             size_t                             ignore_block_num,
                             int64_t                            request_id,
                             std::map<std::string, std::string> extra_metas) -> bool {
            EXPECT_EQ(actual_cache_keys.size(), 4);
            EXPECT_EQ(actual_cache_keys, (std::vector<int64_t>(cache_keys.begin(), cache_keys.begin() + 4)));
            // remote allocation should fill the next free blocks {3,4}
            EXPECT_EQ(block_indices.size(), 2);
            EXPECT_EQ(block_indices, (std::vector<int32_t>{3, 4}));
            EXPECT_EQ(ignore_block_num, match_result.block_indices.size());
            EXPECT_EQ(request_id, malloc_info.request_id);
            return true;
        }));

    cache_manager.matchInDistKvCache(malloc_info, match_result);
    ASSERT_EQ(match_result.block_indices.size(), 4u);
    // first two indices remain the original local ones
    ASSERT_EQ(match_result.block_indices, std::vector<int>({1, 2, 3, 4}));
}

// empty keys => return true, do not call dist_kvcache_
TEST_F(CacheManagerTest, testPutToDistKvCache_EmptyKeys_ReturnsTrue_NoCall) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    auto mock_ptr               = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_ = mock_ptr;

    std::vector<int64_t> cache_keys{};  // empty
    std::vector<int32_t> block_indices{1, 2, 3};
    size_t               ignore_block_num = 0;
    int64_t              request_id       = 100;
    std::string          adapter_name     = "test_adapter_name";

    EXPECT_CALL(*mock_ptr, putForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    bool ok = cache_manager.putToDistKvCache(cache_keys, block_indices, ignore_block_num, request_id, adapter_name);
    ASSERT_TRUE(ok);
}

// size mismatch => return false, do not call dist_kvcache_
TEST_F(CacheManagerTest, testPutToDistKvCache_SizeMismatch_ReturnsFalse_NoCall) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    auto mock_ptr               = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_ = mock_ptr;

    std::vector<int64_t> cache_keys{1, 2};
    std::vector<int32_t> block_indices{1};  // mismatch
    size_t               ignore_block_num = 0;
    int64_t              request_id       = 101;
    std::string          adapter_name     = "test_adapter_name";

    EXPECT_CALL(*mock_ptr, putForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .Times(0);

    bool ok = cache_manager.putToDistKvCache(cache_keys, block_indices, ignore_block_num, request_id, adapter_name);
    ASSERT_FALSE(ok);
}

// dist_kvcache_ == nullptr => return false
TEST_F(CacheManagerTest, testPutToDistKvCache_NoDistKvCache_ReturnsFalse) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    std::vector<int64_t> cache_keys{10, 20};
    std::vector<int32_t> block_indices{1, 2};
    size_t               ignore_block_num = 1;
    int64_t              request_id       = 102;
    std::string          adapter_name     = "adapterC";

    bool ok = cache_manager.putToDistKvCache(cache_keys, block_indices, ignore_block_num, request_id, adapter_name);
    ASSERT_FALSE(ok);
}

// failure path => returns false when dist returns false
TEST_F(CacheManagerTest, testPutToDistKvCache_CallsAndFails) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    auto mock_ptr               = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_ = mock_ptr;

    std::vector<int64_t> cache_keys{7, 8};
    std::vector<int32_t> block_indices{9, 10};
    size_t               ignore_block_num = 0;
    int64_t              request_id       = 104;
    std::string          adapter_name     = "test_adapter_name";

    EXPECT_CALL(*mock_ptr, putForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(Return(false));

    bool ok = cache_manager.putToDistKvCache(cache_keys, block_indices, ignore_block_num, request_id, adapter_name);
    ASSERT_FALSE(ok);
}

// success path => calls dist_kvcache_->putForAllRank and returns true
TEST_F(CacheManagerTest, testPutToDistKvCache_CallsAndSucceeds) {
    auto cache_config               = initConfig();
    cache_config.block_nums         = 10;
    cache_config.seq_size_per_block = 1;
    CacheManager cache_manager(cache_config, device_);

    auto mock_ptr               = std::make_shared<MockDistKvCache>();
    cache_manager.dist_kvcache_ = mock_ptr;

    std::vector<int64_t> cache_keys{100, 300, 600};
    std::vector<int32_t> block_indices{3, 4, 5};
    size_t               ignore_block_num      = 1;
    int64_t              request_id            = 103;
    std::string          adapter_name          = "test_adapter_name";
    cache_manager.lora_info_map_[adapter_name] = "test_lora_ckpt_path";

    EXPECT_CALL(*mock_ptr, putForAllRank(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillOnce(Invoke([&](const std::vector<int64_t>& actual_cache_keys,
                             const std::vector<int32_t>& actual_block_indices,
                             size_t                      actual_ignore_block_num,
                             int64_t                     actual_request_id,
                             MapStrStr                   extra_metas) -> bool {
            EXPECT_EQ(actual_cache_keys, cache_keys);
            EXPECT_EQ(actual_block_indices, block_indices);
            EXPECT_EQ(actual_ignore_block_num, ignore_block_num);
            EXPECT_EQ(actual_request_id, request_id);
            EXPECT_TRUE(extra_metas.at("LORA_CKPT_PATH")
                        == std::to_string(std::hash<std::string>()("test_lora_ckpt_path")));
            return true;
        }));

    bool ok = cache_manager.putToDistKvCache(cache_keys, block_indices, ignore_block_num, request_id, adapter_name);
    ASSERT_TRUE(ok);
}

}  // namespace rtp_llm
