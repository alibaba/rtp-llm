#include "rtp_llm/cpp/cache/CacheManager.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;


namespace rtp_llm {

class CacheManagerTest: public DeviceTestBase {
protected:
    CacheConfig initConfig() {
        // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
        CacheConfig config(KVCacheParam({1, 4, 1, 1, 1, rtp_llm::TYPE_INT8}));
        return config;
    }

    std::vector<int64_t> constructCacheKey(CacheManager& cache_manager, const vector<int>& token_ids) {
        auto seq_size_per_block = cache_manager.config_.seq_size_per_block;
        auto total_blocks = token_ids.size() / seq_size_per_block;
        vector<int64_t> cache_keys;
        int64_t hash = 0;
        for (int index = 0; index < total_blocks; index++) {
            auto start_pos = token_ids.begin() + index * seq_size_per_block;
            hash = std::accumulate(start_pos, start_pos + seq_size_per_block, hash, std::plus<int>());
            cache_keys.push_back(hash);
        }
        return cache_keys;
    }

    CacheManager::MatchInfo mallocWithCache(CacheManager& cache_manager, const vector<int>& token_ids,
                                            const vector<vector<int>>& mm_bounds = {}, bool need_loss = false, int need_block_num = -1) {
        if (need_block_num == -1) {
            need_block_num = token_ids.size();
        }
        auto cache_keys = constructCacheKey(cache_manager, token_ids);
        CacheManager::AdvancedMallocInfo malloc_info(request_id, token_ids, cache_keys, mm_bounds, need_loss);
        auto match_info = cache_manager.mallocWithCache(malloc_info);
        if (match_info.cache_blocks.size() < need_block_num) {
            auto [success, index] = cache_manager.mallocIndex({request_id, uint32_t(need_block_num - match_info.cache_blocks.size())});
            if (success) {
                match_info.cache_blocks.insert(match_info.cache_blocks.end(), index.begin(), index.end());
            } else {
                cache_manager.free(match_info.cache_blocks);
                return {0, {}, {}};
            }
        }
        return match_info;
    }

    void freeWithCache(CacheManager& cache_manager, const std::vector<int>& block_indices,
                       const vector<int>& token_ids, const vector<float>& loss = {}) {
        auto cache_keys = constructCacheKey(cache_manager, token_ids);
        CacheManager::FreeInfo free_info(request_id, token_ids, cache_keys, block_indices, loss);
        cache_manager.freeWithCache(free_info);
    }

    void insertResidentCache(CacheManager& cache_manager, const std::vector<int>& block_indices,
                             const vector<int>& token_ids) {
        auto cache_keys = constructCacheKey(cache_manager, token_ids);
        CacheManager::FreeInfo free_info(request_id, token_ids, cache_keys, block_indices);
        cache_manager.insertResidentCache(free_info);
    }

protected:
    int64_t request_id = 0;
};

TEST_F(CacheManagerTest, testSimple) {
    auto         cache_config = initConfig();
    CacheManager cache_manager(cache_config, device_);

    ASSERT_EQ(cache_manager.freeBlockNums(), 3);

    auto [success1, index1] = cache_manager.mallocIndex({request_id, 1});
    ASSERT_TRUE(success1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);

    auto [success2, index2] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_TRUE(success2);
    auto [success3, _] = cache_manager.mallocIndex({request_id, 1});
    ASSERT_FALSE(success3);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);

    cache_manager.free(index1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 0);

    cache_manager.free(index2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);

    ASSERT_EQ(cache_manager.freeBlockNums(), 3);
}

TEST_F(CacheManagerTest, testAllocateWithFreeCache) {
    auto         cache_config = initConfig();
    CacheManager cache_manager(cache_config, device_);

    auto match_info = mallocWithCache(cache_manager, {1000, 2000, 3000});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
    ASSERT_FALSE(match_info.reuse_length);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);

    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 2000, 3000});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);

    auto [success2, index2] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_EQ(index2, std::vector<int>({1, 2}));
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.freeBlockNums(), 1);

    // paritial fallback case
    freeWithCache(cache_manager, index2, {1000, 2000, 3000, 4000, 5000});
    ASSERT_TRUE(cache_manager.blockCache().hasKey({1000, 2000}));
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

    ASSERT_EQ(cache_manager.freeBlockNums(), 3);
    auto [success1, index1] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_EQ(index1, std::vector<int>({1, 2}));
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 0);

    freeWithCache(cache_manager, index1, {1000, 1002});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 1);

    auto match_info = mallocWithCache(cache_manager, {1000, 1002});
    ASSERT_EQ(cache_manager.freeBlockNums(), 1);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2}));
    ASSERT_EQ(match_info.reuse_length, 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);

    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);

    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(match_info.reuse_length, 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.cacheItemNum(), 1);

    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002, 1003});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 2);
}

TEST_F(CacheManagerTest, testAllocateWithMultimodalReuse) {
    auto         cache_config = initConfig();
    cache_config.block_nums = 10;
    cache_config.seq_size_per_block = 2;
    CacheManager cache_manager(cache_config, device_);

    ASSERT_EQ(cache_manager.freeBlockNums(), 9);
    auto [success1, index1] = cache_manager.mallocIndex({request_id, 4});
    ASSERT_EQ(index1, std::vector<int>({1, 2, 3, 4}));
    ASSERT_EQ(cache_manager.cacheItemNum(), 0);

    freeWithCache(cache_manager, index1, {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 1);

    auto match_info = mallocWithCache(cache_manager, {1000, 1001, 1002, 1003, 1004, 1005, 1006}, {{5, 2}}, false, 4);
    ASSERT_EQ(cache_manager.freeBlockNums(), 4);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 4, 5}));
    ASSERT_EQ(match_info.reuse_length, 4);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(5), 1);

    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1001, 1002, 1003, 1004, 1005, 1006});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 1);

    match_info = mallocWithCache(cache_manager, {1000, 1001, 1002, 1003, 1004, 1015}, {{1, 2}, {3, 2}}, false, 3);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({4, 5, 6}));
    ASSERT_EQ(match_info.reuse_length, 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(5), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(6), 1);
    ASSERT_EQ(cache_manager.cacheItemNum(), 1);

    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1001, 1002, 1003, 1004, 1015});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(5), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(6), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 2);
}

TEST_F(CacheManagerTest, testMatchMaxLen) {
    auto cache_config       = initConfig();
    cache_config.block_nums = 100;
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.freeBlockNums(), 99);

    // malloc cache item 1
    auto match_info = mallocWithCache(cache_manager, {1000, 1002});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2}));
    ASSERT_EQ(match_info.reuse_length, 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);

    // insert cache item 1
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);

    // malloc cache item 2
    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(match_info.reuse_length, 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 0);

    // insert cache item 2
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002, 1003});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 0);

    // malloc cache item 3
    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003, 1004});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3, 4}));
    ASSERT_EQ(match_info.reuse_length, 2);  // Assuming 2 blocks were reused, replace with actual logic
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 3);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(6), 0);

    // insert cache item 3
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002, 1003, 1004});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 3);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(6), 0);

    // trigger match max len cache item
    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003, 1004});
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3, 4}));
    ASSERT_EQ(match_info.reuse_length, 3);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 4);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 3);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(6), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(7), 0);
}

TEST_F(CacheManagerTest, testPopNoResidentCacheItem) {
    auto         cache_config = initConfig();
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.freeBlockNums(), 3);

    // malloc cache item 1
    auto [success1, index1] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_TRUE(success1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);

    // insert cache item 1
    freeWithCache(cache_manager, index1, {1000, 1002});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);

    // trigger reuse cache, pop cache item 1, malloc from free failed
    auto match_info = mallocWithCache(cache_manager, {1000, 1002, 1003, 1004});
    ASSERT_EQ(match_info.reuse_length, 0);
    ASSERT_EQ(cache_manager.freeBlockNums(), 3);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);

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
    ASSERT_EQ(cache_manager.freeBlockNums(), 5);

    // Insert resident cache item
    auto [success1, index1] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_EQ(index1, std::vector<int>({1, 2}));
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);

    insertResidentCache(cache_manager, index1, {1000, 1002});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);
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
    ASSERT_EQ(cache_manager.freeBlockNums(), 99);

    // Malloc for resident block
    auto [success1, index1] = cache_manager.mallocIndex({request_id, 2});
    ASSERT_EQ(index1, std::vector<int>({1, 2}));
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);

    // Insert resident cache item
    insertResidentCache(cache_manager, index1, {1000, 1002});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);
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
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);

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
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    // Free cache item 2
    freeWithCache(cache_manager, match_info.cache_blocks, {1000, 1002, 1003});
    ASSERT_TRUE(cache_manager.blockCache().hasKey({1000, 1002}));

    // Malloc cache item 3
    match_info = mallocWithCache(cache_manager, {1000, 1002, 1003, 1004, 1005}, {}, false, 3);
    ASSERT_EQ(match_info.cache_blocks, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(match_info.reuse_length, 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
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
        auto host_vbuffer       = device_->clone({*vbuffer, AllocationType::HOST});
        ASSERT_EQ(cache_config.kv_block_size, host_kbuffer->size());
        ASSERT_EQ(cache_config.kv_block_size, host_vbuffer->size());
        for (size_t i = 0; i < host_kbuffer->size(); i++) {
            ASSERT_EQ(block_value, host_kbuffer->data<int8_t>()[i]);
            ASSERT_EQ(block_value, host_vbuffer->data<int8_t>()[i]);
        }

        for (size_t layer_id = 0; layer_id < cache_config.layer_num; layer_id++) {
            auto [kbuffer, vbuffer] = cache_manager.getKVBlockValue(block_index, layer_id);
            auto host_kbuffer = device_->clone({*kbuffer, AllocationType::HOST});
            auto host_vbuffer = device_->clone({*vbuffer, AllocationType::HOST});
            ASSERT_EQ(cache_config.k_block_stride, host_kbuffer->size());
            ASSERT_EQ(cache_config.k_block_stride, host_vbuffer->size());
            for (size_t i = 0; i < host_kbuffer->size(); i++) {
                ASSERT_EQ(block_value, host_kbuffer->data<int8_t>()[i]);
                ASSERT_EQ(block_value, host_vbuffer->data<int8_t>()[i]);
            }
        }
    };
    testFunc(1, 1);

    cache_manager.blockCopy(1, 3);
    testFunc(3, 1);
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

}  // namespace rtp_llm
