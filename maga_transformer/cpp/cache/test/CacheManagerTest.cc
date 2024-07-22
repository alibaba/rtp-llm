
#include "gtest/gtest.h"

#define private public
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/testing/TestBase.h"

#include <chrono>
#include <memory>
#include <thread>

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

class CacheManagerTest: public DeviceTestBase {
protected:
    CacheConfig init_config() {
        // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
        CacheConfig config(1, 4, 1, 1, 1, ft::TYPE_INT8);
        return config;
    }

    CacheConfig init_complex_config() {
        // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
        CacheConfig config(2, 4, 3, 5, 6, ft::TYPE_INT8);
        return config;
    }

protected:
};

TEST_F(CacheManagerTest, testSimple) {
    auto            cache_config = init_config();
    CacheManager    cache_manager(cache_config, device_);

    ASSERT_EQ(cache_manager.freeBlockNums(), 3);

    auto [success1, index1] = cache_manager.mallocIndex(1);
    ASSERT_TRUE(success1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);

    auto [success2, index2] = cache_manager.mallocIndex(2);
    ASSERT_TRUE(success2);
    auto [success3, _] = cache_manager.mallocIndex(1);
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
    auto            cache_config = init_config();
    CacheManager    cache_manager(cache_config, device_);

    auto [success1, index1, reuse_len] = cache_manager.mallocWithCacheImpl(3, {1000, 2000, 3000});
    ASSERT_EQ(index1, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);

    cache_manager.freeWithCache(index1, {1000, 2000, 3000});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);

    auto [success2, index2] = cache_manager.mallocIndex(2);
    ASSERT_EQ(index2, std::vector<int>({1, 2}));
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.freeBlockNums(), 1);

    // paritial fallback case
    cache_manager.freeWithCache(index2, {1000, 2000, 3000, 4000, 5000});
    ASSERT_TRUE(cache_manager.blockCache().hasKey({1000, 2000}));
}

TEST_F(CacheManagerTest, testAllocateWithReuse) {
    auto            cache_config = init_config();
    CacheManager    cache_manager(cache_config, device_);

    ASSERT_EQ(cache_manager.freeBlockNums(), 3);
    auto [success1, index1] = cache_manager.mallocIndex(2);
    ASSERT_EQ(index1, std::vector<int>({1, 2}));
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 0);

    cache_manager.freeWithCache(index1, {1000, 1002});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 1);

    auto [success2, index2, reuseNum] = cache_manager.mallocWithCacheImpl(2, {1000, 1002});
    ASSERT_EQ(cache_manager.freeBlockNums(), 1);
    ASSERT_EQ(index2, std::vector<int>({1, 2}));
    ASSERT_EQ(reuseNum, 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);

    cache_manager.freeWithCache(index2, {1000, 1002});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);

    auto [sucecss3, index3, reuseNum3] = cache_manager.mallocWithCacheImpl(3, {1000, 1002, 1003});
    ASSERT_EQ(index3, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(reuseNum3, 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.cacheItemNum(), 1);

    cache_manager.freeWithCache(index3, {1000, 1002, 1003});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.cacheItemNum(), 2);
}

TEST_F(CacheManagerTest, testMatchMaxLen) {
    auto cache_config       = init_config();
    cache_config.block_nums = 100;
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.freeBlockNums(), 99);

    // malloc cache item 1
    auto [success1, index1, reuseNum] = cache_manager.mallocWithCacheImpl(2, {1000, 1002});
    ASSERT_EQ(index1, std::vector<int>({1, 2}));
    ASSERT_EQ(reuseNum, 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);

    // insert cache item 1
    cache_manager.freeWithCache(index1, {1000, 1002});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);

    // malloc cache item 2
    auto [success2, index2, reuseNum2] = cache_manager.mallocWithCacheImpl(3, {1000, 1002, 1003});
    ASSERT_EQ(index2, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(reuseNum2, 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 0);

    // insert cache item 2
    cache_manager.freeWithCache(index2, {1000, 1002, 1003});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 0);

    // malloc cache item 3
    auto [success3, index3, reuseNum3] = cache_manager.mallocWithCacheImpl(4, {1000, 1002, 1003, 1004});
    ASSERT_EQ(index3, (std::vector<int>{1, 2, 3, 4}));
    ASSERT_EQ(reuseNum3, 2);  // Assuming 2 blocks were reused, replace with actual logic
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 3);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(6), 0);

    // insert cache item 3
    cache_manager.freeWithCache(index3, {1000, 1002, 1003, 1004});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 3);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(6), 0);

    // trigger match max len cache item
    auto [success4, index4, reuseNum4] = cache_manager.mallocWithCacheImpl(4, {1000, 1002, 1003, 1004});
    ASSERT_EQ(index4, std::vector<int>({1, 2, 3, 4}));
    ASSERT_EQ(reuseNum4, 3);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 4);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 3);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(4), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(5), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(6), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(7), 0);
}

TEST_F(CacheManagerTest, testPopNoResidentCacheItem) {
    auto         cache_config = init_config();
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.freeBlockNums(), 3);

    // malloc cache item 1
    auto [success1, index1] = cache_manager.mallocIndex(2);
    ASSERT_TRUE(success1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);

    // insert cache item 1
    cache_manager.freeWithCache(index1, {1000, 1002});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);

    // trigger reuse cache, pop cache item 1, malloc from free failed
    auto [success2, index2, reuseNum2] = cache_manager.mallocWithCacheImpl(4, {1000, 1002, 1003, 1004});
    ASSERT_FALSE(success2);
    ASSERT_EQ(cache_manager.freeBlockNums(), 3);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);

    // trigger malloc block from free failed
    auto [success3, index3, reuseNum3] = cache_manager.mallocWithCacheImpl(4, {1000, 1002, 1003, 1004});
    ASSERT_FALSE(success3);

    // insert cache item 2
    auto [success4, index4, reuseNum4] = cache_manager.mallocWithCacheImpl(2, {100, 1002});
    ASSERT_EQ(index4, std::vector<int>({1, 2}));
    ASSERT_EQ(reuseNum4, 0);
    cache_manager.freeWithCache(index4, {1000, 1002});
    ASSERT_EQ(cache_manager.freeBlockNums(), 2);

    // trigger pop cache item 2 from cache, malloc success
    auto [success5, index5, reuseNum5] = cache_manager.mallocWithCacheImpl(3, {2000, 2002, 2003});
    ASSERT_EQ(index5, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(reuseNum5, 0);
    ASSERT_EQ(cache_manager.freeBlockNums(), 0);
}

TEST_F(CacheManagerTest, testPopTwoCache) {
    auto cache_config       = init_config();
    cache_config.block_nums = 7;
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.freeBlockNums(), 6);

    // insert cache item 1
    auto [success1, index1, reuseNum1] = cache_manager.mallocWithCacheImpl(2, {1000, 1002});
    ASSERT_EQ(index1, std::vector<int>({1, 2}));
    ASSERT_EQ(reuseNum1, 0);
    cache_manager.freeWithCache(index1, {1000, 1002});
    ASSERT_EQ(cache_manager.freeBlockNums(), 5);
    ASSERT_TRUE(cache_manager.blockCache().hasKey({1000}));

    // insert cache item 2
    auto [success2, index2, reuseNum2] = cache_manager.mallocWithCacheImpl(3, {2000, 2002, 2003});
    ASSERT_EQ(index2, std::vector<int>({2, 3, 4}));
    ASSERT_EQ(reuseNum2, 0);
    cache_manager.freeWithCache(index2, {2000, 2002, 2003});
    ASSERT_EQ(cache_manager.freeBlockNums(), 3);
    ASSERT_TRUE(cache_manager.blockCache().hasKey({2000, 2002}));

    // malloc cache item 3 lead to pop cache item 2
    auto [success3, index3, reuseNum3] = cache_manager.mallocWithCacheImpl(5, {1000, 1002, 1003, 1004, 1005});
    ASSERT_EQ(index3, std::vector<int>({1, 2, 3, 4, 5}));
    ASSERT_EQ(reuseNum3, 1);

    // cache item 1 is in cache
    ASSERT_TRUE(cache_manager.blockCache().hasKey({1000}));

    // cache item 2 is not in cache
    ASSERT_FALSE(cache_manager.blockCache().hasKey({2000, 2002}));
}

TEST_F(CacheManagerTest, testPopWithResident) {
    auto cache_config       = init_config();
    cache_config.block_nums = 6;
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.freeBlockNums(), 5);

    // Insert resident cache item
    auto [success1, index1] = cache_manager.mallocIndex(2);
    ASSERT_EQ(index1, std::vector<int>({1, 2}));
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    cache_manager.insertResidentCache(index1, {1000, 1002});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);
    ASSERT_TRUE(cache_manager.blockCache().isResident({1000}));

    // Insert cache item 2
    auto [success2, index2, reuseNum] = cache_manager.mallocWithCacheImpl(3, {2000, 2002, 2003});
    ASSERT_EQ(index2, std::vector<int>({2, 3, 4}));
    ASSERT_EQ(reuseNum, 0);
    cache_manager.freeWithCache(index2, {2000, 2002, 2003});
    ASSERT_EQ(cache_manager.freeBlockNums(), 2);
    ASSERT_TRUE(cache_manager.blockCache().hasKey({2000, 2002}));
    ASSERT_TRUE(cache_manager.blockCache().isResident({1000}));

    // Malloc cache item 3 lead to pop cache item 2
    auto [success3, index3, reuseNum3] = cache_manager.mallocWithCacheImpl(5, {2000, 2002, 2003, 2004, 2005});
    ASSERT_FALSE(success3);
    // Cache item 1 is in cache
    ASSERT_TRUE(cache_manager.blockCache().hasKey({1000}));
    ASSERT_TRUE(cache_manager.blockCache().isResident({1000}));
    // Cache item 2 is not in cache
    ASSERT_FALSE(cache_manager.blockCache().hasKey({2000, 2002}));
}

TEST_F(CacheManagerTest, testResident) {
    auto cache_config       = init_config();
    cache_config.block_nums = 100;
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.freeBlockNums(), 99);

    // Malloc for resident block
    auto [success1, index1] = cache_manager.mallocIndex(2);
    ASSERT_EQ(index1, std::vector<int>({1, 2}));
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);

    // Insert resident cache item
    cache_manager.insertResidentCache(index1, {1000, 1002});
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 0);
    ASSERT_TRUE(cache_manager.blockCache().isResident({1000}));

    // Put not pop resident cache item
    auto [success2, index2, reuseNum2] = cache_manager.mallocWithCacheImpl(2, {1000, 1002});
    ASSERT_EQ(index2, std::vector<int>({1, 2}));
    ASSERT_EQ(reuseNum2, 1);
    cache_manager.freeWithCache(index2, {1000, 1002});
    ASSERT_TRUE(cache_manager.blockCache().isResident({1000}));

    // Match resident cache item
    auto [success3, index3, reuseNum3] = cache_manager.mallocWithCacheImpl(3, {1000, 1002, 1003});
    ASSERT_EQ(index3, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(reuseNum3, 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);

    // Put not pop resident cache item
    cache_manager.freeWithCache(index3, {1000, 1002, 1003});
    ASSERT_TRUE(cache_manager.blockCache().isResident({1000}));

    // Not match
    auto [success4, index4, reuseNum4] = cache_manager.mallocWithCacheImpl(3, {2000, 2002, 2003});
    ASSERT_EQ(index4, std::vector<int>({3, 4, 5}));
    ASSERT_EQ(reuseNum4, 0);
}

TEST_F(CacheManagerTest, testSeqSizePerBlock) {
    auto cache_config               = init_config();
    cache_config.block_nums         = 100;
    cache_config.seq_size_per_block = 2;
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.freeBlockNums(), 99);

    // Malloc cache item 1
    auto [success1, index1, reuseNum] = cache_manager.mallocWithCacheImpl(1, {1000, 1002});
    ASSERT_EQ(index1, std::vector<int>({1}));
    ASSERT_EQ(reuseNum, 0);
    // Insert cache item 1
    cache_manager.freeWithCache(index1, {1000, 1002});
    ASSERT_FALSE(cache_manager.blockCache().hasKey({1000, 1002}));

    // Malloc cache item 2
    auto [success2, index2, reuseNum2] = cache_manager.mallocWithCacheImpl(2, {1000, 1002, 1003});
    ASSERT_EQ(index2, std::vector<int>({1, 2}));
    ASSERT_EQ(reuseNum2, 0);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    // Free cache item 2
    cache_manager.freeWithCache(index2, {1000, 1002, 1003});
    ASSERT_TRUE(cache_manager.blockCache().hasKey({1000, 1002}));

    // Malloc cache item 3
    auto [success3, index3, reuseNum3] = cache_manager.mallocWithCacheImpl(2, {1000, 1002, 1003, 1004});
    ASSERT_EQ(index3, std::vector<int>({1, 2}));
    ASSERT_EQ(reuseNum3, 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(1), 2);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(2), 1);
    ASSERT_EQ(cache_manager.blockRefCounter().getRefCounter(3), 0);
    // Free cache item 3
    cache_manager.freeWithCache(index3, {1000, 1002, 1003, 1004, 1005});

    auto [success4, index4, reuseNum4] = cache_manager.mallocWithCacheImpl(3, {1000, 1002, 1003, 1004, 1005});
    ASSERT_EQ(index4, std::vector<int>({1, 2, 3}));
    ASSERT_EQ(reuseNum4, 4);
}


TEST_F(CacheManagerTest, testSetBlockValue) {
    // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
    CacheConfig cache_config(2, 4, 1, 1, 2, ft::TYPE_INT8);
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.freeBlockNums(), 3);

    vector<int8_t> k_vec(cache_config.kv_block_size, 1);
    vector<int8_t> v_vec(cache_config.kv_block_size, 1);
    auto k_buffer = ft::vector2Buffer(k_vec);
    auto v_buffer = ft::vector2Buffer(v_vec);
    cache_manager.setKVBlockValue(1, 1, k_buffer, v_buffer);

    auto testFunc = [&](int block_index, int block_value){
        auto [kbuffer, vbuffer] = cache_manager.getKVBlockValue(block_index);
        auto host_kbuffer = device_->clone({*kbuffer, AllocationType::HOST});
        auto host_vbuffer = device_->clone({*vbuffer, AllocationType::HOST});
        ASSERT_EQ(cache_config.kv_block_size, host_kbuffer->size());
        ASSERT_EQ(cache_config.kv_block_size, host_vbuffer->size());
        for (size_t i = 0; i < host_kbuffer->size(); i++) {
            ASSERT_EQ(block_value, host_kbuffer->data<int8_t>()[i]);
            ASSERT_EQ(block_value, host_vbuffer->data<int8_t>()[i]);
        }
    };
    testFunc(1, 1);

    cache_manager.blockCopy(1, 3);
    testFunc(3, 1);
}

TEST_F(CacheManagerTest, testBlockCacheHoldBlockNums) {
    // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
    CacheConfig cache_config(2, 10, 1, 1, 1, ft::TYPE_INT8);
    CacheManager cache_manager(cache_config, device_);
    ASSERT_EQ(cache_manager.block_cache_.holdBlockNums(), 0);

    auto [success1, index1, reuse_len] = cache_manager.mallocWithCacheImpl(3, {1000, 2000, 3000});
    ASSERT_EQ(index1, std::vector<int>({1, 2, 3}));

    cache_manager.freeWithCache(index1, {1000, 2000, 3000});
    ASSERT_EQ(cache_manager.block_cache_.holdBlockNums(), 2);

    auto [sucecss2, index2, reuseNum2] = cache_manager.mallocWithCacheImpl(3, {1000, 1002, 1003});
    ASSERT_EQ(index2, std::vector<int>({1, 3, 4}));
    ASSERT_EQ(reuseNum2, 1);

    cache_manager.freeWithCache(index2, {1000, 1002, 1003});
    ASSERT_EQ(cache_manager.block_cache_.holdBlockNums(), 3);

    cache_manager.block_cache_.pop();
    ASSERT_EQ(cache_manager.block_cache_.holdBlockNums(), 2);
}

}  // namespace rtp_llm
