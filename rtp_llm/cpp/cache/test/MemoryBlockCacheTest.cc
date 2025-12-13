#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "rtp_llm/cpp/cache/MemoryBlockCache.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

// 定义MemoryBlockCacheOp枚举（如果不存在的话）
namespace rtp_llm {
enum class MemoryBlockCacheOp {
    MEMORY_CACHE_COPY_FROM_GPU = 0,
    MEMORY_CACHE_COPY_TO_GPU   = 1
};
}  // namespace rtp_llm

namespace rtp_llm {
namespace test {

class MemoryBlockCacheTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();

        // 创建单TP的配置
        parallelism_config_.tp_size = 1;
        parallelism_config_.tp_rank = 0;

        gpu_allocator_ = std::make_unique<KVCacheAllocator>(gpu_config_, device_);
        ASSERT_TRUE(gpu_allocator_->init());

        memory_cache_ = std::make_unique<MemoryBlockCache>(
            cpu_config_, device_, gpu_allocator_.get(), parallelism_config_, kv_cache_config_, runtime_config_);
        ASSERT_TRUE(memory_cache_->init());
    }

    void TearDown() override {
        memory_cache_.reset();
        gpu_allocator_.reset();
        DeviceTestBase::TearDown();
    }

    // 创建测试用的KV数据
    void
    createTestKVData(int block_id, int layer_id, float k_value, float v_value, KVCacheAllocator* allocator = nullptr) {
        if (allocator == nullptr) {
            allocator = gpu_allocator_.get();
        }

        auto k_shape = gpu_config_.getKeyShape();
        auto v_shape = gpu_config_.getValueShape();

        // 创建K buffer
        auto   k_buffer   = device_->allocateBuffer({gpu_config_.dtype, {k_shape}, rtp_llm::AllocationType::HOST});
        float* k_data_ptr = k_buffer->data<float>();
        for (size_t i = 0; i < k_shape; ++i) {
            k_data_ptr[i] = k_value;
        }

        // 创建V buffer
        auto   v_buffer   = device_->allocateBuffer({gpu_config_.dtype, {v_shape}, rtp_llm::AllocationType::HOST});
        float* v_data_ptr = v_buffer->data<float>();
        for (size_t i = 0; i < v_shape; ++i) {
            v_data_ptr[i] = v_value;
        }

        // 设置到指定的allocator中
        allocator->setKVBlockValue(block_id, layer_id, *k_buffer, *v_buffer);
        device_->syncAndCheck();
    }

    // 验证KV数据
    void verifyKVData(int               block_id,
                      int               layer_id,
                      float             expected_k_value,
                      float             expected_v_value,
                      KVCacheAllocator* allocator = nullptr) {
        if (allocator == nullptr) {
            allocator = gpu_allocator_.get();
        }

        auto [success, k_buffer, v_buffer] = allocator->getKVBlockValue(block_id, layer_id);
        ASSERT_TRUE(success);
        ASSERT_NE(k_buffer, nullptr);
        ASSERT_NE(v_buffer, nullptr);
        device_->syncAndCheck();

        auto k_shape = gpu_config_.getKeyShape();
        // auto v_shape = gpu_config_.getValueShape();

        // 检查是否是GPU allocator，如果是则需要拷贝到主机内存
        if (allocator == gpu_allocator_.get()) {
            // 为GPU数据创建主机内存buffer
            auto host_k_buffer = device_->allocateBuffer({gpu_config_.dtype, {k_shape}, rtp_llm::AllocationType::HOST});
            // auto host_v_buffer = device_->allocateBuffer({gpu_config_.dtype, {v_shape},
            // rtp_llm::AllocationType::HOST});

            // 从GPU拷贝数据到主机内存
            device_->copy({*host_k_buffer, *k_buffer});
            // device_->copy({*host_v_buffer, *v_buffer});
            device_->syncAndCheck();

            // 使用主机内存数据进行验证
            float* k_data_ptr = host_k_buffer->data<float>();
            // float* v_data_ptr = host_v_buffer->data<float>();

            for (size_t i = 0; i < k_shape; ++i) {
                EXPECT_FLOAT_EQ(k_data_ptr[i], expected_k_value) << "K buffer value mismatch at index " << i;
            }

            // for (size_t i = 0; i < v_shape; ++i) {
            //     EXPECT_FLOAT_EQ(v_data_ptr[i], expected_v_value) << "V buffer value mismatch at index " << i;
            // }
        } else {
            // 对于主机内存allocator，可以直接访问数据
            float* k_data_ptr = k_buffer->data<float>();
            // float* v_data_ptr = v_buffer->data<float>();

            for (size_t i = 0; i < k_shape; ++i) {
                EXPECT_FLOAT_EQ(k_data_ptr[i], expected_k_value) << "K buffer value mismatch at index " << i;
            }

            // for (size_t i = 0; i < v_shape; ++i) {
            //     EXPECT_FLOAT_EQ(v_data_ptr[i], expected_v_value) << "V buffer value mismatch at index " << i;
            // }
        }
    }

    // 验证匹配结果（数据内容和losses）
    void verifyMatchResult(const MemoryMatchResult&                    result,
                           int                                         expect_match_len,
                           const std::vector<int>&                     expect_block_ids,
                           const std::vector<std::pair<float, float>>& expect_block_values,
                           const std::vector<float>&                   expect_losses) {
        // 验证匹配长度
        ASSERT_EQ(result.matched_len, expect_match_len);

        // 验证block_ids
        ASSERT_EQ(result.block_indices.size(), expect_block_ids.size());
        for (size_t i = 0; i < expect_block_ids.size(); ++i) {
            EXPECT_EQ(result.block_indices[i], expect_block_ids[i]) << "Block ID mismatch at index " << i;
        }

        // 验证losses
        ASSERT_EQ(result.losses.size(), expect_losses.size());
        for (size_t i = 0; i < expect_losses.size(); ++i) {
            EXPECT_FLOAT_EQ(result.losses[i], expect_losses[i]) << "Loss mismatch at index " << i;
        }
        // 验证每个匹配的block中的数据内容
        for (size_t i = 0; i < result.block_indices.size(); ++i) {
            int memory_block_id = result.block_indices[i];

            // 从expect_block_values中获取期望的K和V值
            if (i >= expect_block_values.size()) {
                FAIL() << "expect_block_values size too small for block " << i;
                return;
            }
            float expected_k_value = expect_block_values[i].first;
            // float expected_v_value = expect_block_values[i].second;
            // 验证GPU中的KV数据与期望值一致
            for (uint32_t layer_id = 0; layer_id < gpu_config_.layer_num; ++layer_id) {
                // 直接从内存缓存获取数据并验证
                auto [mem_success, mem_k_buffer, mem_v_buffer] =
                    gpu_allocator_->getKVBlockValue(memory_block_id, layer_id);
                ASSERT_TRUE(mem_success);
                ASSERT_NE(mem_k_buffer, nullptr);
                ASSERT_NE(mem_v_buffer, nullptr);
                device_->syncAndCheck();
                // 从内存缓存获取实际值
                auto [actual_k_value, actual_v_value] = getGPUExpectedValues(mem_k_buffer, mem_v_buffer);
                // 直接比较期望值和实际值
                EXPECT_FLOAT_EQ(actual_k_value, expected_k_value)
                    << "K value mismatch for block " << memory_block_id << ", layer " << layer_id;
                // EXPECT_FLOAT_EQ(actual_v_value, expected_v_value)
                //     << "V value mismatch for block " << memory_block_id << ", layer " << layer_id;
            }
        }
    }

    // 从GPU buffer获取期望值（拷贝到主机内存）
    std::pair<float, float> getGPUExpectedValues(const BufferPtr& k_buffer, const BufferPtr& v_buffer) {
        auto k_shape = gpu_config_.getKeyShape();
        // auto v_shape = gpu_config_.getValueShape();

        auto host_k_buffer = device_->allocateBuffer({gpu_config_.dtype, {k_shape}, rtp_llm::AllocationType::HOST});
        // auto host_v_buffer = device_->allocateBuffer({gpu_config_.dtype, {v_shape}, rtp_llm::AllocationType::HOST});

        device_->copy({*host_k_buffer, *k_buffer});
        // device_->copy({*host_v_buffer, *v_buffer});
        device_->syncAndCheck();

        float expected_k_value = host_k_buffer->data<float>()[0];
        // float expected_v_value = host_v_buffer->data<float>()[0];

        return {expected_k_value, 0};
    }

    CacheConfig initGPUConfig() {
        // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
        CacheConfig config(KVCacheParam({2, 20, 4, 64, 1, rtp_llm::TYPE_FP32}));
        return config;
    }

    CacheConfig initCPUConfig() {
        // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
        CacheConfig config(KVCacheParam({2, 8, 4, 64, 1, rtp_llm::TYPE_FP32}));
        return config;
    }

    CacheConfig                       gpu_config_ = initGPUConfig();
    CacheConfig                       cpu_config_ = initCPUConfig();
    std::unique_ptr<KVCacheAllocator> gpu_allocator_;
    ParallelismConfig                 parallelism_config_;
    KVCacheConfig                     kv_cache_config_;
    RuntimeConfig                     runtime_config_;
    std::unique_ptr<MemoryBlockCache> memory_cache_;
    int64_t                           request_id_ = 0;
};

// ==================== 基础功能测试 ====================

TEST_F(MemoryBlockCacheTest, ConstructorAndInitTest) {
    // 测试构造函数和初始化
    EXPECT_NE(memory_cache_, nullptr);

    // 测试单TP配置
    EXPECT_EQ(parallelism_config_.tp_size, 1);
    EXPECT_EQ(parallelism_config_.tp_rank, 0);
    EXPECT_EQ(memory_cache_->size(), 0);
    EXPECT_EQ(memory_cache_->capacity(), 7);
}

// ==================== Match方法测试 ====================

TEST_F(MemoryBlockCacheTest, MatchEmptyCacheTest) {
    // 测试空缓存的匹配
    std::vector<int64_t> cache_keys    = {101, 102, 103};
    std::vector<int>     gpu_block_ids = {1, 2, 3};

    auto result = memory_cache_->match(cache_keys, gpu_block_ids, request_id_);
    EXPECT_EQ(result.matched_len, 0);
    EXPECT_TRUE(result.block_indices.empty());
    EXPECT_TRUE(result.losses.empty());
}

TEST_F(MemoryBlockCacheTest, MatchWithDataTest) {
    // 先放入一些数据
    std::vector<int64_t> cache_keys    = {101, 102, 103};
    std::vector<int>     gpu_block_ids = {1, 2, 3};
    std::vector<float>   losses        = {0.1f, 0.2f, 0.3f};

    // 为GPU block创建测试数据
    for (int i = 0; i < 4; ++i) {
        for (uint32_t layer_id = 0; layer_id < gpu_config_.layer_num; ++layer_id) {
            createTestKVData(gpu_block_ids[i], layer_id, 1.0f + i, 2.0f + i);
        }
    }

    memory_cache_->put(cache_keys, gpu_block_ids, losses, false, request_id_);

    // 测试完全匹配并验证数据内容和losses
    std::vector<int> match_gpu_ids = {10, 11, 12};  // 使用不同的gpu_block_ids，确保数据从内存拷贝到GPU
    auto             result        = memory_cache_->match(cache_keys, match_gpu_ids, request_id_);
    std::vector<std::pair<float, float>> expect_block_values = {
        {1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}};  // block 0,1,2的K,V值
    verifyMatchResult(result, 3, match_gpu_ids, expect_block_values, losses);

    // 测试部分匹配, from start
    std::vector<int64_t> partial_keys    = {101, 102};
    std::vector<int>     partial_gpu_ids = {10, 11};  // 使用不同的gpu_block_ids
    auto                 partial_result  = memory_cache_->match(partial_keys, partial_gpu_ids, request_id_);
    EXPECT_EQ(partial_result.matched_len, 2);
    EXPECT_EQ(partial_result.block_indices.size(), 2);
    verifyMatchResult(partial_result, 2, partial_gpu_ids, {{1.0f, 2.0f}, {2.0f, 3.0f}}, {0.1f, 0.2f});

    // from middle
    std::vector<int64_t> partial_keys_middle    = {102, 103};
    std::vector<int>     partial_gpu_ids_middle = {12, 11};  // 使用不同的gpu_block_ids
    auto partial_result_middle = memory_cache_->match(partial_keys_middle, partial_gpu_ids_middle, request_id_);
    EXPECT_EQ(partial_result_middle.matched_len, 2);
    EXPECT_EQ(partial_result_middle.block_indices.size(), 2);
    verifyMatchResult(partial_result_middle, 2, partial_gpu_ids_middle, {{2.0f, 3.0f}, {3.0f, 4.0f}}, {0.2f, 0.3f});

    // 测试不匹配
    std::vector<int64_t> non_match_keys    = {14, 15, 16};
    std::vector<int>     non_match_gpu_ids = {3, 4, 5};
    auto                 non_match_result  = memory_cache_->match(non_match_keys, non_match_gpu_ids, request_id_);
    EXPECT_EQ(non_match_result.matched_len, 0);
    EXPECT_TRUE(non_match_result.block_indices.empty());

    // 测试单个block匹配
    std::vector<int64_t> single_key    = {102};
    std::vector<int>     single_gpu_id = {5};  // 使用不同的gpu_block_id
    auto                 single_result = memory_cache_->match(single_key, single_gpu_id, request_id_);
    EXPECT_EQ(single_result.matched_len, 1);
    EXPECT_EQ(single_result.block_indices.size(), 1);
    verifyMatchResult(single_result, 1, single_gpu_id, {{2.0f, 3.0f}}, {0.2f});
}

// ==================== Put方法测试 ====================

TEST_F(MemoryBlockCacheTest, PutBasicTest) {
    // 测试基本的put功能
    std::vector<int64_t> cache_keys    = {101, 102, 103};
    std::vector<int>     gpu_block_ids = {1, 2, 3};
    std::vector<float>   losses        = {0.1f, 0.2f, 0.3f};

    // 为GPU block创建测试数据
    for (int i = 0; i < 4; ++i) {
        for (uint32_t layer_id = 0; layer_id < gpu_config_.layer_num; ++layer_id) {
            createTestKVData(gpu_block_ids[i], layer_id, 1.0f + i, 2.0f + i);
        }
    }

    memory_cache_->put(cache_keys, gpu_block_ids, losses, false, request_id_);

    // 验证数据已经放入缓存并验证数据内容和losses
    std::vector<int>                     match_gpu_ids = {10, 11, 12};  // 使用不同的gpu_block_ids
    auto                                 result        = memory_cache_->match(cache_keys, match_gpu_ids, request_id_);
    std::vector<std::pair<float, float>> expect_block_values = {
        {1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}};  // block 0,1,2的K,V值
    verifyMatchResult(result, 3, match_gpu_ids, expect_block_values, losses);
}

TEST_F(MemoryBlockCacheTest, PutWithEmptyInputsTest) {
    // 测试空输入的put
    std::vector<int64_t> empty_keys;
    std::vector<int>     empty_gpu_ids;
    std::vector<float>   empty_losses;

    memory_cache_->put(empty_keys, empty_gpu_ids, empty_losses, false, request_id_);
    EXPECT_EQ(memory_cache_->size(), 0);

    // 测试部分空输入
    std::vector<int64_t> keys    = {101};
    std::vector<int>     gpu_ids = {1};
    std::vector<float>   losses  = {0.1f};

    memory_cache_->put(keys, empty_gpu_ids, losses, false, request_id_);
    memory_cache_->put(empty_keys, gpu_ids, losses, false, request_id_);
    EXPECT_EQ(memory_cache_->size(), 0);
}

TEST_F(MemoryBlockCacheTest, PutWithMismatchedSizesTest) {
    // 测试输入参数大小不匹配的情况
    std::vector<int64_t> keys    = {101, 102};
    std::vector<int>     gpu_ids = {1};  // 大小不匹配
    std::vector<float>   losses  = {0.1f, 0.2f, 0.3f};

    memory_cache_->put(keys, gpu_ids, losses, false, request_id_);

    // 测试losses大小不匹配
    std::vector<float> wrong_losses = {0.1f, 0.2f};  // 大小不匹配
    memory_cache_->put(keys, gpu_ids, wrong_losses, false, request_id_);
}

// ==================== copyKVData方法测试 ====================

TEST_F(MemoryBlockCacheTest, copyKVDataTest) {
    // 测试从GPU拷贝数据到内存缓存
    std::vector<int32_t> gpu_block_ids    = {2, 1};
    std::vector<int32_t> memory_block_ids = {1, 2};

    // 为GPU block创建测试数据
    for (int i = 0; i < 2; ++i) {
        for (uint32_t layer_id = 0; layer_id < gpu_config_.layer_num; ++layer_id) {
            createTestKVData(gpu_block_ids[i], layer_id, 1.0f + i, 2.0f + i);
        }
    }

    // 验证GPU中的数据
    verifyKVData(2, 0, 1.0f, 2.0f);
    verifyKVData(1, 0, 2.0f, 3.0f);

    // 执行拷贝
    EXPECT_TRUE(memory_cache_->copyKVData(
        memory_block_ids, gpu_block_ids, MemoryBlockCache::CopyDirection::FROM_GPU, request_id_));

    verifyKVData(1, 0, 1.0f, 2.0f, memory_cache_->allocator_.get());
    verifyKVData(2, 0, 2.0f, 3.0f, memory_cache_->allocator_.get());
}

TEST_F(MemoryBlockCacheTest, CopyWithMismatchedSizesTest) {
    // 测试拷贝时大小不匹配的情况
    std::vector<int32_t> gpu_block_ids    = {2, 1};
    std::vector<int32_t> memory_block_ids = {1};  // 大小不匹配

    EXPECT_FALSE(memory_cache_->copyKVData(
        memory_block_ids, gpu_block_ids, MemoryBlockCache::CopyDirection::FROM_GPU, request_id_));
    EXPECT_FALSE(memory_cache_->copyKVData(
        memory_block_ids, gpu_block_ids, MemoryBlockCache::CopyDirection::TO_GPU, request_id_));
}

// ==================== 内存管理测试 ====================

TEST_F(MemoryBlockCacheTest, LRUEvictionTest) {
    // 测试LRU淘汰机制
    // 创建一个较小的配置来测试淘汰
    CacheConfig small_config(KVCacheParam({2, 4, 4, 64, 1, rtp_llm::TYPE_FP32}));  // 只有3个block
    auto        small_memory_cache =
        std::make_unique<MemoryBlockCache>(small_config, device_, gpu_allocator_.get(), parallelism_config_, kv_cache_config_, runtime_config_);
    ASSERT_TRUE(small_memory_cache->init());
    ASSERT_EQ(3, small_memory_cache->capacity());

    // 尝试放入4个block，但容量只有3个
    // 放入缓存
    small_memory_cache->put({101, 102, 103}, {1, 2, 3}, {0.1f, 0.2f, 0.3f}, false, request_id_);
    EXPECT_EQ(small_memory_cache->size(), 3);

    // 验证只有最新的3个block在缓存中（因为容量限制， 只存储部分block）
    auto result = small_memory_cache->match({101, 102, 103}, {1, 2, 3}, request_id_);
    EXPECT_EQ(result.matched_len, 3);  // 应该只有3个匹配
    EXPECT_EQ(result.block_indices, std::vector<int>({1, 2, 3}));
    EXPECT_EQ(small_memory_cache->availableBlockNum(), 3);

    // 验证104没有存
    auto partial_result = small_memory_cache->match({104}, {0}, request_id_);
    EXPECT_EQ(partial_result.matched_len, 0);  // 第一个block应该被淘汰

    // 放入新的缓存， 23被淘汰
    small_memory_cache->put({105, 106}, {5, 6}, {0.5f, 0.6f}, false, request_id_);
    auto result2 = small_memory_cache->match({105, 106}, {5, 6}, request_id_);
    EXPECT_EQ(result2.matched_len, 2);  // 应该只有3个匹配
    EXPECT_EQ(result2.block_indices, std::vector<int>({5, 6}));

    // 验证102/ 103被淘汰
    auto result3 = small_memory_cache->match({102, 103}, {2, 3}, request_id_);
    EXPECT_EQ(result3.matched_len, 0);
    EXPECT_EQ(result3.block_indices, std::vector<int>({}));
}

}  // namespace test
}  // namespace rtp_llm
