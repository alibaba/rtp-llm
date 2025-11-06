#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/utils/Exception.h"

#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <cstring>

using namespace std;

namespace rtp_llm {

class KVCacheAllocatorTest: public DeviceTestBase {
protected:
    CacheConfig initConfig(rtp_llm::DataType dtype = rtp_llm::TYPE_FP16) {
        // layer_num, block_nums, local_head_num_kv, size_per_head, seq_size_per_block, dtype
        CacheConfig config(KVCacheParam({2, 10, 4, 64, 1, dtype}));
        return config;
    }

    CacheConfig initSmallConfig() {
        // 较小的配置用于测试边界情况
        CacheConfig config(KVCacheParam({1, 5, 2, 32, 1, rtp_llm::TYPE_FP16}));
        return config;
    }

    CacheConfig initMlaConfig() {
        // MLA配置
        CacheConfig config(MlaCacheParam({2, 8, 16, 64, 1, rtp_llm::TYPE_FP16}));
        return config;
    }

    CacheConfig initMlaSmallConfig() {
        // 较小的MLA配置用于测试边界情况
        CacheConfig config(MlaCacheParam({1, 4, 8, 32, 1, rtp_llm::TYPE_FP16}));
        return config;
    }

protected:
    int64_t request_id_ = 0;
};

// 测试基本初始化和状态获取
TEST_F(KVCacheAllocatorTest, testInitAndStatus) {
    auto config    = initConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);

    ASSERT_TRUE(allocator->init());

    // 验证总块数和空闲块数
    // block 0 是保留的，所以总块数是 block_nums - 1
    EXPECT_EQ(allocator->totalBlocks(), 9);    // 10 - 1
    EXPECT_EQ(allocator->freeBlockNums(), 9);  // 初始时所有块都是空闲的

    // 验证缓存缓冲区
    const auto& kv_buffer = allocator->kvCacheBuffer();
    EXPECT_NE(kv_buffer.k_blocks, nullptr);
    EXPECT_NE(kv_buffer.v_blocks, nullptr);
}

// 测试基本的malloc和free功能
TEST_F(KVCacheAllocatorTest, testBasicMallocAndFree) {
    auto config    = initConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);
    ASSERT_TRUE(allocator->init());

    // 初始状态
    EXPECT_EQ(allocator->freeBlockNums(), 9);

    // 分配0个块
    auto [success0, resource0] = allocator->malloc({request_id_, 0, false});
    EXPECT_TRUE(success0);
    EXPECT_EQ(resource0.block_id.size(), 0);
    EXPECT_EQ(allocator->freeBlockNums(), 9);

    // 释放空资源
    allocator->free({resource0});
    EXPECT_EQ(allocator->freeBlockNums(), 9);

    // 分配1个块
    auto [success1, resource1] = allocator->malloc({request_id_, 1, false});
    EXPECT_TRUE(success1);
    EXPECT_EQ(resource1.block_id.size(), 1);
    EXPECT_EQ(allocator->freeBlockNums(), 8);

    // 分配2个块
    auto [success2, resource2] = allocator->malloc({request_id_, 2, false});
    EXPECT_TRUE(success2);
    EXPECT_EQ(resource2.block_id.size(), 2);
    EXPECT_EQ(allocator->freeBlockNums(), 6);

    // 尝试分配9个块，应该失败
    auto [success3, resource3] = allocator->malloc({request_id_, 9, true});
    EXPECT_FALSE(success3);
    EXPECT_EQ(resource3.block_id.size(), 0);
    EXPECT_EQ(allocator->freeBlockNums(), 6);  // 空闲块数不变

    // 释放第一个资源
    allocator->free({resource1});
    EXPECT_EQ(allocator->freeBlockNums(), 7);

    // 分配7个块，应该成功
    auto [success4, resource4] = allocator->malloc({request_id_, 7, false});
    EXPECT_TRUE(success4);
    EXPECT_EQ(resource4.block_id.size(), 7);
    EXPECT_EQ(allocator->freeBlockNums(), 0);

    // 再次尝试分配，应该失败
    auto [success5, resource5] = allocator->malloc({request_id_, 1, false});
    EXPECT_FALSE(success5);
    EXPECT_EQ(resource5.block_id.size(), 0);

    // 释放第二个资源
    allocator->free({resource2});
    EXPECT_EQ(allocator->freeBlockNums(), 2);

    // 释放第四个资源
    allocator->free({resource4});
    EXPECT_EQ(allocator->freeBlockNums(), 9);
}

// 测试通过索引释放
TEST_F(KVCacheAllocatorTest, testFreeByIndices) {
    auto config    = initConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);
    ASSERT_TRUE(allocator->init());

    // 释放空的索引列表
    std::vector<int> empty_indices;
    allocator->free(empty_indices);
    EXPECT_EQ(allocator->freeBlockNums(), 9);

    // 分配资源
    auto [success, resource] = allocator->malloc({request_id_, 3, false});
    EXPECT_TRUE(success);
    EXPECT_EQ(allocator->freeBlockNums(), 6);

    // 通过索引释放
    allocator->free(resource.block_id);
    EXPECT_EQ(allocator->freeBlockNums(), 9);
}

// 测试并发malloc和free
TEST_F(KVCacheAllocatorTest, testConcurrentMallocAndFree) {
    auto config    = initConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);
    ASSERT_TRUE(allocator->init());

    std::vector<std::thread>     threads;
    std::vector<KVCacheResource> resources;
    std::mutex                   resources_mutex;

    // 启动多个线程进行malloc
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back([&, i]() {
            auto [success, resource] = allocator->malloc({request_id_ + i, 2, false});
            if (success) {
                std::lock_guard<std::mutex> lock(resources_mutex);
                resources.push_back(resource);
            }
        });
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 验证结果
    EXPECT_EQ(resources.size(), 3);
    EXPECT_EQ(allocator->freeBlockNums(), 3);  // 9 - 3*2 = 3

    // 释放所有资源
    allocator->free(resources);
    EXPECT_EQ(allocator->freeBlockNums(), 9);
}

// 测试重复释放
TEST_F(KVCacheAllocatorTest, testDoubleFree) {
    auto config    = initConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);
    ASSERT_TRUE(allocator->init());

    // 分配资源
    auto [success, resource] = allocator->malloc({request_id_, 2, false});
    EXPECT_TRUE(success);
    EXPECT_EQ(allocator->freeBlockNums(), 7);

    // 第一次释放
    allocator->free({resource});
    EXPECT_EQ(allocator->freeBlockNums(), 9);

    // 重复释放，应该失败
    EXPECT_THROW(allocator->free({resource}), rtp_llm::RTPException);
    EXPECT_EQ(allocator->freeBlockNums(), 9);
}

// ==================== MLA相关测试 ====================
// 测试MLA基本初始化和状态获取
TEST_F(KVCacheAllocatorTest, testMlaInitAndStatus) {
    auto config    = initMlaConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);

    ASSERT_TRUE(allocator->init());

    // 验证总块数和空闲块数
    // block 0 是保留的，所以总块数是 block_nums - 1
    EXPECT_EQ(allocator->totalBlocks(), 7);    // 8 - 1
    EXPECT_EQ(allocator->freeBlockNums(), 7);  // 初始时所有块都是空闲的

    // 验证缓存缓冲区
    const auto& kv_buffer = allocator->kvCacheBuffer();
    EXPECT_NE(kv_buffer.k_blocks, nullptr);
    EXPECT_NE(kv_buffer.v_blocks, nullptr);

    // MLA模式下应该没有scale缓冲区（除非是INT8或FP8）
    if (config.dtype != rtp_llm::TYPE_INT8 && config.dtype != rtp_llm::TYPE_FP8_E4M3) {
        EXPECT_EQ(kv_buffer.k_scale, nullptr);
        EXPECT_EQ(kv_buffer.v_scale, nullptr);
    }
}

// 测试MLA基本的malloc和free功能
TEST_F(KVCacheAllocatorTest, testMlaBasicMallocAndFree) {
    auto config    = initMlaConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);
    ASSERT_TRUE(allocator->init());

    // 初始状态
    EXPECT_EQ(allocator->freeBlockNums(), 7);

    // 分配1个块
    auto [success1, resource1] = allocator->malloc({request_id_, 1, false});
    EXPECT_TRUE(success1);
    EXPECT_EQ(resource1.block_id.size(), 1);
    EXPECT_EQ(allocator->freeBlockNums(), 6);

    // 分配2个块
    auto [success2, resource2] = allocator->malloc({request_id_, 2, false});
    EXPECT_TRUE(success2);
    EXPECT_EQ(resource2.block_id.size(), 2);
    EXPECT_EQ(allocator->freeBlockNums(), 4);

    // 尝试分配5个块，应该失败
    auto [success3, resource3] = allocator->malloc({request_id_, 5, true});
    EXPECT_FALSE(success3);
    EXPECT_EQ(resource3.block_id.size(), 0);
    EXPECT_EQ(allocator->freeBlockNums(), 4);  // 空闲块数不变

    // 分配3个块，应该成功
    auto [success4, resource4] = allocator->malloc({request_id_, 4, false});
    EXPECT_TRUE(success4);
    EXPECT_EQ(resource4.block_id.size(), 4);
    EXPECT_EQ(allocator->freeBlockNums(), 0);

    // 再次尝试分配，应该失败
    auto [success5, resource5] = allocator->malloc({request_id_, 1, false});
    EXPECT_FALSE(success5);
    EXPECT_EQ(resource5.block_id.size(), 0);

    // 释放第一个资源
    allocator->free({resource1});
    EXPECT_EQ(allocator->freeBlockNums(), 1);

    // 释放第二个资源
    allocator->free({resource2});
    EXPECT_EQ(allocator->freeBlockNums(), 3);

    // 释放第四个资源
    allocator->free({resource4});
    EXPECT_EQ(allocator->freeBlockNums(), 7);
}

// 测试MLA并发malloc和free
TEST_F(KVCacheAllocatorTest, testMlaConcurrentMallocAndFree) {
    auto config    = initMlaConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);
    ASSERT_TRUE(allocator->init());

    std::vector<std::thread>     threads;
    std::vector<KVCacheResource> resources;
    std::mutex                   resources_mutex;

    // 启动多个线程进行malloc
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back([&, i]() {
            auto [success, resource] = allocator->malloc({request_id_ + i, 1, false});
            if (success) {
                std::lock_guard<std::mutex> lock(resources_mutex);
                resources.push_back(resource);
            }
        });
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 验证结果
    EXPECT_EQ(resources.size(), 3);
    EXPECT_EQ(allocator->freeBlockNums(), 4);  // 7 - 3*1 = 4

    // 释放所有资源
    allocator->free(resources);
    EXPECT_EQ(allocator->freeBlockNums(), 7);
}

// 测试MLA边界情况
TEST_F(KVCacheAllocatorTest, testMlaEdgeCases) {
    auto config    = initMlaSmallConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);
    ASSERT_TRUE(allocator->init());

    // 分配0个块
    auto [success1, resource1] = allocator->malloc({request_id_, 0, false});
    EXPECT_TRUE(success1);
    EXPECT_EQ(resource1.block_id.size(), 0);
    EXPECT_EQ(allocator->freeBlockNums(), 3);

    // 释放空资源
    allocator->free({resource1});
    EXPECT_EQ(allocator->freeBlockNums(), 3);

    // 释放空的索引列表
    std::vector<int> empty_indices;
    allocator->free(empty_indices);
    EXPECT_EQ(allocator->freeBlockNums(), 3);

    // 分配所有可用块
    auto [success2, resource2] = allocator->malloc({request_id_, 3, false});
    EXPECT_TRUE(success2);
    EXPECT_EQ(resource2.block_id.size(), 3);
    EXPECT_EQ(allocator->freeBlockNums(), 0);

    // 释放所有块
    allocator->free({resource2});
    EXPECT_EQ(allocator->freeBlockNums(), 3);
}

// 测试MLA重复释放
TEST_F(KVCacheAllocatorTest, testMlaDoubleFree) {
    auto config    = initMlaConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);
    ASSERT_TRUE(allocator->init());

    // 分配资源
    auto [success, resource] = allocator->malloc({request_id_, 2, false});
    EXPECT_TRUE(success);
    EXPECT_EQ(allocator->freeBlockNums(), 5);

    // 第一次释放
    allocator->free({resource});
    EXPECT_EQ(allocator->freeBlockNums(), 7);

    // 重复释放，应该失败
    EXPECT_THROW(allocator->free({resource}), rtp_llm::RTPException);
    EXPECT_EQ(allocator->freeBlockNums(), 7);
}

// ==================== KV块值操作测试 ====================

// 测试setKVBlockValue和getKVBlockValue（单层，使用HOST内存）
TEST_F(KVCacheAllocatorTest, testSetAndGetKVBlockValueSingleLayer) {
    auto config    = initConfig(rtp_llm::TYPE_FP32);
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_, rtp_llm::AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    // 分配一个块
    auto [success, resource] = allocator->malloc({request_id_, 1, false});
    EXPECT_TRUE(success);
    EXPECT_EQ(resource.block_id.size(), 1);

    int block_index = resource.block_id[0];
    int layer_id    = 0;

    // 创建测试数据
    auto k_shape = config.getKeyShape();
    auto v_shape = config.getValueShape();

    // 从device中allocateBuffer然后memset填充数据
    auto k_buffer = device_->allocateBuffer({config.dtype, {k_shape}, rtp_llm::AllocationType::HOST});
    auto v_buffer = device_->allocateBuffer({config.dtype, {v_shape}, rtp_llm::AllocationType::HOST});

    // 填充具体的测试数据
    float* k_data_ptr = static_cast<float*>(k_buffer->data());
    float* v_data_ptr = static_cast<float*>(v_buffer->data());
    for (size_t i = 0; i < k_shape; ++i) {
        k_data_ptr[i] = 1.0f;
    }
    for (size_t i = 0; i < v_shape; ++i) {
        v_data_ptr[i] = 2.0f;
    }

    // 设置KV块值
    bool set_success = allocator->setKVBlockValue(block_index, layer_id, *k_buffer, *v_buffer);
    EXPECT_TRUE(set_success) << "setKVBlockValue failed for valid block_index: " << block_index;

    // 立即检查set后的值是否正确
    auto [check_success, check_k, check_v] = allocator->getKVBlockValue(block_index, layer_id);
    EXPECT_TRUE(check_success) << "getKVBlockValue failed for checking set result";
    EXPECT_NE(check_k, nullptr);
    EXPECT_NE(check_v, nullptr);
    EXPECT_EQ(check_k->sizeBytes(), k_buffer->sizeBytes());
    EXPECT_EQ(check_v->sizeBytes(), v_buffer->sizeBytes());

    // 验证set后的数据值
    float* check_k_data = static_cast<float*>(check_k->data());
    float* check_v_data = static_cast<float*>(check_v->data());
    for (size_t i = 0; i < k_shape; ++i) {
        EXPECT_FLOAT_EQ(check_k_data[i], 1.0f) << "K buffer value mismatch after set at index " << i;
    }
    for (size_t i = 0; i < v_shape; ++i) {
        EXPECT_FLOAT_EQ(check_v_data[i], 2.0f) << "V buffer value mismatch after set at index " << i;
    }

    // 获取KV块值（用于后续验证）
    auto [get_success, retrieved_k, retrieved_v] = allocator->getKVBlockValue(block_index, layer_id);
    EXPECT_TRUE(get_success) << "getKVBlockValue failed for valid block_index: " << block_index;

    // 验证获取的数据
    EXPECT_NE(retrieved_k, nullptr);
    EXPECT_NE(retrieved_v, nullptr);
    EXPECT_EQ(retrieved_k->sizeBytes(), k_buffer->sizeBytes());
    EXPECT_EQ(retrieved_v->sizeBytes(), v_buffer->sizeBytes());

    // 验证buffer值：直接比较HOST内存中的数据
    float* retrieved_k_data = static_cast<float*>(retrieved_k->data());
    float* retrieved_v_data = static_cast<float*>(retrieved_v->data());

    // 验证数据值
    for (size_t i = 0; i < k_shape; ++i) {
        EXPECT_FLOAT_EQ(retrieved_k_data[i], 1.0f) << "K buffer value mismatch at index " << i;
    }
    for (size_t i = 0; i < v_shape; ++i) {
        EXPECT_FLOAT_EQ(retrieved_v_data[i], 2.0f) << "V buffer value mismatch at index " << i;
    }

    // 释放资源
    allocator->free({resource});
}

// 测试setKVBlockValue和getKVBlockValue（多层，使用HOST内存）
TEST_F(KVCacheAllocatorTest, testSetAndGetKVBlockValueMultiLayer) {
    auto config    = initConfig(rtp_llm::TYPE_FP32);
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_, rtp_llm::AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    // 分配一个块
    auto [success, resource] = allocator->malloc({request_id_, 1, false});
    EXPECT_TRUE(success);
    EXPECT_EQ(resource.block_id.size(), 1);

    int block_index = resource.block_id[0];

    // 创建多层测试数据
    auto k_shape = config.getKeyShape();
    auto v_shape = config.getValueShape();

    // 从device中allocateBuffer然后memset填充数据
    auto k_buffer =
        device_->allocateBuffer({config.dtype, {config.layer_num * k_shape}, rtp_llm::AllocationType::HOST});
    auto v_buffer =
        device_->allocateBuffer({config.dtype, {config.layer_num * v_shape}, rtp_llm::AllocationType::HOST});

    // 为每一层填充不同的测试数据
    float* k_data_ptr = static_cast<float*>(k_buffer->data());
    float* v_data_ptr = static_cast<float*>(v_buffer->data());
    for (uint32_t layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        size_t k_offset = layer_id * k_shape;
        size_t v_offset = layer_id * v_shape;

        for (size_t i = 0; i < k_shape; ++i) {
            k_data_ptr[k_offset + i] = 1.0f + layer_id;
        }
        for (size_t i = 0; i < v_shape; ++i) {
            v_data_ptr[v_offset + i] = 2.0f + layer_id;
        }
    }

    // 设置KV块值（多层）
    bool set_success = allocator->setKVBlockValue(block_index, *k_buffer, *v_buffer);
    EXPECT_TRUE(set_success) << "setKVBlockValue failed for valid block_index: " << block_index;

    // 立即检查set后的值是否正确（多层）
    auto [check_success, check_k, check_v] = allocator->getKVBlockValue(block_index);
    EXPECT_TRUE(check_success) << "getKVBlockValue failed for checking set result";
    EXPECT_NE(check_k, nullptr);
    EXPECT_NE(check_v, nullptr);
    EXPECT_EQ(check_k->sizeBytes(), k_buffer->sizeBytes());
    EXPECT_EQ(check_v->sizeBytes(), v_buffer->sizeBytes());

    // 验证set后的数据值（多层）
    float* check_k_data = static_cast<float*>(check_k->data());
    float* check_v_data = static_cast<float*>(check_v->data());
    for (uint32_t layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        size_t k_offset = layer_id * k_shape;
        size_t v_offset = layer_id * v_shape;

        for (size_t i = 0; i < k_shape; ++i) {
            float expected_k_value = 1.0f + layer_id;
            EXPECT_FLOAT_EQ(check_k_data[k_offset + i], expected_k_value)
                << "K buffer value mismatch after set at layer " << layer_id << ", index " << i;
        }
        for (size_t i = 0; i < v_shape; ++i) {
            float expected_v_value = 2.0f + layer_id;
            EXPECT_FLOAT_EQ(check_v_data[v_offset + i], expected_v_value)
                << "V buffer value mismatch after set at layer " << layer_id << ", index " << i;
        }
    }

    // 获取KV块值（用于后续验证）
    auto [get_success, retrieved_k, retrieved_v] = allocator->getKVBlockValue(block_index);
    EXPECT_TRUE(get_success) << "getKVBlockValue failed for valid block_index: " << block_index;

    // 验证获取的数据
    EXPECT_NE(retrieved_k, nullptr);
    EXPECT_NE(retrieved_v, nullptr);
    EXPECT_EQ(retrieved_k->sizeBytes(), k_buffer->sizeBytes());
    EXPECT_EQ(retrieved_v->sizeBytes(), v_buffer->sizeBytes());

    // 验证buffer值：直接比较HOST内存中的数据
    float* retrieved_k_data = static_cast<float*>(retrieved_k->data());
    float* retrieved_v_data = static_cast<float*>(retrieved_v->data());

    // 验证每一层的数据值
    for (uint32_t layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        size_t k_offset = layer_id * k_shape;
        size_t v_offset = layer_id * v_shape;

        for (size_t i = 0; i < k_shape; ++i) {
            float expected_k_value = 1.0f + layer_id;
            EXPECT_FLOAT_EQ(retrieved_k_data[k_offset + i], expected_k_value)
                << "K buffer value mismatch at layer " << layer_id << ", index " << i;
        }
        for (size_t i = 0; i < v_shape; ++i) {
            float expected_v_value = 2.0f + layer_id;
            EXPECT_FLOAT_EQ(retrieved_v_data[v_offset + i], expected_v_value)
                << "V buffer value mismatch at layer " << layer_id << ", index " << i;
        }
    }

    // 释放资源
    allocator->free({resource});
}

// 测试blockCopy功能（使用HOST内存）
TEST_F(KVCacheAllocatorTest, testBlockCopy) {
    auto config    = initConfig(rtp_llm::TYPE_FP32);
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_, rtp_llm::AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    // 分配两个块
    auto [success, resource] = allocator->malloc({request_id_, 2, false});
    EXPECT_TRUE(success);
    EXPECT_EQ(resource.block_id.size(), 2);

    int src_block_index  = resource.block_id[0];
    int dest_block_index = resource.block_id[1];

    // 为源块设置测试数据
    auto k_shape = config.getKeyShape();
    auto v_shape = config.getValueShape();

    // 从device中allocateBuffer然后memset填充数据
    auto k_buffer =
        device_->allocateBuffer({config.dtype, {config.layer_num * k_shape}, rtp_llm::AllocationType::HOST});
    auto v_buffer =
        device_->allocateBuffer({config.dtype, {config.layer_num * v_shape}, rtp_llm::AllocationType::HOST});

    // 填充源块数据
    float* k_data_ptr = static_cast<float*>(k_buffer->data());
    float* v_data_ptr = static_cast<float*>(v_buffer->data());
    for (uint32_t layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        size_t k_offset = layer_id * k_shape;
        size_t v_offset = layer_id * v_shape;

        for (size_t i = 0; i < k_shape; ++i) {
            k_data_ptr[k_offset + i] = 10.0f + layer_id;
        }
        for (size_t i = 0; i < v_shape; ++i) {
            v_data_ptr[v_offset + i] = 20.0f + layer_id;
        }
    }

    // 设置源块的值
    bool set_success = allocator->setKVBlockValue(src_block_index, *k_buffer, *v_buffer);
    EXPECT_TRUE(set_success) << "setKVBlockValue failed for valid src_block_index: " << src_block_index;

    // 立即检查set后的源块值是否正确
    auto [check_success, check_k, check_v] = allocator->getKVBlockValue(src_block_index);
    EXPECT_TRUE(check_success) << "getKVBlockValue failed for checking source block set result";
    EXPECT_NE(check_k, nullptr);
    EXPECT_NE(check_v, nullptr);
    EXPECT_EQ(check_k->sizeBytes(), k_buffer->sizeBytes());
    EXPECT_EQ(check_v->sizeBytes(), v_buffer->sizeBytes());

    // 验证源块set后的数据值
    float* check_k_data = static_cast<float*>(check_k->data());
    float* check_v_data = static_cast<float*>(check_v->data());
    for (uint32_t layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        size_t k_offset = layer_id * k_shape;
        size_t v_offset = layer_id * v_shape;

        for (size_t i = 0; i < k_shape; ++i) {
            float expected_k_value = 10.0f + layer_id;
            EXPECT_FLOAT_EQ(check_k_data[k_offset + i], expected_k_value)
                << "Source K buffer value mismatch after set at layer " << layer_id << ", index " << i;
        }
        for (size_t i = 0; i < v_shape; ++i) {
            float expected_v_value = 20.0f + layer_id;
            EXPECT_FLOAT_EQ(check_v_data[v_offset + i], expected_v_value)
                << "Source V buffer value mismatch after set at layer " << layer_id << ", index " << i;
        }
    }

    // 执行块复制
    allocator->blockCopy(src_block_index, dest_block_index);

    // 验证目标块的数据
    auto [get_success, dest_k, dest_v] = allocator->getKVBlockValue(dest_block_index);
    EXPECT_TRUE(get_success) << "getKVBlockValue failed for valid dest_block_index: " << dest_block_index;
    EXPECT_NE(dest_k, nullptr);
    EXPECT_NE(dest_v, nullptr);
    EXPECT_EQ(dest_k->sizeBytes(), k_buffer->sizeBytes());
    EXPECT_EQ(dest_v->sizeBytes(), v_buffer->sizeBytes());

    // 验证buffer值：直接比较HOST内存中的数据
    float* dest_k_data = static_cast<float*>(dest_k->data());
    float* dest_v_data = static_cast<float*>(dest_v->data());

    // 验证每一层的数据值（应该与源块相同）
    for (uint32_t layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        size_t k_offset = layer_id * k_shape;
        size_t v_offset = layer_id * v_shape;

        for (size_t i = 0; i < k_shape; ++i) {
            float expected_k_value = 10.0f + layer_id;
            EXPECT_FLOAT_EQ(dest_k_data[k_offset + i], expected_k_value)
                << "Copied K buffer value mismatch at layer " << layer_id << ", index " << i;
        }
        for (size_t i = 0; i < v_shape; ++i) {
            float expected_v_value = 20.0f + layer_id;
            EXPECT_FLOAT_EQ(dest_v_data[v_offset + i], expected_v_value)
                << "Copied V buffer value mismatch at layer " << layer_id << ", index " << i;
        }
    }

    // 释放资源
    allocator->free({resource});
}

// 测试边界情况：无效的block_index（使用HOST内存）
TEST_F(KVCacheAllocatorTest, testInvalidBlockIndex) {
    auto config    = initConfig(rtp_llm::TYPE_FP32);
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_, rtp_llm::AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    // 创建测试数据
    auto k_shape = config.getKeyShape();
    auto v_shape = config.getValueShape();

    // 从device中allocateBuffer然后memset填充数据
    auto k_buffer = device_->allocateBuffer({config.dtype, {k_shape}, rtp_llm::AllocationType::HOST});
    auto v_buffer = device_->allocateBuffer({config.dtype, {v_shape}, rtp_llm::AllocationType::HOST});

    // 测试无效的block_index
    int invalid_block_index = 999;
    int layer_id            = 0;

    // 这些操作应该返回false，表示操作失败
    bool set_success = allocator->setKVBlockValue(invalid_block_index, layer_id, *k_buffer, *v_buffer);
    EXPECT_FALSE(set_success) << "setKVBlockValue should fail for invalid block_index: " << invalid_block_index;

    auto [get_success, retrieved_k, retrieved_v] = allocator->getKVBlockValue(invalid_block_index, layer_id);
    EXPECT_FALSE(get_success) << "getKVBlockValue should fail for invalid block_index: " << invalid_block_index;
    EXPECT_EQ(retrieved_k, nullptr) << "getKVBlockValue should return nullptr for invalid block_index";
    EXPECT_EQ(retrieved_v, nullptr) << "getKVBlockValue should return nullptr for invalid block_index";
}

// 测试边界情况：无效的layer_id（使用HOST内存）
TEST_F(KVCacheAllocatorTest, testInvalidLayerId) {
    auto config    = initConfig(rtp_llm::TYPE_FP32);
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_, rtp_llm::AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    // 分配一个块
    auto [success, resource] = allocator->malloc({request_id_, 1, false});
    EXPECT_TRUE(success);

    int block_index      = resource.block_id[0];
    int invalid_layer_id = 999;

    // 创建测试数据
    auto k_shape = config.getKeyShape();
    auto v_shape = config.getValueShape();

    // 从device中allocateBuffer然后memset填充数据
    auto k_buffer = device_->allocateBuffer({config.dtype, {k_shape}, rtp_llm::AllocationType::HOST});
    auto v_buffer = device_->allocateBuffer({config.dtype, {v_shape}, rtp_llm::AllocationType::HOST});

    // 测试无效的layer_id
    bool set_success = allocator->setKVBlockValue(block_index, invalid_layer_id, *k_buffer, *v_buffer);
    EXPECT_FALSE(set_success) << "setKVBlockValue should fail for invalid layer_id: " << invalid_layer_id;

    auto [get_success, retrieved_k, retrieved_v] = allocator->getKVBlockValue(block_index, invalid_layer_id);
    EXPECT_FALSE(get_success) << "getKVBlockValue should fail for invalid layer_id: " << invalid_layer_id;
    EXPECT_EQ(retrieved_k, nullptr) << "getKVBlockValue should return nullptr for invalid layer_id";
    EXPECT_EQ(retrieved_v, nullptr) << "getKVBlockValue should return nullptr for invalid layer_id";

    // 释放资源
    allocator->free({resource});
}

// 测试blockCopy的无效参数处理
TEST_F(KVCacheAllocatorTest, testBlockCopyInvalidParameters) {
    auto config    = initConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_, rtp_llm::AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    // 分配一个块
    auto [success, resource] = allocator->malloc({request_id_, 1, false});
    EXPECT_TRUE(success);
    EXPECT_EQ(resource.block_id.size(), 1);

    int valid_block_index   = resource.block_id[0];
    int invalid_block_index = 999;

    // 测试无效的源块索引
    allocator->blockCopy(invalid_block_index, valid_block_index);

    // 测试无效的目标块索引
    allocator->blockCopy(valid_block_index, invalid_block_index);

    // 测试两个都是无效的块索引
    allocator->blockCopy(invalid_block_index, invalid_block_index);

    // 释放资源
    allocator->free({resource});
}

// ==================== BlockRefCounter相关测试 ====================

// 测试BlockRefCounter基本功能
TEST_F(KVCacheAllocatorTest, testBlockRefCounter) {
    auto config    = initConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);
    ASSERT_TRUE(allocator->init());

    // 初始状态：所有块的引用计数应该为0
    for (int i = 1; i < config.block_nums; ++i) {
        EXPECT_EQ(allocator->blockRefCounter().getRefCounter(i), 0);
    }

    // 分配2个块
    auto [success, resource] = allocator->malloc({request_id_, 2, false});
    ASSERT_TRUE(success);
    EXPECT_EQ(resource.block_id.size(), 2);

    // 验证引用计数增加
    for (int block_id : resource.block_id) {
        EXPECT_EQ(allocator->blockRefCounter().getRefCounter(block_id), 1);
    }

    // 增加引用计数
    allocator->incrBlockRefCounter(resource.block_id);
    for (int block_id : resource.block_id) {
        EXPECT_EQ(allocator->blockRefCounter().getRefCounter(block_id), 2);
    }

    // 减少引用计数
    allocator->decrBlockRefCounter(resource.block_id);
    for (int block_id : resource.block_id) {
        EXPECT_EQ(allocator->blockRefCounter().getRefCounter(block_id), 1);
    }

    // 释放资源
    allocator->free({resource});

    // 验证引用计数归零
    for (int block_id : resource.block_id) {
        EXPECT_EQ(allocator->blockRefCounter().getRefCounter(block_id), 0);
    }
}

// ==================== 地址转换功能测试 ====================

// 测试地址转换功能（Normal模式）
TEST_F(KVCacheAllocatorTest, testConvertIndexToAddrNormal) {
    auto config    = initConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);
    ASSERT_TRUE(allocator->init());

    // 分配一个块
    auto [success, resource] = allocator->malloc({request_id_, 1, false});
    EXPECT_TRUE(success);
    int block_index = resource.block_id[0];

    // 测试每一层的地址转换
    for (uint32_t layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        auto addr_info = allocator->convertIndexToAddr(block_index, layer_id);

        // 验证地址不为空
        EXPECT_NE(addr_info.k_addr, nullptr);
        EXPECT_NE(addr_info.v_addr, nullptr);

        // 验证地址在合理范围内
        const auto& kv_buffer = allocator->kvCacheBuffer();
        EXPECT_GE(addr_info.k_addr, kv_buffer.k_blocks->data());
        EXPECT_LT(addr_info.k_addr, (char*)kv_buffer.k_blocks->data() + kv_buffer.k_blocks->sizeBytes());
        EXPECT_GE(addr_info.v_addr, kv_buffer.v_blocks->data());
        EXPECT_LT(addr_info.v_addr, (char*)kv_buffer.v_blocks->data() + kv_buffer.v_blocks->sizeBytes());

        // 验证地址偏移计算
        size_t expected_k_offset = config.getKeyOffset(block_index, layer_id);
        size_t expected_v_offset = config.getValueOffset(block_index, layer_id);

        size_t actual_k_offset = (char*)addr_info.k_addr - (char*)kv_buffer.k_blocks->data();
        size_t actual_v_offset = (char*)addr_info.v_addr - (char*)kv_buffer.v_blocks->data();

        EXPECT_EQ(actual_k_offset, expected_k_offset);
        EXPECT_EQ(actual_v_offset, expected_v_offset);
    }

    // 释放资源
    allocator->free({resource});
}

// 测试地址转换功能（MLA模式）
TEST_F(KVCacheAllocatorTest, testConvertIndexToAddrMla) {
    auto config    = initMlaConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);
    ASSERT_TRUE(allocator->init());

    // 分配一个块
    auto [success, resource] = allocator->malloc({request_id_, 1, false});
    EXPECT_TRUE(success);
    int block_index = resource.block_id[0];

    // 测试每一层的地址转换
    for (uint32_t layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        auto addr_info = allocator->convertIndexToAddr(block_index, layer_id);

        // 验证地址不为空
        EXPECT_NE(addr_info.k_addr, nullptr);
        EXPECT_NE(addr_info.v_addr, nullptr);

        // 验证地址在合理范围内
        const auto& kv_buffer = allocator->kvCacheBuffer();
        EXPECT_GE(addr_info.k_addr, kv_buffer.k_blocks->data());
        EXPECT_LT(addr_info.k_addr, (char*)kv_buffer.k_blocks->data() + kv_buffer.k_blocks->sizeBytes());
        EXPECT_GE(addr_info.v_addr, kv_buffer.v_blocks->data());
        EXPECT_LT(addr_info.v_addr, (char*)kv_buffer.v_blocks->data() + kv_buffer.v_blocks->sizeBytes());

        // 验证MLA模式下的地址偏移计算
        size_t expected_k_offset = config.getKeyOffset(block_index, layer_id);
        size_t expected_v_offset = config.getValueOffset(block_index, layer_id);

        size_t actual_k_offset = (char*)addr_info.k_addr - (char*)kv_buffer.k_blocks->data();
        size_t actual_v_offset = (char*)addr_info.v_addr - (char*)kv_buffer.v_blocks->data();

        EXPECT_EQ(actual_k_offset, expected_k_offset);
        EXPECT_EQ(actual_v_offset, expected_v_offset);
    }

    // 释放资源
    allocator->free({resource});
}

// 测试地址转换的边界情况
TEST_F(KVCacheAllocatorTest, testConvertIndexToAddrEdgeCases) {
    auto config    = initConfig();
    auto allocator = std::make_unique<KVCacheAllocator>(config, device_);
    ASSERT_TRUE(allocator->init());

    // 测试无效的block_index
    int invalid_block_index = 999;
    int layer_id            = 0;

    auto addr_info = allocator->convertIndexToAddr(invalid_block_index, layer_id);
    // 注意：当前实现没有边界检查，这可能需要改进
    // 这里我们只是验证函数不会崩溃

    // 测试无效的layer_id
    int block_index      = 1;
    int invalid_layer_id = 999;

    addr_info = allocator->convertIndexToAddr(block_index, invalid_layer_id);
    // 同样，当前实现没有边界检查
}

}  // namespace rtp_llm
