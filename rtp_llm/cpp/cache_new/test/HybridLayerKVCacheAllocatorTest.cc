#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <set>
#include <torch/torch.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache_new/HybridLayerKVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/CacheConfigCreator.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/cache_new/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {
namespace test {

CacheConfig createHybridLayerTestConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8) {
    CacheConfig config;
    config.layer_type_num     = 1;
    config.layer_num          = layer_num;
    config.block_num          = block_num;
    config.seq_size_per_block = seq_size_per_block;

    auto mha_spec                = std::make_shared<MHAKVCacheSpec>();
    mha_spec->layer_num          = layer_num;
    mha_spec->block_nums         = block_num;
    mha_spec->local_head_num_kv  = 8;
    mha_spec->size_per_head      = 128;
    mha_spec->seq_size_per_block = seq_size_per_block;
    mha_spec->dtype              = rtp_llm::DataType::TYPE_FP16;
    mha_spec->type               = KVCacheType::MultiHeadAttention;

    config.layer_type_params.push_back(mha_spec);

    std::vector<int> layer_ids(layer_num);
    for (int i = 0; i < layer_num; ++i) {
        layer_ids[i] = i;
    }
    config.layer_ids.push_back(layer_ids);

    return config;
}

CompleteTokenIdsPtr createCompleteTokenIds(int batch_size, int seq_length) {
    auto device = createDevice();
    // CompleteTokenIds(device, batch_size, max_batch_size, max_seq_len, seq_size_per_block)
    auto complete_token_ids = std::make_shared<CompleteTokenIds>(device, batch_size, batch_size, seq_length + 100, 8);

    auto input_ids = device->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {(size_t)seq_length}, rtp_llm::AllocationType::HOST}, {});
    auto* token_data = input_ids->data<int32_t>();
    for (int i = 0; i < seq_length; ++i) {
        token_data[i] = i + 1;
    }

    auto generate_input             = std::make_shared<GenerateInput>();
    generate_input->input_ids       = input_ids;
    generate_input->generate_config = std::make_shared<GenerateConfig>();

    complete_token_ids->init(generate_input);

    return complete_token_ids;
}

BatchKVCacheResourcePtr createBatchKVCacheResource(int batch_size, int block_num_per_batch = 0) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->batch_block_id.resize(batch_size);
    resource->cache_keys.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        resource->batch_block_id[i] = std::vector<int>(block_num_per_batch);
        resource->cache_keys[i]     = std::vector<size_t>(block_num_per_batch, i * 100);
    }
    return resource;
}

class HybridLayerKVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        device_ = createDevice();
    }

    void TearDown() override {
        allocator_.reset();
    }

    rtp_llm::DeviceBase*                         device_;
    std::shared_ptr<HybridLayerKVCacheAllocator> allocator_;
};

// Test init
// TEST_F(HybridLayerKVCacheAllocatorTest, ConstructorAndInit) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     ASSERT_NE(allocator_, nullptr);

//     bool init_result = allocator_->init();
//     EXPECT_TRUE(init_result);

//     EXPECT_EQ(allocator_->totalBlocksNums(), config.block_num);
//     EXPECT_EQ(allocator_->freeBlocksNums(), config.block_num - 1);  // reserve 1 block
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, InitWithDifferentLayerNum) {
//     auto config = createHybridLayerTestConfig(8, 20, 16);
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);

//     bool init_result = allocator_->init();
//     EXPECT_TRUE(init_result);

//     EXPECT_EQ(allocator_->totalBlocksNums(), 20);
// }

// // Test malloc
// TEST_F(HybridLayerKVCacheAllocatorTest, MallocSingleBatch) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     int  seq_length         = 16;
//     auto batch_resource     = createBatchKVCacheResource(1);
//     auto complete_token_ids = createCompleteTokenIds(1, seq_length);

//     MallocInfo malloc_info(batch_resource, complete_token_ids);
//     auto       result = allocator_->malloc(malloc_info);

//     EXPECT_TRUE(result.success);
//     EXPECT_EQ(batch_resource->batch_block_id[0].size(), 2);
//     EXPECT_LT(allocator_->freeBlocksNums(), config.block_num);

//     seq_length         = 160;
//     complete_token_ids = createCompleteTokenIds(1, seq_length);
//     MallocInfo malloc_info2(batch_resource, complete_token_ids);
//     auto       result2 = allocator_->malloc(malloc_info2);
//     EXPECT_FALSE(result2.success);
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, MallocMultipleBatches) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     int  batch_size         = 3;
//     int  seq_length         = 16;
//     auto batch_resource     = createBatchKVCacheResource(batch_size);
//     auto complete_token_ids = createCompleteTokenIds(batch_size, seq_length);

//     MallocInfo malloc_info(batch_resource, complete_token_ids);
//     auto       result = allocator_->malloc(malloc_info);

//     EXPECT_TRUE(result.success);
//     for (int i = 0; i < batch_size; ++i) {
//         EXPECT_EQ(batch_resource->batch_block_id[i].size(), 2);
//     }
//     EXPECT_EQ(allocator_->freeBlocksNums(), config.block_num - 7);  // 3 batches * 2 blocks + 1 reserved
// }

// // TEST_F(HybridLayerKVCacheAllocatorTest, MallocWithInsufficientBlocks) {
// //     auto config = createHybridLayerTestConfig(4, 5, 8);
// //     allocator_ = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
// //     allocator_->init();

// //     int batch_size = 3;
// //     int seq_length = 16;  // 3 batches * 2 blocks
// //     auto batch_resource = createBatchKVCacheResource(batch_size);
// //     auto complete_token_ids = createCompleteTokenIds(batch_size, seq_length);

// //     MallocInfo malloc_info(batch_resource, complete_token_ids);
// //     auto result = allocator_->malloc(malloc_info);

// //     EXPECT_LE(allocator_->freeBlocksNums(), 5);
// // }

// // Test free
// TEST_F(HybridLayerKVCacheAllocatorTest, FreeSingleBatch) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     int  seq_length         = 16;
//     auto batch_resource     = createBatchKVCacheResource(1);
//     auto complete_token_ids = createCompleteTokenIds(1, seq_length);

//     MallocInfo malloc_info(batch_resource, complete_token_ids);
//     allocator_->malloc(malloc_info);

//     size_t free_before = allocator_->freeBlocksNums();

//     FreeInfo free_info(batch_resource, complete_token_ids);
//     auto     result = allocator_->free(free_info);

//     EXPECT_TRUE(result.success);
//     EXPECT_GT(allocator_->freeBlocksNums(), free_before);
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, FreeMultipleBatches) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     int  batch_size         = 3;
//     int  seq_length         = 16;
//     auto batch_resource     = createBatchKVCacheResource(batch_size);
//     auto complete_token_ids = createCompleteTokenIds(batch_size, seq_length);

//     MallocInfo malloc_info(batch_resource, complete_token_ids);
//     allocator_->malloc(malloc_info);

//     FreeInfo free_info(batch_resource, complete_token_ids);
//     auto     result = allocator_->free(free_info);

//     EXPECT_TRUE(result.success);
//     EXPECT_EQ(allocator_->freeBlocksNums(), config.block_num - 1);  // reserve 1 block
// }

// // Test malloc free cycle
// TEST_F(HybridLayerKVCacheAllocatorTest, MallocFreeCycle) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     for (int i = 0; i < 5; ++i) {
//         int  seq_length         = 16;
//         auto batch_resource     = createBatchKVCacheResource(1);
//         auto complete_token_ids = createCompleteTokenIds(1, seq_length);

//         MallocInfo malloc_info(batch_resource, complete_token_ids);
//         auto       malloc_result = allocator_->malloc(malloc_info);
//         EXPECT_TRUE(malloc_result.success);

//         FreeInfo free_info(batch_resource, complete_token_ids);
//         auto     free_result = allocator_->free(free_info);
//         EXPECT_TRUE(free_result.success);

//         EXPECT_EQ(allocator_->freeBlocksNums(), config.block_num - 1);  // reserve 1 block
//     }
// }

// // Test insert into cache
// TEST_F(HybridLayerKVCacheAllocatorTest, InsertIntoCache) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     int  seq_length         = 16;
//     auto batch_resource     = createBatchKVCacheResource(1);
//     auto complete_token_ids = createCompleteTokenIds(1, seq_length);

//     MallocInfo malloc_info(batch_resource, complete_token_ids);
//     allocator_->malloc(malloc_info);

//     InsertInfo insert_info(batch_resource, complete_token_ids, false);
//     auto       result = allocator_->insertIntoCache(insert_info);

//     EXPECT_TRUE(result.success);
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, InsertIntoCacheAsResident) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     int  seq_length         = 16;
//     auto batch_resource     = createBatchKVCacheResource(1);
//     auto complete_token_ids = createCompleteTokenIds(1, seq_length);

//     MallocInfo malloc_info(batch_resource, complete_token_ids);
//     allocator_->malloc(malloc_info);

//     InsertInfo insert_info(batch_resource, complete_token_ids, true);
//     auto       result = allocator_->insertIntoCache(insert_info);

//     EXPECT_TRUE(result.success);
// }

// // Test convert index to addr
// TEST_F(HybridLayerKVCacheAllocatorTest, ConvertIndexToAddr) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
//         for (int block_id = 0; block_id < 3; ++block_id) {
//             auto addr_info = allocator_->convertIndexToAddr(layer_id, block_id);
//             EXPECT_NE(addr_info.k_addr, nullptr);
//             EXPECT_NE(addr_info.v_addr, nullptr);
//         }
//     }
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, ConvertIndexToAddrInvalidLayer) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     int  invalid_layer = config.layer_num + 10;
//     auto addr_info     = allocator_->convertIndexToAddr(invalid_layer, 0);
//     EXPECT_EQ(addr_info.k_addr, nullptr);
//     EXPECT_EQ(addr_info.v_addr, nullptr);
// }

// // Test convert index to buffer
// TEST_F(HybridLayerKVCacheAllocatorTest, ConvertIndexToBuffer) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     auto buffer_info = allocator_->convertIndexToBuffer(0, 0);
//     EXPECT_NE(buffer_info.k_addr, nullptr);
//     EXPECT_NE(buffer_info.v_addr, nullptr);
// }

// // Test layer cache base
// TEST_F(HybridLayerKVCacheAllocatorTest, LayerCacheBase) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     auto layout = allocator_->layerCacheBase();
//     EXPECT_EQ(layout.layers_to_buffer_ptrs.size(), config.layer_num);

//     for (size_t i = 0; i < layout.layers_to_buffer_ptrs.size(); ++i) {
//         EXPECT_TRUE(layout.layers_to_buffer_ptrs[i].defined());
//         EXPECT_GT(layout.layers_to_buffer_ptrs[i].numel(), 0);
//     }
// }

// // Test block copy
// TEST_F(HybridLayerKVCacheAllocatorTest, BlockCopySingle) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_, AllocationType::HOST);
//     allocator_->init();

//     int src_block = 0;
//     int dst_block = 1;

//     auto&  spec         = config.layer_type_params[0];
//     size_t k_block_size = spec->k_block_size();
//     size_t v_block_size = spec->v_block_size();

//     for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
//         auto src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
//         ASSERT_NE(src_addr.k_addr, nullptr) << "K addr is null for layer " << layer_id << ", block " << src_block;
//         ASSERT_NE(src_addr.v_addr, nullptr) << "V addr is null for layer " << layer_id << ", block " << src_block;

//         auto dst_addr = allocator_->convertIndexToAddr(layer_id, dst_block);
//         ASSERT_NE(dst_addr.k_addr, nullptr) << "K addr is null for layer " << layer_id << ", block " << dst_block;
//         ASSERT_NE(dst_addr.v_addr, nullptr) << "V addr is null for layer " << layer_id << ", block " << dst_block;

//         auto* k_data = static_cast<uint8_t*>(src_addr.k_addr);
//         auto* v_data = static_cast<uint8_t*>(src_addr.v_addr);

//         for (size_t i = 0; i < k_block_size; ++i) {
//             k_data[i] = static_cast<uint8_t>((layer_id * 100 + src_block * 10 + i) % 256);
//         }
//         for (size_t i = 0; i < v_block_size; ++i) {
//             v_data[i] = static_cast<uint8_t>((layer_id * 100 + src_block * 10 + i + 128) % 256);
//         }
//     }

//     EXPECT_NO_THROW(allocator_->blockCopy(src_block, dst_block));

//     for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
//         auto src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
//         auto dst_addr = allocator_->convertIndexToAddr(layer_id, dst_block);

//         auto* src_k_data = static_cast<uint8_t*>(src_addr.k_addr);
//         auto* dst_k_data = static_cast<uint8_t*>(dst_addr.k_addr);
//         auto* src_v_data = static_cast<uint8_t*>(src_addr.v_addr);
//         auto* dst_v_data = static_cast<uint8_t*>(dst_addr.v_addr);

//         for (size_t i = 0; i < k_block_size; ++i) {
//             EXPECT_EQ(dst_k_data[i], src_k_data[i]) << "K cache mismatch at layer " << layer_id << ", offset " << i;
//         }

//         for (size_t i = 0; i < v_block_size; ++i) {
//             EXPECT_EQ(dst_v_data[i], src_v_data[i]) << "V cache mismatch at layer " << layer_id << ", offset " << i;
//         }
//     }
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, BlockBatchCopyVector) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_, AllocationType::HOST);
//     allocator_->init();

//     std::vector<BlockIdPair> copy_mapping;
//     copy_mapping.push_back({0, 1});
//     copy_mapping.push_back({2, 3});
//     copy_mapping.push_back({4, 5});

//     auto&  spec         = config.layer_type_params[0];
//     size_t k_block_size = spec->k_block_size();
//     size_t v_block_size = spec->v_block_size();

//     for (const auto& pair : copy_mapping) {
//         for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
//             auto src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
//             ASSERT_NE(src_addr.k_addr, nullptr);
//             ASSERT_NE(src_addr.v_addr, nullptr);

//             auto* k_data = static_cast<uint8_t*>(src_addr.k_addr);
//             auto* v_data = static_cast<uint8_t*>(src_addr.v_addr);
//             for (size_t i = 0; i < k_block_size; ++i) {
//                 k_data[i] = static_cast<uint8_t>((layer_id * 100 + pair.src * 10 + i) % 256);
//             }
//             for (size_t i = 0; i < v_block_size; ++i) {
//                 v_data[i] = static_cast<uint8_t>((layer_id * 100 + pair.src * 10 + i + 128) % 256);
//             }
//         }
//     }

//     EXPECT_NO_THROW(allocator_->blockBatchCopy(copy_mapping));

//     // 验证每个 block 的数据正确性
//     for (const auto& pair : copy_mapping) {
//         for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
//             auto src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
//             auto dst_addr = allocator_->convertIndexToAddr(layer_id, pair.dst);

//             auto* src_k_data = static_cast<uint8_t*>(src_addr.k_addr);
//             auto* dst_k_data = static_cast<uint8_t*>(dst_addr.k_addr);
//             auto* src_v_data = static_cast<uint8_t*>(src_addr.v_addr);
//             auto* dst_v_data = static_cast<uint8_t*>(dst_addr.v_addr);

//             for (size_t i = 0; i < k_block_size; ++i) {
//                 EXPECT_EQ(dst_k_data[i], src_k_data[i]) << "K cache mismatch at block pair (" << pair.src << "->"
//                                                         << pair.dst << "), layer " << layer_id << ", offset " << i;
//             }

//             for (size_t i = 0; i < v_block_size; ++i) {
//                 EXPECT_EQ(dst_v_data[i], src_v_data[i]) << "V cache mismatch at block pair (" << pair.src << "->"
//                                                         << pair.dst << "), layer " << layer_id << ", offset " << i;
//             }
//         }
//     }
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, BlockBatchCopyEmpty) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     std::vector<BlockIdPair> empty_mapping;

//     EXPECT_NO_THROW(allocator_->blockBatchCopy(empty_mapping));
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, BlockBatchCopyPointers) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_, AllocationType::HOST);
//     allocator_->init();

//     BlockIdPair pairs[] = {{0, 1}, {2, 3}};

//     auto&  spec         = config.layer_type_params[0];
//     size_t k_block_size = spec->k_block_size();
//     size_t v_block_size = spec->v_block_size();

//     for (const auto& pair : pairs) {
//         for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
//             auto  src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
//             auto* k_data   = static_cast<uint8_t*>(src_addr.k_addr);
//             auto* v_data   = static_cast<uint8_t*>(src_addr.v_addr);
//             for (size_t i = 0; i < k_block_size; ++i) {
//                 k_data[i] = static_cast<uint8_t>((layer_id * 50 + pair.src * 20 + i) % 256);
//             }
//             for (size_t i = 0; i < v_block_size; ++i) {
//                 v_data[i] = static_cast<uint8_t>((layer_id * 50 + pair.src * 20 + i + 64) % 256);
//             }
//         }
//     }

//     EXPECT_NO_THROW(allocator_->blockBatchCopy(pairs, pairs + 2));

//     for (const auto& pair : pairs) {
//         for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
//             auto src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
//             auto dst_addr = allocator_->convertIndexToAddr(layer_id, pair.dst);

//             auto* src_k_data = static_cast<uint8_t*>(src_addr.k_addr);
//             auto* dst_k_data = static_cast<uint8_t*>(dst_addr.k_addr);
//             auto* src_v_data = static_cast<uint8_t*>(src_addr.v_addr);
//             auto* dst_v_data = static_cast<uint8_t*>(dst_addr.v_addr);

//             EXPECT_EQ(memcmp(dst_k_data, src_k_data, k_block_size), 0)
//                 << "K cache mismatch for block pair (" << pair.src << "->" << pair.dst << "), layer " << layer_id;
//             EXPECT_EQ(memcmp(dst_v_data, src_v_data, v_block_size), 0)
//                 << "V cache mismatch for block pair (" << pair.src << "->" << pair.dst << "), layer " << layer_id;
//         }
//     }
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, BlockBatchCopyBuffer) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_, AllocationType::HOST);
//     allocator_->init();

//     std::vector<int32_t> data   = {0, 1, 2, 3, 4, 5};  // 3 pairs: (0->1, 2->3, 4->5)
//     auto                 buffer = device_->allocateBuffer({rtp_llm::TYPE_INT32, {3, 2}, AllocationType::HOST});
//     std::memcpy(buffer->data(), data.data(), data.size() * sizeof(int32_t));

//     auto&  spec         = config.layer_type_params[0];
//     size_t k_block_size = spec->k_block_size();
//     size_t v_block_size = spec->v_block_size();

//     for (size_t i = 0; i < data.size(); i += 2) {
//         int src_block = data[i];
//         for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
//             auto  src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
//             auto* k_data   = static_cast<uint8_t*>(src_addr.k_addr);
//             auto* v_data   = static_cast<uint8_t*>(src_addr.v_addr);
//             for (size_t j = 0; j < k_block_size; ++j) {
//                 k_data[j] = static_cast<uint8_t>((layer_id * 70 + src_block * 15 + j) % 256);
//             }
//             for (size_t j = 0; j < v_block_size; ++j) {
//                 v_data[j] = static_cast<uint8_t>((layer_id * 70 + src_block * 15 + j + 96) % 256);
//             }
//         }
//     }

//     EXPECT_NO_THROW(allocator_->blockBatchCopy(*buffer));

//     for (size_t i = 0; i < data.size(); i += 2) {
//         int src_block = data[i];
//         int dst_block = data[i + 1];
//         for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
//             auto src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
//             auto dst_addr = allocator_->convertIndexToAddr(layer_id, dst_block);

//             auto* src_k_data = static_cast<uint8_t*>(src_addr.k_addr);
//             auto* dst_k_data = static_cast<uint8_t*>(dst_addr.k_addr);
//             auto* src_v_data = static_cast<uint8_t*>(src_addr.v_addr);
//             auto* dst_v_data = static_cast<uint8_t*>(dst_addr.v_addr);

//             EXPECT_EQ(memcmp(dst_k_data, src_k_data, k_block_size), 0)
//                 << "K cache mismatch for block pair (" << src_block << "->" << dst_block << "), layer " << layer_id;
//             EXPECT_EQ(memcmp(dst_v_data, src_v_data, v_block_size), 0)
//                 << "V cache mismatch for block pair (" << src_block << "->" << dst_block << "), layer " << layer_id;
//         }
//     }
// }

// // Test getter methods
// TEST_F(HybridLayerKVCacheAllocatorTest, FreeBlocksNums) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     EXPECT_EQ(allocator_->freeBlocksNums(), config.block_num - 1);  // reserve 1 block
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, AvailableBlocksNums) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     EXPECT_EQ(allocator_->availableBlocksNums(), config.block_num - 1);  // reserve 1 block
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, TotalBlocksNums) {
//     auto config = createHybridLayerTestConfig(4, 20);
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     EXPECT_EQ(allocator_->totalBlocksNums(), 20);
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, MaxSeqLen) {
//     auto config = createHybridLayerTestConfig(4, 10, 8);
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     EXPECT_EQ(allocator_->maxSeqLen(), 10 * 8);  // block_num * seq_size_per_block
// }

// // Test boundary conditions

// TEST_F(HybridLayerKVCacheAllocatorTest, MallocWithZeroSeqLength) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     auto batch_resource     = createBatchKVCacheResource(1);
//     auto complete_token_ids = createCompleteTokenIds(1, 0);

//     MallocInfo malloc_info(batch_resource, complete_token_ids);
//     auto       result = allocator_->malloc(malloc_info);
//     // not crash
//     EXPECT_TRUE(result.success || !result.success);
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, FreeEmptyBatchResource) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     auto batch_resource     = createBatchKVCacheResource(0);
//     auto complete_token_ids = createCompleteTokenIds(0, 0);

//     FreeInfo free_info(batch_resource, complete_token_ids);
//     auto     result = allocator_->free(free_info);

//     EXPECT_TRUE(result.success);
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, MallocWithNullBatchResource) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     auto complete_token_ids = createCompleteTokenIds(1, 16);

//     MallocInfo malloc_info(nullptr, complete_token_ids);
//     auto       result = allocator_->malloc(malloc_info);

//     EXPECT_FALSE(result.success);
// }

// TEST_F(HybridLayerKVCacheAllocatorTest, FreeWithNullBatchResource) {
//     auto config = createHybridLayerTestConfig();
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     auto complete_token_ids = createCompleteTokenIds(1, 16);

//     FreeInfo free_info(nullptr, complete_token_ids);
//     auto     result = allocator_->free(free_info);

//     EXPECT_FALSE(result.success);
// }

// // ==================== 压力测试 ====================

// TEST_F(HybridLayerKVCacheAllocatorTest, MixedOperations) {
//     auto config = createHybridLayerTestConfig(4, 20);
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
//     allocator_->init();

//     std::vector<BatchKVCacheResourcePtr> resources;
//     std::vector<CompleteTokenIdsPtr>     token_ids_list;

//     for (int i = 0; i < 5; ++i) {
//         auto batch_resource     = createBatchKVCacheResource(2);
//         auto complete_token_ids = createCompleteTokenIds(2, 16);

//         MallocInfo malloc_info(batch_resource, complete_token_ids);
//         auto       result = allocator_->malloc(malloc_info);
//         EXPECT_TRUE(result.success);

//         resources.push_back(batch_resource);
//         token_ids_list.push_back(complete_token_ids);
//     }

//     for (int i = 0; i < 3; ++i) {
//         FreeInfo free_info(resources[i], token_ids_list[i]);
//         auto     result = allocator_->free(free_info);
//         EXPECT_TRUE(result.success);
//     }

//     for (int i = 0; i < 2; ++i) {
//         auto batch_resource     = createBatchKVCacheResource(1);
//         auto complete_token_ids = createCompleteTokenIds(1, 16);

//         MallocInfo malloc_info(batch_resource, complete_token_ids);
//         auto       result = allocator_->malloc(malloc_info);
//         EXPECT_TRUE(result.success);
//     }

//     EXPECT_GT(allocator_->freeBlocksNums(), 0);
// }

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
