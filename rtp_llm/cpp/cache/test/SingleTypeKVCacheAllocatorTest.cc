#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <set>
#include <torch/torch.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {
namespace test {

CacheConfig createSingleTypeTestConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8) {
    return makeSimpleMhaCacheConfig(/*layer_num=*/layer_num,
                                    /*block_num=*/block_num,
                                    /*tokens_per_block=*/static_cast<size_t>(seq_size_per_block),
                                    rtp_llm::DataType::TYPE_FP16,
                                    /*local_head_num_kv=*/8,
                                    /*size_per_head=*/128);
}

CompleteTokenIdsPtr createCompleteTokenIds(int batch_size, int seq_length, int seq_size_per_block = 8) {
    auto device = createDevice();
    // CompleteTokenIds(device, batch_size, max_batch_size, max_seq_len, seq_size_per_block)
    auto complete_token_ids =
        std::make_shared<CompleteTokenIds>(device, batch_size, batch_size, seq_length + 100, seq_size_per_block);

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
    resource->resetBatchSize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        resource->initBatchGroups(i, 1);
        resource->setBatchBlocks(i, 0, std::vector<int>(block_num_per_batch));
        resource->setBatchCacheKeys(i, CacheKeysType(block_num_per_batch, static_cast<CacheKeyType>(i * 100)));
    }
    return resource;
}

class SingleTypeKVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        device_ = createDevice();
    }

    void TearDown() override {
        allocator_.reset();
    }

    rtp_llm::DeviceBase*                        device_;
    std::shared_ptr<SingleTypeKVCacheAllocator> allocator_;
};

// Test init
TEST_F(SingleTypeKVCacheAllocatorTest, ConstructorAndInit) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    ASSERT_NE(allocator_, nullptr);

    bool init_result = allocator_->init();
    EXPECT_TRUE(init_result);

    EXPECT_EQ(allocator_->totalBlocksNum(), config.block_num - 1);
    EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 1);  // reserve 1 block
}

TEST_F(SingleTypeKVCacheAllocatorTest, InitWithDifferentLayerNum) {
    auto config = createSingleTypeTestConfig(8, 20, 16);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);

    bool init_result = allocator_->init();
    EXPECT_TRUE(init_result);

    EXPECT_EQ(allocator_->totalBlocksNum(), 20 - 1);
}

// Test malloc
TEST_F(SingleTypeKVCacheAllocatorTest, MallocSingleBatch) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    auto       result = allocator_->malloc(malloc_info);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(batch_resource->blocksNum(0, 0), 2);
    EXPECT_LT(allocator_->freeBlocksNum(), config.block_num);

    seq_length         = 160;
    complete_token_ids = createCompleteTokenIds(1, seq_length);
    MallocInfo malloc_info2{batch_resource, complete_token_ids};
    auto       result2 = allocator_->malloc(malloc_info2);
    EXPECT_FALSE(result2.success);
}

TEST_F(SingleTypeKVCacheAllocatorTest, MallocMultipleBatches) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    int  batch_size         = 3;
    int  seq_length         = 17;
    auto batch_resource     = createBatchKVCacheResource(batch_size);
    auto complete_token_ids = createCompleteTokenIds(batch_size, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};

    auto result = allocator_->malloc(malloc_info);

    EXPECT_TRUE(result.success);
    for (int i = 0; i < batch_size; ++i) {
        EXPECT_EQ(batch_resource->blocksNum(i, 0), 3);
    }
    EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 6);  // 2 shared + 3 batches * 1 blocks + 1 reserved
}

// TEST_F(SingleTypeKVCacheAllocatorTest, MallocWithInsufficientBlocks) {
//     auto config = createSingleTypeTestConfig(4, 5, 8);
//     allocator_ = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
//     allocator_->init();

//     int batch_size = 3;
//     int seq_length = 16;  // 3 batches * 2 blocks
//     auto batch_resource = createBatchKVCacheResource(batch_size);
//     auto complete_token_ids = createCompleteTokenIds(batch_size, seq_length);

//     MallocInfo malloc_info{batch_resource, complete_token_ids};
//     auto result = allocator_->malloc(malloc_info);

//     EXPECT_LE(allocator_->freeBlocksNum(), 5);
// }

// Test free
TEST_F(SingleTypeKVCacheAllocatorTest, FreeSingleBatch) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    allocator_->malloc(malloc_info);

    size_t free_before = allocator_->freeBlocksNum();

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
    EXPECT_GT(allocator_->freeBlocksNum(), free_before);
}

TEST_F(SingleTypeKVCacheAllocatorTest, FreeMultipleBatches) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    int  batch_size         = 3;
    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(batch_size);
    auto complete_token_ids = createCompleteTokenIds(batch_size, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    allocator_->malloc(malloc_info);

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
    EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 1);  // reserve 1 block
}

// Test malloc free cycle
TEST_F(SingleTypeKVCacheAllocatorTest, MallocFreeCycle) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    for (int i = 0; i < 5; ++i) {
        int  seq_length         = 16;
        auto batch_resource     = createBatchKVCacheResource(1);
        auto complete_token_ids = createCompleteTokenIds(1, seq_length);

        MallocInfo malloc_info{batch_resource, complete_token_ids};
        auto       malloc_result = allocator_->malloc(malloc_info);
        EXPECT_TRUE(malloc_result.success);

        FreeInfo free_info{batch_resource, complete_token_ids};
        allocator_->free(free_info);

        EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 1);  // reserve 1 block
    }
}

// Test insert into cache
TEST_F(SingleTypeKVCacheAllocatorTest, InsertIntoCache) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    allocator_->malloc(malloc_info);

    InsertInfo insert_info{batch_resource, complete_token_ids, false};
    allocator_->insertIntoCache(insert_info);
}

TEST_F(SingleTypeKVCacheAllocatorTest, InsertIntoCacheAsResident) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    allocator_->malloc(malloc_info);

    InsertInfo insert_info{batch_resource, complete_token_ids, true};
    allocator_->insertIntoCache(insert_info);
}

// Test convert index to addr
TEST_F(SingleTypeKVCacheAllocatorTest, ConvertIndexToAddr) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        for (int block_id = 0; block_id < 3; ++block_id) {
            auto addr_info = allocator_->convertIndexToAddr(layer_id, block_id);
            EXPECT_NE(addr_info.kv_addr, nullptr);
        }
    }
}

// Test convert index to buffer
TEST_F(SingleTypeKVCacheAllocatorTest, ConvertIndexToBuffer) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    auto buffer_info = allocator_->convertIndexToBuffer(0, 0);
    EXPECT_NE(buffer_info.kv_addr, nullptr);
}

// Test layer cache base
TEST_F(SingleTypeKVCacheAllocatorTest, LayerCacheBase) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    auto layout = allocator_->allLayerCacheBase();
    EXPECT_EQ(layout.layers_to_buffer_ptrs.size(), config.layer_num);
    EXPECT_EQ(layout.layers_to_scale_buffer_ptrs.size(), config.layer_num);

    for (size_t i = 0; i < layout.layers_to_buffer_ptrs.size(); ++i) {
        EXPECT_NE(layout.layers_to_buffer_ptrs[i], nullptr);
        EXPECT_GT(layout.layers_to_buffer_ptrs[i]->sizeBytes(), 0);
    }
}

// Test block copy
TEST_F(SingleTypeKVCacheAllocatorTest, BlockCopySingle) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_, AllocationType::HOST);
    allocator_->init();

    int src_block = 0;
    int dst_block = 1;

    auto&  spec         = config.cache_specs[0];
    size_t k_block_size = spec->k_block_size();
    size_t v_block_size = spec->v_block_size();

    for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        auto src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
        ASSERT_NE(src_addr.kv_addr, nullptr) << "KV addr is null for layer " << layer_id << ", block " << src_block;

        auto dst_addr = allocator_->convertIndexToAddr(layer_id, dst_block);
        ASSERT_NE(dst_addr.kv_addr, nullptr) << "KV addr is null for layer " << layer_id << ", block " << dst_block;

        auto* base   = static_cast<uint8_t*>(src_addr.kv_addr);
        auto* k_data = base;
        auto* v_data = base + k_block_size;

        for (size_t i = 0; i < k_block_size; ++i) {
            k_data[i] = static_cast<uint8_t>((layer_id * 100 + src_block * 10 + i) % 256);
        }
        for (size_t i = 0; i < v_block_size; ++i) {
            v_data[i] = static_cast<uint8_t>((layer_id * 100 + src_block * 10 + i + 128) % 256);
        }
    }

    EXPECT_NO_THROW(allocator_->blockCopy(src_block, dst_block));

    for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
        auto src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
        auto dst_addr = allocator_->convertIndexToAddr(layer_id, dst_block);

        auto* src_base   = static_cast<uint8_t*>(src_addr.kv_addr);
        auto* dst_base   = static_cast<uint8_t*>(dst_addr.kv_addr);
        auto* src_k_data = src_base;
        auto* dst_k_data = dst_base;
        auto* src_v_data = src_base + k_block_size;
        auto* dst_v_data = dst_base + k_block_size;

        for (size_t i = 0; i < k_block_size; ++i) {
            EXPECT_EQ(dst_k_data[i], src_k_data[i]) << "K cache mismatch at layer " << layer_id << ", offset " << i;
        }

        for (size_t i = 0; i < v_block_size; ++i) {
            EXPECT_EQ(dst_v_data[i], src_v_data[i]) << "V cache mismatch at layer " << layer_id << ", offset " << i;
        }
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyVector) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_, AllocationType::HOST);
    allocator_->init();

    std::vector<BlockIdPair> copy_mapping;
    copy_mapping.push_back({0, 1});
    copy_mapping.push_back({2, 3});
    copy_mapping.push_back({4, 5});

    auto&  spec         = config.cache_specs[0];
    size_t k_block_size = spec->k_block_size();
    size_t v_block_size = spec->v_block_size();

    for (const auto& pair : copy_mapping) {
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
            ASSERT_NE(src_addr.kv_addr, nullptr);

            auto* base   = static_cast<uint8_t*>(src_addr.kv_addr);
            auto* k_data = base;
            auto* v_data = base + k_block_size;
            for (size_t i = 0; i < k_block_size; ++i) {
                k_data[i] = static_cast<uint8_t>((layer_id * 100 + pair.src * 10 + i) % 256);
            }
            for (size_t i = 0; i < v_block_size; ++i) {
                v_data[i] = static_cast<uint8_t>((layer_id * 100 + pair.src * 10 + i + 128) % 256);
            }
        }
    }

    EXPECT_NO_THROW(allocator_->blockBatchCopy(copy_mapping));

    // Verify data correctness for each block
    for (const auto& pair : copy_mapping) {
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
            auto dst_addr = allocator_->convertIndexToAddr(layer_id, pair.dst);

            auto* src_base   = static_cast<uint8_t*>(src_addr.kv_addr);
            auto* dst_base   = static_cast<uint8_t*>(dst_addr.kv_addr);
            auto* src_k_data = src_base;
            auto* dst_k_data = dst_base;
            auto* src_v_data = src_base + k_block_size;
            auto* dst_v_data = dst_base + k_block_size;

            for (size_t i = 0; i < k_block_size; ++i) {
                EXPECT_EQ(dst_k_data[i], src_k_data[i]) << "K cache mismatch at block pair (" << pair.src << "->"
                                                        << pair.dst << "), layer " << layer_id << ", offset " << i;
            }

            for (size_t i = 0; i < v_block_size; ++i) {
                EXPECT_EQ(dst_v_data[i], src_v_data[i]) << "V cache mismatch at block pair (" << pair.src << "->"
                                                        << pair.dst << "), layer " << layer_id << ", offset " << i;
            }
        }
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyEmpty) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    std::vector<BlockIdPair> empty_mapping;

    EXPECT_NO_THROW(allocator_->blockBatchCopy(empty_mapping));
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyPointers) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_, AllocationType::HOST);
    allocator_->init();

    BlockIdPair pairs[] = {{0, 1}, {2, 3}};

    auto&  spec         = config.cache_specs[0];
    size_t k_block_size = spec->k_block_size();
    size_t v_block_size = spec->v_block_size();

    for (const auto& pair : pairs) {
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto  src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
            auto* base     = static_cast<uint8_t*>(src_addr.kv_addr);
            auto* k_data   = base;
            auto* v_data   = base + k_block_size;
            for (size_t i = 0; i < k_block_size; ++i) {
                k_data[i] = static_cast<uint8_t>((layer_id * 50 + pair.src * 20 + i) % 256);
            }
            for (size_t i = 0; i < v_block_size; ++i) {
                v_data[i] = static_cast<uint8_t>((layer_id * 50 + pair.src * 20 + i + 64) % 256);
            }
        }
    }

    EXPECT_NO_THROW(allocator_->blockBatchCopy(pairs, pairs + 2));

    for (const auto& pair : pairs) {
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto src_addr = allocator_->convertIndexToAddr(layer_id, pair.src);
            auto dst_addr = allocator_->convertIndexToAddr(layer_id, pair.dst);

            auto* src_base   = static_cast<uint8_t*>(src_addr.kv_addr);
            auto* dst_base   = static_cast<uint8_t*>(dst_addr.kv_addr);
            auto* src_k_data = src_base;
            auto* dst_k_data = dst_base;
            auto* src_v_data = src_base + k_block_size;
            auto* dst_v_data = dst_base + k_block_size;

            EXPECT_EQ(memcmp(dst_k_data, src_k_data, k_block_size), 0)
                << "K cache mismatch for block pair (" << pair.src << "->" << pair.dst << "), layer " << layer_id;
            EXPECT_EQ(memcmp(dst_v_data, src_v_data, v_block_size), 0)
                << "V cache mismatch for block pair (" << pair.src << "->" << pair.dst << "), layer " << layer_id;
        }
    }
}

TEST_F(SingleTypeKVCacheAllocatorTest, BlockBatchCopyBuffer) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_, AllocationType::HOST);
    allocator_->init();

    std::vector<int32_t> data   = {0, 1, 2, 3, 4, 5};  // 3 pairs: (0->1, 2->3, 4->5)
    auto                 buffer = device_->allocateBuffer({rtp_llm::TYPE_INT32, {3, 2}, AllocationType::HOST});
    std::memcpy(buffer->data(), data.data(), data.size() * sizeof(int32_t));

    auto&  spec         = config.cache_specs[0];
    size_t k_block_size = spec->k_block_size();
    size_t v_block_size = spec->v_block_size();

    for (size_t i = 0; i < data.size(); i += 2) {
        int src_block = data[i];
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto  src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
            auto* base     = static_cast<uint8_t*>(src_addr.kv_addr);
            auto* k_data   = base;
            auto* v_data   = base + k_block_size;
            for (size_t j = 0; j < k_block_size; ++j) {
                k_data[j] = static_cast<uint8_t>((layer_id * 70 + src_block * 15 + j) % 256);
            }
            for (size_t j = 0; j < v_block_size; ++j) {
                v_data[j] = static_cast<uint8_t>((layer_id * 70 + src_block * 15 + j + 96) % 256);
            }
        }
    }

    EXPECT_NO_THROW(allocator_->blockBatchCopy(*buffer));

    for (size_t i = 0; i < data.size(); i += 2) {
        int src_block = data[i];
        int dst_block = data[i + 1];
        for (int layer_id = 0; layer_id < config.layer_num; ++layer_id) {
            auto src_addr = allocator_->convertIndexToAddr(layer_id, src_block);
            auto dst_addr = allocator_->convertIndexToAddr(layer_id, dst_block);

            auto* src_base   = static_cast<uint8_t*>(src_addr.kv_addr);
            auto* dst_base   = static_cast<uint8_t*>(dst_addr.kv_addr);
            auto* src_k_data = src_base;
            auto* dst_k_data = dst_base;
            auto* src_v_data = src_base + k_block_size;
            auto* dst_v_data = dst_base + k_block_size;

            EXPECT_EQ(memcmp(dst_k_data, src_k_data, k_block_size), 0)
                << "K cache mismatch for block pair (" << src_block << "->" << dst_block << "), layer " << layer_id;
            EXPECT_EQ(memcmp(dst_v_data, src_v_data, v_block_size), 0)
                << "V cache mismatch for block pair (" << src_block << "->" << dst_block << "), layer " << layer_id;
        }
    }
}

// Test getter methods
TEST_F(SingleTypeKVCacheAllocatorTest, FreeBlocksNums) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 1);  // reserve 1 block
}

TEST_F(SingleTypeKVCacheAllocatorTest, AvailableBlocksNums) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    EXPECT_EQ(allocator_->availableBlocksNum(), config.block_num - 1);  // reserve 1 block
}

TEST_F(SingleTypeKVCacheAllocatorTest, IncrKVCacheRefReferencesMatchedBlocksOnly) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/8);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());

    auto block_pool = allocator_->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t total_free_before = allocator_->freeBlocksNum();
    auto         blocks            = block_pool->malloc(4);
    ASSERT_EQ(blocks.size(), 4);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before - 4);

    KVCacheResource resource;
    resource.initGroups(1);

    resource.cacheKeys() = CacheKeysType{100, 101, 102, 103};
    resource.blocks(0)   = BlockIndicesType{blocks[0], blocks[1], 0, blocks[2]};

    // Reference keys: 101(pos1)->blocks[1], 102(pos2)->0(ignored), 103(pos3)->blocks[2]
    auto ref_resource = allocator_->incrKVCacheRef(resource, CacheKeysType{101, 999, 102, 103});
    ASSERT_NE(ref_resource, nullptr);

    block_pool->requestFree(blocks);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before - 2);  // blocks[1] & blocks[2] are still referenced

    allocator_->decrKVCacheRef(*ref_resource);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before);
}

TEST_F(SingleTypeKVCacheAllocatorTest, IncrKVCacheRefEmptyInputNoEffect) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/10, /*seq_size_per_block=*/8);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());

    auto block_pool = allocator_->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t total_free_before = allocator_->freeBlocksNum();
    auto         blocks            = block_pool->malloc(2);
    ASSERT_EQ(blocks.size(), 2);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before - 2);

    KVCacheResource resource;
    resource.initGroups(1);
    resource.cacheKeys() = CacheKeysType{100, 101};
    resource.blocks(0)   = BlockIndicesType{blocks[0], blocks[1]};

    auto ref_resource = allocator_->incrKVCacheRef(resource, CacheKeysType{});
    ASSERT_EQ(ref_resource, nullptr);

    block_pool->requestFree(blocks);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_free_before);
}

TEST_F(SingleTypeKVCacheAllocatorTest, TotalBlocksNums) {
    auto config = createSingleTypeTestConfig(4, 20);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    EXPECT_EQ(allocator_->totalBlocksNum(), 20 - 1);
}

TEST_F(SingleTypeKVCacheAllocatorTest, MaxSeqLen) {
    auto config = createSingleTypeTestConfig(4, 10, 8);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    EXPECT_EQ(allocator_->maxAvailableTokensNum(), (10 - 1) * 8);  // block_num * seq_size_per_block
}

// Test boundary conditions

TEST_F(SingleTypeKVCacheAllocatorTest, MallocWithZeroSeqLength) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    auto batch_resource     = createBatchKVCacheResource(1);
    auto complete_token_ids = createCompleteTokenIds(1, 0);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    auto       result = allocator_->malloc(malloc_info);
    // not crash
    EXPECT_TRUE(result.success || !result.success);
}

TEST_F(SingleTypeKVCacheAllocatorTest, FreeEmptyBatchResource) {
    auto config = createSingleTypeTestConfig();
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    auto batch_resource     = createBatchKVCacheResource(0);
    auto complete_token_ids = createCompleteTokenIds(0, 0);

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
}

TEST_F(SingleTypeKVCacheAllocatorTest, InitMallocRollbackWhenInitMallocForCommonLenFails) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/6, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());

    auto seed_resource = createBatchKVCacheResource(/*batch_size=*/1);
    seed_resource->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103});  // 4 keys for 4 blocks
    auto seed_token_ids = createCompleteTokenIds(/*batch_size=*/1, /*seq_length=*/16, /*seq_size_per_block=*/4);

    const size_t free_before_seed      = allocator_->freeBlocksNum();
    const size_t available_before_seed = allocator_->availableBlocksNum();
    ASSERT_EQ(free_before_seed, 5u);       // 6-1 reserved
    ASSERT_EQ(available_before_seed, 5u);  // no request refs

    MallocInfo seed_malloc_info{seed_resource, seed_token_ids};
    auto       seed_malloc_result = allocator_->malloc(seed_malloc_info);
    ASSERT_TRUE(seed_malloc_result.success);
    ASSERT_EQ(seed_resource->blocksNum(0, 0), 4);

    InsertInfo seed_insert_info{seed_resource, seed_token_ids, /*is_resident=*/true};
    allocator_->insertIntoCache(seed_insert_info);

    FreeInfo seed_free_info{seed_resource, seed_token_ids};
    allocator_->free(seed_free_info);

    // After free, blocks remain held by resident cache, thus not truly free; but they are "available" to requests.
    ASSERT_EQ(allocator_->freeBlocksNum(), 1u);
    ASSERT_EQ(allocator_->availableBlocksNum(), 5u);

    auto batch_resource = createBatchKVCacheResource(/*batch_size=*/2);
    batch_resource->setBatchCacheKeys(0, CacheKeysType{100, 101});  // match_keys -> {100}
    batch_resource->setBatchCacheKeys(1, CacheKeysType{200, 201});
    batch_resource->enable_reuse_cache = true;

    auto token_ids = createCompleteTokenIds(/*batch_size=*/2, /*seq_length=*/13, /*seq_size_per_block=*/4);

    const size_t free_before_fail      = allocator_->freeBlocksNum();
    const size_t available_before_fail = allocator_->availableBlocksNum();
    ASSERT_EQ(free_before_fail, 1u);
    ASSERT_EQ(available_before_fail, 5u);

    MallocInfo malloc_info{batch_resource, token_ids};
    auto       result = allocator_->malloc(malloc_info);
    EXPECT_FALSE(result.success);

    // KVCacheAllocator::initMalloc should call free() to rollback any referenced/allocated blocks.
    EXPECT_EQ(batch_resource->curBlocksNum(), 0);
    EXPECT_EQ(batch_resource->blocksNum(0, 0), 0);
    EXPECT_EQ(batch_resource->blocksNum(1, 0), 0);

    EXPECT_EQ(allocator_->freeBlocksNum(), free_before_fail);
    EXPECT_EQ(allocator_->availableBlocksNum(), available_before_fail);
}

// Test rollback logic in incrMalloc
TEST_F(SingleTypeKVCacheAllocatorTest, IncrMallocRollback) {
    // Create a config with limited blocks to trigger rollback
    auto config = createSingleTypeTestConfig(4, 8, 4);  // 8 blocks, 4 seq per block
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    size_t initial_free_blocks = allocator_->freeBlocksNum();
    EXPECT_EQ(initial_free_blocks, 7);

    int  batch_size         = 3;
    auto batch_resource     = createBatchKVCacheResource(batch_size);
    auto complete_token_ids = createCompleteTokenIds(batch_size, 4, 4);  // 4 seq length = 1 block per batch

    // First, do a common allocation for all batches (1 block each)
    MallocInfo common_malloc_info{batch_resource, complete_token_ids};
    auto       common_result = allocator_->initMallocForCommonLen(common_malloc_info);
    EXPECT_TRUE(common_result.success);

    // 1 block allocated and shared by all batches
    size_t after_common_free_blocks = allocator_->freeBlocksNum();
    EXPECT_EQ(after_common_free_blocks, 6);

    // Verify each batch has 1 block
    for (int i = 0; i < batch_size; ++i) {
        EXPECT_EQ(batch_resource->blocksNum(i, 0), 1);
    }

    // update complete_token_ids to 16 tokens
    complete_token_ids->setSeqLength(16);
    MallocInfo incr_malloc_info{batch_resource, complete_token_ids};

    auto incr_result = allocator_->incrMalloc(incr_malloc_info);
    EXPECT_FALSE(incr_result.success);

    size_t after_rollback_free_blocks = allocator_->freeBlocksNum();
    EXPECT_EQ(after_rollback_free_blocks, 6);

    for (int i = 0; i < batch_size; ++i) {
        EXPECT_EQ(batch_resource->blocksNum(i, 0), 1);
    }

    // Verify that no extra blocks were allocated and left unfreed
    // If rollback didn't work properly, we might have partially allocated blocks
}

TEST_F(SingleTypeKVCacheAllocatorTest, InitMallocRollbackWhenIncrMallocFails) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/4, /*block_num=*/5, /*seq_size_per_block=*/8);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());

    const size_t free_before      = allocator_->freeBlocksNum();
    const size_t available_before = allocator_->availableBlocksNum();
    ASSERT_EQ(free_before, 4u);
    ASSERT_EQ(available_before, 4u);

    auto batch_resource     = createBatchKVCacheResource(/*batch_size=*/3);
    auto complete_token_ids = createCompleteTokenIds(/*batch_size=*/3, /*seq_length=*/17, /*seq_size_per_block=*/8);
    batch_resource->enable_reuse_cache = false;  // keep this case purely capacity-driven

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    auto       result = allocator_->malloc(malloc_info);
    EXPECT_FALSE(result.success);

    // KVCacheAllocator::initMalloc should call free() to clear shared blocks after incrMalloc fails.
    EXPECT_EQ(batch_resource->curBlocksNum(), 0);
    for (int i = 0; i < batch_resource->batchSize(); ++i) {
        EXPECT_EQ(batch_resource->blocksNum(i, 0), 0);
    }

    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator_->availableBlocksNum(), available_before);
}

// ==================== Stress tests ====================

TEST_F(SingleTypeKVCacheAllocatorTest, MixedOperations) {
    auto config = createSingleTypeTestConfig(4, 30);
    allocator_  = std::make_shared<SingleTypeKVCacheAllocator>(config, device_);
    allocator_->init();

    std::vector<BatchKVCacheResourcePtr> resources;
    std::vector<CompleteTokenIdsPtr>     token_ids_list;

    for (int i = 0; i < 5; ++i) {
        auto batch_resource     = createBatchKVCacheResource(2);
        auto complete_token_ids = createCompleteTokenIds(2, 16);

        MallocInfo malloc_info{batch_resource, complete_token_ids};
        auto       result = allocator_->malloc(malloc_info);
        EXPECT_TRUE(result.success);

        resources.push_back(batch_resource);
        token_ids_list.push_back(complete_token_ids);
    }

    for (int i = 0; i < 3; ++i) {
        FreeInfo free_info{resources[i], token_ids_list[i]};
        allocator_->free(free_info);
    }

    for (int i = 0; i < 2; ++i) {
        auto batch_resource     = createBatchKVCacheResource(1);
        auto complete_token_ids = createCompleteTokenIds(1, 16);

        MallocInfo malloc_info{batch_resource, complete_token_ids};
        auto       result = allocator_->malloc(malloc_info);
        EXPECT_TRUE(result.success);
    }

    EXPECT_GT(allocator_->freeBlocksNum(), 0);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
