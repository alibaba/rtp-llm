#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <set>
#include <torch/torch.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/HybridLayerKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {
namespace test {

CacheConfig createHybridLayerTestConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8) {
    CacheConfig config;
    config.layer_num          = layer_num;
    config.block_num          = block_num;
    config.seq_size_per_block = seq_size_per_block;

    auto mha_spec                = std::make_shared<MHAKVCacheSpec>();
    mha_spec->layer_num          = layer_num;
    mha_spec->local_head_num_kv  = 8;
    mha_spec->size_per_head      = 128;
    mha_spec->seq_size_per_block = seq_size_per_block;
    mha_spec->dtype              = rtp_llm::DataType::TYPE_FP16;
    mha_spec->type               = KVCacheType::MultiHeadAttention;

    config.cache_specs.push_back(mha_spec);

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

BatchKVCacheResourcePtr
createBatchKVCacheResource(int batch_size, int group_nums = 0, int layer_num = 0, int cache_key_nums = 0) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->batch_resource.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < cache_key_nums; ++j) {
            resource->batch_resource[i].cache_keys.push_back(100 + j);
        }
        resource->batch_resource[i].initGroups(group_nums, layer_num);
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

TEST_F(HybridLayerKVCacheAllocatorTest, ConstructorAndInit) {
    auto config = createHybridLayerTestConfig();
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    ASSERT_NE(allocator_, nullptr);

    bool init_result = allocator_->init();
    EXPECT_TRUE(init_result);

    EXPECT_EQ(allocator_->totalBlocksNum(), config.block_num - 1);
    EXPECT_EQ(allocator_->freeBlocksNum(), config.block_num - 1);  // reserve 1 block
}

// TEST_F(HybridLayerKVCacheAllocatorTest, InitWithDifferentLayerNum) {
//     auto config = createHybridLayerTestConfig(8, 20, 16);
//     allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);

//     bool init_result = allocator_->init();
//     EXPECT_TRUE(init_result);

//     EXPECT_EQ(allocator_->totalBlocksNum(), 20);
// }

TEST_F(HybridLayerKVCacheAllocatorTest, ReuseCache) {
    auto config = createHybridLayerTestConfig();
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();
    auto block_pool  = allocator_->getBlockPool();
    auto block_cache = block_pool->blockCache();

    BlockCache::CacheItem item1   = {101, 0, 1, false};
    auto                  result1 = block_cache->put(item1);
    EXPECT_TRUE(result1);

    BlockCache::CacheItem item2   = {102, 0, 2, false};
    auto                  result2 = block_cache->put(item2);
    EXPECT_TRUE(result2);

    BlockCache::CacheItem item3   = {103, 0, 3, false};
    auto                  result3 = block_cache->put(item3);
    EXPECT_TRUE(result3);

    BlockCache::CacheItem item4   = {102, 1, 4, false};
    auto                  result4 = block_cache->put(item4);
    EXPECT_TRUE(result4);

    CacheKeysType     cache_keys{101, 102, 103, 104};
    KVCacheResourceV1 resource;
    auto&             group_ids = resource.groupBlocks();
    ;
    for (int i = 0; i < 3; i++) {
        group_ids.push_back(std::make_shared<BlockIds>());
    }

    // full group 和 linear group，混合来做匹配。
    int reuse_blocks1 = allocator_->reuseCache(cache_keys, resource);
    ASSERT_EQ(reuse_blocks1, 0);

    BlockCache::CacheItem item5   = {102, 2, 5, false};
    auto                  result5 = block_cache->put(item5);
    EXPECT_TRUE(result5);

    int reuse_blocks2 = allocator_->reuseCache(cache_keys, resource);
    ASSERT_EQ(reuse_blocks2, 2);

    ASSERT_EQ(group_ids[0]->block_indices.size(), 2);
    ASSERT_EQ(group_ids[0]->block_indices[0], 1);
    ASSERT_EQ(group_ids[0]->block_indices[1], 2);

    ASSERT_EQ(group_ids[1]->block_indices.size(), 2);
    ASSERT_EQ(group_ids[1]->block_indices[0], -1);
    ASSERT_EQ(group_ids[1]->block_indices[1], 4);

    ASSERT_EQ(group_ids[2]->block_indices.size(), 2);
    ASSERT_EQ(group_ids[2]->block_indices[0], -1);
    ASSERT_EQ(group_ids[2]->block_indices[1], 5);
}

TEST_F(HybridLayerKVCacheAllocatorTest, IncrMallocSingleBatch) {
    auto config = createHybridLayerTestConfig();
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();
    auto total_blocks = allocator_->freeBlocksNum();

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(1, 3);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    auto       result = allocator_->incrMalloc(malloc_info);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(batch_resource->batch_resource[0].blocksNum(), 2);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - 3 * 2);

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);
}

TEST_F(HybridLayerKVCacheAllocatorTest, IncrMallocMultiBatch) {
    auto config = createHybridLayerTestConfig(4, 40);
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();
    auto total_blocks = allocator_->freeBlocksNum();

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(2, 3);
    auto complete_token_ids = createCompleteTokenIds(2, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    auto       result = allocator_->incrMalloc(malloc_info);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(batch_resource->batch_resource[0].blocksNum(), 2);
    EXPECT_EQ(batch_resource->batch_resource[1].blocksNum(), 2);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - 2 * 3 * 2);

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);
}

TEST_F(HybridLayerKVCacheAllocatorTest, initMallocForCommonLenSingleBatch) {
    auto config = createHybridLayerTestConfig(4, 40, 4);
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();
    auto total_blocks = allocator_->freeBlocksNum();

    int  seq_length         = 17;
    auto batch_resource     = createBatchKVCacheResource(1, 3);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    malloc_info.batch_kv_cache_resource->enable_reuse_cache = false;
    malloc_info.common_seq_len                              = 16;
    auto malloc_result1                                     = allocator_->initMallocForCommonLen(malloc_info);
    EXPECT_TRUE(malloc_result1.success);
    EXPECT_EQ(batch_resource->batch_resource[0].blocksNum(), 4);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - 3 * 4);

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);

    auto                  block_pool  = allocator_->getBlockPool();
    auto                  block_cache = block_pool->blockCache();
    BlockCache::CacheItem item1       = {101, 0, 1, false};
    auto                  result1     = block_cache->put(item1);
    EXPECT_TRUE(result1);
    BlockCache::CacheItem item2   = {102, 0, 2, false};
    auto                  result2 = block_cache->put(item2);
    EXPECT_TRUE(result2);
    BlockCache::CacheItem item3   = {102, 1, 3, false};
    auto                  result3 = block_cache->put(item3);
    EXPECT_TRUE(result3);
    BlockCache::CacheItem item4   = {102, 2, 4, false};
    auto                  result4 = block_cache->put(item4);
    EXPECT_TRUE(result4);

    auto batch_resource2                = createBatchKVCacheResource(1, 3);
    malloc_info.batch_kv_cache_resource = batch_resource2;
    auto& cache_keys                    = malloc_info.batch_kv_cache_resource->cacheKeys();
    cache_keys                          = {101, 102, 103, 104};

    malloc_info.batch_kv_cache_resource->enable_reuse_cache = true;
    auto malloc_result2                                     = allocator_->initMallocForCommonLen(malloc_info);
    EXPECT_TRUE(malloc_result2.success);
    EXPECT_EQ(malloc_result2.reuse_len, 8);
    EXPECT_EQ(batch_resource2->batch_resource[0].blocksNum(), 4);
    EXPECT_EQ(batch_resource2->blocksNum(0, 0), 4);
    EXPECT_EQ(batch_resource2->blocksNum(0, 1), 4);
    EXPECT_EQ(batch_resource2->blocksNum(0, 2), 4);
    EXPECT_EQ(batch_resource2->batch_resource[0].group_block_ids[0]->block_indices[0], 1);
    EXPECT_EQ(batch_resource2->batch_resource[0].group_block_ids[0]->block_indices[1], 2);
    EXPECT_EQ(batch_resource2->batch_resource[0].group_block_ids[1]->block_indices[0], -1);
    EXPECT_EQ(batch_resource2->batch_resource[0].group_block_ids[1]->block_indices[1], 3);
    EXPECT_EQ(batch_resource2->batch_resource[0].group_block_ids[2]->block_indices[0], -1);
    EXPECT_EQ(batch_resource2->batch_resource[0].group_block_ids[2]->block_indices[1], 4);

    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - 3 * 2);

    FreeInfo free_info_2{batch_resource2, complete_token_ids};
    allocator_->free(free_info_2);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);
}

TEST_F(HybridLayerKVCacheAllocatorTest, initMallocForCommonLenMultiBatch) {
    auto config = createHybridLayerTestConfig(4, 40, 4);
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();
    auto total_blocks = allocator_->freeBlocksNum();

    int  seq_length         = 17;
    auto batch_resource     = createBatchKVCacheResource(2, 3);
    auto complete_token_ids = createCompleteTokenIds(2, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    malloc_info.common_seq_len                              = 16;
    malloc_info.batch_kv_cache_resource->enable_reuse_cache = false;
    auto malloc_result1                                     = allocator_->initMallocForCommonLen(malloc_info);
    EXPECT_TRUE(malloc_result1.success);
    EXPECT_EQ(batch_resource->batch_resource[0].blocksNum(), 4);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - (3 * 4));

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);

    auto                  block_pool  = allocator_->getBlockPool();
    auto                  block_cache = block_pool->blockCache();
    BlockCache::CacheItem item1       = {101, 0, 1, false};
    auto                  result1     = block_cache->put(item1);
    EXPECT_TRUE(result1);
    BlockCache::CacheItem item2   = {102, 0, 2, false};
    auto                  result2 = block_cache->put(item2);
    EXPECT_TRUE(result2);
    BlockCache::CacheItem item3   = {102, 1, 3, false};
    auto                  result3 = block_cache->put(item3);
    EXPECT_TRUE(result3);
    BlockCache::CacheItem item4   = {102, 2, 4, false};
    auto                  result4 = block_cache->put(item4);
    EXPECT_TRUE(result4);

    auto batch_resource2                = createBatchKVCacheResource(2, 3);
    malloc_info.batch_kv_cache_resource = batch_resource2;
    auto& cache_keys                    = malloc_info.batch_kv_cache_resource->cacheKeys();
    cache_keys                          = {101, 102, 103, 104};
    auto& cache_keys_2                  = malloc_info.batch_kv_cache_resource->cacheKeys();
    cache_keys_2                        = {101, 102, 103, 104};

    malloc_info.batch_kv_cache_resource->enable_reuse_cache = true;
    auto malloc_result2                                     = allocator_->initMallocForCommonLen(malloc_info);
    EXPECT_TRUE(malloc_result2.success);
    auto& batch_id_0 = batch_resource2->batch_resource[0];
    auto& batch_id_1 = batch_resource2->batch_resource[1];

    EXPECT_EQ(batch_id_0.blocksNum(), 4);
    EXPECT_EQ(batch_id_0.blocksNum(0), 4);
    EXPECT_EQ(batch_id_0.blocksNum(1), 4);
    EXPECT_EQ(batch_id_0.blocksNum(2), 4);
    EXPECT_EQ(batch_id_0.group_block_ids[0]->block_indices[0], 1);
    EXPECT_EQ(batch_id_0.group_block_ids[0]->block_indices[1], 2);
    EXPECT_EQ(batch_id_0.group_block_ids[1]->block_indices[0], -1);
    EXPECT_EQ(batch_id_0.group_block_ids[1]->block_indices[1], 3);
    EXPECT_EQ(batch_id_0.group_block_ids[2]->block_indices[0], -1);
    EXPECT_EQ(batch_id_0.group_block_ids[2]->block_indices[1], 4);

    EXPECT_EQ(batch_id_1.blocksNum(), 4);
    EXPECT_EQ(batch_id_1.blocksNum(0), 4);
    EXPECT_EQ(batch_id_1.blocksNum(1), 4);
    EXPECT_EQ(batch_id_1.blocksNum(2), 4);
    EXPECT_EQ(batch_id_1.group_block_ids[0]->block_indices[0], 1);
    EXPECT_EQ(batch_id_1.group_block_ids[0]->block_indices[1], 2);
    EXPECT_EQ(batch_id_1.group_block_ids[1]->block_indices[0], -1);
    EXPECT_EQ(batch_id_1.group_block_ids[1]->block_indices[1], 3);
    EXPECT_EQ(batch_id_1.group_block_ids[2]->block_indices[0], -1);
    EXPECT_EQ(batch_id_1.group_block_ids[2]->block_indices[1], 4);

    EXPECT_EQ(batch_id_0.group_block_ids[0]->block_indices[2], batch_id_1.group_block_ids[0]->block_indices[2]);
    EXPECT_EQ(batch_id_0.group_block_ids[0]->block_indices[3], batch_id_1.group_block_ids[0]->block_indices[3]);
    EXPECT_EQ(batch_id_0.group_block_ids[1]->block_indices[2], batch_id_1.group_block_ids[1]->block_indices[2]);
    EXPECT_EQ(batch_id_0.group_block_ids[1]->block_indices[3], batch_id_1.group_block_ids[1]->block_indices[3]);
    EXPECT_EQ(batch_id_0.group_block_ids[2]->block_indices[2], batch_id_1.group_block_ids[2]->block_indices[2]);
    EXPECT_EQ(batch_id_0.group_block_ids[2]->block_indices[3], batch_id_1.group_block_ids[2]->block_indices[3]);

    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - (3 * 2));

    FreeInfo free_info_2{batch_resource2, complete_token_ids};
    allocator_->free(free_info_2);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);
}

TEST_F(HybridLayerKVCacheAllocatorTest, initMallocSingleBatch) {
    auto config = createHybridLayerTestConfig(4, 40, 4);
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();
    auto total_blocks = allocator_->freeBlocksNum();

    int  seq_length         = 17;
    auto batch_resource     = createBatchKVCacheResource(1, 3);
    auto complete_token_ids = createCompleteTokenIds(1, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    malloc_info.batch_kv_cache_resource->enable_reuse_cache = false;
    malloc_info.common_seq_len                              = 16;
    auto malloc_result1                                     = allocator_->malloc(malloc_info);
    EXPECT_TRUE(malloc_result1.success);
    EXPECT_EQ(batch_resource->batch_resource[0].blocksNum(), 5);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - 3 * 5);

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);

    auto                  block_pool  = allocator_->getBlockPool();
    auto                  block_cache = block_pool->blockCache();
    BlockCache::CacheItem item1       = {101, 0, 1, false};
    auto                  result1     = block_cache->put(item1);
    EXPECT_TRUE(result1);
    BlockCache::CacheItem item2   = {102, 0, 2, false};
    auto                  result2 = block_cache->put(item2);
    EXPECT_TRUE(result2);
    BlockCache::CacheItem item3   = {102, 1, 3, false};
    auto                  result3 = block_cache->put(item3);
    EXPECT_TRUE(result3);
    BlockCache::CacheItem item4   = {102, 2, 4, false};
    auto                  result4 = block_cache->put(item4);
    EXPECT_TRUE(result4);

    auto batch_resource2                = createBatchKVCacheResource(1, 3);
    malloc_info.batch_kv_cache_resource = batch_resource2;
    auto& cache_keys                    = malloc_info.batch_kv_cache_resource->cacheKeys();
    cache_keys                          = {101, 102, 103, 104};

    malloc_info.batch_kv_cache_resource->enable_reuse_cache = true;
    auto malloc_result2                                     = allocator_->malloc(malloc_info);
    EXPECT_TRUE(malloc_result2.success);
    EXPECT_EQ(malloc_result2.reuse_len, 8);
    EXPECT_EQ(batch_resource2->batch_resource[0].blocksNum(), 5);
    EXPECT_EQ(batch_resource2->blocksNum(0, 0), 5);
    EXPECT_EQ(batch_resource2->blocksNum(0, 1), 5);
    EXPECT_EQ(batch_resource2->blocksNum(0, 2), 5);
    EXPECT_EQ(batch_resource2->batch_resource[0].group_block_ids[0]->block_indices[0], 1);
    EXPECT_EQ(batch_resource2->batch_resource[0].group_block_ids[0]->block_indices[1], 2);
    EXPECT_EQ(batch_resource2->batch_resource[0].group_block_ids[1]->block_indices[0], -1);
    EXPECT_EQ(batch_resource2->batch_resource[0].group_block_ids[1]->block_indices[1], 3);
    EXPECT_EQ(batch_resource2->batch_resource[0].group_block_ids[2]->block_indices[0], -1);
    EXPECT_EQ(batch_resource2->batch_resource[0].group_block_ids[2]->block_indices[1], 4);

    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - 3 * 3);

    FreeInfo free_info_2{batch_resource2, complete_token_ids};
    allocator_->free(free_info_2);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);
}

TEST_F(HybridLayerKVCacheAllocatorTest, initMallocMultiBatch) {
    auto config = createHybridLayerTestConfig(4, 40, 4);
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();
    auto total_blocks = allocator_->freeBlocksNum();

    int  seq_length         = 17;
    auto batch_resource     = createBatchKVCacheResource(2, 3);
    auto complete_token_ids = createCompleteTokenIds(2, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    malloc_info.common_seq_len                              = 16;
    malloc_info.batch_kv_cache_resource->enable_reuse_cache = false;
    auto malloc_result1                                     = allocator_->malloc(malloc_info);
    EXPECT_TRUE(malloc_result1.success);
    EXPECT_EQ(batch_resource->batch_resource[0].blocksNum(), 5);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - (3 * 5 + 3 * 1));

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);

    auto                  block_pool  = allocator_->getBlockPool();
    auto                  block_cache = block_pool->blockCache();
    BlockCache::CacheItem item1       = {101, 0, 1, false};
    auto                  result1     = block_cache->put(item1);
    EXPECT_TRUE(result1);
    BlockCache::CacheItem item2   = {102, 0, 2, false};
    auto                  result2 = block_cache->put(item2);
    EXPECT_TRUE(result2);
    BlockCache::CacheItem item3   = {102, 1, 3, false};
    auto                  result3 = block_cache->put(item3);
    EXPECT_TRUE(result3);
    BlockCache::CacheItem item4   = {102, 2, 4, false};
    auto                  result4 = block_cache->put(item4);
    EXPECT_TRUE(result4);

    auto batch_resource2                = createBatchKVCacheResource(2, 3);
    malloc_info.batch_kv_cache_resource = batch_resource2;
    auto& cache_keys                    = malloc_info.batch_kv_cache_resource->cacheKeys();
    cache_keys                          = {101, 102, 103, 104};
    auto& cache_keys_2                  = malloc_info.batch_kv_cache_resource->cacheKeys();
    cache_keys_2                        = {101, 102, 103, 104};

    malloc_info.batch_kv_cache_resource->enable_reuse_cache = true;
    auto malloc_result2                                     = allocator_->malloc(malloc_info);
    EXPECT_TRUE(malloc_result2.success);
    auto& batch_id_0 = batch_resource2->batch_resource[0];
    auto& batch_id_1 = batch_resource2->batch_resource[1];

    EXPECT_EQ(batch_id_0.blocksNum(), 5);
    EXPECT_EQ(batch_id_0.blocksNum(0), 5);
    EXPECT_EQ(batch_id_0.blocksNum(1), 5);
    EXPECT_EQ(batch_id_0.blocksNum(2), 5);
    EXPECT_EQ(batch_id_0.group_block_ids[0]->block_indices[0], 1);
    EXPECT_EQ(batch_id_0.group_block_ids[0]->block_indices[1], 2);
    EXPECT_EQ(batch_id_0.group_block_ids[1]->block_indices[0], -1);
    EXPECT_EQ(batch_id_0.group_block_ids[1]->block_indices[1], 3);
    EXPECT_EQ(batch_id_0.group_block_ids[2]->block_indices[0], -1);
    EXPECT_EQ(batch_id_0.group_block_ids[2]->block_indices[1], 4);

    EXPECT_EQ(batch_id_1.blocksNum(), 5);
    EXPECT_EQ(batch_id_1.blocksNum(0), 5);
    EXPECT_EQ(batch_id_1.blocksNum(1), 5);
    EXPECT_EQ(batch_id_1.blocksNum(2), 5);
    EXPECT_EQ(batch_id_1.group_block_ids[0]->block_indices[0], 1);
    EXPECT_EQ(batch_id_1.group_block_ids[0]->block_indices[1], 2);
    EXPECT_EQ(batch_id_1.group_block_ids[1]->block_indices[0], -1);
    EXPECT_EQ(batch_id_1.group_block_ids[1]->block_indices[1], 3);
    EXPECT_EQ(batch_id_1.group_block_ids[2]->block_indices[0], -1);
    EXPECT_EQ(batch_id_1.group_block_ids[2]->block_indices[1], 4);

    EXPECT_EQ(batch_id_0.group_block_ids[0]->block_indices[2], batch_id_1.group_block_ids[0]->block_indices[2]);
    EXPECT_EQ(batch_id_0.group_block_ids[0]->block_indices[3], batch_id_1.group_block_ids[0]->block_indices[3]);
    EXPECT_NE(batch_id_0.group_block_ids[0]->block_indices[4], batch_id_1.group_block_ids[0]->block_indices[4]);
    EXPECT_EQ(batch_id_0.group_block_ids[1]->block_indices[2], batch_id_1.group_block_ids[1]->block_indices[2]);
    EXPECT_EQ(batch_id_0.group_block_ids[1]->block_indices[3], batch_id_1.group_block_ids[1]->block_indices[3]);
    EXPECT_NE(batch_id_0.group_block_ids[1]->block_indices[4], batch_id_1.group_block_ids[1]->block_indices[4]);
    EXPECT_EQ(batch_id_0.group_block_ids[2]->block_indices[2], batch_id_1.group_block_ids[2]->block_indices[2]);
    EXPECT_EQ(batch_id_0.group_block_ids[2]->block_indices[3], batch_id_1.group_block_ids[2]->block_indices[3]);
    EXPECT_NE(batch_id_0.group_block_ids[2]->block_indices[4], batch_id_1.group_block_ids[2]->block_indices[4]);

    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - (3 * 3 + 3 * 1));

    FreeInfo free_info_2{batch_resource2, complete_token_ids};
    allocator_->free(free_info_2);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);
}

TEST_F(HybridLayerKVCacheAllocatorTest, MallocWithInsufficientBlocks) {
    auto config = createHybridLayerTestConfig(4, 5, 4);
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();
    auto total_blocks = allocator_->freeBlocksNum();

    int  seq_length         = 17;
    auto batch_resource     = createBatchKVCacheResource(2, 3);
    auto complete_token_ids = createCompleteTokenIds(2, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    malloc_info.common_seq_len                              = 16;
    malloc_info.batch_kv_cache_resource->enable_reuse_cache = false;
    auto malloc_result1                                     = allocator_->malloc(malloc_info);
    EXPECT_FALSE(malloc_result1.success);
    EXPECT_EQ(malloc_result1.reuse_len, 0);

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);
}

// Test malloc free cycle
TEST_F(HybridLayerKVCacheAllocatorTest, MallocFreeCycle) {
    auto config = createHybridLayerTestConfig(4, 50, 4);
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();

    for (int i = 0; i < 5; ++i) {
        auto total_blocks       = allocator_->freeBlocksNum();
        int  seq_length         = 17;
        auto batch_resource     = createBatchKVCacheResource(2, 3);
        auto complete_token_ids = createCompleteTokenIds(2, seq_length);

        MallocInfo malloc_info{batch_resource, complete_token_ids};

        auto malloc_result = allocator_->malloc(malloc_info);
        EXPECT_TRUE(malloc_result.success);

        FreeInfo free_info{batch_resource, complete_token_ids};
        allocator_->free(free_info);

        EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);
    }
}

// Test insert into cache
TEST_F(HybridLayerKVCacheAllocatorTest, InsertIntoCache) {
    auto config = createHybridLayerTestConfig(4, 50, 4);
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();
    auto block_pool   = allocator_->getBlockPool();
    auto block_cache  = block_pool->blockCache();
    auto total_blocks = allocator_->freeBlocksNum();

    int  seq_length         = 16;
    auto batch_resource     = createBatchKVCacheResource(2, 3, 1, 4);
    auto complete_token_ids = createCompleteTokenIds(2, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    malloc_info.common_seq_len                              = 16;
    malloc_info.batch_kv_cache_resource->enable_reuse_cache = false;
    auto malloc_result1                                     = allocator_->malloc(malloc_info);
    EXPECT_TRUE(malloc_result1.success);

    // batch 0: 12个block，9个block可以insert to cache。
    // batch 1: 12个block，在前面都insert to cache过了。
    InsertInfo insert_info1{batch_resource, complete_token_ids, false};
    allocator_->insertIntoCache(insert_info1);
    ASSERT_EQ(block_cache->size(), 9);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - (3 * 4));

    // batch 0: 12个block，12个block可以insert to cache。
    // batch 1: 12个block，在前面都insert to cache过了。
    seq_length                      = 17;
    auto       complete_token_ids_2 = createCompleteTokenIds(2, seq_length);
    InsertInfo insert_info2{batch_resource, complete_token_ids_2, false};
    allocator_->insertIntoCache(insert_info2);
    ASSERT_EQ(block_cache->size(), 12);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - (3 * 4));
}

TEST_F(HybridLayerKVCacheAllocatorTest, InsertIntoCacheAsResident) {
    auto config = createHybridLayerTestConfig(4, 50, 4);
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();
    auto total_blocks = allocator_->freeBlocksNum();

    int  seq_length         = 17;
    auto batch_resource     = createBatchKVCacheResource(2, 3, 1, 4);
    auto complete_token_ids = createCompleteTokenIds(2, seq_length);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    malloc_info.common_seq_len = 16;
    allocator_->malloc(malloc_info);

    InsertInfo insert_info{batch_resource, complete_token_ids, true};
    allocator_->insertIntoCache(insert_info);

    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks - (3 * 5 + 3 * 1));
}

TEST_F(HybridLayerKVCacheAllocatorTest, MaxSeqLen) {
    auto config = createHybridLayerTestConfig(4, 10, 8);
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();

    EXPECT_EQ(allocator_->maxAvailableTokensNum(), (10 - 1) * 8);  // block_num * seq_size_per_block
}

TEST_F(HybridLayerKVCacheAllocatorTest, MallocWithZeroSeqLength) {
    auto config = createHybridLayerTestConfig();
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();
    auto total_blocks = allocator_->freeBlocksNum();

    auto batch_resource     = createBatchKVCacheResource(1, 3);
    auto complete_token_ids = createCompleteTokenIds(1, 0);

    MallocInfo malloc_info{batch_resource, complete_token_ids};
    auto       result = allocator_->malloc(malloc_info);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);
}

TEST_F(HybridLayerKVCacheAllocatorTest, FreeEmptyBatchResource) {
    auto config = createHybridLayerTestConfig();
    allocator_  = std::make_shared<HybridLayerKVCacheAllocator>(config, device_);
    allocator_->init();
    auto total_blocks = allocator_->freeBlocksNum();

    auto batch_resource     = createBatchKVCacheResource(0);
    auto complete_token_ids = createCompleteTokenIds(0, 0);

    FreeInfo free_info{batch_resource, complete_token_ids};
    allocator_->free(free_info);

    EXPECT_EQ(allocator_->freeBlocksNum(), total_blocks);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
