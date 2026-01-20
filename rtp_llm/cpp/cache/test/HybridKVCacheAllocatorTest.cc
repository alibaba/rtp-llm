#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

static CacheConfig makeTinyHybridConfig() {
    // 4 layers: [0,1] linear, [2,3] full. gcd(2,2)=2 => group_size=2.
    CacheConfig config;
    config.dtype              = rtp_llm::DataType::TYPE_FP16;
    config.layer_num          = 4;
    config.layer_all_num      = 4;
    config.block_num          = 10;
    config.seq_size_per_block = 4;
    config.linear_step        = 2;
    config.group_size         = 2;

    // Linear spec (small but valid).
    auto linear_spec   = std::make_shared<LinearKVCacheSpec>();
    linear_spec->type  = KVCacheType::LinearAttention;
    linear_spec->dtype = config.dtype;
    linear_spec->layer_num          = 2;
    linear_spec->local_num_k_heads  = 1;
    linear_spec->local_num_v_heads  = 1;
    linear_spec->head_k_dim         = 1;
    linear_spec->head_v_dim         = 1;
    linear_spec->conv_kernel_dim    = 2;
    linear_spec->local_head_num_kv  = 1;
    linear_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    // Full spec.
    auto full_spec                = std::make_shared<MHAKVCacheSpec>();
    full_spec->type               = KVCacheType::MultiHeadAttention;
    full_spec->dtype              = config.dtype;
    full_spec->layer_num          = 2;
    full_spec->local_head_num_kv  = 1;
    full_spec->size_per_head      = 1;
    full_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    // Order matters: linear groups first, then full groups (as in CacheConfigCreator).
    config.layer_ids        = {{0, 1}, {2, 3}};
    config.global_layer_ids = config.layer_ids;
    config.cache_specs      = {linear_spec, full_spec};
    config.linear_group_num = 1;
    config.full_group_num   = 1;

    // Physical block strides: take max between full and linear.
    config.kv_block_stride       = std::max(full_spec->block_size(), linear_spec->block_size());
    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());

    config.kv_block_size       = static_cast<size_t>(config.group_size) * config.kv_block_stride;
    config.kv_block_size_bytes = static_cast<size_t>(config.group_size) * config.kv_block_stride_bytes;

    // No kv scale for fp16.
    config.kv_scale_stride       = 0;
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size         = 0;
    config.kv_scale_size_bytes   = 0;

    config.block_stride       = config.kv_block_stride + config.kv_scale_stride;
    config.block_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.block_size         = config.kv_block_size + config.kv_scale_size;
    config.block_size_bytes   = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    config.layer_to_group_id.assign(static_cast<size_t>(config.layer_num), 0);
    for (size_t gid = 0; gid < config.layer_ids.size(); ++gid) {
        for (int layer_id : config.layer_ids[gid]) {
            config.layer_to_group_id[static_cast<size_t>(layer_id)] = static_cast<int>(gid);
        }
    }
    return config;
}

static CompleteTokenIdsPtr makeCompleteTokenIds(rtp_llm::DeviceBase* device, int batch_size, int seq_length, int seq_size_per_block) {
    auto complete_token_ids =
        std::make_shared<CompleteTokenIds>(device, batch_size, batch_size, seq_length + 64, seq_size_per_block);
    auto input_ids = device->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {(size_t)seq_length}, rtp_llm::AllocationType::HOST}, {});
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

static BatchKVCacheResourcePtr makeBatchResource(int                batch_size,
                                                 int                group_nums,
                                                 int                layer_num,
                                                 const std::vector<int>& layer_to_group_id,
                                                 CacheKeysType       keys) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    res->initGroups(group_nums, layer_num, layer_to_group_id);
    for (int b = 0; b < batch_size; ++b) {
        res->setBatchCacheKeys(b, keys);
    }
    return res;
}

static std::vector<BlockIdxType> allocateAndCache(BlockPoolPtr block_pool,
                                                  BlockCachePtr block_cache,
                                                  int group_id,
                                                  const CacheKeysType& keys,
                                                  bool is_resident = true) {
    auto blocks = block_pool->malloc(static_cast<int>(keys.size()));
    EXPECT_EQ(blocks.size(), keys.size());

    for (size_t i = 0; i < keys.size(); ++i) {
        BlockCache::CacheItem item;
        item.cache_key   = keys[i];
        item.group_id    = group_id;
        item.block_index = blocks[i];
        item.is_resident = is_resident;
        EXPECT_TRUE(block_cache->put(item));
        block_pool->blockCacheReference(blocks[i]);
    }

    // Drop request references so these blocks behave like "cached but available" blocks.
    block_pool->requestFree(blocks);
    return blocks;
}

class HybridKVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        device_ = createDevice();
        ASSERT_NE(device_, nullptr);
    }

    rtp_llm::DeviceBase* device_ = nullptr;
};

TEST_F(HybridKVCacheAllocatorTest, InitAndAddressLookupSmoke) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridLayerKVCacheAllocator>(config, device_, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    EXPECT_EQ(allocator->seqSizePerBlock(), 4);
    EXPECT_EQ(allocator->totalBlocksNum(), config.block_num - 1);
    EXPECT_EQ(allocator->freeBlocksNum(), config.block_num - 1);

    // Should be able to fetch address for any global layer and non-zero block id.
    auto addr0 = allocator->convertIndexToAddr(/*layer_id=*/0, /*block_id=*/1);
    auto addr3 = allocator->convertIndexToAddr(/*layer_id=*/3, /*block_id=*/1);
    EXPECT_NE(addr0.kv_addr, nullptr);
    EXPECT_NE(addr3.kv_addr, nullptr);
}

TEST_F(HybridKVCacheAllocatorTest, JointReuseUsesFullPrefixAndLinearTailOnly) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridLayerKVCacheAllocator>(config, device_, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool  = allocator->getBlockPool();
    auto block_cache = block_pool->blockCache();
    ASSERT_NE(block_pool, nullptr);
    ASSERT_NE(block_cache, nullptr);

    // Config order: gid=0 linear, gid=1 full.
    const int gid_linear = 0;
    const int gid_full   = 1;

    // Full group has prefix matches for {100,101,102}.
    CacheKeysType full_keys = {100, 101, 102};
    auto          full_blocks = allocateAndCache(block_pool, block_cache, gid_full, full_keys);

    // Linear group only matches key 101 (so joint match should backoff to pos=1 => reuse_blocks_len=2).
    CacheKeysType linear_keys = {101};
    auto          linear_blocks = allocateAndCache(block_pool, block_cache, gid_linear, linear_keys);
    ASSERT_EQ(linear_blocks.size(), 1u);

    // Request has 4 keys, but allocator drops the last for matching.
    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_to_group_id=*/config.layer_to_group_id,
                                       CacheKeysType{100, 101, 102, 103});
    batch_res->enable_reuse_cache = true;

    // seq_len=12 => 3 slots (4 tokens per block).
    auto token_ids = makeCompleteTokenIds(device_, /*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    auto       result = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Full group: should reuse the first 2 blocks and allocate the third.
    const auto& full_out = batch_res->blocks(0, gid_full);
    ASSERT_EQ(full_out.size(), 3u);
    EXPECT_EQ(full_out[0], full_blocks[0]);
    EXPECT_EQ(full_out[1], full_blocks[1]);
    EXPECT_FALSE(isNullBlockIdx(full_out[2]));

    // Linear group: only the tail slot of the reused prefix is filled; earlier slots stay NULL.
    const auto& linear_out = batch_res->blocks(0, gid_linear);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_EQ(linear_out[1], linear_blocks[0]);  // reused tail at pos=1
    EXPECT_FALSE(isNullBlockIdx(linear_out[2])); // allocated tail for common length
}

TEST_F(HybridKVCacheAllocatorTest, DisableReuseKeepsOnlyLinearTailOnInitMalloc) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridLayerKVCacheAllocator>(config, device_, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_to_group_id=*/config.layer_to_group_id,
                                       CacheKeysType{100, 101, 102, 103});
    batch_res->enable_reuse_cache = false;

    auto token_ids = makeCompleteTokenIds(device_, /*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    auto       result = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Linear group should keep only the tail block across common length slots.
    const auto& linear_out = batch_res->blocks(0, /*group_id=*/0);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_TRUE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


