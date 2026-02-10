#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/BlockCache.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
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
    config.group_layer_num    = 2;

    // Linear spec (small but valid).
    auto linear_spec                = std::make_shared<LinearKVCacheSpec>();
    linear_spec->type               = KVCacheSpecType::LinearAttention;
    linear_spec->dtype              = config.dtype;
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
    full_spec->type               = KVCacheSpecType::MultiHeadAttention;
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
    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;

    // No kv scale for fp16.
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;

    config.block_size_bytes = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    config.layer_to_group_id.assign(static_cast<size_t>(config.layer_num), 0);
    for (size_t gid = 0; gid < config.layer_ids.size(); ++gid) {
        for (int layer_id : config.layer_ids[gid]) {
            config.layer_to_group_id[static_cast<size_t>(layer_id)] = static_cast<int>(gid);
        }
    }
    return config;
}

static CompleteTokenIdsPtr
makeCompleteTokenIds(rtp_llm::DeviceBase* device, int batch_size, int seq_length, int seq_size_per_block) {
    auto complete_token_ids =
        std::make_shared<CompleteTokenIds>(device, batch_size, batch_size, seq_length + 64, seq_size_per_block);
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

static BatchKVCacheResourcePtr makeBatchResource(
    int batch_size, int group_nums, int layer_num, const std::vector<int>& layer_to_group_id, CacheKeysType keys) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    res->initGroups(group_nums, layer_num, layer_to_group_id);
    for (int b = 0; b < batch_size; ++b) {
        res->setBatchCacheKeys(b, keys);
    }
    return res;
}

static std::vector<BlockIdxType> allocateAndCache(BlockPoolPtr         block_pool,
                                                  BlockCachePtr        block_cache,
                                                  int                  group_id,
                                                  const CacheKeysType& keys,
                                                  bool                 is_resident = true) {
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

static std::vector<BlockIdxType> allocateAndCacheKeepAllocated(BlockPoolPtr         block_pool,
                                                               BlockCachePtr        block_cache,
                                                               int                  group_id,
                                                               const CacheKeysType& keys,
                                                               bool                 is_resident = true) {
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

    // NOTE: intentionally keep these blocks allocated/unavailable to avoid accidental reuse via malloc().
    return blocks;
}

static size_t countValidBlocks(const BlockIndicesType& blocks) {
    size_t n = 0;
    for (auto b : blocks) {
        if (!isNullBlockIdx(b)) {
            ++n;
        }
    }
    return n;
}

class HybridTypeKVCacheAllocatorTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        device_ = createDevice();
        ASSERT_NE(device_, nullptr);
    }

    rtp_llm::DeviceBase* device_ = nullptr;
};

TEST_F(HybridTypeKVCacheAllocatorTest, InitAndAddressLookupSmoke) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, device_, AllocationType::DEVICE);
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

TEST_F(HybridTypeKVCacheAllocatorTest, GetNeedBlocksUsesGroupGetNeedBlocksAndReuseFlag) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, device_, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    // batch=2, seq_len=12 (3 slots), reserve_step=2
    auto token_ids = makeCompleteTokenIds(device_, /*batch_size=*/2, /*seq_length=*/12, /*seq_size_per_block=*/4);
    token_ids->setReserveStep(2);

    // Reuse disabled: linear group keeps only tail for common blocks; reserve_step contributes extra blocks.
    // full group contributes common=3, extra=1.
    {
        auto       batch_res = makeBatchResource(/*batch_size=*/2,
                                           /*group_nums=*/2,
                                           /*layer_num=*/static_cast<int>(config.layer_all_num),
                                           /*layer_to_group_id=*/config.layer_to_group_id,
                                           CacheKeysType{100, 101, 102, 103});
        MallocInfo info{batch_res, token_ids};
        info.enable_device_cache = false;
        // common_total = full(3) + linear(1) = 4
        // extra_total  = full(1) + linear(reserve_step=2) = 3
        // total = 4 + 2*3 = 10
        EXPECT_EQ(allocator->getNeedBlocks(info), 10);
    }

    // Reuse enabled but no existing blocks: linear group uses sparse counting from begin=0.
    {
        auto       batch_res = makeBatchResource(/*batch_size=*/2,
                                           /*group_nums=*/2,
                                           /*layer_num=*/static_cast<int>(config.layer_all_num),
                                           /*layer_to_group_id=*/config.layer_to_group_id,
                                           CacheKeysType{100, 101, 102, 103});
        MallocInfo info{batch_res, token_ids};
        info.enable_device_cache = true;
        // full: common=3 extra=1
        // linear: common=count(0,3]=2, extra=reserve_step(=2)
        // common_total = 3 + 2 = 5
        // extra_total  = 1 + 2 = 3
        // total = 5 + 2*3 = 11
        EXPECT_EQ(allocator->getNeedBlocks(info), 11);
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, JointReuseUsesFullPrefixAndLinearTailOnly) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, device_, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool  = allocator->getBlockPool();
    auto block_cache = block_pool->blockCache();
    ASSERT_NE(block_pool, nullptr);
    ASSERT_NE(block_cache, nullptr);

    // Config order: gid=0 linear, gid=1 full.
    const int gid_linear = 0;
    const int gid_full   = 1;

    // Full group has prefix matches for {100,101,102}.
    CacheKeysType full_keys   = {100, 101, 102};
    auto          full_blocks = allocateAndCache(block_pool, block_cache, gid_full, full_keys);

    // Linear group only matches key 101 (so joint match should backoff to pos=1 => reuse_blocks_len=2).
    CacheKeysType linear_keys   = {101};
    auto          linear_blocks = allocateAndCache(block_pool, block_cache, gid_linear, linear_keys);
    ASSERT_EQ(linear_blocks.size(), 1u);

    // Request has 4 keys, but allocator drops the last for matching.
    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_to_group_id=*/config.layer_to_group_id,
                                       CacheKeysType{100, 101, 102, 103});
    // Enable device cache reuse for joint match.

    // seq_len=12 => 3 slots (4 tokens per block).
    auto token_ids = makeCompleteTokenIds(device_, /*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache = true;
    auto result              = allocator->malloc(info);
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
    EXPECT_EQ(linear_out[1], linear_blocks[0]);   // reused tail at pos=1
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));  // allocated tail for common length
}

TEST_F(HybridTypeKVCacheAllocatorTest, DisableReuseKeepsOnlyLinearTailOnInitMalloc) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, device_, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_to_group_id=*/config.layer_to_group_id,
                                       CacheKeysType{100, 101, 102, 103});
    // Disable device cache reuse.

    auto token_ids = makeCompleteTokenIds(device_, /*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache = false;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Linear group should keep only the tail block across common length slots.
    const auto& linear_out = batch_res->blocks(0, /*group_id=*/0);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_TRUE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
}

TEST_F(HybridTypeKVCacheAllocatorTest, DisableDeviceCacheSkipsReuseMatchAndAllocatesOnlyLinearTail) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, device_, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool  = allocator->getBlockPool();
    auto block_cache = block_pool->blockCache();
    ASSERT_NE(block_pool, nullptr);
    ASSERT_NE(block_cache, nullptr);

    // Config order: gid=0 linear, gid=1 full.
    const int gid_linear = 0;
    const int gid_full   = 1;

    // Prepare cached blocks for full group; keep them allocated so allocator's malloc() cannot accidentally return same
    // ids.
    CacheKeysType full_keys   = {100, 101, 102};
    auto          full_blocks = allocateAndCacheKeepAllocated(block_pool, block_cache, gid_full, full_keys);
    ASSERT_EQ(full_blocks.size(), 3u);

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_to_group_id=*/config.layer_to_group_id,
                                       CacheKeysType{100, 101, 102, 103});
    // Disable device cache reuse: allocator should skip reuse match even if cache exists.

    auto token_ids =
        makeCompleteTokenIds(device_, /*batch_size=*/1, /*seq_length=*/12, /*seq_size_per_block=*/4);  // 3 slots

    MallocInfo info{batch_res, token_ids};
    info.enable_device_cache = false;
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Device cache disabled => must not reuse match.
    EXPECT_EQ(result.reuse_len, 0);

    // Full group should allocate fresh blocks (not reuse cached ones).
    const auto& full_out = batch_res->blocks(0, gid_full);
    ASSERT_EQ(full_out.size(), 3u);
    EXPECT_FALSE(isNullBlockIdx(full_out[0]));
    EXPECT_FALSE(isNullBlockIdx(full_out[1]));
    EXPECT_FALSE(isNullBlockIdx(full_out[2]));
    EXPECT_NE(full_out[0], full_blocks[0]);
    EXPECT_NE(full_out[1], full_blocks[1]);
    EXPECT_NE(full_out[2], full_blocks[2]);

    // Linear group keeps only tail block (others NULL) when reuse is disabled.
    const auto& linear_out = batch_res->blocks(0, gid_linear);
    ASSERT_EQ(linear_out.size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(linear_out[0]));
    EXPECT_TRUE(isNullBlockIdx(linear_out[1]));
    EXPECT_FALSE(isNullBlockIdx(linear_out[2]));
    EXPECT_EQ(countValidBlocks(linear_out), 1u);
}

TEST_F(HybridTypeKVCacheAllocatorTest, IncrDecrKVCacheRefReferencesOnlyMatchedValidBlocksAcrossGroups) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, device_, AllocationType::HOST);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator->freeBlocksNum();
    auto         blocks      = block_pool->malloc(4);
    ASSERT_EQ(blocks.size(), 4u);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 4);

    KVCacheResource resource;
    resource.initGroups(/*group_nums=*/2,
                        /*layer_num=*/static_cast<int>(config.layer_all_num),
                        /*layer_to_group_id=*/config.layer_to_group_id);
    resource.cacheKeys()       = CacheKeysType{100, 101, 102};
    resource.blocks(/*gid=*/0) = BlockIndicesType{blocks[0], 0, blocks[1]};  // linear group (contains a 0)
    resource.blocks(/*gid=*/1) = BlockIndicesType{blocks[2], blocks[3], 0};  // full group (contains a 0)

    // keys: 101(pos1)->gid0:0(ignore), gid1:blocks[3](ref); 102(pos2)->gid0:blocks[1](ref), gid1:0(ignore)
    auto ref = allocator->incrKVCacheRef(resource, CacheKeysType{101, 999, 102});
    ASSERT_NE(ref, nullptr);
    ASSERT_EQ(ref->groupNums(), 2);
    ASSERT_EQ(ref->cacheKeys().size(), 3u);
    ASSERT_EQ(ref->blocks(0).size(), 2u);
    ASSERT_EQ(ref->blocks(1).size(), 2u);

    block_pool->requestFree(blocks);
    EXPECT_EQ(allocator->freeBlocksNum(), free_before - 2) << "Only blocks[1] and blocks[3] should remain referenced";

    ref.reset();
    EXPECT_EQ(allocator->freeBlocksNum(), free_before);
}

TEST_F(HybridTypeKVCacheAllocatorTest, InsertIntoCacheInsertsOnlyFullBlocks) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, device_, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool  = allocator->getBlockPool();
    auto block_cache = block_pool->blockCache();
    ASSERT_NE(block_pool, nullptr);
    ASSERT_NE(block_cache, nullptr);

    // gid=0 linear, gid=1 full.
    const int gid_linear = 0;
    const int gid_full   = 1;

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_to_group_id=*/config.layer_to_group_id,
                                       CacheKeysType{100, 101, 102});
    // Disable device cache reuse.

    // seq_len=10 => 3 slots, full_blocks_num = floor(10/4)=2 -> only first 2 keys inserted.
    auto token_ids = makeCompleteTokenIds(device_, /*batch_size=*/1, /*seq_length=*/10, /*seq_size_per_block=*/4);

    MallocInfo malloc_info{batch_res, token_ids};
    malloc_info.enable_device_cache = false;
    auto malloc_result              = allocator->malloc(malloc_info);
    ASSERT_TRUE(malloc_result.success);
    ASSERT_EQ(batch_res->blocksNum(0, gid_full), 3);
    ASSERT_EQ(batch_res->blocksNum(0, gid_linear), 3);

    InsertInfo insert_info{batch_res, token_ids, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    // Full group should have cached first two keys.
    EXPECT_TRUE(block_cache->contains(100, gid_full));
    EXPECT_TRUE(block_cache->contains(101, gid_full));
    EXPECT_FALSE(block_cache->contains(102, gid_full));

    // Linear group has NULL in early slots when reuse disabled, thus should not insert these full blocks.
    EXPECT_FALSE(block_cache->contains(100, gid_linear));
    EXPECT_FALSE(block_cache->contains(101, gid_linear));
}

TEST_F(HybridTypeKVCacheAllocatorTest, ConvertIndexToBufferAndAllLayerCacheBaseSmoke) {
    auto config    = makeTinyHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, device_, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    KVCacheAllocator* base = allocator.get();
    auto              buf0 = base->convertIndexToBuffer(/*layer_id=*/0, /*block_id=*/1);
    ASSERT_FALSE(buf0.empty());
    EXPECT_NE(buf0[0].addr, nullptr);

    auto layout = allocator->allLayerCacheBase();
    EXPECT_EQ(layout.layers_to_kv_buffer_ptrs.size(), static_cast<size_t>(config.layer_num));
    for (size_t i = 0; i < layout.layers_to_kv_buffer_ptrs.size(); ++i) {
        EXPECT_NE(layout.layers_to_kv_buffer_ptrs[i], nullptr);
    }
}

TEST_F(HybridTypeKVCacheAllocatorTest, IncrMallocRollbackFreesPartiallyAllocatedBlocks) {
    auto config      = makeTinyHybridConfig();
    config.block_num = 6;  // free=5
    auto allocator   = std::make_shared<HybridTypeKVCacheAllocator>(config, device_, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto block_pool = allocator->getBlockPool();
    ASSERT_NE(block_pool, nullptr);

    auto batch_res = makeBatchResource(/*batch_size=*/1,
                                       /*group_nums=*/2,
                                       /*layer_num=*/static_cast<int>(config.layer_all_num),
                                       /*layer_to_group_id=*/config.layer_to_group_id,
                                       CacheKeysType{100, 101, 102});
    // Disable device cache reuse (makes linear group allocate only tail for new slots).

    // Initial small allocation: seq_len=4 => 1 slot per group.
    auto       token_ids = makeCompleteTokenIds(device_, /*batch_size=*/1, /*seq_length=*/4, /*seq_size_per_block=*/4);
    MallocInfo init_info{batch_res, token_ids};
    init_info.enable_device_cache = false;
    auto init_result              = allocator->malloc(init_info);
    ASSERT_TRUE(init_result.success);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 1);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 1);

    const auto linear_block_before = batch_res->blocks(0, /*gid=*/0)[0];
    const auto full_block_before   = batch_res->blocks(0, /*gid=*/1)[0];

    // Leave exactly 1 free block in pool, so linear allocates 1 and full fails on the next allocation.
    const size_t free_before_incr = block_pool->freeBlocksNum();
    ASSERT_GE(free_before_incr, 1u);
    auto keep = block_pool->malloc(static_cast<int>(free_before_incr - 1));
    ASSERT_EQ(block_pool->freeBlocksNum(), 1u);

    // Incr to seq_len=9 => 3 slots per group. Linear adds 2 slots but allocates only 1 real block; full needs 2.
    token_ids->setSeqLength(9);
    MallocInfo incr_info{batch_res, token_ids};
    incr_info.enable_device_cache = false;
    auto incr_result              = allocator->malloc(incr_info);
    EXPECT_FALSE(incr_result.success);

    // Rollback should restore original sizes and keep original blocks.
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/0), 1);
    ASSERT_EQ(batch_res->blocksNum(0, /*gid=*/1), 1);
    EXPECT_EQ(batch_res->blocks(0, /*gid=*/0)[0], linear_block_before);
    EXPECT_EQ(batch_res->blocks(0, /*gid=*/1)[0], full_block_before);

    // Free blocks count should return to 1 (no leaks).
    EXPECT_EQ(block_pool->freeBlocksNum(), 1u);

    // Cleanup.
    block_pool->requestFree(keep);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
