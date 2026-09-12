// CP-shard (Stage 5, Plan A) UTs for HybridKVCacheAllocator.
//
// These exercise the cp_slot_mapper plumbing in initMallocForCommonLen,
// incrMalloc, insertIntoCache, and getNeedBlocks. The shape of the tests
// piggybacks on the helpers in HybridTypeKVCacheAllocatorTest.cc but
// keeps the configuration self-contained so the two files build cleanly
// alongside each other.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

namespace {

// Two-group hybrid: gid=0 linear (won't be exercised here), gid=1 full (the CP-shard target).
CacheConfig makeCPHybridConfig() {
    CacheConfig config;
    config.dtype                     = rtp_llm::DataType::TYPE_FP16;
    config.layer_num                 = 4;
    config.layer_all_num             = 4;
    config.block_num                 = 32;  // headroom for cp_size=2 expansion
    config.seq_size_per_block        = 4;
    config.kernel_seq_size_per_block = 2;
    config.linear_step               = 2;
    config.group_layer_num           = 2;

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

    auto full_spec                = std::make_shared<MHAKVCacheSpec>();
    full_spec->type               = KVCacheSpecType::MultiHeadAttention;
    full_spec->dtype              = config.dtype;
    full_spec->layer_num          = 2;
    full_spec->local_head_num_kv  = 1;
    full_spec->size_per_head      = 1;
    full_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    config.layer_ids        = {{0, 1}, {2, 3}};
    config.global_layer_ids = config.layer_ids;
    config.cache_specs      = {linear_spec, full_spec};
    config.linear_group_num = 1;
    config.full_group_num   = 1;

    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    config.layer_to_group_id.assign(static_cast<size_t>(config.layer_num), 0);
    for (size_t gid = 0; gid < config.layer_ids.size(); ++gid) {
        for (int layer_id : config.layer_ids[gid]) {
            config.layer_to_group_id[static_cast<size_t>(layer_id)] = static_cast<int>(gid);
        }
    }
    return config;
}

CacheConfig makeCompactLinearCPHybridConfig() {
    auto config                               = makeCPHybridConfig();
    config.group_types                        = {CacheGroupType::LINEAR, CacheGroupType::FULL};
    config.group_seq_size_per_block           = {8, 4};
    config.cache_specs[0]->seq_size_per_block = 8;
    config.linear_step                        = 1;
    return config;
}

CompleteTokenIdsPtr makeTokens(int batch_size, int seq_length, int seq_size_per_block, int max_batch_size = 0) {
    auto tokens = std::make_shared<CompleteTokenIds>(
        batch_size, max_batch_size > 0 ? max_batch_size : batch_size, seq_length + 64, seq_size_per_block);
    auto  ids = torch::empty({(int64_t)seq_length}, torch::kInt32);
    auto* p   = ids.data_ptr<int32_t>();
    for (int i = 0; i < seq_length; ++i) {
        p[i] = i + 1;
    }
    auto gen             = std::make_shared<GenerateInput>();
    gen->input_ids       = ids;
    gen->generate_config = std::make_shared<GenerateConfig>();
    tokens->init(gen);
    return tokens;
}

BatchKVCacheResourcePtr makeBatchRes(
    int batch_size, int group_nums, int layer_num, const std::vector<int>& layer_to_group_id, CacheKeysType keys) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    res->initGroups(group_nums, layer_num, layer_to_group_id);
    for (int b = 0; b < batch_size; ++b) {
        res->setBatchCacheKeys(b, keys);
    }
    return res;
}

// Cache (key, group-slot) pairs into SharedBlockCache and drop request refs so blocks are reusable.
std::vector<BlockIdxType> seedCache(
    BlockPoolPtr block_pool, SharedBlockCachePtr shared_cache, int group_num, int group_id, const CacheKeysType& keys) {
    auto blocks = block_pool->malloc(static_cast<int>(keys.size()));
    EXPECT_EQ(blocks.size(), keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<BlockIdxType> group_slots(static_cast<size_t>(group_num), NULL_BLOCK_IDX);
        group_slots[static_cast<size_t>(group_id)] = blocks[i];
        shared_cache->put(keys[i], group_slots, true);
    }
    block_pool->requestFree(blocks);
    return blocks;
}

}  // namespace

class HybridKVCacheAllocatorCPShardTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

// 1) When cp_slot_mapper is null/passthrough, behavior is identical to the non-CP baseline:
//    a request occupying 4 logical blocks allocates 4 blocks in the full group.
TEST_F(HybridKVCacheAllocatorCPShardTest, NullMapperIsPassthrough) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    const int gid_full  = 1;
    auto      batch_res = makeBatchRes(/*batch_size=*/1,
                                  /*group_nums=*/2,
                                  /*layer_num=*/static_cast<int>(config.layer_all_num),
                                  /*layer_to_group_id=*/config.layer_to_group_id,
                                  CacheKeysType{100, 101, 102, 103});
    // seq_len=16 => 4 slots @ block_size=4
    auto       tokens = makeTokens(/*batch=*/1, /*seq_len=*/16, /*sspb=*/4);
    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    // cp_slot_mapper intentionally left null.
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, gid_full), 4);
}

// 2) With cp_slot_mapper(cp_rank=0, cp_size=2, block_size=4): a 4-block request allocates ceil(4/2)=2
//    physical blocks on this rank for the full group.
TEST_F(HybridKVCacheAllocatorCPShardTest, ShardedAllocHalvesFullGroup) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    const int gid_full  = 1;
    auto      batch_res = makeBatchRes(
        1, 2, static_cast<int>(config.layer_all_num), config.layer_to_group_id, CacheKeysType{100, 101, 102, 103});
    auto tokens = makeTokens(1, 16, 4);  // 4 logical blocks worth

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    info.cp_slot_mapper      = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, gid_full), 2)
        << "cp_size=2 should halve allocation to ceil(4/2)=2 physical blocks per rank";
}

// 3) Reuse path: cache the last-rank canonical key and confirm a second malloc hits it,
//    returning reuse_len in units of virtualBlockSize (= block_size * cp_size).
TEST_F(HybridKVCacheAllocatorCPShardTest, ReuseHitOnLastRankCanonicalKey) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto block_pool   = allocator->getBlockPool();
    auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(block_pool, nullptr);
    ASSERT_NE(shared_cache, nullptr);

    const int gid_linear = 0;
    const int gid_full   = 1;
    const int group_num  = 2;
    // Full keys for 4 blocks: {100,101,102,103}.
    // localCacheKeys(cp_rank=cp_size-1=1, cp_size=2) selects indices {1,3} => {101, 103}.
    // initMallocForCommonLen drops the last for matching => match_keys = {101}.
    // Joint match requires the linear group's tail to also resolve, so seed both groups with key 101.
    seedCache(block_pool, shared_cache, group_num, gid_full, CacheKeysType{101});
    seedCache(block_pool, shared_cache, group_num, gid_linear, CacheKeysType{101});

    auto batch_res = makeBatchRes(
        1, 2, static_cast<int>(config.layer_all_num), config.layer_to_group_id, CacheKeysType{100, 101, 102, 103});
    auto tokens = makeTokens(1, 16, 4);

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    info.cp_slot_mapper      = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Expect 1 reuse virtual-block * virtualBlockSize(=8 tokens).
    EXPECT_EQ(result.reuse_len, 8);
    // Per-rank physical blocks for full group still = ceil(4/2) = 2.
    EXPECT_EQ(batch_res->blocksNum(0, gid_full), 2);
}

// 4) When reuse is disabled, cp_slot_mapper still translates seq_len for malloc and skips the match.
TEST_F(HybridKVCacheAllocatorCPShardTest, ShardedAllocSkipsReuseWhenDisabled) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto block_pool   = allocator->getBlockPool();
    auto shared_cache = allocator->sharedBlockCache();

    const int gid_full = 1;
    seedCache(block_pool, shared_cache, /*group_num=*/2, gid_full, CacheKeysType{101});

    auto batch_res = makeBatchRes(
        1, 2, static_cast<int>(config.layer_all_num), config.layer_to_group_id, CacheKeysType{100, 101, 102, 103});
    auto tokens = makeTokens(1, 16, 4);

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    info.cp_slot_mapper      = std::make_shared<CPSlotMapper>(0, 2, 4);
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(batch_res->blocksNum(0, gid_full), 2);
}

// 5) insertIntoCache uses last-rank canonical keys and virtualBlockSize when sharded:
//    a 12-token request (full_blocks_num = floor(12/8)=1 virtual block) inserts only key {103}
//    (= last-rank canonical key at index cp_size-1=1 of the first virtual block window).
TEST_F(HybridKVCacheAllocatorCPShardTest, InsertIntoCacheUsesCanonicalKeysAndVirtualBlockSize) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);

    const int gid_full  = 1;
    auto      batch_res = makeBatchRes(
        1, 2, static_cast<int>(config.layer_all_num), config.layer_to_group_id, CacheKeysType{100, 101, 102, 103});

    // seq_len=16 => allocator computes 4 logical blocks; cp_size=2 keeps 2 per rank.
    auto       tokens = makeTokens(1, 16, 4);
    MallocInfo malloc_info{batch_res, tokens};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    malloc_info.cp_slot_mapper      = std::make_shared<CPSlotMapper>(0, 2, 4);
    ASSERT_TRUE(allocator->malloc(malloc_info).success);
    ASSERT_EQ(batch_res->blocksNum(0, gid_full), 2);

    // CompleteTokenIds reflects token-len 16, so token_len-1 = 15. virtualBlockSize=8 =>
    // full_blocks_num = floor(15/8) = 1. n = min(local_keys.size()=2, 1) = 1.
    // local_keys = {101, 103}; first key is 101.
    InsertInfo insert_info{batch_res, tokens, /*is_resident=*/false};
    insert_info.cp_slot_mapper = malloc_info.cp_slot_mapper;
    allocator->insertIntoCache(insert_info);

    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(101, gid_full)));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(100, gid_full)));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(102, gid_full)));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(103, gid_full)));
}

TEST_F(HybridKVCacheAllocatorCPShardTest, CompactLinearReuseKeepsCanonicalVirtualCoordinates) {
    auto config    = makeCompactLinearCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto block_pool   = allocator->getBlockPool();
    auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(block_pool, nullptr);
    ASSERT_NE(shared_cache, nullptr);

    constexpr int gid_linear    = 0;
    constexpr int gid_full      = 1;
    constexpr int group_num     = 2;
    const auto    full_blocks   = seedCache(block_pool, shared_cache, group_num, gid_full, CacheKeysType{101, 103});
    const auto    linear_blocks = seedCache(block_pool, shared_cache, group_num, gid_linear, CacheKeysType{101, 103});

    auto batch_res = makeBatchRes(1,
                                  group_num,
                                  static_cast<int>(config.layer_all_num),
                                  config.layer_to_group_id,
                                  CacheKeysType{100, 101, 102, 103, 104});
    auto tokens    = makeTokens(/*batch_size=*/1, /*seq_length=*/17, /*seq_size_per_block=*/4);

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    info.cp_slot_mapper      = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
    const auto result        = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_EQ(result.reuse_len, 16);
    ASSERT_EQ(batch_res->blocksNum(0, gid_full), 3u);
    ASSERT_EQ(batch_res->blocksNum(0, gid_linear), 3u);
    EXPECT_EQ(batch_res->blocks(0, gid_full)[1], full_blocks[1]);
    EXPECT_EQ(batch_res->blocks(0, gid_linear)[1], linear_blocks[1]);
}

TEST_F(HybridKVCacheAllocatorCPShardTest, CompactLinearReuseDoesNotRepublishEvictedHistoricalCheckpoints) {
    auto config    = makeCompactLinearCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto cache     = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(cache);
    ASSERT_TRUE(allocator->init());
    auto pool   = allocator->getBlockPool();
    auto seeded = pool->malloc(4);
    ASSERT_EQ(seeded.size(), 4u);
    cache->put(101, {seeded[2], seeded[0]}, false);
    cache->put(103, {seeded[3], seeded[1]}, false);
    pool->requestFree(seeded);

    auto res =
        makeBatchRes(1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102, 103, 104});
    auto       tokens = makeTokens(1, 17, 4);
    MallocInfo info{res, tokens};
    info.cp_slot_mapper = std::make_shared<CPSlotMapper>(0, 2, 4);
    const auto result   = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.reuse_len, 16);
    ASSERT_EQ(res->blocksNum(0, 0), 3u);
    EXPECT_EQ(res->blocks(0, 0)[0], NULL_BLOCK_IDX);
    EXPECT_EQ(res->blocks(0, 0)[1], seeded[3]);
    EXPECT_NE(res->blocks(0, 0)[2], NULL_BLOCK_IDX);

    // Prefix=16, suffix=1 and LINEAR span=8 compute only slot 2. Evict
    // the old checkpoints so SharedBlockCache's merge cannot mask pollution.
    cache->evictAndFreeForGroup(0, 2);
    ASSERT_EQ(cache->matchGroup(101, 0), NULL_BLOCK_IDX);
    allocator->insertIntoCache(InsertInfo{res, tokens, false, info.cp_slot_mapper});
    EXPECT_EQ(cache->matchGroup(101, 0), NULL_BLOCK_IDX);
    EXPECT_EQ(cache->matchGroup(103, 0), seeded[3]);

    auto short_res = makeBatchRes(1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102});
    auto short_tokens = makeTokens(1, 9, 4);
    MallocInfo short_info{short_res, short_tokens};
    short_info.cp_slot_mapper = info.cp_slot_mapper;
    const auto short_result   = allocator->malloc(short_info);
    ASSERT_TRUE(short_result.success);
    EXPECT_EQ(short_result.reuse_len, 0) << "the shorter checkpoint was evicted and must be recomputed";
    allocator->free(FreeInfo{short_res, short_tokens});

    // A request reaching the retained checkpoint still reuses its valid state.
    auto long_res =
        makeBatchRes(1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102, 103, 104});
    MallocInfo long_info{long_res, tokens};
    long_info.cp_slot_mapper = info.cp_slot_mapper;
    const auto long_result   = allocator->malloc(long_info);
    ASSERT_TRUE(long_result.success);
    EXPECT_EQ(long_result.reuse_len, 16);
    EXPECT_EQ(long_res->blocks(0, 0)[1], seeded[3]);
    allocator->free(FreeInfo{long_res, tokens});
    allocator->free(FreeInfo{res, tokens});
}

TEST_F(HybridKVCacheAllocatorCPShardTest, CompactLinearReuseAllocatesOnlyComputedSuffixUnderPressure) {
    for (size_t available : {1u, 2u}) {
        SCOPED_TRACE(available);
        auto config    = makeCompactLinearCPHybridConfig();
        auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
        allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
        ASSERT_TRUE(allocator->init());
        auto pool  = allocator->getBlockPool();
        auto cache = allocator->sharedBlockCache();
        seedCache(pool, cache, 2, 1, CacheKeysType{101, 103});
        seedCache(pool, cache, 2, 0, CacheKeysType{101, 103});
        auto       held        = pool->malloc(pool->freeBlocksNum() - available);
        const auto refs_before = pool->requestRefBlocksNum();
        auto       res =
            makeBatchRes(1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102, 103, 104});
        auto       tokens = makeTokens(1, 17, 4);
        MallocInfo info{res, tokens};
        info.cp_slot_mapper = std::make_shared<CPSlotMapper>(0, 2, 4);
        const auto result   = allocator->malloc(info);
        // Exactly one new LINEAR state and one FULL page are needed.
        EXPECT_EQ(result.success, available == 2u);
        if (result.success) {
            EXPECT_EQ(result.reuse_len, 16);
            EXPECT_EQ(res->blocks(0, 0)[0], NULL_BLOCK_IDX);
            allocator->free(FreeInfo{res, tokens});
        } else {
            EXPECT_EQ(res->blocksNum(0, 0), 0u);
            EXPECT_EQ(res->blocksNum(0, 1), 0u);
        }
        EXPECT_EQ(pool->requestRefBlocksNum(), refs_before);
        pool->requestFree(held);
    }
}

TEST_F(HybridKVCacheAllocatorCPShardTest, ReplicatedCompactLinearEstimatesReuseInEachGroupsCoordinates) {
    auto config    = makeCompactLinearCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());
    auto pool  = allocator->getBlockPool();
    auto cache = allocator->sharedBlockCache();
    seedCache(pool, cache, 2, 1, CacheKeysType{100, 101, 102, 103});
    seedCache(pool, cache, 2, 0, CacheKeysType{101, 103});
    auto res =
        makeBatchRes(1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102, 103, 104});
    auto       tokens = makeTokens(1, 17, 4);
    MallocInfo info{res, tokens};

    // Replicated Decode uses four FULL pages but only two LINEAR slots
    // for the same 16-token cached prefix.
    ASSERT_EQ(allocator->reuseCache(CacheKeysType{100, 101, 102, 103}, *res, nullptr), 4);
    ASSERT_EQ(res->blocksNum(0, 0), 2);
    ASSERT_EQ(res->blocksNum(0, 1), 4);
    EXPECT_EQ(allocator->getNeedBlocks(info), 2);  // One LINEAR state and one FULL page.
}

TEST_F(HybridKVCacheAllocatorCPShardTest, CompactLinearReusedCommonPhasePreservesHistoricalNullTail) {
    auto config    = makeCompactLinearCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto cache     = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(cache);
    ASSERT_TRUE(allocator->init());
    auto pool   = allocator->getBlockPool();
    auto seeded = pool->malloc(4);
    ASSERT_EQ(seeded.size(), 4u);
    cache->put(101, {seeded[2], seeded[0]}, false);
    cache->put(103, {seeded[3], seeded[1]}, false);
    pool->requestFree(seeded);
    auto res =
        makeBatchRes(1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102, 103, 104});
    // Initial beam prefill still has one active batch, but capacity for two
    // beams rounds the common phase down to 16 tokens: exactly the reused prefix.
    auto tokens = makeTokens(1, 17, 4, /*max_batch_size=*/2);
    ASSERT_EQ(tokens->commonSeqLength(), 16);
    MallocInfo info{res, tokens};
    info.cp_slot_mapper = std::make_shared<CPSlotMapper>(0, 2, 4);
    const auto result   = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.reuse_len, 16);
    ASSERT_EQ(res->blocksNum(0, 0), 3);
    EXPECT_EQ(res->blocks(0, 0)[0], NULL_BLOCK_IDX);
    EXPECT_EQ(res->blocks(0, 0)[1], seeded[3]);
    EXPECT_NE(res->blocks(0, 0)[2], NULL_BLOCK_IDX);
    cache->evictAndFreeForGroup(0, 2);
    ASSERT_EQ(cache->matchGroup(101, 0), NULL_BLOCK_IDX);
    allocator->insertIntoCache(InsertInfo{res, tokens, false, info.cp_slot_mapper});
    EXPECT_EQ(cache->matchGroup(101, 0), NULL_BLOCK_IDX);
    auto short_res = makeBatchRes(1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102});
    auto short_tokens = makeTokens(1, 9, 4);
    MallocInfo short_info{short_res, short_tokens};
    short_info.cp_slot_mapper = info.cp_slot_mapper;
    const auto short_result   = allocator->malloc(short_info);
    ASSERT_TRUE(short_result.success);
    EXPECT_EQ(short_result.reuse_len, 0);
    allocator->free(FreeInfo{short_res, short_tokens});
    allocator->free(FreeInfo{res, tokens});
}

TEST_F(HybridKVCacheAllocatorCPShardTest, ReplicatedCompactLinearReusePreservesReserveBudget) {
    auto config    = makeCompactLinearCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(
        config, AllocationType::DEVICE, nullptr, /*reserve_block_ratio=*/10);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());
    ASSERT_EQ(allocator->reserveBlockNum(), 3u);
    auto pool  = allocator->getBlockPool();
    auto cache = allocator->sharedBlockCache();
    seedCache(pool, cache, 2, 1, CacheKeysType{100, 101, 102, 103});
    seedCache(pool, cache, 2, 0, CacheKeysType{101, 103});
    auto       held        = pool->malloc(pool->freeBlocksNum() - 3);
    const auto refs_before = pool->requestRefBlocksNum();
    auto       res =
        makeBatchRes(1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102, 103, 104});
    auto       tokens = makeTokens(1, 17, 4);
    MallocInfo info{res, tokens};

    // After referencing the reused tail and FULL prefix, four blocks remain
    // available (three free plus the historical LINEAR checkpoint). Two new
    // blocks would leave fewer than the three-block reserve.
    const auto result = allocator->malloc(info);
    EXPECT_FALSE(result.success);
    if (result.success) {
        EXPECT_GE(allocator->availableBlocksNum(), allocator->reserveBlockNum());
        allocator->free(FreeInfo{res, tokens});
    }
    EXPECT_EQ(res->blocksNum(0, 0), 0);
    EXPECT_EQ(res->blocksNum(0, 1), 0);
    EXPECT_EQ(pool->requestRefBlocksNum(), refs_before);
    pool->requestFree(held);
}

TEST_F(HybridKVCacheAllocatorCPShardTest, PdLinearPrefixLoadKeepsOnlyTerminalStateAcrossDecodeSteps) {
    auto config    = makeCompactLinearCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto cache     = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(cache);
    ASSERT_TRUE(allocator->init());
    auto res =
        makeBatchRes(1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102, 103, 104});
    auto tokens = makeTokens(1, 17, 4);
    tokens->setReserveStep(3);
    MallocInfo info{res, tokens};
    info.enable_device_cache       = false;
    info.linear_prefix_load_tokens = 17;
    EXPECT_EQ(allocator->getNeedBlocks(info), 8);  // Five FULL pages, terminal LINEAR state and two reserve slots.
    ASSERT_TRUE(allocator->malloc(info).success);
    ASSERT_EQ(res->blocksNum(0, 0), 5);
    EXPECT_EQ(res->blocks(0, 0)[0], NULL_BLOCK_IDX);
    EXPECT_EQ(res->blocks(0, 0)[1], NULL_BLOCK_IDX);
    const auto terminal = res->blocks(0, 0)[2];
    ASSERT_NE(terminal, NULL_BLOCK_IDX);
    EXPECT_NE(res->blocks(0, 0)[3], NULL_BLOCK_IDX);
    EXPECT_NE(res->blocks(0, 0)[4], NULL_BLOCK_IDX);

    // The transfer writes only this terminal state. Later decode steps do
    // not recompute either historical checkpoint, including tail-1 at 18/19.
    auto state = torch::from_blob(allocator->convertIndexToAddr(0, terminal).kv_addr,
                                  {1},
                                  torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    state.fill_(0x11223344);
    for (int seq_len : {18, 19, 25}) {
        tokens->setSeqLength(seq_len);
        MallocInfo incr_info{res, tokens};
        ASSERT_TRUE(allocator->malloc(incr_info).success);
        EXPECT_EQ(res->blocks(0, 0)[0], NULL_BLOCK_IDX);
        EXPECT_EQ(res->blocks(0, 0)[1], NULL_BLOCK_IDX);
        EXPECT_EQ(res->blocks(0, 0)[2], terminal);
    }
    // Model execution completes the next checkpoint in the retained slot.
    state.fill_(0x55667788);
    res->setBatchCacheKeys(0, CacheKeysType{100, 101, 102, 103, 104, 105, 106});
    allocator->insertIntoCache(InsertInfo{res, tokens, false});
    EXPECT_EQ(cache->matchGroup(101, 0), NULL_BLOCK_IDX);
    EXPECT_EQ(cache->matchGroup(103, 0), NULL_BLOCK_IDX);
    EXPECT_EQ(cache->matchGroup(105, 0), terminal);

    auto short_res = makeBatchRes(1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102});
    auto short_tokens       = makeTokens(1, 9, 4);
    const auto short_result = allocator->malloc(MallocInfo{short_res, short_tokens});
    ASSERT_TRUE(short_result.success);
    EXPECT_EQ(short_result.reuse_len, 0);
    allocator->free(FreeInfo{short_res, short_tokens});

    auto later_res = makeBatchRes(
        1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102, 103, 104, 105, 106});
    const auto later_result = allocator->malloc(MallocInfo{later_res, tokens});
    ASSERT_TRUE(later_result.success);
    EXPECT_EQ(later_result.reuse_len, 24);
    EXPECT_EQ(later_res->blocks(0, 0)[2], terminal);
    EXPECT_EQ(state.cpu().item<int32_t>(), 0x55667788);
    allocator->free(FreeInfo{later_res, tokens});
    allocator->free(FreeInfo{res, tokens});
}

TEST_F(HybridKVCacheAllocatorCPShardTest, PdLinearPrefixLoadRollsBackAndHonorsExactBudget) {
    for (bool reuse_cache : {false, true}) {
        for (size_t available : {5u, 6u}) {
            SCOPED_TRACE(::testing::Message() << "reuse=" << reuse_cache << " available=" << available);
            auto config    = makeCompactLinearCPHybridConfig();
            auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
            allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
            ASSERT_TRUE(allocator->init());
            auto       pool        = allocator->getBlockPool();
            auto       held        = pool->malloc(pool->freeBlocksNum() - available);
            const auto refs_before = pool->requestRefBlocksNum();
            auto       res         = makeBatchRes(
                1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102, 103, 104});
            auto       tokens = makeTokens(1, 17, 4);
            MallocInfo info{res, tokens};
            info.enable_device_cache       = false;
            info.reuse_cache               = reuse_cache;
            info.linear_prefix_load_tokens = 17;
            EXPECT_EQ(allocator->getNeedBlocks(info), 6);
            const auto result = allocator->malloc(info);
            EXPECT_EQ(result.success, available == 6u);
            if (result.success) {
                EXPECT_EQ(res->blocks(0, 0)[0], NULL_BLOCK_IDX);
                EXPECT_EQ(res->blocks(0, 0)[1], NULL_BLOCK_IDX);
                allocator->free(FreeInfo{res, tokens});
            }
            EXPECT_EQ(res->blocksNum(0, 0), 0);
            EXPECT_EQ(res->blocksNum(0, 1), 0);
            EXPECT_EQ(pool->requestRefBlocksNum(), refs_before);
            pool->requestFree(held);
        }
    }
}

TEST_F(HybridKVCacheAllocatorCPShardTest, PdAlignedLinearPrefixLoadPreservesEarlierValidReuse) {
    auto config    = makeCompactLinearCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    auto cache     = std::make_shared<SharedBlockCache>();
    allocator->setSharedBlockCache(cache);
    ASSERT_TRUE(allocator->init());
    auto seed        = makeBatchRes(1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102});
    auto seed_tokens = makeTokens(1, 9, 4);
    ASSERT_TRUE(allocator->malloc(MallocInfo{seed, seed_tokens}).success);
    const auto valid_checkpoint = seed->blocks(0, 0)[0];
    allocator->insertIntoCache(InsertInfo{seed, seed_tokens, false});
    allocator->free(FreeInfo{seed, seed_tokens});

    auto res =
        makeBatchRes(1, 2, config.layer_all_num, config.layer_to_group_id, CacheKeysType{100, 101, 102, 103, 104, 105});
    auto       tokens = makeTokens(1, 24, 4);
    MallocInfo info{res, tokens};
    info.linear_prefix_load_tokens = 24;
    const auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.reuse_len, 8);
    ASSERT_EQ(res->blocksNum(0, 0), 3);
    EXPECT_EQ(res->blocks(0, 0)[0], valid_checkpoint);
    EXPECT_EQ(res->blocks(0, 0)[1], NULL_BLOCK_IDX);
    EXPECT_NE(res->blocks(0, 0)[2], NULL_BLOCK_IDX);
    tokens->setSeqLength(25);
    ASSERT_TRUE(allocator->malloc(MallocInfo{res, tokens}).success);
    EXPECT_EQ(res->blocks(0, 0)[0], valid_checkpoint);
    EXPECT_EQ(res->blocks(0, 0)[1], NULL_BLOCK_IDX);
    EXPECT_NE(res->blocks(0, 0)[3], NULL_BLOCK_IDX);
    allocator->free(FreeInfo{res, tokens});
}

// 6) Two-malloc smoke: cp_size=4 sharding, request occupies 8 logical blocks ⇒ 2 per rank.
TEST_F(HybridKVCacheAllocatorCPShardTest, ShardedAllocCpSize4) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    const int     gid_full = 1;
    CacheKeysType keys;
    for (int i = 0; i < 8; ++i) {
        keys.push_back(200 + i);
    }
    auto batch_res = makeBatchRes(1, 2, static_cast<int>(config.layer_all_num), config.layer_to_group_id, keys);
    auto tokens    = makeTokens(1, /*seq_len=*/32, 4);  // 8 logical blocks

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    info.cp_slot_mapper      = std::make_shared<CPSlotMapper>(/*cp_rank=*/2, /*cp_size=*/4, /*block_size=*/4);
    auto result              = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, gid_full), 2);  // ceil(8/4)=2
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
