// CP-shard (Stage 5, Plan A) UTs for HybridPoolKVCacheAllocator.
//
// These exercise the cp_slot_mapper plumbing in initMallocForCommonLen,
// incrMalloc, insertIntoCache, and getNeedBlocks. The shape of the tests
// piggybacks on the helpers in HybridPoolKVCacheAllocatorTest.cc but
// keeps the configuration self-contained so the two files build cleanly
// alongside each other.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/SharedBlockCache.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

namespace {

// Two-group hybrid: the "linear" group (not exercised here) and the "full" group (the CP-shard target).
CacheConfig makeCPHybridConfig() {
    CacheConfig config;
    config.dtype              = rtp_llm::DataType::TYPE_FP16;
    config.layer_num          = 4;
    config.block_num          = 32;  // headroom for cp_size=2 expansion
    config.seq_size_per_block = 4;
    config.linear_step        = 2;

    auto linear_spec = makeResolvedLinearSpec(config.dtype,
                                              1,
                                              1,
                                              1,
                                              1,
                                              2,
                                              static_cast<uint32_t>(config.seq_size_per_block),
                                              config.dtype,
                                              config.dtype,
                                              "linear");
    auto full_spec = makeResolvedMhaSpec(config.dtype, 1, 1, static_cast<uint32_t>(config.seq_size_per_block), "full");

    rtp_llm::test::assignCacheConfigFromGroupedSpecs(config,
                                                     config.layer_num,
                                                     {linear_spec, full_spec},
                                                     {{0, 1}, {2, 3}},
                                                     {CacheGroupType::LINEAR, CacheGroupType::FULL},
                                                     {"linear", "full"});

    config.finalizeBlockNums(config.block_num, RuntimeConfig{});
    return config;
}

CompleteTokenIdsPtr makeTokens(int batch_size, int seq_length, int seq_size_per_block) {
    auto  tokens = std::make_shared<CompleteTokenIds>(batch_size, batch_size, seq_length + 64, seq_size_per_block);
    auto  ids    = torch::empty({(int64_t)seq_length}, torch::kInt32);
    auto* p      = ids.data_ptr<int32_t>();
    for (int i = 0; i < seq_length; ++i) {
        p[i] = i + 1;
    }
    auto gen             = std::make_shared<GenerateInput>();
    gen->input_ids       = ids;
    gen->generate_config = std::make_shared<GenerateConfig>();
    tokens->init(gen);
    return tokens;
}

BatchKVCacheResourcePtr makeBatchRes(int batch_size, const CacheConfig& config, CacheKeysType keys) {
    auto res = std::make_shared<BatchKVCacheResource>();
    res->resetBatchSize(batch_size);
    res->initGroups(config);
    for (int b = 0; b < batch_size; ++b) {
        res->setBatchCacheKeys(b, keys);
    }
    return res;
}

// Cache (key, group-block) pairs into SharedBlockCache and drop request refs so blocks are reusable.
std::vector<BlockIdxType>
seedCache(BlockPoolPtr block_pool, SharedBlockCachePtr shared_cache, std::string_view tag, const CacheKeysType& keys) {
    auto blocks = block_pool->malloc(static_cast<int>(keys.size()));
    EXPECT_EQ(blocks.size(), keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        shared_cache->put(keys[i], {{std::string(tag), blocks[i]}}, true);
    }
    block_pool->requestFree(blocks);
    return blocks;
}

}  // namespace

class HybridPoolKVCacheAllocatorCPShardTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

// 1) When cp_slot_mapper is null/passthrough, behavior is identical to the non-CP baseline:
//    a request occupying 4 logical blocks allocates 4 blocks in the full group.
TEST_F(HybridPoolKVCacheAllocatorCPShardTest, NullMapperIsPassthrough) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchRes(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    // seq_len=16 => 4 slots @ block_size=4
    auto       tokens = makeTokens(/*batch=*/1, /*seq_len=*/16, /*sspb=*/4);
    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    // cp_slot_mapper intentionally left null.
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 4);
}

// 2) With cp_slot_mapper(cp_rank=0, cp_size=2, block_size=4): a 4-block request allocates ceil(4/2)=2
//    physical blocks on this rank for the full group.
TEST_F(HybridPoolKVCacheAllocatorCPShardTest, ShardedAllocHalvesFullGroup) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto tokens    = makeTokens(1, 16, 4);  // 4 logical blocks worth

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 2)
        << "cp_size=2 should halve allocation to ceil(4/2)=2 physical blocks per rank";
}

TEST_F(HybridPoolKVCacheAllocatorCPShardTest, HybridPoolCoordinatorPreservesShardedAllocation) {
    auto config = makeCPHybridConfig();
    setGroupBlockLayout(config,
                        {32, 32},
                        {config.group("linear").kv_block_stride_bytes, config.group("full").kv_block_stride_bytes},
                        {0, 0});
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto       batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto       tokens    = makeTokens(1, 16, 4);
    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    ASSERT_TRUE(allocator->malloc(info).success);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 2);
}

// 3) Reuse path: cache the last-rank canonical key and confirm a second malloc hits it,
//    returning reuse_len in units of virtualBlockSize (= block_size * cp_size).
TEST_F(HybridPoolKVCacheAllocatorCPShardTest, ReuseHitOnLastRankCanonicalKey) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);

    // Full keys for 4 blocks: {100,101,102,103}.
    // localCacheKeys(cp_rank=cp_size-1=1, cp_size=2) selects indices {1,3} => {101, 103}.
    // initMallocForCommonLen drops the last for matching => match_keys = {101}.
    // Joint match requires the linear group's tail to also resolve, so seed both groups with key 101.
    seedCache(allocator->blockPool("full"), shared_cache, "full", CacheKeysType{101});
    seedCache(allocator->blockPool("linear"), shared_cache, "linear", CacheKeysType{101});

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto tokens    = makeTokens(1, 16, 4);

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    // Expect 1 reuse virtual-block * virtualBlockSize(=8 tokens).
    EXPECT_EQ(result.reuse_len, 8);
    // Per-rank physical blocks for full group still = ceil(4/2) = 2.
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 2);
}

// 4) When reuse is disabled, cp_slot_mapper still translates seq_len for malloc and skips the match.
TEST_F(HybridPoolKVCacheAllocatorCPShardTest, ShardedAllocSkipsReuseWhenDisabled) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto shared_cache = allocator->sharedBlockCache();

    seedCache(allocator->blockPool("full"), shared_cache, "full", CacheKeysType{101});

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto tokens    = makeTokens(1, 16, 4);

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(0, 2, 4));
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 2);
}

// 5) insertIntoCache uses last-rank canonical keys and virtualBlockSize when sharded:
//    a 12-token request (full_blocks_num = floor(12/8)=1 virtual block) inserts only key {103}
//    (= last-rank canonical key at index cp_size-1=1 of the first virtual block window).
TEST_F(HybridPoolKVCacheAllocatorCPShardTest, InsertIntoCacheUsesCanonicalKeysAndVirtualBlockSize) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    auto shared_cache = allocator->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});

    // seq_len=16 => allocator computes 4 logical blocks; cp_size=2 keeps 2 per rank.
    auto       tokens = makeTokens(1, 16, 4);
    MallocInfo malloc_info{batch_res, tokens};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(0, 2, 4));
    ASSERT_TRUE(allocator->malloc(malloc_info).success);
    ASSERT_EQ(batch_res->blocksNum(0, "full"), 2);

    // CompleteTokenIds reflects token-len 16, so token_len-1 = 15. virtualBlockSize=8 =>
    // full_blocks_num = floor(15/8) = 1. n = min(local_keys.size()=2, 1) = 1.
    // local_keys = {101, 103}; first key is 101.
    InsertInfo insert_info{batch_res, tokens, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(101, "full")));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(100, "full")));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(102, "full")));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(103, "full")));
}

// 6) Two-malloc smoke: cp_size=4 sharding, request occupies 8 logical blocks ⇒ 2 per rank.
TEST_F(HybridPoolKVCacheAllocatorCPShardTest, ShardedAllocCpSize4) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<HybridPoolKVCacheAllocator>(config, AllocationType::DEVICE);
    allocator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(allocator->init());

    CacheKeysType keys;
    for (int i = 0; i < 8; ++i) {
        keys.push_back(200 + i);
    }
    auto batch_res = makeBatchRes(1, config, keys);
    auto tokens    = makeTokens(1, /*seq_len=*/32, 4);  // 8 logical blocks

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/2, /*cp_size=*/4, /*block_size=*/4));
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 2);  // ceil(8/4)=2
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
