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
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

using TestHybridTypeKVCacheAllocator = BlockTreeCacheTestAllocator<HybridTypeKVCacheAllocator>;

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

    config.fromGroupedSpecs(
        {linear_spec, full_spec}, {{0, 1}, {2, 3}}, {CacheGroupType::LINEAR, CacheGroupType::FULL}, {"linear", "full"});

    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;
    config.layer_to_block_stride_bytes.assign(
        config.layer_all_num, static_cast<int>(config.kv_block_stride_bytes + config.kv_scale_stride_bytes));

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
    res->initGroups(config.topologyPtr());
    for (int b = 0; b < batch_size; ++b) {
        res->setBatchCacheKeys(b, keys);
    }
    return res;
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
    auto allocator = std::make_shared<TestHybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const int gid_full  = 1;
    auto      batch_res = makeBatchRes(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
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
    auto allocator = std::make_shared<TestHybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const int gid_full  = 1;
    auto      batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto      tokens    = makeTokens(1, 16, 4);  // 4 logical blocks worth

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, gid_full), 2)
        << "cp_size=2 should halve allocation to ceil(4/2)=2 physical blocks per rank";
}

// 3) Reuse path: cache the last-rank canonical key and confirm a second malloc hits it,
//    returning reuse_len in units of virtualBlockSize (= block_size * cp_size).
TEST_F(HybridKVCacheAllocatorCPShardTest, ReuseHitOnLastRankCanonicalKey) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const int gid_full = 1;
    // Full keys for 4 blocks: {100,101,102,103}.
    // localCacheKeys(cp_rank=cp_size-1=1, cp_size=2) selects indices {1,3} => {101, 103}.
    // initMallocForCommonLen drops the last for matching => match_keys = {101}.
    // Joint match requires a physically complete declarative coordinate for key 101.
    const auto seeded = seedCompleteBlockTreePath(allocator, CacheKeysType{101});
    ASSERT_TRUE(seeded.success);

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
    EXPECT_EQ(batch_res->blocksNum(0, gid_full), 2);
}

// 4) When reuse is disabled, cp_slot_mapper still translates seq_len for malloc and skips the match.
TEST_F(HybridKVCacheAllocatorCPShardTest, ShardedAllocSkipsReuseWhenDisabled) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const int  gid_full = 1;
    const auto seeded   = seedCompleteBlockTreePath(allocator, CacheKeysType{101});
    ASSERT_TRUE(seeded.success);

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto tokens    = makeTokens(1, 16, 4);

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(0, 2, 4));
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(batch_res->blocksNum(0, gid_full), 2);
}

// 5) insertIntoCache publishes the allocator's complete rank-local blocks on
//    the last-rank canonical key chain. KVCacheManager owns partial-tail
//    trimming before this allocator boundary.
TEST_F(HybridKVCacheAllocatorCPShardTest, InsertIntoCacheUsesCanonicalKeysAndVirtualBlockSize) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const int gid_full  = 1;
    auto      batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});

    // seq_len=16 => allocator computes 4 logical blocks; cp_size=2 keeps 2 per rank.
    auto       tokens = makeTokens(1, 16, 4);
    MallocInfo malloc_info{batch_res, tokens};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    allocator->setCPSlotMapper(std::make_shared<CPSlotMapper>(0, 2, 4));
    ASSERT_TRUE(allocator->malloc(malloc_info).success);
    ASSERT_EQ(batch_res->blocksNum(0, gid_full), 2);
    const auto full_blocks = batch_res->blocks(0, gid_full);

    InsertInfo insert_info{batch_res, tokens, /*is_resident=*/false};
    allocator->insertIntoCache(insert_info);

    const auto snapshot = allocator->blockTreeCacheOwner()->getKeySnapshot(10);
    EXPECT_EQ(snapshot.keys, (CacheKeysType{101, 103}));

    auto match = allocator->blockTreeCacheOwner()->match(CacheKeysType{101, 103});
    ASSERT_EQ(match.matched_blocks, 2u);
    ASSERT_EQ(match.group_block_indices.at(config.tagForGroup(gid_full)), full_blocks);
    allocator->blockTreeCacheOwner()->releaseMatchedBlocks(match.matched_block_sets);
}

// 6) Two-malloc smoke: cp_size=4 sharding, request occupies 8 logical blocks ⇒ 2 per rank.
TEST_F(HybridKVCacheAllocatorCPShardTest, ShardedAllocCpSize4) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const int     gid_full = 1;
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
    EXPECT_EQ(batch_res->blocksNum(0, gid_full), 2);  // ceil(8/4)=2
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
