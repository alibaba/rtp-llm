// CP-shard (Stage 5, Plan A) UTs for CoordinatorCacheManager.
//
// These exercise the cp_slot_mapper plumbing in initMallocForCommonLen,
// incrMalloc, insertIntoCache, and getNeedBlocks. The shape of the tests
// piggybacks on the helpers in CoordinatorCacheManagerTest.cc but
// keeps the configuration self-contained so the two files build cleanly
// alongside each other.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
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
                                                     /*main_layer_num=*/4,
                                                     {linear_spec, full_spec},
                                                     {{0, 1}, {2, 3}},
                                                     {CacheGroupType::LINEAR, CacheGroupType::FULL},
                                                     {"linear", "full"});

    config.finalizeBlockNums(config.block_num, RuntimeConfig{});
    return config;
}

CacheConfig makeMixedFullCPConfig(uint32_t logical_group_block_size = 4,
                                  bool     reverse_groups           = false,
                                  uint32_t cache_key_block_size     = 4) {
    CacheConfig config;
    config.dtype              = DataType::TYPE_FP16;
    config.block_num          = 32;
    config.seq_size_per_block = cache_key_block_size;

    auto rr_spec              = makeResolvedMhaSpec(config.dtype, 1, 1, cache_key_block_size, "rr");
    auto logical_spec         = makeResolvedMhaSpec(config.dtype, 1, 1, logical_group_block_size, "logical");
    auto rr_policy            = defaultCacheGroupPolicy(CacheGroupType::FULL);
    rr_policy.cp_mapping      = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    auto logical_policy       = defaultCacheGroupPolicy(CacheGroupType::FULL);
    logical_policy.cp_mapping = CpBlockMappingMode::NONE;

    if (reverse_groups) {
        assignCacheConfigFromGroupedSpecs(config,
                                          /*main_layer_num=*/2,
                                          {logical_spec, rr_spec},
                                          {{1}, {0}},
                                          {CacheGroupType::FULL, CacheGroupType::FULL},
                                          {"logical", "rr"},
                                          {logical_policy, rr_policy});
    } else {
        assignCacheConfigFromGroupedSpecs(config,
                                          /*main_layer_num=*/2,
                                          {rr_spec, logical_spec},
                                          {{0}, {1}},
                                          {CacheGroupType::FULL, CacheGroupType::FULL},
                                          {"rr", "logical"},
                                          {rr_policy, logical_policy});
    }
    config.finalizeBlockNums(config.block_num, RuntimeConfig{});
    return config;
}

CacheConfig makeLogicalFullConfig(uint32_t physical_block_size) {
    CacheConfig config;
    config.dtype              = DataType::TYPE_FP16;
    config.block_num          = 32;
    config.seq_size_per_block = 4;

    auto spec         = makeResolvedMhaSpec(config.dtype, 1, 1, physical_block_size, "logical");
    auto policy       = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.cp_mapping = CpBlockMappingMode::NONE;
    assignCacheConfigFromGroupedSpecs(config,
                                      /*main_layer_num=*/1,
                                      {spec},
                                      {{0}},
                                      {CacheGroupType::FULL},
                                      {"logical"},
                                      {policy});
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
        shared_cache->put(keys[i], {{std::string(tag), blocks[i]}}, {}, true, BlockDependency{});
    }
    block_pool->requestFree(blocks);
    return blocks;
}

}  // namespace

class CoordinatorCacheManagerCPShardTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

// 1) When cp_slot_mapper is null/passthrough, behavior is identical to the non-CP baseline:
//    a request occupying 4 logical blocks allocates 4 blocks in the full group.
TEST_F(CoordinatorCacheManagerCPShardTest, NullMapperIsPassthrough) {
    auto config                    = makeCPHybridConfig();
    auto coordinator_cache_manager = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    coordinator_cache_manager->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(coordinator_cache_manager->init());

    auto batch_res = makeBatchRes(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    // seq_len=16 => 4 slots @ block_size=4
    auto       tokens = makeTokens(/*batch=*/1, /*seq_len=*/16, /*sspb=*/4);
    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    // cp_slot_mapper intentionally left null.
    auto result = coordinator_cache_manager->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 4);
}

// 2) With cp_slot_mapper(cp_rank=0, cp_size=2, block_size=4): a 4-block request allocates ceil(4/2)=2
//    physical blocks on this rank for the full group.
TEST_F(CoordinatorCacheManagerCPShardTest, ShardedAllocHalvesFullGroup) {
    auto config                    = makeCPHybridConfig();
    auto coordinator_cache_manager = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    coordinator_cache_manager->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(coordinator_cache_manager->init());

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto tokens    = makeTokens(1, 16, 4);  // 4 logical blocks worth

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    coordinator_cache_manager->setCPSlotMapper(
        std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));
    auto result = coordinator_cache_manager->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 2)
        << "cp_size=2 should halve allocation to ceil(4/2)=2 physical blocks per rank";
}

TEST_F(CoordinatorCacheManagerCPShardTest, HybridPoolCoordinatorPreservesShardedAllocation) {
    auto config = makeCPHybridConfig();
    setGroupBlockLayout(config,
                        {32, 32},
                        {config.group("linear").kv_block_stride_bytes, config.group("full").kv_block_stride_bytes},
                        {0, 0});
    auto coordinator_cache_manager = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    coordinator_cache_manager->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(coordinator_cache_manager->init());

    auto       batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto       tokens    = makeTokens(1, 16, 4);
    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    coordinator_cache_manager->setCPSlotMapper(
        std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));

    ASSERT_TRUE(coordinator_cache_manager->malloc(info).success);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 2);
}

// 3) Reuse path: four CP-canonical hits still report eight global cache-key blocks.
TEST_F(CoordinatorCacheManagerCPShardTest, ReuseHitOnLastRankCanonicalKey) {
    auto config                    = makeCPHybridConfig();
    auto coordinator_cache_manager = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    coordinator_cache_manager->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(coordinator_cache_manager->init());

    auto shared_cache = coordinator_cache_manager->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);

    const CacheKeysType keys{100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    // The final canonical key is intentionally left as the dropped tail.
    const CacheKeysType reusable_keys{101, 103, 105, 107};
    seedCache(coordinator_cache_manager->blockPool("full"), shared_cache, "full", reusable_keys);
    seedCache(coordinator_cache_manager->blockPool("linear"), shared_cache, "linear", reusable_keys);

    auto batch_res = makeBatchRes(1, config, keys);
    auto tokens    = makeTokens(1, 40, 4);

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    coordinator_cache_manager->setCPSlotMapper(
        std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4));
    auto result = coordinator_cache_manager->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_EQ(batch_res->cacheResource(0).deviceReuseBlockNum(), 8u);
    EXPECT_EQ(result.reuse_len, 32);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 5);
}

TEST_F(CoordinatorCacheManagerCPShardTest, InsertFreeReusePreservesCanonicalBlockIdentity) {
    auto config      = makeCPHybridConfig();
    auto coordinator = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    coordinator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(coordinator->init());
    auto mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
    coordinator->setCPSlotMapper(mapper);

    const CacheKeysType full_keys{100, 101, 102, 103};
    auto                seed_resource = makeBatchRes(/*batch_size=*/1, config, full_keys);
    auto                seed_tokens   = makeTokens(/*batch_size=*/1, /*seq_length=*/16, /*seq_size_per_block=*/4);
    MallocInfo          seed_malloc{seed_resource, seed_tokens};
    seed_malloc.enable_device_cache = false;
    seed_malloc.reuse_cache         = true;
    ASSERT_TRUE(coordinator->malloc(seed_malloc).success);
    ASSERT_EQ(seed_resource->blocksNum(0, "full"), 2);
    const auto reused_full_block = seed_resource->blocks(0, "full").front();

    coordinator->insertIntoCache(InsertInfo{seed_resource, seed_tokens, /*is_resident=*/false});
    coordinator->free(FreeInfo{seed_resource, seed_tokens});
    EXPECT_EQ(coordinator->blockPool("full")->requestRefBlocksNum(), 0u);
    EXPECT_EQ(coordinator->blockPool("full")->blockCacheRefBlocksNum(), 1u);
    EXPECT_EQ(coordinator->sharedBlockCache()->matchGroup(101, "full"), reused_full_block);
    EXPECT_FALSE(isNullBlockIdx(coordinator->sharedBlockCache()->matchGroup(101, "linear")));

    auto       hit_resource = makeBatchRes(/*batch_size=*/1, config, full_keys);
    auto       hit_tokens   = makeTokens(/*batch_size=*/1, /*seq_length=*/16, /*seq_size_per_block=*/4);
    MallocInfo hit_malloc{hit_resource, hit_tokens};
    hit_malloc.enable_device_cache = true;
    hit_malloc.reuse_cache         = true;
    auto result                    = coordinator->malloc(hit_malloc);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, mapper->virtualBlockSize());
    ASSERT_EQ(hit_resource->blocksNum(0, "full"), 2);
    EXPECT_EQ(hit_resource->blocks(0, "full").front(), reused_full_block);
    EXPECT_EQ(coordinator->blockPool("full")->requestRefBlocksNum(), 2u);
    coordinator->free(FreeInfo{hit_resource, hit_tokens});
    EXPECT_EQ(coordinator->blockPool("full")->requestRefBlocksNum(), 0u);
}

TEST_F(CoordinatorCacheManagerCPShardTest, MixedFullGroupsReuseCanonicalUnitsByPhysicalProjection) {
    for (const bool reverse_groups : {false, true}) {
        SCOPED_TRACE(reverse_groups ? "reverse group declaration" : "original group declaration");
        auto config      = makeMixedFullCPConfig(/*logical_group_block_size=*/4, reverse_groups);
        auto coordinator = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
        coordinator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
        ASSERT_TRUE(coordinator->init());
        auto mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
        coordinator->setCPSlotMapper(mapper);

        const CacheKeysType keys{100, 101, 102, 103, 104, 105};
        auto                seed_resource = makeBatchRes(/*batch_size=*/1, config, keys);
        auto                seed_tokens   = makeTokens(/*batch_size=*/1, /*seq_length=*/24, /*seq_size_per_block=*/4);
        MallocInfo          seed_malloc{seed_resource, seed_tokens};
        seed_malloc.enable_device_cache = false;
        seed_malloc.reuse_cache         = false;
        ASSERT_TRUE(coordinator->malloc(seed_malloc).success);
        ASSERT_EQ(seed_resource->blocksNum(0, "rr"), 3);
        ASSERT_EQ(seed_resource->blocksNum(0, "logical"), 6);
        const auto seed_rr      = seed_resource->blocks(0, "rr");
        const auto seed_logical = seed_resource->blocks(0, "logical");

        coordinator->insertIntoCache(InsertInfo{seed_resource, seed_tokens, /*is_resident=*/false});
        coordinator->free(FreeInfo{seed_resource, seed_tokens});

        auto       hit_resource = makeBatchRes(/*batch_size=*/1, config, keys);
        auto       hit_tokens   = makeTokens(/*batch_size=*/1, /*seq_length=*/24, /*seq_size_per_block=*/4);
        MallocInfo hit_malloc{hit_resource, hit_tokens};
        hit_malloc.enable_device_cache = true;
        hit_malloc.reuse_cache         = true;
        const auto result              = coordinator->malloc(hit_malloc);
        ASSERT_TRUE(result.success);
        EXPECT_EQ(result.reuse_len, 2 * mapper->virtualBlockSize());

        const auto& hit_rr      = hit_resource->blocks(0, "rr");
        const auto& hit_logical = hit_resource->blocks(0, "logical");
        ASSERT_GE(hit_rr.size(), 2u);
        ASSERT_GE(hit_logical.size(), 4u);
        EXPECT_EQ(BlockIndicesType(hit_rr.begin(), hit_rr.begin() + 2),
                  BlockIndicesType(seed_rr.begin(), seed_rr.begin() + 2));
        EXPECT_EQ(BlockIndicesType(hit_logical.begin(), hit_logical.begin() + 4),
                  BlockIndicesType(seed_logical.begin(), seed_logical.begin() + 4));
        coordinator->free(FreeInfo{hit_resource, hit_tokens});
        EXPECT_EQ(coordinator->blockPool("rr")->requestRefBlocksNum(), 0u);
        EXPECT_EQ(coordinator->blockPool("logical")->requestRefBlocksNum(), 0u);
    }
}

TEST_F(CoordinatorCacheManagerCPShardTest, BeamForkCopiesRoundRobinTailUntilVirtualBlockBoundary) {
    const auto run_case = [](int previous_seq_len, bool expect_round_robin_copy) {
        auto config      = makeMixedFullCPConfig(/*logical_group_block_size=*/64,
                                            /*reverse_groups=*/false,
                                            /*cache_key_block_size=*/64);
        auto coordinator = std::make_shared<CoordinatorCacheManager>(config, AllocationType::HOST);
        ASSERT_TRUE(coordinator->init());
        coordinator->setCPSlotMapper(std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/64));

        const auto rr_blocks      = coordinator->blockPool("rr")->malloc(1);
        const auto logical_blocks = coordinator->blockPool("logical")->malloc(1);
        ASSERT_EQ(rr_blocks.size(), 1u);
        ASSERT_EQ(logical_blocks.size(), 1u);

        auto resource = makeBatchRes(/*batch_size=*/1, config, CacheKeysType{100});
        resource->mutableBlockIds(0, "rr").assign(rr_blocks);
        resource->mutableBlockIds(0, "logical").assign(logical_blocks);

        std::vector<TaggedBlockIdPair> update_mapping;
        ASSERT_TRUE(coordinator->updateKVBlock(resource, /*block_src_batch=*/{0, 0}, previous_seq_len, update_mapping));

        if (expect_round_robin_copy) {
            ASSERT_EQ(update_mapping.size(), 1u);
            EXPECT_EQ(update_mapping.front().tag, "rr");
            EXPECT_EQ(update_mapping.front().src, rr_blocks.front());
            EXPECT_NE(update_mapping.front().dst, rr_blocks.front());
            EXPECT_NE(resource->blocks(0, "rr").back(), rr_blocks.front());
        } else {
            EXPECT_TRUE(update_mapping.empty());
            EXPECT_EQ(resource->blocks(0, "rr").back(), rr_blocks.front());
        }
        EXPECT_EQ(resource->blocks(1, "rr").back(), rr_blocks.front());
        EXPECT_EQ(resource->blocks(0, "logical").back(), logical_blocks.front());
        EXPECT_EQ(resource->blocks(1, "logical").back(), logical_blocks.front());

        coordinator->free(FreeInfo{resource, nullptr});
        EXPECT_EQ(coordinator->blockPool("rr")->requestRefBlocksNum(), 0u);
        EXPECT_EQ(coordinator->blockPool("logical")->requestRefBlocksNum(), 0u);
    };

    // The rank-local RR slot spans 64 * CP=2 global tokens and remains writable at token 64.
    run_case(/*previous_seq_len=*/64, /*expect_round_robin_copy=*/true);
    run_case(/*previous_seq_len=*/128, /*expect_round_robin_copy=*/false);
}

TEST_F(CoordinatorCacheManagerCPShardTest, MissingLogicalPhysicalBlockTruncatesCanonicalReuseUnit) {
    auto config      = makeMixedFullCPConfig();
    auto coordinator = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    coordinator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(coordinator->init());
    auto mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
    coordinator->setCPSlotMapper(mapper);

    const auto rr_blocks =
        seedCache(coordinator->blockPool("rr"), coordinator->sharedBlockCache(), "rr", CacheKeysType{101, 103});
    const auto logical_blocks = seedCache(
        coordinator->blockPool("logical"), coordinator->sharedBlockCache(), "logical", CacheKeysType{100, 101, 103});
    auto       resource = makeBatchRes(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103, 104, 105});
    auto       tokens   = makeTokens(/*batch_size=*/1, /*seq_length=*/24, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, tokens};
    malloc_info.enable_device_cache = true;
    malloc_info.reuse_cache         = true;
    const auto result               = coordinator->malloc(malloc_info);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, mapper->virtualBlockSize());
    ASSERT_GE(resource->blocks(0, "rr").size(), 1u);
    ASSERT_GE(resource->blocks(0, "logical").size(), 2u);
    EXPECT_EQ(resource->blocks(0, "rr")[0], rr_blocks[0]);
    EXPECT_EQ(resource->blocks(0, "logical")[0], logical_blocks[0]);
    EXPECT_EQ(resource->blocks(0, "logical")[1], logical_blocks[1]);
    coordinator->free(FreeInfo{resource, tokens});
    EXPECT_EQ(coordinator->blockPool("rr")->requestRefBlocksNum(), 0u);
    EXPECT_EQ(coordinator->blockPool("logical")->requestRefBlocksNum(), 0u);
}

TEST_F(CoordinatorCacheManagerCPShardTest, IncompleteFirstCanonicalUnitLeavesNoMatchedReferences) {
    auto config      = makeMixedFullCPConfig();
    auto coordinator = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    coordinator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(coordinator->init());
    auto mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
    coordinator->setCPSlotMapper(mapper);

    seedCache(coordinator->blockPool("rr"), coordinator->sharedBlockCache(), "rr", CacheKeysType{101});
    seedCache(coordinator->blockPool("logical"), coordinator->sharedBlockCache(), "logical", CacheKeysType{100});
    auto       resource = makeBatchRes(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103, 104, 105});
    auto       tokens   = makeTokens(/*batch_size=*/1, /*seq_length=*/24, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, tokens};
    malloc_info.enable_device_cache = true;
    malloc_info.reuse_cache         = true;
    const auto result               = coordinator->malloc(malloc_info);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(coordinator->blockPool("rr")->requestRefBlocksNum(), resource->blocksNum(0, "rr"));
    EXPECT_EQ(coordinator->blockPool("logical")->requestRefBlocksNum(), resource->blocksNum(0, "logical"));
    coordinator->free(FreeInfo{resource, tokens});
    EXPECT_EQ(coordinator->blockPool("rr")->requestRefBlocksNum(), 0u);
    EXPECT_EQ(coordinator->blockPool("logical")->requestRefBlocksNum(), 0u);
}

TEST_F(CoordinatorCacheManagerCPShardTest, LogicalPhysicalBlockCanSpanMultipleCacheKeyBlocks) {
    auto config      = makeMixedFullCPConfig(/*logical_group_block_size=*/8);
    auto coordinator = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    coordinator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(coordinator->init());
    auto mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
    coordinator->setCPSlotMapper(mapper);

    const CacheKeysType keys{100, 101, 102, 103, 104, 105};
    auto                seed_resource = makeBatchRes(/*batch_size=*/1, config, keys);
    auto                seed_tokens   = makeTokens(/*batch_size=*/1, /*seq_length=*/24, /*seq_size_per_block=*/4);
    MallocInfo          seed_malloc{seed_resource, seed_tokens};
    seed_malloc.enable_device_cache = false;
    seed_malloc.reuse_cache         = false;
    ASSERT_TRUE(coordinator->malloc(seed_malloc).success);
    const auto seed_rr      = seed_resource->blocks(0, "rr");
    const auto seed_logical = seed_resource->blocks(0, "logical");
    coordinator->insertIntoCache(InsertInfo{seed_resource, seed_tokens, /*is_resident=*/false});
    coordinator->free(FreeInfo{seed_resource, seed_tokens});
    EXPECT_EQ(coordinator->sharedBlockCache()->matchGroup(101, "logical"), seed_logical[0]);
    EXPECT_EQ(coordinator->sharedBlockCache()->matchGroup(103, "logical"), seed_logical[1]);

    auto       hit_resource = makeBatchRes(/*batch_size=*/1, config, keys);
    auto       hit_tokens   = makeTokens(/*batch_size=*/1, /*seq_length=*/24, /*seq_size_per_block=*/4);
    MallocInfo hit_malloc{hit_resource, hit_tokens};
    hit_malloc.enable_device_cache = true;
    hit_malloc.reuse_cache         = true;
    const auto result              = coordinator->malloc(hit_malloc);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 2 * mapper->virtualBlockSize());
    ASSERT_GE(hit_resource->blocks(0, "rr").size(), 2u);
    ASSERT_GE(hit_resource->blocks(0, "logical").size(), 2u);
    EXPECT_EQ(hit_resource->blocks(0, "rr")[0], seed_rr[0]);
    EXPECT_EQ(hit_resource->blocks(0, "rr")[1], seed_rr[1]);
    EXPECT_EQ(hit_resource->blocks(0, "logical")[0], seed_logical[0]);
    EXPECT_EQ(hit_resource->blocks(0, "logical")[1], seed_logical[1]);
}

TEST_F(CoordinatorCacheManagerCPShardTest, LogicalReuseStopsAtCompletePhysicalBlockBoundary) {
    auto config      = makeLogicalFullConfig(/*physical_block_size=*/8);
    auto coordinator = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    coordinator->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(coordinator->init());

    const auto cached_blocks = seedCache(
        coordinator->blockPool("logical"), coordinator->sharedBlockCache(), "logical", CacheKeysType{101, 103, 105});
    auto       resource = makeBatchRes(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103, 104, 105});
    auto       tokens   = makeTokens(/*batch_size=*/1, /*seq_length=*/24, /*seq_size_per_block=*/4);
    MallocInfo malloc_info{resource, tokens};
    malloc_info.enable_device_cache = true;
    malloc_info.reuse_cache         = true;
    const auto result               = coordinator->malloc(malloc_info);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 16);
    EXPECT_EQ(resource->cacheResource(0).deviceReuseBlockNum(), 4u);
    EXPECT_EQ(resource->cacheResource(0).deviceReuseBlockNum() * config.seq_size_per_block,
              static_cast<size_t>(result.reuse_len));
    const auto& blocks = resource->blocks(0, "logical");
    ASSERT_EQ(blocks.size(), 3u);
    EXPECT_EQ(blocks[0], cached_blocks[0]);
    EXPECT_EQ(blocks[1], cached_blocks[1]);
    EXPECT_NE(blocks[2], cached_blocks[2]);
    EXPECT_EQ(coordinator->blockPool("logical")->requestRefBlocksNum(), blocks.size());
    coordinator->free(FreeInfo{resource, tokens});
    EXPECT_EQ(coordinator->blockPool("logical")->requestRefBlocksNum(), 0u);
    EXPECT_EQ(coordinator->sharedBlockCache()->matchGroup(105, "logical"), cached_blocks[2]);
}

// 4) When reuse is disabled, cp_slot_mapper still translates seq_len for malloc and skips the match.
TEST_F(CoordinatorCacheManagerCPShardTest, ShardedAllocSkipsReuseWhenDisabled) {
    auto config                    = makeCPHybridConfig();
    auto coordinator_cache_manager = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    coordinator_cache_manager->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(coordinator_cache_manager->init());

    auto shared_cache = coordinator_cache_manager->sharedBlockCache();

    seedCache(coordinator_cache_manager->blockPool("full"), shared_cache, "full", CacheKeysType{101});

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto tokens    = makeTokens(1, 16, 4);

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    coordinator_cache_manager->setCPSlotMapper(std::make_shared<CPSlotMapper>(0, 2, 4));
    auto result = coordinator_cache_manager->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 2);
}

// 5) insertIntoCache uses last-rank canonical keys and virtualBlockSize when sharded:
//    a 12-token request (full_blocks_num = floor(12/8)=1 virtual block) inserts only key {103}
//    (= last-rank canonical key at index cp_size-1=1 of the first virtual block window).
TEST_F(CoordinatorCacheManagerCPShardTest, InsertIntoCacheUsesCanonicalKeysAndVirtualBlockSize) {
    auto config                    = makeCPHybridConfig();
    auto coordinator_cache_manager = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    coordinator_cache_manager->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(coordinator_cache_manager->init());

    auto shared_cache = coordinator_cache_manager->sharedBlockCache();
    ASSERT_NE(shared_cache, nullptr);

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});

    // seq_len=16 => coordinator_cache_manager computes 4 logical blocks; cp_size=2 keeps 2 per rank.
    auto       tokens = makeTokens(1, 16, 4);
    MallocInfo malloc_info{batch_res, tokens};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    coordinator_cache_manager->setCPSlotMapper(std::make_shared<CPSlotMapper>(0, 2, 4));
    ASSERT_TRUE(coordinator_cache_manager->malloc(malloc_info).success);
    ASSERT_EQ(batch_res->blocksNum(0, "full"), 2);

    // CompleteTokenIds reflects token-len 16, so token_len-1 = 15. virtualBlockSize=8 =>
    // full_blocks_num = floor(15/8) = 1. n = min(local_keys.size()=2, 1) = 1.
    // local_keys = {101, 103}; first key is 101.
    InsertInfo insert_info{batch_res, tokens, /*is_resident=*/false};
    coordinator_cache_manager->insertIntoCache(insert_info);

    EXPECT_FALSE(isNullBlockIdx(shared_cache->matchGroup(101, "full")));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(100, "full")));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(102, "full")));
    EXPECT_TRUE(isNullBlockIdx(shared_cache->matchGroup(103, "full")));
}

// 6) Two-malloc smoke: cp_size=4 sharding, request occupies 8 logical blocks ⇒ 2 per rank.
TEST_F(CoordinatorCacheManagerCPShardTest, ShardedAllocCpSize4) {
    auto config                    = makeCPHybridConfig();
    auto coordinator_cache_manager = std::make_shared<CoordinatorCacheManager>(config, AllocationType::DEVICE);
    coordinator_cache_manager->setSharedBlockCache(std::make_shared<SharedBlockCache>());
    ASSERT_TRUE(coordinator_cache_manager->init());

    CacheKeysType keys;
    for (int i = 0; i < 8; ++i) {
        keys.push_back(200 + i);
    }
    auto batch_res = makeBatchRes(1, config, keys);
    auto tokens    = makeTokens(1, /*seq_len=*/32, 4);  // 8 logical blocks

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    coordinator_cache_manager->setCPSlotMapper(
        std::make_shared<CPSlotMapper>(/*cp_rank=*/2, /*cp_size=*/4, /*block_size=*/4));
    auto result = coordinator_cache_manager->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 2);  // ceil(8/4)=2
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
