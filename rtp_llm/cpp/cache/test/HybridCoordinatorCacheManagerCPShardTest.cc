// CP-shard request and storage-layout UTs for CoordinatorCacheManager.
//
// These exercise cache-owned mappings in initMallocForCommonLen,
// incrMalloc, insertIntoCache, and getNeedBlocks. The shape of the tests
// piggybacks on the helpers in HybridTypeCoordinatorCacheManagerTest.cc but
// keeps the configuration self-contained so the two files build cleanly
// alongside each other.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace test {

using TestHybridTypeCoordinatorCacheManager = BlockTreeCacheTestAllocator<CoordinatorCacheManager>;

namespace {

// Two-group hybrid: group_id=0 linear (won't be exercised here), group_id=1 full (the CP-shard target).
CacheConfig makeCPHybridConfig(uint32_t kernel_block_size = 2) {
    CacheConfig config;
    config.dtype              = rtp_llm::DataType::TYPE_FP16;
    config.layer_num          = 4;
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
    full_spec->kernel_seq_size_per_block = kernel_block_size;

    auto linear_policy = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    auto full_policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    full_policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    config.fromGroupedSpecs({linear_spec, full_spec},
                            {{0, 1}, {2, 3}},
                            {CacheGroupType::LINEAR, CacheGroupType::FULL},
                            {"linear", "full"},
                            {linear_policy, full_policy});

    config.finalizeBlockNums(32, RuntimeConfig{});  // headroom for cp_size=2 expansion
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

void resolveTestCpLayout(CacheConfig& config, int cp_size, int cp_rank = 0) {
    ParallelismConfig parallelism;
    parallelism.tp_rank = cp_rank;
    parallelism.tp_size = cp_size;
    parallelism.prefill_cp_config.kv_cache_sharded = cp_size > 1;
    config.finalizeStorageLayout(parallelism);
}

std::vector<size_t> canonicalPositions(const CacheConfig& config, const std::string& tag, size_t key_count) {
    std::vector<size_t> positions;
    positions.reserve(key_count);
    for (size_t index = 0; index < key_count; ++index) {
        positions.push_back(config.canonicalBlockPosition(tag, index));
    }
    return positions;
}

}  // namespace

class HybridCoordinatorCacheManagerCPShardTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }
};

// 1) Without a sharded request projection, behavior follows the non-CP baseline:
//    a request occupying 4 logical blocks allocates 4 blocks in the full group.
TEST_F(HybridCoordinatorCacheManagerCPShardTest, OrdinaryRequestIsPassthrough) {
    auto config    = makeCPHybridConfig();
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchRes(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    // seq_len=16 => 4 resources @ block_size=4
    auto       tokens = makeTokens(/*batch=*/1, /*seq_len=*/16, /*sspb=*/4);
    MallocInfo info{batch_res, tokens};
    info.enable_cache_lookup = false;
    info.reuse_cache         = false;
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 4);
    EXPECT_EQ(result.block_aligned_input_length, 16);
}

// 2) With CP2 storage at B=4, a 4-block request allocates ceil(4/2)=2
//    physical blocks on this rank for the full group.
TEST_F(HybridCoordinatorCacheManagerCPShardTest, ShardedAllocHalvesFullGroup) {
    auto config    = makeCPHybridConfig();
    resolveTestCpLayout(config, 2);
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto tokens    = makeTokens(1, 16, 4);  // 4 logical blocks worth

    MallocInfo info{batch_res, tokens};
    info.enable_cache_lookup = false;
    info.reuse_cache         = false;
    info.request_layout      = snapshotGlobalCacheRequest(*batch_res, *tokens);
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 2)
        << "cp_size=2 should halve allocation to ceil(4/2)=2 physical blocks per rank";
    EXPECT_EQ(batch_res->cacheKeys(0), (CacheKeysType{100, 101, 102, 103}));
    EXPECT_EQ(result.block_aligned_input_length, 16);
}

TEST_F(HybridCoordinatorCacheManagerCPShardTest, ShardedConfigRequiresGlobalSnapshotAndDerivesInsertKeys) {
    auto config = makeCPHybridConfig();
    resolveTestCpLayout(config, 2);
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    auto resource = makeBatchRes(1, config, CacheKeysType{100, 101});
    auto tokens = makeTokens(1, 8, 4);
    MallocInfo info{resource, tokens};
    info.enable_cache_lookup = false;
    EXPECT_ANY_THROW((void)allocator->malloc(info));
    EXPECT_EQ(resource->blocksNum(0, "full"), 0u);

    info.request_layout = snapshotGlobalCacheRequest(*resource, *tokens);
    ASSERT_TRUE(allocator->malloc(info).success);
    InsertInfo insert_info{resource, tokens, /*is_resident=*/false};
    size_t     resident_prefix_length = 0;
    EXPECT_NO_THROW(allocator->insertIntoCache(insert_info, resident_prefix_length));
    allocator->free(FreeInfo{resource, tokens});
}

// 3) Reuse path with CP canonical keys: reuse_len must stay in whole logical blocks
//    strictly below the query length, while every complete logical prefix block below
//    that bound is still reused. Each case seeds the full canonical chain the request
//    could match, so a lookup that keeps too many keys is visible here.
void expectCpCanonicalReuse(int                  seq_len,
                            const CacheKeysType& request_keys,
                            const CacheKeysType& cached_canonical_keys,
                            int                  expected_reuse_len,
                            int                  expected_local_blocks,
                            int                  expected_input_length,
                            int                  cp_size = 2) {
    auto config    = makeCPHybridConfig();
    resolveTestCpLayout(config, cp_size);
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());
    ASSERT_TRUE(seedCompleteBlockTreePath(allocator, cached_canonical_keys).success);

    auto batch_res = makeBatchRes(1, config, request_keys);
    auto tokens    = makeTokens(1, seq_len, 4);

    MallocInfo info{batch_res, tokens};
    info.enable_cache_lookup = true;
    info.reuse_cache         = true;
    info.request_layout      = snapshotGlobalCacheRequest(*batch_res, *tokens);
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);

    EXPECT_EQ(result.reuse_len, expected_reuse_len) << "seq_len=" << seq_len;
    EXPECT_EQ(result.query_length, seq_len) << "global N must not use rank-local block count";
    EXPECT_EQ(result.full_key_hit_length, expected_reuse_len) << "H uses the canonical, N-1-clipped key path";
    EXPECT_EQ(batch_res->cacheKeys(0), request_keys);
    EXPECT_EQ(result.block_aligned_input_length, expected_input_length) << "seq_len=" << seq_len;
    EXPECT_EQ(batch_res->cacheResource(0).reuseBlockNum(), static_cast<size_t>(expected_reuse_len / (4 * cp_size)))
        << "seq_len=" << seq_len;
    EXPECT_EQ(batch_res->blocksNum(0, "full"), expected_local_blocks) << "seq_len=" << seq_len;
}

TEST_F(HybridCoordinatorCacheManagerCPShardTest, ReuseOnCanonicalKeysStaysBelowQueryLength) {
    // block_size=4, cp_size=2 => one canonical key covers 8 tokens.
    expectCpCanonicalReuse(
        16, {100, 101, 102, 103}, {101, 103}, /*reuse_len=*/8, /*local_blocks=*/2, /*input_length=*/16);
    expectCpCanonicalReuse(12, {100, 101, 102}, {101}, /*reuse_len=*/8, /*local_blocks=*/2, /*input_length=*/12);
    expectCpCanonicalReuse(8, {100, 101}, {101}, /*reuse_len=*/0, /*local_blocks=*/1, /*input_length=*/8);
    expectCpCanonicalReuse(
        17, {100, 101, 102, 103, 104}, {101, 103}, /*reuse_len=*/16, /*local_blocks=*/3, /*input_length=*/20);
}

TEST_F(HybridCoordinatorCacheManagerCPShardTest, CpFourCanonicalQueryKeepsLastRankKeyAndExcludesLastToken) {
    // At P=4 the only canonical key among five B=4 pages is key 103. A
    // 16-token request may not reuse all its tokens; at 20 tokens it may use
    // that first 16-token bundle, while key 104 is never a canonical key.
    expectCpCanonicalReuse(16,
                           {100, 101, 102, 103},
                           {103},
                           /*reuse_len=*/0,
                           /*local_blocks=*/1,
                           /*input_length=*/16,
                           /*cp_size=*/4);
    expectCpCanonicalReuse(20,
                           {100, 101, 102, 103, 104},
                           {103},
                           /*reuse_len=*/16,
                           /*local_blocks=*/2,
                           /*input_length=*/20,
                           /*cp_size=*/4);
}

// 4) When reuse is disabled, the request projection still sizes malloc and skips the match.
TEST_F(HybridCoordinatorCacheManagerCPShardTest, ShardedAllocSkipsReuseWhenDisabled) {
    auto config    = makeCPHybridConfig();
    resolveTestCpLayout(config, 2);
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    const auto seeded = seedCompleteBlockTreePath(allocator, CacheKeysType{101});
    ASSERT_TRUE(seeded.success);

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto tokens    = makeTokens(1, 16, 4);

    MallocInfo info{batch_res, tokens};
    info.enable_cache_lookup = false;
    info.reuse_cache         = false;
    info.request_layout      = snapshotGlobalCacheRequest(*batch_res, *tokens);
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 2);
}

// 5) insertIntoCache publishes the allocator's complete rank-local blocks on
//    the last-rank canonical key chain. KVCacheManager owns partial-tail
//    trimming before this allocator boundary.
TEST_F(HybridCoordinatorCacheManagerCPShardTest, InsertIntoCacheUsesCanonicalKeysAndVirtualBlockSize) {
    auto config    = makeCPHybridConfig();
    resolveTestCpLayout(config, 2);
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    batch_res->setLastBlockAligned(true);

    // seq_len=16 => allocator computes 4 logical blocks; cp_size=2 keeps 2 per rank.
    auto       tokens = makeTokens(1, 16, 4);
    MallocInfo malloc_info{batch_res, tokens};
    malloc_info.enable_cache_lookup = false;
    malloc_info.reuse_cache         = false;
    malloc_info.request_layout      = snapshotGlobalCacheRequest(*batch_res, *tokens);
    ASSERT_TRUE(allocator->malloc(malloc_info).success);
    ASSERT_EQ(batch_res->blocksNum(0, "full"), 2);
    const auto full_blocks = batch_res->blocks(0, "full");

    InsertInfo insert_info{batch_res, tokens, /*is_resident=*/false};
    {
        size_t resident_prefix_length = 0;
        allocator->insertIntoCache(insert_info, resident_prefix_length);
    }

    const auto snapshot = allocator->blockTreeCacheOwner()->getKeySnapshot();
    EXPECT_EQ(snapshot.keys, (CacheKeysType{101, 103}));

    auto match = allocator->blockTreeCacheOwner()->match(CacheKeysType{101, 103});
    ASSERT_EQ(match.matched_device_blocks, 2u);
    ASSERT_EQ(allocator->blockTreeCacheOwner()->matchedBlocksForGroup("full", match.matched_device_resources),
              full_blocks);
    block_tree_cache_test::releaseRequestRefsForTest(*allocator->blockTreeCacheOwner(), match.matched_device_resources);
}

// 6) Two-malloc smoke: cp_size=4 sharding, request occupies 8 logical blocks ⇒ 2 per rank.
TEST_F(HybridCoordinatorCacheManagerCPShardTest, ShardedAllocCpSize4) {
    auto config    = makeCPHybridConfig();
    resolveTestCpLayout(config, 4, 2);
    auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
    ASSERT_TRUE(allocator->init());

    CacheKeysType keys;
    for (int i = 0; i < 8; ++i) {
        keys.push_back(200 + i);
    }
    auto batch_res = makeBatchRes(1, config, keys);
    auto tokens    = makeTokens(1, /*seq_len=*/32, 4);  // 8 logical blocks

    MallocInfo info{batch_res, tokens};
    info.enable_cache_lookup = false;
    info.reuse_cache         = false;
    info.request_layout      = snapshotGlobalCacheRequest(*batch_res, *tokens);
    auto result = allocator->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(batch_res->blocksNum(0, "full"), 2);  // ceil(8/4)=2
}

// Frozen request-boundary vector: B is four tokens while K may be two or four.
// Five logical pages leave an incomplete CP bundle at P=2 and P=4. Every
// rank still receives the same number of local slots; the final slot is not
// evidence that every rank owns valid payload in the incomplete bundle.
TEST_F(HybridCoordinatorCacheManagerCPShardTest, PartialBundleAllocationAndReleaseIgnoreKernelPageSize) {
    for (uint32_t kernel_block_size : {2u, 4u}) {
        for (int cp_size : {1, 2, 4}) {
            auto config    = makeCPHybridConfig(kernel_block_size);
            resolveTestCpLayout(config, cp_size);
            auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
            ASSERT_TRUE(allocator->init());

            auto       batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103, 104});
            auto       tokens    = makeTokens(1, /*seq_length=*/20, /*seq_size_per_block=*/4);
            MallocInfo info{batch_res, tokens};
            info.enable_cache_lookup = false;
            info.reuse_cache         = false;
            info.request_layout      = snapshotGlobalCacheRequest(*batch_res, *tokens);
            const size_t free_before = allocator->freeBlocksNum();

            const auto result = allocator->malloc(info);
            ASSERT_TRUE(result.success) << "P=" << cp_size << " K=" << kernel_block_size;
            const size_t expected_slots = cp_size == 1 ? 5u : (cp_size == 2 ? 3u : 2u);
            EXPECT_EQ(batch_res->blocksNum(0, "full"), expected_slots);
            EXPECT_EQ(result.reuse_len, 0);
            EXPECT_EQ(result.block_aligned_input_length, 20);

            allocator->free(FreeInfo{batch_res, tokens});
            EXPECT_EQ(allocator->freeBlocksNum(), free_before) << "P=" << cp_size << " K=" << kernel_block_size;
        }
    }
}

TEST_F(HybridCoordinatorCacheManagerCPShardTest, CacheLayoutAllocatesAndPublishesFrozenCpVectors) {
    for (uint32_t kernel_block_size : {2u, 4u}) {
        for (int cp_size : {1, 2, 4}) {
            auto config = makeCPHybridConfig(kernel_block_size);
            ParallelismConfig parallelism;
            parallelism.tp_size = cp_size;
            parallelism.prefill_cp_config.kv_cache_sharded = true;
            config.finalizeStorageLayout(parallelism);

            auto allocator = std::make_shared<TestHybridTypeCoordinatorCacheManager>(config, AllocationType::DEVICE);
            ASSERT_TRUE(allocator->init());
            auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103, 104});
            auto tokens = makeTokens(1, 20, 4);
            MallocInfo info{batch_res, tokens};
            info.reuse_cache = false;
            info.enable_cache_lookup = false;
            info.request_layout      = snapshotGlobalCacheRequest(*batch_res, *tokens);
            const std::vector<size_t> expected_full_positions =
                cp_size == 1 ? std::vector<size_t>{0, 1, 2, 3, 4} :
                cp_size == 2 ? std::vector<size_t>{0, 1} : std::vector<size_t>{0};
            const std::vector<size_t> expected_linear_positions =
                cp_size == 1 ? std::vector<size_t>{0, 1, 2, 3, 4} :
                cp_size == 2 ? std::vector<size_t>{1, 3} : std::vector<size_t>{3};
            const auto match_key_count = config.canonicalCacheKeys(batch_res->cacheKeys(0)).size();
            EXPECT_EQ(canonicalPositions(config, "full", match_key_count), expected_full_positions);
            EXPECT_EQ(canonicalPositions(config, "linear", match_key_count), expected_linear_positions);
            auto insert_keys = batch_res->cacheKeys(0);
            insert_keys.pop_back();
            const auto insert_key_count = config.canonicalCacheKeys(insert_keys).size();
            // This resource has an incomplete final page: insertion drops it
            // before taking the CP canonical keys.
            const std::vector<size_t> expected_insert_full =
                cp_size == 1 ? std::vector<size_t>{0, 1, 2, 3} :
                cp_size == 2 ? std::vector<size_t>{0, 1} : std::vector<size_t>{0};
            const std::vector<size_t> expected_insert_linear =
                cp_size == 1 ? std::vector<size_t>{0, 1, 2, 3} :
                cp_size == 2 ? std::vector<size_t>{1, 3} : std::vector<size_t>{3};
            EXPECT_EQ(canonicalPositions(config, "full", insert_key_count), expected_insert_full);
            EXPECT_EQ(canonicalPositions(config, "linear", insert_key_count), expected_insert_linear);

            const size_t free_before = allocator->freeBlocksNum();
            ASSERT_TRUE(allocator->malloc(info).success) << "P=" << cp_size << " K=" << kernel_block_size;
            EXPECT_EQ(batch_res->blocksNum(0, "full"), cp_size == 1 ? 5 : (cp_size == 2 ? 3 : 2));
            EXPECT_EQ(batch_res->cacheKeys(0), (CacheKeysType{100, 101, 102, 103, 104}));
            EXPECT_EQ(config.topology().group("full").seqSizePerBlock(), 4u);
            EXPECT_EQ(config.topology().group("full").kernelSeqSizePerBlock(), kernel_block_size);
            EXPECT_EQ(config.topology().group("full").cacheKeyTokenStride(), static_cast<size_t>(4 * cp_size));

            if (cp_size == 2) {
                InsertInfo insert_info{batch_res, tokens, false};
                size_t resident_prefix_length = 0;
                allocator->insertIntoCache(insert_info, resident_prefix_length);
                EXPECT_EQ(allocator->blockTreeCacheOwner()->getKeySnapshot().keys, (CacheKeysType{101, 103}));
            }
            allocator->free(FreeInfo{batch_res, tokens});
            EXPECT_EQ(allocator->freeBlocksNum(), free_before - (cp_size == 2 ? 2u : 0u));
        }
    }
}

TEST_F(HybridCoordinatorCacheManagerCPShardTest, CacheLayoutPreservesNonRoundRobinFullGroup) {
    auto config = makeCPHybridConfig();
    std::vector<CacheGroupPolicy> policies;
    for (const auto& group : config.topology().groups()) {
        auto policy = group.policy;
        if (policy.group_type == CacheGroupType::FULL) {
            policy.cp_mapping = CpBlockMappingMode::NONE;
        }
        policies.push_back(policy);
    }
    setTestGroupPolicies(config, policies);
    ParallelismConfig parallelism;
    parallelism.tp_size = 2;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    config.finalizeStorageLayout(parallelism);
    auto resource = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103, 104});
    auto tokens = makeTokens(1, 20, 4);
    const auto request  = snapshotGlobalCacheRequest(*resource, *tokens);

    EXPECT_EQ(config.localAllocTokens("full", 20), 20);
    EXPECT_EQ(config.localAllocTokens("linear", request->request_incr_seq_len), 20);
    EXPECT_EQ(canonicalPositions(config, "linear", config.canonicalCacheKeys(request->match_keys).size()),
              (std::vector<size_t>{1, 3}));
    EXPECT_EQ(config.reuse_unit_tokens, 4);
    EXPECT_EQ(config.canonicalCacheKeys(request->match_keys), (CacheKeysType{101, 103}));
    EXPECT_EQ(config.topology().group("full").cacheKeyTokenStride(), 8u);
    EXPECT_EQ(config.topology().group("full").seqSizePerBlock(), 4u);
}

TEST_F(HybridCoordinatorCacheManagerCPShardTest, CacheLayoutPreservesNoncompactSwaSparsePositions) {
    CacheConfig config;
    config.dtype = DataType::TYPE_FP16;
    config.layer_num = 2;
    config.seq_size_per_block = 4;
    auto swa_spec = makeResolvedMhaSpec(config.dtype, 1, 1, 4, "swa");
    auto full_spec = makeResolvedMhaSpec(config.dtype, 1, 1, 4, "full");
    auto swa_policy = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_policy.cp_mapping = CpBlockMappingMode::NONE;
    swa_policy.sliding_window_size = 16;
    config.fromGroupedSpecs({swa_spec, full_spec},
                            {{0}, {1}},
                            {CacheGroupType::SWA, CacheGroupType::FULL},
                            {"swa", "full"},
                            {swa_policy, defaultCacheGroupPolicy(CacheGroupType::FULL)});
    ParallelismConfig parallelism;
    parallelism.tp_size = 4;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    config.finalizeStorageLayout(parallelism);
    auto resource = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103, 104, 105, 106, 107});
    auto tokens = makeTokens(1, 32, 4);
    const auto request         = snapshotGlobalCacheRequest(*resource, *tokens);
    const auto match_key_count = config.canonicalCacheKeys(request->match_keys).size();
    EXPECT_EQ(config.canonicalCacheKeys(request->match_keys), (CacheKeysType{103, 107}));
    EXPECT_EQ(canonicalPositions(config, "swa", match_key_count), (std::vector<size_t>{3, 7}));
    EXPECT_EQ(canonicalPositions(config, "full", match_key_count), (std::vector<size_t>{0, 1}));
    auto insert_keys = resource->cacheKeys(0);
    insert_keys.pop_back();
    EXPECT_EQ(canonicalPositions(config, "swa", config.canonicalCacheKeys(insert_keys).size()),
              (std::vector<size_t>{3}));
}

TEST_F(HybridCoordinatorCacheManagerCPShardTest, ExplicitSwaRoundRobinProjectsAllocationForEveryGroup) {
    CacheConfig config;
    config.dtype = DataType::TYPE_FP16;
    config.layer_num = 2;
    config.seq_size_per_block = 4;
    auto swa_spec = makeResolvedMhaSpec(config.dtype, 1, 1, 4, "swa");
    auto full_spec = makeResolvedMhaSpec(config.dtype, 1, 1, 4, "full");
    auto swa_policy = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    swa_policy.sliding_window_size = 16;
    config.fromGroupedSpecs({swa_spec, full_spec},
                            {{0}, {1}},
                            {CacheGroupType::SWA, CacheGroupType::FULL},
                            {"swa", "full"},
                            {swa_policy, defaultCacheGroupPolicy(CacheGroupType::FULL)});
    ParallelismConfig parallelism;
    parallelism.tp_size = 4;
    parallelism.prefill_cp_config.kv_cache_sharded = true;
    config.finalizeStorageLayout(parallelism);
    auto resource = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103, 104});
    auto tokens = makeTokens(1, 20, 4);
    const auto request  = snapshotGlobalCacheRequest(*resource, *tokens);
    // Five B=4 pages use two slots on every rank under P=4.
    EXPECT_EQ(config.localAllocTokens("swa", 20), 8);
    EXPECT_EQ(config.localAllocTokens("swa", request->request_incr_seq_len), 8);
    EXPECT_EQ(config.localAllocTokens("swa", request->request_common_seq_len), 8);
    EXPECT_EQ(config.canonicalCacheKeys(request->match_keys), (CacheKeysType{103}));
    EXPECT_EQ(canonicalPositions(config, "swa", config.canonicalCacheKeys(request->match_keys).size()),
              (std::vector<size_t>{3}));
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
