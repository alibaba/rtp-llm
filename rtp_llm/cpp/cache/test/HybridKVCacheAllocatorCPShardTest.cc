// CP-shard (Stage 5, Plan A) UTs for HybridKVCacheAllocator.
//
// These exercise the cp_slot_mapper plumbing in initMallocForCommonLen,
// incrMalloc, insertIntoCache, and getNeedBlocks. The shape of the tests
// piggybacks on the helpers in HybridTypeKVCacheAllocatorTest.cc but
// keeps the configuration self-contained so the two files build cleanly
// alongside each other.

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/allocator/HybridTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"
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
    linear_spec->local_num_k_heads  = 1;
    linear_spec->local_num_v_heads  = 1;
    linear_spec->head_k_dim         = 1;
    linear_spec->head_v_dim         = 1;
    linear_spec->conv_kernel_dim    = 2;
    linear_spec->local_head_num_kv  = 1;
    linear_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);
    // These tests isolate CP-sharded paged-prefix behavior. Keep LINEAR in the
    // allocation topology, but exclude it from prefix reuse so the canonical
    // tree namespace is owned by the CP-shardable FULL group.
    linear_spec->skip_prefix_reuse = true;

    auto full_spec                = std::make_shared<MHAKVCacheSpec>();
    full_spec->type               = KVCacheSpecType::MultiHeadAttention;
    full_spec->dtype              = config.dtype;
    full_spec->local_head_num_kv  = 1;
    full_spec->size_per_head      = 1;
    full_spec->seq_size_per_block = static_cast<uint32_t>(config.seq_size_per_block);

    config.fromGroupedSpecs({linear_spec, full_spec},
                            {{0, 1}, {2, 3}},
                            {CacheGroupType::LINEAR, CacheGroupType::FULL},
                            {"linear", "full"});

    config.kv_block_stride_bytes = std::max(full_spec->block_size_bytes(), linear_spec->block_size_bytes());
    config.kv_block_size_bytes   = static_cast<size_t>(config.group_layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    const size_t per_layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(config.layer_all_num),
                                              static_cast<int>(per_layer_stride_bytes));

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
    res->initGroups(config.groupNums(),
                    static_cast<int>(config.layer_all_num),
                    config.layerGroupIdsSnapshot(),
                    config.kernelBlocksPerKvBlock(),
                    config.groupTypesSnapshot());
    for (int b = 0; b < batch_size; ++b) {
        res->setBatchCacheKeys(b, keys);
    }
    return res;
}

CacheKeysType canonicalCPKeys(const std::shared_ptr<CPSlotMapper>& mapper, const CacheKeysType& full_keys) {
    if (!mapper || !mapper->isSharded()) {
        return full_keys;
    }
    CacheKeysType canonical_keys;
    for (int i = mapper->cpSize() - 1; i < static_cast<int>(full_keys.size()); i += mapper->cpSize()) {
        canonical_keys.push_back(full_keys[static_cast<size_t>(i)]);
    }
    return canonical_keys;
}

struct CPRealCacheSeed {
    CacheKeysType                 canonical_keys;
    std::vector<BlockIndicesType> group_blocks;
};

CPRealCacheSeed seedCPRealCachePath(const std::shared_ptr<HybridTypeKVCacheAllocator>& allocator,
                                    const CacheConfig&                                config,
                                    const std::shared_ptr<CPSlotMapper>&               mapper,
                                    const CacheKeysType&                              full_keys,
                                    int                                               seq_length) {
    CPRealCacheSeed seed;
    EXPECT_EQ(allocator->cpSlotMapper(), mapper);
    seed.canonical_keys = canonicalCPKeys(mapper, full_keys);

    auto resource = makeBatchRes(/*batch_size=*/1, config, full_keys);
    auto tokens   = makeTokens(
        /*batch_size=*/1, seq_length, static_cast<int>(config.seq_size_per_block));
    MallocInfo malloc_info{resource, tokens};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    const auto result               = allocator->malloc(malloc_info);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.async_context, nullptr);
    if (!result.success) {
        return seed;
    }

    seed.group_blocks.reserve(static_cast<size_t>(resource->groupNums()));
    for (int gid = 0; gid < resource->groupNums(); ++gid) {
        seed.group_blocks.push_back(resource->blocks(/*batch_id=*/0, gid));
    }
    allocator->insertIntoCache(InsertInfo{resource, tokens, /*is_resident=*/false});
    allocator->free(FreeInfo{resource, tokens});
    EXPECT_EQ(resource->curBlocksNum(), 0u);
    return seed;
}

}  // namespace

class HybridKVCacheAllocatorCPShardTest: public ::testing::Test {
protected:
    void SetUp() override {
        rtp_llm::initLogger();
        createDevice();
    }

    void TearDown() override {
        allocator_.reset();
        block_tree_cache_.reset();
    }

    bool initWithBlockTreeCache(const CacheConfig& config, const std::shared_ptr<CPSlotMapper>& mapper = nullptr) {
        allocator_ = std::make_shared<HybridTypeKVCacheAllocator>(config, AllocationType::DEVICE);
        if (mapper) {
            allocator_->setCPSlotMapper(mapper);
        }
        if (!allocator_->init()) {
            return false;
        }

        const size_t total_before  = allocator_->totalBlocksNum();
        const size_t free_before   = allocator_->freeBlocksNum();
        const size_t active_before = allocator_->activeTreeCachedBlocksNum();
        KVCacheConfig kv_cache_config;
        block_tree_cache_ = createBlockTreeCache(config, kv_cache_config, allocator_);
        if (!block_tree_cache_) {
            return false;
        }
        allocator_->setBlockTreeCache(block_tree_cache_.get());
        return allocator_->blockTreeCache() == block_tree_cache_.get()
               && allocator_->totalBlocksNum() == total_before && allocator_->freeBlocksNum() == free_before
               && allocator_->activeTreeCachedBlocksNum() == active_before;
    }

    std::shared_ptr<HybridTypeKVCacheAllocator> allocator_;
    BlockTreeCachePtr                           block_tree_cache_;
};

// 1) When cp_slot_mapper is null/passthrough, behavior is identical to the non-CP baseline:
//    a request occupying 4 logical blocks allocates 4 blocks in the full group.
TEST_F(HybridKVCacheAllocatorCPShardTest, NullMapperIsPassthrough) {
    auto config = makeCPHybridConfig();
    ASSERT_TRUE(initWithBlockTreeCache(config));
    ASSERT_EQ(allocator_->blockTreeCache(), block_tree_cache_.get());

    const int gid_full  = 1;
    auto      batch_res = makeBatchRes(/*batch_size=*/1, config, CacheKeysType{100, 101, 102, 103});
    const size_t free_before = allocator_->freeBlocksNum();
    ASSERT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);
    // seq_len=16 => 4 slots @ block_size=4
    auto       tokens = makeTokens(/*batch=*/1, /*seq_len=*/16, /*sspb=*/4);
    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    // cp_slot_mapper intentionally left null.
    auto result = allocator_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_EQ(batch_res->blocksNum(0, gid_full), 4);

    FreeInfo free_info{batch_res, tokens};
    allocator_->free(free_info);
    EXPECT_EQ(batch_res->curBlocksNum(), 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);
}

// 2) With cp_slot_mapper(cp_rank=0, cp_size=2, block_size=4): a 4-block request allocates ceil(4/2)=2
//    physical blocks on this rank for the full group.
TEST_F(HybridKVCacheAllocatorCPShardTest, ShardedAllocHalvesFullGroup) {
    auto config = makeCPHybridConfig();
    auto mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
    ASSERT_TRUE(initWithBlockTreeCache(config, mapper));
    ASSERT_EQ(allocator_->blockTreeCache(), block_tree_cache_.get());

    const int gid_full  = 1;
    auto      batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto      tokens    = makeTokens(1, 16, 4);  // 4 logical blocks worth
    const size_t free_before = allocator_->freeBlocksNum();
    ASSERT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    auto result              = allocator_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_EQ(batch_res->blocksNum(0, gid_full), 2)
        << "cp_size=2 should halve allocation to ceil(4/2)=2 physical blocks per rank";

    FreeInfo free_info{batch_res, tokens};
    allocator_->free(free_info);
    EXPECT_EQ(batch_res->curBlocksNum(), 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);
}

TEST_F(HybridKVCacheAllocatorCPShardTest, ReuseHitOnLastRankCanonicalKey) {
    auto config = makeCPHybridConfig();
    auto mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
    ASSERT_TRUE(initWithBlockTreeCache(config, mapper));
    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator_->freeBlocksNum();
    const auto   seed =
        seedCPRealCachePath(allocator_, config, mapper, CacheKeysType{100, 101}, /*seq_length=*/8);
    ASSERT_EQ(seed.canonical_keys, (CacheKeysType{101}));
    ASSERT_EQ(seed.group_blocks.size(), 2u);
    ASSERT_EQ(seed.group_blocks[1].size(), 1u);
    ASSERT_FALSE(isNullBlockIdx(seed.group_blocks[1][0]));
    EXPECT_EQ(block_pool->refCount(seed.group_blocks[1][0]), 1u);

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto tokens    = makeTokens(1, 16, 4);

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = true;
    info.reuse_cache         = true;
    auto result              = allocator_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 8);
    EXPECT_EQ(result.async_context, nullptr);

    const auto& full_blocks = batch_res->blocks(/*batch_id=*/0, /*gid=*/1);
    ASSERT_EQ(full_blocks.size(), 2u);
    EXPECT_EQ(full_blocks[0], seed.group_blocks[1][0]);
    EXPECT_FALSE(isNullBlockIdx(full_blocks[1]));
    EXPECT_NE(full_blocks[1], seed.group_blocks[1][0]);
    EXPECT_EQ(block_pool->refCount(seed.group_blocks[1][0]), 2u);

    allocator_->free(FreeInfo{batch_res, tokens});
    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(block_pool->refCount(seed.group_blocks[1][0]), 1u);
    EXPECT_GT(allocator_->blockTreeCache()->reclaimBlocks(config.block_num, Tier::DEVICE), 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

TEST_F(HybridKVCacheAllocatorCPShardTest, ShardedAllocSkipsReuseWhenDisabled) {
    auto config = makeCPHybridConfig();
    auto mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
    ASSERT_TRUE(initWithBlockTreeCache(config, mapper));
    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator_->freeBlocksNum();
    const auto   seed =
        seedCPRealCachePath(allocator_, config, mapper, CacheKeysType{100, 101}, /*seq_length=*/8);
    ASSERT_EQ(seed.canonical_keys, (CacheKeysType{101}));
    ASSERT_EQ(seed.group_blocks.size(), 2u);
    ASSERT_EQ(seed.group_blocks[1].size(), 1u);

    auto batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto tokens    = makeTokens(1, 16, 4);

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    auto result              = allocator_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 0);
    EXPECT_EQ(result.async_context, nullptr);

    const auto& full_blocks = batch_res->blocks(/*batch_id=*/0, /*gid=*/1);
    ASSERT_EQ(full_blocks.size(), 2u);
    for (auto block : full_blocks) {
        EXPECT_FALSE(isNullBlockIdx(block));
        EXPECT_NE(block, seed.group_blocks[1][0]);
    }

    allocator_->free(FreeInfo{batch_res, tokens});
    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(block_pool->refCount(seed.group_blocks[1][0]), 1u);
    EXPECT_GT(allocator_->blockTreeCache()->reclaimBlocks(config.block_num, Tier::DEVICE), 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

TEST_F(HybridKVCacheAllocatorCPShardTest, InsertIntoCacheUsesCanonicalKeysAndVirtualBlockSize) {
    auto config = makeCPHybridConfig();
    auto mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/0, /*cp_size=*/2, /*block_size=*/4);
    ASSERT_TRUE(initWithBlockTreeCache(config, mapper));
    auto block_pool = allocator_->getDeviceBlockPool();
    ASSERT_NE(block_pool, nullptr);

    const size_t free_before = allocator_->freeBlocksNum();
    auto         batch_res = makeBatchRes(1, config, CacheKeysType{100, 101, 102, 103});
    auto         tokens    = makeTokens(1, 16, 4);
    MallocInfo malloc_info{batch_res, tokens};
    malloc_info.enable_device_cache = false;
    malloc_info.reuse_cache         = false;
    const auto malloc_result         = allocator_->malloc(malloc_info);
    ASSERT_TRUE(malloc_result.success);
    EXPECT_EQ(malloc_result.async_context, nullptr);
    ASSERT_EQ(batch_res->blocksNum(/*batch_id=*/0, /*gid=*/1), 2);
    const auto full_blocks = batch_res->blocks(/*batch_id=*/0, /*gid=*/1);

    allocator_->insertIntoCache(InsertInfo{batch_res, tokens, /*is_resident=*/false});
    allocator_->free(FreeInfo{batch_res, tokens});
    EXPECT_EQ(batch_res->curBlocksNum(), 0u);
    EXPECT_EQ(mapper->virtualBlockSize(), 8);
    EXPECT_EQ(canonicalCPKeys(mapper, CacheKeysType{100, 101, 102, 103}), (CacheKeysType{101, 103}));

    auto noncanonical_match = block_tree_cache_->match(CacheKeysType{100});
    EXPECT_EQ(noncanonical_match.matched_blocks, 0u);
    EXPECT_TRUE(noncanonical_match.matched_block_sets.empty());

    auto canonical_match = block_tree_cache_->match(CacheKeysType{101, 103});
    ASSERT_EQ(canonical_match.matched_blocks, 2u);
    ASSERT_EQ(canonical_match.group_block_indices.at(/*gid=*/1), full_blocks);
    block_tree_cache_->releaseMatchedBlocks(canonical_match.matched_block_sets);
    canonical_match.matched_block_sets.clear();
    for (auto block : full_blocks) {
        EXPECT_EQ(block_pool->refCount(block), 1u);
    }

    EXPECT_GT(allocator_->blockTreeCache()->reclaimBlocks(config.block_num, Tier::DEVICE), 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
}

// 6) Two-malloc smoke: cp_size=4 sharding, request occupies 8 logical blocks ⇒ 2 per rank.
TEST_F(HybridKVCacheAllocatorCPShardTest, ShardedAllocCpSize4) {
    auto config = makeCPHybridConfig();
    auto mapper = std::make_shared<CPSlotMapper>(/*cp_rank=*/2, /*cp_size=*/4, /*block_size=*/4);
    ASSERT_TRUE(initWithBlockTreeCache(config, mapper));
    ASSERT_EQ(allocator_->blockTreeCache(), block_tree_cache_.get());

    const int     gid_full = 1;
    CacheKeysType keys;
    for (int i = 0; i < 8; ++i) {
        keys.push_back(200 + i);
    }
    auto batch_res = makeBatchRes(1, config, keys);
    auto tokens    = makeTokens(1, /*seq_len=*/32, 4);  // 8 logical blocks
    const size_t free_before = allocator_->freeBlocksNum();
    ASSERT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);

    MallocInfo info{batch_res, tokens};
    info.enable_device_cache = false;
    info.reuse_cache         = false;
    auto result              = allocator_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.async_context, nullptr);
    EXPECT_EQ(batch_res->blocksNum(0, gid_full), 2);  // ceil(8/4)=2

    FreeInfo free_info{batch_res, tokens};
    allocator_->free(free_info);
    EXPECT_EQ(batch_res->curBlocksNum(), 0);
    EXPECT_EQ(allocator_->freeBlocksNum(), free_before);
    EXPECT_EQ(allocator_->activeTreeCachedBlocksNum(), 0u);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
