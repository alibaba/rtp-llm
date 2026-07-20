#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"

#include <memory>
#include <type_traits>

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/FullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

namespace rtp_llm {
namespace {

static_assert(
    std::is_same_v<decltype(BlockPoolConfigHelper::createConfig(std::declval<const CacheConfig&>())), BlockPoolConfig>);

BlockPoolConfig makeConfig() {
    constexpr int        kLayerNum       = 4;
    constexpr int        kBlockNum       = 10;
    constexpr size_t     kTokensPerBlock = 1;
    rtp_llm::CacheConfig cache_config    = rtp_llm::test::makeSimpleMhaCacheConfig(kLayerNum,
                                                                                kBlockNum,
                                                                                kTokensPerBlock,
                                                                                rtp_llm::TYPE_FP16,
                                                                                /*local_head_num_kv=*/1,
                                                                                /*size_per_head=*/64);
    auto                 config          = BlockPoolConfigHelper::createConfig(cache_config);
    config.pool_name                     = "device";
    return config;
}

BlockPoolConfig makeMixedScaleConfig() {
    constexpr int    kBlockNum       = 8;
    constexpr size_t kTokensPerBlock = 1;

    rtp_llm::CacheConfig scaled_cfg =
        rtp_llm::test::makeSimpleMhaCacheConfig(2, kBlockNum, kTokensPerBlock, rtp_llm::TYPE_INT8, 1, 64);
    rtp_llm::CacheConfig plain_cfg =
        rtp_llm::test::makeSimpleMhaCacheConfig(3, kBlockNum, kTokensPerBlock, rtp_llm::TYPE_FP16, 1, 64);

    BlockPoolConfig scaled_pool = BlockPoolConfigHelper::createConfig(scaled_cfg);
    BlockPoolConfig plain_pool  = BlockPoolConfigHelper::createConfig(plain_cfg);

    MemoryLayoutConfig l0    = scaled_pool.memory_layouts[0];
    MemoryLayoutConfig l1    = plain_pool.memory_layouts[0];
    l1.kv_cache_offset_bytes = l0.total_size_bytes + l1.kv_cache_offset_bytes;
    l1.kv_scale_offset_bytes = l0.total_size_bytes + l1.kv_scale_offset_bytes;

    BlockPoolConfig config;
    config.pool_name        = "mixed_scale_device";
    config.block_num        = l0.block_num;
    config.total_size_bytes = l0.total_size_bytes + l1.total_size_bytes;
    config.memory_layouts   = {l0, l1};
    return config;
}

BlockPoolPtr makeBackingPool(const BlockPoolConfig& config) {
    auto pool = std::make_shared<BlockPool>(config);
    RTP_LLM_CHECK(pool->init());
    return pool;
}

class MutationCountingFullGroup: public FullKVCacheGroup {
public:
    MutationCountingFullGroup(GroupBase cache_group, BlockPoolPtr block_pool):
        FullKVCacheGroup(std::move(cache_group), std::move(block_pool), /*group_id=*/0) {}

    bool malloc(BlockIds&, int, bool, int, std::vector<size_t>*) override {
        ++malloc_calls;
        return false;
    }

    void insertIntoCache(const CacheKeysType&, const BlockIndicesType&, bool) override {
        ++insert_calls;
    }

    void free(const BlockIndicesType&) override {
        ++free_calls;
    }

    size_t malloc_calls = 0;
    size_t insert_calls = 0;
    size_t free_calls   = 0;
};

std::shared_ptr<MutationCountingFullGroup>
makeAddressView(const BlockPoolPtr& backing, std::string tag, std::vector<int> global_layer_ids) {
    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                = tag;
    spec->seq_size_per_block = 1;

    GroupBase group;
    group.tag                       = std::move(tag);
    group.spec                      = spec;
    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.layer_ids                 = std::move(global_layer_ids);
    group.block_num                 = static_cast<uint32_t>(backing->totalBlocksNum() + 1);
    group.seq_size_per_block        = 1;
    group.kernel_seq_size_per_block = 1;

    auto address_view = std::make_shared<MutationCountingFullGroup>(std::move(group), backing);
    RTP_LLM_CHECK(address_view->init());
    return address_view;
}

}  // namespace

TEST(DeviceBlockPoolTest, InitKeepsBlockZeroInvalid) {
    auto            config = makeConfig();
    DeviceBlockPool pool(makeBackingPool(config));

    ASSERT_TRUE(pool.init());
    EXPECT_FALSE(pool.isAllocated(0));
    EXPECT_FALSE(pool.validBlock(0));
    EXPECT_EQ(pool.totalBlocksNum(), config.block_num - 1);

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_NE(*block, 0);
}

TEST(DeviceBlockPoolTest, ValidBlockUsesBackingUsableDomainBoundaries) {
    const auto      backing = makeBackingPool(makeConfig());
    DeviceBlockPool pool(backing);
    ASSERT_TRUE(pool.init());

    const BlockIdxType usable = static_cast<BlockIdxType>(backing->totalBlocksNum());
    ASSERT_GT(usable, 1);
    EXPECT_EQ(pool.totalBlocksNum(), backing->totalBlocksNum());
    EXPECT_EQ(pool.freeBlocksNum(), backing->freeBlocksNum());

    EXPECT_FALSE(pool.validBlock(static_cast<BlockIdxType>(-1)));
    EXPECT_FALSE(pool.validBlock(0));
    EXPECT_TRUE(pool.validBlock(1));
    EXPECT_TRUE(pool.validBlock(usable));
    EXPECT_FALSE(pool.validBlock(usable + 1));
}

TEST(DeviceBlockPoolTest, ConvertIndexToBufferReturnsBlockInfo) {
    auto            config = makeConfig();
    DeviceBlockPool pool(makeBackingPool(config));
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_NE(*block, 0);

    auto buffers = pool.convertIndexToBuffer(0, *block);
    ASSERT_FALSE(buffers.empty());
    for (const auto& buffer : buffers) {
        EXPECT_NE(buffer.addr, nullptr);
        EXPECT_GT(buffer.size_bytes, 0u);
    }
}

TEST(DeviceBlockPoolTest, PartitionedConvertIndexToBufferReturnsBlockInfo) {
    auto            config = makeConfig();
    DeviceBlockPool pool(makeBackingPool(config));
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());

    auto buffers = pool.convertIndexToBuffer(0, *block, 1, 0);
    ASSERT_FALSE(buffers.empty());
    for (const auto& buffer : buffers) {
        EXPECT_NE(buffer.addr, nullptr);
        EXPECT_GT(buffer.size_bytes, 0u);
    }
}

TEST(DeviceBlockPoolTest, LifecycleStartsAllocatedBlockWithZeroRefCount) {
    auto            config = makeConfig();
    DeviceBlockPool pool(makeBackingPool(config));
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 0u);

    pool.incRef(*block);
    pool.decRef(*block);
    EXPECT_FALSE(pool.isAllocated(*block));
}

TEST(DeviceBlockPoolTest, LifecycleUsesIBlockPoolSemantics) {
    auto            config = makeConfig();
    DeviceBlockPool pool(makeBackingPool(config));
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    EXPECT_TRUE(pool.isAllocated(*block));
    EXPECT_EQ(pool.refCount(*block), 0u);

    pool.incRef(*block);
    EXPECT_EQ(pool.refCount(*block), 1u);
    pool.decRef(*block);
    EXPECT_FALSE(pool.isAllocated(*block));
}

TEST(DeviceBlockPoolTest, ExposesAllocatorFacingLayerTensorsAndBuffers) {
    auto            config = makeConfig();
    DeviceBlockPool pool(makeBackingPool(config));
    ASSERT_TRUE(pool.init());

    auto cache_tensors = pool.allLayerCacheBase();
    ASSERT_FALSE(cache_tensors.empty());
    EXPECT_TRUE(cache_tensors[0].defined());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    auto addr = pool.convertIndexToAddr(/*layer_id=*/0, *block);
    EXPECT_NE(addr.kv_addr, nullptr);

    auto infos = pool.convertIndexToBuffer(/*layer_id=*/0, *block);
    ASSERT_FALSE(infos.empty());
    EXPECT_NE(infos[0].addr, nullptr);
    EXPECT_GT(infos[0].size_bytes, 0u);
    EXPECT_TRUE(infos[0].is_cuda);
}

TEST(DeviceBlockPoolTest, RegUserMrWithoutCacheStoreIsNoOp) {
    auto            config = makeConfig();
    DeviceBlockPool pool(makeBackingPool(config));
    ASSERT_TRUE(pool.init());

    // No cache store wired: MR registration must be a no-op, not a crash.
    pool.regUserMr(/*model_id=*/0, /*cache_store=*/nullptr);
    EXPECT_EQ(pool.getMrCostTimeMs(), 0);

    // Idempotent / safe to call again and to deregister when nothing was registered.
    pool.regUserMr(/*model_id=*/0, /*cache_store=*/nullptr);
    pool.deregUserMr();
    EXPECT_EQ(pool.getMrCostTimeMs(), 0);
}

TEST(DeviceBlockPoolTest, AllLayerScaleCacheBaseStaysAlignedWithPartialScale) {
    auto            config = makeMixedScaleConfig();
    DeviceBlockPool pool(makeBackingPool(config));
    ASSERT_TRUE(pool.init());

    const auto kv_bases    = pool.allLayerCacheBase();
    const auto scale_bases = pool.allLayerScaleCacheBase();

    ASSERT_EQ(kv_bases.size(), 5u);
    ASSERT_EQ(scale_bases.size(), 5u);

    for (const auto& kv : kv_bases) {
        EXPECT_TRUE(kv.defined());
        EXPECT_GT(kv.numel(), 0);
    }

    EXPECT_TRUE(scale_bases[0].defined());
    EXPECT_GT(scale_bases[0].numel(), 0);
    EXPECT_TRUE(scale_bases[1].defined());
    EXPECT_GT(scale_bases[1].numel(), 0);
    EXPECT_FALSE(scale_bases[2].defined() && scale_bases[2].numel() > 0);
    EXPECT_FALSE(scale_bases[3].defined() && scale_bases[3].numel() > 0);
    EXPECT_FALSE(scale_bases[4].defined() && scale_bases[4].numel() > 0);
}

TEST(DeviceBlockPoolTest, MallocAllocatesLowestBlockFirst) {
    auto            config = makeConfig();
    DeviceBlockPool pool(makeBackingPool(config));
    ASSERT_TRUE(pool.init());

    auto b1 = pool.malloc();
    auto b2 = pool.malloc();
    auto b3 = pool.malloc();
    ASSERT_TRUE(b1.has_value() && b2.has_value() && b3.has_value());
    EXPECT_EQ(*b1, 1);
    EXPECT_EQ(*b2, 2);
    EXPECT_EQ(*b3, 3);
}

TEST(DeviceBlockPoolTest, TreeHoldDoesNotDisturbRequestOrConnectorOwnership) {
    auto            backing = makeBackingPool(makeConfig());
    DeviceBlockPool pool(backing);
    ASSERT_TRUE(pool.init());

    const auto request_blocks = backing->malloc(1);
    ASSERT_EQ(request_blocks.size(), 1u);
    const BlockIdxType block = request_blocks[0];
    backing->connectorReference(block);
    EXPECT_EQ(backing->requestRefBlocksNum(), 1u);
    EXPECT_EQ(backing->connectorRefBlocksNum(), 1u);
    EXPECT_EQ(backing->blockCacheRefBlocksNum(), 0u);

    pool.incRef(block);
    EXPECT_EQ(backing->requestRefBlocksNum(), 1u);
    EXPECT_EQ(backing->connectorRefBlocksNum(), 1u);
    EXPECT_EQ(backing->blockCacheRefBlocksNum(), 1u);

    pool.decRef(block);
    EXPECT_EQ(backing->requestRefBlocksNum(), 1u);
    EXPECT_EQ(backing->connectorRefBlocksNum(), 1u);
    EXPECT_EQ(backing->blockCacheRefBlocksNum(), 0u);
    EXPECT_EQ(backing->freeBlocksNum(), backing->totalBlocksNum() - 1);

    backing->connectorFree(block);
    backing->requestFree(block);
    EXPECT_EQ(backing->freeBlocksNum(), backing->totalBlocksNum());
}

TEST(DeviceBlockPoolTest, ExternalAndSecondTreeHoldersDynamicallyControlCacheOnlyState) {
    auto            backing = makeBackingPool(makeConfig());
    DeviceBlockPool pool(backing);
    ASSERT_TRUE(pool.init());

    auto allocate_request_block = [&]() {
        const auto blocks = backing->malloc(1);
        EXPECT_EQ(blocks.size(), 1u);
        return blocks.empty() ? NULL_BLOCK_IDX : blocks.front();
    };

    const BlockIdxType request_held = allocate_request_block();
    ASSERT_NE(request_held, NULL_BLOCK_IDX);
    pool.incRef(request_held);
    EXPECT_EQ(pool.refCount(request_held), 1u);
    EXPECT_TRUE(pool.hasRequestHold(request_held));
    EXPECT_TRUE(pool.hasExternalHolder(request_held));
    EXPECT_FALSE(pool.isCacheOnly(request_held));
    EXPECT_EQ(pool.activeTreeCachedBlocksNum(), 1u);
    backing->requestFree(request_held);
    EXPECT_FALSE(pool.hasRequestHold(request_held));
    EXPECT_FALSE(pool.hasExternalHolder(request_held));
    EXPECT_TRUE(pool.isCacheOnly(request_held));
    EXPECT_EQ(pool.activeTreeCachedBlocksNum(), 0u);
    pool.decRef(request_held);

    const BlockIdxType connector_held = allocate_request_block();
    ASSERT_NE(connector_held, NULL_BLOCK_IDX);
    backing->connectorReference(connector_held);
    pool.incRef(connector_held);
    backing->requestFree(connector_held);
    EXPECT_TRUE(pool.hasExternalHolder(connector_held));
    EXPECT_FALSE(pool.isCacheOnly(connector_held));
    EXPECT_EQ(pool.activeTreeCachedBlocksNum(), 1u);
    backing->connectorFree(connector_held);
    EXPECT_FALSE(pool.hasExternalHolder(connector_held));
    EXPECT_TRUE(pool.isCacheOnly(connector_held));
    EXPECT_EQ(pool.activeTreeCachedBlocksNum(), 0u);
    pool.decRef(connector_held);

    const BlockIdxType legacy_cache_held = allocate_request_block();
    ASSERT_NE(legacy_cache_held, NULL_BLOCK_IDX);
    pool.incRef(legacy_cache_held);
    backing->blockCacheReference(legacy_cache_held);
    backing->requestFree(legacy_cache_held);
    EXPECT_TRUE(pool.hasExternalHolder(legacy_cache_held));
    EXPECT_FALSE(pool.isCacheOnly(legacy_cache_held));
    EXPECT_EQ(pool.activeTreeCachedBlocksNum(), 1u);
    backing->blockCacheFree(legacy_cache_held);
    EXPECT_FALSE(pool.hasExternalHolder(legacy_cache_held));
    EXPECT_TRUE(pool.isCacheOnly(legacy_cache_held));
    EXPECT_EQ(pool.activeTreeCachedBlocksNum(), 0u);
    pool.decRef(legacy_cache_held);

    const BlockIdxType two_tree_holders = allocate_request_block();
    ASSERT_NE(two_tree_holders, NULL_BLOCK_IDX);
    pool.incRef(two_tree_holders);
    backing->requestFree(two_tree_holders);
    EXPECT_EQ(pool.activeTreeCachedBlocksNum(), 0u);
    pool.incRef(two_tree_holders);
    EXPECT_EQ(pool.refCount(two_tree_holders), 2u);
    EXPECT_FALSE(pool.hasExternalHolder(two_tree_holders));
    EXPECT_FALSE(pool.isCacheOnly(two_tree_holders));
    EXPECT_EQ(pool.activeTreeCachedBlocksNum(), 1u);
    pool.decRef(two_tree_holders);
    EXPECT_TRUE(pool.isCacheOnly(two_tree_holders));
    EXPECT_EQ(pool.activeTreeCachedBlocksNum(), 0u);
    pool.decRef(two_tree_holders);

    EXPECT_EQ(pool.treeCachedBlocksNum(), 0u);
    EXPECT_EQ(pool.activeTreeCachedBlocksNum(), 0u);
    EXPECT_EQ(backing->freeBlocksNum(), backing->totalBlocksNum());
}

TEST(DeviceBlockPoolTest, AddressViewProjectsNonContiguousGlobalLayersWithoutMutatingGroup) {
    auto            backing      = makeBackingPool(makeConfig());
    auto            address_view = makeAddressView(backing, "hybrid_full", {/*global_layer_ids=*/2, 7});
    DeviceBlockPool pool(backing, address_view);
    ASSERT_TRUE(pool.init());

    auto block = pool.malloc();
    ASSERT_TRUE(block.has_value());
    const size_t request_refs_before   = backing->requestRefBlocksNum();
    const size_t cache_refs_before     = backing->blockCacheRefBlocksNum();
    const size_t connector_refs_before = backing->connectorRefBlocksNum();

    const auto projected_first = pool.convertIndexToAddr(/*global_layer_id=*/2, *block);
    const auto backing_first   = backing->convertIndexToAddr(/*local_layer_id=*/0, *block);
    EXPECT_EQ(projected_first.kv_addr, backing_first.kv_addr);

    const auto projected_second = pool.convertIndexToAddr(/*global_layer_id=*/7, *block);
    const auto backing_second   = backing->convertIndexToAddr(/*local_layer_id=*/1, *block);
    EXPECT_EQ(projected_second.kv_addr, backing_second.kv_addr);

    const auto projected_buffers = pool.convertIndexToBuffer(/*global_layer_id=*/7, *block);
    const auto backing_buffers   = backing->convertIndexToBuffer(/*local_layer_id=*/1, *block);
    ASSERT_EQ(projected_buffers.size(), backing_buffers.size());
    for (size_t i = 0; i < projected_buffers.size(); ++i) {
        EXPECT_EQ(projected_buffers[i].addr, backing_buffers[i].addr);
        EXPECT_EQ(projected_buffers[i].size_bytes, backing_buffers[i].size_bytes);
    }

    EXPECT_EQ(backing->requestRefBlocksNum(), request_refs_before);
    EXPECT_EQ(backing->blockCacheRefBlocksNum(), cache_refs_before);
    EXPECT_EQ(backing->connectorRefBlocksNum(), connector_refs_before);
    EXPECT_EQ(address_view->malloc_calls, 0u);
    EXPECT_EQ(address_view->insert_calls, 0u);
    EXPECT_EQ(address_view->free_calls, 0u);

    pool.free(*block);
}

TEST(DeviceBlockPoolTest, RejectsAddressViewBackedByDifferentPool) {
    auto backing      = makeBackingPool(makeConfig());
    auto other        = makeBackingPool(makeConfig());
    auto address_view = makeAddressView(other, "hybrid_full", {/*global_layer_ids=*/2, 7});

    EXPECT_ANY_THROW(DeviceBlockPool(backing, address_view));
}

}  // namespace rtp_llm
