#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/FullCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/BlockPoolTestHelper.h"

namespace rtp_llm {
namespace test {

using DeviceFullCacheManager = FullCacheManager;

class DeviceFullCacheManagerTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(DeviceFullCacheManagerTest, OffloadRoutesStagedLongRequestBeforeAllocatingShortRequest) {
    autil::EnvGuard guard("RTP_LLM_SPARSE_KV_OFFLOAD_HBM_SHARE_DENOMINATOR", "3");
    auto            cache = makeSimpleMhaCacheConfig(1, 30, 4, TYPE_FP16, 1, 64);
    const auto&     tag   = cache.topology().groups().front().tag;
    cache.tiered_cache_configs.emplace(tag, TieredCacheConfig{30, 10, 16});
    auto pool = std::make_shared<DeviceBlockPool>(std::make_shared<DeviceBlockPoolConfig>(
        DeviceBlockPoolConfigHelper::createConfigForGroup(cache, cache.group(tag))));
    ASSERT_TRUE(pool->init());
    DeviceFullCacheManager group(cache.group(tag), pool, 0);
    ASSERT_TRUE(group.init(cache.layerIdsForGroup(tag)));

    BlockIds long_request;
    // Allocate only the common prefix first, but route from the full 48-token request.
    ASSERT_TRUE(group.malloc(long_request, 4, false, 0, nullptr, {}, 48));
    EXPECT_EQ(long_request.blocks(), (BlockIndicesType{10}));
    ASSERT_TRUE(group.malloc(long_request, 48, false, 0, nullptr, {}, 48));
    EXPECT_TRUE(
        std::all_of(long_request.blocks().begin(), long_request.blocks().end(), [](int block) { return block >= 10; }));
    BlockIds short_request;
    ASSERT_TRUE(group.malloc(short_request, 8));
    EXPECT_EQ(short_request.blocks(), (BlockIndicesType{1, 2}));
    ASSERT_TRUE(group.malloc(long_request, 52));
    EXPECT_GE(long_request.blocks().back(), 10);
    group.unreference(long_request.blocks());
    group.unreference(short_request.blocks());
    EXPECT_EQ(pool->freeBlocksNum(), 29u);
}

TEST_F(DeviceFullCacheManagerTest, OffloadInitialRoutingPreservesReusedPrefixAndBackfillsPlaceholders) {
    autil::EnvGuard guard("RTP_LLM_SPARSE_KV_OFFLOAD_HBM_SHARE_DENOMINATOR", "3");
    auto            cache = makeSimpleMhaCacheConfig(1, 20, 4, TYPE_FP16, 1, 64);
    const auto&     tag   = cache.topology().groups().front().tag;
    cache.tiered_cache_configs.emplace(tag, TieredCacheConfig{20, 10, 16});
    auto pool = std::make_shared<DeviceBlockPool>(std::make_shared<DeviceBlockPoolConfig>(
        DeviceBlockPoolConfigHelper::createConfigForGroup(cache, cache.group(tag))));
    ASSERT_TRUE(pool->init());
    DeviceFullCacheManager group(cache.group(tag), pool, 0);
    ASSERT_TRUE(group.init(cache.layerIdsForGroup(tag)));

    auto prefix = pool->mallocForRequest(1, 1);
    ASSERT_TRUE(prefix.has_value());
    ASSERT_EQ(prefix->front(), 1);
    pool->incRef(*prefix);
    BlockIds request;
    request.assign({prefix->front(), NULL_BLOCK_IDX});
    std::vector<size_t> filled;
    ASSERT_TRUE(group.malloc(request, 16, true, 0, &filled, {}, 16));
    EXPECT_EQ(request.blocks(), (BlockIndicesType{1, 10, 11, 12}));
    EXPECT_EQ(filled, (std::vector<size_t>{1}));
    EXPECT_EQ(pool->refCount(1), 1u);
    group.unreference(request.blocks());
    EXPECT_EQ(pool->freeBlocksNum(), 19u);
}

// ==================== Basic functionality tests ====================

TEST_F(DeviceFullCacheManagerTest, NeedBlocksNumTest) {
    auto block_pool = createDeviceBlockPool();
    block_pool->init();

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    DeviceFullCacheManager group1({}, spec, block_pool, 0);
    ASSERT_EQ(2, group1.needBlocksNum(10, 1));
    ASSERT_EQ(0, group1.needBlocksNum(10, 5));
    ASSERT_EQ(1, group1.needBlocksNum(1, 0));
    ASSERT_EQ(0, group1.needBlocksNum(2, 1));
}

TEST_F(DeviceFullCacheManagerTest, GetNeedBlocksTest) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    DeviceFullCacheManager group({}, spec, block_pool, 0);

    // common=8 => 2 blocks, seq=12 reserve=3 => ceil(15/4)=4 blocks => extra=2
    const auto need =
        group.getNeedBlocks(/*common_seq_len=*/8, /*seq_len=*/12, /*reserve_step=*/3, /*reuse_blocks_len=*/0, false);
    EXPECT_EQ(need.common_blocks, 2);
    EXPECT_EQ(need.extra_blocks, 2);

    // no reserve: common=12 => 3, seq=12 => 3 => extra=0
    const auto need2 =
        group.getNeedBlocks(/*common_seq_len=*/12, /*seq_len=*/12, /*reserve_step=*/0, /*reuse_blocks_len=*/0, false);
    EXPECT_EQ(need2.common_blocks, 3);
    EXPECT_EQ(need2.extra_blocks, 0);
}

TEST_F(DeviceFullCacheManagerTest, RemoveSkippedBlocksTest) {
    auto block_pool = createDeviceBlockPool();
    block_pool->init();

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 4;

    DeviceFullCacheManager group1({}, spec, block_pool, 0);

    BlockIndicesType old_indices = {1, 2, 3, 4};
    BlockIds         block_ids(/*kernel_blocks_per_kv_block=*/1);
    block_ids.assign(old_indices);
    group1.removeSkippedBlocks(block_ids);
    ASSERT_EQ(old_indices, block_ids.blocks());
}

TEST_F(DeviceFullCacheManagerTest, MallocFreeTest) {
    auto block_pool = createDeviceBlockPool();
    block_pool->init();
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    DeviceFullCacheManager group1({}, spec, block_pool, 0);

    CacheKeysType cache_keys = {101, 102, 103};
    BlockIds      block_ids(/*kernel_blocks_per_kv_block=*/1);

    ASSERT_TRUE(group1.malloc(block_ids, 7));
    ASSERT_EQ(block_pool->freeBlocksNum(), 5);
    ASSERT_EQ(block_ids.blocks().size(), 4);

    BlockIndicesType expected_result = {1, 2, 3, 4};
    ASSERT_EQ(block_ids.blocks(), expected_result);

    group1.unreference(block_ids.blocks());
    ASSERT_EQ(block_pool->freeBlocksNum(), 9);

    BlockIds block_ids2(/*kernel_blocks_per_kv_block=*/1);
    ASSERT_FALSE(group1.malloc(block_ids2, 180));
}

TEST_F(DeviceFullCacheManagerTest, MallocBackfillsMatchedLoadPlaceholder) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;
    DeviceFullCacheManager group({}, spec, block_pool, 0);

    auto resident = block_pool->malloc();
    ASSERT_TRUE(resident.has_value());
    block_pool->incRef(*resident);

    BlockIds block_ids(/*kernel_blocks_per_kv_block=*/1);
    block_ids.assign({NULL_BLOCK_IDX, *resident});
    ASSERT_TRUE(group.malloc(block_ids, /*seq_len=*/4));

    ASSERT_EQ(block_ids.blocksNum(), 2u);
    EXPECT_FALSE(isNullBlockIdx(block_ids.blocks()[0]));
    EXPECT_EQ(block_ids.blocks()[1], *resident);
    EXPECT_EQ(block_pool->refCount(block_ids.blocks()[0]), 1u);
    EXPECT_EQ(block_pool->refCount(*resident), 1u);

    group.unreference(block_ids.blocks());
}

// Single-count co-hold: a block held by both a request (via group malloc) and a cache
// holder (extra incRef) must survive the request release and only free on the final
// decRef.
TEST_F(DeviceFullCacheManagerTest, RequestReleaseKeepsCacheHeldBlock) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;

    DeviceFullCacheManager group1({}, spec, block_pool, 0);

    BlockIds block_ids(/*kernel_blocks_per_kv_block=*/1);
    ASSERT_TRUE(group1.malloc(block_ids, /*seq_len=*/2));
    ASSERT_FALSE(block_ids.blocks().empty());
    const auto block = block_ids.blocks()[0];
    EXPECT_EQ(block_pool->refCount(block), 1u);  // request holder

    block_pool->incTreeRef(block, BlockTreeRefType::CACHE);
    EXPECT_EQ(block_pool->refCount(block), 2u);

    group1.unreference(BlockIndicesType{block});  // release request holder
    EXPECT_TRUE(block_pool->isAllocated(block));
    EXPECT_EQ(block_pool->refCount(block), 1u);

    block_pool->decTreeRef(block, BlockTreeRefType::CACHE);
    EXPECT_FALSE(block_pool->isAllocated(block));
}

TEST_F(DeviceFullCacheManagerTest, ReleaseFiltersNullBlocksAndPreservesOtherReferences) {
    auto block_pool = createDeviceBlockPool();
    ASSERT_TRUE(block_pool->init());

    auto spec                = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block = 2;
    DeviceFullCacheManager group({}, spec, block_pool, 7);

    BlockIds block_ids(/*kernel_blocks_per_kv_block=*/1);
    ASSERT_TRUE(group.malloc(block_ids, /*seq_len=*/2));
    ASSERT_EQ(block_ids.blocksNum(), 1u);
    const BlockIdxType block = block_ids.blocks()[0];
    block_pool->incTreeRef(block, BlockTreeRefType::CACHE);

    group.unreference(BlockIndicesType{NULL_BLOCK_IDX, block});

    EXPECT_TRUE(block_pool->isAllocated(block));
    EXPECT_EQ(block_pool->refCount(block), 1u);

    block_pool->decTreeRef(block, BlockTreeRefType::CACHE);
    EXPECT_FALSE(block_pool->isAllocated(block));
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
