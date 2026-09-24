#include <gtest/gtest.h>

#include <array>
#include <future>
#include <memory>
#include <set>
#include <vector>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"

namespace rtp_llm::test {
namespace {

using TestAllocator = BlockTreeCacheTestAllocator<CoordinatorCacheManager>;

BlockIndicesType allocateRequestBlocks(const DeviceBlockPoolPtr& pool, size_t count) {
    auto blocks = pool->malloc(count);
    if (!blocks.has_value()) {
        return {};
    }
    pool->incRef(*blocks);
    return std::move(*blocks);
}

CacheConfig createSingleTypeTestConfig(int layer_num, int block_num, int seq_size_per_block) {
    return makeSimpleMhaCacheConfig(layer_num, block_num, seq_size_per_block, DataType::TYPE_FP16);
}

BatchKVCacheResourcePtr createBatchKVCacheResource(int batch_size, const CacheConfig& config) {
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(batch_size);
    resource->initGroups(config.topologyPtr());
    return resource;
}

class CoordinatorCacheManagerAtomicUpdateTest: public ::testing::Test {
protected:
    std::shared_ptr<TestAllocator> allocator_;
};

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, InvalidExchangeLeavesEarlierReferencesUnchanged) {
    auto config = createSingleTypeTestConfig(1, 4, 4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto       pool   = allocator_->groupBlockPools().front();
    const auto blocks = allocateRequestBlocks(pool, 2);
    ASSERT_EQ(blocks.size(), 2u);
    BlockIndicesType replacements;
    int              required_free_blocks = 0;
    EXPECT_ANY_THROW(pool->tryReplaceRequestReferences(
        {{blocks[0], 1, 0}, {blocks[1], 2, 0}}, 1, replacements, required_free_blocks));
    EXPECT_TRUE(replacements.empty());
    EXPECT_EQ(pool->freeBlocksNum(), 1u);
    EXPECT_EQ(pool->refCount(blocks[0]), 1u);
    EXPECT_EQ(pool->refCount(blocks[1]), 1u);
    pool->decRef(blocks);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, ExchangeNotifiesCapacityOutsidePoolLock) {
    auto config = createSingleTypeTestConfig(1, 3, 4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto       pool   = allocator_->groupBlockPools().front();
    const auto blocks = allocateRequestBlocks(pool, 2);
    ASSERT_EQ(blocks.size(), 2u);
    int notifications = 0;
    pool->setCapacityChangeCallback([&] {
        ++notifications;
        EXPECT_EQ(pool->freeBlocksNum(), 1u);
        EXPECT_EQ(pool->activeBlocksNum(), 1u);
        EXPECT_EQ(pool->availableBlocksNum(), 1u);
    });
    BlockIndicesType replacements;
    int              required_free_blocks = 0;
    EXPECT_TRUE(pool->tryReplaceRequestReferences(
        {{blocks[0], 1, 0}, {blocks[1], 1, 0}}, 1, replacements, required_free_blocks));
    EXPECT_EQ(replacements, (BlockIndicesType{blocks[0]}));
    EXPECT_EQ(notifications, 1);
    pool->setCapacityChangeCallback({});
    pool->decRef(replacements);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, UpdateKVBlockReservationFailureLeavesResourceUnchanged) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/1, /*block_num=*/6, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool   = allocator_->groupBlockPools().front();
    auto blocks = allocateRequestBlocks(pool, 4);
    ASSERT_EQ(blocks.size(), 4u);

    auto resource = createBatchKVCacheResource(/*batch_size=*/2, config);
    resource->setBatchBlocks(0, "default", {blocks[0], blocks[1]});
    resource->setBatchBlocks(1, "default", {blocks[0], blocks[2]});
    pool->incRef(blocks[0]);
    resource->setBatchCacheKeys(0, {100, 101});
    resource->setBatchCacheKeys(1, {100, 102});
    resource->cacheResource(0).setDeviceReuseBlockNum(1);
    resource->markCacheKeysInitialized();
    const auto* original_block_ids = &resource->mutableBlockIds(0, "default");
    const auto  refs_before        = pool->referencedBlocksNum();
    ASSERT_EQ(pool->freeBlocksNum(), 1u);

    // Three replacements are needed: one dropped tail can be transferred and
    // one block is free, but the entire exchange must fail without allocating.
    std::vector<TaggedBlockIdPair> mapping{{"stale", 1, 2}};
    EXPECT_FALSE(allocator_->updateKVBlock(resource, {0, 0, 0, 0}, /*copy_last_block=*/true, mapping));
    EXPECT_TRUE(mapping.empty());
    ASSERT_EQ(resource->batchSize(), 2);
    EXPECT_EQ(resource->blocks(0, "default"), (BlockIndicesType{blocks[0], blocks[1]}));
    EXPECT_EQ(resource->blocks(1, "default"), (BlockIndicesType{blocks[0], blocks[2]}));
    EXPECT_EQ(resource->cacheKeys(0), (CacheKeysType{100, 101}));
    EXPECT_EQ(resource->cacheKeys(1), (CacheKeysType{100, 102}));
    EXPECT_EQ(pool->freeBlocksNum(), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(), refs_before);
    EXPECT_EQ(&resource->mutableBlockIds(0, "default"), original_block_ids);
    EXPECT_EQ(resource->cacheResource(0).deviceReuseBlockNum(), 1u);
    EXPECT_TRUE(resource->cacheKeysInitialized());

    // The original resource remains usable after failure, including its dropped
    // tail. A smaller expansion can consume that tail and the untouched free block.
    ASSERT_TRUE(allocator_->updateKVBlock(resource, {0, 0, 0}, /*copy_last_block=*/true, mapping));
    ASSERT_EQ(mapping.size(), 2u);
    EXPECT_EQ(mapping[0].src, blocks[1]);
    EXPECT_EQ(mapping[0].dst, blocks[2]);
    EXPECT_EQ(mapping[1].src, blocks[1]);
    EXPECT_NE(mapping[1].dst, mapping[0].dst);
    EXPECT_EQ(pool->freeBlocksNum(), 0u);
    allocator_->free(FreeInfo{resource, nullptr});
    pool->decRef(blocks[3]);
    EXPECT_EQ(pool->freeBlocksNum(), 5u);
    EXPECT_EQ(pool->referencedBlocksNum(), 0u);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, UpdateKVBlockReusesDroppedTailAtFullCapacity) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/1, /*block_num=*/4, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool   = allocator_->groupBlockPools().front();
    auto blocks = allocateRequestBlocks(pool, 3);
    ASSERT_EQ(blocks.size(), 3u);
    auto resource = createBatchKVCacheResource(/*batch_size=*/2, config);
    resource->setBatchBlocks(0, "default", {blocks[0], blocks[1]});
    resource->setBatchBlocks(1, "default", {blocks[0], blocks[2]});
    pool->incRef(blocks[0]);
    resource->setBatchCacheKeys(0, {100, 101});
    resource->setBatchCacheKeys(1, {100, 102});
    ASSERT_EQ(pool->freeBlocksNum(), 0u);

    std::vector<TaggedBlockIdPair> mapping;
    ASSERT_TRUE(allocator_->updateKVBlock(resource, {1, 1}, /*copy_last_block=*/true, mapping));
    ASSERT_EQ(mapping.size(), 1u);
    EXPECT_EQ(mapping[0].tag, config.topology().soleGroupForLayer(0).tag);
    EXPECT_EQ(mapping[0].src, blocks[2]);
    EXPECT_EQ(mapping[0].dst, blocks[1]);
    EXPECT_EQ(resource->blocks(0, "default"), (BlockIndicesType{blocks[0], blocks[1]}));
    EXPECT_EQ(resource->blocks(1, "default"), (BlockIndicesType{blocks[0], blocks[2]}));
    EXPECT_EQ(resource->cacheKeys(0), (CacheKeysType{100, 102}));
    EXPECT_EQ(resource->cacheKeys(1), (CacheKeysType{100, 102}));
    EXPECT_EQ(pool->freeBlocksNum(), 0u);
    allocator_->free(FreeInfo{resource, nullptr});
    EXPECT_EQ(pool->freeBlocksNum(), 3u);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, UpdateKVBlockDoesNotTransferRetainedTail) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/1, /*block_num=*/3, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool   = allocator_->groupBlockPools().front();
    auto blocks = allocateRequestBlocks(pool, 2);
    ASSERT_EQ(blocks.size(), 2u);
    auto resource = createBatchKVCacheResource(/*batch_size=*/2, config);
    resource->setBatchBlocks(0, "default", blocks);
    resource->setBatchBlocks(1, "default", blocks);
    pool->incRef(blocks);

    std::vector<TaggedBlockIdPair> mapping;
    EXPECT_FALSE(allocator_->updateKVBlock(resource, {0, 0}, /*copy_last_block=*/true, mapping));
    EXPECT_TRUE(mapping.empty());
    EXPECT_EQ(resource->blocks(0, "default"), blocks);
    EXPECT_EQ(resource->blocks(1, "default"), blocks);
    EXPECT_EQ(pool->freeBlocksNum(), 0u);
    allocator_->free(FreeInfo{resource, nullptr});
    EXPECT_EQ(pool->freeBlocksNum(), 2u);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, UpdateKVBlockEmptyResourcesNeedNoReplacements) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/1, /*block_num=*/2, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool   = allocator_->groupBlockPools().front();
    auto blocks = allocateRequestBlocks(pool, 1);
    ASSERT_EQ(blocks.size(), 1u);
    auto resource = createBatchKVCacheResource(/*batch_size=*/1, config);

    std::vector<TaggedBlockIdPair> mapping;
    ASSERT_TRUE(allocator_->updateKVBlock(resource, {0, 0}, /*copy_last_block=*/true, mapping));
    EXPECT_TRUE(mapping.empty());
    EXPECT_EQ(resource->batchSize(), 2);
    EXPECT_TRUE(resource->blocks(0, "default").empty());
    EXPECT_TRUE(resource->blocks(1, "default").empty());
    pool->decRef(blocks);
    EXPECT_EQ(pool->freeBlocksNum(), 1u);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, UpdateKVBlockDoesNotTransferExternallyReferencedTail) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/1, /*block_num=*/3, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool   = allocator_->groupBlockPools().front();
    auto blocks = allocateRequestBlocks(pool, 2);
    ASSERT_EQ(blocks.size(), 2u);
    auto resource = createBatchKVCacheResource(/*batch_size=*/2, config);
    resource->setBatchBlocks(0, "default", {blocks[0]});
    resource->setBatchBlocks(1, "default", {blocks[1]});

    // Being unique within this batch does not imply exclusive ownership in the
    // pool. Another request or any tree reference may still hold the tail.
    for (int owner = 0; owner <= static_cast<int>(BlockTreeRefType::COUNT); ++owner) {
        if (owner == 0) {
            pool->incRef(blocks[1]);
        } else {
            pool->incTreeRef(blocks[1], static_cast<BlockTreeRefType>(owner - 1));
        }

        std::vector<TaggedBlockIdPair> mapping;
        EXPECT_FALSE(allocator_->updateKVBlock(resource, {0, 0}, /*copy_last_block=*/true, mapping));
        EXPECT_TRUE(mapping.empty());
        EXPECT_EQ(resource->blocks(0, "default"), (BlockIndicesType{blocks[0]}));
        EXPECT_EQ(resource->blocks(1, "default"), (BlockIndicesType{blocks[1]}));
        EXPECT_EQ(pool->freeBlocksNum(), 0u);

        if (owner == 0) {
            pool->decRef(blocks[1]);
        } else {
            pool->decTreeRef(blocks[1], static_cast<BlockTreeRefType>(owner - 1));
        }
    }

    allocator_->free(FreeInfo{resource, nullptr});
    EXPECT_EQ(pool->freeBlocksNum(), 2u);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, TransfersTailSharedOnlyByDroppedBeams) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/1, /*block_num=*/3, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool   = allocator_->groupBlockPools().front();
    auto blocks = allocateRequestBlocks(pool, 2);
    ASSERT_EQ(blocks.size(), 2u);
    auto resource = createBatchKVCacheResource(3, config);
    resource->setBatchBlocks(0, "default", {blocks[0]});
    resource->setBatchBlocks(1, "default", {blocks[1]});
    resource->setBatchBlocks(2, "default", {blocks[1]});
    pool->incRef(blocks[1]);

    std::vector<TaggedBlockIdPair> mapping;
    ASSERT_TRUE(allocator_->updateKVBlock(resource, {0, 0}, true, mapping));
    ASSERT_EQ(mapping.size(), 1u);
    EXPECT_EQ(mapping[0].src, blocks[0]);
    EXPECT_EQ(mapping[0].dst, blocks[1]);
    ASSERT_EQ(resource->batchSize(), 2);
    EXPECT_EQ(resource->blocks(0, "default"), (BlockIndicesType{blocks[1]}));
    EXPECT_EQ(resource->blocks(1, "default"), (BlockIndicesType{blocks[0]}));
    EXPECT_TRUE(pool->isExclusiveRequestBlock(blocks[0]));
    EXPECT_TRUE(pool->isExclusiveRequestBlock(blocks[1]));
    EXPECT_EQ(pool->freeBlocksNum(), 0u);
    allocator_->free(FreeInfo{resource, nullptr});
    EXPECT_EQ(pool->freeBlocksNum(), 2u);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, ReusesDroppedPrivatePrefixAndTailAtFullCapacity) {
    auto config = createSingleTypeTestConfig(1, 5, 4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool   = allocator_->groupBlockPools().front();
    auto blocks = allocateRequestBlocks(pool, 4);
    ASSERT_EQ(blocks.size(), 4u);
    auto resource = createBatchKVCacheResource(2, config);
    resource->setBatchBlocks(0, "default", {blocks[0], blocks[1]});
    resource->setBatchBlocks(1, "default", {blocks[2], blocks[3]});
    resource->setBatchCacheKeys(0, {100, 101});
    resource->setBatchCacheKeys(1, {102, 103});
    resource->cacheResource(0).setDeviceReuseBlockNum(1);
    ASSERT_EQ(pool->freeBlocksNum(), 0u);

    std::vector<TaggedBlockIdPair> mapping;
    ASSERT_TRUE(allocator_->updateKVBlock(resource, {0, 0, 0}, true, mapping));
    ASSERT_EQ(mapping.size(), 2u);
    EXPECT_EQ(mapping[0].src, blocks[1]);
    EXPECT_EQ(mapping[0].dst, blocks[3]);
    EXPECT_EQ(mapping[1].src, blocks[1]);
    EXPECT_EQ(mapping[1].dst, blocks[2]);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(resource->blocks(i, "default"), (BlockIndicesType{blocks[0], i < 2 ? mapping[i].dst : blocks[1]}));
        EXPECT_EQ(resource->kernelBlocks(i, "default"), resource->blocks(i, "default"));
        EXPECT_EQ(resource->cacheKeys(i), (CacheKeysType{100, 101}));
    }
    EXPECT_EQ(resource->cacheResource(2).deviceReuseBlockNum(), 1u);
    EXPECT_FALSE(pool->isExclusiveRequestBlock(blocks[0]));
    EXPECT_EQ(pool->freeBlocksNum(), 0u);
    allocator_->free(FreeInfo{resource, nullptr});
    EXPECT_EQ(pool->freeBlocksNum(), 4u);
    EXPECT_EQ(pool->referencedBlocksNum(), 0u);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, ReusesPrefixSharedOnlyByDroppedBeams) {
    auto config = createSingleTypeTestConfig(1, 5, 4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool   = allocator_->groupBlockPools().front();
    auto blocks = allocateRequestBlocks(pool, 4);
    ASSERT_EQ(blocks.size(), 4u);
    auto resource = createBatchKVCacheResource(3, config);
    resource->setBatchBlocks(0, "default", {blocks[0]});
    resource->setBatchBlocks(1, "default", {blocks[1], blocks[2]});
    resource->setBatchBlocks(2, "default", {blocks[1], blocks[3]});
    pool->incRef(blocks[1]);

    std::vector<TaggedBlockIdPair> mapping;
    ASSERT_TRUE(allocator_->updateKVBlock(resource, {0, 0, 0, 0}, true, mapping));
    ASSERT_EQ(mapping.size(), 3u);
    EXPECT_EQ(mapping[0].dst, blocks[2]);
    EXPECT_EQ(mapping[1].dst, blocks[3]);
    EXPECT_EQ(mapping[2].dst, blocks[1]);
    for (const auto& copy : mapping) {
        EXPECT_EQ(copy.src, blocks[0]);
        EXPECT_TRUE(pool->isExclusiveRequestBlock(copy.dst));
    }
    allocator_->free(FreeInfo{resource, nullptr});
    EXPECT_EQ(pool->freeBlocksNum(), 4u);
    EXPECT_EQ(pool->referencedBlocksNum(), 0u);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, EvictsCachedBlocksBeforeRetryingExchange) {
    auto config = createSingleTypeTestConfig(1, 4, 4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool = allocator_->groupBlockPools().front();
    // Seed below the automatic eviction watermark, then fill the remaining
    // capacity so this test exercises the allocation-triggered eviction retry.
    const auto seeded = seedCompleteBlockTreePath(allocator_, CacheKeysType{100, 101});
    ASSERT_TRUE(seeded.success);
    auto blocks = allocateRequestBlocks(pool, 1);
    ASSERT_EQ(blocks.size(), 1u);
    const auto& cached = seeded.blocks_by_tag.at(config.topology().soleGroupForLayer(0).tag);
    blocks.insert(blocks.end(), cached.begin(), cached.end());
    auto resource = createBatchKVCacheResource(1, config);
    resource->setBatchBlocks(0, "default", {blocks[0]});
    ASSERT_EQ(pool->freeBlocksNum(), 0u);
    ASSERT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 2u);

    std::vector<TaggedBlockIdPair> mapping;
    ASSERT_TRUE(allocator_->updateKVBlock(resource, {0, 0, 0}, true, mapping));
    ASSERT_EQ(mapping.size(), 2u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 0u);
    EXPECT_EQ(pool->freeBlocksNum(), 0u);
    EXPECT_EQ((std::set<BlockIdxType>{mapping[0].dst, mapping[1].dst}), (std::set<BlockIdxType>{blocks[1], blocks[2]}));
    allocator_->free(FreeInfo{resource, nullptr});
    EXPECT_EQ(pool->freeBlocksNum(), 3u);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, ConcurrentUpdatesReclaimOwnBlocksAtFullCapacity) {
    auto config = createSingleTypeTestConfig(1, 9, 4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool = allocator_->groupBlockPools().front();
    for (int iteration = 0; iteration < 32; ++iteration) {
        SCOPED_TRACE(iteration);
        auto blocks = allocateRequestBlocks(pool, 8);
        ASSERT_EQ(blocks.size(), 8u);
        std::array<BatchKVCacheResourcePtr, 2>        resources;
        std::array<std::vector<TaggedBlockIdPair>, 2> mappings;
        std::array<std::future<bool>, 2>              results;
        std::promise<void>                            start;
        const auto                                    gate = start.get_future().share();
        for (int i = 0; i < 2; ++i) {
            resources[i] = createBatchKVCacheResource(2, config);
            resources[i]->setBatchBlocks(0, "default", {blocks[4 * i], blocks[4 * i + 1]});
            resources[i]->setBatchBlocks(1, "default", {blocks[4 * i + 2], blocks[4 * i + 3]});
            results[i] = std::async(std::launch::async, [&, i, gate] {
                gate.wait();
                return allocator_->updateKVBlock(resources[i], {0, 0, 0}, true, mappings[i]);
            });
        }
        start.set_value();
        const std::array<bool, 2> succeeded{results[0].get(), results[1].get()};
        ASSERT_TRUE(succeeded[0]);
        ASSERT_TRUE(succeeded[1]);
        std::set<BlockIdxType> destinations;
        for (int i = 0; i < 2; ++i) {
            ASSERT_EQ(mappings[i].size(), 2u);
            EXPECT_EQ(mappings[i][0].dst, blocks[4 * i + 3]);
            EXPECT_EQ(mappings[i][1].dst, blocks[4 * i + 2]);
            for (const auto& copy : mappings[i]) {
                EXPECT_EQ(copy.src, blocks[4 * i + 1]);
                EXPECT_TRUE(destinations.insert(copy.dst).second);
            }
        }
        EXPECT_EQ(pool->freeBlocksNum(), 0u);
        for (auto& resource : resources) {
            allocator_->free(FreeInfo{resource, nullptr});
        }
        EXPECT_EQ(pool->freeBlocksNum(), 8u);
        EXPECT_EQ(pool->referencedBlocksNum(), 0u);
    }
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, ConcurrentUpdatesReserveMultipleBlocksAllOrNothing) {
    auto config = createSingleTypeTestConfig(1, 5, 4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool = allocator_->groupBlockPools().front();
    for (int iteration = 0; iteration < 32; ++iteration) {
        SCOPED_TRACE(iteration);
        auto blocks = allocateRequestBlocks(pool, 2);
        ASSERT_EQ(blocks.size(), 2u);
        std::array<BatchKVCacheResourcePtr, 2>        resources;
        std::array<std::vector<TaggedBlockIdPair>, 2> mappings;
        std::array<std::future<bool>, 2>              results;
        std::promise<void>                            start;
        const auto                                    gate = start.get_future().share();
        for (int i = 0; i < 2; ++i) {
            resources[i] = createBatchKVCacheResource(1, config);
            resources[i]->setBatchBlocks(0, "default", {blocks[i]});
            results[i] = std::async(std::launch::async, [&, i, gate] {
                gate.wait();
                return allocator_->updateKVBlock(resources[i], {0, 0, 0}, true, mappings[i]);
            });
        }
        start.set_value();
        const std::array<bool, 2> succeeded{results[0].get(), results[1].get()};
        EXPECT_NE(succeeded[0], succeeded[1]);
        EXPECT_EQ(pool->freeBlocksNum(), 0u);
        for (int i = 0; i < 2; ++i) {
            EXPECT_EQ(resources[i]->batchSize(), succeeded[i] ? 3 : 1);
            EXPECT_EQ(mappings[i].size(), succeeded[i] ? 2u : 0u);
            if (succeeded[i]) {
                ASSERT_EQ(mappings[i].size(), 2u);
                EXPECT_NE(mappings[i][0].dst, mappings[i][1].dst);
                for (const auto& copy : mappings[i]) {
                    EXPECT_EQ(copy.src, blocks[i]);
                    EXPECT_NE(copy.dst, blocks[0]);
                    EXPECT_NE(copy.dst, blocks[1]);
                }
            } else {
                EXPECT_EQ(resources[i]->blocks(0, "default"), (BlockIndicesType{blocks[i]}));
                EXPECT_TRUE(pool->isExclusiveRequestBlock(blocks[i]));
            }
            allocator_->free(FreeInfo{resources[i], nullptr});
        }
        EXPECT_EQ(pool->freeBlocksNum(), 4u);
        EXPECT_EQ(pool->referencedBlocksNum(), 0u);
    }
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, RetainedBeamsMoveBlockViewsAndMetadata) {
    auto config = createSingleTypeTestConfig(2, 5, 4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto       pool   = allocator_->groupBlockPools().front();
    const auto blocks = allocateRequestBlocks(pool, 4);
    ASSERT_EQ(blocks.size(), 4u);
    auto resource = createBatchKVCacheResource(2, config);
    resource->setBatchBlocks(0, "default", {blocks[0], blocks[1]});
    resource->setBatchBlocks(1, "default", {blocks[2], blocks[3]});
    resource->setBatchCacheKeys(0, {100, 101});
    resource->setBatchCacheKeys(1, {102, 103});
    resource->cacheResource(0).setDeviceReuseBlockNum(1);
    resource->cacheResource(1).setDeviceReuseBlockNum(2);
    std::array<const BlockIds*, 2> before{&resource->mutableBlockIds(0, "default"),
                                          &resource->mutableBlockIds(1, "default")};

    std::vector<TaggedBlockIdPair> mapping;
    ASSERT_TRUE(allocator_->updateKVBlock(resource, {1, 0}, true, mapping));
    EXPECT_TRUE(mapping.empty());
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(&resource->mutableBlockIds(i, "default"), before[1 - i]);
        EXPECT_EQ(resource->cacheResource(i).deviceReuseBlockNum(), static_cast<size_t>(2 - i));
        EXPECT_EQ(resource->cacheResource(i).layerBlocks()[0].get(), before[1 - i]);
        EXPECT_EQ(resource->cacheResource(i).layerBlocks()[1].get(), before[1 - i]);
    }
    EXPECT_EQ(resource->cacheKeys(0), (CacheKeysType{102, 103}));
    EXPECT_EQ(resource->cacheKeys(1), (CacheKeysType{100, 101}));
    EXPECT_EQ(pool->freeBlocksNum(), 0u);
    allocator_->free(FreeInfo{resource, nullptr});
    EXPECT_EQ(pool->freeBlocksNum(), 4u);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, WeightedExchangePreservesOwnersAndReleasesSurplus) {
    auto config = createSingleTypeTestConfig(1, 6, 4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto       pool   = allocator_->groupBlockPools().front();
    const auto blocks = allocateRequestBlocks(pool, 5);
    ASSERT_EQ(blocks.size(), 5u);
    pool->incRef(blocks[0]);  // Another request retains this block.
    pool->incTreeRef(blocks[1], BlockTreeRefType::LOAD);
    pool->incTreeRef(blocks[2], BlockTreeRefType::CACHE);
    pool->incRef(blocks[3]);  // Two dropped beams share this block.

    BlockIndicesType replacements;
    int              required_free_blocks = 0;
    ASSERT_TRUE(pool->tryReplaceRequestReferences({{blocks[0], 1, 0},
                                                   {blocks[1], 1, 0},
                                                   {blocks[2], 1, 0},
                                                   {blocks[3], 1, 0},
                                                   {blocks[3], 1, 0},
                                                   {blocks[4], 1, 0}},
                                                  1,
                                                  replacements,
                                                  required_free_blocks));
    EXPECT_EQ(replacements, (BlockIndicesType{blocks[3]}));
    EXPECT_TRUE(pool->isExclusiveRequestBlock(blocks[3]));
    EXPECT_EQ(pool->freeBlocksNum(), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(), 2u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::LOAD), 1u);
    EXPECT_EQ(pool->referencedBlocksNum(BlockTreeRefType::CACHE), 1u);
    EXPECT_EQ(pool->availableBlocksNum(), 2u);
    pool->decRef({blocks[0], blocks[3]});
    pool->decTreeRef(blocks[1], BlockTreeRefType::LOAD);
    pool->decTreeRef(blocks[2], BlockTreeRefType::CACHE);
    EXPECT_EQ(pool->freeBlocksNum(), 5u);
    EXPECT_EQ(pool->referencedBlocksNum(), 0u);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, SharesAlignedTailAndFreesDroppedBeams) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/1, /*block_num=*/4, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool   = allocator_->groupBlockPools().front();
    auto blocks = allocateRequestBlocks(pool, 3);
    ASSERT_EQ(blocks.size(), 3u);
    auto resource = createBatchKVCacheResource(3, config);
    for (int i = 0; i < 3; ++i)
        resource->setBatchBlocks(i, "default", {blocks[i]});
    resource->setBatchCacheKeys(2, {102});

    std::vector<TaggedBlockIdPair> mapping{{"stale", 1, 2}};
    ASSERT_TRUE(allocator_->updateKVBlock(resource, {2, 2}, false, mapping));
    EXPECT_TRUE(mapping.empty());
    ASSERT_EQ(resource->batchSize(), 2);
    EXPECT_EQ(resource->blocks(0, "default"), (BlockIndicesType{blocks[2]}));
    EXPECT_EQ(resource->blocks(1, "default"), (BlockIndicesType{blocks[2]}));
    EXPECT_EQ(resource->cacheKeys(0), (CacheKeysType{102}));
    EXPECT_EQ(resource->cacheKeys(1), (CacheKeysType{102}));
    EXPECT_EQ(pool->freeBlocksNum(), 2u);
    allocator_->free(FreeInfo{resource, nullptr});
    EXPECT_EQ(pool->freeBlocksNum(), 3u);
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, ConcurrentUpdatesCompeteForLastFreeBlock) {
    auto config = createSingleTypeTestConfig(/*layer_num=*/1, /*block_num=*/4, /*seq_size_per_block=*/4);
    allocator_  = std::make_shared<TestAllocator>(config, AllocationType::HOST);
    ASSERT_TRUE(allocator_->init());
    auto pool = allocator_->groupBlockPools().front();

    for (int iteration = 0; iteration < 32; ++iteration) {
        SCOPED_TRACE(iteration);
        const auto original = allocateRequestBlocks(pool, 2);
        ASSERT_EQ(original.size(), 2u);
        std::array<BatchKVCacheResourcePtr, 2>        resources;
        std::array<std::vector<TaggedBlockIdPair>, 2> mappings;
        std::array<std::future<bool>, 2>              results;
        std::promise<void>                            start;
        const auto                                    gate = start.get_future().share();
        for (int i = 0; i < 2; ++i) {
            resources[i] = createBatchKVCacheResource(1, config);
            resources[i]->setBatchBlocks(0, "default", {original[i]});
            resources[i]->setBatchCacheKeys(0, {100 + i});
            results[i] = std::async(std::launch::async, [&, i, gate] {
                gate.wait();
                return allocator_->updateKVBlock(resources[i], {0, 0}, true, mappings[i]);
            });
        }
        start.set_value();
        const std::array<bool, 2> succeeded{results[0].get(), results[1].get()};
        // Keep successful reservations until both workers finish, so exactly
        // one update can succeed regardless of their allocation interleaving.
        EXPECT_NE(succeeded[0], succeeded[1]);
        EXPECT_EQ(pool->freeBlocksNum(), 0u);
        for (int i = 0; i < 2; ++i) {
            if (succeeded[i]) {
                ASSERT_EQ(resources[i]->batchSize(), 2);
                ASSERT_EQ(mappings[i].size(), 1u);
                EXPECT_EQ(mappings[i][0].src, original[i]);
                EXPECT_NE(mappings[i][0].dst, original[0]);
                EXPECT_NE(mappings[i][0].dst, original[1]);
                EXPECT_EQ(resources[i]->blocks(0, "default"), (BlockIndicesType{mappings[i][0].dst}));
                EXPECT_EQ(resources[i]->blocks(1, "default"), (BlockIndicesType{original[i]}));
            } else {
                EXPECT_TRUE(mappings[i].empty());
                ASSERT_EQ(resources[i]->batchSize(), 1);
                EXPECT_EQ(resources[i]->blocks(0, "default"), (BlockIndicesType{original[i]}));
            }
            EXPECT_EQ(resources[i]->cacheKeys(0), (CacheKeysType{100 + i}));
            allocator_->free(FreeInfo{resources[i], nullptr});
        }
        EXPECT_EQ(pool->freeBlocksNum(), 3u);
        EXPECT_EQ(pool->referencedBlocksNum(), 0u);
    }
}

TEST_F(CoordinatorCacheManagerAtomicUpdateTest, Expands128To1200BeamsWithSharedOrPrivatePrefixes) {
    constexpr int old_beams = 128;
    constexpr int new_beams = 1200;
    for (bool shared_prefix : {true, false}) {
        SCOPED_TRACE(shared_prefix);
        const int initial_blocks = shared_prefix ? old_beams + 1 : old_beams * 2;
        auto      config         = createSingleTypeTestConfig(1, initial_blocks + new_beams - old_beams + 1, 4);
        allocator_               = std::make_shared<TestAllocator>(config, AllocationType::HOST);
        ASSERT_TRUE(allocator_->init());
        auto       pool     = allocator_->groupBlockPools().front();
        const auto original = allocateRequestBlocks(pool, initial_blocks);
        ASSERT_EQ(original.size(), static_cast<size_t>(initial_blocks));
        auto resource = createBatchKVCacheResource(old_beams, config);
        for (int i = 0; i < old_beams; ++i) {
            resource->setBatchBlocks(i,
                                     "default",
                                     shared_prefix ? BlockIndicesType{original[0], original[i + 1]} :
                                                     BlockIndicesType{original[2 * i], original[2 * i + 1]});
            if (shared_prefix && i > 0)
                pool->incRef(original[0]);
            resource->setBatchCacheKeys(i, {100, 1000 + i});
        }
        std::vector<int> parents(new_beams);
        for (int i = 0; i < new_beams; ++i)
            parents[i] = i % old_beams;
        std::vector<TaggedBlockIdPair> mapping;
        ASSERT_TRUE(allocator_->updateKVBlock(resource, parents, true, mapping));
        ASSERT_EQ(resource->batchSize(), new_beams);
        EXPECT_EQ(mapping.size(), static_cast<size_t>(new_beams - old_beams));
        EXPECT_EQ(pool->freeBlocksNum(), 0u);
        std::set<BlockIdxType> tails;
        for (int i = 0; i < new_beams; ++i) {
            const auto& blocks = resource->blocks(i, "default");
            ASSERT_EQ(blocks.size(), 2u);
            EXPECT_EQ(blocks[0], original[shared_prefix ? 0 : 2 * parents[i]]);
            EXPECT_TRUE(tails.insert(blocks[1]).second);
            EXPECT_EQ(resource->cacheKeys(i), (CacheKeysType{100, 1000 + parents[i]}));
        }
        allocator_->free(FreeInfo{resource, nullptr});
        EXPECT_EQ(pool->freeBlocksNum(), pool->totalBlocksNum());
        EXPECT_EQ(pool->referencedBlocksNum(), 0u);
    }
}

}  // namespace
}  // namespace rtp_llm::test
