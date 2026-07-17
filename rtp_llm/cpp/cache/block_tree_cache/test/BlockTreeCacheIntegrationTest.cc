#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {
using namespace block_tree_cache_test;

class BlockTreeCacheIntegrationTest: public ::testing::Test {
protected:
    void SetUp() override {
        auto tree                             = std::make_unique<BlockTree>(1);
        auto full_group                       = std::make_shared<FullComponentGroup>();
        full_group->component_group_id        = 0;
        std::vector<ComponentGroupPtr> groups = {full_group};
        cache_ = makeBlockTreeCacheForTest(std::move(tree), std::move(groups), std::vector<Component>{});
    }

    std::unique_ptr<BlockTreeCache> cache_;
};

constexpr size_t kPathLength = 4;
constexpr size_t kPoolSize   = 16;

enum class DemotionFailureStage {
    D2H,
    H2DISK,
};

std::string tierParamName(const ::testing::TestParamInfo<Tier>& info) {
    return info.param == Tier::HOST ? "Host" : "Disk";
}

std::string demotionFailureParamName(const ::testing::TestParamInfo<DemotionFailureStage>& info) {
    return info.param == DemotionFailureStage::D2H ? "D2H" : "H2Disk";
}

void demoteTo(FullSWAEnvironment& environment, Tier target_tier) {
    environment.demoteAll(Tier::DEVICE);
    ASSERT_TRUE(environment.allSlotsAtTier(Tier::HOST));
    if (target_tier == Tier::DISK) {
        environment.demoteAll(Tier::HOST);
        ASSERT_TRUE(environment.allSlotsAtTier(Tier::DISK));
    }
}

void expectUnpublishedResult(const BlockTreeMatchResult& result) {
    EXPECT_EQ(result.matched_blocks, 0u);
    EXPECT_EQ(result.matched_node, nullptr);
    EXPECT_TRUE(result.group_block_indices.empty());
    EXPECT_TRUE(result.matched_block_sets.empty());
    EXPECT_EQ(result.async_context, nullptr);
}

void expectAggregatedReadyResult(const BlockTreeMatchResult& result, size_t full_blocks, size_t swa_blocks) {
    ASSERT_EQ(result.group_block_indices.size(), 3u);
    ASSERT_EQ(result.group_block_indices.count(0), 1u);
    ASSERT_EQ(result.group_block_indices.count(1), 1u);
    ASSERT_EQ(result.group_block_indices.count(2), 1u);
    EXPECT_EQ(result.group_block_indices.at(0).size(), full_blocks);
    EXPECT_EQ(result.group_block_indices.at(1).size(), full_blocks);
    EXPECT_EQ(result.group_block_indices.at(2).size(), swa_blocks);

    ASSERT_EQ(result.matched_block_sets.size(), 2u);
    EXPECT_EQ(result.matched_block_sets[0].component_group_id, 0);
    EXPECT_EQ(result.matched_block_sets[0].tier, Tier::DEVICE);
    EXPECT_EQ(result.matched_block_sets[0].per_node.size(), full_blocks);
    EXPECT_EQ(result.matched_block_sets[1].component_group_id, 1);
    EXPECT_EQ(result.matched_block_sets[1].tier, Tier::DEVICE);
    EXPECT_EQ(result.matched_block_sets[1].per_node.size(), swa_blocks);
}

void expectPlanningSourceRefCounts(const FullSWAEnvironment& environment, Tier tier) {
    for (size_t path_index = 0; path_index < environment.keys.size(); ++path_index) {
        const std::vector<GroupSlot> slots = environment.slotsForPathNode(path_index);
        ASSERT_EQ(slots.size(), 2u);
        for (size_t group_id = 0; group_id < slots.size(); ++group_id) {
            const BlockIdxType block = tier == Tier::HOST ? slots[group_id].host_block : slots[group_id].disk_slot;
            const IBlockPool&  pool  = tier == Tier::HOST ?
                                           static_cast<const IBlockPool&>(*environment.host_pools[group_id]) :
                                           static_cast<const IBlockPool&>(*environment.disk_pools[group_id]);
            const bool         in_swa_window = group_id == 0 || path_index + 2 >= environment.keys.size();
            const uint32_t     expected      = in_swa_window ? 2u : 1u;
            EXPECT_EQ(pool.refCount(block), expected);
        }
    }
}

size_t ticketItemCountForGroup(const std::shared_ptr<LoadBackTicket>& ticket, int group_id) {
    if (ticket == nullptr) {
        return 0;
    }
    return static_cast<size_t>(
        std::count_if(ticket->items().begin(), ticket->items().end(), [group_id](const PendingLoadBackItem& item) {
            return item.group_id == group_id;
        }));
}

std::vector<BlockIdxType> exhaustPool(IBlockPool& pool) {
    std::vector<BlockIdxType> blocks;
    while (true) {
        const BlockIdxType block = poolMalloc(pool);
        if (block == NULL_BLOCK_IDX) {
            break;
        }
        pool.incRef(block);
        blocks.push_back(block);
    }
    return blocks;
}

void releaseBlocks(IBlockPool& pool, const std::vector<BlockIdxType>& blocks) {
    for (BlockIdxType block : blocks) {
        pool.decRef(block);
    }
}

void runSingleMaintenance(FullSWAEnvironment& environment, Tier tier, double ratio) {
    environment.cache->setTierWatermark(tier, ratio, 0);
    environment.runMaintenance();
    environment.cache->setTierWatermark(tier, 0.0, 0);
}

TEST_F(BlockTreeCacheIntegrationTest, WatermarkDemotionCopiesHostBlockToDisk) {
    auto host_pool = makeHostPool(256, 8);

    auto disk_pool = makeDiskPool(256, 8, std::make_unique<MemoryDiskBlockIO>());

    auto tree                = std::make_unique<BlockTree>(1);
    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setHostPool(host_pool);
    full->setDiskPool(disk_pool);
    auto host_block = full->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(host_block, NULL_BLOCK_IDX);
    std::vector<ComponentGroupPtr> groups = {full};

    BlockTreeCacheConfig cfg;
    cfg.enable_device_cache = false;
    cfg.enable_memory_cache = true;
    cfg.enable_disk_cache   = true;

    auto cache = makeBlockTreeCacheForTest(
        std::move(tree), std::move(groups), std::vector<Component>{}, std::move(cfg));

    std::vector<std::vector<GroupSlot>> slots(1, std::vector<GroupSlot>(1));
    slots[0][0].host_block = host_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, slots));

    auto before = cache->tree()->findNode({100});
    ASSERT_NE(before.matched_node, nullptr);
    ASSERT_EQ(cache->getStats().host_heap_total_size, 1u);

    cache->setTierWatermark(Tier::HOST, 0.01, 0);
    std::vector<std::vector<GroupSlot>> trigger_slots(1, std::vector<GroupSlot>(1));
    cache->insert(nullptr, {200}, trigger_slots);
    cache->waitForPendingTasks();

    auto find = cache->tree()->findNode({100});
    ASSERT_NE(find.matched_node, nullptr);
    const auto& slot = find.matched_node->group_slots[0];
    EXPECT_EQ(slot.transfer_state, SlotTransferState::IDLE);
    EXPECT_FALSE(slot.has_value(Tier::HOST));
    EXPECT_TRUE(slot.has_value(Tier::DISK));
    EXPECT_NE(slot.disk_slot, NULL_BLOCK_IDX);
    EXPECT_FALSE(host_pool->isAllocated(host_block));
    EXPECT_TRUE(disk_pool->isAllocated(slot.disk_slot));
    EXPECT_EQ(disk_pool->refCount(slot.disk_slot), 1u);
    EXPECT_EQ(host_pool->freeBlocksNum(), 8u);
    EXPECT_EQ(disk_pool->freeBlocksNum(), 7u);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 0u);
    EXPECT_EQ(cache->getStats().disk_heap_total_size, 1u);
}

TEST_F(BlockTreeCacheIntegrationTest, LoadBackDeviceAllocationFailureRollsBackAllItems) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    DeviceBlockPoolPtr first_device_pool      = makeDevicePool({{1, 0}}, 1, "load_back_atomic_first");
    DeviceBlockPoolPtr exhausted_device_pool  = makeDevicePool({{1, 0}}, 1, "load_back_atomic_exhausted");
    const BlockIdxType exhausted_device_block = poolMalloc(*exhausted_device_pool);
    ASSERT_NE(exhausted_device_block, NULL_BLOCK_IDX);
    exhausted_device_pool->incRef(exhausted_device_block);

    std::shared_ptr<HostBlockPool> first_host_pool  = makeHostPool(1, 2);
    std::shared_ptr<HostBlockPool> second_host_pool = makeHostPool(1, 2);

    std::shared_ptr<FullComponentGroup> first_group = std::make_shared<FullComponentGroup>();
    first_group->component_group_id                 = 0;
    first_group->setDevicePools({first_device_pool});
    first_group->setHostPool(first_host_pool);
    std::shared_ptr<FullComponentGroup> second_group = std::make_shared<FullComponentGroup>();
    second_group->component_group_id                 = 1;
    second_group->setDevicePools({exhausted_device_pool});
    second_group->setHostPool(second_host_pool);

    const BlockIdxType first_host_block  = first_group->allocateSingleBlock(Tier::HOST);
    const BlockIdxType second_host_block = second_group->allocateSingleBlock(Tier::HOST);
    ASSERT_NE(first_host_block, NULL_BLOCK_IDX);
    ASSERT_NE(second_host_block, NULL_BLOCK_IDX);

    std::vector<ComponentGroupPtr>  component_groups = {first_group, second_group};
    BlockTreeCacheConfig           config;
    config.enable_memory_cache = true;
    std::unique_ptr<BlockTreeCache> cache = makeBlockTreeCacheForTest(
        std::make_unique<BlockTree>(2), std::move(component_groups), std::vector<Component>{}, std::move(config));
    ASSERT_NE(cache, nullptr);

    std::vector<std::vector<GroupSlot>> first_slots(1, std::vector<GroupSlot>(2));
    first_slots[0][0].host_block = first_host_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {100}, first_slots));
    std::vector<std::vector<GroupSlot>> second_slots(1, std::vector<GroupSlot>(2));
    second_slots[0][1].host_block = second_host_block;
    ASSERT_TRUE(BlockTreeCacheTestUtil::insertComponentGroupSlots(*cache, nullptr, {200}, second_slots));

    BlockTreeFindResult first_find  = cache->tree()->findNode({100});
    BlockTreeFindResult second_find = cache->tree()->findNode({200});
    ASSERT_NE(first_find.matched_node, nullptr);
    ASSERT_NE(second_find.matched_node, nullptr);
    ASSERT_EQ(cache->getStats().host_heap_total_size, 2u);

    first_group->referenceBlocks(GroupBlockSet{0, Tier::HOST, {{first_host_block}}});
    second_group->referenceBlocks(GroupBlockSet{1, Tier::HOST, {{second_host_block}}});

    BlockTreeCache*                 cache_pointer = cache.get();
    std::shared_ptr<LoadBackTicket> ticket        = std::make_shared<LoadBackTicket>(
        [cache_pointer](const std::vector<PendingLoadBackItem>& items) { return cache_pointer->commitLoadBack(items); },
        [cache_pointer](const std::vector<PendingLoadBackItem>& items) { cache_pointer->abortLoadBack(items); });
    ticket->items().push_back(PendingLoadBackItem{first_find.matched_node, 0, Tier::HOST, {first_host_block}});
    ticket->items().push_back(PendingLoadBackItem{second_find.matched_node, 1, Tier::HOST, {second_host_block}});

    std::shared_ptr<AsyncContext> context = ticket->commit();
    EXPECT_EQ(context, nullptr);
    EXPECT_EQ(first_device_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(exhausted_device_pool->freeBlocksNum(), 0u);
    EXPECT_EQ(first_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(second_host_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(first_find.matched_node->group_slots[0].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(second_find.matched_node->group_slots[1].transfer_state, SlotTransferState::IDLE);
    EXPECT_EQ(first_find.matched_node->group_slots[0].host_block, first_host_block);
    EXPECT_EQ(second_find.matched_node->group_slots[1].host_block, second_host_block);
    EXPECT_EQ(cache->getStats().host_heap_total_size, 2u);

    EXPECT_EQ(cache->reclaimBlocks(2, Tier::HOST), 2);
    cache->waitForPendingTasks();
    const CacheStats stats = cache->getStats();
    EXPECT_EQ(stats.device_heap_total_size, 0u);
    EXPECT_EQ(stats.host_heap_total_size, 0u);
    EXPECT_EQ(stats.disk_heap_total_size, 0u);
    EXPECT_EQ(stats.tree_node_count, 0u);
    EXPECT_EQ(first_host_pool->freeBlocksNum(), 2u);
    EXPECT_EQ(second_host_pool->freeBlocksNum(), 2u);
    EXPECT_FALSE(first_host_pool->isAllocated(first_host_block));
    EXPECT_FALSE(second_host_pool->isAllocated(second_host_block));

    exhausted_device_pool->decRef(exhausted_device_block);
    EXPECT_EQ(first_device_pool->freeBlocksNum(), 1u);
    EXPECT_EQ(exhausted_device_pool->freeBlocksNum(), 1u);
}

TEST_F(BlockTreeCacheIntegrationTest, UncommittedLoadBackTicketReleasesSourceReferences) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    std::unique_ptr<FullSWAEnvironment> environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, Tier::HOST);

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_FALSE(result.load_back_ticket->empty());
    expectPlanningSourceRefCounts(*environment, Tier::HOST);

    result.load_back_ticket.reset();
    for (size_t path_index = 0; path_index < environment->keys.size(); ++path_index) {
        const std::vector<GroupSlot> slots = environment->slotsForPathNode(path_index);
        ASSERT_EQ(slots.size(), 2u);
        for (size_t group_id = 0; group_id < slots.size(); ++group_id) {
            EXPECT_EQ(environment->host_pools[group_id]->refCount(slots[group_id].host_block), 1u);
        }
    }

    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST_F(BlockTreeCacheIntegrationTest, Evictor_SkipsRequestPinnedBlock) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    ASSERT_TRUE(environment->allSlotsAtTier(Tier::DEVICE));
    environment->expectPayloads();
    environment->expectPoolFreeCounts({12, 12, 12}, {16, 16}, {16, 16});

    environment->scripted_copy_engine->clear();
    environment->demoteAll(Tier::DEVICE);
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 0u);
    EXPECT_TRUE(environment->allSlotsAtTier(Tier::DEVICE));
    environment->expectPayloads();
    environment->expectPoolFreeCounts({12, 12, 12}, {16, 16}, {16, 16});

    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    EXPECT_TRUE(environment->allSlotsAtTier(Tier::HOST));
    EXPECT_GT(environment->scripted_copy_engine->submitCount(), 0u);
    environment->expectPayloads();
    environment->expectPoolFreeCounts({16, 16, 16}, {12, 12}, {16, 16});
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

class BlockTreeCacheDemotionFailureTest: public ::testing::TestWithParam<DemotionFailureStage> {};

TEST_P(BlockTreeCacheDemotionFailureTest, Evictor_DemotionFailure_RestoresSourceAndHeap) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    environment->releaseRequestRefs();

    const Tier source_tier = GetParam() == DemotionFailureStage::D2H ? Tier::DEVICE : Tier::HOST;
    const Tier target_tier = GetParam() == DemotionFailureStage::D2H ? Tier::HOST : Tier::DISK;
    if (source_tier == Tier::HOST) {
        environment->demoteAll(Tier::DEVICE);
        ASSERT_TRUE(environment->allSlotsAtTier(Tier::HOST));
    }

    environment->scripted_copy_engine->clear();
    for (size_t attempt = 0; attempt < 128; ++attempt) {
        environment->scripted_copy_engine->enqueue(
            GetParam() == DemotionFailureStage::D2H ? CopyStatus::DEVICE_IO_ERROR : CopyStatus::DISK_IO_ERROR);
    }
    environment->demoteAll(source_tier);

    EXPECT_TRUE(environment->allSlotsAtTier(source_tier));
    EXPECT_GT(environment->scripted_copy_engine->submitCount(), 0u);
    for (const TransferDescriptor& descriptor : environment->scripted_copy_engine->descriptors()) {
        EXPECT_EQ(descriptor.source_tier, source_tier);
        EXPECT_EQ(descriptor.target_tier, target_tier);
    }
    environment->expectPayloads();
    if (source_tier == Tier::DEVICE) {
        environment->expectPoolFreeCounts({12, 12, 12}, {16, 16}, {16, 16});
        EXPECT_GT(environment->cache->getStats().device_heap_total_size, 0u);
    } else {
        environment->expectPoolFreeCounts({16, 16, 16}, {12, 12}, {16, 16});
        EXPECT_GT(environment->cache->getStats().host_heap_total_size, 0u);
    }

    environment->scripted_copy_engine->clear();
    environment->demoteAll(source_tier);
    EXPECT_TRUE(environment->allSlotsAtTier(target_tier));
    EXPECT_GT(environment->scripted_copy_engine->submitCount(), 0u);
    for (const TransferDescriptor& descriptor : environment->scripted_copy_engine->descriptors()) {
        EXPECT_EQ(descriptor.source_tier, source_tier);
        EXPECT_EQ(descriptor.target_tier, target_tier);
    }
    environment->expectPayloads();
    if (target_tier == Tier::HOST) {
        environment->expectPoolFreeCounts({16, 16, 16}, {12, 12}, {16, 16});
    } else {
        environment->expectPoolFreeCounts({16, 16, 16}, {16, 16}, {12, 12});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

INSTANTIATE_TEST_SUITE_P(D2H,
                         BlockTreeCacheDemotionFailureTest,
                         ::testing::Values(DemotionFailureStage::D2H),
                         demotionFailureParamName);

// BTC-ISSUE-20: after a successful H2Disk retry, Full eviction only promotes the
// parent into the Device heap. The new Host leaf is never returned to the Host heap,
// so only one of four nodes can be demoted and final reclaim remains incomplete.
// Re-enable when H2Disk completion promotes the parent at the source tier.
INSTANTIATE_TEST_SUITE_P(DISABLED_H2DiskRetry,
                         BlockTreeCacheDemotionFailureTest,
                         ::testing::Values(DemotionFailureStage::H2DISK),
                         demotionFailureParamName);

class BlockTreeCacheLowerTierTest: public ::testing::TestWithParam<Tier> {};

// BTC-ISSUE-03/05: match currently treats Host/Disk residency as publicly ready.
// Re-enable when public match results use Device-ready residency while planning load-back from the logical path.
TEST_P(BlockTreeCacheLowerTierTest, DISABLED_FullSWA_MatchLowerTierOnlyReturnsTicketWithoutPublishing) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, GetParam());
    environment->expectPayloads();

    environment->scripted_copy_engine->clear();
    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    expectUnpublishedResult(result);
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_FALSE(result.load_back_ticket->empty());
    EXPECT_EQ(result.load_back_blocks, 6u);
    EXPECT_EQ(result.host_load_back_blocks, GetParam() == Tier::HOST ? 6u : 0u);
    EXPECT_EQ(result.disk_load_back_blocks, GetParam() == Tier::DISK ? 6u : 0u);
    EXPECT_EQ(ticketItemCountForGroup(result.load_back_ticket, 0), 4u);
    EXPECT_EQ(ticketItemCountForGroup(result.load_back_ticket, 1), 2u);
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 0u);
    expectPlanningSourceRefCounts(*environment, GetParam());
    if (GetParam() == Tier::HOST) {
        environment->expectPoolFreeCounts({16, 16, 16}, {12, 12}, {16, 16});
    } else {
        environment->expectPoolFreeCounts({16, 16, 16}, {16, 16}, {12, 12});
    }

    std::shared_ptr<AsyncContext> context = result.load_back_ticket->commit();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(result.load_back_ticket->commit(), nullptr);
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    expectUnpublishedResult(result);
    EXPECT_EQ(result.async_context, nullptr);

    const size_t submits_after_commit = environment->scripted_copy_engine->submitCount();
    EXPECT_GT(submits_after_commit, 0u);
    EXPECT_EQ(result.load_back_ticket->commit(), nullptr);
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), submits_after_commit);

    BlockTreeMatchResult rematch = environment->cache->match(environment->keys);
    EXPECT_EQ(rematch.matched_blocks, kPathLength);
    ASSERT_NE(rematch.matched_node, nullptr);
    expectAggregatedReadyResult(rematch, /*full_blocks=*/4, /*swa_blocks=*/2);
    environment->expectPayloads();
    environment->releaseMatch(rematch);

    if (GetParam() == Tier::HOST) {
        environment->expectPoolFreeCounts({12, 12, 14}, {16, 14}, {16, 16});
    } else {
        environment->expectPoolFreeCounts({12, 12, 14}, {16, 16}, {16, 14});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}
// BTC-ISSUE-05: validators currently count lower-tier slots even when load-back is disabled.
// Re-enable when disabled load-back exposes no lower-tier hit and allocates no recovery resources.
TEST_F(BlockTreeCacheIntegrationTest, DISABLED_LoadBackDisabled_DoesNotReportLowerTierHit) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.enable_load_back = false;
    auto environment         = FullSWAEnvironment::create(options);
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, Tier::HOST);
    environment->scripted_copy_engine->clear();

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    expectUnpublishedResult(result);
    EXPECT_EQ(result.load_back_ticket, nullptr);
    EXPECT_EQ(result.load_back_blocks, 0u);
    EXPECT_EQ(result.host_load_back_blocks, 0u);
    EXPECT_EQ(result.disk_load_back_blocks, 0u);
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 0u);
    EXPECT_TRUE(environment->allSlotsAtTier(Tier::HOST));
    environment->expectPayloads();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

INSTANTIATE_TEST_SUITE_P(HostAndDisk,
                         BlockTreeCacheLowerTierTest,
                         ::testing::Values(Tier::HOST, Tier::DISK),
                         tierParamName);
// BTC-ISSUE-03/05: the validator currently lets lower-tier SWA blocks extend the public boundary.
// Re-enable when only the prefix ready in every group is published and the suffix is represented by the ticket.
TEST_F(BlockTreeCacheIntegrationTest, DISABLED_FullSWA_MatchPublishesOnlyReadyBoundary) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    environment->releaseRequestRefsForGroup(1);
    runSingleMaintenance(*environment, Tier::DEVICE, 0.20);

    for (size_t path_index = 0; path_index < kPathLength; ++path_index) {
        const std::vector<GroupSlot> slots = environment->slotsForPathNode(path_index);
        ASSERT_EQ(slots.size(), 2u);
        EXPECT_TRUE(slots[0].has_value(Tier::DEVICE));
        if (path_index < 2) {
            EXPECT_TRUE(slots[1].has_value(Tier::DEVICE));
        } else {
            EXPECT_TRUE(slots[1].has_value(Tier::HOST));
            EXPECT_FALSE(slots[1].has_value(Tier::DEVICE));
        }
    }

    environment->scripted_copy_engine->clear();
    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    EXPECT_EQ(result.matched_blocks, 2u);
    ASSERT_EQ(result.matched_node->cache_key, environment->keys[1]);
    expectAggregatedReadyResult(result, /*full_blocks=*/2, /*swa_blocks=*/2);
    ASSERT_NE(result.load_back_ticket, nullptr);
    EXPECT_EQ(ticketItemCountForGroup(result.load_back_ticket, 0), 0u);
    EXPECT_EQ(ticketItemCountForGroup(result.load_back_ticket, 1), 2u);
    EXPECT_EQ(environment->scripted_copy_engine->submitCount(), 0u);
    environment->releaseMatch(result);
    result.load_back_ticket.reset();
    environment->releaseRequestRefs();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}
// BTC-ISSUE-17: Disk load-back currently allocates staging from the resident Host cache pool.
// Re-enable when staging has an independent transient allocation path and is released on both outcomes.
TEST_F(BlockTreeCacheIntegrationTest, DISABLED_DiskLoadBack_HostPoolFullStillUsesTransientStaging) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto environment = FullSWAEnvironment::create();
    environment->insertRequestPath();
    environment->releaseRequestRefs();
    demoteTo(*environment, Tier::DISK);

    std::vector<std::vector<BlockIdxType>> resident_host_fill(2);
    for (size_t group_id = 0; group_id < 2; ++group_id) {
        resident_host_fill[group_id] = exhaustPool(*environment->host_pools[group_id]);
        ASSERT_EQ(environment->host_pools[group_id]->freeBlocksNum(), 0u);
    }
    environment->scripted_copy_engine->clear();

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    expectUnpublishedResult(result);
    ASSERT_NE(result.load_back_ticket, nullptr);
    std::shared_ptr<AsyncContext> context = result.load_back_ticket->commit();
    ASSERT_NE(context, nullptr);
    context->waitDone();
    ASSERT_TRUE(context->success());
    EXPECT_EQ(environment->host_pools[0]->freeBlocksNum(), 0u);
    EXPECT_EQ(environment->host_pools[1]->freeBlocksNum(), 0u);
    environment->expectPayloads();

    BlockTreeMatchResult rematch = environment->cache->match(environment->keys);
    EXPECT_EQ(rematch.matched_blocks, kPathLength);
    expectAggregatedReadyResult(rematch, /*full_blocks=*/4, /*swa_blocks=*/2);
    environment->releaseMatch(rematch);
    EXPECT_EQ(environment->disk_pools[0]->freeBlocksNum(), 16u);
    EXPECT_EQ(environment->disk_pools[1]->freeBlocksNum(), 14u);

    for (size_t group_id = 0; group_id < 2; ++group_id) {
        releaseBlocks(*environment->host_pools[group_id], resident_host_fill[group_id]);
        EXPECT_EQ(environment->host_pools[group_id]->freeBlocksNum(), kPoolSize);
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}
}  // namespace
}  // namespace rtp_llm
