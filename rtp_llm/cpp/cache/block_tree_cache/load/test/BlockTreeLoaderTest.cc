#include "rtp_llm/cpp/cache/block_tree_cache/load/BlockTreeLoader.h"

#include <atomic>
#include <chrono>
#include <future>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

namespace rtp_llm {
namespace {

using block_tree_cache_test::FullSWAEnvironment;
using block_tree_cache_test::FullSWAEnvironmentOptions;
using block_tree_cache_test::cudaAvailable;
using block_tree_cache_test::insertGroupSetResources;
using block_tree_cache_test::makeHostPool;
using block_tree_cache_test::makeStructuralDevicePool;
using block_tree_cache_test::releaseDeviceBlocks;

class ReadFailureBackend final: public StorageBackend {
public:
    ~ReadFailureBackend() override {
        shutdown();
    }

    std::atomic<size_t> read_calls{0};

protected:
    bool initImpl() override {
        return true;
    }
    StorageMatchResult matchImpl(const StorageRequest& request) override {
        return {request.keys->size(), nullptr};
    }
    void readImpl(const StorageRequest&, const std::shared_ptr<StorageBackendMatchMeta>&) override {
        ++read_calls;
        throw std::runtime_error("injected remote read failure");
    }
    void writeImpl(const StorageRequest&) override {}
};

TEST(BlockTreeLoaderTest, CrcDeviceReuseDoesNotPromoteUnprotectedBackendFailure) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);
    environment->insertRequestPath();
    environment->releaseRequestRefs();

    // Only the ordinary group needs a HOST restore. The CRC-capable group's
    // existing DEVICE value participates in the same match without any CRC I/O.
    using Peer = block_tree_cache_test::BlockTreeCacheTestPeer;
    ASSERT_TRUE(Peer::demoteOneForGroupSetForTest(*environment->cache, 1, Tier::DEVICE));
    Peer::waitForTaskPoolIdleForTest(*environment->cache);
    auto resources = environment->resourcesForPathNode(0);
    ASSERT_TRUE(resources[0].hasTier(Tier::DEVICE));
    ASSERT_TRUE(resources[1].hasTier(Tier::HOST));
    environment->groups[0]->enable_crc_ = true;

    auto                       backend = std::make_shared<ReadFailureBackend>();
    StorageBackend::PoolsByTag pools;
    for (const auto& group : environment->groups) {
        for (size_t member = 0; member < group->groupTags().size(); ++member) {
            pools.emplace(group->groupTags()[member], group->devicePools()[member]);
        }
    }
    ASSERT_TRUE(backend->init(
        environment->topology, pools, [](int, const std::string&, int) { return std::vector<BlockInfo>{}; }));
    environment->cache->loader_.storage_backend_ = backend;
    CacheKeysType keys                           = environment->keys;
    keys.push_back(keys.back() + 1);
    auto result  = environment->cache->match(keys);
    auto context = result.async_context;
    ASSERT_NE(context, nullptr);
    ASSERT_TRUE(context->needBackendMatch());
    ASSERT_EQ(context->loadDescs().size(), 2u);
    EXPECT_EQ(context->loadDescs()[0].source_tier, Tier::DEVICE);
    EXPECT_EQ(context->loadDescs()[1].source_tier, Tier::HOST);

    // Keep each allocation's request reference until the backend and loader
    // have both completed. DEVICE descriptors already own their reuse refs.
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_refs;
    for (size_t i = 0; i < context->loadDescs().size(); ++i) {
        const auto&               desc    = context->loadDescs()[i];
        const auto&               group   = environment->groups[desc.group_set_id];
        std::vector<BlockIdxType> targets = desc.source_blocks;
        if (desc.source_tier != Tier::DEVICE) {
            targets.clear();
            for (const auto& pool : group->devicePools()) {
                const auto blocks = pool->malloc(1).value();
                pool->incRef(blocks);
                targets.push_back(blocks.front());
            }
        }
        for (size_t member = 0; member < targets.size(); ++member) {
            request_refs.emplace_back(group->devicePools()[member], targets[member]);
        }
        context->setTargetBlocks(i, std::move(targets));
    }
    for (size_t handle = 0; handle < context->backendHandles()[1].size(); ++handle) {
        const auto& pool   = pools.at(context->backendHandles()[1][handle].tag);
        const auto  blocks = pool->malloc(1).value();
        pool->incRef(blocks);
        request_refs.emplace_back(pool, blocks.front());
        context->setBackendTargetBlock(1, handle, blocks.front());
    }
    context->setMatchCallback([](LoadAsyncContext& current, size_t matched) {
        EXPECT_EQ(matched, 2u);
        return current.commit();
    });
    context->startBackendMatch();
    context->waitDone();
    backend->shutdown();
    EXPECT_EQ(backend->read_calls.load(), 1u);
    EXPECT_FALSE(context->success());
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    resources = environment->resourcesForPathNode(0);
    EXPECT_TRUE(resources[0].hasTier(Tier::DEVICE));
    EXPECT_TRUE(resources[1].hasTier(Tier::HOST));
    EXPECT_FALSE(resources[1].hasTier(Tier::DEVICE));
    for (const auto& resource : resources) {
        EXPECT_TRUE(resource.isMatchUsable());
    }

    result.async_context.reset();
    context.reset();
    for (const auto& [pool, block] : request_refs) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, HostLoadInstallsAllocatorBoundDeviceTargets) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 2;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::HOST));
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::HOST), 1u);

    BlockTreeMatchResult result = environment->cache->match(environment->keys);
    EXPECT_EQ(result.matched_device_blocks, 0u);
    std::shared_ptr<LoadAsyncContext> load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    EXPECT_EQ(load_context->matchedBlocks(), 2u);
    EXPECT_EQ(load_context->matchedBlocks(Tier::HOST), 2u);
    EXPECT_EQ(load_context->matchedBlocks(Tier::DISK), 0u);

    for (const TransferDescriptor& desc : load_context->loadDescs()) {
        const auto source_pool = environment->groups.at(desc.group_set_id)->hostPool();
        ASSERT_NE(source_pool, nullptr);
        for (BlockIdxType block : desc.source_blocks) {
            EXPECT_EQ(source_pool->treeRefCount(block), 2u);
        }
    }
    for (const auto& source_pool : environment->host_pools) {
        EXPECT_EQ(source_pool->referencedBlocksNum(BlockTreeRefType::CACHE), options.path_length);
        EXPECT_EQ(source_pool->referencedBlocksNum(BlockTreeRefType::LOAD), options.path_length);
    }

    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
        std::vector<BlockIdxType> targets;
        const size_t              group_set_id = load_context->loadDescs()[desc_index].group_set_id;
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(group_set_id)->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks);
            targets.push_back(blocks.front());
            request_targets.emplace_back(pool, blocks.front());
        }
        load_context->setTargetBlocks(desc_index, std::move(targets));
    }

    ASSERT_TRUE(load_context->commit());
    std::shared_ptr<AsyncContext> context = load_context;
    context->waitDone();
    ASSERT_TRUE(context->done());
    EXPECT_TRUE(context->success());
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::HOST), 0u);
    EXPECT_EQ(environment->cache->evictor_.candidateCount(/*group_set_id=*/0, Tier::DEVICE), 1u);
    environment->expectPayloads();

    result.async_context.reset();
    load_context.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : request_targets) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, HostLoadUsesReservedAdmissionWhenBackgroundQueueIsFull) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::HOST));

    constexpr size_t worker_count = 2;
    constexpr size_t queue_size   = BlockTreeTaskPool::kLoadReservedSlots + 2;
    auto replacement = std::make_unique<BlockTreeTaskPool>(worker_count, queue_size, "load_reserved_admission");
    ASSERT_TRUE(replacement->start());
    environment->cache->task_pool_->shutdown();
    environment->cache->task_pool_          = std::move(replacement);
    environment->cache->evictor_.task_pool_ = environment->cache->task_pool_.get();
    environment->cache->loader_.task_pool_  = environment->cache->task_pool_.get();
    environment->cache->storer_.task_pool_  = environment->cache->task_pool_.get();

    auto release_promise = std::make_shared<std::promise<void>>();
    auto release_future  = release_promise->get_future().share();
    auto release_once    = std::make_shared<std::once_flag>();
    auto release_workers = [release_promise, release_once]() {
        std::call_once(*release_once, [release_promise]() { release_promise->set_value(); });
    };
    [[maybe_unused]] auto release_guard =
        std::shared_ptr<void>(nullptr, [release_workers](void*) { release_workers(); });

    auto  entered_count   = std::make_shared<std::atomic<size_t>>(0);
    auto  entered_promise = std::make_shared<std::promise<void>>();
    auto  entered_future  = entered_promise->get_future();
    auto* task_pool       = environment->cache->task_pool_.get();
    for (size_t worker = 0; worker < worker_count; ++worker) {
        if (!task_pool->submit(BlockTreeTaskClass::BACKGROUND,
                               [release_future, entered_count, entered_promise]() {
                                   if (entered_count->fetch_add(1) + 1 == worker_count) {
                                       entered_promise->set_value();
                                   }
                                   release_future.wait();
                               })) {
            FAIL() << "failed to submit background task";
        }
    }
    ASSERT_EQ(entered_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    ASSERT_TRUE(environment->cache->task_pool_->submit(BlockTreeTaskClass::BACKGROUND, [] {}));
    ASSERT_TRUE(environment->cache->task_pool_->submit(BlockTreeTaskClass::BACKGROUND, [] {}));
    EXPECT_FALSE(environment->cache->task_pool_->submit(BlockTreeTaskClass::BACKGROUND, [] {}));

    BlockTreeMatchResult result       = environment->cache->match(environment->keys);
    auto                 load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
        std::vector<BlockIdxType> targets;
        const size_t              group_set_id = load_context->loadDescs()[desc_index].group_set_id;
        for (const DeviceBlockPoolPtr& pool : environment->groups.at(group_set_id)->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            ASSERT_EQ(blocks.size(), 1u);
            pool->incRef(blocks);
            targets.push_back(blocks.front());
            request_targets.emplace_back(pool, blocks.front());
        }
        load_context->setTargetBlocks(desc_index, std::move(targets));
    }

    auto completion_promise = std::make_shared<std::promise<ErrorInfo>>();
    auto completion_future  = completion_promise->get_future();
    load_context->onDone([completion_promise](ErrorInfo error) { completion_promise->set_value(std::move(error)); });
    ASSERT_TRUE(load_context->commit());
    {
        std::lock_guard<std::mutex> lock(environment->cache->task_pool_->lifecycle_mutex_);
        EXPECT_EQ(environment->cache->task_pool_->load_queue_.size(), 1u);
        EXPECT_EQ(environment->cache->task_pool_->background_queue_.size(), 2u);
    }

    release_workers();
    ASSERT_EQ(completion_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(completion_future.get().ok());
    EXPECT_TRUE(load_context->done());
    EXPECT_TRUE(load_context->success());
    EXPECT_TRUE(environment->allResourcesAtTier(Tier::DEVICE));

    result.async_context.reset();
    load_context.reset();
    environment->reclaimAll();
    for (const auto& [pool, block] : request_targets) {
        releaseDeviceBlocks(*environment->cache, pool, {block});
    }
    environment->reclaimAll();
    environment->expectFullyReclaimed();
    EXPECT_EQ(environment->cache->task_pool_->pending_tasks_.load(), 0);
}


TEST(BlockTreeLoaderTest, MatchRefreshesOnlyReusedSuffixForEachGroup) {
    constexpr size_t path_length = 4;
    auto             full        = std::make_shared<FullGroupSet>(
        std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(0)}, makeHostPool(1, path_length), nullptr);
    auto swa    = std::make_shared<SWAGroupSet>(/*sliding_window_size=*/2,
                                             /*seq_size_per_block=*/1,
                                             std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(1)},
                                             makeHostPool(1, path_length),
                                             nullptr);
    auto linear = std::make_shared<LinearGroupSet>(
        std::vector<DeviceBlockPoolPtr>{makeStructuralDevicePool(2)}, makeHostPool(1, path_length), nullptr);
    std::vector<GroupSetPtr> groups = {full, swa, linear};

    BlockTreeCacheConfig config;
    config.enable_device_cache = true;
    config.enable_host_cache   = true;
    config.enable_disk_cache   = false;
    auto cache                 = block_tree_cache_test::makeBlockTreeCacheForTest(groups, config);
    ASSERT_NE(cache, nullptr);

    const CacheKeysType                        keys = {100, 200, 300, 400};
    std::vector<std::vector<GroupSetResource>> resources(path_length, std::vector<GroupSetResource>(groups.size()));
    for (size_t path_index = 0; path_index < path_length; ++path_index) {
        for (size_t group_set_id = 0; group_set_id < groups.size(); ++group_set_id) {
            resources[path_index][group_set_id].host_block =
                groups[group_set_id]->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            ASSERT_NE(resources[path_index][group_set_id].host_block, NULL_BLOCK_IDX);
        }
    }
    ASSERT_TRUE(insertGroupSetResources(*cache, keys, resources));

    const std::vector<TreeNode*> path = cache->tree()->findNode(keys);
    ASSERT_EQ(path.size(), path_length);
    std::vector<std::vector<uint64_t>> access_before(path.size());
    for (size_t node_index = 0; node_index < path.size(); ++node_index) {
        access_before[node_index].reserve(groups.size());
        for (const GroupSetResource& resource : path[node_index]->group_set_resources) {
            EXPECT_EQ(resource.candidate_meta.hit_count, 0u);
            access_before[node_index].push_back(resource.candidate_meta.last_access_seq);
        }
    }

    BlockTreeMatchResult result = cache->match(keys);
    EXPECT_EQ(result.matched_device_blocks, 0u);
    auto load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    EXPECT_EQ(load_context->matchedBlocks(), path_length);

    const std::vector<size_t> reused_suffix_starts = {0, 2, 3};
    for (size_t group_set_id = 0; group_set_id < groups.size(); ++group_set_id) {
        for (size_t node_index = 0; node_index < path.size(); ++node_index) {
            const CandidateMeta& meta = path[node_index]->group_set_resources[group_set_id].candidate_meta;
            if (node_index < reused_suffix_starts[group_set_id]) {
                EXPECT_EQ(meta.last_access_seq, access_before[node_index][group_set_id]);
                EXPECT_EQ(meta.hit_count, 0u);
            } else {
                EXPECT_GT(meta.last_access_seq, access_before[node_index][group_set_id]);
                EXPECT_EQ(meta.hit_count, 1u);
            }
        }
    }

    result.async_context.reset();
    load_context.reset();
}

TEST(BlockTreeLoaderTest, DiskTransferFailureInstallsNoLoadTargets) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    environment->demoteAll(Tier::HOST);
    ASSERT_TRUE(environment->allResourcesAtTier(Tier::DISK));

    environment->scripted_per_rank_transfer_engine->clear();
    environment->scripted_per_rank_transfer_engine->enqueue(true);
    environment->scripted_per_rank_transfer_engine->enqueue(false);

    BlockTreeMatchResult result       = environment->cache->match(environment->keys);
    auto                 load_context = std::dynamic_pointer_cast<LoadAsyncContext>(result.async_context);
    ASSERT_NE(load_context, nullptr);
    ASSERT_EQ(load_context->loadDescs().size(), 2u);
    std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> request_targets;
    for (size_t desc_index = 0; desc_index < load_context->loadDescs().size(); ++desc_index) {
        const size_t              group_set_id = load_context->loadDescs()[desc_index].group_set_id;
        std::vector<BlockIdxType> targets;
        for (const DeviceBlockPoolPtr& pool : environment->groups[group_set_id]->devicePools()) {
            const BlockIdList blocks = pool->malloc(1).value();
            pool->incRef(blocks);
            targets.push_back(blocks.front());
            request_targets.emplace_back(pool, blocks.front());
        }
        load_context->setTargetBlocks(desc_index, std::move(targets));
    }

    ASSERT_TRUE(load_context->commit());
    load_context->waitDone();
    EXPECT_FALSE(load_context->success());
    EXPECT_EQ(environment->scripted_per_rank_transfer_engine->submittedBatchCount(), 2u);

    const auto resources = environment->resourcesForPathNode(0);
    ASSERT_EQ(resources.size(), 2u);
    for (const GroupSetResource& resource : resources) {
        EXPECT_FALSE(resource.hasTier(Tier::DEVICE));
        EXPECT_TRUE(resource.hasTier(Tier::DISK));
        EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);
    }
    for (const auto& [pool, block] : request_targets) {
        EXPECT_EQ(pool->refCount(block), 1u);
        pool->decRef(block);
        EXPECT_EQ(pool->freeBlocksNum(), options.usable_device_blocks);
    }

    result.async_context.reset();
    load_context.reset();
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, LoadStateMachineRejectsDuplicateTransitionAndRestoresSource) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    auto find_result = environment->cache->tree()->findNode(environment->keys);
    ASSERT_FALSE(find_result.empty());

    constexpr size_t  group_set_id = 0;
    GroupSetResource& resource     = find_result.back()->group_set_resources[group_set_id];

    ASSERT_TRUE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::IDLE, GroupSetTransferState::LOAD_PENDING));
    environment->cache->evictor_.suspendCandidate(find_result.back(), group_set_id, Tier::HOST);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOAD_PENDING);
    ASSERT_TRUE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);
    EXPECT_FALSE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::LOAD_PENDING, GroupSetTransferState::LOADING));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::LOADING);

    ASSERT_TRUE(environment->cache->loader_.changeTransferState(
        find_result.back(), group_set_id, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE));
    environment->cache->evictor_.admitCandidate(find_result.back(), group_set_id, Tier::HOST);
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::IDLE);

    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, ChangeTransferStateDoesNotOverwriteForeignTransferState) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }

    FullSWAEnvironmentOptions options;
    options.path_length = 1;
    options.enable_disk = false;
    auto environment    = FullSWAEnvironment::create(options);
    ASSERT_NE(environment, nullptr);

    environment->insertRequestPath();
    environment->releaseRequestRefs();
    environment->demoteAll(Tier::DEVICE);
    auto find_result = environment->cache->tree()->findNode(environment->keys);
    ASSERT_FALSE(find_result.empty());

    GroupSetResource& resource = find_result.back()->group_set_resources[0];
    resource.transfer_state    = GroupSetTransferState::DEMOTING;
    EXPECT_FALSE(environment->cache->loader_.changeTransferState(
        find_result.back(), 0, GroupSetTransferState::LOADING, GroupSetTransferState::IDLE));
    EXPECT_EQ(resource.transfer_state, GroupSetTransferState::DEMOTING);

    resource.transfer_state = GroupSetTransferState::IDLE;
    environment->reclaimAll();
    environment->expectFullyReclaimed();
}

class DeferredLoadEngine final: public PerRankBlockTransferEngine {
public:
    explicit DeferredLoadEngine(const std::vector<GroupSetPtr>& groups): PerRankBlockTransferEngine(groups) {}

    std::shared_ptr<AsyncContext> execute(TransferTask task) override {
        auto context = std::make_shared<TransferBatchAsyncContext>();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push_back(std::move(task));
            contexts_.push_back(context);
        }
        cv_.notify_all();
        return context;
    }

    bool waitForBatches(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [&] { return tasks_.size() >= count; });
    }

    size_t batchCount() {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }

    void complete(size_t first, size_t end, int corrupt_group = -1, bool whole_batch = false) {
        std::vector<std::pair<TransferTask, std::shared_ptr<TransferBatchAsyncContext>>> batches;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (size_t i = first; i < end; ++i) {
                batches.emplace_back(tasks_.at(i), contexts_.at(i));
            }
        }
        for (const auto& [task, context] : batches) {
            bool success = true;
            for (const auto& desc : task.descriptors()) {
                if (static_cast<int>(desc.group_set_id) == corrupt_group) {
                    desc.markCorrupted();
                    success = false;
                    if (!whole_batch) {
                        break;
                    }
                }
            }
            context->complete(success ? ErrorInfo::OkStatus() :
                                        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected corrupt record"));
        }
    }

private:
    std::mutex                                              mutex_;
    std::condition_variable                                 cv_;
    std::vector<TransferTask>                               tasks_;
    std::vector<std::shared_ptr<TransferBatchAsyncContext>> contexts_;
};

using RequestRefs = std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>>;

RequestRefs bindTargets(FullSWAEnvironment& env, const std::shared_ptr<LoadAsyncContext>& context) {
    RequestRefs refs;
    for (size_t i = 0; i < context->loadDescs().size(); ++i) {
        const auto&               desc   = context->loadDescs()[i];
        std::vector<BlockIdxType> blocks = desc.source_tier == Tier::DEVICE ? desc.source_blocks : desc.target_blocks;
        const auto&               pools  = env.groups[desc.group_set_id]->devicePools();
        if (desc.source_tier != Tier::DEVICE && !context->joinedLoads()[i]) {
            blocks.clear();
            for (const auto& pool : pools) {
                auto allocated = pool->malloc(1).value();
                pool->incRef(allocated);
                blocks.push_back(allocated.front());
            }
            context->setTargetBlocks(i, blocks);
        }
        for (size_t member = 0; member < pools.size(); ++member) {
            refs.emplace_back(pools[member], blocks[member]);
        }
    }
    return refs;
}

void releaseTargets(const RequestRefs& refs) {
    for (const auto& [pool, block] : refs) {
        pool->decRef(block);
    }
}

TEST(BlockTreeLoaderTest, SwaCorruptionDetachesItsSubtreeAndPreservesEarlierPrefix) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA not available";
    FullSWAEnvironmentOptions options;
    options.path_length = 4;
    options.enable_disk = false;
    auto env            = FullSWAEnvironment::create(options);
    ASSERT_NE(env, nullptr);
    env->insertRequestPath();
    env->releaseRequestRefs();
    env->demoteAll(Tier::DEVICE);
    auto engine                                        = std::make_shared<DeferredLoadEngine>(env->groups);
    env->cache->transfer_dispatcher_->per_rank_engine_ = engine;
    auto owner                                         = env->cache->match(env->keys).async_context;
    ASSERT_NE(owner, nullptr);
    auto refs = bindTargets(*env, owner);
    ASSERT_TRUE(owner->commit());
    ASSERT_TRUE(engine->waitForBatches(2));
    const auto bad = *std::find_if(
        owner->loadDescs().begin(), owner->loadDescs().end(), [](const auto& desc) { return desc.group_set_id == 1; });
    ASSERT_EQ(bad.path_index, 2u);
    const auto prefix    = env->cache->tree_->findNode({env->keys[0], env->keys[1]});
    const auto bad_block = bad.source_blocks.front();
    engine->complete(0, 2, 1);
    owner->waitDone();
    EXPECT_FALSE(owner->success());
    EXPECT_TRUE(env->cache->tree_->detached_nodes_.empty());
    EXPECT_EQ(env->cache->tree_->findNode(env->keys), prefix);
    EXPECT_FALSE(env->host_pools[1]->isAllocated(bad_block));
    EXPECT_EQ(env->cache->tree_->size(), 2u);
    for (const auto& pool : env->host_pools) {
        EXPECT_EQ(pool->usedBlocksNum(), 2u);
    }
    // Matching stops before the corrupt SWA subtree, including its healthy FULL records.
    auto later = env->cache->match(env->keys).async_context;
    ASSERT_NE(later, nullptr);
    EXPECT_EQ(later->matchedBlocks(), 2u);
    EXPECT_TRUE(later->abortPending());
    releaseTargets(refs);
    env->reclaimAll();
    env->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, SwaBatchCorruptionReclaimsOverlappingSubtreesOnce) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA not available";
    FullSWAEnvironmentOptions options;
    options.path_length = 4;
    options.enable_disk = false;
    auto env            = FullSWAEnvironment::create(options);
    ASSERT_NE(env, nullptr);
    env->insertRequestPath();
    env->releaseRequestRefs();
    env->demoteAll(Tier::DEVICE);
    auto engine                                        = std::make_shared<DeferredLoadEngine>(env->groups);
    env->cache->transfer_dispatcher_->per_rank_engine_ = engine;
    auto owner                                         = env->cache->match({env->keys[0], env->keys[1]}).async_context;
    ASSERT_NE(owner, nullptr);
    auto refs = bindTargets(*env, owner);
    ASSERT_TRUE(owner->commit());
    ASSERT_TRUE(engine->waitForBatches(2));
    engine->complete(0, 2, 1, /*whole_batch=*/true);
    owner->waitDone();
    EXPECT_FALSE(owner->success());
    size_t invalidated = 0;
    for (const auto& desc : owner->loadDescs()) {
        if (desc.group_set_id == 1) {
            ++invalidated;
            EXPECT_TRUE(desc.corrupted());
            EXPECT_FALSE(env->host_pools[1]->isAllocated(desc.source_blocks.front()));
        }
    }
    EXPECT_EQ(invalidated, 2u);
    EXPECT_EQ(env->cache->tree_->size(), 0u);
    EXPECT_TRUE(env->cache->tree_->detached_nodes_.empty());
    auto later = env->cache->match(env->keys);
    EXPECT_TRUE(later.async_context == nullptr || later.async_context->empty());
    releaseTargets(refs);
    env->reclaimAll();
    env->expectFullyReclaimed();
}

class BlockTreeLoaderCorruptionTest: public ::testing::TestWithParam<size_t> {};

TEST_P(BlockTreeLoaderCorruptionTest, CorruptionDetachesBeforeDescendantCopyAndAllowsReplacement) {
    if (!cudaAvailable())
        GTEST_SKIP() << "CUDA not available";
    FullSWAEnvironmentOptions options;
    options.path_length    = 3;
    options.enable_disk    = false;
    options.task_pool_size = 1;
    auto env               = FullSWAEnvironment::create(options);
    ASSERT_NE(env, nullptr);
    env->insertRequestPath();
    // An older request can still hold valid GPU blocks after the cache is demoted.
    env->demoteAll(Tier::DEVICE);
    auto engine                                        = std::make_shared<DeferredLoadEngine>(env->groups);
    env->cache->transfer_dispatcher_->per_rank_engine_ = engine;
    auto owner                                         = env->cache->match({env->keys[0], env->keys[1]}).async_context;
    ASSERT_NE(owner, nullptr);
    auto owner_refs = bindTargets(*env, owner);
    ASSERT_TRUE(owner->commit());
    ASSERT_TRUE(engine->waitForBatches(2));
    auto child = env->cache->match(env->keys).async_context;
    ASSERT_NE(child, nullptr);
    auto      child_refs = bindTargets(*env, child);
    TreeNode* old_leaf   = env->cache->tree_->findNode(env->keys).back();
    ASSERT_TRUE(child->commit());
    ASSERT_TRUE(engine->waitForBatches(4));
    // Multiple corrupt ancestors in the same batch detach the subtree only once.
    engine->complete(0, 2, GetParam(), /*whole_batch=*/true);
    owner->waitDone();
    EXPECT_FALSE(owner->success());
    EXPECT_FALSE(child->done());
    EXPECT_EQ(env->cache->tree_->size(), 0u);
    EXPECT_EQ(env->cache->tree_->detached_nodes_.size(), 1u);
    EXPECT_EQ(old_leaf->parent, nullptr);
    for (const auto& pool : env->host_pools) {
        EXPECT_EQ(pool->usedBlocksNum(), 1u);
    }
    auto retry = env->cache->match(env->keys);
    EXPECT_TRUE(retry.async_context == nullptr || retry.async_context->empty());
    std::vector<std::vector<GroupSetResource>> replacement_resources;
    for (size_t i = 0; i < env->keys.size(); ++i) {
        std::vector<GroupSetResource> resources(env->groups.size());
        for (size_t group = 0; group < env->groups.size(); ++group) {
            resources[group].device_blocks = env->request_blocks[group][i];
        }
        replacement_resources.push_back(std::move(resources));
    }
    env->cache->insert(env->keys, replacement_resources, Tier::DEVICE);
    auto replacement = env->cache->tree_->findNode(env->keys);
    ASSERT_EQ(replacement.size(), env->keys.size());
    EXPECT_NE(replacement.back(), old_leaf);
    auto matched = env->cache->match(env->keys);
    EXPECT_EQ(matched.matched_device_blocks, env->keys.size());
    env->releaseMatch(matched);
    // A second failure on an already detached descendant must not touch the replacement.
    engine->complete(2, 4, GetParam());
    child->waitDone();
    EXPECT_FALSE(child->success());
    EXPECT_EQ(env->cache->tree_->findNode(env->keys), replacement);
    EXPECT_TRUE(env->cache->tree_->detached_nodes_.empty());
    for (const auto& pool : env->host_pools) {
        EXPECT_EQ(pool->usedBlocksNum(), 0u);
    }
    for (size_t group = 0; group < env->groups.size(); ++group) {
        for (const auto& blocks : env->request_blocks[group]) {
            for (size_t member = 0; member < blocks.size(); ++member) {
                EXPECT_EQ(env->groups[group]->devicePools()[member]->refCount(blocks[member]), 2u);
            }
        }
    }
    releaseTargets(owner_refs);
    releaseTargets(child_refs);
    env->releaseRequestRefs();
    env->reclaimAll();
    env->expectFullyReclaimed();
}

TEST_P(BlockTreeLoaderCorruptionTest, CorruptionRacingPendingCommitReleasesEachReferenceOnce) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    FullSWAEnvironmentOptions options;
    options.path_length = 3;
    options.enable_disk = false;
    auto env            = FullSWAEnvironment::create(options);
    ASSERT_NE(env, nullptr);
    env->insertRequestPath();
    env->releaseRequestRefs();
    env->demoteAll(Tier::DEVICE);
    auto engine                                        = std::make_shared<DeferredLoadEngine>(env->groups);
    env->cache->transfer_dispatcher_->per_rank_engine_ = engine;
    auto owner                                         = env->cache->match({env->keys[0], env->keys[1]}).async_context;
    ASSERT_NE(owner, nullptr);
    auto owner_refs = bindTargets(*env, owner);
    ASSERT_TRUE(owner->commit());
    ASSERT_TRUE(engine->waitForBatches(2));
    auto pending = env->cache->match(env->keys).async_context;
    ASSERT_NE(pending, nullptr);
    auto               pending_refs = bindTargets(*env, pending);
    std::promise<void> start;
    auto               ready  = start.get_future().share();
    auto               commit = std::async(std::launch::async, [pending, ready] {
        ready.wait();
        return pending->commit();
    });
    auto               fail   = std::async(std::launch::async, [engine, ready, group = GetParam()] {
        ready.wait();
        engine->complete(0, 2, group);
    });
    start.set_value();
    EXPECT_EQ(commit.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    commit.get();  // Either commit or invalidation may claim the pending reservation.
    fail.get();
    owner->waitDone();
    env->cache->task_pool_->waitForIdle();
    EXPECT_EQ(env->cache->getStats().tree_node_count, 0u);
    // If commit won, its copy still owns the retired node until this completion.
    engine->complete(2, engine->batchCount());
    pending->waitDone();
    env->cache->task_pool_->waitForIdle();
    EXPECT_FALSE(owner->success());
    EXPECT_FALSE(pending->success());
    EXPECT_TRUE(env->cache->tree_->detached_nodes_.empty());
    releaseTargets(owner_refs);
    releaseTargets(pending_refs);
    env->expectFullyReclaimed();
}

TEST_P(BlockTreeLoaderCorruptionTest, CorruptionReclaimsIdleDescendantsOutsideTheLoadBatch) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    FullSWAEnvironmentOptions options;
    options.path_length = 4;
    options.enable_disk = false;
    auto env            = FullSWAEnvironment::create(options);
    ASSERT_NE(env, nullptr);
    env->insertRequestPath();
    env->releaseRequestRefs();
    env->demoteAll(Tier::DEVICE);
    auto engine                                        = std::make_shared<DeferredLoadEngine>(env->groups);
    env->cache->transfer_dispatcher_->per_rank_engine_ = engine;
    auto owner                                         = env->cache->match({env->keys[0], env->keys[1]}).async_context;
    ASSERT_NE(owner, nullptr);
    auto refs = bindTargets(*env, owner);
    ASSERT_TRUE(owner->commit());
    ASSERT_TRUE(engine->waitForBatches(2));

    // Both descriptors of the selected GroupSet fail; their idle descendants have no load completion of their own.
    engine->complete(0, 2, GetParam(), /*whole_batch=*/true);
    owner->waitDone();
    env->cache->task_pool_->waitForIdle();
    EXPECT_FALSE(owner->success());
    EXPECT_EQ(env->cache->tree_->size(), 0u);
    EXPECT_TRUE(env->cache->tree_->detached_nodes_.empty());
    EXPECT_TRUE(env->cache->loader_.load_join_registry_.records_.empty());
    for (const auto& pool : env->host_pools) {
        EXPECT_EQ(pool->usedBlocksNum(), 0u);
    }
    releaseTargets(refs);
    env->expectFullyReclaimed();
}

TEST_P(BlockTreeLoaderCorruptionTest, CorruptionCancelsPendingDescendantAndReclaimsItsNode) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    FullSWAEnvironmentOptions options;
    options.path_length = 3;
    options.enable_disk = false;
    auto env            = FullSWAEnvironment::create(options);
    ASSERT_NE(env, nullptr);
    env->insertRequestPath();
    env->releaseRequestRefs();
    env->demoteAll(Tier::DEVICE);
    auto engine                                        = std::make_shared<DeferredLoadEngine>(env->groups);
    env->cache->transfer_dispatcher_->per_rank_engine_ = engine;
    auto owner                                         = env->cache->match({env->keys[0], env->keys[1]}).async_context;
    ASSERT_NE(owner, nullptr);
    auto owner_refs = bindTargets(*env, owner);
    ASSERT_TRUE(owner->commit());
    ASSERT_TRUE(engine->waitForBatches(2));
    auto pending = env->cache->match(env->keys).async_context;
    ASSERT_NE(pending, nullptr);
    auto pending_refs = bindTargets(*env, pending);

    engine->complete(0, 2, GetParam(), /*whole_batch=*/true);
    owner->waitDone();
    env->cache->task_pool_->waitForIdle();
    EXPECT_FALSE(owner->success());
    EXPECT_TRUE(pending->done());
    EXPECT_FALSE(pending->success());
    EXPECT_EQ(engine->batchCount(), 2u);
    EXPECT_EQ(env->cache->tree_->size(), 0u);
    EXPECT_TRUE(env->cache->tree_->detached_nodes_.empty());
    EXPECT_TRUE(env->cache->loader_.load_join_registry_.records_.empty());
    releaseTargets(owner_refs);
    releaseTargets(pending_refs);
    env->expectFullyReclaimed();
}

INSTANTIATE_TEST_SUITE_P(FullAndSwa, BlockTreeLoaderCorruptionTest, ::testing::Values(size_t{0}, size_t{1}));

TEST(BlockTreeLoaderTest, BatchReclaimsOldDetachedNodesBeforeInvalidatingNewCorruption) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    FullSWAEnvironmentOptions options;
    options.path_length = 3;
    options.enable_disk = false;
    auto env            = FullSWAEnvironment::create(options);
    ASSERT_NE(env, nullptr);
    env->insertRequestPath();
    env->releaseRequestRefs();
    env->demoteAll(Tier::DEVICE);
    auto engine                                        = std::make_shared<DeferredLoadEngine>(env->groups);
    env->cache->transfer_dispatcher_->per_rank_engine_ = engine;
    auto owner                                         = env->cache->match(env->keys).async_context;
    ASSERT_NE(owner, nullptr);
    auto refs = bindTargets(*env, owner);
    ASSERT_TRUE(owner->commit());
    ASSERT_TRUE(engine->waitForBatches(2));
    {
        std::lock_guard<std::mutex> lock(env->cache->mutex_);
        TreeNode*                   leaf = env->cache->tree_->findNode(env->keys).back();
        // Retire the leaf while this batch still owns its transfer, then fail the remaining prefix too.
        const TransferDescriptor source(
            leaf, 0, 2, Tier::HOST, Tier::DEVICE, leaf->group_set_resources[0].getBlocks(Tier::HOST));
        EXPECT_FALSE(env->cache->evictor_.invalidateSources({source}));
        EXPECT_EQ(env->cache->tree_->detached_nodes_.size(), 1u);
        EXPECT_EQ(env->cache->tree_->size(), 2u);
    }
    engine->complete(0, 2, 0, /*whole_batch=*/true);
    owner->waitDone();
    env->cache->task_pool_->waitForIdle();
    EXPECT_FALSE(owner->success());
    EXPECT_TRUE(env->cache->tree_->detached_nodes_.empty());
    EXPECT_TRUE(env->cache->loader_.load_join_registry_.records_.empty());
    releaseTargets(refs);
    env->expectFullyReclaimed();
}

TEST(BlockTreeLoaderTest, ConcurrentFullAndSwaCorruptionCancelsPendingLoadsOnBothChains) {
    if (!cudaAvailable()) {
        GTEST_SKIP() << "CUDA not available";
    }
    FullSWAEnvironmentOptions options;
    options.path_length    = 3;
    options.enable_disk    = false;
    options.task_pool_size = 2;
    auto env               = FullSWAEnvironment::create(options);
    ASSERT_NE(env, nullptr);
    env->insertRequestPath();
    env->releaseRequestRefs();
    env->demoteAll(Tier::DEVICE);
    const CacheKeysType                        other_keys{200, 201, 202};
    std::vector<std::vector<GroupSetResource>> resources(3, std::vector<GroupSetResource>(env->groups.size()));
    for (auto& per_node : resources) {
        for (size_t group = 0; group < env->groups.size(); ++group) {
            per_node[group].host_block = env->groups[group]->allocateSingleBlock(Tier::HOST, BlockTreeRefType::CACHE);
            ASSERT_FALSE(isNullBlockIdx(per_node[group].host_block));
        }
    }
    ASSERT_TRUE(insertGroupSetResources(*env->cache, other_keys, resources));
    auto engine                                        = std::make_shared<DeferredLoadEngine>(env->groups);
    env->cache->transfer_dispatcher_->per_rank_engine_ = engine;
    auto first                                         = env->cache->match({env->keys[0], env->keys[1]}).async_context;
    auto second = env->cache->match({other_keys[0], other_keys[1]}).async_context;
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    auto first_refs  = bindTargets(*env, first);
    auto second_refs = bindTargets(*env, second);
    ASSERT_TRUE(first->commit());
    ASSERT_TRUE(engine->waitForBatches(2));
    ASSERT_TRUE(second->commit());
    ASSERT_TRUE(engine->waitForBatches(4));
    auto first_pending  = env->cache->match(env->keys).async_context;
    auto second_pending = env->cache->match(other_keys).async_context;
    ASSERT_NE(first_pending, nullptr);
    ASSERT_NE(second_pending, nullptr);
    auto first_pending_refs  = bindTargets(*env, first_pending);
    auto second_pending_refs = bindTargets(*env, second_pending);

    std::promise<void> start;
    auto               ready       = start.get_future().share();
    auto               fail_first  = std::async(std::launch::async, [engine, ready] {
        ready.wait();
        engine->complete(0, 2, 0, /*whole_batch=*/true);
    });
    auto               fail_second = std::async(std::launch::async, [engine, ready] {
        ready.wait();
        engine->complete(2, 4, 1, /*whole_batch=*/true);
    });
    start.set_value();
    fail_first.get();
    fail_second.get();
    first->waitDone();
    second->waitDone();
    env->cache->task_pool_->waitForIdle();
    EXPECT_FALSE(first->success());
    EXPECT_FALSE(second->success());
    EXPECT_TRUE(first_pending->done());
    EXPECT_TRUE(second_pending->done());
    EXPECT_FALSE(first_pending->success());
    EXPECT_FALSE(second_pending->success());
    EXPECT_EQ(engine->batchCount(), 4u);
    EXPECT_TRUE(env->cache->tree_->detached_nodes_.empty());
    EXPECT_TRUE(env->cache->loader_.load_join_registry_.records_.empty());
    releaseTargets(first_refs);
    releaseTargets(second_refs);
    releaseTargets(first_pending_refs);
    releaseTargets(second_pending_refs);
    env->expectFullyReclaimed();
}

}  // namespace
}  // namespace rtp_llm
