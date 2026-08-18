#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

#include "rtp_llm/cpp/cache/BlockReleaseBatch.h"

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/EvictionTaskRunner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::block_tree_cache_test {

void CallbackBarrier::enterAndWait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++entered_count_;
    cv_.notify_all();
    cv_.wait(lock, [this] { return released_; });
}

void CallbackBarrier::waitUntilEntered(size_t expected_count) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this, expected_count] { return entered_count_ >= expected_count; });
}

void CallbackBarrier::release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    cv_.notify_all();
}

const char* transferCopyActionName(TransferCopyAction action) {
    switch (action) {
        case TransferCopyAction::Succeed:
            return "succeed";
        case TransferCopyAction::Fail:
            return "fail";
        case TransferCopyAction::Throw:
            return "throw";
    }
    return "unknown";
}

ControlledPerRankBlockTransferEngine::ControlledPerRankBlockTransferEngine(const std::vector<GroupSetPtr>&  groups,
                                                                           TransferCopyAction               action,
                                                                           std::shared_ptr<CallbackBarrier> barrier):
    PerRankBlockTransferEngine(groups), action_(action), barrier_(std::move(barrier)) {}

std::shared_ptr<AsyncContext>
ControlledPerRankBlockTransferEngine::submit(const std::vector<TransferDescriptor>& descriptors) {
    submit_count_.fetch_add(1);
    if (barrier_ != nullptr) {
        barrier_->enterAndWait();
    }
    if (action_ == TransferCopyAction::Throw) {
        throw std::runtime_error("injected copy failure");
    }
    if (action_ == TransferCopyAction::Fail) {
        return std::make_shared<CompletedAsyncContext>(
            ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "injected copy failure"));
    }
    return PerRankBlockTransferEngine::submit(descriptors);
}

size_t ControlledPerRankBlockTransferEngine::submittedBatchCount() const {
    return submit_count_.load();
}

std::shared_ptr<HostBlockPool> makeHostPool(size_t payload_bytes, size_t usable_count) {
    return block_transfer_engine_test::makeHostPool(payload_bytes, usable_count, /*enable_pinned=*/true);
}

DiskBlockIOStatus MemoryDiskBlockIO::openAndPreallocate(const std::string&, size_t bytes, bool) {
    data_.assign(bytes, 0);
    return DiskBlockIOStatus::OK;
}

DiskBlockIOStatus MemoryDiskBlockIO::read(uint64_t offset, void* dst, size_t bytes) {
    if (offset + bytes > data_.size()) {
        return DiskBlockIOStatus::INVALID_SIZE;
    }
    std::memcpy(dst, data_.data() + offset, bytes);
    return DiskBlockIOStatus::OK;
}

DiskBlockIOStatus MemoryDiskBlockIO::write(uint64_t offset, const void* src, size_t bytes) {
    if (offset + bytes > data_.size()) {
        return DiskBlockIOStatus::INVALID_SIZE;
    }
    std::memcpy(data_.data() + offset, src, bytes);
    return DiskBlockIOStatus::OK;
}

DiskBlockIOStatus MemoryDiskBlockIO::read(const std::vector<DiskRead>& reads) {
    for (const auto& read_request : reads) {
        const auto status = read(read_request.offset, read_request.buffer, read_request.bytes);
        if (status != DiskBlockIOStatus::OK) {
            return status;
        }
    }
    return DiskBlockIOStatus::OK;
}

DiskBlockIOStatus MemoryDiskBlockIO::write(const std::vector<DiskWrite>& writes) {
    for (const auto& write_request : writes) {
        const auto status = write(write_request.offset, write_request.buffer, write_request.bytes);
        if (status != DiskBlockIOStatus::OK) {
            return status;
        }
    }
    return DiskBlockIOStatus::OK;
}

void MemoryDiskBlockIO::close() {}

std::string MemoryDiskBlockIO::debugString() const {
    return "MemoryDiskBlockIO";
}

std::shared_ptr<BlockTreeDiskBlockPool>
makeDiskPool(size_t payload_bytes, size_t usable_count, std::unique_ptr<DiskBlockIO> io) {
    return block_transfer_engine_test::makeDiskPool(
        payload_bytes, usable_count, "/tmp", std::move(io), "block_tree_cache_disk");
}

bool cudaAvailable() {
    try {
        return torch::cuda::is_available();
    } catch (const std::exception&) {
        return false;
    }
}

DeviceBlockPoolPtr
makeDevicePool(const std::vector<DeviceLayerBufferSpec>& specs, size_t usable_count, const std::string& pool_name) {
    const auto physical_block_count = usable_count + 1;

    auto config                     = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DEVICE;
    config->pool_name               = pool_name;
    config->physical_block_count    = physical_block_count;
    config->use_cuda_malloc_backing = false;

    size_t offset = 0;
    for (const auto& spec : specs) {
        MemoryLayoutConfig layout;
        layout.layer_num                = 1;
        layout.block_num                = static_cast<uint32_t>(physical_block_count);
        layout.dtype                    = TYPE_INT8;
        layout.kv_cache_offset_bytes    = offset;
        layout.kv_block_stride_bytes    = spec.kv_bytes;
        layout.kv_block_pool_size_bytes = physical_block_count * spec.kv_bytes;
        layout.block_stride_bytes       = spec.kv_bytes + spec.scale_bytes;
        layout.total_size_bytes         = layout.kv_block_pool_size_bytes;
        offset += layout.kv_block_pool_size_bytes;

        if (spec.scale_bytes > 0) {
            layout.enable_kv_scale          = true;
            layout.kv_scale_offset_bytes    = offset;
            layout.kv_scale_stride_bytes    = spec.scale_bytes;
            layout.kv_scale_pool_size_bytes = physical_block_count * spec.scale_bytes;
            layout.total_size_bytes += layout.kv_scale_pool_size_bytes;
            offset += layout.kv_scale_pool_size_bytes;
        }

        layout.local_head_num_kv          = 1;
        layout.seq_size_per_block         = 1;
        layout.kernel_blocks_per_kv_block = 1;
        config->memory_layouts.push_back(layout);
    }
    config->total_size_bytes = offset;

    auto pool = std::make_shared<DeviceBlockPool>(config);
    RTP_LLM_CHECK(pool->init());
    return pool;
}

BlockIdxType poolMalloc(IBlockPool& pool) {
    return block_transfer_engine_test::poolMalloc(pool);
}

MultiNodeBlocks allocateDeviceBlocksForTest(GroupSet& group_set, size_t count, BlockRefType ref_type) {
    MultiNodeBlocks resource(count);
    for (const auto& pool : group_set.devicePools()) {
        const auto blocks = pool->malloc(count);
        if (!blocks.has_value()) {
            for (size_t i = 0; i < count; ++i) {
                for (size_t pool_index = 0; pool_index < resource[i].size(); ++pool_index) {
                    group_set.devicePools()[pool_index]->decRef(resource[i][pool_index], ref_type);
                }
            }
            return {};
        }
        pool->incRef(*blocks, ref_type);
        for (size_t i = 0; i < count; ++i) {
            resource[i].push_back((*blocks)[i]);
        }
    }
    return resource;
}

void referenceDeviceBlocksForTest(GroupSet& group_set, const MultiNodeBlocks& blocks, BlockRefType ref_type) {
    for (const auto& node_blocks : blocks) {
        RTP_LLM_CHECK(node_blocks.size() == group_set.devicePools().size());
        for (size_t pool_index = 0; pool_index < node_blocks.size(); ++pool_index) {
            group_set.devicePools()[pool_index]->incRef(node_blocks[pool_index], ref_type);
        }
    }
}

void unreferenceDeviceBlocksForTest(GroupSet& group_set, const MultiNodeBlocks& blocks, BlockRefType ref_type) {
    for (const auto& node_blocks : blocks) {
        RTP_LLM_CHECK(node_blocks.size() == group_set.devicePools().size());
        for (size_t pool_index = 0; pool_index < node_blocks.size(); ++pool_index) {
            group_set.devicePools()[pool_index]->decRef(node_blocks[pool_index], ref_type);
        }
    }
}

MultiNodeResource makeMultiNodeResourceForTest(size_t                        group_set_id,
                                               Tier                          tier,
                                               const std::vector<TreeNode*>& nodes,
                                               const MultiNodeBlocks&        blocks) {
    RTP_LLM_CHECK(nodes.size() == blocks.size());
    MultiNodeResource resource{group_set_id, tier};
    for (size_t i = 0; i < nodes.size(); ++i) {
        RTP_LLM_CHECK(nodes[i] != nullptr);
        resource.node_blocks.emplace_back(nodes[i], blocks[i]);
    }
    return resource;
}

size_t unreferencedBlocksNum(const IBlockPool& pool) {
    std::lock_guard<std::mutex> lock(pool.mutex_);
    size_t                      count = 0;
    for (size_t block = 1; block < pool.allocated_.size(); ++block) {
        if (pool.allocated_[block] != 0 && pool.refcounts_[block] == 0) {
            ++count;
        }
    }
    return count;
}

size_t treeCachedBlocksNum(const IBlockPool& pool) {
    std::lock_guard<std::mutex> lock(pool.mutex_);
    size_t                      count = 0;
    for (size_t block = 1; block < pool.allocated_.size(); ++block) {
        if (pool.allocated_[block] != 0 && pool.refcounts_[block] > 0) {
            ++count;
        }
    }
    return count;
}

void releaseRequestRefsForTest(BlockTreeCache& cache, const std::vector<MultiNodeResource>& resources) {
    BlockReleaseBatch releases;
    const auto&       group_sets = cache.groupSets();
    for (const MultiNodeResource& resource : resources) {
        RTP_LLM_CHECK(resource.tier == Tier::DEVICE);
        RTP_LLM_CHECK(resource.group_set_id < group_sets.size());
        const GroupSetPtr& group_set = group_sets[resource.group_set_id];
        const auto&        group_ids = group_set->groupIds();
        const auto&        pools     = group_set->devicePools();
        RTP_LLM_CHECK(group_ids.size() == pools.size());
        for (const auto& [_, blocks] : resource.node_blocks) {
            RTP_LLM_CHECK(blocks.size() == pools.size());
            for (size_t member_group_id = 0; member_group_id < blocks.size(); ++member_group_id) {
                releases.append(group_ids[member_group_id],
                                pools[member_group_id]->decRefWithResult(
                                    {blocks[member_group_id]}, BlockRefType::REQUEST));
            }
        }
    }
    cache.onBlocksReleased(releases.finish());
}

void releaseDeviceBlocksAndNotify(BlockTreeCache&           cache,
                                  const DeviceBlockPoolPtr& pool,
                                  const BlockIdList&        blocks,
                                  BlockRefType              ref_type) {
    for (const GroupSetPtr& group_set : cache.groupSets()) {
        for (size_t member_group_id = 0; member_group_id < group_set->devicePools().size(); ++member_group_id) {
            if (group_set->devicePools()[member_group_id] != pool) {
                continue;
            }
            BlockReleaseBatch releases;
            releases.append(group_set->groupIds()[member_group_id], pool->decRefWithResult(blocks, ref_type));
            cache.onBlocksReleased(releases.finish());
            return;
        }
    }
    RTP_LLM_CHECK(false);
}

DeviceBlockPoolPtr makeStructuralDevicePool(size_t group_set_id) {
    constexpr size_t physical_block_count = 1024;
    constexpr size_t block_bytes          = 1;

    MemoryLayoutConfig layout;
    layout.layer_num                  = 1;
    layout.block_num                  = static_cast<uint32_t>(physical_block_count);
    layout.dtype                      = TYPE_INT8;
    layout.kv_cache_offset_bytes      = 0;
    layout.kv_block_stride_bytes      = block_bytes;
    layout.kv_block_pool_size_bytes   = physical_block_count * block_bytes;
    layout.block_stride_bytes         = block_bytes;
    layout.total_size_bytes           = layout.kv_block_pool_size_bytes;
    layout.local_head_num_kv          = 1;
    layout.seq_size_per_block         = 1;
    layout.kernel_blocks_per_kv_block = 1;

    auto config                     = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DEVICE;
    config->pool_name               = "block_tree_cache_test_" + std::to_string(group_set_id);
    config->physical_block_count    = physical_block_count;
    config->total_size_bytes        = layout.total_size_bytes;
    config->memory_layouts          = {layout};
    config->use_cuda_malloc_backing = false;

    auto pool = std::make_shared<DeviceBlockPool>(config);
    // Structural tree/eviction tests only exercise block-id ownership and do not
    // access payload memory. Mark the logical pool ready without allocating a
    // CUDA tensor so these tests remain runnable on CPU-only test hosts.
    pool->markInitialized();
    auto structural_blocks = pool->malloc(physical_block_count - 1);
    RTP_LLM_CHECK(structural_blocks.has_value());
    return pool;
}

namespace {

void prepareGroupSets(std::vector<GroupSetPtr>& group_sets) {
    const bool has_uninitialized = std::any_of(group_sets.begin(), group_sets.end(), [](const GroupSetPtr& group_set) {
        return group_set != nullptr && group_set->groupIds().empty();
    });
    if (!has_uninitialized) {
        return;
    }

    std::vector<GroupBase> groups;
    groups.reserve(group_sets.size());
    for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
        const GroupSetPtr& group_set = group_sets[group_set_id];
        RTP_LLM_CHECK(group_set != nullptr && group_set->groupIds().empty());

        CacheGroupType type               = CacheGroupType::FULL;
        size_t         seq_size_per_block = 1;
        if (const auto* swa = dynamic_cast<SWAGroupSet*>(group_set.get()); swa != nullptr) {
            type               = CacheGroupType::SWA;
            seq_size_per_block = swa->seqSizePerBlock();
        } else if (dynamic_cast<LinearGroupSet*>(group_set.get()) != nullptr) {
            type = CacheGroupType::LINEAR;
        }
        auto policy                = defaultCacheGroupPolicy(type);
        policy.enable_prefix_reuse = true;
        if (const auto* swa = dynamic_cast<SWAGroupSet*>(group_set.get()); swa != nullptr) {
            policy.sliding_window_size = static_cast<int>(swa->slidingWindowSize());
        }
        size_t payload_bytes = 1;
        if (group_set->hostPool() != nullptr) {
            payload_bytes = group_set->hostPool()->payloadBytes();
        } else if (group_set->diskPool() != nullptr) {
            payload_bytes = group_set->diskPool()->payloadBytes();
        }
        groups.push_back(block_transfer_engine_test::makeTestGroupBase(
            policy, {static_cast<int>(group_set_id)}, payload_bytes, 0, 128, seq_size_per_block));
    }

    auto topology = block_transfer_engine_test::makeTestTopology(std::move(groups));
    for (size_t group_set_id = 0; group_set_id < group_sets.size(); ++group_set_id) {
        group_sets[group_set_id]->initialize(group_set_id, topology, {group_set_id});
    }
}

}  // namespace

void prepareGroupSetsForTest(std::vector<GroupSetPtr>& group_sets) {
    prepareGroupSets(group_sets);
}

std::unique_ptr<BlockTreeCache> makeBlockTreeCacheForTest(std::vector<GroupSetPtr>          group_sets,
                                                          BlockTreeCacheConfig              config,
                                                          std::shared_ptr<StorageBackend>   storage_backend,
                                                          std::shared_ptr<BroadcastManager> broadcast_manager) {
    prepareGroupSets(group_sets);
    if (!config.enable_remote_cache) {
        storage_backend = nullptr;
    }
    std::shared_ptr<const CacheTopology>     storage_topology;
    std::vector<std::shared_ptr<IBlockPool>> storage_device_pools;
    StorageBackend::BufferResolver           storage_buffer_resolver;
    if (storage_backend != nullptr) {
        storage_topology = group_sets.front()->topologyPtr();
        std::vector<DeviceBlockPoolPtr> device_pools(storage_topology->groups().size());
        for (const auto& group_set : group_sets) {
            for (size_t member = 0; member < group_set->groupIds().size(); ++member) {
                device_pools[group_set->groupIds()[member]] = group_set->devicePools()[member];
            }
        }
        storage_device_pools.assign(device_pools.begin(), device_pools.end());
        storage_buffer_resolver = [topology     = storage_topology,
                                   device_pools = std::move(device_pools)](int layer_id, int group_id, int block_id) {
            const auto& layers = topology->groupById(static_cast<size_t>(group_id)).layer_ids;
            const auto  layer  = std::find(layers.begin(), layers.end(), layer_id);
            RTP_LLM_CHECK(layer != layers.end());
            return device_pools[static_cast<size_t>(group_id)]->convertIndexToBuffer(
                static_cast<int>(std::distance(layers.begin(), layer)), block_id);
        };
    }
    auto per_rank_engine = std::make_shared<PerRankBlockTransferEngine>(group_sets);
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine;
    if (broadcast_manager != nullptr) {
        multi_rank_engine = std::make_shared<MultiRankBlockTransferEngine>(group_sets, std::move(broadcast_manager));
    }
    auto transfer_dispatcher =
        std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine), std::move(multi_rank_engine));
    auto task_pool =
        std::make_unique<BlockTreeTaskPool>(static_cast<size_t>(config.task_pool_size), 1000, "BlockTreeCacheTaskPool");
    auto tree  = std::make_unique<BlockTree>(std::move(group_sets));
    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(config),
                                                  std::move(storage_backend),
                                                  std::move(transfer_dispatcher),
                                                  std::move(task_pool));
    if (cache->storageBackend()) {
        RTP_LLM_CHECK_WITH_INFO(
            cache->storageBackend()->init(std::move(storage_topology),
                                          std::move(storage_device_pools),
                                          std::move(storage_buffer_resolver),
                                          [cache_ptr = cache.get()](const std::vector<BlockReleaseReceipt>& receipts) {
                                              cache_ptr->onBlocksReleased(receipts);
                                          }),
            "StorageBackend init failed");
    }
    if (!cache->init()) {
        return nullptr;
    }
    return cache;
}

bool insertGroupSetResources(BlockTreeCache&                                   cache,
                             const CacheKeysType&                              cache_keys,
                             const std::vector<std::vector<GroupSetResource>>& resources) {
    BlockTree* tree = cache.tree();
    if (tree == nullptr) {
        return false;
    }
    const BlockTreeInsertResult insert_result = tree->insertNode(cache_keys, resources, /*collect_path=*/false);
    releaseLowerTierSeedRefs(tree->groupSets(), resources);
    cache.evictor_.onInserted(insert_result);
    return !insert_result.inserted_nodes.empty() || !insert_result.adopted_nodes.empty();
}

void releaseLowerTierSeedRefs(const std::vector<GroupSetPtr>&                   group_sets,
                              const std::vector<std::vector<GroupSetResource>>& resources) {
    for (const std::vector<GroupSetResource>& per_key_resources : resources) {
        for (size_t group_set_id = 0; group_set_id < per_key_resources.size(); ++group_set_id) {
            const GroupSetResource& resource = per_key_resources[group_set_id];
            const Tier              tier     = resource.getTopTier();
            if (tier != Tier::HOST && tier != Tier::DISK) {
                continue;
            }
            group_sets[group_set_id]->releaseSingleBlock(
                tier, resource.getBlocks(tier).front(), BlockRefType::BLOCK_CACHE);
        }
    }
}

void BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(
    BlockTreeCache& cache, PerRankBlockTransferEnginePtr per_rank_transfer_engine) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    if (per_rank_transfer_engine == nullptr) {
        ADD_FAILURE() << "test PerRankBlockTransferEngine must not be null";
        return;
    }
    if (cache.task_pool_->pending_tasks_.load() != 0 || cache.tree_->size() != 0) {
        ADD_FAILURE() << "test PerRankBlockTransferEngine must be installed before any cache work starts";
        return;
    }
    cache.transfer_dispatcher_->per_rank_engine_ = std::move(per_rank_transfer_engine);
}

void BlockTreeCacheTestPeer::setTierWatermarkForTest(BlockTreeCache& cache, Tier tier, double ratio) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    switch (tier) {
        case Tier::DEVICE:
            cache.config_.watermark_device.ratio = ratio;
            break;
        case Tier::HOST:
            cache.config_.watermark_host.ratio = ratio;
            break;
        case Tier::DISK:
            cache.config_.watermark_disk.ratio = ratio;
            break;
        default:
            break;
    }
}

size_t BlockTreeCacheTestPeer::pendingEvictionReleasesForTest(const BlockTreeCache& cache) {
    std::lock_guard<std::mutex> lock(cache.evictor_.pending_release_mutex_);
    size_t                      pending = 0;
    for (const auto& [_, count] : cache.evictor_.pending_release_counts_) {
        pending += count;
    }
    return pending;
}

void BlockTreeCacheTestPeer::runMaintenanceForTest(BlockTreeCache& cache) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    cache.checkWatermark();
}

void BlockTreeCacheTestPeer::beginStoreShutdownForTest(BlockTreeCache& cache) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    cache.storer_.stopAdmissionLocked();
}

bool BlockTreeCacheTestPeer::demoteOneForGroupSetForTest(BlockTreeCache& cache,
                                                         size_t          group_set_id,
                                                         Tier            tier,
                                                         bool            force_drop) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    if (!cache.config_.isTierEnabled(tier)) {
        return false;
    }
    return cache.evictor_.evictLocked(group_set_id, tier, force_drop);
}

int BlockTreeCacheTestPeer::reclaimBlocksForTest(BlockTreeCache& cache, size_t num_blocks, Tier tier) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    if (!cache.config_.isTierEnabled(tier)) {
        return 0;
    }

    int total_evicted = 0;
    for (size_t attempt = 0; attempt < num_blocks; ++attempt) {
        bool evicted = false;
        for (const GroupSetPtr& group_set : cache.tree_->groupSets()) {
            if (cache.evictor_.evictLocked(group_set->groupSetId(), tier, /*force_drop=*/true)) {
                evicted = true;
                break;
            }
        }
        if (!evicted) {
            break;
        }
        ++total_evicted;
    }
    return total_evicted;
}

BlockTreeCacheTestPeer::ScopedQueueRejectionGuard::ScopedQueueRejectionGuard(BlockTreeCache& cache):
    cache_(&cache), armed_(BlockTreeCacheTestPeer::armQueueRejectionForTest(cache)) {
    if (!armed_) {
        cache_ = nullptr;
    }
}

BlockTreeCacheTestPeer::ScopedQueueRejectionGuard::~ScopedQueueRejectionGuard() {
    (void)restore();
}

bool BlockTreeCacheTestPeer::ScopedQueueRejectionGuard::armed() const {
    return armed_;
}

bool BlockTreeCacheTestPeer::ScopedQueueRejectionGuard::restore() {
    if (!armed_ || cache_ == nullptr) {
        return false;
    }
    BlockTreeCache* cache = cache_;
    cache_                = nullptr;
    armed_                = false;
    return BlockTreeCacheTestPeer::restoreQueueAfterRejectionForTest(*cache);
}

int BlockTreeCacheTestPeer::pendingTasksForTest(const BlockTreeCache& cache) {
    return cache.task_pool_->pending_tasks_.load();
}

void BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(const BlockTreeCache& cache) {
    cache.task_pool_->waitForIdle();
}

bool BlockTreeCacheTestPeer::armQueueRejectionForTest(BlockTreeCache& cache) {
    waitForTaskPoolIdleForTest(cache);
    if (cache.task_pool_->pending_tasks_.load() != 0) {
        ADD_FAILURE() << "queue-rejection guard requires zero pending cache tasks";
        return false;
    }
    cache.task_pool_->shutdown();
    return true;
}

bool BlockTreeCacheTestPeer::restoreQueueAfterRejectionForTest(BlockTreeCache& cache) {
    try {
        auto replacement = std::make_unique<BlockTreeTaskPool>(
            static_cast<size_t>(cache.config_.task_pool_size), 1000, "BlockTreeCacheTaskPool");
        if (!replacement->start()) {
            ADD_FAILURE() << "queue-rejection guard failed to start replacement task pool";
            cache.task_pool_.reset();
            return false;
        }
        cache.task_pool_          = std::move(replacement);
        cache.evictor_.task_pool_ = cache.task_pool_.get();
        cache.loader_.task_pool_  = cache.task_pool_.get();
        cache.storer_.task_pool_  = cache.task_pool_.get();
        return true;
    } catch (const std::exception& error) {
        ADD_FAILURE() << "queue-rejection guard failed to restore thread pool: " << error.what();
    } catch (...) {
        ADD_FAILURE() << "queue-rejection guard failed to restore thread pool with unknown exception";
    }
    cache.task_pool_.reset();
    return false;
}

ScriptedPerRankBlockTransferEngine::ScriptedPerRankBlockTransferEngine(const std::vector<GroupSetPtr>& groups,
                                                                       bool perform_successful_transfers):
    PerRankBlockTransferEngine(groups), perform_successful_transfers_(perform_successful_transfers) {}

std::shared_ptr<AsyncContext>
ScriptedPerRankBlockTransferEngine::submit(const std::vector<TransferDescriptor>& descriptors) {
    bool success = true;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++submitted_batch_count_;
        descriptors_.insert(descriptors_.end(), descriptors.begin(), descriptors.end());
        if (!results_.empty()) {
            success = results_.front();
            results_.pop_front();
        }
    }
    if (success && perform_successful_transfers_) {
        return PerRankBlockTransferEngine::submit(descriptors);
    }
    if (success) {
        return std::make_shared<CompletedAsyncContext>(ErrorInfo::OkStatus());
    }
    return std::make_shared<CompletedAsyncContext>(
        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "scripted transfer failure"));
}

void ScriptedPerRankBlockTransferEngine::enqueue(bool success) {
    std::lock_guard<std::mutex> lock(mutex_);
    results_.push_back(success);
}

void ScriptedPerRankBlockTransferEngine::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    results_.clear();
    descriptors_.clear();
    submitted_batch_count_ = 0;
}

std::vector<TransferDescriptor> ScriptedPerRankBlockTransferEngine::descriptors() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return descriptors_;
}

size_t ScriptedPerRankBlockTransferEngine::submittedBatchCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return submitted_batch_count_;
}

size_t ScriptedPerRankBlockTransferEngine::submittedDescriptorCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return descriptors_.size();
}

namespace {

constexpr size_t kGroupPayloadBytes = 16;

uint8_t payloadPattern(size_t pool_id, size_t path_index) {
    return static_cast<uint8_t>(0x10 + pool_id * 0x20 + path_index);
}

void fillDeviceBlock(const DeviceBlockPoolPtr& pool, BlockIdxType block, uint8_t pattern) {
    for (const auto& buffer : pool->convertIndexToBuffer(0, block)) {
        auto view = torch::from_blob(buffer.addr,
                                     {static_cast<int64_t>(buffer.size_bytes)},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        view.fill_(pattern);
        // Materialize once on Host so the pattern is visible before an eviction worker
        // starts a copy on a different CUDA stream.
        (void)view.cpu();
    }
}

void expectDeviceBlock(const DeviceBlockPoolPtr& pool, BlockIdxType block, uint8_t pattern) {
    for (const auto& buffer : pool->convertIndexToBuffer(0, block)) {
        auto                view = torch::from_blob(buffer.addr,
                                                    {static_cast<int64_t>(buffer.size_bytes)},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        const torch::Tensor host = view.cpu();
        const auto*         data = host.data_ptr<uint8_t>();
        for (size_t index = 0; index < buffer.size_bytes; ++index) {
            EXPECT_EQ(data[index], pattern);
        }
    }
}

void expectBytes(const uint8_t* data, size_t bytes, uint8_t pattern) {
    ASSERT_NE(data, nullptr);
    for (size_t index = 0; index < bytes; ++index) {
        EXPECT_EQ(data[index], pattern);
    }
}

}  // namespace

namespace {

std::vector<TreeNode*> topologyPath(BlockTree& tree, const CacheKeysType& keys) {
    std::vector<TreeNode*> path;
    TreeNode*              current = tree.root();
    path.reserve(keys.size());
    for (CacheKeyType key : keys) {
        const auto child = current->children.find(key);
        if (child == current->children.end() || child->second == nullptr) {
            break;
        }
        current = child->second;
        path.push_back(current);
    }
    return path;
}

}  // namespace

FullSWAEnvironment::FullSWAEnvironment(FullSWAEnvironmentOptions options): options_(std::move(options)) {}

std::unique_ptr<FullSWAEnvironment> FullSWAEnvironment::create(const FullSWAEnvironmentOptions& options) {
    auto environment = std::unique_ptr<FullSWAEnvironment>(new FullSWAEnvironment(options));

    environment->device_pools = {
        makeDevicePool({{kGroupPayloadBytes, 0}}, options.usable_device_blocks, "p1_full_kv"),
        makeDevicePool({{kGroupPayloadBytes, 0}}, options.usable_device_blocks, "p1_full_aux"),
        makeDevicePool({{kGroupPayloadBytes, 0}}, options.usable_device_blocks, "p1_swa_kv"),
    };
    environment->host_pools = {
        makeHostPool(2 * kGroupPayloadBytes, options.usable_host_blocks),
        makeHostPool(kGroupPayloadBytes, options.usable_host_blocks),
    };
    if (options.enable_disk) {
        environment->disk_pools = {
            makeDiskPool(2 * kGroupPayloadBytes, options.usable_disk_blocks, std::make_unique<MemoryDiskBlockIO>()),
            makeDiskPool(kGroupPayloadBytes, options.usable_disk_blocks, std::make_unique<MemoryDiskBlockIO>()),
        };
    }

    auto full_policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    full_policy.enable_prefix_reuse = true;
    auto swa_policy                 = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_policy.enable_prefix_reuse  = true;
    swa_policy.sliding_window_size  = 2;
    environment->topology           = block_transfer_engine_test::makeTestTopology({
        block_transfer_engine_test::makeTestGroupBase(full_policy, {0}, kGroupPayloadBytes),
        block_transfer_engine_test::makeTestGroupBase(full_policy, {0}, kGroupPayloadBytes),
        block_transfer_engine_test::makeTestGroupBase(swa_policy, {0}, kGroupPayloadBytes),
    });

    auto full =
        block_transfer_engine_test::makeTestGroupSet(0,
                                                     environment->topology,
                                                     {0, 1},
                                                     {environment->device_pools[0], environment->device_pools[1]},
                                                     environment->host_pools[0],
                                                     options.enable_disk ? environment->disk_pools[0] : nullptr);
    auto swa            = block_transfer_engine_test::makeTestGroupSet(1,
                                                            environment->topology,
                                                                       {2},
                                                                       {environment->device_pools[2]},
                                                            environment->host_pools[1],
                                                            options.enable_disk ? environment->disk_pools[1] : nullptr);
    environment->groups = {full, swa};
    BlockTreeCacheConfig config;
    config.enable_device_cache = true;
    config.enable_host_cache   = true;
    config.enable_disk_cache   = options.enable_disk;

    environment->scripted_per_rank_transfer_engine =
        std::make_shared<ScriptedPerRankBlockTransferEngine>(environment->groups);

    std::vector<GroupSetPtr> cache_groups = environment->groups;
    environment->cache                    = makeBlockTreeCacheForTest(std::move(cache_groups), std::move(config));
    if (environment->cache == nullptr) {
        ADD_FAILURE() << "failed to initialize BlockTreeCache test environment";
        return nullptr;
    }
    BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(*environment->cache,
                                                                 environment->scripted_per_rank_transfer_engine);

    environment->keys.resize(options.path_length);
    std::iota(environment->keys.begin(), environment->keys.end(), static_cast<CacheKeyType>(100));
    environment->request_refs_released_.assign(2, true);
    return environment;
}

void FullSWAEnvironment::insertRequestPath() {
    ASSERT_TRUE(request_blocks.empty());
    request_blocks = {
        allocateDeviceBlocksForTest(*groups[0], options_.path_length, BlockRefType::REQUEST),
        allocateDeviceBlocksForTest(*groups[1], options_.path_length, BlockRefType::REQUEST),
    };
    ASSERT_EQ(request_blocks[0].size(), options_.path_length);
    ASSERT_EQ(request_blocks[1].size(), options_.path_length);
    request_refs_released_.assign(2, false);
    fillRequestPayloads();

    std::vector<std::vector<GroupSetResource>> resources(options_.path_length, std::vector<GroupSetResource>(2));
    for (size_t path_index = 0; path_index < options_.path_length; ++path_index) {
        resources[path_index][0].device_blocks = request_blocks[0][path_index];
        resources[path_index][1].device_blocks = request_blocks[1][path_index];
    }
    cache->insert(keys, resources, Tier::DEVICE);
}

void FullSWAEnvironment::releaseRequestRefs() {
    for (size_t group_id = 0; group_id < request_blocks.size(); ++group_id) {
        releaseRequestRefsForGroup(static_cast<int>(group_id));
    }
}

void FullSWAEnvironment::releaseRequestRefsForGroup(int group_id) {
    ASSERT_GE(group_id, 0);
    ASSERT_LT(static_cast<size_t>(group_id), request_blocks.size());
    if (request_refs_released_[static_cast<size_t>(group_id)]) {
        return;
    }
    const std::vector<TreeNode*> path = topologyPath(*cache->tree(), keys);
    ASSERT_EQ(path.size(), options_.path_length);
    MultiNodeResource released_blocks = makeMultiNodeResourceForTest(
        static_cast<size_t>(group_id), Tier::DEVICE, path, request_blocks[static_cast<size_t>(group_id)]);
    block_tree_cache_test::releaseRequestRefsForTest(*cache, {released_blocks});
    request_refs_released_[static_cast<size_t>(group_id)] = true;
}

void FullSWAEnvironment::releaseMatch(BlockTreeMatchResult& result) {
    block_tree_cache_test::releaseRequestRefsForTest(*cache, result.matched_device_resources);
    result.matched_device_resources.clear();
}

void FullSWAEnvironment::setTierWatermark(Tier tier, double ratio) {
    for (Tier candidate : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        BlockTreeCacheTestPeer::setTierWatermarkForTest(*cache, candidate, candidate == tier ? ratio : 0.0);
    }
}

void FullSWAEnvironment::runMaintenance() {
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
}

void FullSWAEnvironment::demoteAll(Tier tier) {
    setTierWatermark(tier, 0.01);
    for (size_t attempt = 0; attempt < options_.path_length * 4; ++attempt) {
        runMaintenance();
        bool source_present = false;
        for (size_t path_index = 0; path_index < options_.path_length; ++path_index) {
            for (const GroupSetResource& resource : resourcesForPathNode(path_index)) {
                source_present = source_present || resource.hasTier(tier);
            }
        }
        if (!source_present) {
            break;
        }
    }
    setTierWatermark(tier, 0.0);
}

void FullSWAEnvironment::reclaimAll() {
    releaseRequestRefs();
    for (Tier tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        for (size_t attempt = 0; attempt < options_.path_length * groups.size() * 4; ++attempt) {
            if (BlockTreeCacheTestPeer::reclaimBlocksForTest(*cache, 1, tier) == 0) {
                break;
            }
            block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
        }
    }
    block_tree_cache_test::BlockTreeCacheTestPeer::waitForTaskPoolIdleForTest(*cache);
}

bool FullSWAEnvironment::allResourcesAtTier(Tier tier) const {
    for (size_t path_index = 0; path_index < options_.path_length; ++path_index) {
        for (const GroupSetResource& resource : resourcesForPathNode(path_index)) {
            if (!resource.hasTier(tier)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<BlockIdxType> FullSWAEnvironment::blocksForDevicePool(size_t pool_id) const {
    std::vector<BlockIdxType> result;
    if (pool_id >= 3 || request_blocks.size() != 2) {
        return result;
    }
    const size_t group_id   = pool_id < 2 ? 0 : 1;
    const size_t pool_index = pool_id < 2 ? pool_id : 0;
    for (const auto& node_blocks : request_blocks[group_id]) {
        result.push_back(node_blocks[pool_index]);
    }
    return result;
}

std::vector<GroupSetResource> FullSWAEnvironment::resourcesForPathNode(size_t path_index) const {
    if (path_index >= keys.size()) {
        return {};
    }
    const std::vector<TreeNode*> path = topologyPath(*cache->tree(), keys);
    if (path_index >= path.size()) {
        return {};
    }
    return path[path_index]->group_set_resources;
}

void FullSWAEnvironment::fillRequestPayloads() {
    for (size_t pool_id = 0; pool_id < 3; ++pool_id) {
        const std::vector<BlockIdxType> blocks = blocksForDevicePool(pool_id);
        for (size_t path_index = 0; path_index < blocks.size(); ++path_index) {
            fillDeviceBlock(device_pools[pool_id], blocks[path_index], payloadPattern(pool_id, path_index));
        }
    }
}

void FullSWAEnvironment::expectPayloads() const {
    for (size_t path_index = 0; path_index < options_.path_length; ++path_index) {
        const std::vector<GroupSetResource> node_resources = resourcesForPathNode(path_index);
        ASSERT_EQ(node_resources.size(), 2u);
        for (size_t group_id = 0; group_id < node_resources.size(); ++group_id) {
            const GroupSetResource& resource = node_resources[group_id];
            if (resource.hasTier(Tier::DEVICE)) {
                const size_t device_pool_begin = group_id == 0 ? 0 : 2;
                for (size_t pool_index = 0; pool_index < resource.device_blocks.size(); ++pool_index) {
                    const size_t device_pool_id = device_pool_begin + pool_index;
                    expectDeviceBlock(device_pools[device_pool_id],
                                      resource.device_blocks[pool_index],
                                      payloadPattern(device_pool_id, path_index));
                }
            } else if (resource.hasTier(Tier::HOST)) {
                const auto   buffer            = host_pools[group_id]->blockBuffer(resource.host_block);
                const auto*  data              = static_cast<const uint8_t*>(buffer.addr);
                const size_t device_pool_begin = group_id == 0 ? 0 : 2;
                const size_t device_pool_count = group_id == 0 ? 2 : 1;
                for (size_t local = 0; local < device_pool_count; ++local) {
                    expectBytes(data + local * kGroupPayloadBytes,
                                kGroupPayloadBytes,
                                payloadPattern(device_pool_begin + local, path_index));
                }
            } else if (resource.hasTier(Tier::DISK)) {
                std::vector<uint8_t> data(disk_pools[group_id]->payloadBytes());
                ASSERT_EQ(disk_pools[group_id]->read(resource.disk_slot, data.data(), data.size()), BlockIOStatus::OK);
                const size_t device_pool_begin = group_id == 0 ? 0 : 2;
                const size_t device_pool_count = group_id == 0 ? 2 : 1;
                for (size_t local = 0; local < device_pool_count; ++local) {
                    expectBytes(data.data() + local * kGroupPayloadBytes,
                                kGroupPayloadBytes,
                                payloadPattern(device_pool_begin + local, path_index));
                }
            } else {
                ADD_FAILURE() << "node " << path_index << " group " << group_id << " has no resident tier";
            }
        }
    }
}

void FullSWAEnvironment::expectPoolFreeCounts(const std::vector<size_t>& device_free,
                                              const std::vector<size_t>& host_free,
                                              const std::vector<size_t>& disk_free) const {
    ASSERT_EQ(device_free.size(), device_pools.size());
    ASSERT_EQ(host_free.size(), host_pools.size());
    ASSERT_EQ(disk_free.size(), disk_pools.size());
    for (size_t index = 0; index < device_pools.size(); ++index) {
        EXPECT_EQ(device_pools[index]->freeBlocksNum(), device_free[index]);
    }
    for (size_t index = 0; index < host_pools.size(); ++index) {
        EXPECT_EQ(host_pools[index]->freeBlocksNum(), host_free[index]);
    }
    for (size_t index = 0; index < disk_pools.size(); ++index) {
        EXPECT_EQ(disk_pools[index]->freeBlocksNum(), disk_free[index]);
    }
}

void FullSWAEnvironment::expectFullyReclaimed() const {
    EXPECT_EQ(cache->getStats().tree_node_count, 0u);
    expectPoolFreeCounts(std::vector<size_t>(device_pools.size(), options_.usable_device_blocks),
                         std::vector<size_t>(host_pools.size(), options_.usable_host_blocks),
                         std::vector<size_t>(disk_pools.size(), options_.usable_disk_blocks));
}

}  // namespace rtp_llm::block_tree_cache_test
