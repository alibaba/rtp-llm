#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <numeric>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::block_tree_cache_test {

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

namespace {

DeviceBlockPoolPtr makeStructuralDevicePool(const std::string& tag) {
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
    config->pool_name               = "block_tree_cache_test_" + tag;
    config->physical_block_count    = physical_block_count;
    config->total_size_bytes        = layout.total_size_bytes;
    config->memory_layouts          = {layout};
    config->use_cuda_malloc_backing = false;

    auto pool = std::make_shared<DeviceBlockPool>(config);
    RTP_LLM_CHECK(pool->init());
    auto structural_blocks = pool->malloc(physical_block_count - 1);
    RTP_LLM_CHECK(structural_blocks.has_value());
    return pool;
}

void prepareGroupSets(std::vector<GroupSetPtr>& group_sets) {
    const bool has_uninitialized = std::any_of(group_sets.begin(), group_sets.end(), [](const GroupSetPtr& group_set) {
        return group_set != nullptr && group_set->topology() == nullptr;
    });
    if (!has_uninitialized) {
        return;
    }

    std::vector<block_transfer_engine_test::TestGroupSpec> specs;
    specs.reserve(group_sets.size());
    for (size_t group_set_index = 0; group_set_index < group_sets.size(); ++group_set_index) {
        const GroupSetPtr& group_set = group_sets[group_set_index];
        RTP_LLM_CHECK(group_set != nullptr && group_set->topology() == nullptr);

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
        specs.push_back({"tag_" + std::to_string(group_set_index),
                         policy,
                         {static_cast<int>(group_set_index)},
                         payload_bytes,
                         0,
                         128,
                         seq_size_per_block});
    }

    auto topology = block_transfer_engine_test::makeTestTopology(std::move(specs));
    for (size_t group_set_index = 0; group_set_index < group_sets.size(); ++group_set_index) {
        const std::string tag = "tag_" + std::to_string(group_set_index);
        group_sets[group_set_index]->initialize(
            group_set_index, topology, {group_set_index}, {makeStructuralDevicePool(tag)});
    }
}

}  // namespace

std::unique_ptr<BlockTreeCache> makeBlockTreeCacheForTest(std::unique_ptr<BlockTree>        tree,
                                                          std::vector<GroupSetPtr>          group_sets,
                                                          BlockTreeCacheConfig              config,
                                                          std::shared_ptr<StorageBackend>   storage_backend,
                                                          std::shared_ptr<BroadcastManager> broadcast_manager) {
    prepareGroupSets(group_sets);
    auto per_rank_engine = std::make_shared<PerRankBlockTransferEngine>(group_sets);
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine;
    if (broadcast_manager != nullptr) {
        multi_rank_engine = std::make_shared<MultiRankBlockTransferEngine>(group_sets, std::move(broadcast_manager));
    }
    auto transfer_dispatcher =
        std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine), std::move(multi_rank_engine));
    auto task_pool = std::make_unique<BlockTreeTaskPool>(
        static_cast<size_t>(config.eviction_thread_pool_size), 1000, "BlockTreeEvictionPool");
    auto cache = std::make_unique<BlockTreeCache>(std::move(tree),
                                                  std::move(group_sets),
                                                  std::move(config),
                                                  std::move(storage_backend),
                                                  std::move(transfer_dispatcher),
                                                  std::move(task_pool));
    if (!cache->init()) {
        return nullptr;
    }
    return cache;
}

bool insertGroupSetSlots(BlockTreeCache&                                   cache,
                         TreeNode*                                         parent,
                         const CacheKeysType&                              cache_keys,
                         const std::vector<std::vector<GroupSetResource>>& resources) {
    BlockTree* tree = cache.tree();
    if (tree == nullptr) {
        return false;
    }
    const BlockTreeInsertResult     insert_result = tree->insertNode(parent, cache_keys, resources);
    const std::vector<GroupSetPtr>& group_sets    = cache.groupSets();
    for (const BlockTreeInsertedNode& inserted : insert_result.inserted_nodes) {
        TreeNode* node = inserted.node;
        if (node == nullptr) {
            continue;
        }
        for (const GroupSetPtr& group : group_sets) {
            if (group == nullptr) {
                continue;
            }
            const size_t group_set_id = group->groupSetId();
            if (group_set_id >= node->group_set_resources.size()) {
                continue;
            }
            GroupSetResource& resource = node->group_set_resources[group_set_id];
            if (!group->hasCompleteDeviceValue(resource)) {
                continue;
            }
            const auto blocks = group->getBlocks(resource, Tier::DEVICE);
            group->referenceBlocks(MultiNodeResource{group_set_id, Tier::DEVICE, {blocks}}, BlockRefType::BLOCK_CACHE);
        }
    }
    for (const BlockTreeAdoptedSlot& adopted : insert_result.adopted_slots) {
        if (adopted.node == nullptr || adopted.group_set_id >= group_sets.size()
            || group_sets[adopted.group_set_id] == nullptr
            || adopted.group_set_id >= adopted.node->group_set_resources.size()) {
            continue;
        }
        const GroupSetPtr& group = group_sets[adopted.group_set_id];
        const auto blocks = group->getBlocks(adopted.node->group_set_resources[adopted.group_set_id], Tier::DEVICE);
        if (!blocks.empty()) {
            group->referenceBlocks(MultiNodeResource{adopted.group_set_id, Tier::DEVICE, {blocks}},
                                   BlockRefType::BLOCK_CACHE);
        }
    }
    cache.evictor_.onInsertCommitted(insert_result);
    return insert_result.leaf != nullptr;
}

void BlockTreeCacheTestPeer::setPerRankBlockTransferEngineForTest(
    BlockTreeCache& cache, PerRankBlockTransferEnginePtr per_rank_transfer_engine) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    if (per_rank_transfer_engine == nullptr) {
        ADD_FAILURE() << "test PerRankBlockTransferEngine must not be null";
        return;
    }
    if (cache.task_pool_ == nullptr || cache.task_pool_->pending_tasks_.load() != 0 || cache.tree_->nodeCount() != 0) {
        ADD_FAILURE() << "test PerRankBlockTransferEngine must be installed before any cache work starts";
        return;
    }
    cache.transfer_dispatcher_->per_rank_engine_ = std::move(per_rank_transfer_engine);
}

void BlockTreeCacheTestPeer::runMaintenanceForTest(BlockTreeCache& cache) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    cache.checkWatermark();
}

bool BlockTreeCacheTestPeer::demoteOneForGroupForTest(BlockTreeCache& cache, size_t group_set_id, Tier tier) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    if (!cache.config_.isTierEnabled(tier)) {
        return false;
    }
    auto eviction_move = cache.evictor_.chooseVictim(group_set_id, tier);
    if (!eviction_move.has_value()) {
        return false;
    }
    return cache.evictor_.submitLocked(*eviction_move);
}

int BlockTreeCacheTestPeer::reclaimBlocksForTest(BlockTreeCache& cache, size_t num_blocks, Tier tier) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    if (!cache.config_.isTierEnabled(tier)) {
        return 0;
    }

    int total_evicted = 0;
    for (size_t attempt = 0; attempt < num_blocks; ++attempt) {
        auto eviction_move = cache.evictor_.chooseVictim(tier);
        if (!eviction_move.has_value()) {
            break;
        }

        // Tests use this entry to trigger eviction state transitions without
        // exposing a direct-reclaim operation on the production cache API.
        eviction_move->target_tier = Tier::NONE;
        if (cache.evictor_.submitLocked(*eviction_move)) {
            ++total_evicted;
        }
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
    return cache.task_pool_ == nullptr ? 0 : cache.task_pool_->pending_tasks_.load();
}

bool BlockTreeCacheTestPeer::armQueueRejectionForTest(BlockTreeCache& cache) {
    cache.waitForPendingTasks();
    if (cache.task_pool_ != nullptr && cache.task_pool_->pending_tasks_.load() != 0) {
        ADD_FAILURE() << "queue-rejection guard requires zero pending cache tasks";
        return false;
    }
    if (cache.task_pool_ == nullptr) {
        ADD_FAILURE() << "queue-rejection guard requires an initialized task pool";
        return false;
    }
    cache.task_pool_->shutdown();
    return true;
}

bool BlockTreeCacheTestPeer::restoreQueueAfterRejectionForTest(BlockTreeCache& cache) {
    try {
        auto replacement = std::make_unique<BlockTreeTaskPool>(
            static_cast<size_t>(cache.config_.eviction_thread_pool_size), 1000, "BlockTreeEvictionPool");
        if (!replacement->start()) {
            ADD_FAILURE() << "queue-rejection guard failed to start replacement task pool";
            cache.task_pool_.reset();
            return false;
        }
        cache.task_pool_ = std::move(replacement);
        return true;
    } catch (const std::exception& error) {
        ADD_FAILURE() << "queue-rejection guard failed to restore thread pool: " << error.what();
    } catch (...) {
        ADD_FAILURE() << "queue-rejection guard failed to restore thread pool with unknown exception";
    }
    cache.task_pool_.reset();
    return false;
}

ScriptedPerRankBlockTransferEngine::ScriptedPerRankBlockTransferEngine(const std::vector<GroupSetPtr>& groups):
    PerRankBlockTransferEngine(groups) {}

std::shared_ptr<AsyncContext> ScriptedPerRankBlockTransferEngine::submit(const TransferDescriptor& descriptor) {
    bool success = true;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        descriptors_.push_back(descriptor);
        if (!results_.empty()) {
            success = results_.front();
            results_.pop_front();
        }
    }
    if (success) {
        return PerRankBlockTransferEngine::submit(descriptor);
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
}

std::vector<TransferDescriptor> ScriptedPerRankBlockTransferEngine::descriptors() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return descriptors_;
}

size_t ScriptedPerRankBlockTransferEngine::submitCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return descriptors_.size();
}

namespace {

constexpr size_t kGroupPayloadBytes = 16;

uint8_t payloadPattern(size_t tag_id, size_t path_index) {
    return static_cast<uint8_t>(0x10 + tag_id * 0x20 + path_index);
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
        {"tag_0", full_policy, {0}, kGroupPayloadBytes, 0},
        {"tag_1", full_policy, {0}, kGroupPayloadBytes, 0},
        {"tag_2", swa_policy, {0}, kGroupPayloadBytes, 0},
    });

    auto full = block_transfer_engine_test::makeTestGroupSet(
        0, environment->topology, {0, 1}, {environment->device_pools[0], environment->device_pools[1]});
    full->setHostPool(environment->host_pools[0]);
    if (options.enable_disk) {
        full->setDiskPool(environment->disk_pools[0]);
    }
    auto swa =
        block_transfer_engine_test::makeTestGroupSet(1, environment->topology, {2}, {environment->device_pools[2]});
    swa->setHostPool(environment->host_pools[1]);
    if (options.enable_disk) {
        swa->setDiskPool(environment->disk_pools[1]);
    }
    environment->groups = {full, swa};
    BlockTreeCacheConfig config;
    config.enable_device_cache     = true;
    config.enable_memory_cache     = true;
    config.enable_disk_cache       = options.enable_disk;
    config.enable_load             = options.enable_load;
    config.enable_reverse_eviction = options.enable_reverse_eviction;

    environment->scripted_per_rank_transfer_engine =
        std::make_shared<ScriptedPerRankBlockTransferEngine>(environment->groups);

    std::vector<GroupSetPtr> cache_groups = environment->groups;
    environment->cache =
        makeBlockTreeCacheForTest(std::make_unique<BlockTree>(2), std::move(cache_groups), std::move(config));
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
        groups[0]->allocateBlocks(Tier::DEVICE, options_.path_length, BlockRefType::REQUEST),
        groups[1]->allocateBlocks(Tier::DEVICE, options_.path_length, BlockRefType::REQUEST),
    };
    ASSERT_EQ(request_blocks[0].per_node.size(), options_.path_length);
    ASSERT_EQ(request_blocks[1].per_node.size(), options_.path_length);
    request_refs_released_.assign(2, false);
    fillRequestPayloads();

    std::vector<std::vector<GroupSetResource>> slots(options_.path_length, std::vector<GroupSetResource>(2));
    for (size_t path_index = 0; path_index < options_.path_length; ++path_index) {
        slots[path_index][0].device_blocks = request_blocks[0].per_node[path_index];
        slots[path_index][1].device_blocks = request_blocks[1].per_node[path_index];
    }
    cache->insert(nullptr, keys, slots);
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
    MultiNodeResource released_blocks = request_blocks[static_cast<size_t>(group_id)];
    released_blocks.tree_nodes        = path;
    cache->releaseMatchedResources({released_blocks});
    request_refs_released_[static_cast<size_t>(group_id)] = true;
}

void FullSWAEnvironment::releaseMatch(BlockTreeMatchResult& result) {
    cache->releaseMatchedResources(result.matched_resources);
    result.matched_resources.clear();
}

void FullSWAEnvironment::setTierWatermark(Tier tier, double ratio) {
    for (Tier candidate : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        cache->setTierWatermark(candidate, candidate == tier ? ratio : 0.0, 0);
    }
}

void FullSWAEnvironment::runMaintenance() {
    BlockTreeCacheTestPeer::runMaintenanceForTest(*cache);
    cache->waitForPendingTasks();
}

void FullSWAEnvironment::demoteAll(Tier tier) {
    setTierWatermark(tier, 0.01);
    for (size_t attempt = 0; attempt < options_.path_length * 4; ++attempt) {
        runMaintenance();
        bool source_present = false;
        for (size_t path_index = 0; path_index < options_.path_length; ++path_index) {
            for (const GroupSetResource& slot : slotsForPathNode(path_index)) {
                source_present = source_present || slot.hasTier(tier);
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
            cache->waitForPendingTasks();
        }
    }
    cache->waitForPendingTasks();
}

bool FullSWAEnvironment::allSlotsAtTier(Tier tier) const {
    for (size_t path_index = 0; path_index < options_.path_length; ++path_index) {
        for (const GroupSetResource& slot : slotsForPathNode(path_index)) {
            if (!slot.hasTier(tier)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<BlockIdxType> FullSWAEnvironment::blocksForTag(size_t tag_id) const {
    std::vector<BlockIdxType> result;
    if (tag_id >= 3 || request_blocks.size() != 2) {
        return result;
    }
    const size_t group_id   = tag_id < 2 ? 0 : 1;
    const size_t pool_index = tag_id < 2 ? tag_id : 0;
    for (const auto& node_blocks : request_blocks[group_id].per_node) {
        result.push_back(node_blocks[pool_index]);
    }
    return result;
}

std::vector<GroupSetResource> FullSWAEnvironment::slotsForPathNode(size_t path_index) const {
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
    for (size_t tag_id = 0; tag_id < 3; ++tag_id) {
        const std::vector<BlockIdxType> blocks = blocksForTag(tag_id);
        for (size_t path_index = 0; path_index < blocks.size(); ++path_index) {
            fillDeviceBlock(device_pools[tag_id], blocks[path_index], payloadPattern(tag_id, path_index));
        }
    }
}

void FullSWAEnvironment::expectPayloads() const {
    for (size_t path_index = 0; path_index < options_.path_length; ++path_index) {
        const std::vector<GroupSetResource> node_slots = slotsForPathNode(path_index);
        ASSERT_EQ(node_slots.size(), 2u);
        for (size_t group_id = 0; group_id < node_slots.size(); ++group_id) {
            const GroupSetResource& slot = node_slots[group_id];
            if (slot.hasTier(Tier::DEVICE)) {
                const size_t tag_begin = group_id == 0 ? 0 : 2;
                for (size_t pool_index = 0; pool_index < slot.device_blocks.size(); ++pool_index) {
                    const size_t tag_id = tag_begin + pool_index;
                    expectDeviceBlock(
                        device_pools[tag_id], slot.device_blocks[pool_index], payloadPattern(tag_id, path_index));
                }
            } else if (slot.hasTier(Tier::HOST)) {
                const auto   buffer    = host_pools[group_id]->blockBuffer(slot.host_block);
                const auto*  data      = static_cast<const uint8_t*>(buffer.addr);
                const size_t tag_begin = group_id == 0 ? 0 : 2;
                const size_t tag_count = group_id == 0 ? 2 : 1;
                for (size_t local = 0; local < tag_count; ++local) {
                    expectBytes(data + local * kGroupPayloadBytes,
                                kGroupPayloadBytes,
                                payloadPattern(tag_begin + local, path_index));
                }
            } else if (slot.hasTier(Tier::DISK)) {
                std::vector<uint8_t> data(disk_pools[group_id]->payloadBytes());
                ASSERT_EQ(disk_pools[group_id]->read(slot.disk_slot, data.data(), data.size()), BlockIOStatus::OK);
                const size_t tag_begin = group_id == 0 ? 0 : 2;
                const size_t tag_count = group_id == 0 ? 2 : 1;
                for (size_t local = 0; local < tag_count; ++local) {
                    expectBytes(data.data() + local * kGroupPayloadBytes,
                                kGroupPayloadBytes,
                                payloadPattern(tag_begin + local, path_index));
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
