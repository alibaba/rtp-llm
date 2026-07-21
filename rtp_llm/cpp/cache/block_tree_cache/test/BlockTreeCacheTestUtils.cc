#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtils.h"

#include "rtp_llm/cpp/cache/block_tree_cache/test/BlockTreeCacheTestUtil.h"

#include <gtest/gtest.h>

#include <cstring>
#include <exception>
#include <numeric>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/test/CopyEngineTestUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::block_tree_cache_test {

std::shared_ptr<HostBlockPool> makeHostPool(size_t payload_bytes, size_t usable_count) {
    return copy_engine_test::makeHostPool(payload_bytes, usable_count, /*enable_pinned=*/true);
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

std::shared_ptr<DiskBlockPool>
makeDiskPool(size_t payload_bytes, size_t usable_count, std::unique_ptr<DiskBlockIO> io) {
    return copy_engine_test::makeDiskPool(payload_bytes, usable_count, "/tmp", std::move(io), "block_tree_cache_disk");
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
    return copy_engine_test::poolMalloc(pool);
}

std::unique_ptr<BlockTreeCache> makeBlockTreeCacheForTest(std::unique_ptr<BlockTree>        tree,
                                                          std::vector<ComponentGroupPtr>    component_groups,
                                                          std::vector<Component>            components,
                                                          BlockTreeCacheConfig              config,
                                                          std::shared_ptr<StorageBackend>   storage_backend,
                                                          std::shared_ptr<BroadcastManager> broadcast_manager) {
    return BlockTreeCacheTestUtil::makeBlockTreeCache(std::move(tree),
                                                      std::move(component_groups),
                                                      std::move(components),
                                                      std::move(config),
                                                      std::move(storage_backend),
                                                      std::move(broadcast_manager));
}

void BlockTreeCacheTestPeer::setCopyEngineForTest(BlockTreeCache& cache, CopyEnginePtr copy_engine) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    if (copy_engine == nullptr) {
        ADD_FAILURE() << "test CopyEngine must not be null";
        return;
    }
    if (cache.pending_tasks_.load() != 0 || cache.tree_->nodeCount() != 0) {
        ADD_FAILURE() << "test CopyEngine must be installed before any cache work starts";
        return;
    }
    cache.copy_engine_ = std::move(copy_engine);
}

void BlockTreeCacheTestPeer::runMaintenanceForTest(BlockTreeCache& cache) {
    std::lock_guard<std::mutex> lock(cache.mutex_);
    cache.checkWatermark();
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
    return cache.pending_tasks_.load();
}

bool BlockTreeCacheTestPeer::armQueueRejectionForTest(BlockTreeCache& cache) {
    cache.waitForPendingTasks();
    if (cache.pending_tasks_.load() != 0) {
        ADD_FAILURE() << "queue-rejection guard requires zero pending cache tasks";
        return false;
    }
    if (cache.thread_pool_ == nullptr) {
        ADD_FAILURE() << "queue-rejection guard requires an initialized thread pool";
        return false;
    }
    cache.thread_pool_->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
    cache.thread_pool_->join();
    return true;
}

bool BlockTreeCacheTestPeer::restoreQueueAfterRejectionForTest(BlockTreeCache& cache) {
    try {
        auto replacement = std::make_shared<autil::LockFreeThreadPool>(
            static_cast<size_t>(cache.config_.eviction_thread_pool_size), 1000, nullptr, "BlockTreeEvictionPool");
        if (!replacement->start()) {
            ADD_FAILURE() << "queue-rejection guard failed to start replacement thread pool";
            cache.thread_pool_.reset();
            return false;
        }
        cache.thread_pool_ = std::move(replacement);
        return true;
    } catch (const std::exception& error) {
        ADD_FAILURE() << "queue-rejection guard failed to restore thread pool: " << error.what();
    } catch (...) {
        ADD_FAILURE() << "queue-rejection guard failed to restore thread pool with unknown exception";
    }
    cache.thread_pool_.reset();
    return false;
}

ScriptedCopyEngine::ScriptedCopyEngine(const std::vector<ComponentGroupPtr>& groups,
                                       const std::vector<Component>&         components):
    CopyEngine(groups, std::make_shared<const std::vector<Component>>(components)) {}

TransferHandle ScriptedCopyEngine::submit(const TransferDescriptor& descriptor) {
    CopyStatus status = CopyStatus::OK;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        descriptors_.push_back(descriptor);
        if (!statuses_.empty()) {
            status = statuses_.front();
            statuses_.pop_front();
        }
    }
    if (status == CopyStatus::OK) {
        return CopyEngine::submit(descriptor);
    }
    return TransferHandle::completed(status);
}

void ScriptedCopyEngine::enqueue(CopyStatus status) {
    std::lock_guard<std::mutex> lock(mutex_);
    statuses_.push_back(status);
}

void ScriptedCopyEngine::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    statuses_.clear();
    descriptors_.clear();
}

std::vector<TransferDescriptor> ScriptedCopyEngine::descriptors() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return descriptors_;
}

size_t ScriptedCopyEngine::submitCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return descriptors_.size();
}

namespace {

constexpr size_t kComponentBytes = 16;

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

FullSWAEnvironment::FullSWAEnvironment(FullSWAEnvironmentOptions options): options_(std::move(options)) {}

std::unique_ptr<FullSWAEnvironment> FullSWAEnvironment::create(const FullSWAEnvironmentOptions& options) {
    auto environment = std::unique_ptr<FullSWAEnvironment>(new FullSWAEnvironment(options));

    environment->device_pools = {
        makeDevicePool({{kComponentBytes, 0}}, options.usable_device_blocks, "p1_full_kv"),
        makeDevicePool({{kComponentBytes, 0}}, options.usable_device_blocks, "p1_full_aux"),
        makeDevicePool({{kComponentBytes, 0}}, options.usable_device_blocks, "p1_swa_kv"),
    };
    environment->host_pools = {
        makeHostPool(2 * kComponentBytes, options.usable_host_blocks),
        makeHostPool(kComponentBytes, options.usable_host_blocks),
    };
    if (options.enable_disk) {
        environment->disk_pools = {
            makeDiskPool(2 * kComponentBytes, options.usable_disk_blocks, std::make_unique<MemoryDiskBlockIO>()),
            makeDiskPool(kComponentBytes, options.usable_disk_blocks, std::make_unique<MemoryDiskBlockIO>()),
        };
    }

    environment->components.resize(3);
    environment->components[0]      = copy_engine_test::makeSchemaComponent(0, 0, "full_kv", {kComponentBytes});
    environment->components[1]      = copy_engine_test::makeSchemaComponent(1, 0, "full_aux", {kComponentBytes});
    environment->components[2]      = copy_engine_test::makeSchemaComponent(2, 1, "swa_kv", {kComponentBytes});
    environment->components[2].type = CacheGroupType::SWA;

    auto full                = std::make_shared<FullComponentGroup>();
    full->component_group_id = 0;
    full->setDevicePools({environment->device_pools[0], environment->device_pools[1]});
    full->setHostPool(environment->host_pools[0]);
    if (options.enable_disk) {
        full->setDiskPool(environment->disk_pools[0]);
    }
    RTP_LLM_CHECK(full->finalizeLayout({0, 1}, environment->components));

    auto swa                = std::make_shared<SWAComponentGroup>(/*sliding_window_size=*/2,
                                                   /*seq_size_per_block=*/1);
    swa->component_group_id = 1;
    swa->setDevicePools({environment->device_pools[2]});
    swa->setHostPool(environment->host_pools[1]);
    if (options.enable_disk) {
        swa->setDiskPool(environment->disk_pools[1]);
    }
    RTP_LLM_CHECK(swa->finalizeLayout({2}, environment->components));
    environment->groups = {full, swa};

    BlockTreeCacheConfig config;
    config.enable_device_cache     = true;
    config.enable_memory_cache     = true;
    config.enable_disk_cache       = options.enable_disk;
    config.enable_load_back        = options.enable_load_back;
    config.enable_reverse_eviction = options.enable_reverse_eviction;

    environment->scripted_copy_engine =
        std::make_shared<ScriptedCopyEngine>(environment->groups, environment->components);

    std::vector<ComponentGroupPtr> cache_groups = environment->groups;
    environment->cache                          = makeBlockTreeCacheForTest(
        std::make_unique<BlockTree>(2), std::move(cache_groups), environment->components, std::move(config));
    if (environment->cache == nullptr) {
        ADD_FAILURE() << "failed to initialize BlockTreeCache test environment";
        return nullptr;
    }
    BlockTreeCacheTestPeer::setCopyEngineForTest(*environment->cache, environment->scripted_copy_engine);

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

    std::vector<std::vector<GroupSlot>> slots(options_.path_length, std::vector<GroupSlot>(3));
    for (size_t path_index = 0; path_index < options_.path_length; ++path_index) {
        slots[path_index][0].device_blocks = {request_blocks[0].per_node[path_index][0]};
        slots[path_index][1].device_blocks = {request_blocks[0].per_node[path_index][1]};
        slots[path_index][2].device_blocks = {request_blocks[1].per_node[path_index][0]};
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
    BlockTreeFindResult find = cache->tree()->findNode(keys);
    ASSERT_EQ(find.path.size(), options_.path_length);
    GroupBlockSet released_blocks = request_blocks[static_cast<size_t>(group_id)];
    released_blocks.nodes         = find.path;
    cache->releaseMatchedBlocks({released_blocks});
    request_refs_released_[static_cast<size_t>(group_id)] = true;
}

void FullSWAEnvironment::releaseMatch(BlockTreeMatchResult& result) {
    cache->releaseMatchedBlocks(result.matched_block_sets);
    result.matched_block_sets.clear();
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
            for (const GroupSlot& slot : slotsForPathNode(path_index)) {
                source_present = source_present || slot.has_value(tier);
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
            if (cache->reclaimBlocks(1, tier) == 0) {
                break;
            }
            cache->waitForPendingTasks();
        }
    }
    cache->waitForPendingTasks();
}

bool FullSWAEnvironment::allSlotsAtTier(Tier tier) const {
    for (size_t path_index = 0; path_index < options_.path_length; ++path_index) {
        for (const GroupSlot& slot : slotsForPathNode(path_index)) {
            if (!slot.has_value(tier)) {
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

std::vector<GroupSlot> FullSWAEnvironment::slotsForPathNode(size_t path_index) const {
    if (path_index >= keys.size()) {
        return {};
    }
    CacheKeysType       prefix(keys.begin(), keys.begin() + static_cast<ptrdiff_t>(path_index + 1));
    BlockTreeFindResult find = cache->tree()->findNode(prefix);
    if (find.matched_node == nullptr) {
        return {};
    }
    return find.matched_node->group_slots;
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
        const std::vector<GroupSlot> node_slots = slotsForPathNode(path_index);
        ASSERT_EQ(node_slots.size(), 2u);
        for (size_t group_id = 0; group_id < node_slots.size(); ++group_id) {
            const GroupSlot& slot = node_slots[group_id];
            if (slot.has_value(Tier::DEVICE)) {
                const size_t tag_begin = group_id == 0 ? 0 : 2;
                for (size_t pool_index = 0; pool_index < slot.device_blocks.size(); ++pool_index) {
                    const size_t tag_id = tag_begin + pool_index;
                    expectDeviceBlock(
                        device_pools[tag_id], slot.device_blocks[pool_index], payloadPattern(tag_id, path_index));
                }
            } else if (slot.has_value(Tier::HOST)) {
                const auto   buffer    = host_pools[group_id]->blockBuffer(slot.host_block);
                const auto*  data      = static_cast<const uint8_t*>(buffer.addr);
                const size_t tag_begin = group_id == 0 ? 0 : 2;
                const size_t tag_count = group_id == 0 ? 2 : 1;
                for (size_t local = 0; local < tag_count; ++local) {
                    expectBytes(
                        data + local * kComponentBytes, kComponentBytes, payloadPattern(tag_begin + local, path_index));
                }
            } else if (slot.has_value(Tier::DISK)) {
                std::vector<uint8_t> data(disk_pools[group_id]->payloadBytes());
                ASSERT_EQ(disk_pools[group_id]->read(slot.disk_slot, data.data(), data.size()), BlockIOStatus::OK);
                const size_t tag_begin = group_id == 0 ? 0 : 2;
                const size_t tag_count = group_id == 0 ? 2 : 1;
                for (size_t local = 0; local < tag_count; ++local) {
                    expectBytes(data.data() + local * kComponentBytes,
                                kComponentBytes,
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
