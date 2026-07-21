#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_set>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTransferConverter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

// AsyncContext for load_back: waits until all copy tasks complete.
class LoadBackAsyncContext: public AsyncContext {
public:
    void addTask() {
        remaining_++;
    }
    void onTaskComplete(bool ok) {
        if (!ok)
            all_success_.store(false);
        if (--remaining_ <= 0) {
            std::lock_guard<std::mutex> lk(mu_);
            cv_.notify_all();
        }
    }
    void waitDone() override {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return remaining_.load() <= 0; });
    }
    bool done() const override {
        return remaining_.load() <= 0;
    }
    bool success() const override {
        return all_success_.load();
    }

private:
    std::atomic<int>        remaining_{0};
    std::atomic<bool>       all_success_{true};
    std::mutex              mu_;
    std::condition_variable cv_;
};

}  // anonymous namespace

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>         tree,
                               std::vector<ComponentGroupPtr>     component_groups,
                               std::vector<Component>             components,
                               BlockTreeCacheConfig               config,
                               std::shared_ptr<StorageBackend>    storage_backend,
                               std::shared_ptr<BroadcastManager>  broadcast_manager,
                               std::vector<DeviceKVCacheGroupPtr> per_tag_device_groups,
                               std::vector<PerTagMapping>         per_tag_mapping):
    config_(std::move(config)),
    tree_(std::move(tree)),
    component_groups_(std::move(component_groups)),
    components_(std::make_shared<const std::vector<Component>>(std::move(components))),
    per_tag_device_groups_(std::move(per_tag_device_groups)),
    per_tag_mapping_(std::move(per_tag_mapping)),
    load_back_ticket_registry_(std::make_shared<LoadBackTicketRegistry>(
        [this](const std::vector<PendingLoadBackItem>& items) { return commitLoadBack(items); },
        [this](const std::vector<PendingLoadBackItem>& items) { abortLoadBack(items); })),
    storage_backend_(std::move(storage_backend)),
    broadcast_manager_(std::move(broadcast_manager)),
    evictor_(
        component_groups_,
        [this](const TransferDescriptor& descriptor) { return executeTransfer(descriptor); },
        config_.enable_reverse_eviction) {}

bool BlockTreeCache::init() {
    if (initialized_) {
        RTP_LLM_LOG_ERROR("BlockTreeCache::init: cache is already initialized");
        return false;
    }
    if (!validateConfiguration()) {
        RTP_LLM_LOG_ERROR("BlockTreeCache::init: invalid configuration");
        return false;
    }
    if (!initDeviceGroupIds()) {
        return false;
    }
    if (!evictor_.init(config_.device_eviction_policy, config_.host_eviction_policy, config_.disk_eviction_policy)) {
        RTP_LLM_LOG_ERROR("BlockTreeCache::init: failed to initialize BlockTreeEvictor");
        return false;
    }
    copy_engine_ = std::make_shared<CopyEngine>(component_groups_, components_);

    thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        static_cast<size_t>(config_.eviction_thread_pool_size), 1000, nullptr, "BlockTreeEvictionPool");
    if (!thread_pool_->start()) {
        RTP_LLM_LOG_ERROR("BlockTreeCache::init: failed to start eviction thread pool, size=%d",
                          config_.eviction_thread_pool_size);
        thread_pool_.reset();
        return false;
    }
    RTP_LLM_LOG_INFO("BlockTreeCache: initialized with %zu component groups, %zu components, "
                     "pool_threads=%d, storage_backend=%s, "
                     "device=%s, host=%s, disk=%s, remote=%s",
                     component_groups_.size(),
                     components_->size(),
                     config_.eviction_thread_pool_size,
                     storage_backend_ ? "enabled" : "null",
                     config_.enable_device_cache ? "on" : "off",
                     config_.enable_memory_cache ? "on" : "off",
                     config_.enable_disk_cache ? "on" : "off",
                     config_.enable_remote_cache ? "on" : "off");
    for (const ComponentGroupPtr& component_group : component_groups_) {
        RTP_LLM_LOG_INFO("BlockTreeCache:   group[%d] type=%s host_pool=%s disk_pool=%s",
                         component_group->component_group_id,
                         cacheGroupTypeName(component_group->group_type),
                         component_group->hostPool() ? "enabled" : "null",
                         component_group->diskPool() ? "enabled" : "null");
    }
    // Wire eviction through the complete per-tag registry, including NON_REUSABLE tags.
    for (DeviceKVCacheGroupPtr& device_group : per_tag_device_groups_) {
        if (device_group != nullptr) {
            device_group->setEvictionCallback(
                [this](int group_id, size_t num_blocks) { return evictForGroup(group_id, num_blocks); });
        }
    }
    initialized_ = true;
    return true;
}

bool BlockTreeCache::initDeviceGroupIds() {
    device_group_ids_.clear();
    device_group_ids_.resize(component_groups_.size());
    const bool per_tag_registries_absent = per_tag_mapping_.empty() && per_tag_device_groups_.empty();
    for (size_t group_id = 0; group_id < component_groups_.size(); ++group_id) {
        const ComponentGroupPtr& component_group = component_groups_[group_id];
        if (component_group == nullptr || component_group->component_group_id != static_cast<int>(group_id)
            || component_group->devicePoolCount() == 0) {
            RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupIds: invalid component group, group=%zu", group_id);
            return false;
        }
        if (!per_tag_registries_absent) {
            device_group_ids_[group_id].assign(component_group->devicePoolCount(), -1);
        }
    }
    if (per_tag_registries_absent) {
        return true;
    }

    for (size_t device_group_id = 0; device_group_id < per_tag_mapping_.size(); ++device_group_id) {
        const PerTagMapping& mapping = per_tag_mapping_[device_group_id];
        if (mapping.component_group_id == -1) {
            if (mapping.local_pool_index != -1) {
                RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupIds: invalid non-reusable mapping, gid=%zu",
                                  device_group_id);
                return false;
            }
            continue;
        }
        if (mapping.component_group_id < 0 || mapping.local_pool_index < 0) {
            RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupIds: negative mapping index, gid=%zu", device_group_id);
            return false;
        }
        const size_t group_id      = static_cast<size_t>(mapping.component_group_id);
        const size_t local_pool_id = static_cast<size_t>(mapping.local_pool_index);
        if (group_id >= device_group_ids_.size() || local_pool_id >= device_group_ids_[group_id].size()
            || device_group_ids_[group_id][local_pool_id] != -1) {
            RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupIds: invalid mapping, gid=%zu group=%zu local=%zu",
                              device_group_id,
                              group_id,
                              local_pool_id);
            return false;
        }
        device_group_ids_[group_id][local_pool_id] = static_cast<int>(device_group_id);
    }

    for (size_t group_id = 0; group_id < device_group_ids_.size(); ++group_id) {
        const std::vector<int>& device_group_ids = device_group_ids_[group_id];
        for (int device_group_id : device_group_ids) {
            if (device_group_id < 0) {
                RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupIds: incomplete mapping, group=%zu", group_id);
                return false;
            }
        }
    }
    return true;
}

bool BlockTreeCache::validateConfiguration() const {
    if (tree_ == nullptr || component_groups_.empty() || components_ == nullptr) {
        RTP_LLM_LOG_ERROR("BlockTreeCache: tree, component groups, and component registry must be initialized");
        return false;
    }
    if (config_.enable_disk_cache && !config_.enable_memory_cache) {
        RTP_LLM_LOG_ERROR("BlockTreeCache: disk cache requires memory cache");
        return false;
    }
    if (config_.enable_load_back && !config_.enable_memory_cache) {
        RTP_LLM_LOG_ERROR("BlockTreeCache: load back requires memory cache");
        return false;
    }

    for (size_t group_index = 0; group_index < component_groups_.size(); ++group_index) {
        const ComponentGroupPtr& group = component_groups_[group_index];
        if (group == nullptr || group->component_group_id != static_cast<int>(group_index)) {
            RTP_LLM_LOG_ERROR("BlockTreeCache: component group must be non-null and indexed by id, index=%zu",
                              group_index);
            return false;
        }
        if (!group->hasLayout()) {
            RTP_LLM_LOG_ERROR("BlockTreeCache: component group %zu has no finalized layout", group_index);
            return false;
        }

        const auto& membership = group->componentIndices();
        const auto& layout     = group->layout();
        if (membership.empty() || membership.size() != layout.componentCount()
            || membership.size() != group->devicePoolCount()) {
            RTP_LLM_LOG_ERROR("BlockTreeCache: group %zu membership/layout/pool counts differ: %zu/%zu/%zu",
                              group_index,
                              membership.size(),
                              layout.componentCount(),
                              group->devicePoolCount());
            return false;
        }

        std::unordered_set<std::string> tags;
        size_t                          slice_index     = 0;
        size_t                          expected_offset = 0;
        for (size_t component_position = 0; component_position < membership.size(); ++component_position) {
            const int component_index = membership[component_position];
            if (component_index < 0 || static_cast<size_t>(component_index) >= components_->size()) {
                RTP_LLM_LOG_ERROR(
                    "BlockTreeCache: group %zu has invalid component index %d", group_index, component_index);
                return false;
            }
            const Component& component = (*components_)[static_cast<size_t>(component_index)];
            if (component.component_id != component_index
                || component.component_group_id != static_cast<int>(group_index) || component.tag.empty()
                || !tags.insert(component.tag).second || component.layerCount() == 0
                || component.model_layer_ids.size() != component.layerCount()) {
                RTP_LLM_LOG_ERROR("BlockTreeCache: invalid component binding index=%d group=%zu pool_position=%zu",
                                  component_index,
                                  group_index,
                                  component_position);
                return false;
            }
            for (size_t layer_index = 0; layer_index < component.layerCount(); ++layer_index) {
                if (slice_index >= layout.slices().size()) {
                    RTP_LLM_LOG_ERROR("BlockTreeCache: group %zu layout has too few slices", group_index);
                    return false;
                }
                const auto&  slice = layout.slices()[slice_index++];
                const size_t bytes = component.layerBytes(layer_index);
                if (bytes == 0 || bytes > std::numeric_limits<size_t>::max() - expected_offset
                    || slice.component_idx != component_position || slice.layer_idx != layer_index
                    || slice.offset_bytes != expected_offset) {
                    RTP_LLM_LOG_ERROR("BlockTreeCache: group %zu layout drift at component=%zu layer=%zu",
                                      group_index,
                                      component_position,
                                      layer_index);
                    return false;
                }
                expected_offset += bytes;
            }
        }
        if (slice_index != layout.slices().size() || expected_offset != layout.payloadBytes()) {
            RTP_LLM_LOG_ERROR("BlockTreeCache: group %zu layout slice count or payload drift", group_index);
            return false;
        }

        const auto host_pool = group->hostPool();
        const auto disk_pool = group->diskPool();
        if (config_.enable_memory_cache && host_pool == nullptr) {
            RTP_LLM_LOG_ERROR("BlockTreeCache: memory cache group %zu has no host pool", group_index);
            return false;
        }
        if (config_.enable_disk_cache && disk_pool == nullptr) {
            RTP_LLM_LOG_ERROR("BlockTreeCache: disk cache group %zu has no disk pool", group_index);
            return false;
        }
        if (host_pool != nullptr && host_pool->payloadBytes() != layout.payloadBytes()) {
            RTP_LLM_LOG_ERROR("BlockTreeCache: group %zu host/layout payload mismatch: %zu/%zu",
                              group_index,
                              host_pool->payloadBytes(),
                              layout.payloadBytes());
            return false;
        }
        if (disk_pool != nullptr && disk_pool->payloadBytes() != layout.payloadBytes()) {
            RTP_LLM_LOG_ERROR("BlockTreeCache: group %zu disk/layout payload mismatch: %zu/%zu",
                              group_index,
                              disk_pool->payloadBytes(),
                              layout.payloadBytes());
            return false;
        }
    }
    return true;
}

bool BlockTreeCache::validateDeviceGroupIdsForComponentGroup(int                     component_group_id,
                                                             const std::vector<int>& device_group_ids) const {
    if (!initialized_ || component_group_id < 0
        || static_cast<size_t>(component_group_id) >= device_group_ids_.size()) {
        RTP_LLM_LOG_WARNING("BlockTreeCache: invalid component group mapping request, group=%d", component_group_id);
        return false;
    }
    const std::vector<int>& expected = device_group_ids_[static_cast<size_t>(component_group_id)];
    if (device_group_ids != expected) {
        RTP_LLM_LOG_WARNING("BlockTreeCache: device group mapping mismatch, group=%d expected=%zu actual=%zu",
                            component_group_id,
                            expected.size(),
                            device_group_ids.size());
        return false;
    }
    return true;
}

BlockTreeCache::~BlockTreeCache() {
    RTP_LLM_LOG_INFO("BlockTreeCache: destroying, closing load-back tickets...");
    load_back_ticket_registry_->shutdown();
    RTP_LLM_LOG_INFO("BlockTreeCache: load-back tickets closed, waiting for pending tasks...");
    waitForPendingTasks();
    if (thread_pool_) {
        thread_pool_->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
        thread_pool_->join();
    }
    if (initialized_) {
        drainTreeHolds();
    }
    RTP_LLM_LOG_INFO("BlockTreeCache: destroyed");
}

void BlockTreeCache::drainTreeHolds() {
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK_WITH_INFO(tree_ != nullptr && tree_->root() != nullptr,
                            "BlockTreeCache::drainTreeHolds: tree and root must be valid");

    const auto drain_node = [this](TreeNode* node) {
        RTP_LLM_CHECK_WITH_INFO(node != nullptr, "BlockTreeCache::drainTreeHolds: node must be valid");
        RTP_LLM_CHECK_WITH_INFO(node->group_slots.size() == component_groups_.size(),
                                "BlockTreeCache::drainTreeHolds: slot count mismatch, slots=%zu groups=%zu",
                                node->group_slots.size(),
                                component_groups_.size());

        for (size_t component_group_index = 0; component_group_index < component_groups_.size();
             ++component_group_index) {
            const ComponentGroupPtr& component_group = component_groups_[component_group_index];
            RTP_LLM_CHECK_WITH_INFO(component_group != nullptr
                                        && component_group->component_group_id
                                               == static_cast<int>(component_group_index),
                                    "BlockTreeCache::drainTreeHolds: component group is not indexed by id, index=%zu",
                                    component_group_index);

            GroupSlot&                      slot          = node->group_slots[component_group_index];
            const std::vector<BlockIdxType> device_blocks = slot.device_blocks;
            if (!device_blocks.empty()) {
                RTP_LLM_CHECK_WITH_INFO(
                    device_blocks.size() == component_group->devicePoolCount(),
                    "BlockTreeCache::drainTreeHolds: device slot/pool count mismatch, group=%zu slots=%zu pools=%zu",
                    component_group_index,
                    device_blocks.size(),
                    component_group->devicePoolCount());
                // Keep shutdown symmetric with referenceBlocks/unreferenceBlocks:
                // pool-less structural slots carry no hold, while real pools are released exactly once.
                component_group->unreferenceBlocks(
                    GroupBlockSet{static_cast<int>(component_group_index), Tier::DEVICE, {device_blocks}});
                std::fill(slot.device_blocks.begin(), slot.device_blocks.end(), NULL_BLOCK_IDX);
            }

            if (!isNullBlockIdx(slot.host_block)) {
                const BlockIdxType host_block = slot.host_block;
                component_group->unreferenceBlocks(
                    GroupBlockSet{static_cast<int>(component_group_index), Tier::HOST, {{host_block}}});
                slot.host_block = NULL_BLOCK_IDX;
            }

            if (!isNullBlockIdx(slot.disk_slot)) {
                const BlockIdxType disk_block = slot.disk_slot;
                component_group->unreferenceBlocks(
                    GroupBlockSet{static_cast<int>(component_group_index), Tier::DISK, {{disk_block}}});
                slot.disk_slot = NULL_BLOCK_IDX;
            }

            slot.transfer_state = SlotTransferState::IDLE;
        }
    };

    drain_node(tree_->root());
    for (const std::unique_ptr<TreeNode>& node : tree_->nodes()) {
        drain_node(node.get());
    }
}

CopyStatus BlockTreeCache::executeTransfer(const TransferDescriptor& descriptor) {
    if (!copy_engine_) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::executeTransfer: copy engine is not initialized");
        return CopyStatus::INVALID_ARGS;
    }

    TransferHandle handle = copy_engine_->submit(descriptor);
    return handle.status();
}

bool BlockTreeCache::broadcastTransfer(const ::MemoryOperationRequestPB& request, int timeout_ms) const {
    if (!broadcast_manager_) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: broadcast manager is not initialized");
        return false;
    }
    if (request.copy_items_size() == 0 || timeout_ms <= 0) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: invalid request, item_count=%d, timeout_ms=%d",
                            request.copy_items_size(),
                            timeout_ms);
        return false;
    }

    const size_t worker_count = broadcast_manager_->workerNum();
    if (worker_count == 0) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: no worker configured");
        return false;
    }

    FunctionRequestPB         function_request;
    MemoryOperationRequestPB* memory_request = function_request.mutable_mem_request();
    if (memory_request == nullptr) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: failed to create memory request");
        return false;
    }
    memory_request->CopyFrom(request);
    std::vector<FunctionRequestPB> requests(worker_count, function_request);

    std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>> broadcast_result =
        broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
            requests,
            timeout_ms,
            [](const std::shared_ptr<RpcService::Stub>&    stub,
               const std::shared_ptr<grpc::ClientContext>& context,
               const FunctionRequestPB&                    rpc_request,
               grpc::CompletionQueue*                      completion_queue) {
                return stub->AsyncExecuteFunction(context.get(), rpc_request, completion_queue);
            });
    if (!broadcast_result) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: failed to start broadcast");
        return false;
    }

    broadcast_result->waitDone();
    if (!broadcast_result->success()) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: worker RPC failed");
        return false;
    }

    const std::vector<FunctionResponsePB> responses = broadcast_result->responses();
    if (responses.size() != worker_count) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: response count mismatch, expected=%zu, actual=%zu",
                            worker_count,
                            responses.size());
        return false;
    }
    for (size_t rank = 0; rank < responses.size(); ++rank) {
        const FunctionResponsePB& response = responses[rank];
        if (!response.has_mem_response() || !response.mem_response().success()) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::broadcastTransfer: worker transfer failed, rank=%zu", rank);
            return false;
        }
    }
    return true;
}

BlockTreeMatchResult BlockTreeCache::match(const CacheKeysType& cache_keys) {
    BlockTreeMatchResult result;
    if (cache_keys.empty()) {
        RTP_LLM_LOG_DEBUG("BlockTreeCache::match: empty cache_keys, returning empty result");
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    BlockTreeFindResult         tree_find_result = tree_->findNode(cache_keys);
    if (tree_find_result.matched_node == nullptr) {
        RTP_LLM_LOG_DEBUG("BlockTreeCache::match: no match found for %zu cache_keys", cache_keys.size());
        return result;
    }

    size_t            valid_matched_block_count = 0;
    std::vector<bool> candidate_logically_valid;
    candidate_logically_valid.reserve(tree_find_result.path.size());
    std::vector<std::unique_ptr<MatchValidator>> match_validators;
    match_validators.reserve(component_groups_.size());
    for (ComponentGroupPtr& component_group : component_groups_) {
        match_validators.push_back(component_group->createMatchValidator());
    }
    for (size_t i = 0; i < tree_find_result.path.size(); ++i) {
        TreeNode* path_node        = tree_find_result.path[i];
        bool      all_groups_valid = true;
        for (size_t group_index = 0; group_index < component_groups_.size(); ++group_index) {
            const size_t component_group_index =
                static_cast<size_t>(component_groups_[group_index]->component_group_id);
            GroupSlot& group_slot  = path_node->group_slots[component_group_index];
            const bool group_valid = match_validators[group_index]->validate(path_node, group_slot);
            if (!group_valid) {
                all_groups_valid = false;
            }
        }
        if (all_groups_valid) {
            valid_matched_block_count = i + 1;
        }
        candidate_logically_valid.push_back(all_groups_valid);
    }

    std::vector<TreeNode*> matched_path(tree_find_result.path.begin(),
                                        tree_find_result.path.begin()
                                            + static_cast<ptrdiff_t>(valid_matched_block_count));
    candidate_logically_valid.resize(valid_matched_block_count);
    std::vector<PendingLoadBackItem> pending_load_back_items;
    prepareMatchedBlocks(matched_path, candidate_logically_valid, result, pending_load_back_items);
    if (config_.enable_load_back && !pending_load_back_items.empty()) {
        result.load_back_ticket =
            load_back_ticket_registry_->createTicket(pending_load_back_items, valid_matched_block_count);
        if (result.load_back_ticket == nullptr) {
            abortLoadBackUnsafe(pending_load_back_items, /*prepared_item_count=*/0);
            result.load_back_blocks      = 0;
            result.host_load_back_blocks = 0;
            result.disk_load_back_blocks = 0;
        }
    }

    RTP_LLM_LOG_DEBUG("BlockTreeCache::match: matched %zu blocks, cache_keys=%zu, tree_nodes=%zu",
                      result.matched_blocks,
                      cache_keys.size(),
                      tree_->nodeCount());
    return result;
}

void BlockTreeCache::insert(TreeNode*                                  parent,
                            const CacheKeysType&                       cache_keys,
                            const std::vector<std::vector<GroupSlot>>& slots) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (cache_keys.empty()) {
        return;
    }

    // Input slots are indexed by per-tag gid; tree slots are indexed by component group.
    const size_t                        component_group_count = component_groups_.size();
    std::vector<std::vector<GroupSlot>> translated_slots(cache_keys.size(),
                                                         std::vector<GroupSlot>(component_group_count));
    for (size_t i = 0; i < cache_keys.size(); ++i) {
        for (size_t component_group_index = 0; component_group_index < component_group_count; ++component_group_index) {
            const size_t device_pool_count = component_groups_[component_group_index]->devicePoolCount();
            translated_slots[i][component_group_index].device_blocks.assign(device_pool_count, NULL_BLOCK_IDX);
        }
        if (i >= slots.size()) {
            continue;
        }
        const std::vector<GroupSlot>& per_tag_slots = slots[i];
        for (size_t tag_group_index = 0;
             tag_group_index < per_tag_slots.size() && tag_group_index < per_tag_mapping_.size();
             ++tag_group_index) {
            const PerTagMapping& mapping = per_tag_mapping_[tag_group_index];
            if (mapping.component_group_id < 0) {
                continue;
            }
            const std::vector<BlockIdxType>& source_blocks = per_tag_slots[tag_group_index].device_blocks;
            if (source_blocks.empty() || isNullBlockIdx(source_blocks.front())) {
                continue;
            }
            std::vector<BlockIdxType>& target_blocks =
                translated_slots[i][static_cast<size_t>(mapping.component_group_id)].device_blocks;
            target_blocks[static_cast<size_t>(mapping.local_pool_index)] = source_blocks.front();
        }
    }

    BlockTreeInsertResult insert_result = tree_->insertNode(parent, cache_keys, translated_slots);

    // incRef cache-hold on new nodes' device blocks (balanced by unreferenceBlocks on
    // eviction). Reused nodes keep theirs; their demoted data comes from load_back.
    for (const BlockTreeInsertedNode& inserted : insert_result.inserted_nodes) {
        TreeNode* node = inserted.node;
        for (ComponentGroupPtr& group : component_groups_) {
            const size_t gid = static_cast<size_t>(group->component_group_id);
            if (gid >= node->group_slots.size())
                continue;
            GroupSlot& slot = node->group_slots[gid];
            if (slot.has_value(Tier::DEVICE)) {
                const std::vector<BlockIdxType> blocks = group->getBlocks(slot, Tier::DEVICE);
                if (!blocks.empty()) {
                    group->referenceBlocks(GroupBlockSet{group->component_group_id, Tier::DEVICE, {blocks}});
                }
            }
        }
    }

    // Report the commit so the evictor stamps all new nodes and refreshes their
    // candidacy plus the first new node's existing direct parent when needed.
    evictor_.onInsertCommitted(insert_result);

    RTP_LLM_LOG_DEBUG(
        "BlockTreeCache::insert: inserted %zu cache_keys, tree_nodes=%zu", cache_keys.size(), tree_->nodeCount());
    ++mutation_version_;

    // Phase 3: check watermark after insert
    checkWatermark();
}

std::shared_ptr<HostBlockPool> BlockTreeCache::hostPoolForGroup(int component_group_id) const {
    auto gid = static_cast<size_t>(component_group_id);
    return gid < component_groups_.size() ? component_groups_[gid]->hostPool() : nullptr;
}

std::shared_ptr<DiskBlockPool> BlockTreeCache::diskPoolForGroup(int component_group_id) const {
    auto gid = static_cast<size_t>(component_group_id);
    return gid < component_groups_.size() ? component_groups_[gid]->diskPool() : nullptr;
}

// reclaimBlocks: directly reclaim (drop) num_blocks blocks at the given tier.
// Forces target_tier=NONE, so no copy/demotion happens; block content is
// discarded instead of being moved down to a lower tier.
int BlockTreeCache::reclaimBlocks(size_t num_blocks, Tier tier) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.isTierEnabled(tier)) {
        RTP_LLM_LOG_DEBUG("BlockTreeCache::reclaimBlocks: tier %s is disabled, skipping", tierName(tier));
        return 0;
    }

    int total_reclaimed = 0;
    for (size_t attempt = 0; attempt < num_blocks; ++attempt) {
        auto eviction_move = evictor_.chooseVictim(tier);
        if (!eviction_move.has_value()) {
            RTP_LLM_LOG_DEBUG("BlockTreeCache::reclaimBlocks: no more candidates at tier=%s, reclaimed %d/%zu blocks",
                              tierName(tier),
                              total_reclaimed,
                              num_blocks);
            break;
        }

        // Force direct reclaim: no demotion, no copy; drop block content directly.
        eviction_move->target_tier = Tier::NONE;

        if (submitEvictionLocked(*eviction_move)) {
            ++total_reclaimed;
        }
    }

    RTP_LLM_LOG_INFO(
        "BlockTreeCache::reclaimBlocks: reclaimed %d blocks from %s tier", total_reclaimed, tierName(tier));
    return total_reclaimed;
}

int BlockTreeCache::evictForGroup(int component_group_id, size_t num_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.isTierEnabled(Tier::DEVICE)) {
        return 0;
    }
    const int resolved = resolveComponentGroupId(component_group_id);
    if (resolved < 0) {
        return 0;
    }
    const auto gid = static_cast<size_t>(resolved);
    if (gid >= component_groups_.size()) {
        return 0;
    }
    int total_reclaimed = 0;
    for (size_t attempt = 0; attempt < num_blocks; ++attempt) {
        auto eviction_move = evictor_.chooseVictim(resolved, Tier::DEVICE);
        if (!eviction_move.has_value()) {
            break;
        }
        // Force direct reclaim: drop block content, do not demote to a lower tier.
        eviction_move->target_tier = Tier::NONE;
        if (submitEvictionLocked(*eviction_move)) {
            ++total_reclaimed;
        }
    }
    RTP_LLM_LOG_DEBUG("BlockTreeCache::evictForGroup: group[%d] reclaimed %d/%zu device blocks",
                      component_group_id,
                      total_reclaimed,
                      num_blocks);
    return total_reclaimed;
}

void BlockTreeCache::releaseMatchedBlocks(const std::vector<GroupBlockSet>& sets) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& set : sets) {
        auto gid = static_cast<size_t>(set.component_group_id);
        if (gid < component_groups_.size()) {
            component_groups_[gid]->unreferenceBlocks(set);
            // Releasing a match reference may make the node evictable again.
            evictor_.refreshCandidatesAfterRelease(set);
        }
    }
}

CacheStats BlockTreeCache::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats                  stats;
    stats.tree_node_count           = tree_->nodeCount();
    const CandidateStats candidates = evictor_.candidateStats();
    stats.device_heap_total_size    = candidates.device_candidates;
    stats.host_heap_total_size      = candidates.host_candidates;
    stats.disk_heap_total_size      = candidates.disk_candidates;
    return stats;
}

BlockTreeKeySnapshot BlockTreeCache::getKeySnapshot(size_t limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    BlockTreeKeySnapshot        snapshot;
    snapshot.version = mutation_version_;
    if (limit == 0 || !tree_ || !tree_->root()) {
        return snapshot;
    }

    std::vector<const TreeNode*> pending;
    pending.reserve(tree_->nodeCount());
    for (const auto& [cache_key, child] : tree_->root()->children) {
        (void)cache_key;
        if (child) {
            pending.push_back(child);
        }
    }
    while (!pending.empty() && snapshot.keys.size() < limit) {
        const TreeNode* node = pending.back();
        pending.pop_back();
        const bool reusable = std::any_of(
            node->group_slots.begin(), node->group_slots.end(), [](const GroupSlot& slot) { return !slot.is_empty(); });
        if (reusable) {
            snapshot.keys.push_back(node->cache_key);
        }
        for (const auto& [cache_key, child] : node->children) {
            (void)cache_key;
            if (child) {
                pending.push_back(child);
            }
        }
    }
    return snapshot;
}

void BlockTreeCache::waitForPendingTasks() {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    bool                         wait_observer_invoked = false;
    wait_cv_.wait(lock, [this, &wait_observer_invoked] {
        const int pending_tasks = pending_tasks_.load();
        if (pending_tasks > 0 && !wait_observer_invoked) {
            wait_observer_invoked                          = true;
            const auto pending_task_wait_observer_for_test = pending_task_wait_observer_for_test_;
            if (pending_task_wait_observer_for_test) {
                pending_task_wait_observer_for_test();
            }
        }
        return pending_tasks <= 0;
    });
}

void BlockTreeCache::onBlocksReleased() {
    std::lock_guard<std::mutex> lock(mutex_);
    // After external refcount changes (e.g. request free), blocks that were
    // non-evictable at insert time (refcount > 1) may now have refcount == 1
    // and thus become eviction candidates.  Refresh the eviction heap before
    // checking watermark so that pending evictions can find victims.
    evictor_.refreshAllCandidates(*tree_);
    checkWatermark();
}

void BlockTreeCache::taskStarted() {
    pending_tasks_++;
}

void BlockTreeCache::taskFinished() {
    int remaining = --pending_tasks_;
    if (remaining <= 0) {
        std::lock_guard<std::mutex> lock(wait_mutex_);
        wait_cv_.notify_all();
    }
}

void BlockTreeCache::prepareMatchedBlocks(const std::vector<TreeNode*>&     matched_path,
                                          const std::vector<bool>&          candidate_logically_valid,
                                          BlockTreeMatchResult&             result,
                                          std::vector<PendingLoadBackItem>& pending_load_back_items) {
    const size_t logical_matched_block_count = matched_path.size();
    if (logical_matched_block_count == 0) {
        return;
    }
    if (candidate_logically_valid.size() != logical_matched_block_count) {
        RTP_LLM_LOG_WARNING(
            "size mismatch, path=%zu valid=%zu", logical_matched_block_count, candidate_logically_valid.size());
        return;
    }

    const size_t ready_matched_block_count = computeReadyMatchedBlockCount(matched_path, candidate_logically_valid);
    if (ready_matched_block_count > 0) {
        result.matched_node   = matched_path[ready_matched_block_count - 1];
        result.matched_blocks = ready_matched_block_count;
        evictor_.onMatched(std::vector<TreeNode*>(
            matched_path.begin(), matched_path.begin() + static_cast<ptrdiff_t>(ready_matched_block_count)));
    }

    for (size_t group_index = 0; group_index < component_groups_.size(); ++group_index) {
        ComponentGroupPtr&      component_group       = component_groups_[group_index];
        const size_t            component_group_index = static_cast<size_t>(component_group->component_group_id);
        GroupBlockSet           matched_device_blocks{component_group->component_group_id, Tier::DEVICE};
        const std::vector<int>& device_group_ids = device_group_ids_[component_group_index];

        const size_t ready_reuse_count =
            std::min(component_group->computeReuseBlockCount(ready_matched_block_count, matched_path),
                     ready_matched_block_count);
        const size_t ready_reuse_begin = ready_matched_block_count - ready_reuse_count;
        for (size_t i = ready_reuse_begin; i < ready_matched_block_count; ++i) {
            TreeNode*                       path_node     = matched_path[i];
            GroupSlot&                      group_slot    = path_node->group_slots[component_group_index];
            const std::vector<BlockIdxType> device_blocks = component_group->getBlocks(group_slot, Tier::DEVICE);
            for (size_t tag_group_index = 0; tag_group_index < per_tag_mapping_.size(); ++tag_group_index) {
                const PerTagMapping& tag_mapping = per_tag_mapping_[tag_group_index];
                if (tag_mapping.component_group_id != component_group->component_group_id
                    || tag_mapping.local_pool_index < 0) {
                    continue;
                }
                const size_t local_pool_index = static_cast<size_t>(tag_mapping.local_pool_index);
                if (local_pool_index >= device_blocks.size() || device_blocks[local_pool_index] == NULL_BLOCK_IDX) {
                    continue;
                }
                result.group_block_indices[static_cast<int>(tag_group_index)].push_back(
                    device_blocks[local_pool_index]);
            }
            matched_device_blocks.per_node.push_back(device_blocks);
            matched_device_blocks.nodes.push_back(path_node);
        }

        if (!matched_device_blocks.per_node.empty()) {
            component_group->referenceBlocks(matched_device_blocks);
            result.matched_block_sets.push_back(std::move(matched_device_blocks));
        }

        if (!config_.enable_load_back) {
            continue;
        }
        const size_t logical_reuse_count =
            std::min(component_group->computeReuseBlockCount(logical_matched_block_count, matched_path),
                     logical_matched_block_count);
        for (size_t i = logical_matched_block_count - logical_reuse_count; i < logical_matched_block_count; ++i) {
            if (i >= ready_reuse_begin && i < ready_matched_block_count) {
                continue;
            }
            TreeNode*  path_node  = matched_path[i];
            GroupSlot& group_slot = path_node->group_slots[component_group_index];
            prepareMatchedLoadBackItem(
                path_node, component_group, group_slot, i, device_group_ids, result, pending_load_back_items);
        }
    }
}

size_t BlockTreeCache::computeReadyMatchedBlockCount(const std::vector<TreeNode*>& matched_path,
                                                     const std::vector<bool>&      candidate_logically_valid) const {
    size_t ready_matched_block_count = 0;
    for (size_t candidate_count = matched_path.size(); candidate_count > 0; --candidate_count) {
        if (!candidate_logically_valid[candidate_count - 1]) {
            continue;
        }
        bool all_groups_ready = true;
        for (const ComponentGroupPtr& component_group : component_groups_) {
            const size_t reuse_count =
                std::min(component_group->computeReuseBlockCount(candidate_count, matched_path), candidate_count);
            const size_t component_group_index = static_cast<size_t>(component_group->component_group_id);
            for (size_t path_index = candidate_count - reuse_count; path_index < candidate_count; ++path_index) {
                TreeNode* path_node = matched_path[path_index];
                if (component_group_index >= path_node->group_slots.size()
                    || !path_node->group_slots[component_group_index].has_value(Tier::DEVICE)) {
                    all_groups_ready = false;
                    break;
                }
            }
            if (!all_groups_ready) {
                break;
            }
        }
        if (all_groups_ready) {
            ready_matched_block_count = candidate_count;
            break;
        }
    }
    return ready_matched_block_count;
}

void BlockTreeCache::prepareMatchedLoadBackItem(TreeNode*                         path_node,
                                                const ComponentGroupPtr&          component_group,
                                                const GroupSlot&                  group_slot,
                                                size_t                            path_index,
                                                const std::vector<int>&           device_group_ids,
                                                BlockTreeMatchResult&             result,
                                                std::vector<PendingLoadBackItem>& pending_load_back_items) {
    Tier source_tier = Tier::NONE;
    if (group_slot.has_value(Tier::DEVICE)) {
        source_tier = Tier::DEVICE;
    } else if (group_slot.has_value(Tier::HOST)) {
        source_tier = Tier::HOST;
    } else if (group_slot.has_value(Tier::DISK)) {
        source_tier = Tier::DISK;
    }
    if (source_tier == Tier::NONE) {
        return;
    }

    const std::vector<BlockIdxType> source_blocks = component_group->getBlocks(group_slot, source_tier);
    if (source_blocks.empty()) {
        return;
    }

    // Keep the source non-evictable until the ticket is committed or aborted.
    component_group->referenceBlocks(GroupBlockSet{component_group->component_group_id, source_tier, {source_blocks}});
    PendingLoadBackItem pending_item;
    pending_item.node             = path_node;
    pending_item.group_id         = component_group->component_group_id;
    pending_item.path_index       = path_index;
    pending_item.source_tier      = source_tier;
    pending_item.source_blocks    = source_blocks;
    pending_item.device_group_ids = device_group_ids;
    pending_load_back_items.push_back(std::move(pending_item));

    if (source_tier == Tier::HOST) {
        result.host_load_back_blocks++;
        result.load_back_blocks++;
    } else if (source_tier == Tier::DISK) {
        result.disk_load_back_blocks++;
        result.load_back_blocks++;
    }

    RTP_LLM_LOG_DEBUG("BlockTreeCache::match: planned logical settlement from %s group[%d] node_key=%ld",
                      tierName(source_tier),
                      component_group->component_group_id,
                      path_node->cache_key);
}

std::shared_ptr<AsyncContext> BlockTreeCache::commitLoadBack(const std::vector<PendingLoadBackItem>& items) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (items.empty()) {
        return nullptr;
    }

    bool preflight_succeeded = true;
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const PendingLoadBackItem& item        = items[item_index];
        const bool                 device_item = item.source_tier == Tier::DEVICE;
        if (item.group_id < 0 || static_cast<size_t>(item.group_id) >= component_groups_.size()
            || component_groups_[static_cast<size_t>(item.group_id)] == nullptr
            || (!device_item && item.node == nullptr)
            || (item.source_tier != Tier::DEVICE && item.source_tier != Tier::HOST && item.source_tier != Tier::DISK)) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::commitLoadBack: boundary=core_batch_preflight "
                                "failure=invalid_item_shape item_index=%zu component_group_id=%d "
                                "source_tier=%s actual_source_blocks=%zu",
                                item_index,
                                item.group_id,
                                tierName(item.source_tier),
                                item.source_blocks.size());
            preflight_succeeded = false;
            break;
        }

        const size_t             component_group_index = static_cast<size_t>(item.group_id);
        const ComponentGroupPtr& component_group       = component_groups_[component_group_index];
        if (!validateDeviceGroupIdsForComponentGroup(item.group_id, item.device_group_ids)) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::commitLoadBack: invalid device group mapping, item=%zu group=%d",
                                item_index,
                                item.group_id);
            preflight_succeeded = false;
            break;
        }

        if (component_group->devicePoolCount() == 0
            || (!device_item && component_group_index >= item.node->group_slots.size())) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::commitLoadBack: boundary=core_batch_preflight "
                                "failure=invalid_component_group_shape item_index=%zu component_group_id=%d "
                                "expected_device_pools_nonzero=1 actual_device_pools=%zu",
                                item_index,
                                item.group_id,
                                component_group->devicePoolCount());
            preflight_succeeded = false;
            break;
        }
        const size_t expected_source_blocks = item.source_tier == Tier::DEVICE ? component_group->devicePoolCount() : 1;
        if (item.source_blocks.size() != expected_source_blocks) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::commitLoadBack: boundary=core_batch_preflight "
                                "failure=source_cardinality_mismatch item_index=%zu component_group_id=%d "
                                "expected=%zu actual=%zu",
                                item_index,
                                item.group_id,
                                expected_source_blocks,
                                item.source_blocks.size());
            preflight_succeeded = false;
            break;
        }

        if (!device_item) {
            const GroupSlot& source_slot = item.node->group_slots[component_group_index];
            if (source_slot.transfer_state != SlotTransferState::IDLE
                || component_group->getTopTier(source_slot) != item.source_tier
                || component_group->getBlocks(source_slot, item.source_tier) != item.source_blocks) {
                RTP_LLM_LOG_WARNING("BlockTreeCache::commitLoadBack: boundary=core_batch_preflight "
                                    "failure=source_shape_changed item_index=%zu component_group_id=%d "
                                    "expected_source_tier=%s actual_source_tier=%s transfer_state=%d",
                                    item_index,
                                    item.group_id,
                                    tierName(item.source_tier),
                                    tierName(component_group->getTopTier(source_slot)),
                                    static_cast<int>(source_slot.transfer_state));
                preflight_succeeded = false;
                break;
            }
        }

        const auto& device_pools = component_group->devicePools();
        if (item.target_device_blocks.size() != component_group->devicePoolCount()
            || device_pools.size() != component_group->devicePoolCount()) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::commitLoadBack: boundary=core_batch_preflight "
                                "failure=target_cardinality_mismatch item_index=%zu component_group_id=%d "
                                "expected=%zu actual=%zu",
                                item_index,
                                item.group_id,
                                component_group->devicePoolCount(),
                                item.target_device_blocks.size());
            preflight_succeeded = false;
            break;
        }
        if (device_item && item.target_device_blocks != item.source_blocks) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::commitLoadBack: boundary=core_batch_preflight "
                                "failure=resident_identity_changed item_index=%zu component_group_id=%d",
                                item_index,
                                item.group_id);
            preflight_succeeded = false;
            break;
        }
        for (size_t local_pool_index = 0; local_pool_index < item.target_device_blocks.size(); ++local_pool_index) {
            const BlockIdxType block = item.target_device_blocks[local_pool_index];
            const auto&        pool  = device_pools[local_pool_index];
            if (!pool || isNullBlockIdx(block) || !pool->validBlock(block) || !pool->isAllocated(block)
                || pool->refCount(block) == 0) {
                RTP_LLM_LOG_WARNING("BlockTreeCache::commitLoadBack: boundary=core_batch_preflight "
                                    "failure=invalid_target_block item_index=%zu component_group_id=%d "
                                    "expected=%zu actual=%zu local_pool_index=%zu offending_gid=%d block=%d",
                                    item_index,
                                    item.group_id,
                                    component_group->devicePoolCount(),
                                    item.target_device_blocks.size(),
                                    local_pool_index,
                                    item.device_group_ids[local_pool_index],
                                    block);
                preflight_succeeded = false;
                break;
            }
        }
        if (!preflight_succeeded) {
            break;
        }
    }

    if (!preflight_succeeded) {
        abortLoadBackUnsafe(items, /*prepared_item_count=*/0);
        RTP_LLM_LOG_WARNING("BlockTreeCache::commitLoadBack: boundary=core_batch_preflight "
                            "failure=batch_rejected released_source_protection=%zu",
                            items.size());
        return nullptr;
    }

    auto async_items = std::make_shared<std::vector<LoadBackItem>>();
    async_items->reserve(items.size());
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const PendingLoadBackItem& item                  = items[item_index];
        const size_t               component_group_index = static_cast<size_t>(item.group_id);
        ComponentGroupPtr&         component_group       = component_groups_[component_group_index];
        if (item.source_tier == Tier::DEVICE) {
            async_items->push_back(
                LoadBackItem{nullptr, item.group_id, item.source_tier, item.source_blocks, item.target_device_blocks});
            continue;
        }
        if (!evictor_.beginLoadBack(item.node, item.group_id, item.source_tier)) {
            abortLoadBackUnsafe(items, async_items->size());
            RTP_LLM_LOG_WARNING("BlockTreeCache::commitLoadBack: failed to prepare load_back items, count=%zu",
                                items.size());
            return nullptr;
        }

        // Add a tree/copy holder. The request already owns these targets; on
        // failure only this additional holder is released.
        component_group->referenceBlocks(GroupBlockSet{item.group_id, Tier::DEVICE, {item.target_device_blocks}});
        async_items->push_back(
            LoadBackItem{item.node, item.group_id, item.source_tier, item.source_blocks, item.target_device_blocks});
    }

    auto lb_ctx = std::make_shared<LoadBackAsyncContext>();
    lb_ctx->addTask();

    taskStarted();
    autil::LambdaWorkItem*              work_item = new autil::LambdaWorkItem([this, async_items, lb_ctx]() {
        performLoadBack(std::move(*async_items), lb_ctx);
        taskFinished();
    });
    const autil::ThreadPool::ERROR_TYPE error     = thread_pool_->pushWorkItem(work_item);
    if (error != autil::ThreadPool::ERROR_NONE) {
        work_item->destroy();
        abortLoadBackUnsafe(items, items.size());
        lb_ctx->onTaskComplete(false);
        taskFinished();
        return lb_ctx;
    }
    return lb_ctx;
}

void BlockTreeCache::abortLoadBack(const std::vector<PendingLoadBackItem>& items) {
    std::lock_guard<std::mutex> lock(mutex_);
    abortLoadBackUnsafe(items, /*prepared_item_count=*/0);
}

void BlockTreeCache::abortLoadBackUnsafe(const std::vector<PendingLoadBackItem>& items, size_t prepared_item_count) {
    bool global_refresh_required = false;
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const PendingLoadBackItem& item = items[item_index];
        const size_t gid = static_cast<size_t>(item.group_id);
        if (item.group_id < 0 || gid >= component_groups_.size() || component_groups_[gid] == nullptr) {
            continue;
        }
        if (item_index < prepared_item_count && item.source_tier != Tier::DEVICE) {
            component_groups_[gid]->unreferenceBlocks(
                GroupBlockSet{item.group_id, Tier::DEVICE, {item.target_device_blocks}});
        }
        // Release the source reference taken while preparing the match.
        GroupBlockSet source_set{item.group_id, item.source_tier, {item.source_blocks}};
        if (item.source_tier != Tier::DEVICE) {
            source_set.nodes = {item.node};
        }
        component_groups_[gid]->unreferenceBlocks(source_set);
        if (item.source_tier != Tier::DEVICE) {
            if (item_index < prepared_item_count) {
                evictor_.finishLoadBack(item.node, item.group_id, item.source_tier, false);
            } else {
                evictor_.refreshCandidatesAfterRelease(source_set);
            }
        } else {
            global_refresh_required = true;
        }
    }
    if (global_refresh_required) {
        evictor_.refreshAllCandidates(*tree_);
        checkWatermark();
    }
}

bool BlockTreeCache::executeLoadBackTransferBatch(const std::vector<TransferDescriptor>& descriptors, int timeout_ms) {
    if (descriptors.empty()) {
        return true;
    }

    if (broadcast_manager_ == nullptr) {
        for (const TransferDescriptor& descriptor : descriptors) {
            const CopyStatus status = executeTransfer(descriptor);
            if (status != CopyStatus::OK) {
                RTP_LLM_LOG_WARNING("BlockTreeCache::executeLoadBackTransferBatch: local transfer failed, "
                                    "group=%d source=%s target=%s status=%d",
                                    descriptor.component_group_id,
                                    tierName(descriptor.source_tier),
                                    tierName(descriptor.target_tier),
                                    static_cast<int>(status));
                return false;
            }
        }
        return true;
    }

    MemoryOperationRequestPB request;
    for (const TransferDescriptor& descriptor : descriptors) {
        const bool appended = BlockTreeTransferConverter::appendTransfer(descriptor, component_groups_, request);
        if (!appended) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::executeLoadBackTransferBatch: failed to encode transfer, "
                                "group=%d source=%s target=%s",
                                descriptor.component_group_id,
                                tierName(descriptor.source_tier),
                                tierName(descriptor.target_tier));
            return false;
        }
    }
    return broadcastTransfer(request, timeout_ms);
}

void BlockTreeCache::performLoadBack(std::vector<LoadBackItem> items, std::shared_ptr<AsyncContext> ctx) {
    std::shared_ptr<LoadBackAsyncContext> load_back_context = std::dynamic_pointer_cast<LoadBackAsyncContext>(ctx);
    std::vector<BlockIdxType>             staging_host_blocks(items.size(), NULL_BLOCK_IDX);
    std::vector<TransferDescriptor>       disk_to_host_descriptors;
    std::vector<TransferDescriptor>       host_to_device_descriptors;
    bool                                  prepared = !items.empty();

    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        LoadBackItem& item = items[item_index];
        if (item.group_id < 0 || static_cast<size_t>(item.group_id) >= component_groups_.size()) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: invalid group id, group=%d", item.group_id);
            prepared = false;
            continue;
        }

        ComponentGroupPtr& group = component_groups_[static_cast<size_t>(item.group_id)];
        if (item.target_device_blocks.empty()) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: invalid item, group=%d", item.group_id);
            prepared = false;
            continue;
        }
        if (item.source_tier == Tier::DEVICE) {
            if (item.source_blocks.empty() || item.source_blocks != item.target_device_blocks) {
                RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: resident identity changed, group=%d",
                                    item.group_id);
                prepared = false;
            }
            continue;
        }
        if (item.node == nullptr) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: invalid copy item node, group=%d", item.group_id);
            prepared = false;
            continue;
        }
        if ((item.source_tier != Tier::HOST && item.source_tier != Tier::DISK) || item.source_blocks.size() != 1) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: invalid copy item, group=%d source=%s",
                                item.group_id,
                                tierName(item.source_tier));
            prepared = false;
            continue;
        }

        BlockIdxType source_host_block = NULL_BLOCK_IDX;
        if (item.source_tier == Tier::HOST && group->hostPool() != nullptr) {
            source_host_block = item.source_blocks[0];
        } else if (item.source_tier == Tier::DISK && group->hostPool() != nullptr && group->diskPool() != nullptr) {
            source_host_block = group->allocateSingleBlock(Tier::HOST);
            if (isNullBlockIdx(source_host_block) && reclaimOneForGroup(item.group_id, Tier::HOST)) {
                // Disk load-back needs a temporary host staging block. Apply
                // target-tier pressure once before failing the whole request.
                source_host_block = group->allocateSingleBlock(Tier::HOST);
            }
            if (!isNullBlockIdx(source_host_block)) {
                staging_host_blocks[item_index] = source_host_block;
                disk_to_host_descriptors.push_back(
                    TransferDescriptor::diskToHost(item.group_id, item.source_blocks[0], source_host_block));
            }
        }

        if (isNullBlockIdx(source_host_block)) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::performLoadBack: failed to prepare source, group=%d source=%s",
                                item.group_id,
                                tierName(item.source_tier));
            prepared = false;
            continue;
        }
        host_to_device_descriptors.push_back(
            TransferDescriptor::hostToDevice(item.group_id, source_host_block, item.target_device_blocks));
    }

    bool copy_success = prepared;
    if (copy_success) {
        copy_success =
            executeLoadBackTransferBatch(disk_to_host_descriptors, config_.memory_cache_disk_sync_timeout_ms);
    }
    if (copy_success) {
        copy_success = executeLoadBackTransferBatch(host_to_device_descriptors, config_.memory_cache_sync_timeout_ms);
    }

    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        LoadBackItem& item = items[item_index];
        if (item.group_id < 0 || static_cast<size_t>(item.group_id) >= component_groups_.size()) {
            continue;
        }
        ComponentGroupPtr& group = component_groups_[static_cast<size_t>(item.group_id)];
        if (!isNullBlockIdx(staging_host_blocks[item_index])) {
            group->releaseSingleBlock(Tier::HOST, staging_host_blocks[item_index]);
        }
        group->unreferenceBlocks(GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}});
        if (!copy_success && item.source_tier != Tier::DEVICE) {
            group->unreferenceBlocks(GroupBlockSet{item.group_id, Tier::DEVICE, {item.target_device_blocks}});
        }
    }

    const bool has_copy_items = std::any_of(
        items.begin(), items.end(), [](const LoadBackItem& item) { return item.source_tier != Tier::DEVICE; });
    const bool has_device_items = std::any_of(
        items.begin(), items.end(), [](const LoadBackItem& item) { return item.source_tier == Tier::DEVICE; });

    // Phase 3: re-acquire lock to update tree state and reset transfer state.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& item : items) {
            if (item.source_tier == Tier::DEVICE) {
                continue;
            }
            auto gid = static_cast<size_t>(item.group_id);
            if (gid >= component_groups_.size() || item.node == nullptr)
                continue;

            auto& group = component_groups_[gid];
            auto& slot  = item.node->group_slots[gid];

            if (copy_success) {
                // Move to DEVICE, then retire source: release its cache-hold (saved
                // ids) before evictFromTier clears the slot. load_back is async.
                group->setBlocks(slot, Tier::DEVICE, item.target_device_blocks);
                group->unreferenceBlocks(GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}});
                group->evictFromTier(item.node, slot, item.source_tier);
            }
            // Clear LOADING_BACK and re-evaluate candidacy (DEVICE on success, else source).
            evictor_.finishLoadBack(item.node, item.group_id, item.source_tier, copy_success);
        }
        if (has_device_items) {
            evictor_.refreshAllCandidates(*tree_);
        }
        if (has_copy_items) {
            ++mutation_version_;
        }
        if (has_device_items || has_copy_items) {
            checkWatermark();
        }
    }

    if (load_back_context != nullptr) {
        load_back_context->onTaskComplete(copy_success);
    }
}

bool BlockTreeCache::reclaimOneForGroup(int component_group_id, Tier tier) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (component_group_id < 0 || static_cast<size_t>(component_group_id) >= component_groups_.size()) {
        return false;
    }
    auto eviction_move = evictor_.chooseVictim(component_group_id, tier);
    if (!eviction_move.has_value()) {
        return false;
    }
    eviction_move->target_tier = Tier::NONE;
    return submitEvictionLocked(*eviction_move);
}

bool BlockTreeCache::submitEvictionLocked(EvictionMove& eviction_move) {
    if (eviction_move.target_tier != Tier::NONE && !config_.isTierEnabled(eviction_move.target_tier)) {
        eviction_move.target_tier = Tier::NONE;
    }

    auto plan = evictor_.buildPlan(eviction_move);
    if (!plan.has_value()) {
        return false;
    }

    if (!plan->needsCopy()) {
        BlockTreeEvictor::CopyResultSet results;
        results.primary_success = true;
        results.cascade_success.assign(plan->cascade_moves.size(), true);
        evictor_.complete(*tree_, *plan, results);
        ++mutation_version_;
        return true;
    }

    auto plan_ptr = std::make_shared<BlockTreeEvictor::EvictionPlan>(std::move(*plan));
    taskStarted();
    auto* work_item = new autil::LambdaWorkItem([this, plan_ptr]() { performEvictionCopy(*plan_ptr); });
    auto  err       = thread_pool_->pushWorkItem(work_item);
    if (err != autil::ThreadPool::ERROR_NONE) {
        work_item->destroy();
        evictor_.rollbackPreparedPlan(*plan_ptr);
        taskFinished();
        return false;
    }
    return true;
}

void BlockTreeCache::performEvictionCopy(const BlockTreeEvictor::EvictionPlan& plan) {
    BlockTreeEvictor::CopyResultSet copy_results;
    if (broadcast_manager_ == nullptr) {
        copy_results = evictor_.performCopy(plan);
    } else {
        MemoryOperationRequestPB request;
        const bool               request_ready = buildEvictionTransferRequest(plan, request);
        const bool copy_success      = request_ready && broadcastTransfer(request, evictionTransferTimeoutMs(plan));
        copy_results.primary_success = copy_success;
        copy_results.cascade_success.assign(plan.cascade_moves.size(), copy_success);
    }

    bool         copy_ok          = copy_results.primary_success;
    CacheKeyType remote_cache_key = 0;
    int          remote_group_id  = -1;
    if (copy_ok && plan.primary.node != nullptr) {
        remote_cache_key = plan.primary.node->cache_key;
        remote_group_id  = plan.primary.component_group_id;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        evictor_.complete(*tree_, plan, copy_results);
        const bool mutated = copy_results.primary_success
                             || std::any_of(copy_results.cascade_success.begin(),
                                            copy_results.cascade_success.end(),
                                            [](bool success) { return success; });
        if (mutated) {
            ++mutation_version_;
            // A completed device->host or host->disk move changes the target
            // tier's pressure. Continue the cascade without waiting for another
            // request insert/release event. A fully failed plan is restored to
            // its source heap and must not be retried in a tight loop here.
            checkWatermark();
        }
    }

    if (copy_ok && config_.enable_remote_cache && remote_group_id >= 0) {
        evictor_.writeRemoteThrough(storage_backend_, remote_cache_key, remote_group_id);
    }
    taskFinished();
}

bool BlockTreeCache::buildEvictionTransferRequest(const BlockTreeEvictor::EvictionPlan& plan,
                                                  MemoryOperationRequestPB&             request) const {
    TransferDescriptor primary_descriptor;
    if (!BlockTreeEvictor::buildTransferDescriptor(plan.primary, primary_descriptor)
        || !BlockTreeTransferConverter::appendTransfer(primary_descriptor, component_groups_, request)) {
        return false;
    }

    for (const EvictionMove& cascade_move : plan.cascade_moves) {
        TransferDescriptor cascade_descriptor;
        if (!BlockTreeEvictor::buildTransferDescriptor(cascade_move, cascade_descriptor)
            || !BlockTreeTransferConverter::appendTransfer(cascade_descriptor, component_groups_, request)) {
            return false;
        }
    }
    return true;
}

int BlockTreeCache::evictionTransferTimeoutMs(const BlockTreeEvictor::EvictionPlan& plan) const {
    bool uses_disk = plan.primary.source_tier == Tier::DISK || plan.primary.target_tier == Tier::DISK;
    for (const EvictionMove& cascade_move : plan.cascade_moves) {
        if (cascade_move.source_tier == Tier::DISK || cascade_move.target_tier == Tier::DISK) {
            uses_disk = true;
            break;
        }
    }
    if (!uses_disk) {
        return config_.memory_cache_sync_timeout_ms;
    }
    return std::max(config_.memory_cache_sync_timeout_ms, config_.memory_cache_disk_sync_timeout_ms);
}

void BlockTreeCache::checkWatermark() {
    if (config_.enable_device_cache && config_.device_min_free_blocks > 0) {
        for (auto& group : component_groups_) {
            const size_t excess = group->devicePoolMaxExcessForMinFree(config_.device_min_free_blocks);
            for (size_t i = 0; i < excess; ++i) {
                auto eviction_move = evictor_.chooseVictim(group->component_group_id, Tier::DEVICE);
                if (!eviction_move.has_value()) {
                    break;
                }
                submitEvictionLocked(*eviction_move);
            }
        }
    }

    for (auto tier : {Tier::DEVICE, Tier::HOST, Tier::DISK}) {
        if (tier == Tier::DEVICE && config_.device_min_free_blocks > 0) {
            continue;
        }
        auto wm = config_.watermarkForTier(tier);
        if (wm.ratio <= 0.0 || !config_.isTierEnabled(tier))
            continue;

        for (auto& group : component_groups_) {
            auto victims = evictor_.chooseWatermarkVictims(*group, tier, wm.ratio);
            for (auto& eviction_move : victims) {
                submitEvictionLocked(eviction_move);
            }
        }
    }
}

}  // namespace rtp_llm
