#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <limits>
#include <set>
#include <unordered_map>
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

template<typename Cleanup>
class ScopeRollback {
public:
    explicit ScopeRollback(Cleanup cleanup): cleanup_(std::move(cleanup)) {}

    ~ScopeRollback() {
        run();
    }

    ScopeRollback(const ScopeRollback&)            = delete;
    ScopeRollback& operator=(const ScopeRollback&) = delete;
    ScopeRollback(ScopeRollback&&)                 = delete;
    ScopeRollback& operator=(ScopeRollback&&)      = delete;

    void run() {
        if (!active_) {
            return;
        }
        active_ = false;
        cleanup_();
    }

    void dismiss() noexcept {
        active_ = false;
    }

private:
    Cleanup cleanup_;
    bool    active_{true};
};

// AsyncContext for load_back: waits until all copy tasks complete.
class LoadBackAsyncContext: public AsyncContext {
public:
    enum class State : int {
        PENDING          = 0,
        CANCEL_REQUESTED = 1,
        SUCCEEDED        = 2,
        FAILED           = 3,
        CANCELLED        = 4
    };

    void addTask() {}

    bool requestCancel() {
        State expected = State::PENDING;
        if (state_.compare_exchange_strong(expected, State::CANCEL_REQUESTED)) {
            return true;
        }
        return expected == State::CANCEL_REQUESTED;
    }

    bool cancelRequested() const {
        return state_.load() == State::CANCEL_REQUESTED;
    }

    void onTaskComplete(bool ok) {
        State expected = State::PENDING;
        State terminal = ok ? State::SUCCEEDED : State::FAILED;
        if (!state_.compare_exchange_strong(expected, terminal)) {
            if (expected == State::CANCEL_REQUESTED) {
                state_.store(State::CANCELLED);
            }
        }
        std::lock_guard<std::mutex> lock(mu_);
        cv_.notify_all();
    }

    void waitDone() override {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return done(); });
    }

    bool done() const override {
        const State state = state_.load();
        return state == State::SUCCEEDED || state == State::FAILED || state == State::CANCELLED;
    }

    bool success() const override {
        return state_.load() == State::SUCCEEDED;
    }

private:
    std::atomic<State>      state_{State::PENDING};
    mutable std::mutex      mu_;
    std::condition_variable cv_;
};

}  // anonymous namespace

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>         tree,
                               std::vector<ComponentGroupPtr>     component_groups,
                               std::vector<Component>             components,
                               BlockTreeCacheConfig               config,
                               std::shared_ptr<StorageBackend>    storage_backend,
                               std::shared_ptr<BroadcastManager>  broadcast_manager,
                               std::vector<std::string>           per_tag_tags,
                               std::vector<DeviceKVCacheGroupPtr> per_tag_device_groups,
                               std::vector<PerTagMapping>         per_tag_mapping):
    config_(std::move(config)),
    tree_(std::move(tree)),
    component_groups_(std::move(component_groups)),
    components_(std::make_shared<const std::vector<Component>>(std::move(components))),
    per_tag_tags_(std::move(per_tag_tags)),
    per_tag_device_groups_(std::move(per_tag_device_groups)),
    per_tag_mapping_(std::move(per_tag_mapping)),
    load_back_ticket_registry_(std::make_shared<LoadBackTicketRegistry>(
        [this](const LoadBackTicket& ticket) { return commitLoadBack(ticket); },
        [this](const LoadBackTicket& ticket) { abortLoadBack(ticket); })),
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
    if (!initDeviceGroupTags()) {
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
    initialized_ = true;
    return true;
}

bool BlockTreeCache::initDeviceGroupTags() {
    device_group_tags_.clear();
    device_group_tags_.resize(component_groups_.size());
    if (per_tag_mapping_.empty() || per_tag_tags_.empty()) {
        RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupTags: declarative mapping registry must not be empty");
        return false;
    }
    if (per_tag_tags_.size() != per_tag_mapping_.size()) {
        RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupTags: tag/mapping size mismatch, tags=%zu mappings=%zu",
                          per_tag_tags_.size(),
                          per_tag_mapping_.size());
        return false;
    }
    for (size_t group_id = 0; group_id < component_groups_.size(); ++group_id) {
        const ComponentGroupPtr& component_group = component_groups_[group_id];
        if (component_group == nullptr || component_group->component_group_id != static_cast<int>(group_id)
            || component_group->devicePoolCount() == 0) {
            RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupTags: invalid component group, group=%zu", group_id);
            return false;
        }
        device_group_tags_[group_id].assign(component_group->devicePoolCount(), {});
    }

    for (size_t tag_index = 0; tag_index < per_tag_mapping_.size(); ++tag_index) {
        const PerTagMapping& mapping = per_tag_mapping_[tag_index];
        if (mapping.component_group_id == -1) {
            if (mapping.local_pool_index != -1) {
                RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupTags: invalid non-reusable mapping, tag=%s",
                                  per_tag_tags_[tag_index].c_str());
                return false;
            }
            continue;
        }
        if (mapping.component_group_id < 0 || mapping.local_pool_index < 0) {
            RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupTags: negative mapping index, tag=%s",
                              per_tag_tags_[tag_index].c_str());
            return false;
        }
        const size_t group_id      = static_cast<size_t>(mapping.component_group_id);
        const size_t local_pool_id = static_cast<size_t>(mapping.local_pool_index);
        if (group_id >= device_group_tags_.size() || local_pool_id >= device_group_tags_[group_id].size()
            || !device_group_tags_[group_id][local_pool_id].empty() || per_tag_tags_[tag_index].empty()) {
            RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupTags: invalid mapping, tag=%s group=%zu local=%zu",
                              per_tag_tags_[tag_index].c_str(),
                              group_id,
                              local_pool_id);
            return false;
        }
        device_group_tags_[group_id][local_pool_id] = per_tag_tags_[tag_index];
    }

    for (size_t group_id = 0; group_id < device_group_tags_.size(); ++group_id) {
        const std::vector<std::string>& device_group_tags = device_group_tags_[group_id];
        if (std::any_of(device_group_tags.begin(), device_group_tags.end(), [](const std::string& tag) {
                return tag.empty();
            })) {
            RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupTags: incomplete mapping, group=%zu", group_id);
            return false;
        }
        if (device_group_tags != component_groups_[group_id]->tags()) {
            RTP_LLM_LOG_ERROR("BlockTreeCache::initDeviceGroupTags: component group tag mapping mismatch, group=%zu",
                              group_id);
            return false;
        }
    }
    return true;
}

bool BlockTreeCache::validateConfiguration() const {
    if (tree_ == nullptr || components_ == nullptr) {
        RTP_LLM_LOG_ERROR("BlockTreeCache: tree and component registry must be initialized");
        return false;
    }
    if (component_groups_.empty() && !components_->empty()) {
        RTP_LLM_LOG_ERROR("BlockTreeCache: empty component groups require an empty component registry");
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

BlockTreeCache::~BlockTreeCache() {
    RTP_LLM_LOG_INFO("BlockTreeCache: destroying, closing load-back tickets...");
    load_back_ticket_registry_->shutdown();
    RTP_LLM_LOG_INFO("BlockTreeCache: load-back tickets closed, waiting for pending tasks...");
    waitForPendingTasks();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        RTP_LLM_CHECK_WITH_INFO(
            in_flight_device_release_credits_.empty(),
            "BlockTreeCache: in-flight DEVICE release credits remain after pending tasks drained: %zu",
            in_flight_device_release_credits_.size());
    }
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
    LoadBackTicket::PendingLoadBackItems pending_load_back_items;
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
    insertImpl(parent, cache_keys, slots, false);
}

void BlockTreeCache::insertSparse(TreeNode*                                  parent,
                                  const CacheKeysType&                       cache_keys,
                                  const std::vector<std::vector<GroupSlot>>& slots) {
    insertImpl(parent, cache_keys, slots, true);
}

void BlockTreeCache::insertImpl(TreeNode*                                  parent,
                                const CacheKeysType&                       cache_keys,
                                const std::vector<std::vector<GroupSlot>>& slots,
                                bool                                       allow_sparse_slots) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (cache_keys.empty()) {
        return;
    }

    if (slots.size() != cache_keys.size()) {
        RTP_LLM_LOG_WARNING(
            "BlockTreeCache::insert: key/slot size mismatch, keys=%zu slots=%zu", cache_keys.size(), slots.size());
        return;
    }
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].size() != component_groups_.size()) {
            RTP_LLM_LOG_WARNING("BlockTreeCache::insert: component slot mismatch, index=%zu expected=%zu actual=%zu",
                                i,
                                component_groups_.size(),
                                slots[i].size());
            return;
        }
        for (size_t component_group_index = 0; component_group_index < component_groups_.size();
             ++component_group_index) {
            const auto& group                  = component_groups_[component_group_index];
            const auto& slot                   = slots[i][component_group_index];
            const bool  structurally_absent    = slot.device_blocks.empty() && slot.is_empty();
            const bool  allowed_sparse_absence = allow_sparse_slots && group != nullptr
                                                && group->group_type != CacheGroupType::FULL && structurally_absent;
            if (group == nullptr
                || (!allowed_sparse_absence
                    && (slot.device_blocks.size() != group->devicePoolCount()
                        || std::any_of(slot.device_blocks.begin(), slot.device_blocks.end(), [](BlockIdxType block) {
                               return isNullBlockIdx(block);
                           })))) {
                RTP_LLM_LOG_WARNING(
                    "BlockTreeCache::insert: device slot mismatch, index=%zu component=%zu expected=%zu actual=%zu",
                    i,
                    component_group_index,
                    group ? group->devicePoolCount() : 0,
                    slot.device_blocks.size());
                return;
            }
        }
    }

    BlockTreeInsertResult insert_result = tree_->insertNode(parent, cache_keys, slots);

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

    // Existing nodes may independently refill one empty component. Take a tree
    // holder only for the exact adopted component; other slots already own theirs.
    for (const BlockTreeAdoptedSlot& adopted : insert_result.adopted_slots) {
        if (adopted.node == nullptr || adopted.component_group_id < 0) {
            continue;
        }
        const size_t gid = static_cast<size_t>(adopted.component_group_id);
        if (gid >= component_groups_.size() || component_groups_[gid] == nullptr
            || gid >= adopted.node->group_slots.size()) {
            continue;
        }
        auto&      group  = component_groups_[gid];
        const auto blocks = group->getBlocks(adopted.node->group_slots[gid], Tier::DEVICE);
        if (!blocks.empty()) {
            group->referenceBlocks(GroupBlockSet{adopted.component_group_id, Tier::DEVICE, {blocks}});
        }
    }

    const bool changed = !insert_result.inserted_nodes.empty() || !insert_result.adopted_slots.empty();
    if (!changed) {
        return;
    }

    // Stamp and refresh only newly created nodes and exact adopted components.
    evictor_.onInsertCommitted(insert_result);
    ++mutation_version_;
    RTP_LLM_LOG_DEBUG("BlockTreeCache::insert: created=%zu adopted=%zu tree_nodes=%zu",
                      insert_result.inserted_nodes.size(),
                      insert_result.adopted_slots.size(),
                      tree_->nodeCount());
    checkWatermark();
}

int BlockTreeCache::evictForTag(const std::string& tag, size_t num_blocks) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.isTierEnabled(Tier::DEVICE) || tag.empty()) {
        return 0;
    }
    const auto tag_it = std::find(per_tag_tags_.begin(), per_tag_tags_.end(), tag);
    if (tag_it == per_tag_tags_.end()) {
        return 0;
    }
    const size_t tag_index = static_cast<size_t>(std::distance(per_tag_tags_.begin(), tag_it));
    if (tag_index >= per_tag_mapping_.size()) {
        return 0;
    }
    const PerTagMapping& mapping = per_tag_mapping_[tag_index];
    if (mapping.component_group_id < 0 || mapping.local_pool_index < 0) {
        return 0;
    }
    const size_t gid = static_cast<size_t>(mapping.component_group_id);
    if (gid >= component_groups_.size() || component_groups_[gid] == nullptr) {
        return 0;
    }
    const auto&  device_pools     = component_groups_[gid]->devicePools();
    const size_t local_pool_index = static_cast<size_t>(mapping.local_pool_index);
    if (local_pool_index >= device_pools.size() || device_pools[local_pool_index] == nullptr) {
        return 0;
    }

    const size_t initial_free = device_pools[local_pool_index]->freeBlocksNum();
    size_t       reclaimed    = 0;
    while (reclaimed < num_blocks) {
        auto eviction_move = evictor_.chooseVictim(mapping.component_group_id, Tier::DEVICE);
        if (!eviction_move.has_value()) {
            break;
        }
        eviction_move->target_tier = Tier::NONE;
        if (!submitEvictionLocked(*eviction_move)) {
            break;
        }
        const size_t current_free = device_pools[local_pool_index]->freeBlocksNum();
        reclaimed                 = current_free > initial_free ? current_free - initial_free : 0;
    }
    RTP_LLM_LOG_DEBUG("BlockTreeCache::evictForTag: tag=%s component_group[%d] reclaimed %zu/%zu device blocks",
                      tag.c_str(),
                      mapping.component_group_id,
                      reclaimed,
                      num_blocks);
    return static_cast<int>(reclaimed);
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

bool BlockTreeCache::cancelLoadBack(const std::shared_ptr<AsyncContext>& context) {
    auto load_back_context = std::dynamic_pointer_cast<LoadBackAsyncContext>(context);
    if (!load_back_context) {
        RTP_LLM_LOG_WARNING("BlockTreeCache::cancelLoadBack: context is not owned by BlockTreeCache");
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return !load_back_context->done() && load_back_context->requestCancel();
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
                                          LoadBackTicket::PendingLoadBackItems& pending_load_back_items) {
    const size_t logical_matched_block_count = matched_path.size();
    if (logical_matched_block_count == 0) {
        return;
    }
    if (candidate_logically_valid.size() != logical_matched_block_count) {
        RTP_LLM_LOG_WARNING(
            "BlockTreeCache::prepareMatchedBlocks: candidate validity size mismatch, path=%zu valid=%zu",
            logical_matched_block_count,
            candidate_logically_valid.size());
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
        ComponentGroupPtr& component_group       = component_groups_[group_index];
        const size_t       component_group_index = static_cast<size_t>(component_group->component_group_id);
        GroupBlockSet      matched_device_blocks{component_group->component_group_id, Tier::DEVICE};
        const std::vector<std::string>& device_group_tags = device_group_tags_[component_group_index];

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
                result.group_block_indices[per_tag_tags_[tag_group_index]].push_back(device_blocks[local_pool_index]);
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
                path_node, component_group, group_slot, i, device_group_tags, result, pending_load_back_items);
        }
    }
}

size_t BlockTreeCache::computeReadyMatchedBlockCount(const std::vector<TreeNode*>& matched_path,
                                                     const std::vector<bool>&      candidate_logically_valid) const {
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
            return candidate_count;
        }
    }
    return 0;
}

void BlockTreeCache::prepareMatchedLoadBackItem(TreeNode*                         path_node,
                                                const ComponentGroupPtr&          component_group,
                                                const GroupSlot&                  group_slot,
                                                size_t                            path_index,
                                                const std::vector<std::string>&   device_group_tags,
                                                BlockTreeMatchResult&             result,
                                                LoadBackTicket::PendingLoadBackItems& pending_load_back_items) {
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
    LoadBackTicket::PendingLoadBackItem pending_item;
    pending_item.node              = path_node;
    pending_item.group_id          = component_group->component_group_id;
    pending_item.path_index        = path_index;
    pending_item.source_tier       = source_tier;
    pending_item.source_blocks     = source_blocks;
    pending_item.device_group_tags = device_group_tags;
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

bool BlockTreeCache::validateDeviceGroupTagsForComponentGroup(int                             component_group_id,
                                                              const std::vector<std::string>& device_group_tags) const {
    if (!initialized_ || component_group_id < 0
        || static_cast<size_t>(component_group_id) >= device_group_tags_.size()) {
        RTP_LLM_LOG_WARNING("BlockTreeCache: invalid component group mapping request, group=%d", component_group_id);
        return false;
    }
    const std::vector<std::string>& expected = device_group_tags_[static_cast<size_t>(component_group_id)];
    if (device_group_tags != expected) {
        RTP_LLM_LOG_WARNING("BlockTreeCache: device group tag mapping mismatch, group=%d expected=%zu actual=%zu",
                            component_group_id,
                            expected.size(),
                            device_group_tags.size());
        return false;
    }
    return true;
}

std::shared_ptr<AsyncContext> BlockTreeCache::commitLoadBack(const LoadBackTicket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto&                 items = ticket.items_;

    size_t prepared_item_count         = 0;
    bool   partial_item_claimed        = false;
    bool   partial_target_holder_added = false;
    auto rollback_action = [this, &items, &prepared_item_count, &partial_item_claimed, &partial_target_holder_added]() {
        abortLoadBackUnsafe(items, prepared_item_count, partial_item_claimed, partial_target_holder_added);
    };
    ScopeRollback<decltype(rollback_action)> rollback_guard(std::move(rollback_action));

    // Materialize every allocation-backed payload before changing a slot state
    // or acquiring a target holder. The rollback guard already owns all source
    // planning holds if any of these copies throws.
    auto async_items = std::make_shared<std::vector<LoadBackItem>>();
    async_items->reserve(items.size());
    std::vector<GroupBlockSet> target_holder_sets;
    target_holder_sets.reserve(items.size());
    for (const auto& item : items) {
        async_items->push_back(LoadBackItem{item.source_tier == Tier::DEVICE ? nullptr : item.node,
                                            item.group_id,
                                            item.source_tier,
                                            item.source_blocks,
                                            item.target_device_blocks});
        if (item.source_tier == Tier::DEVICE) {
            target_holder_sets.emplace_back();
        } else {
            target_holder_sets.push_back(GroupBlockSet{item.group_id, Tier::DEVICE, {item.target_device_blocks}});
        }
    }

    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const auto& item                  = items[item_index];
        const size_t component_group_index = static_cast<size_t>(item.group_id);
        ComponentGroupPtr& component_group = component_groups_[component_group_index];
        if (item.source_tier == Tier::DEVICE) {
            ++prepared_item_count;
            continue;
        }
        if (!evictor_.beginLoadBack(item.node, item.group_id, item.source_tier)) {
            rollback_guard.run();
            RTP_LLM_LOG_WARNING("BlockTreeCache::commitLoadBack: failed to prepare load_back items, count=%zu",
                                items.size());
            return nullptr;
        }
        partial_item_claimed = true;

        // Add a tree/copy holder. The request already owns these targets; on
        // failure only this additional holder is released.
        component_group->referenceBlocks(target_holder_sets[item_index]);
        partial_target_holder_added = true;

        ++prepared_item_count;
        partial_item_claimed        = false;
        partial_target_holder_added = false;
    }

    auto lb_ctx = std::make_shared<LoadBackAsyncContext>();
    lb_ctx->addTask();

    autil::LambdaWorkItem* work_item                = new autil::LambdaWorkItem([this, async_items, lb_ctx]() {
        performLoadBack(std::move(*async_items), lb_ctx);
        taskFinished();
    });
    auto                   work_item_cleanup_action = [&work_item]() { work_item->destroy(); };
    ScopeRollback<decltype(work_item_cleanup_action)> work_item_cleanup_guard(std::move(work_item_cleanup_action));

    taskStarted();
    auto                                         task_cleanup_action = [this]() { taskFinished(); };
    ScopeRollback<decltype(task_cleanup_action)> task_cleanup_guard(std::move(task_cleanup_action));

    const autil::ThreadPool::ERROR_TYPE error = thread_pool_->pushWorkItem(work_item);
    if (error != autil::ThreadPool::ERROR_NONE) {
        work_item_cleanup_guard.run();
        rollback_guard.run();
        lb_ctx->onTaskComplete(false);
        task_cleanup_guard.run();
        return lb_ctx;
    }
    work_item_cleanup_guard.dismiss();
    task_cleanup_guard.dismiss();
    rollback_guard.dismiss();
    return lb_ctx;
}

void BlockTreeCache::abortLoadBack(const LoadBackTicket& ticket) {
    std::lock_guard<std::mutex> lock(mutex_);
    abortLoadBackUnsafe(ticket.items_, /*prepared_item_count=*/0);
}

void BlockTreeCache::abortLoadBackUnsafe(const LoadBackTicket::PendingLoadBackItems& items,
                                         size_t prepared_item_count,
                                         bool   partial_item_claimed,
                                         bool   partial_target_holder_added) {
    bool global_refresh_required = false;
    for (size_t item_index = 0; item_index < items.size(); ++item_index) {
        const auto& item = items[item_index];
        const size_t gid  = static_cast<size_t>(item.group_id);
        if (item.group_id < 0 || gid >= component_groups_.size() || component_groups_[gid] == nullptr) {
            continue;
        }
        const bool fully_prepared     = item_index < prepared_item_count;
        const bool partially_prepared = item_index == prepared_item_count && partial_item_claimed;
        if (item.source_tier != Tier::DEVICE
            && (fully_prepared || (partially_prepared && partial_target_holder_added))) {
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
            if (fully_prepared || partially_prepared) {
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
        const LoadBackItem& item = items[item_index];
        if (item.group_id < 0 || static_cast<size_t>(item.group_id) >= component_groups_.size()) {
            continue;
        }
        ComponentGroupPtr& group = component_groups_[static_cast<size_t>(item.group_id)];
        if (group != nullptr && !isNullBlockIdx(staging_host_blocks[item_index])) {
            group->releaseSingleBlock(Tier::HOST, staging_host_blocks[item_index]);
        }
    }

    const bool has_copy_items = std::any_of(
        items.begin(), items.end(), [](const LoadBackItem& item) { return item.source_tier != Tier::DEVICE; });
    const bool has_device_items = std::any_of(
        items.begin(), items.end(), [](const LoadBackItem& item) { return item.source_tier == Tier::DEVICE; });

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool                  effective_success =
            copy_success && load_back_context != nullptr && !load_back_context->cancelRequested();
        for (auto& item : items) {
            const auto gid = static_cast<size_t>(item.group_id);
            if (item.group_id < 0 || gid >= component_groups_.size() || component_groups_[gid] == nullptr) {
                continue;
            }

            auto&         group = component_groups_[gid];
            GroupBlockSet source_protection{item.group_id, item.source_tier, {item.source_blocks}};
            if (item.source_tier != Tier::DEVICE) {
                source_protection.nodes = {item.node};
            }
            group->unreferenceBlocks(source_protection);

            if (item.source_tier == Tier::DEVICE) {
                continue;
            }
            if (item.node == nullptr) {
                continue;
            }
            auto& slot = item.node->group_slots[gid];
            if (effective_success) {
                group->setBlocks(slot, Tier::DEVICE, item.target_device_blocks);
                group->unreferenceBlocks(GroupBlockSet{item.group_id, item.source_tier, {item.source_blocks}});
                group->evictFromTier(item.node, slot, item.source_tier);
            } else {
                group->unreferenceBlocks(GroupBlockSet{item.group_id, Tier::DEVICE, {item.target_device_blocks}});
            }
            evictor_.finishLoadBack(item.node, item.group_id, item.source_tier, effective_success);
        }
        if (has_device_items) {
            evictor_.refreshAllCandidates(*tree_);
        }
        if (effective_success && has_copy_items) {
            ++mutation_version_;
        }
        if (has_device_items || has_copy_items) {
            checkWatermark();
        }
        if (load_back_context != nullptr) {
            load_back_context->onTaskComplete(effective_success);
        }
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

void BlockTreeCache::reserveInFlightDeviceReleaseCreditsLocked(
    const std::vector<DeviceReleaseCredit>& release_credits) {
    for (const DeviceReleaseCredit& credit : release_credits) {
        if (credit.pool != nullptr) {
            ++in_flight_device_release_credits_[credit.pool];
        }
    }
}

void BlockTreeCache::settleInFlightDeviceReleaseCreditsLocked(const std::vector<DeviceReleaseCredit>& release_credits) {
    for (const DeviceReleaseCredit& credit : release_credits) {
        const auto it = in_flight_device_release_credits_.find(credit.pool);
        if (it == in_flight_device_release_credits_.end() || it->second == 0) {
            RTP_LLM_LOG_ERROR("BlockTreeCache: missing in-flight DEVICE release credit while settling pool=%p block=%d",
                              static_cast<void*>(credit.pool.get()),
                              credit.block);
            continue;
        }
        if (--it->second == 0) {
            in_flight_device_release_credits_.erase(it);
        }
    }
}

bool BlockTreeCache::submitEvictionLocked(EvictionMove&                     eviction_move,
                                          std::vector<DeviceReleaseCredit>* release_credits) {
    if (release_credits != nullptr) {
        release_credits->clear();
    }
    if (eviction_move.target_tier != Tier::NONE && !config_.isTierEnabled(eviction_move.target_tier)) {
        eviction_move.target_tier = Tier::NONE;
    }

    auto plan = evictor_.buildPlan(eviction_move);
    if (!plan.has_value()) {
        return false;
    }

    std::vector<DeviceReleaseCredit>                    accepted_release_credits;
    std::set<std::pair<DeviceBlockPool*, BlockIdxType>> accepted_physical_releases;
    auto                                                collect_device_credits = [&](const EvictionMove& move) {
        if (move.source_tier != Tier::DEVICE || move.component_group_id < 0) {
            return;
        }
        const size_t gid = static_cast<size_t>(move.component_group_id);
        if (gid >= component_groups_.size() || component_groups_[gid] == nullptr) {
            return;
        }
        const auto&  pools = component_groups_[gid]->devicePools();
        const size_t slot_count = std::min(pools.size(), move.source_blocks.size());
        for (size_t i = 0; i < slot_count; ++i) {
            const auto& pool = pools[i];
            if (!isNullBlockIdx(move.source_blocks[i]) && pool != nullptr
                && accepted_physical_releases.emplace(pool.get(), move.source_blocks[i]).second) {
                accepted_release_credits.push_back({pool, move.source_blocks[i]});
            }
        }
    };
    collect_device_credits(plan->primary);
    for (const EvictionMove& cascade_move : plan->cascade_moves) {
        collect_device_credits(cascade_move);
    }

    if (!plan->needsCopy()) {
        BlockTreeEvictor::CopyResultSet results;
        results.primary_success = true;
        results.cascade_success.assign(plan->cascade_moves.size(), true);
        evictor_.complete(*tree_, *plan, results);
        ++mutation_version_;
        if (release_credits != nullptr) {
            *release_credits = std::move(accepted_release_credits);
        }
        return true;
    }

    auto plan_ptr                  = std::make_shared<BlockTreeEvictor::EvictionPlan>(std::move(*plan));
    auto in_flight_release_credits = accepted_release_credits;
    taskStarted();
    auto* work_item =
        new autil::LambdaWorkItem([this, plan_ptr, in_flight_release_credits = std::move(in_flight_release_credits)]() {
            performEvictionCopy(*plan_ptr, in_flight_release_credits);
        });
    auto err = thread_pool_->pushWorkItem(work_item);
    if (err != autil::ThreadPool::ERROR_NONE) {
        work_item->destroy();
        evictor_.rollbackPreparedPlan(*plan_ptr);
        taskFinished();
        return false;
    }
    reserveInFlightDeviceReleaseCreditsLocked(accepted_release_credits);
    if (release_credits != nullptr) {
        *release_credits = std::move(accepted_release_credits);
    }
    return true;
}

void BlockTreeCache::performEvictionCopy(const BlockTreeEvictor::EvictionPlan&   plan,
                                         const std::vector<DeviceReleaseCredit>& release_credits) {
    BlockTreeEvictor::CopyResultSet copy_results;
    copy_results.primary_success = false;
    copy_results.cascade_success.assign(plan.cascade_moves.size(), false);

    auto worker_finalization_action = [this, &plan, &release_credits, &copy_results]() noexcept {
        bool task_finish_attempted = false;
        auto task_finish_action    = [this, &task_finish_attempted]() noexcept {
            if (task_finish_attempted) {
                return;
            }
            task_finish_attempted = true;
            try {
                taskFinished();
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("BlockTreeCache: accepted eviction task finalization failed: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("BlockTreeCache: accepted eviction task finalization failed with unknown exception");
            }
        };
        ScopeRollback<decltype(task_finish_action)> task_finish_guard(std::move(task_finish_action));

        bool credit_settlement_attempted = false;
        auto credit_settlement_action    = [this, &release_credits, &credit_settlement_attempted]() noexcept {
            if (credit_settlement_attempted) {
                return;
            }
            credit_settlement_attempted = true;
            try {
                std::lock_guard<std::mutex> lock(mutex_);
                settleInFlightDeviceReleaseCreditsLocked(release_credits);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("BlockTreeCache: DEVICE release-credit settlement failed: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("BlockTreeCache: DEVICE release-credit settlement failed with unknown exception");
            }
        };
        ScopeRollback<decltype(credit_settlement_action)> credit_settlement_guard(std::move(credit_settlement_action));

        bool completion_succeeded = false;
        bool plan_terminalized    = false;
        bool plan_succeeded       = false;
        bool copy_ok              = copy_results.primary_success;

        CacheKeyType remote_cache_key = 0;
        int          remote_group_id  = -1;
        if (copy_ok && plan.primary.node != nullptr) {
            remote_cache_key = plan.primary.node->cache_key;
            remote_group_id  = plan.primary.component_group_id;
        }

        try {
            std::lock_guard<std::mutex> lock(mutex_);
            try {
                evictor_.complete(*tree_, plan, copy_results);
                completion_succeeded = true;
                plan_terminalized    = true;
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("BlockTreeCache: eviction completion failed; rolling back accepted plan: %s",
                                  error.what());
                try {
                    evictor_.rollbackPreparedPlan(plan);
                    plan_terminalized = true;
                } catch (const std::exception& rollback_error) {
                    RTP_LLM_LOG_ERROR("BlockTreeCache: accepted eviction rollback failed: %s", rollback_error.what());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("BlockTreeCache: accepted eviction rollback failed with unknown exception");
                }
            } catch (...) {
                RTP_LLM_LOG_ERROR("BlockTreeCache: eviction completion failed with unknown exception; rolling back "
                                  "accepted plan");
                try {
                    evictor_.rollbackPreparedPlan(plan);
                    plan_terminalized = true;
                } catch (const std::exception& rollback_error) {
                    RTP_LLM_LOG_ERROR("BlockTreeCache: accepted eviction rollback failed: %s", rollback_error.what());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("BlockTreeCache: accepted eviction rollback failed with unknown exception");
                }
            }

            // Credits are accounting-only. The completed or rolled-back evictor plan above owns all
            // pool reference transitions; settlement must never add another decRef.
            credit_settlement_attempted = true;
            try {
                settleInFlightDeviceReleaseCreditsLocked(release_credits);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("BlockTreeCache: DEVICE release-credit settlement failed: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("BlockTreeCache: DEVICE release-credit settlement failed with unknown exception");
            }

            const bool mutated = plan_terminalized && completion_succeeded
                                 && (copy_results.primary_success
                                     || std::any_of(copy_results.cascade_success.begin(),
                                                    copy_results.cascade_success.end(),
                                                    [](bool success) { return success; }));
            plan_succeeded = plan_terminalized && completion_succeeded && copy_results.primary_success
                             && copy_results.cascade_success.size() == plan.cascade_moves.size()
                             && std::all_of(copy_results.cascade_success.begin(),
                                            copy_results.cascade_success.end(),
                                            [](bool success) { return success; });
            if (mutated) {
                ++mutation_version_;
            }
            if (plan_succeeded) {
                // A fully completed device->host or host->disk plan changes the target tier's
                // pressure. This remains under the cache lock, after this plan's credits settle.
                try {
                    checkWatermark();
                } catch (const std::exception& error) {
                    RTP_LLM_LOG_ERROR("BlockTreeCache: post-eviction watermark check failed: %s", error.what());
                } catch (...) {
                    RTP_LLM_LOG_ERROR("BlockTreeCache: post-eviction watermark check failed with unknown exception");
                }
            }
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("BlockTreeCache: eviction terminalization lock/follow-up failed: %s", error.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("BlockTreeCache: eviction terminalization lock/follow-up failed with unknown exception");
        }

        // If an exception escaped before the in-lock settlement attempt, perform that accounting step
        // now. The no-throw guard records one attempt and prevents a duplicate decrement.
        credit_settlement_guard.run();

        if (plan_terminalized && completion_succeeded && copy_ok && config_.enable_remote_cache
            && remote_group_id >= 0) {
            try {
                evictor_.writeRemoteThrough(storage_backend_, remote_cache_key, remote_group_id);
            } catch (const std::exception& error) {
                RTP_LLM_LOG_ERROR("BlockTreeCache: remote eviction write-through failed: %s", error.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("BlockTreeCache: remote eviction write-through failed with unknown exception");
            }
        }

        // This is deliberately last and no-throw: pending-task decrement is attempted exactly once
        // after the credit settlement attempt, even when terminalization or settlement reports failure.
        task_finish_guard.run();
    };
    ScopeRollback<decltype(worker_finalization_action)> worker_finalization_guard(
        std::move(worker_finalization_action));

    try {
        if (broadcast_manager_ == nullptr) {
            copy_results = evictor_.performCopy(plan);
        } else {
            MemoryOperationRequestPB request;
            const bool               request_ready = buildEvictionTransferRequest(plan, request);
            const bool copy_success      = request_ready && broadcastTransfer(request, evictionTransferTimeoutMs(plan));
            copy_results.primary_success = copy_success;
            copy_results.cascade_success.assign(plan.cascade_moves.size(), copy_success);
        }
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("BlockTreeCache: eviction copy failed with exception: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("BlockTreeCache: eviction copy failed with unknown exception");
    }
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
        struct PoolDeficit {
            DeviceBlockPoolPtr pool;
            size_t             deficit{0};
            size_t             accepted_credits{0};
        };
        std::vector<PoolDeficit>                     pool_deficits;
        std::unordered_map<DeviceBlockPool*, size_t> pool_indices;
        for (const auto& group : component_groups_) {
            if (group == nullptr) {
                continue;
            }
            for (const auto& pool : group->devicePools()) {
                if (pool == nullptr || pool_indices.count(pool.get()) != 0) {
                    continue;
                }
                const size_t capacity     = pool->totalBlocksNum();
                const size_t min_free     = std::min(config_.device_min_free_blocks, capacity);
                const size_t free_blocks  = pool->freeBlocksNum();
                const size_t deficit      = free_blocks < min_free ? min_free - free_blocks : 0;
                const auto   in_flight_it = in_flight_device_release_credits_.find(pool);
                const size_t in_flight_credits =
                    in_flight_it == in_flight_device_release_credits_.end() ? 0 : in_flight_it->second;
                pool_indices.emplace(pool.get(), pool_deficits.size());
                pool_deficits.push_back({pool, deficit, in_flight_credits});
            }
        }

        auto has_uncovered_deficit = [&]() {
            return std::any_of(pool_deficits.begin(), pool_deficits.end(), [](const PoolDeficit& state) {
                return state.accepted_credits < state.deficit;
            });
        };
        auto group_has_uncovered_deficit = [&](const ComponentGroupPtr& group) {
            if (group == nullptr) {
                return false;
            }
            for (const auto& pool : group->devicePools()) {
                const auto it = pool_indices.find(pool.get());
                if (it != pool_indices.end()) {
                    const auto& state = pool_deficits[it->second];
                    if (state.accepted_credits < state.deficit) {
                        return true;
                    }
                }
            }
            return false;
        };

        std::vector<bool> unavailable(component_groups_.size(), false);
        while (has_uncovered_deficit()) {
            bool round_progress = false;
            for (size_t group_index = 0; group_index < component_groups_.size(); ++group_index) {
                const auto& group = component_groups_[group_index];
                if (unavailable[group_index] || !group_has_uncovered_deficit(group)) {
                    continue;
                }
                auto eviction_move = evictor_.chooseVictim(group->component_group_id, Tier::DEVICE);
                if (!eviction_move.has_value()) {
                    unavailable[group_index] = true;
                    continue;
                }
                std::vector<DeviceReleaseCredit> release_credits;
                if (!submitEvictionLocked(*eviction_move, &release_credits)) {
                    unavailable[group_index] = true;
                    continue;
                }
                bool credited_uncovered_pool = false;
                for (const DeviceReleaseCredit& credit : release_credits) {
                    const auto it = pool_indices.find(credit.pool.get());
                    if (it == pool_indices.end()) {
                        continue;
                    }
                    auto& state = pool_deficits[it->second];
                    if (state.accepted_credits < state.deficit) {
                        ++state.accepted_credits;
                        credited_uncovered_pool = true;
                    }
                }
                if (credited_uncovered_pool) {
                    round_progress = true;
                } else {
                    unavailable[group_index] = true;
                }
            }
            if (!round_progress) {
                break;
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
