#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"

#include <algorithm>
#include <deque>
#include <future>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheLockGuard.h"

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/FullPrefixInvariantScanner.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/MockDiskCacheSource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/MockMemoryCacheSource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/remote_mock/RemoteMockAdapter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

// This state uses the existing completion pool. It owns only bounded logical
// metadata and source pins; RemoteMockWriteback owns each physical Host window.
struct BlockTreeCache::RemoteCompletionState: std::enable_shared_from_this<RemoteCompletionState> {
    using PreparedCompletion = BlockTreeTaskPool::PreparedCompletion;
    struct Intent {
        PreparedCompletion                                       progress;
        std::vector<TransferDescriptor>                          descriptors;
        std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>> pins;
        size_t                                                   next{0};
        bool                                                     settled{false};
        ~Intent() {
            releasePins();
        }
        void releasePins() {
            for (const auto& pin : pins) {
                pin.first->decTreeRef(pin.second, BlockTreeRefType::STORE);
            }
            pins.clear();
        }
    };
    BlockTreeCache*                     cache;
    std::mutex                          mutex;
    RemoteMockCompletionStats           counters;
    size_t                              descriptor_limit{0};
    bool                                stopping{false}, running{false}, physical_drain{false};
    std::deque<std::shared_ptr<Intent>> queue;
    std::vector<std::function<void()>>  drained;

    explicit RemoteCompletionState(BlockTreeCache* cache): cache(cache) {
        if (!cache->config_.business_queue_max_size) {
            throw std::invalid_argument("remote completion queue must be positive");
        }
        for (const auto& group : cache->groupSets()) {
            size_t capacity = SIZE_MAX;
            for (const auto& pool : group->devicePools()) {
                capacity = std::min(capacity, pool->totalBlocksNum());
            }
            if (capacity == SIZE_MAX || capacity > SIZE_MAX - descriptor_limit) {
                throw std::invalid_argument("invalid remote completion source capacity");
            }
            descriptor_limit += capacity;
        }
    }
    void post(std::function<void()> fn) {
        RTP_LLM_CHECK(cache->task_pool_->submitCompletion(std::move(fn)));
    }
    void postPrepared(PreparedCompletion& node) noexcept {
        // Teardown waits for the final barrier before shutting down the pool.
        RTP_LLM_CHECK(cache->task_pool_->submitPreparedCompletion(node));
    }
    PreparedCompletion prepareProgress() {
        auto self     = shared_from_this();
        auto final    = BlockTreeTaskPool::prepareCompletion([self] {
            std::vector<std::function<void()>> callbacks;
            {
                std::lock_guard<std::mutex> lock(self->mutex);
                self->physical_drain = false;
                if (!self->counters.pending_requests && !self->counters.pending_intents) {
                    callbacks.swap(self->drained);
                }
            }
            for (auto& callback : callbacks) {
                try {
                    callback();
                } catch (...) {
                    RTP_LLM_LOG_ERROR("remote completion drain callback threw");
                }
            }
        });
        auto physical = cache->remote_adapter_->prepareDrain(
            [self, final = std::move(final)]() mutable { self->postPrepared(final); });
        return BlockTreeTaskPool::prepareCompletion([self, physical = std::move(physical)]() mutable {
            self->startNext();
            {
                std::lock_guard<std::mutex> lock(self->mutex);
                if (self->drained.empty() || self->counters.pending_requests || self->counters.pending_intents
                    || self->physical_drain) {
                    return;
                }
                self->physical_drain = true;
            }
            physical();
        });
    }
    void retireRequest(PreparedCompletion& progress) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        --counters.pending_requests;
        if (!drained.empty()) {
            postPrepared(progress);
        }
    }
    bool append(const std::shared_ptr<Intent>& intent, size_t group_id, const BlockIndicesType& blocks) {
        if (group_id >= cache->groupSets().size()) {
            return false;
        }
        const auto& pools = cache->groupSets()[group_id]->devicePools();
        if (blocks.size() != pools.size()) {
            return false;
        }
        for (size_t member = 0; member < pools.size(); ++member) {
            if (!pools[member]->validBlock(blocks[member]) || !pools[member]->isAllocated(blocks[member])) {
                return false;
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (counters.retained_descriptors >= descriptor_limit) {
                return false;
            }
            ++counters.retained_descriptors;
        }
        // Reserve metadata before any source pin. All descriptors (including
        // future batches) are captured/pinned before the request can free them.
        try {
            intent->descriptors.push_back(TransferDescriptor::deviceToRemote(
                group_id, cache->remote_adapter_->session(), intent->descriptors.size(), blocks));
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            --counters.retained_descriptors;
            throw;
        }
        for (size_t member = 0; member < pools.size(); ++member) {
            const auto duplicate = std::find_if(intent->pins.begin(), intent->pins.end(), [&](const auto& pin) {
                return pin.first == pools[member] && pin.second == blocks[member];
            });
            if (duplicate == intent->pins.end()) {
                intent->pins.emplace_back(pools[member], blocks[member]);
                pools[member]->incTreeRef(blocks[member], BlockTreeRefType::STORE);
            }
        }
        return true;
    }
    void settle(const std::shared_ptr<Intent>& intent, bool success, bool was_running) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (intent->settled) {
                return;
            }
            intent->settled = true;
        }
        intent->releasePins();
        const auto count = intent->descriptors.size();
        std::vector<TransferDescriptor>().swap(intent->descriptors);
        std::vector<std::pair<DeviceBlockPoolPtr, BlockIdxType>>().swap(intent->pins);
        {
            std::lock_guard<std::mutex> lock(mutex);
            counters.retained_descriptors -= count;
            --counters.pending_intents;
            if (success) {
                ++counters.completed;
            } else {
                ++counters.failed;
            }
            if (was_running) {
                running = false;
            }
        }
        if (!success) {
            RTP_LLM_LOG_WARNING("accepted remote request completion failed; write coverage is incomplete");
        }
        postPrepared(intent->progress);
    }
    void runBatch(const std::shared_ptr<Intent>& intent) {
        if (intent->next == intent->descriptors.size()) {
            settle(intent, true, true);
            return;
        }
        const auto begin = intent->next;
        const auto group = intent->descriptors[begin].group_set_id;
        const auto limit = cache->remote_adapter_->maxWritebackDescriptors();
        size_t     end   = begin;
        while (end < intent->descriptors.size() && end - begin < limit
               && intent->descriptors[end].group_set_id == group) {
            ++end;
        }
        if (end == begin || cache->remoteMockSafety().faulted) {
            settle(intent, false, true);
            return;
        }
        try {
            std::vector<TransferDescriptor> batch_descs(intent->descriptors.begin() + begin,
                                                        intent->descriptors.begin() + end);
            TransferTask batch(batch_descs, std::chrono::milliseconds(cache->config_.mock_remote_cache_timeout_ms));
            auto         self                           = shared_from_this();
            TransferAsyncContext::DoneCallback terminal = [self, intent, descriptors = std::move(batch_descs)](
                                                              TransferResult result) {
                self->cache->tree_->recordPressureWorkflowFailure(CacheTransferOperation::STORE, descriptors, result);
                try {
                    self->post([self, intent, success = result.ok()] {
                        if (success) {
                            self->runBatch(intent);
                        } else {
                            self->settle(intent, false, true);
                        }
                    });
                } catch (...) {
                    // The dispatcher terminal proves all batches are done (or
                    // physically quarantined); posting failed before enqueue.
                    self->settle(intent, false, true);
                }
            };
            intent->next = end;
            cache->transfer_dispatcher_->runTransfer(
                CacheTransferOperation::STORE, std::move(batch), std::move(terminal));
        } catch (...) {
            // Dispatcher preparation is transactional: escaping exceptions
            // here precede every physical launch for this logical batch.
            settle(intent, false, true);
        }
    }

    void startNext() {
        std::shared_ptr<Intent> intent;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (running || queue.empty()) {
                return;
            }
            running = true;
            intent  = std::move(queue.front());
            queue.pop_front();
        }
        runBatch(intent);
    }
};

struct BlockTreeCache::RemoteCompletionRequest {
    std::shared_ptr<RemoteCompletionState>    state;
    RemoteCompletionState::PreparedCompletion progress;
    RemoteCompletionRequest(std::shared_ptr<RemoteCompletionState>    owner,
                            RemoteCompletionState::PreparedCompletion node):
        state(std::move(owner)), progress(std::move(node)) {}
    bool captured{false};  // Serialized by state->mutex.
    ~RemoteCompletionRequest() {
        state->retireRequest(progress);
    }
};

std::shared_ptr<BlockTreeCache::RemoteCompletionRequest> BlockTreeCache::enrollRemoteCompletion() {
    const auto state = remote_completions_;
    if (!state) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->stopping) {
        return nullptr;
    }
    auto request = std::make_shared<RemoteCompletionRequest>(state, state->prepareProgress());
    ++state->counters.pending_requests;
    return request;
}

void BlockTreeCache::completeRemoteRequest(const std::shared_ptr<RemoteCompletionRequest>&          request,
                                           const std::function<bool(const RemoteGeometryVisitor&)>& capture) {
    const auto state = remote_completions_;
    if (!state) {
        return;
    }
    std::shared_ptr<RemoteCompletionState::Intent> intent;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (request && request->captured) {
            return;
        }
        if (request) {
            request->captured = true;
        }
        if (!request || request->state != state || remoteMockSafety().faulted
            || state->counters.pending_intents >= config_.business_queue_max_size
            || state->counters.retained_descriptors >= state->descriptor_limit) {
            ++state->counters.rejected;
            RTP_LLM_LOG_WARNING("remote request completion rejected before admission; write coverage is incomplete");
            return;
        }
        // Intent metadata is part of admission. A failure here has accepted no
        // work and acquired no pins, so reject without provisional counters.
        try {
            intent           = std::make_shared<RemoteCompletionState::Intent>();
            intent->progress = state->prepareProgress();
        } catch (...) {
            ++state->counters.rejected;
            return;
        }
        ++state->counters.submitted;
        ++state->counters.pending_intents;
    }
    bool valid = false;
    try {
        valid =
            capture([&](size_t group, const BlockIndicesType& blocks) { return state->append(intent, group, blocks); });
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("remote completion capture failed: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("remote completion capture failed");
    }
    if (!valid) {
        state->settle(intent, false, false);
        return;
    }
    if (intent->descriptors.empty()) {
        // No canonical cacheable record is a legitimate no-op, not a write or
        // coverage failure. Retire the provisional admission without rejection.
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            --state->counters.submitted;
            --state->counters.pending_intents;
        }
        state->postPrepared(intent->progress);
        return;
    }
    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->queue.push_back(intent);
        try {
            // Hold the ownership lock through scheduling. Another startNext
            // cannot take this intent until both queue insertion and post commit.
            state->post([state] { state->startNext(); });
        } catch (...) {
            state->queue.pop_back();
            throw;
        }
    } catch (...) {
        state->settle(intent, false, false);
    }
}

void BlockTreeCache::stopRemoteCompletionAdmission() {
    if (remote_completions_) {
        std::lock_guard<std::mutex> lock(remote_completions_->mutex);
        remote_completions_->stopping = true;
    }
}

void BlockTreeCache::drainRemoteCompletions(std::function<void()> callback) {
    const auto state = remote_completions_;
    if (!state) {
        auto physical = remote_adapter_ ? remote_adapter_->prepareDrain(std::move(callback)) : std::move(callback);
        auto node     = BlockTreeTaskPool::prepareCompletion(std::move(physical));
        RTP_LLM_CHECK(task_pool_->submitPreparedCompletion(node));
        return;
    }
    auto progress = state->prepareProgress();
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->drained.push_back(std::move(callback));
    }
    state->postPrepared(progress);
}

BlockTreeCache::BlockTreeCache(std::unique_ptr<BlockTree>                     tree,
                               BlockTreeCacheConfig                           config,
                               std::shared_ptr<StorageBackend>                storage_backend,
                               std::unique_ptr<BlockTransferDispatcher>       transfer_dispatcher,
                               std::unique_ptr<BlockTreeTaskPool>             task_pool,
                               std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter,
                               std::shared_ptr<MockMemoryCacheSource>         mock_memory_cache_source,
                               std::shared_ptr<MockDiskCacheSource>           mock_disk_cache_source,
                               std::shared_ptr<RemoteMockAdapter>             remote_adapter):
    config_(std::move(config)),
    tree_(std::move(tree)),
    storage_backend_(std::move(storage_backend)),
    mock_memory_cache_source_(std::move(mock_memory_cache_source)),
    mock_disk_cache_source_(std::move(mock_disk_cache_source)),
    remote_adapter_(std::move(remote_adapter)),
    transfer_dispatcher_(std::move(transfer_dispatcher)),
    task_pool_(std::move(task_pool)),
    metrics_reporter_(std::move(metrics_reporter)),
    evictor_(
        tree_.get(),
        config_.device_eviction_policy,
        config_.host_eviction_policy,
        config_.disk_eviction_policy,
        transfer_dispatcher_.get(),
        task_pool_.get(),
        *metrics_reporter_,
        mutex_,
        config_.host_cache_sync_timeout_ms,
        config_.disk_cache_evict_timeout_ms,
        config_.max_descriptors_per_transfer_batch,
        [this](Tier tier) { return config_.isTierEnabled(tier); },
        [this](bool check_watermark) { onWorkflowSettledLocked(check_watermark); },
        config_.disk_cache_evict_max_pending_bytes),
    loader_(tree_.get(),
            evictor_,
            transfer_dispatcher_.get(),
            task_pool_.get(),
            *metrics_reporter_,
            mutex_,
            config_.disk_cache_sync_timeout_ms,
            config_.host_cache_sync_timeout_ms,
            config_.enable_device_cache,
            storage_backend_,
            [this](bool check_watermark) { onWorkflowSettledLocked(check_watermark); },
            mock_memory_cache_source_,
            mock_disk_cache_source_,
            remote_adapter_.get(),
            config_.mock_remote_cache_timeout_ms),
    storer_(tree_.get(),
            evictor_,
            transfer_dispatcher_.get(),
            task_pool_.get(),
            *metrics_reporter_,
            mutex_,
            config_.host_cache_sync_timeout_ms,
            config_.disk_cache_sync_timeout_ms,
            storage_backend_,
            [this](bool check_watermark) { onWorkflowSettledLocked(check_watermark); }) {
    metrics_reporter_->initEvictionMetricGroupTypes(tree_->groupSets());
    if (storage_backend_) {
        storage_backend_->setStageMetricsReporter(metrics_reporter_);
    }
    transfer_dispatcher_->setLocalCopyObserver(
        [tree = tree_.get()](const TransferTask& task, const TransferResult& result) {
            tree->recordPressureCopy(*task.operation, task.descriptors(), result);
        });
}

bool BlockTreeCache::init() {
    if (initialized_) {
        RTP_LLM_LOG_ERROR("cache is already initialized");
        return false;
    }
    if (!task_pool_->start()) {
        RTP_LLM_LOG_ERROR("failed to start task pool, size=%d", config_.task_pool_size);
        return false;
    }
    RTP_LLM_LOG_INFO("initialized with %zu group sets, %zu reusable topology groups, "
                     "pool_threads=%d, storage_backend=%s, "
                     "device=%s, host=%s, disk=%s, remote=%s",
                     tree_->groupSets().size(),
                     tree_->reusableGroupCount(),
                     config_.task_pool_size,
                     storage_backend_ ? "enabled" : "null",
                     config_.enable_device_cache ? "on" : "off",
                     config_.enable_host_cache ? "on" : "off",
                     config_.enable_disk_cache ? "on" : "off",
                     config_.enable_remote_cache ? "on" : "off");
    for (const GroupSetPtr& group_set : tree_->groupSets()) {
        RTP_LLM_LOG_INFO("  group_set[%zu] type=%s host_pool=%s disk_pool=%s",
                         group_set->groupSetId(),
                         cacheGroupTypeName(group_set->groupType()),
                         group_set->hostPool() ? "enabled" : "null",
                         group_set->diskPool() ? "enabled" : "null");
    }
    if (config_.full_prefix_scan_interval_ms > 0) {
        auto scanner = std::make_unique<FullPrefixInvariantScanner>(
            *tree_, mutex_, config_.full_prefix_scan_interval_ms, metrics_reporter_.get());
        if (scanner->start()) {
            full_prefix_scanner_ = std::move(scanner);
            RTP_LLM_LOG_INFO("FULL prefix invariant scanner enabled, interval_ms=%ld",
                             config_.full_prefix_scan_interval_ms);
        }
    }
    if (remote_adapter_ && config_.mock_memory_cache_coordinator) {
        remote_completions_ = std::make_shared<RemoteCompletionState>(this);
    }
    initialized_ = true;
    if (config_.mock_memory_cache) {
        metrics_reporter_->updateMockMemorySource(tree_->groupSets());
    }
    return true;
}

BlockTreeCache::~BlockTreeCache() {
    stopping_.store(true);
    stopRemoteCompletionAdmission();
    if (remote_completions_) {
        std::promise<void> drained;
        auto               done = drained.get_future();
        drainRemoteCompletions([&drained] { drained.set_value(); });
        done.wait();
    }
    // Permanent stop precedes both request settlement and quarantine release.
    // External references to these pools cannot allocate after cache teardown.
    if (remote_adapter_) {
        for (const auto& group : tree_->groupSets()) {
            for (const auto& pool : group->devicePools()) {
                pool->stopAllocation();
            }
        }
        remote_adapter_->stopAccepting();
    }
    full_prefix_scanner_.reset();
    RTP_LLM_LOG_INFO("destroying, closing load tickets...");
    shutdownLoads();
    if (storage_backend_) {
        storage_backend_->shutdown();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        storer_.stopAdmissionLocked();
    }
    if (remote_adapter_) {
        remote_adapter_->stopAccepting();
    }
    RTP_LLM_LOG_INFO("load tickets closed, draining cache tasks...");
    task_pool_->stopAdmission();
    transfer_dispatcher_->cancelPendingStagingTransfers();
    task_pool_->waitForIdle();
    if (remote_adapter_) {
        std::promise<void> drained;
        auto               completed = drained.get_future();
        remote_adapter_->drain([&drained] { drained.set_value(); });
        completed.wait();
    }
    // Task-body idleness does not imply asynchronous RPC/transfer completion.
    // Keep dispatcher and cache state alive until transfer callbacks return,
    // then drain the settlement tasks those callbacks submitted.
    transfer_dispatcher_->shutdown();
    task_pool_->waitForIdle();
    remote_adapter_.reset();
    transfer_dispatcher_.reset();
    task_pool_.reset();
    RTP_LLM_LOG_INFO("destroyed");
}

RemoteMockSafetySnapshot BlockTreeCache::remoteMockSafety() const {
    return transfer_dispatcher_->remoteMockSafety();
}

void BlockTreeCache::shutdownLoads() {
    loader_.shutdown();
}

void BlockTreeCache::onCapacityChanged() {
    if (stopping_.load() || watermark_check_pending_.exchange(true)) {
        return;
    }
    // Capacity callbacks can run under mutex_ while tree references are released.
    // Queue the check and coalesce wakeups instead of reacquiring that mutex inline.
    if (!task_pool_->submitCompletion([this] {
            watermark_check_pending_.store(false);
            std::lock_guard<std::mutex> lock(mutex_);
            if (!stopping_.load()) {
                checkWatermark();
            }
        })) {
        watermark_check_pending_.store(false);
    }
}

std::vector<std::shared_ptr<FrozenTask>> BlockTreeCache::frozenTasks() const {
    return transfer_dispatcher_ ? transfer_dispatcher_->frozenTasks() : std::vector<std::shared_ptr<FrozenTask>>{};
}

TransferTaskRegistry::ForgetResult BlockTreeCache::forgetLocalTransferCompletion(uint64_t trace_id) const {
    return transfer_dispatcher_ ? transfer_dispatcher_->forgetLocalCompletion(trace_id) :
                                  TransferTaskRegistry::ForgetResult::FAILED;
}

void BlockTreeCache::setCompletionCleanup(std::shared_ptr<TransferCompletionCleanup> cleanup) {
    if (transfer_dispatcher_)
        transfer_dispatcher_->setCompletionCleanup(std::move(cleanup));
}

TransferTaskRegistry::State BlockTreeCache::localTransferCompletion(uint64_t trace_id) const {
    if (!transfer_dispatcher_) {
        return TransferTaskRegistry::State::UNKNOWN;
    }
    return transfer_dispatcher_->queryLocalCompletion(trace_id);
}

TransferResult BlockTreeCache::executeTransfer(TransferTask task) {
    auto context = transfer_dispatcher_->executePerRank(std::move(task));
    context->waitDone();
    auto result = context->result();
    if (!result.ok()) {
        RTP_LLM_LOG_WARNING("per-rank block transfer failed: %s", result.status.ToString().c_str());
    }
    return result;
}

BlockTreeMatchResult BlockTreeCache::match(const CacheKeysType&               cache_keys,
                                           const std::function<bool(size_t)>& valid_prefix,
                                           bool                               enable_storage_backend,
                                           std::optional<int64_t>             request_id,
                                           bool                               skip_linear) {
    BlockTreeMatchResult result;
    {
        CacheLockGuard lock(mutex_, *metrics_reporter_, "match");
        result = loader_.matchLocked(cache_keys, valid_prefix, enable_storage_backend, request_id, skip_linear);
    }
    metrics_reporter_->reportCacheReuseTimeMetrics(result.reuse_time_metrics);
    return result;
}

BlockTreeMatchResult BlockTreeCache::matchMockCache(const CacheKeysType&                     cache_keys,
                                                    const std::function<bool(size_t)>&       valid_prefix,
                                                    const BlockTreeLoader::MockCachePlanner& planner) {
    if (!config_.mock_memory_cache_coordinator) {
        return {};
    }
    BlockTreeMatchResult result;
    {
        CacheLockGuard lock(mutex_, *metrics_reporter_, "match");
        if (remoteMockSafety().faulted) {
            result.mock_load_failed = true;
            RTP_LLM_LOG_ERROR("remote mock TP transport outcome unknown; full engine restart required");
            return result;
        }
        result = loader_.matchMockCacheLocked(
            cache_keys,
            valid_prefix,
            [&](size_t eligible, const auto& valid_gpu) {
                auto decision = planner(eligible, valid_gpu);
                metrics_reporter_->updateMockCachePlan(decision);
                return decision;
            },
            metrics_reporter_);
    }
    if (result.mock_load_failed) {
        metrics_reporter_->recordMockCacheLoad(
            result.mock_memory_blocks, result.mock_disk_blocks, false, result.mock_remote_blocks);
    }
    metrics_reporter_->reportCacheReuseTimeMetrics(result.reuse_time_metrics);
    return result;
}

void BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                            const std::vector<std::vector<GroupSetResource>>& resources,
                            Tier                                              target_tier) {
    (void)insert(cache_keys, resources, target_tier, false);
}

size_t BlockTreeCache::insert(const CacheKeysType&                              cache_keys,
                              const std::vector<std::vector<GroupSetResource>>& resources,
                              Tier                                              target_tier,
                              bool                                              is_resident,
                              std::optional<int64_t>                            request_id,
                              size_t                                            remote_reuse_begin,
                              size_t                                            remote_reuse_end,
                              bool                                              mock_memory_cache,
                              bool                                              mock_disk_cache) {
    size_t           resident_prefix_length = 0;
    StorageWriteTask storage_write;
    {
        CacheLockGuard lock(mutex_, *metrics_reporter_, "insert");
        storage_write = storer_.storeLocked(cache_keys,
                                            resources,
                                            target_tier,
                                            is_resident,
                                            resident_prefix_length,
                                            request_id,
                                            remote_reuse_begin,
                                            remote_reuse_end,
                                            mock_memory_cache,
                                            mock_disk_cache);
    }
    if (storage_write) {
        storage_backend_->write(std::move(storage_write));
    }
    return resident_prefix_length;
}

int BlockTreeCache::evictForGroup(std::string_view group_tag, size_t num_blocks) {
    CacheLockGuard lock(mutex_,
                        "force_drop",
                        CacheStageContext::forLock(metrics_reporter_.get(), "force_drop", Tier::DEVICE, Tier::NONE));
    if (!config_.isTierEnabled(Tier::DEVICE)) {
        return 0;
    }
    const ReusableGroupLocation* location = tree_->reusableGroupLocation(group_tag);
    if (location == nullptr) {
        return 0;
    }
    const size_t reclaimed =
        evictor_.reclaimDeviceBlocksLocked(location->group_set_id, location->member_group_id, num_blocks);
    RTP_LLM_LOG_DEBUG("tag=%.*s group_set[%zu] reclaimed %zu/%zu device blocks",
                      static_cast<int>(group_tag.size()),
                      group_tag.data(),
                      location->group_set_id,
                      reclaimed,
                      num_blocks);
    return static_cast<int>(reclaimed);
}

BlockIndicesType BlockTreeCache::matchedBlocksForGroup(std::string_view                      group_tag,
                                                       const std::vector<MultiNodeResource>& matched_resources) const {
    return loader_.matchedBlocksForGroup(group_tag, matched_resources);
}

CacheStats BlockTreeCache::getStats() const {
    CacheLockGuard lock(mutex_, *metrics_reporter_, "other");
    CacheStats     stats;
    stats.tree_node_count           = tree_->size();
    const CandidateStats candidates = evictor_.candidateStats();
    stats.device_heap_total_size    = candidates.device_candidates;
    stats.host_heap_total_size      = candidates.host_candidates;
    stats.disk_heap_total_size      = candidates.disk_candidates;
    return stats;
}

std::vector<CachePoolMetrics> BlockTreeCache::poolMetrics() const {
    CacheLockGuard lock(mutex_, *metrics_reporter_, "other");
    return metrics_reporter_->collectPoolMetrics(tree_->groupSets());
}

CachePressureSnapshot BlockTreeCache::pressureSnapshot() const {
    CachePressureSnapshot snapshot;
    snapshot.ownership          = config_.pressure_ownership;
    snapshot.remote_mock_safety = remoteMockSafety();
    if (remote_adapter_) {
        snapshot.remote_runtime                          = remote_adapter_->runtimeSnapshot();
        snapshot.remote_read_payload_bytes               = remote_adapter_->stats().read_payload_bytes;
        snapshot.remote_write_acknowledged_payload_bytes = remote_adapter_->writebackStats().acknowledged_payload_bytes;
    }
    if (remote_completions_) {
        std::lock_guard<std::mutex> lock(remote_completions_->mutex);
        snapshot.mock_remote_write = remote_completions_->counters;
    }
    if ((config_.mock_memory_cache || config_.mock_disk_cache || config_.mock_remote_cache)
        && config_.mock_memory_cache_coordinator) {
        // Acquire the stream-release frontier before reading physical counters.
        // A released request has already returned from all its local insertions.
        if (config_.mock_memory_cache || config_.mock_disk_cache) {
            snapshot.local_store = metrics_reporter_->localStoreSnapshot();
        }
        snapshot.mock_cache      = metrics_reporter_->mockCacheMetricsSnapshot();
        snapshot.mock_disk_write = metrics_reporter_->mockDiskWriteMetricsSnapshot();
    }
    if (config_.mock_memory_cache && config_.mock_memory_cache_coordinator) {
        snapshot.mock_memory = metrics_reporter_->mockMemoryMetricsSnapshot();
    }
    // Lock order: cache -> individual pool / pending-release / ledger mutex.
    // Each subordinate lock is released before acquiring another. Copy callbacks
    // take only the ledger lock. No serialization or tree walk occurs here.
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& physical : tree_->pressurePools()) {
        const auto*               pool = physical.pool;
        CachePressurePoolSnapshot record;
        static_cast<BlockTreePoolMetricsSnapshot&>(record) =
            BlockTreeCacheMetricsReporter::snapshotPool(physical.tier, *pool);
        if (physical.tier == Tier::DISK) {
            const auto* disk        = static_cast<const DiskBlockPool*>(pool);
            record.disk_read_bytes  = disk->readBytes();
            record.disk_write_bytes = disk->writeBytes();
        }
        record.pool_id                      = physical.id;
        record.request_ref_blocks_available = physical.tier == Tier::DEVICE;
        record.pending_release_blocks       = evictor_.pendingReleaseBlocks(pool);
        record.published_host_blocks        = tree_->pressureState().pool(physical.id).published_host_blocks;
        snapshot.pools.push_back(std::move(record));
    }
    for (const auto& group : tree_->groupSets()) {
        CachePressureGroupSnapshot record;
        record.group_id      = group->groupSetId();
        record.group_type    = group->groupType();
        record.group_tags    = group->groupTags();
        record.block_tokens  = group->group(group->groupTags().front()).seqSizePerBlock();
        record.payload_bytes = group->payloadBytes();
        record.storage_bytes = group->storageBytes();
        record.crc_enabled   = group->crcEnabled();
        std::vector<const IBlockPool*> pools;
        for (const auto& pool : group->devicePools()) {
            if (pool) {
                record.device_pool_ids.push_back(tree_->pressurePoolId(pool.get()));
                pools.push_back(pool.get());
            }
        }
        if (group->hostPool()) {
            record.host_pool_id = tree_->pressurePoolId(group->hostPool().get());
            pools.push_back(group->hostPool().get());
        }
        if (group->diskPool()) {
            record.disk_pool_id = tree_->pressurePoolId(group->diskPool().get());
            pools.push_back(group->diskPool().get());
        }
        record.device_watermark        = config_.watermark_device;
        record.host_watermark          = config_.watermark_host;
        record.disk_watermark          = config_.watermark_disk;
        record.watermark_trigger_count = tree_->pressureState().watermarkTriggers(record.group_id);
        snapshot.groups.push_back(record);
        // Transfer evidence uses real physical endpoints and bounded tier directions.
        // Deduplicate member pools shared within a group as well as across groups.
        std::unordered_set<const IBlockPool*> visited;
        for (const auto* pool : pools) {
            if (!visited.insert(pool).second) {
                continue;
            }
            const auto& id        = tree_->pressurePoolId(pool);
            Tier        pool_tier = Tier::DEVICE;
            if (group->hostPool().get() == pool) {
                pool_tier = Tier::HOST;
            } else if (group->diskPool().get() == pool) {
                pool_tier = Tier::DISK;
            }
            for (auto operation :
                 {CacheTransferOperation::LOAD, CacheTransferOperation::EVICT, CacheTransferOperation::STORE}) {
                for (auto source : {Tier::DEVICE, Tier::HOST, Tier::DISK, Tier::REMOTE}) {
                    for (auto target : {Tier::DEVICE, Tier::HOST, Tier::DISK, Tier::REMOTE}) {
                        if ((source == Tier::REMOTE || target == Tier::REMOTE)
                            && (!remote_adapter_ || pool_tier != Tier::DEVICE)) {
                            continue;
                        }
                        if (source == target || (pool_tier != source && pool_tier != target)) {
                            continue;
                        }
                        const auto evidence =
                            tree_->pressureState().transfer({record.group_id, id, operation, source, target});
                        if (!evidence.copied_blocks && !evidence.published_blocks && !evidence.failed_operations
                            && !evidence.physical_failed_operations) {
                            continue;
                        }
                        CachePressureTransferSnapshot transfer;
                        static_cast<TransferEvidence&>(transfer) = evidence;
                        transfer.group_id                        = record.group_id;
                        transfer.pool_id                         = id;
                        transfer.pool_tier                       = pool_tier;
                        transfer.operation                       = operation;
                        transfer.source_tier                     = source;
                        transfer.target_tier                     = target;
                        snapshot.transfers.push_back(std::move(transfer));
                    }
                }
            }
        }
    }
    snapshot.complete = transfer_dispatcher_->localCopyEvidenceComplete() && tree_->workflowEvidenceComplete();
    return snapshot;
}

void BlockTreeCache::cleanupExpiredMissing() const {
    if (transfer_dispatcher_)
        transfer_dispatcher_->cleanupExpiredMissing();
}

void BlockTreeCache::reportUnfreezeMetrics(const FrozenTask::ExecutionInfo& execution, bool forced) const {
    metrics_reporter_->reportUnfreezeMetrics(execution, forced);
}

void BlockTreeCache::reportUnfreezeZeroMetrics(const std::vector<CachePoolMetrics>& pools) const {
    metrics_reporter_->reportUnfreezeZeroMetrics(pools);
}

void BlockTreeCache::reportMetrics() const {
    EvictionVictimMetrics metrics_data;
    {
        CacheLockGuard lock(mutex_, *metrics_reporter_, "other");
        metrics_data = evictor_.getAndResetEvictionVictimMetrics();
    }
    metrics_reporter_->reportEvictionMetrics(metrics_data);
    metrics_reporter_->reportQueueBacklog(task_pool_->queueSizes(), "business");
    metrics_reporter_->reportQueueBacklog(transfer_dispatcher_->queueSizes(), "transfer");
    if (storage_backend_)
        metrics_reporter_->reportQueueBacklog(storage_backend_->queueSizes(), "remote");
    if (config_.mock_memory_cache) {
        metrics_reporter_->reportMockMemoryMetrics();
    }
}

void BlockTreeCache::setEventPublisher(KVCacheEventPublisherPtr        publisher,
                                       const std::vector<std::string>& required_group_tags) {
    CacheLockGuard lock(mutex_, *metrics_reporter_, "other");
    tree_->setEventPublisher(std::move(publisher), required_group_tags);
}

KVCacheSnapshot BlockTreeCache::logicalCacheSnapshot() const {
    CacheLockGuard lock(mutex_, *metrics_reporter_, "snapshot");
    return tree_->logicalCacheSnapshot();
}

BlockTreeKeySnapshot BlockTreeCache::getKeySnapshot(int64_t latest_version) const {
    CacheLockGuard       lock(mutex_, *metrics_reporter_, "snapshot");
    BlockTreeKeySnapshot snapshot;
    snapshot.version = tree_->keyVersion();
    if (latest_version != snapshot.version) {
        snapshot.keys = tree_->cacheKeys();
    }
    return snapshot;
}

bool BlockTreeCache::abortPendingLoad(const std::shared_ptr<AsyncContext>& context) {
    return loader_.abortPendingLoad(context);
}

void BlockTreeCache::onWorkflowSettledLocked(bool check_watermark) {
    if (check_watermark) {
        checkWatermark();
    }
}

void BlockTreeCache::checkWatermark() {
    for (Tier tier : {Tier::DISK, Tier::HOST, Tier::DEVICE}) {
        const auto watermark = config_.watermarkForTier(tier);
        if (!config_.isTierEnabled(tier)) {
            continue;
        }
        evictor_.scheduleWatermarkEvictionsLocked(tier, watermark);
    }
}

}  // namespace rtp_llm
