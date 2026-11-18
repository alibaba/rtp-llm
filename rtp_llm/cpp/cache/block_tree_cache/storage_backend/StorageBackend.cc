#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

#include <algorithm>
#include <mutex>
#include <exception>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageTimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"

namespace rtp_llm::storage_backend_detail {

thread_local const StorageBackend* completing_backend = nullptr;

uint64_t nextTraceId() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

template<typename Callback>
bool invokeCallback(const StorageBackend* backend, Callback&& callback) noexcept {
    struct CompletionScope {
        const StorageBackend* previous = completing_backend;
        ~CompletionScope() {
            completing_backend = previous;
        }
    } scope;
    completing_backend = backend;
    try {
        callback();
        return true;
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("StorageBackend completion failed: %s", error.what());
    } catch (...) { RTP_LLM_LOG_ERROR("StorageBackend completion failed with an unknown exception"); }
    return false;
}

struct StorageTaskState {
    struct Pin {
        DeviceBlockPoolPtr pool;
        BlockIdxType       block;
    };

    StorageRequest   request;
    uint64_t                          trace_id{0};
    std::vector<Pin> pins;
    // Parallel to pins for writes and protected reads; one task sidecar, never a per-pin field.
    // Inline optional container: ordinary reads allocate no token storage;
    // writes avoid a separate heap allocation for the vector object itself.
    std::optional<std::vector<RemotePublicationPin>> publication_pins;
    // Backend-owned unresolved-I/O token. Backend shutdown keeps its original
    // transport owner alive until the token and protected tasks are discharged.
    std::shared_ptr<void> completion_resources;
    std::shared_ptr<StorageTaskState> quarantine_next;
    bool completion_protected{false};
    bool quarantined{false};
    std::once_flag   finish_once;
    std::atomic<bool>                 completion_started{false};
    void             finish() {
        std::call_once(finish_once, [this] {
            if (publication_pins) {
                publication_pins->clear();
            } else {
                for (Pin& pin : pins)
                    pin.pool->decRef(pin.block, DeviceBlockRefType::REMOTE_CONNECTOR);
            }
            pins.clear();
            request.source_owner.reset(); // Endpoint connector pins also retire after native freeze/completion.
            request.completion_admission.reset();
            completion_resources.reset();
        });
    }

    ~StorageTaskState() {
        finish();
    }
};

// Keep one copy of the settlement state across the worker and RPC callback.
template<typename Result, typename Settle>
auto makeIoCompletion(std::shared_ptr<StorageBackendExecutor> executor,
                      std::shared_ptr<StorageTaskState> state, Settle settle) {
    // Prepare the shared settlement owner before I/O, without reserving a
    // completion slot. A queue allocation failure still uses the inline fallback.
    auto continuation = std::make_shared<Settle>(std::move(settle));
    return [executor = std::move(executor), state = std::move(state), continuation](Result result, bool asynchronous, std::exception_ptr error) mutable {
        if (state->completion_started.exchange(true))
            return;
        if (asynchronous) {
            try {
                if (executor->submitCompletion([continuation, result, error]() mutable {
                        (*continuation)(std::move(result), true, false, std::move(error));
                    }))
                    return;
            } catch (...) {}
        }
        (*continuation)(std::move(result), !asynchronous, false, std::move(error));
    };
}

}  // namespace rtp_llm::storage_backend_detail

namespace rtp_llm {
namespace {

struct BlockKey {
    DeviceBlockPool* pool;
    BlockIdxType     block;
    bool             operator==(const BlockKey& other) const {
        return pool == other.pool && block == other.block;
    }
};

struct BlockKeyHash {
    size_t operator()(const BlockKey& key) const {
        return std::hash<DeviceBlockPool*>{}(key.pool) ^ (std::hash<BlockIdxType>{}(key.block) << 1U);
    }
};

}  // namespace

StorageWriteTask::StorageWriteTask(std::shared_ptr<storage_backend_detail::StorageTaskState> state):
    state_(std::move(state)) {}

StorageBackend::StorageBackend(std::shared_ptr<StorageBackendExecutor>        executor,
                               std::shared_ptr<BlockTreeCacheMetricsReporter> reporter,
                               int64_t                                        world_rank,
                               std::chrono::milliseconds                      timeout):
    executor_(std::move(executor)),
    stage_metrics_reporter_(std::move(reporter)),
    stage_metrics_enabled_(stage_metrics_reporter_ && stage_metrics_reporter_->stageMetricsEnabled()),
    world_rank_(world_rank),
    timeout_(timeout) {}

void StorageBackend::setStageMetricsReporter(std::shared_ptr<BlockTreeCacheMetricsReporter> reporter) {
    const bool enabled = reporter && reporter->stageMetricsEnabled();
    std::atomic_store(&stage_metrics_reporter_, std::move(reporter));
    stage_metrics_enabled_.store(enabled, std::memory_order_release);
}

SharedTransferStageTimer StorageBackend::stageTimer(const char*           stage,
                                                    const StorageRequest& request,
                                                    const char*           operation,
                                                    uint64_t              trace_id,
                                                    AsyncProfileKind      kind) const noexcept {
    return TransferStageTimer::create(
        stage_metrics_enabled_.load(std::memory_order_acquire) ? std::atomic_load(&stage_metrics_reporter_) : nullptr,
        stage,
        request.request_id.value_or(-1),
        0,
        request.handles.size(),
        world_rank_,
        "storage_backend",
        trace_id,
        CacheStageContext::storage(operation),
        kind);
}

TransferStageTimer StorageBackend::stageValue(const char*           stage,
                                              const StorageRequest& request,
                                              const char*           operation,
                                              uint64_t              trace_id) const noexcept {
    return TransferStageTimer(
        stage_metrics_enabled_.load(std::memory_order_acquire) ? std::atomic_load(&stage_metrics_reporter_) : nullptr,
        stage,
        request.request_id.value_or(-1),
        0,
        request.handles.size(),
        world_rank_,
        "storage_backend",
        trace_id,
        CacheStageContext::storage(operation));
}

StorageBackend::ExternalIoGuard::ExternalIoGuard(ExternalIoGuard&& other) noexcept:
    backend_(std::exchange(other.backend_, nullptr)) {}

StorageBackend::ExternalIoGuard& StorageBackend::ExternalIoGuard::operator=(ExternalIoGuard&& other) noexcept {
    if (this != &other) {
        if (backend_) { backend_->taskFinished(); }
        backend_ = std::exchange(other.backend_, nullptr);
    }
    return *this;
}

StorageBackend::ExternalIoGuard::~ExternalIoGuard() {
    if (backend_) { backend_->taskFinished(); }
}

StorageBackend::ExternalIoGuard StorageBackend::trackExternalIo() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (lifecycle_ != Lifecycle::ACCEPTING) { return {}; }
    ++in_flight_;
    return ExternalIoGuard(this);
}

StorageBackend::~StorageBackend() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    RTP_LLM_CHECK_WITH_INFO(lifecycle_ == Lifecycle::CREATED || lifecycle_ == Lifecycle::STOPPED,
                            "StorageBackend must be shutdown before derived destruction");
}

bool StorageBackend::init(std::shared_ptr<const CacheTopology> topology,
                          PoolsByTag                           pools_by_tag,
                          BufferResolver                       buffer_resolver) {
    if (init_attempted_) {
        RTP_LLM_LOG_ERROR("StorageBackend initialization has already been attempted");
        return false;
    }
    RTP_LLM_CHECK(topology && buffer_resolver);
    RTP_LLM_CHECK(pools_by_tag.size() == topology->groups().size());
    std::unordered_set<const DeviceBlockPool*> seen_pools;
    for (const auto& [tag, pool] : pools_by_tag) {
        (void)topology->group(tag);
        RTP_LLM_CHECK_WITH_INFO(pool != nullptr, "null storage pool for tag=%s", tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            seen_pools.emplace(pool.get()).second, "storage tags must own distinct pools: tag=%s", tag.c_str());
    }
    init_attempted_ = true;
    if (executor_ == nullptr) {
        executor_ = makeDefaultStorageBackendExecutor();
    }
    if (executor_->bound_to_backend_.exchange(true)) {
        RTP_LLM_LOG_ERROR("StorageBackend executor cannot be shared between backends");
        return false;
    }
    topology_            = std::move(topology);
    pools_by_tag_        = std::move(pools_by_tag);
    buffer_resolver_     = std::move(buffer_resolver);
    const auto fail_init = [this] {
        shutdownImpl();
        buffer_resolver_ = {};
        pools_by_tag_.clear();
        topology_.reset();
        return false;
    };
    bool impl_initialized = false;
    try {
        impl_initialized = initImpl();
    } catch (...) {}
    if (!impl_initialized) {
        // The executor has never started and may still be used by another backend.
        executor_->bound_to_backend_.store(false);
        return fail_init();
    }
    bool started = false;
    try {
        started = executor_->start();
    } catch (...) {}
    if (!started) {
        executor_->shutdown();
        // Keep the binding: shutdown executors are terminal and must not be reused.
        return fail_init();
    }
    initialized_ = true;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        lifecycle_ = Lifecycle::ACCEPTING;
    }
    return true;
}

bool StorageBackend::dispatch(Operation operation, BlockTreeTaskClass task_class, bool run_inline,
                              std::optional<StorageBackendExecutor::Clock::time_point> io_deadline) {
    const auto once = std::make_shared<std::once_flag>();
    const auto finish_once = std::make_shared<std::once_flag>();
    Lifecycle  outcome;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        outcome = lifecycle_;
        if (outcome == Lifecycle::FINALIZING) {
            outcome = Lifecycle::STOPPED;
        } else if (outcome != Lifecycle::STOPPED) {
            RTP_LLM_CHECK(outcome != Lifecycle::CREATED);
            ++in_flight_;
        }
    }
    auto complete = [this, once, finish_once, counted = outcome != Lifecycle::STOPPED,
                     operation = std::move(operation)](Lifecycle result) mutable {
        std::call_once(*once, [&] {
            auto finished = [this, finish_once, counted] {
                if (counted)
                    std::call_once(*finish_once, [this] { taskFinished(); });
            };
            if (!storage_backend_detail::invokeCallback(this, [&] { operation(result, finished); })) {
                finished();
            }
        });
    };
    if (outcome == Lifecycle::STOPPED) {
        complete(outcome);
        return false;
    }
    if (outcome != Lifecycle::ACCEPTING) {
        complete(Lifecycle::STOPPING);
        return false;
    }
    // Local-only completions must not queue behind stalled remote I/O or
    // become allocation errors when that executor is full.
    if (run_inline) {
        complete(Lifecycle::ACCEPTING);
        return true;
    }
    try {
        if (executor_->submitWithDeadline(
                [complete]() mutable { complete(Lifecycle::ACCEPTING); }, task_class, io_deadline)) {
            return true;
        }
    } catch (...) {}
    complete(Lifecycle::STOPPING);
    return false;
}

void StorageBackend::taskFinished() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    RTP_LLM_CHECK(in_flight_ > 0);
    if (--in_flight_ == 0) {
        lifecycle_cv_.notify_all();
    }
}

void StorageBackend::shutdown() {
    RTP_LLM_CHECK_WITH_INFO(storage_backend_detail::completing_backend != this,
                            "StorageBackend shutdown cannot run from its callback");
    std::shared_ptr<StorageBackendExecutor> executor;
    {
        std::unique_lock<std::mutex> lock(lifecycle_mutex_);
        if (lifecycle_ == Lifecycle::CREATED || lifecycle_ == Lifecycle::STOPPED) {
            return;
        }
        if (lifecycle_ != Lifecycle::ACCEPTING) {
            lifecycle_cv_.wait(lock, [this] { return lifecycle_ == Lifecycle::STOPPED; });
            return;
        }
        lifecycle_ = Lifecycle::STOPPING;
        executor   = executor_;
    }
    // RPCs may outlive their submitting worker. Keep the completion queue open
    // until the last operation has released its pins or recorded quarantine.
    executor->stopAdmission();
    {
        std::unique_lock<std::mutex> lock(lifecycle_mutex_);
        lifecycle_cv_.wait(lock, [this] { return in_flight_ == 0; });
        lifecycle_ = Lifecycle::FINALIZING;
    }
    executor->shutdown();
    std::shared_ptr<storage_backend_detail::StorageTaskState> retired;
    {
        std::unique_lock<std::mutex> lock(quarantine_mutex_);
        // No forced teardown: an unresolved operation keeps this complete
        // backend session alive, including the actual SDK registration owner.
        quarantine_cv_.wait(lock, [this] {
            return (!protected_quarantine_ && !hasUnresolvedExternalIo()) || quiesceUnknownOperations();
        });
        retired = std::move(protected_quarantine_);
    }
    while (retired) {
        auto next = std::move(retired->quarantine_next);
        retired->finish();
        retired = std::move(next);
    }
    shutdownImpl();
    uint64_t quarantine_generation;
    {
        std::lock_guard<std::mutex> lock(quarantine_mutex_);
        quarantined_task_count_  = 0;
        quarantined_block_count_ = 0;
        quarantined_breakdown_   = {};
        quarantine_generation    = ++quarantine_generation_;
    }
    onQuarantineChanged(quarantine_generation, /*task_count=*/0, /*block_count=*/0, {});
    // FROZEN references remain until the block pools are destroyed.
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        lifecycle_ = Lifecycle::STOPPED;
    }
    lifecycle_cv_.notify_all();
}

void StorageBackend::quarantineTask(
    const std::shared_ptr<storage_backend_detail::StorageTaskState>& state,
    bool                                                             is_write,
    StorageQuarantineReason                                          reason,
    StorageLocalTargetProtection                                     local_target_protection) {
    if (!state) {
        return;
    }
    const size_t pinned_block_count = state->pins.size();
    if (state->completion_protected) {
        // Anchor before any log, observer, allocation or cleanup. This link
        // only copies/moves already-existing shared ownership and is acyclic.
        std::lock_guard<std::mutex> lock(quarantine_mutex_);
        if (state->quarantined) {
            return;
        }
        state->quarantine_next = std::move(protected_quarantine_);
        protected_quarantine_ = state;
        state->quarantined = true;
    } else {
        if (local_target_protection == StorageLocalTargetProtection::LEGACY) {
            for (const auto& pin : state->pins) {
                pin.pool->freezeBlock(pin.block);
            }
        }
        state->finish();
    }
    size_t task_count;
    size_t block_count;
    uint64_t generation;
    StorageQuarantineBreakdown breakdown;
    {
        std::lock_guard<std::mutex> lock(quarantine_mutex_);
        quarantined_block_count_ += pinned_block_count;
        task_count  = ++quarantined_task_count_;
        block_count = quarantined_block_count_;
        generation  = ++quarantine_generation_;
        auto& counts = quarantined_breakdown_[is_write ? 1 : 0][static_cast<size_t>(reason)];
        ++counts.task_count;
        counts.block_count += pinned_block_count;
        breakdown = quarantined_breakdown_;
    }
    RTP_LLM_LOG_WARNING("quarantine storage request after transfer completion became unknown: tasks=%zu blocks=%zu",
                        task_count,
                        block_count);
    // Count each operation even if its state snapshot is superseded, and before
    // onQuarantineChanged may abort due to the configured capacity limit.
    onQuarantined(is_write, reason, pinned_block_count);
    onQuarantineChanged(generation, task_count, block_count, breakdown);
}

std::shared_ptr<storage_backend_detail::StorageTaskState>
StorageBackend::prepare(StorageRequest request, bool for_write, uint64_t trace_id) {
    if (trace_id == 0)
        trace_id = storage_backend_detail::nextTraceId();
    auto timer =
        stageValue(for_write ? "write_prepare" : "read_prepare", request, for_write ? "write" : "read", trace_id);
    validateRequest(request, /*allow_null_blocks=*/false, for_write);
    auto state     = std::make_shared<storage_backend_detail::StorageTaskState>();
    state->request = std::move(request);
    state->trace_id = trace_id;
    RTP_LLM_CHECK(initialized_);

    // Decode source blocks use the same native FROZEN custody as ordinary
    // storage tasks. A backend must explicitly opt into additional ownership.
    state->completion_protected = protectUnknownCompletion();
    if (state->completion_protected) {
        state->completion_resources = completionResourceOwner();
        RTP_LLM_CHECK_WITH_INFO(state->completion_resources != nullptr,
                                "protected storage I/O requires a backend unresolved-I/O token");
    }
    if (for_write || state->completion_protected) {
        // All metadata allocation and deduplication precedes pin acquisition.
        // In particular, recording an acquired pin cannot allocate or lose it.
        std::unordered_set<BlockKey, BlockKeyHash> seen;
        std::vector<storage_backend_detail::StorageTaskState::Pin> sources;
        for (size_t key = 0; key < state->request.handles.size(); ++key) {
            const auto& row = state->request.handles[key];
            for (const auto& handle : row) {
                const auto& pool = devicePool(handle.tag);
                auto add = [&](BlockIdxType block) {
                    if (seen.insert({pool.get(), block}).second) {
                        sources.push_back({pool, block});
                    }
                };
                add(handle.block);
                if (state->request.decode_peer_blocks) {
                    const auto found = state->request.decode_peer_blocks->find(handle.tag);
                    if (found != state->request.decode_peer_blocks->end())
                        for (const auto block : found->second.at(key)) add(block);
                }
            }
        }
        state->publication_pins.emplace();
        state->publication_pins->reserve(sources.size());
        state->pins.reserve(sources.size());
        for (auto& pin : sources) {
            state->publication_pins->push_back(pin.pool->pinForRemotePublication(pin.block));
            state->pins.push_back(std::move(pin));
        }
        timer.finish(true);
        return state;
    }

    std::unordered_set<BlockKey, BlockKeyHash> pinned;
    for (const auto& key_handles : state->request.handles) {
        for (const StorageBlockHandle& handle : key_handles) {
            const auto&    pool = devicePool(handle.tag);
            const BlockKey key{pool.get(), handle.block};
            if (pinned.insert(key).second) {
                pool->incRef(handle.block, DeviceBlockRefType::REMOTE_CONNECTOR);
                state->pins.push_back({pool, handle.block});
            }
        }
    }
    timer.finish(true);
    return state;
}

void StorageBackend::validateRequest(const StorageRequest& request, bool allow_null_blocks, bool for_write) const {
    RTP_LLM_CHECK_WITH_INFO(request.keys != nullptr, "storage request requires cache keys");
    RTP_LLM_CHECK_WITH_INFO(request.handles.size() == request.keys->size(),
                            "storage request key/handle count mismatch: keys=%zu handles=%zu",
                            request.keys->size(),
                            request.handles.size());
    RTP_LLM_CHECK_WITH_INFO(!request.decode_completion || for_write, "Decode publication is write-only");
    if (request.decode_peer_blocks) {
        RTP_LLM_CHECK_WITH_INFO(for_write && request.decode_completion,
                                "peer source blocks require a completed Decode write");
        for (const auto& [tag, rows] : *request.decode_peer_blocks) {
            RTP_LLM_CHECK_WITH_INFO(rows.size() == request.handles.size(), "Decode peer key count mismatch");
            for (size_t key = 0; key < rows.size(); ++key) {
                if (rows[key].empty()) continue;
                const auto& handles = request.handles[key];
                const auto found = std::find_if(handles.begin(), handles.end(),
                                                [&](const auto& handle) { return handle.tag == tag; });
                RTP_LLM_CHECK_WITH_INFO(found != handles.end(), "Decode peers have no source handle");
                for (const auto peer : rows[key])
                    RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(peer), "null Decode peer source block");
            }
        }
    }
    for (size_t key_index = 0; key_index < request.handles.size(); ++key_index) {
        std::unordered_set<std::string> seen_tags;
        for (const auto& handle : request.handles[key_index]) {
            RTP_LLM_CHECK_WITH_INFO(!handle.tag.empty(), "storage handle has empty tag at key=%zu", key_index);
            (void)topology().group(handle.tag);
            RTP_LLM_CHECK_WITH_INFO(seen_tags.emplace(handle.tag).second,
                                    "storage request has duplicate tag=%s at key=%zu",
                                    handle.tag.c_str(),
                                    key_index);
            RTP_LLM_CHECK_WITH_INFO(allow_null_blocks || !isNullBlockIdx(handle.block),
                                    "storage request has null block for tag=%s at key=%zu",
                                    handle.tag.c_str(),
                                    key_index);
        }
    }
}

const CacheTopology& StorageBackend::topology() const {
    RTP_LLM_CHECK(topology_ != nullptr);
    return *topology_;
}

const DeviceBlockPoolPtr& StorageBackend::devicePool(const std::string& tag) const {
    return pools_by_tag_.at(tag);
}

std::vector<BlockInfo> StorageBackend::convertIndexToBuffer(int layer_id, const std::string& tag, int block_id) const {
    RTP_LLM_CHECK(static_cast<bool>(buffer_resolver_));
    (void)topology().group(tag);
    return buffer_resolver_(layer_id, tag, block_id);
}

bool StorageBackend::isHandleRequired(size_t key_index, size_t matched_key_count, std::string_view tag) const {
    RTP_LLM_CHECK(key_index < matched_key_count);
    const size_t reuse_count = topology().group(tag).reuseBlockCount(matched_key_count);
    return matched_key_count - key_index <= reuse_count;
}

void StorageBackend::match(StorageRequest request, MatchDone done) {
    RTP_LLM_CHECK(initialized_);
    validateRequest(request, /*allow_null_blocks=*/true);
    if (!request.transfer_task) {
        request.transfer_task.emplace(std::vector<TransferDescriptor>{}, timeout_, request.request_id);
    }
    const auto trace_id = storage_backend_detail::nextTraceId();
    auto       total    = stageTimer("match_total", request, "match", trace_id, AsyncProfileKind::Lifecycle);
    auto       queue    = stageTimer("match_queue_wait", request, "match", trace_id, AsyncProfileKind::Wait);
    const bool skip_remote = !remoteCacheEnabled();
    dispatch([this, skip_remote, request = std::move(request), done = std::move(done), trace_id, total, queue](
                 Lifecycle outcome, Task finished) mutable {
        block_tree_cache_detail::ScopeRollback retire(std::move(finished));
        queue.finish(outcome == Lifecycle::ACCEPTING);
        StorageMatchResult result;
        bool               success = outcome == Lifecycle::ACCEPTING && !request.transfer_task->expired();
        if (success) {
            auto io = stageValue("match_io", request, "match", trace_id);
            try {
                const bool admitted = !skip_remote && remoteCacheEnabled();
                if (admitted) {
                    result = matchImpl(request);
                }
                // A query may have been in flight when the control changed.
                // Return a successful local-only match, never unfilled remote hits.
                if (!admitted || !remoteCacheEnabled()) {
                    onRemoteCacheSkipped("match");
                    result = {request.local_matched_blocks_num, nullptr};
                }
                success = !request.transfer_task->expired();
            } catch (...) {
                success = false;
            }
            io.finish(success);
        }
        if (done) {
            auto callback = stageValue("match_callback", request, "match", trace_id);
            done(success ? result.matched_blocks_num : 0, success ? std::move(result.match_meta) : nullptr, success);
            callback.finish(true);
        }
        total.finish(success);
    }, BlockTreeTaskClass::LOAD, skip_remote);
}

void StorageBackend::read(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta, Done done) {
    if (!request.transfer_task) {
        request.transfer_task.emplace(std::vector<TransferDescriptor>{}, timeout_, request.request_id);
    }
    const auto trace_id = storage_backend_detail::nextTraceId();
    auto       total    = stageTimer("read_total", request, "read", trace_id, AsyncProfileKind::Lifecycle);
    auto       state    = prepare(std::move(request), false, trace_id);
    auto       queue    = stageTimer("read_queue_wait", state->request, "read", trace_id, AsyncProfileKind::Wait);
    const auto deadline    = state->request.transfer_task->deadline();
    const bool skip_remote = !remoteCacheEnabled();
    dispatch(
        [this,
         skip_remote,
         state      = std::move(state),
         match_meta = std::move(match_meta),
         done       = std::move(done),
         total,
         queue](Lifecycle outcome, Task finished) mutable {
            queue.finish(outcome == Lifecycle::ACCEPTING && !state->request.transfer_task->expired());
            ErrorInfo error = outcome == Lifecycle::ACCEPTING ?
                                  ErrorInfo::OkStatus() :
                                  ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "storage backend is not accepting reads");
            if (error.ok() && state->request.transfer_task->expired()) {
                RTP_LLM_LOG_WARNING("storage read deadline exceeded before I/O: trace=%llu target_key_count=%zu",
                                    static_cast<unsigned long long>(state->trace_id), state->request.handles.size());
                error = ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "storage read deadline exceeded before I/O");
            }
            if (error.ok() && (skip_remote || !remoteCacheEnabled())) {
                onRemoteCacheSkipped("read");
                error =
                    ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "remote cache disabled by KVCM before read submission");
            }
            const auto io_kind = usesAsyncPayload(state->request, /*is_write=*/false) ? AsyncProfileKind::Lifecycle :
                                                                                        AsyncProfileKind::Execution;
            auto io = error.ok() ? stageTimer("read_io", state->request, "read", state->trace_id, io_kind) :
                                   SharedTransferStageTimer{};
            auto settle_read = [this, state, error, done = std::move(done), total, io, io_kind,
                                finished = std::move(finished)](Task result, bool, bool before_io = false,
                                                               std::exception_ptr failure = {}) mutable {
                block_tree_cache_detail::ScopeRollback retire(std::move(finished));
                storage_backend_detail::invokeCallback(this, [&] {
                    std::optional<StorageQuarantineReason> quarantine_reason;
                    auto local_target_protection = StorageLocalTargetProtection::LEGACY;
                    if (error.ok()) {
                        try {
                            std::optional<TransferStageTimer> complete;
                            if (io_kind == AsyncProfileKind::Lifecycle)
                                complete.emplace(nullptr, "read_complete", state->request.request_id.value_or(-1), 0,
                                                 state->request.handles.size(), world_rank_, "storage_backend",
                                                 state->trace_id, CacheStageContext::storage("read"));
                            if (failure) std::rethrow_exception(failure);
                            result();
                            if (state->request.transfer_task->expired()) {
                                error = ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "storage read deadline exceeded");
                            }
                        } catch (const StorageOperationTimeout& exception) {
                            if (!before_io) {
                                quarantine_reason       = exception.reason();
                                local_target_protection = exception.localTargetProtection();
                                quarantineTask(state, /*is_write=*/false, *quarantine_reason, local_target_protection);
                            }
                            error = ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, exception.what());
                        } catch (const StorageOperationCompletionUnknown& exception) {
                            if (!before_io) {
                                quarantine_reason       = exception.reason();
                                local_target_protection = exception.localTargetProtection();
                                quarantineTask(state, /*is_write=*/false, *quarantine_reason, local_target_protection);
                            }
                            error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what());
                        } catch (const StorageOperationBufferSafeFailure& exception) {
                            error = ErrorInfo(state->request.transfer_task->expired() ? ErrorCode::DEADLINE_EXCEEDED :
                                                                                        ErrorCode::EXECUTION_EXCEPTION,
                                              exception.what());
                        } catch (const std::exception& exception) {
                            if (!before_io && (state->completion_protected || state->request.decode_completion)) {
                                quarantine_reason = StorageQuarantineReason::INVALID_RESPONSE;
                                quarantineTask(state, /*is_write=*/false, *quarantine_reason);
                            }
                            error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what());
                        } catch (...) {
                            if (!before_io && (state->completion_protected || state->request.decode_completion)) {
                                quarantine_reason = StorageQuarantineReason::INVALID_RESPONSE;
                                quarantineTask(state, /*is_write=*/false, *quarantine_reason);
                            }
                            error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown storage read failure");
                        }
                        io.finish(error.ok());
                    }
                    auto settle = stageValue("read_settle", state->request, "read", state->trace_id);
                    // A recorded reason was already quarantined before fallible reporting.
                    if (!quarantine_reason) state->finish();
                    settle.finish(true);
                    const bool success = error.ok();
                    if (done) {
                        auto callback = stageValue("read_callback", state->request, "read", state->trace_id);
                        done(std::move(error));
                        callback.finish(true);
                    }
                    total.finish(success);
                });
            };
            std::function<void(Task, bool, std::exception_ptr)> complete_read;
            try {
                complete_read = storage_backend_detail::makeIoCompletion<Task>(
                    executor_, state, settle_read);
            } catch (...) {
                // Completion preparation failed before readAsyncImpl/readImpl.
                // No external I/O can own the destination, even on bad_alloc.
                settle_read({}, false, true, std::current_exception());
                return;
            }
            if (!error.ok()) {
                complete_read({}, false, {});
                return;
            }
            try {
                // Execution scopes feed the trace while I/O timers report stage metrics.
                std::optional<TransferStageTimer> submit;
                if (io_kind == AsyncProfileKind::Lifecycle)
                    submit.emplace(nullptr, "read_submit", state->request.request_id.value_or(-1), 0,
                                   state->request.handles.size(), world_rank_, "storage_backend",
                                   state->trace_id, CacheStageContext::storage("read"));
                if (readAsyncImpl(state->request, match_meta, [complete_read](Task result) mutable {
                        complete_read(std::move(result), true, {});
                    }))
                    return;
                complete_read([this, state, match_meta] { readImpl(state->request, match_meta); }, false, {});
            } catch (...) {
                complete_read({}, false, std::current_exception());
            }
        },
        BlockTreeTaskClass::LOAD,
        skip_remote,
        deadline);
}

StorageWriteTask StorageBackend::prepareWrite(StorageRequest request) {
    RTP_LLM_CHECK(initialized_);
    if (!remoteCacheEnabled()) {
        onRemoteCacheSkipped("write");
        return {};
    }
    if (request.empty()) {
        return {};
    }
    if (!request.transfer_task) {
        request.transfer_task.emplace(std::vector<TransferDescriptor>{}, timeout_, request.request_id);
    }
    return StorageWriteTask(prepare(std::move(request), /*for_write=*/true));
}

bool StorageBackend::write(StorageWriteTask task) {
    RTP_LLM_CHECK(initialized_);
    RTP_LLM_CHECK(task.state_ != nullptr);
    auto state = std::move(task.state_);
    auto total = stageTimer("write_total", state->request, "write", state->trace_id, AsyncProfileKind::Lifecycle);
    auto queue = stageTimer("write_queue_wait", state->request, "write", state->trace_id, AsyncProfileKind::Wait);
    const bool skip_remote = !remoteCacheEnabled();
    const bool admitted = dispatch([this, skip_remote, state, total, queue](Lifecycle outcome, Task finished) {
        const bool accepting = outcome == Lifecycle::ACCEPTING;
        queue.finish(accepting && !state->request.transfer_task->expired());
        const bool disabled = accepting && (skip_remote || !remoteCacheEnabled());
        if (disabled)
            onRemoteCacheSkipped("write");
        const bool run = accepting && !disabled && !state->request.transfer_task->expired();
        const auto io_kind = usesAsyncPayload(state->request, /*is_write=*/true) ? AsyncProfileKind::Lifecycle :
                                                                                   AsyncProfileKind::Execution;
        auto io = run ? stageTimer("write_io", state->request, "write", state->trace_id, io_kind) :
                        SharedTransferStageTimer{};
        auto settle_write = [this, state, total, io, io_kind, run, finished = std::move(finished)](
                                WriteResult result, bool on_worker, bool before_io = false,
                                std::exception_ptr failure = {}) mutable {
            block_tree_cache_detail::ScopeRollback retire(std::move(finished));
            storage_backend_detail::invokeCallback(this, [&] {
                bool success = false;
                std::optional<StorageQuarantineReason> quarantine_reason;
                auto local_target_protection = StorageLocalTargetProtection::LEGACY;
                if (run) {
                    try {
                        std::optional<TransferStageTimer> complete;
                        if (io_kind == AsyncProfileKind::Lifecycle)
                            complete.emplace(nullptr, "write_complete", state->request.request_id.value_or(-1), 0,
                                             state->request.handles.size(), world_rank_, "storage_backend",
                                             state->trace_id, CacheStageContext::storage("write"));
                        if (failure) std::rethrow_exception(failure);
                        if (result)
                            result(on_worker);
                        success = !state->request.transfer_task->expired();
                    } catch (const StorageOperationTimeout& error) {
                        if (!before_io) {
                            quarantine_reason       = error.reason();
                            local_target_protection = error.localTargetProtection();
                            quarantineTask(state, /*is_write=*/true, *quarantine_reason, local_target_protection);
                        }
                        RTP_LLM_LOG_WARNING("StorageBackend write timed out: %s", error.what());
                    } catch (const StorageOperationCompletionUnknown& error) {
                        if (!before_io) {
                            quarantine_reason       = error.reason();
                            local_target_protection = error.localTargetProtection();
                            quarantineTask(state, /*is_write=*/true, *quarantine_reason, local_target_protection);
                        }
                        RTP_LLM_LOG_WARNING("StorageBackend write completion is unknown: %s", error.what());
                    } catch (const StorageOperationBufferSafeFailure& error) {
                        RTP_LLM_LOG_WARNING("StorageBackend payload-safe write failure: %s", error.what());
                    } catch (const std::exception& error) {
                        if (!before_io && (state->completion_protected || state->request.decode_completion)) {
                            quarantine_reason = StorageQuarantineReason::INVALID_RESPONSE;
                            quarantineTask(state, /*is_write=*/true, *quarantine_reason);
                        }
                        RTP_LLM_LOG_ERROR("StorageBackend write failed: %s", error.what());
                    } catch (...) {
                        if (!before_io && (state->completion_protected || state->request.decode_completion)) {
                            quarantine_reason = StorageQuarantineReason::INVALID_RESPONSE;
                            quarantineTask(state, /*is_write=*/true, *quarantine_reason);
                        }
                        RTP_LLM_LOG_ERROR("StorageBackend write failed with an unknown exception");
                    }
                    io.finish(success);
                }
                auto settle = stageValue("write_settle", state->request, "write", state->trace_id);
                // A recorded reason was already quarantined before fallible reporting.
                if (!quarantine_reason) state->finish();
                settle.finish(true);
                total.finish(success);
            });
        };
        if (!run) {
            settle_write({}, true);
            return;
        }
        std::function<void(WriteResult, bool, std::exception_ptr)> complete;
        try {
            complete = storage_backend_detail::makeIoCompletion<WriteResult>(
                executor_, state, settle_write);
        } catch (...) {
            // No startWriteImpl/writeImpl call has occurred. Preserve the
            // original error without manufacturing an UNKNOWN buffer owner.
            settle_write({}, false, true, std::current_exception());
            return;
        }
        try {
            std::optional<TransferStageTimer> submit;
            if (io_kind == AsyncProfileKind::Lifecycle)
                submit.emplace(nullptr, "write_submit", state->request.request_id.value_or(-1), 0,
                               state->request.handles.size(), world_rank_, "storage_backend",
                               state->trace_id, CacheStageContext::storage("write"));
            auto retain_pins = [this, state](const std::vector<size_t>& key_indices) {
                if (key_indices.size() == state->request.handles.size())
                    return;
                // Decode peers are a request sidecar; keep every retained key's peer pages too.
                const auto* peers = state->request.decode_peer_blocks.get();
                std::unordered_set<BlockKey, BlockKeyHash> required;
                for (const auto key_index : key_indices) {
                    for (const auto& handle : state->request.handles.at(key_index)) {
                        const auto& pool = devicePool(handle.tag);
                        required.insert({pool.get(), handle.block});
                        if (!peers)
                            continue;
                        const auto found = peers->find(handle.tag);
                        if (found != peers->end())
                            for (const auto block : found->second.at(key_index))
                                required.insert({pool.get(), block});
                    }
                }
                // pins and publication_pins are parallel and unique per (pool, block).
                auto& pins        = state->pins;
                auto* publication = state->publication_pins ? &*state->publication_pins : nullptr;
                RTP_LLM_CHECK(!publication || publication->size() == pins.size());
                size_t kept = 0;
                for (size_t i = 0; i < pins.size(); ++i) {
                    if (required.count({pins[i].pool.get(), pins[i].block})) {
                        if (kept != i) {
                            pins[kept] = std::move(pins[i]);
                            if (publication)
                                (*publication)[kept] = std::move((*publication)[i]);
                        }
                        ++kept;
                    } else if (publication) {
                        (*publication)[i] = {};
                    } else {
                        pins[i].pool->decRef(pins[i].block, DeviceBlockRefType::REMOTE_CONNECTOR);
                    }
                }
                pins.erase(pins.begin() + kept, pins.end());
                if (publication)
                    publication->erase(publication->begin() + kept, publication->end());
            };
            if (startWriteImpl(state->request,
                               [complete](WriteResult result) { complete(std::move(result), true, {}); },
                               std::move(retain_pins)))
                return;
        } catch (...) {
            complete({}, false, std::current_exception());
            return;
        }
        complete({}, false, {});
    }, BlockTreeTaskClass::BACKGROUND, skip_remote, state->request.transfer_task->deadline());
    return admitted && !skip_remote;
}

}  // namespace rtp_llm
