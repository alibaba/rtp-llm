#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackend.h"

#include <algorithm>
#include <mutex>
#include <exception>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/IBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/CacheDiagnostics.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheObserver.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StoragePrefixCoverage.h"
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
        BTC_LOG_ERROR("StorageBackend completion failed: %s", error.what());
    } catch (...) {
        BTC_LOG_ERROR("StorageBackend completion failed with an unknown exception");
    }
    return false;
}

struct StorageTaskState {
    struct Pin {
        std::shared_ptr<IBlockPool> pool;
        BlockIdxType          block;
        std::function<void()> release;
    };

    StorageRequest    request;
    uint64_t          trace_id{0};
    std::vector<Pin>  pins;
    std::once_flag    finish_once;
    std::atomic<bool> completion_started{false};
    void              finish(bool freeze_pins = false) {
        std::call_once(finish_once, [this, freeze_pins] {
            if (freeze_pins) {
                for (const Pin& pin : pins)
                    pin.pool->freezeBlock(pin.block);
            }
            for (const Pin& pin : pins) {
                if (pin.release)
                    pin.release();
            }
            pins.clear();
        });
    }

    ~StorageTaskState() {
        finish();
    }
};

// Keep one copy of the settlement state across the worker and RPC callback.
template<typename Result, typename Settle>
auto makeIoCompletion(std::shared_ptr<StorageBackendExecutor> executor,
                      std::shared_ptr<StorageTaskState>       state,
                      Settle                                  settle) {
    auto continuation = std::make_shared<Settle>(std::move(settle));
    return [executor = std::move(executor), state = std::move(state), continuation](Result result, bool asynchronous) {
        if (state->completion_started.exchange(true))
            return;
        if (asynchronous) {
            try {
                if (executor->submitCompletion(
                        [continuation, result]() mutable { (*continuation)(std::move(result), true); }))
                    return;
            } catch (...) {}
        }
        (*continuation)(std::move(result), !asynchronous);
    };
}

}  // namespace rtp_llm::storage_backend_detail

namespace rtp_llm {
namespace {

struct BlockKey {
    const IBlockPool* pool;
    BlockIdxType     block;
    bool             operator==(const BlockKey& other) const {
        return pool == other.pool && block == other.block;
    }
};

struct BlockKeyHash {
    size_t operator()(const BlockKey& key) const {
        return std::hash<const IBlockPool*>{}(key.pool) ^ (std::hash<BlockIdxType>{}(key.block) << 1U);
    }
};

}  // namespace

StorageWriteTask::StorageWriteTask(std::shared_ptr<storage_backend_detail::StorageTaskState> state):
    state_(std::move(state)) {}

StorageBackend::StorageBackend(std::shared_ptr<StorageBackendExecutor>        executor,
                               std::shared_ptr<CacheObserver>                  reporter,
                               int64_t                                        world_rank,
                               std::optional<std::chrono::milliseconds>       timeout):
    executor_(std::move(executor)),
    stage_metrics_reporter_(std::move(reporter)),
    stage_metrics_enabled_(stage_metrics_reporter_ && stage_metrics_reporter_->stageMetricsEnabled()),
    world_rank_(world_rank),
    timeout_(timeout) {}

void StorageBackend::setStageMetricsReporter(std::shared_ptr<CacheObserver> reporter) {
    const bool enabled = reporter && reporter->stageMetricsEnabled();
    std::atomic_store(&stage_metrics_reporter_, std::move(reporter));
    stage_metrics_enabled_.store(enabled, std::memory_order_release);
}

void StorageBackend::applyDeadline(StorageRequest& request) const {
    if (timeout_ && !request.transfer_task) {
        request.transfer_task.emplace(std::vector<TransferDescriptor>{}, *timeout_, request.request_id);
    }
}

std::optional<StorageBackendExecutor::Clock::time_point> StorageBackend::deadline(const StorageRequest& request) const {
    return timeout_ ? std::optional{request.transfer_task->deadline()} : std::nullopt;
}

bool StorageBackend::expired(const StorageRequest& request) const {
    return timeout_.has_value() && request.transfer_task->expired();
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

StorageBackend::~StorageBackend() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    BTC_CHECK_WITH_INFO(lifecycle_ == Lifecycle::CREATED || lifecycle_ == Lifecycle::STOPPED,
                            "StorageBackend must be shutdown before derived destruction");
}

const StorageBackend::PageAccess& StorageBackend::GroupAccess::forTier(Tier tier) const {
    if (tier == Tier::DEVICE) {
        return device;
    }
    if (tier == Tier::HOST && host.pool) {
        return host;
    }
    throw std::invalid_argument("storage operation has no pool for its tier");
}

bool StorageBackend::init(Groups groups) {
    if (init_attempted_) {
        BTC_LOG_ERROR("StorageBackend initialization has already been attempted");
        return false;
    }
    for (const auto& [tag, group] : groups) {
        if (tag.empty() || !group.device.pool || !group.device.pin || !group.required
            || bool(group.host.pool) != bool(group.host.pin)) {
            throw std::invalid_argument("incomplete storage group binding for tag=" + tag);
        }
    }
    init_attempted_ = true;
    if (executor_ == nullptr) {
        executor_ = makeDefaultStorageBackendExecutor();
    }
    if (executor_->bound_to_backend_.exchange(true)) {
        BTC_LOG_ERROR("StorageBackend executor cannot be shared between backends");
        return false;
    }
    groups_ = std::move(groups);
    const auto fail_init = [this] {
        shutdownImpl();
        groups_.clear();
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

bool StorageBackend::dispatch(Operation                                                operation,
                              BlockTreeTaskClass                                       task_class,
                              bool                                                     run_inline,
                              std::optional<StorageBackendExecutor::Clock::time_point> io_deadline) {
    const auto once        = std::make_shared<std::once_flag>();
    const auto finish_once = std::make_shared<std::once_flag>();
    Lifecycle  outcome;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        outcome = lifecycle_;
        if (outcome == Lifecycle::FINALIZING) {
            outcome = Lifecycle::STOPPED;
        } else if (outcome != Lifecycle::STOPPED) {
            BTC_CHECK(outcome != Lifecycle::CREATED);
            ++in_flight_;
        }
    }
    auto complete =
        [this, once, finish_once, counted = outcome != Lifecycle::STOPPED, operation = std::move(operation)](
            Lifecycle result) mutable {
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
    BTC_CHECK(in_flight_ > 0);
    if (--in_flight_ == 0) {
        lifecycle_cv_.notify_all();
    }
}

void StorageBackend::shutdown() {
    BTC_CHECK_WITH_INFO(storage_backend_detail::completing_backend != this,
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
    executor->stopAdmission();
    {
        std::unique_lock<std::mutex> lock(lifecycle_mutex_);
        lifecycle_cv_.wait(lock, [this] { return in_flight_ == 0; });
        lifecycle_ = Lifecycle::FINALIZING;
    }
    executor->shutdown();
    shutdownImpl();
    // FROZEN references remain until the block pools are destroyed.
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        lifecycle_ = Lifecycle::STOPPED;
    }
    lifecycle_cv_.notify_all();
}

std::shared_ptr<storage_backend_detail::StorageTaskState>
StorageBackend::prepare(StorageRequest request, bool for_write, uint64_t trace_id) {
    if (trace_id == 0)
        trace_id = storage_backend_detail::nextTraceId();
    auto timer =
        stageValue(for_write ? "write_prepare" : "read_prepare", request, for_write ? "write" : "read", trace_id);
    validateRequest(request, /*allow_null_blocks=*/false);
    auto state      = std::make_shared<storage_backend_detail::StorageTaskState>();
    state->request  = std::move(request);
    state->trace_id = trace_id;
    BTC_CHECK(initialized_);

    std::unordered_set<BlockKey, BlockKeyHash> pinned;
    std::vector<std::pair<const PageAccess*, BlockIdxType>> targets;
    for (const auto& key_handles : state->request.handles) {
        for (const StorageBlockHandle& handle : key_handles) {
            const auto& group = groups_.at(handle.tag).forTier(handle.tier);
            const BlockKey key{group.pool.get(), handle.block};
            if (pinned.insert(key).second) {
                targets.emplace_back(&group, handle.block);
            }
        }
    }
    // Finish all metadata allocation before acquiring any pin. A partial pin
    // acquisition then releases only the closures that were successfully prepared.
    state->pins.reserve(targets.size());
    for (const auto& [group, block] : targets) {
        state->pins.push_back({group->pool, block, group->pin(block)});
    }
    timer.finish(true);
    return state;
}

void StorageBackend::validateRequest(const StorageRequest& request, bool allow_null_blocks) const {
    BTC_CHECK_WITH_INFO(static_cast<bool>(request.keys) != static_cast<bool>(request.opaque_keys),
                            "storage request requires one cache key sequence");
    BTC_CHECK_WITH_INFO(request.handles.size() == request.cacheKeyCount(),
                            "storage request key/handle count mismatch: keys=%zu handles=%zu",
                            request.cacheKeyCount(),
                            request.handles.size());
    for (size_t key_index = 0; key_index < request.handles.size(); ++key_index) {
        std::unordered_set<std::string> seen_tags;
        for (const auto& handle : request.handles[key_index]) {
            BTC_CHECK_WITH_INFO(!handle.tag.empty(), "storage handle has empty tag at key=%zu", key_index);
            (void)groups_.at(handle.tag);
            BTC_CHECK_WITH_INFO(seen_tags.emplace(handle.tag).second,
                                    "storage request has duplicate tag=%s at key=%zu",
                                    handle.tag.c_str(),
                                    key_index);
            BTC_CHECK_WITH_INFO(allow_null_blocks || !isNullBlockIdx(handle.block),
                                    "storage request has null block for tag=%s at key=%zu",
                                    handle.tag.c_str(),
                                    key_index);
        }
    }
}

bool StorageBackend::isHandleRequired(size_t key_index, size_t matched_key_count, const std::string& tag) const {
    BTC_CHECK(key_index < matched_key_count);
    return groups_.at(tag).required(key_index, matched_key_count);
}

size_t StorageBackend::coveredPrefix(
    const StorageRequest& request,
    const std::function<bool(size_t, const StorageBlockHandle&)>& available) const {
    StoragePrefixCoverage coverage;
    auto matched = request.local_matched_blocks_num;
    for (size_t row = matched; row < request.handles.size(); ++row) {
        for (const auto& handle : request.handles[row]) {
            coverage.observe(row, handle.tag, available(row, handle));
        }
        if (coverage.covers(row + 1, [&](size_t index, size_t end, const std::string& tag) {
                return isHandleRequired(index, end, tag);
            })) {
            matched = row + 1;
        }
    }
    return matched;
}

void StorageBackend::match(StorageRequest request, MatchDone done) {
    BTC_CHECK(initialized_);
    validateRequest(request, /*allow_null_blocks=*/true);
    applyDeadline(request);
    const auto trace_id    = storage_backend_detail::nextTraceId();
    auto       total       = stageTimer("match_total", request, "match", trace_id, AsyncProfileKind::Lifecycle);
    auto       queue       = stageTimer("match_queue_wait", request, "match", trace_id, AsyncProfileKind::Wait);
    const bool skip_remote = !remoteCacheEnabled();
    auto       state       = std::make_shared<storage_backend_detail::StorageTaskState>();
    state->request        = std::move(request);
    state->trace_id       = trace_id;
    dispatch(
        [this, skip_remote, state = std::move(state), done = std::move(done), total, queue](
            Lifecycle outcome, Task finished) mutable {
            queue.finish(outcome == Lifecycle::ACCEPTING);
            const auto& request = state->request;
            const bool can_run = outcome == Lifecycle::ACCEPTING && !expired(request);
            const bool admitted = can_run && !skip_remote && remoteCacheEnabled();
            const auto io_kind = usesAsyncMatch(request) ? AsyncProfileKind::Lifecycle : AsyncProfileKind::Execution;
            auto io = can_run ? stageTimer("match_io", request, "match", state->trace_id, io_kind) :
                                SharedTransferStageTimer{};
            auto settle = [this, state, can_run, admitted, done = std::move(done), io, total,
                           finished = std::move(finished)](MatchResult result, bool) mutable {
                block_tree_cache_detail::ScopeRollback retire(std::move(finished));
                storage_backend_detail::invokeCallback(this, [&] {
                    const auto& request = state->request;
                    StorageMatchResult match;
                    bool success = can_run;
                    if (success) {
                        try {
                            if (admitted) {
                                match = result();
                            }
                            // An in-flight query cannot install remote data after admission closes.
                            if (!admitted || !remoteCacheEnabled()) {
                                onRemoteCacheSkipped("match");
                                match = {request.local_matched_blocks_num, nullptr};
                            }
                            success = !expired(request);
                        } catch (...) {
                            success = false;
                        }
                        io.finish(success);
                    }
                    if (done) {
                        auto callback = stageValue("match_callback", request, "match", state->trace_id);
                        done(success ? match.matched_blocks_num : 0, success ? std::move(match.match_meta) : nullptr, success);
                        callback.finish(true);
                    }
                    total.finish(success);
                });
            };
            auto complete = storage_backend_detail::makeIoCompletion<MatchResult>(executor_, state, std::move(settle));
            if (!admitted) {
                complete({}, false);
                return;
            }
            try {
                if (matchAsyncImpl(request, [complete](MatchResult result) { complete(std::move(result), true); })) {
                    return;
                }
                complete([this, state] { return matchImpl(state->request); }, false);
            } catch (...) {
                complete([error = std::current_exception()]() -> StorageMatchResult { std::rethrow_exception(error); },
                         false);
            }
        },
        BlockTreeTaskClass::LOAD,
        skip_remote);
}

void StorageBackend::read(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta, Done done) {
    TransferDone report;
    if (done) {
        report = [done = std::move(done)](StorageTransferOutcome outcome) {
            done(std::move(outcome.error));
        };
    }
    readOwned(std::move(request), std::move(match_meta), std::move(report));
}

void StorageBackend::readOwned(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta,
                               TransferDone done) {
    applyDeadline(request);
    const auto trace_id    = storage_backend_detail::nextTraceId();
    auto       total       = stageTimer("read_total", request, "read", trace_id, AsyncProfileKind::Lifecycle);
    auto       state       = prepare(std::move(request), false, trace_id);
    auto       queue       = stageTimer("read_queue_wait", state->request, "read", trace_id, AsyncProfileKind::Wait);
    const auto io_deadline = deadline(state->request);
    const bool skip_remote = !remoteCacheEnabled();
    dispatch(
        [this,
         skip_remote,
         state      = std::move(state),
         match_meta = std::move(match_meta),
         done       = std::move(done),
         total,
         queue](Lifecycle outcome, Task finished) mutable {
            queue.finish(outcome == Lifecycle::ACCEPTING && !expired(state->request));
            ErrorInfo error = outcome == Lifecycle::ACCEPTING ?
                                  ErrorInfo::OkStatus() :
                                  ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "storage backend is not accepting reads");
            if (error.ok() && expired(state->request)) {
                BTC_LOG_WARNING("storage read deadline exceeded before I/O: trace=%llu target_key_count=%zu",
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
            auto       io      = error.ok() ? stageTimer("read_io", state->request, "read", state->trace_id, io_kind) :
                                              SharedTransferStageTimer{};
            auto       settle_read =
                [this, state, error, done = std::move(done), total, io, io_kind, finished = std::move(finished)](
                    Task result, bool) mutable {
                    block_tree_cache_detail::ScopeRollback retire(std::move(finished));
                    storage_backend_detail::invokeCallback(this, [&] {
                        bool freeze_pins = false;
                        bool safe = true;
                        std::shared_ptr<const FrozenTask> recovery;
                        if (error.ok()) {
                            try {
                                std::optional<TransferStageTimer> complete;
                                if (io_kind == AsyncProfileKind::Lifecycle)
                                    complete.emplace(nullptr,
                                                     "read_complete",
                                                     state->request.request_id.value_or(-1),
                                                     0,
                                                     state->request.handles.size(),
                                                     world_rank_,
                                                     "storage_backend",
                                                     state->trace_id,
                                                     CacheStageContext::storage("read"));
                                result();
                                if (expired(state->request)) {
                                    error = ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "storage read deadline exceeded");
                                }
                            } catch (const StorageDeadlineExceeded& exception) {
                                error = ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, exception.what());
                            } catch (const StorageOperationTimeout& exception) {
                                freeze_pins = exception.localTargetProtection() == StorageLocalTargetProtection::LEGACY;
                                safe = false;
                                recovery = exception.recovery();
                                error       = ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, exception.what());
                            } catch (const StorageOperationCompletionUnknown& exception) {
                                freeze_pins = exception.localTargetProtection() == StorageLocalTargetProtection::LEGACY;
                                safe = false;
                                recovery = exception.recovery();
                                error       = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what());
                            } catch (const StorageOperationBufferSafeFailure& exception) {
                                error =
                                    ErrorInfo(expired(state->request) ? ErrorCode::DEADLINE_EXCEEDED :
                                                                                        ErrorCode::EXECUTION_EXCEPTION,
                                              exception.what());
                            } catch (const std::exception& exception) {
                                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what());
                            } catch (...) {
                                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown storage read failure");
                            }
                            io.finish(error.ok());
                        }
                        auto settle = stageValue("read_settle", state->request, "read", state->trace_id);
                        state->finish(freeze_pins);
                        settle.finish(true);
                        const bool success = error.ok();
                        if (done) {
                            auto callback = stageValue("read_callback", state->request, "read", state->trace_id);
                            done({std::move(error), safe, std::move(recovery)});
                            callback.finish(true);
                        }
                        total.finish(success);
                    });
                };
            auto complete_read =
                storage_backend_detail::makeIoCompletion<Task>(executor_, state, std::move(settle_read));
            if (!error.ok()) {
                complete_read({}, false);
                return;
            }
            try {
                // Execution scopes feed the trace while I/O timers report stage metrics.
                std::optional<TransferStageTimer> submit;
                if (io_kind == AsyncProfileKind::Lifecycle)
                    submit.emplace(nullptr,
                                   "read_submit",
                                   state->request.request_id.value_or(-1),
                                   0,
                                   state->request.handles.size(),
                                   world_rank_,
                                   "storage_backend",
                                   state->trace_id,
                                   CacheStageContext::storage("read"));
                if (readAsyncImpl(state->request, match_meta, [complete_read](Task result) mutable {
                        complete_read(std::move(result), true);
                    }))
                    return;
                complete_read([this, state, match_meta] { readImpl(state->request, match_meta); }, false);
            } catch (...) {
                complete_read([error = std::current_exception()] { std::rethrow_exception(error); }, false);
            }
        },
        BlockTreeTaskClass::LOAD,
        skip_remote,
        io_deadline);
}

StorageWriteTask StorageBackend::prepareWrite(StorageRequest request) {
    BTC_CHECK(initialized_);
    if (!remoteCacheEnabled()) {
        onRemoteCacheSkipped("write");
        return {};
    }
    if (request.empty()) {
        return {};
    }
    applyDeadline(request);
    return StorageWriteTask(prepare(std::move(request), true));
}

bool StorageBackend::write(StorageWriteTask task, TransferDone done) {
    BTC_CHECK(initialized_);
    BTC_CHECK(task.state_ != nullptr);
    auto       state = std::move(task.state_);
    auto       total = stageTimer("write_total", state->request, "write", state->trace_id, AsyncProfileKind::Lifecycle);
    auto       queue = stageTimer("write_queue_wait", state->request, "write", state->trace_id, AsyncProfileKind::Wait);
    const bool skip_remote = !remoteCacheEnabled();
    const bool admitted    = dispatch(
        [this, skip_remote, state, done = std::move(done), total, queue](Lifecycle outcome, Task finished) mutable {
            const bool accepting = outcome == Lifecycle::ACCEPTING;
            queue.finish(accepting && !expired(state->request));
            const bool disabled = accepting && (skip_remote || !remoteCacheEnabled());
            if (disabled)
                onRemoteCacheSkipped("write");
            const bool run     = accepting && !disabled && !expired(state->request);
            const auto io_kind = usesAsyncPayload(state->request, /*is_write=*/true) ? AsyncProfileKind::Lifecycle :
                                                                                          AsyncProfileKind::Execution;
            auto       io      = run ? stageTimer("write_io", state->request, "write", state->trace_id, io_kind) :
                                          SharedTransferStageTimer{};
            auto       settle_write = [this, state, done = std::move(done), total, io, io_kind, run,
                                       finished = std::move(finished)](
                                    WriteResult result, bool on_worker) mutable {
                block_tree_cache_detail::ScopeRollback retire(std::move(finished));
                storage_backend_detail::invokeCallback(this, [&] {
                    bool success     = false;
                    bool freeze_pins = false;
                    bool safe        = true;
                    bool timeout     = false;
                    std::shared_ptr<const FrozenTask> recovery;
                    if (run) {
                        try {
                            std::optional<TransferStageTimer> complete;
                            if (io_kind == AsyncProfileKind::Lifecycle)
                                complete.emplace(nullptr,
                                                 "write_complete",
                                                 state->request.request_id.value_or(-1),
                                                 0,
                                                 state->request.handles.size(),
                                                 world_rank_,
                                                 "storage_backend",
                                                 state->trace_id,
                                                 CacheStageContext::storage("write"));
                            if (result)
                                result(on_worker);
                            success = !expired(state->request);
                        } catch (const StorageOperationTimeout& error) {
                            freeze_pins = error.localTargetProtection() == StorageLocalTargetProtection::LEGACY;
                            safe = false;
                            timeout = true;
                            recovery = error.recovery();
                            BTC_LOG_WARNING("StorageBackend write timed out: %s", error.what());
                        } catch (const StorageOperationCompletionUnknown& error) {
                            freeze_pins = error.localTargetProtection() == StorageLocalTargetProtection::LEGACY;
                            safe = false;
                            recovery = error.recovery();
                            BTC_LOG_WARNING("StorageBackend write completion is unknown: %s", error.what());
                        } catch (const StorageOperationBufferSafeFailure& error) {
                            BTC_LOG_WARNING("StorageBackend payload-safe write failure: %s", error.what());
                        } catch (const std::exception& error) {
                            BTC_LOG_ERROR("StorageBackend write failed: %s", error.what());
                        } catch (...) {
                            BTC_LOG_ERROR("StorageBackend write failed with an unknown exception");
                        }
                        io.finish(success);
                    }
                    auto settle = stageValue("write_settle", state->request, "write", state->trace_id);
                    state->finish(freeze_pins);
                    settle.finish(true);
                    if (done) {
                        auto failure = timeout || expired(state->request)
                            ? ErrorCode::DEADLINE_EXCEEDED : ErrorCode::EXECUTION_EXCEPTION;
                        auto report = success && remoteCacheEnabled()
                            ? ErrorInfo::OkStatus() : ErrorInfo(failure, "storage write failed");
                        done({std::move(report), safe, std::move(recovery)});
                    }
                    total.finish(success);
                });
            };
            if (!run) {
                settle_write({}, true);
                return;
            }
            auto complete =
                storage_backend_detail::makeIoCompletion<WriteResult>(executor_, state, std::move(settle_write));
            try {
                std::optional<TransferStageTimer> submit;
                if (io_kind == AsyncProfileKind::Lifecycle)
                    submit.emplace(nullptr,
                                   "write_submit",
                                   state->request.request_id.value_or(-1),
                                   0,
                                   state->request.handles.size(),
                                   world_rank_,
                                   "storage_backend",
                                   state->trace_id,
                                   CacheStageContext::storage("write"));
                auto retain_pins = [this, state](const std::vector<size_t>& key_indices) {
                    if (key_indices.size() == state->request.handles.size())
                        return;
                    std::unordered_set<BlockKey, BlockKeyHash> required;
                    for (const auto key_index : key_indices) {
                        for (const auto& handle : state->request.handles.at(key_index)) {
                            const auto& group = groups_.at(handle.tag).forTier(handle.tier);
                            required.insert({group.pool.get(), handle.block});
                        }
                    }
                    auto& pins = state->pins;
                    pins.erase(std::remove_if(pins.begin(),
                                              pins.end(),
                                              [&](const auto& pin) {
                                                  if (required.count({pin.pool.get(), pin.block}))
                                                      return false;
                                                  pin.release();
                                                  return true;
                                              }),
                               pins.end());
                };
                if (startWriteImpl(
                        state->request,
                        [complete](WriteResult result) { complete(std::move(result), true); },
                        std::move(retain_pins)))
                    return;
            } catch (...) {
                complete([error = std::current_exception()](bool) { std::rethrow_exception(error); }, false);
                return;
            }
            complete({}, false);
        },
        BlockTreeTaskClass::BACKGROUND,
        skip_remote,
        deadline(state->request));
    return admitted && !skip_remote;
}

}  // namespace rtp_llm
