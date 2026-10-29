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

namespace rtp_llm::storage_backend_detail {

thread_local const StorageBackend* completing_backend = nullptr;

uint64_t nextTraceId() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

template<typename Callback>
void invokeCallback(const StorageBackend* backend, Callback&& callback) noexcept {
    struct CompletionScope {
        const StorageBackend* previous = completing_backend;
        ~CompletionScope() {
            completing_backend = previous;
        }
    } scope;
    completing_backend = backend;
    try {
        callback();
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("StorageBackend completion failed: %s", error.what());
    } catch (...) { RTP_LLM_LOG_ERROR("StorageBackend completion failed with an unknown exception"); }
}

struct StorageTaskState {
    struct Pin {
        DeviceBlockPoolPtr pool;
        BlockIdxType       block;
        std::function<void()> publication_release;
    };

    StorageRequest   request;
    uint64_t                          trace_id{0};
    std::vector<Pin> pins;
    std::shared_ptr<void> completion_resources;
    std::shared_ptr<StorageTaskState> quarantine_next;
    bool completion_protected{false};
    bool quarantined{false};
    std::once_flag   finish_once;
    void             finish() {
        std::call_once(finish_once, [this] {
            for (const Pin& pin : pins) {
                if (pin.publication_release) {
                    pin.publication_release();
                } else {
                    pin.pool->decRef(pin.block, DeviceBlockRefType::REMOTE_CONNECTOR);
                }
            }
            pins.clear();
            request.completion_budget_lease.reset();
            request.completion_admission.reset();
            completion_resources.reset();
        });
    }

    ~StorageTaskState() {
        finish();
    }
};

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
                               int64_t                                        world_rank):
    executor_(std::move(executor)),
    stage_metrics_reporter_(std::move(reporter)),
    stage_metrics_enabled_(stage_metrics_reporter_ && stage_metrics_reporter_->stageMetricsEnabled()),
    world_rank_(world_rank) {}

void StorageBackend::setStageMetricsReporter(std::shared_ptr<BlockTreeCacheMetricsReporter> reporter) {
    const bool enabled = reporter && reporter->stageMetricsEnabled();
    std::atomic_store(&stage_metrics_reporter_, std::move(reporter));
    stage_metrics_enabled_.store(enabled, std::memory_order_release);
}

SharedTransferStageTimer StorageBackend::stageTimer(const char*           stage,
                                                    const StorageRequest& request,
                                                    const char*           operation,
                                                    uint64_t              trace_id) const noexcept {
    return TransferStageTimer::create(
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

bool StorageBackend::dispatch(Operation operation) {
    const auto once = std::make_shared<std::once_flag>();
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
    if (outcome == Lifecycle::STOPPED) {
        try {
            operation(outcome);
        } catch (...) {}
        return false;
    }
    auto complete = [this, once, operation = std::move(operation)](Lifecycle result) mutable {
        std::call_once(*once, [&] {
            storage_backend_detail::invokeCallback(this, [&] { operation(result); });
            // invokeCallback isolates all exceptions, so every admitted task
            // reaches this exactly-once accounting boundary.
            taskFinished();
        });
    };
    if (outcome != Lifecycle::ACCEPTING) {
        complete(Lifecycle::STOPPING);
        return false;
    }
    try {
        if (executor_->submit([complete]() mutable { complete(Lifecycle::ACCEPTING); })) {
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
    executor->shutdown();
    {
        std::unique_lock<std::mutex> lock(lifecycle_mutex_);
        lifecycle_cv_.wait(lock, [this] { return in_flight_ == 0; });
        lifecycle_ = Lifecycle::FINALIZING;
    }
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

void StorageBackend::quarantineTask(const std::shared_ptr<storage_backend_detail::StorageTaskState>& state,
                                    bool                                                             is_write,
                                    StorageQuarantineReason                                          reason) {
    if (!state) {
        return;
    }
    const size_t pinned_block_count = state->pins.size();
    if (state->request.completion_admission)
        state->request.completion_admission->onQuarantined();
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
        for (const auto& pin : state->pins) {
            pin.pool->freezeBlock(pin.block);
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

    state->completion_protected = state->request.decode_completion || protectUnknownCompletion();
    if (state->completion_protected) {
        state->completion_resources = completionResourceOwner();
        RTP_LLM_CHECK_WITH_INFO(state->completion_resources != nullptr,
                                "protected storage I/O requires an actual transport resource owner");
        // All metadata allocation and deduplication precedes pin acquisition.
        // In particular, recording an acquired pin cannot allocate or lose it.
        std::unordered_set<BlockKey, BlockKeyHash> seen;
        std::vector<storage_backend_detail::StorageTaskState::Pin> sources;
        for (const auto& row : state->request.handles) {
            for (const auto& handle : row) {
                const auto& pool = devicePool(handle.tag);
                auto add = [&](BlockIdxType block) {
                    if (seen.insert({pool.get(), block}).second) {
                        sources.push_back({pool, block, {}});
                    }
                };
                add(handle.block);
                for (const auto block : handle.peer_blocks) {
                    add(block);
                }
            }
        }
        state->pins.reserve(sources.size());
        for (auto& pin : sources) {
            pin.publication_release = pin.pool->pinForRemotePublication(pin.block);
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
            RTP_LLM_CHECK_WITH_INFO(handle.peer_blocks.empty() || (for_write && request.decode_completion),
                                    "peer source blocks require a completed Decode write");
            for (const auto peer : handle.peer_blocks) {
                RTP_LLM_CHECK_WITH_INFO(!isNullBlockIdx(peer), "null Decode peer source block");
            }
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
    const auto trace_id = storage_backend_detail::nextTraceId();
    auto       total    = stageTimer("match_total", request, "match", trace_id);
    auto       queue    = stageTimer("match_queue_wait", request, "match", trace_id);
    dispatch([this, request = std::move(request), done = std::move(done), trace_id, total, queue](
                 Lifecycle outcome) mutable {
        queue.finish(outcome == Lifecycle::ACCEPTING);
        StorageMatchResult result;
        bool               success = outcome == Lifecycle::ACCEPTING;
        if (success) {
            auto io = stageValue("match_io", request, "match", trace_id);
            try {
                result = matchImpl(request);
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
    });
}

void StorageBackend::read(StorageRequest request, std::shared_ptr<StorageBackendMatchMeta> match_meta, Done done) {
    const auto trace_id = storage_backend_detail::nextTraceId();
    auto       total    = stageTimer("read_total", request, "read", trace_id);
    auto       state    = prepare(std::move(request), false, trace_id);
    auto       queue    = stageTimer("read_queue_wait", state->request, "read", trace_id);
    dispatch([this, state = std::move(state), match_meta = std::move(match_meta), done = std::move(done), total, queue](
                 Lifecycle outcome) mutable {
        queue.finish(outcome == Lifecycle::ACCEPTING);
        ErrorInfo error = outcome == Lifecycle::ACCEPTING ?
                              ErrorInfo::OkStatus() :
                              ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "storage backend is not accepting reads");
        std::optional<StorageQuarantineReason> quarantine_reason;
        if (error.ok()) {
            auto io = stageValue("read_io", state->request, "read", state->trace_id);
            try {
                readImpl(state->request, match_meta);
            } catch (const StorageOperationTimeout& exception) {
                if (state->completion_protected) {
                    quarantineTask(state, /*is_write=*/false, exception.reason());
                }
                error             = ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, exception.what());
                quarantine_reason = exception.reason();
            } catch (const StorageOperationCompletionUnknown& exception) {
                if (state->completion_protected) {
                    quarantineTask(state, /*is_write=*/false, exception.reason());
                }
                error             = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what());
                quarantine_reason = exception.reason();
            } catch (const StorageOperationBufferSafeFailure& exception) {
                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what());
            } catch (const std::exception& exception) {
                if (state->completion_protected) {
                    quarantineTask(state, /*is_write=*/false, StorageQuarantineReason::INVALID_RESPONSE);
                    quarantine_reason = StorageQuarantineReason::INVALID_RESPONSE;
                }
                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, exception.what());
            } catch (...) {
                if (state->completion_protected) {
                    quarantineTask(state, /*is_write=*/false, StorageQuarantineReason::INVALID_RESPONSE);
                    quarantine_reason = StorageQuarantineReason::INVALID_RESPONSE;
                }
                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown storage read failure");
            }
            io.finish(error.ok());
        }
        auto settle = stageValue("read_settle", state->request, "read", state->trace_id);
        if (quarantine_reason) {
            quarantineTask(state, /*is_write=*/false, *quarantine_reason);
        } else {
            state->finish();
        }
        settle.finish(true);
        const bool success = error.ok();
        if (done) {
            auto callback = stageValue("read_callback", state->request, "read", state->trace_id);
            done(std::move(error));
            callback.finish(true);
        }
        total.finish(success);
    });
}

StorageWriteTask StorageBackend::prepareWrite(StorageRequest request) {
    RTP_LLM_CHECK(initialized_);
    if (request.empty()) {
        return {};
    }
    return StorageWriteTask(prepare(std::move(request), /*for_write=*/true));
}

bool StorageBackend::write(StorageWriteTask task) {
    RTP_LLM_CHECK(initialized_);
    RTP_LLM_CHECK(task.state_ != nullptr);
    auto state = std::move(task.state_);
    auto total = stageTimer("write_total", state->request, "write", state->trace_id);
    auto queue = stageTimer("write_queue_wait", state->request, "write", state->trace_id);
    if (state->request.completion_admission)
        state->request.completion_admission->onQueued();
    return dispatch([this, state, total, queue](Lifecycle outcome) {
        queue.finish(outcome == Lifecycle::ACCEPTING);
        bool                                   success = false;
        std::optional<StorageQuarantineReason> quarantine_reason;
        if (outcome == Lifecycle::ACCEPTING) {
            auto io = stageValue("write_io", state->request, "write", state->trace_id);
            if (state->request.completion_admission)
                state->request.completion_admission->onRunning();
            try {
                writeImpl(state->request);
                success = true;
            } catch (const StorageOperationTimeout& error) {
                quarantine_reason = error.reason();
                if (state->completion_protected) {
                    quarantineTask(state, /*is_write=*/true, *quarantine_reason);
                }
                RTP_LLM_LOG_WARNING("StorageBackend write timed out; quarantining request pins: %s", error.what());
            } catch (const StorageOperationCompletionUnknown& error) {
                quarantine_reason = error.reason();
                if (state->completion_protected) {
                    quarantineTask(state, /*is_write=*/true, *quarantine_reason);
                }
                RTP_LLM_LOG_WARNING("StorageBackend write completion is unknown; quarantining request pins: %s",
                                    error.what());
            } catch (const StorageOperationBufferSafeFailure& error) {
                RTP_LLM_LOG_WARNING("StorageBackend payload-safe write failure: %s", error.what());
            } catch (const std::exception& error) {
                if (state->completion_protected) {
                    quarantine_reason = StorageQuarantineReason::INVALID_RESPONSE;
                    quarantineTask(state, /*is_write=*/true, *quarantine_reason);
                }
                RTP_LLM_LOG_ERROR("StorageBackend write failed: %s", error.what());
            } catch (...) {
                if (state->completion_protected) {
                    quarantine_reason = StorageQuarantineReason::INVALID_RESPONSE;
                    quarantineTask(state, /*is_write=*/true, *quarantine_reason);
                }
                RTP_LLM_LOG_ERROR("StorageBackend write failed with an unknown exception");
            }
            io.finish(success);
        }
        auto settle = stageValue("write_settle", state->request, "write", state->trace_id);
        if (quarantine_reason) {
            quarantineTask(state, /*is_write=*/true, *quarantine_reason);
        } else {
            state->finish();
        }
        settle.finish(true);
        total.finish(success);
    });
}

}  // namespace rtp_llm
