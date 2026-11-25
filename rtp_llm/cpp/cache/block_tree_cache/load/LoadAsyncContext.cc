#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadOperation.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/BlockTreeCacheMetricsReporter.h"

#include <algorithm>
#include <exception>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace {

void invokeDoneCallback(const LoadAsyncContext::DoneCallback& callback, ErrorInfo error) {
    try {
        callback(std::move(error));
    } catch (const std::exception& exception) {
        RTP_LLM_LOG_WARNING("load context completion callback failed: %s", exception.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("load context completion callback failed");
    }
}

}  // namespace

LoadAsyncContext::LoadAsyncContext(std::vector<TransferDescriptor>                load_descs,
                                   std::vector<bool>                              joined_load,
                                   std::vector<std::shared_ptr<LoadOperation>>    load_operations,
                                   size_t                                         local_matched_blocks,
                                   size_t                                         matched_device_blocks,
                                   size_t                                         matched_device_host_blocks,
                                   std::function<bool(size_t)>                    valid_prefix,
                                   uint64_t                                       context_id,
                                   const std::shared_ptr<LoadContextCoordinator>& coordinator,
                                   std::shared_ptr<StorageBackend>                storage_backend,
                                   StorageRequest                                 storage_request,
                                   const BlockTreeCacheMetricsReporter*           metrics_reporter):
    coordinator_(coordinator),
    metrics_reporter_(metrics_reporter),
    context_id_(context_id),
    load_descs_(std::move(load_descs)),
    joined_load_(std::move(joined_load)),
    load_operations_(std::move(load_operations)),
    local_matched_blocks_(local_matched_blocks),
    matched_blocks_(local_matched_blocks),
    matched_device_host_blocks_(matched_device_host_blocks),
    device_ready_blocks_(matched_device_blocks),
    storage_backend_(std::move(storage_backend)),
    request_id_(storage_request.request_id),
    storage_request_(std::move(storage_request)),
    backend_task_({}, storage_backend_ ? storage_backend_->timeout_ : std::chrono::milliseconds(0), request_id_),
    valid_prefix_(std::move(valid_prefix)),
    need_backend_match_(storage_backend_ && !storage_request_.empty()),
    backend_pending_(need_backend_match_),
    backend_deadline_state_(need_backend_match_ ? State::PENDING : State::SUCCEEDED),
    remaining_transfer_count_(std::count_if(
        load_descs_.begin(),
        load_descs_.end(),
        [](const auto& desc) { return desc.source_tier == Tier::HOST || desc.source_tier == Tier::DISK; })),
    remaining_join_count_(std::count(joined_load_.begin(), joined_load_.end(), true)) {
    RTP_LLM_CHECK(device_ready_blocks_ <= matched_device_host_blocks_
                  && matched_device_host_blocks_ <= local_matched_blocks_);
    load_operations_.resize(load_descs_.size());
    new_load_indices_.reserve(remaining_transfer_count_ - remaining_join_count_);
    workflow_span_ = AsyncProfileSpan("workflow_total",
                                      request_id_.value_or(-1),
                                      0,
                                      load_descs_.size(),
                                      0,
                                      -1,
                                      workflowId(),
                                      "load",
                                      AsyncProfileKind::Lifecycle);
    if (need_backend_match_) {
        storage_request_.transfer_task = backend_task_.subtask({});
    }
    for (size_t index = 0; index < load_descs_.size(); ++index) {
        const auto& desc = load_descs_[index];
        if (desc.source_tier != Tier::DEVICE && !joined_load_[index]) {
            new_load_indices_.emplace_back(index);
        }
        if (desc.source_tier < Tier::DEVICE || desc.source_tier > Tier::DISK) {
            continue;
        }
        auto [it, inserted] = reuse_tier_by_path_.emplace(desc.path_index, desc.source_tier);
        if (!inserted && desc.source_tier > it->second) {
            it->second = desc.source_tier;
        }
    }
    for (const auto& [_, tier] : reuse_tier_by_path_) {
        ++matched_blocks_by_tier_[static_cast<size_t>(tier)];
    }
    if (remaining_join_count_ != 0) {
        if (metrics_reporter_) {
            metrics_reporter_->reportLoadJoin(remaining_join_count_);
        }
        join_span_ = AsyncProfileSpan(
            "join_wait", request_id_.value_or(-1), 0, 0, 0, -1, workflowId(), "load", AsyncProfileKind::Wait);
    }
}

LoadAsyncContext::~LoadAsyncContext() {
    if (!abortPending() && state_.load() == State::PENDING) {
        unsubscribeFromJoinedLoads();
    }
}

bool LoadAsyncContext::empty() const {
    return load_descs_.empty() && !need_backend_match_;
}

uint64_t LoadAsyncContext::contextId() const {
    return context_id_;
}

size_t LoadAsyncContext::localMatchedBlocks() const {
    return local_matched_blocks_;
}

size_t LoadAsyncContext::matchedBlocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return matched_blocks_;
}

size_t LoadAsyncContext::matchedBlocks(Tier tier) const {
    return tier >= Tier::DEVICE && tier <= Tier::DISK ? matched_blocks_by_tier_[static_cast<size_t>(tier)] : 0;
}

size_t LoadAsyncContext::matchedBlocks(Tier tier, size_t prefix) const {
    if (prefix == local_matched_blocks_) {
        return matchedBlocks(tier);
    }
    return std::count_if(reuse_tier_by_path_.begin(), reuse_tier_by_path_.end(), [=](const auto& item) {
        return item.first < prefix && item.second == tier;
    });
}

size_t LoadAsyncContext::deviceReadyBlocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return device_ready_blocks_;
}

bool LoadAsyncContext::hostTransfersSucceeded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return settlement_ready_ && !has_host_failure_;
}

bool LoadAsyncContext::needBackendMatch() const {
    return need_backend_match_;
}

void LoadAsyncContext::setMatchCallback(MatchCallback callback) {
    RTP_LLM_CHECK(need_backend_match_ && callback && !match_callback_ && !backend_started_);
    match_callback_ = std::move(callback);
}

void LoadAsyncContext::setSettlementReadyCallback(SettlementReadyCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK(callback && state_.load() == State::PENDING && !settlement_ready_ && !settlement_ready_callback_);
    settlement_ready_callback_ = std::move(callback);
}

void LoadAsyncContext::startBackendMatch() {
    RTP_LLM_CHECK(need_backend_match_ && match_callback_ && !backend_started_);
    if (done()) {
        return;
    }
    backend_started_                     = true;
    std::weak_ptr<LoadAsyncContext> weak = weak_from_this();
    storage_backend_->match(
        storage_request_,
        [weak](size_t matched_blocks_num, std::shared_ptr<StorageBackendMatchMeta> match_meta, bool success) {
            if (auto context = weak.lock()) {
                context->onBackendMatch(matched_blocks_num, std::move(match_meta), success);
            }
        });
}

void LoadAsyncContext::setTargetBlocks(size_t desc_index, std::vector<BlockIdxType> target_blocks) {
    load_descs_[desc_index].target_blocks = std::move(target_blocks);
}

bool LoadAsyncContext::createLoadOperations(BlockTree& tree) {
    std::vector<MultiNodeResource> targets;
    bool                           success = true;
    for (size_t desc_index : new_load_indices_) {
        const auto& desc = load_descs_[desc_index];
        load_operations_[desc_index] =
            std::make_shared<LoadOperation>(desc, tree.groupSets()[desc.group_set_id], workflowId());
        appendBlocks(targets, desc.group_set_id, Tier::DEVICE, desc.target_blocks);
        if (desc.isCurrentOwner()) {
            auto& resource = desc.node->group_set_resources[desc.group_set_id];
            if (resource.transfer_state != GroupSetTransferState::LOAD_PENDING) {
                RTP_LLM_LOG_ERROR("committed load source is not LOAD_PENDING, group_set_id=%zu", desc.group_set_id);
                success = false;
                break;
            }
            resource.transfer_state = GroupSetTransferState::LOADING;
            resource.load_operation = load_operations_[desc_index];
            tree.refreshPublishedState(desc.node);
        }
    }
    referenceBlocks(tree.groupSets(), targets, BlockTreeRefType::LOAD);
    return success;
}

void LoadAsyncContext::subscribeToJoinedLoads() {
    if (remaining_join_count_ == 0) {
        return;
    }
    const int64_t start_time_us = currentTimeUs();
    const auto    weak          = weak_from_this();
    for (size_t index = 0; index < load_operations_.size(); ++index) {
        if (!joined_load_[index]) {
            continue;
        }
        const auto& operation = load_operations_[index];
        AsyncProfileSession::taskLink(
            workflowId(), workflowId(), 0, request_id_.value_or(-1), "load", operation->workflowId());
        operation->subscribe(
            workflowId(), [weak, source = load_descs_[index].source_tier, start_time_us](ErrorInfo error) {
                if (auto context = weak.lock()) {
                    context->completeTransfers(1, error.ok(), source, start_time_us);
                }
            });
    }
}

void LoadAsyncContext::unsubscribeFromJoinedLoads() {
    if (remaining_join_count_.load(std::memory_order_acquire) == 0) {
        return;
    }
    const auto subscriber_id = workflowId();
    for (size_t index = 0; index < load_operations_.size(); ++index) {
        if (joined_load_[index]) {
            load_operations_[index]->unsubscribe(subscriber_id);
        }
    }
}

void LoadAsyncContext::setBackendTargetBlock(size_t key_index, size_t handle_index, BlockIdxType target_block) {
    storage_request_.handles[key_index][handle_index].block = target_block;
}

const std::vector<TransferDescriptor>& LoadAsyncContext::loadDescs() const {
    return load_descs_;
}

const std::vector<bool>& LoadAsyncContext::joinedLoads() const {
    return joined_load_;
}

const std::vector<std::vector<StorageBlockHandle>>& LoadAsyncContext::backendHandles() const {
    return storage_request_.handles;
}

void LoadAsyncContext::onBackendMatch(size_t                                   matched_blocks_num,
                                      std::shared_ptr<StorageBackendMatchMeta> match_meta,
                                      bool                                     success) {
    if (!coordinator_->beginActiveCallback()) {
        return;
    }
    block_tree_cache_detail::ScopeRollback active_callback_guard(
        [coordinator = coordinator_] { coordinator->retireActiveCallback(); });
    std::lock_guard<std::mutex> backend_match_lock(backend_match_mutex_);
    if (done()) {
        failBeforeCommit();
        return;
    }
    if (!success) {
        matched_blocks_num = local_matched_blocks_;
    }
    RTP_LLM_CHECK(storage_request_.keys && storage_request_.keys->size() == storage_request_.handles.size()
                  && storage_request_.local_matched_blocks_num == local_matched_blocks_
                  && matched_blocks_num >= local_matched_blocks_
                  && matched_blocks_num <= storage_request_.handles.size());
    while (matched_blocks_num > local_matched_blocks_ && valid_prefix_ && !valid_prefix_(matched_blocks_num)) {
        --matched_blocks_num;
    }
    backend_matched_blocks_ = matched_blocks_num;
    if (matched_blocks_num < storage_request_.handles.size()) {
        storage_request_.keys = std::make_shared<CacheKeysType>(storage_request_.keys->begin(),
                                                                storage_request_.keys->begin() + matched_blocks_num);
        storage_request_.handles.resize(matched_blocks_num);
    }
    for (size_t key_index = 0; key_index < storage_request_.handles.size(); ++key_index) {
        auto& handles = storage_request_.handles[key_index];
        if (key_index < local_matched_blocks_) {
            handles.clear();
            continue;
        }
        handles.erase(std::remove_if(handles.begin(),
                                     handles.end(),
                                     [&](const StorageBlockHandle& handle) {
                                         return !storage_backend_->isHandleRequired(
                                             key_index, matched_blocks_num, handle.tag);
                                     }),
                      handles.end());
    }
    const LoadMatchResult match_result = match_callback_(*this, matched_blocks_num);
    if (!match_result.success) {
        malloc_status_.store(match_result.malloc_status, std::memory_order_release);
        failBeforeCommit();
        return;
    }
    if (storage_request_.empty()) {
        onBackendRead(ErrorInfo::OkStatus());
        return;
    }
    if (backend_task_.expired()) {
        onBackendRead(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "match-to-ready deadline exceeded"));
        return;
    }
    std::weak_ptr<LoadAsyncContext> weak = weak_from_this();
    storage_backend_->read(std::move(storage_request_), std::move(match_meta), [weak](ErrorInfo error) {
        if (auto context = weak.lock()) {
            context->onBackendRead(std::move(error));
        }
    });
}

void LoadAsyncContext::onBackendRead(ErrorInfo error) {
    bool                    notify = false;
    SettlementReadyCallback settlement_ready_callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_.load() != State::PENDING || !backend_pending_) {
            return;
        }
        (void)done();
        State expected = State::PENDING;
        backend_deadline_state_.compare_exchange_strong(expected, State::SUCCEEDED);
        if (deadlineExceeded()) {
            error = ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "match-to-ready deadline exceeded");
        }
        if (error.ok()) {
            matched_blocks_ = backend_matched_blocks_;
        } else {
            backend_error_ = error;
            error_info_    = std::move(error);
        }
        backend_pending_ = false;
        finishIfReadyLocked(notify, settlement_ready_callback);
    }
    dispatchCompletion(notify, std::move(settlement_ready_callback));
}

void LoadAsyncContext::failBeforeCommit() {
    coordinator_->abort(*this);
    onTaskFail();
}

void LoadAsyncContext::failCommit() {
    malloc_status_.store(MallocStatus::INTERNAL_ERROR, std::memory_order_release);
    onTaskFail();
}

bool LoadAsyncContext::commit() {
    if (done() && deadlineExceeded()) {
        return false;
    }
    bool expected = false;
    if (!commit_started_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return false;
    }
    if (!coordinator_->commit(context_id_)) {
        // The coordinator can reject before resolving the weak context. Make
        // that path terminal as well; failCommit is idempotent if a rejected
        // callback already invoked it.
        failCommit();
        return false;
    }
    bool                    notify = false;
    SettlementReadyCallback settlement_ready_callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        committed_ = true;
        finishIfReadyLocked(notify, settlement_ready_callback);
    }
    dispatchCompletion(notify, std::move(settlement_ready_callback));
    return true;
}

void LoadAsyncContext::markAborted() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const State                 state = state_.load();
        if (state != State::PENDING) {
            return;
        }
        State expected = State::PENDING;
        backend_deadline_state_.compare_exchange_strong(expected, State::SUCCEEDED);
        remaining_transfer_count_ = 0;
        backend_pending_          = false;
        if (error_info_.ok()) {
            error_info_ = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "load context aborted");
        }
        completion_time_us_.store(autil::TimeUtility::currentTimeInMicroSeconds());
        state_.store(State::FAILED);
    }
    notifyCompletion();
}

bool LoadAsyncContext::abortPending() {
    std::lock_guard<std::mutex> backend_match_lock(backend_match_mutex_);
    if (!coordinator_->abort(*this)) {
        return false;
    }
    markAborted();
    return true;
}

bool LoadAsyncContext::completeTransfers(size_t                 count,
                                         bool                   success,
                                         Tier                   source_tier,
                                         std::optional<int64_t> join_start_time_us) {
    bool                    notify = false;
    bool                    last_join = false;
    SettlementReadyCallback settlement_ready_callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (join_start_time_us) {
            last_join = remaining_join_count_.fetch_sub(count, std::memory_order_acq_rel) == count;
        }
        const State                 state = state_.load();
        if (state != State::PENDING || count == 0 || count > remaining_transfer_count_) {
            return false;
        }
        has_local_failure_ = has_local_failure_ || !success;
        has_host_failure_  = has_host_failure_ || (!success && source_tier != Tier::DISK);
        if (!success && error_info_.ok()) {
            error_info_ = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "load transfer failed");
        }
        remaining_transfer_count_ -= count;
        finishIfReadyLocked(notify, settlement_ready_callback);
    }
    dispatchCompletion(notify, std::move(settlement_ready_callback));
    if (last_join) {
        if (!join_trace_finished_.exchange(true)) {
            join_span_.finish();
        }
        if (metrics_reporter_) {
            metrics_reporter_->reportLoadJoinWait(currentTimeUs() - *join_start_time_us);
        }
    }
    return true;
}

void LoadAsyncContext::cancelUnsubmittedTransfers() {
    coordinator_->abort_callback_(*this);
    std::lock_guard<std::mutex> lock(mutex_);
    has_local_failure_        = true;
    has_host_failure_         = true;
    remaining_transfer_count_ = 0;
}

bool LoadAsyncContext::aggregateSuccess() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return settlement_ready_ && !has_local_failure_ && backend_error_.ok();
}

bool LoadAsyncContext::localTransfersSucceeded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return settlement_ready_ && !has_local_failure_;
}

bool LoadAsyncContext::localLoadSucceeded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done() && local_load_succeeded_;
}

bool LoadAsyncContext::backendTimedOut() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return backend_error_.code() == ErrorCode::DEADLINE_EXCEEDED;
}

bool LoadAsyncContext::settle(bool success, bool host_success) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_.load() != State::PENDING || !settlement_ready_) {
            return false;
        }
        local_load_succeeded_ = success && !has_local_failure_;
        if (local_load_succeeded_) {
            device_ready_blocks_ = local_matched_blocks_;
        } else if (host_success && !has_host_failure_) {
            device_ready_blocks_ = matched_device_host_blocks_;
        }
        const bool completed_successfully = local_load_succeeded_ && backend_error_.ok();
        if (!completed_successfully && error_info_.ok()) {
            error_info_ = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "load settlement failed");
        }
        completion_time_us_.store(autil::TimeUtility::currentTimeInMicroSeconds());
        state_.store(completed_successfully ? State::SUCCEEDED : State::FAILED);
    }
    notifyCompletion();
    return true;
}

bool LoadAsyncContext::onTaskFail() {
    SettlementReadyCallback discarded_callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const State                 state = state_.load();
        if (state != State::PENDING) {
            return false;
        }
        State expected = State::PENDING;
        backend_deadline_state_.compare_exchange_strong(expected, State::SUCCEEDED);
        remaining_transfer_count_ = 0;
        backend_pending_          = false;
        if (error_info_.ok()) {
            error_info_ = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "load task failed");
        }
        completion_time_us_.store(autil::TimeUtility::currentTimeInMicroSeconds());
        discarded_callback = std::move(settlement_ready_callback_);
        state_.store(State::FAILED);
    }
    for (size_t index : new_load_indices_) {
        if (load_operations_[index]) {
            load_operations_[index]->complete(false);
        }
    }
    notifyCompletion();
    return true;
}

void LoadAsyncContext::finishIfReadyLocked(bool& notify, SettlementReadyCallback& settlement_ready_callback) {
    if (!committed_ || remaining_transfer_count_ != 0 || backend_pending_) {
        return;
    }
    const State state = state_.load();
    if (state != State::PENDING || settlement_ready_) {
        return;
    }
    if (settlement_ready_callback_) {
        settlement_ready_         = true;
        settlement_ready_callback = std::move(settlement_ready_callback_);
        return;
    }
    local_load_succeeded_ = !has_local_failure_;
    if (local_load_succeeded_) {
        device_ready_blocks_ = local_matched_blocks_;
    } else if (!has_host_failure_) {
        device_ready_blocks_ = matched_device_host_blocks_;
    }
    completion_time_us_.store(autil::TimeUtility::currentTimeInMicroSeconds());
    state_.store(local_load_succeeded_ && backend_error_.ok() ? State::SUCCEEDED : State::FAILED);
    notify = true;
}

void LoadAsyncContext::dispatchCompletion(bool notify, SettlementReadyCallback settlement_ready_callback) {
    if (notify) {
        notifyCompletion();
    }
    if (settlement_ready_callback) {
        settlement_ready_callback(shared_from_this());
    }
}

void LoadAsyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!done()) {
        if (backend_pending_) {
            cv_.wait_until(lock, backend_task_.deadline(), [this] { return done(); });
        } else {
            cv_.wait(lock, [this] { return done(); });
        }
    }
}

void LoadAsyncContext::onDone(DoneCallback callback) {
    if (!callback) {
        return;
    }
    bool      run_now = false;
    ErrorInfo error   = ErrorInfo::OkStatus();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done()) {
            run_now = true;
            error   = completionErrorLocked();
        } else {
            callbacks_.push_back(std::move(callback));
        }
    }
    if (run_now) {
        invokeDoneCallback(callback, std::move(error));
    }
}

ErrorInfo LoadAsyncContext::completionErrorLocked() const {
    if (deadlineExceeded()) {
        return ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "match-to-ready deadline exceeded");
    }
    if (state_.load() == State::FAILED && error_info_.ok()) {
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "load async context failed");
    }
    return error_info_;
}

void LoadAsyncContext::notifyCompletion() {
    unsubscribeFromJoinedLoads();
    block_tree_cache_detail::ScopeRollback finish_trace([this] {
        if (!workflow_trace_finished_.exchange(true))
            workflow_span_.finish();
    });
    if (!join_trace_finished_.exchange(true))
        join_span_.finish();
    std::vector<DoneCallback> callbacks;
    ErrorInfo                 error = ErrorInfo::OkStatus();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        error = completionErrorLocked();
        callbacks.swap(callbacks_);
    }
    cv_.notify_all();
    for (auto& callback : callbacks) {
        invokeDoneCallback(callback, error);
    }
}

bool LoadAsyncContext::done() const {
    const State state = state_.load();
    if (state == State::PENDING && backend_deadline_state_.load() == State::PENDING && backend_task_.expired()) {
        State expected = State::PENDING;
        backend_deadline_state_.compare_exchange_strong(expected, State::FAILED);
    }
    return deadlineExceeded() || state_.load() != State::PENDING;
}

bool LoadAsyncContext::success() const {
    return state_.load() == State::SUCCEEDED && !deadlineExceeded();
}

ErrorInfo LoadAsyncContext::errorInfo() const {
    (void)done();
    std::lock_guard<std::mutex> lock(mutex_);
    return completionErrorLocked();
}

void LoadAsyncContext::deferRelease(std::shared_ptr<BatchKVCacheResource> resource) {
    if (!coordinator_->defer_cleanup_
        || !coordinator_->defer_cleanup_([self = shared_from_this(), resource = std::move(resource)] {
               std::lock_guard<std::mutex> lock(self->backend_match_mutex_);
               if (self->coordinator_->abort(*self)) {
                   self->markAborted();
               } else if (self->deadlineExceeded()) {
                   // I/O pins remain with StorageBackend until the payload has settled.
                   self->onBackendRead(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "match-to-ready deadline exceeded"));
               }
               for (int batch = 0; batch < resource->batchSize(); ++batch) {
                   for (const auto& tag : self->storage_backend_->cacheConfig().allGroupTags()) {
                       for (const auto block : resource->blocks(batch, tag)) {
                           if (!isNullBlockIdx(block)) {
                               self->storage_backend_->devicePool(tag)->decRef(block, DeviceBlockRefType::REQUEST);
                           }
                       }
                   }
               }
               resource->clearBlocks();
           })) {
        // Admission has closed. Keep request ownership rather than reuse an I/O target.
        RTP_LLM_LOG_WARNING("load cleanup queue closed; retaining abandoned request block references");
    }
}

MallocStatus LoadAsyncContext::mallocStatus() const {
    return malloc_status_.load(std::memory_order_acquire);
}

LoadContextCoordinator::LoadContextCoordinator(CommitCallback                             commit_callback,
                                               AbortCallback                              abort_callback,
                                               std::function<bool(std::function<void()>)> defer_cleanup,
                                               const BlockTreeCacheMetricsReporter*       metrics_reporter):
    commit_callback_(std::move(commit_callback)),
    abort_callback_(std::move(abort_callback)),
    defer_cleanup_(std::move(defer_cleanup)),
    metrics_reporter_(metrics_reporter) {}

std::shared_ptr<LoadAsyncContext>
LoadContextCoordinator::create(std::vector<TransferDescriptor>             load_descs,
                               std::vector<bool>                           joined_load,
                               std::vector<std::shared_ptr<LoadOperation>> load_operations,
                               size_t                                      local_matched_blocks,
                               size_t                                      matched_device_blocks,
                               size_t                                      matched_device_host_blocks,
                               std::function<bool(size_t)>                 valid_prefix,
                               std::shared_ptr<StorageBackend>             storage_backend,
                               StorageRequest                              storage_request) {
    uint64_t context_id = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!accepting_) {
            return nullptr;
        }
        context_id = next_context_id_++;
    }
    auto context = std::make_shared<LoadAsyncContext>(std::move(load_descs),
                                                      std::move(joined_load),
                                                      std::move(load_operations),
                                                      local_matched_blocks,
                                                      matched_device_blocks,
                                                      matched_device_host_blocks,
                                                      std::move(valid_prefix),
                                                      context_id,
                                                      shared_from_this(),
                                                      std::move(storage_backend),
                                                      std::move(storage_request),
                                                      metrics_reporter_);
    context->subscribeToJoinedLoads();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!accepting_) {
            return nullptr;
        }
        pending_contexts_.emplace(context_id, context);
    }
    return context;
}

bool LoadContextCoordinator::beginActiveCallback() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!accepting_) {
        return false;
    }
    ++active_callbacks_;
    return true;
}

bool LoadContextCoordinator::commit(uint64_t context_id) {
    std::shared_ptr<LoadAsyncContext> context;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  pending = pending_contexts_.find(context_id);
        if (!accepting_ || pending == pending_contexts_.end() || !(context = pending->second.lock())) {
            return false;
        }
        pending_contexts_.erase(pending);
        ++active_callbacks_;
    }
    block_tree_cache_detail::ScopeRollback callback_guard([this] { retireActiveCallback(); });
    if (commit_callback_(context)) {
        return true;
    }
    context->failCommit();
    return false;
}

bool LoadContextCoordinator::abort(LoadAsyncContext& context) noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  pending = pending_contexts_.find(context.contextId());
        if (pending == pending_contexts_.end()) {
            return false;
        }
        pending_contexts_.erase(pending);
        ++active_callbacks_;
    }
    block_tree_cache_detail::ScopeRollback callback_guard([this] { retireActiveCallback(); });
    abort_callback_(context);
    return true;
}

void LoadContextCoordinator::retireActiveCallback() {
    std::lock_guard<std::mutex> lock(mutex_);
    --active_callbacks_;
    if (active_callbacks_ == 0 || pending_contexts_.empty()) {
        cv_.notify_all();
    }
}

void LoadContextCoordinator::shutdown() {
    std::vector<std::shared_ptr<LoadAsyncContext>> contexts;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        accepting_ = false;
        for (const auto& [_, weak] : pending_contexts_) {
            if (auto context = weak.lock()) {
                contexts.push_back(std::move(context));
            }
        }
    }
    for (const auto& context : contexts) {
        context->abortPending();
    }

    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return pending_contexts_.empty() && active_callbacks_ == 0; });
    commit_callback_ = {};
    abort_callback_  = {};
}

}  // namespace rtp_llm
