#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

LoadAsyncContext::LoadAsyncContext(std::vector<TransferDescriptor>                load_descs,
                                   std::vector<bool>                              joined_load,
                                   size_t                                         local_matched_blocks,
                                   uint64_t                                       context_id,
                                   const std::shared_ptr<LoadContextCoordinator>& coordinator,
                                   std::shared_ptr<StorageBackend>                storage_backend,
                                   StorageRequest                                 storage_request,
                                   size_t                                         reuse_unit_tokens):
    coordinator_(coordinator),
    context_id_(context_id),
    load_descs_(std::move(load_descs)),
    joined_load_(std::move(joined_load)),
    local_matched_blocks_(local_matched_blocks),
    matched_blocks_(local_matched_blocks),
    reuse_unit_tokens_(reuse_unit_tokens),
    storage_backend_(std::move(storage_backend)),
    storage_request_(std::move(storage_request)),
    remote_base_blocks_(storage_request_.local_matched_blocks_num),
    remote_base_token_endpoint_(remote_base_blocks_ * reuse_unit_tokens_),
    need_backend_match_(storage_backend_ && !storage_request_.empty()),
    backend_pending_(need_backend_match_),
    remaining_transfer_count_(std::count_if(load_descs_.begin(), load_descs_.end(), [](const auto& desc) {
        return desc.source_tier == Tier::HOST || desc.source_tier == Tier::DISK;
    })) {
    remaining_join_count_.store(static_cast<size_t>(std::count(joined_load_.begin(), joined_load_.end(), true)),
                                std::memory_order_relaxed);
    rebuildMatchedBlocksByTier();
}

LoadAsyncContext::~LoadAsyncContext() {
    abortPending();
}

void LoadAsyncContext::rebuildMatchedBlocksByTier() {
    matched_blocks_by_tier_.fill(0);
    std::unordered_map<size_t, Tier> reuse_tier_by_path;
    for (const TransferDescriptor& desc : load_descs_) {
        if (desc.source_tier < Tier::DEVICE || desc.source_tier > Tier::DISK) {
            continue;
        }
        auto [it, inserted] = reuse_tier_by_path.emplace(desc.path_index, desc.source_tier);
        if (!inserted && desc.source_tier > it->second) {
            it->second = desc.source_tier;
        }
    }
    for (const auto& [_, tier] : reuse_tier_by_path) {
        ++matched_blocks_by_tier_[static_cast<size_t>(tier)];
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

void LoadAsyncContext::setLocalTokenEndpoints(size_t ready_endpoint, size_t target_endpoint) {
    RTP_LLM_CHECK(!commit_started_ && !backend_started_ && ready_endpoint <= target_endpoint);
    ready_token_endpoint_ = ready_endpoint;
    local_token_endpoint_ = target_endpoint;
    rebuildMatchedTokensByTier();
}

size_t LoadAsyncContext::targetTokenEndpoint() const {
    if (localFallbackSelected()) {
        return local_fallback_endpoint_;
    }
    return remote_token_endpoint_ == 0 ? local_token_endpoint_ : remote_token_endpoint_;
}

size_t LoadAsyncContext::matchedTokenEndpoint() const {
    return success() ? targetTokenEndpoint() : ready_token_endpoint_;
}

size_t LoadAsyncContext::matchedTokens(Tier tier) const {
    if (tier == Tier::REMOTE) {
        return success() && remote_token_endpoint_ != 0 ? remote_token_endpoint_ - remote_base_token_endpoint_ : 0;
    }
    return success() && tier >= Tier::DEVICE && tier <= Tier::DISK ?
               matched_tokens_by_tier_[static_cast<size_t>(tier)] :
               0;
}

void LoadAsyncContext::rebuildMatchedTokensByTier() {
    const size_t endpoint = local_token_endpoint_;
    matched_tokens_by_tier_.fill(0);
    struct Boundary {
        size_t token;
        Tier   tier;
        int    delta;
    };
    std::vector<Boundary> boundaries;
    for (const auto& desc : load_descs_) {
        if (desc.token_end <= desc.token_begin || desc.source_tier < Tier::DEVICE || desc.source_tier > Tier::DISK) {
            continue;
        }
        boundaries.push_back({std::min(desc.token_begin, endpoint), desc.source_tier, 1});
        boundaries.push_back({std::min(desc.token_end, endpoint), desc.source_tier, -1});
    }
    std::sort(boundaries.begin(), boundaries.end(), [](const auto& left, const auto& right) {
        return left.token < right.token;
    });
    std::array<int, 3> active{};
    size_t             previous = 0;
    for (const auto& boundary : boundaries) {
        const Tier deepest = active[static_cast<size_t>(Tier::DISK)] > 0 ? Tier::DISK :
                             active[static_cast<size_t>(Tier::HOST)] > 0 ? Tier::HOST :
                                                                           Tier::DEVICE;
        matched_tokens_by_tier_[static_cast<size_t>(deepest)] += boundary.token - previous;
        active[static_cast<size_t>(boundary.tier)] += boundary.delta;
        previous = boundary.token;
    }
    matched_tokens_by_tier_[static_cast<size_t>(Tier::DEVICE)] += endpoint - previous;
}

void LoadAsyncContext::setLocalFallback(size_t token_endpoint) {
    RTP_LLM_CHECK(need_backend_match_ && !backend_started_ && token_endpoint > 0);
    local_fallback_endpoint_ = token_endpoint;
}

bool LoadAsyncContext::needBackendMatch() const {
    return need_backend_match_;
}

PrefixCacheRemoteObservation LoadAsyncContext::remoteObservation() const {
    std::lock_guard<std::mutex> lock(backend_match_mutex_);
    PrefixCacheRemoteObservation observation;
    observation.query_status = remote_query_status_;
    if (remote_query_status_ == PrefixCacheRemoteQueryStatus::HIT) {
        observation.candidate_pos = remote_candidate_endpoint_;
    }
    return observation;
}

void LoadAsyncContext::setRemoteSelectionCallback(std::function<bool(LoadAsyncContext&, size_t)> callback,
                                                  bool                                           discard_local) {
    RTP_LLM_CHECK(need_backend_match_ && !backend_started_ && callback);
    remote_selection_callback_ = std::move(callback);
    discard_local_on_remote_   = discard_local;
}

size_t LoadAsyncContext::backendMatchedBlocks() const {
    return !localFallbackSelected() && backend_matched_blocks_ > remote_base_blocks_ ?
               backend_matched_blocks_ - remote_base_blocks_ :
               0;
}

void LoadAsyncContext::setRemoteSources(std::vector<TransferDescriptor> sources, std::vector<bool> joined) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (discard_local_on_remote_) {
        load_descs_.clear();
        joined_load_.clear();
        local_matched_blocks_     = 0;
        matched_blocks_           = 0;
        local_token_endpoint_     = 0;
        ready_token_endpoint_     = 0;
        remaining_transfer_count_ = 0;
        remaining_join_count_.store(0, std::memory_order_release);
        join_generation_.fetch_add(1, std::memory_order_acq_rel);
        has_failure_ = false;
        error_info_  = ErrorInfo::OkStatus();
    }
    remote_source_count_ = sources.size();
    for (size_t i = 0; i < sources.size(); ++i) {
        remaining_transfer_count_ += sources[i].source_tier != Tier::DEVICE;
        if (joined[i]) {
            remaining_join_count_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    load_descs_.insert(load_descs_.end(), sources.begin(), sources.end());
    joined_load_.insert(joined_load_.end(), joined.begin(), joined.end());
    rebuildMatchedBlocksByTier();
    rebuildMatchedTokensByTier();
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
    backend_started_                     = true;
    std::weak_ptr<LoadAsyncContext> weak = weak_from_this();
    storage_backend_->match(storage_request_,
                             [weak](size_t matched_blocks_num,
                                    std::shared_ptr<StorageBackendMatchMeta> match_meta,
                                    bool success,
                                    PrefixCacheRemoteQueryStatus query_status) {
                                 if (auto context = weak.lock()) {
                                     context->onBackendMatch(
                                         matched_blocks_num, std::move(match_meta), success, query_status);
                                 }
                             });
}

void LoadAsyncContext::setTargetBlocks(size_t desc_index, std::vector<BlockIdxType> target_blocks) {
    load_descs_[desc_index].target_blocks = std::move(target_blocks);
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
                                      bool                                     success,
                                      PrefixCacheRemoteQueryStatus             query_status) {
    if (!coordinator_->beginActiveCallback()) {
        return;
    }
    block_tree_cache_detail::ScopeRollback active_callback_guard(
        [coordinator = coordinator_] { coordinator->retireActiveCallback(); });
    std::lock_guard<std::mutex> backend_match_lock(backend_match_mutex_);
    if (state_.load() != State::PENDING) {
        return;
    }
    remote_query_status_ = success ? query_status : PrefixCacheRemoteQueryStatus::ERROR;
    if (!success) {
        malloc_status_.store(MallocStatus::INTERNAL_ERROR, std::memory_order_release);
        failBeforeCommit();
        return;
    }
    RTP_LLM_CHECK(storage_request_.keys && storage_request_.keys->size() == storage_request_.handles.size()
                  && matched_blocks_num >= storage_request_.local_matched_blocks_num
                  && matched_blocks_num <= storage_request_.handles.size());
    remote_candidate_endpoint_ = matched_blocks_num * reuse_unit_tokens_;
    size_t candidate_endpoint  = remote_candidate_endpoint_;
    for (;;) {
        if (local_fallback_endpoint_ > candidate_endpoint) {
            local_fallback_selected_.store(true, std::memory_order_release);
            matched_blocks_num = local_matched_blocks_;
            for (auto& handles : storage_request_.handles) {
                handles.clear();
            }
            break;
        }
        if (matched_blocks_num <= remote_base_blocks_ || !remote_selection_callback_
            || remote_selection_callback_(*this, candidate_endpoint)) {
            break;
        }
        --matched_blocks_num;
        candidate_endpoint -= reuse_unit_tokens_;
    }
    backend_matched_blocks_ = matched_blocks_num;
    if (!localFallbackSelected() && matched_blocks_num > remote_base_blocks_) {
        remote_token_endpoint_ = candidate_endpoint;
    }
    if (matched_blocks_num < storage_request_.handles.size()) {
        storage_request_.keys = std::make_shared<CacheKeysType>(storage_request_.keys->begin(),
                                                                storage_request_.keys->begin() + matched_blocks_num);
        storage_request_.handles.resize(matched_blocks_num);
    }
    for (size_t key_index = 0; key_index < storage_request_.handles.size(); ++key_index) {
        auto& handles = storage_request_.handles[key_index];
        if (localFallbackSelected() || key_index < storage_request_.local_matched_blocks_num) {
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
        if (error.ok()) {
            matched_blocks_ = backend_matched_blocks_;
        } else {
            has_failure_ = true;
            error_info_  = std::move(error);
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
        remaining_transfer_count_ = 0;
        backend_pending_          = false;
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

void LoadAsyncContext::startJoinWait(int64_t start_time_us) {
    assert(remaining_join_count_.load(std::memory_order_relaxed) != 0);
    if (join_start_time_us_ == 0) {
        join_start_time_us_ = start_time_us;
    }
}

bool LoadAsyncContext::completeJoinedOne(bool     success,
                                         bool&    join_completed,
                                         int64_t& join_wait_latency_us,
                                         uint64_t generation) {
    join_completed                 = false;
    join_wait_latency_us = 0;
    bool                    notify = false;
    SettlementReadyCallback settlement_ready_callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (generation != join_generation_.load(std::memory_order_acquire)) {
            return true;
        }
        if (state_.load() != State::PENDING || remaining_transfer_count_ == 0
            || remaining_join_count_.load(std::memory_order_relaxed) == 0) {
            return false;
        }
        has_failure_ = has_failure_ || !success;
        --remaining_transfer_count_;
        if (remaining_join_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            assert(join_start_time_us_ != 0);
            join_completed       = true;
            join_wait_latency_us = currentTimeUs() - join_start_time_us_;
        }
        finishIfReadyLocked(notify, settlement_ready_callback);
    }
    dispatchCompletion(notify, std::move(settlement_ready_callback));
    return true;
}

bool LoadAsyncContext::completeTransfers(size_t count, bool success) {
    bool                    notify = false;
    SettlementReadyCallback settlement_ready_callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const State                 state = state_.load();
        if (state != State::PENDING || count == 0 || count > remaining_transfer_count_) {
            return false;
        }
        has_failure_ = has_failure_ || !success;
        remaining_transfer_count_ -= count;
        finishIfReadyLocked(notify, settlement_ready_callback);
    }
    dispatchCompletion(notify, std::move(settlement_ready_callback));
    return true;
}

void LoadAsyncContext::cancelUnsubmittedTransfers() {
    // Called only before commit, after removing reservations and join registrations.
    // Already delivered join completions need not be counted a second time.
    std::lock_guard<std::mutex> lock(mutex_);
    assert(!committed_);
    has_failure_              = true;
    remaining_transfer_count_ = 0;
}

bool LoadAsyncContext::aggregateSuccess() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return settlement_ready_ && !has_failure_;
}

bool LoadAsyncContext::settle(bool success) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_.load() != State::PENDING || !settlement_ready_) {
            return false;
        }
        state_.store(success && !has_failure_ ? State::SUCCEEDED : State::FAILED);
    }
    notifyCompletion();
    return true;
}

bool LoadAsyncContext::onTaskFail() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const State                 state = state_.load();
        if (state != State::PENDING) {
            return false;
        }
        remaining_transfer_count_ = 0;
        backend_pending_          = false;
        state_.store(State::FAILED);
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
    state_.store(has_failure_ ? State::FAILED : State::SUCCEEDED);
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
    cv_.wait(lock, [this] { return done(); });
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
        callback(std::move(error));
    }
}

ErrorInfo LoadAsyncContext::completionErrorLocked() const {
    if (state_.load() == State::FAILED && error_info_.ok()) {
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "load async context failed");
    }
    return error_info_;
}

void LoadAsyncContext::notifyCompletion() {
    std::vector<DoneCallback> callbacks;
    ErrorInfo                 error = ErrorInfo::OkStatus();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        error = completionErrorLocked();
        callbacks.swap(callbacks_);
    }
    cv_.notify_all();
    for (auto& callback : callbacks) {
        callback(error);
    }
}

bool LoadAsyncContext::done() const {
    const State state = state_.load();
    return state == State::SUCCEEDED || state == State::FAILED;
}

bool LoadAsyncContext::success() const {
    return state_.load() == State::SUCCEEDED;
}

ErrorInfo LoadAsyncContext::errorInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completionErrorLocked();
}

MallocStatus LoadAsyncContext::mallocStatus() const {
    return malloc_status_.load(std::memory_order_acquire);
}

LoadContextCoordinator::LoadContextCoordinator(CommitCallback commit_callback, AbortCallback abort_callback):
    commit_callback_(std::move(commit_callback)), abort_callback_(std::move(abort_callback)) {}

std::shared_ptr<LoadAsyncContext> LoadContextCoordinator::create(std::vector<TransferDescriptor> load_descs,
                                                                 std::vector<bool>               joined_load,
                                                                 size_t                          matched_blocks,
                                                                 std::shared_ptr<StorageBackend> storage_backend,
                                                                 StorageRequest                  storage_request,
                                                                 size_t                          reuse_unit_tokens) {
    const auto coordinator = shared_from_this();
    uint64_t   context_id  = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!accepting_) {
            return nullptr;
        }
        context_id = next_context_id_++;
    }
    return std::make_shared<LoadAsyncContext>(std::move(load_descs),
                                              std::move(joined_load),
                                              matched_blocks,
                                              context_id,
                                              coordinator,
                                              std::move(storage_backend),
                                              std::move(storage_request),
                                              reuse_unit_tokens);
}

bool LoadContextCoordinator::registerContext(const std::shared_ptr<LoadAsyncContext>& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    return accepting_ && pending_contexts_.emplace(context->contextId(), context).second;
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
    if (!commit_callback_(context)) {
        // Publish the typed failure before FAILED. A scheduler may poll the
        // context as soon as onTaskFail makes it terminal.
        context->failCommit();
        return false;
    }
    return true;
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
