#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

LoadAsyncContext::LoadAsyncContext(std::vector<TransferDescriptor>                load_descs,
                                   std::vector<bool>                              joined_load,
                                   size_t                                         local_matched_blocks,
                                   uint64_t                                       context_id,
                                   const std::shared_ptr<LoadContextCoordinator>& coordinator,
                                   std::shared_ptr<StorageBackend>                storage_backend,
                                   StorageRequest                                 storage_request):
    coordinator_(coordinator),
    context_id_(context_id),
    load_descs_(std::move(load_descs)),
    joined_load_(std::move(joined_load)),
    local_matched_blocks_(local_matched_blocks),
    matched_blocks_(local_matched_blocks),
    storage_backend_(std::move(storage_backend)),
    storage_request_(std::move(storage_request)),
    need_backend_match_(storage_backend_ && !storage_request_.empty()),
    backend_pending_(need_backend_match_),
    remaining_transfer_count_(std::count_if(load_descs_.begin(), load_descs_.end(), [](const auto& desc) {
        return desc.source_tier == Tier::HOST || desc.source_tier == Tier::DISK;
    })) {
    rebuildMatchedBlocksByTier();
}

LoadAsyncContext::~LoadAsyncContext() {
    abort();
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

bool LoadAsyncContext::needBackendMatch() const {
    return need_backend_match_;
}

void LoadAsyncContext::setMatchCallback(MatchCallback callback) {
    RTP_LLM_CHECK(need_backend_match_ && callback && !match_callback_ && !backend_started_);
    match_callback_ = std::move(callback);
}

void LoadAsyncContext::startBackendMatch() {
    RTP_LLM_CHECK(need_backend_match_ && match_callback_ && !backend_started_);
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
    bool callback_started = false;
    {
        std::lock_guard<std::mutex> lock(match_callback_mutex_);
        if (!isRequestCanceled()) {
            match_callback_running_ = true;
            callback_started        = true;
        }
    }
    if (!callback_started) {
        malloc_status_.store(MallocStatus::INTERNAL_ERROR, std::memory_order_release);
        failBeforeCommit();
        return;
    }
    block_tree_cache_detail::ScopeRollback match_callback_guard([this] { finishMatchCallback(); });
    if (!success) {
        malloc_status_.store(MallocStatus::INTERNAL_ERROR, std::memory_order_release);
        failBeforeCommit();
        return;
    }
    RTP_LLM_CHECK(storage_request_.keys && storage_request_.keys->size() == storage_request_.handles.size()
                  && storage_request_.local_matched_blocks_num == local_matched_blocks_
                  && matched_blocks_num >= local_matched_blocks_
                  && matched_blocks_num <= storage_request_.handles.size());
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
                                             key_index, matched_blocks_num, handle.group_id);
                                     }),
                      handles.end());
    }
    LoadMatchResult match_result{false};
    try {
        match_result = match_callback_(*this, matched_blocks_num);
    } catch (...) {}
    if (!match_result.success) {
        malloc_status_.store(match_result.malloc_status, std::memory_order_release);
        failBeforeCommit();
        return;
    }
    if (storage_request_.empty()) {
        onBackendRead(/*success=*/true);
        return;
    }
    std::weak_ptr<LoadAsyncContext> weak = weak_from_this();
    storage_backend_->read(std::move(storage_request_), std::move(match_meta), [weak](bool success) {
        if (auto context = weak.lock()) {
            context->onBackendRead(success);
        }
    });
}

void LoadAsyncContext::finishMatchCallback() {
    {
        std::lock_guard<std::mutex> lock(match_callback_mutex_);
        match_callback_running_ = false;
    }
    match_callback_cv_.notify_all();
}

void LoadAsyncContext::onBackendRead(bool success) {
    if (!success) {
        onTaskFail();
        return;
    }
    bool notify = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        matched_blocks_  = backend_matched_blocks_;
        backend_pending_ = false;
        finishIfReadyLocked(notify);
    }
    if (notify) {
        cv_.notify_all();
    }
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
    if (!coordinator_->commit(context_id_)) {
        // The coordinator can reject before resolving the weak context. Make
        // that path terminal as well; failCommit is idempotent if a rejected
        // callback already invoked it.
        failCommit();
        return false;
    }
    bool notify = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        committed_ = true;
        finishIfReadyLocked(notify);
    }
    if (notify) {
        cv_.notify_all();
    }
    return true;
}

void LoadAsyncContext::abort() {
    requestCancel();
    if (coordinator_->abort(*this)) {
        markAborted();
    }
}

void LoadAsyncContext::markAborted() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const State                 state = state_.load();
        if (state != State::PENDING && state != State::CANCEL_REQUESTED) {
            return;
        }
        remaining_transfer_count_ = 0;
        backend_pending_          = false;
        state_.store(State::CANCELLED);
    }
    cv_.notify_all();
}

bool LoadAsyncContext::requestCancel() {
    std::unique_lock<std::mutex> lock(match_callback_mutex_);
    const State                  initial_state = state_.load();
    if (initial_state == State::PENDING) {
        state_.store(State::CANCEL_REQUESTED);
    }
    match_callback_cv_.wait(lock, [this] { return !match_callback_running_; });
    return initial_state == State::PENDING || initial_state == State::CANCEL_REQUESTED;
}

bool LoadAsyncContext::isRequestCanceled() const {
    const State state = state_.load();
    return state == State::CANCEL_REQUESTED || state == State::CANCELLED;
}

bool LoadAsyncContext::completeOne(bool success) {
    bool notify = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const State                 state = state_.load();
        if ((state != State::PENDING && state != State::CANCEL_REQUESTED) || remaining_transfer_count_ == 0) {
            return false;
        }
        has_failure_ = has_failure_ || !success;
        --remaining_transfer_count_;
        finishIfReadyLocked(notify);
    }
    if (notify) {
        cv_.notify_all();
    }
    return true;
}

bool LoadAsyncContext::onTaskFail() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const State                 state = state_.load();
        if (state != State::PENDING && state != State::CANCEL_REQUESTED) {
            return false;
        }
        remaining_transfer_count_ = 0;
        backend_pending_          = false;
        state_.store(state == State::CANCEL_REQUESTED ? State::CANCELLED : State::FAILED);
    }
    cv_.notify_all();
    return true;
}

void LoadAsyncContext::finishIfReadyLocked(bool& notify) {
    if (!committed_ || remaining_transfer_count_ != 0 || backend_pending_) {
        return;
    }
    const State state = state_.load();
    if (state != State::PENDING && state != State::CANCEL_REQUESTED) {
        return;
    }
    state_.store(state == State::CANCEL_REQUESTED ? State::CANCELLED : has_failure_ ? State::FAILED : State::SUCCEEDED);
    notify = true;
}

void LoadAsyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return done(); });
}

bool LoadAsyncContext::done() const {
    const State state = state_.load();
    return state == State::SUCCEEDED || state == State::FAILED || state == State::CANCELLED;
}

bool LoadAsyncContext::success() const {
    return state_.load() == State::SUCCEEDED;
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
                                                                 StorageRequest                  storage_request) {
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
                                              std::move(storage_request));
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
        context->abort();
    }

    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return pending_contexts_.empty() && active_callbacks_ == 0; });
    commit_callback_ = {};
    abort_callback_  = {};
}

}  // namespace rtp_llm
