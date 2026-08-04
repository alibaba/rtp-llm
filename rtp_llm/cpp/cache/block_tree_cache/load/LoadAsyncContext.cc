#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"

namespace rtp_llm {

LoadAsyncContext::LoadAsyncContext(std::vector<TransferDescriptor>                load_descs,
                                   std::vector<bool>                              joined_load,
                                   size_t                                         matched_blocks,
                                   uint64_t                                       context_id,
                                   const std::shared_ptr<LoadContextCoordinator>& coordinator):
    coordinator_(coordinator),
    context_id_(context_id),
    load_descs_(std::move(load_descs)),
    joined_load_(std::move(joined_load)),
    matched_blocks_(matched_blocks),
    remaining_transfer_count_(std::count_if(load_descs_.begin(), load_descs_.end(), [](const TransferDescriptor& desc) {
        return desc.source_tier == Tier::HOST || desc.source_tier == Tier::DISK;
    })) {
    rebuildMatchedBlocksByTier();
    if (remaining_transfer_count_ == 0) {
        state_.store(State::SUCCEEDED);
    }
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
        const std::pair<std::unordered_map<size_t, Tier>::iterator, bool> insert_result =
            reuse_tier_by_path.emplace(desc.path_index, desc.source_tier);
        if (!insert_result.second
            && (desc.source_tier == Tier::DISK
                || (desc.source_tier == Tier::HOST && insert_result.first->second == Tier::DEVICE))) {
            insert_result.first->second = desc.source_tier;
        }
    }
    for (const std::pair<const size_t, Tier>& reuse_tier : reuse_tier_by_path) {
        ++matched_blocks_by_tier_[static_cast<size_t>(reuse_tier.second)];
    }
}

bool LoadAsyncContext::empty() const {
    return load_descs_.empty();
}

uint64_t LoadAsyncContext::contextId() const {
    return context_id_;
}

size_t LoadAsyncContext::matchedBlocks() const {
    return matched_blocks_;
}

size_t LoadAsyncContext::matchedBlocks(Tier tier) const {
    if (tier < Tier::DEVICE || tier > Tier::DISK) {
        return 0;
    }
    return matched_blocks_by_tier_[static_cast<size_t>(tier)];
}

void LoadAsyncContext::setTargetBlocks(size_t desc_index, std::vector<BlockIdxType> target_blocks) {
    load_descs_[desc_index].target_blocks = std::move(target_blocks);
}

const std::vector<TransferDescriptor>& LoadAsyncContext::loadDescs() const {
    return load_descs_;
}

const std::vector<bool>& LoadAsyncContext::joinedLoads() const {
    return joined_load_;
}

bool LoadAsyncContext::commit() {
    if (coordinator_ == nullptr || context_id_ == 0 || empty()) {
        return false;
    }
    return coordinator_->commit(context_id_);
}

void LoadAsyncContext::abort() {
    if (coordinator_ == nullptr || context_id_ == 0) {
        return;
    }
    const bool aborted = coordinator_->abort(context_id_, *this);
    if (aborted) {
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
        state_.store(State::CANCELLED);
    }
    cv_.notify_all();
}

bool LoadAsyncContext::requestCancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    const State                 state = state_.load();
    if (state == State::PENDING) {
        state_.store(State::CANCEL_REQUESTED);
        return true;
    }
    return state == State::CANCEL_REQUESTED;
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
        if (remaining_transfer_count_ == 0) {
            if (state == State::CANCEL_REQUESTED) {
                state_.store(State::CANCELLED);
            } else if (has_failure_) {
                state_.store(State::FAILED);
            } else {
                state_.store(State::SUCCEEDED);
            }
            notify = true;
        }
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
        if (state == State::CANCEL_REQUESTED) {
            state_.store(State::CANCELLED);
        } else {
            state_.store(State::FAILED);
        }
    }
    cv_.notify_all();
    return true;
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

LoadContextCoordinator::LoadContextCoordinator(CommitCallback commit_callback, AbortCallback abort_callback):
    commit_callback_(std::move(commit_callback)), abort_callback_(std::move(abort_callback)) {}

std::shared_ptr<LoadAsyncContext> LoadContextCoordinator::create(std::vector<TransferDescriptor> load_descs,
                                                                 std::vector<bool>               joined_load,
                                                                 size_t                          matched_blocks) {
    const std::shared_ptr<LoadContextCoordinator> coordinator = shared_from_this();

    uint64_t context_id = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!accepting_) {
            return nullptr;
        }
        context_id = next_context_id_;
        ++next_context_id_;
    }

    return std::make_shared<LoadAsyncContext>(
        std::move(load_descs), std::move(joined_load), matched_blocks, context_id, coordinator);
}

bool LoadContextCoordinator::registerContext(const std::shared_ptr<LoadAsyncContext>& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!accepting_) {
        return false;
    }

    const std::pair<PendingContextMap::iterator, bool> inserted =
        pending_contexts_.emplace(context->contextId(), std::weak_ptr<LoadAsyncContext>(context));
    if (!inserted.second) {
        return false;
    }
    return true;
}

bool LoadContextCoordinator::commit(uint64_t context_id) {
    std::shared_ptr<LoadAsyncContext> context;
    {
        std::lock_guard<std::mutex>       lock(mutex_);
        const PendingContextMap::iterator pending_it = pending_contexts_.find(context_id);
        if (!accepting_ || pending_it == pending_contexts_.end()) {
            return false;
        }
        context = pending_it->second.lock();
        if (context == nullptr || context->contextId() != context_id) {
            return false;
        }
        pending_contexts_.erase(pending_it);
        ++active_callbacks_;
    }

    block_tree_cache_detail::ScopeRollback callback_guard([this]() { retireActiveCallback(); });
    const bool                             committed = commit_callback_ && commit_callback_(context);
    if (!committed) {
        const bool failure_recorded = context->onTaskFail();
        if (!failure_recorded) {
            assert(context->done());
        }
    }
    return committed;
}

bool LoadContextCoordinator::abort(uint64_t context_id, LoadAsyncContext& context) noexcept {
    if (context_id == 0 || context.contextId() != context_id) {
        return false;
    }
    {
        std::lock_guard<std::mutex>       lock(mutex_);
        const PendingContextMap::iterator pending_it = pending_contexts_.find(context_id);
        if (pending_it == pending_contexts_.end()) {
            return false;
        }
        pending_contexts_.erase(pending_it);
        ++active_callbacks_;
    }

    block_tree_cache_detail::ScopeRollback callback_guard([this]() { retireActiveCallback(); });
    if (abort_callback_) {
        abort_callback_(context);
    }
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
    // Keep contexts alive only while shutdown explicitly aborts their pending operations.
    std::vector<std::shared_ptr<LoadAsyncContext>> live_contexts;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (accepting_) {
            accepting_ = false;
            for (const std::pair<const uint64_t, std::weak_ptr<LoadAsyncContext>>& pending_context :
                 pending_contexts_) {
                std::shared_ptr<LoadAsyncContext> context = pending_context.second.lock();
                if (context != nullptr) {
                    live_contexts.push_back(std::move(context));
                }
            }
        }
    }

    for (const std::shared_ptr<LoadAsyncContext>& context : live_contexts) {
        context->abort();
    }

    std::unique_lock<std::mutex> lock(mutex_);
    bool                         wait_observer_invoked = false;
    cv_.wait(lock, [this, &wait_observer_invoked] {
        if ((!pending_contexts_.empty() || active_callbacks_ != 0) && !wait_observer_invoked) {
            wait_observer_invoked                                       = true;
            const std::function<void()> shutdown_wait_observer_for_test = shutdown_wait_observer_for_test_;
            if (shutdown_wait_observer_for_test) {
                shutdown_wait_observer_for_test();
            }
        }
        return pending_contexts_.empty() && active_callbacks_ == 0;
    });
    commit_callback_ = {};
    abort_callback_  = {};
}

}  // namespace rtp_llm
