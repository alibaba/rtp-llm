#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeReclaimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include <limits>
namespace rtp_llm {
BlockTreeReclaimer::BlockTreeReclaimer(BlockTreeCache& tree): tree_(tree), completion_cleanup_(std::make_shared<TransferCompletionCleanup>()) {}
BlockTreeReclaimer::~BlockTreeReclaimer() {
    stop();
}

void BlockTreeReclaimer::configure(int rank, int64_t interval, int64_t timeout, int64_t force_release_timeout, std::shared_ptr<BroadcastManager> broadcast) {
    rank_ = rank;
    interval_ms_ = interval;
    timeout_ms_ = timeout;
    force_release_timeout_ms_ = force_release_timeout;
    transfer_broadcast_manager_ = std::move(broadcast);
}

bool BlockTreeReclaimer::validConfig(int64_t interval, int64_t timeout, int64_t force_release_timeout) {
    return interval >= 0 && interval <= std::numeric_limits<int>::max() && timeout > 0 && timeout <= std::numeric_limits<int>::max()
           && force_release_timeout >= 0 && force_release_timeout <= std::numeric_limits<int>::max();
}

bool BlockTreeReclaimer::start() {
    if (!validConfig(interval_ms_, timeout_ms_, force_release_timeout_ms_))
        return false;
    if (cleanup_thread_.joinable())
        return true;
    stopped_ = false;
    if (rank_ == 0) {
        tree_.setCompletionCleanup(completion_cleanup_);
        if (auto backend = tree_.storageBackend())
            backend->setCompletionCleanup(completion_cleanup_);
    }
    try {
        cleanup_thread_ = std::thread([this] { cleanupLoop(); });
        if (startFrozenTaskRecovery())
            return true;
    } catch (...) {
        RTP_LLM_LOG_ERROR("failed to start block tree reclaimer");
    }
    stop();
    return false;
}

void BlockTreeReclaimer::stop() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
    }
    cv_.notify_all();
    stopFrozenTaskRecovery();
    if (cleanup_thread_.joinable())
        cleanup_thread_.join();
}

bool BlockTreeReclaimer::startFrozenTaskRecovery() {
    if (rank_ != 0 || interval_ms_ == 0 || frozen_task_recovery_)
        return true;
    auto recovery = std::make_unique<FrozenTaskRecovery>(*completion_cleanup_,
        [this] { return frozenTasks(); },
        [this](const auto& execution, int timeout) { return queryTransferCompletion(execution, timeout); },
        FrozenTaskRecovery::Cleanup{},
        [this](const auto& execution, bool forced) { tree_.reportUnfreezeMetrics(execution, forced); },
        static_cast<int>(force_release_timeout_ms_));
    if (!recovery->start(static_cast<int>(interval_ms_), static_cast<int>(timeout_ms_)))
        return false;
    frozen_task_recovery_ = std::move(recovery);
    return true;
}

void BlockTreeReclaimer::stopFrozenTaskRecovery() noexcept {
    if (frozen_task_recovery_) {
        frozen_task_recovery_->stop();
        frozen_task_recovery_.reset();
    }
}

void BlockTreeReclaimer::cleanupLoop() noexcept {
    auto next_missing = std::chrono::steady_clock::now();
    const auto stopped = [this] { return stopped_.load(); };
    while (!stopped()) {
        try {
            const auto now = std::chrono::steady_clock::now();
            if (now >= next_missing) {
                tree_.cleanupExpiredMissing();
                if (auto backend = tree_.storageBackend())
                    backend->cleanupExpiredMissing();
                next_missing = now + std::chrono::minutes(1);
            }
            if (rank_ == 0) {
                completion_cleanup_->drain(
                    [this](const auto& target, const auto& traces, int timeout) {
                        return forgetTransferCompletion(target, traces, timeout);
                    }, 1000, stopped);
            }
        } catch (...) {
            RTP_LLM_LOG_WARNING("block tree daily cleanup failed; retrying next pass");
        }
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::milliseconds(100), stopped);
    }
}

std::vector<std::shared_ptr<FrozenTask>> BlockTreeReclaimer::frozenTasks() const {
    auto tasks = tree_.frozenTasks();
    auto backend = tree_.storageBackend();
    if (backend) {
        const auto remote_tasks = backend->frozenTasks();
        tasks.insert(tasks.end(), remote_tasks.begin(), remote_tasks.end());
    }
    return tasks;
}


TransferTaskRegistry::State BlockTreeReclaimer::localTransferCompletion(const TransferCompletionKey& key) const {
    if (!validTransferCompletionKey(key)) {
        return TransferTaskRegistry::State::UNKNOWN;
    }
    if (key.registry_type == RegistryType::BLOCK_TRANSFER) {
        return tree_.localTransferCompletion(key.numericId());
    }
    auto backend = tree_.storageBackend();
    if (backend) {
        return backend->queryReadCompletion(key.trace_id);
    }
    return TransferTaskRegistry::State::UNKNOWN;
}


TransferTaskRegistry::ForgetResult
BlockTreeReclaimer::localForgetTransferCompletion(const TransferCompletionKey& key) const {
    using Result = TransferTaskRegistry::ForgetResult;
    if (!validTransferCompletionKey(key)) {
        RTP_LLM_LOG_WARNING("completion forget requires a nonempty execution trace");
        return Result::INVALID;
    }
    if (key.registry_type == RegistryType::BLOCK_TRANSFER)
        return tree_.forgetLocalTransferCompletion(key.numericId());
    auto backend = tree_.storageBackend();
    if (backend)
        return backend->forgetReadCompletion(key.trace_id);
    return Result::FAILED;
}


TransferCompletionForgetResult
BlockTreeReclaimer::forgetTransferCompletion(const TransferCompletionTarget& target,
                                        const std::vector<TransferCompletionKey>& traces, int timeout_ms) const {
    return TransferCompletionForget(transfer_broadcast_manager_,
                                    [this](const TransferCompletionKey& trace) {
                                        return localForgetTransferCompletion(trace);
                                    }).forget(target, traces, timeout_ms);
}


TransferCompletionQueryResult BlockTreeReclaimer::queryTransferCompletion(const FrozenTask::ExecutionInfo& execution,
                                                                    int timeout_ms) const {
    TransferCompletionIdentity identity;
    identity.registry_type       = execution.registry_type;
    identity.trace_id            = execution.trace_id;
    identity.remote              = execution.uses_rpc;
    identity.expected_rank_count = execution.rank_count;
    return TransferCompletionQuery(transfer_broadcast_manager_,
                                   [this](const TransferCompletionKey& key) {
                                       return localTransferCompletion(key);
                                   })
        .query(identity, timeout_ms);
}


bool BlockTreeReclaimer::executeFunction(BlockTreeReclaimer* owner, int rank, const FunctionRequestPB& request, FunctionResponsePB& response) {
    if (request.has_completion_forget()) {
        auto* forget_response = response.mutable_completion_forget();
        forget_response->Clear();
        forget_response->set_rank(static_cast<uint32_t>(rank));
        const auto& keys = request.completion_forget().keys();
        if (static_cast<size_t>(keys.size()) > kTransferCompletionForgetBatchSize) {
            RTP_LLM_LOG_WARNING("completion forget batch exceeds limit: %d", keys.size());
            return false;
        }
        for (const auto& key : keys) {
            auto* entry = forget_response->add_entries();
            entry->mutable_key()->CopyFrom(key);
            auto result = TransferTaskRegistry::ForgetResult::INVALID;
            switch (key.trace_case()) {
                case TransferCompletionKeyPB::kTransferTraceId:
                    result = owner ? owner->localForgetTransferCompletion(TransferCompletionKey{key.transfer_trace_id()}) : TransferTaskRegistry::ForgetResult::FAILED;
                    break;
                case TransferCompletionKeyPB::kKvcmTraceId:
                    result = owner ? owner->localForgetTransferCompletion(TransferCompletionKey{key.kvcm_trace_id()}) : TransferTaskRegistry::ForgetResult::FAILED;
                    break;
                default:
                    RTP_LLM_LOG_WARNING("completion forget key has no trace namespace");
                    break;
            }
            switch (result) {
                case TransferTaskRegistry::ForgetResult::DONE:
                    entry->set_status(TRANSFER_FORGET_DONE);
                    break;
                case TransferTaskRegistry::ForgetResult::RUNNING:
                    entry->set_status(TRANSFER_FORGET_RUNNING);
                    break;
                case TransferTaskRegistry::ForgetResult::INVALID:
                    entry->set_status(TRANSFER_FORGET_INVALID);
                    break;
                default:
                    entry->set_status(TRANSFER_FORGET_FAILED);
                    break;
            }
        }
        return true;
    }
    if (request.has_completion_query()) {
        auto* query_response = response.mutable_completion_query();
        query_response->Clear();
        query_response->set_rank(static_cast<uint32_t>(rank));
        for (const auto& key : request.completion_query().keys()) {
            auto* entry = query_response->add_entries();
            entry->mutable_key()->CopyFrom(key);
            auto state = TransferTaskRegistry::State::UNKNOWN;
            switch (key.trace_case()) {
                case TransferCompletionKeyPB::kTransferTraceId:
                    state = owner ? owner->localTransferCompletion(TransferCompletionKey{key.transfer_trace_id()}) : TransferTaskRegistry::State::UNKNOWN;
                    break;
                case TransferCompletionKeyPB::kKvcmTraceId:
                    state = owner ? owner->localTransferCompletion(TransferCompletionKey{key.kvcm_trace_id()}) : TransferTaskRegistry::State::UNKNOWN;
                    break;
                default:
                    break;
            }
            switch (state) {
                case TransferTaskRegistry::State::REJECTED:
                    entry->set_state(TRANSFER_COMPLETION_UNKNOWN);
                    entry->set_admission_closed(true);
                    break;
                case TransferTaskRegistry::State::RUNNING:
                    entry->set_state(TRANSFER_COMPLETION_RUNNING);
                    break;
                case TransferTaskRegistry::State::BUFFER_SAFE:
                    entry->set_state(TRANSFER_COMPLETION_BUFFER_SAFE);
                    break;
                default:
                    entry->set_state(TRANSFER_COMPLETION_UNKNOWN);
                    break;
            }
        }
        return true;
    }
    return false;
}

}
