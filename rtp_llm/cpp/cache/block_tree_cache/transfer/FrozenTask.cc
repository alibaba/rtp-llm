#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionTypes.h"

#include <chrono>
#include <unordered_map>
#include <unordered_set>

namespace rtp_llm {
namespace {

int64_t wallTimeUs() noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace

FrozenTask::PreparedExecution FrozenTask::prepare(ExecutionInfo execution) {
    RTP_LLM_CHECK_WITH_INFO(validTransferCompletionKey(execution.registry_type, execution.trace_id),
                            "frozen execution requires a valid trace");
    std::vector<TargetPool>                        targets;
    std::unordered_map<IBlockPool*, size_t>        pool_indices;
    std::unordered_map<IBlockPool*, std::unordered_set<BlockIdxType>> seen;
    size_t                                        target_block_count = 0;
    for (const auto& target : execution.target_blocks) {
        RTP_LLM_CHECK_WITH_INFO(target.pool != nullptr, "frozen target requires a physical pool");
        if (!seen[target.pool.get()].insert(target.block).second)
            continue;
        auto [position, inserted] = pool_indices.emplace(target.pool.get(), targets.size());
        if (inserted)
            targets.push_back(TargetPool{target.pool, {}, {}});
        targets[position->second].blocks.push_back(target.block);
        ++target_block_count;
    }
    for (auto& target : targets) {
        target.release = target.pool->prepareFrozenReferenceRelease(target.blocks);
    }
    return PreparedExecution(std::move(execution), std::move(targets), target_block_count);
}

FrozenTask::FrozenTask(PreparedExecution&& prepared, TransferResult failure,
                       TransferCompletionCleanup::Prepared cleanup):
    execution_(std::move(prepared.execution_)),
    targets_(std::move(prepared.targets_)),
    failure_(std::move(failure)),
    target_block_count_(prepared.target_block_count_),
    created_at_us_(wallTimeUs()),
    cleanup_(std::move(cleanup)) {
    RTP_LLM_CHECK_WITH_INFO(failure_.checkNeedFreeze(),
                            "frozen execution requires unproven buffer safety");
}

void FrozenTask::freeze() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != State::PREPARED) {
        return;
    }
    for (const auto& target : targets_) {
        for (const auto block : target.blocks) {
            target.pool->freezeBlock(block);
        }
    }
    frozen_at_us_ = wallTimeUs();
    state_        = State::FROZEN;
}

bool FrozenTask::completeRecovery(TransferCompletionCleanup& cleanup_queue) noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != State::FROZEN) {
            return false;
        }
        state_ = State::RELEASING;
    }
    // Each prepared closure releases one reference per target, never source
    // ownership. No task/table mutex is held while pool notifications run.
    for (const auto& target : targets_) {
        target.release();
    }
    // Cleanup acknowledges every participant after physical release. Workers
    // delete safe completion records but retain admission bans on forget.
    cleanup_queue.enqueue(std::move(cleanup_));
    {
        std::lock_guard<std::mutex> lock(mutex_);
        released_at_us_ = wallTimeUs();
        state_          = State::RELEASED;
    }
    return true;
}

void FrozenTask::recordRecoveryQuery(std::shared_ptr<const TransferCompletionQueryResult> query) noexcept {
    if (!query) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++recovery_.query_count;
        recovery_.last_query_at_us = wallTimeUs();
        recovery_.last_query.swap(query);
    }
    // Release the previous observation after dropping the task mutex.
}

FrozenTask::RecoverySnapshot FrozenTask::recoverySnapshot() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return recovery_;
}

FrozenTask::Snapshot FrozenTask::snapshot() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return {state_, created_at_us_, frozen_at_us_, released_at_us_, target_block_count_};
}

}  // namespace rtp_llm
