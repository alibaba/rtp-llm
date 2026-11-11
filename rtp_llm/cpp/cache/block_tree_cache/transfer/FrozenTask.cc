#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionTypes.h"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace rtp_llm {
namespace {

int64_t wallTimeUs() noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

const char* stateName(FrozenTask::State state) {
    switch (state) {
        case FrozenTask::State::PREPARED:
            return "PREPARED";
        case FrozenTask::State::FROZEN:
            return "FROZEN";
        case FrozenTask::State::RELEASING:
            return "RELEASING";
        case FrozenTask::State::RELEASED:
            return "RELEASED";
    }
    return "UNKNOWN";
}

const char* dispatchName(TransferRpcDispatchState state) {
    switch (state) {
        case TransferRpcDispatchState::NOT_SUBMITTED:
            return "NOT_SUBMITTED";
        case TransferRpcDispatchState::POSSIBLY_SUBMITTED:
            return "POSSIBLY_SUBMITTED";
        case TransferRpcDispatchState::SUBMITTED:
            return "SUBMITTED";
    }
    return "UNKNOWN";
}

const char* completionStateName(TransferTaskRegistry::State state) {
    switch (state) {
        case TransferTaskRegistry::State::UNKNOWN:
            return "UNKNOWN";
        case TransferTaskRegistry::State::RUNNING:
            return "RUNNING";
        case TransferTaskRegistry::State::BUFFER_SAFE:
            return "BUFFER_SAFE";
    }
    return "UNKNOWN";
}

template<typename T>
void writeOptional(std::ostream& out, const std::optional<T>& value) {
    if (value) {
        out << *value;
    } else {
        out << "unknown";
    }
}

void writeErrors(std::ostream& out, const std::vector<CacheCopyError>& errors) {
    out << '[';
    for (size_t i = 0; i < errors.size(); ++i) {
        if (i != 0)
            out << ',';
        out << cacheCopyErrorName(errors[i]);
    }
    out << ']';
}

void writeBlocks(std::ostream& out, const std::vector<FrozenTask::BlockInfo>& blocks) {
    out << '[';
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (i != 0)
            out << ',';
        const auto& block = blocks[i];
        out << "{pool=" << std::quoted(block.pool_name) << ",pool_address=" << block.pool.get()
            << ",block=" << block.block << ",member=" << block.member_index << '}';
    }
    out << ']';
}

}  // namespace

FrozenTask::PreparedExecution FrozenTask::prepare(ExecutionInfo execution) {
    RTP_LLM_CHECK_WITH_INFO(validTransferExecutionTrace(execution.trace_id),
                            "frozen execution requires a valid trace");
    std::vector<TargetPool>                        targets;
    std::unordered_map<IBlockPool*, size_t>        pool_indices;
    std::unordered_map<IBlockPool*, std::unordered_set<BlockIdxType>> seen;
    size_t                                        target_block_count = 0;
    for (auto& descriptor : execution.descriptors) {
        for (auto& source : descriptor.sources) {
            // Source metadata can be unavailable on a diagnostic-only/mock
            // path. Sources never contribute frozen references.
            if (source.pool) {
                source.pool_name = source.pool->poolName();
            }
        }
        for (auto& target : descriptor.targets) {
            RTP_LLM_CHECK_WITH_INFO(target.pool != nullptr, "frozen target requires a physical pool");
            target.pool_name = target.pool->poolName();
            if (!seen[target.pool.get()].insert(target.block).second) {
                continue;
            }
            auto [position, inserted] = pool_indices.emplace(target.pool.get(), targets.size());
            if (inserted) {
                targets.push_back(TargetPool{target.pool, {}, {}});
            }
            targets[position->second].blocks.push_back(target.block);
            ++target_block_count;
        }
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
    RTP_LLM_CHECK_WITH_INFO(failure_.status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN,
                            "only completion-unknown executions can create a frozen task");
    RTP_LLM_CHECK_WITH_INFO(!failure_.execution_diagnostics
                               || failure_.execution_diagnostics->trace_id == execution_.trace_id,
                            "frozen execution diagnostics belong to a different trace");
}

void FrozenTask::freeze() noexcept {
    // Snapshot before taking mutex_: debugString() also locks it. The pool
    // prints this context only if a frozen-block threshold triggers abort.
    std::string abort_context;
    try {
        abort_context = debugString();
    } catch (...) {
        // Diagnostics must not prevent acquiring the frozen references.
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != State::PREPARED) {
        return;
    }
    for (const auto& target : targets_) {
        for (const auto block : target.blocks) {
            target.pool->freezeBlock(block, abort_context.empty() ? nullptr : abort_context.c_str());
        }
    }
    frozen_at_us_ = wallTimeUs();
    state_        = State::FROZEN;
}

bool FrozenTask::unfreeze() noexcept {
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
    // Cleanup can forget only worker evidence. Transfer its preallocated node
    // after every physical release, without holding task or pool locks.
    cleanup_.submit();
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

std::string FrozenTask::debugString() const {
    const auto         current = snapshot();
    const auto         recovery = recoverySnapshot();
    std::ostringstream out;
    out << std::boolalpha << "FrozenTask{trace=";
    if (const auto* numeric = std::get_if<uint64_t>(&execution_.trace_id)) {
        out << *numeric << ",execution_type=transfer";
    } else {
        out << std::quoted(std::get<std::string>(execution_.trace_id)) << ",execution_type=kvcm_read";
    }
    out << ",operation=" << std::quoted(execution_.operation) << ",request=";
    writeOptional(out, execution_.request_id);
    out << ",workflow=" << execution_.workflow_id << ",parent_trace=" << execution_.parent_trace_id
        << ",remote=" << execution_.remote << ",expected_rank_count=" << execution_.expected_rank_count
        << ",state=" << stateName(current.state) << ",created_at_us=" << current.created_at_us
        << ",frozen_at_us=" << current.frozen_at_us << ",released_at_us=" << current.released_at_us
        << ",target_block_count=" << current.target_block_count << ",error_code="
        << static_cast<int64_t>(failure_.status.code()) << ",error=" << std::quoted(failure_.status.ToString())
        << ",copy_error=" << cacheCopyErrorName(failure_.copy_error) << ",buffer_safe=" << failure_.buffer_safe
        << ",block_errors=";
    writeErrors(out, failure_.block_errors);
    out << ",expected_peers=[";
    for (size_t i = 0; i < execution_.expected_peers.size(); ++i) {
        if (i != 0)
            out << ',';
        const auto& peer = execution_.expected_peers[i];
        out << "{rank=" << peer.rank << ",address=" << std::quoted(peer.address) << '}';
    }
    out << "],descriptors=[";
    for (size_t i = 0; i < execution_.descriptors.size(); ++i) {
        if (i != 0)
            out << ',';
        const auto& descriptor = execution_.descriptors[i];
        out << "{index=" << i << ",group_set=" << descriptor.group_set_id << ",path_index=" << descriptor.path_index
            << ",source_tier=" << tierName(descriptor.source_tier)
            << ",target_tier=" << tierName(descriptor.target_tier) << ",sources=";
        writeBlocks(out, descriptor.sources);
        out << ",targets=";
        writeBlocks(out, descriptor.targets);
        if (descriptor.rank) {
            out << ",rank=" << *descriptor.rank << ",group_tag=" << std::quoted(descriptor.group_tag)
                << ",source_uri=" << std::quoted(descriptor.source_uri);
        }
        out << '}';
    }
    out << "],failure_reason=";
    if (failure_.execution_diagnostics) {
        out << std::quoted(failure_.execution_diagnostics->failure_reason);
    } else {
        out << "unknown";
    }
    out << ",rank_snapshots=";
    if (!failure_.execution_diagnostics) {
        out << "unknown";
    } else {
        out << '[';
        const auto& ranks = failure_.execution_diagnostics->ranks;
        for (size_t i = 0; i < ranks.size(); ++i) {
            if (i != 0)
                out << ',';
            const auto& rank = ranks[i];
            out << "{rank=" << rank.rank << ",address=" << std::quoted(rank.server_address)
                << ",remote=" << rank.remote << ",dispatch_state=" << dispatchName(rank.dispatch_state)
                << ",dispatch_error=" << std::quoted(rank.dispatch_error) << ",cq_received=" << rank.cq_received
                << ",cq_ok=" << rank.cq_ok << ",grpc_code=";
            writeOptional(out, rank.grpc_code);
            out << ",grpc_message=" << std::quoted(rank.grpc_message)
                << ",response_received=" << rank.response_received << ",response_valid=" << rank.response_valid
                << ",response_error=" << std::quoted(rank.response_error) << ",worker_error_code=";
            writeOptional(out, rank.worker_error_code);
            out << ",worker_error_message=" << std::quoted(rank.worker_error_message) << ",worker_copy_error=";
            if (rank.worker_copy_error) {
                out << cacheCopyErrorName(*rank.worker_copy_error);
            } else {
                out << "unknown";
            }
            out << ",block_errors=";
            writeErrors(out, rank.block_errors);
            out << ",buffer_safe=";
            writeOptional(out, rank.buffer_safe);
            out << ",worker_transfer_status=";
            writeOptional(out, rank.worker_transfer_status);
            out << ",sdk_failed=";
            writeOptional(out, rank.sdk_failed);
            out << '}';
        }
        out << ']';
    }
    out << ",recovery_query_count=" << recovery.query_count
        << ",last_recovery_query_at_us=" << recovery.last_query_at_us << ",last_recovery_query=";
    if (!recovery.last_query) {
        out << "unknown";
    } else {
        const auto& query = *recovery.last_query;
        out << "{trace=";
        if (const auto* numeric = std::get_if<uint64_t>(&query.trace_id)) {
            out << *numeric;
        } else {
            out << std::quoted(std::get<std::string>(query.trace_id));
        }
        out << ",all_safe=" << query.all_safe << ",error=" << std::quoted(query.error) << ",ranks=[";
        for (size_t i = 0; i < query.ranks.size(); ++i) {
            if (i != 0)
                out << ',';
            const auto& rank = query.ranks[i];
            out << "{rank=" << rank.rank << ",address=" << std::quoted(rank.server_address)
                << ",state=" << completionStateName(rank.state) << ",grpc_code=";
            writeOptional(out, rank.grpc_code);
            out << ",error=" << std::quoted(rank.error) << ",cq_received=" << rank.cq_received
                << ",cq_ok=" << rank.cq_ok << '}';
        }
        out << "]}";
    }
    out << '}';
    return out.str();
}

}  // namespace rtp_llm
