#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTask.h"

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
    RTP_LLM_CHECK_WITH_INFO(execution.trace_id != 0, "frozen execution requires a nonzero trace");
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

FrozenTask::FrozenTask(PreparedExecution&& prepared, TransferResult failure):
    execution_(std::move(prepared.execution_)),
    targets_(std::move(prepared.targets_)),
    failure_(std::move(failure)),
    target_block_count_(prepared.target_block_count_),
    created_at_us_(wallTimeUs()) {
    RTP_LLM_CHECK_WITH_INFO(failure_.status.code() == ErrorCode::TRANSFER_COMPLETION_UNKNOWN,
                            "only completion-unknown executions can create a frozen task");
    RTP_LLM_CHECK_WITH_INFO(!failure_.execution_diagnostics
                               || failure_.execution_diagnostics->trace_id == execution_.trace_id,
                            "frozen execution diagnostics belong to a different trace");
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

void FrozenTask::unfreeze() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != State::FROZEN) {
            return;
        }
        state_ = State::RELEASING;
    }
    // Each prepared closure releases one reference per target, never source
    // ownership. No task/table mutex is held while pool notifications run.
    for (const auto& target : targets_) {
        target.release();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        released_at_us_ = wallTimeUs();
        state_          = State::RELEASED;
    }
}

FrozenTask::Snapshot FrozenTask::snapshot() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return {state_, created_at_us_, frozen_at_us_, released_at_us_, target_block_count_};
}

std::string FrozenTask::debugString() const {
    const auto         current = snapshot();
    std::ostringstream out;
    out << std::boolalpha << "FrozenTask{trace=" << execution_.trace_id
        << ",operation=" << std::quoted(execution_.operation) << ",request=";
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
        out << '}';
    }
    out << "],rank_snapshots=";
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
            out << '}';
        }
        out << ']';
    }
    out << '}';
    return out.str();
}

}  // namespace rtp_llm
