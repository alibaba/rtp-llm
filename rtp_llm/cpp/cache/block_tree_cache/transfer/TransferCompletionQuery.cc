#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionQuery.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <utility>

#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {
namespace {

using State = TransferTaskRegistry::State;
using Clock = std::chrono::steady_clock;

void encodeKey(const TransferExecutionTrace& trace, TransferCompletionKeyPB& key) {
    if (const auto* numeric = std::get_if<uint64_t>(&trace)) {
        key.set_transfer_trace_id(*numeric);
    } else {
        key.set_kvcm_trace_id(std::get<std::string>(trace));
    }
}

bool sameKey(const TransferCompletionKeyPB& key, const TransferExecutionTrace& trace) {
    if (const auto* numeric = std::get_if<uint64_t>(&trace)) {
        return key.trace_case() == TransferCompletionKeyPB::kTransferTraceId && key.transfer_trace_id() == *numeric;
    }
    return key.trace_case() == TransferCompletionKeyPB::kKvcmTraceId
           && key.kvcm_trace_id() == std::get<std::string>(trace);
}

bool validState(State state) {
    return state == State::UNKNOWN || state == State::RUNNING || state == State::BUFFER_SAFE
           || state == State::REJECTED;
}

void decodeResponse(const FunctionResponsePB& response,
                    const TransferExecutionTrace& trace,
                    TransferCompletionRankResult& row) {
    if (!response.has_completion_query()) {
        row.error = "missing completion query response";
        return;
    }
    const auto& query = response.completion_query();
    if (query.worker_case() != TransferCompletionQueryResponsePB::kRank || query.rank() != row.rank) {
        row.error = "completion query worker rank mismatch";
        return;
    }
    if (query.entries_size() != 1 || !sameKey(query.entries(0).key(), trace)) {
        row.error = "completion query key count or identity mismatch";
        return;
    }
    switch (query.entries(0).state()) {
        case TRANSFER_COMPLETION_UNKNOWN:
            row.state = State::UNKNOWN;
            break;
        case TRANSFER_COMPLETION_RUNNING:
            row.state = State::RUNNING;
            break;
        case TRANSFER_COMPLETION_BUFFER_SAFE:
            row.state = State::BUFFER_SAFE;
            break;
        default:
            row.error = "invalid completion query state";
            break;
    }
    row.admission_closed = query.entries(0).admission_closed();
    if (row.admission_closed && row.state != State::UNKNOWN)
        row.error = "admission ban requires UNKNOWN completion state";
}

int remainingTimeout(Clock::time_point deadline) {
    const auto now = Clock::now();
    return now >= deadline ? 0 : static_cast<int>(std::chrono::ceil<std::chrono::milliseconds>(deadline - now).count());
}

void fail(TransferCompletionQueryResult& result, const std::string& error) {
    result.all_safe = false;
    result.error = error;
    for (auto& row : result.ranks) {
        row.state = State::UNKNOWN;
        row.admission_closed = false;
        row.error = error;
    }
}

}  // namespace

TransferCompletionQuery::TransferCompletionQuery(std::shared_ptr<BroadcastManager> broadcast_manager,
                                                 LocalReader local_reader):
    broadcast_manager_(std::move(broadcast_manager)), local_reader_(std::move(local_reader)) {}

TransferCompletionQueryResult
TransferCompletionQuery::query(const FrozenTask::ExecutionInfo& execution, int timeout_ms) const {
    TransferCompletionQueryResult result;
    result.trace_id = execution.trace_id;
    result.ranks.resize(execution.expected_rank_count);
    for (size_t rank = 0; rank < result.ranks.size(); ++rank)
        result.ranks[rank].rank = rank;
    for (const auto& peer : execution.expected_peers) {
        if (peer.rank < result.ranks.size())
            result.ranks[peer.rank].server_address = peer.address;
    }
    if (timeout_ms <= 0 || !validTransferExecutionTrace(execution.trace_id)) {
        fail(result, "completion query requires a valid trace and positive timeout");
        return result;
    }
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    const auto rpc_deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);
    if (result.ranks.empty() || execution.expected_peers.size() != result.ranks.size()) {
        fail(result, "completion query requires the complete original participant set");
        return result;
    }
    std::vector<bool> seen(result.ranks.size(), false);
    for (const auto& peer : execution.expected_peers) {
        if (peer.rank >= seen.size() || seen[peer.rank] || peer.address.empty()) {
            fail(result, "completion query participants contain a missing, duplicate, or invalid rank");
            return result;
        }
        seen[peer.rank] = true;
    }

    if (!execution.remote) {
        if (result.ranks.size() != 1 || !local_reader_) {
            fail(result, "local completion query requires one participant and a local reader");
            return result;
        }
        try {
            const auto state = local_reader_(execution.trace_id);
            if (!validState(state)) {
                fail(result, "local completion reader returned an invalid state");
            } else if (remainingTimeout(deadline) == 0) {
                fail(result, "completion query deadline exceeded");
            } else {
                auto& row = result.ranks.front();
                row.admission_closed = state == State::REJECTED;
                row.state = row.admission_closed ? State::UNKNOWN : state;
                result.all_safe = row.bufferSafe();
            }
        } catch (const std::exception& error) {
            fail(result, error.what());
        } catch (...) {
            fail(result, "local completion reader failed");
        }
        return result;
    }

    if (!broadcast_manager_ || broadcast_manager_->workerAddresses().size() != result.ranks.size()) {
        fail(result, "completion query manager does not match the original participant count");
        return result;
    }
    for (const auto& row : result.ranks) {
        if (broadcast_manager_->workerAddresses()[row.rank] != row.server_address) {
            fail(result, "completion query manager does not match the original participant addresses");
            return result;
        }
    }
    try {
        std::vector<FunctionRequestPB> requests(result.ranks.size());
        for (auto& request : requests)
            encodeKey(execution.trace_id, *request.mutable_completion_query()->add_keys());
        const int submit_timeout = remainingTimeout(deadline);
        if (submit_timeout == 0) {
            fail(result, "completion query deadline exceeded before submission");
            return result;
        }
        auto rpc = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
            requests, submit_timeout,
            [rpc_deadline](const auto& stub, const auto& context, const auto& request, auto* queue) {
                context->set_deadline(rpc_deadline);
                return stub->AsyncExecuteFunction(context.get(), request, queue);
            },
            execution.request_id, "completion_query", /*fatal_on_error=*/false);
        if (!rpc) {
            fail(result, "completion query could not be dispatched");
            return result;
        }
        const int wait_timeout = remainingTimeout(deadline);
        // Zero means an infinite wait to BroadcastResult; never pass it here.
        const bool completed = rpc->done() || (wait_timeout > 0 && rpc->waitDone(wait_timeout, false));
        if (!completed)
            result.error = "completion query deadline exceeded";
        for (const auto& snapshot : rpc->rankSnapshots()) {
            auto& row = result.ranks.at(snapshot.rank);
            row.cq_received = snapshot.cq_received;
            row.cq_ok = snapshot.cq_ok;
            row.grpc_code = snapshot.grpc_code;
            if (!snapshot.cq_received || !snapshot.cq_ok || !snapshot.grpc_code) {
                row.error = snapshot.dispatch_error.empty() ? "completion query has no successful CQ event" :
                                                              snapshot.dispatch_error;
            } else if (*snapshot.grpc_code != grpc::StatusCode::OK) {
                row.error = snapshot.grpc_message.empty() ? "completion query RPC failed" : snapshot.grpc_message;
            } else if (!snapshot.response) {
                row.error = "completion query response is unavailable";
            } else {
                decodeResponse(*snapshot.response, execution.trace_id, row);
            }
        }
        if (remainingTimeout(deadline) == 0) {
            result.error = "completion query deadline exceeded";
        }
        if (result.error.empty()
            && std::any_of(result.ranks.begin(), result.ranks.end(), [](const auto& row) { return !row.error.empty(); })) {
            result.error = "one or more completion query ranks failed";
        }
        result.all_safe = result.error.empty()
                          && std::all_of(result.ranks.begin(), result.ranks.end(),
                                         [](const auto& row) { return row.bufferSafe(); });
    } catch (const std::exception& error) {
        fail(result, error.what());
    } catch (...) {
        fail(result, "completion query failed");
    }
    return result;
}

}  // namespace rtp_llm
