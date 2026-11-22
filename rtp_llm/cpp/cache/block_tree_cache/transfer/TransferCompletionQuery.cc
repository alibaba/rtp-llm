#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionQuery.h"

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionWire.h"

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

bool validState(State state) {
    return state == State::UNKNOWN || state == State::RUNNING || state == State::BUFFER_SAFE
           || state == State::REJECTED;
}

void decodeResponse(const FunctionResponsePB& response,
                    const TransferCompletionKey& trace,
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
    if (query.entries_size() != 1 || decodeCompletionTraceId(query.entries(0).key()) != trace.trace_id) {
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
TransferCompletionQuery::query(const TransferCompletionIdentity& execution, const std::vector<size_t>& worker_ids, int timeout_ms) const {
    TransferCompletionQueryResult result;
    const auto& key = execution.key;
    result.trace_id = key.trace_id;
    result.ranks.resize(worker_ids.size());
    for (size_t rank = 0; rank < result.ranks.size(); ++rank)
        result.ranks[rank].rank = worker_ids[rank];
    if (timeout_ms <= 0 || key.trace_id.empty()) {
        fail(result, "completion query requires a valid trace and positive timeout");
        return result;
    }
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    const auto rpc_deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);
    if (worker_ids.empty() || worker_ids.back() >= execution.expected_rank_count
        || std::adjacent_find(worker_ids.begin(), worker_ids.end(), std::greater_equal<size_t>()) != worker_ids.end()) {
        fail(result, "completion query requires valid original participant ranks");
        return result;
    }
    if (!execution.remote) {
        if (execution.expected_rank_count != 1 || !local_reader_) {
            fail(result, "local completion query requires one participant and a local reader");
            return result;
        }
        try {
            const auto state = local_reader_(key);
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

    if (!broadcast_manager_ || broadcast_manager_->workerNum() != execution.expected_rank_count) {
        fail(result, "completion query manager does not match the original participant count");
        return result;
    }
    try {
        FunctionRequestPB request;
        encodeCompletionKey(key, *request.mutable_completion_query()->add_keys());
        const int submit_timeout = remainingTimeout(deadline);
        if (submit_timeout == 0) {
            fail(result, "completion query deadline exceeded before submission");
            return result;
        }
        auto rpc = broadcast_manager_->broadcastToRanks<FunctionRequestPB, FunctionResponsePB>(
            worker_ids, request, submit_timeout,
            [rpc_deadline](const auto& stub, const auto& context, const auto& request, auto* queue) {
                context->set_deadline(rpc_deadline);
                return stub->AsyncExecuteFunction(context.get(), request, queue);
            },
            execution.request_id, "completion_query");
        if (!rpc) {
            fail(result, "completion query could not be dispatched");
            return result;
        }
        const int wait_timeout = remainingTimeout(deadline);
        // Zero means an infinite wait to BroadcastResult; never pass it here.
        if (!rpc->done() && wait_timeout > 0)
            rpc->waitDone(wait_timeout, false);
        for (size_t i = 0; i < result.ranks.size(); ++i) {
            auto& row = result.ranks[i];
            if (const auto response = rpc->responseIfSucceeded(i, deadline)) {
                row.grpc_code = grpc::StatusCode::OK;
                decodeResponse(*response, key, row);
            } else {
                row.error = "completion query worker did not respond successfully before the deadline";
            }
        }
        result.all_safe = result.error.empty()
                          && std::all_of(result.ranks.begin(), result.ranks.end(),
                                         [](const auto& row) { return row.error.empty() && row.bufferSafe(); });
    } catch (const std::exception& error) {
        fail(result, error.what());
    } catch (...) {
        fail(result, "completion query failed");
    }
    return result;
}

}  // namespace rtp_llm
