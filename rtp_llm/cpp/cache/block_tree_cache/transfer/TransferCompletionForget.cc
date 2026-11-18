#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionForget.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <utility>

#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {
namespace {

using Outcome = TransferTaskRegistry::ForgetResult;
using Clock = std::chrono::steady_clock;

void encodeKey(const TransferExecutionTrace& trace, TransferCompletionKeyPB& key) {
    if (const auto* numeric = std::get_if<uint64_t>(&trace))
        key.set_transfer_trace_id(*numeric);
    else
        key.set_kvcm_trace_id(std::get<std::string>(trace));
}

bool sameKey(const TransferCompletionKeyPB& key, const TransferExecutionTrace& trace) {
    if (const auto* numeric = std::get_if<uint64_t>(&trace))
        return key.trace_case() == TransferCompletionKeyPB::kTransferTraceId && key.transfer_trace_id() == *numeric;
    return key.trace_case() == TransferCompletionKeyPB::kKvcmTraceId
           && key.kvcm_trace_id() == std::get<std::string>(trace);
}

int remainingTimeout(Clock::time_point deadline) {
    const auto now = Clock::now();
    return now >= deadline ? 0 : static_cast<int>(std::chrono::ceil<std::chrono::milliseconds>(deadline - now).count());
}

void fail(TransferCompletionForgetResult& result, const std::string& error) {
    result.error = error;
    for (auto& entry : result.entries) {
        entry.all_done = false;
        for (auto& row : entry.ranks) {
            row.result = Outcome::FAILED;
            row.error = error;
        }
    }
}

void setOutcome(TransferCompletionForgetRankResult& row, Outcome outcome) {
    switch (outcome) {
        case Outcome::DONE:
        case Outcome::RUNNING:
            row.result = outcome;
            break;
        case Outcome::FAILED:
            row.error = "worker completion forget failed";
            break;
        case Outcome::INVALID:
            row.result = outcome;
            row.error = "worker rejected completion forget key";
            break;
        default:
            row.error = "invalid completion forget status";
            break;
    }
}

void decodeEntry(const TransferCompletionForgetEntryPB& wire, const TransferExecutionTrace& trace,
                 TransferCompletionForgetRankResult& row) {
    if (!sameKey(wire.key(), trace)) {
        row.error = "completion forget key identity mismatch";
        return;
    }
    switch (wire.status()) {
        case TRANSFER_FORGET_DONE: setOutcome(row, Outcome::DONE); break;
        case TRANSFER_FORGET_RUNNING: setOutcome(row, Outcome::RUNNING); break;
        case TRANSFER_FORGET_INVALID: setOutcome(row, Outcome::INVALID); break;
        case TRANSFER_FORGET_FAILED: setOutcome(row, Outcome::FAILED); break;
        default: row.error = "invalid completion forget status"; break;
    }
}

void summarize(TransferCompletionForgetResult& result, bool within_deadline) {
    for (auto& entry : result.entries) {
        entry.all_done = within_deadline && !entry.ranks.empty()
                         && std::all_of(entry.ranks.begin(), entry.ranks.end(), [](const auto& row) {
                                return row.result == Outcome::DONE && row.error.empty();
                            });
        if (result.error.empty()
            && std::any_of(entry.ranks.begin(), entry.ranks.end(), [](const auto& row) { return !row.error.empty(); }))
            result.error = "one or more completion forget entries failed";
    }
}

}  // namespace

TransferCompletionForget::TransferCompletionForget(std::shared_ptr<BroadcastManager> broadcast_manager,
                                                   LocalForget local_forget):
    broadcast_manager_(std::move(broadcast_manager)), local_forget_(std::move(local_forget)) {}

TransferCompletionForgetResult
TransferCompletionForget::forget(const TransferCompletionTarget& target,
                                  const std::vector<TransferExecutionTrace>& traces, int timeout_ms) const {
    TransferCompletionForgetResult result;
    if (traces.size() > kTransferCompletionForgetBatchSize) {
        result.error = "completion forget batch exceeds the maximum size";
        return result;
    }
    result.entries.resize(traces.size());
    for (size_t index = 0; index < traces.size(); ++index) {
        auto& entry = result.entries[index];
        entry.trace_id = traces[index];
        entry.ranks.resize(target.peers.size());
        for (size_t rank = 0; rank < entry.ranks.size(); ++rank)
            entry.ranks[rank].rank = rank;
        for (const auto& peer : target.peers) {
            if (peer.rank < entry.ranks.size())
                entry.ranks[peer.rank].server_address = peer.address;
        }
    }
    if (timeout_ms <= 0 || traces.empty()) {
        fail(result, "completion forget requires a nonempty batch and positive timeout");
        return result;
    }
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    const auto rpc_deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);
    if (target.peers.empty()) {
        fail(result, "completion forget requires the original participants");
        return result;
    }
    std::vector<bool> seen(target.peers.size(), false);
    for (const auto& peer : target.peers) {
        if (peer.rank >= seen.size() || seen[peer.rank] || peer.address.empty()) {
            fail(result, "completion forget participants contain a missing, duplicate, or invalid rank");
            return result;
        }
        seen[peer.rank] = true;
    }
    if (!target.remote && (target.peers.size() != 1 || !local_forget_)) {
        fail(result, "local completion forget requires one participant and a local handler");
        return result;
    }
    if (target.remote) {
        if (!broadcast_manager_ || broadcast_manager_->workerAddresses().size() != target.peers.size()) {
            fail(result, "completion forget manager does not match the original participant count");
            return result;
        }
        for (const auto& peer : target.peers) {
            if (broadcast_manager_->workerAddresses()[peer.rank] != peer.address) {
                fail(result, "completion forget manager does not match the original participant addresses");
                return result;
            }
        }
    }
    std::vector<size_t> valid_indices;
    valid_indices.reserve(traces.size());
    for (size_t index = 0; index < traces.size(); ++index) {
        if (validTransferExecutionTrace(traces[index])) {
            valid_indices.push_back(index);
        } else {
            for (auto& row : result.entries[index].ranks) {
                row.result = Outcome::INVALID;
                row.error = "completion forget requires a nonempty execution trace";
            }
        }
    }
    if (!target.remote) {
        for (const auto index : valid_indices) {
            auto& row = result.entries[index].ranks.front();
            if (remainingTimeout(deadline) == 0) {
                row.error = "completion forget deadline exceeded";
                continue;
            }
            try {
                setOutcome(row, local_forget_(traces[index]));
            } catch (const std::exception& error) {
                row.error = error.what();
            } catch (...) {
                row.error = "local completion forget failed";
            }
        }
        const bool within_deadline = remainingTimeout(deadline) > 0;
        if (!within_deadline)
            result.error = "completion forget deadline exceeded";
        summarize(result, within_deadline);
        return result;
    }
    if (valid_indices.empty()) {
        summarize(result, true);
        return result;
    }
    try {
        std::vector<FunctionRequestPB> requests(target.peers.size());
        for (auto& request : requests) {
            for (const auto index : valid_indices)
                encodeKey(traces[index], *request.mutable_completion_forget()->add_keys());
        }
        const int submit_timeout = remainingTimeout(deadline);
        if (submit_timeout == 0) {
            fail(result, "completion forget deadline exceeded before submission");
            return result;
        }
        auto rpc = broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(
            requests, submit_timeout,
            [rpc_deadline](const auto& stub, const auto& context, const auto& request, auto* queue) {
                context->set_deadline(rpc_deadline);
                return stub->AsyncExecuteFunction(context.get(), request, queue);
            },
            std::nullopt, "completion_forget", /*fatal_on_error=*/false);
        if (!rpc) {
            fail(result, "completion forget could not be dispatched");
            return result;
        }
        const int wait_timeout = remainingTimeout(deadline);
        // Zero is an infinite wait in BroadcastResult; never pass it here.
        const bool completed = rpc->done() || (wait_timeout > 0 && rpc->waitDone(wait_timeout, false));
        if (!completed)
            result.error = "completion forget deadline exceeded";
        for (const auto& snapshot : rpc->rankSnapshots()) {
            const TransferCompletionForgetResponsePB* response = nullptr;
            std::string rank_error;
            if (!snapshot.cq_received || !snapshot.cq_ok || !snapshot.grpc_code)
                rank_error = snapshot.dispatch_error.empty() ? "completion forget has no successful CQ event" :
                                                               snapshot.dispatch_error;
            else if (*snapshot.grpc_code != grpc::StatusCode::OK)
                rank_error = snapshot.grpc_message.empty() ? "completion forget RPC failed" : snapshot.grpc_message;
            else if (!snapshot.response || !snapshot.response->has_completion_forget())
                rank_error = "missing completion forget response";
            else {
                response = &snapshot.response->completion_forget();
                if (response->worker_case() != TransferCompletionForgetResponsePB::kRank
                    || response->rank() != snapshot.rank)
                    rank_error = "completion forget worker rank mismatch";
                else if (static_cast<size_t>(response->entries_size()) != valid_indices.size())
                    rank_error = "completion forget key count mismatch";
            }
            for (size_t wire_index = 0; wire_index < valid_indices.size(); ++wire_index) {
                const auto index = valid_indices[wire_index];
                auto& row = result.entries[index].ranks.at(snapshot.rank);
                row.cq_received = snapshot.cq_received;
                row.cq_ok = snapshot.cq_ok;
                row.grpc_code = snapshot.grpc_code;
                if (!rank_error.empty())
                    row.error = rank_error;
                else
                    decodeEntry(response->entries(wire_index), traces[index], row);
            }
        }
        const bool within_deadline = completed && remainingTimeout(deadline) > 0;
        if (!within_deadline)
            result.error = "completion forget deadline exceeded";
        summarize(result, within_deadline);
    } catch (const std::exception& error) {
        fail(result, error.what());
    } catch (...) {
        fail(result, "completion forget failed");
    }
    return result;
}

}  // namespace rtp_llm
