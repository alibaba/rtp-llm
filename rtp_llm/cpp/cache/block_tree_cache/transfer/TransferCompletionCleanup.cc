#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionCleanup.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

bool sameTarget(const TransferCompletionTarget& first, const TransferCompletionTarget& second) noexcept {
    if (first.remote != second.remote || first.peers.size() != second.peers.size())
        return false;
    for (size_t rank = 0; rank < first.peers.size(); ++rank) {
        if (first.peers[rank].rank != second.peers[rank].rank
            || first.peers[rank].address != second.peers[rank].address)
            return false;
    }
    return true;
}

}  // namespace

void TransferCompletionCleanup::Prepared::submit() noexcept {
    if (node_.empty())
        return;
    auto owner = std::move(owner_);
    owner->submit(node_);
}

TransferCompletionCleanup::Prepared
TransferCompletionCleanup::prepare(TransferCompletionTarget target, TransferExecutionTrace trace_id) {
    if (!validTransferExecutionTrace(trace_id))
        throw std::invalid_argument("completion cleanup requires a nonempty execution trace");
    if (target.peers.empty() || (!target.remote && target.peers.size() != 1))
        throw std::invalid_argument("completion cleanup requires the original participants");
    std::sort(target.peers.begin(), target.peers.end(),
              [](const auto& first, const auto& second) { return first.rank < second.rank; });
    for (size_t rank = 0; rank < target.peers.size(); ++rank) {
        if (target.peers[rank].rank != rank || target.peers[rank].address.empty())
            throw std::invalid_argument("completion cleanup requires unique original ranks and addresses");
    }
    Prepared prepared;
    prepared.owner_ = shared_from_this();
    prepared.node_.push_back(Entry{std::move(target), std::move(trace_id)});
    return prepared;
}

size_t TransferCompletionCleanup::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return outstanding_;
}

void TransferCompletionCleanup::submit(std::list<Entry>& node) noexcept {
    size_t count = 0;
    size_t threshold = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        outstanding_ += node.size();
        pending_.splice(pending_.end(), node);
        if (next_warning_ != 0 && outstanding_ >= next_warning_) {
            count = outstanding_;
            threshold = next_warning_;
            do {
                next_warning_ = next_warning_ > std::numeric_limits<size_t>::max() / 2 ? 0 : next_warning_ * 2;
            } while (next_warning_ != 0 && outstanding_ >= next_warning_);
        }
    }
    if (threshold != 0) {
        try {
            RTP_LLM_LOG_WARNING("completion cleanup retains %zu entries (warning threshold=%zu); admission remains open",
                                count, threshold);
        } catch (...) {
            // Diagnostics cannot interrupt ownership transfer of a prepared node.
        }
    }
}

void TransferCompletionCleanup::requeue(std::list<Entry>& entries) noexcept {
    if (entries.empty())
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.splice(pending_.end(), entries);
}

void TransferCompletionCleanup::logFailure(
    const char* reason, const Entry* entry,
    const TransferCompletionForgetEntryResult* response) noexcept {
    try {
        size_t pending;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto now = std::chrono::steady_clock::now();
            if (now < next_failure_log_)
                return;
            next_failure_log_ = now + std::chrono::seconds(60);
            pending = outstanding_;
        }
        std::ostringstream context;
        if (entry) {
            context << " remote=" << entry->target.remote << " trace=";
            std::visit([&](const auto& trace) { context << trace; }, entry->trace_id);
            for (const auto& peer : entry->target.peers)
                context << " peer=" << peer.rank << "@" << peer.address;
        }
        if (response) {
            for (const auto& rank : response->ranks) {
                if (rank.result == TransferTaskRegistry::ForgetResult::DONE && rank.error.empty())
                    continue;
                context << " failed_rank=" << rank.rank << " address=" << rank.server_address
                        << " status=" << static_cast<int>(rank.result)
                        << " grpc_code=" << rank.grpc_code.value_or(-1)
                        << " error=" << rank.error;
                break;  // One representative failed rank keeps the diagnostic bounded.
            }
        }
        RTP_LLM_LOG_WARNING("completion cleanup failed: reason=%s pending=%zu%s", reason, pending,
                            context.str().c_str());
    } catch (...) {
        // Diagnostics must never prevent retry or change cleanup ownership.
    }
}

TransferCompletionCleanup::Summary
TransferCompletionCleanup::drain(const Forget& forget, int timeout_ms,
                                  const std::function<bool()>& should_stop) noexcept {
    Summary summary;
    if (!forget || timeout_ms <= 0) {
        summary.failed = true;
        logFailure("missing forget callback or nonpositive timeout");
        summary.pending = size();
        return summary;
    }
    const auto stopped = [&]() noexcept {
        if (should_stop) {
            try {
                if (!should_stop())
                    return false;
            } catch (const std::exception& error) {
                summary.failed = true;
                logFailure(error.what());
            } catch (...) {
                summary.failed = true;
                logFailure("stop predicate threw an unknown exception");
            }
            summary.stopped = true;
            return true;
        }
        return false;
    };
    if (stopped()) {
        summary.pending = size();
        return summary;
    }
    std::list<Entry> work;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        work.splice(work.end(), pending_);
    }
    std::list<Entry> batch;
    while (!work.empty()) {
        if (stopped())
            break;
        batch.splice(batch.end(), work, work.begin());
        const auto& target = batch.front().target;
        for (auto it = work.begin(); it != work.end() && batch.size() < kTransferCompletionForgetBatchSize;) {
            const auto current = it++;
            if (sameTarget(target, current->target))
                batch.splice(batch.end(), work, current);
        }
        try {
            std::vector<TransferExecutionTrace> traces;
            traces.reserve(batch.size());
            for (const auto& entry : batch)
                traces.push_back(entry.trace_id);
            if (stopped())
                break;
            summary.attempted += batch.size();
            const auto result = forget(target, traces, timeout_ms);
            // Prefer the per-entry rank error over a generic batch failure.
            if (!result.error.empty())
                summary.failed = true;
            if (result.entries.size() != batch.size()) {
                summary.failed = true;
                logFailure(result.error.empty() ? "forget acknowledgement count mismatch" : result.error.c_str(),
                           &batch.front());
            } else {
                size_t index = 0;
                size_t forgotten = 0;
                for (auto it = batch.begin(); it != batch.end(); ++index) {
                    const auto& response = result.entries[index];
                    if (response.trace_id == it->trace_id && response.all_done) {
                        it = batch.erase(it);
                        ++forgotten;
                    } else {
                        summary.failed = true;
                        logFailure(response.trace_id != it->trace_id ? "forget acknowledgement trace mismatch" :
                                   (result.error.empty() ? "forget entry not done" : result.error.c_str()),
                                   &*it, &response);
                        ++it;
                    }
                }
                if (forgotten != 0) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    outstanding_ -= forgotten;
                    summary.forgotten += forgotten;
                }
            }
            if (!result.error.empty())
                logFailure(result.error.c_str(), batch.empty() ? nullptr : &batch.front());
        } catch (const std::exception& error) {
            summary.failed = true;
            logFailure(error.what(), batch.empty() ? nullptr : &batch.front());
        } catch (...) {
            logFailure("forget threw an unknown exception", batch.empty() ? nullptr : &batch.front());
            // No acknowledgement means no deletion, even if the remote worker
            // applied the request before the exception or transport failure.
            summary.failed = true;
        }
        requeue(batch);
    }
    requeue(batch);
    requeue(work);
    summary.pending = size();
    return summary;
}

}  // namespace rtp_llm
