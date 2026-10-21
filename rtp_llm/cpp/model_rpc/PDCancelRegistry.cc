#include "rtp_llm/cpp/model_rpc/PDCancelRegistry.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void PDCancelRegistry::scheduleExpirationLocked(const Handle& handle, int64_t expires_at_ms) {
    const auto current = entries_.find(handle->identity.request_id);
    if (current == entries_.end() || current->second != handle) {
        return;
    }
    // Keep one scheduled expiration per record, even when downstream completion
    // shortens an earlier deadline-based retention period.
    entry_expirations_.erase({handle->retain_until_ms, handle->identity.request_id});
    handle->retain_until_ms = expires_at_ms;
    entry_expirations_.emplace(expires_at_ms, handle->identity.request_id);
}

void PDCancelRegistry::trimLocked(int64_t now_ms) {
    size_t expired_remote_retries = 0;
    size_t fences_trimmed         = 0;
    while (!fence_expirations_.empty() && fence_expirations_.begin()->first <= now_ms
           && fences_trimmed < kMaxExpirationsPerPass) {
        ++fences_trimmed;
        const auto [expires_at_ms, request_id] = *fence_expirations_.begin();
        fence_expirations_.erase(fence_expirations_.begin());
        auto it = absent_fences_.find(request_id);
        if (it != absent_fences_.end() && it->second.expires_at_ms == expires_at_ms) {
            absent_fences_.erase(it);
        }
    }
    size_t entries_trimmed = 0;
    while (!entry_expirations_.empty() && entry_expirations_.begin()->first <= now_ms
           && entries_trimmed < kMaxExpirationsPerPass) {
        ++entries_trimmed;
        const auto [expires_at_ms, request_id] = *entry_expirations_.begin();
        entry_expirations_.erase(entry_expirations_.begin());
        auto it = entries_.find(request_id);
        if (it == entries_.end()) {
            continue;
        }
        const auto& e = it->second;
        if (!e->local_done || (e->canceled.load() && !e->terminal) || e->retain_until_ms != expires_at_ms) {
            continue;
        }
        if (e->canceled.load()) {
            if (!e->downstream_done) {
                // complete() already proved local physical cleanup. Retiring this
                // lightweight retry record must never masquerade as a remote ACK.
                ++expired_remote_retries;
            }
            auto key = canceled_keys_.find(e->unique_key);
            if (key != canceled_keys_.end() && --key->second == 0) {
                canceled_keys_.erase(key);
            }
        }
        pending_cancels_.erase(request_id);
        entries_.erase(it);
    }
    if (expired_remote_retries != 0) {
        RTP_LLM_LOG_WARNING("PD cancel retry retention expired after local physical cleanup, unconfirmed_peers=%zu",
                            expired_remote_retries);
    }
}

ErrorInfo PDCancelRegistry::admit(const GenerateInputPB& request,
                                  const std::string&     key,
                                  const std::string&     downstream_address,
                                  int64_t                deadline_ms,
                                  Handle&                handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    trimLocked(currentTimeMs());
    auto existing = entries_.find(request.request_id());
    auto fence    = absent_fences_.find(request.request_id());
    if (fence != absent_fences_.end())
        return fence->second.reason;
    if (existing != entries_.end() && existing->second->canceled.load())
        return existing->second->cancel_reason;
    if (existing != entries_.end()) {
        return {ErrorCode::INVALID_PARAMS, "duplicate PD request_id"};
    }
    if (entries_.size() + absent_fences_.size() >= record_capacity_) {
        return {ErrorCode::CONCURRENCY_LIMIT_ERROR, "PD cancellation registry capacity exhausted"};
    }
    handle = std::make_shared<Entry>(
        TaskIdentity{request.request_id(), request.has_group_id() ? request.group_id().value() : -1});
    handle->unique_key         = key;
    handle->downstream_address = downstream_address;
    handle->deadline_ms        = deadline_ms;
    handle->downstream_done    = downstream_address.empty();
    entries_.emplace(request.request_id(), handle);
    return ErrorInfo::OkStatus();
}

PDCancelRegistry::Handle PDCancelRegistry::find(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        found = entries_.find(request_id);
    return found == entries_.end() ? nullptr : found->second;
}

void PDCancelRegistry::attach(const Handle& handle, const GenerateStreamPtr& stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    handle->stream = stream;
    if (stream && handle->canceled.load()) {
        stream->reportError(handle->cancel_reason.code(), handle->cancel_reason.ToString());
    }
}

void PDCancelRegistry::finishLocal(const Handle& handle) {
    GenerateStreamPtr stream;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stream = handle->stream;
    }
    if (stream && stream->getStatus() == StreamState::FINISHED && !stream->isDeferredReleasePending()) {
        // A scheduler's FINISHED transition can precede its resource cleanup.
        // Complete the idempotent release before recording local completion.
        stream->releaseResource();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    handle->local_done      = true;
    const auto now_ms = currentTimeMs();
    scheduleExpirationLocked(handle,
                             (handle->downstream_done ? now_ms : std::max(now_ms, handle->deadline_ms)) + kRetentionMs);
    if (!handle->canceled.load() && stream && stream->getStatus() == StreamState::FINISHED
        && !stream->isDeferredReleasePending() && !stream->hasPendingP2PResourceHold())
        handle->stream.reset();
}

absl::StatusOr<CancelStatusPB> PDCancelRegistry::cancel(int64_t request_id, const ErrorInfo& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    trimLocked(currentTimeMs());
    auto found = entries_.find(request_id);
    if (found == entries_.end()) {
        if (absent_fences_.count(request_id)) {
            return CANCEL_STATUS_TOMBSTONED;
        }
        if (entries_.size() + absent_fences_.size() >= record_capacity_) {
            return absl::ResourceExhaustedError("PD cancellation registry capacity exhausted");
        }
        const auto expires_at_ms = currentTimeMs() + kRetentionMs;
        const bool inserted = absent_fences_.emplace(request_id, CancelFence{expires_at_ms, reason}).second;
        if (inserted) {
            fence_expirations_.emplace(expires_at_ms, request_id);
        }
        return CANCEL_STATUS_TOMBSTONED;
    }
    const auto& e = found->second;
    const auto prior_error = e->stream ? e->stream->firstError().error : ErrorInfo::OkStatus();
    if (reason.code() == ErrorCode::PRIORITY_PREEMPTED
        && ((e->canceled.load() && e->cancel_reason.code() != ErrorCode::PRIORITY_PREEMPTED)
            || (!e->canceled.load() && prior_error.hasError()
                && prior_error.code() != ErrorCode::PRIORITY_PREEMPTED))) {
        // An earlier outcome already won. ACCEPTED would promise a typed
        // preemption terminal that this record cannot publish; let FlexLB
        // reconcile the original worker outcome through its NOT_FOUND path.
        return CANCEL_STATUS_NOT_FOUND;
    }
    if (e->canceled.load())
        return CANCEL_STATUS_ACCEPTED;
    if (e->local_done && e->downstream_done && !active_reads_.count(e->unique_key)
        && (!e->stream
            || (e->stream->getStatus() == StreamState::FINISHED && !e->stream->isDeferredReleasePending()
                && !e->stream->hasPendingP2PResourceHold())))
        return CANCEL_STATUS_NOT_FOUND;
    e->cancel_reason = reason;
    // Preserve the first local failure if it preceded the cleanup request.
    if (prior_error.hasError())
        e->cancel_reason = prior_error;
    e->canceled.store(true);
    ++canceled_keys_[e->unique_key];
    pending_cancels_.emplace(request_id, e);
    meta_->markCancellationPending(e->identity, e->cancel_reason.code() == ErrorCode::PRIORITY_PREEMPTED);
    if (e->stream)
        e->stream->reportError(e->cancel_reason.code(), e->cancel_reason.ToString());
    return CANCEL_STATUS_ACCEPTED;
}

bool PDCancelRegistry::isCanceled(const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    return canceled_keys_.count(unique_key) != 0;
}

std::vector<PDCancelRegistry::Handle> PDCancelRegistry::pending() {
    std::lock_guard<std::mutex> lock(mutex_);
    trimLocked(currentTimeMs());
    std::vector<Handle> result;
    result.reserve(pending_cancels_.size());
    for (const auto& [id, e] : pending_cancels_) {
        result.push_back(e);
    }
    return result;
}

void PDCancelRegistry::finishDownstream(const Handle& handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  current = entries_.find(handle->identity.request_id);
    if (current == entries_.end() || current->second != handle || handle->downstream_done) {
        return;
    }
    handle->downstream_done = true;
    if (handle->terminal) {
        pending_cancels_.erase(handle->identity.request_id);
    }
    if (handle->local_done) {
        scheduleExpirationLocked(handle, currentTimeMs() + kRetentionMs);
    }
}

void PDCancelRegistry::beginRead(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++active_reads_[key];
}

void PDCancelRegistry::endRead(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        found = active_reads_.find(key);
    if (found != active_reads_.end() && --found->second == 0)
        active_reads_.erase(found);
}

bool PDCancelRegistry::complete(const Handle& handle) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (handle->terminal || !handle->canceled.load() || !handle->local_done)
        return false;
    if (active_reads_.count(handle->unique_key))
        return false;
    auto stream = handle->stream;
    if (stream && (stream->getStatus() != StreamState::FINISHED || stream->isDeferredReleasePending()))
        return false;
    // Resource cleanup may wait for transport callbacks; do not block unrelated
    // admission/Cancel operations under the registry lock.
    lock.unlock();
    if (stream) {
        stream->releaseResource();
        if (stream->hasPendingP2PResourceHold())
            return false;
    }
    lock.lock();
    if (handle->terminal || active_reads_.count(handle->unique_key))
        return false;
    if (!meta_->markCancellationComplete(handle->identity.request_id,
                                         static_cast<int64_t>(handle->cancel_reason.code()),
                                         handle->cancel_reason.ToString(),
                                         stream))
        return false;
    handle->terminal = true;
    // Local terminal is independent of remote control-plane availability. Keep
    // the cancellation fence and bounded retry record independently of Prefill
    // availability. Retention expiry only retires bookkeeping after this local proof.
    if (handle->downstream_done) {
        pending_cancels_.erase(handle->identity.request_id);
    }
    handle->stream.reset();
    scheduleExpirationLocked(handle, currentTimeMs() + kRetentionMs);
    return true;
}

}  // namespace rtp_llm
