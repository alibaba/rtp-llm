#include "rtp_llm/cpp/model_rpc/PDCancelRegistry.h"

namespace rtp_llm {

void PDCancelRegistry::trimLocked(int64_t now_ms) {
    for (auto it = absent_fences_.begin(); it != absent_fences_.end();) {
        if (it->second.expires_at_ms <= now_ms)
            it = absent_fences_.erase(it);
        else
            ++it;
    }
    for (auto it = entries_.begin(); it != entries_.end();) {
        const auto& e = it->second;
        if (e->local_done && (!e->canceled.load() || e->terminal) && e->retain_until_ms <= now_ms)
            it = entries_.erase(it);
        else
            ++it;
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
    handle->retain_until_ms = std::max(currentTimeMs(), handle->deadline_ms) + kRetentionMs;
    if (!handle->canceled.load() && stream && stream->getStatus() == StreamState::FINISHED
        && !stream->isDeferredReleasePending() && !stream->hasPendingP2PResourceHold())
        handle->stream.reset();
}

CancelStatusPB PDCancelRegistry::cancel(int64_t request_id, const ErrorInfo& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    trimLocked(currentTimeMs());
    auto found = entries_.find(request_id);
    if (found == entries_.end()) {
        absent_fences_.emplace(request_id, CancelFence{currentTimeMs() + kRetentionMs, reason});
        return CANCEL_STATUS_TOMBSTONED;
    }
    const auto& e = found->second;
    if (e->canceled.load())
        return CANCEL_STATUS_ACCEPTED;
    if (e->local_done && e->downstream_done && !active_reads_.count(e->unique_key)
        && (!e->stream
            || (e->stream->getStatus() == StreamState::FINISHED && !e->stream->isDeferredReleasePending()
                && !e->stream->hasPendingP2PResourceHold())))
        return CANCEL_STATUS_NOT_FOUND;
    e->cancel_reason = reason;
    // Preserve the first local failure if it preceded the cleanup request.
    if (e->stream && e->stream->firstError().error.hasError())
        e->cancel_reason = e->stream->firstError().error;
    e->canceled.store(true);
    meta_->markCancellationPending(e->identity, e->cancel_reason.code() == ErrorCode::PRIORITY_PREEMPTED);
    if (e->stream)
        e->stream->reportError(e->cancel_reason.code(), e->cancel_reason.ToString());
    return CANCEL_STATUS_ACCEPTED;
}

bool PDCancelRegistry::isCanceled(const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [id, e] : entries_) {
        if (e->unique_key == unique_key)
            return e->canceled.load();
    }
    return false;
}

std::vector<PDCancelRegistry::Handle> PDCancelRegistry::pending() {
    std::lock_guard<std::mutex> lock(mutex_);
    trimLocked(currentTimeMs());
    std::vector<Handle> result;
    for (const auto& [id, e] : entries_) {
        if (e->canceled.load() && !e->terminal)
            result.push_back(e);
    }
    return result;
}

void PDCancelRegistry::finishDownstream(const Handle& handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    handle->downstream_done = true;
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
    if (handle->terminal || !handle->canceled.load() || !handle->local_done || !handle->downstream_done)
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
    if (!meta_->markCancellationComplete(
            handle->identity.request_id, handle->cancel_reason.code(), handle->cancel_reason.ToString(), stream))
        return false;
    handle->terminal = true;
    handle->stream.reset();
    handle->retain_until_ms = currentTimeMs() + kRetentionMs;
    return true;
}

}  // namespace rtp_llm
