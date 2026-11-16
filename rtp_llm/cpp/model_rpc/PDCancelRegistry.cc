#include "rtp_llm/cpp/model_rpc/PDCancelRegistry.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/RpcCompletionQueue.h"
#include <chrono>
#include <limits>

namespace rtp_llm {

PDCancelRegistry::~PDCancelRegistry() {
    if (cancel_expiration_) {
        cancel_expiration_();
    }
}

void PDCancelRegistry::armExpirationLocked() {
    auto weak = weak_from_this();
    if (weak.expired()) {
        return;
    }
    const auto now  = currentTimeMs();
    auto       next = std::numeric_limits<int64_t>::max();
    if (!entry_expirations_.empty())
        next = entry_expirations_.begin()->first;
    if (!fence_expirations_.empty())
        next = std::min(next, fence_expirations_.begin()->first);
    if (!overdue_cancels_.empty())
        next = std::min(next, now + 30 * 1000);
    if (cancel_expiration_ && expiration_ms_ == next) {
        return;
    }
    if (cancel_expiration_) {
        cancel_expiration_();
        cancel_expiration_ = {};
    }
    const auto generation = ++expiration_generation_;
    if (next == std::numeric_limits<int64_t>::max()) {
        return;
    }
    expiration_ms_ = next;
    cancel_expiration_ =
        RpcCompletionQueue::instance().scheduleAfter(std::max<int64_t>(0, next - now), [weak, generation](bool ok) {
            auto registry = weak.lock();
            if (!ok || !registry)
                return;
            std::lock_guard<std::mutex> lock(registry->mutex_);
            if (generation != registry->expiration_generation_)
                return;
            registry->cancel_expiration_ = {};
            registry->trimLocked(currentTimeMs());
            registry->reportPressureLocked();
            registry->armExpirationLocked();
        });
}

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
    armExpirationLocked();
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
        auto expired                            = entry_expirations_.extract(entry_expirations_.begin());
        const auto& [expires_at_ms, request_id] = expired.value();
        auto it = entries_.find(request_id);
        if (it == entries_.end()) {
            continue;
        }
        const auto& e = it->second;
        if (e->retain_until_ms != expires_at_ms) {
            continue;
        }
        if (e->canceled.load() && !e->terminal) {
            overdue_cancels_.insert(std::move(expired));
            continue;
        }
        if (!e->local_done || e->stream) {
            continue;
        }
        if (active_reads_.count(e->unique_key)) {
            // An outstanding StartLoad still needs its request ID to resolve
            // a late cancellation. Retry after this bounded sweep.
            scheduleExpirationLocked(e, now_ms + kActiveReadExpiryRetryMs);
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
        if (e->control_only) {
            --control_record_count_;
        }
        entries_.erase(it);
    }
    if (expired_remote_retries != 0) {
        RTP_LLM_LOG_WARNING("PD cancel retry retention expired after local physical cleanup, unconfirmed_peers=%zu",
                            expired_remote_retries);
    }
}

void PDCancelRegistry::reportPressureLocked() {
    if (overdue_cancels_.empty() && request_rejections_ == 0 && control_rejections_ == 0) {
        return;
    }
    const auto now_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    if (last_pressure_log_ms_ != 0 && now_ms - last_pressure_log_ms_ < 30000) {
        return;
    }
    // Each overdue index key is the first cancellation time plus kRetentionMs.
    const int64_t oldest_cancel_age_ms =
        overdue_cancels_.empty() ?
            0 :
            kRetentionMs + std::max<int64_t>(0, currentTimeMs() - overdue_cancels_.begin()->first);
    RTP_LLM_LOG_WARNING(
        "PD cancellation registry pressure: request_records=%zu request_capacity=%zu control_records=%zu "
        "control_capacity=%zu request_rejections=%zu control_rejections=%zu overdue_cancels=%zu "
        "oldest_overdue_cancel_age_ms=%ld",
        entries_.size() - control_record_count_,
        record_capacity_,
        control_record_count_ + absent_fences_.size(),
        control_capacity_,
        request_rejections_,
        control_rejections_,
        overdue_cancels_.size(),
        oldest_cancel_age_ms);
    last_pressure_log_ms_ = now_ms;
    request_rejections_   = 0;
    control_rejections_   = 0;
}

ErrorInfo PDCancelRegistry::admit(const GenerateInputPB& request,
                                  const std::string&     key,
                                  const std::string&     downstream_address,
                                  int64_t                deadline_ms,
                                  Handle&                handle,
                                  bool                   control_only) {
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
    const bool capacity_exhausted = control_only ? control_record_count_ + absent_fences_.size() >= control_capacity_ :
                                                   entries_.size() - control_record_count_ >= record_capacity_;
    if (capacity_exhausted) {
        ++(control_only ? control_rejections_ : request_rejections_);
        reportPressureLocked();
        return {ErrorCode::CONCURRENCY_LIMIT_ERROR,
                control_only ? "PD early cancellation capacity exhausted" : "PD request registry capacity exhausted"};
    }
    handle = std::make_shared<Entry>(
        TaskIdentity{request.request_id(), request.has_group_id() ? request.group_id().value() : -1});
    handle->unique_key         = key;
    handle->downstream_address = downstream_address;
    handle->deadline_ms        = deadline_ms;
    handle->control_only       = control_only;
    handle->downstream_done    = downstream_address.empty();
    entries_.emplace(request.request_id(), handle);
    if (control_only) {
        ++control_record_count_;
    }
    return ErrorInfo::OkStatus();
}

PDCancelRegistry::Handle PDCancelRegistry::find(int64_t request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        found = entries_.find(request_id);
    return found == entries_.end() ? nullptr : found->second;
}

void PDCancelRegistry::attach(const Handle& handle, const GenerateStreamPtr& stream) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        handle->stream = stream;
        if (stream && handle->canceled.load()) {
            stream->reportError(handle->cancel_reason.code(), handle->cancel_reason.ToString());
        }
    }
    auto registry = weak_from_this();
    if (stream && !registry.expired()) {
        stream->onLocalCleanupProgress([registry, weak_handle = std::weak_ptr<Entry>(handle)] {
            auto tracker = registry.lock();
            auto entry   = weak_handle.lock();
            if (tracker && entry)
                tracker->scheduleLocalProgress(entry);
        });
    }
}

void PDCancelRegistry::scheduleLocalProgress(const Handle& handle) {
    auto weak = weak_from_this();
    if (weak.expired() || handle->local_event_queued.exchange(true)) {
        return;
    }
    if (!RpcCompletionQueue::instance().scheduleAfter(0, [weak, entry = std::weak_ptr<Entry>(handle)](bool ok) {
            auto registry = weak.lock();
            auto handle   = entry.lock();
            if (!registry || !handle)
                return;
            handle->local_event_queued.store(false);
            if (!ok || !handle->local_done)
                return;
            if (handle->canceled.load()) {
                registry->complete(handle);
            } else {
                registry->releaseFinishedStream(handle);
            }
        })) {
        handle->local_event_queued.store(false);
    }
}

void PDCancelRegistry::finishLocal(const Handle& handle) {
    GenerateStreamPtr stream;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stream = handle->stream;
    }
    const bool cleanup_complete = stream && stream->tryCompleteLocalCleanup();
    std::lock_guard<std::mutex> lock(mutex_);
    handle->local_done      = true;
    if (!handle->canceled.load()) {
        const auto now_ms = currentTimeMs();
        scheduleExpirationLocked(
            handle, (handle->downstream_done ? now_ms : std::max(now_ms, handle->deadline_ms)) + kRetentionMs);
    }
    if (handle->canceled.load() || !stream || handle->stream != stream) {
        if (handle->canceled.load())
            scheduleLocalProgress(handle);
        return;
    }
    const auto request_id = handle->identity.request_id;
    const auto it = entries_.find(request_id);
    if (it == entries_.end() || it->second != handle) {
        return;
    }
    if (cleanup_complete) {
        handle->stream.reset();
    } else {
        // A stream event may already have run while RPC teardown was in progress.
        scheduleLocalProgress(handle);
    }
}

absl::StatusOr<CancelStatusPB> PDCancelRegistry::cancel(int64_t                                   request_id,
                                                        const ErrorInfo&                          reason,
                                                        const std::function<void(const Handle&)>& fence_local_prefill) {
    std::lock_guard<std::mutex> lock(mutex_);
    trimLocked(currentTimeMs());
    auto found = entries_.find(request_id);
    if (found == entries_.end()) {
        if (absent_fences_.count(request_id)) {
            return CANCEL_STATUS_TOMBSTONED;
        }
        if (control_record_count_ + absent_fences_.size() >= control_capacity_) {
            ++control_rejections_;
            reportPressureLocked();
            return absl::ResourceExhaustedError("PD early cancellation capacity exhausted");
        }
        const auto expires_at_ms = currentTimeMs() + kRetentionMs;
        const bool inserted = absent_fences_.emplace(request_id, CancelFence{expires_at_ms, reason}).second;
        if (inserted) {
            fence_expirations_.emplace(expires_at_ms, request_id);
            armExpirationLocked();
        }
        return CANCEL_STATUS_TOMBSTONED;
    }
    const auto& e = found->second;
    if (fence_local_prefill) {
        if (e->terminal.load()) {
            return CANCEL_STATUS_NOT_FOUND;
        }
        fence_local_prefill(e);
    }
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
        && (!e->stream || !e->stream->hasPendingLocalCleanup()))
        return CANCEL_STATUS_NOT_FOUND;
    const bool priority_preemption = reason.code() == ErrorCode::PRIORITY_PREEMPTED;
    // WorkerStatus may already have published completion while RPC cleanup is still running.
    // Preserve that outcome for preemption while allowing ordinary resource cleanup.
    if (!meta_->markCancellationPending(e->identity, priority_preemption, priority_preemption)) {
        return CANCEL_STATUS_NOT_FOUND;
    }
    e->cancel_reason = reason;
    // Preserve the first local failure if it preceded the cleanup request.
    if (prior_error.hasError())
        e->cancel_reason = prior_error;
    e->canceled.store(true);
    // The first scheduled deadline observes cleanup age. Only complete()
    // can start retirement for a canceled record.
    scheduleExpirationLocked(e, currentTimeMs() + kRetentionMs);
    ++canceled_keys_[e->unique_key];
    if (e->stream)
        e->stream->reportError(e->cancel_reason.code(), e->cancel_reason.ToString());
    scheduleLocalProgress(e);
    if (cancel_started_ && !e->downstream_done) {
        cancel_started_(e);
    }
    return CANCEL_STATUS_ACCEPTED;
}

bool PDCancelRegistry::isCanceled(const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    return canceled_keys_.count(unique_key) != 0;
}

void PDCancelRegistry::releaseFinishedStream(const Handle& handle) {
    const auto        request_id = handle->identity.request_id;
    GenerateStreamPtr stream;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = entries_.find(request_id);
        if (it == entries_.end() || it->second != handle || !handle->local_done || handle->canceled.load()) {
            return;
        }
        stream = handle->stream;
    }
    if (!stream || !stream->tryCompleteLocalCleanup()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = entries_.find(request_id);
    if (it != entries_.end() && it->second == handle && handle->stream == stream && !handle->canceled.load()) {
        // Restore the expiry index if retention elapsed while release was pending.
        entry_expirations_.emplace(handle->retain_until_ms, request_id);
        handle->stream.reset();
        armExpirationLocked();
    }
}

void PDCancelRegistry::finishDownstream(const Handle& handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  current = entries_.find(handle->identity.request_id);
    if (current == entries_.end() || current->second != handle || handle->downstream_done) {
        return;
    }
    handle->downstream_done = true;
    if (handle->local_done && (!handle->canceled.load() || handle->terminal)) {
        scheduleExpirationLocked(handle, currentTimeMs() + kRetentionMs);
    }
}

bool PDCancelRegistry::needsDownstreamCancel(const Handle& handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  current = entries_.find(handle->identity.request_id);
    return current != entries_.end() && current->second == handle && handle->canceled.load()
           && !handle->downstream_done;
}

void PDCancelRegistry::beginRead(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++active_reads_[key];
}

void PDCancelRegistry::endRead(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        found = active_reads_.find(key);
    if (found != active_reads_.end() && --found->second == 0) {
        active_reads_.erase(found);
        auto waiting = waiting_on_reads_.extract(key);
        if (!waiting.empty()) {
            for (auto request_id : waiting.mapped()) {
                auto entry = entries_.find(request_id);
                if (entry != entries_.end())
                    scheduleLocalProgress(entry->second);
            }
        }
    }
}

bool PDCancelRegistry::complete(const Handle& handle) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (handle->terminal || !handle->canceled.load() || !handle->local_done)
        return false;
    if (active_reads_.count(handle->unique_key)) {
        waiting_on_reads_[handle->unique_key].insert(handle->identity.request_id);
        return false;
    }
    auto stream = handle->stream;
    // Resource cleanup may wait for transport callbacks; do not block unrelated
    // admission/Cancel operations under the registry lock.
    lock.unlock();
    if (stream && !stream->tryCompleteLocalCleanup())
        return false;
    lock.lock();
    if (handle->terminal || handle->stream != stream)
        return false;
    if (active_reads_.count(handle->unique_key)) {
        waiting_on_reads_[handle->unique_key].insert(handle->identity.request_id);
        return false;
    }
    if (!meta_->markCancellationComplete(handle->identity.request_id,
                                         static_cast<int64_t>(handle->cancel_reason.code()),
                                         handle->cancel_reason.ToString(),
                                         stream))
        return false;
    handle->terminal = true;
    overdue_cancels_.erase({handle->retain_until_ms, handle->identity.request_id});
    // Local terminal is independent of remote control-plane availability. Keep
    // the cancellation fence and bounded retry record independently of Prefill
    // availability. Retention expiry only retires bookkeeping after this local proof.
    handle->stream.reset();
    scheduleExpirationLocked(handle, currentTimeMs() + kRetentionMs);
    return true;
}

}  // namespace rtp_llm
