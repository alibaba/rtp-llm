#include "rtp_llm/cpp/engine_base/sleep/DrainManager.h"

#include <algorithm>
#include <chrono>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

void DrainManager::registerCounter(const std::string& name, CounterFn fn, CounterKind kind) {
    if (!fn) {
        RTP_LLM_LOG_WARNING("drain manager: reject null counter provider [%s]", name.c_str());
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        counters_[name] = CounterEntry{std::move(fn), kind};
    }
    RTP_LLM_LOG_INFO("drain manager: registered counter [%s] kind=%s",
                     name.c_str(),
                     kind == CounterKind::REQUEST ? "REQUEST" : "CACHE_TRANSFER");
}

void DrainManager::unregisterCounter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    counters_.erase(name);
}

void DrainManager::setCancelCallback(CancelFn fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    cancel_callback_ = std::move(fn);
}

std::vector<std::pair<std::string, DrainManager::CounterEntry>> DrainManager::snapshotCounters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {counters_.begin(), counters_.end()};
}

bool DrainManager::drained() const {
    for (const auto& [name, entry] : snapshotCounters()) {
        if (entry.fn() != 0) {
            return false;
        }
    }
    return true;
}

std::string DrainManager::pendingCountersDebugString() const {
    std::string result;
    for (const auto& [name, entry] : snapshotCounters()) {
        const size_t value = entry.fn();
        if (value != 0) {
            if (!result.empty()) {
                result += ", ";
            }
            result += name + "=" + std::to_string(value);
        }
    }
    return result;
}

bool DrainManager::waitDrained(int64_t timeout_ms) {
    if (drained()) {
        return true;
    }
    if (timeout_ms <= 0) {
        RTP_LLM_LOG_WARNING("drain manager: not drained on immediate check (timeout_ms=%ld), pending: [%s]",
                            timeout_ms,
                            pendingCountersDebugString().c_str());
        return false;
    }

    int64_t poll_interval_ms = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        poll_interval_ms = poll_interval_ms_;
    }

    const auto deadline      = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    auto       next_log_time = std::chrono::steady_clock::now();
    while (true) {
        if (drained()) {
            RTP_LLM_LOG_INFO("drain manager: drained");
            return true;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            RTP_LLM_LOG_WARNING("drain manager: drain timed out after %ld ms, pending: [%s]",
                                timeout_ms,
                                pendingCountersDebugString().c_str());
            return false;
        }
        if (now >= next_log_time) {
            RTP_LLM_LOG_INFO("drain manager: waiting for drain, pending: [%s]", pendingCountersDebugString().c_str());
            next_log_time = now + std::chrono::seconds(1);
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        const auto wait_span =
            std::min<std::chrono::milliseconds>(remaining, std::chrono::milliseconds(poll_interval_ms));
        // Correctness is by polling, not by the condvar: the drain counters live in
        // external atomics that are decremented off this mutex, so notifyDrainProgress()
        // is a best-effort early wakeup, not a lossless signal. A missed notify only
        // delays observation until this bounded wait_for elapses and the loop re-reads
        // drained() -- so wait_for MUST stay bounded by poll_interval_ms.
        std::unique_lock<std::mutex> lock(wait_mutex_);
        wait_cv_.wait_for(lock, wait_span);
    }
}

bool DrainManager::drain(const SleepOptions& opt) {
    const bool abort = opt.mode == "abort";
    RTP_LLM_LOG_INFO("drain manager: start drain, mode=%s abort=%d timeout_ms=%ld",
                     opt.mode.c_str(),
                     static_cast<int>(abort),
                     opt.timeout_ms);
    if (abort) {
        forceCancel();
    }
    return waitDrained(opt.timeout_ms);
}

void DrainManager::forceCancel() {
    CancelFn callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        callback = cancel_callback_;
    }
    if (!callback) {
        RTP_LLM_LOG_WARNING("drain manager: abort drain requested but no cancel callback injected");
        return;
    }
    RTP_LLM_LOG_INFO("drain manager: invoking abort callback (streaming requests are exempted by provider)");
    callback();
    notifyDrainProgress();
}

int64_t DrainManager::sumByKind(CounterKind kind) const {
    int64_t total = 0;
    for (const auto& [name, entry] : snapshotCounters()) {
        if (entry.kind == kind) {
            total += static_cast<int64_t>(entry.fn());
        }
    }
    return total;
}

int64_t DrainManager::activeRequestCount() const {
    return sumByKind(CounterKind::REQUEST);
}

int64_t DrainManager::activeCacheTransferCount() const {
    return sumByKind(CounterKind::CACHE_TRANSFER);
}

void DrainManager::installHooks(SleepHooks& hooks) {
    hooks.drain                    = [this](const SleepOptions& opt) { return drain(opt); };
    hooks.activeRequestCount       = [this]() { return activeRequestCount(); };
    hooks.activeCacheTransferCount = [this]() { return activeCacheTransferCount(); };
}

void DrainManager::notifyDrainProgress() {
    // Best-effort early wakeup only. The counters this unblocks are not guarded by
    // wait_mutex_, so this notify carries no lossless-wakeup guarantee; waitDrained()
    // guarantees progress by re-polling drained() after each bounded wait_for.
    wait_cv_.notify_all();
}

void DrainManager::setPollIntervalMs(int64_t interval_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    poll_interval_ms_ = std::max<int64_t>(1, interval_ms);
}

}  // namespace rtp_llm
