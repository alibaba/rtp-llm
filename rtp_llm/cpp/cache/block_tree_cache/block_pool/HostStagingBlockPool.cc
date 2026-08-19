#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

constexpr std::chrono::milliseconds kInitialBackoff{1};
constexpr std::chrono::milliseconds kMaxBackoff{64};

}  // namespace

HostStagingBlockPool::HostStagingBlockPool(size_t block_count, size_t stride_bytes, bool try_pin_memory):
    block_count_(block_count),
    stride_bytes_(stride_bytes),
    backing_(block_count_ * stride_bytes_, kAlignment, try_pin_memory, "host staging block pool") {
    const size_t total_bytes = block_count_ * stride_bytes_;
    free_id_list_.reserve(block_count_);
    for (size_t block_id = 0; block_id < block_count_; ++block_id) {
        free_id_list_.push_back(block_id);
    }
    RTP_LLM_LOG_INFO("host staging block pool ready: blocks=%zu stride=%zu total_bytes=%zu pinned=%d",
                     block_count_,
                     stride_bytes_,
                     total_bytes,
                     static_cast<int>(backing_.isPinned()));
}

std::optional<HostStagingBlockPool::HostStagingBlockLease> HostStagingBlockPool::malloc() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_id_list_.empty()) {
        return std::nullopt;
    }
    const size_t block_id = free_id_list_.back();
    free_id_list_.pop_back();
    return HostStagingBlockLease(this, block_id);
}

std::optional<HostStagingBlockPool::HostStagingBlockLease>
HostStagingBlockPool::mallocWithBackoff(std::chrono::milliseconds timeout) {
    if (auto lease = malloc(); lease.has_value()) {
        return lease;
    }

    const auto wait_start = std::chrono::steady_clock::now();
    const auto deadline   = wait_start + timeout;
    auto       delay      = kInitialBackoff;

    while (true) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            reportAcquireTimeout(wait_start);
            return std::nullopt;
        }

        const auto remaining = deadline - now;
        const auto sleep_duration =
            std::min(std::chrono::duration_cast<std::chrono::steady_clock::duration>(delay), remaining);
        std::this_thread::sleep_for(sleep_duration);

        // Final non-blocking try at the deadline; never start a new sleep past it.
        if (auto lease = malloc(); lease.has_value()) {
            return lease;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            reportAcquireTimeout(wait_start);
            return std::nullopt;
        }

        if (delay >= kMaxBackoff / 2) {
            delay = kMaxBackoff;
        } else {
            delay = std::min(delay * 2, kMaxBackoff);
        }
    }
}

// Rate-limited to ~1 log/s.
void HostStagingBlockPool::reportAcquireTimeout(std::chrono::steady_clock::time_point wait_start) {
    const auto        now       = std::chrono::steady_clock::now();
    const int64_t     waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - wait_start).count();
    const size_t      total     = timeout_count_.fetch_add(1) + 1;
    const int64_t     now_ns    = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    int64_t           last_ns   = last_timeout_log_ns_.load();
    constexpr int64_t kLogIntervalNs = 1'000'000'000;
    if (now_ns - last_ns < kLogIntervalNs || !last_timeout_log_ns_.compare_exchange_strong(last_ns, now_ns)) {
        return;
    }
    RTP_LLM_LOG_WARNING("host staging acquire timed out after %ld ms: blocks=%zu total_timeouts=%zu; "
                        "consider raising staging block count or investigating disk latency",
                        waited_ms,
                        block_count_,
                        total);
}

void HostStagingBlockPool::free(size_t block_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_id_list_.push_back(block_id);
}

HostBufferView HostStagingBlockPool::blockBuffer(size_t block_id, size_t payload_bytes) const {
    void* base = backing_.data() + block_id * stride_bytes_;
    return HostBufferView{base, payload_bytes, stride_bytes_};
}

}  // namespace rtp_llm
