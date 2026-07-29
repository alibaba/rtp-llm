#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

constexpr size_t kStagingAlignment = 4096;

// Internal wait budget for a staging lease; not user-configurable.
constexpr std::chrono::milliseconds kStagingAcquireTimeout{1000};
constexpr std::chrono::milliseconds kInitialBackoff{1};
constexpr std::chrono::milliseconds kMaxBackoff{64};

size_t alignUp(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

std::chrono::milliseconds saturatingDouble(std::chrono::milliseconds delay, std::chrono::milliseconds max_backoff) {
    if (delay >= max_backoff / 2) {
        return max_backoff;
    }
    return std::min(delay * 2, max_backoff);
}

}  // namespace

DeviceDiskTransferExecutor::TransientHostStagingPool::TransientHostStagingPool(size_t block_count,
                                                                               size_t stride_bytes,
                                                                               bool   try_pin_memory):
    block_count_(block_count), stride_bytes_(stride_bytes) {
    RTP_LLM_CHECK_WITH_INFO(block_count_ > 0, "staging block_count must be > 0");
    RTP_LLM_CHECK_WITH_INFO(stride_bytes_ > 0, "staging stride_bytes must be > 0");
    RTP_LLM_CHECK_WITH_INFO(block_count_ <= (std::numeric_limits<size_t>::max() - kStagingAlignment) / stride_bytes_,
                            "staging backing size overflow: blocks=%zu stride=%zu",
                            block_count_,
                            stride_bytes_);

    const size_t total_bytes = block_count_ * stride_bytes_;
    RTP_LLM_CHECK_WITH_INFO(total_bytes + kStagingAlignment <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                            "staging backing exceeds int64 tensor dim: total_bytes=%zu",
                            total_bytes);
    auto cpu = torch::empty({static_cast<int64_t>(total_bytes + kStagingAlignment)},
                            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    if (try_pin_memory) {
        try {
            backing_ = cpu.pin_memory();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("pin staging memory failed, fallback to pageable CPU memory, error=%s", e.what());
        }
    }
    if (!backing_.defined() || !backing_.is_pinned()) {
        backing_ = cpu;
    }
    RTP_LLM_CHECK(backing_.defined() && backing_.is_contiguous());
    const auto raw_base = reinterpret_cast<uintptr_t>(backing_.data_ptr<uint8_t>());
    base_ptr_           = reinterpret_cast<uint8_t*>(alignUp(raw_base, kStagingAlignment));

    free_id_list_.reserve(block_count_);
    for (size_t block_id = 0; block_id < block_count_; ++block_id) {
        free_id_list_.push_back(block_id);
    }
    RTP_LLM_LOG_INFO("device-disk staging pool ready: blocks=%zu stride=%zu total_bytes=%zu pinned=%d",
                     block_count_,
                     stride_bytes_,
                     total_bytes,
                     static_cast<int>(backing_.is_pinned()));
}

std::optional<DeviceDiskTransferExecutor::TransientHostStagingPool::Lease>
DeviceDiskTransferExecutor::TransientHostStagingPool::tryAcquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_id_list_.empty()) {
        return std::nullopt;
    }
    const size_t block_id = free_id_list_.back();
    free_id_list_.pop_back();
    return Lease(this, block_id);
}

std::optional<DeviceDiskTransferExecutor::TransientHostStagingPool::Lease>
DeviceDiskTransferExecutor::TransientHostStagingPool::acquireWithBackoff(std::chrono::milliseconds timeout) {
    if (auto lease = tryAcquire(); lease.has_value()) {
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
        if (auto lease = tryAcquire(); lease.has_value()) {
            return lease;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            reportAcquireTimeout(wait_start);
            return std::nullopt;
        }

        delay = saturatingDouble(delay, kMaxBackoff);
    }
}

// Rate-limited to ~1 log/s.
void DeviceDiskTransferExecutor::TransientHostStagingPool::reportAcquireTimeout(
    std::chrono::steady_clock::time_point wait_start) {
    const auto        now       = std::chrono::steady_clock::now();
    const int64_t     waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - wait_start).count();
    const size_t      total     = timeout_count_.fetch_add(1) + 1;
    const int64_t     now_ns    = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    int64_t           last_ns   = last_timeout_log_ns_.load();
    constexpr int64_t kLogIntervalNs = 1'000'000'000;
    if (now_ns - last_ns < kLogIntervalNs || !last_timeout_log_ns_.compare_exchange_strong(last_ns, now_ns)) {
        return;
    }
    RTP_LLM_LOG_WARNING("device-disk staging acquire timed out after %ld ms: blocks=%zu total_timeouts=%zu; "
                        "consider raising staging block count or investigating disk latency",
                        waited_ms,
                        block_count_,
                        total);
}

void DeviceDiskTransferExecutor::TransientHostStagingPool::release(size_t block_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_id_list_.push_back(block_id);
}

HostBufferView DeviceDiskTransferExecutor::TransientHostStagingPool::viewFor(size_t block_id,
                                                                             size_t payload_bytes) const {
    RTP_LLM_CHECK(block_id < block_count_);
    RTP_LLM_CHECK_WITH_INFO(
        payload_bytes <= stride_bytes_, "staging payload %zu exceeds stride %zu", payload_bytes, stride_bytes_);
    void* base = base_ptr_ + block_id * stride_bytes_;
    return HostBufferView{base, payload_bytes, stride_bytes_};
}

DeviceDiskTransferExecutor::DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                                                       HostDiskTransferExecutor&       host_disk_executor,
                                                       const std::vector<GroupSetPtr>& group_sets,
                                                       size_t                          staging_block_count):
    device_host_executor_(device_host_executor), host_disk_executor_(host_disk_executor) {
    RTP_LLM_CHECK_WITH_INFO(staging_block_count > 0, "device_disk_staging_block_count must be > 0");
    size_t max_payload_bytes = 0;
    for (const auto& group_set : group_sets) {
        if (group_set != nullptr) {
            max_payload_bytes = std::max(max_payload_bytes, group_set->payloadBytes());
        }
    }
    RTP_LLM_CHECK_WITH_INFO(max_payload_bytes > 0, "device-disk staging requires a non-zero group payload");
    const size_t stride_bytes = alignUp(max_payload_bytes, kStagingAlignment);
    staging_pool_             = std::make_unique<TransientHostStagingPool>(staging_block_count, stride_bytes);
}

DeviceDiskTransferExecutor::~DeviceDiskTransferExecutor() = default;

TransferStatus DeviceDiskTransferExecutor::execute(const TransferDescriptor& desc, const GroupSet& group) {
    RTP_LLM_CHECK(staging_pool_ != nullptr);

    const bool device_to_disk = desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::DISK;
    const bool disk_to_device = desc.source_tier == Tier::DISK && desc.target_tier == Tier::DEVICE;
    if (!device_to_disk && !disk_to_device) {
        RTP_LLM_LOG_WARNING("device-disk executor received unsupported direction source=%s target=%s",
                            tierName(desc.source_tier),
                            tierName(desc.target_tier));
        return TransferStatus::INVALID_ARGS;
    }

    auto lease = staging_pool_->acquireWithBackoff(kStagingAcquireTimeout);
    if (!lease.has_value()) {
        return TransferStatus::RESOURCE_EXHAUSTED;
    }
    const HostBufferView host = lease->view(group.payloadBytes());

    if (device_to_disk) {
        const TransferStatus stage_one = device_host_executor_.deviceToHost(desc, group, host);
        if (stage_one != TransferStatus::OK) {
            return stage_one;
        }
        return host_disk_executor_.hostToDisk(host, desc, group);
    }

    const TransferStatus stage_one = host_disk_executor_.diskToHost(desc, group, host);
    if (stage_one != TransferStatus::OK) {
        return stage_one;
    }
    return device_host_executor_.hostToDevice(host, desc, group);
}

}  // namespace rtp_llm
