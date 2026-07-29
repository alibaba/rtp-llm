#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <utility>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

constexpr size_t kStagingAlignment = 4096;

size_t alignUp(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace

// Fixed-size execution pool; leases own buffer reuse.
class DeviceDiskTransferExecutor::TransientHostStagingPool {
public:
    class Lease {
    public:
        Lease() = default;
        Lease(TransientHostStagingPool* pool, size_t block_id): pool_(pool), block_id_(block_id) {}
        Lease(const Lease&)            = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept: pool_(other.pool_), block_id_(other.block_id_) {
            other.pool_ = nullptr;
        }
        Lease& operator=(Lease&& other) noexcept {
            if (this != &other) {
                reset();
                pool_       = other.pool_;
                block_id_   = other.block_id_;
                other.pool_ = nullptr;
            }
            return *this;
        }
        ~Lease() {
            reset();
        }

        HostBufferView view(size_t payload_bytes) const {
            RTP_LLM_CHECK(pool_ != nullptr);
            return pool_->viewFor(block_id_, payload_bytes);
        }

    private:
        void reset() {
            if (pool_ != nullptr) {
                pool_->release(block_id_);
                pool_ = nullptr;
            }
        }

        TransientHostStagingPool* pool_{nullptr};
        size_t                    block_id_{0};
    };

    TransientHostStagingPool(size_t block_count, size_t stride_bytes, StagingPinMemoryFn pin_memory):
        block_count_(block_count), stride_bytes_(stride_bytes) {
        RTP_LLM_CHECK_WITH_INFO(block_count_ > 0, "staging block_count must be > 0");
        RTP_LLM_CHECK_WITH_INFO(stride_bytes_ > 0, "staging stride_bytes must be > 0");

        const size_t total_bytes = block_count_ * stride_bytes_;
        auto         cpu         = torch::empty({static_cast<int64_t>(total_bytes + kStagingAlignment)},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        try {
            backing_ = pin_memory ? pin_memory(cpu) : cpu.pin_memory();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("pin staging memory failed, fallback to pageable CPU memory, error=%s", e.what());
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

    std::optional<Lease> tryAcquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (free_id_list_.empty()) {
            return std::nullopt;
        }
        const size_t block_id = free_id_list_.back();
        free_id_list_.pop_back();
        return Lease(this, block_id);
    }

private:
    friend class Lease;

    void release(size_t block_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        free_id_list_.push_back(block_id);
    }

    HostBufferView viewFor(size_t block_id, size_t payload_bytes) const {
        RTP_LLM_CHECK(block_id < block_count_);
        RTP_LLM_CHECK_WITH_INFO(
            payload_bytes <= stride_bytes_, "staging payload %zu exceeds stride %zu", payload_bytes, stride_bytes_);
        void* base = base_ptr_ + block_id * stride_bytes_;
        return HostBufferView{base, payload_bytes, stride_bytes_};
    }

    torch::Tensor       backing_;
    uint8_t*            base_ptr_{nullptr};
    std::vector<size_t> free_id_list_;
    mutable std::mutex  mutex_;
    size_t              block_count_{0};
    size_t              stride_bytes_{0};
};

DeviceDiskTransferExecutor::DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                                                       HostDiskTransferExecutor&       host_disk_executor,
                                                       const std::vector<GroupSetPtr>& group_sets,
                                                       size_t                          staging_block_count,
                                                       StagingPinMemoryFn              pin_memory):
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
    staging_pool_ =
        std::make_unique<TransientHostStagingPool>(staging_block_count, stride_bytes, std::move(pin_memory));
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

    auto lease = staging_pool_->tryAcquire();
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
