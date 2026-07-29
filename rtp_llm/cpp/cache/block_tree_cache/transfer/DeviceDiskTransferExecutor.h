#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

class DeviceHostTransferExecutor;
class HostDiskTransferExecutor;

// D2Disk: staging lease -> D2H -> H2Disk. Disk2D: staging lease -> Disk2H -> H2D.
class DeviceDiskTransferExecutor {
public:
    // Fixed-size execution pool; leases own buffer reuse.
    class TransientHostStagingPool {
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

        // try_pin_memory=false is a test seam to force pageable backing.
        TransientHostStagingPool(size_t block_count, size_t stride_bytes, bool try_pin_memory = true);

        std::optional<Lease> tryAcquire();

        // Exponential backoff with deadline; never sleeps while holding mutex_.
        std::optional<Lease> acquireWithBackoff(std::chrono::milliseconds timeout);

    private:
        friend class Lease;

        void reportAcquireTimeout(std::chrono::steady_clock::time_point wait_start);

        void release(size_t block_id);

        HostBufferView viewFor(size_t block_id, size_t payload_bytes) const;

        torch::Tensor        backing_;
        uint8_t*             base_ptr_{nullptr};
        std::vector<size_t>  free_id_list_;
        mutable std::mutex   mutex_;
        size_t               block_count_{0};
        size_t               stride_bytes_{0};
        std::atomic<size_t>  timeout_count_{0};
        std::atomic<int64_t> last_timeout_log_ns_{0};
    };

    DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                               HostDiskTransferExecutor&       host_disk_executor,
                               const std::vector<GroupSetPtr>& group_sets,
                               size_t                          staging_block_count);
    ~DeviceDiskTransferExecutor();

    DeviceDiskTransferExecutor(const DeviceDiskTransferExecutor&)            = delete;
    DeviceDiskTransferExecutor& operator=(const DeviceDiskTransferExecutor&) = delete;

    TransferStatus execute(const TransferDescriptor& desc, const GroupSet& group);

private:
    DeviceHostTransferExecutor&               device_host_executor_;
    HostDiskTransferExecutor&                 host_disk_executor_;
    std::unique_ptr<TransientHostStagingPool> staging_pool_;
};

}  // namespace rtp_llm
