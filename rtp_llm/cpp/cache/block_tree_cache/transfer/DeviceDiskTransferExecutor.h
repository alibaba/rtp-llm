#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

class DeviceHostTransferExecutor;
class HostDiskTransferExecutor;
class BlockTreeTaskPool;

class DeviceDiskTransferExecutor {
public:
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

        TransientHostStagingPool(size_t block_count, size_t stride_bytes, bool try_pin_memory = true);

        std::optional<Lease> tryAcquire();

    private:
        friend class Lease;

        void release(size_t block_id);

        HostBufferView viewFor(size_t block_id, size_t payload_bytes) const;

        torch::Tensor        backing_;
        uint8_t*             base_ptr_{nullptr};
        std::vector<size_t>  free_id_list_;
        mutable std::mutex   mutex_;
        size_t               block_count_{0};
        size_t               stride_bytes_{0};
    };

    DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                               HostDiskTransferExecutor&       host_disk_executor,
                               const std::vector<GroupSetPtr>& group_sets,
                               size_t                          staging_block_count);
    ~DeviceDiskTransferExecutor();

    DeviceDiskTransferExecutor(const DeviceDiskTransferExecutor&)            = delete;
    DeviceDiskTransferExecutor& operator=(const DeviceDiskTransferExecutor&) = delete;

    TransferStatus                execute(const TransferDescriptor& desc, const GroupSet& group);
    std::shared_ptr<AsyncContext> diskToDevice(const std::vector<TransferDescriptor>& descriptors,
                                               const std::vector<const GroupSet*>&    group_sets,
                                               std::shared_ptr<void>                  completion_guard = nullptr);

private:
    enum class StagingPoolState {
        FREE,
        STAGE1_FILLING,
        STAGE1_IN_FLIGHT,
        STAGE2_READY,
        STAGE2_IN_FLIGHT,
    };

    struct PipelineBatchState;
    struct PipelineSlice;

    std::optional<size_t> acquireStagingPool(std::chrono::milliseconds timeout);
    void                  setStagingPoolState(size_t pool_index, StagingPoolState state);
    void                  releaseStagingPool(size_t pool_index);

    DeviceHostTransferExecutor&                              device_host_executor_;
    HostDiskTransferExecutor&                                host_disk_executor_;
    std::array<std::unique_ptr<TransientHostStagingPool>, 2> staging_pools_;
    size_t                                                   active_staging_pool_count_{0};
    size_t                                                   staging_pool_capacity_{0};
    size_t                                                   next_staging_pool_index_{0};
    std::array<StagingPoolState, 2>                          staging_pool_states_;
    std::mutex                                               staging_state_mutex_;
    std::condition_variable                                  staging_state_cv_;
    std::unique_ptr<BlockTreeTaskPool>                       staging_to_device_task_pool_;
    std::unique_ptr<BlockTreeTaskPool>                       disk_to_staging_task_pool_;
};

}  // namespace rtp_llm
