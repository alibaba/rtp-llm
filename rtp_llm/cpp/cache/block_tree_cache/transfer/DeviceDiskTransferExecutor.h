#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTreeTaskPool;
class DeviceHostTransferExecutor;
class HostDiskTransferExecutor;

class DeviceDiskTransferExecutor {
public:
    DeviceDiskTransferExecutor(DeviceHostTransferExecutor&     device_host_executor,
                               HostDiskTransferExecutor&       host_disk_executor,
                               const std::vector<GroupSetPtr>& group_sets,
                               size_t                          staging_block_count);
    ~DeviceDiskTransferExecutor();

    DeviceDiskTransferExecutor(const DeviceDiskTransferExecutor&)            = delete;
    DeviceDiskTransferExecutor& operator=(const DeviceDiskTransferExecutor&) = delete;

    std::shared_ptr<AsyncContext> execute(const std::vector<TransferDescriptor>& descriptors,
                                          const std::vector<const GroupSet*>&    group_sets,
                                          std::shared_ptr<void>                  completion_guard = nullptr);

    TransferStatus execute(const TransferDescriptor& descriptor, const GroupSet& group_set);

private:
    enum class PoolState {
        FREE,
        STAGE1_IN_FLIGHT,
        STAGE2_READY,
        STAGE2_IN_FLIGHT,
    };

    struct BatchState;
    struct Slice;
    struct PoolWork;

    void                  drainStageOne();
    std::optional<size_t> acquirePool(std::chrono::milliseconds timeout);
    void                  setPoolState(size_t pool_index, PoolState state);
    void                  releasePool(size_t pool_index);

    DeviceHostTransferExecutor&                    device_host_executor_;
    HostDiskTransferExecutor&                      host_disk_executor_;
    std::array<std::unique_ptr<HostStagingBlockPool>, 2> staging_pools_;
    size_t                                         staging_pool_capacity_{0};
    size_t                                         next_pool_index_{0};
    std::array<PoolState, 2>                       pool_states_;
    std::mutex                                     pool_mutex_;
    std::condition_variable                        pool_cv_;
    std::mutex                                     pending_mutex_;
    std::deque<std::shared_ptr<BatchState>>        pending_batches_;
    bool                                           stage_one_scheduled_{false};
    std::unique_ptr<BlockTreeTaskPool>             disk_to_staging_task_pool_;
    std::unique_ptr<BlockTreeTaskPool>             staging_to_device_task_pool_;
};

}  // namespace rtp_llm
