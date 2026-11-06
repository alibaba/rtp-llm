#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/legacy/DeviceHostCopyOptions.h"

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskIOExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcTransferService.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferWorkLoop.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class LegacyTransferEngine;

class PerRankBlockTransferEngine {
public:
    explicit PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                        bool                     enable_disk_cache                        = false,
                                        DeviceHostCopyOptions    device_host_options                      = {},
                                        size_t                   device_disk_staging_block_count          = 128,
                                        size_t                   max_descriptors_per_batch                = 32,
                                        size_t                   transfer_worker_count                    = 1,
                                        size_t                   transfer_queue_max_size                  = 10000,
                                        std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter   = nullptr,
                                        int64_t                                        world_rank         = 0,
                                        size_t                                         btc_crc_slot_count = 8,
                                        bool                                           btc_transfer_engine_async = true,
                                        DiskIOExecutorConfig                           disk_io_config            = {});
    PerRankBlockTransferEngine() = delete;
    virtual ~PerRankBlockTransferEngine();

    virtual std::shared_ptr<TransferAsyncContext> execute(TransferTask task);
    // Initialized once from M and all async staging capacities, shared by both
    // directions. Legacy keeps M; Device->Disk keeps its single-item contract.
    size_t              batchLimit(Tier source, Tier target) const noexcept;
    void                cancelPendingStagingTransfers();
    BlockTreeQueueSizes queueSizes() const;
    void                stopAdmission();
    // External lifecycle thread only, never from a WorkLoop callback.
    // The caller serializes drain() and destruction; sequential repeats are safe.
    void drain();
    // External thread waits for callbacks and cleanup without closing admission.
    void waitForIdle();
    // Engine-wide disk scheduling configuration, shared by every WorkLoop.
    const DiskIOExecutor* diskIOExecutor() const {
        return disk_io_executor_.get();
    }
    bool asyncEnabled() const {
        return async_;
    }

    // WorkLoop owners or legacy workers per direction.
    size_t transferWorkerCount() const {
        return transfer_worker_count_;
    }

    // Slots per device in each async CRC service.
    size_t crcSlotCount() const {
        return crc_slot_count_;
    }

private:
    std::vector<GroupSetPtr> group_sets_;
    size_t                   max_descriptors_per_batch_;

    std::unique_ptr<DiskIOExecutor> disk_io_executor_;

    std::unique_ptr<LegacyTransferEngine>          legacy_;
    std::vector<std::unique_ptr<TransferWorkLoop>> load_loops_, offload_loops_;
    std::atomic<size_t>                            next_load_{0}, next_offload_{0};
    bool                                           async_{false}, enable_disk_cache_{false};
    size_t                                         transfer_worker_count_{1};
    bool                                           track_decode_origins_{false};
    size_t                                         crc_slot_count_;
};

using PerRankBlockTransferEnginePtr = std::shared_ptr<PerRankBlockTransferEngine>;

}  // namespace rtp_llm
