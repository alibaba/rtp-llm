#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheLayout.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskIOExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/CrcCopyExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferWorkLoop.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class RemoteMockAdapter;

class PerRankBlockTransferEngine {
public:
    // Disk queue sizes are derived from Q and the configured batch limit;
    // disk_io_config supplies the shared read/write thread counts.
    explicit PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                        bool                     enable_disk_cache                      = false,
                                        size_t                   device_disk_staging_block_count        = 128,
                                        size_t                   max_descriptors_per_batch              = 32,
                                        size_t                   transfer_worker_count                  = 2,
                                        size_t                   transfer_queue_max_size                = 10000,
                                        std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter = nullptr,
                                        int64_t                                        world_rank       = 0,
                                        size_t               btc_crc_slot_count = kDefaultCrcSlotsPerWorkLoop,
                                        DiskIOExecutorConfig disk_io_config     = {});
    PerRankBlockTransferEngine() = delete;
    virtual ~PerRankBlockTransferEngine();

    // Group a logical request by direction and GroupSet, then submit bounded batches.
    virtual std::shared_ptr<TransferAsyncContext> execute(TransferTask task);
    // External pinned staging does not occupy a Host cache block. The owner retains
    // its storage and any device references through actual copy completion.
    std::shared_ptr<TransferAsyncContext> executeHostBuffers(TransferTask                       task,
                                                             std::vector<HostBufferView>        hosts,
                                                             std::shared_ptr<void>              owner,
                                                             TransferAsyncContext::DoneCallback terminal = {});
    // Startup attachment: workers execute owner indices without local allocation authority.
    void setRemoteMockAdapter(const std::shared_ptr<RemoteMockAdapter>& adapter, bool owns_device_allocations = true);
    RemoteMockWritebackStats remoteWritebackStats() const;
    // Staging pools are validated at initialization to hold a full configured batch.
    size_t batchLimit() const noexcept {
        return max_descriptors_per_batch_;
    }
    const std::vector<GroupSetPtr>& groupSets() const noexcept {
        return group_sets_;
    }
    void cancelPendingStagingTransfers();
    // Report outstanding batches including callbacks. Evict uses the existing background field.
    BlockTreeQueueSizes queueSizes() const;
    void                stopAdmission();
    // External lifecycle thread only, never from a WorkLoop callback.
    // Closes admission and waits for all local transfers, callbacks and cleanup.
    // The caller serializes shutdown() and destruction; sequential repeats are safe.
    void shutdown();
    // External thread waits for callbacks and cleanup without closing admission.
    void waitForIdle();
    // Engine-wide disk scheduling configuration, shared by every WorkLoop.
    const DiskIOExecutor* diskIOExecutor() const {
        return disk_io_executor_.get();
    }
    // WorkLoop owners per direction.
    size_t transferWorkerCount() const {
        return transfer_worker_count_;
    }

    // Slots per device in each CRC service.
    size_t crcSlotCount() const {
        return crc_slot_count_;
    }

private:
    virtual std::shared_ptr<TransferAsyncContext> executeBatch(TransferTask task);
    std::shared_ptr<TransferAsyncContext>         executeRemote(TransferTask task);
    std::shared_ptr<TransferAsyncContext>         executeDeviceToDisk(TransferTask task);

    std::weak_ptr<RemoteMockAdapter> remote_adapter_;
    bool                             owns_device_allocations_{true};

    std::vector<GroupSetPtr> group_sets_;
    size_t                   max_descriptors_per_batch_;

    std::unique_ptr<DiskIOExecutor> disk_io_executor_;

    std::vector<std::unique_ptr<TransferWorkLoop>> load_loops_, evict_loops_;
    std::atomic<size_t>                            next_load_{0}, next_evict_{0};
    bool                                           enable_disk_cache_{false};
    size_t                                         transfer_worker_count_{2};
    size_t                                         crc_slot_count_;
};

using PerRankBlockTransferEnginePtr = std::shared_ptr<PerRankBlockTransferEngine>;

}  // namespace rtp_llm
