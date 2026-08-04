#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheMetricsReporter.h"
#include "rtp_llm/cpp/cache/block_tree_cache/evict/BlockTreeEvictor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeTaskPool;

class BlockTreeStorer {
public:
    // Invoked with the shared cache mutex held.
    using SettledFn = std::function<void(bool tree_data_mutated, bool check_watermark)>;

    BlockTreeStorer(BlockTree*                     tree,
                    BlockTreeEvictor&              evictor,
                    BlockTransferDispatcher*       transfer_dispatcher,
                    BlockTreeTaskPool*             task_pool,
                    BlockTreeCacheMetricsReporter& metrics_reporter,
                    std::mutex&                    mutex,
                    int                            host_timeout_ms,
                    int                            disk_timeout_ms,
                    SettledFn                      settled);

    void storeLocked(const CacheKeysType&                              cache_keys,
                     const std::vector<std::vector<GroupSetResource>>& resources,
                     Tier                                              target_tier);
    void stopAdmissionLocked();

private:
    struct StoreTask {
        struct Entry {
            size_t                    key_index{0};
            size_t                    group_set_id{0};
            std::vector<BlockIdxType> source_device_blocks;
            BlockIdxType              target_block{NULL_BLOCK_IDX};
        };

        Tier                            target_tier{Tier::NONE};
        CacheKeysType                   cache_keys;
        std::vector<Entry>              entries;
        std::vector<TransferDescriptor> descriptors;
    };
    using StoreTaskPtr = std::shared_ptr<StoreTask>;

    void   publishDeviceLocked(const CacheKeysType&                              cache_keys,
                               const std::vector<std::vector<GroupSetResource>>& resources);
    void   submitLowerTierLocked(const CacheKeysType&                              cache_keys,
                                 const std::vector<std::vector<GroupSetResource>>& resources,
                                 Tier                                              target_tier);
    void   runStoreTask(const StoreTaskPtr& task);
    void   settleTask(const StoreTask& task, bool copy_success);
    size_t settleLocked(const StoreTask& task, bool publish);
    void   releaseSourceLocked(const GroupSet& group_set, const StoreTask::Entry& entry);

    BlockTree*                     tree_;
    BlockTreeEvictor&              evictor_;
    BlockTransferDispatcher*       transfer_dispatcher_;
    BlockTreeTaskPool*             task_pool_;
    BlockTreeCacheMetricsReporter& metrics_reporter_;
    std::mutex&                    mutex_;
    int                            host_timeout_ms_{0};
    int                            disk_timeout_ms_{0};
    SettledFn                      settled_;
    std::atomic<bool>              stopping_{false};
};

}  // namespace rtp_llm
