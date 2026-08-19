#pragma once

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/test/PerRankBlockTransferEngineTestUtils.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockIO.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"

namespace rtp_llm {
class BroadcastManager;
}

namespace rtp_llm::block_tree_cache_test {

struct DeviceLayerBufferSpec {
    size_t kv_bytes{0};
    size_t scale_bytes{0};
};

std::shared_ptr<HostBlockPool> makeHostPool(size_t payload_bytes, size_t usable_count);

class MemoryDiskBlockIO: public DiskBlockIO {
public:
    DiskBlockIOStatus openAndPreallocate(const std::string&, size_t bytes, bool) override;
    DiskBlockIOStatus read(uint64_t offset, void* dst, size_t bytes) override;
    DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) override;
    DiskBlockIOStatus read(const std::vector<DiskRead>& reads) override;
    DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) override;
    void              close() override;
    std::string       debugString() const override;

private:
    std::vector<char> data_;
};

std::shared_ptr<BlockTreeDiskBlockPool>
makeDiskPool(size_t payload_bytes, size_t usable_count, std::unique_ptr<DiskBlockIO> io = nullptr);

bool cudaAvailable();

DeviceBlockPoolPtr
makeDevicePool(const std::vector<DeviceLayerBufferSpec>& specs, size_t usable_count, const std::string& pool_name);

using MultiNodeBlocks = std::vector<std::vector<BlockIdxType>>;

BlockIdxType    poolMalloc(IBlockPool& pool);
MultiNodeBlocks allocateDeviceBlocksForTest(GroupSet& group_set, size_t count, BlockRefType ref_type);
void            referenceDeviceBlocksForTest(GroupSet& group_set, const MultiNodeBlocks& blocks, BlockRefType ref_type);
void unreferenceDeviceBlocksForTest(GroupSet& group_set, const MultiNodeBlocks& blocks, BlockRefType ref_type);
MultiNodeResource makeMultiNodeResourceForTest(size_t                        group_set_id,
                                               Tier                          tier,
                                               const std::vector<TreeNode*>& nodes,
                                               const MultiNodeBlocks&        blocks);

size_t             unreferencedBlocksNum(const IBlockPool& pool);
size_t             treeCachedBlocksNum(const IBlockPool& pool);
DeviceBlockPoolPtr makeStructuralDevicePool(size_t group_set_id);
void               releaseDeviceBlocks(BlockTreeCache&           cache,
                                       const DeviceBlockPoolPtr& pool,
                                       const BlockIdList&        blocks,
                                       BlockRefType              ref_type);
void releaseRequestRefsForTest(BlockTreeCache& cache, const std::vector<MultiNodeResource>& resources);

void prepareGroupSetsForTest(std::vector<GroupSetPtr>& group_sets);

class CallbackBarrier {
public:
    void enterAndWait();
    void waitUntilEntered(size_t expected_count = 1);
    void release();

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    size_t                  entered_count_{0};
    bool                    released_{false};
};

enum class TransferCopyAction {
    Succeed,
    Fail,
    Throw
};

const char* transferCopyActionName(TransferCopyAction action);

class ControlledPerRankBlockTransferEngine final: public PerRankBlockTransferEngine {
public:
    ControlledPerRankBlockTransferEngine(const std::vector<GroupSetPtr>&  groups,
                                         TransferCopyAction               action,
                                         std::shared_ptr<CallbackBarrier> barrier = nullptr);

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override;

    size_t submittedBatchCount() const;

private:
    const TransferCopyAction         action_;
    std::shared_ptr<CallbackBarrier> barrier_;
    std::atomic<size_t>              submit_count_{0};
};

std::unique_ptr<BlockTreeCache>
makeBlockTreeCacheForTest(std::vector<GroupSetPtr>          group_sets,
                          BlockTreeCacheConfig              config            = {},
                          std::shared_ptr<StorageBackend>   storage_backend   = nullptr,
                          std::shared_ptr<BroadcastManager> broadcast_manager = nullptr);

bool insertGroupSetResources(BlockTreeCache&                                   cache,
                             const CacheKeysType&                              cache_keys,
                             const std::vector<std::vector<GroupSetResource>>& resources);

// Accepted resources keep the tree hold after temporary seed holds are released.
void releaseLowerTierSeedRefs(const std::vector<GroupSetPtr>&                   group_sets,
                              const std::vector<std::vector<GroupSetResource>>& resources);

class BlockTreeCacheTestPeer {
public:
    class ScopedQueueRejectionGuard {
    public:
        explicit ScopedQueueRejectionGuard(BlockTreeCache& cache);
        ~ScopedQueueRejectionGuard();

        ScopedQueueRejectionGuard(const ScopedQueueRejectionGuard&)            = delete;
        ScopedQueueRejectionGuard& operator=(const ScopedQueueRejectionGuard&) = delete;
        ScopedQueueRejectionGuard(ScopedQueueRejectionGuard&&)                 = delete;
        ScopedQueueRejectionGuard& operator=(ScopedQueueRejectionGuard&&)      = delete;

        bool armed() const;
        bool restore();

    private:
        BlockTreeCache* cache_{nullptr};
        bool            armed_{false};
    };

    static void setPerRankBlockTransferEngineForTest(BlockTreeCache&               cache,
                                                     PerRankBlockTransferEnginePtr per_rank_transfer_engine);
    static void   setTierWatermarkForTest(BlockTreeCache& cache, Tier tier, double ratio);
    static void   refreshCandidateForTest(BlockTreeCache& cache, TreeNode* node, size_t group_set_id);
    static void   markPathMatchedForTest(BlockTreeCache& cache, const std::vector<TreeNode*>& path);
    static size_t pendingEvictionReleasesForTest(const BlockTreeCache& cache);
    static void runMaintenanceForTest(BlockTreeCache& cache);
    static void beginStoreShutdownForTest(BlockTreeCache& cache);
    static bool
    demoteOneForGroupSetForTest(BlockTreeCache& cache, size_t group_set_id, Tier tier, bool force_drop = false);
    static int  reclaimBlocksForTest(BlockTreeCache& cache, size_t num_blocks, Tier tier = Tier::DEVICE);
    static int  pendingTasksForTest(const BlockTreeCache& cache);
    static void waitForTaskPoolIdleForTest(const BlockTreeCache& cache);

private:
    static bool armQueueRejectionForTest(BlockTreeCache& cache);
    static bool restoreQueueAfterRejectionForTest(BlockTreeCache& cache);
};

class ScriptedPerRankBlockTransferEngine: public PerRankBlockTransferEngine {
public:
    explicit ScriptedPerRankBlockTransferEngine(const std::vector<GroupSetPtr>& groups,
                                                bool perform_successful_transfers = true);

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override;

    // Scripts the outcome of upcoming submits. Successful submits delegate to
    // the real engine unless perform_successful_transfers is false.
    void enqueue(bool success);
    void clear();

    std::vector<TransferDescriptor> descriptors() const;
    size_t                          submittedBatchCount() const;
    size_t                          submittedDescriptorCount() const;

private:
    mutable std::mutex              mutex_;
    std::deque<bool>                results_;
    std::vector<TransferDescriptor> descriptors_;
    size_t                          submitted_batch_count_{0};
    bool                            perform_successful_transfers_{true};
};

struct FullSWAEnvironmentOptions {
    size_t path_length{4};
    size_t usable_device_blocks{16};
    size_t usable_host_blocks{16};
    size_t usable_disk_blocks{16};
    bool   enable_disk{true};
};

class FullSWAEnvironment {
public:
    static std::unique_ptr<FullSWAEnvironment> create(const FullSWAEnvironmentOptions& options = {});

    void insertRequestPath();
    void releaseRequestRefs();
    void releaseRequestRefsForGroup(int group_id);
    void releaseMatch(BlockTreeMatchResult& result);

    void demoteAll(Tier tier);
    void runMaintenance();
    void reclaimAll();

    bool allResourcesAtTier(Tier tier) const;
    void expectPayloads() const;
    void expectFullyReclaimed() const;
    void expectPoolFreeCounts(const std::vector<size_t>& device_free,
                              const std::vector<size_t>& host_free,
                              const std::vector<size_t>& disk_free) const;

    std::vector<BlockIdxType>     blocksForDevicePool(size_t pool_id) const;
    std::vector<GroupSetResource> resourcesForPathNode(size_t path_index) const;

    CacheKeysType                                        keys;
    std::vector<GroupSetPtr>                             groups;
    std::vector<DeviceBlockPoolPtr>                      device_pools;
    std::vector<std::shared_ptr<HostBlockPool>>          host_pools;
    std::vector<std::shared_ptr<BlockTreeDiskBlockPool>> disk_pools;
    std::shared_ptr<const CacheTopology>                 topology;
    std::vector<MultiNodeBlocks>                         request_blocks;
    std::shared_ptr<ScriptedPerRankBlockTransferEngine>  scripted_per_rank_transfer_engine;
    std::unique_ptr<BlockTreeCache>                      cache;

private:
    explicit FullSWAEnvironment(FullSWAEnvironmentOptions options);

    void fillRequestPayloads();
    void setTierWatermark(Tier tier, double ratio);

    FullSWAEnvironmentOptions options_;
    std::vector<bool>         request_refs_released_;
};

}  // namespace rtp_llm::block_tree_cache_test
