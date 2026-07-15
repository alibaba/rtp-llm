#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockIO.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"

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

std::shared_ptr<DiskBlockPool>
makeDiskPool(size_t payload_bytes, size_t usable_count, std::unique_ptr<DiskBlockIO> io = nullptr);

bool cudaAvailable();

DeviceBlockPoolPtr
makeDevicePool(const std::vector<DeviceLayerBufferSpec>& specs, size_t usable_count, const std::string& pool_name);

BlockIdxType poolMalloc(IBlockPool& pool);

std::unique_ptr<BlockTreeCache>
makeBlockTreeCacheForTest(std::unique_ptr<BlockTree>        tree,
                          std::vector<ComponentGroupPtr>    component_groups,
                          std::vector<Component>            components,
                          BlockTreeCacheConfig              config            = {},
                          std::shared_ptr<StorageBackend>   storage_backend   = nullptr,
                          std::shared_ptr<BroadcastManager> broadcast_manager = nullptr);

class BlockTreeCacheTestPeer {
public:
    static void setCopyEngineForTest(BlockTreeCache& cache, CopyEnginePtr copy_engine);
    static void runMaintenanceForTest(BlockTreeCache& cache);
};

class ScriptedCopyEngine: public CopyEngine {
public:
    ScriptedCopyEngine(const std::vector<ComponentGroupPtr>& groups, const std::vector<Component>& components);

    TransferHandle submit(const TransferDescriptor& descriptor) override;

    void enqueue(CopyStatus status);
    void clear();

    std::vector<TransferDescriptor> descriptors() const;
    size_t                          submitCount() const;

private:
    mutable std::mutex              mutex_;
    std::deque<CopyStatus>          statuses_;
    std::vector<TransferDescriptor> descriptors_;
};

struct FullSWAEnvironmentOptions {
    size_t path_length{4};
    size_t usable_device_blocks{16};
    size_t usable_host_blocks{16};
    size_t usable_disk_blocks{16};
    bool   enable_disk{true};
    bool   enable_load_back{true};
    bool   enable_reverse_eviction{true};
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

    bool allSlotsAtTier(Tier tier) const;
    void expectPayloads() const;
    void expectFullyReclaimed() const;
    void expectPoolFreeCounts(const std::vector<size_t>& device_free,
                              const std::vector<size_t>& host_free,
                              const std::vector<size_t>& disk_free) const;

    std::vector<BlockIdxType> blocksForTag(size_t tag_id) const;
    std::vector<GroupSlot>    slotsForPathNode(size_t path_index) const;

    CacheKeysType                               keys;
    std::vector<ComponentGroupPtr>              groups;
    std::vector<DeviceBlockPoolPtr>             device_pools;
    std::vector<std::shared_ptr<HostBlockPool>> host_pools;
    std::vector<std::shared_ptr<DiskBlockPool>> disk_pools;
    std::vector<Component>                      components;
    std::vector<GroupBlockSet>                  request_blocks;
    std::shared_ptr<ScriptedCopyEngine>         scripted_copy_engine;
    std::unique_ptr<BlockTreeCache>             cache;

private:
    explicit FullSWAEnvironment(FullSWAEnvironmentOptions options);

    void fillRequestPayloads();
    void setTierWatermark(Tier tier, double ratio);

    FullSWAEnvironmentOptions options_;
    std::vector<bool>         request_refs_released_;
};

}  // namespace rtp_llm::block_tree_cache_test
