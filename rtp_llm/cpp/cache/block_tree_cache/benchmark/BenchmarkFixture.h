#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/ModelProfile.h"

namespace rtp_llm::benchmark {

struct PhaseTiming {
    int64_t bootstrap_ns{0};
    int64_t profile_load_ns{0};
    int64_t allocation_ns{0};
    int64_t setup_ns{0};
    int64_t warmup_ns{0};
    int64_t measured_ns{0};
    int64_t sync_drain_ns{0};
    int64_t teardown_ns{0};
};

struct ResourceBudget {
    int64_t estimated_device_bytes{0};
    int64_t estimated_host_bytes{0};
    int64_t estimated_peak_host_bytes{0};
    int64_t estimated_disk_bytes{0};
    int64_t estimated_request_template_bytes{0};
    int64_t available_device_bytes{0};
    int64_t available_host_or_cgroup_bytes{0};
    int64_t available_disk_bytes{0};
    bool    sufficient{false};
};

class BenchmarkFixture {
public:
    BenchmarkFixture(const ModelProfile& profile, uint64_t seed);
    ~BenchmarkFixture();

    // Pool builders (static, no test utility dependency). layer_stride_bytes
    // is the per-layer stride; a block holds layer_num consecutive layers.
    static DeviceBlockPoolPtr
    createDevicePool(size_t layer_stride_bytes, size_t layer_num, size_t usable_count, const std::string& pool_name);

    static std::shared_ptr<HostBlockPool> createHostPool(size_t payload_bytes, size_t usable_count, bool enable_pinned);

    static BlockTreeDiskBlockPoolPtr createDiskPool(size_t             payload_bytes,
                                                    size_t             usable_count,
                                                    const std::string& work_dir,
                                                    const std::string& pool_name,
                                                    bool               buffered_io);

    // GroupSet builders (share one global CacheTopology across GroupSets so
    // group ids are unique in the BlockTree). Payload is derived by
    // GroupSet::initialize from the topology's layer layout.
    static GroupSetPtr createFullGroupSet(std::vector<DeviceBlockPoolPtr>      device_pools,
                                          std::shared_ptr<HostBlockPool>       host_pool,
                                          BlockTreeDiskBlockPoolPtr            disk_pool,
                                          size_t                               group_set_id,
                                          std::shared_ptr<const CacheTopology> topology,
                                          const std::vector<size_t>&           group_ids);

    static GroupSetPtr createSWAGroupSet(std::vector<DeviceBlockPoolPtr>      device_pools,
                                         std::shared_ptr<HostBlockPool>       host_pool,
                                         BlockTreeDiskBlockPoolPtr            disk_pool,
                                         size_t                               group_set_id,
                                         std::shared_ptr<const CacheTopology> topology,
                                         const std::vector<size_t>&           group_ids,
                                         size_t                               sliding_window_size);

    // Build a shared CacheTopology: one group per entry with a unique tag.
    // layer_stride_bytes_per_group is the per-layer stride; each group gets
    // layer_counts_per_group[i] layers (default 1) with globally unique ids.
    static std::shared_ptr<const CacheTopology>
    createTopology(const std::vector<std::pair<std::string, rtp_llm::CacheGroupType>>& group_specs,
                   const std::vector<size_t>&                                          layer_stride_bytes_per_group,
                   const std::vector<size_t>&                                          layer_counts_per_group = {});

    // Cache builder. `task_pool_size` sizes the shared store/load/evict async
    // task pool; production configures it separately from request concurrency.
    // `device_watermark_ratio`/`host_watermark_ratio` arm event-driven
    // watermark eviction (0.0 = tier watermark disabled).
    static std::unique_ptr<BlockTreeCache> createCache(std::vector<GroupSetPtr> group_sets,
                                                       bool                     enable_host            = false,
                                                       bool                     enable_disk            = false,
                                                       size_t                   task_pool_size         = 4,
                                                       double                   device_watermark_ratio = 0.0,
                                                       double                   host_watermark_ratio   = 0.0);

    // Resource preflight
    static ResourceBudget preflightResources(const ModelProfile& profile,
                                             const std::string&  payload_mode,
                                             size_t              tree_node_count,
                                             size_t              transfer_concurrency,
                                             const std::string&  host_memory_type,
                                             const std::string&  disk_path,
                                             double              max_device_memory_fraction);

    // Payload scaling
    static size_t scaleStride(size_t original_stride);
    static size_t computeScaledPayload(size_t original_payload);

    // Phase timing
    const PhaseTiming& timing() const {
        return timing_;
    }
    void setPhaseTime(const std::string& phase, int64_t ns);

    // Resource budget
    const ResourceBudget& budget() const {
        return budget_;
    }
    void setBudget(const ResourceBudget& budget) {
        budget_ = budget;
    }

    const ModelProfile& profile() const {
        return profile_;
    }

private:
    const ModelProfile& profile_;
    uint64_t            seed_;
    PhaseTiming         timing_;
    ResourceBudget      budget_;
};

// Utility: compute SHA-256 of a file
std::string fileSha256(const std::string& path);

}  // namespace rtp_llm::benchmark