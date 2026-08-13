#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {

class BlockTreeCache;
struct CacheTopology;
class DeviceBlockPool;
class HostBlockPool;
class BlockTreeDiskBlockPool;
class GroupSet;
enum class CacheGroupType : int8_t;

namespace benchmark {

struct ModelProfile;
struct OnlineTreeWorkloadConfig;

struct ResourceBudget {
    int64_t estimated_device_bytes{0};
    int64_t estimated_host_bytes{0};
    int64_t available_device_bytes{0};
    int64_t raw_available_device_bytes{0};
    int64_t available_host_or_cgroup_bytes{0};
    bool    sufficient{false};
};

class BenchmarkFixture {
public:
    // Pool builders (static, no test utility dependency). layer_stride_bytes
    // is the per-layer stride; a block holds layer_num consecutive layers.
    static std::shared_ptr<DeviceBlockPool>
    createDevicePool(size_t layer_stride_bytes, size_t layer_num, size_t usable_count, const std::string& pool_name);

    static std::shared_ptr<HostBlockPool> createHostPool(size_t             payload_bytes,
                                                         size_t             usable_count,
                                                         bool               enable_pinned,
                                                         const std::string& pool_name = "benchmark_host");

    static std::shared_ptr<BlockTreeDiskBlockPool> createDiskPool(size_t             payload_bytes,
                                                                  size_t             usable_count,
                                                                  const std::string& work_dir,
                                                                  const std::string& pool_name,
                                                                  bool               buffered_io);

    // GroupSet builders (share one global CacheTopology across GroupSets so
    // group ids are unique in the BlockTree). Payload is derived by
    // GroupSet::initialize from the topology's layer layout.
    static std::shared_ptr<GroupSet> createFullGroupSet(std::vector<std::shared_ptr<DeviceBlockPool>> device_pools,
                                                        std::shared_ptr<HostBlockPool>                host_pool,
                                                        std::shared_ptr<BlockTreeDiskBlockPool>       disk_pool,
                                                        size_t                                        group_set_id,
                                                        std::shared_ptr<const CacheTopology>          topology,
                                                        const std::vector<size_t>&                    group_ids);

    static std::shared_ptr<GroupSet> createSWAGroupSet(std::vector<std::shared_ptr<DeviceBlockPool>> device_pools,
                                                       std::shared_ptr<HostBlockPool>                host_pool,
                                                       std::shared_ptr<BlockTreeDiskBlockPool>       disk_pool,
                                                       size_t                                        group_set_id,
                                                       std::shared_ptr<const CacheTopology>          topology,
                                                       const std::vector<size_t>&                    group_ids,
                                                       size_t sliding_window_size);

    // Build a shared CacheTopology: one group per entry with a unique tag.
    // layer_stride_bytes_per_group is the per-layer stride; each group gets
    // layer_counts_per_group[i] layers (default 1) with globally unique ids.
    static std::shared_ptr<const CacheTopology>
    createTopology(const std::vector<std::pair<std::string, rtp_llm::CacheGroupType>>& group_specs,
                   const std::vector<size_t>&                                          layer_stride_bytes_per_group,
                   const std::vector<size_t>&                                          layer_counts_per_group = {},
                   const std::vector<size_t>&                                          sliding_windows        = {});

    // Cache builder. `task_pool_size` sizes the shared store/load/evict async
    // task pool; production configures it separately from request concurrency.
    // `device_watermark_ratio`/`host_watermark_ratio` arm event-driven
    // watermark eviction (0.0 = tier watermark disabled).
    static std::unique_ptr<BlockTreeCache> createCache(std::vector<std::shared_ptr<GroupSet>> group_sets,
                                                       bool                                   enable_host    = false,
                                                       bool                                   enable_disk    = false,
                                                       size_t                                 task_pool_size = 4,
                                                       double device_watermark_ratio                         = 0.0,
                                                       double host_watermark_ratio                           = 0.0);

    // Resource preflight for the online Tree workload: estimates from the
    // fixed device/host pool sizes plus the admission preparation peak
    // (active-token budget / tokens-per-block logical blocks), never from the
    // initial cache node count.
    static ResourceBudget preflightTreeResources(const ModelProfile&             profile,
                                                 const OnlineTreeWorkloadConfig& config,
                                                 int                             cuda_device,
                                                 double                          max_device_memory_fraction);

    // Payload scaling
    static size_t computeScaledPayload(size_t original_payload);
};

}  // namespace benchmark

}  // namespace rtp_llm
