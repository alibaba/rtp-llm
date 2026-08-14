#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/BenchmarkFixture.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/statvfs.h>
#include <chrono>

#include <cuda_runtime.h>
#include <openssl/sha.h>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::benchmark {

namespace {

constexpr size_t kPoolAlignment   = 4096;
constexpr size_t kScaleFactor     = 256;
constexpr size_t kMinScaledStride = 64;
constexpr size_t kScaledAlignment = 64;

size_t alignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

int64_t getAvailableHostMemory() {
    std::ifstream cgroup_file("/sys/fs/cgroup/memory/memory.limit_in_bytes");
    if (cgroup_file) {
        int64_t limit;
        cgroup_file >> limit;
        if (limit > 0 && limit < static_cast<int64_t>(1ULL << 50)) {
            return limit;
        }
    }
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) {
        return static_cast<int64_t>(rl.rlim_cur);
    }
    return static_cast<int64_t>(128ULL << 30);
}

int64_t getAvailableDiskSpace(const std::string& path) {
    struct statvfs stat;
    if (statvfs(path.c_str(), &stat) == 0) {
        return static_cast<int64_t>(stat.f_frsize) * static_cast<int64_t>(stat.f_bavail);
    }
    return -1;
}

}  // anonymous namespace

BenchmarkFixture::BenchmarkFixture(const ModelProfile& profile, uint64_t seed): profile_(profile), seed_(seed) {}

BenchmarkFixture::~BenchmarkFixture() = default;

DeviceBlockPoolPtr BenchmarkFixture::createDevicePool(size_t             layer_stride_bytes,
                                                      size_t             layer_num,
                                                      size_t             usable_count,
                                                      const std::string& pool_name) {
    const size_t physical_block_count = usable_count + 1;
    const size_t block_stride_bytes   = layer_stride_bytes * layer_num;

    auto config                     = std::make_shared<DeviceBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DEVICE;
    config->pool_name               = pool_name;
    config->physical_block_count    = physical_block_count;
    config->use_cuda_malloc_backing = false;
    config->total_size_bytes        = physical_block_count * block_stride_bytes;

    MemoryLayoutConfig layout;
    layout.layer_num                  = static_cast<uint32_t>(layer_num);
    layout.block_num                  = static_cast<uint32_t>(physical_block_count);
    layout.dtype                      = TYPE_INT8;
    layout.kv_cache_offset_bytes      = 0;
    layout.kv_block_stride_bytes      = layer_stride_bytes;
    layout.kv_block_pool_size_bytes   = physical_block_count * block_stride_bytes;
    layout.block_stride_bytes         = block_stride_bytes;
    layout.total_size_bytes           = layout.kv_block_pool_size_bytes;
    layout.local_head_num_kv          = 1;
    layout.seq_size_per_block         = 1;
    layout.kernel_blocks_per_kv_block = 1;
    config->memory_layouts.push_back(layout);

    auto pool = std::make_shared<DeviceBlockPool>(config);
    RTP_LLM_CHECK(pool->init());
    return pool;
}

std::shared_ptr<HostBlockPool>
BenchmarkFixture::createHostPool(size_t payload_bytes, size_t usable_count, bool enable_pinned) {
    auto config                  = std::make_shared<HostBlockPoolConfig>();
    config->pool_type            = BlockPoolType::HOST;
    config->pool_name            = "benchmark_host";
    config->physical_block_count = usable_count + 1;
    config->payload_bytes        = payload_bytes;
    config->stride_bytes         = alignUp(payload_bytes, kPoolAlignment);
    config->enable_pinned        = enable_pinned;
    config->alignment            = kPoolAlignment;

    auto pool = std::make_shared<HostBlockPool>(config);
    RTP_LLM_CHECK(pool->init());
    return pool;
}

BlockTreeDiskBlockPoolPtr BenchmarkFixture::createDiskPool(size_t             payload_bytes,
                                                           size_t             usable_count,
                                                           const std::string& work_dir,
                                                           const std::string& pool_name,
                                                           bool               buffered_io) {
    const size_t stride_bytes = alignUp(payload_bytes, kPoolAlignment);

    auto config             = std::make_shared<BlockTreeDiskBlockPoolConfig>();
    config->pool_type       = BlockPoolType::DISK;
    config->pool_name       = pool_name;
    config->work_dir        = work_dir;
    config->local_rank      = 0;
    config->world_rank      = 0;
    config->disk_size_bytes = stride_bytes * (usable_count + 1);
    config->payload_bytes   = payload_bytes;
    config->stride_bytes    = stride_bytes;
    config->buffered_io     = buffered_io;

    auto pool = std::make_shared<BlockTreeDiskBlockPool>(config, nullptr);
    RTP_LLM_CHECK(pool->init());
    return pool;
}

GroupSetPtr BenchmarkFixture::createFullGroupSet(std::vector<DeviceBlockPoolPtr>      device_pools,
                                                 std::shared_ptr<HostBlockPool>       host_pool,
                                                 BlockTreeDiskBlockPoolPtr            disk_pool,
                                                 size_t                               group_set_id,
                                                 std::shared_ptr<const CacheTopology> topology,
                                                 const std::vector<size_t>&           group_ids) {
    auto group_set = std::make_shared<FullGroupSet>(device_pools, host_pool, disk_pool);
    group_set->initialize(group_set_id, std::move(topology), group_ids);
    return group_set;
}

GroupSetPtr BenchmarkFixture::createSWAGroupSet(std::vector<DeviceBlockPoolPtr>      device_pools,
                                                std::shared_ptr<HostBlockPool>       host_pool,
                                                BlockTreeDiskBlockPoolPtr            disk_pool,
                                                size_t                               group_set_id,
                                                std::shared_ptr<const CacheTopology> topology,
                                                const std::vector<size_t>&           group_ids,
                                                size_t                               sliding_window_size) {
    auto group_set = std::make_shared<SWAGroupSet>(sliding_window_size, 1, device_pools, host_pool, disk_pool);
    group_set->initialize(group_set_id, std::move(topology), group_ids);
    return group_set;
}

std::shared_ptr<const CacheTopology>
BenchmarkFixture::createTopology(const std::vector<std::pair<std::string, rtp_llm::CacheGroupType>>& group_specs,
                                 const std::vector<size_t>& layer_stride_bytes_per_group,
                                 const std::vector<size_t>& layer_counts_per_group) {
    RTP_LLM_CHECK(group_specs.size() == layer_stride_bytes_per_group.size());
    RTP_LLM_CHECK(layer_counts_per_group.empty() || layer_counts_per_group.size() == group_specs.size());

    std::vector<GroupBase> groups;
    std::vector<LayerBase> layers;
    groups.reserve(group_specs.size());
    layers.reserve(group_specs.size());

    int next_layer_id = 0;
    for (size_t i = 0; i < group_specs.size(); ++i) {
        const auto& [tag, type]  = group_specs[i];
        const size_t layer_count = layer_counts_per_group.empty() ? 1 : layer_counts_per_group[i];

        GroupBase group_base;
        group_base.tag    = tag;
        auto spec         = std::make_shared<MHAKVCacheSpec>();
        spec->tag         = tag;
        group_base.spec   = spec;
        group_base.policy = defaultCacheGroupPolicy(type);
        if (type == rtp_llm::CacheGroupType::SWA) {
            group_base.policy.sliding_window_size = 128;
        }
        group_base.block_num                 = 0;
        group_base.local_kv_head_num         = 1;
        group_base.seq_size_per_block        = 1;
        group_base.kernel_seq_size_per_block = 1;
        group_base.kv_block_stride_bytes     = layer_stride_bytes_per_group[i];
        group_base.kv_scale_stride_bytes     = 0;
        for (size_t l = 0; l < layer_count; ++l) {
            group_base.layer_ids.push_back(next_layer_id);
            LayerBase layer;
            layer.layer_id = next_layer_id;
            layer.group_tags.push_back(tag);
            layers.push_back(std::move(layer));
            ++next_layer_id;
        }
        groups.push_back(std::move(group_base));
    }

    return CacheTopology::create(std::move(groups), std::move(layers));
}

std::unique_ptr<BlockTreeCache> BenchmarkFixture::createCache(std::vector<GroupSetPtr> group_sets,
                                                              bool                     enable_host,
                                                              bool                     enable_disk,
                                                              size_t                   task_pool_size,
                                                              double                   device_watermark_ratio,
                                                              double                   host_watermark_ratio) {
    BlockTreeCacheConfig config;
    config.enable_device_cache = true;
    config.enable_host_cache   = enable_host;
    config.enable_disk_cache   = enable_disk;
    config.task_pool_size      = static_cast<int>(task_pool_size);
    // Event-driven eviction: insert commits trigger checkWatermark, which
    // demotes overflow down to the per-tier ratio watermark (0.0 = disabled).
    if (device_watermark_ratio > 0.0) {
        config.watermark_device.ratio = device_watermark_ratio;
    }
    if (host_watermark_ratio > 0.0) {
        config.watermark_host.ratio = host_watermark_ratio;
    }

    auto engine     = std::make_shared<PerRankBlockTransferEngine>(group_sets);
    auto dispatcher = std::make_unique<BlockTransferDispatcher>(engine);
    // Watermark eviction submits one async demote task per victim in a burst
    // (a single insert commit can exceed the watermark by thousands of nodes);
    // the queue must absorb that burst or task submission blocks while holding the
    // cache mutex, deadlocking against task completion.
    auto task_pool =
        std::make_unique<BlockTreeTaskPool>(config.task_pool_size, 1000000, "BlockTreeCacheBenchmarkTaskPool");
    auto tree = std::make_unique<BlockTree>(group_sets);

    auto cache =
        std::make_unique<BlockTreeCache>(std::move(tree), config, nullptr, std::move(dispatcher), std::move(task_pool));

    if (!cache->init()) {
        throw std::runtime_error("Failed to initialize BlockTreeCache");
    }
    return cache;
}

ResourceBudget BenchmarkFixture::preflightResources(const ModelProfile& profile,
                                                    const std::string&  payload_mode,
                                                    size_t              tree_node_count,
                                                    size_t              transfer_concurrency,
                                                    const std::string&  host_memory_type,
                                                    const std::string&  disk_path,
                                                    double              max_device_memory_fraction) {
    ResourceBudget budget;

    size_t total_payload_per_coordinate = 0;
    for (const auto& gs : profile.group_sets) {
        total_payload_per_coordinate += gs.payload_bytes;
    }
    for (const auto& tag : profile.device_only_groups) {
        const auto* group = profile.findGroup(tag);
        if (group)
            total_payload_per_coordinate += group->group_payload_bytes;
    }

    if (payload_mode == "scaled") {
        total_payload_per_coordinate = computeScaledPayload(total_payload_per_coordinate);
    }

    size_t device_block_count = tree_node_count + std::max(static_cast<size_t>(64), tree_node_count / 10);
    size_t transfer_blocks    = 2 * transfer_concurrency;

    budget.estimated_device_bytes =
        static_cast<int64_t>((device_block_count + transfer_blocks) * total_payload_per_coordinate);
    budget.estimated_host_bytes      = static_cast<int64_t>((host_memory_type == "pinned" ? device_block_count / 2 : 0)
                                                       * total_payload_per_coordinate);
    budget.estimated_peak_host_bytes = budget.estimated_host_bytes;
    budget.estimated_request_template_bytes = static_cast<int64_t>(tree_node_count * 64);

    int device = 0;
    cudaSetDevice(device);
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    budget.available_device_bytes = static_cast<int64_t>(static_cast<double>(free_bytes) * max_device_memory_fraction);
    budget.available_host_or_cgroup_bytes = getAvailableHostMemory();

    if (!disk_path.empty()) {
        budget.available_disk_bytes = getAvailableDiskSpace(disk_path);
    }

    budget.sufficient = true;
    if (budget.estimated_device_bytes > budget.available_device_bytes) {
        budget.sufficient = false;
    }
    if (budget.estimated_host_bytes > budget.available_host_or_cgroup_bytes) {
        budget.sufficient = false;
    }

    return budget;
}

size_t BenchmarkFixture::scaleStride(size_t original_stride) {
    return alignUp(std::max(kMinScaledStride, original_stride / kScaleFactor), kScaledAlignment);
}

size_t BenchmarkFixture::computeScaledPayload(size_t original_payload) {
    return alignUp(std::max(kMinScaledStride, original_payload / kScaleFactor), kScaledAlignment);
}

void BenchmarkFixture::setPhaseTime(const std::string& phase, int64_t ns) {
    if (phase == "bootstrap")
        timing_.bootstrap_ns = ns;
    else if (phase == "profile_load")
        timing_.profile_load_ns = ns;
    else if (phase == "allocation")
        timing_.allocation_ns = ns;
    else if (phase == "setup")
        timing_.setup_ns = ns;
    else if (phase == "warmup")
        timing_.warmup_ns = ns;
    else if (phase == "measured")
        timing_.measured_ns = ns;
    else if (phase == "sync_drain")
        timing_.sync_drain_ns = ns;
    else if (phase == "teardown")
        timing_.teardown_ns = ns;
}

std::string fileSha256(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return "";

    SHA256_CTX ctx;
    SHA256_Init(&ctx);

    char buffer[8192];
    while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
        SHA256_Update(&ctx, buffer, file.gcount());
    }

    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_Final(hash, &ctx);

    char hex[SHA256_DIGEST_LENGTH * 2 + 1];
    for (size_t i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        std::snprintf(hex + i * 2, 3, "%02x", hash[i]);
    }
    return std::string(hex, SHA256_DIGEST_LENGTH * 2);
}

}  // namespace rtp_llm::benchmark
