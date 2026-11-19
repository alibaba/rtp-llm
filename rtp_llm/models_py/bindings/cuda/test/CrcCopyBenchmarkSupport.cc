#include "rtp_llm/models_py/bindings/cuda/test/CrcCopyBenchmarkSupport.h"

#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/cuda/test/MainStagedCopyBaseline.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <numeric>
#include <limits>
#include <sstream>

namespace rtp_llm::crc_copy_benchmark {
namespace {
CacheConfig flashCacheConfig() {
    // Flash-0731: this benchmark disables speculative decoding and measures only the 43 target layers.
    ModelConfig model;
    model.num_layers                          = 43;
    model.hidden_size                         = 4096;
    model.attn_config.head_num                = 64;
    model.attn_config.kv_head_num             = 1;
    model.attn_config.size_per_head           = 512;
    model.attn_config.rope_head_dim           = 64;
    model.attn_config.sliding_window          = 128;
    model.attn_config.indexer_head_dim        = 128;
    model.attn_config.indexer_head_num        = 64;
    model.attn_config.indexer_topk            = 512;
    model.attn_config.o_groups                = 8;
    model.attn_config.o_lora_rank             = 1024;
    model.attn_config.tokens_per_block        = 128;
    model.attn_config.kernel_tokens_per_block = 128;
    model.attn_config.kv_cache_dtype          = KvCacheDataType::FP8;
    model.attn_config.layer_compress_ratios   = {0, 0};
    for (int layer = 2; layer < 43; ++layer)
        model.attn_config.layer_compress_ratios.push_back(layer % 2 == 0 ? 4 : 128);
    model.hybrid_attention_config.enable_hybrid_attention = true;
    test::setDsv4KvCacheSpecs(model, model.attn_config.layer_compress_ratios);

    ParallelismConfig parallel;
    parallel.role_type                          = RoleType::PREFILL;
    parallel.tp_size                            = 4;
    parallel.world_size                         = 4;
    parallel.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    parallel.prefill_cp_config.kv_cache_sharded = true;
    parallel.prefill_cp_config.prefill_cp_size  = 4;
    return CacheConfigCreator::createWarmupConfig(model, parallel, /*gen_num_per_cycle=*/0);
}

Layout deriveLayout(const CacheConfig& config, bool full) {
    Layout result;
    result.name            = full ? "full" : "prefill_cp4_no_spec_swa";
    const auto type        = full ? CacheGroupType::FULL : CacheGroupType::SWA;
    size_t     pool_offset = 0, member = 0;
    for (const auto& group : config.topology().groups()) {
        if (!group.policy.enable_prefix_reuse || group.policy.group_type != type)
            continue;
        const auto  pool        = DeviceBlockPoolConfigHelper::createConfigForGroup(config, group);
        const auto& layer_ids   = config.layerIdsForGroup(group.tag);
        size_t      local_layer = 0;
        for (const auto& memory : pool.memory_layouts) {
            for (size_t layer = 0; layer < memory.layer_num; ++layer, ++local_layer) {
                const auto append = [&](size_t offset, size_t width) {
                    if (!width)
                        return;
                    if (offset % pool.physical_block_count != 0)
                        throw std::runtime_error("physical pool offset cannot be scaled by capacity");
                    result.geometry.push_back({group.tag,
                                               member,
                                               local_layer,
                                               layer_ids.at(local_layer),
                                               width,
                                               pool_offset + offset / pool.physical_block_count + layer * width});
                    result.sizes.push_back(width);
                };
                append(memory.kv_cache_offset_bytes, memory.kv_block_stride_bytes);
                append(memory.kv_scale_offset_bytes, memory.kv_scale_stride_bytes);
            }
        }
        if (local_layer != layer_ids.size() || pool.total_size_bytes % pool.physical_block_count != 0)
            throw std::runtime_error("physical layer/pool geometry mismatch");
        pool_offset += pool.total_size_bytes / pool.physical_block_count;
        ++member;
    }
    auto sorted = result.geometry;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.offset_per_pool_block < b.offset_per_pool_block;
    });
    size_t covered = 0;
    for (const auto& tile : sorted) {
        if (tile.offset_per_pool_block != covered)
            throw std::runtime_error("physical pool has a gap or overlapping copy tiles");
        covered += tile.bytes;
    }
    if (covered == 0 || covered != pool_offset || covered != result.payload())
        throw std::runtime_error("physical tiles do not cover the reusable backing");
    return result;
}
}  // namespace

size_t Layout::payload() const {
    return std::accumulate(sizes.begin(), sizes.end(), size_t(0));
}

std::string Layout::geometryJson() const {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < geometry.size(); ++i) {
        const auto& tile = geometry[i];
        if (i)
            os << ',';
        os << "{\"tag\":\"" << tile.tag << "\",\"member\":" << tile.member << ",\"local_layer\":" << tile.local_layer
           << ",\"model_layer\":" << tile.model_layer << ",\"bytes\":" << tile.bytes
           << ",\"offset_per_pool_block\":" << tile.offset_per_pool_block << '}';
    }
    os << ']';
    return os.str();
}

const Layout& deepSeekV4FlashLayout(bool full) {
    static const auto config      = flashCacheConfig();
    static const auto full_layout = deriveLayout(config, true);
    static const auto swa_layout  = deriveLayout(config, false);
    return full ? full_layout : swa_layout;
}

size_t maximumLayoutTiles() {
    return std::max(deepSeekV4FlashLayout(true).sizes.size(), deepSeekV4FlashLayout(false).sizes.size());
}

size_t maximumLayoutPayload() {
    return std::max(deepSeekV4FlashLayout(true).payload(), deepSeekV4FlashLayout(false).payload());
}

namespace {
struct CopyTile {
    void*  host;
    void*  device;
    size_t bytes;
    size_t member;
    size_t local_layer;
};

StagedMemoryCopyParams stagedParams(const std::vector<CopyTile>& tiles, void* host_base, int device, bool evict) {
    StagedMemoryCopyParams params;
    params.host_base    = host_base;
    params.device_index = device;
    params.direction    = evict ? StagedMemoryCopyDirection::D2H : StagedMemoryCopyDirection::H2D;
    params.tiles.reserve(tiles.size());
    params.host_segments.reserve(tiles.size());
    for (const auto& tile : tiles) {
        if (params.host_bytes > std::numeric_limits<size_t>::max() - 15)
            throw std::overflow_error("staged copy alignment overflow");
        const size_t offset = (params.host_bytes + 15) & ~size_t{15};
        if (tile.bytes > std::numeric_limits<size_t>::max() - offset)
            throw std::overflow_error("staged copy size overflow");
        params.tiles.push_back({tile.device, offset, tile.bytes});
        params.host_bytes = offset + tile.bytes;
        if (!params.host_segments.empty()) {
            auto& prev = params.host_segments.back();
            if (static_cast<uint8_t*>(prev.host) + prev.bytes == tile.host && prev.host_offset + prev.bytes == offset) {
                prev.bytes += tile.bytes;
                continue;
            }
        }
        params.host_segments.push_back({tile.host, offset, tile.bytes});
    }
    return params;
}
}  // namespace

struct FrameworkCopyPlan::Impl {
    void*                 host_base{nullptr};
    size_t                payload_bytes{0};
    size_t                capacity_bytes{0};
    std::vector<CopyTile> tiles;

    Impl(const std::vector<CrcCopyItem>& items, const Layout& layout) {
        if (!items.empty()) {
            host_base      = items.front().host;
            payload_bytes  = items.front().payload_bytes;
            capacity_bytes = items.front().capacity_bytes;
        }
        tiles.reserve(items.size() * layout.geometry.size());
        for (const auto& item : items) {
            if (item.tiles.size() != layout.geometry.size())
                throw std::runtime_error("framework plan does not match derived physical geometry");
            for (size_t i = 0; i < item.tiles.size(); ++i) {
                const auto& tile = item.tiles[i];
                tiles.push_back({static_cast<uint8_t*>(item.host) + tile.offset,
                                 tile.device,
                                 tile.bytes,
                                 layout.geometry[i].member,
                                 layout.geometry[i].local_layer});
            }
        }
    }
};

FrameworkCopyPlan::FrameworkCopyPlan(const std::vector<CrcCopyItem>& items, const Layout& layout):
    impl_(std::make_unique<Impl>(items, layout)) {}
FrameworkCopyPlan::~FrameworkCopyPlan() = default;

uintptr_t FrameworkCopyPlan::touchMetadata() const {
    uintptr_t sum = reinterpret_cast<uintptr_t>(impl_->host_base) ^ impl_->payload_bytes ^ impl_->capacity_bytes;
    for (const auto& tile : impl_->tiles)
        sum += reinterpret_cast<uintptr_t>(tile.host) ^ reinterpret_cast<uintptr_t>(tile.device) ^ tile.bytes
               ^ tile.member ^ tile.local_layer;
    return sum;
}

struct FrameworkCopies::Impl {
    const int                              device;
    std::shared_ptr<DeviceHostCopyStreams> copy_streams;
    StagedMemoryCopyScratch                staged;
    main_baseline::StagedCopy              main_staged;
    at::cuda::CUDAStream                   stream;

    explicit Impl(int device):
        device(device),
        copy_streams(acquireDeviceHostCopyStreams(device)),
        stream(at::cuda::getStreamFromPool(false, device)) {}

    ~Impl() {
        releaseStagedMemoryCopyScratch(staged);
    }
};

FrameworkCopies::FrameworkCopies(int device) {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    impl_ = std::make_unique<Impl>(device);
}

FrameworkCopies::~FrameworkCopies() = default;

void FrameworkCopies::copyBatch(const FrameworkCopyPlan& plan, bool evict) {
    BatchedMemoryCopyParams params;
    params.device_index = impl_->device;
    params.direction    = evict ? DeviceHostCopyDirection::D2H : DeviceHostCopyDirection::H2D;
    params.tiles.reserve(plan.impl_->tiles.size());
    for (const auto& tile : plan.impl_->tiles) {
        params.tiles.push_back({evict ? tile.host : tile.device, evict ? tile.device : tile.host, tile.bytes});
    }
    const DeviceHostCopyExecutionContext context{impl_->copy_streams, params.direction};
    if (execBatchedMemoryCopy(params, context) != BatchedMemoryCopyStatus::SUCCESS)
        throw std::runtime_error("CUDA 1D batch copy failed");
}

void FrameworkCopies::copyStaged(const FrameworkCopyPlan& plan, bool evict) {
    const auto params = stagedParams(plan.impl_->tiles, plan.impl_->host_base, impl_->device, evict);
    if (!execStagedMemoryCopy(params, &impl_->staged))
        throw std::runtime_error("staged copy failed");
}

void FrameworkCopies::copyMainStaged(const FrameworkCopyPlan& plan, bool evict) {
    const auto params = stagedParams(plan.impl_->tiles, plan.impl_->host_base, impl_->device, evict);
    if (!impl_->main_staged.copy(params))
        throw std::runtime_error("main staged copy failed");
}

const char* mainStagedSourceCommit() {
    return main_baseline::kSourceCommit;
}

cudaStream_t FrameworkCopies::copy3dStream() const {
    return impl_->stream.stream();
}

}  // namespace rtp_llm::crc_copy_benchmark
