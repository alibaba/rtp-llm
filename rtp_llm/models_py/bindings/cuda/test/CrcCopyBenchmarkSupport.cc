#include "rtp_llm/models_py/bindings/cuda/test/CrcCopyBenchmarkSupport.h"

#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"
#include "rtp_llm/cpp/cache/CacheConfigCreator.h"
#include "rtp_llm/cpp/cache/DeviceBlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <numeric>
#include <sstream>

namespace rtp_llm::crc_copy_benchmark {
namespace {
void requireDone(const StrategyResult& result, const char* name) {
    if (result.status != StrategyStatus::DONE) {
        throw std::runtime_error(std::string(name) + " did not complete; fallback is forbidden; strategy_status="
                                 + std::to_string(static_cast<int>(result.status))
                                 + " copy_status=" + std::to_string(static_cast<int>(result.copy_status)));
    }
}

CacheConfig proCacheConfig() {
    ModelConfig model;
    model.num_layers                          = 61;
    model.hidden_size                         = 7168;
    model.attn_config.head_num                = 128;
    model.attn_config.kv_head_num             = 1;
    model.attn_config.size_per_head           = 512;
    model.attn_config.rope_head_dim           = 64;
    model.attn_config.sliding_window          = 128;
    model.attn_config.indexer_head_dim        = 128;
    model.attn_config.indexer_head_num        = 64;
    model.attn_config.indexer_topk            = 1024;
    model.attn_config.o_groups                = 16;
    model.attn_config.o_lora_rank             = 1024;
    model.attn_config.tokens_per_block        = 128;
    model.attn_config.kernel_tokens_per_block = 128;
    model.attn_config.kv_cache_dtype          = KvCacheDataType::FP8;
    model.attn_config.layer_compress_ratios   = {128, 128};
    for (int layer = 2; layer < 61; ++layer)
        model.attn_config.layer_compress_ratios.push_back(layer % 2 == 0 ? 4 : 128);
    model.hybrid_attention_config.enable_hybrid_attention = true;
    test::setDsv4KvCacheSpecs(model, model.attn_config.layer_compress_ratios);

    ParallelismConfig parallel;
    parallel.role_type                          = RoleType::PREFILL;
    parallel.tp_size                            = 8;
    parallel.world_size                         = 8;
    parallel.prefill_cp_config.method           = CPRotateMethod::ALL_GATHER;
    parallel.prefill_cp_config.kv_cache_sharded = true;
    parallel.prefill_cp_config.prefill_cp_size  = 8;
    return CacheConfigCreator::createWarmupConfig(model, parallel, /*gen_num_per_cycle=*/0);
}

Layout deriveLayout(const CacheConfig& config, bool full) {
    Layout result;
    result.name            = full ? "full" : "prefill_cp8_no_spec_swa";
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

const Layout& deepSeekV4ProLayout(bool full) {
    static const auto config      = proCacheConfig();
    static const auto full_layout = deriveLayout(config, true);
    static const auto swa_layout  = deriveLayout(config, false);
    return full ? full_layout : swa_layout;
}

size_t maximumLayoutTiles() {
    return std::max(deepSeekV4ProLayout(true).sizes.size(), deepSeekV4ProLayout(false).sizes.size());
}

size_t maximumLayoutPayload() {
    return std::max(deepSeekV4ProLayout(true).payload(), deepSeekV4ProLayout(false).payload());
}

struct FrameworkCopyPlan::Impl {
    DeviceHostCopyPlan store;
    DeviceHostCopyPlan load;

    Impl(const std::vector<CrcCopyItem>& items, const Layout& layout) {
        store.device_to_host = true;
        load.device_to_host  = false;
        if (!items.empty()) {
            store.host = load.host = {items.front().host, items.front().payload_bytes, items.front().capacity_bytes};
        }
        size_t count = 0;
        for (const auto& item : items)
            count += item.tiles.size();
        store.copy_tiles.reserve(count);
        load.copy_tiles.reserve(count);
        for (const auto& item : items) {
            if (item.tiles.size() != layout.geometry.size())
                throw std::runtime_error("framework plan does not match derived physical geometry");
            for (size_t i = 0; i < item.tiles.size(); ++i) {
                const auto&        tile = item.tiles[i];
                DeviceHostCopyTile copy;
                copy.host_addr         = static_cast<unsigned char*>(item.host) + tile.offset;
                copy.device_addr       = tile.device;
                copy.host_offset       = tile.offset;
                copy.bytes             = tile.bytes;
                copy.device_index      = 0;
                copy.member_group_id   = layout.geometry[i].member;
                copy.local_layer_index = layout.geometry[i].local_layer;
                store.copy_tiles.push_back(copy);
                load.copy_tiles.push_back(copy);
            }
        }
    }
};

FrameworkCopyPlan::FrameworkCopyPlan(const std::vector<CrcCopyItem>& items, const Layout& layout):
    impl_(std::make_unique<Impl>(items, layout)) {}
FrameworkCopyPlan::~FrameworkCopyPlan() = default;

uintptr_t FrameworkCopyPlan::touchMetadata() const {
    uintptr_t sum = 0;
    for (const auto* plan : {&impl_->store, &impl_->load}) {
        sum += plan->device_to_host ^ plan->group_set_id ^ reinterpret_cast<uintptr_t>(plan->host.base)
               ^ plan->host.payload_bytes ^ plan->host.capacity_bytes;
        for (const auto& tile : plan->copy_tiles)
            sum += reinterpret_cast<uintptr_t>(tile.host_addr) ^ reinterpret_cast<uintptr_t>(tile.device_addr)
                   ^ tile.host_offset ^ tile.bytes ^ uintptr_t(tile.device_index) ^ tile.member_group_id
                   ^ tile.local_layer_index;
    }
    return sum;
}

struct FrameworkCopies::Impl {
    CudaBatchDeviceHostCopyStrategy batch;
    StagedSmDeviceHostCopyStrategy  staged;
    at::cuda::CUDAStream            stream;

    explicit Impl(int device): stream(at::cuda::getStreamFromPool(false, device)) {}
};

FrameworkCopies::FrameworkCopies(int device) {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    impl_ = std::make_unique<Impl>(device);
}

FrameworkCopies::~FrameworkCopies() = default;

void FrameworkCopies::copyBatch(const FrameworkCopyPlan& plan, bool store) {
    requireDone(impl_->batch.tryExecute(store ? plan.impl_->store : plan.impl_->load, DeviceHostCopyOptions{}),
                "CUDA 1D batch");
}

void FrameworkCopies::copyStaged(const FrameworkCopyPlan& plan, bool store) {
    requireDone(impl_->staged.tryExecute(store ? plan.impl_->store : plan.impl_->load, DeviceHostCopyOptions{}),
                "rebased production staged without CRC");
}

cudaStream_t FrameworkCopies::copy3dStream() const {
    return impl_->stream.stream();
}

}  // namespace rtp_llm::crc_copy_benchmark
