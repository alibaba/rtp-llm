#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"

#include <limits>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

#if USING_CUDA
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace rtp_llm {

bool Cuda3DBatchDeviceHostCopyStrategy::isSupported() {
#if USING_CUDA && !USE_PPU && CUDART_VERSION >= 13000 && CUDART_VERSION < 14000
    static const bool supported = [] {
        int driver_version  = 0;
        int runtime_version = 0;
        return cudaDriverGetVersion(&driver_version) == cudaSuccess
               && cudaRuntimeGetVersion(&runtime_version) == cudaSuccess && driver_version >= 13000
               && runtime_version >= 13000 && runtime_version < 14000;
    }();
    return supported;
#else
    return false;
#endif
}

Cuda3DBatchDeviceHostCopyStrategy::Cuda3DBatchDeviceHostCopyStrategy(
    const std::vector<std::shared_ptr<GroupSet>>& group_sets, bool enabled) {
#if USING_CUDA && !USE_PPU && CUDART_VERSION >= 13000 && CUDART_VERSION < 14000
    if (!enabled || group_sets.empty() || !isSupported()) {
        return;
    }
    for (const auto& group : group_sets) {
        if (group->crcEnabled()) {
            continue;
        }
        auto templates = initializeCopy3DTemplates(*group);
        if (!templates.empty()) {
            templates_.emplace(group.get(), std::move(templates));
        }
    }
#endif
}

std::vector<DeviceHostCopyTemplate>
Cuda3DBatchDeviceHostCopyStrategy::initializeCopy3DTemplates(const GroupSet& group) {
    const auto&                         pools = group.devicePools();
    const auto&                         tags  = group.groupTags();
    std::vector<DeviceHostCopyTemplate> templates;
    size_t                              host_offset  = 0;
    int                                 device_index = -1;
    for (size_t member = 0; member < pools.size(); ++member) {
        const auto& pool       = pools[member];
        const auto& group_base = group.group(tags[member]);
        const auto  layer_ids  = group.topologyPtr()->layerIdsForGroup(tags[member]);
        if (!pool->getBaseAddress() || pool->deviceIndex() < 0 || group_base.kvScaleStrideBytes() != 0) {
            return {};
        }
        if (device_index < 0) {
            device_index = pool->deviceIndex();
        } else if (device_index != pool->deviceIndex()) {
            return {};
        }
        const size_t width    = group_base.kvBlockStrideBytes();
        const size_t capacity = pool->getTotalSizeBytes();
        if (reinterpret_cast<uintptr_t>(pool->getBaseAddress()) > UINTPTR_MAX - capacity) {
            return {};
        }
        size_t layer_begin = 0;
        for (const auto& layout : pool->config().memory_layouts) {
            if (width == 0 || layout.block_num == 0 || layout.layer_num == 0 || layout.kv_block_stride_bytes != width
                || layout.kv_scale_stride_bytes != 0 || layout.hasScale()
                || layout.block_num > std::numeric_limits<size_t>::max() / width || layer_begin > layer_ids.size()
                || layout.layer_num > layer_ids.size() - layer_begin) {
                return {};
            }
            const size_t pitch  = layout.block_num * width;
            const size_t offset = layout.kv_cache_offset_bytes;
            if (offset > capacity || layout.layer_num > (capacity - offset) / pitch
                || layout.layer_num > layout.kv_block_pool_size_bytes / pitch || host_offset > group.payloadBytes()
                || layout.layer_num > (group.payloadBytes() - host_offset) / width) {
                return {};
            }
            auto* base = static_cast<uint8_t*>(pool->getBaseAddress()) + offset;
            templates.push_back({member, base, host_offset, width, layout.layer_num, pitch});
            host_offset += width * layout.layer_num;
            layer_begin += layout.layer_num;
        }
        if (layer_begin != layer_ids.size()) {
            return {};
        }
    }
    if (host_offset != group.payloadBytes()) {
        return {};
    }
    return templates;
}

StrategyResult Cuda3DBatchDeviceHostCopyStrategy::tryExecute(const std::vector<HostBufferView>&     hosts,
                                                             const std::vector<TransferDescriptor>& descriptors,
                                                             const std::vector<const GroupSet*>&    group_sets,
                                                             const DeviceHostCopyExecutionContext&  context) {
#if USING_CUDA && !USE_PPU && CUDART_VERSION >= 13000 && CUDART_VERSION < 14000
    const bool                       d2h          = descriptors.front().target_tier != Tier::DEVICE;
    int                              device_index = -1;
    std::vector<cudaMemcpy3DBatchOp> ops;
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const auto& group = *group_sets[i];
        const auto  found = templates_.find(&group);
        if (found == templates_.end()) {
            return StrategyResult::notApplicable();
        }
        const auto&  templates = found->second;
        const auto&  pools     = group.devicePools();
        const size_t payload   = group.payloadBytes();
        if (!isValidHostBufferView(hosts[i], payload, payload)
            || reinterpret_cast<uintptr_t>(hosts[i].base) > UINTPTR_MAX - payload) {
            return StrategyResult::failed(hosts[i].base == nullptr ? TransferStatus::DEVICE_IO_ERROR :
                                                                     TransferStatus::INVALID_ARGS);
        }
        const auto& blocks             = descriptors[i].blocksAt(Tier::DEVICE);
        const int   group_device_index = pools.front()->deviceIndex();
        if (device_index >= 0 && device_index != group_device_index) {
            return StrategyResult::notApplicable();
        }
        device_index = group_device_index;
        for (const auto& t : templates) {
            const auto block = blocks[t.member_index];
            if (block < 0
                || static_cast<size_t>(block) >= pools[t.member_index]->config().memory_layouts.front().block_num) {
                return StrategyResult::failed(TransferStatus::INVALID_ARGS);
            }
            auto* host   = static_cast<uint8_t*>(hosts[i].base) + t.host_offset;
            auto* device = static_cast<uint8_t*>(t.device_base) + static_cast<size_t>(block) * t.width_bytes;
            cudaMemcpy3DBatchOp op{};
            op.src.type               = cudaMemcpyOperandTypePointer;
            op.dst.type               = cudaMemcpyOperandTypePointer;
            op.src.op.ptr.ptr         = d2h ? device : host;
            op.dst.op.ptr.ptr         = d2h ? host : device;
            op.src.op.ptr.rowLength   = d2h ? t.device_pitch : t.width_bytes;
            op.dst.op.ptr.rowLength   = d2h ? t.width_bytes : t.device_pitch;
            op.src.op.ptr.layerHeight = t.layer_count;
            op.dst.op.ptr.layerHeight = t.layer_count;
            op.extent                 = {t.width_bytes, t.layer_count, 1};
            op.srcAccessOrder         = cudaMemcpySrcAccessOrderStream;
            ops.push_back(op);
        }
    }

    if (!context.owner || context.deviceIndex() != device_index || context.stream() == 0
        || context.direction != (d2h ? DeviceHostCopyDirection::D2H : DeviceHostCopyDirection::H2D)) {
        return StrategyResult::failed(TransferStatus::INVALID_ARGS);
    }
    c10::cuda::CUDAGuard         guard(device_index);
    auto                         stream = reinterpret_cast<cudaStream_t>(context.stream());
    std::unique_lock<std::mutex> submit_lock(cudaBatchSubmitMutex(device_index));
    const auto                   submit = cudaMemcpy3DBatchAsync(ops.size(), ops.data(), 0, stream);
    submit_lock.unlock();
    // Drain even a failed submission: a prefix may already be in flight.
    const auto completion = cudaStreamSynchronize(stream);
    RTP_LLM_CHECK_WITH_INFO(completion == cudaSuccess,
                            "3D batch completion failed: device=%d error=%s",
                            device_index,
                            cudaGetErrorString(completion));
    if (submit != cudaSuccess) {
        RTP_LLM_LOG_WARNING("3D batch submission failed: device=%d ops=%zu error=%s",
                            device_index,
                            ops.size(),
                            cudaGetErrorString(submit));
        return StrategyResult::failed(TransferStatus::DEVICE_IO_ERROR);
    }
    return StrategyResult::done();
#else
    return StrategyResult::notApplicable();
#endif
}

}  // namespace rtp_llm
