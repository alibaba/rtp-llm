#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyStrategy.h"

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

#if USING_CUDA
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace rtp_llm {

StrategyResult Cuda3DBatchDeviceHostCopyStrategy::tryExecute(const std::vector<HostBufferView>&     hosts,
                                                             const std::vector<TransferDescriptor>& descriptors,
                                                             const std::vector<const GroupSet*>&    group_sets) {
#if USING_CUDA && !USE_PPU && CUDART_VERSION >= 13000 && CUDART_VERSION < 14000
    static const bool supported = [] {
        int driver_version  = 0;
        int runtime_version = 0;
        return cudaDriverGetVersion(&driver_version) == cudaSuccess
               && cudaRuntimeGetVersion(&runtime_version) == cudaSuccess && driver_version >= 13000
               && runtime_version >= 13000 && runtime_version < 14000;
    }();
    if (!supported) {
        return StrategyResult::notApplicable();
    }

    const bool                       d2h          = descriptors.front().target_tier != Tier::DEVICE;
    int                              device_index = -1;
    std::vector<cudaMemcpy3DBatchOp> ops;
    for (size_t i = 0; i < descriptors.size(); ++i) {
        const auto& group     = *group_sets[i];
        const auto& templates = group.copy3DTemplates();
        const auto& pools     = group.devicePools();
        if (templates.empty()) {
            return StrategyResult::notApplicable();
        }
        const size_t payload = group.payloadBytes();
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
            if (block < 0 || static_cast<size_t>(block) >= pools[t.member_index]->memoryLayouts().front().block_num) {
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

    c10::cuda::CUDAGuard guard(device_index);
    // Per-thread CUDA stream avoids sharing a copy stream across concurrent executor workers.
    auto                         stream = cudaStreamPerThread;
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
