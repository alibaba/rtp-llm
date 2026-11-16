#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>
#include <unordered_map>

namespace rtp_llm {

namespace {

at::cuda::CUDAStream getNoBlockCopyStream(int device_index) {
    static thread_local std::unordered_map<int, at::cuda::CUDAStream> streams;
    auto stream = streams.find(device_index);
    if (stream == streams.end()) {
        stream = streams.emplace(device_index,
                                 at::cuda::getStreamFromPool(/*isHighPriority=*/false, device_index)).first;
    }
    return stream->second;
}

void execNoBlockCopyImpl(const MultiCopyParams& params, const DeviceHostCopyExecutionContext* context) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

    if (params.multi_src.empty()) {
        return;
    }
    int copy_device = -1;
    if (context != nullptr) {
        if (context->owner == nullptr || context->deviceIndex() < 0) {
            throw std::invalid_argument("invalid PPU copy device context");
        }
        copy_device = context->deviceIndex();
    }
    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        for (const auto* tensor : {&params.multi_src[i], &params.multi_dst[i]}) {
            if (!tensor->is_cuda()) {
                continue;
            }
            const int tensor_device = tensor->get_device();
            if (copy_device >= 0 && copy_device != tensor_device) {
                throw std::invalid_argument("PPU copy tensors and context must use one device");
            }
            copy_device = tensor_device;
        }
    }
    if (copy_device < 0) {
        copy_device = c10::cuda::current_device();
    }
    // Transfer-pool threads do not inherit the model thread's current device.
    // Keep both the selected device and the cached stream tied to the payload.
    c10::cuda::CUDAGuard device_guard(copy_device);
    auto stream = getNoBlockCopyStream(copy_device).stream();

    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        check_cuda_value(cudaMemcpyAsync(params.multi_dst[i].data_ptr(),
                                         params.multi_src[i].data_ptr(),
                                         params.multi_src[i].nbytes(),
                                         cudaMemcpyDefault,
                                         stream));
    }
    check_cuda_value(cudaStreamSynchronize(stream));
    check_cuda_error();
}

}  // namespace

std::shared_ptr<DeviceHostCopyStreams> acquireDeviceHostCopyStreams(int device_index) {
    if (device_index < 0) {
        throw std::invalid_argument("invalid PPU copy stream device");
    }
    return std::make_shared<DeviceHostCopyStreams>(device_index, 0, 0);
}

void execNoBlockCopy(const MultiCopyParams& params) {
    execNoBlockCopyImpl(params, nullptr);
}

void execNoBlockCopy(const MultiCopyParams& params, const DeviceHostCopyExecutionContext& context) {
    execNoBlockCopyImpl(params, &context);
}

void warmupNoBlockCopy() {}

// PPU has no batched/staged fast path; callers use per-item copies.
BatchedMemoryCopyStatus execBatchedMemoryCopy(const BatchedMemoryCopyParams& params,
                                              const DeviceHostCopyExecutionContext&) {
    return params.tiles.empty() ? BatchedMemoryCopyStatus::SUCCESS : BatchedMemoryCopyStatus::NOT_SUPPORTED;
}

bool execStagedMemoryCopy(const StagedMemoryCopyParams& params, StagedMemoryCopyScratch*) {
    return params.tiles.empty();
}

void releaseStagedMemoryCopyScratch(StagedMemoryCopyScratch&) {}

}  // namespace rtp_llm
