#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <unordered_map>

namespace rtp_llm {

namespace {

// Collect runtime evidence only after a failed enqueue; a preceding asynchronous
// failure may also be surfaced by these best-effort queries. Never retry here:
// copyBlocks must report the original error and drain the stream on unwind.
void logFailedCopyContext(const void* src, const void* dst, cudaStream_t stream) {
    int                   current_device = -1;
    unsigned int          stream_flags = 0;
    cudaPointerAttributes src_attrs{};
    cudaPointerAttributes dst_attrs{};
    const auto            device_rc = cudaGetDevice(&current_device);
    const auto            stream_rc = cudaStreamGetFlags(stream, &stream_flags);
    const auto            src_rc = cudaPointerGetAttributes(&src_attrs, src);
    const auto            dst_rc = cudaPointerGetAttributes(&dst_attrs, dst);
    RTP_LLM_LOG_ERROR("PPU copy runtime diagnostics: device_rc=%d current_device=%d "
                      "stream_flags_rc=%d stream_flags=%u "
                      "src_attr_rc=%d src_type=%d src_device=%d "
                      "dst_attr_rc=%d dst_type=%d dst_device=%d",
                      static_cast<int>(device_rc), device_rc == cudaSuccess ? current_device : -1,
                      static_cast<int>(stream_rc), stream_flags,
                      static_cast<int>(src_rc), src_rc == cudaSuccess ? static_cast<int>(src_attrs.type) : -1,
                      src_rc == cudaSuccess ? src_attrs.device : -1,
                      static_cast<int>(dst_rc), dst_rc == cudaSuccess ? static_cast<int>(dst_attrs.type) : -1,
                      dst_rc == cudaSuccess ? dst_attrs.device : -1);
}

at::cuda::CUDAStream getNoBlockCopyStream(int device) {
    // A worker can serve more than one device over its lifetime.
    static thread_local std::unordered_map<int, at::cuda::CUDAStream> streams;
    auto                                                              it = streams.find(device);
    if (it == streams.end()) {
        it = streams.emplace(device, at::cuda::getStreamFromPool(false, device)).first;
    }
    return it->second;
}

void copyBlocks(const MultiCopyParams& params, const DeviceHostCopyExecutionContext* context) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

    // Validate the whole submission before starting asynchronous work. These are
    // raw byte copies (not tensor broadcasting/casting); views must be contiguous.
    int  device           = -1;
    bool multiple_devices = false;
    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        const auto& src = params.multi_src[i];
        const auto& dst = params.multi_dst[i];
        TORCH_CHECK(src.defined() && dst.defined(), "PPU copy has undefined tensor at index ", i);
        TORCH_CHECK(
            src.is_contiguous() && dst.is_contiguous(), "PPU byte copy requires contiguous tensors at index ", i);
        TORCH_CHECK(dst.nbytes() >= src.nbytes(),
                    "PPU copy destination is too small at index ",
                    i,
                    ": src_bytes=",
                    src.nbytes(),
                    " dst_bytes=",
                    dst.nbytes());
        for (const auto* tensor : {&src, &dst}) {
            TORCH_CHECK(tensor->is_cpu() || tensor->is_cuda(), "PPU copy requires CPU or CUDA tensors");
            if (tensor->is_cuda()) {
                const int tensor_device = tensor->get_device();
                multiple_devices |= device >= 0 && device != tensor_device;
                device = tensor_device;
                if (context != nullptr) {
                    TORCH_CHECK(context->owner && context->deviceIndex() == tensor_device,
                                "PPU copy device does not match context");
                }
            }
        }
        if (context != nullptr) {
            TORCH_CHECK(context->owner, "PPU copy context has no owner");
            const bool h2d = context->direction == DeviceHostCopyDirection::H2D;
            TORCH_CHECK(h2d ? (src.is_cpu() && dst.is_cuda()) : (src.is_cuda() && dst.is_cpu()),
                        "PPU copy direction does not match context at index ",
                        i);
        }
    }
    if (context != nullptr && device >= 0) {
        TORCH_CHECK(context->deviceIndex() == device, "PPU copy device does not match context");
    }
    if (device < 0) {
        for (size_t i = 0; i < params.multi_src.size(); ++i) {
            if (params.multi_src[i].nbytes() != 0) {
                std::memmove(
                    params.multi_dst[i].data_ptr(), params.multi_src[i].data_ptr(), params.multi_src[i].nbytes());
            }
        }
        return;
    }

    // The no-context API historically accepts mixed-device batches. Keep its
    // current-device/default-kind behavior rather than turning that capability
    // into a new rejection while fixing the device-host staging path.
    if (multiple_devices) {
        device = at::cuda::current_device();
    }

    const c10::cuda::CUDAGuard guard(device);
    auto                       stream    = getNoBlockCopyStream(device).stream();
    bool                       submitted = false;
    try {
        for (size_t i = 0; i < params.multi_src.size(); ++i) {
            const auto& src = params.multi_src[i];
            const auto& dst = params.multi_dst[i];
            if (src.nbytes() == 0) {
                continue;
            }
            // For a CPU <-> single-device submission metadata already gives
            // the direction, including interior pool addresses. Retain the
            // original default-kind behavior for peer/device/mixed copies.
            const bool device_host = src.is_cuda() != dst.is_cuda();
            const auto kind        = !multiple_devices && device_host ?
                                         (src.is_cuda() ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice) :
                                         cudaMemcpyDefault;
            submitted              = true;
            const auto error       = cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), src.nbytes(), kind, stream);
            if (error != cudaSuccess) {
                RTP_LLM_LOG_ERROR("PPU byte copy failed index=%zu device=%d kind=%d src=%p dst=%p "
                                  "src_bytes=%zu dst_bytes=%zu stream=%p error=%s",
                                  i,
                                  device,
                                  static_cast<int>(kind),
                                  src.data_ptr(),
                                  dst.data_ptr(),
                                  src.nbytes(),
                                  dst.nbytes(),
                                  stream,
                                  cudaGetErrorString(error));
                logFailedCopyContext(src.data_ptr(), dst.data_ptr(), stream);
            }
            check_cuda_value(error);
        }
        check_cuda_value(cudaStreamSynchronize(stream));
    } catch (...) {
        // Earlier items may still reference pinned staging owned by the caller.
        // Drain before stack unwinding releases that storage, as CUDA does.
        if (submitted) {
            const auto error = cudaStreamSynchronize(stream);
            if (error != cudaSuccess) {
                RTP_LLM_LOG_ERROR(
                    "PPU copy drain failed device=%d stream=%p error=%s", device, stream, cudaGetErrorString(error));
            }
        }
        throw;
    }
}

}  // namespace

std::shared_ptr<DeviceHostCopyStreams> acquireDeviceHostCopyStreams(int device_index) {
    return std::make_shared<DeviceHostCopyStreams>(device_index, 0, 0);
}

void execNoBlockCopy(const MultiCopyParams& params) {
    copyBlocks(params, nullptr);
}

void execNoBlockCopy(const MultiCopyParams& params, const DeviceHostCopyExecutionContext& context) {
    copyBlocks(params, &context);
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
