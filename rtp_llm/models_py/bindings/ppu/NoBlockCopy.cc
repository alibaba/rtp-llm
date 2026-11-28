#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/cuda/cuda_host_utils.h"

#include <cuda_runtime.h>
#include <c10/core/DeviceGuard.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <utility>

namespace rtp_llm {

namespace {

cudaError_t copyPpuResult(const char* phase, cudaError_t result) {
    if (auto probe = no_block_copy_detail::cuda_result_for_test) {
        return static_cast<cudaError_t>(probe(phase, static_cast<int>(result)));
    }
    return result;
}

void drainPpuCopy(cudaStream_t stream, const char* phase) noexcept {
    const auto result = copyPpuResult(phase, cudaStreamSynchronize(stream));
    if (result != cudaSuccess) {
        // Completion cannot be established. Do not let an exception release
        // buffers which an earlier asynchronous copy may still be accessing.
        std::fprintf(
            stderr, "PPU copy completion unconfirmed phase=%s error_code=%d\n", phase, static_cast<int>(result));
        std::fflush(stderr);
        std::abort();
    }
}

void copyBlocks(const MultiCopyParams& params, const DeviceHostCopyExecutionContext* context) {
    RTP_LLM_CHECK_WITH_INFO(params.multi_src.size() == params.multi_dst.size(),
                            "multi_src.size(%zu) != multi_dst.size(%zu)",
                            params.multi_src.size(),
                            params.multi_dst.size());

    // These are byte copies. Reject an invalid later item before submitting
    // any earlier copy in the same request.
    for (size_t i = 0; i < params.multi_src.size(); ++i) {
        const auto& src = params.multi_src[i];
        const auto& dst = params.multi_dst[i];
        TORCH_CHECK(src.defined() && dst.defined(), "PPU copy has undefined tensor at index ", i);
        TORCH_CHECK((src.is_cpu() || src.is_cuda()) && (dst.is_cpu() || dst.is_cuda()),
                    "PPU copy requires CPU or CUDA tensors at index ", i);
        TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "PPU byte copy requires contiguous tensors at index ", i);
        TORCH_CHECK(dst.nbytes() >= src.nbytes(), "PPU copy destination is too small at index ", i);
        if (context) {
            const bool h2d = context->direction == DeviceHostCopyDirection::H2D;
            TORCH_CHECK(h2d ? (src.is_cpu() && dst.is_cuda()) : (src.is_cuda() && dst.is_cpu()),
                        "PPU copy direction does not match context at index ", i);
            TORCH_CHECK(context->owner && context->deviceIndex() == (h2d ? dst.get_device() : src.get_device()),
                        "PPU copy device does not match context at index ", i);
        }
    }

    c10::OptionalDeviceGuard    device_guard;
    std::optional<cudaStream_t> stream;
    int                         device_index = -1;
    const auto                  drain        = [&](const char* phase) {
        if (auto pending = std::exchange(stream, std::nullopt)) {
            drainPpuCopy(*pending, phase);
        }
    };
    try {
        for (size_t i = 0; i < params.multi_src.size(); ++i) {
            const auto& src   = params.multi_src[i];
            const auto& dst   = params.multi_dst[i];
            const auto  bytes = src.nbytes();
            if (bytes == 0) {
                continue;
            }
            if (!src.is_cuda() && !dst.is_cuda()) {
                drain("generic_completion");
                std::memmove(dst.data_ptr(), src.data_ptr(), bytes);
                continue;
            }
            const int copy_device = getCopyDevice(dst, src);
            if (src.is_cpu() || dst.is_cpu()) {
                // Host KV slots can be interior, non-owning byte ranges. The
                // PPU runtime rejects side-stream DMA for some of these ranges.
                // As in the PPU TCP transport, complete on the device's current
                // stream before publishing the slot. Preserve raw byte-copy
                // semantics even if the two tensors have different dtypes.
                drain("generic_completion");
                device_guard.reset_device(torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(copy_device)));
                auto src_bytes = src.reshape({-1}).view(torch::kUInt8);
                auto dst_bytes = dst.reshape({-1}).view(torch::kUInt8).narrow(0, 0, src_bytes.numel());
                dst_bytes.copy_(src_bytes, /*non_blocking=*/false);
                continue;
            }
            if (!stream || copy_device != device_index) {
                drain("generic_completion");
                device_guard.reset_device(torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(copy_device)));
                device_index = copy_device;
                stream       = getNoBlockCopyStream(device_index).stream();
            }
            check_cuda_value(copyPpuResult(
                "generic_submit", cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), bytes, cudaMemcpyDeviceToDevice, *stream)));
        }
        drain("generic_completion");
        if (device_index >= 0) {
            check_cuda_error();
        }
    } catch (...) {
        drain("generic_exception_drain");
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
