#include "rtp_llm/cpp/multimodal_processor/CudaPinnedCopy.h"

#if USING_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <map>
#include <mutex>

namespace rtp_llm {
namespace {

c10::cuda::CUDAStream multimodalCopyStream(c10::DeviceIndex device) {
    static std::mutex                                        mutex;
    static std::map<c10::DeviceIndex, c10::cuda::CUDAStream> streams;
    std::lock_guard<std::mutex>                              lock(mutex);
    if (const auto stream = streams.find(device); stream != streams.end()) {
        return stream->second;
    }
    const c10::cuda::CUDAGuard guard(device);
    cudaStream_t               raw = nullptr;
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&raw, cudaStreamNonBlocking));
    try {
        const auto stream = c10::cuda::getStreamFromExternal(raw, device);
        // Keep a dedicated stream alive with the CUDA context. Never borrow a
        // pool slot that may also carry model or collective work.
        streams.emplace(device, stream);
        return stream;
    } catch (...) {
        (void)cudaStreamDestroy(raw);
        throw;
    }
}

}  // namespace

torch::Tensor copyCudaTensorToPinnedHost(const torch::Tensor& tensor, const c10::cuda::CUDAStream& producer) {
    TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == producer.device_index(),
                "multimodal copy producer does not match the tensor device");
    const c10::cuda::CUDAGuard guard(tensor.device());
    auto options = torch::TensorOptions().dtype(tensor.scalar_type()).device(torch::kCPU).pinned_memory(true);
    // An explicit pinned destination avoids the pageable allocation made by
    // Tensor::to(options, /*non_blocking=*/false).
    auto host = tensor.is_non_overlapping_and_dense() ? torch::empty_strided(tensor.sizes(), tensor.strides(), options) :
                                                        torch::empty(tensor.sizes(), options);
    const auto copy_stream = multimodalCopyStream(tensor.get_device());
    at::cuda::CUDAEvent ready;
    ready.record(producer);
    ready.block(copy_stream);
    const c10::cuda::CUDAStreamGuard copy_guard(copy_stream);
    c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), copy_stream);
    host.copy_(tensor, /*non_blocking=*/true);
    at::cuda::CUDAEvent copied;
    copied.record(copy_stream);
    copied.synchronize();
    return host;
}

}  // namespace rtp_llm
#endif
