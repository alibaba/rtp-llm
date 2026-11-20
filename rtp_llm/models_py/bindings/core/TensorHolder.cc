#include "rtp_llm/models_py/bindings/core/TensorHolder.h"

#if USING_CUDA
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace rtp_llm {

#if USING_CUDA
struct TensorHolder::DeferredRelease {
    std::vector<torch::Tensor> tensors;
    at::cuda::CUDAEvent        ready{cudaEventDisableTiming};

    ~DeferredRelease() {
        // Teardown is the only blocking path. A borrowed host allocation must
        // outlive its consumer even if the holder is destroyed between steps.
        if (!ready.query()) {
            ready.synchronize();
        }
    }
};
#endif

void TensorHolder::collectCompleted() {
#if USING_CUDA
    while (!deferred_releases_.empty() && deferred_releases_.front()->ready.query()) {
        deferred_releases_.pop();
    }
#endif
}

void TensorHolder::release(const torch::Stream& consumer_stream) {
#if USING_CUDA
    const c10::cuda::CUDAStream stream(consumer_stream);
    collectCompleted();
    std::vector<torch::Tensor> unregistered;
    const auto                 retire = [&stream, &unregistered](std::vector<torch::Tensor>& held) {
        for (const auto& tensor : held) {
            if (!tensor.defined() || tensor.numel() == 0) {
                continue;
            }
            const auto& storage = tensor.storage().data_ptr();
            if (tensor.is_cuda()) {
                c10::cuda::CUDACachingAllocator::recordStream(storage, stream);
            } else if (tensor.device().is_cpu()) {
                if (!at::cuda::CachingHostAllocator_recordEvent(tensor.data_ptr(), storage.get_context(), stream)) {
                    unregistered.push_back(tensor);
                }
            }
        }
        held.clear();
    };
    retire(tensors);
    while (!clear_tensors.empty()) {
        retire(clear_tensors.front());
        clear_tensors.pop();
    }
    if (!unregistered.empty()) {
        // External/pageable host storage cannot use PyTorch's allocator
        // tracking. Retain its owner until the GPU event completes instead
        // of draining the engine stream at every phase boundary.
        auto pending     = std::make_shared<DeferredRelease>();
        pending->tensors = std::move(unregistered);
        pending->ready.record(stream);
        deferred_releases_.push(std::move(pending));
    }
#else
    consumer_stream.synchronize();
    release();
#endif
}

}  // namespace rtp_llm
