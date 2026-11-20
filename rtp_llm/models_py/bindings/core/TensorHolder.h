#pragma once

#include <memory>
#include <queue>
#include <vector>

#include <torch/extension.h>

namespace rtp_llm {

struct TensorHolder {
    static constexpr size_t kReleasedHoldRounds = 2;

    std::vector<torch::Tensor>              tensors;
    std::queue<std::vector<torch::Tensor>> clear_tensors;

    void hold_host(const torch::Tensor& tensor) {
        if (tensor.defined() && tensor.device().is_cpu()) {
            tensors.push_back(tensor);
        }
    }

    void hold(const torch::Tensor& tensor) {
        if (tensor.defined()) {
            tensors.push_back(tensor);
        }
    }

    void release() {
        collectCompleted();
        // Move the current hold set into clear_tensors. Keep two released
        // rounds alive so tensors created for async H2D/D2H copies or CUDA
        // kernels are not freed until the third release point.
        clear_tensors.push(std::move(tensors));
        tensors.clear();
        while (clear_tensors.size() > kReleasedHoldRounds) {
            clear_tensors.pop();
        }
    }

    // Release against the last consumer stream instead of counting CPU
    // iterations. The PyTorch allocators defer storage reuse until that
    // stream completes, including raw fused copies from pinned host buffers.
    void release(const torch::Stream& consumer_stream);

private:
    struct DeferredRelease;
    std::queue<std::shared_ptr<DeferredRelease>> deferred_releases_;
    void                                         collectCompleted();
};

}  // namespace rtp_llm
