#pragma once

#include <queue>
#include <vector>

#include <torch/extension.h>

namespace rtp_llm {

struct TensorHolder {
    static constexpr size_t kReleasedHoldRounds = 2;

    std::vector<torch::Tensor>             tensors;
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
        releaseDeferred();
    }

    // Advance the same retention window, leaving expired references to the caller.
    std::vector<torch::Tensor> releaseDeferred() {
        // Move the current hold set into clear_tensors. Keep two released
        // rounds alive so tensors created for async H2D/D2H copies or CUDA
        // kernels remain retained until the third release point.
        clear_tensors.push(std::move(tensors));
        tensors.clear();
        std::vector<torch::Tensor> expired;
        if (clear_tensors.size() > kReleasedHoldRounds) {
            expired = std::move(clear_tensors.front());
            clear_tensors.pop();
        }
        return expired;
    }
};

}  // namespace rtp_llm
