#pragma once

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

    void release(bool defer_reclaim = false) {
        // Finish any deferred reclamation left by an interrupted phase before
        // advancing the retention window again.
        reclaim();
        // Move the current hold set into clear_tensors. Keep two released
        // rounds alive so tensors created for async H2D/D2H copies or CUDA
        // kernels remain retained until the third release point.
        clear_tensors.push(std::move(tensors));
        tensors.clear();
        if (!defer_reclaim) {
            reclaim();
        }
    }

    // Reclaim only expired rounds; the two retained rounds stay alive.
    void reclaim() {
        while (clear_tensors.size() > kReleasedHoldRounds) {
            clear_tensors.pop();
        }
    }
};

}  // namespace rtp_llm
