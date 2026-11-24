#pragma once

#include <ATen/core/Tensor.h>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

// The batch is immutable and readers receive independent CPU tensor copies.
class EmbHashTensorBatch {
public:
    explicit EmbHashTensorBatch(const std::vector<at::Tensor>& inputs) {
        hashes_.reserve(inputs.size());
        for (const auto& input : inputs) {
            hashes_.push_back(input.detach().to(input.options().dtype(at::kInt).device(at::kCPU),
                                                /*non_blocking=*/false,
                                                /*copy=*/true));
        }
    }

    std::optional<size_t> metadataRows(size_t budget) const {
        if (hashes_.empty()
            || std::any_of(hashes_.begin(), hashes_.end(), [](const auto& hash) { return hash.numel() == 0; })) {
            return {};
        }
        size_t rows = 0;
        for (const auto& hash : hashes_) {
            const auto count = static_cast<size_t>(hash.numel());
            if (count > budget - rows) {
                throw std::invalid_argument("multimodal metadata response exceeds row limit");
            }
            rows += count;
        }
        return rows;
    }

    std::vector<at::Tensor> cloneHashes() const {
        std::vector<at::Tensor> copies;
        copies.reserve(hashes_.size());
        for (const auto& hash : hashes_) {
            copies.push_back(hash.clone());
        }
        return copies;
    }

private:
    std::vector<at::Tensor> hashes_;
};

}  // namespace rtp_llm
