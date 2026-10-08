#pragma once

#include <array>
#include <torch/extension.h>

namespace rtp_llm {

// Request rows stay on device; group-indexed fields use physical cache group IDs.
struct LinearReplayInputs {
    torch::Tensor slot_ids;
    torch::Tensor slot_generations;
    torch::Tensor active_block_ids;
    torch::Tensor prev_accept_lengths;
    torch::Tensor history_valid_lengths;
    torch::Tensor history_epochs;
    torch::Tensor verify_epochs;
    torch::Tensor init_kinds;
    torch::Tensor state_read_block_ids;
    torch::Tensor anchor_processed_lengths;

    std::array<torch::Tensor*, 10> tensors() {
        return {&slot_ids,
                &slot_generations,
                &active_block_ids,
                &prev_accept_lengths,
                &history_valid_lengths,
                &history_epochs,
                &verify_epochs,
                &init_kinds,
                &state_read_block_ids,
                &anchor_processed_lengths};
    }

    LinearReplayInputs slice(int64_t start, int64_t count) const {
        auto result = *this;
        for (auto* tensor : result.tensors()) {
            *tensor = tensor->narrow(tensor->dim() == 2 ? 1 : 0, start, count);
        }
        return result;
    }

    LinearReplayInputs padToBatch(int64_t physical_batch) const {
        const int64_t logical_batch = slot_ids.numel();
        TORCH_CHECK(physical_batch >= logical_batch, "cannot shrink LINEAR replay request rows");
        auto result = *this;
        for (auto* tensor : result.tensors()) {
            TORCH_CHECK(tensor->defined() && (tensor->dim() == 1 || tensor->dim() == 2),
                        "LINEAR replay metadata must have one or two dimensions");
            const int64_t dim = tensor->dim() - 1;
            TORCH_CHECK(tensor->size(dim) == logical_batch, "LINEAR replay request counts disagree");
            if (physical_batch == logical_batch) {
                continue;
            }
            const int64_t fill = tensor == &result.slot_ids || tensor == &result.active_block_ids
                                         || tensor == &result.state_read_block_ids ? -1 : 0;
            auto shape = tensor->sizes().vec();
            shape[dim] = physical_batch - logical_batch;
            *tensor = torch::cat({*tensor, torch::full(shape, fill, tensor->options())}, dim);
        }
        // Keep the original logical views for finalize and request bookkeeping.
        return result;
    }

    static LinearReplayInputs allocate(int64_t batch, int64_t groups, torch::Device device = torch::kCUDA) {
        LinearReplayInputs result;
        const auto         i32          = torch::TensorOptions().dtype(torch::kInt32).device(device);
        const auto         i64          = i32.dtype(torch::kInt64);
        result.slot_ids                 = torch::empty({batch}, i32);
        result.slot_generations         = torch::empty({batch}, i64);
        result.active_block_ids         = torch::empty({groups, batch}, i32);
        result.prev_accept_lengths      = torch::empty({batch}, i32);
        result.history_valid_lengths    = torch::empty({batch}, i32);
        result.history_epochs           = torch::empty({batch}, i64);
        result.verify_epochs            = torch::empty({batch}, i64);
        result.init_kinds               = torch::empty({batch}, i32);
        result.state_read_block_ids     = torch::empty({groups, batch}, i32);
        result.anchor_processed_lengths = torch::empty({batch}, i32);
        return result;
    }
};

}  // namespace rtp_llm

namespace torch_ext {

struct LinearReplayLayerCache {
    torch::Tensor k;
    torch::Tensor u;
    torch::Tensor g;
    torch::Tensor conv_inputs;
    torch::Tensor slot_generations;
    torch::Tensor log_epochs;
    torch::Tensor valid_counts;
    torch::Tensor error_flags;
};

}  // namespace torch_ext
