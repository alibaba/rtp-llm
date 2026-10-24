#pragma once

#include <torch/types.h>
#include <vector>

namespace torch_ext {

// Optional native extension for prefill-only, host-metadata models. Implementations
// must be request-local/thread-safe and must not access Python or GPU state. Input
// tensors are borrowed for this call; returned tensors own/retain their storage.
// Models opting in must disable graph/microbatch execution and retain host inputs.
class HostInputMetadataBuilder {
public:
    virtual ~HostInputMetadataBuilder() = default;
    virtual std::vector<torch::Tensor> build(const torch::Tensor& input_ids,
                                             const torch::Tensor& input_lengths,
                                             const torch::Tensor& text_mask,
                                             const torch::Tensor& cu_seqlens) const = 0;
};

}  // namespace torch_ext
