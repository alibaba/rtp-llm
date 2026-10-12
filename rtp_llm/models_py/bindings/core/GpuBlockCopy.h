#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <torch/torch.h>

namespace rtp_llm {

struct GpuBlockCopyPlane {
    torch::Tensor blocks;
    size_t        copy_bytes;
};

// GPU copy for a single cache group. Owns the backing tensors, layout descriptors,
// and two event-protected staging slots. No per-layer CPU expansion per step.
class GpuBlockCopy {
public:
    // Non-CUDA builds return nullptr; callers retain their existing copy path.
    static std::unique_ptr<GpuBlockCopy> create(std::vector<GpuBlockCopyPlane> planes);
    ~GpuBlockCopy();

    // Enqueue before forward on the current CUDA stream, outside graph capture.
    // The caller can release/mutate cpu_mappings on return. KV storage must not
    // be reused until dependent GPU work completes. Destinations must be unique
    // and disjoint from other mappings' sources (beam tail copy-on-write).
    void copy(const torch::Tensor& cpu_mappings);

private:
    struct Impl;
    explicit GpuBlockCopy(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
