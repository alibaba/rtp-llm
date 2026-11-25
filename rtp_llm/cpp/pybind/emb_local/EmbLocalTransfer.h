#pragma once

#include "rtp_llm/cpp/pybind/emb_local/EmbLocalIndex.h"

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>

namespace rtp_llm::emb_local {

class Ready {
public:
    explicit Ready(c10::cuda::CUDAStream stream): fence_(cudaEventDisableTiming | cudaEventBlockingSync) {
        fence_.record(stream);
    }

    void wait() const {
        fence_.synchronize();
    }

private:
    at::cuda::CUDAEvent fence_;
};

inline std::vector<at::Tensor> views(const Resident& object) {
    std::vector<at::Tensor> result;
    result.reserve(object.slices.size());
    for (const auto& slice : object.slices) {
        const auto& block  = object.storage[tierIndex(slice.placement)];
        const auto  device = slice.placement == Tier::Cpu ? at::Device(at::kCPU) : slice.original;
        if (slice.nbytes == 0) {
            result.push_back(at::empty(slice.sizes, at::TensorOptions().device(device).dtype(slice.type)));
        } else {
            result.push_back(
                block->bytes().narrow(0, slice.offset, slice.nbytes).view(slice.type).reshape(slice.sizes));
        }
    }
    return result;
}

// Both source and destination extents stay owned throughout the CUDA wait.
// Copies return synchronously to the request, with independent output tensors.
inline void copyAndWait(const std::vector<at::Tensor>&       from,
                        const std::vector<at::Tensor>&       to,
                        std::optional<c10::cuda::CUDAStream> stream,
                        const Blocks&                        source      = {},
                        const Blocks&                        destination = {}) {
    TORCH_CHECK(from.size() == to.size(), "local EMB copy layout mismatch");
    std::optional<c10::cuda::CUDAStreamGuard> guard;
    if (stream) {
        guard.emplace(*stream);
    }
    bool submitted = false;
    try {
        for (size_t index = 0; index < from.size(); ++index) {
            const auto& input  = from[index];
            const auto& output = to[index];
            const bool  cuda   = input.is_cuda() || output.is_cuda();
            TORCH_CHECK(!cuda || stream, "local EMB CUDA copies require a stream");
            if (cuda) {
                submitted = true;
                if (input.is_cuda()) {
                    input.record_stream(stream->unwrap());
                }
                if (output.is_cuda()) {
                    output.record_stream(stream->unwrap());
                }
            }
            output.copy_(input.detach(), /*non_blocking=*/cuda);
        }
        if (submitted) {
            Ready(*stream).wait();
        }
    } catch (...) {
        if (submitted) {
            // These blocks may still be on the failed stream. Dropping a
            // Tensor reference cannot make that exact slab interval safe.
            for (const auto* blocks : {&source, &destination}) {
                for (const auto& block : *blocks) {
                    if (block) {
                        block->quarantine();
                    }
                }
            }
        }
        throw;
    }
}

inline std::vector<at::Tensor> copyOut(const std::shared_ptr<const Resident>& object,
                                       std::optional<c10::cuda::CUDAStream>   stream) {
    // Allocate output on the stream that first writes it. The CUDA allocator
    // may reuse a block whose prior operations are ordered on that stream.
    std::optional<c10::cuda::CUDAStreamGuard> guard;
    if (stream) {
        guard.emplace(*stream);
    }
    auto                    from = views(*object);
    std::vector<at::Tensor> requested;
    requested.reserve(from.size());
    for (const auto& slice : object->slices) {
        requested.push_back(at::empty(slice.sizes, at::TensorOptions().device(slice.original).dtype(slice.type)));
    }
    copyAndWait(from, requested, stream, object->storage);
    return requested;
}

}  // namespace rtp_llm::emb_local
