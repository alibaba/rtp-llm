#include "rtp_llm/models_py/bindings/cuda/PagedKvCacheCopy.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace rtp_llm {

void copyPagedKvCacheAsync(const torch::Tensor&                       source,
                           const torch::Tensor&                       destination,
                           const std::vector<std::array<int64_t, 3>>& runs) {
    TORCH_CHECK(source.dim() == 3 && destination.dim() == 3, "page copies require three-dimensional tensors");
    TORCH_CHECK(source.is_contiguous() && destination.is_contiguous(), "page copies require contiguous tensors");
    TORCH_CHECK(source.scalar_type() == destination.scalar_type() && source.size(1) == destination.size(1)
                    && source.size(2) == destination.size(2),
                "page copies require matching dtype and page shape");
    TORCH_CHECK(source.is_cuda() != destination.is_cuda(), "page copies require one CUDA tensor");
    const auto& host = source.is_cuda() ? destination : source;
    const auto& gpu  = source.is_cuda() ? source : destination;
    TORCH_CHECK(host.device().is_cpu() && host.is_pinned(), "page copies require pinned host storage");
    const c10::cuda::CUDAGuard guard(gpu.device());
    const auto                 stream     = c10::cuda::getCurrentCUDAStream(gpu.get_device()).stream();
    const size_t               page_bytes = source.stride(0) * source.element_size();
    std::vector<void*>         destinations;
    std::vector<const void*>   sources;
    std::vector<size_t>        sizes;
    destinations.reserve(runs.size());
    sources.reserve(runs.size());
    sizes.reserve(runs.size());
    int64_t previous_destination_end = 0;
    for (const auto& run : runs) {
        const auto first = run[0], target = run[1], count = run[2];
        TORCH_CHECK(first >= 0 && target >= 0 && count > 0 && first <= source.size(0) && count <= source.size(0) - first
                        && target <= destination.size(0) && count <= destination.size(0) - target,
                    "page copy run exceeds tensor bounds");
        TORCH_CHECK(target >= previous_destination_end, "page copy destinations must be sorted and disjoint");
        previous_destination_end = target + count;
        sources.push_back(static_cast<const char*>(source.data_ptr()) + first * page_bytes);
        destinations.push_back(static_cast<char*>(destination.data_ptr()) + target * page_bytes);
        sizes.push_back(count * page_bytes);
    }
    if (runs.empty()) {
        return;
    }
    // Submit all fragments together, leaving the CPU free to enqueue the
    // owner layer while the copy engine fills the next layer's staging slot.
#if CUDART_VERSION >= 12080
    cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t attribute_index    = 0;
#if CUDART_VERSION >= 13000
    C10_CUDA_CHECK(cudaMemcpyBatchAsync(
        destinations.data(), sources.data(), sizes.data(), sizes.size(), &attributes, &attribute_index, 1, stream));
#else
    std::vector<void*> mutable_sources;
    mutable_sources.reserve(sources.size());
    for (const auto* source_pointer : sources) {
        mutable_sources.push_back(const_cast<void*>(source_pointer));
    }
    size_t failure_index = 0;
    C10_CUDA_CHECK(cudaMemcpyBatchAsync(destinations.data(),
                                        mutable_sources.data(),
                                        sizes.data(),
                                        sizes.size(),
                                        &attributes,
                                        &attribute_index,
                                        1,
                                        &failure_index,
                                        stream));
#endif
#else
    const auto direction = source.is_cuda() ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
    for (size_t i = 0; i < sizes.size(); ++i) {
        C10_CUDA_CHECK(cudaMemcpyAsync(destinations[i], sources[i], sizes[i], direction, stream));
    }
#endif
}

}  // namespace rtp_llm
