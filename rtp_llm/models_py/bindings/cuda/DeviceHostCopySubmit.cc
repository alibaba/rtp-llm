#include "rtp_llm/models_py/bindings/cuda/DeviceHostCopySubmit.h"

#include <vector>

namespace rtp_llm {

DeviceHostBatchSupport cudaDeviceHostBatchSupport() noexcept {
    DeviceHostBatchSupport support;
#if CUDART_VERSION >= 12080
    constexpr int minimum_driver = CUDART_VERSION >= 13000 ? 13000 : 12080;
    support.query_error          = cudaDriverGetVersion(&support.driver_version);
    if (support.query_error != cudaSuccess || support.driver_version < minimum_driver) {
        return support;
    }
    support.query_error = cudaRuntimeGetVersion(&support.runtime_version);
    support.available   = support.query_error == cudaSuccess && support.runtime_version >= 12080
                        && (CUDART_VERSION >= 13000) == (support.runtime_version >= 13000);
#endif
    return support;
}

cudaError_t submitCudaDeviceHostBatch(void**       destinations,
                                    const void** sources,
                                    size_t*      bytes,
                                    size_t       count,
                                    cudaStream_t stream) {
#if CUDART_VERSION >= 12080
    cudaMemcpyAttributes attrs{};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t attr_idx      = 0;
#if CUDART_VERSION >= 13000
    return cudaMemcpyBatchAsync(destinations, sources, bytes, count, &attrs, &attr_idx, 1, stream);
#else
    std::vector<void*> mutable_sources;
    mutable_sources.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        mutable_sources.push_back(const_cast<void*>(sources[i]));
    }
    size_t fail_idx = 0;
    return cudaMemcpyBatchAsync(
        destinations, mutable_sources.data(), bytes, count, &attrs, &attr_idx, 1, &fail_idx, stream);
#endif
#else
    (void)destinations;
    (void)sources;
    (void)bytes;
    (void)count;
    (void)stream;
    return cudaErrorNotSupported;
#endif
}

cudaError_t submitCudaDeviceHostTile(void* destination, const void* source, size_t bytes, cudaStream_t stream) noexcept {
    return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDefault, stream);
}

}  // namespace rtp_llm
