#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace rtp_llm {

// These are submission primitives. Callers own the stream, ordering and proof
// of completion, including after a failed or partially submitted copy.
struct DeviceHostBatchSupport {
    int         driver_version{0};
    int         runtime_version{0};
    cudaError_t query_error{cudaSuccess};
    bool        available{false};
};

DeviceHostBatchSupport cudaDeviceHostBatchSupport() noexcept;
cudaError_t            submitCudaDeviceHostBatch(void**       destinations,
                                                 const void** sources,
                                                 size_t*      bytes,
                                                 size_t       count,
                                                 cudaStream_t stream);
cudaError_t submitCudaDeviceHostTile(void* destination, const void* source, size_t bytes, cudaStream_t stream) noexcept;

}  // namespace rtp_llm
