#include "aios/network/rdma/accl/CudaUtil.h"

#ifdef GOOGLE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#endif

namespace arpc {

AUTIL_LOG_SETUP(arpc, CudaUtil);

bool CudaUtil::IsGPUMemory(void* p) {

#ifdef GOOGLE_CUDA
    if (!cudaAvailable) {
        return false;
    }

    cudaPointerAttributes attr;
    auto                  result = cudaPointerGetAttributes(&attr, p);
    if (result == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
        return true;
    }
    if (result == cudaErrorStubLibrary || result == cudaErrorInvalidDevice) {  // environment do not have gpu, eg:
        cudaAvailable = false;
        AUTIL_LOG(INFO, "cuda not available, ret is %d", result);
    }
#endif
    return false;  // not compile with cuda, consider this is not a GPU Memory
}

bool CudaUtil::CudaMemcpyGPU2CPU(void* dst, void* src, uint32_t len) {
#ifdef GOOGLE_CUDA
    if (!cudaAvailable) {
        return false;
    }
    return cudaMemcpy(dst, src, len, cudaMemcpyDeviceToHost) == cudaSuccess;
#else
    return false;
#endif
}

bool CudaUtil::CudaMemcpyGPU2GPU(void* dst, void* src, uint32_t len) {
#ifdef GOOGLE_CUDA
    if (!cudaAvailable) {
        return false;
    }
    return cudaMemcpy(dst, src, len, cudaMemcpyDeviceToDevice) == cudaSuccess;
#else
    return false;
#endif
}

}  // namespace arpc