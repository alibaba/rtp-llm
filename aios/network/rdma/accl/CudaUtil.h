#pragma once

#include "autil/Log.h"

namespace arpc {

/*
 * Cuda needs conditional compile, move all cuda codes here for better style and test
 */
class CudaUtil {
public:
    virtual ~CudaUtil() = default;

    // virtual for unittest
    virtual bool IsGPUMemory(void* p);
    bool         CudaMemcpyGPU2CPU(void* dst, void* src, uint32_t len);
    virtual bool CudaMemcpyGPU2GPU(void* dst, void* src, uint32_t len);

private:
#ifdef GOOGLE_CUDA
    bool cudaAvailable{true};
#endif

    AUTIL_LOG_DECLARE();
};

}  // namespace arpc