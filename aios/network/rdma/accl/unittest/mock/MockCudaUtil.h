#pragma once

#include "aios/network/rdma/accl/CudaUtil.h"
#include "unittest/unittest.h"

namespace arpc {

class MockCudaUtil: public CudaUtil {

public:
    virtual ~MockCudaUtil() = default;

    MOCK_METHOD1(IsGPUMemory, bool(void*));
    MOCK_METHOD3(CudaMemcpyGPU2GPU, bool(void*, void*, uint32_t));
};

}  // namespace arpc