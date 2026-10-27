#include "aios/network/rdma/accl/CudaUtil.h"

#include "unittest/unittest.h"

#ifdef GOOGLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using namespace ::testing;

namespace arpc {

class CudaUtilTest: public TESTBASE {};

TEST_F(CudaUtilTest, test_IsGPUMemory) {
    CudaUtil cudaUtil;

    void* p = nullptr;
    ASSERT_FALSE(cudaUtil.IsGPUMemory(p));

    p = (void*)0x1243;
    ASSERT_FALSE(cudaUtil.IsGPUMemory(p));

    p = malloc(1024);
    ASSERT_TRUE(p != nullptr);
    ASSERT_FALSE(cudaUtil.IsGPUMemory(p));
    free(p);

#ifdef GOOGLE_CUDA
    ASSERT_EQ(cudaSuccess, cudaMalloc(&p, 1024));
    ASSERT_TRUE(cudaUtil.IsGPUMemory(p));
    ASSERT_EQ(cudaSuccess, cudaFree(p));

    ASSERT_EQ(cudaSuccess, cudaMallocHost(&p, 1024));
    ASSERT_FALSE(cudaUtil.IsGPUMemory(p));
    ASSERT_EQ(cudaSuccess, cudaFreeHost(p));
#endif
}

TEST_F(CudaUtilTest, test_CudaMemcpyGPU2CPU) {
    CudaUtil cudaUtil;

    const int size = 1024;
    void*     src  = nullptr;
    void*     dst  = nullptr;

#ifdef GOOGLE_CUDA
    ASSERT_EQ(cudaSuccess, cudaMalloc(&src, size));
    ASSERT_EQ(cudaSuccess, cudaMemset(src, 'a', size));

    ASSERT_FALSE(cudaUtil.CudaMemcpyGPU2CPU(dst, src, size));

    dst = malloc(1024);
    ASSERT_TRUE(dst != nullptr);
    ASSERT_TRUE(cudaUtil.CudaMemcpyGPU2CPU(dst, src, size));

    for (int i = 0; i < size; i++) {
        ASSERT_EQ('a', ((char*)dst)[i]);
    }

    free(dst);
    ASSERT_EQ(cudaSuccess, cudaFree(src));

#else
    ASSERT_FALSE(cudaUtil.CudaMemcpyGPU2CPU(dst, src, size));

#endif
}

TEST_F(CudaUtilTest, test_CudaMemcpyGPU2GPU) {
    CudaUtil cudaUtil;

    const int size = 1024;
    void*     src  = nullptr;
    void*     dst  = nullptr;

#ifdef GOOGLE_CUDA
    ASSERT_EQ(cudaSuccess, cudaMalloc(&src, size));
    ASSERT_EQ(cudaSuccess, cudaMemset(src, 'a', size));

    void* dstDevice = nullptr;
    ASSERT_FALSE(cudaUtil.CudaMemcpyGPU2GPU(dstDevice, src, size));

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dstDevice, size));
    ASSERT_TRUE(cudaUtil.CudaMemcpyGPU2GPU(dstDevice, src, size));

    dst = malloc(1024);
    ASSERT_TRUE(dst != nullptr);
    ASSERT_TRUE(cudaUtil.CudaMemcpyGPU2CPU(dst, src, size));

    for (int i = 0; i < size; i++) {
        ASSERT_EQ('a', ((char*)dst)[i]);
    }
    free(dst);
    ASSERT_EQ(cudaSuccess, cudaFree(dstDevice));
    ASSERT_EQ(cudaSuccess, cudaFree(src));

#else
    ASSERT_FALSE(cudaUtil.CudaMemcpyGPU2GPU(dst, src, size));

#endif
}
}  // namespace arpc