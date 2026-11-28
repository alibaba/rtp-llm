#include "rtp_llm/cpp/utils/TorchCudaOom.h"

#include <stdexcept>
#include <string>

#include <c10/util/Exception.h>
#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TEST(TorchCudaOomTest, RecognizesTypedAndBackendSpecificOomErrors) {
    try {
        C10_THROW_ERROR(OutOfMemoryError, "allocator marker");
    } catch (const std::exception& exception) {
        EXPECT_TRUE(isTorchCudaOom(exception));
    }

    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("CUDA out of memory")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("CUDA error: out of memory")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("[ERROR] CUDA runtime error: out of memory model.cc:123")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("[ERROR] CUDA runtime error: CUDA_ERROR_OUT_OF_MEMORY")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("[ERROR] CUDA runtime error: CUBLAS_STATUS_ALLOC_FAILED")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("cudaErrorMemoryAllocation")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("HIP out of memory")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("HIP error: out of memory")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("HIP runtime error: out of memory")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("[ROCm][ERROR] out of memory Assertion fail")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("hipErrorOutOfMemory")));
    EXPECT_TRUE(isTorchCudaOom(std::runtime_error("TrackerAllocator failed to allocate 1073741824 bytes")));
    EXPECT_TRUE(isTorchCudaOomMessage("pybind11 error: CUDA out of memory. Tried to allocate 2.00 GiB"));
    EXPECT_FALSE(isTorchCudaOomMessage("CUDA error: an illegal memory access was encountered"));
}

TEST(TorchCudaOomTest, RejectsUnrelatedExceptionsContainingGenericOomWords) {
    EXPECT_FALSE(isTorchCudaOom(std::runtime_error("database out of memory")));
    EXPECT_FALSE(isTorchCudaOom(std::runtime_error("host allocator out of memory")));
    EXPECT_FALSE(isTorchCudaOom(std::runtime_error("not out of memory")));
    EXPECT_FALSE(isTorchCudaOom(std::runtime_error("CUDA illegal memory access")));
}

TEST(TorchCudaOomTest, RetriesRecognizedOomOnlyWithSupportedCacheRecovery) {
    int operation_calls = 0;
    int retry_callbacks = 0;

    EXPECT_THROW(
        retryOnceOnTorchCudaOom(
            [&]() {
                ++operation_calls;
                throw std::runtime_error("HIP out of memory");
            },
            [&](const std::exception&) { ++retry_callbacks; }),
        std::runtime_error);

#if USING_CUDA
    EXPECT_TRUE(torchGpuOomRecoveryRetrySupported());
    EXPECT_EQ(operation_calls, 2);
    EXPECT_EQ(retry_callbacks, 1);
#else
    EXPECT_FALSE(torchGpuOomRecoveryRetrySupported());
    EXPECT_EQ(operation_calls, 1);
    EXPECT_EQ(retry_callbacks, 0);
#endif
}

TEST(TorchCudaOomTest, DoesNotRetryUnrelatedFailure) {
    int operation_calls = 0;
    int retry_callbacks = 0;

    EXPECT_THROW(
        retryOnceOnTorchCudaOom(
            [&]() {
                ++operation_calls;
                throw std::runtime_error("out of memory while parsing a database response");
            },
            [&](const std::exception&) { ++retry_callbacks; }),
        std::runtime_error);

    EXPECT_EQ(operation_calls, 1);
    EXPECT_EQ(retry_callbacks, 0);
}

TEST(TorchCudaOomTest, SuccessfulRecoveryIsCudaOnly) {
    int operation_calls = 0;
    int retry_callbacks = 0;
    auto operation       = [&]() {
        ++operation_calls;
        if (operation_calls == 1) {
            throw std::runtime_error("cudaErrorMemoryAllocation");
        }
    };
    auto before_retry = [&](const std::exception&) { ++retry_callbacks; };

#if USING_CUDA
    EXPECT_NO_THROW(retryOnceOnTorchCudaOom(operation, before_retry));
    EXPECT_EQ(operation_calls, 2);
    EXPECT_EQ(retry_callbacks, 1);
#else
    EXPECT_THROW(retryOnceOnTorchCudaOom(operation, before_retry), std::runtime_error);
    EXPECT_EQ(operation_calls, 1);
    EXPECT_EQ(retry_callbacks, 0);
#endif
}

}  // namespace
}  // namespace rtp_llm
