#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/TorchCudaOom.h"

namespace rtp_llm {
namespace {

constexpr size_t kMiB             = 1024 * 1024;
constexpr size_t kGraphAllocBytes = 512 * kMiB;
constexpr size_t kCachedBytes     = 1024 * kMiB;
constexpr size_t kTargetFreeBytes = 1 * kMiB;
constexpr size_t kMaxFillBytes    = 4ULL * 1024 * kMiB;
constexpr size_t kMinFillBytes    = 64 * 1024;

struct ExceptionSnapshot {
    const std::exception* address = nullptr;
    std::string           type;
    std::string           what;
    std::string           backtrace;
};

ExceptionSnapshot snapshotException(const std::exception& exception) {
    ExceptionSnapshot snapshot{&exception, typeid(exception).name(), exception.what(), {}};
    if (const auto* c10_error = dynamic_cast<const c10::Error*>(&exception)) {
        if (const auto& backtrace = c10_error->backtrace()) {
            snapshot.backtrace = backtrace->get();
        }
    }
    return snapshot;
}

void expectBareRethrowPreservesException(const std::exception_ptr& exception, const ExceptionSnapshot& expected) {
    ExceptionSnapshot inner_catch;
    ExceptionSnapshot outer_catch;
    try {
        try {
            std::rethrow_exception(exception);
        } catch (const std::exception& caught) {
            inner_catch = snapshotException(caught);
            throw;
        }
    } catch (const std::exception& caught) {
        outer_catch = snapshotException(caught);
    }

    EXPECT_EQ(inner_catch.address, expected.address);
    EXPECT_EQ(outer_catch.address, expected.address);
    EXPECT_EQ(inner_catch.type, expected.type);
    EXPECT_EQ(outer_catch.type, expected.type);
    EXPECT_EQ(inner_catch.what, expected.what);
    EXPECT_EQ(outer_catch.what, expected.what);
    EXPECT_EQ(inner_catch.backtrace, expected.backtrace);
    EXPECT_EQ(outer_catch.backtrace, expected.backtrace);
}

class CudaMemoryPressure final {
public:
    ~CudaMemoryPressure() {
        for (auto it = allocations_.rbegin(); it != allocations_.rend(); ++it) {
            (void)cudaFree(*it);
        }
        cuda_graph::graphEmptyCache();
    }

    void fillUntil(size_t target_free_bytes) {
        size_t max_allocation = kMaxFillBytes;
        while (freeBytes() > target_free_bytes + kMinFillBytes) {
            const size_t request    = std::min(max_allocation, freeBytes() - target_free_bytes);
            void*        allocation = nullptr;
            const auto   error      = cudaMalloc(&allocation, request);
            if (error == cudaSuccess) {
                allocations_.push_back(allocation);
                max_allocation = kMaxFillBytes;
                continue;
            }

            (void)cudaGetLastError();
            if (request <= kMinFillBytes) {
                break;
            }
            max_allocation = std::max(kMinFillBytes, request / 2);
        }
    }

    static size_t freeBytes() {
        size_t free_bytes  = 0;
        size_t total_bytes = 0;
        C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        return free_bytes;
    }

private:
    std::vector<void*> allocations_;
};

bool manualTestEnabled() {
    const char* value = std::getenv("RTP_LLM_RUN_REAL_CUDA_GRAPH_OOM_TEST");
    return value != nullptr && std::string(value) == "1";
}

}  // namespace

TEST(CudaGraphReplayRealOomManualTest, RetryAndBareRethrowPreserveRealCudaApiOomBacktraces) {
    if (!manualTestEnabled()) {
        GTEST_SKIP() << "Set RTP_LLM_RUN_REAL_CUDA_GRAPH_OOM_TEST=1 and expose one dedicated GPU to run";
    }
    ASSERT_TRUE(torch::cuda::is_available());

    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    cuda_graph::graphEmptyCache();
    C10_CUDA_CHECK(cudaDeviceGraphMemTrim(device));
    cuda_graph::graphDeviceSynchronize();

    auto                stream = cuda_graph::graphGetStreamFromPool(/*is_high_priority=*/false);
    at::cuda::CUDAGraph graph;
    {
        at::cuda::CUDAStreamGuard stream_guard(stream);
        void*                     graph_allocation = nullptr;
        cuda_graph::graphCaptureBegin(graph, cuda_graph::graphPoolHandle());
        C10_CUDA_CHECK(cudaMallocAsync(&graph_allocation, kGraphAllocBytes, stream.stream()));
        C10_CUDA_CHECK(cudaMemsetAsync(graph_allocation, 0, kGraphAllocBytes, stream.stream()));
        C10_CUDA_CHECK(cudaFreeAsync(graph_allocation, stream.stream()));
        graph.capture_end();
    }

    const size_t free_before_cache = CudaMemoryPressure::freeBytes();
    ASSERT_GT(free_before_cache, kCachedBytes + kGraphAllocBytes);

    CudaMemoryPressure pressure;
    auto               cached = torch::empty({static_cast<int64_t>(kCachedBytes)},
                               torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    cached                    = torch::Tensor();
    cuda_graph::graphDeviceSynchronize();

    const size_t reserved  = cuda_graph::graphReservedBytes();
    const size_t allocated = cuda_graph::graphAllocatedBytes();
    ASSERT_GE(reserved - allocated, kCachedBytes)
        << "Torch did not retain the released block; this scenario cannot validate emptyCache recovery";

    pressure.fillUntil(kTargetFreeBytes);
    const size_t free_before_replay = CudaMemoryPressure::freeBytes();
    ASSERT_LT(free_before_replay, kGraphAllocBytes)
        << "Failed to create enough real device-memory pressure to force graph replay OOM";

    ExceptionSnapshot  malloc_oom;
    std::exception_ptr malloc_oom_ptr;
    try {
        void* unexpected_allocation = nullptr;
        C10_CUDA_CHECK(cudaMalloc(&unexpected_allocation, kGraphAllocBytes));
        (void)cudaFree(unexpected_allocation);
        FAIL() << "cudaMalloc unexpectedly succeeded under forced device-memory pressure";
    } catch (const std::exception& exception) {
        malloc_oom     = snapshotException(exception);
        malloc_oom_ptr = std::current_exception();
    }
    ASSERT_TRUE(malloc_oom_ptr);
    ASSERT_TRUE(isTorchCudaOom(*malloc_oom.address));
    ASSERT_FALSE(malloc_oom.backtrace.empty());
    std::cout << "[REAL_OOM][cudaMalloc] original exception and backtrace:\n" << malloc_oom.what << std::endl;
    expectBareRethrowPreservesException(malloc_oom_ptr, malloc_oom);

    ExceptionSnapshot  graph_oom;
    std::exception_ptr graph_oom_ptr;
    size_t             free_after_empty_cache = 0;
    try {
        graph.replay();
    } catch (const std::exception& exception) {
        if (!isTorchCudaOom(exception)) {
            throw;
        }
        graph_oom     = snapshotException(exception);
        graph_oom_ptr = std::current_exception();
        cuda_graph::graphEmptyCache();
        free_after_empty_cache = CudaMemoryPressure::freeBytes();
    }

    ASSERT_TRUE(graph_oom_ptr) << "cudaGraphLaunch did not return a synchronous OOM under real memory pressure";
    ASSERT_TRUE(isTorchCudaOom(*graph_oom.address));
    ASSERT_FALSE(graph_oom.backtrace.empty());
    std::cout << "[REAL_OOM][cudaGraphLaunch] original exception and backtrace:\n" << graph_oom.what << std::endl;
    expectBareRethrowPreservesException(graph_oom_ptr, graph_oom);
    EXPECT_GE(free_after_empty_cache, free_before_replay + kCachedBytes);
    EXPECT_NO_THROW(graph.replay());
    EXPECT_NO_THROW(cuda_graph::graphDeviceSynchronize());
}

}  // namespace rtp_llm
