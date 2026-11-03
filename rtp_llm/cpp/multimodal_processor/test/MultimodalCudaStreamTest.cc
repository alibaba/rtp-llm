#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"

#if USING_CUDA
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace rtp_llm {
namespace {

// No model, network, NCCL communicator, or request content is needed: one pending
// dependency on the model's default stream reproduces the unsafe RPC coupling.
// The memory wait is released by a CPU store, never by another CUDA API (which
// itself can wait on the driver lock held by a synchronous D2H).
class DefaultStreamGate {
public:
    ~DefaultStreamGate() {
        release();
        if (watchdog_.joinable()) {
            watchdog_.join();
        }
        if (host_ != nullptr) {
            cudaStreamSynchronize(nullptr);
            cudaFreeHost(host_);
        }
    }

    cudaError_t allocate() {
        void* allocation = nullptr;
        auto  result     = cudaHostAlloc(&allocation, sizeof(uint32_t), cudaHostAllocMapped);
        if (result != cudaSuccess) {
            return result;
        }
        host_ = static_cast<uint32_t*>(allocation);
        __atomic_store_n(host_, 0u, __ATOMIC_SEQ_CST);
        return cudaHostGetDevicePointer(&device_, host_, 0);
    }

    CUresult block() {
        // All allocation and lazy kernel initialization happens before this.
        auto result = cuStreamWaitValue32(nullptr, reinterpret_cast<CUdeviceptr>(device_), 1u, CU_STREAM_WAIT_VALUE_EQ);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        watchdog_ = std::thread([this]() {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!condition_.wait_for(lock, std::chrono::seconds(8), [this]() { return released_; })) {
                deadline_released_.store(true, std::memory_order_release);
                // Pinned mapped memory is coherent with the GPU. A direct host
                // store also recovers the pre-fix path without taking CUDA locks.
                __atomic_store_n(host_, 1u, __ATOMIC_SEQ_CST);
            }
        });
        return CUDA_SUCCESS;
    }

    bool deadlineReleased() const {
        return deadline_released_.load(std::memory_order_acquire);
    }

    bool isStillClosed() const {
        return __atomic_load_n(host_, __ATOMIC_SEQ_CST) == 0u;
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        if (host_ != nullptr) {
            __atomic_store_n(host_, 1u, __ATOMIC_SEQ_CST);
        }
        condition_.notify_all();
    }

private:
    uint32_t*               host_   = nullptr;
    void*                   device_ = nullptr;
    std::mutex              mutex_;
    std::condition_variable condition_;
    bool                    released_ = false;
    std::atomic<bool>       deadline_released_{false};
    std::thread             watchdog_;
};

class GpuOutputProcessor final: public MultimodalProcessor {
public:
    GpuOutputProcessor():
        MultimodalProcessor(py::object{}, MMModelConfig{true, {{1}}, false}, 32),
        embedding_(torch::empty({2, 64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))),
        extra_(torch::empty({2, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))),
        hashes_(torch::tensor({-11, -12}, torch::kInt32)) {}

    bool fail = false;

private:
    ErrorResult<MultimodalOutput>
    MultimodalEmbedding(const std::vector<MultimodalInput>, std::string, int64_t, grpc::ServerContext*) override {
        if (fail) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "intentional stream restoration test");
        }
        const auto   consumer_stream = c10::cuda::getCurrentCUDAStream(0);
        unsigned int consumer_flags  = 0;
        const auto   stream_status   = cudaStreamGetFlags(consumer_stream.stream(), &consumer_flags);
        if (stream_status != cudaSuccess) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, cudaGetErrorString(stream_status));
        }
        std::fprintf(stderr,
                     "MM CUDA fetch: stream_id=%lld handle=%p flags=%u\n",
                     static_cast<long long>(consumer_stream.id()),
                     reinterpret_cast<void*>(consumer_stream.stream()),
                     consumer_flags);
        std::fflush(stderr);
        // Real GPU writes on the stream selected by the production entry point.
        // The later CPU assertions must observe these writes, not stale storage.
        embedding_.fill_(17.0f);
        extra_.fill_(23.0f);
        MultimodalOutput output;
        output.mm_features       = {embedding_};
        output.mm_extra_input    = std::vector<torch::Tensor>{extra_};
        output.mm_feature_hashes = std::vector<torch::Tensor>{hashes_};
        return output;
    }

    torch::Tensor embedding_;
    torch::Tensor extra_;
    torch::Tensor hashes_;
};

std::shared_ptr<GenerateInput> makeInput() {
    auto input               = std::make_shared<GenerateInput>();
    input->input_ids         = torch::tensor({0, 1, 2}, torch::kInt32);
    input->multimodal_inputs = std::vector<MultimodalInput>{MultimodalInput("synthetic-gpu-output")};
    return input;
}

void assertHostReady(const std::shared_ptr<GenerateInput>& input) {
    ASSERT_TRUE(input->multimodal_features.has_value());
    ASSERT_TRUE(input->mm_extra_input.has_value());
    ASSERT_EQ(input->multimodal_features->size(), 1u);
    ASSERT_EQ(input->mm_extra_input->size(), 1u);
    const auto& embedding = input->multimodal_features->at(0);
    const auto& extra     = input->mm_extra_input->at(0);
    ASSERT_TRUE(embedding.device().is_cpu());
    ASSERT_TRUE(extra.device().is_cpu());
    ASSERT_TRUE(embedding.is_pinned());
    ASSERT_TRUE(extra.is_pinned());
    // Direct host reads intentionally perform no CUDA synchronization.
    for (int64_t i = 0; i < embedding.numel(); ++i) {
        ASSERT_EQ(embedding.data_ptr<float>()[i], 17.0f);
    }
    for (int64_t i = 0; i < extra.numel(); ++i) {
        ASSERT_EQ(extra.data_ptr<float>()[i], 23.0f);
    }
}

class MultimodalCudaStreamTest: public ::testing::TestWithParam<bool> {};

TEST_P(MultimodalCudaStreamTest, CompletesWhileDefaultStreamHasPendingDependency) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    c10::cuda::CUDAStreamGuard caller(c10::cuda::getDefaultCUDAStream(0));
    const auto                 original_stream = c10::cuda::getCurrentCUDAStream(0);
    DefaultStreamGate          gate;
    ASSERT_EQ(gate.allocate(), cudaSuccess);
    GpuOutputProcessor processor;
    {
        // Warm the exact GPU fill/copy kernels and caching allocator bins first.
        auto warm_input = makeInput();
        ASSERT_TRUE(processor.updateMultimodalFeatures(warm_input).ok());
        assertHostReady(warm_input);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    if (GetParam()) {
        // Cold pinned allocation must be safe too; a warm-only test would hide
        // cudaHostAlloc/registration synchronization in the production path.
        at::cuda::CachingHostAllocator_emptyCache();
    }
    auto       input       = makeInput();
    const auto gate_status = gate.block();
    if (gate_status == CUDA_ERROR_NOT_SUPPORTED) {
        GTEST_SKIP() << "device does not support CUDA stream memory operations";
    }
    ASSERT_EQ(gate_status, CUDA_SUCCESS);
    std::fprintf(stderr, "MM CUDA gate closed: pinned_allocator=%s\n", GetParam() ? "cold" : "warm");
    std::fflush(stderr);
    const auto result                   = processor.updateMultimodalFeatures(input);
    const bool completed_before_release = !gate.deadlineReleased() && gate.isStillClosed();
    const auto restored_stream          = c10::cuda::getCurrentCUDAStream(0);
    gate.release();
    EXPECT_TRUE(result.ok()) << result.ToString();
    EXPECT_TRUE(completed_before_release) << "MM preprocessing waited for the unrelated default-stream dependency ("
                                          << (GetParam() ? "cold" : "warm") << " pinned allocator)";
    EXPECT_EQ(restored_stream, original_stream);
    if (result.ok()) {
        assertHostReady(input);
    }
}

INSTANTIATE_TEST_SUITE_P(PinnedAllocator,
                         MultimodalCudaStreamTest,
                         ::testing::Values(false, true),
                         [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "Cold" : "Warm"; });

TEST(MultimodalCudaStreamRestorationTest, RestoresCallerStreamOnError) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    c10::cuda::CUDAStreamGuard caller(c10::cuda::getStreamFromPool(false, 0));
    const auto                 original_stream = c10::cuda::getCurrentCUDAStream(0);
    GpuOutputProcessor         processor;
    processor.fail = true;
    auto input     = makeInput();
    EXPECT_FALSE(processor.updateMultimodalFeatures(input).ok());
    EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), original_stream);
}

}  // namespace
}  // namespace rtp_llm
#else
TEST(MultimodalCudaStreamTest, RequiresCuda) {
    GTEST_SKIP() << "CUDA-only regression";
}
#endif
