#include <gtest/gtest.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <set>

#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
namespace {
struct HostGate {
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    released = false;
    static void CUDART_CB   wait(void* data) {
        auto&                        gate = *static_cast<HostGate*>(data);
        std::unique_lock<std::mutex> lock(gate.mutex);
        gate.cv.wait(lock, [&] { return gate.released; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
};

TEST(CopyStreamIsolationTest, CopiesDoNotWaitForUnrelatedPooledStreams) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    constexpr size_t bytes = 4096;
    void*            host  = nullptr;
    void*            gpu   = nullptr;
    ASSERT_EQ(cudaHostAlloc(&host, bytes, cudaHostAllocDefault), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&gpu, bytes), cudaSuccess);
    std::memset(host, 0x5a, bytes);
    ASSERT_EQ(cudaMemcpy(gpu, host, bytes, cudaMemcpyHostToDevice), cudaSuccess);
    auto host_tensor = torch::from_blob(host, {bytes}, torch::TensorOptions().dtype(torch::kUInt8));
    auto gpu_tensor =
        torch::from_blob(gpu, {bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0));
    for (int kind = 0; kind < 4; ++kind) {
        for (bool h2d : {true, false}) {
            SCOPED_TRACE(testing::Message() << "kind=" << kind << " h2d=" << h2d);
            StagedMemoryCopyScratch scratch;
            auto                    copy = [&] {
                auto dst = h2d ? gpu_tensor : host_tensor;
                auto src = h2d ? host_tensor : gpu_tensor;
                if (kind == 0) {
                    execNoBlockCopy(CopyParams{dst, src});
                } else if (kind == 1) {
                    MultiCopyParams params;
                    params.multi_dst = {dst};
                    params.multi_src = {src};
                    execNoBlockCopy(params);
                } else if (kind == 2) {
                    BatchedMemoryCopyParams params;
                    params.device_index = 0;
                    params.tiles        = {{dst.data_ptr(), src.data_ptr(), bytes}};
                    EXPECT_TRUE(execBatchedMemoryCopy(params));
                } else {
                    StagedMemoryCopyParams params;
                    params.host_base    = host;
                    params.host_bytes   = bytes;
                    params.tiles        = {{gpu, 0, bytes}};
                    params.device_index = 0;
                    params.direction = h2d ? StagedMemoryCopyDirection::H2D : StagedMemoryCopyDirection::D2H;
                    EXPECT_TRUE(execStagedMemoryCopy(params, &scratch));
                }
            };
            std::promise<void> ready, start;
            auto               start_future = start.get_future();
            auto               worker       = std::async(std::launch::async, [&] {
                EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
                copy();  // Initialize stream and scratch on the actual worker.
                ready.set_value();
                start_future.wait();
                copy();
            });
            ready.get_future().wait();
            std::set<cudaStream_t> pooled;
            for (size_t i = 0; i < 256; ++i) {
                auto stream = at::cuda::getStreamFromPool(false, 0).stream();
                if (!pooled.insert(stream).second)
                    break;
            }
            EXPECT_FALSE(pooled.empty());
            cudaStream_t gate_stream;
            cudaEvent_t  gate_event;
            EXPECT_EQ(cudaStreamCreateWithFlags(&gate_stream, cudaStreamNonBlocking), cudaSuccess);
            EXPECT_EQ(cudaEventCreateWithFlags(&gate_event, cudaEventDisableTiming), cudaSuccess);
            HostGate gate;
            EXPECT_EQ(cudaLaunchHostFunc(gate_stream, &HostGate::wait, &gate), cudaSuccess);
            EXPECT_EQ(cudaEventRecord(gate_event, gate_stream), cudaSuccess);
            for (auto stream : pooled)
                EXPECT_EQ(cudaStreamWaitEvent(stream, gate_event, 0), cudaSuccess);
            start.set_value();
            const auto status = worker.wait_for(std::chrono::seconds(1));
            // Always release before reporting failure: the old implementation
            // aliases this pool and must be allowed to finish without a hang.
            gate.release();
            worker.get();
            EXPECT_EQ(status, std::future_status::ready);
            EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            EXPECT_EQ(cudaMemcpy(host, gpu, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
            const auto* values = static_cast<const unsigned char*>(host);
            for (size_t i = 0; i < bytes; ++i)
                EXPECT_EQ(values[i], 0x5a);
            EXPECT_EQ(cudaEventDestroy(gate_event), cudaSuccess);
            EXPECT_EQ(cudaStreamDestroy(gate_stream), cudaSuccess);
            releaseStagedMemoryCopyScratch(scratch);
        }
    }
    EXPECT_EQ(cudaFree(gpu), cudaSuccess);
    EXPECT_EQ(cudaFreeHost(host), cudaSuccess);
}
}  // namespace
}  // namespace rtp_llm
