#include <c10/util/Exception.h>
#include <torch/torch.h>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/TorchCudaOom.h"

namespace rtp_llm {

TEST(CudaGraphReplayRetryIntegrationTest, RecoverableOomKeepsCapturedGraphReplayable) {
    ASSERT_TRUE(torch::cuda::is_available());

    auto                stream = cuda_graph::graphGetStreamFromPool(false);
    auto                output = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA));
    auto                one    = torch::ones_like(output);
    at::cuda::CUDAGraph graph;
    cuda_graph::graphDeviceSynchronize();

    {
        cuda_graph::GraphStreamGuard stream_guard(stream);
        output.add_(one);
        output.zero_();
        cuda_graph::graphDeviceSynchronize();
        cuda_graph::graphCaptureBegin(graph, cuda_graph::graphPoolHandle());
        output.add_(one);
        graph.capture_end();
    }

    int attempts   = 0;
    int recoveries = 0;
    // A typed signal before launch deterministically exercises allocator
    // recovery without exhausting the shared worker's GPU memory.
    auto replay = [&]() {
        if (++attempts == 1) {
            C10_THROW_ERROR(OutOfMemoryError, "injected recoverable graph launch OOM");
        }
        graph.replay();
    };
    auto recover = [&](const std::exception& exception) {
        EXPECT_TRUE(isTorchCudaOom(exception));
        ++recoveries;
        cuda_graph::graphEmptyCache();
    };

    EXPECT_NO_THROW(retryOnceOnTorchCudaOom(replay, recover));
    cuda_graph::graphDeviceSynchronize();
    EXPECT_EQ(attempts, 2);
    EXPECT_EQ(recoveries, 1);
    EXPECT_EQ(output.cpu().item<float>(), 1.0F);

    EXPECT_NO_THROW(retryOnceOnTorchCudaOom(replay, recover));
    cuda_graph::graphDeviceSynchronize();
    EXPECT_EQ(attempts, 3);
    EXPECT_EQ(recoveries, 1);
    EXPECT_EQ(output.cpu().item<float>(), 2.0F);
}

}  // namespace rtp_llm
