#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/torch.h>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/TorchCudaOom.h"

namespace rtp_llm {

TEST(CudaGraphReplayRetryIntegrationTest, SameCapturedGraphCanReplayAfterRecoverableOomSignal) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA required";
    }

    auto                stream = cuda_graph::graphGetStreamFromPool(/*is_high_priority=*/false);
    auto                output = torch::zeros({1}, torch::TensorOptions().device(torch::kCUDA));
    auto                one    = torch::ones({1}, torch::TensorOptions().device(torch::kCUDA));
    at::cuda::CUDAGraph graph;

    {
        at::cuda::CUDAStreamGuard stream_guard(stream);
        output.add_(one);
        output.zero_();
        cuda_graph::graphDeviceSynchronize();
        cuda_graph::graphCaptureBegin(graph, cuda_graph::graphPoolHandle());
        output.add_(one);
        graph.capture_end();
    }

    int replay_calls = 0;
    try {
        graph.replay();
        ++replay_calls;
        C10_THROW_ERROR(OutOfMemoryError, "injected recoverable CUDA OOM after graph launch");
    } catch (const std::exception& exception) {
        ASSERT_TRUE(isTorchCudaOom(exception));
        cuda_graph::graphEmptyCache();
    }
    graph.replay();
    ++replay_calls;

    cuda_graph::graphDeviceSynchronize();
    EXPECT_EQ(replay_calls, 2);
    EXPECT_EQ(output.cpu().item<float>(), 2.0F);
}

}  // namespace rtp_llm
