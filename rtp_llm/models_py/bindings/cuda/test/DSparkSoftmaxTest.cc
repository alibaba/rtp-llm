#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <array>
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"

namespace {
struct SoftmaxBuffers {
    torch::Tensor input, output, workspace;
    cudaDeviceProp properties{};
    size_t bytes;

    SoftmaxBuffers(int rows, int vocab) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        input = torch::randn({rows, vocab}, options) * 4;
        output = torch::empty_like(input);
        int device = 0;
        TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess);
        TORCH_CHECK(cudaGetDeviceProperties(&properties, device) == cudaSuccess);
        bytes = rtp_llm::dsparkSoftmaxWorkspaceBytes(input.data_ptr<float>(), output.data_ptr<float>(),
                                                    rows, vocab, properties.major, properties.minor);
        workspace = torch::empty({static_cast<int64_t>(bytes)}, options.dtype(torch::kUInt8));
    }
    cudaError_t launch(cudaStream_t stream) {
        return rtp_llm::invokeDSparkSoftmax(input.data_ptr<float>(), output.data_ptr<float>(),
                                          input.size(0), input.size(1), properties.major, properties.minor,
                                          properties.multiProcessorCount,
                                          bytes ? workspace.data_ptr() : nullptr, bytes, stream);
    }
    void check() {
        auto expected = torch::softmax(input, -1);
        ASSERT_TRUE(torch::isfinite(output).all().item<bool>());
        ASSERT_TRUE(torch::allclose(output, expected, 2e-5, 2e-6));
        ASSERT_TRUE(torch::allclose(output.sum(-1), torch::ones({input.size(0)}, input.options()), 2e-5, 2e-5));
    }
};
}

TEST(DSparkSoftmaxTest, EagerCachedAndUnsupportedShapes) {
    for (auto shape : {std::array<int, 2>{1, 196608}, {3, 196608}, {5, 196608}, {16, 196608},
                       {4, 200064}, {24, 200064}, {1, 8}, {8, 262144},
                       {5, 196607}, {129, 32000}, {1, 262152}}) {
        SoftmaxBuffers buffers(shape[0], shape[1]);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(buffers.launch(nullptr), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        buffers.check();
    }
}

TEST(DSparkSoftmaxTest, GraphReplaysChangedLogitsAndPaddedRows) {
    for (int vocab : {200064, 196607}) {
        SoftmaxBuffers buffers(8, vocab);
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        // Warm all lazy kernel attributes before capture.
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(buffers.launch(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
        ASSERT_EQ(buffers.launch(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
        ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
        for (int live : {8, 5, 3, 8}) {
            buffers.input.copy_(torch::randn_like(buffers.input) * 8);
            buffers.input.narrow(0, live, 8 - live).zero_();
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            ASSERT_EQ(cudaGraphLaunch(executable, stream), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            buffers.check();
        }
        ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
}

TEST(DSparkSoftmaxTest, ConcurrentStreamsKeepPrivateFallbackScratch) {
    SoftmaxBuffers first(5, 196607), second(8, 196607);
    ASSERT_GT(first.bytes, 0);
    ASSERT_GT(second.bytes, 0);
    ASSERT_NE(first.workspace.data_ptr(), second.workspace.data_ptr());
    cudaStream_t a, b;
    ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int repeat = 0; repeat < 8; ++repeat) {
        ASSERT_EQ(first.launch(a), cudaSuccess);
        ASSERT_EQ(second.launch(b), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(b), cudaSuccess);
    first.check();
    second.check();
    ASSERT_EQ(cudaStreamDestroy(a), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(b), cudaSuccess);
}
