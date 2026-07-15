#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif

#include "rtp_llm/models_py/bindings/common/kernels/vocab_prune/mapping.h"

class VocabPruneMappingKernelTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }

    void TearDown() override {
        EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
    }

    cudaStream_t stream_ = nullptr;

    static torch::TensorOptions intCuda() {
        return torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    }
};

TEST_F(VocabPruneMappingKernelTest, MappingDraft2Target_Basic) {
    const int batch_size   = 2;
    const int token_stride = 4;
    const int token_offset = 0;
    const int map_size     = 8;

    // d2t_map: draft token i → target token (i * 10)
    std::vector<int64_t> map_host(map_size);
    for (int i = 0; i < map_size; ++i)
        map_host[i] = i * 10;
    auto d2t_map = torch::from_blob(map_host.data(), {map_size}, torch::kInt64).to(torch::kCUDA);

    // tokens: [[0,1,2,3], [4,5,6,7]]
    auto tokens = torch::tensor({{0, 1, 2, 3}, {4, 5, 6, 7}}, intCuda());

    rtp_llm::invokeMappingDraft2Target<int32_t>(tokens.data_ptr<int32_t>(),
                                                batch_size,
                                                token_offset,
                                                token_stride,
                                                d2t_map.data_ptr<int64_t>(),
                                                map_size,
                                                stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    auto tokens_h = tokens.to(torch::kCPU);
    EXPECT_EQ(tokens_h[0][0].item<int>(), 0);
    EXPECT_EQ(tokens_h[0][1].item<int>(), 10);
    EXPECT_EQ(tokens_h[0][2].item<int>(), 20);
    EXPECT_EQ(tokens_h[0][3].item<int>(), 30);
    EXPECT_EQ(tokens_h[1][0].item<int>(), 40);
    EXPECT_EQ(tokens_h[1][1].item<int>(), 50);
    EXPECT_EQ(tokens_h[1][2].item<int>(), 60);
    EXPECT_EQ(tokens_h[1][3].item<int>(), 70);
}

TEST_F(VocabPruneMappingKernelTest, MappingDraft2Target_WithOffset) {
    const int batch_size   = 1;
    const int token_stride = 4;
    const int token_offset = 2;
    const int map_size     = 8;

    std::vector<int64_t> map_host(map_size);
    for (int i = 0; i < map_size; ++i)
        map_host[i] = i + 100;
    auto d2t_map = torch::from_blob(map_host.data(), {map_size}, torch::kInt64).to(torch::kCUDA);

    // tokens: [0, 1, 2, 3]; only positions 2,3 should be mapped
    auto tokens = torch::tensor({0, 1, 2, 3}, intCuda());

    rtp_llm::invokeMappingDraft2Target<int32_t>(tokens.data_ptr<int32_t>(),
                                                batch_size,
                                                token_offset,
                                                token_stride,
                                                d2t_map.data_ptr<int64_t>(),
                                                map_size,
                                                stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    auto tokens_h = tokens.to(torch::kCPU);
    EXPECT_EQ(tokens_h[0].item<int>(), 0);    // unchanged (before offset)
    EXPECT_EQ(tokens_h[1].item<int>(), 1);    // unchanged (before offset)
    EXPECT_EQ(tokens_h[2].item<int>(), 102);  // mapped: d2t_map[2]
    EXPECT_EQ(tokens_h[3].item<int>(), 103);  // mapped: d2t_map[3]
}

TEST_F(VocabPruneMappingKernelTest, MappingDraft2Target_NegativeTokensUnchanged) {
    const int batch_size   = 1;
    const int token_stride = 4;
    const int token_offset = 0;
    const int map_size     = 8;

    std::vector<int64_t> map_host(map_size);
    for (int i = 0; i < map_size; ++i)
        map_host[i] = i + 100;
    auto d2t_map = torch::from_blob(map_host.data(), {map_size}, torch::kInt64).to(torch::kCUDA);

    // tokens contain -1 (padding) — should be left unchanged
    auto tokens = torch::tensor({2, -1, 5, -1}, intCuda());

    rtp_llm::invokeMappingDraft2Target<int32_t>(tokens.data_ptr<int32_t>(),
                                                batch_size,
                                                token_offset,
                                                token_stride,
                                                d2t_map.data_ptr<int64_t>(),
                                                map_size,
                                                stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    auto tokens_h = tokens.to(torch::kCPU);
    EXPECT_EQ(tokens_h[0].item<int>(), 102);  // mapped
    EXPECT_EQ(tokens_h[1].item<int>(), -1);   // negative → unchanged
    EXPECT_EQ(tokens_h[2].item<int>(), 105);  // mapped
    EXPECT_EQ(tokens_h[3].item<int>(), -1);   // negative → unchanged
}

TEST_F(VocabPruneMappingKernelTest, MappingDraft2Target_OutOfRangeUnchanged) {
    const int batch_size   = 1;
    const int token_stride = 3;
    const int token_offset = 0;
    const int map_size     = 4;

    std::vector<int64_t> map_host = {100, 101, 102, 103};
    auto                 d2t_map  = torch::from_blob(map_host.data(), {map_size}, torch::kInt64).to(torch::kCUDA);

    // Token 10 is out of range for map_size=4 — should be left unchanged
    auto tokens = torch::tensor({1, 10, 3}, intCuda());

    rtp_llm::invokeMappingDraft2Target<int32_t>(tokens.data_ptr<int32_t>(),
                                                batch_size,
                                                token_offset,
                                                token_stride,
                                                d2t_map.data_ptr<int64_t>(),
                                                map_size,
                                                stream_);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    auto tokens_h = tokens.to(torch::kCPU);
    EXPECT_EQ(tokens_h[0].item<int>(), 101);  // mapped
    EXPECT_EQ(tokens_h[1].item<int>(), 10);   // out of range → unchanged
    EXPECT_EQ(tokens_h[2].item<int>(), 103);  // mapped
}
