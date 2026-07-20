#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>

#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#include "rtp_llm/models_py/bindings/common/kernels/vocab_prune/mapping.h"

class SpeculativeSamplingKernelTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
    }

    void TearDown() override {
        cudaStreamDestroy(stream_);
    }

    cudaStream_t stream_ = nullptr;

    static torch::TensorOptions floatCuda() {
        return torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
    }
    static torch::TensorOptions intCuda() {
        return torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    }
    static torch::TensorOptions boolCuda() {
        return torch::TensorOptions(torch::kBool).device(torch::kCUDA);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// invokeRejectionSampling tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_AllAccept) {
    const int batch_size    = 2;
    const int num_spec      = 3;
    const int vocab_size    = 16;
    const int target_stride = 1;

    auto draft_probs  = torch::zeros({batch_size, num_spec, vocab_size}, floatCuda());
    auto target_probs = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());

    // Both draft and target assign probability 1.0 to token 5
    draft_probs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 5}, 1.0f);
    target_probs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 5}, 1.0f);

    auto draft_token_ids = torch::full({batch_size, num_spec}, 5, intCuda());

    // target_token_ids: [batch, num_spec+1, target_stride], last column holds the argmax
    auto target_token_ids = torch::full({batch_size, num_spec + 1, target_stride}, 5, intCuda());

    auto uniform_samples     = torch::zeros({batch_size, num_spec + 1}, floatCuda());
    auto output_token_ids    = torch::full({batch_size, num_spec + 1}, -1, intCuda());
    auto output_accepted_num = torch::zeros({batch_size}, intCuda());
    auto do_sample           = torch::ones({batch_size}, boolCuda());

    auto status = rtp_llm::invokeRejectionSampling<float, int>(draft_probs.data_ptr<float>(),
                                                               draft_token_ids.data_ptr<int>(),
                                                               uniform_samples.data_ptr<float>(),
                                                               target_probs.data_ptr<float>(),
                                                               target_token_ids.data_ptr<int>(),
                                                               target_stride,
                                                               output_token_ids.data_ptr<int>(),
                                                               output_accepted_num.data_ptr<int>(),
                                                               do_sample.data_ptr<bool>(),
                                                               batch_size,
                                                               num_spec,
                                                               vocab_size,
                                                               stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    auto out_ids_h = output_token_ids.to(torch::kCPU);
    auto acc_num_h = output_accepted_num.to(torch::kCPU);

    for (int b = 0; b < batch_size; ++b) {
        // All speculative tokens accepted + bonus token = num_spec + 1
        EXPECT_EQ(acc_num_h[b].item<int>(), num_spec + 1);
        // First num_spec tokens should be draft token (5)
        for (int s = 0; s < num_spec; ++s) {
            EXPECT_EQ(out_ids_h[b][s].item<int>(), 5);
        }
        // Bonus token is target_token_ids[..., -1] = 5
        EXPECT_EQ(out_ids_h[b][num_spec].item<int>(), 5);
    }
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_ImmediateReject) {
    const int batch_size    = 1;
    const int num_spec      = 3;
    const int vocab_size    = 16;
    const int target_stride = 1;

    // Draft picks token 3, target picks token 7 — immediate mismatch
    auto draft_probs  = torch::zeros({batch_size, num_spec, vocab_size}, floatCuda());
    auto target_probs = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());

    draft_probs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 3}, 1.0f);
    target_probs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 7}, 1.0f);

    auto draft_token_ids  = torch::full({batch_size, num_spec}, 3, intCuda());
    auto target_token_ids = torch::full({batch_size, num_spec + 1, target_stride}, 7, intCuda());

    // u = 0.5, p(draft_id=3) in target = 0 => u*p=0 which is NOT < q=0, so rejection
    // Actually: same_token is false, do_sample is false => reject
    auto uniform_samples     = torch::full({batch_size, num_spec + 1}, 0.5f, floatCuda());
    auto output_token_ids    = torch::full({batch_size, num_spec + 1}, -1, intCuda());
    auto output_accepted_num = torch::zeros({batch_size}, intCuda());
    auto do_sample           = torch::zeros({batch_size}, boolCuda());

    auto status = rtp_llm::invokeRejectionSampling<float, int>(draft_probs.data_ptr<float>(),
                                                               draft_token_ids.data_ptr<int>(),
                                                               uniform_samples.data_ptr<float>(),
                                                               target_probs.data_ptr<float>(),
                                                               target_token_ids.data_ptr<int>(),
                                                               target_stride,
                                                               output_token_ids.data_ptr<int>(),
                                                               output_accepted_num.data_ptr<int>(),
                                                               do_sample.data_ptr<bool>(),
                                                               batch_size,
                                                               num_spec,
                                                               vocab_size,
                                                               stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    auto acc_num_h = output_accepted_num.to(torch::kCPU);
    auto out_ids_h = output_token_ids.to(torch::kCPU);

    // Rejected at position 0, so accepted count = 0 + 1 = 1 (the resampled token)
    EXPECT_EQ(acc_num_h[0].item<int>(), 1);
    // The resampled token at pos 0 should come from relu(q-p); target_probs is all on token 7,
    // so the resampled token should be 7
    EXPECT_EQ(out_ids_h[0][0].item<int>(), 7);
    // Remaining positions padded with -1
    EXPECT_EQ(out_ids_h[0][1].item<int>(), -1);
    EXPECT_EQ(out_ids_h[0][2].item<int>(), -1);
    EXPECT_EQ(out_ids_h[0][3].item<int>(), -1);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_PartialAccept) {
    const int batch_size    = 1;
    const int num_spec      = 3;
    const int vocab_size    = 16;
    const int target_stride = 1;

    auto draft_probs  = torch::zeros({batch_size, num_spec, vocab_size}, floatCuda());
    auto target_probs = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());

    // Position 0: draft=5, target argmax=5 → same_token → accept
    // Position 1: draft=5, target argmax=5 → same_token → accept
    // Position 2: draft=3, target argmax=7 → mismatch, do_sample=false → reject
    draft_probs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 5}, 1.0f);
    draft_probs.index_put_({torch::indexing::Slice(), 2, 5}, 0.0f);
    draft_probs.index_put_({torch::indexing::Slice(), 2, 3}, 1.0f);

    target_probs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 7}, 1.0f);

    auto draft_token_ids = torch::full({batch_size, num_spec}, 5, intCuda());
    draft_token_ids.index_put_({torch::indexing::Slice(), 2}, 3);

    auto target_token_ids = torch::full({batch_size, num_spec + 1, target_stride}, 5, intCuda());
    // Position 0,1 target matches draft (token 5)
    // Position 2 target is different (token 7)
    target_token_ids.index_put_({torch::indexing::Slice(), 2, 0}, 7);
    target_token_ids.index_put_({torch::indexing::Slice(), 3, 0}, 7);

    auto uniform_samples     = torch::full({batch_size, num_spec + 1}, 0.5f, floatCuda());
    auto output_token_ids    = torch::full({batch_size, num_spec + 1}, -1, intCuda());
    auto output_accepted_num = torch::zeros({batch_size}, intCuda());
    auto do_sample           = torch::zeros({batch_size}, boolCuda());

    auto status = rtp_llm::invokeRejectionSampling<float, int>(draft_probs.data_ptr<float>(),
                                                               draft_token_ids.data_ptr<int>(),
                                                               uniform_samples.data_ptr<float>(),
                                                               target_probs.data_ptr<float>(),
                                                               target_token_ids.data_ptr<int>(),
                                                               target_stride,
                                                               output_token_ids.data_ptr<int>(),
                                                               output_accepted_num.data_ptr<int>(),
                                                               do_sample.data_ptr<bool>(),
                                                               batch_size,
                                                               num_spec,
                                                               vocab_size,
                                                               stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    auto acc_num_h = output_accepted_num.to(torch::kCPU);
    auto out_ids_h = output_token_ids.to(torch::kCPU);

    // Accepted positions 0, 1, rejected at 2 → count = 2 + 1 = 3
    EXPECT_EQ(acc_num_h[0].item<int>(), 3);
    EXPECT_EQ(out_ids_h[0][0].item<int>(), 5);
    EXPECT_EQ(out_ids_h[0][1].item<int>(), 5);
    // Resampled from relu(q-p) at position 2; target has all prob on token 7
    EXPECT_EQ(out_ids_h[0][2].item<int>(), 7);
    EXPECT_EQ(out_ids_h[0][3].item<int>(), -1);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_BatchSizeZero) {
    auto status = rtp_llm::invokeRejectionSampling<float, int>(
        nullptr, nullptr, nullptr, nullptr, nullptr, 1, nullptr, nullptr, nullptr, 0, 3, 16, stream_);
    ASSERT_EQ(status, cudaSuccess);
}

// ─────────────────────────────────────────────────────────────────────────────
// invokeMappingDraft2Target tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(SpeculativeSamplingKernelTest, MappingDraft2Target_Basic) {
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
    cudaStreamSynchronize(stream_);

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

TEST_F(SpeculativeSamplingKernelTest, MappingDraft2Target_WithOffset) {
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
    cudaStreamSynchronize(stream_);

    auto tokens_h = tokens.to(torch::kCPU);
    EXPECT_EQ(tokens_h[0].item<int>(), 0);    // unchanged (before offset)
    EXPECT_EQ(tokens_h[1].item<int>(), 1);    // unchanged (before offset)
    EXPECT_EQ(tokens_h[2].item<int>(), 102);  // mapped: d2t_map[2]
    EXPECT_EQ(tokens_h[3].item<int>(), 103);  // mapped: d2t_map[3]
}

TEST_F(SpeculativeSamplingKernelTest, MappingDraft2Target_NegativeTokensUnchanged) {
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
    cudaStreamSynchronize(stream_);

    auto tokens_h = tokens.to(torch::kCPU);
    EXPECT_EQ(tokens_h[0].item<int>(), 102);  // mapped
    EXPECT_EQ(tokens_h[1].item<int>(), -1);   // negative → unchanged
    EXPECT_EQ(tokens_h[2].item<int>(), 105);  // mapped
    EXPECT_EQ(tokens_h[3].item<int>(), -1);   // negative → unchanged
}

TEST_F(SpeculativeSamplingKernelTest, MappingDraft2Target_OutOfRangeUnchanged) {
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
    cudaStreamSynchronize(stream_);

    auto tokens_h = tokens.to(torch::kCPU);
    EXPECT_EQ(tokens_h[0].item<int>(), 101);  // mapped
    EXPECT_EQ(tokens_h[1].item<int>(), 10);   // out of range → unchanged
    EXPECT_EQ(tokens_h[2].item<int>(), 103);  // mapped
}
