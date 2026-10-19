#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>
#include <limits>

#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/markov_softmax.h"
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

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_GreedyTargetUsesDirectFallbackWithFullQ) {
    const int batch_size    = 1;
    const int num_spec      = 1;
    const int vocab_size    = 16;
    const int target_stride = 1;

    auto draft_probs  = torch::zeros({batch_size, num_spec, vocab_size}, floatCuda());
    auto target_probs = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());
    draft_probs.index_put_({0, 0, 3}, 0.4f);
    draft_probs.index_put_({0, 0, 7}, 0.6f);
    target_probs.index_put_({0, 0, 7}, 0.51f);
    target_probs.index_put_({0, 0, 8}, 0.49f);

    auto draft_token_ids     = torch::full({batch_size, num_spec}, 3, intCuda());
    auto target_token_ids    = torch::full({batch_size, num_spec + 1, target_stride}, 7, intCuda());
    auto uniform_samples     = torch::full({batch_size, num_spec + 1}, 0.99f, floatCuda());
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

    // Greedy verification must emit its exact argmax token, not sample the
    // stochastic residual target-draft distribution.
    EXPECT_EQ(output_accepted_num.cpu()[0].item<int>(), 1);
    EXPECT_EQ(output_token_ids.cpu()[0][0].item<int>(), 7);
    EXPECT_EQ(output_token_ids.cpu()[0][1].item<int>(), -1);
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
    // Greedy verification falls back directly to target token 7.
    EXPECT_EQ(out_ids_h[0][2].item<int>(), 7);
    EXPECT_EQ(out_ids_h[0][3].item<int>(), -1);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_StochasticSameTokenStillUsesRatio) {
    const int batch_size    = 1;
    const int num_spec      = 1;
    const int vocab_size    = 16;
    const int target_stride = 1;

    auto draft_probs  = torch::zeros({batch_size, num_spec, vocab_size}, floatCuda());
    auto target_probs = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());
    draft_probs.index_put_({0, 0, 5}, 0.9f);
    draft_probs.index_put_({0, 0, 7}, 0.1f);
    target_probs.index_put_({0, 0, 5}, 0.2f);
    target_probs.index_put_({0, 0, 7}, 0.8f);

    auto draft_token_ids     = torch::full({batch_size, num_spec}, 5, intCuda());
    auto target_token_ids    = torch::full({batch_size, num_spec + 1, target_stride}, 5, intCuda());
    auto uniform_samples     = torch::full({batch_size, num_spec + 1}, 0.5f, floatCuda());
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

    // Target argmax happens to match, but standard rejection sampling uses
    // only min(1, p_target/q_draft): 0.5 * 0.9 >= 0.2, so reject and sample
    // the residual distribution, which is concentrated on token 7.
    EXPECT_EQ(output_accepted_num.cpu()[0].item<int>(), 1);
    EXPECT_EQ(output_token_ids.cpu()[0][0].item<int>(), 7);
    EXPECT_EQ(output_token_ids.cpu()[0][1].item<int>(), -1);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_PointMassStochasticSameTokenCanReject) {
    const int batch_size    = 1;
    const int num_spec      = 1;
    const int vocab_size    = 16;
    const int target_stride = 1;

    auto target_probs = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());
    target_probs.index_put_({0, 0, 5}, 0.2f);
    target_probs.index_put_({0, 0, 7}, 0.8f);

    auto draft_token_ids     = torch::full({batch_size, num_spec}, 5, intCuda());
    auto target_token_ids    = torch::full({batch_size, num_spec + 1, target_stride}, 5, intCuda());
    auto uniform_samples     = torch::full({batch_size, num_spec + 1}, 0.5f, floatCuda());
    auto output_token_ids    = torch::full({batch_size, num_spec + 1}, -1, intCuda());
    auto output_accepted_num = torch::zeros({batch_size}, intCuda());
    auto do_sample           = torch::ones({batch_size}, boolCuda());

    auto status = rtp_llm::invokeRejectionSampling<float, int>(nullptr,
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
                                                               stream_,
                                                               true);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    EXPECT_EQ(output_accepted_num.cpu()[0].item<int>(), 1);
    EXPECT_EQ(output_token_ids.cpu()[0][0].item<int>(), 7);
    EXPECT_EQ(output_token_ids.cpu()[0][1].item<int>(), -1);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_ImplicitPointMassDraft) {
    const int batch_size    = 1;
    const int num_spec      = 1;
    const int vocab_size    = 16;
    const int target_stride = 1;

    auto target_probs = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());
    target_probs.index_put_({0, 0, 5}, 0.2f);
    target_probs.index_put_({0, 0, 7}, 0.8f);

    auto draft_token_ids     = torch::full({batch_size, num_spec}, 5, intCuda());
    auto target_token_ids    = torch::full({batch_size, num_spec + 1, target_stride}, 7, intCuda());
    auto uniform_samples     = torch::full({batch_size, num_spec + 1}, 0.5f, floatCuda());
    auto output_token_ids    = torch::full({batch_size, num_spec + 1}, -1, intCuda());
    auto output_accepted_num = torch::zeros({batch_size}, intCuda());
    auto do_sample           = torch::ones({batch_size}, boolCuda());

    auto status = rtp_llm::invokeRejectionSampling<float, int>(nullptr,
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
                                                               stream_,
                                                               true);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    EXPECT_EQ(output_accepted_num.cpu()[0].item<int>(), 1);
    EXPECT_EQ(output_token_ids.cpu()[0][0].item<int>(), 7);
    EXPECT_EQ(output_token_ids.cpu()[0][1].item<int>(), -1);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_BatchSizeZero) {
    auto status = rtp_llm::invokeRejectionSampling<float, int>(
        nullptr, nullptr, nullptr, nullptr, nullptr, 1, nullptr, nullptr, nullptr, 0, 3, 16, stream_);
    ASSERT_EQ(status, cudaSuccess);
}

// ─────────────────────────────────────────────────────────────────────────────
// invokeMarkovSoftmaxStep tests
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Reference of the pre-fusion eager chain: FP32-cast bias, add, temperature
// divide, softmax.
torch::Tensor markovSoftmaxReference(const torch::Tensor& logits, const torch::Tensor& bias, const torch::Tensor& t) {
    auto x = logits + bias.to(torch::kFloat32);
    x.div_(t.unsqueeze(1));
    return torch::softmax(x, -1);
}

}  // namespace

TEST_F(SpeculativeSamplingKernelTest, MarkovSoftmaxStep_StridedViewsMatchTorchChainBF16Bias) {
    // Production-shaped: [B, gamma, vocab] select() views over a TP-padded
    // logits matrix, BF16 mm output, mixed greedy/stochastic temperatures.
    const int          batch_size = 32, gamma = 5, step = 2, vocab_size = 8192, padded = vocab_size + 128;
    auto               base_logits = torch::randn({(int64_t)batch_size * gamma, padded}, floatCuda()) * 3.0f;
    auto               bias        = (torch::randn({batch_size, vocab_size}, floatCuda()) * 0.5f).to(torch::kBFloat16);
    std::vector<float> temps(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        temps[i] = (i % 2 == 0) ? 1.0e-6f : 0.5f + 0.05f * i;
    }
    auto temperature = torch::tensor(temps, floatCuda());

    auto proposal  = base_logits.narrow(1, 0, vocab_size).view({batch_size, gamma, vocab_size});
    auto logits_v  = proposal.select(1, step);
    auto all_probs = torch::full({batch_size, gamma, (int64_t)vocab_size}, -777.0f, floatCuda());
    auto q_all_v   = all_probs.select(1, step);
    auto q_contig  = torch::empty({batch_size, (int64_t)vocab_size}, floatCuda());

    const int splits   = rtp_llm::markovSoftmaxStepSplits(batch_size, vocab_size);
    auto      partials = torch::empty({2, batch_size, splits}, floatCuda());
    auto      status   = rtp_llm::invokeMarkovSoftmaxStep(logits_v.data_ptr<float>(),
                                                   logits_v.stride(0),
                                                   bias.data_ptr(),
                                                   rtp_llm::MarkovBiasDType::kBFloat16,
                                                   temperature.data_ptr<float>(),
                                                   q_all_v.data_ptr<float>(),
                                                   q_all_v.stride(0),
                                                   q_contig.data_ptr<float>(),
                                                   batch_size,
                                                   vocab_size,
                                                   partials.data_ptr<float>(),
                                                   stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    auto reference = markovSoftmaxReference(logits_v, bias, temperature);
    EXPECT_TRUE(torch::allclose(q_all_v, reference, 1e-5, 1e-6));
    EXPECT_TRUE(torch::allclose(q_contig, reference, 1e-5, 1e-6));
    EXPECT_TRUE(torch::equal(q_all_v, q_contig));
    EXPECT_TRUE(torch::allclose(q_all_v.sum(-1), torch::ones({batch_size}, floatCuda()), 1e-5, 1e-5));
    // Only this step's slice may be written; sibling steps keep the sentinel.
    for (int other = 0; other < gamma; ++other) {
        if (other == step) {
            continue;
        }
        EXPECT_TRUE(
            torch::equal(all_probs.select(1, other), torch::full({batch_size, vocab_size}, -777.0f, floatCuda())));
    }
}

TEST_F(SpeculativeSamplingKernelTest, MarkovSoftmaxStep_ScalarPathOddVocabFP32Bias) {
    const int batch_size = 3, vocab_size = 1023;  // not a multiple of 4
    auto      logits      = torch::randn({batch_size, vocab_size}, floatCuda()) * 2.0f;
    auto      bias        = torch::randn({batch_size, vocab_size}, floatCuda()) * 0.5f;
    auto      temperature = torch::tensor({1.0e-6f, 0.8f, 1.0f}, floatCuda());
    auto      q_all       = torch::empty({batch_size, vocab_size}, floatCuda());
    auto      q_contig    = torch::empty({batch_size, vocab_size}, floatCuda());
    const int splits      = rtp_llm::markovSoftmaxStepSplits(batch_size, vocab_size);
    auto      partials    = torch::empty({2, batch_size, splits}, floatCuda());
    auto      status      = rtp_llm::invokeMarkovSoftmaxStep(logits.data_ptr<float>(),
                                                   logits.stride(0),
                                                   bias.data_ptr(),
                                                   rtp_llm::MarkovBiasDType::kFloat32,
                                                   temperature.data_ptr<float>(),
                                                   q_all.data_ptr<float>(),
                                                   q_all.stride(0),
                                                   q_contig.data_ptr<float>(),
                                                   batch_size,
                                                   vocab_size,
                                                   partials.data_ptr<float>(),
                                                   stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    auto reference = markovSoftmaxReference(logits, bias, temperature);
    EXPECT_TRUE(torch::allclose(q_all, reference, 1e-5, 1e-6));
    EXPECT_TRUE(torch::equal(q_all, q_contig));
    EXPECT_TRUE(torch::allclose(q_all.sum(-1), torch::ones({batch_size}, floatCuda()), 1e-5, 1e-5));
}

TEST_F(SpeculativeSamplingKernelTest, MarkovSoftmaxStep_HalfBiasGreedyCollapsesToArgmaxOneHot) {
    const int batch_size = 4, vocab_size = 4096;
    auto      logits = torch::randn({batch_size, vocab_size}, floatCuda()) * 2.0f;
    auto      bias   = (torch::randn({batch_size, vocab_size}, floatCuda()) * 0.5f).to(torch::kHalf);
    logits.select(1, 23).fill_(100.0f);
    auto      temperature = torch::full({batch_size}, 1.0e-6f, floatCuda());
    auto      q_all       = torch::empty({batch_size, vocab_size}, floatCuda());
    auto      q_contig    = torch::empty({batch_size, vocab_size}, floatCuda());
    const int splits      = rtp_llm::markovSoftmaxStepSplits(batch_size, vocab_size);
    auto      partials    = torch::empty({2, batch_size, splits}, floatCuda());
    auto      status      = rtp_llm::invokeMarkovSoftmaxStep(logits.data_ptr<float>(),
                                                   logits.stride(0),
                                                   bias.data_ptr(),
                                                   rtp_llm::MarkovBiasDType::kFloat16,
                                                   temperature.data_ptr<float>(),
                                                   q_all.data_ptr<float>(),
                                                   q_all.stride(0),
                                                   q_contig.data_ptr<float>(),
                                                   batch_size,
                                                   vocab_size,
                                                   partials.data_ptr<float>(),
                                                   stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    auto reference = markovSoftmaxReference(logits, bias, temperature);
    EXPECT_TRUE(torch::allclose(q_all, reference, 1e-5, 1e-6));
    auto expected_argmax = reference.argmax(-1);
    EXPECT_TRUE(torch::equal(q_all.argmax(-1), expected_argmax));
    // One-hot: the argmax column dominates and every other entry underflows.
    auto gathered = q_all.gather(1, expected_argmax.unsqueeze(1));
    EXPECT_TRUE(torch::allclose(gathered, torch::ones({batch_size, 1}, floatCuda()), 1e-5, 1e-5));
}

TEST_F(SpeculativeSamplingKernelTest, MarkovSoftmaxStep_MaskedElementsAndEmptyPartials) {
    for (const int vocab_size : {8192, 8193}) {
        SCOPED_TRACE(vocab_size);
        constexpr int batch_size = 3;
        auto logits = torch::full({batch_size, vocab_size}, -std::numeric_limits<float>::infinity(), floatCuda());
        // Entire splits and individual threads see only masked elements.
        logits[0][17]             = 0.5f;
        logits[0][2065]           = 2.0f;
        logits[0][vocab_size - 3] = -1.0f;
        logits[1][vocab_size - 1] = 1.0f;
        auto bias                 = torch::zeros_like(logits);
        auto temperature          = torch::tensor({0.7f, 1.0f, 1.0f}, floatCuda());
        auto q_all                = torch::empty_like(logits);
        auto q_contig             = torch::empty_like(logits);
        auto partials =
            torch::empty({2, batch_size, rtp_llm::markovSoftmaxStepSplits(batch_size, vocab_size)}, floatCuda());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(rtp_llm::invokeMarkovSoftmaxStep(logits.data_ptr<float>(),
                                                   logits.stride(0),
                                                   bias.data_ptr(),
                                                   rtp_llm::MarkovBiasDType::kFloat32,
                                                   temperature.data_ptr<float>(),
                                                   q_all.data_ptr<float>(),
                                                   q_all.stride(0),
                                                   q_contig.data_ptr<float>(),
                                                   batch_size,
                                                   vocab_size,
                                                   partials.data_ptr<float>(),
                                                   stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        auto reference = markovSoftmaxReference(logits, bias, temperature);
        EXPECT_TRUE(torch::allclose(q_all.narrow(0, 0, 2), reference.narrow(0, 0, 2), 1e-5, 1e-6));
        EXPECT_TRUE(torch::equal(q_all.narrow(0, 0, 2), q_contig.narrow(0, 0, 2)));
        EXPECT_TRUE(q_all.narrow(0, 0, 2).isfinite().all().item<bool>());
        EXPECT_TRUE(q_all[0].masked_select(logits[0].isneginf()).eq(0).all().item<bool>());
        // An entirely masked row has no distribution, as in torch::softmax.
        EXPECT_TRUE(q_all[2].isnan().all().item<bool>());
        EXPECT_TRUE(q_contig[2].isnan().all().item<bool>());
    }
}

TEST_F(SpeculativeSamplingKernelTest, MarkovSoftmaxStep_UnalignedContiguousStorageOffsets) {
    constexpr int batch_size = 2, vocab_size = 8192;
    for (const auto dtype : {torch::kFloat32, torch::kBFloat16, torch::kHalf}) {
        const auto bias_dtype = dtype == torch::kFloat32  ? rtp_llm::MarkovBiasDType::kFloat32 :
                                dtype == torch::kBFloat16 ? rtp_llm::MarkovBiasDType::kBFloat16 :
                                                            rtp_llm::MarkovBiasDType::kFloat16;
        for (int offset_arg = 0; offset_arg < 4; ++offset_arg) {
            SCOPED_TRACE(offset_arg);
            std::vector<torch::Tensor> args;
            for (int i = 0; i < 4; ++i) {
                auto storage =
                    torch::randn({batch_size * vocab_size + 1}, floatCuda().dtype(i == 1 ? dtype : torch::kFloat32));
                args.push_back(
                    storage.narrow(0, i == offset_arg ? 1 : 0, batch_size * vocab_size).view({batch_size, vocab_size}));
                ASSERT_TRUE(args.back().is_contiguous());
            }
            auto temperature = torch::tensor({0.7f, 1.3f}, floatCuda());
            auto partials =
                torch::empty({2, batch_size, rtp_llm::markovSoftmaxStepSplits(batch_size, vocab_size)}, floatCuda());
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            ASSERT_EQ(rtp_llm::invokeMarkovSoftmaxStep(args[0].data_ptr<float>(),
                                                       args[0].stride(0),
                                                       args[1].data_ptr(),
                                                       bias_dtype,
                                                       temperature.data_ptr<float>(),
                                                       args[2].data_ptr<float>(),
                                                       args[2].stride(0),
                                                       args[3].data_ptr<float>(),
                                                       batch_size,
                                                       vocab_size,
                                                       partials.data_ptr<float>(),
                                                       stream_),
                      cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
            auto reference = markovSoftmaxReference(args[0], args[1], temperature);
            EXPECT_TRUE(torch::allclose(args[2], reference, 1e-5, 1e-6));
            EXPECT_TRUE(torch::equal(args[2], args[3]));
        }
    }
}

TEST_F(SpeculativeSamplingKernelTest, MarkovSoftmaxStep_BatchSizeZeroIsNoop) {
    auto status = rtp_llm::invokeMarkovSoftmaxStep(nullptr,
                                                   0,
                                                   nullptr,
                                                   rtp_llm::MarkovBiasDType::kBFloat16,
                                                   nullptr,
                                                   nullptr,
                                                   0,
                                                   nullptr,
                                                   0,
                                                   128,
                                                   nullptr,
                                                   stream_);
    ASSERT_EQ(status, cudaSuccess);
}

TEST_F(SpeculativeSamplingKernelTest, MarkovSoftmaxStep_RejectsBadArguments) {
    const int batch_size = 2, vocab_size = 64;
    auto      logits      = torch::randn({batch_size, vocab_size}, floatCuda());
    auto      bias        = torch::randn({batch_size, vocab_size}, floatCuda());
    auto      temperature = torch::ones({batch_size}, floatCuda());
    auto      q_all       = torch::empty({batch_size, vocab_size}, floatCuda());
    const int splits      = rtp_llm::markovSoftmaxStepSplits(batch_size, vocab_size);
    auto      partials    = torch::empty({2, batch_size, splits}, floatCuda());
    // Row stride smaller than the row length.
    auto status = rtp_llm::invokeMarkovSoftmaxStep(logits.data_ptr<float>(),
                                                   vocab_size - 1,
                                                   bias.data_ptr(),
                                                   rtp_llm::MarkovBiasDType::kFloat32,
                                                   temperature.data_ptr<float>(),
                                                   q_all.data_ptr<float>(),
                                                   vocab_size,
                                                   nullptr,
                                                   batch_size,
                                                   vocab_size,
                                                   partials.data_ptr<float>(),
                                                   stream_);
    ASSERT_EQ(status, cudaErrorInvalidValue);
    // Missing partials scratch.
    status = rtp_llm::invokeMarkovSoftmaxStep(logits.data_ptr<float>(),
                                              vocab_size,
                                              bias.data_ptr(),
                                              rtp_llm::MarkovBiasDType::kFloat32,
                                              temperature.data_ptr<float>(),
                                              q_all.data_ptr<float>(),
                                              vocab_size,
                                              nullptr,
                                              batch_size,
                                              vocab_size,
                                              nullptr,
                                              stream_);
    ASSERT_EQ(status, cudaErrorInvalidValue);
}

// ─────────────────────────────────────────────────────────────────────────────
// invokeMarkovArgmaxStep tests
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Reference of the fused kernel's value function: FP32-cast bias, add,
// temperature divide, first-maximum argmax.
torch::Tensor markovArgmaxReference(const torch::Tensor& logits, const torch::Tensor& bias, const torch::Tensor& t) {
    auto x = logits + bias.to(torch::kFloat32);
    x.div_(t.unsqueeze(1));
    return torch::argmax(x, -1).to(torch::kInt32);
}

torch::Tensor runMarkovArgmax(const torch::Tensor&     logits_v,
                              const torch::Tensor&     bias,
                              rtp_llm::MarkovBiasDType bias_dtype,
                              const torch::Tensor&     temperature,
                              cudaStream_t             stream) {
    const int batch_size = static_cast<int>(logits_v.size(0));
    const int vocab_size = static_cast<int>(logits_v.size(1));
    auto      tokens     = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    const int splits     = rtp_llm::markovSoftmaxStepSplits(batch_size, vocab_size);
    auto      partials =
        torch::empty({2, batch_size, splits}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto status = rtp_llm::invokeMarkovArgmaxStep(logits_v.data_ptr<float>(),
                                                  logits_v.stride(0),
                                                  bias.data_ptr(),
                                                  bias_dtype,
                                                  temperature.data_ptr<float>(),
                                                  tokens.data_ptr<int32_t>(),
                                                  batch_size,
                                                  vocab_size,
                                                  partials.data_ptr<float>(),
                                                  stream);
    EXPECT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream);
    return tokens;
}

}  // namespace

TEST_F(SpeculativeSamplingKernelTest, MarkovArgmaxStep_StridedViewsMatchTorchArgmaxBF16Bias) {
    // Production-shaped: [B, gamma, vocab] select() views over a TP-padded
    // logits matrix, BF16 mm output, the pinned greedy draft temperature.
    const int batch_size = 32, gamma = 5, step = 2, vocab_size = 8192, padded = vocab_size + 128;
    auto      base_logits = torch::randn({(int64_t)batch_size * gamma, padded}, floatCuda()) * 3.0f;
    auto      bias        = (torch::randn({batch_size, vocab_size}, floatCuda()) * 0.5f).to(torch::kBFloat16);
    auto      temperature = torch::full({batch_size}, 1.0e-6f, floatCuda());

    auto proposal  = base_logits.narrow(1, 0, vocab_size).view({batch_size, gamma, vocab_size});
    auto logits_v  = proposal.select(1, step);
    auto tokens    = runMarkovArgmax(logits_v, bias, rtp_llm::MarkovBiasDType::kBFloat16, temperature, stream_);
    auto reference = markovArgmaxReference(logits_v.clone(), bias, temperature);
    EXPECT_TRUE(torch::equal(tokens, reference));
}

TEST_F(SpeculativeSamplingKernelTest, MarkovArgmaxStep_ScalarPathOddVocabFP32Bias) {
    const int batch_size = 3, vocab_size = 1023;  // not a multiple of 4
    auto      logits      = torch::randn({batch_size, vocab_size}, floatCuda()) * 2.0f;
    auto      bias        = torch::randn({batch_size, vocab_size}, floatCuda()) * 0.5f;
    auto      temperature = torch::tensor({1.0e-6f, 0.8f, 1.0f}, floatCuda());
    auto      tokens      = runMarkovArgmax(logits, bias, rtp_llm::MarkovBiasDType::kFloat32, temperature, stream_);
    auto      reference   = markovArgmaxReference(logits.clone(), bias, temperature);
    EXPECT_TRUE(torch::equal(tokens, reference));
}

TEST_F(SpeculativeSamplingKernelTest, MarkovArgmaxStep_HalfBiasTiesResolveToLowestIndex) {
    const int batch_size = 3, vocab_size = 8192;
    // Row 0: tie across splits -> lowest index wins.
    // Row 1: all-equal row -> column 0 wins.
    // Row 2: unique maximum after the bias, ties elsewhere.
    auto logits      = torch::zeros({batch_size, vocab_size}, floatCuda());
    logits[0][5]     = 2.0f;
    logits[0][4096]  = 2.0f;
    logits[0][7]     = 1.0f;
    logits[2][10]    = 0.5f;
    logits[2][11]    = 0.5f;
    logits[2][63]    = 3.0f;
    auto bias        = torch::zeros({batch_size, vocab_size}, floatCuda()).to(torch::kHalf);
    auto temperature = torch::full({batch_size}, 1.0e-6f, floatCuda());
    auto tokens      = runMarkovArgmax(logits, bias, rtp_llm::MarkovBiasDType::kFloat16, temperature, stream_);
    EXPECT_EQ(tokens.cpu()[0].item<int32_t>(), 5);
    EXPECT_EQ(tokens.cpu()[1].item<int32_t>(), 0);
    EXPECT_EQ(tokens.cpu()[2].item<int32_t>(), 63);
    auto q = markovSoftmaxReference(logits, bias, temperature);
    EXPECT_FLOAT_EQ(q[0][5].item<float>(), 0.5f);
    EXPECT_FLOAT_EQ(q[0][4096].item<float>(), 0.5f);
}

TEST_F(SpeculativeSamplingKernelTest, MarkovArgmaxStep_NaNsAndTiesMatchTorch) {
    // Small rows exercise idle threads; wide rows revisit threads and cross splits.
    for (const int vocab_size : {4, 5, 131072, 131073}) {
        SCOPED_TRACE(vocab_size);
        auto logits      = torch::zeros({1, vocab_size}, floatCuda());
        auto bias        = torch::zeros_like(logits);
        auto temperature = torch::ones({1}, floatCuda());
        auto check       = [&](int expected) {
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            auto tokens = runMarkovArgmax(logits, bias, rtp_llm::MarkovBiasDType::kFloat32, temperature, stream_);
            EXPECT_EQ(tokens[0].item<int32_t>(), expected);
            EXPECT_TRUE(torch::equal(tokens, markovArgmaxReference(logits, bias, temperature)));
        };
        for (const int first_nan : {0, 1, 2, 3, 4096, vocab_size / 2, vocab_size - 1}) {
            if (first_nan >= vocab_size) {
                continue;
            }
            SCOPED_TRACE(first_nan);
            logits.zero_();
            logits[0][1]         = 2.0f;
            logits[0][2]         = 1.0f;
            logits[0][first_nan] = std::numeric_limits<float>::quiet_NaN();
            check(first_nan);
            logits[0][vocab_size - 1] = std::numeric_limits<float>::quiet_NaN();
            check(first_nan);
        }
        for (const float value : {-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()}) {
            logits.fill_(value);
            check(0);
        }
        logits.zero_();
        for (const int index : {1, 2, vocab_size / 2, vocab_size - 1}) {
            logits[0][index] = 5.0f;
        }
        check(1);
    }
}

TEST_F(SpeculativeSamplingKernelTest, MarkovArgmaxStep_GreedyMatchesDenseChainToken) {
    // Well-separated maxima collapse under tiny temperature in this fixture.
    const int batch_size = 8, vocab_size = 4096;
    auto      logits = torch::randn({batch_size, vocab_size}, floatCuda()) * 2.0f;
    auto      bias   = (torch::randn({batch_size, vocab_size}, floatCuda()) * 0.5f).to(torch::kBFloat16);
    logits.select(1, 23).fill_(100.0f);
    auto temperature = torch::full({batch_size}, 1.0e-6f, floatCuda());
    auto tokens      = runMarkovArgmax(logits, bias, rtp_llm::MarkovBiasDType::kBFloat16, temperature, stream_);

    auto q        = markovSoftmaxReference(logits.clone(), bias, temperature);
    auto expected = q.argmax(-1).to(torch::kInt32);
    EXPECT_TRUE(torch::equal(tokens, expected));
    EXPECT_TRUE(torch::equal(tokens, markovArgmaxReference(logits.clone(), bias, temperature)));
    // This fixture is one-hot; ties and close logits need not be.
    auto gathered = q.gather(1, expected.to(torch::kLong).unsqueeze(1));
    EXPECT_TRUE(torch::allclose(gathered, torch::ones({batch_size, 1}, floatCuda()), 1e-5, 1e-5));
}

TEST_F(SpeculativeSamplingKernelTest, MarkovArgmaxStep_UnalignedContiguousStorageOffsets) {
    constexpr int batch_size = 2, vocab_size = 8192;
    for (const auto dtype : {torch::kFloat32, torch::kBFloat16, torch::kHalf}) {
        const auto bias_dtype = dtype == torch::kFloat32  ? rtp_llm::MarkovBiasDType::kFloat32 :
                                dtype == torch::kBFloat16 ? rtp_llm::MarkovBiasDType::kBFloat16 :
                                                            rtp_llm::MarkovBiasDType::kFloat16;
        for (int offset_arg = 0; offset_arg < 2; ++offset_arg) {
            SCOPED_TRACE(offset_arg);
            auto logits = torch::randn({batch_size * vocab_size + 1}, floatCuda())
                              .narrow(0, offset_arg == 0 ? 1 : 0, batch_size * vocab_size)
                              .view({batch_size, vocab_size});
            auto bias = torch::randn({batch_size * vocab_size + 1}, floatCuda().dtype(dtype))
                            .narrow(0, offset_arg == 1 ? 1 : 0, batch_size * vocab_size)
                            .view({batch_size, vocab_size});
            auto temperature = torch::full({batch_size}, 1.0e-6f, floatCuda());
            ASSERT_TRUE(logits.is_contiguous());
            ASSERT_TRUE(bias.is_contiguous());
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            auto tokens = runMarkovArgmax(logits, bias, bias_dtype, temperature, stream_);
            EXPECT_TRUE(torch::equal(tokens, markovArgmaxReference(logits, bias, temperature)));
        }
    }
}

TEST_F(SpeculativeSamplingKernelTest, MarkovArgmaxStep_BatchSizeZeroIsNoop) {
    auto status = rtp_llm::invokeMarkovArgmaxStep(
        nullptr, 0, nullptr, rtp_llm::MarkovBiasDType::kBFloat16, nullptr, nullptr, 0, 128, nullptr, stream_);
    ASSERT_EQ(status, cudaSuccess);
}

TEST_F(SpeculativeSamplingKernelTest, MarkovArgmaxStep_RejectsBadArguments) {
    const int batch_size = 2, vocab_size = 64;
    auto      logits      = torch::randn({batch_size, vocab_size}, floatCuda());
    auto      bias        = torch::randn({batch_size, vocab_size}, floatCuda());
    auto      temperature = torch::ones({batch_size}, floatCuda());
    auto      tokens   = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    const int splits   = rtp_llm::markovSoftmaxStepSplits(batch_size, vocab_size);
    auto      partials = torch::empty({2, batch_size, splits}, floatCuda());
    // Row stride smaller than the row length.
    auto status = rtp_llm::invokeMarkovArgmaxStep(logits.data_ptr<float>(),
                                                  vocab_size - 1,
                                                  bias.data_ptr(),
                                                  rtp_llm::MarkovBiasDType::kFloat32,
                                                  temperature.data_ptr<float>(),
                                                  tokens.data_ptr<int32_t>(),
                                                  batch_size,
                                                  vocab_size,
                                                  partials.data_ptr<float>(),
                                                  stream_);
    ASSERT_EQ(status, cudaErrorInvalidValue);
    // Missing partials scratch.
    status = rtp_llm::invokeMarkovArgmaxStep(logits.data_ptr<float>(),
                                             vocab_size,
                                             bias.data_ptr(),
                                             rtp_llm::MarkovBiasDType::kFloat32,
                                             temperature.data_ptr<float>(),
                                             tokens.data_ptr<int32_t>(),
                                             batch_size,
                                             vocab_size,
                                             nullptr,
                                             stream_);
    ASSERT_EQ(status, cudaErrorInvalidValue);
    // vocab >= 2^24 breaks the float index slot contract (validated before
    // any memory is touched).
    status = rtp_llm::invokeMarkovArgmaxStep(logits.data_ptr<float>(),
                                             (1 << 24) + 1,
                                             bias.data_ptr(),
                                             rtp_llm::MarkovBiasDType::kFloat32,
                                             temperature.data_ptr<float>(),
                                             tokens.data_ptr<int32_t>(),
                                             batch_size,
                                             (1 << 24) + 1,
                                             partials.data_ptr<float>(),
                                             stream_);
    ASSERT_EQ(status, cudaErrorInvalidValue);
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
