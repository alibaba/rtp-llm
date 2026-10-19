#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>
#include <limits>

#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/markov_softmax.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/greedy_top1_renorm.h"
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

namespace {

torch::TensorOptions fxIntCuda() {
    return torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
}
torch::TensorOptions fxFloatCuda() {
    return torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
}
torch::TensorOptions fxBoolCuda() {
    return torch::TensorOptions(torch::kBool).device(torch::kCUDA);
}

// Mixed-outcome fixture for the folded finalization tests. Five rows cover
// every kernel exit path and both force states:
//   row 0: forced, greedy, rejects at position 1 (early-exit path, overridden)
//   row 1: unforced, greedy, accepts all + bonus (early-exit path)
//   row 2: forced, stochastic, accepts all (early-exit path, overridden)
//   row 3: unforced, stochastic, rejects at position 1 -> residual sampling
//          (residual exit path, -1 padding reset by the fold)
//   row 4: unforced, greedy, rejects at position 0 (direct target fallback)
// target_token_ids uses stride 3 with non-final columns filled with a decoy
// value, pinning the production contract that only the last column is read.
struct FinalizeFixture {
    static constexpr int batch_size    = 5;
    static constexpr int num_spec      = 3;
    static constexpr int vocab_size    = 64;
    static constexpr int target_stride = 3;
    static constexpr int sentinel      = 0x5A5A5A5A;

    torch::Tensor draft_probs;
    torch::Tensor target_probs;
    torch::Tensor draft_token_ids;
    torch::Tensor target_token_ids;
    torch::Tensor uniform_samples;
    torch::Tensor do_sample;
    torch::Tensor force_mask;  // device bool [batch]

    static FinalizeFixture make() {
        FinalizeFixture f;
        const int       B = batch_size, S = num_spec, V = vocab_size, W = target_stride;
        f.draft_token_ids = torch::empty({B, S}, fxIntCuda());
        auto draft_h      = f.draft_token_ids.cpu();
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
                // Distinct per row/position; token 0 avoided so the -1 -> 0
                // reset is observable against real tokens.
                draft_h[b][s] = 3 + b * S + s;
            }
        }
        f.draft_token_ids = draft_h.to(fxIntCuda());

        // Per (row, position) target argmax (last column):
        //   rows 1, 2: match the draft everywhere (all accept).
        //   row 0: match at 0, mismatch at 1 (greedy reject, forced).
        //   row 3: match at 0, mismatch at 1 (stochastic reject -> residual).
        //   row 4: mismatch at 0 (greedy reject).
        // Mismatch values 2+s never collide with any draft token (3+3b+s).
        auto target_argmax = torch::empty({B, S + 1}, torch::TensorOptions().dtype(torch::kInt32));
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
                const bool matches  = (b == 1 || b == 2) || (b == 0 && s == 0) || (b == 3 && s == 0);
                target_argmax[b][s] = matches ? draft_h[b][s].item<int>() : (2 + s);
            }
            target_argmax[b][S] = 40 + b;  // bonus token, distinct per row
        }
        f.target_token_ids = torch::full({B, S + 1, W}, 7777, fxIntCuda());  // decoy in non-final columns
        auto ttid_h        = f.target_token_ids.cpu();
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s <= S; ++s) {
                ttid_h[b][s][W - 1] = target_argmax[b][s];
            }
        }
        f.target_token_ids = ttid_h.to(fxIntCuda());

        // One-hot materialized probs consistent with the argmax columns; p is
        // one-hot on the draft token (p(draft) == 1), so stochastic rows accept
        // when the target matches (q=1 > u) and reject when it does not (q=0).
        f.draft_probs  = torch::zeros({B, S, V}, fxFloatCuda());
        f.target_probs = torch::zeros({B, S + 1, V}, fxFloatCuda());
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
                f.draft_probs.index_put_({b, s, draft_h[b][s].item<int>()}, 1.0f);
                f.target_probs.index_put_({b, s, target_argmax[b][s].item<int>()}, 1.0f);
            }
            f.target_probs.index_put_({b, S, target_argmax[b][S].item<int>()}, 1.0f);
        }

        f.uniform_samples = torch::full({B, S + 1}, 0.5f, fxFloatCuda());
        f.do_sample       = torch::tensor({false, false, true, true, false}, fxBoolCuda());
        f.force_mask      = torch::tensor({true, false, true, false, false}, fxBoolCuda());
        return f;
    }

    // The legacy eager finalization exactly as SpeculativeSampler::batchSample
    // ran it before the fold: where(force, draft+bonus) then -1 -> 0 reset.
    static std::pair<torch::Tensor, torch::Tensor> legacyChain(torch::Tensor        out_ids,
                                                               torch::Tensor        acc_num,
                                                               const torch::Tensor& draft_ids,
                                                               const torch::Tensor& ttids,
                                                               const torch::Tensor& force) {
        if (force.defined() && force.any().item<bool>()) {
            auto target_bonus_t = ttids.reshape({batch_size, num_spec + 1, target_stride});
            auto target_bonus   = target_bonus_t.select(1, num_spec).select(1, target_stride - 1).unsqueeze(1);
            auto forced_tokens  = torch::cat({draft_ids, target_bonus}, 1);
            auto force_mask_2d  = force.unsqueeze(1).expand_as(out_ids);
            out_ids             = torch::where(force_mask_2d, forced_tokens, out_ids);
            acc_num             = torch::where(force, torch::full_like(acc_num, num_spec + 1), acc_num);
        }
        out_ids.index_put_({out_ids == -1}, 0);
        return {out_ids, acc_num};
    }
};

}  // namespace

// The folded finalize phase must produce byte-identical outputs to the legacy
// eager chain (where + full_like + index_put_) for a mixed force/do_sample /
// accept-reject batch covering both kernel exit paths.
TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_FinalizeMatchesLegacyForceWhereChain) {
    auto f = FinalizeFixture::make();

    // Legacy: kernel without finalize, zeros-init outputs, then eager chain.
    auto legacy_ids = torch::zeros({FinalizeFixture::batch_size, FinalizeFixture::num_spec + 1}, intCuda());
    auto legacy_acc = torch::zeros({FinalizeFixture::batch_size}, intCuda());
    auto status     = rtp_llm::invokeRejectionSampling<float, int>(f.draft_probs.data_ptr<float>(),
                                                               f.draft_token_ids.data_ptr<int>(),
                                                               f.uniform_samples.data_ptr<float>(),
                                                               f.target_probs.data_ptr<float>(),
                                                               f.target_token_ids.data_ptr<int>(),
                                                               FinalizeFixture::target_stride,
                                                               legacy_ids.data_ptr<int>(),
                                                               legacy_acc.data_ptr<int>(),
                                                               f.do_sample.data_ptr<bool>(),
                                                               FinalizeFixture::batch_size,
                                                               FinalizeFixture::num_spec,
                                                               FinalizeFixture::vocab_size,
                                                               stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);
    auto [legacy_ids_ref, legacy_acc_ref] =
        FinalizeFixture::legacyChain(legacy_ids, legacy_acc, f.draft_token_ids, f.target_token_ids, f.force_mask);

    // Folded: sentinel-filled outputs (as if torch::empty garbage), finalize on.
    auto folded_ids =
        torch::full({FinalizeFixture::batch_size, FinalizeFixture::num_spec + 1}, FinalizeFixture::sentinel, intCuda());
    auto folded_acc = torch::full({FinalizeFixture::batch_size}, FinalizeFixture::sentinel, intCuda());
    auto status2    = rtp_llm::invokeRejectionSampling<float, int>(f.draft_probs.data_ptr<float>(),
                                                                f.draft_token_ids.data_ptr<int>(),
                                                                f.uniform_samples.data_ptr<float>(),
                                                                f.target_probs.data_ptr<float>(),
                                                                f.target_token_ids.data_ptr<int>(),
                                                                FinalizeFixture::target_stride,
                                                                folded_ids.data_ptr<int>(),
                                                                folded_acc.data_ptr<int>(),
                                                                f.do_sample.data_ptr<bool>(),
                                                                FinalizeFixture::batch_size,
                                                                FinalizeFixture::num_spec,
                                                                FinalizeFixture::vocab_size,
                                                                stream_,
                                                                /*draft_probs_point_mass=*/false,
                                                                /*target_probs_point_mass=*/false,
                                                                f.force_mask.data_ptr<bool>(),
                                                                /*finalize_outputs=*/true);
    ASSERT_EQ(status2, cudaSuccess);
    cudaStreamSynchronize(stream_);

    EXPECT_TRUE(torch::equal(legacy_ids_ref.cpu(), folded_ids.cpu())) << "legacy:\n"
                                                                      << legacy_ids_ref.cpu() << "\nfolded:\n"
                                                                      << folded_ids.cpu();
    EXPECT_TRUE(torch::equal(legacy_acc_ref.cpu(), folded_acc.cpu())) << "legacy:\n"
                                                                      << legacy_acc_ref.cpu() << "\nfolded:\n"
                                                                      << folded_acc.cpu();
    // Forced rows: draft tokens + bonus, full accept count (explicit spot-check).
    auto ids_h = folded_ids.cpu(), acc_h = folded_acc.cpu();
    for (int b : {0, 2}) {
        EXPECT_EQ(acc_h[b].item<int>(), FinalizeFixture::num_spec + 1);
        for (int s = 0; s < FinalizeFixture::num_spec; ++s) {
            EXPECT_EQ(ids_h[b][s].item<int>(), f.draft_token_ids.cpu()[b][s].item<int>());
        }
        EXPECT_EQ(ids_h[b][FinalizeFixture::num_spec].item<int>(), 40 + b);
    }
    // Unforced stochastic reject row 3 (residual exit path): accepted draft 12
    // at position 0, residual-sampled the one-hot target 3 at position 1,
    // padding reset to 0 by the fold.
    EXPECT_EQ(ids_h[3][0].item<int>(), 12);
    EXPECT_EQ(ids_h[3][1].item<int>(), 3);
    EXPECT_EQ(ids_h[3][2].item<int>(), 0);
    EXPECT_EQ(ids_h[3][3].item<int>(), 0);
    EXPECT_EQ(acc_h[3].item<int>(), 2);
    // Unforced greedy reject row 4: target fallback 2 at position 0, padding
    // reset to 0.
    EXPECT_EQ(ids_h[4][0].item<int>(), 2);
    EXPECT_EQ(ids_h[4][1].item<int>(), 0);
    EXPECT_EQ(acc_h[4].item<int>(), 1);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_FinalizeForcedResidualWithPointMassInputs) {
    auto f = FinalizeFixture::make();
    // Force every row, including the stochastic residual exit in row 3.
    f.force_mask.fill_(true);
    auto bonus = f.target_token_ids.select(1, FinalizeFixture::num_spec)
                     .select(1, FinalizeFixture::target_stride - 1)
                     .unsqueeze(1);
    auto expected = torch::cat({f.draft_token_ids, bonus}, 1);
    for (bool draft_point_mass : {false, true}) {
        for (bool target_point_mass : {false, true}) {
            SCOPED_TRACE(draft_point_mass);
            SCOPED_TRACE(target_point_mass);
            auto ids = torch::full_like(expected, FinalizeFixture::sentinel);
            auto acc = torch::full({FinalizeFixture::batch_size}, FinalizeFixture::sentinel, intCuda());
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            auto status = rtp_llm::invokeRejectionSampling<float, int>(
                draft_point_mass ? nullptr : f.draft_probs.data_ptr<float>(),
                f.draft_token_ids.data_ptr<int>(),
                f.uniform_samples.data_ptr<float>(),
                target_point_mass ? nullptr : f.target_probs.data_ptr<float>(),
                f.target_token_ids.data_ptr<int>(),
                FinalizeFixture::target_stride,
                ids.data_ptr<int>(),
                acc.data_ptr<int>(),
                f.do_sample.data_ptr<bool>(),
                FinalizeFixture::batch_size,
                FinalizeFixture::num_spec,
                FinalizeFixture::vocab_size,
                stream_,
                draft_point_mass,
                target_point_mass,
                f.force_mask.data_ptr<bool>(),
                true);
            ASSERT_EQ(status, cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
            EXPECT_TRUE(torch::equal(ids, expected));
            EXPECT_TRUE(acc.eq(FinalizeFixture::num_spec + 1).all().item<bool>());
        }
    }
}

// Without a force mask the folded finalize is exactly the -1 -> 0 reset:
// byte-identical to the legacy kernel output + index_put_.
TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_FinalizeResetParityNoForce) {
    auto f = FinalizeFixture::make();

    auto plain_ids = torch::zeros({FinalizeFixture::batch_size, FinalizeFixture::num_spec + 1}, intCuda());
    auto plain_acc = torch::zeros({FinalizeFixture::batch_size}, intCuda());
    auto status    = rtp_llm::invokeRejectionSampling<float, int>(f.draft_probs.data_ptr<float>(),
                                                               f.draft_token_ids.data_ptr<int>(),
                                                               f.uniform_samples.data_ptr<float>(),
                                                               f.target_probs.data_ptr<float>(),
                                                               f.target_token_ids.data_ptr<int>(),
                                                               FinalizeFixture::target_stride,
                                                               plain_ids.data_ptr<int>(),
                                                               plain_acc.data_ptr<int>(),
                                                               f.do_sample.data_ptr<bool>(),
                                                               FinalizeFixture::batch_size,
                                                               FinalizeFixture::num_spec,
                                                               FinalizeFixture::vocab_size,
                                                               stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);
    plain_ids.index_put_({plain_ids == -1}, 0);

    auto reset_ids =
        torch::full({FinalizeFixture::batch_size, FinalizeFixture::num_spec + 1}, FinalizeFixture::sentinel, intCuda());
    auto reset_acc = torch::full({FinalizeFixture::batch_size}, FinalizeFixture::sentinel, intCuda());
    auto status2   = rtp_llm::invokeRejectionSampling<float, int>(f.draft_probs.data_ptr<float>(),
                                                                f.draft_token_ids.data_ptr<int>(),
                                                                f.uniform_samples.data_ptr<float>(),
                                                                f.target_probs.data_ptr<float>(),
                                                                f.target_token_ids.data_ptr<int>(),
                                                                FinalizeFixture::target_stride,
                                                                reset_ids.data_ptr<int>(),
                                                                reset_acc.data_ptr<int>(),
                                                                f.do_sample.data_ptr<bool>(),
                                                                FinalizeFixture::batch_size,
                                                                FinalizeFixture::num_spec,
                                                                FinalizeFixture::vocab_size,
                                                                stream_,
                                                                /*draft_probs_point_mass=*/false,
                                                                /*target_probs_point_mass=*/false,
                                                                /*force_mask=*/nullptr,
                                                                /*finalize_outputs=*/true);
    ASSERT_EQ(status2, cudaSuccess);
    cudaStreamSynchronize(stream_);

    EXPECT_TRUE(torch::equal(plain_ids.cpu(), reset_ids.cpu()));
    EXPECT_TRUE(torch::equal(plain_acc.cpu(), reset_acc.cpu()));
    // No row is forced: row 3 keeps its residual-sampled token at position 1
    // and its -1 padding is reset to 0 by the fold.
    EXPECT_EQ(reset_ids.cpu()[3][1].item<int>(), 3);
    EXPECT_EQ(reset_ids.cpu()[3][2].item<int>(), 0);
}

// The finalize contract that licenses torch::empty output buffers in
// batchSample: with finalize on, the kernel writes EVERY element of both
// output buffers on every path — no sentinel may survive a mixed batch.
TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_FinalizeWritesEveryOutputElement) {
    auto f = FinalizeFixture::make();

    auto ids =
        torch::full({FinalizeFixture::batch_size, FinalizeFixture::num_spec + 1}, FinalizeFixture::sentinel, intCuda());
    auto acc    = torch::full({FinalizeFixture::batch_size}, FinalizeFixture::sentinel, intCuda());
    auto status = rtp_llm::invokeRejectionSampling<float, int>(f.draft_probs.data_ptr<float>(),
                                                               f.draft_token_ids.data_ptr<int>(),
                                                               f.uniform_samples.data_ptr<float>(),
                                                               f.target_probs.data_ptr<float>(),
                                                               f.target_token_ids.data_ptr<int>(),
                                                               FinalizeFixture::target_stride,
                                                               ids.data_ptr<int>(),
                                                               acc.data_ptr<int>(),
                                                               f.do_sample.data_ptr<bool>(),
                                                               FinalizeFixture::batch_size,
                                                               FinalizeFixture::num_spec,
                                                               FinalizeFixture::vocab_size,
                                                               stream_,
                                                               /*draft_probs_point_mass=*/false,
                                                               /*target_probs_point_mass=*/false,
                                                               f.force_mask.data_ptr<bool>(),
                                                               /*finalize_outputs=*/true);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    EXPECT_EQ((ids.cpu() == FinalizeFixture::sentinel).sum().item<int64_t>(), 0);
    EXPECT_EQ((acc.cpu() == FinalizeFixture::sentinel).sum().item<int64_t>(), 0);
}

// Default-off contract: with no finalize flag a provided force mask is inert
// and -1 padding survives — every existing caller keeps the legacy bytes.
TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_FinalizeDefaultsPreserveLegacy) {
    auto f = FinalizeFixture::make();

    auto ids = torch::zeros({FinalizeFixture::batch_size, FinalizeFixture::num_spec + 1}, intCuda());
    auto acc = torch::zeros({FinalizeFixture::batch_size}, intCuda());
    // force_mask non-null but finalize_outputs=false (the default): the mask
    // must be ignored and the raw rejection output (with -1 padding) kept.
    auto status = rtp_llm::invokeRejectionSampling<float, int>(f.draft_probs.data_ptr<float>(),
                                                               f.draft_token_ids.data_ptr<int>(),
                                                               f.uniform_samples.data_ptr<float>(),
                                                               f.target_probs.data_ptr<float>(),
                                                               f.target_token_ids.data_ptr<int>(),
                                                               FinalizeFixture::target_stride,
                                                               ids.data_ptr<int>(),
                                                               acc.data_ptr<int>(),
                                                               f.do_sample.data_ptr<bool>(),
                                                               FinalizeFixture::batch_size,
                                                               FinalizeFixture::num_spec,
                                                               FinalizeFixture::vocab_size,
                                                               stream_,
                                                               /*draft_probs_point_mass=*/false,
                                                               /*target_probs_point_mass=*/false,
                                                               f.force_mask.data_ptr<bool>(),
                                                               /*finalize_outputs=*/false);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    auto ids_h = ids.cpu();
    // Unforced greedy reject row 4 keeps -1 padding (no reset without finalize).
    EXPECT_EQ(ids_h[4][0].item<int>(), 2);
    EXPECT_EQ(ids_h[4][1].item<int>(), -1);
    EXPECT_EQ(ids_h[4][2].item<int>(), -1);
    EXPECT_EQ(ids_h[4][3].item<int>(), -1);
    EXPECT_EQ(acc.cpu()[4].item<int>(), 1);
    // Forced row 0 keeps its raw rejection result (accepted draft 3 at
    // position 0, greedy target fallback 3 at position 1, -1 padding), not
    // the draft+bonus override.
    EXPECT_EQ(ids_h[0][0].item<int>(), 3);
    EXPECT_EQ(ids_h[0][1].item<int>(), 3);
    EXPECT_EQ(ids_h[0][2].item<int>(), -1);
    EXPECT_EQ(ids_h[0][3].item<int>(), -1);
    // Row 0: accepted at position 0 (draft==target), greedy-rejected at 1
    // -> pos=1, accept count = pos+1 = 2.
    EXPECT_EQ(acc.cpu()[0].item<int>(), 2);
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

// ─────────────────────────────────────────────────────────────────────────────
// invokeGreedyTop1Renorm tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(SpeculativeSamplingKernelTest, GreedyTop1Renorm_MatchesTorchArgmaxOneHot) {
    // The fused op must reproduce exactly what the replaced chain produced for
    // tie-free rows: token = torch::argmax(logits) (== argmax(softmax(logits))
    // by strict monotonicity) and all_probs = one-hot at that index
    // (== flashinfer top_k_renorm_probs(k=1)).
    const int batch_size = 32, vocab_size = 4096;
    auto      logits     = torch::randn({batch_size, vocab_size}, floatCuda()) * 3.0f;
    auto      logits_ref = logits.clone();
    auto      onehot     = torch::full({batch_size, vocab_size}, -777.0f, floatCuda());
    auto      samples    = torch::full({batch_size}, -1, intCuda());
    const int splits     = rtp_llm::greedyTop1RenormSplits(batch_size, vocab_size);
    auto      partials   = torch::empty({2, batch_size, splits}, floatCuda());

    auto status = rtp_llm::invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                                  onehot.data_ptr<float>(),
                                                  samples.data_ptr<int32_t>(),
                                                  batch_size,
                                                  vocab_size,
                                                  partials.data_ptr<float>(),
                                                  stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    auto expected = logits_ref.argmax(-1);  // first maximum, matching the kernel
    EXPECT_TRUE(torch::equal(samples.to(torch::kLong), expected));
    auto expected_onehot = torch::zeros({batch_size, vocab_size}, floatCuda());
    expected_onehot.scatter_(1, expected.unsqueeze(1), 1.0f);
    EXPECT_TRUE(torch::equal(onehot, expected_onehot));
    EXPECT_TRUE(torch::equal(onehot.sum(-1), torch::ones({batch_size}, floatCuda())));
    // Read-only contract: logits must be untouched (no probs writeback).
    EXPECT_TRUE(torch::equal(logits, logits_ref));
}

TEST_F(SpeculativeSamplingKernelTest, GreedyTop1Renorm_ProductionShapeWideVocab) {
    // The MTP verify shape: [B*(gamma+1), vocab] fp32 rows.
    const int batch_size = 192, vocab_size = 129280;
    auto      logits     = torch::randn({batch_size, vocab_size}, floatCuda());
    auto      logits_ref = logits.clone();
    auto      onehot     = torch::full({batch_size, vocab_size}, -777.0f, floatCuda());
    auto      samples    = torch::full({batch_size}, -1, intCuda());
    const int splits     = rtp_llm::greedyTop1RenormSplits(batch_size, vocab_size);
    auto      partials   = torch::empty({2, batch_size, splits}, floatCuda());

    auto status = rtp_llm::invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                                  onehot.data_ptr<float>(),
                                                  samples.data_ptr<int32_t>(),
                                                  batch_size,
                                                  vocab_size,
                                                  partials.data_ptr<float>(),
                                                  stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    auto expected = logits_ref.argmax(-1);
    EXPECT_TRUE(torch::equal(samples.to(torch::kLong), expected));
    auto expected_onehot = torch::zeros({batch_size, vocab_size}, floatCuda());
    expected_onehot.scatter_(1, expected.unsqueeze(1), 1.0f);
    EXPECT_TRUE(torch::equal(onehot, expected_onehot));
    EXPECT_TRUE(torch::equal(logits, logits_ref));
}

TEST_F(SpeculativeSamplingKernelTest, GreedyTop1Renorm_ScalarPathOddVocab) {
    const int batch_size = 8, vocab_size = 4097;  // not a multiple of 4
    auto      logits     = torch::randn({batch_size, vocab_size}, floatCuda()) * 2.0f;
    auto      logits_ref = logits.clone();
    auto      onehot     = torch::full({batch_size, vocab_size}, -777.0f, floatCuda());
    auto      samples    = torch::full({batch_size}, -1, intCuda());
    const int splits     = rtp_llm::greedyTop1RenormSplits(batch_size, vocab_size);
    auto      partials   = torch::empty({2, batch_size, splits}, floatCuda());

    auto status = rtp_llm::invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                                  onehot.data_ptr<float>(),
                                                  samples.data_ptr<int32_t>(),
                                                  batch_size,
                                                  vocab_size,
                                                  partials.data_ptr<float>(),
                                                  stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    auto expected = logits_ref.argmax(-1);
    EXPECT_TRUE(torch::equal(samples.to(torch::kLong), expected));
    auto expected_onehot = torch::zeros({batch_size, vocab_size}, floatCuda());
    expected_onehot.scatter_(1, expected.unsqueeze(1), 1.0f);
    EXPECT_TRUE(torch::equal(onehot, expected_onehot));
}

TEST_F(SpeculativeSamplingKernelTest, GreedyTop1Renorm_TiesResolveToFirstMaximum) {
    // torch::argmax returns the first maximal value; the fused kernel must
    // match, including rows whose maximum repeats across split boundaries and
    // degenerate all-equal rows.
    auto      logits     = torch::tensor({{0.5f, 3.0f, 3.0f, 1.0f, 3.0f, 2.0f},
                                          {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                                          {-1.0f, -5.0f, -1.0f, -5.0f, -5.0f, -5.0f},
                                          {2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f}},
                                floatCuda());
    const int batch_size = 4, vocab_size = 6;
    auto      onehot   = torch::full({batch_size, vocab_size}, -777.0f, floatCuda());
    auto      samples  = torch::full({batch_size}, -1, intCuda());
    const int splits   = rtp_llm::greedyTop1RenormSplits(batch_size, vocab_size);
    auto      partials = torch::empty({2, batch_size, splits}, floatCuda());

    auto status = rtp_llm::invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                                  onehot.data_ptr<float>(),
                                                  samples.data_ptr<int32_t>(),
                                                  batch_size,
                                                  vocab_size,
                                                  partials.data_ptr<float>(),
                                                  stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    EXPECT_TRUE(torch::equal(samples.to(torch::kLong), logits.argmax(-1)));
    auto expected_onehot = torch::zeros({batch_size, vocab_size}, floatCuda());
    expected_onehot.scatter_(1, logits.argmax(-1).unsqueeze(1), 1.0f);
    EXPECT_TRUE(torch::equal(onehot, expected_onehot));
}

TEST_F(SpeculativeSamplingKernelTest, GreedyTop1Renorm_NaNsAndTiesMatchTorch) {
    for (const int vocab_size : {4, 5, 131072, 131073}) {
        SCOPED_TRACE(vocab_size);
        auto logits   = torch::zeros({1, vocab_size}, floatCuda());
        auto onehot   = torch::empty_like(logits);
        auto samples  = torch::empty({1}, intCuda());
        auto partials = torch::empty({2, 1, rtp_llm::greedyTop1RenormSplits(1, vocab_size)}, floatCuda());
        auto check    = [&](int expected) {
            for (bool write_onehot : {false, true}) {
                SCOPED_TRACE(write_onehot);
                onehot.fill_(-777.0f);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                ASSERT_EQ(rtp_llm::invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                                          onehot.data_ptr<float>(),
                                                          samples.data_ptr<int32_t>(),
                                                          1,
                                                          vocab_size,
                                                          partials.data_ptr<float>(),
                                                          stream_,
                                                          write_onehot),
                          cudaSuccess);
                ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
                EXPECT_EQ(samples[0].item<int32_t>(), expected);
                auto reference = logits.argmax(-1);
                EXPECT_TRUE(torch::equal(samples, reference.to(torch::kInt32)));
                if (write_onehot) {
                    EXPECT_TRUE(
                        torch::equal(onehot, torch::zeros_like(logits).scatter_(1, reference.unsqueeze(1), 1.0f)));
                } else {
                    EXPECT_TRUE(onehot.eq(-777.0f).all().item<bool>());
                }
            }
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

TEST_F(SpeculativeSamplingKernelTest, GreedyTop1Renorm_UnalignedContiguousStorageOffsets) {
    constexpr int batch_size = 3, vocab_size = 8192;
    for (int offset_arg = 0; offset_arg < 2; ++offset_arg) {
        for (bool write_onehot : {false, true}) {
            SCOPED_TRACE(offset_arg);
            SCOPED_TRACE(write_onehot);
            auto logits = torch::randn({batch_size * vocab_size + 1}, floatCuda())
                              .narrow(0, offset_arg == 0 ? 1 : 0, batch_size * vocab_size)
                              .view({batch_size, vocab_size});
            auto onehot = torch::full({batch_size * vocab_size + 1}, -777.0f, floatCuda())
                              .narrow(0, offset_arg == 1 ? 1 : 0, batch_size * vocab_size)
                              .view({batch_size, vocab_size});
            auto samples = torch::empty({batch_size}, intCuda());
            auto partials =
                torch::empty({2, batch_size, rtp_llm::greedyTop1RenormSplits(batch_size, vocab_size)}, floatCuda());
            ASSERT_TRUE(logits.is_contiguous());
            ASSERT_TRUE(onehot.is_contiguous());
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            ASSERT_EQ(rtp_llm::invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                                      onehot.data_ptr<float>(),
                                                      samples.data_ptr<int32_t>(),
                                                      batch_size,
                                                      vocab_size,
                                                      partials.data_ptr<float>(),
                                                      stream_,
                                                      write_onehot),
                      cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
            auto expected = logits.argmax(-1);
            EXPECT_TRUE(torch::equal(samples, expected.to(torch::kInt32)));
            if (write_onehot) {
                EXPECT_TRUE(torch::equal(onehot, torch::zeros_like(logits).scatter_(1, expected.unsqueeze(1), 1.0f)));
            } else {
                EXPECT_TRUE(onehot.eq(-777.0f).all().item<bool>());
            }
        }
    }
}

TEST_F(SpeculativeSamplingKernelTest, GreedyTop1Renorm_BatchSizeZeroIsNoop) {
    auto status = rtp_llm::invokeGreedyTop1Renorm(nullptr, nullptr, nullptr, 0, 128, nullptr, stream_);
    ASSERT_EQ(status, cudaSuccess);
}

TEST_F(SpeculativeSamplingKernelTest, GreedyTop1Renorm_RejectsBadArguments) {
    const int batch_size = 2, vocab_size = 64;
    auto      logits   = torch::randn({batch_size, vocab_size}, floatCuda());
    auto      onehot   = torch::empty({batch_size, vocab_size}, floatCuda());
    auto      samples  = torch::empty({batch_size}, intCuda());
    const int splits   = rtp_llm::greedyTop1RenormSplits(batch_size, vocab_size);
    auto      partials = torch::empty({2, batch_size, splits}, floatCuda());
    // Missing partials scratch.
    auto status = rtp_llm::invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                                  onehot.data_ptr<float>(),
                                                  samples.data_ptr<int32_t>(),
                                                  batch_size,
                                                  vocab_size,
                                                  nullptr,
                                                  stream_);
    ASSERT_EQ(status, cudaErrorInvalidValue);
    // Missing output.
    status = rtp_llm::invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                             nullptr,
                                             samples.data_ptr<int32_t>(),
                                             batch_size,
                                             vocab_size,
                                             partials.data_ptr<float>(),
                                             stream_);
    ASSERT_EQ(status, cudaErrorInvalidValue);
    // Vocab beyond the exact float-index bound.
    status = rtp_llm::invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                             onehot.data_ptr<float>(),
                                             samples.data_ptr<int32_t>(),
                                             batch_size,
                                             (1 << 24),
                                             partials.data_ptr<float>(),
                                             stream_);
    ASSERT_EQ(status, cudaErrorInvalidValue);
}

// ─────────────────────────────────────────────────────────────────────────────
// Point-mass contract tests: the token-only greedy mode and the rejection
// kernel's target_probs_point_mass synthesis must be bit-identical to the
// materialized one-hot matrix path.
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(SpeculativeSamplingKernelTest, GreedyTop1Renorm_TokenOnlySkipsOnehot) {
    // write_onehot=false: samples still carry the first argmax, the one-hot
    // destination is left untouched (sentinel survives), logits stay raw.
    const int batch_size = 8, vocab_size = 4097;  // scalar path (odd vocab)
    auto      logits     = torch::randn({batch_size, vocab_size}, floatCuda()) * 2.0f;
    auto      logits_ref = logits.clone();
    auto      onehot     = torch::full({batch_size, vocab_size}, -777.0f, floatCuda());
    auto      samples    = torch::full({batch_size}, -1, intCuda());
    const int splits     = rtp_llm::greedyTop1RenormSplits(batch_size, vocab_size);
    auto      partials   = torch::empty({2, batch_size, splits}, floatCuda());

    auto status = rtp_llm::invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                                  onehot.data_ptr<float>(),
                                                  samples.data_ptr<int32_t>(),
                                                  batch_size,
                                                  vocab_size,
                                                  partials.data_ptr<float>(),
                                                  stream_,
                                                  /*write_onehot=*/false);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    EXPECT_TRUE(torch::equal(samples.to(torch::kLong), logits_ref.argmax(-1)));
    EXPECT_TRUE(torch::equal(onehot, torch::full({batch_size, vocab_size}, -777.0f, floatCuda())));
    EXPECT_TRUE(torch::equal(logits, logits_ref));

    // The same run on the vector path (vocab multiple of 4) must agree with a
    // full-write run's tokens.
    const int vocab_vec = 4096;
    auto      logits_v  = torch::randn({batch_size, vocab_vec}, floatCuda());
    auto      onehot_v  = torch::full({batch_size, vocab_vec}, -777.0f, floatCuda());
    auto      samples_v = torch::full({batch_size}, -1, intCuda());
    auto      partials_v =
        torch::empty({2, batch_size, rtp_llm::greedyTop1RenormSplits(batch_size, vocab_vec)}, floatCuda());
    ASSERT_EQ(rtp_llm::invokeGreedyTop1Renorm(logits_v.data_ptr<float>(),
                                              onehot_v.data_ptr<float>(),
                                              samples_v.data_ptr<int32_t>(),
                                              batch_size,
                                              vocab_vec,
                                              partials_v.data_ptr<float>(),
                                              stream_,
                                              /*write_onehot=*/false),
              cudaSuccess);
    cudaStreamSynchronize(stream_);
    EXPECT_TRUE(torch::equal(samples_v.to(torch::kLong), logits_v.argmax(-1)));
    EXPECT_TRUE(torch::equal(onehot_v, torch::full({batch_size, vocab_vec}, -777.0f, floatCuda())));
}

TEST_F(SpeculativeSamplingKernelTest, GreedyTop1Renorm_TokenOnlyAllowsNullOnehot) {
    const int batch_size = 4, vocab_size = 64;
    auto      logits   = torch::randn({batch_size, vocab_size}, floatCuda());
    auto      samples  = torch::full({batch_size}, -1, intCuda());
    const int splits   = rtp_llm::greedyTop1RenormSplits(batch_size, vocab_size);
    auto      partials = torch::empty({2, batch_size, splits}, floatCuda());
    auto      status   = rtp_llm::invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                                  nullptr,
                                                  samples.data_ptr<int32_t>(),
                                                  batch_size,
                                                  vocab_size,
                                                  partials.data_ptr<float>(),
                                                  stream_,
                                                  /*write_onehot=*/false);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);
    EXPECT_TRUE(torch::equal(samples.to(torch::kLong), logits.argmax(-1)));
    // With write_onehot=true a null destination stays invalid.
    ASSERT_EQ(rtp_llm::invokeGreedyTop1Renorm(logits.data_ptr<float>(),
                                              nullptr,
                                              samples.data_ptr<int32_t>(),
                                              batch_size,
                                              vocab_size,
                                              partials.data_ptr<float>(),
                                              stream_),
              cudaErrorInvalidValue);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_TargetPointMass_GreedyParity) {
    // Greedy rows (do_sample=false): synthesized q must reproduce the accept /
    // direct-target-fallback decisions of the materialized one-hot matrix
    // bit-for-bit, with a multi-column target_token_ids stride.
    torch::manual_seed(1234);
    const int batch_size = 8, num_spec = 5, vocab_size = 1000, target_stride = 3;

    auto target_token_ids = torch::randint(0, vocab_size, {batch_size, num_spec + 1, target_stride}, intCuda());
    auto target_last      = target_token_ids.select(2, target_stride - 1);  // [B, num_spec+1]
    // Force a deterministic mix of matches and mismatches against the greedy
    // target column so both accept and reject branches are exercised.
    auto draft_token_ids = torch::randint(0, vocab_size, {batch_size, num_spec}, torch::kInt32).contiguous();
    {
        auto     target_last_h = target_last.to(torch::kCPU).contiguous();
        int32_t* dp            = draft_token_ids.data_ptr<int32_t>();
        auto     tp            = target_last_h.data_ptr<int32_t>();
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < num_spec; ++s) {
                const int32_t tgt = tp[b * (num_spec + 1) + s];
                // Odd (b + s): guaranteed mismatch — shift within the vocab.
                dp[b * num_spec + s] =
                    ((b + s) % 2 == 0) ? tgt : static_cast<int32_t>((tgt + 1 + (s % 7)) % vocab_size);
            }
        }
        draft_token_ids = draft_token_ids.to(intCuda());
    }

    auto target_probs_onehot = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());
    target_probs_onehot.scatter_(2, target_last.to(torch::kLong).unsqueeze(-1), 1.0f);

    auto draft_probs     = torch::rand({batch_size, num_spec, vocab_size}, floatCuda());
    draft_probs          = draft_probs / draft_probs.sum(-1, true);
    auto uniform_samples = torch::rand({batch_size, num_spec + 1}, floatCuda());
    auto do_sample       = torch::zeros({batch_size}, boolCuda());

    auto run_once = [&](const torch::Tensor& target_probs, bool target_point_mass) {
        auto output_token_ids = torch::full({batch_size, num_spec + 1}, -1, intCuda());
        auto output_acc_num   = torch::zeros({batch_size}, intCuda());
        auto status           = rtp_llm::invokeRejectionSampling<float, int>(draft_probs.data_ptr<float>(),
                                                                   draft_token_ids.data_ptr<int>(),
                                                                   uniform_samples.data_ptr<float>(),
                                                                   target_probs.data_ptr<float>(),
                                                                   target_token_ids.data_ptr<int>(),
                                                                   target_stride,
                                                                   output_token_ids.data_ptr<int>(),
                                                                   output_acc_num.data_ptr<int>(),
                                                                   do_sample.data_ptr<bool>(),
                                                                   batch_size,
                                                                   num_spec,
                                                                   vocab_size,
                                                                   stream_,
                                                                   /*draft_probs_point_mass=*/false,
                                                                   target_point_mass);
        EXPECT_EQ(status, cudaSuccess);
        cudaStreamSynchronize(stream_);
        return std::make_pair(output_token_ids.to(torch::kCPU), output_acc_num.to(torch::kCPU));
    };

    auto [ids_matrix, acc_matrix] = run_once(target_probs_onehot, false);
    auto garbage                  = torch::full({batch_size, num_spec + 1, vocab_size}, -777.0f, floatCuda());
    auto [ids_point, acc_point]   = run_once(garbage, true);

    EXPECT_TRUE(torch::equal(ids_matrix, ids_point));
    EXPECT_TRUE(torch::equal(acc_matrix, acc_point));
    // Non-vacuous: the forced mismatches must reject some rows before the end
    // while other rows accept at least their first proposal.
    EXPECT_TRUE((acc_matrix < num_spec + 1).any().item<bool>());
    EXPECT_TRUE((acc_matrix > 1).any().item<bool>());
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_TargetPointMass_StochasticResidualParity) {
    // Stochastic rows (do_sample=true) with dense draft probabilities: the
    // residual relu(q - p) sampling must produce identical tokens whether q
    // comes from the one-hot matrix or is synthesized from target_token_ids.
    torch::manual_seed(987);
    const int batch_size = 8, num_spec = 3, vocab_size = 1000, target_stride = 1;

    auto target_token_ids = torch::randint(0, vocab_size, {batch_size, num_spec + 1, target_stride}, intCuda());
    auto draft_token_ids  = torch::randint(0, vocab_size, {batch_size, num_spec}, intCuda());
    auto target_last      = target_token_ids.select(2, target_stride - 1);

    auto target_probs_onehot = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());
    target_probs_onehot.scatter_(2, target_last.to(torch::kLong).unsqueeze(-1), 1.0f);

    auto draft_probs     = torch::rand({batch_size, num_spec, vocab_size}, floatCuda());
    draft_probs          = draft_probs / draft_probs.sum(-1, true);
    auto uniform_samples = torch::rand({batch_size, num_spec + 1}, floatCuda());
    auto do_sample       = torch::ones({batch_size}, boolCuda());

    auto run_once = [&](const torch::Tensor& target_probs, bool target_point_mass) {
        auto output_token_ids = torch::full({batch_size, num_spec + 1}, -1, intCuda());
        auto output_acc_num   = torch::zeros({batch_size}, intCuda());
        auto status           = rtp_llm::invokeRejectionSampling<float, int>(draft_probs.data_ptr<float>(),
                                                                   draft_token_ids.data_ptr<int>(),
                                                                   uniform_samples.data_ptr<float>(),
                                                                   target_probs.data_ptr<float>(),
                                                                   target_token_ids.data_ptr<int>(),
                                                                   target_stride,
                                                                   output_token_ids.data_ptr<int>(),
                                                                   output_acc_num.data_ptr<int>(),
                                                                   do_sample.data_ptr<bool>(),
                                                                   batch_size,
                                                                   num_spec,
                                                                   vocab_size,
                                                                   stream_,
                                                                   /*draft_probs_point_mass=*/false,
                                                                   target_point_mass);
        EXPECT_EQ(status, cudaSuccess);
        cudaStreamSynchronize(stream_);
        return std::make_pair(output_token_ids.to(torch::kCPU), output_acc_num.to(torch::kCPU));
    };

    auto [ids_matrix, acc_matrix] = run_once(target_probs_onehot, false);
    auto garbage                  = torch::full({batch_size, num_spec + 1, vocab_size}, -777.0f, floatCuda());
    auto [ids_point, acc_point]   = run_once(garbage, true);

    EXPECT_TRUE(torch::equal(ids_matrix, ids_point));
    EXPECT_TRUE(torch::equal(acc_matrix, acc_point));
    // Non-vacuous: with dense random draft probs and random uniforms, the
    // residual path must trigger (some row rejects before the bonus slot).
    EXPECT_TRUE((acc_matrix < num_spec + 1).any().item<bool>());
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_BothPointMass_Parity) {
    // DSpARK-style draft (no draft_probs at all) combined with the point-mass
    // target: accept reduces to same_token and the residual concentrates on
    // the target token — identical to the matrix-based run.
    torch::manual_seed(55);
    const int batch_size = 4, num_spec = 3, vocab_size = 512, target_stride = 2;

    auto target_token_ids = torch::randint(0, vocab_size, {batch_size, num_spec + 1, target_stride}, intCuda());
    auto target_last      = target_token_ids.select(2, target_stride - 1);
    // Half the proposals match the greedy target token.
    auto draft_token_ids = torch::randint(0, vocab_size, {batch_size, num_spec}, torch::kInt32).contiguous();
    {
        auto     target_last_h = target_last.to(torch::kCPU).contiguous();
        int32_t* dp            = draft_token_ids.data_ptr<int32_t>();
        auto     tp            = target_last_h.data_ptr<int32_t>();
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < num_spec; ++s) {
                if ((b + s) % 2 == 0) {
                    dp[b * num_spec + s] = tp[b * (num_spec + 1) + s];
                }
            }
        }
        draft_token_ids = draft_token_ids.to(intCuda());
    }

    auto target_probs_onehot = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());
    target_probs_onehot.scatter_(2, target_last.to(torch::kLong).unsqueeze(-1), 1.0f);

    auto uniform_samples = torch::rand({batch_size, num_spec + 1}, floatCuda());
    auto do_sample       = torch::ones({batch_size}, boolCuda());

    auto run_once = [&](const torch::Tensor& target_probs, bool target_point_mass) {
        auto output_token_ids = torch::full({batch_size, num_spec + 1}, -1, intCuda());
        auto output_acc_num   = torch::zeros({batch_size}, intCuda());
        auto status           = rtp_llm::invokeRejectionSampling<float, int>(nullptr,
                                                                   draft_token_ids.data_ptr<int>(),
                                                                   uniform_samples.data_ptr<float>(),
                                                                   target_probs.data_ptr<float>(),
                                                                   target_token_ids.data_ptr<int>(),
                                                                   target_stride,
                                                                   output_token_ids.data_ptr<int>(),
                                                                   output_acc_num.data_ptr<int>(),
                                                                   do_sample.data_ptr<bool>(),
                                                                   batch_size,
                                                                   num_spec,
                                                                   vocab_size,
                                                                   stream_,
                                                                   /*draft_probs_point_mass=*/true,
                                                                   target_point_mass);
        EXPECT_EQ(status, cudaSuccess);
        cudaStreamSynchronize(stream_);
        return std::make_pair(output_token_ids.to(torch::kCPU), output_acc_num.to(torch::kCPU));
    };

    auto [ids_matrix, acc_matrix] = run_once(target_probs_onehot, false);
    auto garbage                  = torch::full({batch_size, num_spec + 1, vocab_size}, -777.0f, floatCuda());
    auto [ids_point, acc_point]   = run_once(garbage, true);

    EXPECT_TRUE(torch::equal(ids_matrix, ids_point));
    EXPECT_TRUE(torch::equal(acc_matrix, acc_point));
    EXPECT_TRUE((acc_matrix < num_spec + 1).any().item<bool>());
}
