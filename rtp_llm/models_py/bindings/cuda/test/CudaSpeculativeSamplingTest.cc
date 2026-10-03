#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>
#include <limits>
#include <array>
#include <algorithm>
#include <cmath>
#include <functional>

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
    static torch::TensorOptions bfloatCuda() {
        return torch::TensorOptions(torch::kBFloat16).device(torch::kCUDA);
    }
};

// Finite weighted-uniform enumeration calls the REAL CUDA planner/rejection.
// Scope: conditional probability inputs, adaptive planning and production cap;
// NOT model logits/Markov GEMM, RNG, confidence projection, EOS/KV or Graph.

TEST_F(SpeculativeSamplingKernelTest, DSparkAdaptiveMarkovJointDistribution) {
    constexpr int gamma = 3, horizon = 4, outcomes = 1 << horizon;
    constexpr int vocab = 2, paths = 64, variants = 16;
    constexpr int rows       = paths * 2 * variants;
    const float   P[2][2][2] = {{{.7f, .3f}, {.25f, .75f}}, {{.4f, .6f}, {.8f, .2f}}};
    const float   Q[2][2][2] = {{{.2f, .8f}, {.9f, .1f}}, {{.85f, .15f}, {.1f, .9f}}};
    const float   C[2][2]    = {{.96f, .12f}, {.22f, .94f}};
    const int     anchors[2] = {0, 1};
    using Distribution       = std::array<double, outcomes>;
    auto cpuFloat            = torch::TensorOptions().dtype(torch::kFloat32);
    auto cpuInt              = torch::TensorOptions().dtype(torch::kInt32);
    auto token               = [](int path, int request, int step) { return (path >> (request * gamma + step)) & 1; };
    auto pathWeight          = [&](int path, int request, const float matrix[2][2][2]) {
        double weight   = 1.;
        int    previous = anchors[request];
        for (int j = 0; j < gamma; ++j) {
            int x = token(path, request, j);
            weight *= matrix[request][previous][x];
            previous = x;
        }
        return weight;
    };
    auto finish = [&](int request, const std::vector<int>& prefix, double weight, Distribution& result) {
        std::function<void(int, int, int, double)> visit;
        visit = [&](int size, int previous, int code, double probability) {
            if (size == horizon) {
                result[code] += probability;
                return;
            }
            for (int x = 0; x < 2; ++x)
                visit(size + 1, x, code | (x << size), probability * P[request][previous][x]);
        };
        int code = 0, size = std::min<int>(prefix.size(), horizon);
        for (int j = 0; j < size; ++j)
            code |= prefix[j] << j;
        visit(size, size ? prefix[size - 1] : anchors[request], code, weight);
    };
    Distribution expected0{}, expected1{};
    finish(0, {}, 1., expected0);
    finish(1, {}, 1., expected1);

    auto confidenceCpu = torch::empty({paths, 2, gamma}, cpuFloat);
    auto ca            = confidenceCpu.accessor<float, 3>();
    for (int path = 0; path < paths; ++path)
        for (int r = 0; r < 2; ++r)
            for (int j = 0; j < gamma; ++j)
                ca[path][r][j] = C[r][j ? token(path, r, j - 1) : anchors[r]];
    auto confidence = confidenceCpu.to(torch::kCUDA);
    for (int budget = 0; budget <= 2 * gamma; ++budget) {
        SCOPED_TRACE(budget);
        auto capsGpu = torch::empty({paths, 2}, intCuda());
        auto mapGpu  = torch::empty({paths, 2 + budget}, intCuda());
        // All Torch default-stream staging must finish before custom stream_ use.
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (int path = 0; path < paths; ++path)
            ASSERT_EQ(rtp_llm::invokeDSparkVerifyPlan(confidence.data_ptr<float>() + path * 2 * gamma,
                                                      capsGpu.data_ptr<int>() + path * 2,
                                                      mapGpu.data_ptr<int>() + path * (2 + budget),
                                                      2,
                                                      gamma,
                                                      budget,
                                                      stream_),
                      cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        auto capsCpu = capsGpu.cpu(), mapCpu = mapGpu.cpu();
        auto caps    = capsCpu.accessor<int, 2>();
        auto mapping = mapCpu.accessor<int, 2>();
        for (int path = 0; path < paths; ++path) {
            struct Candidate {
                float survival;
                int   r, j;
            };
            std::vector<Candidate> candidates;
            for (int r = 0; r < 2; ++r) {
                float survival = 1.f;
                for (int j = 0; j < gamma; ++j) {
                    survival *= ca[path][r][j];
                    candidates.push_back({survival, r, j});
                }
            }
            std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                if (a.survival != b.survival)
                    return a.survival > b.survival;
                if (a.j != b.j)
                    return a.j < b.j;
                return a.r < b.r;
            });
            int lengths[2] = {1, 1};
            for (int k = 0; k < budget; ++k)
                ++lengths[candidates[k].r];
            int offset = 0;
            for (int r = 0; r < 2; ++r) {
                ASSERT_EQ(caps[path][r], lengths[r]);
                for (int j = 0; j < lengths[r]; ++j)
                    ASSERT_EQ(mapping[path][offset++], r * (gamma + 1) + j);
            }
        }

        auto                draftCpu    = torch::empty({rows, gamma, vocab}, cpuFloat);
        auto                targetCpu   = torch::empty({rows, gamma + 1, vocab}, cpuFloat);
        auto                idsCpu      = torch::empty({rows, gamma}, cpuInt);
        auto                drawsCpu    = torch::empty({rows, gamma + 1}, cpuInt);
        auto                uniformsCpu = torch::empty({rows, gamma + 1}, cpuFloat);
        auto                activeCpu   = torch::empty({rows}, cpuInt);
        auto                dp          = draftCpu.accessor<float, 3>();
        auto                tp          = targetCpu.accessor<float, 3>();
        auto                ids         = idsCpu.accessor<int, 2>();
        auto                draws       = drawsCpu.accessor<int, 2>();
        auto                uniforms    = uniformsCpu.accessor<float, 2>();
        auto                active      = activeCpu.accessor<int, 1>();
        std::vector<double> weights(rows, 1.);
        for (int path = 0; path < paths; ++path)
            for (int r = 0; r < 2; ++r)
                for (int variant = 0; variant < variants; ++variant) {
                    int row = (path * 2 + r) * variants + variant, previous = anchors[r];
                    active[row] = caps[path][r];
                    for (int j = 0; j <= gamma; ++j) {
                        for (int x = 0; x < 2; ++x)
                            tp[row][j][x] = P[r][previous][x];
                        double split;
                        if (j < gamma) {
                            int x       = token(path, r, j);
                            ids[row][j] = x;
                            // Deliberately exercise the CDF branch, not target-ID bonus reuse.
                            draws[row][j] = 1 - x;
                            for (int y = 0; y < 2; ++y)
                                dp[row][j][y] = Q[r][previous][y];
                            split    = std::min(1., double(P[r][previous][x]) / Q[r][previous][x]);
                            previous = x;
                        } else {
                            draws[row][j] = 0;
                            split         = P[r][previous][0];
                        }
                        bool   upper = (variant >> j) & 1;
                        double lo = upper ? split : 0., hi = upper ? 1. : split;
                        uniforms[row][j] = float((lo + hi) * .5);
                        weights[row] *= hi - lo;
                        // Zero-weight intervals at split=1 cannot use u=1 (invalid RNG).
                        if (hi == lo)
                            uniforms[row][j] = .5f;
                    }
                }
        auto draft = draftCpu.to(torch::kCUDA), target = targetCpu.to(torch::kCUDA);
        auto idsD = idsCpu.to(torch::kCUDA), drawsD = drawsCpu.to(torch::kCUDA);
        auto uniformsD = uniformsCpu.to(torch::kCUDA), activeD = activeCpu.to(torch::kCUDA);
        auto output  = torch::full({rows, gamma + 1}, -1, intCuda());
        auto lengths = torch::zeros({rows}, intCuda());
        auto sample = torch::ones({rows}, boolCuda()), success = torch::ones({rows}, boolCuda());
        auto targetSuccess = torch::ones({rows, gamma + 1}, boolCuda());
        auto workspace = torch::empty({rtp_llm::rejectionValidationWorkspaceElements(rows, gamma, vocab)}, floatCuda());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ((rtp_llm::invokeRejectionSampling<float, int>(draft.data_ptr<float>(),
                                                                idsD.data_ptr<int>(),
                                                                uniformsD.data_ptr<float>(),
                                                                target.data_ptr<float>(),
                                                                drawsD.data_ptr<int>(),
                                                                1,
                                                                output.data_ptr<int>(),
                                                                lengths.data_ptr<int>(),
                                                                sample.data_ptr<bool>(),
                                                                false,
                                                                rows,
                                                                gamma,
                                                                vocab,
                                                                stream_,
                                                                true,
                                                                success.data_ptr<bool>(),
                                                                targetSuccess.data_ptr<bool>(),
                                                                activeD.data_ptr<int>(),
                                                                workspace.data_ptr<float>())),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        // MtpExecutor::capDSparkVerifyLengths: active lengths alone do not cap.
        auto                                    committed = torch::minimum(lengths, activeD).cpu();
        auto                                    outCpu = output.cpu(), okCpu = success.cpu();
        auto                                    out   = outCpu.accessor<int, 2>();
        auto                                    count = committed.accessor<int, 1>();
        auto                                    ok    = okCpu.accessor<bool, 1>();
        std::array<double, outcomes * outcomes> joint{};
        std::array<double, outcomes * outcomes> corruptedBonusJoint{};
        for (int path = 0; path < paths; ++path) {
            Distribution distribution[2]{};
            Distribution corruptedBonus{};
            for (int r = 0; r < 2; ++r)
                for (int variant = 0; variant < variants; ++variant) {
                    int row = (path * 2 + r) * variants + variant;
                    if (weights[row] == 0.)
                        continue;
                    ASSERT_TRUE(ok[row]);
                    ASSERT_GE(count[row], 1);
                    ASSERT_LE(count[row], caps[path][r]);
                    std::vector<int> prefix;
                    for (int j = 0; j < count[row]; ++j) {
                        ASSERT_GE(out[row][j], 0);
                        ASSERT_LT(out[row][j], vocab);
                        prefix.push_back(out[row][j]);
                    }
                    finish(r, prefix, weights[row], distribution[r]);
                    if (r == 0) {
                        // Negative oracle control: a legal but wrong full-accept
                        // bonus must fail the distribution check, not just bounds.
                        if (budget == 2 * gamma && path == 0 && variant == 8) {
                            ASSERT_EQ(count[row], gamma + 1);
                            ASSERT_EQ(prefix[gamma], 1);
                            prefix[gamma] = 0;
                        }
                        finish(r, prefix, weights[row], corruptedBonus);
                    }
                }
            double proposalWeight = pathWeight(path, 0, Q) * pathWeight(path, 1, Q);
            for (int a = 0; a < outcomes; ++a)
                for (int b = 0; b < outcomes; ++b) {
                    joint[a * outcomes + b] += proposalWeight * distribution[0][a] * distribution[1][b];
                    corruptedBonusJoint[a * outcomes + b] += proposalWeight * corruptedBonus[a] * distribution[1][b];
                }
        }
        double mass = 0.;
        for (int a = 0; a < outcomes; ++a)
            for (int b = 0; b < outcomes; ++b) {
                mass += joint[a * outcomes + b];
                EXPECT_NEAR(joint[a * outcomes + b], expected0[a] * expected1[b], 2.e-6) << a << "," << b;
            }
        EXPECT_NEAR(mass, 1., 2.e-6);
        if (budget == 2 * gamma) {
            double corruptedMass = 0., maxError = 0.;
            for (int a = 0; a < outcomes; ++a)
                for (int b = 0; b < outcomes; ++b) {
                    corruptedMass += corruptedBonusJoint[a * outcomes + b];
                    maxError = std::max(maxError,
                                        std::abs(corruptedBonusJoint[a * outcomes + b] - expected0[a] * expected1[b]));
                }
            EXPECT_NEAR(corruptedMass, 1., 2.e-6);
            EXPECT_GT(maxError, 2.e-6);
        }
    }
}

TEST_F(SpeculativeSamplingKernelTest, DSparkConfidenceMatchesFeatureReference) {
    constexpr int64_t batch       = 2;
    constexpr int64_t gamma       = 3;
    constexpr int64_t hidden_dim  = 8;
    constexpr int64_t markov_rank = 4;
    constexpr int64_t vocab       = 11;

    auto hidden =
        (torch::arange(batch * gamma * hidden_dim, floatCuda()).reshape({batch * gamma, hidden_dim}) / 97.0f - 0.2f)
            .to(torch::kBFloat16)
            .contiguous();
    auto markov = (torch::arange(vocab * markov_rank, floatCuda()).reshape({vocab, markov_rank}) / 53.0f - 0.3f)
                      .to(torch::kBFloat16)
                      .contiguous();
    auto weight =
        (torch::arange(hidden_dim + markov_rank, floatCuda()) / 41.0f - 0.1f).to(torch::kBFloat16).contiguous();
    auto bias    = torch::tensor({0.125f}, bfloatCuda()).contiguous();
    auto anchors = torch::tensor({2, 7}, intCuda()).contiguous();
    auto sampled = torch::tensor({{4, 5, 6}, {8, 9, 10}}, intCuda()).contiguous();
    auto output  = torch::empty({batch, gamma}, floatCuda());

    auto status =
        rtp_llm::invokeDSparkConfidence(reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
                                        anchors.data_ptr<int32_t>(),
                                        sampled.data_ptr<int32_t>(),
                                        reinterpret_cast<const __nv_bfloat16*>(markov.data_ptr<at::BFloat16>()),
                                        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                                        reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
                                        output.data_ptr<float>(),
                                        batch,
                                        gamma,
                                        hidden_dim,
                                        markov_rank,
                                        stream_);
    ASSERT_EQ(status, cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    auto previous = torch::cat({anchors.view({batch, 1}), sampled.narrow(1, 0, gamma - 1)}, 1).reshape({-1});
    auto features = torch::cat(
        {hidden.to(torch::kFloat32), markov.index_select(0, previous.to(torch::kLong)).to(torch::kFloat32)}, 1);
    auto raw      = (features.matmul(weight.to(torch::kFloat32)) + bias.to(torch::kFloat32)).to(torch::kBFloat16);
    auto expected = torch::sigmoid(raw.to(torch::kFloat32)).reshape({batch, gamma});
    EXPECT_TRUE(torch::allclose(output, expected, 1.0e-6, 1.0e-6));
}

TEST_F(SpeculativeSamplingKernelTest, DSparkVerifyPlanSelectsGlobalSurvivalBudget) {
    constexpr int64_t batch  = 3;
    constexpr int64_t gamma  = 4;
    constexpr int64_t budget = 5;
    auto              confidence =
        torch::tensor({{0.90f, 0.90f, 0.50f, 0.50f}, {0.80f, 0.50f, 0.50f, 0.50f}, {0.95f, 0.20f, 0.20f, 0.20f}},
                      floatCuda())
            .contiguous();
    auto lengths = torch::empty({batch}, intCuda());
    auto mapping = torch::empty({batch + budget}, intCuda());

    auto status = rtp_llm::invokeDSparkVerifyPlan(confidence.data_ptr<float>(),
                                                  lengths.data_ptr<int32_t>(),
                                                  mapping.data_ptr<int32_t>(),
                                                  batch,
                                                  gamma,
                                                  budget,
                                                  stream_);
    ASSERT_EQ(status, cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    // Survival scores: request0=[.9,.81,.405,.2025], request1=[.8,.4,...],
    // request2=[.95,.19,...]. Global top5 therefore allocate [3,1,1].
    EXPECT_TRUE(torch::equal(lengths.cpu(), torch::tensor({4, 2, 2}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(mapping.cpu(), torch::tensor({0, 1, 2, 3, 5, 6, 10, 11}, torch::kInt32)));
}

TEST_F(SpeculativeSamplingKernelTest, DSparkVerifyPlanCoversZeroAndFullBudget) {
    constexpr int64_t batch      = 2;
    constexpr int64_t gamma      = 3;
    auto              confidence = torch::full({batch, gamma}, 0.5f, floatCuda());
    for (const int64_t budget : {int64_t{0}, batch * gamma}) {
        auto lengths = torch::empty({batch}, intCuda());
        auto mapping = torch::empty({batch + budget}, intCuda());
        ASSERT_EQ(rtp_llm::invokeDSparkVerifyPlan(confidence.data_ptr<float>(),
                                                  lengths.data_ptr<int32_t>(),
                                                  mapping.data_ptr<int32_t>(),
                                                  batch,
                                                  gamma,
                                                  budget,
                                                  stream_),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        auto expected_lengths = torch::full({batch}, budget == 0 ? 1 : gamma + 1, torch::kInt32);
        EXPECT_TRUE(torch::equal(lengths.cpu(), expected_lengths));
        auto expected_mapping =
            budget == 0 ? torch::tensor({0, 4}, torch::kInt32) : torch::arange(batch * (gamma + 1), torch::kInt32);
        EXPECT_TRUE(torch::equal(mapping.cpu(), expected_mapping));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Additional DSpARK verify-planner coverage
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(SpeculativeSamplingKernelTest, DSparkVerifyPlanBreaksTiesByPrefixPosition) {
    constexpr int64_t batch  = 2;
    constexpr int64_t gamma  = 3;
    constexpr int64_t budget = 3;
    // Every cumulative survival value is exactly one. The stable reference
    // order selects position 0 for both requests before position 1, rather
    // than exhausting request 0 first.
    auto confidence = torch::ones({batch, gamma}, floatCuda());
    auto lengths    = torch::empty({batch}, intCuda());
    auto mapping    = torch::empty({batch + budget}, intCuda());

    ASSERT_EQ(rtp_llm::invokeDSparkVerifyPlan(confidence.data_ptr<float>(),
                                              lengths.data_ptr<int32_t>(),
                                              mapping.data_ptr<int32_t>(),
                                              batch,
                                              gamma,
                                              budget,
                                              stream_),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    EXPECT_TRUE(torch::equal(lengths.cpu(), torch::tensor({3, 2}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(mapping.cpu(), torch::tensor({0, 1, 2, 4, 5}, torch::kInt32)));
}

// ─────────────────────────────────────────────────────────────────────────────
// invokeRejectionSampling tests
// ───────────────────────────────────────────────────────────────────────────

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_GreedyAllAccept) {
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
                                                               false,
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

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_SampledPointMassMatchesTargetDistribution) {
    const int batch_size    = 1000;
    const int num_spec      = 1;
    const int vocab_size    = 16;
    const int target_stride = 1;

    auto draft_probs  = torch::zeros({batch_size, num_spec, vocab_size}, floatCuda());
    auto target_probs = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());
    draft_probs.index_put_({torch::indexing::Slice(), 0, 5}, 1.0f);
    target_probs.index_put_({torch::indexing::Slice(), 0, 5}, 0.25f);
    target_probs.index_put_({torch::indexing::Slice(), 0, 7}, 0.75f);
    target_probs.index_put_({torch::indexing::Slice(), 1, 7}, 1.0f);

    auto draft_token_ids  = torch::full({batch_size, num_spec}, 5, intCuda());
    auto target_token_ids = torch::full({batch_size, num_spec + 1, target_stride}, 7, intCuda());
    auto uniform_samples  = torch::zeros({batch_size, num_spec + 1}, floatCuda());
    uniform_samples.index_put_({torch::indexing::Slice(), 0},
                               (torch::arange(batch_size, floatCuda()) + 0.5f) / static_cast<float>(batch_size));
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
                                                               false,
                                                               batch_size,
                                                               num_spec,
                                                               vocab_size,
                                                               stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    auto first_tokens = output_token_ids.index({torch::indexing::Slice(), 0}).to(torch::kCPU);
    auto accepted_num = output_accepted_num.to(torch::kCPU);
    EXPECT_EQ(first_tokens.eq(5).sum().item<int64_t>(), 250);
    EXPECT_EQ(first_tokens.eq(7).sum().item<int64_t>(), 750);
    EXPECT_EQ(accepted_num.eq(2).sum().item<int64_t>(), 250);
    EXPECT_EQ(accepted_num.eq(1).sum().item<int64_t>(), 750);
    auto second_tokens = output_token_ids.index({torch::indexing::Slice(), 1}).to(torch::kCPU);
    EXPECT_EQ(second_tokens.eq(7).sum().item<int64_t>(), 250);
    EXPECT_EQ(second_tokens.eq(-1).sum().item<int64_t>(), 750);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_ExactMatchIgnoresRequestDoSampleAndUniform) {
    const int batch_size    = 2;
    const int num_spec      = 2;
    const int vocab_size    = 16;
    const int target_stride = 1;

    auto draft_probs         = torch::zeros({batch_size, num_spec, vocab_size}, floatCuda());
    auto target_probs        = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());
    auto draft_token_ids     = torch::tensor({{5, 3}, {5, 3}}, intCuda());
    auto target_token_ids    = torch::tensor({{{5}, {7}, {9}}, {{5}, {7}, {9}}}, intCuda());
    auto uniform_samples     = torch::tensor({{0.0f, 0.0f, 0.0f}, {0.99f, 0.99f, 0.99f}}, floatCuda());
    auto output_token_ids    = torch::full({batch_size, num_spec + 1}, -1, intCuda());
    auto output_accepted_num = torch::zeros({batch_size}, intCuda());
    auto do_sample           = torch::tensor({false, true}, boolCuda());

    auto status = rtp_llm::invokeRejectionSampling<float, int>(draft_probs.data_ptr<float>(),
                                                               draft_token_ids.data_ptr<int>(),
                                                               uniform_samples.data_ptr<float>(),
                                                               target_probs.data_ptr<float>(),
                                                               target_token_ids.data_ptr<int>(),
                                                               target_stride,
                                                               output_token_ids.data_ptr<int>(),
                                                               output_accepted_num.data_ptr<int>(),
                                                               do_sample.data_ptr<bool>(),
                                                               true,
                                                               batch_size,
                                                               num_spec,
                                                               vocab_size,
                                                               stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    EXPECT_TRUE(output_token_ids[0].eq(output_token_ids[1]).all().item<bool>());
    EXPECT_TRUE(output_accepted_num[0].eq(output_accepted_num[1]).item<bool>());
    EXPECT_EQ(output_accepted_num[0].item<int>(), 2);
    EXPECT_EQ(output_token_ids[0][0].item<int>(), 5);
    EXPECT_EQ(output_token_ids[0][1].item<int>(), 7);
    EXPECT_EQ(output_token_ids[0][2].item<int>(), -1);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_GreedyImmediateMismatchUsesTargetToken) {
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
                                                               false,
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
    // Greedy verification emits the target top-1 token directly.
    EXPECT_EQ(out_ids_h[0][0].item<int>(), 7);
    // Remaining positions padded with -1
    EXPECT_EQ(out_ids_h[0][1].item<int>(), -1);
    EXPECT_EQ(out_ids_h[0][2].item<int>(), -1);
    EXPECT_EQ(out_ids_h[0][3].item<int>(), -1);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_GreedyRejectUsesTargetTop1NotResidualSample) {
    const int batch_size    = 1;
    const int num_spec      = 3;
    const int vocab_size    = 16;
    const int target_stride = 1;

    auto draft_probs  = torch::zeros({batch_size, num_spec, vocab_size}, floatCuda());
    auto target_probs = torch::zeros({batch_size, num_spec + 1, vocab_size}, floatCuda());

    // Position 0: draft picks token 3, target top1 is token 7. If the greedy
    // mismatch path incorrectly samples relu(target-draft), u=0.9 would pick
    // token 8 from the residual distribution instead of target top1 token 7.
    draft_probs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 3}, 1.0f);
    target_probs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 7}, 0.6f);
    target_probs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 8}, 0.4f);

    auto draft_token_ids  = torch::full({batch_size, num_spec}, 3, intCuda());
    auto target_token_ids = torch::full({batch_size, num_spec + 1, target_stride}, 7, intCuda());

    auto uniform_samples     = torch::full({batch_size, num_spec + 1}, 0.9f, floatCuda());
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
                                                               false,
                                                               batch_size,
                                                               num_spec,
                                                               vocab_size,
                                                               stream_);
    ASSERT_EQ(status, cudaSuccess);
    cudaStreamSynchronize(stream_);

    auto acc_num_h = output_accepted_num.to(torch::kCPU);
    auto out_ids_h = output_token_ids.to(torch::kCPU);

    EXPECT_EQ(acc_num_h[0].item<int>(), 1);
    EXPECT_EQ(out_ids_h[0][0].item<int>(), 7);
    EXPECT_EQ(out_ids_h[0][1].item<int>(), -1);
    EXPECT_EQ(out_ids_h[0][2].item<int>(), -1);
    EXPECT_EQ(out_ids_h[0][3].item<int>(), -1);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_LegacyPartialAccept) {
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
                                                               false,
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
    // Greedy verification emits the target top-1 token directly.
    EXPECT_EQ(out_ids_h[0][2].item<int>(), 7);
    EXPECT_EQ(out_ids_h[0][3].item<int>(), -1);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_SampledSameTokenStillTestsProbabilityRatio) {
    // Nondegenerate P(A,B)=(.8,.2), Q(A,B)=(.2,.8). At u=.5,
    // .5*.8 >= .2: reject A even if the independent target draw is A.
    // The normalized positive residual is point mass B, not target draw A.
    constexpr int batch  = 5;
    constexpr int vocab  = 16;
    constexpr int stride = 2;
    for (int steps : {1, 4, 7}) {
        SCOPED_TRACE(steps);
        auto draft_probs = torch::zeros({batch, steps, vocab}, floatCuda());
        draft_probs.select(2, 0).fill_(0.8f);
        draft_probs.select(2, 1).fill_(0.2f);
        auto target_probs = torch::zeros({batch, steps + 1, vocab}, floatCuda());
        target_probs.narrow(1, 0, steps).select(2, 0).fill_(0.2f);
        target_probs.narrow(1, 0, steps).select(2, 1).fill_(0.8f);
        target_probs.select(1, steps).select(1, 7).fill_(1.0f);
        auto drafts  = torch::zeros({batch, steps}, intCuda());
        auto targets = torch::full({batch, steps + 1, stride}, 9, intCuda());
        targets.select(2, stride - 1).fill_(0);
        targets.select(1, steps).select(1, stride - 1).fill_(7);
        // Row4 exercises full acceptance when target draws differ from draft;
        // row3 exercises the all-same fast path. Both must emit bonus7.
        targets[4].narrow(0, 0, steps).select(1, stride - 1).fill_(1);
        auto uniforms = torch::full({batch, steps + 1}, 0.1f, floatCuda());
        uniforms[0][0].fill_(0.5f);
        uniforms[1].fill_(0.5f);  // Greedy row ignores the ratio.
        uniforms[2][steps / 2].fill_(0.5f);
        auto modes   = torch::tensor({true, false, true, true, true}, boolCuda());
        auto output  = torch::full({batch, steps + 1}, -1, intCuda());
        auto lengths = torch::zeros({batch}, intCuda());
        // Tensor initialization uses Torch's stream; the fixture launches on
        // its own CUDA stream. Establish setup completion explicitly.
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ((rtp_llm::invokeRejectionSampling<float, int>(draft_probs.data_ptr<float>(),
                                                                drafts.data_ptr<int>(),
                                                                uniforms.data_ptr<float>(),
                                                                target_probs.data_ptr<float>(),
                                                                targets.data_ptr<int>(),
                                                                stride,
                                                                output.data_ptr<int>(),
                                                                lengths.data_ptr<int>(),
                                                                modes.data_ptr<bool>(),
                                                                false,
                                                                batch,
                                                                steps,
                                                                vocab,
                                                                stream_,
                                                                true)),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        const auto out = output.cpu();
        const auto len = lengths.cpu();
        for (int row = 0; row < batch; ++row) {
            const int accepted_drafts = row == 0 ? 0 : row == 2 ? steps / 2 : steps;
            EXPECT_EQ(len[row].item<int>(), accepted_drafts + 1) << "row=" << row;
            for (int col = 0; col < accepted_drafts; ++col) {
                EXPECT_EQ(out[row][col].item<int>(), 0) << "row=" << row << " col=" << col;
            }
            EXPECT_EQ(out[row][accepted_drafts].item<int>(), accepted_drafts == steps ? 7 : 1) << "row=" << row;
            for (int col = accepted_drafts + 1; col <= steps; ++col) {
                EXPECT_EQ(out[row][col].item<int>(), -1) << "row=" << row << " col=" << col;
            }
        }
        // The default ABI preserves historical MTP same-token acceptance.
        ASSERT_EQ((rtp_llm::invokeRejectionSampling<float, int>(draft_probs.data_ptr<float>(),
                                                                drafts.data_ptr<int>(),
                                                                uniforms.data_ptr<float>(),
                                                                target_probs.data_ptr<float>(),
                                                                targets.data_ptr<int>(),
                                                                stride,
                                                                output.data_ptr<int>(),
                                                                lengths.data_ptr<int>(),
                                                                modes.data_ptr<bool>(),
                                                                false,
                                                                batch,
                                                                steps,
                                                                vocab,
                                                                stream_)),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        const auto legacy_out     = output.cpu();
        const auto legacy_lengths = lengths.cpu();
        for (int row = 0; row < batch; ++row) {
            EXPECT_EQ(legacy_lengths[row].item<int>(), steps + 1);
            EXPECT_EQ(legacy_out[row][steps].item<int>(), 7);
        }
    }
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_DSparkInvalidDistributionFailsClosed) {
    constexpr int batch = 14;
    for (const auto& geometry : std::vector<std::pair<int, int>>{{3, 16}, {7, 200064}}) {
        const int steps = geometry.first, vocab = geometry.second;
        auto      draft  = torch::zeros({batch, steps, vocab}, floatCuda());
        auto      target = torch::zeros({batch, steps + 1, vocab}, floatCuda());
        draft.select(2, 0).fill_(1.0f);
        target.select(2, 0).fill_(1.0f);
        auto ids            = torch::zeros({batch, steps}, intCuda());
        auto draws          = torch::zeros({batch * (steps + 1), 1}, intCuda());
        auto uniforms       = torch::full({batch, steps + 1}, 0.5f, floatCuda());
        auto output         = torch::zeros({batch, steps + 1}, intCuda());
        auto lengths        = torch::zeros({batch}, intCuda());
        auto success        = torch::ones({batch}, boolCuda());
        auto target_success = torch::ones({batch, steps + 1}, boolCuda());
        auto sample         = torch::ones({batch}, boolCuda());
        auto caps           = torch::full({batch}, steps + 1, intCuda());
        auto workspace      = torch::full({rtp_llm::rejectionValidationWorkspaceElements(batch, steps, vocab)},
                                     std::numeric_limits<float>::quiet_NaN(),
                                     floatCuda());
        // Poison a non-selected entry in the final vocab tile: rejection itself
        // does not inspect it, so tiled validation must detect the invalid row.
        target[0][0][vocab - 1].fill_(std::numeric_limits<float>::quiet_NaN());
        target[1][steps].zero_();
        target[2][0][vocab - 1].fill_(-0.25f);
        draft[3][0][vocab - 1].fill_(std::numeric_limits<float>::infinity());
        draft[4][0].zero_();
        target_success[5][0].fill_(false);
        uniforms[6][0].fill_(1.0f);  // Identical P,Q; rejection has zero residual mass.
        // Row7 is a healthy full-accept control.
        // Row8 rejects immediately; poisoned future rows must remain unread.
        draft[8][0][0].fill_(0.8f);
        draft[8][0][1].fill_(0.2f);
        target[8][0][0].fill_(0.2f);
        target[8][0][1].fill_(0.8f);
        target[8][1].fill_(std::numeric_limits<float>::quiet_NaN());
        target_success[8][1].fill_(false);
        // Row9 reaches only an adaptive prefix; invalid dense future rows are not live.
        caps[9].fill_(1);
        target[9][1].fill_(std::numeric_limits<float>::quiet_NaN());
        target_success[9][1].fill_(false);
        // Row10 is greedy with invalid target probabilities, despite a valid argmax ID.
        sample[10].fill_(false);
        target[10][0].fill_(std::numeric_limits<float>::quiet_NaN());
        // Row11 has finite positive residual mass but no CDF hit (invalid draw).
        draft[11][0][0].fill_(0.8f);
        draft[11][0][1].fill_(0.2f);
        target[11][0][0].fill_(0.2f);
        target[11][0][1].fill_(0.8f);
        uniforms[11][1].fill_(std::numeric_limits<float>::quiet_NaN());
        draft[12][0].fill_(std::numeric_limits<float>::quiet_NaN());
        // Row13 is a healthy middle reject control.
        draft[13][1][0].fill_(0.8f);
        draft[13][1][1].fill_(0.2f);
        target[13][1][0].fill_(0.2f);
        target[13][1][1].fill_(0.8f);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ((rtp_llm::invokeRejectionSampling<float, int>(draft.data_ptr<float>(),
                                                                ids.data_ptr<int>(),
                                                                uniforms.data_ptr<float>(),
                                                                target.data_ptr<float>(),
                                                                draws.data_ptr<int>(),
                                                                1,
                                                                output.data_ptr<int>(),
                                                                lengths.data_ptr<int>(),
                                                                sample.data_ptr<bool>(),
                                                                false,
                                                                batch,
                                                                steps,
                                                                vocab,
                                                                stream_,
                                                                true,
                                                                success.data_ptr<bool>(),
                                                                target_success.data_ptr<bool>(),
                                                                caps.data_ptr<int>(),
                                                                workspace.data_ptr<float>())),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        auto ok     = success.cpu();
        auto tokens = output.cpu();
        auto count  = lengths.cpu();
        for (int row = 0; row < batch; ++row) {
            const bool expected_success = (row >= 7 && row <= 9) || row == 13;
            EXPECT_EQ(ok[row].item<bool>(), expected_success) << row;
            if (!expected_success) {
                EXPECT_EQ(count[row].item<int>(), 1) << row;
                EXPECT_TRUE(tokens[row].eq(0).all().item<bool>()) << row;
            }
        }
        EXPECT_EQ(count[7].item<int>(), steps + 1);
        EXPECT_EQ(count[8].item<int>(), 1);
        EXPECT_EQ(tokens[8][0].item<int>(), 1);
        EXPECT_EQ(count[13].item<int>(), 2);
        EXPECT_EQ(tokens[13][0].item<int>(), 0);
        EXPECT_EQ(tokens[13][1].item<int>(), 1);
        // The inactive suffix has been overwritten with neutral records even
        // though this workspace began as NaN poison.
        const auto tiles = (vocab + rtp_llm::kRejectionValidationTileSize - 1) / rtp_llm::kRejectionValidationTileSize;
        EXPECT_TRUE(workspace.reshape({batch, steps + 1, tiles, 4})[8].narrow(0, 1, steps).eq(0).all().item<bool>());
    }
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_DSparkDeterministicValidatesUsedRows) {
    constexpr int batch = 2, steps = 2, vocab = 16;
    auto          draft  = torch::zeros({batch, steps, vocab}, floatCuda());
    auto          target = torch::zeros({batch, steps + 1, vocab}, floatCuda());
    draft.select(2, 0).fill_(1);
    target.select(2, 0).fill_(1);
    target[0][0].fill_(std::numeric_limits<float>::infinity());
    auto ids       = torch::zeros({batch, steps}, intCuda());
    auto draws     = torch::zeros({batch * (steps + 1), 1}, intCuda());
    auto uniform   = torch::zeros({batch, steps + 1}, floatCuda());
    auto output    = torch::zeros({batch, steps + 1}, intCuda());
    auto lengths   = torch::zeros({batch}, intCuda());
    auto success   = torch::ones({batch}, boolCuda());
    auto sample    = torch::ones({batch}, boolCuda());
    auto workspace = torch::empty({rtp_llm::rejectionValidationWorkspaceElements(batch, steps, vocab)}, floatCuda());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ((rtp_llm::invokeRejectionSampling<float, int>(draft.data_ptr<float>(),
                                                            ids.data_ptr<int>(),
                                                            uniform.data_ptr<float>(),
                                                            target.data_ptr<float>(),
                                                            draws.data_ptr<int>(),
                                                            1,
                                                            output.data_ptr<int>(),
                                                            lengths.data_ptr<int>(),
                                                            sample.data_ptr<bool>(),
                                                            true,
                                                            batch,
                                                            steps,
                                                            vocab,
                                                            stream_,
                                                            true,
                                                            success.data_ptr<bool>(),
                                                            nullptr,
                                                            nullptr,
                                                            workspace.data_ptr<float>())),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    EXPECT_FALSE(success.cpu()[0].item<bool>());
    EXPECT_TRUE(success.cpu()[1].item<bool>());
    EXPECT_EQ(lengths.cpu()[0].item<int>(), 1);
    EXPECT_EQ(lengths.cpu()[1].item<int>(), steps + 1);
}

TEST_F(SpeculativeSamplingKernelTest, RejectionSampling_BatchSizeZero) {
    auto status = rtp_llm::invokeRejectionSampling<float, int>(
        nullptr, nullptr, nullptr, nullptr, nullptr, 1, nullptr, nullptr, nullptr, false, 0, 3, 16, stream_);
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
