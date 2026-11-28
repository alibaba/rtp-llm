#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/cpp/normal_engine/speculative/DSparkSampleSeed.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

#include <limits>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <gtest/gtest.h>
#include <limits>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <torch/all.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
namespace speculative {
namespace {

TEST(FastTopKSamplerTest, TopKOneReturnsArgmaxWithPointMassProbability) {
    FastTopKSampler sampler;
    auto            logits = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}});
    auto            out    = sampler.forward(logits, 1);

    ASSERT_EQ(out.token_ids.dim(), 2);
    ASSERT_EQ(out.token_ids.size(0), 1);
    ASSERT_EQ(out.token_ids.size(1), 1);
    EXPECT_EQ(out.token_ids.scalar_type(), torch::kInt32);
    EXPECT_EQ(out.token_ids[0][0].item<int64_t>(), 2);
    EXPECT_TRUE(torch::equal(out.all_probs, torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})));
}

TEST(FastTopKSamplerTest, TopKGreaterThanOneReturnsTopKIndices) {
    FastTopKSampler sampler;
    auto            logits = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}});
    auto            out    = sampler.forward(logits, 2);

    ASSERT_EQ(out.token_ids.dim(), 2);
    ASSERT_EQ(out.token_ids.size(0), 1);
    ASSERT_EQ(out.token_ids.size(1), 2);
    EXPECT_EQ(out.token_ids.scalar_type(), torch::kInt32);
    // softmax preserves ordering: indices 2 (5.0) then 3 (3.0).
    EXPECT_EQ(out.token_ids[0][0].item<int64_t>(), 2);
    EXPECT_EQ(out.token_ids[0][1].item<int64_t>(), 3);
}

TEST(ExecRejectionSamplingTest, FixedCandidatesUseTargetAcceptanceAndResidualDistribution) {
    constexpr int64_t batch  = 7;
    constexpr int64_t width  = 4;
    const auto        i32    = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    const auto        f32    = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
    auto              drafts = torch::ones({batch, width - 1}, i32);
    auto              targets =
        torch::tensor({1, 1, 2, 3, 2, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3}, i32)
            .reshape({batch * width, 1});
    auto probs = torch::zeros({batch, width, 4}, f32);
    probs.select(2, 1).fill_(1);
    auto uniforms = torch::full({batch, width}, 0.1, f32);
    /** Q(candidate)=1. P(candidate)=.25, so u=.1 accepts and u=.5 rejects. */
    for (int64_t row : {3, 4, 5}) {
        probs[row][0].copy_(torch::tensor({0.f, .25f, .50f, .25f}, f32));
    }
    uniforms[4][0] = .5;
    uniforms[5][0] = .5;
    uniforms[4][1] = .5;                                          // Residual CDF: token 2 has mass 2/3.
    uniforms[5][1] = .9;                                          // Token 3 has the remaining 1/3.
    probs[6][0].copy_(torch::tensor({0.f, 0.f, 1.f, 0.f}, f32));  // Processor-masked candidate.
    auto stochastic = torch::tensor({false, false, false, true, true, true, true},
                                    torch::TensorOptions(torch::kBool).device(torch::kCUDA));
    auto output     = torch::full({batch, width}, -7, i32);
    auto lengths    = torch::zeros({batch}, i32);
    execRejectionSampling({{}, drafts, uniforms, probs, targets, output, lengths, stochastic, true});
    EXPECT_TRUE(torch::equal(lengths.cpu(), torch::tensor({3, 1, 4, 4, 1, 1, 1}, torch::kInt32)));
    const std::vector<std::vector<int32_t>> expected{{1, 1, 2}, {2}, {1, 1, 1, 3}, {1, 1, 1, 3}, {2}, {3}, {2}};
    const auto                              cpu = output.cpu();
    for (int64_t row = 0; row < batch; ++row) {
        EXPECT_TRUE(
            torch::equal(cpu[row].narrow(0, 0, expected[row].size()), torch::tensor(expected[row], torch::kInt32)))
            << "row=" << row;
    }

    /** The explicit one-hot must match the implicit path for both acceptance and residual sampling. */
    auto dense_q = torch::zeros({batch, width - 1, 4}, f32).scatter_(-1, drafts.to(torch::kLong).unsqueeze(-1), 1.f);
    auto dense_output  = torch::full_like(output, -7);
    auto dense_lengths = torch::zeros_like(lengths);
    execRejectionSampling({dense_q, drafts, uniforms, probs, targets, dense_output, dense_lengths, stochastic, false});
    EXPECT_TRUE(torch::equal(dense_output, output));
    EXPECT_TRUE(torch::equal(dense_lengths, lengths));
}

class InspectableSpeculativeSampler: public SpeculativeSampler {
public:
    using SpeculativeSampler::SpeculativeSampler;

    const torch::Tensor& mappedDraftProbs() const {
        return draft_probs_padding_buffer_;
    }
};

SpeculativeSamplingParams seededSamplingParams(int64_t batch) {
    SpeculativeSamplingParams params;
    params.do_sample    = torch::ones({batch}, torch::kBool);
    params.force_accept = torch::zeros({batch}, torch::kBool);
    for (int64_t row = 0; row < batch; ++row) {
        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
        generator.set_current_seed(100 + row);
        params.generators.push_back(generator);
    }
    return params;
}

TEST(DraftProbabilitiesTest, MapsDenseQAndReusesBufferWithoutCopyingSameVocabBatch) {
    constexpr int64_t batch     = 4;
    constexpr int64_t steps     = 2;
    const auto        f32       = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto        i64       = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    const auto        compact_q = torch::tensor(
        {{{.1f, .9f}, {.2f, .8f}}, {{.3f, .7f}, {.4f, .6f}}, {{.5f, .5f}, {.6f, .4f}}, {{.7f, .3f}, {.8f, .2f}}}, f32);
    auto dense_q = torch::zeros({batch, steps, 4}, f32);
    dense_q.select(2, 1).copy_(compact_q.select(2, 0));
    dense_q.select(2, 3).copy_(compact_q.select(2, 1));
    SamplerOutput target;
    target.all_probs = torch::tensor({.2f, .3f, .1f, .4f}, f32).reshape({1, 1, 4}).repeat({batch, steps + 1, 1});
    target.token_ids = target.all_probs.argmax(-1).reshape({batch * (steps + 1), 1}).to(torch::kInt32);

    for (bool reduced_vocab : {false, true}) {
        SCOPED_TRACE(reduced_vocab);
        InspectableSpeculativeSampler sampler(reduced_vocab ? torch::tensor({1, 3}, i64) : torch::Tensor(), steps);
        SpeculativeSampler            reference({}, steps);
        float*                        mapped_storage = nullptr;
        for (int iteration = 0; iteration < 2; ++iteration) {
            SCOPED_TRACE(iteration);
            auto          params = seededSamplingParams(batch);
            SamplerOutput draft;
            draft.token_ids              = torch::tensor({1, 3}, torch::kInt32).reshape({1, steps}).repeat({batch, 1});
            const auto source_q          = reduced_vocab ? compact_q : dense_q;
            draft.all_probs              = iteration == 0 ? source_q.clone() : source_q.flip({0});
            const auto    expected_q     = iteration == 0 ? dense_q : dense_q.flip({0});
            const auto    input_snapshot = draft.all_probs.clone();
            const auto    input_storage  = draft.all_probs.data_ptr<float>();
            SamplerOutput expected_draft;
            expected_draft.token_ids = draft.token_ids;
            expected_draft.all_probs = expected_q;
            const auto actual        = sampler.forward(params, draft, target);
            const auto expected      = reference.forward(seededSamplingParams(batch), expected_draft, target);
            EXPECT_TRUE(torch::equal(actual.accept_tokens, expected.accept_tokens));
            EXPECT_TRUE(torch::equal(actual.accept_len, expected.accept_len));
            if (reduced_vocab) {
                EXPECT_TRUE(torch::equal(draft.all_probs, input_snapshot));
                EXPECT_TRUE(torch::equal(sampler.mappedDraftProbs(), expected_q));
                if (mapped_storage) {
                    EXPECT_EQ(sampler.mappedDraftProbs().data_ptr<float>(), mapped_storage);
                }
                mapped_storage = sampler.mappedDraftProbs().data_ptr<float>();
            } else {
                EXPECT_EQ(draft.all_probs.data_ptr<float>(), input_storage);
                EXPECT_TRUE(torch::equal(draft.all_probs, expected_q));
                EXPECT_FALSE(sampler.mappedDraftProbs().defined());
            }
        }
    }
}

TEST(DraftProbabilitiesTest, TargetVocabularyQIsUsedWithoutApplyingDraftMap) {
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    for (bool reduced_vocab : {true, false}) {
        SCOPED_TRACE(reduced_vocab);
        const auto    map = reduced_vocab ? torch::tensor({1, 3}, i64) : torch::tensor({2, 0, 3, 1}, i64);
        SamplerOutput draft;
        draft.token_ids = torch::tensor({reduced_vocab ? 1 : 2, 3, 0, 0}, torch::kInt32).reshape({2, 2});
        /** PP prepares both mapped model Q and handoff delta(0) in the target vocabulary. */
        draft.all_probs = torch::zeros({2, 2, 4}, f32);
        draft.all_probs[0].copy_(reduced_vocab ? torch::tensor({{0.f, .25f, 0.f, .75f}, {0.f, .5f, 0.f, .5f}}, f32) :
                                                 torch::tensor({{.2f, .4f, .1f, .3f}, {.3f, .1f, .4f, .2f}}, f32));
        draft.all_probs[1].select(1, 0).fill_(1);
        const auto    input_snapshot = draft.all_probs.clone();
        const auto    input_storage  = draft.all_probs.data_ptr<float>();
        SamplerOutput target;
        target.all_probs = torch::tensor({.1f, .2f, .3f, .4f}, f32).reshape({1, 1, 4}).repeat({2, 3, 1});
        target.token_ids = target.all_probs.argmax(-1).reshape({6, 1}).to(torch::kInt32);

        /** Row 0 accepts both candidates; row 1 rejects token 0 and samples token 2 from the residual CDF. */
        auto uniforms  = torch::full({2, 3}, .5f, f32);
        uniforms[1][0] = .9f;
        auto tokens    = torch::full({2, 3}, -7, i32);
        auto lengths   = torch::zeros({2}, i32);
        execRejectionSampling({draft.all_probs,
                               draft.token_ids.to(torch::kCUDA),
                               uniforms,
                               target.all_probs,
                               target.token_ids,
                               tokens,
                               lengths,
                               torch::ones({2}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA)),
                               false});
        EXPECT_TRUE(torch::equal(lengths.cpu(), torch::tensor({3, 1}, torch::kInt32)));
        EXPECT_TRUE(torch::equal(
            tokens.cpu(), torch::tensor({reduced_vocab ? 1 : 2, 3, 3, 2, -1, -1}, torch::kInt32).reshape({2, 3})));

        InspectableSpeculativeSampler sampler(map, 2);
        SpeculativeSampler            reference({}, 2);
        const auto                    actual   = sampler.forward(seededSamplingParams(2), draft, target);
        const auto                    expected = reference.forward(seededSamplingParams(2), draft, target);
        EXPECT_TRUE(torch::equal(actual.accept_tokens, expected.accept_tokens));
        EXPECT_TRUE(torch::equal(actual.accept_len, expected.accept_len));
        EXPECT_FALSE(sampler.mappedDraftProbs().defined());
        EXPECT_EQ(draft.all_probs.data_ptr<float>(), input_storage);
        EXPECT_TRUE(torch::equal(draft.all_probs, input_snapshot));
    }
}

TEST(DraftProbabilitiesTest, PointMassCandidatesMatchExplicitQWithoutVocabularyMapping) {
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    for (bool reduced_vocab : {false, true}) {
        SCOPED_TRACE(reduced_vocab);
        InspectableSpeculativeSampler sampler(reduced_vocab ? torch::tensor({1, 3}, i64) : torch::Tensor(), 2);
        SpeculativeSampler            reference({}, 2);
        SamplerOutput                 draft;
        /** MTP candidates are target IDs; token 0 need not be present in the draft vocabulary. */
        draft.token_ids                = torch::zeros({2, 2}, torch::kInt32);
        draft.token_ids_are_point_mass = true;
        SamplerOutput explicit_draft;
        explicit_draft.token_ids = draft.token_ids;
        explicit_draft.all_probs = torch::zeros({2, 2, 4}, f32);
        explicit_draft.all_probs.select(2, 0).fill_(1);
        SamplerOutput target;
        target.all_probs = torch::tensor({.2f, .3f, .1f, .4f}, f32).reshape({1, 1, 4}).repeat({2, 3, 1});
        target.token_ids = target.all_probs.argmax(-1).reshape({6, 1}).to(torch::kInt32);

        const auto actual   = sampler.forward(seededSamplingParams(2), draft, target);
        const auto expected = reference.forward(seededSamplingParams(2), explicit_draft, target);
        EXPECT_TRUE(torch::equal(actual.accept_tokens, expected.accept_tokens));
        EXPECT_TRUE(torch::equal(actual.accept_len, expected.accept_len));
        EXPECT_FALSE(draft.all_probs.defined());
        EXPECT_FALSE(sampler.mappedDraftProbs().defined());
    }
}

TEST(SpeculativeSamplerTest, ExplicitForceAcceptKeepsItsOverride) {
    SpeculativeSampler sampler({}, 3);
    SamplerOutput      draft;
    draft.token_ids                = torch::ones({1, 3}, torch::kInt32);
    draft.token_ids_are_point_mass = true;
    SamplerOutput target;
    target.token_ids = torch::full({4, 1}, 2, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    target.all_probs = torch::zeros({1, 4, 4}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    target.all_probs.select(2, 2).fill_(1);
    SpeculativeSamplingParams params;
    params.do_sample    = torch::zeros({1}, torch::kBool);
    params.force_accept = torch::ones({1}, torch::kBool);
    const auto result   = sampler.forward(params, draft, target);
    EXPECT_EQ(result.accept_len.item<int32_t>(), 4);
    EXPECT_TRUE(torch::equal(result.accept_tokens.cpu(), torch::tensor({{1, 1, 1, 2}}, torch::kInt32)));
}

TEST(DFlashSamplerTest, GreedyUsesExactPointMass) {
    GenerateConfig config;
    config.do_sample = false;
    auto logits      = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}, {5.0f, 1.0f, 0.0f, -1.0f}});
    auto q           = dflashDraftProbabilities(logits, config);
    EXPECT_TRUE(torch::equal(q, torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}})));
    config.do_sample   = true;
    config.temperature = 0.0f;
    EXPECT_TRUE(torch::equal(q, dflashDraftProbabilities(logits, config)));
}

TEST(DFlashSamplerTest, TemperatureTopKAndNucleusMatchReferenceDistribution) {
    // Before filtering the temperature-scaled scores represent [0.4,0.3,0.2,0.1].
    auto           logits = torch::log(torch::tensor({{0.4f, 0.3f, 0.2f, 0.1f}})) * 2.0;
    GenerateConfig config;
    config.temperature = 2.0f;
    config.top_k       = 3;
    config.top_p       = 0.6f;
    // Top-k normalization gives [4/9,3/9,2/9]; nucleus retains its crossing token.
    auto q = dflashDraftProbabilities(logits, config);
    EXPECT_TRUE(torch::allclose(q, torch::tensor({{4.0f / 7, 3.0f / 7, 0.0f, 0.0f}}), 1e-5, 1e-6));
    config.top_k = 0;
    config.top_p = 1.0f;
    EXPECT_TRUE(torch::allclose(
        dflashDraftProbabilities(logits, config), torch::tensor({{0.4f, 0.3f, 0.2f, 0.1f}}), 1e-5, 1e-6));
}

TEST(DFlashSamplerTest, TinyNucleusAlwaysRetainsLargestToken) {
    GenerateConfig config;
    config.top_p = 0.001f;
    EXPECT_TRUE(torch::equal(dflashDraftProbabilities(torch::tensor({{1.0f, 2.0f, 0.0f}}), config),
                             torch::tensor({{0.0f, 1.0f, 0.0f}})));
    // Admission passes top_p = 0 through unchanged (OpenAI passthrough), so
    // it normalizes to "no nucleus cut" (top_p = 1.0) instead of throwing
    // inside the engine loop thread.
    config.top_p = 0.0f;
    auto plain = torch::tensor({{1.0f, 2.0f}});
    EXPECT_TRUE(torch::allclose(dflashDraftProbabilities(plain, config), torch::softmax(plain, -1), 1e-6, 1e-6));
}

TEST(DFlashSamplerTest, NormalizesAdmissionTolerantSamplingParameters) {
    // Admission lets unusable sampling parameters through: the repo's own
    // smoke load sends {temperature: 0, top_p: 0}, the OpenAI endpoint keeps
    // top_p = 0 / > 1 unchanged and is_positive_number accepts inf
    // temperatures. The engine loop thread must normalize all of them instead
    // of throwing; see dflashDraftProbabilities.
    auto           logits = torch::tensor({{0.0f, 1.0f, 2.0f, 3.0f}});
    GenerateConfig config;
    // Smoke-load shape: a greedy request with top_p = 0 stays an exact point mass.
    config.do_sample   = false;
    config.temperature = 0.0f;
    config.top_p       = 0.0f;
    config.top_k       = 1;
    EXPECT_TRUE(torch::equal(dflashDraftProbabilities(logits, config), torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}})));
    // Stochastic with a non-finite temperature, negative top_k and top_p > 1:
    // everything falls back to the plain softmax distribution (1.0 / 0 / 1.0).
    config.do_sample   = true;
    config.temperature = std::numeric_limits<float>::infinity();
    config.top_k       = -3;
    config.top_p       = 2.0f;
    EXPECT_TRUE(
        torch::allclose(dflashDraftProbabilities(logits, config), torch::softmax(logits, -1), 1e-6, 1e-6));
    // A denormal temperature survives the finite/positive checks but would
    // overflow scores/temperature to inf and softmax into NaN; the 1e-6 floor
    // (same as the DSpARK sanitize) keeps the distribution finite and argmax-like.
    config.temperature = std::numeric_limits<float>::denorm_min();
    config.top_k       = 0;
    config.top_p       = 1.0f;
    auto floored       = dflashDraftProbabilities(logits, config);
    EXPECT_TRUE(floored.isfinite().all().item<bool>());
    EXPECT_TRUE(torch::equal(floored, torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}})));
}

TEST(DFlashConfigTest, TypeIsDistinctAndRoundTrips) {
    EXPECT_EQ(SpeculativeExecutionConfig::from_string("dflash"), SP_TYPE_DFLASH);
    EXPECT_EQ(SpeculativeExecutionConfig::to_string(SP_TYPE_DFLASH), "dflash");
    EXPECT_NE(SP_TYPE_DSPARK, SP_TYPE_DFLASH);
    EXPECT_TRUE(isBlockDraftType(SP_TYPE_DFLASH));
    EXPECT_TRUE(isBlockDraftType(SP_TYPE_DSPARK));
    EXPECT_FALSE(isBlockDraftType(SP_TYPE_MTP));
}

TEST(FastTopKSamplerTest, ReturnsInt32MappedTokensWithoutChangingProbabilities) {
    FastTopKSampler sampler(torch::tensor({163839, 163838, 163837, 163836}, torch::kInt64));
    auto            logits = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}});
    auto            out    = sampler.forward(logits, 1);

    EXPECT_EQ(out.token_ids.scalar_type(), torch::kInt32);
    EXPECT_EQ(out.token_ids[0][0].item<int32_t>(), 163837);
    EXPECT_TRUE(torch::equal(out.all_probs, torch::tensor({{0.0f, 0.0f, 1.0f, 0.0f}})));
}

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
