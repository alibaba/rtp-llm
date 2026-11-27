#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

#include <limits>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
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

    size_t graphCaptureCount() const {
        return dsparkGraphCaptureCount();
    }

    size_t graphReplayCount() const {
        return dsparkGraphReplayCount();
    }

    bool captureDirty() const {
        return dsparkCaptureDirty();
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

TEST(ExecSampleFromProbsIntoTest, WritesIntoPreallocatedBuffersWithDeviceRngArrays) {
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto b8  = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
    const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    constexpr int64_t batch = 4;
    constexpr int64_t vocab = 8;
    auto              probs = torch::full({batch, vocab}, 1.0f / static_cast<float>(vocab), f32);
    auto              tokens = torch::empty({batch}, i32);
    auto              valid  = torch::zeros({batch}, b8);

    auto [seed, offset] = execGetSeedAndOffset(batch * 4);
    auto seed_arr       = torch::full({batch}, static_cast<int64_t>(seed), i64);
    auto offset_arr     = torch::empty({batch}, i64);
    for (int64_t b = 0; b < batch; ++b) {
        offset_arr[b] = static_cast<int64_t>(offset + static_cast<uint64_t>(b) * 4ULL);
    }

    execSampleFromProbsInto({probs, tokens, valid, seed_arr, offset_arr});
    EXPECT_TRUE(valid.cpu().all().item<bool>());
    EXPECT_TRUE(tokens.ge(0).all().item<bool>());
    EXPECT_TRUE(tokens.lt(vocab).all().item<bool>());

    // Deterministic replay with identical device seed/offset arrays.
    auto tokens_repeat = torch::empty({batch}, i32);
    auto valid_repeat  = torch::zeros({batch}, b8);
    execSampleFromProbsInto({probs, tokens_repeat, valid_repeat, seed_arr, offset_arr});
    EXPECT_TRUE(torch::equal(tokens, tokens_repeat));
    EXPECT_TRUE(valid_repeat.cpu().all().item<bool>());
}

TEST(DSparkDraftCudaGraphTest, BitExactnessAcrossBatchDtypePaddedVocabAndLargeVocab) {
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    constexpr int64_t gamma       = 3;
    constexpr int64_t markov_rank = 4;

    struct CaseSpec {
        int64_t         batch;
        int64_t         draft_v;
        int64_t         vocab_padded;
        c10::ScalarType base_dtype;
        c10::ScalarType markov_dtype;
    };

    const std::vector<CaseSpec> cases = {
        {1, 5, 11, torch::kFloat32, torch::kFloat32},
        {1, 5, 11, torch::kBFloat16, torch::kBFloat16},
        {3, 5, 13, torch::kFloat32, torch::kFloat32},
        {5, 5, 16, torch::kBFloat16, torch::kFloat32},
        {1, 65536, 65664, torch::kFloat32, torch::kFloat32},
        {3, 65536, 65664, torch::kBFloat16, torch::kBFloat16},
        {5, 65536, 65664, torch::kBFloat16, torch::kFloat32},
    };

    auto generator = at::cuda::detail::getDefaultCUDAGenerator();
    for (size_t idx = 0; idx < cases.size(); ++idx) {
        const auto& c = cases[idx];
        SCOPED_TRACE(idx);
        torch::manual_seed(1000 + static_cast<int64_t>(idx));

        const auto base_opt   = c.base_dtype == torch::kBFloat16 ? bf16 : f32;
        const auto markov_opt = c.markov_dtype == torch::kBFloat16 ? bf16 : f32;
        const int64_t target_v = c.draft_v + 7;

        auto d2t_map = torch::arange(0, c.draft_v, i64);
        auto base_logits = torch::randn({c.batch * gamma, c.vocab_padded}, base_opt) * 0.25f;
        // Fill padded tail [draft_v, vocab_padded) with huge logits to verify truncation to draft_v.
        base_logits.narrow(1, c.draft_v, c.vocab_padded - c.draft_v).fill_(1000.0f);
        // For multi-row batches, pin a dominant peak on row/step so eager vs graph tokens are deterministic too.
        if (c.batch > 1) {
            for (int64_t r = 0; r < c.batch * gamma; ++r) {
                base_logits[r][(r * 3) % c.draft_v] = 40.0f;
            }
        }

        auto anchors     = (torch::arange(0, c.batch, i32) * 2) % target_v;
        auto temperature = torch::linspace(0.5f, 1.3f, c.batch, f32);
        auto markov_w1   = torch::randn({target_v, markov_rank}, markov_opt) * 0.05f;
        auto markov_w2   = torch::randn({c.draft_v, markov_rank}, markov_opt) * 0.05f;

        SpeculativeSampler            eager(d2t_map, gamma);
        InspectableSpeculativeSampler graph(d2t_map, gamma);
        eager.setDSparkCudaGraphEnabled(false);
        graph.setDSparkCudaGraphEnabled(true);

        generator.set_current_seed(7777 + idx);
        auto eager_out = eager.sampleDSparkDraft(base_logits, anchors, temperature, markov_w1, markov_w2, c.draft_v);

        generator.set_current_seed(7777 + idx);
        auto graph_out = graph.sampleDSparkDraft(base_logits, anchors, temperature, markov_w1, markov_w2, c.draft_v);

        EXPECT_EQ(graph.dsparkGraphCacheSize(), 1u);
        EXPECT_EQ(graph.graphCaptureCount(), 1u);
        EXPECT_EQ(graph_out.token_ids.sizes(), eager_out.token_ids.sizes());
        EXPECT_EQ(graph_out.all_probs.sizes(), eager_out.all_probs.sizes());
        EXPECT_TRUE(torch::allclose(graph_out.all_probs, eager_out.all_probs, 1e-5, 1e-6));
        EXPECT_TRUE(torch::equal(graph_out.token_ids, eager_out.token_ids));
        if (c.batch == 1 && c.base_dtype == torch::kFloat32 && c.markov_dtype == torch::kFloat32) {
            EXPECT_TRUE(torch::equal(graph_out.all_probs, eager_out.all_probs));
        }

        // Replay the same key once more and verify cache hit.
        generator.set_current_seed(7777 + idx);
        auto replay_out = graph.sampleDSparkDraft(base_logits, anchors, temperature, markov_w1, markov_w2, c.draft_v);
        EXPECT_EQ(graph.graphCaptureCount(), 1u);
        EXPECT_EQ(graph.graphReplayCount(), 1u);
        EXPECT_TRUE(torch::equal(replay_out.token_ids, graph_out.token_ids));
        EXPECT_TRUE(torch::equal(replay_out.all_probs, graph_out.all_probs));
    }
}

TEST(DSparkDraftCudaGraphTest, RngAdvancesAcrossReplaysAndIsSeedReproducible) {
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    constexpr int64_t batch   = 2;
    constexpr int64_t gamma   = 3;
    constexpr int64_t draft_v = 8;

    auto base_logits = torch::zeros({batch * gamma, draft_v}, f32);
    auto anchors     = torch::zeros({batch}, i32);
    auto temperature = torch::ones({batch}, f32);
    auto markov_w1   = torch::zeros({draft_v, 2}, f32);
    auto markov_w2   = torch::zeros({draft_v, 2}, f32);

    InspectableSpeculativeSampler sampler({}, gamma);
    sampler.setDSparkCudaGraphEnabled(true);

    auto generator = at::cuda::detail::getDefaultCUDAGenerator();
    generator.set_current_seed(20261127);

    std::set<int32_t> observed_tokens;
    int               row_diff_count = 0;
    std::vector<torch::Tensor> run1_tokens;
    for (int iter = 0; iter < 16; ++iter) {
        auto out = sampler.sampleDSparkDraft(base_logits, anchors, temperature, markov_w1, markov_w2, draft_v);
        auto cpu = out.token_ids.cpu();
        for (int64_t r = 0; r < batch; ++r) {
            for (int64_t s = 0; s < gamma; ++s) {
                observed_tokens.insert(cpu[r][s].item<int32_t>());
            }
        }
        if (!torch::equal(cpu[0], cpu[1])) {
            ++row_diff_count;
        }
        if (iter < 4) {
            run1_tokens.push_back(cpu);
        }
    }

    EXPECT_EQ(sampler.graphCaptureCount(), 1u);
    EXPECT_EQ(sampler.graphReplayCount(), 15u);
    EXPECT_GT(observed_tokens.size(), 3u);
    EXPECT_GT(row_diff_count, 0);

    // Resetting seed on another graph-enabled sampler reproduces the exact sequence across capture + replays.
    InspectableSpeculativeSampler sampler_repeat({}, gamma);
    sampler_repeat.setDSparkCudaGraphEnabled(true);
    generator.set_current_seed(20261127);
    for (int iter = 0; iter < 4; ++iter) {
        auto out = sampler_repeat.sampleDSparkDraft(base_logits, anchors, temperature, markov_w1, markov_w2, draft_v);
        EXPECT_TRUE(torch::equal(out.token_ids.cpu(), run1_tokens[iter]));
    }
}

TEST(DSparkDraftCudaGraphTest, BatchShrinkAndGrowIsolatesPaddedTail) {
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    constexpr int64_t gamma   = 3;
    constexpr int64_t draft_v = 6;

    auto markov_w1 = torch::tensor({0.0f, 0.2f, -0.1f, 0.3f, 0.1f, -0.2f}, f32).reshape({draft_v, 1});
    auto markov_w2 = torch::tensor({0.1f, -0.2f, 0.3f, 0.0f, -0.1f, 0.2f}, f32).reshape({draft_v, 1});

    SpeculativeSampler            eager({}, gamma);
    InspectableSpeculativeSampler graph({}, gamma);
    eager.setDSparkCudaGraphEnabled(false);
    graph.setDSparkCudaGraphEnabled(true);

    auto make_peaky_logits = [&](int64_t bsz, int64_t shift) {
        auto logits = torch::full({bsz * gamma, draft_v}, -20.0f, f32);
        for (int64_t b = 0; b < bsz; ++b) {
            for (int64_t s = 0; s < gamma; ++s) {
                logits[b * gamma + s][(b + s + shift) % draft_v] = 30.0f;
            }
        }
        return logits;
    };

    // 1. Capture padded_bsz=4 with B=4
    {
        auto logits  = make_peaky_logits(4, 0);
        auto anchors = torch::tensor({0, 1, 2, 3}, i32);
        auto temps   = torch::tensor({0.5f, 0.8f, 1.1f, 0.4f}, f32);
        auto g_out   = graph.sampleDSparkDraft(logits, anchors, temps, markov_w1, markov_w2, draft_v);
        auto e_out   = eager.sampleDSparkDraft(logits, anchors, temps, markov_w1, markov_w2, draft_v);
        EXPECT_TRUE(torch::equal(g_out.token_ids, e_out.token_ids));
        EXPECT_TRUE(torch::allclose(g_out.all_probs, e_out.all_probs, 1e-5, 1e-6));
    }

    // 2. Shrink to B=3 (reuses padded_bsz=4 graph)
    {
        auto logits  = make_peaky_logits(3, 2);
        auto anchors = torch::tensor({4, 2, 1}, i32);
        auto temps   = torch::tensor({0.7f, 1.3f, 0.9f}, f32);
        auto g_out   = graph.sampleDSparkDraft(logits, anchors, temps, markov_w1, markov_w2, draft_v);
        auto e_out   = eager.sampleDSparkDraft(logits, anchors, temps, markov_w1, markov_w2, draft_v);
        EXPECT_EQ(g_out.token_ids.sizes().vec(), (std::vector<int64_t>{3, gamma}));
        EXPECT_EQ(g_out.all_probs.sizes().vec(), (std::vector<int64_t>{3, gamma, draft_v}));
        EXPECT_TRUE(torch::equal(g_out.token_ids, e_out.token_ids));
        EXPECT_TRUE(torch::allclose(g_out.all_probs, e_out.all_probs, 1e-5, 1e-6));
    }

    // 3. Grow back to B=4 (reuses padded_bsz=4 graph; row 3 must not carry stale clamped tail)
    {
        auto logits  = make_peaky_logits(4, 4);
        auto anchors = torch::tensor({1, 3, 5, 2}, i32);
        auto temps   = torch::tensor({0.6f, 0.9f, 1.2f, 0.5f}, f32);
        auto g_out   = graph.sampleDSparkDraft(logits, anchors, temps, markov_w1, markov_w2, draft_v);
        auto e_out   = eager.sampleDSparkDraft(logits, anchors, temps, markov_w1, markov_w2, draft_v);
        EXPECT_EQ(g_out.token_ids.sizes().vec(), (std::vector<int64_t>{4, gamma}));
        EXPECT_TRUE(torch::equal(g_out.token_ids, e_out.token_ids));
        EXPECT_TRUE(torch::allclose(g_out.all_probs, e_out.all_probs, 1e-5, 1e-6));
    }
    EXPECT_EQ(graph.graphCaptureCount(), 1u);
    EXPECT_EQ(graph.graphReplayCount(), 2u);

    // 4. Repeat shrink/grow inside padded_bsz=8 bucket (B=7 -> B=5 -> B=8)
    for (int64_t bsz : {7, 5, 8}) {
        auto logits  = make_peaky_logits(bsz, bsz);
        auto anchors = torch::arange(0, bsz, i32) % draft_v;
        auto temps   = torch::linspace(0.5f, 1.5f, bsz, f32);
        auto g_out   = graph.sampleDSparkDraft(logits, anchors, temps, markov_w1, markov_w2, draft_v);
        auto e_out   = eager.sampleDSparkDraft(logits, anchors, temps, markov_w1, markov_w2, draft_v);
        EXPECT_EQ(g_out.token_ids.sizes().vec(), (std::vector<int64_t>{bsz, gamma}));
        EXPECT_TRUE(torch::equal(g_out.token_ids, e_out.token_ids));
        EXPECT_TRUE(torch::allclose(g_out.all_probs, e_out.all_probs, 1e-5, 1e-6));
    }
    EXPECT_EQ(graph.graphCaptureCount(), 2u);
    EXPECT_EQ(graph.dsparkGraphCacheSize(), 2u);
}

TEST(DSparkDraftCudaGraphTest, ConsecutiveReplaysReturnOwnedNonAliasedOutputs) {
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    constexpr int64_t batch   = 2;
    constexpr int64_t gamma   = 3;
    constexpr int64_t draft_v = 5;

    auto markov_w1 = torch::zeros({draft_v, 2}, f32);
    auto markov_w2 = torch::zeros({draft_v, 2}, f32);

    InspectableSpeculativeSampler sampler({}, gamma);
    sampler.setDSparkCudaGraphEnabled(true);

    auto logits1 = torch::full({batch * gamma, draft_v}, -30.0f, f32);
    logits1.select(1, 1).fill_(30.0f);
    auto anchors1 = torch::tensor({0, 1}, i32);
    auto temps1   = torch::tensor({0.5f, 0.7f}, f32);

    auto out1           = sampler.sampleDSparkDraft(logits1, anchors1, temps1, markov_w1, markov_w2, draft_v);
    auto out1_tokens_ref = out1.token_ids.cpu().clone();
    auto out1_probs_ref  = out1.all_probs.cpu().clone();

    auto logits2 = torch::full({batch * gamma, draft_v}, -30.0f, f32);
    logits2.select(1, 4).fill_(30.0f);
    auto anchors2 = torch::tensor({2, 3}, i32);
    auto temps2   = torch::tensor({1.3f, 0.9f}, f32);

    auto out2 = sampler.sampleDSparkDraft(logits2, anchors2, temps2, markov_w1, markov_w2, draft_v);
    EXPECT_EQ(sampler.graphCaptureCount(), 1u);
    EXPECT_EQ(sampler.graphReplayCount(), 1u);
    EXPECT_NE(out1.token_ids.data_ptr(), out2.token_ids.data_ptr());
    EXPECT_NE(out1.all_probs.data_ptr(), out2.all_probs.data_ptr());
    EXPECT_TRUE(torch::equal(out1.token_ids.cpu(), out1_tokens_ref));
    EXPECT_TRUE(torch::equal(out1.all_probs.cpu(), out1_probs_ref));
    EXPECT_TRUE(torch::equal(out2.token_ids.cpu(), torch::full({batch, gamma}, 4, torch::kInt32)));
}

TEST(DSparkDraftCudaGraphTest, HandlesCudaInt64Int32AndCpuD2tMapsSafely) {
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    constexpr int64_t gamma    = 2;
    constexpr int64_t draft_v  = 3;
    constexpr int64_t target_v = 6;

    auto base_logits = torch::tensor({-100.0f, -100.0f, 100.0f, 0.0f, 0.0f, 0.0f}, f32).reshape({gamma, draft_v});
    auto anchors     = torch::tensor({0}, i32);
    auto temps       = torch::ones({1}, f32);
    auto markov_w1   = torch::tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, f32).reshape({target_v, 1});
    auto markov_w2   = torch::tensor({-100.0f, 0.0f, 100.0f}, f32).reshape({draft_v, 1});

    for (const auto& d2t : {torch::tensor({1, 3, 5}, i64), torch::tensor({1, 3, 5}, i32)}) {
        InspectableSpeculativeSampler graph(d2t, gamma);
        graph.setDSparkCudaGraphEnabled(true);
        auto out = graph.sampleDSparkDraft(base_logits, anchors, temps, markov_w1, markov_w2, draft_v);
        EXPECT_EQ(graph.dsparkGraphCacheSize(), 1u);
        EXPECT_EQ(graph.graphCaptureCount(), 1u);
        EXPECT_TRUE(torch::equal(out.token_ids.cpu(), torch::tensor({{5, 5}}, torch::kInt32)));
        auto expected_q1 = torch::softmax(torch::tensor({-100.0f, 0.0f, 100.0f}), -1);
        EXPECT_TRUE(torch::allclose(out.all_probs[0][1].cpu(), expected_q1));
    }

    // CPU d2t_map_ cannot run inside stream capture; must fall back cleanly to eager without error.
    auto                          cpu_d2t = torch::tensor({1, 3, 5}, torch::kInt64);
    InspectableSpeculativeSampler cpu_map_sampler(cpu_d2t, gamma);
    cpu_map_sampler.setDSparkCudaGraphEnabled(true);
    auto cpu_out = cpu_map_sampler.sampleDSparkDraft(base_logits, anchors, temps, markov_w1, markov_w2, draft_v);
    EXPECT_EQ(cpu_map_sampler.dsparkGraphCacheSize(), 0u);
    EXPECT_EQ(cpu_map_sampler.graphCaptureCount(), 0u);
    EXPECT_FALSE(cpu_map_sampler.captureDirty());
    EXPECT_TRUE(torch::equal(cpu_out.token_ids.cpu(), torch::tensor({{5, 5}}, torch::kInt32)));
}

TEST(DSparkDraftCudaGraphTest, WeightRebindRecapturesAndCrossStreamReplaysStayRaceFree) {
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    constexpr int64_t batch   = 2;
    constexpr int64_t gamma   = 2;
    constexpr int64_t draft_v = 4;

    auto base_logits = torch::zeros({batch * gamma, draft_v}, f32);
    auto anchors     = torch::tensor({1, 1}, i32);
    auto temps       = torch::ones({batch}, f32);

    auto w1_a = torch::ones({draft_v, 1}, f32);
    auto w2_a = torch::tensor({-40.0f, 40.0f, -40.0f, -40.0f}, f32).reshape({draft_v, 1});
    auto w1_b = torch::ones({draft_v, 1}, f32);
    auto w2_b = torch::tensor({-40.0f, -40.0f, -40.0f, 40.0f}, f32).reshape({draft_v, 1});

    InspectableSpeculativeSampler sampler({}, gamma);
    sampler.setDSparkCudaGraphEnabled(true);

    auto out_a = sampler.sampleDSparkDraft(base_logits, anchors, temps, w1_a, w2_a, draft_v);
    EXPECT_EQ(sampler.graphCaptureCount(), 1u);
    EXPECT_EQ(sampler.dsparkGraphCacheSize(), 1u);
    EXPECT_TRUE(torch::equal(out_a.token_ids.cpu(), torch::full({batch, gamma}, 1, torch::kInt32)));

    // Rebinding Markov weights to new pointers must invalidate the stale graph and re-capture.
    auto out_b = sampler.sampleDSparkDraft(base_logits, anchors, temps, w1_b, w2_b, draft_v);
    EXPECT_EQ(sampler.graphCaptureCount(), 2u);
    EXPECT_EQ(sampler.dsparkGraphCacheSize(), 1u);
    EXPECT_TRUE(torch::equal(out_b.token_ids.cpu(), torch::full({batch, gamma}, 3, torch::kInt32)));

    // Cross-stream alternating replays without host sync between streams.
    auto stream_a = cuda_graph::graphGetStreamFromPool(false);
    auto stream_b = cuda_graph::graphGetStreamFromPool(false);
    auto logits_a = torch::full({batch * gamma, draft_v}, -30.0f, f32);
    logits_a.select(1, 0).fill_(30.0f);
    auto logits_b = torch::full({batch * gamma, draft_v}, -30.0f, f32);
    logits_b.select(1, 2).fill_(30.0f);
    auto w_zero1 = torch::zeros({draft_v, 1}, f32);
    auto w_zero2 = torch::zeros({draft_v, 1}, f32);
    cuda_graph::graphGetCurrentStream().synchronize();

    std::vector<SamplerOutput> outputs_a;
    std::vector<SamplerOutput> outputs_b;
    for (int iter = 0; iter < 4; ++iter) {
        {
            cuda_graph::GraphStreamGuard guard_a(stream_a);
            outputs_a.push_back(sampler.sampleDSparkDraft(logits_a, anchors, temps, w_zero1, w_zero2, draft_v));
        }
        {
            cuda_graph::GraphStreamGuard guard_b(stream_b);
            outputs_b.push_back(sampler.sampleDSparkDraft(logits_b, anchors, temps, w_zero1, w_zero2, draft_v));
        }
    }
    stream_a.synchronize();
    stream_b.synchronize();

    for (int iter = 0; iter < 4; ++iter) {
        EXPECT_TRUE(torch::equal(outputs_a[iter].token_ids.cpu(), torch::full({batch, gamma}, 0, torch::kInt32)));
        EXPECT_TRUE(torch::equal(outputs_b[iter].token_ids.cpu(), torch::full({batch, gamma}, 2, torch::kInt32)));
    }
}

TEST(DSparkDraftCudaGraphTest, NoMarkovWeightsWorkInBothEagerAndGraphModes) {
    const auto f32  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    const auto i32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto i64  = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    constexpr int64_t gamma   = 3;
    constexpr int64_t draft_v = 5;

    for (bool use_d2t : {false, true}) {
        auto d2t = use_d2t ? torch::tensor({2, 4, 6, 8, 10}, i64) : torch::Tensor();
        for (int64_t batch : {1, 3}) {
            for (const auto& opt : {f32, bf16}) {
                auto base_logits = torch::full({batch * gamma, draft_v}, -20.0f, opt);
                for (int64_t r = 0; r < batch * gamma; ++r) {
                    base_logits[r][r % draft_v] = 25.0f;
                }
                auto anchors = torch::zeros({batch}, i32);
                auto temps   = torch::linspace(0.5f, 1.2f, batch, f32);

                SpeculativeSampler            eager(d2t, gamma);
                InspectableSpeculativeSampler graph(d2t, gamma);
                eager.setDSparkCudaGraphEnabled(false);
                graph.setDSparkCudaGraphEnabled(true);

                auto eager_out =
                    eager.sampleDSparkDraft(base_logits, anchors, temps, torch::Tensor(), torch::Tensor(), draft_v);
                auto graph_out =
                    graph.sampleDSparkDraft(base_logits, anchors, temps, torch::Tensor(), torch::Tensor(), draft_v);
                auto replay_out =
                    graph.sampleDSparkDraft(base_logits, anchors, temps, torch::Tensor(), torch::Tensor(), draft_v);

                EXPECT_EQ(graph.dsparkGraphCacheSize(), 1u);
                EXPECT_EQ(graph.graphCaptureCount(), 1u);
                EXPECT_EQ(graph.graphReplayCount(), 1u);
                EXPECT_TRUE(torch::equal(graph_out.token_ids, eager_out.token_ids));
                EXPECT_TRUE(torch::equal(replay_out.token_ids, eager_out.token_ids));
                EXPECT_TRUE(torch::allclose(graph_out.all_probs, eager_out.all_probs, 1e-5, 1e-6));
                EXPECT_TRUE(torch::allclose(replay_out.all_probs, eager_out.all_probs, 1e-5, 1e-6));
            }
        }
    }
}

TEST(DSparkDraftCudaGraphTest, LruCacheEvictsOldestEntriesAtCapacity) {
    const auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    constexpr int64_t gamma = 2;
    InspectableSpeculativeSampler sampler({}, gamma);
    sampler.setDSparkCudaGraphEnabled(true);

    auto anchors = torch::zeros({1}, i32);
    auto temps   = torch::ones({1}, f32);
    for (int64_t draft_v = 4; draft_v < 38; ++draft_v) {
        auto base_logits = torch::zeros({gamma, draft_v}, f32);
        auto out         = sampler.sampleDSparkDraft(base_logits, anchors, temps, torch::Tensor(), torch::Tensor(), draft_v);
        EXPECT_EQ(out.all_probs.size(2), draft_v);
    }
    EXPECT_EQ(sampler.graphCaptureCount(), 34u);
    EXPECT_EQ(sampler.dsparkGraphCacheSize(), 32u);
}

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
