#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <gtest/gtest.h>
#include <limits>
#include <torch/all.h>
#include <vector>

namespace rtp_llm {
namespace speculative {
namespace {

TEST(FastTopKSamplerTest, TopKOneReturnsArgmaxIndex) {
    FastTopKSampler sampler;
    auto            logits = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}});
    auto            out    = sampler.forward(logits, 1);

    ASSERT_EQ(out.token_ids.dim(), 2);
    ASSERT_EQ(out.token_ids.size(0), 1);
    ASSERT_EQ(out.token_ids.size(1), 1);
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
    // softmax preserves ordering: indices 2 (5.0) then 3 (3.0).
    EXPECT_EQ(out.token_ids[0][0].item<int64_t>(), 2);
    EXPECT_EQ(out.token_ids[0][1].item<int64_t>(), 3);
}

struct DSparkConfidenceInputs {
    static constexpr int64_t batch_size = 2, gamma = 3, hidden_dim = 3, markov_rank = 2, vocab_size = 5;
    torch::Tensor            logits;
    torch::Tensor            anchors;
    torch::Tensor            sampling_temperature;
    torch::Tensor            markov_w1;
    torch::Tensor            markov_w2;
    torch::Tensor            hidden;
    torch::Tensor            head_weight;

    static DSparkConfidenceInputs make() {
        const auto             fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        DSparkConfidenceInputs input;
        input.logits = torch::tensor({0.0f,  0.3f, 0.8f, -0.1f, 0.2f,  0.6f, 0.2f,  -0.3f, 0.9f, 0.1f,
                                      -0.2f, 0.4f, 0.1f, 0.3f,  0.8f,  0.8f, 0.1f,  -0.2f, 0.2f, 0.4f,
                                      0.0f,  0.6f, 0.3f, 0.1f,  -0.4f, 0.3f, -0.1f, 0.8f,  0.2f, 0.4f},
                                     fp32)
                           .reshape({batch_size * gamma, vocab_size});
        input.anchors              = torch::tensor({1, 4}, fp32.dtype(torch::kInt32));
        input.sampling_temperature = torch::tensor({0.35f, 1.6f}, fp32);
        input.markov_w1 = torch::tensor({0.0f, 0.5f, 0.25f, -0.5f, 0.5f, 0.75f, -0.75f, 0.25f, 1.0f, -0.25f}, fp32)
                              .reshape({vocab_size, markov_rank});
        input.markov_w2 = torch::tensor({0.1f, -0.2f, 0.3f, 0.1f, -0.1f, 0.4f, 0.2f, -0.3f, 0.4f, 0.2f}, fp32)
                              .reshape({vocab_size, markov_rank});
        input.hidden =
            (torch::arange(batch_size * gamma * hidden_dim, fp32) - 6.0f).reshape({batch_size * gamma, hidden_dim})
            / 8.0f;
        input.head_weight =
            torch::tensor({0.31f, -0.27f, 0.43f, 0.37f, -0.61f}, fp32).reshape({1, hidden_dim + markov_rank});
        return input;
    }

    torch::Tensor hiddenLogits() const {
        return hidden.mm(head_weight.narrow(1, 0, hidden_dim).to(torch::kFloat32).transpose(0, 1));
    }

    SamplerOutput sample(bool observe, bool all_greedy, double confidence_temperature = 0.75) const {
        SpeculativeSampler sampler(torch::Tensor(), gamma);
        return sampler.sampleDSparkDraft(logits,
                                         anchors,
                                         sampling_temperature,
                                         {},
                                         {},
                                         markov_w1,
                                         markov_w2,
                                         vocab_size,
                                         all_greedy,
                                         observe ? hiddenLogits() : torch::Tensor(),
                                         observe ? head_weight : torch::Tensor(),
                                         confidence_temperature);
    }
};

TEST(DSparkConfidenceSamplerTest, MatchesConcatenatedReferenceWithPreviousTokenAndMultipleRequests) {
    constexpr double confidence_temperature = 0.75;
    for (auto weight_dtype : {torch::kFloat32, torch::kBFloat16}) {
        for (bool greedy : {false, true}) {
            auto input        = DSparkConfidenceInputs::make();
            input.head_weight = input.head_weight.to(weight_dtype);
            if (greedy) {
                input.sampling_temperature.fill_(1.0e-6f);
            }
            const auto output = input.sample(true, greedy, confidence_temperature);
            ASSERT_TRUE(output.confidence_rates.defined());
            EXPECT_EQ(output.confidence_rates.scalar_type(), torch::kFloat32);
            EXPECT_EQ(output.confidence_rates.sizes().vec(), (std::vector<int64_t>{input.batch_size, input.gamma}));
            EXPECT_EQ(output.token_ids_are_point_mass, greedy);
            EXPECT_EQ(output.all_probs.defined(), !greedy);

            // Independent scalar dot over the concatenated pre-norm h and the
            // anchor/previous sampled token's embedding, using checkpoint dtype.
            auto hidden     = input.hidden.cpu().contiguous();
            auto embedding  = input.markov_w1.cpu().contiguous();
            auto weight     = input.head_weight.to(torch::kCPU).to(torch::kFloat32).contiguous();
            auto tokens     = output.token_ids.cpu().contiguous();
            auto anchors    = input.anchors.cpu().contiguous();
            auto confidence = output.confidence_rates.cpu().contiguous();
            for (int64_t b = 0; b < input.batch_size; ++b) {
                int32_t previous = anchors.data_ptr<int32_t>()[b];
                for (int64_t step = 0; step < input.gamma; ++step) {
                    double logit = 0.0;
                    for (int64_t d = 0; d < input.hidden_dim; ++d) {
                        logit += static_cast<double>(
                                     hidden.data_ptr<float>()[(b * input.gamma + step) * input.hidden_dim + d])
                                 * weight.data_ptr<float>()[d];
                    }
                    for (int64_t r = 0; r < input.markov_rank; ++r) {
                        logit += static_cast<double>(embedding.data_ptr<float>()[previous * input.markov_rank + r])
                                 * weight.data_ptr<float>()[input.hidden_dim + r];
                    }
                    const double scaled   = std::clamp(logit / confidence_temperature, -60.0, 60.0);
                    const double expected = 1.0 / (1.0 + std::exp(-scaled));
                    EXPECT_NEAR(confidence.data_ptr<float>()[b * input.gamma + step], expected, 1.0e-6)
                        << "batch=" << b << " step=" << step << " greedy=" << greedy;
                    previous = tokens.data_ptr<int32_t>()[b * input.gamma + step];
                }
            }
        }
    }
}

TEST(DSparkConfidenceSamplerTest, ConfidenceTemperatureIsIndependentOfSamplingTemperature) {
    auto input = DSparkConfidenceInputs::make();
    input.sampling_temperature.fill_(1.0e-6f);
    auto greedy                = input.sample(true, true, 0.75);
    input.sampling_temperature = torch::tensor({0.4f, 1.7f}, input.sampling_temperature.options());
    auto stochastic            = input.sample(true, false, 0.75);
    // Position zero has the same hidden and anchor even if later draws differ.
    EXPECT_TRUE(torch::equal(greedy.confidence_rates.select(1, 0), stochastic.confidence_rates.select(1, 0)));
    auto warmer = input.sample(true, false, 2.0);
    EXPECT_FALSE(torch::allclose(stochastic.confidence_rates.select(1, 0), warmer.confidence_rates.select(1, 0)));
}

TEST(DSparkConfidenceSamplerTest, ConfidenceLogitClampMatchesReferenceAndPreservesNan) {
    auto input = DSparkConfidenceInputs::make();
    input.sampling_temperature.fill_(1.0e-6f);
    input.head_weight.narrow(1, input.hidden_dim, input.markov_rank).zero_();
    const float nan           = std::numeric_limits<float>::quiet_NaN();
    auto        hidden_logits = torch::tensor({1000.0f, -1000.0f, nan, 120.0f, -120.0f, 0.0f}, input.logits.options())
                             .reshape({input.batch_size * input.gamma, 1});
    SpeculativeSampler sampler(torch::Tensor(), input.gamma);
    auto               output     = sampler.sampleDSparkDraft(input.logits,
                                            input.anchors,
                                            input.sampling_temperature,
                                                              {},
                                                              {},
                                            input.markov_w1,
                                            input.markov_w2,
                                            input.vocab_size,
                                            true,
                                            hidden_logits,
                                            input.head_weight,
                                            0.5);
    auto               confidence = output.confidence_rates.cpu();
    const double       minimum    = 1.0 / (1.0 + std::exp(60.0));
    EXPECT_FLOAT_EQ(confidence[0][0].item<float>(), 1.0f);
    EXPECT_NEAR(confidence[0][1].item<float>() / minimum, 1.0, 1.0e-6);
    EXPECT_TRUE(std::isnan(confidence[0][2].item<float>()));
    EXPECT_FLOAT_EQ(confidence[1][0].item<float>(), 1.0f);
    EXPECT_NEAR(confidence[1][1].item<float>() / minimum, 1.0, 1.0e-6);
    EXPECT_FLOAT_EQ(confidence[1][2].item<float>(), 0.5f);
}

TEST(DSparkConfidenceSamplerTest, ObservePreservesStochasticTokensExactQAndRngConsumption) {
    auto               input = DSparkConfidenceInputs::make();
    SpeculativeSampler sampler(torch::Tensor(), input.gamma);
    auto               top_k = torch::tensor({3, 4}, input.anchors.options());
    auto               top_p = torch::tensor({0.8f, 0.9f}, input.sampling_temperature.options());
    auto               run   = [&](bool observe) {
        return sampler.sampleDSparkDraft(input.logits,
                                         input.anchors,
                                         input.sampling_temperature,
                                         top_k,
                                         top_p,
                                         input.markov_w1,
                                         input.markov_w2,
                                         input.vocab_size,
                                         false,
                                         observe ? input.hiddenLogits() : torch::Tensor(),
                                         observe ? input.head_weight : torch::Tensor(),
                                         0.65);
    };
    torch::manual_seed(92831);
    auto plain           = run(false);
    auto rng_after_plain = torch::rand({16}, input.logits.options());
    torch::manual_seed(92831);
    auto observed           = run(true);
    auto rng_after_observed = torch::rand({16}, input.logits.options());
    EXPECT_FALSE(plain.confidence_rates.defined());
    EXPECT_TRUE(observed.confidence_rates.defined());
    EXPECT_FALSE(observed.token_ids_are_point_mass);
    ASSERT_TRUE(plain.all_probs.defined());
    ASSERT_TRUE(observed.all_probs.defined());
    EXPECT_TRUE(torch::equal(plain.token_ids, observed.token_ids));
    EXPECT_TRUE(torch::equal(plain.all_probs, observed.all_probs));
    EXPECT_TRUE(torch::equal(rng_after_plain, rng_after_observed));
}

TEST(DSparkConfidenceSamplerTest, RejectsIncompleteOrMalformedConfidenceInputs) {
    auto               input = DSparkConfidenceInputs::make();
    SpeculativeSampler sampler(torch::Tensor(), input.gamma);
    auto run = [&](const torch::Tensor& scalar_logits, const torch::Tensor& weight, double temperature = 1.0) {
        return sampler.sampleDSparkDraft(input.logits,
                                         input.anchors,
                                         input.sampling_temperature,
                                         {},
                                         {},
                                         input.markov_w1,
                                         input.markov_w2,
                                         input.vocab_size,
                                         false,
                                         scalar_logits,
                                         weight,
                                         temperature);
    };
    const auto scalar_logits = input.hiddenLogits();
    EXPECT_THROW(run({}, input.head_weight), std::exception);
    EXPECT_THROW(run(scalar_logits, {}), std::exception);
    EXPECT_THROW(run(input.hidden, input.head_weight), std::exception);
    EXPECT_THROW(run(scalar_logits.flatten(), input.head_weight), std::exception);
    EXPECT_THROW(run(scalar_logits.narrow(0, 0, 1), input.head_weight), std::exception);
    EXPECT_THROW(run(scalar_logits.to(torch::kBFloat16), input.head_weight), std::exception);
    EXPECT_THROW(run(scalar_logits.cpu(), input.head_weight), std::exception);
    auto noncontiguous = torch::zeros({input.batch_size * input.gamma, 2}, input.logits.options()).narrow(1, 0, 1);
    ASSERT_FALSE(noncontiguous.is_contiguous());
    EXPECT_THROW(run(noncontiguous, input.head_weight), std::exception);
    EXPECT_THROW(run(scalar_logits, input.head_weight.flatten()), std::exception);
    EXPECT_THROW(run(scalar_logits, input.head_weight.narrow(1, 0, input.markov_rank)), std::exception);
    EXPECT_THROW(run(scalar_logits, input.head_weight.to(torch::kInt32)), std::exception);
    EXPECT_THROW(run(scalar_logits, input.head_weight.cpu()), std::exception);
    for (double temperature :
         {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        EXPECT_THROW(run(scalar_logits, input.head_weight, temperature), std::exception);
    }
}

// An independent online frontier oracle: it never builds or sorts the complete
// candidate array. Dyadic inputs keep the scalar reference products exact.
std::vector<int32_t> referencePrefixLengths(const std::vector<float>&   rates,
                                            int64_t                     gamma,
                                            const std::vector<int64_t>& stable_order,
                                            int64_t                     budget) {
    const auto           batch_size = stable_order.size();
    std::vector<int32_t> lengths(batch_size, 0);
    std::vector<double>  scores(batch_size, 1.0);
    const auto           clean = [](float value) {
        return std::isfinite(value) && value >= 0.0f && value <= 1.0f ? static_cast<double>(value) : 0.0;
    };
    for (size_t row = 0; row < batch_size; ++row) {
        scores[row] = clean(rates[row * gamma]);
    }
    for (int64_t step = 0; step < budget; ++step) {
        int64_t selected = -1;
        // Strict comparison keeps the first identity on ties. Within a row
        // only its current frontier exists, so its predecessor always wins.
        for (auto row : stable_order) {
            if (lengths[row] < gamma && (selected < 0 || scores[row] > scores[selected])) {
                selected = row;
            }
        }
        ++lengths[selected];
        if (lengths[selected] < gamma) {
            scores[selected] *= clean(rates[selected * gamma + lengths[selected]]);
        }
    }
    return lengths;
}

TEST(DSparkPrefixPlannerTest, MatchesIndependentFrontierForAllSmallDyadicInputs) {
    const std::vector<int64_t> stable_order{1, 0};
    for (auto device : {torch::kCPU, torch::kCUDA}) {
        const auto fp32    = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto       indices = torch::tensor(stable_order, fp32.dtype(torch::kInt64));
        for (int pattern = 0; pattern < 81; ++pattern) {
            std::vector<float> values(4);
            int                remaining = pattern;
            for (auto& value : values) {
                value = static_cast<float>(remaining % 3) * 0.5f;
                remaining /= 3;
            }
            auto rates = torch::tensor(values, fp32).reshape({2, 2});
            for (int64_t budget = 0; budget <= 4; ++budget) {
                SCOPED_TRACE(::testing::Message()
                             << "device=" << device << " pattern=" << pattern << " budget=" << budget);
                auto lengths  = SpeculativeSampler::buildValidDraftLengths(rates, indices, budget);
                auto expected = torch::tensor(referencePrefixLengths(values, 2, stable_order, budget), torch::kInt32);
                EXPECT_EQ(lengths.scalar_type(), torch::kInt32);
                EXPECT_EQ(lengths.device(), rates.device());
                EXPECT_TRUE(lengths.is_contiguous());
                EXPECT_TRUE(torch::equal(lengths.cpu(), expected));
                EXPECT_EQ(lengths.sum().item<int64_t>(), budget);
            }
        }
    }
}

TEST(DSparkPrefixPlannerTest, EmptyAndBoundaryBudgetsDoNotInspectRateValues) {
    for (auto device : {torch::kCPU, torch::kCUDA}) {
        const auto fp32  = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto       empty = SpeculativeSampler::buildValidDraftLengths(
            torch::empty({0, 4}, fp32), torch::empty({0}, fp32.dtype(torch::kInt64)), 0);
        EXPECT_EQ(empty.sizes().vec(), (std::vector<int64_t>{0}));
        EXPECT_EQ(empty.scalar_type(), torch::kInt32);
        EXPECT_EQ(empty.device().type(), device);
        auto rates   = torch::full({3, 1}, std::numeric_limits<float>::quiet_NaN(), fp32);
        auto indices = torch::tensor({2, 0, 1}, fp32.dtype(torch::kInt64));
        EXPECT_TRUE(torch::equal(SpeculativeSampler::buildValidDraftLengths(rates, indices, 0),
                                 torch::zeros({3}, fp32.dtype(torch::kInt32))));
        EXPECT_TRUE(torch::equal(SpeculativeSampler::buildValidDraftLengths(rates, indices, 3),
                                 torch::ones({3}, fp32.dtype(torch::kInt32))));
        EXPECT_TRUE(torch::isnan(rates).all().item<bool>());
    }
}

TEST(DSparkPrefixPlannerTest, SanitizesEachInvalidRateWithoutChangingBudgetOrInput) {
    const float                nan = std::numeric_limits<float>::quiet_NaN();
    const float                inf = std::numeric_limits<float>::infinity();
    const std::vector<float>   values{0.5f, nan, 1.0f, 0.5f, 1.0f, inf, -0.1f, 1.1f, 1.0f};
    const std::vector<int64_t> stable_order{2, 0, 1};
    for (auto device : {torch::kCPU, torch::kCUDA}) {
        auto rates =
            torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32).device(device)).reshape({3, 3});
        auto indices = torch::tensor(stable_order, rates.options().dtype(torch::kInt64));
        for (int64_t budget = 0; budget <= 9; ++budget) {
            auto lengths = SpeculativeSampler::buildValidDraftLengths(rates, indices, budget);
            EXPECT_TRUE(torch::equal(
                lengths.cpu(), torch::tensor(referencePrefixLengths(values, 3, stable_order, budget), torch::kInt32)));
        }
        EXPECT_TRUE(std::isnan(rates[0][1].item<float>()));
        EXPECT_TRUE(std::isinf(rates[1][2].item<float>()));
        EXPECT_FLOAT_EQ(rates[2][0].item<float>(), -0.1f);
        EXPECT_FLOAT_EQ(rates[2][1].item<float>(), 1.1f);
    }
}

TEST(DSparkPrefixPlannerTest, StableIdentitiesSurviveBatchReordering) {
    for (auto device : {torch::kCPU, torch::kCUDA}) {
        auto       rates           = torch::tensor({{0.5f, 1.0f, 0.5f}, {0.5f, 1.0f, 0.5f}, {0.5f, 0.5f, 1.0f}},
                                   torch::TensorOptions().device(device));
        const auto i64             = rates.options().dtype(torch::kInt64);
        auto       original_order  = torch::tensor({1, 2, 0}, i64);
        auto       permutation     = torch::tensor({2, 0, 1}, i64);
        auto       reordered       = rates.index_select(0, permutation);
        auto       reordered_order = torch::tensor({2, 0, 1}, i64);
        for (int64_t budget = 0; budget <= 9; ++budget) {
            auto original = SpeculativeSampler::buildValidDraftLengths(rates, original_order, budget);
            auto changed  = SpeculativeSampler::buildValidDraftLengths(reordered, reordered_order, budget);
            EXPECT_TRUE(torch::equal(changed, original.index_select(0, permutation)));
        }
    }
}

TEST(DSparkPrefixPlannerTest, FutureRatesCannotChangeWhetherCurrentPositionIsSelected) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (auto device : {torch::kCPU, torch::kCUDA}) {
        auto rates = torch::tensor({{0.75f, 0.5f, 0.5f}, {0.5f, 0.75f, 0.5f}}, torch::TensorOptions().device(device));
        auto order = torch::tensor({1, 0}, rates.options().dtype(torch::kInt64));
        for (int64_t position = 0; position < 2; ++position) {
            for (float future : {0.0f, 0.5f, 1.0f, nan}) {
                auto changed = rates.clone();
                changed.select(0, 0).slice(0, position + 1).fill_(future);
                for (int64_t budget = 0; budget <= 6; ++budget) {
                    auto before = SpeculativeSampler::buildValidDraftLengths(rates, order, budget);
                    auto after  = SpeculativeSampler::buildValidDraftLengths(changed, order, budget);
                    EXPECT_EQ(before[0].item<int32_t>() > position, after[0].item<int32_t>() > position);
                }
            }
        }
    }
}

TEST(DSparkPrefixPlannerTest, RoundedProductsAndUnderflowKeepPrefixClosureAtTies) {
    // .01*(.01*.15) and (.01*.01)*.15 differ by an FP32 ulp. The final
    // factor of one must not let position 3 overtake position 2.
    const std::vector<std::vector<float>> cases{
        {0.01f, 0.01f, 0.15f, 1.0f, 0.01f, 0.01f, 0.15f, 1.0f},
        {1.0e-10f, 1.0e-10f, 1.0e-30f, 1.0f, 1.0e-10f, 1.0e-10f, 1.0e-35f, 1.0f}};
    const std::vector<std::vector<int32_t>> expected{
        {0, 0}, {0, 1}, {1, 1}, {1, 2}, {2, 2}, {2, 3}, {2, 4}, {3, 4}, {4, 4}};
    for (auto device : {torch::kCPU, torch::kCUDA}) {
        for (const auto& values : cases) {
            auto rates = torch::tensor(values, torch::TensorOptions().device(device)).reshape({2, 4});
            auto order = torch::tensor({1, 0}, rates.options().dtype(torch::kInt64));
            for (int64_t budget = 0; budget <= 8; ++budget) {
                auto lengths = SpeculativeSampler::buildValidDraftLengths(rates, order, budget);
                EXPECT_TRUE(torch::equal(lengths.cpu(), torch::tensor(expected[budget], torch::kInt32)))
                    << "budget=" << budget;
            }
        }
    }
}

TEST(DSparkPrefixPlannerTest, RejectsMalformedMetadataAndBudget) {
    auto rates = torch::ones({2, 3}, torch::kFloat32);
    auto order = torch::tensor({1, 0}, torch::kInt64);
    for (auto invalid : {torch::Tensor(),
                         rates.flatten(),
                         rates.to(torch::kBFloat16),
                         torch::ones({2, 6}).slice(1, 0, 6, 2),
                         torch::empty({2, 0})}) {
        EXPECT_THROW(SpeculativeSampler::buildValidDraftLengths(invalid, order, 1), std::exception);
    }
    for (auto invalid : {torch::Tensor(),
                         order.reshape({1, 2}),
                         order.to(torch::kInt32),
                         order.narrow(0, 0, 1),
                         torch::arange(4, torch::kInt64).slice(0, 0, 4, 2)}) {
        EXPECT_THROW(SpeculativeSampler::buildValidDraftLengths(rates, invalid, 1), std::exception);
    }
    EXPECT_THROW(SpeculativeSampler::buildValidDraftLengths(rates, order, -1), std::exception);
    EXPECT_THROW(SpeculativeSampler::buildValidDraftLengths(rates, order, 7), std::exception);
    EXPECT_THROW(SpeculativeSampler::buildValidDraftLengths(rates.to(torch::kCUDA), order, 1), std::exception);
}

TEST(DSparkBoundedForwardTest, RejectsMalformedLengthsBeforeReadingStreamState) {
    SpeculativeSampler sampler({}, 3);
    SamplerOutput      draft;
    SamplerOutput      target;
    const auto         i32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    // Null stream sentinels make this a metadata-boundary test: every invalid
    // length tensor must be rejected before any stream or probability access.
    const std::list<GenerateStreamPtr> streams{nullptr, nullptr};
    for (const auto& invalid : {torch::ones({2}, torch::kInt32),
                                torch::ones({2}, i32.dtype(torch::kInt64)),
                                torch::ones({2, 1}, i32),
                                torch::ones({1}, i32),
                                torch::ones({4}, i32).slice(0, 0, 4, 2)}) {
        draft.valid_draft_lengths = invalid;
        try {
            sampler.forward(streams, draft, target);
            FAIL() << "malformed valid draft lengths were accepted";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find("valid draft lengths must"), std::string::npos);
        }
    }
}

TEST(DSparkConfidenceStatsTest, CensorsAfterRejectionAndExcludesCapBonus) {
    auto rates = torch::tensor({{0.1f, 0.2f, 0.3f}, {0.8f, 0.4f, 0.2f}, {0.9f, 0.7f, 0.6f}, {0.5f, 0.1f, 0.9f}});
    // Immediate rejection, rejection at position 1, all accepted, and a
    // grammar/length cap after one accepted draft. The last row's position 1
    // emits a target bonus and is not an observed rejection of a draft.
    auto accept_len = torch::tensor({1, 2, 4, 2}, torch::kInt32);
    auto lengths    = torch::tensor({3, 3, 3, 1}, torch::kInt32);
    auto stats      = SpeculativeSampler::buildConfidenceStats(rates, accept_len, lengths);
    auto expected   = torch::tensor({{4.0f, 3.0f, 2.3f, 0.9f, 0.31f, 0.0f},
                                     {2.0f, 1.0f, 1.1f, 0.7f, 0.25f, 0.0f},
                                     {1.0f, 1.0f, 0.6f, 0.4f, 0.16f, 0.0f}});
    EXPECT_TRUE(torch::allclose(stats, expected, 1.0e-6, 1.0e-6));
    EXPECT_EQ(stats.scalar_type(), torch::kFloat32);
    auto no_cap        = SpeculativeSampler::buildConfidenceStats(rates.narrow(0, 0, 3), accept_len.narrow(0, 0, 3));
    auto explicit_full = SpeculativeSampler::buildConfidenceStats(
        rates.narrow(0, 0, 3), accept_len.narrow(0, 0, 3), lengths.narrow(0, 0, 3));
    EXPECT_TRUE(torch::equal(no_cap, explicit_full));
}

TEST(DSparkConfidenceStatsTest, ExcludesForcedRowsAndCountsOnlyAttemptedInvalidPredictions) {
    const float nan        = std::numeric_limits<float>::quiet_NaN();
    const float inf        = std::numeric_limits<float>::infinity();
    auto        rates      = torch::tensor({{nan, inf, -0.1f}, {1.1f, nan, inf}, {0.9f, 0.8f, 0.7f}});
    auto        accept_len = torch::tensor({4, 1, 4}, torch::kInt32);
    auto        eligible   = torch::tensor({true, true, false}, torch::kBool);
    auto        stats      = SpeculativeSampler::buildConfidenceStats(rates, accept_len, {}, eligible);
    auto        expected   = torch::zeros({3, 6}, torch::kFloat32);
    expected.select(1, 5).copy_(torch::tensor({2.0f, 1.0f, 1.0f}));
    EXPECT_TRUE(torch::equal(stats, expected));
    EXPECT_TRUE(torch::isfinite(stats).all().item<bool>());
}

TEST(DSparkConfidenceStatsTest, ZeroValidLengthAndEmptyBatchHaveNoObservations) {
    auto rates = torch::tensor({{0.5f, 0.5f, 0.5f}});
    auto stats = SpeculativeSampler::buildConfidenceStats(
        rates, torch::tensor({1}, torch::kInt32), torch::tensor({0}, torch::kInt32));
    EXPECT_TRUE(torch::equal(stats, torch::zeros({3, 6}, torch::kFloat32)));
    auto empty = SpeculativeSampler::buildConfidenceStats(torch::empty({0, 3}, torch::kFloat32),
                                                          torch::empty({0}, torch::kInt32));
    EXPECT_TRUE(torch::equal(empty, torch::zeros({3, 6}, torch::kFloat32)));
}

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
