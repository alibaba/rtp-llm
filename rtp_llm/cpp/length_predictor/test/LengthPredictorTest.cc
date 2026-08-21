#include <gtest/gtest.h>

#include <cmath>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/length_predictor/LengthPredictor.h"

namespace rtp_llm {
namespace {

LengthPredictorConfig smallConfig() {
    LengthPredictorConfig config;
    config.hidden_dim         = 24;
    config.feature_dim        = 8;
    config.state_dim          = 6;
    config.adapter_dim        = 6;
    config.time_dim           = 4;
    config.num_bins           = 16;
    config.t_cap              = 1000.0;
    config.history_stride     = 4;
    config.predict_stride     = 20;
    config.hard_anchor_until  = 100.0;
    config.half_life_tokens   = 40.0;
    config.max_history_weight = 1.0;
    return config;
}

std::unordered_map<std::string, torch::Tensor> randomWeights(const LengthPredictorConfig& config, uint64_t seed) {
    torch::manual_seed(seed);
    const auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    std::unordered_map<std::string, torch::Tensor> weights;
    auto normal = [&](std::initializer_list<int64_t> shape) { return torch::randn(shape, opts) * 0.2; };

    weights["encoder_ln_weight"]     = torch::ones({config.hidden_dim}, opts) + normal({config.hidden_dim});
    weights["encoder_ln_bias"]       = normal({config.hidden_dim});
    weights["encoder_linear_weight"] = normal({config.feature_dim, config.hidden_dim});
    weights["encoder_linear_bias"]   = normal({config.feature_dim});
    weights["time_linear_weight"]    = normal({config.time_dim, 2});
    weights["time_linear_bias"]      = normal({config.time_dim});
    weights["fusion1_weight"]        = normal({config.feature_dim, config.feature_dim + config.time_dim});
    weights["fusion1_bias"]          = normal({config.feature_dim});
    weights["fusion2_weight"]        = normal({config.num_bins, config.feature_dim});
    weights["fusion2_bias"]          = normal({config.num_bins});
    weights["adapter_weight"]        = normal({config.adapter_dim, config.feature_dim});
    weights["adapter_bias"]          = normal({config.adapter_dim});
    weights["gru_weight_ih"]         = normal({3 * config.state_dim, config.adapter_dim + 2});
    weights["gru_weight_hh"]         = normal({3 * config.state_dim, config.state_dim});
    weights["gru_bias_ih"]           = normal({3 * config.state_dim});
    weights["gru_bias_hh"]           = normal({3 * config.state_dim});
    weights["modulator1_weight"]     = normal({10, config.state_dim});
    weights["modulator1_bias"]       = normal({10});
    weights["modulator2_weight"]     = normal({2 * config.feature_dim, 10});
    weights["modulator2_bias"]       = normal({2 * config.feature_dim});
    // log1p-space centers must be strictly increasing.
    weights["bin_centers"] = torch::linspace(0.0, 9.0, config.num_bins, opts);
    return weights;
}

torch::Tensor randomFeature(const LengthPredictorConfig& config, uint64_t seed) {
    torch::manual_seed(seed);
    return torch::randn({1, config.feature_dim}, torch::TensorOptions().dtype(torch::kFloat32));
}

void processFeature(const LengthPredictor& predictor,
                    LengthPredictorState&  state,
                    const torch::Tensor&   feature,
                    int64_t                decode_step) {
    predictor.processRowForTest(state, feature.data_ptr<float>(), decode_step);
}

double predictedTotal(const LengthPredictorState& state) {
    return state.predicted_total.load(std::memory_order_relaxed);
}

TEST(LengthPredictorTest, AnchorIsCountdownThroughHardAnchorWindow) {
    auto            config = smallConfig();
    LengthPredictor predictor(config, randomWeights(config, 42));

    LengthPredictorState state;
    processFeature(predictor, state, randomFeature(config, 1), 0);
    ASSERT_TRUE(state.anchor_ready);
    ASSERT_GT(state.anchor_total, 0.0);
    const double anchor = state.anchor_total;

    // All predictions at t <= hard_anchor_until must equal the anchor exactly.
    for (int64_t t = 1; t <= 100; ++t) {
        processFeature(predictor, state, randomFeature(config, 100 + t), t);
        EXPECT_DOUBLE_EQ(predictedTotal(state), anchor) << "t=" << t;
    }
    // The GRU must have consumed observations meanwhile.
    EXPECT_EQ(state.last_obs_step, 97);  // last t with (t-1) % 4 == 0 within 100
}

TEST(LengthPredictorTest, HistoryTakesOverAfterTransition) {
    auto            config = smallConfig();
    LengthPredictor predictor(config, randomWeights(config, 42));

    LengthPredictorState state;
    processFeature(predictor, state, randomFeature(config, 1), 0);
    const double anchor = state.anchor_total;

    for (int64_t t = 1; t <= 400; ++t) {
        processFeature(predictor, state, randomFeature(config, 1000 + t), t);
    }
    // alpha(140) = 0.5, alpha(180) = 0.75, alpha(260) ~ 0.9375.
    EXPECT_NEAR(predictor.alphaAt(140), 0.5, 1e-12);
    EXPECT_NEAR(predictor.alphaAt(180), 0.75, 1e-12);
    EXPECT_NEAR(predictor.alphaAt(260), 0.9375, 1e-12);
    EXPECT_DOUBLE_EQ(predictor.alphaAt(100), 0.0);
    EXPECT_GT(predictor.alphaAt(400), 0.99);

    // After the transition the prediction must have moved off the anchor.
    EXPECT_NE(predictedTotal(state), anchor);
    EXPECT_GT(predictedTotal(state), 0.0);
}

TEST(LengthPredictorTest, FusionIsLog1pInterpolationWithExactEndpoints) {
    auto            config = smallConfig();
    LengthPredictor predictor(config, randomWeights(config, 42));

    EXPECT_DOUBLE_EQ(predictor.fuseTotals(123.0, 456.0, 0.0), 123.0);
    EXPECT_NEAR(predictor.fuseTotals(123.0, 456.0, 1.0), 456.0, 1e-9);
    const double mid      = predictor.fuseTotals(100.0, 400.0, 0.5);
    const double expected = std::expm1(0.5 * std::log1p(100.0) + 0.5 * std::log1p(400.0));
    EXPECT_NEAR(mid, expected, 1e-9);
}

TEST(LengthPredictorTest, PredictionIsCausalAgainstSameStepObservation) {
    // At a step that is both a prediction and an observation point, the
    // prediction must use the GRU state accumulated strictly before this step;
    // the current observation may only enter the state afterwards.
    auto config              = smallConfig();
    config.predict_stride    = 21;   // t=21 is on both grids: (21-1)%4==0 and 21%21==0
    config.hard_anchor_until = 4.0;  // ensure alpha(21) > 0 so the history path runs
    auto            weights = randomWeights(config, 42);
    LengthPredictor predictor(config, weights);

    LengthPredictorState state;
    processFeature(predictor, state, randomFeature(config, 7), 0);
    for (int64_t t = 1; t < 21; ++t) {
        processFeature(predictor, state, randomFeature(config, 5000 + t), t);
    }
    auto state_before_t21 = torch::tensor(state.gru_state).view({1, -1}).clone();

    auto feature_t21 = randomFeature(config, 9999);
    processFeature(predictor, state, feature_t21, 21);

    // Manually reproduce the prediction with the pre-t21 state (torch reference).
    auto inner = torch::nn::functional::gelu(torch::nn::functional::linear(
        state_before_t21, weights.at("modulator1_weight"), weights.at("modulator1_bias")));
    auto raw    = torch::nn::functional::linear(inner, weights.at("modulator2_weight"), weights.at("modulator2_bias"));
    auto chunks = raw.chunk(2, -1);
    auto modulated = feature_t21 * (1.0 + config.scale_limit * torch::tanh(chunks[0]))
                     + config.shift_limit * torch::tanh(chunks[1]);
    const double t     = 21.0;
    auto time_feature  = torch::tensor(
        {{static_cast<float>(t / config.t_cap), static_cast<float>(std::log1p(t) / std::log1p(config.t_cap))}});
    auto encoded_time = torch::nn::functional::gelu(torch::nn::functional::linear(
        time_feature, weights.at("time_linear_weight"), weights.at("time_linear_bias")));
    auto fusion_inner = torch::nn::functional::gelu(torch::nn::functional::linear(
        torch::cat({modulated, encoded_time}, -1), weights.at("fusion1_weight"), weights.at("fusion1_bias")));
    auto logits = torch::nn::functional::linear(fusion_inner, weights.at("fusion2_weight"), weights.at("fusion2_bias"));
    auto probabilities = torch::softmax(logits, -1);
    const double history_remaining =
        std::expm1((probabilities * weights.at("bin_centers").view({1, -1})).sum(-1).item<double>());
    const double alpha    = predictor.alphaAt(21);
    const double expected = predictor.fuseTotals(state.anchor_total, history_remaining + t, alpha);

    EXPECT_NEAR(predictedTotal(state), expected, 1e-4);
    // The same-step observation was consumed only after predicting.
    EXPECT_EQ(state.last_obs_step, 21);
    auto state_after = torch::tensor(state.gru_state).view({1, -1});
    EXPECT_FALSE(torch::allclose(state_after, state_before_t21));
}

TEST(LengthPredictorTest, GruMatchesTorchGRUCell) {
    auto config  = smallConfig();
    auto weights = randomWeights(config, 42);
    LengthPredictor predictor(config, weights);

    torch::nn::GRUCell reference(torch::nn::GRUCellOptions(config.adapter_dim + 2, config.state_dim));
    {
        torch::NoGradGuard no_grad;
        reference->weight_ih.copy_(weights.at("gru_weight_ih"));
        reference->weight_hh.copy_(weights.at("gru_weight_hh"));
        reference->bias_ih.copy_(weights.at("gru_bias_ih"));
        reference->bias_hh.copy_(weights.at("gru_bias_hh"));
    }

    LengthPredictorState state;
    auto feature = randomFeature(config, 3);
    processFeature(predictor, state, feature, 0);

    auto adapted = torch::nn::functional::gelu(
        torch::nn::functional::linear(feature, weights.at("adapter_weight"), weights.at("adapter_bias")));
    auto gru_input = torch::cat({adapted, torch::tensor({{0.0f, 0.0f}})}, -1);
    auto expected  = reference->forward(gru_input, torch::zeros({1, config.state_dim}));

    auto actual = torch::tensor(state.gru_state).view({1, -1});
    EXPECT_TRUE(torch::allclose(actual, expected, /*rtol=*/1e-4, /*atol=*/1e-5));
}

TEST(LengthPredictorTest, EncodeRowMatchesTorchLayerNormLinearGelu) {
    // encodeRowForTest folds gamma/beta into the linear weights; the result
    // must match the unfolded torch reference layer_norm -> linear -> gelu.
    auto config  = smallConfig();
    auto weights = randomWeights(config, 42);
    LengthPredictor predictor(config, weights);

    torch::manual_seed(11);
    auto hidden     = torch::randn({1, config.hidden_dim});
    auto normalized = torch::layer_norm(
        hidden, {config.hidden_dim}, weights.at("encoder_ln_weight"), weights.at("encoder_ln_bias"), 1e-5);
    auto expected = torch::nn::functional::gelu(torch::nn::functional::linear(
        normalized, weights.at("encoder_linear_weight"), weights.at("encoder_linear_bias")));

    std::vector<float> feature(config.feature_dim);
    predictor.encodeRowForTest(hidden.data_ptr<float>(), feature.data());
    auto actual = torch::tensor(feature).view({1, -1});
    EXPECT_TRUE(torch::allclose(actual, expected, /*rtol=*/1e-4, /*atol=*/1e-5));
}

TEST(LengthPredictorTest, OffGridStepsAreNoOps) {
    auto            config = smallConfig();
    LengthPredictor predictor(config, randomWeights(config, 42));

    LengthPredictorState state;
    processFeature(predictor, state, randomFeature(config, 1), 0);
    auto         state_before = state.gru_state;
    const double total_before = predictedTotal(state);

    // t=2,3,4: (t-1)%4 = 1,2,3 and none is a predict point (<20).
    for (int64_t t : {2, 3, 4}) {
        processFeature(predictor, state, randomFeature(config, 60 + t), t);
    }
    EXPECT_EQ(state.gru_state, state_before);
    EXPECT_DOUBLE_EQ(predictedTotal(state), total_before);
    EXPECT_EQ(state.last_obs_step, 0);

    // t=5 is on the observation grid.
    processFeature(predictor, state, randomFeature(config, 65), 5);
    EXPECT_NE(state.gru_state, state_before);
    EXPECT_EQ(state.last_obs_step, 5);
}

TEST(LengthPredictorTest, AsyncSubmitPublishesAnchorAndCountdown) {
    // Exercises the queue/worker path with CPU hidden tensors end to end.
    auto            config = smallConfig();
    LengthPredictor predictor(config, randomWeights(config, 42));

    auto state = std::make_shared<LengthPredictorState>();
    torch::manual_seed(21);

    // Step with t=0 (prefill completion) for batch row 0 of a 3-row batch.
    auto hidden = torch::randn({3, config.hidden_dim});
    predictor.submitStep(hidden, {LengthPredictorEntry{state, state.get(), 0, 0}});
    predictor.drainForTest();
    EXPECT_TRUE(state->anchor_ready);
    const double anchor = state->anchor_total;
    EXPECT_GT(anchor, 0.0);
    EXPECT_DOUBLE_EQ(predictedTotal(*state), anchor);

    // A later on-grid observation advances the GRU without changing the anchor.
    auto hidden2 = torch::randn({1, config.hidden_dim});
    predictor.submitStep(hidden2, {LengthPredictorEntry{state, state.get(), 0, 5}});
    predictor.drainForTest();
    EXPECT_EQ(state->last_obs_step, 5);
    EXPECT_DOUBLE_EQ(predictedTotal(*state), anchor);
    EXPECT_EQ(predictor.droppedPackets(), 0);
}

TEST(LengthPredictorTest, RejectsBadWeightShapes) {
    auto config  = smallConfig();
    auto weights = randomWeights(config, 42);
    weights["fusion2_weight"] = torch::randn({config.num_bins, config.feature_dim + 1});
    EXPECT_THROW(LengthPredictor(config, weights), std::runtime_error);

    weights = randomWeights(config, 42);
    weights.erase("bin_centers");
    EXPECT_THROW(LengthPredictor(config, weights), std::runtime_error);
}

}  // namespace
}  // namespace rtp_llm
