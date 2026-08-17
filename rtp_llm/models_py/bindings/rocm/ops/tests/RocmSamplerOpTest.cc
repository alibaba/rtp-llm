#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <torch/torch.h>

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

using namespace rtp_llm;

namespace {

class HipDeviceGuard {
public:
    explicit HipDeviceGuard(int device) {
        if (hipGetDevice(&previous_device_) != hipSuccess || hipSetDevice(device) != hipSuccess) {
            throw std::runtime_error("failed to select ROCm device");
        }
    }

    ~HipDeviceGuard() {
        static_cast<void>(hipSetDevice(previous_device_));
    }

private:
    int previous_device_ = 0;
};

std::vector<at::Generator> makeGenerators(int64_t batch_size, uint64_t first_seed) {
    std::vector<at::Generator> generators;
    generators.reserve(batch_size);
    for (int64_t row = 0; row < batch_size; ++row) {
        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
        generator.set_current_seed(first_seed + row);
        generators.push_back(generator);
    }
    return generators;
}

SamplerInputs makeSamplerInputs(int64_t batch_size, int32_t top_k_value, float top_p_value, uint64_t seed) {
    constexpr int64_t vocab_size = 8;
    auto              gpu        = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto              logits     = torch::tensor({0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f, 1.4f}, torch::kFloat32)
                      .repeat({batch_size, 1})
                      .to(torch::kCUDA);
    return {logits,
            torch::zeros({batch_size, 1}, torch::kInt32),
            torch::zeros({batch_size}, torch::kInt32),
            torch::zeros({batch_size}, torch::kInt32),
            std::make_shared<LogitsProcessorStates>(),
            vocab_size,
            /*step=*/0,
            static_cast<size_t>(batch_size),
            static_cast<size_t>(batch_size),
            torch::ones({batch_size}, torch::kLong),
            torch::ones({batch_size}, torch::kLong),
            torch::full({batch_size}, top_k_value, torch::kInt32).pin_memory(),
            torch::full({batch_size}, top_p_value, torch::kFloat32).pin_memory(),
            torch::ones({batch_size}, torch::kFloat32).pin_memory(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            /*return_original_all_probs=*/false,
            torch::zeros({batch_size}, torch::kFloat32),
            torch::zeros({batch_size, vocab_size}, gpu),
            makeGenerators(batch_size, seed)};
}

void expectSampleMatchesReturnedProbability(const SamplerOutput& output) {
    auto tokens = output.token_ids.flatten().cpu();
    auto probs  = output.all_probs.cpu();
    auto cum    = output.cum_log_probs.cpu();
    for (int64_t row = 0; row < tokens.numel(); ++row) {
        const auto token = tokens[row].item<int32_t>();
        const auto prob  = probs[row][token].item<float>();
        EXPECT_GT(prob, 0.0f);
        EXPECT_NEAR(probs[row].sum().item<float>(), 1.0f, 1e-5f);
        EXPECT_NEAR(cum[row].item<float>(), std::log(prob), 1e-5f);
    }
}

torch::Tensor jointTopKTopPReference(int64_t top_k, float top_p) {
    auto probs = torch::softmax(torch::tensor({0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f, 1.4f}, torch::kFloat32), 0);
    auto top_k_threshold = std::get<0>(probs.topk(top_k)).min();
    auto sorted_probs    = std::get<0>(probs.sort(/*dim=*/0, /*descending=*/true));
    auto top_p_keep      = (sorted_probs.cumsum(0) - sorted_probs).lt(top_p);
    auto top_p_threshold = sorted_probs.masked_select(top_p_keep).min();
    auto filtered        = probs * probs.ge(top_k_threshold) * probs.ge(top_p_threshold);
    return filtered / filtered.sum();
}

TEST(RocmSamplerOpTest, ProductionSamplerCoversAllDispatchesAndReusesSlots) {
    Sampler sampler(SamplerInitParams{/*max_batch_size=*/16, /*fixed_max_batch_size=*/true});
    const std::vector<std::pair<int32_t, float>> dispatches = {
        {0, 0.75f},  // pure top-p
        {4, 1.0f},   // pure top-k
        {4, 0.75f},  // combined top-k/top-p
        {4, 0.75f},  // fourth call reuses the first persistent buffer slot
    };

    for (size_t iteration = 0; iteration < dispatches.size(); ++iteration) {
        auto inputs = makeSamplerInputs(/*batch_size=*/16,
                                        dispatches[iteration].first,
                                        dispatches[iteration].second,
                                        /*seed=*/1000 + iteration * 100);
        auto output = sampler.forward(inputs);
        runtimeSyncAndCheck();
        expectSampleMatchesReturnedProbability(output);
    }
}

TEST(RocmSamplerOpTest, CombinedSamplingMatchesReturnedDistribution) {
    constexpr int64_t batch_size = 2048;
    Sampler           sampler(SamplerInitParams{/*max_batch_size=*/batch_size, /*fixed_max_batch_size=*/true});
    auto              inputs    = makeSamplerInputs(batch_size, /*top_k=*/4, /*top_p=*/0.75f, /*seed=*/5000);
    auto              reference = jointTopKTopPReference(/*top_k=*/4, /*top_p=*/0.75f);
    auto              output    = sampler.forward(inputs);
    runtimeSyncAndCheck();
    expectSampleMatchesReturnedProbability(output);

    EXPECT_TRUE(torch::allclose(output.all_probs[0].cpu(), reference, /*rtol=*/1e-5, /*atol=*/1e-6));
    auto tokens = output.token_ids.flatten().cpu();
    auto counts =
        torch::bincount(tokens.to(torch::kLong), std::nullopt, reference.numel()).to(torch::kFloat32) / batch_size;
    EXPECT_LT(torch::max(torch::abs(counts - reference)).item<float>(), 0.04f);
}

TEST(RocmSamplerOpTest, CumulativeLogProbWorksWithoutAllProbabilityOutput) {
    Sampler sampler(SamplerInitParams{/*max_batch_size=*/8, /*fixed_max_batch_size=*/true});
    auto    inputs   = makeSamplerInputs(/*batch_size=*/8, /*top_k=*/4, /*top_p=*/0.75f, /*seed=*/6000);
    inputs.all_probs = torch::Tensor();
    auto output      = sampler.forward(inputs);
    runtimeSyncAndCheck();

    EXPECT_FALSE(output.all_probs.defined());
    EXPECT_TRUE(torch::isfinite(output.cum_log_probs).all().item<bool>());
    EXPECT_TRUE(output.cum_log_probs.lt(0).all().item<bool>());
}

std::vector<int32_t> sampleRequestAcrossRounds(int64_t request_row, int64_t batch_size) {
    constexpr int64_t rounds            = 8;
    constexpr int64_t vocab_size        = 8;
    auto              generators        = makeGenerators(batch_size, /*first_seed=*/7000);
    auto              request_generator = torch::make_generator<at::CUDAGeneratorImpl>();
    request_generator.set_current_seed(20260817);
    generators[request_row] = request_generator;

    std::vector<int32_t> samples;
    for (int64_t round = 0; round < rounds; ++round) {
        auto logits = torch::linspace(0.0f, 1.0f, vocab_size, torch::kFloat32).repeat({batch_size, 1}).to(torch::kCUDA);
        auto token_ids        = torch::zeros({batch_size, 1}, torch::kInt32);
        auto input_lengths    = torch::zeros({batch_size}, torch::kInt32);
        auto sequence_lengths = torch::empty({0}, torch::kInt32);
        execSampleGreedy({logits,
                          input_lengths,
                          sequence_lengths,
                          token_ids,
                          /*step=*/0,
                          torch::full({batch_size}, 4, torch::kInt32).pin_memory(),
                          torch::full({batch_size}, 0.8f, torch::kFloat32).pin_memory(),
                          torch::ones({batch_size}, torch::kFloat32).pin_memory(),
                          std::nullopt,
                          std::nullopt,
                          std::nullopt,
                          std::nullopt,
                          /*return_original_all_probs=*/false,
                          std::nullopt,
                          std::nullopt,
                          std::nullopt,
                          std::nullopt,
                          generators});
        runtimeSyncAndCheck();
        samples.push_back(token_ids[request_row].item<int32_t>());
    }
    return samples;
}

TEST(RocmSamplerOpTest, RequestGeneratorIsStableAcrossDynamicBatchTopology) {
    EXPECT_EQ(sampleRequestAcrossRounds(/*request_row=*/0, /*batch_size=*/1),
              sampleRequestAcrossRounds(/*request_row=*/7, /*batch_size=*/16));
}

TEST(RocmSamplerOpTest, RunsProductionSamplerOnNonzeroDevice) {
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "requires at least two ROCm devices";
    }
    HipDeviceGuard guard(/*device=*/1);
    Sampler        sampler(SamplerInitParams{/*max_batch_size=*/8, /*fixed_max_batch_size=*/true});
    auto output = sampler.forward(makeSamplerInputs(/*batch_size=*/8, /*top_k=*/5, /*top_p=*/0.8f, /*seed=*/9000));
    runtimeSyncAndCheck();
    expectSampleMatchesReturnedProbability(output);
}

}  // namespace
