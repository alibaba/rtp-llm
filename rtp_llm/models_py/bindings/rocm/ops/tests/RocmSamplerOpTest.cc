#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/hip/HIPContext.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/rocm/kernels/sampling/sampling.h"

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

class RocmSamplerInvalidProbTest: public ::testing::TestWithParam<std::pair<int32_t, float>> {};

TEST_P(RocmSamplerInvalidProbTest, ReportsInvalidRowsWithoutAffectingHealthyRows) {
    const auto [top_k, top_p] = GetParam();
    const float nan           = std::numeric_limits<float>::quiet_NaN();
    const float inf           = std::numeric_limits<float>::infinity();
    Sampler     sampler(SamplerInitParams{/*max_batch_size=*/5, /*fixed_max_batch_size=*/true});

    // Exercise no probability outputs, normalized outputs, and original outputs.
    for (int output_mode = 0; output_mode < 3; ++output_mode) {
        auto reference = sampler.forward(makeSamplerInputs(5, top_k, top_p, /*seed=*/12000));
        auto inputs    = makeSamplerInputs(5, top_k, top_p, /*seed=*/12000);
        inputs.logits[1][0].fill_(nan);
        inputs.logits[2].fill_(-inf);
        inputs.logits[3][0].fill_(inf);
        if (output_mode == 0) {
            inputs.all_probs     = torch::Tensor();
            inputs.cum_log_probs = torch::Tensor();
        }
        inputs.return_original_all_probs = output_mode == 2;
        auto output                      = sampler.forward(inputs);
        runtimeSyncAndCheck();

        auto success          = output.success.cpu();
        auto tokens           = output.token_ids.flatten().cpu();
        auto reference_tokens = reference.token_ids.flatten().cpu();
        for (int64_t row = 0; row < 5; ++row) {
            const bool valid = row == 0 || row == 4;
            EXPECT_EQ(success[row].item<bool>(), valid);
            if (valid) {
                EXPECT_EQ(tokens[row].item<int32_t>(), reference_tokens[row].item<int32_t>());
            } else {
                EXPECT_EQ(tokens[row].item<int32_t>(), -1);
            }
            if (output_mode != 0) {
                auto        probs = output.all_probs[row].cpu();
                const float cum   = output.cum_log_probs[row].item<float>();
                if (valid) {
                    EXPECT_TRUE(torch::isfinite(probs).all().item<bool>());
                    EXPECT_NEAR(probs.sum().item<float>(), 1.0f, 1e-5f);
                    EXPECT_NEAR(cum, std::log(probs[tokens[row].item<int32_t>()].item<float>()), 1e-5f);
                    if (output_mode == 1) {
                        EXPECT_TRUE(torch::allclose(probs, reference.all_probs[row].cpu()));
                    }
                } else {
                    EXPECT_TRUE(probs.eq(0).all().item<bool>());
                    EXPECT_EQ(cum, -inf);
                }
            }
        }
    }
}

TEST_P(RocmSamplerInvalidProbTest, RejectsSingleInvalidRow) {
    const auto [top_k, top_p] = GetParam();
    Sampler sampler(SamplerInitParams{/*max_batch_size=*/1, /*fixed_max_batch_size=*/true});
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                          -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::infinity()}) {
        auto inputs = makeSamplerInputs(1, top_k, top_p, /*seed=*/13000);
        inputs.logits.fill_(invalid);
        // Also exercise the internal probability buffer allocated for cum_log_probs.
        inputs.all_probs = torch::Tensor();
        auto output      = sampler.forward(inputs);
        runtimeSyncAndCheck();
        EXPECT_FALSE(output.success.cpu()[0].item<bool>());
        EXPECT_EQ(output.token_ids[0][0].item<int32_t>(), -1);
        EXPECT_EQ(output.cum_log_probs[0].item<float>(), -std::numeric_limits<float>::infinity());
    }
}

TEST_P(RocmSamplerInvalidProbTest, ValidatesTailAfterEarlySampleAndFinalizesOnlyFailedRows) {
    const auto [top_k, top_p] = GetParam();
    const float nan           = std::numeric_limits<float>::quiet_NaN();
    const float inf           = std::numeric_limits<float>::infinity();
    const auto  gpu           = torch::TensorOptions().device(torch::kCUDA);
    const auto  stream        = reinterpret_cast<uintptr_t>(at::hip::getCurrentHIPStream().stream());
    for (int64_t vocab : {1025, 32768, 248320}) {
        SCOPED_TRACE(vocab);
        auto host_probs = torch::zeros({8, vocab}, torch::kFloat32);
        host_probs[0][0].fill_(1.0f);
        host_probs[1][vocab - 1].fill_(1.0f);
        // Row 2 has no positive mass. Rows 3-5 sample from the first tile, but
        // must still reject an invalid probability in the last tile.
        for (int64_t row = 3; row < 6; ++row) {
            host_probs[row][0].fill_(1.0f);
        }
        host_probs[3][vocab - 1].fill_(nan);
        host_probs[4][vocab - 1].fill_(inf);
        host_probs[5][vocab - 1].fill_(-0.1f);
        host_probs[6].fill_(nan);
        host_probs[7].fill_(-inf);
        auto probs   = host_probs.to(torch::kCUDA);
        auto samples = torch::empty({8}, gpu.dtype(torch::kInt32));
        auto success = torch::empty({8}, gpu.dtype(torch::kBool));
        auto seeds   = torch::arange(8, gpu.dtype(torch::kInt64)) + 14000;
        for (int round = 0; round < 3; ++round) {
            auto offsets = torch::full({8}, round * 32, gpu.dtype(torch::kInt64));
            samples.fill_(17);
            success.fill_(true);
            if (top_k == 0) {
                top_p_sampling_from_probs(
                    probs, samples, std::nullopt, std::nullopt, top_p, true, seeds, offsets, stream, success);
            } else if (top_p == 1.0f) {
                top_k_sampling_from_probs(
                    probs, samples, std::nullopt, std::nullopt, top_k, true, seeds, offsets, stream, success);
            } else {
                top_k_top_p_sampling_from_probs(probs,
                                                samples,
                                                std::nullopt,
                                                std::nullopt,
                                                top_k,
                                                std::nullopt,
                                                top_p,
                                                true,
                                                seeds,
                                                offsets,
                                                stream,
                                                success);
            }
            auto final_probs = probs.clone();
            auto logs        = torch::empty({8}, gpu.dtype(torch::kFloat32));
            finalize_sampling_probs(final_probs, samples, success, logs, stream);
            runtimeSyncAndCheck();
            auto tokens     = samples.cpu();
            auto status     = success.cpu();
            auto final_host = final_probs.cpu();
            auto log_host   = logs.cpu();
            for (int64_t row = 0; row < 8; ++row) {
                const bool valid = row < 2;
                EXPECT_EQ(status[row].item<bool>(), valid);
                EXPECT_EQ(tokens[row].item<int32_t>(), valid ? (row == 0 ? 0 : vocab - 1) : -1);
                EXPECT_EQ(log_host[row].item<float>(), valid ? 0.0f : -inf);
                if (valid) {
                    EXPECT_TRUE(torch::equal(final_host[row], host_probs[row]));
                } else {
                    EXPECT_TRUE(final_host[row].eq(0).all().item<bool>());
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(AllDispatches,
                         RocmSamplerInvalidProbTest,
                         ::testing::Values(std::make_pair(0, 0.75f),
                                           std::make_pair(4, 1.0f),
                                           std::make_pair(4, 0.75f)));

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
