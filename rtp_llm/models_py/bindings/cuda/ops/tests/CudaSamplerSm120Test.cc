#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/split_softmax.h"
#include "rtp_llm/models_py/bindings/common/kernels/sampling_penalty_kernels.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/ops/tests/CudaTestUtils.h"
#include "rtp_llm/models_py/bindings/common/kernels/banRepeatNgram.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <optional>
#include <limits>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
using namespace rtp_llm;

class CudaSamplerSm120Test: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        const auto* properties = at::cuda::getCurrentDeviceProperties();
        ASSERT_EQ(properties->major, 12) << "requires the SM120 execution pool";
        ASSERT_EQ(properties->minor, 0);
    }
};

TEST_F(CudaSamplerSm120Test, testSplitSoftmaxPreservesSeededTokensAndRngWithCloseProbabilityOutputs) {
    const char* previous = std::getenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX");
    const auto  saved = previous == nullptr ? std::optional<std::string>() : std::make_optional(std::string(previous));
    struct RestoreFlag {
        std::optional<std::string> previous;
        ~RestoreFlag() {
            if (previous.has_value())
                setenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX", previous->c_str(), 1);
            else
                unsetenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX");
        }
    } restore{saved};
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::manual_seed(20261019);
    for (int64_t batch : {1, 4, 17, 32}) {
        for (int64_t vocab : {257, 248320}) {
            for (float gain : {0.0f, 1.0f, 8.0f}) {
                auto logits = torch::randn({batch, vocab}, options) * gain;
                logits.slice(1, vocab - 13).fill_(-std::numeric_limits<float>::infinity());
                auto lengths = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
                auto top_p   = torch::ones({batch}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
                auto top_k   = torch::zeros_like(lengths);
                auto temperature = torch::ones_like(top_p);
                std::vector<torch::Tensor> samples, probabilities, log_probabilities, states, in_place;
                for (const char* flag : {"0", "1"}) {
                    setenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX", flag, 1);
                    std::vector<at::Generator> generators;
                    for (int64_t i = 0; i < batch; ++i) {
                        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                        generator.set_current_seed(210000 + i);
                        generators.push_back(generator);
                    }
                    for (int generation = 0; generation < 4; ++generation) {
                        auto tokens = torch::zeros({batch, 3}, options.dtype(torch::kInt32));
                        tokens.select(1, 0).fill_(3);
                        tokens.select(1, 1).fill_(5);
                        auto         probs   = torch::empty_like(logits);
                        auto         cum     = torch::full({batch}, -2.0, options);
                        auto         mutated = logits.clone();
                        GreedyParams params({mutated,
                                             lengths,
                                             lengths,
                                             tokens,
                                             2,
                                             top_k,
                                             top_p,
                                             temperature,
                                             std::nullopt,
                                             std::nullopt,
                                             cum,
                                             std::nullopt,
                                             false,
                                             probs,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             generators});
                        auto         result = execSampleGreedy(params);
                        check_cuda_error();
                        ASSERT_TRUE(result.success.all().item<bool>());
                        if (flag[0] == '0') {
                            samples.push_back(tokens.clone());
                            probabilities.push_back(probs.clone());
                            log_probabilities.push_back(cum.clone());
                            in_place.push_back(mutated.clone());
                            for (auto& generator : generators)
                                states.push_back(generator.get_state());
                        } else {
                            ASSERT_TRUE(torch::equal(tokens, samples[generation]));
                            ASSERT_TRUE(torch::allclose(probs, probabilities[generation], 3e-6, 1e-8));
                            ASSERT_TRUE(torch::allclose(mutated, in_place[generation], 3e-6, 1e-8));
                            ASSERT_TRUE(torch::allclose(cum, log_probabilities[generation], 3e-6, 1e-6));
                            for (int64_t i = 0; i < batch; ++i)
                                ASSERT_TRUE(torch::equal(generators[i].get_state(), states[generation * batch + i]));
                        }
                    }
                }
            }
        }
    }
}

TEST_F(CudaSamplerSm120Test, testSplitPointMassRejectionMatchesOriginalForAllAcceptancePaths) {
    const char* previous = std::getenv("RTP_LLM_SPLIT_REJECTION");
    const auto  saved = previous == nullptr ? std::optional<std::string>() : std::make_optional(std::string(previous));
    struct RestoreFlag {
        std::optional<std::string> previous;
        ~RestoreFlag() {
            if (previous.has_value())
                setenv("RTP_LLM_SPLIT_REJECTION", previous->c_str(), 1);
            else
                unsetenv("RTP_LLM_SPLIT_REJECTION");
        }
    } restore{saved};
    torch::manual_seed(20261019);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    for (int64_t batch : {1, 3, 8}) {
        for (int64_t stride : {1, 5}) {
            auto probs        = torch::softmax(torch::randn({batch, 4, 248320}, options), -1);
            auto draft        = torch::randint(0, 248320, {batch, 3}, options.dtype(torch::kInt32));
            auto target       = torch::zeros({batch * 4, stride}, options.dtype(torch::kInt32));
            auto expected     = torch::empty({batch, 4}, options.dtype(torch::kInt32));
            auto actual       = torch::empty_like(expected);
            auto expected_len = torch::empty({batch}, options.dtype(torch::kInt32));
            auto actual_len   = torch::empty_like(expected_len);
            for (int generation = 0; generation < 24; ++generation) {
                const int pos     = generation % 4;
                auto      uniform = torch::rand({batch, 4}, options);
                uniform.slice(1, 0, pos).zero_();
                if (pos < 3)
                    uniform.select(1, pos).fill_(1.0f);
                if (generation % 8 == 0)
                    uniform.select(1, std::min(pos + 1, 3)).zero_();
                if (generation % 8 == 1)
                    uniform.select(1, std::min(pos + 1, 3)).fill_(1.0f);
                auto do_sample = torch::ones({batch}, options.dtype(torch::kBool));
                if (generation % 3 == 0)
                    do_sample.select(0, 0).fill_(false);
                target.select(1, stride - 1).copy_(torch::argmax(probs, -1).flatten().to(torch::kInt32));
                for (int i = 0; i < pos; ++i)
                    target.select(1, stride - 1).reshape({batch, 4}).select(1, i).copy_(draft.select(1, i));
                for (int variant = 0; variant < 2; ++variant) {
                    setenv("RTP_LLM_SPLIT_REJECTION", variant ? "1" : "0", 1);
                    auto output  = variant ? actual : expected;
                    auto lengths = variant ? actual_len : expected_len;
                    output.fill_(-999);
                    lengths.fill_(-999);
                    execRejectionSampling(
                        {torch::Tensor(), draft, uniform, probs, target, output, lengths, do_sample, true});
                }
                ASSERT_TRUE(torch::equal(actual, expected))
                    << "batch=" << batch << " stride=" << stride << " generation=" << generation;
                ASSERT_TRUE(torch::equal(actual_len, expected_len));
            }
            // Zero residual mass follows the original last-vocabulary fallback.
            probs.zero_();
            auto uniform    = torch::ones({batch, 4}, options);
            auto stochastic = torch::ones({batch}, options.dtype(torch::kBool));
            setenv("RTP_LLM_SPLIT_REJECTION", "0", 1);
            execRejectionSampling(
                {torch::Tensor(), draft, uniform, probs, target, expected, expected_len, stochastic, true});
            setenv("RTP_LLM_SPLIT_REJECTION", "1", 1);
            execRejectionSampling(
                {torch::Tensor(), draft, uniform, probs, target, actual, actual_len, stochastic, true});
            ASSERT_TRUE(torch::equal(actual, expected));
            ASSERT_TRUE(torch::equal(actual_len, expected_len));
        }
    }
}

TEST_F(CudaSamplerSm120Test, testSplitDenseRejectionMatchesOriginalForAllAcceptancePaths) {
    const char* previous = std::getenv("RTP_LLM_SPLIT_REJECTION");
    const auto  saved = previous == nullptr ? std::optional<std::string>() : std::make_optional(std::string(previous));
    struct RestoreFlag {
        std::optional<std::string> previous;
        ~RestoreFlag() {
            if (previous.has_value())
                setenv("RTP_LLM_SPLIT_REJECTION", previous->c_str(), 1);
            else
                unsetenv("RTP_LLM_SPLIT_REJECTION");
        }
    } restore{saved};
    torch::manual_seed(20261019);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    for (int64_t batch : {1, 3, 8}) {
        for (int64_t stride : {1, 5}) {
            auto probs       = torch::softmax(torch::randn({batch, 4, 248320}, options), -1);
            auto draft_probs = torch::softmax(torch::randn({batch, 3, 248320}, options), -1);
            auto draft       = torch::argmax(draft_probs, -1).to(torch::kInt32);
            // Dense q differs from p throughout the vocabulary. Set a known
            // rejection ratio at each proposed token while retaining normalized rows.
            auto draft_index = draft.to(torch::kLong).unsqueeze(-1);
            probs.slice(1, 0, 3).scatter_(2, draft_index, draft_probs.gather(2, draft_index) * 0.125);
            probs.div_(probs.sum(-1, true));
            auto target       = torch::zeros({batch * 4, stride}, options.dtype(torch::kInt32));
            auto expected     = torch::empty({batch, 4}, options.dtype(torch::kInt32));
            auto actual       = torch::empty_like(expected);
            auto expected_len = torch::empty({batch}, options.dtype(torch::kInt32));
            auto actual_len   = torch::empty_like(expected_len);
            for (int generation = 0; generation < 24; ++generation) {
                const int pos     = generation % 4;
                auto      uniform = torch::rand({batch, 4}, options);
                uniform.slice(1, 0, pos).zero_();
                if (pos < 3)
                    uniform.select(1, pos).fill_(1.0f);
                if (generation % 8 == 0)
                    uniform.select(1, std::min(pos + 1, 3)).zero_();
                if (generation % 8 == 1)
                    uniform.select(1, std::min(pos + 1, 3)).fill_(1.0f);
                auto do_sample = torch::ones({batch}, options.dtype(torch::kBool));
                if (generation % 3 == 0)
                    do_sample.select(0, 0).fill_(false);
                target.select(1, stride - 1).copy_(torch::argmax(probs, -1).flatten().to(torch::kInt32));
                for (int i = 0; i < pos; ++i)
                    target.select(1, stride - 1).reshape({batch, 4}).select(1, i).copy_(draft.select(1, i));
                for (int variant = 0; variant < 2; ++variant) {
                    setenv("RTP_LLM_SPLIT_REJECTION", variant ? "1" : "0", 1);
                    auto output  = variant ? actual : expected;
                    auto lengths = variant ? actual_len : expected_len;
                    output.fill_(-999);
                    lengths.fill_(-999);
                    execRejectionSampling(
                        {draft_probs, draft, uniform, probs, target, output, lengths, do_sample, false});
                }
                ASSERT_TRUE(torch::equal(actual, expected))
                    << "batch=" << batch << " stride=" << stride << " generation=" << generation;
                ASSERT_TRUE(torch::equal(actual_len, expected_len));
            }
            // Zero residual mass follows the original last-vocabulary fallback.
            probs.zero_();
            auto uniform    = torch::ones({batch, 4}, options);
            auto stochastic = torch::ones({batch}, options.dtype(torch::kBool));
            setenv("RTP_LLM_SPLIT_REJECTION", "0", 1);
            execRejectionSampling(
                {draft_probs, draft, uniform, probs, target, expected, expected_len, stochastic, false});
            setenv("RTP_LLM_SPLIT_REJECTION", "1", 1);
            execRejectionSampling({draft_probs, draft, uniform, probs, target, actual, actual_len, stochastic, false});
            ASSERT_TRUE(torch::equal(actual, expected));
            ASSERT_TRUE(torch::equal(actual_len, expected_len));
        }
    }
}

TEST_F(CudaSamplerSm120Test, testTiledTemperaturePreservesSeededTokensProbabilitiesAndRng) {
    const char* previous = std::getenv("RTP_LLM_SAMPLER_TILED_TEMPERATURE");
    const auto  saved = previous == nullptr ? std::optional<std::string>() : std::make_optional(std::string(previous));
    struct RestoreFlag {
        std::optional<std::string> previous;
        ~RestoreFlag() {
            if (previous.has_value())
                setenv("RTP_LLM_SAMPLER_TILED_TEMPERATURE", previous->c_str(), 1);
            else
                unsetenv("RTP_LLM_SAMPLER_TILED_TEMPERATURE");
        }
    } restore{saved};
    const char* softmax_flag = std::getenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX");
    struct RestoreSoftmax {
        std::optional<std::string> flag;
        ~RestoreSoftmax() {
            if (flag)
                setenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX", flag->c_str(), 1);
            else
                unsetenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX");
        }
    } restore_softmax{softmax_flag == nullptr ? std::optional<std::string>() :
                                                std::make_optional(std::string(softmax_flag))};
    setenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX", "1", 1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::manual_seed(20261019);
    for (int64_t batch : {1, 4, 17, 32}) {
        for (int64_t vocab : {257, 248320}) {
            for (float gain : {0.0f, 1.0f, 8.0f}) {
                auto logits = torch::randn({batch, vocab}, options) * gain;
                logits.slice(1, vocab - 13).fill_(-std::numeric_limits<float>::infinity());
                auto lengths = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
                auto top_p   = torch::ones({batch}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
                auto top_k   = torch::zeros_like(lengths);
                auto temperature = torch::full_like(top_p, 0.7);
                std::vector<torch::Tensor> samples, probabilities, log_probabilities, states, in_place;
                for (const char* flag : {"0", "1"}) {
                    setenv("RTP_LLM_SAMPLER_TILED_TEMPERATURE", flag, 1);
                    std::vector<at::Generator> generators;
                    for (int64_t i = 0; i < batch; ++i) {
                        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                        generator.set_current_seed(210000 + i);
                        generators.push_back(generator);
                    }
                    for (int generation = 0; generation < 4; ++generation) {
                        auto tokens = torch::zeros({batch, 3}, options.dtype(torch::kInt32));
                        tokens.select(1, 0).fill_(3);
                        tokens.select(1, 1).fill_(5);
                        auto         probs   = torch::empty_like(logits);
                        auto         cum     = torch::full({batch}, -2.0, options);
                        auto         mutated = logits.clone();
                        GreedyParams params({mutated,
                                             lengths,
                                             lengths,
                                             tokens,
                                             2,
                                             top_k,
                                             top_p,
                                             temperature,
                                             std::nullopt,
                                             std::nullopt,
                                             cum,
                                             std::nullopt,
                                             false,
                                             probs,
                                             std::nullopt,
                                             std::nullopt,
                                             std::nullopt,
                                             generators});
                        auto         result = execSampleGreedy(params);
                        check_cuda_error();
                        ASSERT_TRUE(result.success.all().item<bool>());
                        if (flag[0] == '0') {
                            samples.push_back(tokens.clone());
                            probabilities.push_back(probs.clone());
                            log_probabilities.push_back(cum.clone());
                            in_place.push_back(mutated.clone());
                            for (auto& generator : generators)
                                states.push_back(generator.get_state());
                        } else {
                            ASSERT_TRUE(torch::equal(tokens, samples[generation]));
                            ASSERT_TRUE(torch::equal(probs, probabilities[generation]));
                            ASSERT_TRUE(torch::equal(mutated, in_place[generation]));
                            ASSERT_TRUE(torch::equal(cum, log_probabilities[generation]));
                            for (int64_t i = 0; i < batch; ++i)
                                ASSERT_TRUE(torch::equal(generators[i].get_state(), states[generation * batch + i]));
                        }
                    }
                }
            }
        }
    }
}

TEST_F(CudaSamplerSm120Test, testCompactOutputPreservesTokensProbabilitiesRngAndPenaltyFallback) {
    const char* previous = std::getenv("RTP_LLM_SAMPLER_COMPACT_OUTPUT");
    const auto  saved = previous == nullptr ? std::optional<std::string>() : std::make_optional(std::string(previous));
    struct RestoreFlag {
        std::optional<std::string> previous;
        ~RestoreFlag() {
            if (previous.has_value())
                setenv("RTP_LLM_SAMPLER_COMPACT_OUTPUT", previous->c_str(), 1);
            else
                unsetenv("RTP_LLM_SAMPLER_COMPACT_OUTPUT");
        }
    } restore{saved};
    const char* softmax_flag = std::getenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX");
    struct RestoreSoftmax {
        std::optional<std::string> flag;
        ~RestoreSoftmax() {
            if (flag)
                setenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX", flag->c_str(), 1);
            else
                unsetenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX");
        }
    } restore_softmax{softmax_flag == nullptr ? std::optional<std::string>() :
                                                std::make_optional(std::string(softmax_flag))};
    setenv("RTP_LLM_SAMPLER_SPLIT_SOFTMAX", "1", 1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::manual_seed(20261019);
    for (int64_t batch : {1, 4, 17, 32}) {
        for (int64_t vocab : {257, 248320}) {
            for (int penalty_case = 0; penalty_case < 6; ++penalty_case) {
                const float gain   = penalty_case == 0 ? 0.0f : 1.0f;
                auto        logits = torch::randn({batch, vocab}, options) * gain;
                logits.slice(1, vocab - 13).fill_(-std::numeric_limits<float>::infinity());
                auto lengths =
                    torch::full({batch}, 128, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
                auto top_p = torch::ones({batch}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
                auto top_k = torch::zeros_like(lengths);
                auto temperature = torch::full_like(top_p, 0.7);
                auto repetition  = torch::ones_like(top_p);
                auto presence    = torch::zeros_like(top_p);
                auto frequency   = torch::zeros_like(top_p);
                auto ngram       = torch::zeros_like(lengths);
                if (penalty_case == 2)
                    repetition.fill_(1.2);
                if (penalty_case == 3)
                    presence.fill_(0.2);
                if (penalty_case == 4)
                    frequency.fill_(0.3);
                if (penalty_case == 5)
                    ngram.fill_(2);
                std::vector<torch::Tensor> samples, probabilities, log_probabilities, states, in_place;
                for (const char* flag : {"0", "1"}) {
                    setenv("RTP_LLM_SAMPLER_COMPACT_OUTPUT", flag, 1);
                    std::vector<at::Generator> generators;
                    for (int64_t i = 0; i < batch; ++i) {
                        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
                        generator.set_current_seed(210000 + i);
                        generators.push_back(generator);
                    }
                    for (int generation = 0; generation < 4; ++generation) {
                        auto tokens = torch::zeros({batch, 129}, options.dtype(torch::kInt32));
                        tokens.select(1, 0).fill_(3);
                        tokens.select(1, 1).fill_(5);
                        tokens.select(1, 128).fill_(17);
                        auto         probs   = torch::empty_like(logits);
                        auto         cum     = torch::full({batch}, -2.0, options);
                        auto         mutated = logits.clone();
                        GreedyParams params({mutated,
                                             lengths,
                                             lengths,
                                             tokens,
                                             128,
                                             top_k,
                                             top_p,
                                             temperature,
                                             penalty_case == 0 ? std::nullopt : std::make_optional(repetition),
                                             ngram,
                                             cum,
                                             std::nullopt,
                                             false,
                                             probs,
                                             presence,
                                             frequency,
                                             std::nullopt,
                                             generators});
                        auto         result = execSampleGreedy(params);
                        check_cuda_error();
                        ASSERT_TRUE(result.success.all().item<bool>());
                        if (flag[0] == '0') {
                            samples.push_back(tokens.clone());
                            probabilities.push_back(probs.clone());
                            log_probabilities.push_back(cum.clone());
                            in_place.push_back(mutated.clone());
                            for (auto& generator : generators)
                                states.push_back(generator.get_state());
                        } else {
                            ASSERT_TRUE(torch::equal(tokens, samples[generation]));
                            ASSERT_TRUE(torch::equal(probs, probabilities[generation]));
                            ASSERT_TRUE(torch::equal(mutated, in_place[generation]));
                            ASSERT_TRUE(torch::equal(cum, log_probabilities[generation]));
                            for (int64_t i = 0; i < batch; ++i)
                                ASSERT_TRUE(torch::equal(generators[i].get_state(), states[generation * batch + i]));
                        }
                    }
                }
            }
        }
    }
}
