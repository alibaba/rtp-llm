#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

#include <gtest/gtest.h>
#include <torch/all.h>
#include <array>
#include <mutex>

#if USING_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/models_py/bindings/core/TensorHolder.h"

#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#include "rtp_llm/cpp/models/Sampler.h"
#endif

#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm::speculative {
namespace {

#if USING_CUDA
// Keep this test's RNG manipulation isolated from other sampler tests.
struct CudaGeneratorStateGuard {
    at::Generator generator = at::cuda::detail::getDefaultCUDAGenerator();
    torch::Tensor original;

    CudaGeneratorStateGuard(): original(state()) {}
    ~CudaGeneratorStateGuard() {
        restore(original);
    }

    torch::Tensor state() {
        std::lock_guard<std::mutex> lock(generator.mutex());
        return generator.get_state().clone();
    }
    void restore(const torch::Tensor& value) {
        std::lock_guard<std::mutex> lock(generator.mutex());
        generator.set_state(value);
    }
};

// Test-only metadata scaffolding. The staging bodies and cache members below
// are extracted verbatim from the frozen baseline/candidate MtpExecutor.
class DSparkTemperatureStagingFixture {
public:
    struct Config {
        float temperature;
        bool  greedy;
        bool  top1() const {
            return greedy;
        }
    };
    struct Request {
        Config config;
        int    maxBatchSize() const {
            return 1;
        }
        bool hasNumBeams() const {
            return false;
        }
        int64_t streamId() const {
            return 0;
        }
        const Config* generateConfig() const {
            return &config;
        }
    };
    struct StreamGroups {
        std::vector<std::shared_ptr<Request>> requests;
        size_t                                size() const {
            return requests.size();
        }
        const auto& allStreams() const {
            return requests;
        }
    };

    bool               is_dspark_ = true;
    TensorHolder       buffer_holder_;
    std::vector<float> dspark_temperatures_;
    torch::Tensor      dspark_temperature_gpu_;
    int64_t            dspark_temperature_stream_id_ = -1;

    torch::Tensor fresh(const StreamGroups& stream_groups, const torch::Tensor& base_logits) {

        RTP_LLM_CHECK_WITH_INFO(is_dspark_, "DSpARK draft sampling requires SP_TYPE_DSPARK");
        const auto batch_size = static_cast<int64_t>(stream_groups.size());
        auto       temperature_cpu =
            torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
        auto*           temperatures         = temperature_cpu.data_ptr<float>();
        int64_t         row                  = 0;
        constexpr float kMinDraftTemperature = 1.0e-6f;
        for (const auto& stream : stream_groups.allStreams()) {
            RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1 && !stream->hasNumBeams(),
                                    "DSpARK does not support tiled or beam sampling");
            const auto config      = stream->generateConfig();
            float      temperature = config->temperature;
            if (!std::isfinite(temperature) || temperature < 0.0f) {
                RTP_LLM_LOG_WARNING("DSpARK received invalid sampling temperature=%g for stream=%ld; using 1.0",
                                    temperature,
                                    stream->streamId());
                temperature = 1.0f;
            }
            temperatures[row++] = !config->top1() ? std::max(temperature, kMinDraftTemperature) : kMinDraftTemperature;
        }
        buffer_holder_.hold_host(temperature_cpu);
        auto temperature = temperature_cpu.to(base_logits.device(), /*non_blocking=*/true);

        return temperature;
    }
    torch::Tensor cached(const StreamGroups& stream_groups, const torch::Tensor& base_logits) {

        RTP_LLM_CHECK_WITH_INFO(is_dspark_, "DSpARK draft sampling requires SP_TYPE_DSPARK");
        const auto         batch_size = static_cast<int64_t>(stream_groups.size());
        std::vector<float> temperatures(batch_size);
        int64_t            row                  = 0;
        constexpr float    kMinDraftTemperature = 1.0e-6f;
        for (const auto& stream : stream_groups.allStreams()) {
            RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1 && !stream->hasNumBeams(),
                                    "DSpARK does not support tiled or beam sampling");
            const auto config      = stream->generateConfig();
            float      temperature = config->temperature;
            if (!std::isfinite(temperature) || temperature < 0.0f) {
                RTP_LLM_LOG_WARNING("DSpARK received invalid sampling temperature=%g for stream=%ld; using 1.0",
                                    temperature,
                                    stream->streamId());
                temperature = 1.0f;
            }
            temperatures[row++] = !config->top1() ? std::max(temperature, kMinDraftTemperature) : kMinDraftTemperature;
        }
        // Reuse immutable values only on their producing stream. A miss replaces
        // the tensors instead of overwriting storage still used by async sampling.
        const auto sampling_stream_id = cuda_graph::graphGetCurrentStream().id();
        if (!dspark_temperature_gpu_.defined() || dspark_temperature_gpu_.device() != base_logits.device()
            || dspark_temperature_stream_id_ != sampling_stream_id || dspark_temperatures_ != temperatures) {
            auto temperature_cpu =
                torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
            std::copy(temperatures.begin(), temperatures.end(), temperature_cpu.data_ptr<float>());
            buffer_holder_.hold_host(temperature_cpu);
            auto temperature_gpu          = temperature_cpu.to(base_logits.device(), /*non_blocking=*/true);
            dspark_temperature_gpu_       = std::move(temperature_gpu);
            dspark_temperatures_          = std::move(temperatures);
            dspark_temperature_stream_id_ = sampling_stream_id;
        }

        return dspark_temperature_gpu_;
    }
};

TEST(DSparkSamplerTest, CachedTemperatureMatchesFreshSamplingAndCudaRng) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required";
    }
    CudaGeneratorStateGuard rng;
    constexpr int64_t       gamma = 7, vocab = 31, padded_vocab = 32, markov_rank = 256;
    const auto              options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    // Contains unchanged hits, changed values, B16->1->16, and non-power-of-two
    // live row counts. It does not claim to execute a padded model CUDA Graph.
    const std::vector<int64_t> batches = {16, 16, 16, 16, 16, 1, 16, 3, 3, 5, 9, 15, 16};
    for (const int64_t verify : {4, 7}) {
        for (const bool mapped : {false, true}) {
            for (const bool greedy : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << "verify=" << verify << " mapped=" << mapped << " greedy=" << greedy);
                torch::manual_seed(20260927);
                const int64_t target_vocab = mapped ? 2 * vocab + 1 : vocab;
                auto          w1 = (torch::randn({target_vocab, markov_rank}, options) * 0.1f).to(torch::kBFloat16);
                auto          w2 = (torch::randn({vocab, markov_rank}, options) * 0.1f).to(torch::kBFloat16);
                auto map = mapped ? torch::arange(vocab, options.dtype(torch::kInt64)) * 2 + 1 : torch::Tensor();
                auto original_w1 = w1.clone(), original_w2 = w2.clone();
                auto original_map = map.defined() ? map.clone() : torch::Tensor();
                std::vector<torch::Tensor> logits, original_logits, initial_anchors, original_anchors, next_probs;
                std::vector<DSparkTemperatureStagingFixture::StreamGroups> groups;
                for (size_t round = 0; round < batches.size(); ++round) {
                    const auto batch  = batches[round];
                    auto       values = torch::randn({batch * gamma, padded_vocab}, options);
                    values.select(1, vocab).fill_(10000.0f);
                    logits.push_back(values);
                    original_logits.push_back(values.clone());
                    auto anchors = torch::arange(batch, options.dtype(torch::kInt32));
                    if (mapped)
                        anchors = anchors * 2 + 1;
                    initial_anchors.push_back(anchors);
                    original_anchors.push_back(anchors.clone());
                    next_probs.push_back(torch::softmax(torch::randn({batch, vocab}, options), -1));
                    DSparkTemperatureStagingFixture::StreamGroups current;
                    for (int64_t row = 0; row < batch; ++row) {
                        float temperature = (row % 2) ? 0.7f : 1.0f;
                        if (round == 2 || round == 3) {
                            if (row == 0)
                                temperature = std::nextafter(1.0f, 2.0f);
                        }
                        if (round == 4)
                            temperature = (row % 2) ? 1.0f : 0.7f;  // order/value change
                        if (round == 7 || round == 8) {
                            temperature = row == 0 ? std::numeric_limits<float>::quiet_NaN() : row == 1 ? -1.0f : 0.0f;
                        }
                        current.requests.push_back(std::make_shared<DSparkTemperatureStagingFixture::Request>(
                            DSparkTemperatureStagingFixture::Request{{temperature, greedy}}));
                    }
                    groups.push_back(std::move(current));
                }
                SpeculativeSampler sampler(map, gamma, DraftProposalMode::SAMPLED);
                // Randomized fixtures precede the common RNG snapshot.
                auto                            initial_state = rng.state();
                DSparkTemperatureStagingFixture fresh_stager, cached_stager;
                std::vector<torch::Tensor>      tokens, probabilities, subsequent_samples, draws, states;
                std::vector<torch::Tensor>      fresh_temperatures, temperature_snapshots, cached_temperatures;
                torch::Tensor                   previous_tokens;
                for (size_t round = 0; round < batches.size(); ++round) {
                    SCOPED_TRACE(::testing::Message() << "fresh round=" << round);
                    const auto batch        = batches[round];
                    auto       before_stage = rng.state();
                    auto       temperature  = fresh_stager.fresh(groups[round], logits[round]);
                    EXPECT_TRUE(torch::equal(rng.state(), before_stage));
                    fresh_temperatures.push_back(temperature);
                    temperature_snapshots.push_back(temperature.clone());
                    auto anchors        = previous_tokens.defined() && previous_tokens.size(0) == batch ?
                                              previous_tokens.select(1, verify - 1).contiguous() :
                                              initial_anchors[round];
                    auto anchors_before = anchors.clone();
                    auto output = sampler.sampleDSparkDraft(logits[round], anchors, temperature, w1, w2, vocab, verify);
                    tokens.push_back(output.token_ids);
                    probabilities.push_back(output.all_probs);
                    subsequent_samples.push_back(execSampleFromProbs(next_probs[round]));
                    draws.push_back(torch::rand({batch, 13}, options));
                    states.push_back(rng.state());
                    EXPECT_TRUE(torch::equal(anchors, anchors_before));
                    previous_tokens = output.token_ids;
                    fresh_stager.buffer_holder_.release();
                }
                rng.restore(initial_state);
                previous_tokens = torch::Tensor();
                size_t hits = 0, misses = 0;
                for (size_t round = 0; round < batches.size(); ++round) {
                    SCOPED_TRACE(::testing::Message() << "cached round=" << round);
                    const auto batch        = batches[round];
                    auto       before_stage = rng.state();
                    auto       temperature  = cached_stager.cached(groups[round], logits[round]);
                    EXPECT_TRUE(torch::equal(rng.state(), before_stage));
                    ASSERT_EQ(temperature.sizes(), torch::IntArrayRef({batch}));
                    ASSERT_TRUE(torch::equal(temperature, temperature_snapshots[round]));
                    if (round > 0 && batches[round - 1] == batch
                        && torch::equal(temperature_snapshots[round - 1], temperature_snapshots[round])) {
                        ++hits;
                        EXPECT_EQ(temperature.data_ptr(), cached_temperatures.back().data_ptr());
                    } else {
                        ++misses;
                        if (round > 0) {
                            EXPECT_NE(temperature.data_ptr(), cached_temperatures.back().data_ptr());
                        }
                    }
                    cached_temperatures.push_back(temperature);
                    auto anchors        = previous_tokens.defined() && previous_tokens.size(0) == batch ?
                                              previous_tokens.select(1, verify - 1).contiguous() :
                                              initial_anchors[round];
                    auto anchors_before = anchors.clone();
                    auto output = sampler.sampleDSparkDraft(logits[round], anchors, temperature, w1, w2, vocab, verify);
                    EXPECT_TRUE(torch::equal(output.token_ids, tokens[round]));
                    EXPECT_TRUE(torch::equal(output.all_probs, probabilities[round]));
                    EXPECT_TRUE(torch::equal(execSampleFromProbs(next_probs[round]), subsequent_samples[round]));
                    EXPECT_TRUE(torch::equal(torch::rand({batch, 13}, options), draws[round]));
                    EXPECT_TRUE(torch::equal(rng.state(), states[round]));
                    EXPECT_TRUE(torch::equal(anchors, anchors_before));
                    previous_tokens = output.token_ids;
                    cached_stager.buffer_holder_.release();
                }
                EXPECT_GE(hits, 3u);
                EXPECT_GE(misses, 5u);
                for (size_t round = 0; round < batches.size(); ++round) {
                    EXPECT_TRUE(torch::equal(logits[round], original_logits[round]));
                    EXPECT_TRUE(torch::equal(initial_anchors[round], original_anchors[round]));
                    EXPECT_TRUE(torch::equal(fresh_temperatures[round], temperature_snapshots[round]));
                    EXPECT_TRUE(torch::equal(cached_temperatures[round], temperature_snapshots[round]));
                }
                EXPECT_TRUE(torch::equal(w1, original_w1));
                EXPECT_TRUE(torch::equal(w2, original_w2));
                if (mapped) {
                    EXPECT_TRUE(torch::equal(map, original_map));
                }
            }
        }
    }
}

TEST(DSparkSamplerTest, FusedLogitsMatchLegacyBitsIncludingPaddedViewsAndEdges) {
    if (!torch::cuda::is_available())
        GTEST_SKIP();
    const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    for (int64_t batch : {0, 1, 3, 16}) {
        for (int64_t vocab : {257, 200064}) {
            for (auto dtype : {torch::kBFloat16, torch::kFloat32, torch::kFloat16}) {
                SCOPED_TRACE(::testing::Message() << batch << "/" << vocab << "/" << dtype);
                auto storage     = torch::randn({batch, 7, vocab + 64}, options);
                auto base        = storage.select(1, 3).narrow(1, 0, vocab);
                auto bias        = torch::randn({batch, vocab}, options).to(dtype);
                auto temperature = torch::linspace(0.3f, 1.7f, batch, options);
                if (batch)
                    temperature.select(0, 0).fill_(1e-6f);
                base.narrow(1, 0, 16).copy_(-bias.narrow(1, 0, 16).to(torch::kFloat32));
                auto saved    = storage.clone();
                auto expected = base + bias.to(torch::kFloat32);
                expected.div_(temperature.unsqueeze(1));
                auto actual = execDSparkCombineLogits(base, bias, temperature);
                EXPECT_TRUE(actual.is_contiguous());
                EXPECT_TRUE(torch::equal(actual.view(torch::kInt32), expected.view(torch::kInt32)));
                EXPECT_TRUE(torch::equal(storage, saved));
                EXPECT_TRUE(torch::equal(actual.softmax(-1), expected.softmax(-1)));
            }
        }
    }
    auto values = torch::tensor({0.f, -0.f, 1e-45f, -1e-45f, 1e-38f, -1e-38f, 1e-30f, -1e-30f, 3e38f, -3e38f}, options)
                      .repeat({3, 1});
    auto temperatures = torch::tensor({1e-6f, .3f, 1.3f}, options);
    for (auto dtype : {torch::kBFloat16, torch::kFloat32}) {
        auto bias = torch::zeros_like(values).to(dtype);
        bias.select(1, 1).fill_(-0.f);    // -0 + -0 must retain the sign bit.
        bias.select(1, 2).fill_(1e-40f);  // Nonzero BF16 subnormal bias.
        bias.select(1, 3).fill_(-1e-40f);
        values.select(1, 4).fill_(1.1754943508222875e-38f);
        bias.select(1, 4).fill_(-1.1663108012064884e-38f);
        auto expected = values + bias.to(torch::kFloat32);
        expected.div_(temperatures.unsqueeze(1));
        auto actual = execDSparkCombineLogits(values, bias, temperatures);
        EXPECT_TRUE(torch::equal(actual.view(torch::kInt32), expected.view(torch::kInt32)));
    }
}

TEST(DSparkSamplerTest, FusedLogitsGraphReplaysUpdatedInputsOnNonDefaultStream) {
    if (!torch::cuda::is_available())
        GTEST_SKIP();
    c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromPool());
    const auto                 options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    for (int64_t batch : {3, 16}) {
        constexpr int64_t vocab       = 200064;
        auto              storage     = torch::randn({batch, 7, vocab + 64}, options);
        auto              base        = storage.select(1, 3).narrow(1, 0, vocab);
        auto              bias        = torch::randn({batch, vocab}, options).to(torch::kBFloat16);
        auto              temperature = torch::ones({batch}, options);
        auto              legacy      = [&]() {
            auto value = base + bias.to(torch::kFloat32);
            return value.div_(temperature.unsqueeze(1));
        };
        auto fused = [&]() { return execDSparkCombineLogits(base, bias, temperature); };
        for (int i = 0; i < 5; ++i) {
            legacy();
            fused();
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        at::cuda::CUDAGraph legacy_graph, fused_graph;
        legacy_graph.capture_begin();
        auto expected = legacy();
        legacy_graph.capture_end();
        fused_graph.capture_begin();
        auto actual = fused();
        fused_graph.capture_end();
        auto* output_address = actual.data_ptr<float>();
        for (float temp : {1e-6f, .3f, 1.3f}) {
            base.add_(.125f);
            bias.mul_(.5f);
            temperature.fill_(temp);
            for (int replay = 0; replay < 20; ++replay)
                legacy_graph.replay();
            for (int replay = 0; replay < 20; ++replay)
                fused_graph.replay();
            EXPECT_EQ(actual.data_ptr<float>(), output_address);
            EXPECT_TRUE(torch::equal(actual.view(torch::kInt32), expected.view(torch::kInt32)));
            EXPECT_TRUE(torch::equal(actual.view(torch::kInt32), legacy().view(torch::kInt32)));
        }
    }
}

TEST(DSparkSamplerTest, FusedLogitsPreserveRecurrentSamplesAndRngWithSelectedSoftmax) {
    if (!torch::cuda::is_available())
        GTEST_SKIP();
    CudaGeneratorStateGuard rng;
    constexpr int64_t       gamma = 7, vocab = 257, rank = 256;
    auto                    options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    for (auto dtype : {torch::kBFloat16, torch::kFloat32}) {
        for (bool mapped : {false, true}) {
            auto               map = mapped ? torch::arange(vocab, options.dtype(torch::kInt64)) * 2 : torch::Tensor();
            auto               w1  = (torch::randn({vocab * 2, rank}, options) * .1f).to(dtype);
            auto               w2  = (torch::randn({vocab, rank}, options) * .1f).to(dtype);
            SpeculativeSampler sampler(map, gamma, DraftProposalMode::SAMPLED);
            // Exercise capacity changes without reinitializing the sampler.
            for (int64_t batch : {16, 1, 16, 3}) {
                for (int64_t k : {4, 5, 6, 7}) {
                    auto base        = torch::randn({batch * gamma, vocab + 64}, options);
                    auto temperature = torch::linspace(.3f, 1.5f, batch, options);
                    temperature.select(0, 0).fill_(1e-6f);
                    auto anchors    = torch::arange(batch, options.dtype(torch::kInt32));
                    auto next_probs = torch::randn({batch, vocab}, options).softmax(-1);
                    for (int round = 0; round < 3; ++round) {
                        auto                       before   = rng.state();
                        auto                       previous = anchors.to(torch::kInt64);
                        std::vector<torch::Tensor> ids, probabilities;
                        torch::Tensor softmax_workspace;
                        // Independent legacy Markov/temperature expression; do
                        // not call the fused logits helper. Softmax intentionally
                        // has a different reduction, tested against Torch below.
                        auto view = base.narrow(1, 0, vocab).view({batch, gamma, vocab});
                        for (int64_t step = 0; step < k; ++step) {
                            auto bias = torch::mm(w1.index_select(0, previous), w2.transpose(0, 1)).to(torch::kFloat32);
                            auto logits = view.select(1, step) + bias;
                            logits.div_(temperature.unsqueeze(1));
                            auto q     = execDSparkSoftmax(logits, softmax_workspace);
                            EXPECT_TRUE(torch::allclose(q, logits.softmax(-1), 2.e-5, 2.e-6));
                            auto token = execSampleFromProbs(q).to(torch::kInt32);
                            if (mapped)
                                token = map.index_select(0, token.to(torch::kInt64)).to(torch::kInt32);
                            probabilities.push_back(q);
                            ids.push_back(token);
                            previous = token.to(torch::kInt64);
                        }
                        execReserveSampleFromProbsRng(base, gamma - k);
                        auto expected_ids   = torch::stack(ids, 1);
                        auto expected_probs = torch::stack(probabilities, 1);
                        auto expected_next  = execSampleFromProbs(next_probs);
                        auto expected_rand  = torch::rand({batch, 13}, options);
                        auto expected_state = rng.state();
                        rng.restore(before);
                        auto actual = sampler.sampleDSparkDraft(base, anchors, temperature, w1, w2, vocab, k);
                        EXPECT_TRUE(torch::equal(actual.token_ids, expected_ids));
                        EXPECT_TRUE(
                            torch::equal(actual.all_probs.view(torch::kInt32), expected_probs.view(torch::kInt32)));
                        EXPECT_TRUE(torch::equal(execSampleFromProbs(next_probs), expected_next));
                        EXPECT_TRUE(torch::equal(torch::rand({batch, 13}, options), expected_rand));
                        EXPECT_TRUE(torch::equal(rng.state(), expected_state));
                        anchors = actual.token_ids.select(1, k - 1).contiguous();
                    }
                }
            }
        }
    }
}

TEST(DSparkSamplerTest, VerifiedPrefixPreservesMultiRoundProbabilitiesAndCudaRng) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required";
    }
    CudaGeneratorStateGuard rng;
    constexpr int64_t       gamma = 7, vocab = 31, padded_vocab = 32, rank = 256, rounds = 3;
    const auto              options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    for (const int64_t batch : {1, 3, 16}) {
        for (const int64_t k : {1, 4, 5, 6, 7}) {
            for (const bool mapped : {false, true}) {
                for (const bool greedy : {false, true}) {
                    SCOPED_TRACE(::testing::Message()
                                 << "B=" << batch << " k=" << k << " mapped=" << mapped << " greedy=" << greedy);
                    torch::manual_seed(20260926);
                    const int64_t target_vocab = mapped ? 2 * vocab + 1 : vocab;
                    auto          w1     = (torch::randn({target_vocab, rank}, options) * 0.1f).to(torch::kBFloat16);
                    auto          w2     = (torch::randn({vocab, rank}, options) * 0.1f).to(torch::kBFloat16);
                    auto          logits = torch::randn({batch * gamma, padded_vocab}, options);
                    logits.select(1, vocab).fill_(10000.0f);  // Padding must remain excluded.
                    auto original_logits = logits.clone();
                    auto map = mapped ? (torch::arange(vocab, options.dtype(torch::kInt64)) * 2 + 1) : torch::Tensor();
                    auto initial_anchors = torch::arange(batch, options.dtype(torch::kInt32));
                    if (mapped)
                        initial_anchors = initial_anchors * 2 + 1;
                    auto temperatures =
                        greedy ? torch::full({batch}, 1.0e-6f, options) : torch::linspace(0.3f, 1.5f, batch, options);
                    auto               next_probs = torch::softmax(torch::randn({batch, vocab}, options), -1);
                    SpeculativeSampler sampler(map, gamma, DraftProposalMode::SAMPLED);
                    // Allocate/randomize all fixture inputs before freezing RNG.
                    auto                       initial_state = rng.state();
                    std::vector<torch::Tensor> ids, probs, subsequent_samples, random_draws, states;
                    auto                       anchors = initial_anchors.clone();
                    for (int r = 0; r < rounds; ++r) {
                        auto full = sampler.sampleDSparkDraft(logits, anchors, temperatures, w1, w2, vocab);
                        ids.push_back(full.token_ids.narrow(1, 0, k).contiguous());
                        probs.push_back(full.all_probs.narrow(1, 0, k).contiguous());
                        subsequent_samples.push_back(execSampleFromProbs(next_probs));
                        random_draws.push_back(torch::rand({batch, 13}, options));
                        states.push_back(rng.state());
                        anchors = ids.back().select(1, k - 1).contiguous();
                    }
                    rng.restore(initial_state);
                    anchors = initial_anchors.clone();
                    for (int r = 0; r < rounds; ++r) {
                        auto prefix = sampler.sampleDSparkDraft(logits, anchors, temperatures, w1, w2, vocab, k);
                        ASSERT_EQ(prefix.token_ids.sizes(), torch::IntArrayRef({batch, k}));
                        ASSERT_EQ(prefix.all_probs.sizes(), torch::IntArrayRef({batch, k, vocab}));
                        EXPECT_TRUE(prefix.token_ids.is_contiguous());
                        EXPECT_TRUE(prefix.all_probs.is_contiguous());
                        EXPECT_TRUE(torch::equal(prefix.token_ids, ids[r]));
                        EXPECT_TRUE(torch::equal(prefix.all_probs, probs[r]));
                        EXPECT_TRUE(torch::equal(execSampleFromProbs(next_probs), subsequent_samples[r]));
                        EXPECT_TRUE(torch::equal(torch::rand({batch, 13}, options), random_draws[r]));
                        EXPECT_TRUE(torch::equal(rng.state(), states[r]));
                        anchors = prefix.token_ids.select(1, k - 1).contiguous();
                    }
                    EXPECT_TRUE(torch::equal(logits, original_logits));
                }
            }
        }
    }
}

TEST(DSparkSamplerTest, SkippedProbabilityDrawReservationMatchesRealCalls) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required";
    }
    CudaGeneratorStateGuard rng;
    auto                    probabilities =
        torch::full({3, 17}, 1.0f / 17, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
    auto before = rng.state();
    execReserveSampleFromProbsRng(probabilities, 0);
    EXPECT_TRUE(torch::equal(before, rng.state()));
    for (int i = 0; i < 3; ++i)
        execSampleFromProbs(probabilities);
    auto expected = rng.state();
    rng.restore(before);
    execReserveSampleFromProbsRng(probabilities, 3);
    EXPECT_TRUE(torch::equal(expected, rng.state()));
}

TEST(SpeculativeSamplerTest, ForceAcceptFlagsPreserveRejectionFailureAndRng) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required";
    }
    CudaGeneratorStateGuard rng;
    constexpr int64_t       batch = 3, steps = 2, vocab = 8;
    const auto              cpu_i32 = torch::TensorOptions().dtype(torch::kInt32);
    const auto              cpu_f32 = torch::TensorOptions().dtype(torch::kFloat32);
    for (const auto mode : {DraftProposalMode::LEGACY, DraftProposalMode::SAMPLED}) {
        for (const bool do_sample : {false, true}) {
            for (const bool fail_request : {false, true}) {
                if (fail_request && mode != DraftProposalMode::SAMPLED) {
                    continue;  // Legacy rejection does not expose the sampled-draft failure guard.
                }
                const auto    initial_rng = rng.state();
                torch::Tensor expected_rng;
                for (const int force_mode : {0, 1, 2}) {
                    SCOPED_TRACE(::testing::Message() << "mode=" << static_cast<int>(mode) << "sample=" << do_sample
                                                      << "failure=" << fail_request << "force=" << force_mode);
                    rng.restore(initial_rng);
                    ModelConfig model;
                    model.max_seq_len = 32;
                    model.vocab_size  = vocab;
                    RuntimeConfig                runtime;
                    ResourceContext              resources;
                    std::list<GenerateStreamPtr> streams;
                    for (int64_t b = 0; b < batch; ++b) {
                        auto request                              = std::make_shared<GenerateInput>();
                        request->input_ids                        = torch::tensor({0}, cpu_i32);
                        request->generate_config                  = std::make_shared<GenerateConfig>();
                        request->generate_config->top_k           = do_sample ? 0 : 1;
                        request->generate_config->force_sp_accept = force_mode == 1 || (force_mode == 2 && b != 2);
                        streams.push_back(
                            std::make_shared<NormalGenerateStream>(request, model, runtime, resources, nullptr));
                    }
                    SamplerOutput draft, target;
                    draft.token_ids  = torch::ones({batch, steps}, cpu_i32).cuda();
                    auto draft_probs = torch::zeros({batch, steps, vocab}, cpu_f32);
                    draft_probs.select(2, 1).fill_(1.f);
                    draft.all_probs = draft_probs.cuda();
                    auto target_ids = torch::full({batch, steps + 1}, 2, cpu_i32);
                    target_ids.select(1, steps).fill_(3);
                    target.token_ids  = target_ids.reshape({batch * (steps + 1), 1}).cuda();
                    auto target_probs = torch::zeros({batch, steps + 1, vocab}, cpu_f32);
                    target_probs.select(2, 2).fill_(1.f);
                    target_probs.select(1, steps).zero_();
                    target_probs.select(1, steps).select(1, 3).fill_(1.f);
                    target.all_probs    = target_probs.cuda();
                    auto target_success = torch::ones({batch, steps + 1}, torch::kBool);
                    if (fail_request) {
                        target_success.select(0, 1).fill_(false);
                    }
                    target.success = target_success.reshape({-1}).cuda();
                    SpeculativeSampler verifier(torch::Tensor(), steps, mode);
                    auto               result = verifier.forward(streams, draft, target);
                    result.transfer_done_event->synchronize();
                    for (int64_t b = 0; b < batch; ++b) {
                        const bool failed = fail_request && b == 1;
                        const bool forced = force_mode == 1 || (force_mode == 2 && b != 2);
                        EXPECT_EQ(result.accept_len_cpu.data_ptr<int32_t>()[b], !failed && forced ? steps + 1 : 1);
                        const auto tokens = result.accept_tokens_cpu.select(0, b);
                        if (failed) {
                            ASSERT_TRUE(result.success_cpu.defined());
                            EXPECT_FALSE(result.success_cpu.data_ptr<bool>()[b]);
                            EXPECT_TRUE(tokens.eq(0).all().item<bool>());
                        } else if (forced) {
                            EXPECT_TRUE(torch::equal(tokens, torch::tensor({1, 1, 3}, cpu_i32)));
                        } else {
                            EXPECT_EQ(tokens.data_ptr<int32_t>()[0], 2);
                        }
                    }
                    if (force_mode == 0) {
                        expected_rng = rng.state();
                    } else {
                        EXPECT_TRUE(torch::equal(rng.state(), expected_rng));
                    }
                }
            }
        }
    }
}

TEST(DSparkSamplerTest, CompactTargetSlotsMatchFullHistoryThroughSamplerAndRejection) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required";
    }
    CudaGeneratorStateGuard rng;
    constexpr int64_t       batch = 3, vocab = 32, history = 81920;
    const auto              cpu_i32 = torch::TensorOptions().dtype(torch::kInt32);
    const auto              cpu_f32 = torch::TensorOptions().dtype(torch::kFloat32);
    for (const int64_t verify : {4, 5, 6, 7}) {
        for (const bool do_sample : {false, true}) {
            // 0: normal rejection; 1: all forced; 2: mixed forced/unforced.
            for (const int force_mode : {0, 1, 2}) {
                Sampler            target_sampler(SamplerInitParams{});
                SpeculativeSampler verifier(torch::Tensor(), verify, DraftProposalMode::SAMPLED);
                for (int round = 0; round < 3; ++round) {
                    SCOPED_TRACE(::testing::Message() << "verify=" << verify << " sample=" << do_sample
                                                      << " force=" << force_mode << " round=" << round);
                    const int64_t                rows            = batch * (verify + 1);
                    const std::array<int64_t, 3> rejection       = {0, verify / 2, verify};
                    auto                         draft_ids_cpu   = torch::empty({batch, verify}, cpu_i32);
                    auto                         draft_probs_cpu = torch::zeros({batch, verify, vocab}, cpu_f32);
                    auto logits_cpu = torch::full({rows, vocab}, -std::numeric_limits<float>::infinity(), cpu_f32);
                    for (int64_t b = 0; b < batch; ++b) {
                        const int64_t accepted = rejection[(b + round) % batch];
                        for (int64_t p = 0; p < verify; ++p) {
                            const int32_t id                                  = static_cast<int32_t>((b + p) % 3);
                            draft_ids_cpu.data_ptr<int32_t>()[b * verify + p] = id;
                            draft_probs_cpu.data_ptr<float>()[(b * verify + p) * vocab + id] = 1.0f;
                        }
                        for (int64_t p = 0; p <= verify; ++p) {
                            const int32_t id = p < accepted ? draft_ids_cpu.data_ptr<int32_t>()[b * verify + p] :
                                                              (p == verify ? 7 : 6);
                            logits_cpu.data_ptr<float>()[(b * (verify + 1) + p) * vocab + id] = 0.0f;
                        }
                    }
                    // One-hot distributions give deterministic first/middle/full
                    // rejection oracles while still executing real stochastic paths.
                    auto                                    initial_rng = rng.state();
                    std::vector<SamplerOutput>              targets;
                    std::vector<SpeculativeSamplerOutput>   results;
                    std::vector<torch::Tensor>              global_states;
                    std::vector<std::vector<torch::Tensor>> target_states, request_states;
                    for (const bool compact : {false, true}) {
                        rng.restore(initial_rng);
                        SamplerInputs input;
                        input.batch_size = input.batch_size_out = rows;
                        input.vocab_size                        = vocab;
                        input.step                              = history;
                        input.phase                             = LogitsProcessorPhase::MTP_VERIFY;
                        input.compact_token_ids                 = compact;
                        input.spec_propose_step                 = verify;
                        input.logits                            = logits_cpu.cuda();
                        input.token_ids = torch::full({rows, compact ? 1 : history + 1}, 4, cpu_i32).pin_memory();
                        input.token_ids.select(1, input.token_ids.size(1) - 1).fill_(-1);
                        const auto original_ids    = input.token_ids.clone();
                        input.input_lengths        = torch::full({rows}, history - 4, cpu_i32).pin_memory();
                        input.sequence_lengths     = torch::full({rows}, history, cpu_i32).pin_memory();
                        input.num_beams_in         = torch::ones({rows}, torch::kInt64).pin_memory();
                        input.num_beams_out        = input.num_beams_in.clone().pin_memory();
                        input.top_k                = torch::full({rows}, do_sample ? 0 : 1, cpu_i32).pin_memory();
                        input.top_p                = torch::ones({rows}, cpu_f32).pin_memory();
                        input.temperature          = torch::ones({rows}, cpu_f32).pin_memory();
                        input.repetition_penalty   = torch::ones({rows}, cpu_f32).pin_memory();
                        input.presence_penalty     = torch::zeros({rows}, cpu_f32).pin_memory();
                        input.frequency_penalty    = torch::zeros({rows}, cpu_f32).pin_memory();
                        input.no_repeat_ngram_size = torch::zeros({rows}, cpu_i32).pin_memory();
                        input.do_sample            = torch::full({rows}, do_sample, torch::kBool).pin_memory();
                        input.all_probs            = torch::zeros_like(input.logits);
                        input.cum_log_probs        = torch::full({rows}, -0.5f, input.logits.options());
                        for (int64_t row = 0; row < rows; ++row) {
                            input.generator.push_back(torch::make_generator<at::CUDAGeneratorImpl>());
                            input.generator.back().set_current_seed(1000 + row + round * rows);
                        }
                        std::list<GenerateStreamPtr> streams;
                        ModelConfig                  model;
                        model.max_seq_len = history + verify + 16;
                        model.vocab_size  = vocab;
                        RuntimeConfig   runtime;
                        ResourceContext resources;
                        for (int64_t b = 0; b < batch; ++b) {
                            auto request                              = std::make_shared<GenerateInput>();
                            request->input_ids                        = torch::tensor({0}, cpu_i32);
                            request->generate_config                  = std::make_shared<GenerateConfig>();
                            request->generate_config->top_k           = do_sample ? 0 : 1;
                            request->generate_config->random_seed     = 2000 + b + round * batch;
                            request->generate_config->force_sp_accept = force_mode == 1 || (force_mode == 2 && b != 1);
                            streams.push_back(
                                std::make_shared<NormalGenerateStream>(request, model, runtime, resources, nullptr));
                        }
                        auto target = target_sampler.forward(input);
                        ASSERT_EQ(target.token_ids.size(1), compact ? 1 : history + 1);
                        target.all_probs = target.all_probs.reshape({batch, verify + 1, vocab});
                        auto          target_ids_before_rejection   = target.token_ids.clone();
                        auto          target_probs_before_rejection = target.all_probs.clone();
                        SamplerOutput draft;
                        draft.token_ids = draft_ids_cpu.cuda();
                        draft.all_probs = draft_probs_cpu.cuda();
                        auto result     = verifier.forward(streams, draft, target);
                        result.transfer_done_event->synchronize();
                        EXPECT_TRUE(torch::equal(target.token_ids, target_ids_before_rejection));
                        EXPECT_TRUE(torch::equal(target.all_probs, target_probs_before_rejection));
                        EXPECT_TRUE(torch::equal(input.token_ids, original_ids));
                        EXPECT_TRUE(input.input_lengths.eq(history - 4).all().item<bool>());
                        EXPECT_TRUE(input.sequence_lengths.eq(history).all().item<bool>());
                        EXPECT_TRUE(torch::equal(draft.token_ids.cpu(), draft_ids_cpu));
                        EXPECT_TRUE(torch::equal(draft.all_probs.cpu(), draft_probs_cpu));
                        std::vector<torch::Tensor> target_rng, request_rng;
                        for (const auto& generator : input.generator)
                            target_rng.push_back(generator.get_state().clone());
                        for (const auto& stream : streams)
                            request_rng.push_back(stream->getGenerator().get_state().clone());
                        target_states.push_back(std::move(target_rng));
                        request_states.push_back(std::move(request_rng));
                        global_states.push_back(rng.state());
                        targets.push_back(std::move(target));
                        results.push_back(std::move(result));
                    }
                    EXPECT_TRUE(
                        torch::equal(targets[0].token_ids.select(1, history), targets[1].token_ids.select(1, 0)));
                    EXPECT_TRUE(torch::equal(targets[0].all_probs, targets[1].all_probs));
                    EXPECT_TRUE(torch::equal(targets[0].cum_log_probs, targets[1].cum_log_probs));
                    EXPECT_TRUE(torch::equal(targets[0].success, targets[1].success));
                    EXPECT_TRUE(targets[0].success.all().item<bool>());
                    EXPECT_FALSE(targets[0].beam_index.defined());
                    EXPECT_FALSE(targets[1].beam_index.defined());
                    EXPECT_TRUE(torch::equal(global_states[0], global_states[1]));
                    for (int64_t row = 0; row < rows; ++row)
                        EXPECT_TRUE(torch::equal(target_states[0][row], target_states[1][row]));
                    for (int64_t b = 0; b < batch; ++b)
                        EXPECT_TRUE(torch::equal(request_states[0][b], request_states[1][b]));
                    EXPECT_TRUE(torch::equal(results[0].accept_tokens, results[1].accept_tokens));
                    EXPECT_TRUE(torch::equal(results[0].accept_len, results[1].accept_len));
                    EXPECT_TRUE(torch::equal(results[0].accept_tokens_cpu, results[1].accept_tokens_cpu));
                    EXPECT_TRUE(torch::equal(results[0].accept_len_cpu, results[1].accept_len_cpu));
                    for (const auto& result : results) {
                        EXPECT_TRUE(torch::equal(result.accept_tokens.cpu(), result.accept_tokens_cpu));
                        EXPECT_TRUE(torch::equal(result.accept_len.cpu(), result.accept_len_cpu));
                        for (int64_t b = 0; b < batch; ++b) {
                            const bool    forced   = force_mode == 1 || (force_mode == 2 && b != 1);
                            const int64_t accepted = forced ? verify : rejection[(b + round) % batch];
                            EXPECT_EQ(result.accept_len_cpu.data_ptr<int32_t>()[b], accepted + 1);
                            for (int64_t p = 0; p < accepted; ++p) {
                                EXPECT_EQ(result.accept_tokens_cpu.data_ptr<int32_t>()[b * (verify + 1) + p],
                                          draft_ids_cpu.data_ptr<int32_t>()[b * verify + p]);
                            }
                            EXPECT_EQ(result.accept_tokens_cpu.data_ptr<int32_t>()[b * (verify + 1) + accepted],
                                      accepted == verify ? 7 : 6);
                        }
                    }
                }
            }
        }
    }
}
#endif

class ShortVerifyRejectionTest: public ::testing::TestWithParam<int> {};

TEST_P(ShortVerifyRejectionTest, SamplesGammaSevenPrefixWithCorrectionAndBonus) {
#if !USING_CUDA
    GTEST_SKIP() << "Exact sampled-draft rejection requires CUDA";
#endif
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required";
    }
    constexpr int B            = 3;
    constexpr int gamma        = 7;
    constexpr int target_vocab = 8;
    const int     k            = GetParam();
    const auto    cpu_i32      = torch::TensorOptions().dtype(torch::kInt32);
    const auto    cpu_f32      = torch::TensorOptions().dtype(torch::kFloat32);

    // Exercise probability rejection (SAMPLED + top_k=0) as well as greedy
    // matching. One-hot distributions make both oracles independent of RNG.
    for (bool mapped : {false, true}) {
        for (bool do_sample : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "k=" << k << " mapped=" << mapped << " do_sample=" << do_sample);
            const int                    draft_vocab = mapped ? 3 : target_vocab;
            const std::array<int32_t, 3> mapped_ids  = {1, 3, 5};
            auto               d2t = mapped ? torch::tensor({1, 3, 5}, torch::kInt64).cuda() : torch::Tensor();
            SpeculativeSampler verifier(d2t, k, DraftProposalMode::SAMPLED);
            ModelConfig        model_config;
            model_config.max_seq_len = 32;
            model_config.vocab_size  = target_vocab;
            RuntimeConfig                runtime_config;
            ResourceContext              resource_context;
            std::list<GenerateStreamPtr> streams;
            for (int b = 0; b < B; ++b) {
                auto input                              = std::make_shared<GenerateInput>();
                input->input_ids                        = torch::tensor({0}, cpu_i32);
                input->generate_config                  = std::make_shared<GenerateConfig>();
                input->generate_config->top_k           = do_sample ? 0 : 1;
                input->generate_config->force_sp_accept = false;
                streams.push_back(std::make_shared<NormalGenerateStream>(
                    input, model_config, runtime_config, resource_context, nullptr));
                ASSERT_FALSE(streams.back()->forceSpAccept());
            }

            // A frozen gamma7 proposal: ids already use target vocabulary,
            // probabilities still use draft vocabulary, as sampleDSparkDraft
            // returns them. Only prefix slicing belongs to the executor.
            auto full_ids_cpu   = torch::empty({B, gamma}, cpu_i32);
            auto full_probs_cpu = torch::zeros({B, gamma, draft_vocab}, cpu_f32);
            for (int b = 0; b < B; ++b) {
                for (int p = 0; p < gamma; ++p) {
                    const int local_id                              = (b + p) % 3;
                    full_ids_cpu.data_ptr<int32_t>()[b * gamma + p] = mapped ? mapped_ids[local_id] : local_id;
                    full_probs_cpu.data_ptr<float>()[(b * gamma + p) * draft_vocab + local_id] = 1.0f;
                }
            }
            auto          full_ids   = full_ids_cpu.cuda();
            auto          full_probs = full_probs_cpu.cuda();
            SamplerOutput draft;
            draft.token_ids = full_ids.narrow(1, 0, k).contiguous();
            draft.all_probs = full_probs.narrow(1, 0, k).contiguous();
            ASSERT_EQ(draft.token_ids.size(1), k);
            ASSERT_EQ(draft.all_probs.size(1), k);

            // Reuse the verifier to exercise its mapped-probability workspace.
            // Rotate cases so accepted length is not tied to a batch position.
            for (int round = 0; round < 2; ++round) {
                SCOPED_TRACE(round);
                const std::array<int, B> cases        = {0, k / 2, k};
                auto                     target_probs = torch::zeros({B, k + 1, target_vocab}, cpu_f32);
                // The native contract takes the last column of token_ids;
                // preserve a distinct history sentinel in the first column.
                auto target_ids = torch::full({B * (k + 1), 2}, 4, cpu_i32);
                for (int b = 0; b < B; ++b) {
                    const int accepted_drafts = cases[(b + round) % B];
                    for (int p = 0; p <= k; ++p) {
                        const int token =
                            p < accepted_drafts ? full_ids_cpu.data_ptr<int32_t>()[b * gamma + p] : (p == k ? 7 : 6);
                        target_probs.data_ptr<float>()[(b * (k + 1) + p) * target_vocab + token] = 1.0f;
                        target_ids.data_ptr<int32_t>()[(b * (k + 1) + p) * 2 + 1]                = token;
                    }
                }
                SamplerOutput target;
                target.all_probs = target_probs.cuda();
                target.token_ids = target_ids.cuda();
                auto output      = verifier.forward(streams, draft, target);
                output.transfer_done_event->synchronize();
                ASSERT_EQ(output.accept_tokens.sizes(), torch::IntArrayRef({B, k + 1}));
                ASSERT_EQ(output.accept_len.numel(), B);
                const auto tokens  = output.accept_tokens.cpu();
                const auto lengths = output.accept_len.cpu();
                EXPECT_TRUE(torch::equal(tokens, output.accept_tokens_cpu));
                EXPECT_TRUE(torch::equal(lengths, output.accept_len_cpu));
                for (int b = 0; b < B; ++b) {
                    const int accepted_drafts = cases[(b + round) % B];
                    EXPECT_EQ(lengths.data_ptr<int32_t>()[b], accepted_drafts + 1);
                    for (int p = 0; p < accepted_drafts; ++p) {
                        EXPECT_EQ(tokens.data_ptr<int32_t>()[b * (k + 1) + p],
                                  full_ids_cpu.data_ptr<int32_t>()[b * gamma + p]);
                    }
                    // Rejection emits a correction; full acceptance emits the
                    // target bonus at row k. Neither may be gamma7's suffix.
                    EXPECT_EQ(tokens.data_ptr<int32_t>()[b * (k + 1) + accepted_drafts], accepted_drafts == k ? 7 : 6);
                }
            }
            EXPECT_TRUE(torch::equal(full_ids.cpu(), full_ids_cpu));
            EXPECT_TRUE(torch::equal(full_probs.cpu(), full_probs_cpu));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(IndependentBudget, ShortVerifyRejectionTest, ::testing::Values(1, 4, 5, 6, 7));

TEST(DSparkSamplerTest, AppliesMarkovBiasAutoregressivelyAndMapsDraftTokens) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required";
    }

    constexpr int64_t batch_size   = 1;
    constexpr int64_t propose_step = 3;
    constexpr int64_t target_vocab = 6;
    constexpr int64_t draft_vocab  = 4;
    constexpr int64_t rank         = 2;
    auto              options      = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

    // draft ids [0,1,2,3] map to target ids [1,3,4,5].  The transition
    // sequence is anchor target 0 -> draft 1(target 3) -> draft 2(target 4)
    // -> draft 3(target 5).  Each next Markov lookup must therefore consume
    // the mapped target id, not the previous draft id.
    auto d2t = torch::tensor({1, 3, 4, 5}, torch::TensorOptions().dtype(torch::kInt64)).cuda();
    auto w1  = torch::zeros({target_vocab, rank}, options);
    w1.index_put_({0, 0}, 1.0f);
    w1.index_put_({3, 0}, 2.0f);
    w1.index_put_({4, 0}, 3.0f);
    auto w2 = torch::tensor({{0.0f, 0.0f}, {10.0f, 0.0f}, {20.0f, 0.0f}, {30.0f, 0.0f}}, options);

    // Cancel the larger later columns at each step so that the expected
    // transition is unique after adding W1[token] @ W2.T.
    auto base_logits = torch::tensor(
        {{0.0f, 0.0f, -100.0f, -100.0f}, {0.0f, -100.0f, -10.0f, -100.0f}, {0.0f, -100.0f, -100.0f, -20.0f}}, options);
    auto anchors     = torch::tensor({0}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
    auto temperature = torch::full({batch_size}, 1.0e-6f, options);

    SpeculativeSampler sampler(d2t, propose_step, DraftProposalMode::SAMPLED);
    auto               output = sampler.sampleDSparkDraft(base_logits, anchors, temperature, w1, w2, draft_vocab);

    ASSERT_EQ(output.token_ids.sizes(), torch::IntArrayRef({batch_size, propose_step}));
    EXPECT_TRUE(torch::equal(output.token_ids.cpu(), torch::tensor({{3, 4, 5}}, torch::kInt32)));
    ASSERT_EQ(output.all_probs.sizes(), torch::IntArrayRef({batch_size, propose_step, draft_vocab}));
    EXPECT_TRUE(torch::allclose(output.all_probs.sum(-1).cpu(), torch::ones({batch_size, propose_step})));
}

TEST(DSparkSamplerTest, KeepsSoftmaxDistributionForSamplingRequests) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required";
    }

    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto logits  = torch::tensor({{0.0f, 1.0f, 2.0f}}, options);
    auto anchors = torch::tensor({0}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
    auto temp    = torch::ones({1}, options);
    auto w1      = torch::zeros({3, 2}, options);
    auto w2      = torch::zeros({3, 2}, options);

    SpeculativeSampler sampler(torch::Tensor(), 1, DraftProposalMode::SAMPLED);
    auto               output = sampler.sampleDSparkDraft(logits, anchors, temp, w1, w2, 3);

    auto expected = torch::softmax(logits, -1).reshape({1, 1, 3});
    EXPECT_TRUE(torch::allclose(output.all_probs, expected));
}

#if USING_CUDA
TEST(DSparkSamplerTest, AdaptiveVerifyPlanMatchesStableSortAtBoundaries) {
    struct Candidate {
        float value;
        int   position;
        int   request;
    };
    // Match the CUDA target's FTZ boundary; keep the oracle a full stable
    // sort, independent of the kernel's parallel rank-count implementation.
    const auto flush = [](float value) {
        return std::abs(value) < std::numeric_limits<float>::min() ? std::copysign(0.f, value) : value;
    };
    const auto cpu = torch::TensorOptions().dtype(torch::kFloat32);
    for (const auto& shape :
         std::vector<std::pair<int, int>>{{0, 7}, {1, 1}, {1, 1024}, {3, 4}, {16, 7}, {28, 7}, {256, 4}}) {
        const int batch = shape.first, gamma = shape.second;
        for (int pattern = 0; pattern < 5; ++pattern) {
            auto                   confidence = torch::empty({batch, gamma}, cpu);
            auto*                  values     = confidence.data_ptr<float>();
            std::vector<Candidate> candidates;
            for (int request = 0; request < batch; ++request) {
                float survival = 1.f;
                for (int position = 0; position < gamma; ++position) {
                    float value = 0.f;
                    if (pattern == 1) {
                        value = 1.f;
                    } else if (pattern == 2) {
                        value = static_cast<float>((request * 17 + position * 13) % 37 - 4) / 29.f;
                    } else if (pattern == 3) {
                        const float edges[] = {std::numeric_limits<float>::quiet_NaN(),
                                               std::numeric_limits<float>::infinity(),
                                               -std::numeric_limits<float>::infinity(),
                                               -.2f,
                                               1.2f,
                                               -0.f,
                                               .5f};
                        value               = edges[(request + position) % 7];
                    } else if (pattern == 4) {
                        const float edges[] = {std::numeric_limits<float>::denorm_min(),
                                               std::numeric_limits<float>::min(),
                                               .5f,
                                               -0.f,
                                               0.f,
                                               0x1p-70f};
                        value               = edges[(request + position) % 6];
                    }
                    values[request * gamma + position] = value;
                    value                              = flush(value);
                    const float conditional = std::isfinite(value) ? std::min(std::max(value, 0.f), 1.f) : 0.f;
                    survival                = flush(survival * conditional);
                    candidates.push_back({survival, position, request});
                }
            }
            std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                if (a.value != b.value)
                    return a.value > b.value;
                if (a.position != b.position)
                    return a.position < b.position;
                return a.request < b.request;
            });
            const auto input = confidence.to(torch::kCUDA);
            for (int budget : {0, std::min(1, batch * gamma), std::min(4 * batch, batch * gamma), batch * gamma}) {
                SCOPED_TRACE(::testing::Message() << "batch=" << batch << " gamma=" << gamma << " pattern=" << pattern
                                                  << " budget=" << budget);
                std::vector<int> expected(batch, 1);
                for (int picked = 0; picked < budget; ++picked)
                    ++expected[candidates[picked].request];
                auto plan    = execDSparkVerifyPlan(input, budget);
                auto lengths = plan.first.cpu(), mapping = plan.second.cpu();
                ASSERT_EQ(mapping.numel(), batch + budget);
                int compact = 0;
                for (int request = 0; request < batch; ++request) {
                    ASSERT_EQ(lengths[request].item<int>(), expected[request]);
                    for (int row = 0; row < expected[request]; ++row) {
                        EXPECT_EQ(mapping[compact++].item<int>(), request * (gamma + 1) + row);
                    }
                }
            }
        }
    }
}

#endif

TEST(DSparkSamplerTest, SevenTokenProposalPreservesConditionalProbabilities) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required";
    }
    constexpr int64_t steps        = 7;
    constexpr int64_t vocab        = 31;
    constexpr int64_t padded_vocab = 32;
    constexpr int64_t rank         = 256;
    auto              options      = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    torch::manual_seed(20260926);
    auto               w1 = (torch::randn({vocab, rank}, options) * 0.1f).to(torch::kBFloat16);
    auto               w2 = (torch::randn({vocab, rank}, options) * 0.1f).to(torch::kBFloat16);
    SpeculativeSampler sampler(torch::Tensor(), steps, DraftProposalMode::SAMPLED);
    for (const int64_t batch : {1, 3, 16}) {
        SCOPED_TRACE(batch);
        auto logits = torch::randn({batch * steps, padded_vocab}, options);
        // Padding must never become a sampled token, even with a dominant logit.
        logits.select(1, vocab).fill_(10000.0f);
        auto anchors     = torch::arange(batch, options.dtype(torch::kInt32));
        auto temperature = torch::linspace(0.3f, 1.5f, batch, options);
        auto output      = sampler.sampleDSparkDraft(logits, anchors, temperature, w1, w2, vocab);
        ASSERT_EQ(output.token_ids.sizes(), torch::IntArrayRef({batch, steps}));
        ASSERT_EQ(output.all_probs.sizes(), torch::IntArrayRef({batch, steps, vocab}));
        EXPECT_TRUE(output.token_ids.ge(0).logical_and(output.token_ids.lt(vocab)).all().item<bool>());
        auto previous = anchors.to(torch::kLong);
        auto base     = logits.narrow(1, 0, vocab).view({batch, steps, vocab});
        for (int64_t step = 0; step < steps; ++step) {
            auto bias     = torch::mm(w1.index_select(0, previous), w2.transpose(0, 1)).to(torch::kFloat32);
            auto expected = torch::softmax((base.select(1, step) + bias) / temperature.unsqueeze(1), -1);
            EXPECT_TRUE(torch::allclose(output.all_probs.select(1, step), expected, 1e-5, 1e-6));
            previous = output.token_ids.select(1, step).to(torch::kLong);
        }
    }
}

// A deterministic wiring gate, not a finite-sample distribution proof. Target
// distributions are deliberately chosen after the REAL proposal draw so the
// correction/bonus oracle is independent of which tokens the RNG produced.
TEST(DSparkSamplerTest, RealMarkovConfidenceAdaptiveRejectionChain) {
#if !USING_CUDA
    GTEST_SKIP() << "The real DSpARK sampling chain requires CUDA";
#else
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA is required";
    }
    CudaGeneratorStateGuard rng_guard;
    constexpr int64_t       gamma = 3, vocab = 2, hidden_dim = 2;
    auto                    f32               = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto                    i32               = f32.dtype(torch::kInt32);
    auto                    boolean           = f32.dtype(torch::kBool);
    auto                    w1                = torch::tensor({{-1.f, .5f}, {1.f, -.5f}}, f32).to(torch::kBFloat16);
    auto                    w2                = torch::tensor({{.5f, .25f}, {-.25f, .5f}}, f32).to(torch::kBFloat16);
    auto                    confidence_weight = torch::tensor({1.f, 0.f, .125f, 0.f}, f32).to(torch::kBFloat16);
    auto                    confidence_bias   = torch::zeros({1}, f32).to(torch::kBFloat16);
    SpeculativeSampler      sampler(torch::Tensor(), gamma, DraftProposalMode::SAMPLED);
    for (int64_t batch : {3, 6}) {
        for (int seed : {20261001, 20261002}) {
            SCOPED_TRACE(::testing::Message() << "batch=" << batch << " seed=" << seed);
            torch::manual_seed(seed);
            auto base =
                (torch::arange(batch * gamma * vocab, f32).reshape({batch * gamma, vocab}).remainder(7) - 3.f) / 8.f;
            auto temperature = torch::linspace(.7f, 1.3f, batch, f32);
            auto anchors     = torch::arange(batch, i32).remainder(vocab).contiguous();
            auto draft       = sampler.sampleDSparkDraft(base, anchors, temperature, w1, w2, vocab);
            ASSERT_EQ(draft.token_ids.sizes(), torch::IntArrayRef({batch, gamma}));
            ASSERT_EQ(draft.all_probs.sizes(), torch::IntArrayRef({batch, gamma, vocab}));
            auto ids_cpu = draft.token_ids.cpu();
            ASSERT_TRUE(ids_cpu.ge(0).logical_and(ids_cpu.lt(vocab)).all().item<bool>());
            auto previous = anchors.to(torch::kLong);
            for (int64_t j = 0; j < gamma; ++j) {
                // Mirror the BF16 GEMM output boundary, not the fused combine op.
                auto bias     = torch::mm(w1.index_select(0, previous), w2.transpose(0, 1)).to(torch::kFloat32);
                auto expected = torch::softmax(
                    (base.view({batch, gamma, vocab}).select(1, j) + bias) / temperature.unsqueeze(1), -1);
                EXPECT_TRUE(torch::allclose(draft.all_probs.select(1, j), expected, 1.e-5, 1.e-6));
                previous = draft.token_ids.select(1, j).to(torch::kLong);
            }
            auto hidden_cpu = torch::zeros({batch, gamma, hidden_dim}, torch::kFloat32);
            // High/medium/low survival bands guarantee caps [1,3,4] per group
            // despite the actual previous-token Markov feature +/-0.125.
            for (int64_t b = 0; b < batch; ++b)
                hidden_cpu[b].select(1, 0).fill_(b % 3 == 0 ? -8.f : b % 3 == 1 ? 0.f : 4.f);
            auto hidden     = hidden_cpu.reshape({batch * gamma, hidden_dim}).to(f32.device()).to(torch::kBFloat16);
            auto confidence = sampler.computeDSparkConfidence(
                hidden, anchors, draft.token_ids, w1, confidence_weight, confidence_bias);
            auto prev = torch::cat({anchors.view({batch, 1}), draft.token_ids.narrow(1, 0, gamma - 1)}, 1)
                            .reshape({-1})
                            .to(torch::kLong);
            auto features = torch::cat({hidden.to(torch::kFloat32), w1.index_select(0, prev).to(torch::kFloat32)}, 1);
            auto confidence_reference = torch::sigmoid((features.matmul(confidence_weight.to(torch::kFloat32))
                                                        + confidence_bias.to(torch::kFloat32))
                                                           .to(torch::kBFloat16)
                                                           .to(torch::kFloat32))
                                            .reshape({batch, gamma});
            EXPECT_TRUE(torch::allclose(confidence, confidence_reference, 1.e-6, 1.e-6));
            auto plan     = execDSparkVerifyPlan(confidence, 5 * (batch / 3));
            auto caps_cpu = plan.first.cpu(), mapping_cpu = plan.second.cpu();
            int  compact = 0;
            for (int64_t b = 0; b < batch; ++b) {
                const int expected_cap = b % 3 == 0 ? 1 : b % 3 == 1 ? 3 : 4;
                ASSERT_EQ(caps_cpu[b].item<int>(), expected_cap);
                for (int j = 0; j < expected_cap; ++j)
                    EXPECT_EQ(mapping_cpu[compact++].item<int>(), b * (gamma + 1) + j);
            }
            auto target = torch::zeros({batch, gamma + 1, vocab}, f32);
            target.narrow(1, 0, gamma).copy_(draft.all_probs);
            for (int64_t b = 0; b < batch; ++b) {
                const int reject_position = b % 3 == 0 ? 0 : b % 3 == 1 ? 1 : -1;
                if (reject_position >= 0) {
                    int correction = 1 - ids_cpu[b][reject_position].item<int>();
                    target[b][reject_position].zero_();
                    target[b][reject_position][correction].fill_(1.f);
                }
                int bonus = 1 - ids_cpu[b][gamma - 1].item<int>();
                target[b][gamma][bonus].fill_(1.f);
            }
            // Real target categorical sampling; one-hot correction/bonus rows
            // give exact token oracles while q=p rows remain nontrivial draws.
            auto target_ids =
                execSampleFromProbs(target.reshape({batch * (gamma + 1), vocab})).reshape({batch * (gamma + 1), 1});
            auto output    = torch::full({batch, gamma + 1}, -1, i32);
            auto lengths   = torch::zeros({batch}, i32);
            auto success   = torch::ones({batch}, boolean);
            auto workspace = torch::empty({rtp_llm::rejectionValidationWorkspaceElements(batch, gamma, vocab)}, f32);
            execRejectionSampling({draft.all_probs,
                                   draft.token_ids,
                                   torch::full({batch, gamma + 1}, .5f, f32),
                                   target,
                                   target_ids,
                                   output,
                                   lengths,
                                   torch::ones({batch}, boolean),
                                   false,
                                   true,
                                   success,
                                   torch::ones({batch, gamma + 1}, boolean),
                                   plan.first,
                                   workspace});
            // Same tensor operation as MtpExecutor::capDSparkVerifyLengths.
            auto committed  = torch::minimum(lengths, plan.first).cpu();
            auto output_cpu = output.cpu(), success_cpu = success.cpu();
            for (int64_t b = 0; b < batch; ++b) {
                ASSERT_TRUE(success_cpu[b].item<bool>());
                int expected_length = b % 3 == 0 ? 1 : b % 3 == 1 ? 2 : 4;
                ASSERT_EQ(committed[b].item<int>(), expected_length);
                for (int j = 0; j < expected_length; ++j) {
                    int expected = j == expected_length - 1 ? 1 - ids_cpu[b][j == gamma ? gamma - 1 : j].item<int>() :
                                                              ids_cpu[b][j].item<int>();
                    EXPECT_EQ(output_cpu[b][j].item<int>(), expected) << "b=" << b << " j=" << j;
                }
            }
            // A second valid target gives raw full acceptance on every row,
            // making the adaptive production cap observable (not a no-op).
            target.narrow(1, 0, gamma).copy_(draft.all_probs);
            target_ids =
                execSampleFromProbs(target.reshape({batch * (gamma + 1), vocab})).reshape({batch * (gamma + 1), 1});
            output.fill_(-1);
            lengths.zero_();
            success.fill_(true);
            execRejectionSampling({draft.all_probs,
                                   draft.token_ids,
                                   torch::full({batch, gamma + 1}, .5f, f32),
                                   target,
                                   target_ids,
                                   output,
                                   lengths,
                                   torch::ones({batch}, boolean),
                                   false,
                                   true,
                                   success,
                                   torch::ones({batch, gamma + 1}, boolean),
                                   plan.first,
                                   workspace});
            auto raw_lengths = lengths.cpu();
            committed        = torch::minimum(lengths, plan.first).cpu();
            output_cpu       = output.cpu();
            success_cpu      = success.cpu();
            for (int64_t b = 0; b < batch; ++b) {
                ASSERT_TRUE(success_cpu[b].item<bool>());
                ASSERT_EQ(raw_lengths[b].item<int>(), gamma + 1);
                ASSERT_EQ(committed[b].item<int>(), caps_cpu[b].item<int>());
                for (int j = 0; j < committed[b].item<int>(); ++j)
                    EXPECT_EQ(output_cpu[b][j].item<int>(),
                              j < gamma ? ids_cpu[b][j].item<int>() : 1 - ids_cpu[b][gamma - 1].item<int>());
            }
        }
    }
#endif
}

}  // namespace
}  // namespace rtp_llm::speculative
