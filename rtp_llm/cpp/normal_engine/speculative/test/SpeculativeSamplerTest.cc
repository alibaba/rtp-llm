#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <gtest/gtest.h>
#include <limits>
#include <vector>
#include <torch/all.h>
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#if USING_CUDA
#include <ATen/autocast_mode.h>
#include <c10/util/ScopeExit.h>
#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"
#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"
#endif

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

#if USING_CUDA
TEST(DSparkSamplerTest, RejectionGraphsPreserveMasksAndResultOwnership) {
    const c10::cuda::CUDAGuard guard(0);
    const auto                 ints  = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    constexpr int              vocab = 257;
    for (int steps : {1, 3, 5}) {
        SpeculativeSampler prepared({}, steps);
        prepared.prepareRejectionGraphs(torch::Device(torch::kCUDA, 0), vocab, {1, 7, 64});
        for (int batch : {1, 2, 7, 31, 64}) {
            std::vector<SpeculativeSamplerOutput> retained;
            for (int round = 0; round < 5; ++round) {
                auto draft   = torch::randint(vocab, {batch, steps}, ints);
                auto target  = torch::randint(vocab, {batch * (steps + 1), round % 2 ? 37 : 1}, ints);
                auto uniform = torch::rand({batch, steps + 1}, ints.dtype(torch::kFloat32));
                auto probs   = torch::empty({batch, steps + 1, vocab}, uniform.options());
                auto masks   = torch::empty({2, batch}, torch::kBool);
                for (int row = 0; row < batch; ++row) {
                    masks.data_ptr<bool>()[row]         = (row + round) % 3 == 0;
                    masks.data_ptr<bool>()[batch + row] = (row + round) % 2 == 0;
                }
                auto md       = masks.to(draft.device());
                auto expected = torch::empty({batch, steps + 1}, ints);
                auto lengths  = torch::empty({batch}, ints);
                execRejectionSampling({{},
                                       draft,
                                       uniform,
                                       probs,
                                       target,
                                       expected,
                                       lengths,
                                       md.select(0, 0),
                                       true,
                                       true,
                                       md.select(0, 1),
                                       true});
                SpeculativeSamplerOutput actual;
                const bool replayed = prepared.replayRejectionGraph(actual, masks, draft, target, uniform, probs, true);
                ASSERT_EQ(replayed, round < 3);
                if (!replayed)
                    continue;
                actual.transfer_done_event->synchronize();
                EXPECT_TRUE(torch::equal(actual.accept_tokens, expected));
                EXPECT_TRUE(torch::equal(actual.accept_len, lengths));
                EXPECT_TRUE(torch::equal(actual.accept_tokens_cpu, expected.cpu()));
                EXPECT_TRUE(torch::equal(actual.accept_len_cpu, lengths.cpu()));
                retained.push_back(std::move(actual));
            }
            // Retained outputs occupy all slots even after their GPU events
            // complete. They must remain intact while later calls fall back.
            auto saved = retained.front().accept_tokens_cpu.clone();
            EXPECT_TRUE(torch::equal(retained.front().accept_tokens.cpu(), saved));
            auto previous_event = retained.front().transfer_done_event;
            retained.clear();
            auto                     draft   = torch::zeros({batch, steps}, ints);
            auto                     target  = torch::zeros({batch * (steps + 1), 37}, ints);
            auto                     uniform = torch::zeros({batch, steps + 1}, ints.dtype(torch::kFloat32));
            auto                     probs   = torch::empty({batch, steps + 1, vocab}, uniform.options());
            auto                     masks   = torch::zeros({2, batch}, torch::kBool);
            SpeculativeSamplerOutput reused;
            ASSERT_TRUE(prepared.replayRejectionGraph(reused, masks, draft, target, uniform, probs, true));
            // A logits cap may re-record the public event. It must never be
            // the cache slot's event, including when the same slot is reused.
            EXPECT_NE(reused.transfer_done_event, previous_event);
            reused.transfer_done_event->synchronize();
        }
    }
}

TEST(DSparkSamplerTest, RejectionGraphReplaysChangingAllocatedLengths) {
    const c10::cuda::CUDAGuard guard(0);
    constexpr int              batch = 4, steps = 3, vocab = 17, stride = 5;
    const auto                 ints = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    SpeculativeSampler         prepared({}, steps);
    // A larger capture slot also checks the force-mask offset during replay.
    prepared.prepareRejectionGraphs(torch::Device(torch::kCUDA, 0), vocab, {7});
    auto draft = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}}, ints);
    auto target =
        torch::arange(batch * (steps + 1) * stride, ints).remainder(vocab).reshape({batch * (steps + 1), stride});
    auto uniform        = torch::full({batch, steps + 1}, 0.5f, ints.dtype(torch::kFloat32));
    auto probs          = torch::empty({batch, steps + 1, vocab}, uniform.options());
    auto masks          = torch::tensor({{false, true, false, true}, {true, false, true, false}}, torch::kBool);
    auto md             = masks.to(draft.device());
    auto first_lengths  = torch::tensor({0, 1, 3, 2}, ints);
    auto second_lengths = torch::tensor({3, 2, 0, 1}, ints);
    const std::vector<torch::Tensor> length_inputs{{}, first_lengths, second_lengths, first_lengths, {}};
    void*                            first_slot = nullptr;
    for (size_t round = 0; round < length_inputs.size(); ++round) {
        SCOPED_TRACE(round);
        const auto& valid_lengths = length_inputs[round];
        if (round == 3) {
            // Reuse the same pointer with new device values as well as changing
            // pointers between rounds. The final round restores full width.
            first_lengths.zero_();
        }
        auto expected_tokens  = torch::empty({batch, steps + 1}, ints);
        auto expected_lengths = torch::empty({batch}, ints);
        execRejectionSampling({{},
                               draft,
                               uniform,
                               probs,
                               target,
                               expected_tokens,
                               expected_lengths,
                               md.select(0, 0),
                               true,
                               true,
                               md.select(0, 1),
                               true,
                               valid_lengths});
        SpeculativeSamplerOutput actual;
        ASSERT_TRUE(prepared.replayRejectionGraph(actual, masks, draft, target, uniform, probs, true, valid_lengths));
        actual.transfer_done_event->synchronize();
        if (round == 0) {
            first_slot = actual.graph_buffers.get();
        } else {
            EXPECT_EQ(actual.graph_buffers.get(), first_slot);
        }
        EXPECT_TRUE(torch::equal(actual.accept_tokens, expected_tokens));
        EXPECT_TRUE(torch::equal(actual.accept_len, expected_lengths));
        EXPECT_TRUE(torch::equal(actual.accept_tokens_cpu, expected_tokens.cpu()));
        EXPECT_TRUE(torch::equal(actual.accept_len_cpu, expected_lengths.cpu()));
        auto lengths_cpu = valid_lengths.defined() ? valid_lengths.cpu() : torch::full({batch}, steps, torch::kInt32);
        for (int row : {0, 2}) {
            const int limit = lengths_cpu[row].item<int32_t>();
            EXPECT_EQ(actual.accept_len_cpu[row].item<int32_t>(), limit + 1);
            EXPECT_EQ(actual.accept_tokens_cpu[row][limit].item<int32_t>(),
                      target[row * (steps + 1) + limit][stride - 1].item<int32_t>());
        }
    }
}

TEST(DSparkSamplerTest, StableHeadOutputReplaysSamplingAndPreservesFallbacks) {
    const c10::cuda::CUDAGuard guard(0);
    const auto                 options = torch::TensorOptions().device(torch::kCUDA);
    constexpr int64_t          vocab = 4096, hidden = 256, rank = 256;
    auto                       head = torch::randn({vocab, hidden}, options.dtype(torch::kBFloat16));
    auto                       w1   = torch::randn({vocab, rank}, head.options());
    auto                       w2   = torch::randn_like(w1);
    for (size_t steps : {1, 3, 5}) {
        SpeculativeSampler eager({}, steps), prepared({}, steps);
        auto buffer = torch::zeros({64 * static_cast<int64_t>(steps), vocab}, options.dtype(torch::kFloat32));
        prepared.prepareDSparkDraftGraphs(buffer, w1, w2, vocab, {1, 7, 64});
        for (int64_t batch : {1, 7, 64}) {
            auto input           = torch::randn({batch * static_cast<int64_t>(steps), hidden}, head.options());
            auto expected_logits = torch_ext::cublas_gemm_bf16_bf16_fp32(input, head);
            auto logits = torch_ext::cublas_gemm_bf16_bf16_fp32_out(input, head, buffer.narrow(0, 0, input.size(0)));
            ASSERT_TRUE(torch::equal(logits.view(torch::kInt32), expected_logits.view(torch::kInt32)));
            auto anchors = torch::randint(vocab, {batch}, options.dtype(torch::kInt32));
            for (float value : {1.0e-6f, 0.75f}) {
                auto temperature = torch::full({batch}, value, options.dtype(torch::kFloat32));
                auto expected    = eager.sampleDSparkDraft(logits, anchors, temperature, {}, {}, w1, w2, vocab, true);
                auto actual = prepared.sampleDSparkDraft(logits, anchors, temperature, {}, {}, w1, w2, vocab, true);
                ASSERT_TRUE(torch::equal(actual.token_ids, expected.token_ids));
                ASSERT_TRUE(actual.token_ids_are_point_mass);
                auto saved = actual.token_ids.clone();
                logits.zero_();
                prepared.sampleDSparkDraft(logits, anchors, temperature, {}, {}, w1, w2, vocab, true);
                ASSERT_TRUE(torch::equal(actual.token_ids, saved));
                logits.copy_(expected_logits);
                // A different base allocation or weight allocation must use
                // the original implementation, never stale captured pointers.
                auto other    = logits.clone();
                auto fallback = prepared.sampleDSparkDraft(other, anchors, temperature, {}, {}, w1, w2, vocab, true);
                ASSERT_TRUE(torch::equal(fallback.token_ids, expected.token_ids));
                auto replacement = w1.clone().add_(0.25f);
                auto reloaded =
                    eager.sampleDSparkDraft(logits, anchors, temperature, {}, {}, replacement, w2, vocab, true);
                auto checked =
                    prepared.sampleDSparkDraft(logits, anchors, temperature, {}, {}, replacement, w2, vocab, true);
                ASSERT_TRUE(torch::equal(checked.token_ids, reloaded.token_ids));
                torch::manual_seed(1234);
                auto sampled = eager.sampleDSparkDraft(logits, anchors, temperature, {}, {}, w1, w2, vocab, false);
                torch::manual_seed(1234);
                auto sampled_graph_owner =
                    prepared.sampleDSparkDraft(logits, anchors, temperature, {}, {}, w1, w2, vocab, false);
                ASSERT_FALSE(sampled_graph_owner.token_ids_are_point_mass);
                ASSERT_TRUE(torch::equal(sampled_graph_owner.token_ids, sampled.token_ids));
                ASSERT_TRUE(torch::equal(sampled_graph_owner.all_probs, sampled.all_probs));
            }
        }
    }
}

TEST(DSparkSamplerTest, Int32IndicesPreserveGreedyAndStochasticResults) {
    const c10::cuda::CUDAGuard guard(0);
    const auto                 options = torch::TensorOptions().device(torch::kCUDA);
    constexpr int64_t          vocab = 257, padded_vocab = 384, target_vocab = 275, rank = 16;
    for (auto dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
        for (size_t steps : {1, 3, 5, 8}) {
            for (bool mapped : {false, true}) {
                auto map =
                    mapped ? (torch::arange(vocab, options.dtype(torch::kLong)) * 7) % target_vocab : torch::Tensor();
                SpeculativeSampler sampler(map, steps);
                auto w1 = torch::randn({target_vocab, rank * 2}, options.dtype(dtype)).slice(1, 0, rank * 2, 2);
                auto w2 = torch::randn({vocab, rank * 2}, options.dtype(dtype)).slice(1, 0, rank * 2, 2);
                for (int64_t batch : {1, 7, 64, 65, 129}) {
                    auto logits      = torch::randn({batch * static_cast<int64_t>(steps), padded_vocab},
                                               options.dtype(torch::kFloat32));
                    auto anchors     = torch::randint(target_vocab, {batch}, options.dtype(torch::kInt32));
                    auto temperature = torch::full({batch}, 0.75f, options.dtype(torch::kFloat32));
                    for (bool greedy : {false, true}) {
                        // Frozen pre-optimization reference: every Markov and
                        // vocabulary-map lookup explicitly uses int64 indices.
                        torch::manual_seed(1234);
                        auto previous  = anchors.to(torch::kLong);
                        auto proposals = logits.narrow(1, 0, vocab).view({batch, static_cast<int64_t>(steps), vocab});
                        auto probabilities =
                            torch::empty({batch, static_cast<int64_t>(steps), vocab}, options.dtype(torch::kFloat32));
                        auto scratch = torch::empty({batch, vocab}, options.dtype(torch::kFloat32));
                        std::vector<torch::Tensor> columns;
                        for (int64_t step = 0; step < static_cast<int64_t>(steps); ++step) {
                            auto          bias = torch::mm(w1.index_select(0, previous), w2.transpose(0, 1));
                            torch::Tensor tokens;
                            if (greedy) {
                                tokens = execMarkovArgmaxStep({proposals.select(1, step), bias, temperature});
                            } else {
                                execMarkovSoftmaxStep({proposals.select(1, step),
                                                       bias,
                                                       temperature,
                                                       probabilities.select(1, step),
                                                       scratch});
                                tokens = execSampleFromProbs(scratch).to(torch::kInt32);
                            }
                            if (map.defined()) {
                                tokens = map.index_select(0, tokens.to(torch::kLong)).to(torch::kInt32);
                            }
                            columns.push_back(tokens);
                            previous = tokens.to(torch::kLong);
                        }
                        auto expected = torch::stack(columns, 1);
                        torch::manual_seed(1234);
                        auto actual =
                            sampler.sampleDSparkDraft(logits, anchors, temperature, {}, {}, w1, w2, vocab, greedy);
                        ASSERT_TRUE(torch::equal(expected, actual.token_ids));
                        ASSERT_EQ(actual.token_ids_are_point_mass, greedy);
                        if (!greedy) {
                            ASSERT_TRUE(torch::equal(probabilities, actual.all_probs));
                        }
                    }
                }
            }
        }
    }
}

TEST(DSparkSamplerTest, FusedOutputPreservesEmbeddingBitsAndStrides) {
    const c10::cuda::CUDAGuard guard(0);
    const auto                 options = torch::TensorOptions().device(torch::kCUDA);
    constexpr int64_t          vocab = 128, target_vocab = 139;
    for (auto dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
        const auto bits     = dtype == torch::kFloat32 ? torch::kInt32 : torch::kInt16;
        auto       patterns = dtype == torch::kFloat32 ?
                                  torch::tensor(std::vector<int64_t>{0, 1, 0x7f800001, 0x7fc12345, 0xffffffff, 0x80000000}) :
                                  torch::tensor(std::vector<int64_t>{0, 1, 0x7f81, 0x7fc1, 0xffff, 0x8000});
        for (int64_t rank : {0, 1, 33, 256}) {
            auto indexes = torch::arange(target_vocab * rank * 2, torch::kLong) % patterns.numel();
            auto weights = patterns.index_select(0, indexes)
                               .to(bits)
                               .view(dtype)
                               .reshape({target_vocab, rank * 2})
                               .to(options.device())
                               .slice(1, 0, rank * 2, 2);
            if (rank == 256) {
                weights = weights.contiguous();
            }
            auto map =
                (torch::arange(vocab * 2, options.dtype(torch::kLong)) * 7 % target_vocab).slice(0, 0, vocab * 2, 2);
            for (auto bias_dtype : {torch::kFloat32, torch::kFloat16, torch::kBFloat16}) {
                for (int64_t batch : {0, 1, 7, 64, 65, 257}) {
                    auto logits = torch::zeros({batch, vocab}, options.dtype(torch::kFloat32));
                    logits.select(1, 3).fill_(std::numeric_limits<float>::quiet_NaN());
                    logits.select(1, 17).fill_(std::numeric_limits<float>::quiet_NaN());
                    auto bias        = torch::zeros({batch, vocab}, options.dtype(bias_dtype));
                    auto temperature = torch::ones({batch}, options.dtype(torch::kFloat32));
                    auto tokens      = torch::full({batch, 3}, -99, options.dtype(torch::kInt32));
                    auto next        = torch::empty({batch, rank}, options.dtype(dtype));
                    auto expected_tokens =
                        map.index_select(0, torch::full({batch}, 3, options.dtype(torch::kLong))).to(torch::kInt32);
                    auto expected_embedding = weights.index_select(0, expected_tokens.to(torch::kLong));
                    auto actual =
                        cudaMarkovArgmaxStep({logits, bias, temperature}, tokens.select(1, 1), map, weights, next);
                    ASSERT_TRUE(torch::equal(actual, expected_tokens));
                    ASSERT_TRUE(torch::equal(next.view(bits), expected_embedding.view(bits)));
                    ASSERT_TRUE(torch::equal(tokens.select(1, 0), torch::full_like(actual, -99)));
                    ASSERT_TRUE(torch::equal(tokens.select(1, 2), torch::full_like(actual, -99)));
                }
            }
        }
    }
}

TEST(DSparkSamplerTest, FusedOutputReplaysCurrentInputsAndKeepsFallback) {
    const c10::cuda::CUDAGuard guard(0);
    const auto                 options     = torch::TensorOptions().device(torch::kCUDA);
    auto                       logits      = torch::zeros({7, 128}, options.dtype(torch::kFloat32));
    auto                       bias        = torch::zeros_like(logits);
    auto                       temperature = torch::ones({7}, options.dtype(torch::kFloat32));
    auto                       weights     = torch::arange(128 * 33, options.dtype(torch::kFloat32)).reshape({128, 33});
    auto                       tokens      = torch::empty({7}, options.dtype(torch::kInt32));
    auto                       next        = torch::empty({7, 33}, weights.options());
    at::cuda::getCurrentCUDAStream().synchronize();
    const auto                       stream = at::cuda::getStreamFromPool();
    const c10::cuda::CUDAStreamGuard stream_guard(stream);
    cudaMarkovArgmaxStep({logits, bias, temperature}, tokens, {}, weights, next);
    stream.synchronize();
    at::cuda::CUDAGraph graph;
    graph.capture_begin();
    cudaMarkovArgmaxStep({logits, bias, temperature}, tokens, {}, weights, next);
    graph.capture_end();
    for (int column : {0, 5, 127}) {
        logits.zero_();
        logits.select(1, column).fill_(1);
        graph.replay();
        ASSERT_TRUE(torch::equal(tokens, torch::full_like(tokens, column)));
        ASSERT_TRUE(torch::equal(next, weights.index_select(0, tokens.to(torch::kLong))));
    }
    // Non-int64 maps retain the original standalone Torch mapping path.
    auto               map = torch::arange(128, options.dtype(torch::kInt32));
    SpeculativeSampler sampler(map, 1);
    auto               anchors = torch::zeros({7, 2}, options.dtype(torch::kInt32)).select(1, 0);
    auto               w2      = torch::zeros_like(weights);
    auto               result = sampler.sampleDSparkDraft(logits, anchors, temperature, {}, {}, weights, w2, 128, true);
    ASSERT_TRUE(torch::equal(result.token_ids, torch::full_like(result.token_ids, 127)));
}

TEST(DSparkSamplerTest, FusedGreedySupportsAutocastWithFloat32EmbeddingWeights) {
    const c10::cuda::CUDAGuard guard(0);
    const auto                 options        = torch::TensorOptions().device(torch::kCUDA);
    const bool                 was_enabled    = at::autocast::is_autocast_enabled(at::kCUDA);
    const auto                 previous_dtype = at::autocast::get_autocast_dtype(at::kCUDA);
    auto                       restore        = c10::make_scope_exit([&] {
        at::autocast::set_autocast_enabled(at::kCUDA, was_enabled);
        at::autocast::set_autocast_dtype(at::kCUDA, previous_dtype);
    });
    auto                       weights        = torch::zeros({128, 16}, options.dtype(torch::kFloat32));
    auto                       logits         = torch::arange(128, options.dtype(torch::kFloat32)).repeat({21, 1});
    auto                       anchors        = torch::zeros({7}, options.dtype(torch::kInt32));
    auto                       temperature    = torch::ones({7}, options.dtype(torch::kFloat32));
    SpeculativeSampler         sampler({}, 3);
    for (auto dtype : {torch::kFloat16, torch::kBFloat16}) {
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        at::autocast::set_autocast_dtype(at::kCUDA, dtype);
        ASSERT_EQ(torch::mm(weights, weights.t()).scalar_type(), dtype);
        auto result = sampler.sampleDSparkDraft(logits, anchors, temperature, {}, {}, weights, weights, 128, true);
        ASSERT_TRUE(torch::equal(result.token_ids, torch::full_like(result.token_ids, 127)));
    }
}
#endif

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
                                         0,
                                         0,
                                         true,
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

#if USING_CUDA
TEST(DSparkConfidenceSamplerTest, CapturedProposalKeepsCurrentConfidenceAndBorrowedTokenStorage) {
    const c10::cuda::CUDAGuard guard(0);
    auto                       input = DSparkConfidenceInputs::make();
    input.sampling_temperature.fill_(1.0e-6f);
    auto               map = torch::tensor({4, 2, 0, 3, 1}, input.anchors.options().dtype(torch::kInt64));
    SpeculativeSampler prepared(map, input.gamma), eager(map, input.gamma);
    prepared.prepareDSparkDraftGraphs(
        input.logits, input.markov_w1, input.markov_w2, input.vocab_size, {static_cast<int>(input.batch_size)});
    const void*   captured_tokens = nullptr;
    torch::Tensor previous_rates;
    torch::Tensor previous_rates_snapshot;
    for (int round = 0; round < 3; ++round) {
        SCOPED_TRACE(round);
        input.anchors.add_(1).remainder_(input.vocab_size);
        input.hidden.add_(0.25f);
        auto         hidden_logits = input.hiddenLogits();
        const double temperature   = 0.75 + round * 0.25;
        auto         run           = [&](const SpeculativeSampler& sampler) {
            return sampler.sampleDSparkDraft(input.logits,
                                             input.anchors,
                                             input.sampling_temperature,
                                                               {},
                                                               {},
                                             input.markov_w1,
                                             input.markov_w2,
                                             input.vocab_size,
                                             true,
                                             0,
                                             0,
                                             false,
                                             hidden_logits,
                                             input.head_weight,
                                             temperature);
        };
        auto expected = run(eager);
        auto actual   = run(prepared);
        if (round == 0) {
            captured_tokens = actual.token_ids.data_ptr();
        } else {
            // A stable borrowed output proves the prepared sampler still used
            // its graph with confidence enabled, rather than the eager fallback.
            EXPECT_EQ(actual.token_ids.data_ptr(), captured_tokens);
            EXPECT_TRUE(torch::equal(previous_rates, previous_rates_snapshot));
        }
        EXPECT_TRUE(torch::equal(actual.token_ids, expected.token_ids));
        ASSERT_TRUE(actual.confidence_rates.defined());
        EXPECT_TRUE(torch::allclose(actual.confidence_rates, expected.confidence_rates, 1.0e-6, 1.0e-6));
        previous_rates          = actual.confidence_rates;
        previous_rates_snapshot = actual.confidence_rates.clone();
    }
}
#endif

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
                                            0,
                                            0,
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
    struct SamplingCase {
        const char* name;
        bool        use_top_k;
        bool        use_top_p;
        int         force_topk;
        int         joint_max_top_k;
    };
    const SamplingCase cases[] = {
        {"strided_q", false, false, 0, 0},
        {"compact_q", false, false, 3, 0},
        {"compact_q_top_p", false, true, 3, 0},
        // Compact prefilter combined with an explicit top-k bounded by the
        // compact width. MtpExecutor::sampleDSparkDraft admits this shape, so
        // the compact and full-vocabulary paths must agree on the retained
        // support and on RNG consumption.
        {"compact_q_top_k", true, false, 4, 0},
        {"compact_q_top_k_top_p", true, true, 4, 0},
        {"separate_top_k_top_p", true, true, 0, 0},
        {"joint_top_k_top_p", true, true, 0, 4},
    };
    for (const auto& sampling_case : cases) {
        SCOPED_TRACE(sampling_case.name);
        auto run = [&](bool observe) {
            return sampler.sampleDSparkDraft(input.logits,
                                             input.anchors,
                                             input.sampling_temperature,
                                             sampling_case.use_top_k ? top_k : torch::Tensor(),
                                             sampling_case.use_top_p ? top_p : torch::Tensor(),
                                             input.markov_w1,
                                             input.markov_w2,
                                             input.vocab_size,
                                             false,
                                             sampling_case.force_topk,
                                             sampling_case.joint_max_top_k,
                                             true,
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
        if (!sampling_case.use_top_k && !sampling_case.use_top_p) {
            const int support = sampling_case.force_topk > 0 ? sampling_case.force_topk : input.vocab_size;
            EXPECT_TRUE(observed.all_probs.gt(0).sum(-1).eq(support).all().item<bool>());
        }
    }
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
                                         0,
                                         0,
                                         true,
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
