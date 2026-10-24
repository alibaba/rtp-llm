#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

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

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
