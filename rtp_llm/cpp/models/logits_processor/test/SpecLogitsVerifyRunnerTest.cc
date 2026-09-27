#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <xgrammar/tokenizer_info.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

namespace rtp_llm {
namespace {

std::string tokenizerInfo() {
    std::vector<std::string> vocab;
    for (int i = 0; i < 128; ++i) {
        vocab.emplace_back(1, static_cast<char>(i));
    }
    return xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW, 128, std::vector<int32_t>{0}).SerializeJSON();
}

// Fixed-budget runners match production: changing k on a live runner is not
// exercised here. Both requests have distinct, position-dependent masks.
class SpecLogitsVerifyRunnerTest: public ::testing::TestWithParam<int> {};

TEST_P(SpecLogitsVerifyRunnerTest, RealGrammarCapsAnchorAndAsyncOwnership) {
    const int                        k        = GetParam();
    constexpr int                    B        = 2;
    constexpr int                    V        = 160;  // Also cover tokens outside the grammar vocabulary.
    const std::array<std::string, B> patterns = {"abcdefgh", "mnopqrst"};
    XGrammarBackendOptions           options;
    options.max_compiler_threads = 1;
    XGrammarBackendCpp backend(tokenizerInfo(), options);
    const auto         pinned_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    const auto         pinned_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU).pinned_memory(true);
    const auto         cuda_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto               producer    = cuda_graph::graphGetStreamFromPool(false);
    auto               consumer    = cuda_graph::graphGetStreamFromPool(false);

    for (bool with_anchor : {false, true}) {
        for (const auto& caps : std::vector<std::array<int, B>>{{0, k}, {k - 1, 0}, {k, k - 1}}) {
            SCOPED_TRACE(::testing::Message()
                         << "k=" << k << " anchor=" << with_anchor << " caps=" << caps[0] << "," << caps[1]);
            SpecLogitsVerifyRunner             runner;
            SpecLogitsVerifyRunner::LaunchTask task;
            task.total_streams = B;
            task.propose_step  = k;
            task.vocab_size    = V;
            std::array<std::shared_ptr<GrammarLogitsProcessor>, B> processors;
            for (int b = 0; b < B; ++b) {
                auto compiled = backend.compileNow({"regex", patterns[b].substr(0, k + 1)}).compiled;
                ASSERT_TRUE(compiled);
                processors[b] =
                    std::make_shared<GrammarLogitsProcessor>(backend.createMatcher(compiled, false, std::nullopt), 0);
                SpecLogitsVerifyRunner::ActiveProcessor active;
                active.processor     = processors[b];
                active.stream_idx    = b;
                active.stream_id     = 100 + b;
                active.processor_idx = b;
                task.active.push_back(active);
            }

            torch::Tensor first_mask;
            torch::Tensor first_cap;
            for (int round = 0; round < 2; ++round) {
                SCOPED_TRACE(::testing::Message() << "round=" << round);
                const int cols   = k + static_cast<int>(with_anchor);
                auto      source = torch::empty({B, cols}, pinned_i32);
                auto*     tokens = source.data_ptr<int32_t>();
                for (int b = 0; b < B; ++b) {
                    if (with_anchor) {
                        tokens[b * cols] = '@';  // Invalid: cap would become zero if not stripped.
                    }
                    for (int p = 0; p < k; ++p) {
                        tokens[b * cols + static_cast<int>(with_anchor) + p] = p == caps[b] ? '#' : patterns[b][p];
                    }
                }
                task.draft_tokens             = torch::empty({B, cols}, cuda_i32);
                task.draft_tokens_ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
                {
                    cuda_graph::GraphStreamGuard guard(producer);
                    task.draft_tokens.copy_(source, /*non_blocking=*/true);
                    task.draft_tokens_ready_event->record(producer);
                }
                auto result = runner.buildInline(task);
                ASSERT_TRUE(result.has_active_processor);
                ASSERT_EQ(result.applied_processors.size(), B);
                ASSERT_TRUE(result.ready_event);
                ASSERT_TRUE(result.consumed_event);
                ASSERT_TRUE(result.spec_vocab_mask_cpu_owner.defined());
                ASSERT_TRUE(result.spec_cap_cpu_owner.defined());
                ASSERT_EQ(result.spec_vocab_mask_gpu.size(0), B * (k + 1));
                ASSERT_EQ(result.spec_vocab_mask_gpu.size(1), V);
                ASSERT_EQ(result.spec_cap_gpu.numel(), B);

                auto mask = torch::empty({B * (k + 1), V}, pinned_bool);
                auto cap  = torch::empty({B}, pinned_i32);
                {
                    cuda_graph::GraphStreamGuard guard(consumer);
                    result.ready_event->block(consumer);
                    mask.copy_(result.spec_vocab_mask_gpu, /*non_blocking=*/true);
                    cap.copy_(result.spec_cap_gpu, /*non_blocking=*/true);
                    result.consumed_event->record(consumer);
                }
                // Keep result (including pinned H2D owners) and source alive
                // until the consumer's copies have completed. No global sync.
                result.consumed_event->synchronize();
                for (int b = 0; b < B; ++b) {
                    EXPECT_EQ(cap.data_ptr<int32_t>()[b], caps[b]);
                    EXPECT_EQ(processors[b]->acceptedTokenLen(), 0);
                    for (int p = 0; p <= k; ++p) {
                        for (int token = 0; token < V; ++token) {
                            // Rows beyond the rejection cap are unvisited and
                            // remain all-allow; row cap itself must be valid.
                            const bool expected_mask = p <= caps[b] && token != patterns[b][p];
                            EXPECT_EQ(mask.data_ptr<bool>()[(b * (k + 1) + p) * V + token], expected_mask)
                                << "batch=" << b << " offset=" << p << " token=" << token;
                        }
                    }
                }
                if (round == 0) {
                    first_mask = mask;
                    first_cap  = cap;
                } else {
                    EXPECT_TRUE(torch::equal(first_mask, mask));
                    EXPECT_TRUE(torch::equal(first_cap, cap));
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ShortVerifyBudgets, SpecLogitsVerifyRunnerTest, ::testing::Values(1, 4, 7));

}  // namespace
}  // namespace rtp_llm
