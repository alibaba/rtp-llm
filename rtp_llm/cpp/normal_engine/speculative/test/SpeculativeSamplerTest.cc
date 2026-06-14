#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

#include <gtest/gtest.h>
#include <memory>
#include <torch/all.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/testing/TestBase.h"

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

// Cover the tensorized post-kernel path: CPU→GPU mask + torch::where stitching.

class SpeculativeSamplerTensorizedTest: public DeviceTestBase {
protected:
    GenerateStreamPtr makeStream(bool force_accept) {
        ModelConfig     model_config;
        RuntimeConfig   runtime_config;
        ResourceContext resource_context;
        model_config.max_seq_len = 64;
        model_config.vocab_size  = 4;
        model_config.num_layers  = 1;

        auto query                              = std::make_shared<GenerateInput>();
        query->input_ids                        = torch::tensor(std::vector<int32_t>{1, 2, 3}, torch::kInt32);
        query->generate_config                  = std::make_shared<GenerateConfig>();
        query->generate_config->force_sp_accept = force_accept;
        auto stream =
            std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->setNeedReleaseResource(false);
        return stream;
    }
};

TEST_F(SpeculativeSamplerTensorizedTest, SingleStreamAcceptTokensReturnsLocalBuffer) {
    constexpr size_t   propose_step = 2;
    SpeculativeSampler sampler(propose_step);

    auto streams = std::list<GenerateStreamPtr>{makeStream(/*force_accept=*/false)};

    const int batch_size = 1;
    const int vocab      = 4;

    // Both probs peaked at token 1; assert only post-kernel invariants (shape, range, CPU mirror).
    auto cuda_f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);
    auto cuda_i32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);

    SamplerOutput draft_out;
    draft_out.token_ids = torch::tensor({{1, 1}}, cuda_i32);
    draft_out.all_probs = torch::zeros({batch_size, (long)propose_step, vocab}, cuda_f32);
    draft_out.all_probs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 1}, 1.0f);

    SamplerOutput target_out;
    target_out.token_ids = torch::tensor({{1, 1, 1}}, cuda_i32);
    target_out.all_probs = torch::zeros({batch_size, (long)(propose_step + 1), vocab}, cuda_f32);
    target_out.all_probs.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 1}, 1.0f);
    target_out.all_probs = target_out.all_probs.reshape({batch_size * (long)(propose_step + 1), vocab});

    auto out = sampler.forward(streams, draft_out, target_out);

    ASSERT_TRUE(out.accept_tokens.defined());
    ASSERT_EQ(out.accept_tokens.dim(), 2);
    EXPECT_EQ(out.accept_tokens.size(0), batch_size);
    EXPECT_EQ(out.accept_tokens.size(1), (long)(propose_step + 1));
    EXPECT_TRUE(out.accept_tokens.is_cuda());

    ASSERT_TRUE(out.accept_len.defined());
    ASSERT_EQ(out.accept_len.dim(), 1);
    EXPECT_EQ(out.accept_len.size(0), batch_size);

    // Wait for async H2D before reading CPU mirrors.
    out.transfer_done_event->synchronize();
    auto len_cpu = out.accept_len_cpu.contiguous();
    ASSERT_EQ(len_cpu.numel(), batch_size);
    int len = len_cpu.data_ptr<int32_t>()[0];
    EXPECT_GE(len, 1);
    EXPECT_LE(len, (int)(propose_step + 1));

    auto tok_cpu = out.accept_tokens_cpu.contiguous();
    ASSERT_EQ(tok_cpu.numel(), batch_size * (long)(propose_step + 1));
}

TEST_F(SpeculativeSamplerTensorizedTest, AcceptLenBoundedByProposeStepPlusOne) {
    // Regression for the index_put_ col_idx clamp: even if the kernel returns
    // an emitted_token_num at the upper bound, col_idx = clamp(len-1, 0, P)
    // must stay within [0, P], so accept_tokens shape stays [B, P+1].
    constexpr size_t   propose_step = 3;
    SpeculativeSampler sampler(propose_step);

    auto streams = std::list<GenerateStreamPtr>{makeStream(false), makeStream(false)};

    const int batch_size = 2;
    const int vocab      = 4;

    auto cuda_f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);
    auto cuda_i32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);

    SamplerOutput draft_out;
    draft_out.token_ids = torch::tensor({{1, 2, 3}, {3, 2, 1}}, cuda_i32);
    draft_out.all_probs = torch::ones({batch_size, (long)propose_step, vocab}, cuda_f32) / vocab;

    SamplerOutput target_out;
    target_out.token_ids = torch::tensor({{1, 2, 3, 0}, {3, 2, 1, 0}}, cuda_i32);
    target_out.all_probs = (torch::ones({batch_size, (long)(propose_step + 1), vocab}, cuda_f32) / vocab)
                               .reshape({batch_size * (long)(propose_step + 1), vocab});

    auto out = sampler.forward(streams, draft_out, target_out);
    out.transfer_done_event->synchronize();

    auto len_cpu = out.accept_len_cpu.contiguous();
    ASSERT_EQ(len_cpu.numel(), batch_size);
    for (int b = 0; b < batch_size; ++b) {
        int len = len_cpu.data_ptr<int32_t>()[b];
        EXPECT_GE(len, 1);
        EXPECT_LE(len, (int)(propose_step + 1));
    }

    auto tok_cpu = out.accept_tokens_cpu.contiguous();
    EXPECT_EQ(tok_cpu.numel(), batch_size * (long)(propose_step + 1));
}

#endif  // USING_CUDA

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
