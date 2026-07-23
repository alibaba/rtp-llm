#include <chrono>
#include <cstring>
#include <memory>
#include <limits>
#include "torch/all.h"
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

#define private public
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#undef private
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

template<typename T>
std::vector<T> toVec(const torch::Tensor& t) {
    auto c = t.is_cuda() ? t.cpu().contiguous() : t.contiguous();
    return std::vector<T>(c.data_ptr<T>(), c.data_ptr<T>() + c.numel());
}

void fillScoreTokenIdsWithMemcpy(torch::Tensor&                    token_ids,
                                 const std::vector<torch::Tensor>& complete_token_ids,
                                 const std::vector<int64_t>&       seq_lens,
                                 int64_t                           score_len) {
    int64_t    batch_idx  = 0;
    auto*      dst        = token_ids.data_ptr<int32_t>();
    const auto dst_stride = token_ids.size(1);
    for (size_t stream_idx = 0; stream_idx < complete_token_ids.size(); ++stream_idx) {
        auto* src     = complete_token_ids[stream_idx].data_ptr<int32_t>();
        auto  seq_len = seq_lens[stream_idx];
        for (int64_t i = 0; i < score_len; ++i) {
            std::memcpy(dst + batch_idx * dst_stride, src, seq_len * sizeof(int32_t));
            ++batch_idx;
        }
    }
}

void fillScoreTokenIdsWithTorchCopy(torch::Tensor&                    token_ids,
                                    const std::vector<torch::Tensor>& complete_token_ids,
                                    const std::vector<int64_t>&       seq_lens,
                                    int64_t                           score_len) {
    int64_t batch_idx = 0;
    for (size_t stream_idx = 0; stream_idx < complete_token_ids.size(); ++stream_idx) {
        auto seq_len = seq_lens[stream_idx];
        token_ids.narrow(0, batch_idx, score_len)
            .narrow(1, 0, seq_len)
            .copy_(complete_token_ids[stream_idx].narrow(1, 0, seq_len).expand({score_len, seq_len}));
        batch_idx += score_len;
    }
}

template<typename Func>
double benchmarkUs(Func&& func, int iterations) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

class MtpBatchStreamProcessorTest: public DeviceTestBase {
public:
    void setSpOutputTokens(const SpeculativeExecutorStreamOutputPtr& sp_output_buffer, const vector<int>& token_ids) {
        std::vector<int32_t> token_ids_i32(token_ids.begin(), token_ids.end());
        sp_output_buffer->tokens = torch::tensor(token_ids_i32, torch::TensorOptions().dtype(torch::kInt32))
                                       .reshape({1, (int64_t)token_ids_i32.size()});

        const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        sp_output_buffer->target_token_gpu =
            sp_output_buffer->tokens.narrow(1, 0, 1).to(cuda_i32, /*non_blocking=*/true);
        if (token_ids_i32.size() > 1) {
            sp_output_buffer->propose_tokens_gpu =
                sp_output_buffer->tokens.narrow(1, 1, (int64_t)token_ids_i32.size() - 1)
                    .to(cuda_i32, /*non_blocking=*/true);
        } else {
            sp_output_buffer->propose_tokens_gpu = torch::empty({1, 0}, cuda_i32);
        }
    }

    GenerateStreamPtr createContextStream(const ModelConfig&     model_config,
                                          const RuntimeConfig&   runtime_config,
                                          const ResourceContext& resource_context,
                                          const vector<int>&     input_ids,
                                          const int              block_id,
                                          const vector<int>&     begin_think_token_ids = {},
                                          const vector<int>&     end_think_token_ids   = {},
                                          bool                   return_logprobs       = false,
                                          int                    top_logprobs          = 0,
                                          bool                   in_think_mode         = false) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids       = torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        query->generate_config = make_shared<GenerateConfig>();
        query->generate_config->begin_think_token_ids = begin_think_token_ids;
        query->generate_config->end_think_token_ids   = end_think_token_ids;
        query->generate_config->return_logprobs       = return_logprobs;
        query->generate_config->top_logprobs          = top_logprobs;
        query->generate_config->in_think_mode         = in_think_mode;
        GenerateStreamPtr stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        BatchKVCacheResource addr;
        // New (refactored) BatchKVCacheResource: [batch_id][group_id] -> block_indices
        addr.resetBatchSize(1);
        addr.initGroups(1, 1, {0});
        addr.setBatchBlocks(0, 0, {block_id});
        stream->setKVCache(addr);

        auto        sp_output_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
        vector<int> propose_tokens   = vector<int>(2, -1);
        setSpOutputTokens(sp_output_buffer, propose_tokens);
        stream->setReturnAllProbs(true);
        stream->setSPOutputBuffer(sp_output_buffer);
        stream->generate_status_->status = StreamState::RUNNING;
        stream->setNeedReleaseResource(false);

        return stream;
    }

    void checkOutput(const GenerateStreamPtr& stream,
                     const vector<int>&       expect_token_ids,
                     const vector<int>&       expect_propose_tokens,
                     const vector<float>&     expect_all_probs,
                     const vector<float>&     expect_last_hidden_states) {
        auto token_ids = stream->getCompleteTokenIds()->completeTokenIdsVec(0);
        EXPECT_EQ(expect_token_ids, token_ids);

        auto sp_output_buffer = stream->getSPOutputBuffer();
        auto tokens           = sp_output_buffer->tokens;
        auto tokens_h         = tokens.cpu().clone();
        EXPECT_EQ(expect_propose_tokens, toVec<int>(tokens_h));

        auto all_probs   = sp_output_buffer->all_probs;
        auto all_probs_h = all_probs.is_cuda() ? all_probs.cpu() : all_probs;
        EXPECT_EQ(expect_all_probs, toVec<float>(all_probs_h));

        auto last_hidden_states   = sp_output_buffer->hidden_states;
        auto last_hidden_states_h = last_hidden_states.is_cuda() ? last_hidden_states.cpu() : last_hidden_states;
        EXPECT_EQ(expect_last_hidden_states, toVec<float>(last_hidden_states_h));
    }
};

class TestableMtpBatchStreamProcessor: public MtpBatchStreamProcessor {
public:
    using MtpBatchStreamProcessor::MtpBatchStreamProcessor;
    using MtpBatchStreamProcessor::prepareDecodeSpecUpdateInfo;
    using MtpBatchStreamProcessor::preparePrefillSpecUpdateInfo;
};

TEST_F(MtpBatchStreamProcessorTest, testPrefillTargetLogprobsUseEmittedFirstToken) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};

    model_config.max_seq_len    = 32;
    model_config.vocab_size     = 5;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    ResourceContext resource_context;
    auto            stream = createContextStream(
        model_config, runtime_config, resource_context, {1}, 1, {}, {}, /*return_logprobs=*/true, /*top_logprobs=*/2);
    StreamGroups                    stream_groups({stream});
    TestableMtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    MergedOutput target_output;
    // The emitted token is the last sampler column (token 2), which is not the
    // raw argmax. The selected logprob must still be gathered for token 2.
    target_output.sampler_output.token_ids = torch::tensor({4, 2}, torch::kInt32).reshape({1, 2});

    MergedOutput draft_output;
    draft_output.sampler_output.token_ids = torch::tensor({3L}, torch::kInt64).reshape({1, 1});
    draft_output.sampler_output.all_probs =
        torch::tensor({0.05f, 0.15f, 0.20f, 0.25f, 0.35f}, torch::kFloat32).reshape({1, 5});

    auto raw_logits      = torch::tensor({3.0f, -1.0f, 0.5f, 2.0f, 1.0f}, torch::kFloat32).reshape({1, 5});
    auto target_logprobs = computeMtpTargetLogprobs(raw_logits, 2, /*real_vocab_size=*/5);
    finalizeMtpTargetLogprobs(target_logprobs, torch::tensor({2}, torch::kInt32));
    auto new_tokens_all = torch::empty({1, 1}, torch::kInt32);

    std::vector<StreamSpecUpdateInfo> update_infos;
    processor.preparePrefillSpecUpdateInfo(
        stream_groups, target_output, draft_output, torch::Tensor(), target_logprobs, new_tokens_all, update_infos);

    ASSERT_EQ(update_infos.size(), 1);
    EXPECT_EQ(toVec<int32_t>(update_infos[0].new_tokens), std::vector<int32_t>({2}));
    EXPECT_TRUE(
        torch::allclose(update_infos[0].token_logprobs.cpu(), target_logprobs.token_logprobs.reshape({1, 1}).cpu()));
    EXPECT_TRUE(torch::equal(update_infos[0].top_logprob_token_ids.cpu(),
                             target_logprobs.top_logprob_token_ids.reshape({1, 1, 2}).cpu()));
    EXPECT_TRUE(
        torch::allclose(update_infos[0].top_logprobs.cpu(), target_logprobs.top_logprobs.reshape({1, 1, 2}).cpu()));
    // Draft probabilities remain the independent next-step MTP distribution.
    EXPECT_TRUE(torch::allclose(update_infos[0].draft_token_probs.cpu(), draft_output.sampler_output.all_probs));
}

TEST_F(MtpBatchStreamProcessorTest, testPrefillThinkingTokenNeedsNoTargetLogprobPayload) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types    = {CacheGroupType::FULL};
    model_config.max_seq_len    = 32;
    model_config.vocab_size     = 8;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    ResourceContext resource_context;
    auto            stream                        = createContextStream(model_config,
                                      runtime_config,
                                      resource_context,
                                                                        {0},
                                      1,
                                      /*begin_think_token_ids=*/{7},
                                      /*end_think_token_ids=*/{3, 4},
                                      /*return_logprobs=*/true,
                                      /*top_logprobs=*/2,
                                      /*in_think_mode=*/true);
    stream->generateConfig()->max_thinking_tokens = 10;
    StreamGroups                    stream_groups({stream});
    TestableMtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    MergedOutput target_output;
    target_output.sampler_output.token_ids = torch::tensor({0, 2}, torch::kInt32).reshape({1, 2});
    MergedOutput draft_output;
    draft_output.sampler_output.token_ids = torch::tensor({1L}, torch::kInt64).reshape({1, 1});
    draft_output.sampler_output.all_probs = torch::full({1, 8}, 0.125f, torch::kFloat32);

    std::vector<StreamSpecUpdateInfo> update_infos;
    auto                              new_tokens_all = torch::empty({1, 1}, torch::kInt32);
    processor.preparePrefillSpecUpdateInfo(stream_groups,
                                           target_output,
                                           draft_output,
                                           torch::Tensor(),
                                           /*target_logprobs=*/{},
                                           new_tokens_all,
                                           update_infos);

    ASSERT_EQ(update_infos.size(), 1);
    EXPECT_EQ(update_infos[0].logprobs_offset, 1);
    EXPECT_FALSE(update_infos[0].token_logprobs.defined());
    EXPECT_FALSE(update_infos[0].top_logprob_token_ids.defined());
    EXPECT_FALSE(update_infos[0].top_logprobs.defined());
}

TEST_F(MtpBatchStreamProcessorTest, testDecodeTargetLogprobsCoverReplacementAcceptedAndBonusTokens) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};

    model_config.max_seq_len    = 32;
    model_config.vocab_size     = 5;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 2;

    ResourceContext resource_context;
    auto            stream_replacement =
        createContextStream(model_config, runtime_config, resource_context, {0}, 1, {}, {}, true, /*top_logprobs=*/1);
    auto stream_partial =
        createContextStream(model_config, runtime_config, resource_context, {1}, 2, {}, {}, true, /*top_logprobs=*/2);
    auto stream_bonus =
        createContextStream(model_config, runtime_config, resource_context, {2}, 3, {}, {}, true, /*top_logprobs=*/3);
    StreamGroups                    stream_groups({stream_replacement, stream_partial, stream_bonus});
    TestableMtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    speculative::SpeculativeSamplerOutput spec_output;
    // P=2 target positions per stream. Length 1 is an immediate replacement,
    // length 2 is accepted+replacement, and length 3 includes the bonus row.
    spec_output.accept_len_cpu    = torch::tensor({1, 2, 3}, torch::kInt32);
    spec_output.accept_tokens_cpu = torch::tensor({{4, 0, 0}, {3, 2, 0}, {1, 0, 4}}, torch::kInt32);
    spec_output.accept_len        = spec_output.accept_len_cpu.to(torch::kCUDA);
    spec_output.accept_tokens     = spec_output.accept_tokens_cpu.to(torch::kCUDA);

    MergedOutput draft_prefill_output;
    draft_prefill_output.model_output.all_hidden_states = torch::arange(0, 9, torch::kFloat32).reshape({9, 1});
    draft_prefill_output.sampler_output.token_ids       = torch::tensor({0L, 1L, 2L}, torch::kInt64).reshape({3, 1});
    draft_prefill_output.sampler_output.all_probs       = torch::tensor(
        {{0.10f, 0.20f, 0.30f, 0.15f, 0.25f}, {0.30f, 0.10f, 0.20f, 0.25f, 0.15f}, {0.15f, 0.25f, 0.10f, 0.20f, 0.30f}},
        torch::kFloat32);

    auto target_logits   = torch::tensor({{5.0f, 1.0f, 0.0f, 2.0f, 3.0f},
                                          {0.0f, 5.0f, 1.0f, 3.0f, 2.0f},
                                          {1.0f, 0.0f, 5.0f, 2.0f, 3.0f},
                                          {2.0f, 1.0f, 0.0f, 5.0f, 3.0f},
                                          {3.0f, 2.0f, 1.0f, 0.0f, 5.0f},
                                          {4.0f, 0.0f, 3.0f, 2.0f, 1.0f},
                                          {1.0f, 4.0f, 0.0f, 3.0f, 2.0f},
                                          {4.0f, 3.0f, 2.0f, 1.0f, 0.0f},
                                          {0.0f, 1.0f, 2.0f, 3.0f, 4.0f}},
                                       torch::kFloat32);
    auto target_logprobs = captureMtpTargetLogprobs(target_logits, 3, /*real_vocab_size=*/5);
    EXPECT_FALSE(target_logprobs.row_max.defined());
    EXPECT_FALSE(target_logprobs.row_shifted_logsumexp.defined());
    EXPECT_FALSE(target_logprobs.top_logits.defined());
    EXPECT_FALSE(shouldFinalizeMtpTargetLogprobsEarly(/*stream_async_enabled=*/false, target_logprobs));
    EXPECT_TRUE(shouldFinalizeMtpTargetLogprobsEarly(/*stream_async_enabled=*/true, target_logprobs));

    // Mirror the stream-async identity path: finalize accepted rows before the
    // regular bookkeeping worker assembles StreamSpecUpdateInfo. The latter
    // must consume this compact payload without attempting a second finalize.
    processor.finalizeDecodeTargetLogprobs(stream_groups, spec_output, target_logprobs);
    ASSERT_TRUE(target_logprobs.finalized());
    ASSERT_FALSE(target_logprobs.retainsFullLmHeadStorage());
    auto early_token_logprobs = target_logprobs.token_logprobs.clone();
    auto early_top_ids        = target_logprobs.top_logprob_token_ids.clone();
    auto early_top_logprobs   = target_logprobs.top_logprobs.clone();

    std::vector<StreamSpecUpdateInfo> update_infos;
    processor.prepareDecodeSpecUpdateInfo(
        stream_groups, spec_output, draft_prefill_output, target_logprobs, update_infos);

    ASSERT_EQ(update_infos.size(), 3);
    ASSERT_TRUE(target_logprobs.finalized());
    EXPECT_TRUE(torch::equal(target_logprobs.token_logprobs, early_token_logprobs));
    EXPECT_TRUE(torch::equal(target_logprobs.top_logprob_token_ids, early_top_ids));
    EXPECT_TRUE(torch::equal(target_logprobs.top_logprobs, early_top_logprobs));
    // Only 1+2+3 accepted rows are reduced, rather than all 3*(P+1)=9.
    EXPECT_EQ(target_logprobs.token_logprobs.size(0), 6);
    const std::vector<int64_t> row_offsets = {0, 1, 3};
    const std::vector<int64_t> lengths     = {1, 2, 3};
    const std::vector<int64_t> top_ks      = {1, 2, 3};
    for (size_t i = 0; i < update_infos.size(); ++i) {
        auto expected_selected =
            target_logprobs.token_logprobs.narrow(0, row_offsets[i], lengths[i]).reshape({1, lengths[i]});
        auto expected_top_ids = target_logprobs.top_logprob_token_ids.narrow(0, row_offsets[i], lengths[i])
                                    .reshape({1, lengths[i], 3})
                                    .narrow(2, 0, top_ks[i]);
        auto expected_top_logprobs = target_logprobs.top_logprobs.narrow(0, row_offsets[i], lengths[i])
                                         .reshape({1, lengths[i], 3})
                                         .narrow(2, 0, top_ks[i]);

        EXPECT_EQ(update_infos[i].num_new_tokens, lengths[i]);
        EXPECT_TRUE(torch::allclose(update_infos[i].token_logprobs.cpu(), expected_selected.cpu()));
        EXPECT_TRUE(torch::equal(update_infos[i].top_logprob_token_ids.cpu(), expected_top_ids.cpu()));
        EXPECT_TRUE(torch::allclose(update_infos[i].top_logprobs.cpu(), expected_top_logprobs.cpu()));
        EXPECT_TRUE(torch::allclose(update_infos[i].draft_token_probs.cpu(),
                                    draft_prefill_output.sampler_output.all_probs.narrow(0, i, 1)));
    }
}

TEST_F(MtpBatchStreamProcessorTest, testDecodeTargetLogprobsKeepOnlyContentSuffixInMixedBatch) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types    = {CacheGroupType::FULL};
    model_config.max_seq_len    = 32;
    model_config.vocab_size     = 8;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 5;

    ResourceContext resource_context;
    auto            reasoning_only = createContextStream(model_config,
                                              runtime_config,
                                              resource_context,
                                                         {0},
                                              1,
                                              /*begin_think_token_ids=*/{7},
                                              /*end_think_token_ids=*/{3, 4},
                                              /*return_logprobs=*/true,
                                              /*top_logprobs=*/20,
                                              /*in_think_mode=*/true);
    auto            crossing       = createContextStream(model_config,
                                        runtime_config,
                                        resource_context,
                                                         {0},
                                        2,
                                        /*begin_think_token_ids=*/{7},
                                        /*end_think_token_ids=*/{3, 4},
                                        /*return_logprobs=*/true,
                                        /*top_logprobs=*/0,
                                        /*in_think_mode=*/true);
    auto            content        = createContextStream(
        model_config, runtime_config, resource_context, {0}, 3, {}, {}, /*return_logprobs=*/true, /*top_logprobs=*/1);
    reasoning_only->generateConfig()->max_thinking_tokens = 10;
    crossing->generateConfig()->max_thinking_tokens       = 10;

    StreamGroups                    stream_groups({reasoning_only, crossing, content});
    TestableMtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    speculative::SpeculativeSamplerOutput spec_output;
    // The close token is end_think_token_ids.front() == 3.  It is still a
    // reasoning token; the delimiter tail 4 and token 5 are content.
    spec_output.accept_len_cpu = torch::tensor({2, 5, 1}, torch::kInt32);
    spec_output.accept_tokens_cpu =
        torch::tensor({{1, 2, 0, 0, 0, 0}, {1, 2, 3, 4, 5, 0}, {6, 0, 0, 0, 0, 0}}, torch::kInt32);
    spec_output.accept_len    = spec_output.accept_len_cpu.to(torch::kCUDA);
    spec_output.accept_tokens = spec_output.accept_tokens_cpu.to(torch::kCUDA);

    MergedOutput draft_output;
    draft_output.model_output.all_hidden_states = torch::arange(0, 18, torch::kFloat32).reshape({18, 1});
    draft_output.sampler_output.token_ids       = torch::tensor({0L, 1L, 2L}, torch::kInt64).reshape({3, 1});
    draft_output.sampler_output.all_probs       = torch::full({3, 8}, 0.125f, torch::kFloat32);

    auto target_logits   = torch::arange(0, 144, torch::kFloat32).reshape({18, 8});
    auto target_logprobs = captureMtpDecodeTargetLogprobs(target_logits, /*max_top_logprobs=*/8, /*real_vocab_size=*/8);
    processor.finalizeDecodeTargetLogprobs(stream_groups, spec_output, target_logprobs);

    ASSERT_TRUE(target_logprobs.finalized());
    ASSERT_FALSE(target_logprobs.retainsFullLmHeadStorage());
    // Dense selected rows are crossing[3,4] => [9,10], then content[0] => 12.
    ASSERT_EQ(target_logprobs.token_logprobs.size(0), 3);
    // The thinking-only k=20 stream contributes no row. The crossing stream
    // requests k=0 and the content stream requests k=1, so the shared compact
    // reduction must use width 1 rather than the capture-time/global width 8.
    EXPECT_EQ(target_logprobs.maxTopLogprobs(), 1);
    auto expected_rows   = torch::tensor({9, 10, 12}, torch::kInt64);
    auto expected_logits = target_logits.index_select(0, expected_rows);
    auto expected_ids    = torch::tensor({4, 5, 6}, torch::kInt64);
    auto expected        = torch::log_softmax(expected_logits, -1).gather(1, expected_ids.unsqueeze(1)).squeeze(1);
    EXPECT_TRUE(torch::allclose(target_logprobs.token_logprobs.cpu(), expected.cpu()));

    std::vector<StreamSpecUpdateInfo> update_infos;
    processor.prepareDecodeSpecUpdateInfo(stream_groups, spec_output, draft_output, target_logprobs, update_infos);
    ASSERT_EQ(update_infos.size(), 3);
    EXPECT_EQ(update_infos[0].logprobs_offset, 2);
    EXPECT_FALSE(update_infos[0].token_logprobs.defined());
    EXPECT_EQ(update_infos[1].logprobs_offset, 3);
    EXPECT_EQ(update_infos[1].token_logprobs.sizes(), (torch::IntArrayRef{1, 2}));
    EXPECT_EQ(update_infos[1].top_logprobs.sizes(), (torch::IntArrayRef{1, 2, 0}));
    EXPECT_EQ(update_infos[2].logprobs_offset, 0);
    EXPECT_EQ(update_infos[2].token_logprobs.sizes(), (torch::IntArrayRef{1, 1}));
    EXPECT_EQ(update_infos[2].top_logprobs.sizes(), (torch::IntArrayRef{1, 1, 1}));
}

TEST_F(MtpBatchStreamProcessorTest, testPrefillMaxTopLogprobsIgnoresThinkingStreams) {
    ModelConfig     model_config;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;
    model_config.max_seq_len = 32;
    model_config.vocab_size  = 32;

    auto thinking   = createContextStream(model_config,
                                        runtime_config,
                                        resource_context,
                                          {0},
                                        1,
                                        /*begin_think_token_ids=*/{7},
                                        /*end_think_token_ids=*/{3, 4},
                                        /*return_logprobs=*/true,
                                        /*top_logprobs=*/20,
                                        /*in_think_mode=*/true);
    auto content_k0 = createContextStream(
        model_config, runtime_config, resource_context, {0}, 2, {}, {}, /*return_logprobs=*/true, /*top_logprobs=*/0);
    auto content_k1 = createContextStream(
        model_config, runtime_config, resource_context, {0}, 3, {}, {}, /*return_logprobs=*/true, /*top_logprobs=*/1);

    EXPECT_EQ(maxMtpActiveContentTopLogprobs(StreamGroups({thinking, content_k0})), 0);
    EXPECT_EQ(maxMtpActiveContentTopLogprobs(StreamGroups({thinking, content_k0, content_k1})), 1);
}

TEST_F(MtpBatchStreamProcessorTest, testDecodeTargetLogprobsDeferMixedCompactionUntilAcceptance) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types    = {CacheGroupType::FULL};
    model_config.max_seq_len    = 32;
    model_config.vocab_size     = 5;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    ResourceContext resource_context;
    auto plain_stream   = createContextStream(model_config, runtime_config, resource_context, {0}, 1, {}, {}, false, 0);
    auto logprob_stream = createContextStream(model_config, runtime_config, resource_context, {1}, 2, {}, {}, true, 2);
    StreamGroups                    stream_groups({plain_stream, logprob_stream});
    TestableMtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    speculative::SpeculativeSamplerOutput spec_output;
    spec_output.accept_len_cpu    = torch::tensor({2, 1}, torch::kInt32);
    spec_output.accept_tokens_cpu = torch::tensor({{1, 2}, {3, 0}}, torch::kInt32);
    spec_output.accept_len        = spec_output.accept_len_cpu.to(torch::kCUDA);
    spec_output.accept_tokens     = spec_output.accept_tokens_cpu.to(torch::kCUDA);

    MergedOutput draft_output;
    draft_output.sampler_output.token_ids = torch::tensor({0L, 1L}, torch::kInt64).reshape({2, 1});
    draft_output.sampler_output.all_probs = torch::full({2, 5}, 0.2f, torch::kFloat32);

    auto dense_logits    = torch::tensor({{5.0f, 1.0f, 0.0f, 2.0f, 3.0f},
                                          {0.0f, 5.0f, 1.0f, 3.0f, 2.0f},
                                          {1.0f, 0.0f, 5.0f, 4.0f, 3.0f},
                                          {2.0f, 1.0f, 0.0f, 5.0f, 3.0f}},
                                      torch::kFloat32);
    auto target_logprobs = captureMtpDecodeTargetLogprobs(dense_logits, /*max_top_logprobs=*/2, /*real_vocab_size=*/5);
    EXPECT_EQ(target_logprobs.raw_logits.size(0), 4);
    EXPECT_EQ(target_logprobs.dense_row_count, 4);
    EXPECT_EQ(target_logprobs.raw_logits.data_ptr<float>(), dense_logits.data_ptr<float>());
    EXPECT_TRUE(target_logprobs.captured_dense_row_indices.empty());
    EXPECT_TRUE(target_logprobs.retainsFullLmHeadStorage());
    EXPECT_TRUE(shouldFinalizeMtpTargetLogprobsEarly(/*stream_async_enabled=*/true, target_logprobs));
    EXPECT_FALSE(target_logprobs.row_max.defined());
    EXPECT_FALSE(target_logprobs.row_shifted_logsumexp.defined());
    EXPECT_FALSE(target_logprobs.top_logits.defined());

    // The mixed decode capture stays zero-copy until acceptance. Early finalize
    // then reduces only the one accepted row belonging to the requesting
    // stream; the regular worker consumes the compact result idempotently.
    processor.finalizeDecodeTargetLogprobs(stream_groups, spec_output, target_logprobs);
    ASSERT_TRUE(target_logprobs.finalized());
    ASSERT_FALSE(target_logprobs.retainsFullLmHeadStorage());

    std::vector<StreamSpecUpdateInfo> update_infos;
    processor.prepareDecodeSpecUpdateInfo(stream_groups, spec_output, draft_output, target_logprobs, update_infos);

    ASSERT_EQ(update_infos.size(), 2);
    ASSERT_TRUE(target_logprobs.finalized());
    // The plain request contributes no rows, and the logprob request accepted
    // only its first row. Its dense source row is 2, proving selection was
    // deferred until acceptance rather than captured as a [2,V] mixed copy.
    EXPECT_EQ(target_logprobs.token_logprobs.size(0), 1);
    EXPECT_FALSE(update_infos[0].token_logprobs.defined());
    EXPECT_FALSE(update_infos[0].top_logprob_token_ids.defined());
    EXPECT_FALSE(update_infos[0].top_logprobs.defined());
    ASSERT_TRUE(update_infos[1].token_logprobs.defined());
    EXPECT_TRUE(torch::allclose(update_infos[1].token_logprobs,
                                target_logprobs.token_logprobs.narrow(0, 0, 1).reshape({1, 1})));
    EXPECT_TRUE(torch::equal(update_infos[1].top_logprob_token_ids,
                             target_logprobs.top_logprob_token_ids.narrow(0, 0, 1).reshape({1, 1, 2})));
    auto expected = torch::log_softmax(dense_logits.narrow(0, 2, 1), -1);
    EXPECT_TRUE(
        torch::allclose(target_logprobs.token_logprobs,
                        expected.index({0, spec_output.accept_tokens_cpu.index({1, 0}).item<int64_t>()}).reshape({1})));
}

TEST_F(MtpBatchStreamProcessorTest, testDispatchPrefillStagesCompactLogprobsToCpuAsOneBatch) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types    = {CacheGroupType::FULL};
    model_config.max_seq_len    = 32;
    model_config.vocab_size     = 5;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    ResourceContext resource_context;
    auto            stream = createContextStream(
        model_config, runtime_config, resource_context, {1}, 1, {}, {}, /*return_logprobs=*/true, /*top_logprobs=*/2);
    StreamGroups            stream_groups({stream});
    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    MergedOutput target_output;
    target_output.sampler_output.token_ids =
        torch::tensor({4, 2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)).reshape({1, 2});

    MergedOutput draft_output;
    draft_output.sampler_output.token_ids =
        torch::tensor({3L}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)).reshape({1, 1});
    draft_output.sampler_output.all_probs =
        torch::full({1, 5}, 0.2f, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto raw_logits = torch::tensor({3.0f, -1.0f, 0.5f, 2.0f, 1.0f},
                                    torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA))
                          .reshape({1, 5});
    auto expected        = torch::log_softmax(raw_logits.to(torch::kFloat32), -1);
    auto target_logprobs = computeMtpTargetLogprobs(raw_logits, 2, /*real_vocab_size=*/5);
    finalizeMtpTargetLogprobs(target_logprobs, target_output.sampler_output.token_ids.select(1, 1));

    auto status =
        processor.dispatchPrefill(stream_groups, target_output, draft_output, torch::Tensor(), target_logprobs);
    ASSERT_TRUE(status.ok()) << status.ToString();
    ASSERT_FALSE(stream->hasError());
    EXPECT_FALSE(stream->getTokenLogProbs().is_cuda());
    EXPECT_FALSE(stream->getTopLogprobTokenIds().is_cuda());
    EXPECT_FALSE(stream->getTopLogProbs().is_cuda());
    EXPECT_NEAR(stream->getTokenLogProbs().index({0, 0}).item<float>(), expected.index({0, 2}).item<float>(), 1e-3);
}

TEST_F(MtpBatchStreamProcessorTest, testSpecUpdateLogprobsFollowStopTruncation) {
    ModelConfig   model_config;
    RuntimeConfig runtime_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 8;
    model_config.num_layers  = 1;

    ResourceContext resource_context;
    // specUpdate consults cache geometry before stop-word truncation. Supply
    // the same minimal manager used by the existing decode tests; no cache
    // allocation is needed because this MHA-only config has no linear blocks.
    resource_context.cache_manager =
        std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                                        /*block_num=*/10,
                                                                        /*tokens_per_block=*/2,
                                                                        rtp_llm::TYPE_INT8));
    auto stream                               = createContextStream(model_config,
                                      runtime_config,
                                      resource_context,
                                                                    {0},
                                      1,
                                      /*begin_think_token_ids=*/{6},
                                      /*end_think_token_ids=*/{7},
                                      /*return_logprobs=*/true,
                                      /*top_logprobs=*/1);
    stream->generateConfig()->stop_words_list = {{2}};

    auto new_tokens = torch::tensor({{1, 2, 3}}, torch::kInt32);
    auto selected   = torch::tensor({{-0.1f, -0.2f, -0.3f}}, torch::kFloat32);
    auto top_ids    = torch::tensor({1, 2, 3}, torch::kInt32).reshape({1, 3, 1});
    auto top_values = torch::tensor({-0.01f, -0.02f, -0.03f}, torch::kFloat32).reshape({1, 3, 1});

    stream->specUpdate({new_tokens,
                        3,
                        -1,
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        selected,
                        top_ids,
                        top_values});

    ASSERT_FALSE(stream->hasError());
    EXPECT_EQ(stream->processorAcceptedTokenLen(), 2);
    auto output_result = stream->nextOutput();
    ASSERT_TRUE(output_result.ok());
    const auto& output = output_result.value().generate_outputs[0];
    EXPECT_EQ(toVec<int32_t>(output.output_ids), (std::vector<int32_t>{1, 2}));
    EXPECT_TRUE(torch::allclose(output.token_logprobs.value(), selected.narrow(1, 0, 2).reshape({2})));
    EXPECT_EQ(toVec<int32_t>(output.top_logprob_token_ids.value()), (std::vector<int32_t>{1, 2}));

    // Reaching the logical token cap before a worker update must still drive
    // needFinish() with a zero-width logprob payload. In particular, this path
    // must not read new_tokens[-1] or mutate the speculative output buffer.
    auto capped_stream                              = createContextStream(model_config,
                                             runtime_config,
                                             resource_context,
                                                                          {0},
                                             1,
                                                                          {},
                                                                          {},
                                             /*return_logprobs=*/true,
                                             /*top_logprobs=*/1);
    capped_stream->generateConfig()->max_new_tokens = 0;
    const auto original_spec_tokens                 = capped_stream->getSPOutputBuffer()->tokens.clone();
    capped_stream->specUpdate({new_tokens,
                               3,
                               -1,
                               torch::Tensor(),
                               torch::Tensor(),
                               torch::Tensor(),
                               torch::Tensor(),
                               selected,
                               top_ids,
                               top_values});

    EXPECT_TRUE(capped_stream->hasEvent(StreamEvents::GenerateDone));
    EXPECT_EQ(capped_stream->moveToNext(), StreamState::FINISHED);
    EXPECT_TRUE(capped_stream->isFinished());
    EXPECT_FALSE(capped_stream->hasError());
    EXPECT_EQ(capped_stream->seqLength(), capped_stream->inputLength());
    EXPECT_TRUE(torch::equal(capped_stream->getSPOutputBuffer()->tokens, original_spec_tokens));
    EXPECT_FALSE(capped_stream->getTokenLogProbs().defined());
}

TEST_F(MtpBatchStreamProcessorTest, DISABLED_benchmarkScoreTokenIdsTorchCopyVsMemcpy) {
    constexpr int64_t stream_count = 64;
    constexpr int64_t score_len    = 4;
    constexpr int64_t max_seq_len  = 65536;
    constexpr int     iterations   = 20;

    auto src_storage = torch::empty({stream_count, max_seq_len}, torch::kInt32);
    src_storage.random_(0, 32000);

    std::vector<torch::Tensor> complete_token_ids;
    std::vector<int64_t>       seq_lens;
    complete_token_ids.reserve(stream_count);
    seq_lens.reserve(stream_count);
    for (int64_t i = 0; i < stream_count; ++i) {
        complete_token_ids.push_back(src_storage.narrow(0, i, 1));
        seq_lens.push_back(max_seq_len - (i % 8) * 128);
    }

    auto pinned_i32 = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    auto dst_memcpy = torch::empty({stream_count * score_len, max_seq_len + score_len}, pinned_i32);
    auto dst_torch  = torch::empty({stream_count * score_len, max_seq_len + score_len}, pinned_i32);

    dst_memcpy.fill_(-1);
    dst_torch.fill_(-1);
    fillScoreTokenIdsWithMemcpy(dst_memcpy, complete_token_ids, seq_lens, score_len);
    fillScoreTokenIdsWithTorchCopy(dst_torch, complete_token_ids, seq_lens, score_len);
    ASSERT_TRUE(torch::equal(dst_memcpy, dst_torch));

    auto memcpy_us = benchmarkUs(
        [&]() { fillScoreTokenIdsWithMemcpy(dst_memcpy, complete_token_ids, seq_lens, score_len); }, iterations);
    auto torch_us = benchmarkUs(
        [&]() { fillScoreTokenIdsWithTorchCopy(dst_torch, complete_token_ids, seq_lens, score_len); }, iterations);

    std::cout << "[mtp-score-token-ids-copy] streams=" << stream_count << " score_len=" << score_len
              << " max_seq_len=" << max_seq_len << " iterations=" << iterations << " memcpy_us=" << memcpy_us
              << " torch_copy_us=" << torch_us << " speedup=" << (memcpy_us / torch_us) << std::endl;
}

TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputReplicatesScoreTokenIds) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 3;

    ResourceContext resource_context;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {5}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {6, 7}, 2);
    stream1->setScoreLen(sp_config.gen_num_per_cycle + 1);
    stream2->setScoreLen(sp_config.gen_num_per_cycle + 1);

    auto stream_groups = StreamGroups({stream1, stream2});
    auto processor     = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelInputs  model_inputs;
    GptModelOutputs model_output;
    model_output.logits =
        torch::empty({static_cast<int64_t>(stream_groups.size() * (sp_config.gen_num_per_cycle + 1)), 4},
                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto sampler_inputs_status = processor.gatherSpecSamplerInput(stream_groups, model_inputs, model_output);
    ASSERT_TRUE(sampler_inputs_status.ok());

    auto  token_ids = sampler_inputs_status.value().token_ids;
    auto  stride    = token_ids.size(1);
    auto* data      = token_ids.data_ptr<int32_t>();

    for (int64_t row = 0; row < 4; ++row) {
        EXPECT_EQ(5, data[row * stride]);
    }
    for (int64_t row = 4; row < 8; ++row) {
        EXPECT_EQ(6, data[row * stride]);
        EXPECT_EQ(7, data[row * stride + 1]);
    }
}

TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputPreservesNarrowLogitsWithoutLogprobs) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};

    model_config.max_seq_len    = 32;
    model_config.vocab_size     = 100;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    ResourceContext resource_context;
    auto            stream = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
    StreamGroups            stream_groups({stream});
    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelInputs  model_inputs;
    GptModelOutputs model_output;
    model_output.logits =
        torch::arange(0, 8, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({2, 4});
    auto original_logits = model_output.logits.clone();

    auto sampler_inputs_status = processor.gatherSpecSamplerInput(stream_groups, model_inputs, model_output);
    ASSERT_TRUE(sampler_inputs_status.ok());
    EXPECT_EQ(sampler_inputs_status.value().logits.sizes(), (torch::IntArrayRef{2, 4}));
    EXPECT_TRUE(torch::equal(sampler_inputs_status.value().logits, original_logits));
}

TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputPreservesPaddedLogitsWithoutLogprobs) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};

    model_config.max_seq_len    = 32;
    model_config.vocab_size     = 5;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    ModelConfig small_vocab_model_config = model_config;
    small_vocab_model_config.vocab_size  = 3;

    ResourceContext resource_context;
    auto small_vocab_stream = createContextStream(small_vocab_model_config, runtime_config, resource_context, {1}, 1);
    auto large_vocab_stream = createContextStream(model_config, runtime_config, resource_context, {2}, 2);
    small_vocab_stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
    large_vocab_stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
    StreamGroups stream_groups({small_vocab_stream, large_vocab_stream});

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelInputs  model_inputs;
    GptModelOutputs model_output;
    model_output.logits =
        torch::arange(0, 32, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({4, 8});
    auto original_logits = model_output.logits.clone();

    auto sampler_inputs_status = processor.gatherSpecSamplerInput(stream_groups, model_inputs, model_output);
    ASSERT_TRUE(sampler_inputs_status.ok());
    const auto& sampler_inputs = sampler_inputs_status.value();
    ASSERT_EQ(sampler_inputs.logits.sizes(), (torch::IntArrayRef{4, 8}));
    ASSERT_EQ(sampler_inputs.all_probs.sizes(), (torch::IntArrayRef{4, 8}));
    EXPECT_TRUE(torch::equal(model_output.logits, original_logits));
    EXPECT_TRUE(torch::equal(sampler_inputs.logits, original_logits));
}

TEST_F(MtpBatchStreamProcessorTest, testSpecMaskExcludesPaddedSamplerLogitsWithoutLogprobs) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};

    model_config.max_seq_len    = 32;
    model_config.vocab_size     = 5;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    ResourceContext resource_context;
    auto            stream = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
    StreamGroups            stream_groups({stream});
    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelInputs  model_inputs;
    GptModelOutputs model_output;
    // Real token logits are all negative while padded LM-head columns are zero.
    // Without suffix masking, greedy sampling would select OOV token 5.
    model_output.logits = torch::tensor(
        {{-5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f}, {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, 0.0f, 0.0f, 0.0f}},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto original_logits = model_output.logits.clone();

    SpecLogitsVerifyRunner::LaunchResult spec_mask_result;
    spec_mask_result.has_active_processor = true;
    spec_mask_result.spec_vocab_mask_gpu =
        torch::tensor({{false, false, false, false, true}, {true, false, false, false, false}},
                      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto sampler_inputs_status =
        processor.gatherSpecSamplerInput(stream_groups, model_inputs, model_output, spec_mask_result);
    ASSERT_TRUE(sampler_inputs_status.ok());
    auto sampler_inputs = std::move(sampler_inputs_status.value());

    LogitsProcessorStates states;
    states.batchProcess(sampler_inputs);
    auto masked_logits = sampler_inputs.logits.cpu();

    EXPECT_EQ(masked_logits.index({0, 4}).item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(masked_logits.index({1, 0}).item<float>(), BaseLogitsProcessor::neg_inf);
    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t col = 5; col < 8; ++col) {
            EXPECT_EQ(masked_logits.index({row, col}).item<float>(), BaseLogitsProcessor::neg_inf);
        }
    }
    auto sampled_token_ids = masked_logits.argmax(/*dim=*/1);
    EXPECT_EQ(toVec<int64_t>(sampled_token_ids), (std::vector<int64_t>{3, 1}));
    EXPECT_TRUE(torch::all(sampled_token_ids < model_config.vocab_size).item<bool>());
    EXPECT_TRUE(torch::equal(model_output.logits, original_logits));
}

TEST_F(MtpBatchStreamProcessorTest, testSpecSamplerInputMasksThinkBoundaryTokens) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 16;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 2;
    cache_config.group_types    = {CacheGroupType::FULL};

    ResourceContext resource_context;
    auto stream = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 1, {7}, {8, 9});
    stream->setScoreLen(sp_config.gen_num_per_cycle + 1);

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    StreamGroups stream_groups({stream});

    GptModelInputs  model_input;
    GptModelOutputs model_output;
    model_output.logits = torch::zeros({3, 16}, torch::kFloat32);

    auto sampler_inputs_status = processor.gatherSpecSamplerInput(stream_groups, model_input, model_output);
    ASSERT_TRUE(sampler_inputs_status.ok());
    auto sampler_inputs = sampler_inputs_status.value();

    ASSERT_NE(sampler_inputs.logits_processor_states_ptr, nullptr);
    sampler_inputs.logits_processor_states_ptr->batchProcess(sampler_inputs);

    float neg_inf = -std::numeric_limits<float>::max();
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(neg_inf, sampler_inputs.logits[i][7].item<float>());
        EXPECT_EQ(neg_inf, sampler_inputs.logits[i][8].item<float>());
        EXPECT_EQ(0, sampler_inputs.logits[i][9].item<float>());
    }
}

TEST_F(MtpBatchStreamProcessorTest, testPrefillDispatch) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 4;

    ResourceContext resource_context;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {2}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    StreamGroups stream_groups(streams);

    MergedOutput target_output;
    target_output.model_output.all_hidden_states =
        torch::tensor({0.1f, 0.2f, 1.1f, 1.2f, 1.3f, 1.4f}, torch::kFloat32).reshape({3, 2});
    target_output.sampler_output.token_ids = torch::tensor({2, -1, 1, 1, 2, 3}, torch::kInt32).reshape({2, 3});
    target_output.sampler_output.all_probs = torch::tensor({0.1f, 0.9f, 0.2f, 0.8f}, torch::kFloat32).reshape({2, 2});

    MergedOutput draft_output;
    draft_output.model_output.all_hidden_states =
        torch::tensor({0.3f, 0.4f, 1.5f, 1.6f, 1.7f, 1.8f}, torch::kFloat32).reshape({3, 2});
    draft_output.sampler_output.token_ids = torch::tensor({2L, 0L}, torch::kInt64).reshape({2, 1});
    draft_output.sampler_output.all_probs =
        torch::tensor({0.2f, 0.1f, 0.3f, 0.5f, 0.3f, 0.1f, 0.4f, 0.2f}, torch::kFloat32).reshape({2, 4});

    auto status = processor.dispatchPrefill(stream_groups, target_output, draft_output);
    EXPECT_TRUE(status.ok());
    draft_output.model_output.all_hidden_states.fill_(9.0f);

    checkOutput(stream1, {2, 1}, {1, 2}, {0.2, 0.1, 0.3, 0.5}, {0.3, 0.4});
    checkOutput(stream2, {1, 2, 3}, {3, 0}, {0.3, 0.1, 0.4, 0.2}, {1.7, 1.8});
}

TEST_F(MtpBatchStreamProcessorTest, testPrefillDispatchUsesDraftLastHiddenOverride) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 4;

    ResourceContext resource_context;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {2}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    StreamGroups stream_groups(streams);

    MergedOutput target_output;
    target_output.sampler_output.token_ids = torch::tensor({2, -1, 1, 1, 2, 3}, torch::kInt32).reshape({2, 3});

    MergedOutput draft_output;
    draft_output.model_output.all_hidden_states =
        torch::tensor({0.3f, 0.4f, 1.5f, 1.6f, 1.7f, 1.8f}, torch::kFloat32).reshape({3, 2});
    draft_output.sampler_output.token_ids = torch::tensor({2L, 0L}, torch::kInt64).reshape({2, 1});
    draft_output.sampler_output.all_probs =
        torch::tensor({0.2f, 0.1f, 0.3f, 0.5f, 0.3f, 0.1f, 0.4f, 0.2f}, torch::kFloat32).reshape({2, 4});
    auto draft_last_hidden_states = torch::tensor({9.1f, 9.2f, 8.1f, 8.2f}, torch::kFloat32).reshape({2, 2});

    auto status = processor.dispatchPrefill(stream_groups, target_output, draft_output, draft_last_hidden_states);
    EXPECT_TRUE(status.ok());

    checkOutput(stream1, {2, 1}, {1, 2}, {0.2, 0.1, 0.3, 0.5}, {9.1, 9.2});
    checkOutput(stream2, {1, 2, 3}, {3, 0}, {0.3, 0.1, 0.4, 0.2}, {8.1, 8.2});
}

TEST_F(MtpBatchStreamProcessorTest, testPrefillDispatchSupportsCompactDraftLastHidden) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 4;

    ResourceContext resource_context;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {2}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    StreamGroups stream_groups(streams);

    MergedOutput target_output;
    target_output.sampler_output.token_ids = torch::tensor({2, -1, 1, 1, 2, 3}, torch::kInt32).reshape({2, 3});

    MergedOutput draft_output;
    // CP last-hidden-only prefill returns one hidden row per output batch, not
    // one row per token. Dispatch must treat this compact shape as valid.
    draft_output.model_output.all_hidden_states =
        torch::tensor({9.1f, 9.2f, 8.1f, 8.2f}, torch::kFloat32).reshape({2, 2});
    draft_output.sampler_output.token_ids = torch::tensor({2L, 0L}, torch::kInt64).reshape({2, 1});
    draft_output.sampler_output.all_probs =
        torch::tensor({0.2f, 0.1f, 0.3f, 0.5f, 0.3f, 0.1f, 0.4f, 0.2f}, torch::kFloat32).reshape({2, 4});

    auto status = processor.dispatchPrefill(stream_groups, target_output, draft_output);
    EXPECT_TRUE(status.ok());

    checkOutput(stream1, {2, 1}, {1, 2}, {0.2, 0.1, 0.3, 0.5}, {9.1, 9.2});
    checkOutput(stream2, {1, 2, 3}, {3, 0}, {0.3, 0.1, 0.4, 0.2}, {8.1, 8.2});
}

TEST_F(MtpBatchStreamProcessorTest, testDispatchDecodeStream) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 4;

    ResourceContext resource_context;
    resource_context.cache_manager =
        std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                                        /*block_num=*/10,
                                                                        /*tokens_per_block=*/2,
                                                                        rtp_llm::TYPE_INT8,
                                                                        /*local_head_num_kv=*/128,
                                                                        /*size_per_head=*/256));

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {2, 1}, 2);

    auto stream_groups = StreamGroups({stream1, stream2});

    speculative::SpeculativeSamplerOutput spec_decode_output;
    spec_decode_output.accept_len_cpu    = torch::tensor({5, 1}, torch::kInt32);
    spec_decode_output.accept_tokens_cpu = torch::tensor({{2, 3, 1, 3, 2}, {2, 0, 0, 0, 0}}, torch::kInt32);
    spec_decode_output.accept_len        = spec_decode_output.accept_len_cpu.to(torch::kCUDA);
    spec_decode_output.accept_tokens     = spec_decode_output.accept_tokens_cpu.to(torch::kCUDA);

    MergedOutput draft_prefill_output;
    draft_prefill_output.model_output.all_hidden_states =
        torch::tensor({0.2f, 0.02f, 0.3f, 0.03f, 0.4f, 0.04f, 0.5f, 0.05f, 0.6f, 0.06f, 1.3f, 0.13f}, torch::kFloat32)
            .reshape({6, 2});
    draft_prefill_output.sampler_output.token_ids = torch::tensor({0L, 3L}, torch::kInt64).reshape({2, 1});
    draft_prefill_output.sampler_output.all_probs =
        torch::tensor({0.2f, 0.1f, 0.3f, 0.5f, 0.3f, 0.1f, 0.4f, 0.2f}, torch::kFloat32).reshape({2, 4});

    cache_config.group_types = {CacheGroupType::FULL};
    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    auto status = processor.dispatchDecode(stream_groups, spec_decode_output, draft_prefill_output);
    EXPECT_TRUE(status.ok());
    draft_prefill_output.model_output.all_hidden_states.fill_(9.0f);

    checkOutput(stream1, {1, 2, 3, 1, 3, 2}, {2, 0}, {0.2, 0.1, 0.3, 0.5}, {0.6, 0.06});
    checkOutput(stream2, {2, 1, 2}, {2, 3}, {0.3, 0.1, 0.4, 0.2}, {1.3, 0.13});
    EXPECT_EQ(stream1->getMtpAsyncDeviceState().last_real_seq_len, stream1->seqLength());
    EXPECT_EQ(stream1->getMtpAsyncDeviceState().next_real_seq_len, stream1->seqLength());
    EXPECT_EQ(stream2->getMtpAsyncDeviceState().last_real_seq_len, stream2->seqLength());
    EXPECT_EQ(stream2->getMtpAsyncDeviceState().next_real_seq_len, stream2->seqLength());
}

TEST_F(MtpBatchStreamProcessorTest, testGatherDecodeModelInput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 4;

    auto kv_cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                          /*block_num=*/10,
                                                          /*tokens_per_block=*/2,
                                                          rtp_llm::TYPE_INT8,
                                                          /*local_head_num_kv=*/128,
                                                          /*size_per_head=*/256);
    auto cache_manager   = std::make_shared<KVCacheManager>(kv_cache_config,

                                                          /*warmup=*/false,
                                                          /*metrics_reporter=*/nullptr,
                                                          KVCacheConfig{},
                                                          ParallelismConfig{},
                                                          runtime_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {2}, 2);

    stream1->getSPOutputBuffer()->hidden_states = torch::tensor({{0.1f, 0.2f}});
    stream2->getSPOutputBuffer()->hidden_states = torch::tensor({{1.1f, 1.2f}});

    auto stream_groups = StreamGroups({stream1, stream2});

    cache_config.group_types = {CacheGroupType::FULL};
    auto processor           = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input = processor.gatherDecodeModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input.ok());

    auto          last_hidden_states        = model_input.value().last_hidden_states;
    auto          last_hidden_states_h      = last_hidden_states.cpu().clone();
    vector<float> expect_last_hidden_states = {0.1, 0.2, 1.1, 1.2};
    EXPECT_EQ(expect_last_hidden_states, toVec<float>(last_hidden_states_h));
}

TEST_F(MtpBatchStreamProcessorTest, testPrepareOneStepSpecDecodeModelInput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    auto kv_cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                          /*block_num=*/10,
                                                          /*tokens_per_block=*/2,
                                                          rtp_llm::TYPE_INT8,
                                                          /*local_head_num_kv=*/128,
                                                          /*size_per_head=*/256);
    auto cache_manager   = std::make_shared<KVCacheManager>(kv_cache_config,

                                                          /*warmup=*/false,
                                                          /*metrics_reporter=*/nullptr,
                                                          KVCacheConfig{},
                                                          ParallelismConfig{},
                                                          runtime_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);

    auto context_token_1 = torch::tensor({2}, torch::kInt32).reshape({1, 1});
    auto context_token_2 = torch::tensor({3}, torch::kInt32).reshape({1, 1});

    stream1->update({context_token_1,
                     1,
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor()});
    stream2->update({context_token_2,
                     1,
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor()});

    vector<int> propose_tokens_1 = {2, 3};
    vector<int> propose_tokens_2 = {3, 1};

    setSpOutputTokens(stream1->getSPOutputBuffer(), propose_tokens_1);
    setSpOutputTokens(stream2->getSPOutputBuffer(), propose_tokens_2);

    auto stream_groups = StreamGroups({stream1, stream2});

    cache_config.group_types = {CacheGroupType::FULL};
    auto processor           = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input_status = processor.gatherDecodeModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input            = model_input_status.value();
    model_input.sequence_lengths = torch::tensor({1, 2}, torch::kInt32);

    processor.prepareOneStepSpecDecodeModelInput(stream_groups, model_input, holder);

    auto        combo_tokens        = model_input.combo_tokens;
    vector<int> expect_combo_tokens = {2, 3, 3, 1};
    EXPECT_EQ(expect_combo_tokens, toVec<int>(combo_tokens));

    auto        prefix_lengths        = model_input.prefix_lengths;
    vector<int> expect_prefix_lengths = {1, 2};
    EXPECT_EQ(expect_prefix_lengths, toVec<int>(prefix_lengths));

    auto        input_lengths        = model_input.input_lengths;
    vector<int> expect_input_lengths = {2, 2};
    EXPECT_EQ(expect_input_lengths, toVec<int>(input_lengths));

    auto sequence_lengths = model_input.sequence_lengths;
    EXPECT_TRUE(sequence_lengths.is_cuda());
    EXPECT_EQ(0, sequence_lengths.size(0));

    auto        lm_output_indexes        = model_input.lm_output_indexes;
    vector<int> expect_lm_output_indexes = {0, 1, 2, 3};
    EXPECT_TRUE(lm_output_indexes.is_cuda());
    EXPECT_EQ(expect_lm_output_indexes, toVec<int>(lm_output_indexes));
}

TEST_F(MtpBatchStreamProcessorTest, testPrepareOneStepSpecDecodeModelInputFromDeviceState) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    auto kv_cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                          /*block_num=*/10,
                                                          /*tokens_per_block=*/2,
                                                          rtp_llm::TYPE_INT8,
                                                          /*local_head_num_kv=*/128,
                                                          /*size_per_head=*/256);
    auto cache_manager   = std::make_shared<KVCacheManager>(kv_cache_config,

                                                          /*warmup=*/false,
                                                          /*metrics_reporter=*/nullptr,
                                                          KVCacheConfig{},
                                                          ParallelismConfig{},
                                                          runtime_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);

    auto context_token_1 = torch::tensor({2}, torch::kInt32).reshape({1, 1});
    auto context_token_2 = torch::tensor({3}, torch::kInt32).reshape({1, 1});
    stream1->update({context_token_1,
                     1,
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor()});
    stream2->update({context_token_2,
                     1,
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor()});

    const auto                          cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    GenerateStream::MtpAsyncDeviceState state1;
    state1.accept_len_gpu     = torch::tensor({2}, torch::kInt32).to(torch::kCUDA);
    state1.accept_tokens_gpu  = torch::tensor({{2, 3}}, torch::kInt32).to(torch::kCUDA);
    state1.next_seq_len_gpu   = torch::full({1}, 7, cuda_i32);
    state1.propose_tokens_gpu = torch::tensor({{1}}, torch::kInt32).to(torch::kCUDA);
    stream1->setMtpAsyncDeviceState(std::move(state1));

    GenerateStream::MtpAsyncDeviceState state2;
    state2.accept_len_gpu     = torch::tensor({1}, torch::kInt32).to(torch::kCUDA);
    state2.accept_tokens_gpu  = torch::tensor({{1, 0}}, torch::kInt32).to(torch::kCUDA);
    state2.next_seq_len_gpu   = torch::full({1}, 4, cuda_i32);
    state2.propose_tokens_gpu = torch::tensor({{2}}, torch::kInt32).to(torch::kCUDA);
    stream2->setMtpAsyncDeviceState(std::move(state2));

    auto stream_groups = StreamGroups({stream1, stream2});

    cache_config.group_types = {CacheGroupType::FULL};
    auto processor           = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input_status = processor.gatherDecodeModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input            = model_input_status.value();
    model_input.sequence_lengths = torch::tensor({99, 99}, torch::kInt32);

    processor.prepareOneStepSpecDecodeModelInput(stream_groups, model_input, holder);

    vector<int> expect_combo_tokens = {3, 1, 1, 2};
    EXPECT_TRUE(model_input.combo_tokens.is_cuda());
    EXPECT_EQ(expect_combo_tokens, toVec<int>(model_input.combo_tokens));

    vector<int> expect_prefix_lengths = {6, 3};
    EXPECT_TRUE(model_input.prefix_lengths.is_cuda());
    EXPECT_EQ(expect_prefix_lengths, toVec<int>(model_input.prefix_lengths));
    EXPECT_TRUE(model_input.sequence_lengths.is_cuda());
    EXPECT_EQ(0, model_input.sequence_lengths.size(0));

    vector<int> expect_input_lengths = {2, 2};
    EXPECT_TRUE(model_input.input_lengths.is_cuda());
    EXPECT_EQ(expect_input_lengths, toVec<int>(model_input.input_lengths));
}

TEST_F(MtpBatchStreamProcessorTest, testprepareDecodeDraftModelInput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 2;

    auto kv_cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                          /*block_num=*/10,
                                                          /*tokens_per_block=*/2,
                                                          rtp_llm::TYPE_INT8,
                                                          /*local_head_num_kv=*/128,
                                                          /*size_per_head=*/256);
    auto cache_manager   = std::make_shared<KVCacheManager>(kv_cache_config,

                                                          /*warmup=*/false,
                                                          /*metrics_reporter=*/nullptr,
                                                          KVCacheConfig{},
                                                          ParallelismConfig{},
                                                          runtime_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);

    auto context_token_1 = torch::tensor({2}, torch::kInt32).reshape({1, 1});
    auto context_token_2 = torch::tensor({3}, torch::kInt32).reshape({1, 1});

    stream1->update({context_token_1,
                     1,
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor()});
    stream2->update({context_token_2,
                     1,
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor()});

    vector<int> propose_tokens_1 = {2, 3};
    vector<int> propose_tokens_2 = {3, 1};

    setSpOutputTokens(stream1->getSPOutputBuffer(), propose_tokens_1);
    setSpOutputTokens(stream2->getSPOutputBuffer(), propose_tokens_2);
    stream1->getSPOutputBuffer()->hidden_states = torch::tensor({{0.1f, 0.2f}});
    stream2->getSPOutputBuffer()->hidden_states = torch::tensor({{1.1f, 1.2f}});

    auto stream_groups = StreamGroups({stream1, stream2});

    cache_config.group_types = {CacheGroupType::FULL};
    auto processor           = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input_status = processor.gatherDecodeModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input            = model_input_status.value();
    model_input.sequence_lengths = torch::tensor({1, 2}, torch::kInt32);

    processor.prepareDecodeDraftModelInput(stream_groups, model_input, holder);

    auto        combo_tokens        = model_input.combo_tokens;
    vector<int> expect_combo_tokens = {3, 1};
    EXPECT_EQ(expect_combo_tokens, toVec<int>(combo_tokens));

    auto        lm_output_indexes        = model_input.lm_output_indexes;
    vector<int> expect_lm_output_indexes = {0, 1};
    EXPECT_TRUE(lm_output_indexes.is_cuda());
    EXPECT_EQ(expect_lm_output_indexes, toVec<int>(lm_output_indexes));

    // Normal gather reports the target prefix (the last target-KV row that is
    // already initialized). Multi-step draft decode runs one position ahead,
    // but target verification must retain the original prefix and write that
    // carried token instead of skipping it.
    vector<int> expect_prefix_lengths   = {1, 2};
    vector<int> expect_sequence_lengths = {2, 3};
    EXPECT_TRUE(model_input.prefix_lengths.is_cuda());
    EXPECT_EQ(expect_prefix_lengths, toVec<int>(model_input.prefix_lengths));
    EXPECT_TRUE(model_input.sequence_lengths.is_cuda());
    EXPECT_EQ(expect_sequence_lengths, toVec<int>(model_input.sequence_lengths));

    // The steady-state device path has the same one-token separation. The
    // published length includes the carried target token; target verify starts
    // one row earlier while draft decode consumes the published position.
    GenerateStream::MtpAsyncDeviceState state1;
    state1.propose_tokens_gpu = torch::tensor({{3}}, torch::kInt32).to(torch::kCUDA);
    state1.next_seq_len_gpu   = torch::tensor({7}, torch::kInt32).to(torch::kCUDA);
    stream1->setMtpAsyncDeviceState(std::move(state1));

    GenerateStream::MtpAsyncDeviceState state2;
    state2.propose_tokens_gpu = torch::tensor({{1}}, torch::kInt32).to(torch::kCUDA);
    state2.next_seq_len_gpu   = torch::tensor({4}, torch::kInt32).to(torch::kCUDA);
    stream2->setMtpAsyncDeviceState(std::move(state2));

    model_input.sequence_lengths = torch::tensor({99, 99}, torch::kInt32);
    processor.prepareDecodeDraftModelInput(stream_groups, model_input, holder);

    EXPECT_EQ((vector<int>{6, 3}), toVec<int>(model_input.prefix_lengths));
    EXPECT_EQ((vector<int>{7, 4}), toVec<int>(model_input.sequence_lengths));
}

TEST_F(MtpBatchStreamProcessorTest, testUpdatePrefillPostDraftModelInput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    auto kv_cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                          /*block_num=*/10,
                                                          /*tokens_per_block=*/2,
                                                          rtp_llm::TYPE_INT8,
                                                          /*local_head_num_kv=*/128,
                                                          /*size_per_head=*/256);
    auto cache_manager   = std::make_shared<KVCacheManager>(kv_cache_config,

                                                          /*warmup=*/false,
                                                          /*metrics_reporter=*/nullptr,
                                                          KVCacheConfig{},
                                                          ParallelismConfig{},
                                                          runtime_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);

    auto stream_groups = StreamGroups({stream1, stream2});

    cache_config.group_types = {CacheGroupType::FULL};
    auto processor           = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input_status = processor.gatherModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input            = model_input_status.value();
    model_input.sequence_lengths = torch::tensor({1, 2}, torch::kInt32);

    GptModelOutputs model_output;
    model_output.all_hidden_states =
        torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, torch::kFloat32).reshape({3, 2});

    SamplerOutput sampler_output;
    sampler_output.token_ids = torch::tensor({1, -2, 2, 1, 2, 3}, torch::kInt32).reshape({2, 3});

    processor.updatePrefillPostDraftModelInput(model_input, model_output, sampler_output, holder);

    auto        combo_tokens        = model_input.combo_tokens;
    vector<int> expect_combo_tokens = {2, 2, 3};
    EXPECT_EQ(expect_combo_tokens, toVec<int>(combo_tokens));
}

TEST_F(MtpBatchStreamProcessorTest, testUpdateDecodePostDraftModelInput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 2;

    auto kv_cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                          /*block_num=*/10,
                                                          /*tokens_per_block=*/2,
                                                          rtp_llm::TYPE_INT8,
                                                          /*local_head_num_kv=*/128,
                                                          /*size_per_head=*/256);
    auto cache_manager   = std::make_shared<KVCacheManager>(kv_cache_config,

                                                          /*warmup=*/false,
                                                          /*metrics_reporter=*/nullptr,
                                                          KVCacheConfig{},
                                                          ParallelismConfig{},
                                                          runtime_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);

    auto stream_groups = StreamGroups({stream1, stream2});

    cache_config.group_types = {CacheGroupType::FULL};
    auto processor           = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input_status = processor.gatherModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input = model_input_status.value();

    speculative::SpeculativeSamplerOutput spec_decode_output;
    spec_decode_output.accept_len_cpu    = torch::tensor({3, 1}, torch::kInt32);
    spec_decode_output.accept_tokens_cpu = torch::tensor({{2, 3, 1}, {2, 0, 0}}, torch::kInt32);
    spec_decode_output.accept_len        = spec_decode_output.accept_len_cpu.to(torch::kCUDA);
    spec_decode_output.accept_tokens     = spec_decode_output.accept_tokens_cpu.to(torch::kCUDA);

    torch::Tensor hidden_states_d_t;

    GptModelOutputs model_output;
    model_output.all_hidden_states =
        torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f}, torch::kFloat32)
            .reshape({6, 2});

    processor.updateDecodePostDraftModelInput(
        model_input, model_output, spec_decode_output, 2, hidden_states_d_t, holder);

    auto        combo_tokens        = model_input.combo_tokens.cpu();
    vector<int> expect_combo_tokens = {2, 3, 1, 2, 0, 0};
    EXPECT_EQ(expect_combo_tokens, toVec<int>(combo_tokens));

    EXPECT_TRUE(model_input.lm_output_indexes.is_cuda());
    auto        lm_output_indexes        = model_input.lm_output_indexes.cpu();
    vector<int> expect_lm_output_indexes = {2, 3};
    EXPECT_EQ(expect_lm_output_indexes, toVec<int>(lm_output_indexes));

    auto          last_hidden_states        = model_input.last_hidden_states;
    vector<float> expect_last_hidden_states = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f};
    EXPECT_EQ(expect_last_hidden_states, toVec<float>(last_hidden_states));
}

TEST_F(MtpBatchStreamProcessorTest, testUpdateOneStepDraftSamplerOutput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    auto kv_cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                          /*block_num=*/10,
                                                          /*tokens_per_block=*/2,
                                                          rtp_llm::TYPE_INT8,
                                                          /*local_head_num_kv=*/128,
                                                          /*size_per_head=*/256);
    auto cache_manager   = std::make_shared<KVCacheManager>(kv_cache_config,

                                                          /*warmup=*/false,
                                                          /*metrics_reporter=*/nullptr,
                                                          KVCacheConfig{},
                                                          ParallelismConfig{},
                                                          runtime_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);

    stream1->getSPOutputBuffer()->all_probs = torch::tensor({{0.1f, 0.2f, 0.3f, 0.4f}});
    stream2->getSPOutputBuffer()->all_probs = torch::tensor({{0.5f, 0.6f, 0.7f, 0.8f}});
    setSpOutputTokens(stream1->getSPOutputBuffer(), {1, 2});
    setSpOutputTokens(stream2->getSPOutputBuffer(), {2, 3});

    auto stream_groups = StreamGroups({stream1, stream2});
    auto processor     = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    torch::Tensor draft_token_probs_d_t;
    SamplerOutput sampler_output;
    TensorHolder  holder;

    processor.updateOneStepDraftSamplerOutput(stream_groups, sampler_output, draft_token_probs_d_t, holder);

    vector<int> expect_token_ids = {2, 3};
    EXPECT_EQ(expect_token_ids, toVec<int>(sampler_output.token_ids));

    vector<float> expect_all_probs = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    EXPECT_EQ(expect_all_probs, toVec<float>(sampler_output.all_probs));
}

TEST_F(MtpBatchStreamProcessorTest, testUpdateOneStepDraftSamplerOutputFromDeviceState) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    auto kv_cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                          /*block_num=*/10,
                                                          /*tokens_per_block=*/2,
                                                          rtp_llm::TYPE_INT8,
                                                          /*local_head_num_kv=*/128,
                                                          /*size_per_head=*/256);
    auto cache_manager   = std::make_shared<KVCacheManager>(kv_cache_config,

                                                          /*warmup=*/false,
                                                          /*metrics_reporter=*/nullptr,
                                                          KVCacheConfig{},
                                                          ParallelismConfig{},
                                                          runtime_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);

    stream1->getSPOutputBuffer()->all_probs = torch::tensor({{0.1f, 0.2f, 0.3f, 0.4f}});
    stream2->getSPOutputBuffer()->all_probs = torch::tensor({{0.5f, 0.6f, 0.7f, 0.8f}});
    setSpOutputTokens(stream1->getSPOutputBuffer(), {1, 2});
    setSpOutputTokens(stream2->getSPOutputBuffer(), {2, 3});

    GenerateStream::MtpAsyncDeviceState state1;
    state1.propose_tokens_gpu  = torch::tensor({{3}}, torch::kInt32).to(torch::kCUDA);
    state1.draft_all_probs_gpu = torch::tensor({{0.9f, 0.8f, 0.7f, 0.6f}}).to(torch::kCUDA);
    stream1->setMtpAsyncDeviceState(std::move(state1));

    GenerateStream::MtpAsyncDeviceState state2;
    state2.propose_tokens_gpu  = torch::tensor({{1}}, torch::kInt32).to(torch::kCUDA);
    state2.draft_all_probs_gpu = torch::tensor({{0.4f, 0.3f, 0.2f, 0.1f}}).to(torch::kCUDA);
    stream2->setMtpAsyncDeviceState(std::move(state2));

    auto stream_groups = StreamGroups({stream1, stream2});
    auto processor     = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    torch::Tensor draft_token_probs_d_t;
    SamplerOutput sampler_output;
    TensorHolder  holder;

    processor.updateOneStepDraftSamplerOutput(stream_groups, sampler_output, draft_token_probs_d_t, holder);

    vector<int> expect_token_ids = {3, 1};
    EXPECT_TRUE(sampler_output.token_ids.is_cuda());
    EXPECT_EQ(expect_token_ids, toVec<int>(sampler_output.token_ids));

    vector<float> expect_all_probs = {0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1};
    EXPECT_TRUE(sampler_output.all_probs.is_cuda());
    EXPECT_EQ(expect_all_probs, toVec<float>(sampler_output.all_probs));
}

TEST_F(MtpBatchStreamProcessorTest, updateMultiStepDraftSamplerOutput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 3;

    auto kv_cache_config = test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                                          /*block_num=*/10,
                                                          /*tokens_per_block=*/2,
                                                          rtp_llm::TYPE_INT8,
                                                          /*local_head_num_kv=*/128,
                                                          /*size_per_head=*/256);
    auto cache_manager   = std::make_shared<KVCacheManager>(kv_cache_config,

                                                          /*warmup=*/false,
                                                          /*metrics_reporter=*/nullptr,
                                                          KVCacheConfig{},
                                                          ParallelismConfig{},
                                                          runtime_config);
    ASSERT_TRUE(cache_manager->init());
    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);

    stream1->getSPOutputBuffer()->all_probs = torch::tensor({{0.1f, 0.2f, 0.3f, 0.4f}});
    stream2->getSPOutputBuffer()->all_probs = torch::tensor({{0.5f, 0.6f, 0.7f, 0.8f}});
    setSpOutputTokens(stream1->getSPOutputBuffer(), {1, 2});
    setSpOutputTokens(stream2->getSPOutputBuffer(), {2, 3});

    auto output_token_probs_1 =
        torch::tensor({1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f}, torch::kFloat32).reshape({2, 1, 4});
    auto output_token_probs_2 =
        torch::tensor({2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f}, torch::kFloat32).reshape({2, 1, 4});

    auto draft_token_ids_t = torch::tensor({2, 0, 1, 2, 3, 1, 2, 3}, torch::kInt32).reshape({2, 4});

    auto stream_groups = StreamGroups({stream1, stream2});
    auto processor     = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    torch::Tensor              draft_token_probs_d_t;
    torch::Tensor              draft_token_ids_d_t = draft_token_ids_t;
    torch::Tensor              spec_token_ids_d_t;
    std::vector<torch::Tensor> draft_token_probs_list;
    SamplerOutput              sampler_output;

    draft_token_probs_list.push_back(output_token_probs_1);
    draft_token_probs_list.push_back(output_token_probs_2);

    processor.updateMultiStepDraftSamplerOutput(stream_groups,
                                                sampler_output,
                                                draft_token_ids_d_t,
                                                spec_token_ids_d_t,
                                                draft_token_probs_d_t,
                                                draft_token_probs_list);

    vector<int> expect_token_ids = {0, 1, 2, 1, 2, 3};
    EXPECT_EQ(expect_token_ids, toVec<int>(sampler_output.token_ids));

    vector<float> expect_all_probs = {0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4,
                                      0.5, 0.6, 0.7, 0.8, 1.5, 1.6, 1.7, 1.8, 2.5, 2.6, 2.7, 2.8};
    EXPECT_EQ(expect_all_probs, toVec<float>(sampler_output.all_probs));
}

}  // namespace rtp_llm
