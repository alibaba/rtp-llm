#include <chrono>
#include <csignal>
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
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

using namespace std;

namespace rtp_llm {

// Deliberately does not override requiresTokenHistory: unknown processors
// must retain the conservative base-class contract.
class UnknownHistoryProcessorForCompactTest: public BaseLogitsProcessor {
public:
    void process(const SamplerInputs&, size_t, size_t) override {}
    void updateMultiSeqStatus(const std::vector<int>&) override {}
    void updateStatus(const torch::Tensor&, int32_t) override {}
};

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
                                          const vector<int>&     begin_think_token_ids       = {},
                                          const vector<int>&     end_think_token_ids         = {},
                                          bool                   explicitly_disable_thinking = false) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids       = torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        query->generate_config = make_shared<GenerateConfig>();
        query->generate_config->begin_think_token_ids = begin_think_token_ids;
        query->generate_config->end_think_token_ids   = end_think_token_ids;
        if (explicitly_disable_thinking) {
            query->generate_config->thinking_mode       = ThinkingMode::DISABLED;
            query->generate_config->max_thinking_tokens = 0;
        }
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
        stream->setReturnAllProbs(ReturnAllProbsMode::DEFAULT);
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

TEST_F(MtpBatchStreamProcessorTest, testDSparkInitializedPerfPrefillPreservesAnchor) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_config;
    CacheConfig                 cache_config;
    ResourceContext             resource_context;
    model_config.max_seq_len               = 128;
    model_config.vocab_size                = 32;
    model_config.num_layers                = 1;
    cache_config.group_types               = {CacheGroupType::FULL};
    sp_config.type                         = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle            = 7;
    sp_config.sp_dspark_mask_token_id      = 31;
    sp_config.sp_dspark_sample_from_anchor = true;
    MtpBatchStreamProcessor processor(model_config, pd_sep_config, profiling_config, cache_config, sp_config, false);
    const auto              cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    for (bool perf : {false, true}) {
        auto stream = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3}, 1);
        stream->setPerfTest(perf);
        StreamGroups prefill_group({stream});
        MergedOutput target_output;
        target_output.sampler_output.token_ids = torch::tensor({7}, torch::kInt32).reshape({1, 1});
        MergedOutput draft_output;  // DSpark real prefill is commit-only: no draft sample.
        ASSERT_TRUE(processor.dispatchPrefill(prefill_group, target_output, draft_output).ok());
        ASSERT_FALSE(stream->isContextStream());
        ASSERT_EQ(stream->outputTokenLen(), 1);
        EXPECT_EQ(stream->completeTokenIds().index({0, 3}).item<int32_t>(), perf ? 0 : 7);
        auto sp = stream->getSPOutputBuffer();
        EXPECT_EQ(sp->tokens.index({0, 0}).item<int32_t>(), 7);
        // preparePrefillSpecUpdateInfo omits target_token_gpu; specUpdate clears it.
        ASSERT_FALSE(sp->target_token_gpu.defined());
        ASSERT_FALSE(stream->getMtpAsyncDeviceState().accept_tokens_gpu.defined());

        GptModelInputs input;
        input.sequence_lengths = torch::tensor({3}, cuda_i32);
        TensorHolder holder;
        StreamGroups decode_group({stream});
        auto         round = processor.buildDSparkRoundState(decode_group, input, holder);
        EXPECT_EQ(toVec<int32_t>(round.anchors), (std::vector<int32_t>{7}));
        EXPECT_EQ(toVec<int32_t>(round.committed_ends), (std::vector<int32_t>{3}));

        // Equivalent to ensureSpOutputTokenGpuMirrors in next prepareStreams.
        sp->target_token_gpu = sp->tokens.reshape({-1}).narrow(0, 0, 1).to(cuda_i32);
        round                = processor.buildDSparkRoundState(decode_group, input, holder);
        EXPECT_EQ(toVec<int32_t>(round.anchors), (std::vector<int32_t>{7}));

        // The normal/non-perf fallback must continue to read real token history.
        if (!perf) {
            sp->target_token_gpu.fill_(11);
            round = processor.buildDSparkRoundState(decode_group, input, holder);
            EXPECT_EQ(toVec<int32_t>(round.anchors), (std::vector<int32_t>{7}));
        }

        // Async accepted state remains authoritative over either SP/history source.
        GenerateStream::MtpAsyncDeviceState state;
        state.accept_tokens_gpu = torch::tensor({8, 9}, cuda_i32).reshape({1, 2});
        state.accept_len_gpu    = torch::tensor({2}, cuda_i32);
        state.next_seq_len_gpu  = torch::tensor({6}, cuda_i32);
        stream->setMtpAsyncDeviceState(std::move(state));
        round = processor.buildDSparkRoundState(decode_group, input, holder);
        EXPECT_EQ(toVec<int32_t>(round.anchors), (std::vector<int32_t>{9}));
        EXPECT_EQ(toVec<int32_t>(round.committed_ends), (std::vector<int32_t>{5}));

        stream->setMtpAsyncDeviceState(GenerateStream::MtpAsyncDeviceState{});
        stream->setIsFakeStream(true);
        round = processor.buildDSparkRoundState(decode_group, input, holder);
        EXPECT_EQ(toVec<int32_t>(round.anchors), (std::vector<int32_t>{0}));
    }

    // No emitted token: do not silently substitute the fake SP-buffer token.
    auto direct = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3}, 1);
    direct->setPerfTest(true);
    direct->setIsContextStream(false);
    GptModelInputs input;
    input.sequence_lengths = torch::tensor({2}, cuda_i32);
    TensorHolder holder;
    StreamGroups direct_group({direct});
    auto         round = processor.buildDSparkRoundState(direct_group, input, holder);
    EXPECT_EQ(toVec<int32_t>(round.anchors), (std::vector<int32_t>{3}));
    EXPECT_EQ(toVec<int32_t>(round.committed_ends), (std::vector<int32_t>{2}));
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkBuildsFixedWidthProposalAndVerifyInputs) {
    ModelConfig                 model_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_config;
    CacheConfig                 cache_config;
    cache_config.group_types               = {CacheGroupType::FULL};
    model_config.max_seq_len               = 128;
    model_config.vocab_size                = 16;
    model_config.num_layers                = 1;
    sp_config.type                         = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle            = 3;
    sp_config.sp_dspark_mask_token_id      = 15;
    sp_config.sp_dspark_sample_from_anchor = true;

    MtpBatchStreamProcessor processor(model_config, pd_sep_config, profiling_config, cache_config, sp_config, false);
    TensorHolder            holder;
    GptModelInputs          proposal_input;
    auto                    anchors        = torch::tensor({2, 7}, torch::kInt32);
    auto                    committed_ends = torch::tensor({11, 23}, torch::kInt32);
    processor.buildDSparkProposeInput(proposal_input, anchors, committed_ends, holder);

    EXPECT_EQ(toVec<int32_t>(proposal_input.combo_tokens), (std::vector<int32_t>{2, 15, 15, 7, 15, 15}));
    EXPECT_EQ(toVec<int32_t>(proposal_input.input_lengths), (std::vector<int32_t>{3, 3}));
    EXPECT_EQ(toVec<int32_t>(proposal_input.prefix_lengths), (std::vector<int32_t>{11, 23}));
    EXPECT_EQ(toVec<int32_t>(proposal_input.lm_output_indexes), (std::vector<int32_t>{0, 1, 2, 3, 4, 5}));
    EXPECT_FALSE(proposal_input.last_hidden_states.defined());
    EXPECT_TRUE(proposal_input.is_target_verify);

    GptModelInputs verify_input;
    auto           proposals = torch::tensor({{3, 4, 5}, {8, 9, 10}}, torch::kInt32);
    processor.prepareDSparkTargetVerifyModelInput(verify_input, anchors, committed_ends, proposals, holder);
    EXPECT_EQ(toVec<int32_t>(verify_input.combo_tokens), (std::vector<int32_t>{2, 3, 4, 5, 7, 8, 9, 10}));
    EXPECT_EQ(toVec<int32_t>(verify_input.input_lengths), (std::vector<int32_t>{4, 4}));
    EXPECT_EQ(toVec<int32_t>(verify_input.prefix_lengths), (std::vector<int32_t>{11, 23}));
    EXPECT_EQ(toVec<int32_t>(verify_input.lm_output_indexes), (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
    EXPECT_TRUE(verify_input.is_target_verify);

    auto target_features = torch::zeros({8, 6}, torch::TensorOptions().device(torch::kCUDA));
    processor.updateDecodePostDSparkCommitInput(verify_input, target_features, 2);
    EXPECT_EQ(verify_input.last_hidden_states.data_ptr(), target_features.data_ptr());
    EXPECT_EQ(verify_input.last_hidden_states_layout, MtpHiddenStatesLayout::GLOBAL);
    // Every TP rank must retain the dense verify-row geometry for the commit
    // forward. Shrinking this to one row per request makes the TP lm-head
    // collective use different element counts and deadlock.
    EXPECT_EQ(toVec<int32_t>(verify_input.lm_output_indexes), (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkAdaptiveCompactPositionsAreOptional) {
    ModelConfig                 model_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_config;
    CacheConfig                 cache_config;
    cache_config.group_types    = {CacheGroupType::FULL};
    model_config.max_seq_len    = 128;
    model_config.vocab_size     = 32;
    model_config.num_layers     = 1;
    sp_config.type              = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle = 7;
    MtpBatchStreamProcessor processor(model_config, pd_sep_config, profiling_config, cache_config, sp_config, false);
    const auto              opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    TensorHolder            holder;
    MtpBatchStreamProcessor::DSparkRoundState state{torch::tensor({2, 7}, opts), torch::tensor({11, 20}, opts), {}};
    const auto                                proposals = torch::arange(14, opts).reshape({2, 7}) + 8;
    const auto                                lengths   = torch::tensor({3, 2}, opts);
    const auto                                mapping   = torch::tensor({0, 1, 2, 8, 9}, opts);
    GptModelInputs                            input;
    // Reusing an input must also clear positions from its previous round.
    input.combo_position_ids = torch::ones({5}, opts);
    processor.prepareCompactDSparkTargetVerifyModelInput(state, input, proposals, lengths, mapping, holder);
    EXPECT_FALSE(input.combo_position_ids.defined());
    EXPECT_EQ(toVec<int32_t>(input.combo_tokens), (std::vector<int32_t>{2, 8, 9, 7, 15}));
    EXPECT_EQ(toVec<int32_t>(input.input_lengths), (std::vector<int32_t>{3, 2}));
    EXPECT_TRUE(input.is_ragged_target_verify);

    state.position_bases = torch::tensor({{11}, {20}}, opts);
    processor.prepareCompactDSparkTargetVerifyModelInput(state, input, proposals, lengths, mapping, holder);
    EXPECT_EQ(toVec<int32_t>(input.combo_position_ids), (std::vector<int32_t>{11, 12, 13, 20, 21}));
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkVerifyBudgetKeepsProposalWidthSeven) {
    // This fixture initializes CUDA. Re-exec death-test children instead of
    // running CUDA code in a forked copy of the parent's initialized runtime.
    // GoogleTest restores flag values after this test.
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ModelConfig                 model_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_config;
    CacheConfig                 cache_config;
    cache_config.group_types               = {CacheGroupType::FULL};
    model_config.max_seq_len               = 128;
    model_config.vocab_size                = 32;
    model_config.num_layers                = 1;
    sp_config.type                         = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle            = 7;
    sp_config.sp_dspark_mask_token_id      = 31;
    sp_config.sp_dspark_sample_from_anchor = true;
    const auto cuda_i32                    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    for (int budget : {0, 1, 3, 4, 5, 7}) {
        SCOPED_TRACE(budget);
        sp_config.sp_dspark_verify_tokens = budget;
        const int               steps     = budget == 0 ? 7 : budget;
        const int               width     = steps + 1;
        MtpBatchStreamProcessor processor(
            model_config, pd_sep_config, profiling_config, cache_config, sp_config, false);
        EXPECT_EQ(processor.propose_step_, 7);
        EXPECT_EQ(processor.verify_step_, steps);
        TensorHolder holder;
        // The second row is fake; it still keeps the fixed-width geometry.
        MtpBatchStreamProcessor::DSparkRoundState state{
            torch::tensor({2, 7}, cuda_i32), torch::tensor({11, 0}, cuda_i32), torch::tensor({{11}, {0}}, cuda_i32)};
        GptModelInputs proposal_input;
        processor.prepareDSparkProposeModelInput(state, proposal_input, holder);
        EXPECT_EQ(proposal_input.combo_tokens.numel(), 14);
        EXPECT_EQ(toVec<int32_t>(proposal_input.input_lengths), (std::vector<int32_t>{7, 7}));
        EXPECT_EQ(proposal_input.lm_output_indexes.numel(), 14);
        EXPECT_EQ(proposal_input.combo_position_ids.numel(), 14);

        // Match the executor's prefix view of a full B*7 proposal tensor;
        // for a reduced budget the row stride still belongs to gamma7.
        const auto     full_proposals = torch::arange(14, cuda_i32).reshape({2, 7}) + 8;
        const auto     proposals      = full_proposals.narrow(1, 0, steps);
        GptModelInputs verify_input;
        processor.prepareDSparkTargetVerifyModelInput(state, verify_input, proposals, holder);
        std::vector<int32_t> expected_tokens;
        std::vector<int32_t> expected_positions;
        std::vector<int32_t> expected_indexes;
        for (int row = 0; row < 2; ++row) {
            expected_tokens.push_back(row == 0 ? 2 : 7);
            for (int col = 0; col < steps; ++col) {
                expected_tokens.push_back(8 + row * 7 + col);
            }
            for (int col = 0; col < width; ++col) {
                expected_positions.push_back((row == 0 ? 11 : 0) + col);
                expected_indexes.push_back(row * width + col);
            }
        }
        EXPECT_EQ(toVec<int32_t>(verify_input.combo_tokens), expected_tokens);
        EXPECT_EQ(toVec<int32_t>(verify_input.combo_position_ids), expected_positions);
        EXPECT_EQ(toVec<int32_t>(verify_input.lm_output_indexes), expected_indexes);
        EXPECT_EQ(toVec<int32_t>(verify_input.input_lengths), (std::vector<int32_t>{width, width}));
        EXPECT_EQ(toVec<int32_t>(verify_input.prefix_lengths), (std::vector<int32_t>{11, 0}));
        EXPECT_EQ(verify_input.sequence_lengths.numel(), 0);
        EXPECT_TRUE(verify_input.is_target_verify);

        auto features = torch::zeros({2 * width, 6}, torch::TensorOptions().device(torch::kCUDA));
        processor.updateDecodePostDSparkCommitInput(verify_input, features, 2);
        EXPECT_EQ(verify_input.last_hidden_states.data_ptr(), features.data_ptr());
        EXPECT_EQ(verify_input.last_hidden_states_layout, MtpHiddenStatesLayout::GLOBAL);
        EXPECT_EQ(toVec<int32_t>(verify_input.lm_output_indexes), expected_indexes);
        EXPECT_EQ(toVec<int32_t>(verify_input.combo_position_ids), expected_positions);
        // myAssert aborts when core-dump mode is enabled, rather than throwing.
        // Force that policy only inside the re-executed child, independently
        // of the parent's environment; require SIGABRT, not just any failure.
        EXPECT_EXIT(
            {
                StaticConfig::user_ft_core_dump_on_exception = true;
                processor.prepareDSparkTargetVerifyModelInput(
                    state, verify_input, torch::zeros({2, steps + 1}, cuda_i32), holder);
            },
            ::testing::KilledBySignal(SIGABRT),
            "");
        EXPECT_EXIT(
            {
                StaticConfig::user_ft_core_dump_on_exception = true;
                processor.updateDecodePostDSparkCommitInput(
                    verify_input, torch::zeros({2 * width - 1, 6}, torch::TensorOptions().device(torch::kCUDA)), 2);
            },
            ::testing::KilledBySignal(SIGABRT),
            "");
    }
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkVerifyBudgetRejectsInvalidConfigAndPreservesLegacy) {
    ModelConfig                 model_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_config;
    CacheConfig                 cache_config;
    cache_config.group_types    = {CacheGroupType::FULL};
    sp_config.type              = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle = 7;
    for (int invalid : {-1, 8}) {
        sp_config.sp_dspark_verify_tokens = invalid;
        EXPECT_ANY_THROW(
            MtpBatchStreamProcessor(model_config, pd_sep_config, profiling_config, cache_config, sp_config, false));
    }
    sp_config.type                    = SP_TYPE_MTP;
    sp_config.sp_dspark_verify_tokens = 0;
    MtpBatchStreamProcessor legacy(model_config, pd_sep_config, profiling_config, cache_config, sp_config, false);
    EXPECT_EQ(legacy.propose_step_, 7);
    EXPECT_EQ(legacy.verify_step_, 7);
    sp_config.sp_dspark_verify_tokens = 4;
    EXPECT_ANY_THROW(
        MtpBatchStreamProcessor(model_config, pd_sep_config, profiling_config, cache_config, sp_config, false));
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkVerifyBudgetSamplerGeometry) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_config;
    CacheConfig                 cache_config;
    cache_config.group_types          = {CacheGroupType::FULL};
    model_config.max_seq_len          = 128;
    model_config.vocab_size           = 16;
    model_config.num_layers           = 1;
    sp_config.type                    = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle       = 7;
    sp_config.sp_dspark_mask_token_id = 15;
    ResourceContext resource_context;
    for (int budget : {0, 3, 4, 5, 7}) {
        SCOPED_TRACE(budget);
        sp_config.sp_dspark_verify_tokens = budget;
        const int steps                   = budget == 0 ? 7 : budget;
        const int width                   = steps + 1;
        auto      first                   = createContextStream(model_config, runtime_config, resource_context, {5}, 1);
        auto      second = createContextStream(model_config, runtime_config, resource_context, {6, 7}, 2);
        // This geometry test asserts duplicated history contents. Require an
        // actual history consumer; an empty spec artifact no longer needs it.
        first->generateConfig()->repetition_penalty = 1.1f;
        first->setScoreLen(width);
        second->setScoreLen(width);
        StreamGroups            streams({first, second});
        MtpBatchStreamProcessor processor(
            model_config, pd_sep_config, profiling_config, cache_config, sp_config, false);
        GptModelInputs input;
        input.combo_tokens = torch::zeros({2 * width}, torch::kInt32);
        GptModelOutputs output;
        output.logits = torch::zeros({2 * width, 16}, torch::TensorOptions().device(torch::kCUDA));
        SpecLogitsVerifyRunner::LaunchResult artifact;
        artifact.has_active_processor = true;
        auto sampler                  = processor.gatherSpecSamplerInput(streams, input, output, artifact);
        ASSERT_TRUE(sampler.ok());
        EXPECT_EQ(sampler.value().spec_propose_step, steps);
        EXPECT_EQ(sampler.value().logits.size(0), 2 * width);
        const auto ids = sampler.value().token_ids;
        EXPECT_EQ(ids.size(0), 2 * width);
        for (int row = 0; row < 2 * width; ++row) {
            EXPECT_EQ(ids[row][0].item<int32_t>(), row < width ? 5 : 6);
        }
        output.logits = torch::zeros({2 * width + 1, 16}, torch::TensorOptions().device(torch::kCUDA));
        EXPECT_FALSE(processor.gatherSpecSamplerInput(streams, input, output).ok());
    }
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkCompactVerifySampleSlotsAndHistoryFallback) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_config;
    CacheConfig                 cache_config;
    cache_config.group_types    = {CacheGroupType::FULL};
    model_config.max_seq_len    = 128;
    model_config.vocab_size     = 16;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 7;
    ResourceContext resource_context;
    for (int budget : {4, 5, 6, 7}) {
        for (int history_case = 0; history_case < 12; ++history_case) {
            SCOPED_TRACE(budget);
            SCOPED_TRACE(history_case);
            sp_config.type                    = history_case == 8 ? SP_TYPE_MTP : SP_TYPE_DSPARK;
            sp_config.sp_dspark_verify_tokens = history_case == 8 ? 0 : budget;
            const int width                   = history_case == 8 ? 8 : budget + 1;
            auto      stream                  = createContextStream(model_config,
                                              runtime_config,
                                              resource_context,
                                                                    {5, 6},
                                              1,
                                              history_case == 10 ? vector<int>{7} : vector<int>{},
                                              history_case == 10 ? vector<int>{8, 9} : vector<int>{},
                                              history_case == 10);
            auto&     config                  = *stream->generateConfig();
            if (history_case == 1)
                config.repetition_penalty = 1.1f;
            if (history_case == 2)
                config.presence_penalty = 0.1f;
            if (history_case == 3)
                config.frequency_penalty = 0.1f;
            if (history_case == 4)
                config.no_repeat_ngram_size = 2;
            if (history_case == 0)
                config.no_repeat_ngram_size.reset();
            if (history_case == 9)
                config.no_repeat_ngram_size = 0;
            if (history_case == 10) {
                ASSERT_FALSE(stream->getAllLogitsProcessorPtr().empty());
            }
            if (history_case == 11) {
                auto unknown = std::make_shared<UnknownHistoryProcessorForCompactTest>();
                ASSERT_TRUE(unknown->requiresTokenHistory());
                stream->logits_processor_list_.push_back(unknown);
            }
            stream->setScoreLen(width);
            StreamGroups            streams({stream});
            MtpBatchStreamProcessor processor(
                model_config, pd_sep_config, profiling_config, cache_config, sp_config, false);
            GptModelInputs model_input;
            model_input.combo_tokens = torch::zeros({width}, torch::kInt32);
            GptModelOutputs model_output;
            model_output.logits = torch::zeros({width, 16}, torch::TensorOptions().device(torch::kCUDA));
            SpecLogitsVerifyRunner::LaunchResult artifact;
            if (history_case == 5)
                artifact.has_active_processor = true;
            if (history_case == 6)
                artifact.spec_vocab_mask_gpu = torch::zeros({width, 16}, torch::kBool);
            if (history_case == 7)
                artifact.spec_cap_gpu = torch::zeros({1}, torch::kInt32);
            auto gathered = processor.gatherSpecSamplerInput(streams, model_input, model_output, artifact);
            ASSERT_TRUE(gathered.ok());
            const auto& inputs = gathered.value();
            const bool  expected_compact =
                history_case == 0 || history_case == 5 || history_case == 9 || history_case == 10;
            EXPECT_EQ(inputs.compact_token_ids, expected_compact);
            if (history_case == 0) {
                EXPECT_FALSE(config.no_repeat_ngram_size.has_value());
            }
            if (history_case == 9) {
                ASSERT_EQ(config.no_repeat_ngram_size.value(), 0);
            }
            EXPECT_EQ(inputs.step, 2 + width - 1);
            EXPECT_EQ(inputs.input_lengths[0].item<int32_t>(), 2);
            EXPECT_EQ(inputs.sequence_lengths[0].item<int32_t>(),
                      expected_compact || history_case == 8 ? 2 + width - 1 : 2);
            if (expected_compact) {
                EXPECT_EQ(inputs.token_ids.sizes(), torch::IntArrayRef({width, 1}));
                if (history_case == 5 || history_case == 10) {
                    ASSERT_NE(inputs.logits_processor_states_ptr, nullptr);
                    EXPECT_FALSE(inputs.logits_processor_states_ptr->requiresTokenHistory());
                    if (history_case == 10) {
                        inputs.logits_processor_states_ptr->batchProcess(inputs);
                        EXPECT_EQ(inputs.logits[0][7].item<float>(), -std::numeric_limits<float>::max());
                        EXPECT_EQ(inputs.logits[0][8].item<float>(), -std::numeric_limits<float>::max());
                    }
                } else {
                    EXPECT_FALSE(inputs.logits_processor_states_ptr);
                }
                Sampler sampler(SamplerInitParams{});
                auto    output = sampler.forward(inputs);
                EXPECT_EQ(output.token_ids.sizes(), torch::IntArrayRef({width, 1}));
                EXPECT_TRUE(output.success.cpu().all().item<bool>());
            } else {
                EXPECT_EQ(inputs.token_ids.size(1), inputs.step + 1);
                EXPECT_EQ(inputs.token_ids[0][0].item<int32_t>(), 5);
                EXPECT_EQ(inputs.token_ids[0][1].item<int32_t>(), 6);
                if (history_case == 11) {
                    ASSERT_NE(inputs.logits_processor_states_ptr, nullptr);
                    EXPECT_TRUE(inputs.logits_processor_states_ptr->requiresTokenHistory());
                }
            }
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkHistoryPenaltiesUseEachVerifiedPrefix) {
    ModelConfig model;
    model.max_seq_len = 128;
    model.vocab_size  = 16;
    model.num_layers  = 1;
    RuntimeConfig               runtime;
    ResourceContext             resources;
    PDSepConfig                 pd;
    ProfilingDebugLoggingConfig profiling;
    CacheConfig                 cache;
    cache.group_types = {CacheGroupType::FULL};
    const std::vector<std::vector<int>> histories{{1, 2, 1}, {3, 4}};
    for (int verify : {4, 5, 6, 7}) {
        for (int penalty_case = 0; penalty_case < 5; ++penalty_case) {
            SCOPED_TRACE(::testing::Message() << "verify=" << verify << " penalty=" << penalty_case);
            const int                  width = verify + 1;
            SpeculativeExecutionConfig spec;
            spec.type                    = SP_TYPE_DSPARK;
            spec.gen_num_per_cycle       = 7;
            spec.sp_dspark_verify_tokens = verify;
            std::list<GenerateStreamPtr> streams;
            auto                         verify_ids      = torch::zeros({2, width}, torch::kInt32);
            auto                         expected_logits = torch::full({2 * width, 16}, 2.0f);
            for (int b = 0; b < 2; ++b) {
                auto stream = createContextStream(model, runtime, resources, histories[b], b + 1);
                stream->setScoreLen(width);
                auto& config                = *stream->generateConfig();
                config.do_sample            = true;
                config.top_k                = 0;
                config.top_p                = 1.0f;
                config.temperature          = 1.0f;
                config.repetition_penalty   = penalty_case == 0 || penalty_case == 4 ? 1.5f : 1.0f;
                config.presence_penalty     = penalty_case == 1 || penalty_case == 4 ? 0.2f : 0.0f;
                config.frequency_penalty    = penalty_case == 2 || penalty_case == 4 ? 0.1f : 0.0f;
                config.no_repeat_ngram_size = penalty_case >= 3 ? 2 : 0;
                streams.push_back(stream);
                verify_ids[b][0].fill_(histories[b].back());
                for (int p = 1; p < width; ++p) {
                    verify_ids[b][p].fill_(b == 0 ? (p % 2 ? 2 : 1) : (p % 2 ? 3 : 4));
                }
                auto history = histories[b];
                for (int p = 0; p < width; ++p) {
                    std::vector<int> counts(16, 0);
                    for (int token : history)
                        ++counts[token];
                    for (int token = 0; token < 16; ++token) {
                        if (counts[token]) {
                            expected_logits[b * width + p][token].fill_(2.0f / config.repetition_penalty
                                                                        - config.presence_penalty
                                                                        - config.frequency_penalty * counts[token]);
                        }
                    }
                    if (config.no_repeat_ngram_size.value() == 2) {
                        for (size_t j = 0; j + 1 < history.size(); ++j) {
                            if (history[j] == history.back()) {
                                expected_logits[b * width + p][history[j + 1]].fill_(
                                    -std::numeric_limits<float>::infinity());
                            }
                        }
                    }
                    if (p < verify)
                        history.push_back(verify_ids[b][p + 1].item<int32_t>());
                }
            }
            MtpBatchStreamProcessor processor(model, pd, profiling, cache, spec, false);
            GptModelInputs          model_input;
            // Simulate a CP-mutated input; the preserved global verify tensor
            // must own sampler history, not the post-forward combo tokens.
            model_input.combo_tokens = torch::tensor({15}, torch::kInt32);
            GptModelOutputs output;
            output.logits = torch::full({2 * width, 16}, 2.0f, torch::device(torch::kCUDA));
            auto gathered =
                processor.gatherSpecSamplerInput(StreamGroups(streams), model_input, output, {}, verify_ids.cuda());
            ASSERT_TRUE(gathered.ok()) << gathered.status();
            const auto& input = gathered.value();
            EXPECT_FALSE(input.compact_token_ids);
            EXPECT_TRUE(input.token_history_lengths_are_counts);
            for (int b = 0; b < 2; ++b) {
                auto history = histories[b];
                for (int p = 0; p < width; ++p) {
                    const int row = b * width + p;
                    EXPECT_EQ(input.sequence_lengths[row].item<int32_t>(), history.size());
                    for (size_t j = 0; j < history.size(); ++j) {
                        EXPECT_EQ(input.token_ids[row][j].item<int32_t>(), history[j]);
                    }
                    if (p < verify)
                        history.push_back(verify_ids[b][p + 1].item<int32_t>());
                }
            }
            Sampler sampler(SamplerInitParams{});
            auto    sampled = sampler.forward(input);
            ASSERT_TRUE(sampled.all_probs.defined());
            EXPECT_TRUE(torch::allclose(sampled.all_probs.cpu(), torch::softmax(expected_logits, -1), 1e-5, 1e-6));
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkCompactVerifyMixedBatchProcessorCapabilities) {
    ModelConfig model;
    model.max_seq_len = 128;
    model.vocab_size  = 16;
    model.num_layers  = 1;
    RuntimeConfig               runtime;
    ResourceContext             resources;
    PDSepConfig                 pd;
    ProfilingDebugLoggingConfig profiling;
    CacheConfig                 cache;
    cache.group_types = {CacheGroupType::FULL};
    for (int verify : {4, 5, 6, 7}) {
        for (int processor_case : {0, 1, 2}) {
            const bool think_processor = processor_case == 1;
            SCOPED_TRACE(::testing::Message() << "verify=" << verify << " think=" << think_processor);
            SpeculativeExecutionConfig spec;
            spec.type                    = SP_TYPE_DSPARK;
            spec.gen_num_per_cycle       = 7;
            spec.sp_dspark_verify_tokens = verify;
            const int width              = verify + 1;
            auto      neutral            = createContextStream(model, runtime, resources, {1, 2}, 1);
            auto      history            = createContextStream(model,
                                               runtime,
                                               resources,
                                                               {5, 6},
                                               2,
                                               think_processor ? vector<int>{7} : vector<int>{},
                                               think_processor ? vector<int>{8, 9} : vector<int>{},
                                               think_processor);
            if (processor_case == 0)
                history->generateConfig()->repetition_penalty = 1.1f;
            if (processor_case == 2)
                history->logits_processor_list_.push_back(std::make_shared<UnknownHistoryProcessorForCompactTest>());
            ASSERT_TRUE(neutral->getAllLogitsProcessorPtr().empty());
            if (think_processor) {
                ASSERT_FALSE(history->getAllLogitsProcessorPtr().empty());
            }
            neutral->setScoreLen(width);
            history->setScoreLen(width);
            MtpBatchStreamProcessor processor(model, pd, profiling, cache, spec, false);
            GptModelInputs          model_input;
            model_input.combo_tokens = torch::zeros({2 * width}, torch::kInt32);
            GptModelOutputs output;
            output.logits = torch::zeros({2 * width, 16}, torch::TensorOptions().device(torch::kCUDA));
            auto gathered = processor.gatherSpecSamplerInput(StreamGroups({neutral, history}), model_input, output);
            ASSERT_TRUE(gathered.ok());
            const auto& input = gathered.value();
            EXPECT_EQ(input.compact_token_ids, think_processor);
            EXPECT_EQ(input.token_ids.sizes(), torch::IntArrayRef({2 * width, think_processor ? 1 : 2 + width}));
            if (!think_processor) {
                for (int row = 0; row < 2 * width; ++row) {
                    EXPECT_EQ(input.token_ids[row][0].item<int32_t>(), row < width ? 1 : 5);
                    EXPECT_EQ(input.token_ids[row][1].item<int32_t>(), row < width ? 2 : 6);
                }
            }
            if (think_processor) {
                ASSERT_NE(input.logits_processor_states_ptr, nullptr);
                input.logits_processor_states_ptr->batchProcess(input);
                for (int row = 0; row < 2 * width; ++row) {
                    EXPECT_EQ(input.logits[row][7].item<float>(),
                              row < width ? 0.0f : -std::numeric_limits<float>::max());
                }
            }
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkCompactThinkMatchesFullHistoryWithSpecArtifacts) {
    ModelConfig model;
    model.max_seq_len = 128;
    model.vocab_size  = 16;
    model.num_layers  = 1;
    RuntimeConfig               runtime;
    ResourceContext             resources;
    PDSepConfig                 pd;
    ProfilingDebugLoggingConfig profiling;
    CacheConfig                 cache;
    cache.group_types = {CacheGroupType::FULL};
    for (int verify : {4, 5, 6, 7}) {
        for (bool mixed : {false, true}) {
            for (bool use_spec_artifact : {false, true}) {
                SCOPED_TRACE(::testing::Message()
                             << "verify=" << verify << " mixed=" << mixed << " artifact=" << use_spec_artifact);
                SpeculativeExecutionConfig spec;
                spec.type                    = SP_TYPE_DSPARK;
                spec.gen_num_per_cycle       = 7;
                spec.sp_dspark_verify_tokens = verify;
                const int width = verify + 1, batch = mixed ? 2 : 1, rows = batch * width;
                auto      think = createContextStream(model, runtime, resources, {5, 6}, 1, {7}, {8, 9}, true);
                think->generateConfig()->top_k = 1;
                think->setScoreLen(width);
                auto processor_ptr =
                    std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(think->getAllLogitsProcessorPtr().at(0));
                ASSERT_NE(processor_ptr, nullptr);
                ASSERT_FALSE(processor_ptr->requiresTokenHistory());
                ASSERT_TRUE(think->generateConfig()->enable_think_logits_processor);
                ASSERT_EQ(think->generateConfig()->thinking_mode, ThinkingMode::DISABLED);
                ASSERT_EQ(think->generateConfig()->max_thinking_tokens, 0);
                const auto                   accepted_before = processor_ptr->acceptedTokenLen();
                std::list<GenerateStreamPtr> streams;
                if (mixed) {
                    auto neutral                     = createContextStream(model, runtime, resources, {1, 2}, 2);
                    neutral->generateConfig()->top_k = 1;
                    neutral->setScoreLen(width);
                    streams.push_back(neutral);
                }
                streams.push_back(think);
                SpecLogitsVerifyRunner               runner;
                SpecLogitsVerifyRunner::LaunchResult artifact;
                if (use_spec_artifact) {
                    SpecLogitsVerifyRunner::LaunchTask task;
                    task.total_streams = batch;
                    task.propose_step  = verify;
                    task.vocab_size    = 16;
                    task.draft_tokens  = torch::zeros({batch, verify}, torch::kInt32);
                    task.active.push_back({processor_ptr,
                                           static_cast<size_t>(batch - 1),
                                           0,
                                           static_cast<uint64_t>(think->streamId()),
                                           2,
                                           0});
                    artifact = runner.buildInline(task);
                    ASSERT_TRUE(artifact.has_active_processor);
                    ASSERT_TRUE(artifact.spec_vocab_mask_gpu.defined());
                    ASSERT_TRUE(artifact.spec_cap_gpu.defined());
                    ASSERT_NE(artifact.ready_event, nullptr);
                    ASSERT_NE(artifact.consumed_event, nullptr);
                }
                MtpBatchStreamProcessor processor(model, pd, profiling, cache, spec, false);
                GptModelInputs          model_input;
                GptModelOutputs         model_output;
                model_output.logits = torch::zeros({rows, 16}, torch::TensorOptions().device(torch::kCUDA));
                model_output.logits.select(1, 7).fill_(10.0f);
                model_output.logits.select(1, 8).fill_(9.0f);
                model_output.logits.select(1, 9).fill_(8.0f);
                auto gathered =
                    processor.gatherSpecSamplerInput(StreamGroups(streams), model_input, model_output, artifact);
                ASSERT_TRUE(gathered.ok());
                auto compact = gathered.value();
                ASSERT_TRUE(compact.compact_token_ids);
                ASSERT_NE(compact.logits_processor_states_ptr, nullptr);
                EXPECT_FALSE(compact.logits_processor_states_ptr->requiresTokenHistory());
                if (use_spec_artifact) {
                    EXPECT_EQ(compact.spec_mask_ready_event, artifact.ready_event);
                    EXPECT_EQ(compact.spec_mask_consumed_event, artifact.consumed_event);
                    EXPECT_EQ(compact.spec_applied_processors, artifact.applied_processors);
                    EXPECT_EQ(compact.spec_propose_step, verify);
                    EXPECT_TRUE(torch::equal(compact.spec_vocab_mask_gpu, artifact.spec_vocab_mask_gpu));
                    EXPECT_TRUE(torch::equal(compact.spec_cap_gpu, artifact.spec_cap_gpu));
                }
                // Same live processor snapshot, semantic lengths, row order and
                // artifacts; only physical output stride and its explicit flag differ.
                auto full              = compact;
                full.compact_token_ids = false;
                full.token_ids = torch::zeros({rows, static_cast<int64_t>(full.step + 1)}, torch::kInt32).pin_memory();
                for (int row = 0; row < rows; ++row) {
                    full.token_ids[row][0].fill_(mixed && row < width ? 1 : 5);
                    full.token_ids[row][1].fill_(mixed && row < width ? 2 : 6);
                }
                full.logits = compact.logits.clone();
                if (compact.all_probs.defined())
                    full.all_probs = compact.all_probs.clone();
                if (compact.cum_log_probs.defined())
                    full.cum_log_probs = compact.cum_log_probs.clone();
                // This is greedy sampling; preserve any per-request RNG state
                // nevertheless so the two paths start at identical offsets.
                std::vector<torch::Tensor> generator_states;
                for (auto& generator : compact.generator) {
                    generator_states.push_back(generator.defined() ? generator.get_state().clone() : torch::Tensor());
                }
                Sampler sampler(SamplerInitParams{});
                auto    full_output = sampler.forward(full);
                for (size_t i = 0; i < compact.generator.size(); ++i) {
                    if (generator_states[i].defined())
                        compact.generator[i].set_state(generator_states[i]);
                }
                auto compact_output = sampler.forward(compact);
                EXPECT_TRUE(
                    torch::equal(full_output.token_ids.select(1, full.step), compact_output.token_ids.select(1, 0)));
                EXPECT_TRUE(torch::equal(full_output.success, compact_output.success));
                EXPECT_TRUE(torch::equal(full.logits, compact.logits));
                EXPECT_EQ(full_output.all_probs.defined(), compact_output.all_probs.defined());
                if (full_output.all_probs.defined()) {
                    EXPECT_TRUE(torch::equal(full_output.all_probs, compact_output.all_probs));
                }
                EXPECT_EQ(full_output.cum_log_probs.defined(), compact_output.cum_log_probs.defined());
                if (full_output.cum_log_probs.defined()) {
                    EXPECT_TRUE(torch::equal(full_output.cum_log_probs, compact_output.cum_log_probs));
                }
                for (int row = 0; row < rows; ++row) {
                    EXPECT_EQ(compact_output.token_ids[row][0].item<int32_t>(), mixed && row < width ? 7 : 9);
                }
                EXPECT_EQ(processor_ptr->acceptedTokenLen(), accepted_before);
                if (artifact.consumed_event) {
                    artifact.consumed_event->record(cuda_graph::graphGetCurrentStream());
                    artifact.consumed_event->synchronize();
                }
            }
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testCompactSamplerEntryRejectsHistoryDependentInputs) {
    struct ScopedThrowInsteadOfAbort {
        bool saved = StaticConfig::user_ft_core_dump_on_exception;
        ScopedThrowInsteadOfAbort() {
            StaticConfig::user_ft_core_dump_on_exception = false;
        }
        ~ScopedThrowInsteadOfAbort() {
            StaticConfig::user_ft_core_dump_on_exception = saved;
        }
    } exception_guard;
    Sampler sampler(SamplerInitParams{});
    // All invalid cases must fail at the entry contract, before processors,
    // stream copies, or kernels. No death-test subprocess can inherit CUDA.
    for (int invalid_case = 0; invalid_case < 15; ++invalid_case) {
        SCOPED_TRACE(invalid_case);
        SamplerInputs input;
        input.compact_token_ids = true;
        input.phase             = LogitsProcessorPhase::MTP_VERIFY;
        input.batch_size = input.batch_size_out = 1;
        input.step                              = 81920;
        input.vocab_size                        = 16;
        input.token_ids                         = torch::zeros({1, 1}, torch::kInt32);
        input.num_beams_in                      = torch::ones({1}, torch::kInt64);
        input.num_beams_out                     = torch::ones({1}, torch::kInt64);
        switch (invalid_case) {
            case 0:
                input.phase = LogitsProcessorPhase::NORMAL_DECODE;
                break;
            case 1:
                input.phase = LogitsProcessorPhase::DRAFT_SAMPLE;
                break;
            case 2:
                input.repetition_penalty = torch::full({1}, 1.1f);
                break;
            case 3:
                input.presence_penalty = torch::full({1}, 0.1f);
                break;
            case 4:
                input.frequency_penalty = torch::full({1}, 0.1f);
                break;
            case 5:
                input.no_repeat_ngram_size = torch::full({1}, 2, torch::kInt32);
                break;
            case 6:
                input.num_beams_in.fill_(2);
                break;
            case 7:
                input.num_beams_out.fill_(2);
                break;
            case 8:
                input.batch_size_out = 2;
                break;
            case 9:
                input.logits_processor_states_ptr = std::make_shared<LogitsProcessorStates>();
                input.logits_processor_states_ptr->insert(
                    std::make_shared<UnknownHistoryProcessorForCompactTest>(), 0, 1);
                break;
            case 10:
                input.spec_vocab_mask_gpu = torch::zeros({1, 16}, torch::kBool);
                break;
            case 11:
                input.spec_cap_gpu = torch::zeros({1}, torch::kInt32);
                break;
            case 12:
                input.spec_applied_processors.push_back(SpecLogitsProcessorId{1, 0});
                break;
            case 13:
                input.token_ids = torch::zeros({1, 2}, torch::kInt32);
                break;
            case 14:
                input.token_ids = torch::zeros({1}, torch::kInt32);
                break;
        }
        try {
            sampler.forward(input);
            ADD_FAILURE() << "compact sampler accepted invalid case " << invalid_case;
        } catch (const std::exception& error) {
            // Do not accept an unrelated downstream failure as a gate pass.
            EXPECT_NE(std::string(error.what()).find("compact verify sample slots cannot be used"), std::string::npos);
        } catch (...) {
            ADD_FAILURE() << "unexpected non-standard exception";
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkCanExcludeAnchorFromLmRows) {
    ModelConfig                 model_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_config;
    CacheConfig                 cache_config;
    cache_config.group_types               = {CacheGroupType::FULL};
    model_config.max_seq_len               = 128;
    model_config.vocab_size                = 16;
    model_config.num_layers                = 1;
    sp_config.type                         = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle            = 3;
    sp_config.sp_dspark_mask_token_id      = 15;
    sp_config.sp_dspark_sample_from_anchor = false;

    MtpBatchStreamProcessor processor(model_config, pd_sep_config, profiling_config, cache_config, sp_config, false);
    TensorHolder            holder;
    GptModelInputs          input;
    processor.buildDSparkProposeInput(
        input, torch::tensor({2, 7}, torch::kInt32), torch::tensor({11, 23}, torch::kInt32), holder);
    EXPECT_EQ(toVec<int32_t>(input.combo_tokens), (std::vector<int32_t>{2, 15, 15, 15, 7, 15, 15, 15}));
    EXPECT_EQ(toVec<int32_t>(input.input_lengths), (std::vector<int32_t>{4, 4}));
    EXPECT_EQ(toVec<int32_t>(input.lm_output_indexes), (std::vector<int32_t>{1, 2, 3, 5, 6, 7}));
}

TEST_F(MtpBatchStreamProcessorTest, testSingleRequestDSparkSyntheticRound) {
    ModelConfig                 model_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_config;
    CacheConfig                 cache_config;
    cache_config.group_types               = {CacheGroupType::FULL};
    model_config.max_seq_len               = 128;
    model_config.vocab_size                = 6;
    model_config.num_layers                = 1;
    sp_config.type                         = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle            = 3;
    sp_config.sp_dspark_mask_token_id      = 5;
    sp_config.sp_dspark_sample_from_anchor = true;

    MtpBatchStreamProcessor processor(model_config, pd_sep_config, profiling_config, cache_config, sp_config, false);
    TensorHolder            holder;
    auto                    anchors        = torch::tensor({0}, torch::kInt32);
    auto                    committed_ends = torch::tensor({9}, torch::kInt32);

    GptModelInputs proposal_input;
    processor.buildDSparkProposeInput(proposal_input, anchors, committed_ends, holder);
    EXPECT_EQ(toVec<int32_t>(proposal_input.combo_tokens), (std::vector<int32_t>{0, 5, 5}));

    auto                            float_cuda  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto                            base_logits = torch::tensor({{0.0f, 20.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                                                                 {0.0f, 0.0f, 20.0f, 0.0f, 0.0f, 0.0f},
                                                                 {0.0f, 0.0f, 0.0f, 20.0f, 0.0f, 0.0f}},
                                     float_cuda);
    auto                            temperature = torch::full({1}, 1.0e-6f, float_cuda);
    auto                            markov_w1   = torch::zeros({6, 2}, float_cuda);
    auto                            markov_w2   = torch::zeros({6, 2}, float_cuda);
    speculative::SpeculativeSampler sampler(torch::Tensor(), 3, speculative::DraftProposalMode::LEGACY);
    auto draft = sampler.sampleDSparkDraft(base_logits, anchors.to(torch::kCUDA), temperature, markov_w1, markov_w2, 6);
    EXPECT_EQ(toVec<int32_t>(draft.token_ids), (std::vector<int32_t>{1, 2, 3}));

    GptModelInputs verify_input;
    processor.prepareDSparkTargetVerifyModelInput(verify_input, anchors, committed_ends, draft.token_ids, holder);
    EXPECT_EQ(toVec<int32_t>(verify_input.combo_tokens), (std::vector<int32_t>{0, 1, 2, 3}));

    auto target_features = torch::arange(16, float_cuda).reshape({4, 4});
    processor.updateDecodePostDSparkCommitInput(verify_input, target_features, 1);
    EXPECT_EQ(verify_input.last_hidden_states.data_ptr(), target_features.data_ptr());

    // The next proposal rewrites the complete fixed-width query block: no
    // rejected/stale token from the previous verify row remains reachable.
    processor.buildDSparkProposeInput(
        proposal_input, torch::tensor({2}, torch::kInt32), torch::tensor({11}, torch::kInt32), holder);
    EXPECT_EQ(toVec<int32_t>(proposal_input.combo_tokens), (std::vector<int32_t>{2, 5, 5}));
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

TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputRejectsWrongLogitRows) {
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
    auto            stream1 = createContextStream(model_config, runtime_config, resource_context, {5}, 1);
    auto            stream2 = createContextStream(model_config, runtime_config, resource_context, {6, 7}, 2);
    stream1->setScoreLen(sp_config.gen_num_per_cycle + 1);
    stream2->setScoreLen(sp_config.gen_num_per_cycle + 1);

    auto stream_groups = StreamGroups({stream1, stream2});
    auto processor     = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelInputs  model_inputs;
    GptModelOutputs model_output;
    model_output.logits = torch::empty({7, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto sampler_inputs_status = processor.gatherSpecSamplerInput(stream_groups, model_inputs, model_output);
    ASSERT_FALSE(sampler_inputs_status.ok());
    EXPECT_NE(std::string(sampler_inputs_status.status().message()).find("target verify logits row mismatch"),
              std::string::npos);
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

TEST_F(MtpBatchStreamProcessorTest, testDSparkFailedDecodeDoesNotCommitTokensLengthOrAnchor) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_config;
    CacheConfig                 cache_config;
    cache_config.group_types    = {CacheGroupType::FULL};
    model_config.max_seq_len    = 128;
    model_config.vocab_size     = 16;
    model_config.num_layers     = 1;
    sp_config.type              = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle = 3;
    ResourceContext resource_context;
    auto            failed  = createContextStream(model_config, runtime_config, resource_context, {2, 3}, 1);
    auto            healthy = createContextStream(model_config, runtime_config, resource_context, {4, 5}, 2);
    setSpOutputTokens(failed->getSPOutputBuffer(), {3});
    setSpOutputTokens(healthy->getSPOutputBuffer(), {5});
    failed->setIsContextStream(false);
    healthy->setIsContextStream(false);
    const auto                            old_anchor = failed->getSPOutputBuffer()->target_token_gpu.clone();
    speculative::SpeculativeSamplerOutput output;
    output.accept_tokens_cpu = torch::tensor({{15, 15, 15, 15}, {6, 0, 0, 0}}, torch::kInt32);
    output.accept_len_cpu    = torch::tensor({1, 1}, torch::kInt32);
    output.success_cpu       = torch::tensor({false, true}, torch::kBool);
    output.accept_tokens     = output.accept_tokens_cpu.to(torch::kCUDA);
    output.accept_len        = output.accept_len_cpu.to(torch::kCUDA);
    output.success           = output.success_cpu.to(torch::kCUDA);
    output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
    MtpBatchStreamProcessor processor(model_config, pd_sep_config, profiling_config, cache_config, sp_config, false);
    EXPECT_TRUE(processor.dispatchDecode(StreamGroups({failed, healthy}), output, MergedOutput{}).ok());
    EXPECT_TRUE(failed->hasError());
    EXPECT_EQ(failed->seqLength(), 2);
    EXPECT_EQ(toVec<int32_t>(failed->completeTokenIds().narrow(1, 0, 2)), (vector<int32_t>{2, 3}));
    EXPECT_TRUE(torch::equal(failed->getSPOutputBuffer()->target_token_gpu, old_anchor));
    EXPECT_FALSE(healthy->hasError());
    EXPECT_EQ(healthy->seqLength(), 3);
    EXPECT_EQ(toVec<int32_t>(healthy->completeTokenIds().narrow(1, 0, 3)), (vector<int32_t>{4, 5, 6}));
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
    EXPECT_EQ(model_input.value().last_hidden_states_layout, MtpHiddenStatesLayout::GLOBAL);
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
    EXPECT_EQ(std::vector<int>{}, toVec<int>(sequence_lengths));

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

    // Device-state publishes committed lengths. Target verify starts by
    // replaying the last committed token, whose zero-based position is
    // committed_len - 1.
    vector<int> expect_prefix_lengths = {6, 3};
    EXPECT_TRUE(model_input.prefix_lengths.is_cuda());
    EXPECT_EQ(expect_prefix_lengths, toVec<int>(model_input.prefix_lengths));
    EXPECT_TRUE(model_input.sequence_lengths.is_cuda());
    EXPECT_EQ(std::vector<int>{}, toVec<int>(model_input.sequence_lengths));

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

    const auto                          cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    GenerateStream::MtpAsyncDeviceState state1;
    state1.next_seq_len_gpu   = torch::full({1}, 7, cuda_i32);
    state1.propose_tokens_gpu = torch::tensor({{2, 3}}, torch::kInt32).to(torch::kCUDA);
    stream1->setMtpAsyncDeviceState(std::move(state1));

    GenerateStream::MtpAsyncDeviceState state2;
    state2.next_seq_len_gpu   = torch::full({1}, 4, cuda_i32);
    state2.propose_tokens_gpu = torch::tensor({{3, 1}}, torch::kInt32).to(torch::kCUDA);
    stream2->setMtpAsyncDeviceState(std::move(state2));

    processor.prepareDecodeDraftModelInput(stream_groups, model_input, holder);

    auto        combo_tokens        = model_input.combo_tokens;
    vector<int> expect_combo_tokens = {3, 1};
    EXPECT_EQ(expect_combo_tokens, toVec<int>(combo_tokens));

    auto        lm_output_indexes        = model_input.lm_output_indexes;
    vector<int> expect_lm_output_indexes = {0, 1};
    EXPECT_TRUE(lm_output_indexes.is_cuda());
    EXPECT_EQ(expect_lm_output_indexes, toVec<int>(lm_output_indexes));

    vector<int> expect_sequence_lengths = {7, 4};
    EXPECT_TRUE(model_input.sequence_lengths.is_cuda());
    EXPECT_EQ(expect_sequence_lengths, toVec<int>(model_input.sequence_lengths));
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

TEST_F(MtpBatchStreamProcessorTest, testPrefillMetadataShiftOnlyForMultimodal) {
    ModelConfig                 model_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    SpeculativeExecutionConfig  sp_config;
    sp_config.gen_num_per_cycle = 1;

    auto processor = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder    holder;
    GptModelOutputs model_output;
    SamplerOutput   sampler_output;
    sampler_output.token_ids = torch::tensor({{30}, {40}}, torch::kInt32);

    GptModelInputs multimodal_input;
    multimodal_input.combo_tokens       = torch::tensor({10, 11, 12, 20, 21}, torch::kInt32);
    multimodal_input.input_lengths      = torch::tensor({3, 2}, torch::kInt32);
    multimodal_input.combo_position_ids = torch::tensor({5, 6, 7, 10, 11}, torch::kInt32);
    multimodal_input.text_tokens_mask   = torch::tensor({1, 0, 1, 1, 0}, torch::kInt32);
    multimodal_input.mm_features_locs   = torch::tensor({1, 4}, torch::kInt32);
    multimodal_input.multimodal_features =
        std::vector<torch::Tensor>{torch::ones({1, 4}, torch::kFloat32), torch::ones({1, 4}, torch::kFloat32)};

    processor.updatePrefillPostDraftModelInput(multimodal_input, model_output, sampler_output, holder);
    EXPECT_EQ(std::vector<int>({11, 12, 30, 21, 40}), toVec<int>(multimodal_input.combo_tokens));
    EXPECT_EQ(std::vector<int>({0, 1, 1, 0, 1}), toVec<int>(multimodal_input.text_tokens_mask));
    EXPECT_EQ(std::vector<int>({6, 7, 8, 11, 12}), toVec<int>(multimodal_input.combo_position_ids));
    EXPECT_EQ(std::vector<int>({0, 3}), toVec<int>(multimodal_input.mm_features_locs));

    // A feature beginning at token zero becomes a negative loc after MTP
    // removes the first token; the injector drops the reused feature head.
    GptModelInputs image_first_input;
    image_first_input.combo_tokens       = torch::tensor({10, 11, 12, 20, 21}, torch::kInt32);
    image_first_input.input_lengths      = torch::tensor({3, 2}, torch::kInt32);
    image_first_input.combo_position_ids = torch::tensor({5, 6, 7, 10, 11}, torch::kInt32);
    image_first_input.text_tokens_mask   = torch::tensor({1, 1, 1, 1, 1}, torch::kInt32);
    image_first_input.mm_features_locs   = torch::tensor({0, 3}, torch::kInt32);
    image_first_input.multimodal_features =
        std::vector<torch::Tensor>{torch::ones({2, 4}, torch::kFloat32), torch::ones({2, 4}, torch::kFloat32)};
    image_first_input.mm_extra_input =
        std::vector<torch::Tensor>{torch::arange(16, torch::kFloat32), torch::arange(16, 32, torch::kFloat32)};
    processor.updatePrefillPostDraftModelInput(image_first_input, model_output, sampler_output, holder);
    EXPECT_EQ(std::vector<int>({0, 3}), toVec<int>(image_first_input.mm_features_locs));
    ASSERT_EQ(2U, image_first_input.multimodal_features->size());
    EXPECT_EQ(1, image_first_input.multimodal_features->at(0).size(0));
    EXPECT_EQ(1, image_first_input.multimodal_features->at(1).size(0));
    ASSERT_TRUE(image_first_input.mm_extra_input.has_value());
    ASSERT_EQ(2U, image_first_input.mm_extra_input->size());
    EXPECT_EQ(8, image_first_input.mm_extra_input->at(0).numel());
    EXPECT_EQ(8, image_first_input.mm_extra_input->at(1).numel());

    // MROPE-style position IDs are flattened [tokens, 3] rows. Each appended
    // row follows the same max-component rule as the position generator.
    GptModelInputs mrope_input;
    mrope_input.combo_tokens       = torch::tensor({10, 11, 12, 20, 21}, torch::kInt32);
    mrope_input.input_lengths      = torch::tensor({3, 2}, torch::kInt32);
    mrope_input.combo_position_ids = torch::tensor({0, 0, 0, 1, 2, 3, 4, 5, 6, 10, 10, 10, 11, 12, 13}, torch::kInt32);
    mrope_input.text_tokens_mask   = torch::tensor({1, 1, 1, 1, 1}, torch::kInt32);
    mrope_input.mm_features_locs   = torch::tensor({1, 4}, torch::kInt32);
    mrope_input.multimodal_features =
        std::vector<torch::Tensor>{torch::ones({1, 4}, torch::kFloat32), torch::ones({1, 4}, torch::kFloat32)};
    processor.updatePrefillPostDraftModelInput(mrope_input, model_output, sampler_output, holder);
    EXPECT_EQ(std::vector<int>({1, 2, 3, 4, 5, 6, 7, 7, 7, 11, 12, 13, 14, 14, 14}),
              toVec<int>(mrope_input.combo_position_ids));
    EXPECT_EQ(std::vector<int>({0, 3}), toVec<int>(mrope_input.mm_features_locs));
    // Pure text keeps the existing token update, but does not enter the new
    // multimodal metadata shift path even when explicit positions are present.
    GptModelInputs text_input;
    text_input.combo_tokens       = torch::tensor({10, 11, 12, 20, 21}, torch::kInt32);
    text_input.input_lengths      = torch::tensor({3, 2}, torch::kInt32);
    text_input.combo_position_ids = torch::tensor({5, 6, 7, 10, 11}, torch::kInt32);
    processor.updatePrefillPostDraftModelInput(text_input, model_output, sampler_output, holder);
    EXPECT_EQ(std::vector<int>({11, 12, 30, 21, 40}), toVec<int>(text_input.combo_tokens));
    EXPECT_EQ(std::vector<int>({5, 6, 7, 10, 11}), toVec<int>(text_input.combo_position_ids));
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
    GenerateStreamPtr stream3 = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3}, 3);

    auto stream_groups = StreamGroups({stream1, stream2, stream3});

    cache_config.group_types = {CacheGroupType::FULL};
    auto processor           = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input_status = processor.gatherModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input = model_input_status.value();

    speculative::SpeculativeSamplerOutput spec_decode_output;
    // Cover minimum, intermediate, and full acceptance. The selected dense row
    // is batch_base + accept_len - 1 for every request.
    spec_decode_output.accept_len_cpu    = torch::tensor({1, 2, 3}, torch::kInt32);
    spec_decode_output.accept_tokens_cpu = torch::tensor({{2, 0, 0}, {2, 3, 0}, {2, 3, 1}}, torch::kInt32);
    spec_decode_output.accept_len        = spec_decode_output.accept_len_cpu.to(torch::kCUDA);
    spec_decode_output.accept_tokens     = spec_decode_output.accept_tokens_cpu.to(torch::kCUDA);

    torch::Tensor hidden_states_d_t;

    GptModelOutputs model_output;
    model_output.all_hidden_states = torch::tensor({0.1f,
                                                    0.2f,
                                                    0.3f,
                                                    0.4f,
                                                    0.5f,
                                                    0.6f,
                                                    1.1f,
                                                    1.2f,
                                                    1.3f,
                                                    1.4f,
                                                    1.5f,
                                                    1.6f,
                                                    2.1f,
                                                    2.2f,
                                                    2.3f,
                                                    2.4f,
                                                    2.5f,
                                                    2.6f},
                                                   torch::kFloat32)
                                         .reshape({9, 2});

    processor.updateDecodePostDraftModelInput(
        model_input, model_output, spec_decode_output, 3, hidden_states_d_t, holder);

    auto        combo_tokens        = model_input.combo_tokens.cpu();
    vector<int> expect_combo_tokens = {2, 0, 0, 2, 3, 0, 2, 3, 1};
    EXPECT_EQ(expect_combo_tokens, toVec<int>(combo_tokens));
    EXPECT_EQ((vector<int>{3, 3, 3}), toVec<int>(model_input.input_lengths.cpu()));

    EXPECT_TRUE(model_input.lm_output_indexes.is_cuda());
    auto        lm_output_indexes        = model_input.lm_output_indexes.cpu();
    vector<int> expect_lm_output_indexes = {0, 4, 8};
    EXPECT_EQ(expect_lm_output_indexes, toVec<int>(lm_output_indexes));

    auto          last_hidden_states        = model_input.last_hidden_states;
    vector<float> expect_last_hidden_states = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f};
    EXPECT_EQ(expect_last_hidden_states, toVec<float>(last_hidden_states));
    EXPECT_EQ(model_input.last_hidden_states_layout, MtpHiddenStatesLayout::GLOBAL);

    GptModelOutputs compact_output;
    processor.updateDecodePostDraftModelInput(
        model_input, compact_output, spec_decode_output, 3, hidden_states_d_t, holder, false);
    EXPECT_FALSE(model_input.last_hidden_states.defined());
    EXPECT_FALSE(hidden_states_d_t.defined());
    EXPECT_EQ(model_input.last_hidden_states_layout, MtpHiddenStatesLayout::NONE);
    EXPECT_EQ((vector<int>{3, 3, 3}), toVec<int>(model_input.input_lengths.cpu()));
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
