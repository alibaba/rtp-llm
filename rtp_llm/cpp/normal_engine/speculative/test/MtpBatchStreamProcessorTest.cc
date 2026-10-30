#include <cstdlib>
#include <chrono>
#include <cstring>
#include <memory>
#include <limits>
#include "torch/all.h"
#include "gtest/gtest.h"
#include "autil/EnvUtil.h"

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
#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/LinearBlocksUtil.h"
using namespace std;

namespace rtp_llm {

TEST(EngramHistoryTest, VerifyAndRejectedCandidatesDoNotPolluteNextHistory) {
    auto anchor   = torch::tensor({{10, 9, 8, 7}, {20, 19, 18, 17}}, torch::kInt32);
    auto verify   = torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kInt32);
    auto windows  = MtpBatchStreamProcessor::makeEngramVerifyWindows(anchor, verify);
    auto expected = torch::tensor(
        {{10, 9, 8, 7}, {11, 10, 9, 8}, {12, 11, 10, 9}, {20, 19, 18, 17}, {21, 20, 19, 18}, {22, 21, 20, 19}},
        torch::kInt32);
    EXPECT_TRUE(torch::equal(windows, expected));
    auto accepted = torch::tensor({{30, 31, 32}, {40, 41, 42}}, torch::kInt32);
    auto lengths  = torch::tensor({1, 3}, torch::kInt32);
    auto next     = MtpBatchStreamProcessor::advanceEngramTokenWindows(anchor, accepted, lengths);
    EXPECT_TRUE(torch::equal(next, torch::tensor({{30, 10, 9, 8}, {42, 41, 40, 20}}, torch::kInt32)));
}

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
    static CacheConfig makeProcessorCacheConfig() {
        return test::makeSimpleMhaCacheConfig(/*layer_num=*/1,
                                              /*block_num=*/2,
                                              /*tokens_per_block=*/1,
                                              rtp_llm::TYPE_FP16);
    }

    GenerateStreamPtr createContextStream(const ModelConfig&     model_config,
                                          const RuntimeConfig&   runtime_config,
                                          const ResourceContext& resource_context,
                                          const vector<int>&     input_ids,
                                          const int              block_id,
                                          const vector<int>&     begin_think_token_ids = {},
                                          const vector<int>&     end_think_token_ids   = {},
                                          const int              num_return_sequences  = 1) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids       = torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        query->generate_config = make_shared<GenerateConfig>();
        query->generate_config->begin_think_token_ids = begin_think_token_ids;
        query->generate_config->end_think_token_ids   = end_think_token_ids;
        if (!end_think_token_ids.empty()) {
            query->generate_config->in_think_mode       = true;
            query->generate_config->max_thinking_tokens = 1024;
        }
        query->generate_config->num_return_sequences = num_return_sequences;
        GenerateStreamPtr stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        if (!end_think_token_ids.empty()) {
            // Main's factory routes think constraints through grammar, so this
            // suite attaches the spec-verify think processor directly.
            auto think_processor = ThinkModeLogitsProcessor::fromGenerateInput(query, 1);
            if (think_processor != nullptr) {
                stream->logits_processor_list_.push_back(think_processor);
            }
        }
        BatchKVCacheResource addr;
        // New (refactored) BatchKVCacheResource: [batch_id][group_id] -> block_indices
        addr.resetBatchSize(1);
        addr.initGroups(makeProcessorCacheConfig().topologyPtr());
        addr.setBatchBlocks(0, "default", {block_id});
        stream->setKVCache(addr);

        auto        sp_output_buffer = std::make_shared<SpeculativeExecutorStreamOutput>();
        vector<int> propose_tokens   = vector<int>(2, -1);
        sp_output_buffer->tokens     = torch::tensor(propose_tokens, torch::kInt32).reshape({1, 2});
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

    void checkTargetProbabilityModes(bool verify, const vector<ReturnAllProbsMode>& requested_modes) {
        ModelConfig                 model_config;
        RuntimeConfig               runtime_config;
        SpeculativeExecutionConfig  sp_config;
        PDSepConfig                 pd_sep_config;
        ProfilingDebugLoggingConfig profiling_debug_logging_config;
        model_config.max_seq_len    = 32;
        model_config.vocab_size     = 4;
        model_config.num_layers     = 1;
        sp_config.gen_num_per_cycle = 3;
        ResourceContext         resource_context;
        list<GenerateStreamPtr> streams;
        bool                    capture_original = false;
        for (auto mode : requested_modes) {
            auto stream = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
            stream->generateConfig()->return_all_probs = mode;
            stream->generateConfig()->temperature      = 1.0f;
            stream->generateConfig()->top_k            = 1;
            stream->generateConfig()->top_p            = 1.0f;
            stream->generateConfig()->do_sample        = true;
            stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
            // MtpExecutor::prepareStreams keeps internal probabilities filtered
            // even when the request asks for ORIGINAL response probabilities.
            stream->setReturnAllProbs(ReturnAllProbsMode::DEFAULT);
            capture_original |= mode == ReturnAllProbsMode::ORIGINAL;
            streams.push_back(stream);
        }
        StreamGroups            stream_groups(streams);
        MtpBatchStreamProcessor processor(
            model_config, pd_sep_config, profiling_debug_logging_config, makeProcessorCacheConfig(), sp_config, false);
        const int64_t   rows = requested_modes.size() * (verify ? sp_config.gen_num_per_cycle + 1 : 1);
        GptModelOutputs model_output;
        // An independent numerical oracle: softmax(log(p)) == p. Rejection
        // sampling needs a point mass at token 3 after the top-k=1 filter.
        model_output.logits = torch::tensor({{0.1f, 0.2f, 0.3f, 0.4f}}).log().repeat({rows, 1}).to(torch::kCUDA);
        SamplerInputs inputs;
        if (verify) {
            auto proposals =
                torch::full({static_cast<int64_t>(streams.size()), static_cast<int64_t>(sp_config.gen_num_per_cycle)},
                            3,
                            torch::kInt32);
            inputs = processor.finalizeSpecSamplerInput(
                processor.prepareSpecSamplerInput(stream_groups, proposals), model_output, {});
        } else {
            auto result = processor.gatherSamplerInput(stream_groups, GptModelInputs{}, model_output);
            ASSERT_TRUE(result.ok()) << result.status();
            inputs = std::move(result).value();
        }
        if (!verify) {
            // This is the prefill sampling contract in MtpExecutor::prefillStep.
            inputs.capture_original_probs = capture_original;
        }
        ASSERT_FALSE(inputs.return_original_all_probs);
        ASSERT_EQ(inputs.capture_original_probs, capture_original);
        Sampler sampler(SamplerInitParams{});
        auto    output = sampler.forward(inputs);
        EXPECT_TRUE(torch::allclose(
            output.all_probs.cpu(), torch::tensor({{0.0f, 0.0f, 0.0f, 1.0f}}).repeat({rows, 1}), 1e-6, 1e-6));
        EXPECT_TRUE(output.token_ids.cpu().select(1, output.token_ids.size(1) - 1).eq(3).all().item<bool>());
        ASSERT_EQ(output.original_all_probs.defined(), capture_original);
        if (capture_original) {
            EXPECT_TRUE(torch::allclose(output.original_all_probs.cpu(),
                                        torch::tensor({{0.1f, 0.2f, 0.3f, 0.4f}}).repeat({rows, 1}),
                                        1e-6,
                                        1e-6));
        }
    }
};

TEST_F(MtpBatchStreamProcessorTest, testPrefillKeepsResponseAndRejectionProbabilitiesSeparate) {
    checkTargetProbabilityModes(false, {ReturnAllProbsMode::DEFAULT});
    checkTargetProbabilityModes(false, {ReturnAllProbsMode::ORIGINAL});
    checkTargetProbabilityModes(false,
                                {ReturnAllProbsMode::DEFAULT, ReturnAllProbsMode::ORIGINAL, ReturnAllProbsMode::NONE});
}

TEST_F(MtpBatchStreamProcessorTest, testVerifyKeepsResponseAndRejectionProbabilitiesSeparate) {
    checkTargetProbabilityModes(true, {ReturnAllProbsMode::DEFAULT});
    checkTargetProbabilityModes(true, {ReturnAllProbsMode::ORIGINAL});
    checkTargetProbabilityModes(true,
                                {ReturnAllProbsMode::DEFAULT, ReturnAllProbsMode::ORIGINAL, ReturnAllProbsMode::NONE});
}

TEST_F(MtpBatchStreamProcessorTest, testSpecSamplerHostPreparationKeepsFullHistory) {
    constexpr int64_t history_len = 4096;
    constexpr int64_t score_len   = 4;
    constexpr int64_t token_width = history_len + score_len;
    ModelConfig       model_config;
    model_config.max_seq_len = 8192;
    model_config.vocab_size  = 4;
    model_config.num_layers  = 1;
    RuntimeConfig              runtime_config;
    SpeculativeExecutionConfig sp_config;
    sp_config.gen_num_per_cycle = 3;
    ResourceContext resource_context;
    auto stream =
        createContextStream(model_config, runtime_config, resource_context, std::vector<int>(history_len, 1), 1);
    stream->generateConfig()->top_k = 1;
    stream->setScoreLen(score_len);
    StreamGroups            groups({stream});
    MtpBatchStreamProcessor processor(model_config, {}, {}, makeProcessorCacheConfig(), sp_config, false);
    auto                    inputs = processor.prepareSpecSamplerInput(groups);
    ASSERT_EQ((std::vector<int64_t>{score_len, token_width}), inputs.token_ids.sizes().vec());
    EXPECT_EQ(inputs.token_ids.nbytes(), score_len * token_width * sizeof(int32_t));
    EXPECT_TRUE(inputs.token_ids.narrow(1, 0, history_len).eq(1).all().item<bool>());
    EXPECT_FALSE(inputs.logits.defined());
    EXPECT_FALSE(inputs.all_probs.defined());
    EXPECT_EQ((std::vector<int32_t>{4096, 4097, 4098, 4099}), toVec<int32_t>(inputs.sequence_lengths));
    GptModelOutputs output;
    output.logits = torch::tensor({{0.0f, 1.0f, 2.0f, 3.0f}}).repeat({4, 1}).to(torch::kCUDA);
    processor.prepareSpecSamplerInputDevice(groups, inputs, torch::tensor({1, 2, 3}, torch::kInt32).reshape({1, 3}));
    inputs = processor.finalizeSpecSamplerInput(std::move(inputs), output, {});
    EXPECT_TRUE(inputs.token_ids.narrow(1, 0, history_len).eq(1).all().item<bool>());
    EXPECT_TRUE(torch::equal(inputs.token_ids.narrow(1, history_len, 3).cpu(),
                             torch::tensor({{1, 2, 3}}, torch::kInt32).repeat({score_len, 1})));
    Sampler sampler(SamplerInitParams{});
    auto    sampled = sampler.forward(inputs);
    ASSERT_EQ((std::vector<int64_t>{score_len, token_width}), sampled.token_ids.sizes().vec());
    EXPECT_EQ((std::vector<int32_t>{3, 3, 3, 3}), toVec<int32_t>(sampled.token_ids.select(1, token_width - 1)));
}

TEST_F(MtpBatchStreamProcessorTest, testSpecSamplerPenaltyUsesDraftPrefix) {
    ModelConfig model_config;
    model_config.max_seq_len = 32;
    model_config.vocab_size  = 4;
    model_config.num_layers  = 1;
    RuntimeConfig              runtime_config;
    SpeculativeExecutionConfig sp_config;
    sp_config.gen_num_per_cycle = 2;
    ResourceContext resource_context;
    auto            stream = createContextStream(model_config, runtime_config, resource_context, {0, 1}, 1);
    stream->generateConfig()->do_sample        = true;
    stream->generateConfig()->top_k            = 1;
    stream->generateConfig()->presence_penalty = 2.0f;
    stream->setScoreLen(3);
    StreamGroups            groups({stream});
    MtpBatchStreamProcessor processor(model_config, {}, {}, makeProcessorCacheConfig(), sp_config, false);
    GptModelOutputs         output;
    output.logits = torch::tensor({{3.0f, 1.0f, 2.5f, 2.25f}}).repeat({3, 1}).to(torch::kCUDA);
    auto inputs   = processor.prepareSpecSamplerInput(groups, torch::tensor({2, 3}, torch::kInt32).reshape({1, 2}));
    inputs        = processor.finalizeSpecSamplerInput(std::move(inputs), output, {});
    Sampler sampler(SamplerInitParams{});
    auto    sampled = sampler.forward(inputs);
    // Penalizing committed {0,1}, then prefixes {2} and {2,3}, changes argmax.
    EXPECT_EQ((std::vector<int32_t>{2, 3, 0}),
              toVec<int32_t>(sampled.token_ids.select(1, sampled.token_ids.size(1) - 1)));
}

#if USING_CUDA
TEST_F(MtpBatchStreamProcessorTest, testAsyncSpecSamplerPreparationPreservesHistoryAndMasks) {
    for (int64_t gamma : {1, 3}) {
        ModelConfig model_config;
        model_config.max_seq_len = 64;
        model_config.vocab_size  = 8;
        model_config.num_layers  = 1;
        SpeculativeExecutionConfig sp_config;
        sp_config.gen_num_per_cycle = gamma;
        ResourceContext resource_context;
        auto            first  = createContextStream(model_config, {}, resource_context, {0, 1}, 1);
        auto            second = createContextStream(model_config, {}, resource_context, {0, 1, 2}, 1);
        for (const auto& stream : {first, second}) {
            stream->setScoreLen(gamma + 1);
            stream->setReturnAllProbs(ReturnAllProbsMode::DEFAULT);
            stream->generateConfig()->do_sample        = true;
            stream->generateConfig()->top_k            = 1;
            stream->generateConfig()->presence_penalty = 2.5f;
        }
        StreamGroups            groups({first, second});
        MtpBatchStreamProcessor processor(model_config, {}, {}, makeProcessorCacheConfig(), sp_config, false);
        const auto              main_stream = cuda_graph::graphGetCurrentStream();
        auto                    verify_tokens =
            torch::tensor({{1, 2, 3, 4}, {2, 5, 6, 7}}, torch::kInt32).narrow(1, 0, gamma + 1).to(torch::kCUDA);
        auto draft_ready = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        draft_ready->record(main_stream);
        auto        inputs = std::make_shared<SamplerInputs>();
        AsyncRunner prepare_runner(cuda_graph::graphGetStreamFromPool(true));
        prepare_runner.launch([&processor, groups, inputs, verify_tokens, draft_ready]() {
            *inputs = processor.prepareSpecSamplerInput(groups, verify_tokens.slice(1, 1), draft_ready);
        });

        const int64_t   rows = 2 * (gamma + 1);
        GptModelOutputs output;
        // Two TP-padding columns must be trimmed; token 7 is then masked.
        output.logits     = torch::arange(10, torch::kFloat32).repeat({rows, 1}).to(torch::kCUDA);
        auto logits_ready = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        logits_ready->record(main_stream);
        SpecLogitsVerifyRunner::LaunchResult mask;
        mask.has_active_processor   = true;
        mask.packed_allow_mask_gpu  = torch::full({rows, 1}, 0x7f, torch::kInt32).to(torch::kCUDA);
        mask.logits_row_indices_gpu = torch::arange(rows, torch::kInt32).to(torch::kCUDA);
        mask.ready_event            = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        mask.ready_event->record(main_stream);
        prepare_runner.launch([&processor, inputs, output, mask, logits_ready]() {
            logits_ready->block(cuda_graph::graphGetCurrentStream());
            *inputs = processor.finalizeSpecSamplerInput(std::move(*inputs), output, mask);
        });
        prepare_runner.sync(main_stream);
        for (const auto& tensor : {inputs->token_ids, inputs->logits, inputs->all_probs}) {
            tensor.record_stream(main_stream);
        }
        Sampler       sampler(SamplerInitParams{});
        auto          sampled     = sampler.forward(*inputs);
        auto          tokens      = sampled.token_ids.cpu();
        const int64_t token_width = tokens.size(1);
        for (int64_t row = 0; row <= gamma; ++row) {
            EXPECT_TRUE(torch::equal(tokens[row].narrow(0, 0, 2), torch::tensor({0, 1}, torch::kInt32)));
            EXPECT_TRUE(torch::equal(tokens[row].narrow(0, 2, gamma),
                                     torch::tensor({2, 3, 4}, torch::kInt32).narrow(0, 0, gamma)));
            EXPECT_EQ(tokens[row][token_width - 1].item<int32_t>(), 6);
            const auto other = row + gamma + 1;
            EXPECT_TRUE(torch::equal(tokens[other].narrow(0, 0, 3), torch::tensor({0, 1, 2}, torch::kInt32)));
            EXPECT_TRUE(torch::equal(tokens[other].narrow(0, 3, gamma),
                                     torch::tensor({5, 6, 7}, torch::kInt32).narrow(0, 0, gamma)));
            EXPECT_EQ(tokens[other][token_width - 1].item<int32_t>(), row < 2 ? 6 : 4);
        }
        EXPECT_EQ(inputs->logits.size(1), 8);
        EXPECT_TRUE(torch::equal(output.logits.cpu(), torch::arange(10, torch::kFloat32).repeat({rows, 1})));
        EXPECT_TRUE(sampled.all_probs.select(1, 7).eq(0).all().item<bool>());
    }
}
#endif

class TestableMtpBatchStreamProcessor: public MtpBatchStreamProcessor {
public:
    using MtpBatchStreamProcessor::MtpBatchStreamProcessor;
    using MtpBatchStreamProcessor::overlayMtpCacheSnapshots;
};

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

TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputBuildsPositionSpecificHistories) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 3;
    sp_config.type              = SP_TYPE_DSPARK;

    ResourceContext resource_context;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {5}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {6, 7}, 2);
    stream1->setScoreLen(sp_config.gen_num_per_cycle + 1);
    stream2->setScoreLen(sp_config.gen_num_per_cycle + 1);
    stream1->generateConfig()->do_sample   = true;
    stream1->generateConfig()->top_k       = 0;
    stream1->generateConfig()->top_p       = 0.8f;
    stream1->generateConfig()->temperature = 0.0f;
    stream2->generateConfig()->do_sample   = true;
    stream2->generateConfig()->top_k       = 0;
    stream2->generateConfig()->top_p       = 0.7f;
    stream2->generateConfig()->temperature = 1.0f;

    auto stream_groups = StreamGroups({stream1, stream2});
    auto processor     = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelInputs  model_inputs;
    GptModelOutputs model_output;
    model_output.logits =
        torch::full({static_cast<int64_t>(stream_groups.size() * (sp_config.gen_num_per_cycle + 1)), 7},
                    1000.0f,
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    model_output.logits.narrow(1, 0, model_config.vocab_size).zero_();

    auto draft_token_ids = torch::tensor({1, 2, 3, 3, 2, 1}, torch::kInt32).reshape({2, 3});
    auto sampler_inputs  = processor.finalizeSpecSamplerInput(
        processor.prepareSpecSamplerInput(stream_groups, draft_token_ids), model_output, {});
    EXPECT_EQ((std::vector<int64_t>{8, 4}), sampler_inputs.logits.sizes().vec());
    EXPECT_EQ((std::vector<int64_t>{8, 4}), sampler_inputs.all_probs.sizes().vec());
    EXPECT_TRUE(sampler_inputs.logits.eq(0).all().item<bool>());

    auto  token_ids = sampler_inputs.token_ids.cpu();
    auto  stride    = token_ids.size(1);
    auto* data      = token_ids.data_ptr<int32_t>();

    for (int64_t row = 0; row < 4; ++row) {
        EXPECT_EQ(5, data[row * stride]);
        EXPECT_EQ(1, data[row * stride + 1]);
        EXPECT_EQ(2, data[row * stride + 2]);
        EXPECT_EQ(3, data[row * stride + 3]);
    }
    for (int64_t row = 4; row < 8; ++row) {
        EXPECT_EQ(6, data[row * stride]);
        EXPECT_EQ(7, data[row * stride + 1]);
        EXPECT_EQ(3, data[row * stride + 2]);
        EXPECT_EQ(2, data[row * stride + 3]);
        EXPECT_EQ(1, data[row * stride + 4]);
    }
    EXPECT_EQ((std::vector<int32_t>{1, 2, 3, 4, 2, 3, 4, 5}), toVec<int32_t>(sampler_inputs.sequence_lengths));
    EXPECT_EQ((std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0}), toVec<int32_t>(sampler_inputs.top_k));
    EXPECT_EQ((std::vector<float>{0.8f, 0.8f, 0.8f, 0.8f, 0.7f, 0.7f, 0.7f, 0.7f}), toVec<float>(sampler_inputs.top_p));
    EXPECT_EQ((std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f}),
              toVec<float>(sampler_inputs.temperature));
}

TEST_F(MtpBatchStreamProcessorTest, testSpecSamplerInputMasksThinkBoundaryTokens) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 16;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 2;

    ResourceContext resource_context;
    auto stream = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 1, {7}, {8, 9});
    stream->setScoreLen(sp_config.gen_num_per_cycle + 1);

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    StreamGroups stream_groups({stream});

    GptModelInputs  model_input;
    GptModelOutputs model_output;
    model_output.logits = torch::zeros({3, 16}, torch::kFloat32);

    // Since the spec-logits overlap (#1262), stateful processors (think,
    // grammar) are excluded from the inline score-batch states and applied
    // through the SpecLogitsVerifyRunner vocab mask instead. Build that mask
    // exactly the way MtpExecutor::buildSpecLogitsVerifyInline wires it.
    SpecLogitsVerifyRunner             runner;
    SpecLogitsVerifyRunner::LaunchTask task;
    task.total_streams = 1;
    task.propose_step  = static_cast<int>(sp_config.gen_num_per_cycle);
    task.vocab_size    = model_config.vocab_size;
    task.draft_tokens  = torch::tensor(std::vector<int32_t>{1, 2}, torch::kInt32).reshape({1, 2});
    for (const auto& processor_ptr : stream->getAllLogitsProcessorPtr()) {
        ASSERT_NE(processor_ptr, nullptr);
        ASSERT_EQ(processor_ptr->mtpCapability().mode, MtpProcessorMode::SPEC_VERIFY);
        task.active.push_back({processor_ptr, /*stream_idx=*/0});
    }
    ASSERT_EQ(1u, task.active.size());
    auto spec_result = runner.run(task);
    ASSERT_TRUE(spec_result.has_active_processor);

    auto sampler_inputs = processor.finalizeSpecSamplerInput(
        processor.prepareSpecSamplerInput(stream_groups, task.draft_tokens), model_output, spec_result);

    // Constrained/think processors ride the SpecLogitsVerifyRunner packed
    // allow-mask, which the gather applies to the verify logits inline; the
    // MTP verify sampler still carries no logits-processor states (main #1006
    // design). In think mode with remaining budget, only the begin-think
    // token is blocked while the end-think sequence stays open.
    EXPECT_EQ(sampler_inputs.logits_processor_states_ptr, nullptr);
    const float neg_inf = -std::numeric_limits<float>::max();
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(neg_inf, sampler_inputs.logits[i][7].item<float>());
        EXPECT_EQ(0, sampler_inputs.logits[i][8].item<float>());
        EXPECT_EQ(0, sampler_inputs.logits[i][9].item<float>());
    }
}

TEST_F(MtpBatchStreamProcessorTest, testPrefillDispatch) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

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

    auto status = processor.dispatchPrefill(stream_groups, std::move(target_output), std::move(draft_output));
    EXPECT_TRUE(status.ok());

    checkOutput(stream1, {2, 1}, {1, 2}, {0.2, 0.1, 0.3, 0.5}, {0.3, 0.4});
    checkOutput(stream2, {1, 2, 3}, {3, 0}, {0.3, 0.1, 0.4, 0.2}, {1.7, 1.8});
}

TEST_F(MtpBatchStreamProcessorTest, testDispatchDecodeStream) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

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

    stream1->generateConfig()->is_streaming         = true;
    stream2->generateConfig()->is_streaming         = true;
    stream1->generateConfig()->return_logits        = true;
    stream1->generateConfig()->return_hidden_states = true;
    stream1->generateConfig()->return_all_probs     = ReturnAllProbsMode::DEFAULT;
    stream1->setReturnAllProbs(ReturnAllProbsMode::DEFAULT);
    stream2->setReturnAllProbs(ReturnAllProbsMode::NONE);
    auto stream_groups = StreamGroups({stream1, stream2});

    speculative::SpeculativeSamplerOutput spec_decode_output;
    spec_decode_output.accept_len_cpu    = torch::tensor({5, 1}, torch::kInt32);
    spec_decode_output.accept_tokens_cpu = torch::tensor({{2, 3, 1, 3, 2}, {2, 0, 0, 0, 0}}, torch::kInt32);
    spec_decode_output.accept_len        = spec_decode_output.accept_len_cpu.to(torch::kCUDA);
    spec_decode_output.accept_tokens     = spec_decode_output.accept_tokens_cpu.to(torch::kCUDA);

    spec_decode_output.target_logits =
        torch::tensor({8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}).reshape({2, 4}).to(torch::kCUDA);
    spec_decode_output.target_hidden_states =
        torch::tensor({90.0f, 91.0f, 80.0f, 81.0f}).reshape({2, 2}).to(torch::kCUDA);
    // The sampler publishes one target distribution per verify row. The first
    // four rows select the accepted draft tokens; the final bonus row is flat.
    spec_decode_output.target_probs_cpu =
        torch::tensor({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 0.0f, 1.0f, 0.25f, 0.25f, 0.25f, 0.25f,
                       0.0f, 0.0f, 1.0f, 0.0f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
                       0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f},
                      torch::kFloat32)
            .reshape({2, 5, 4});
    spec_decode_output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());

    MergedOutput draft_prefill_output;
    draft_prefill_output.model_output.all_hidden_states =
        torch::tensor({0.2f, 0.02f, 0.3f, 0.03f, 0.4f, 0.04f, 0.5f, 0.05f, 0.6f, 0.06f,
                       1.3f, 0.13f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, torch::kFloat32)
            .reshape({10, 2});
    draft_prefill_output.sampler_output.token_ids = torch::tensor({0L, 3L}, torch::kInt64).reshape({2, 1});
    draft_prefill_output.sampler_output.all_probs =
        torch::tensor({0.2f, 0.1f, 0.3f, 0.5f, 0.3f, 0.1f, 0.4f, 0.2f}, torch::kFloat32).reshape({2, 4});

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    auto status = processor.dispatchDecode(stream_groups, spec_decode_output, std::move(draft_prefill_output));
    EXPECT_TRUE(status.ok());

    checkOutput(stream1, {1, 2, 3, 1, 3, 2}, {2, 0}, {0.2, 0.1, 0.3, 0.5}, {0.6, 0.06});
    checkOutput(stream2, {2, 1, 2}, {2, 3}, {0.3, 0.1, 0.4, 0.2}, {1.3, 0.13});
    auto output1 = stream1->nextOutput(100);
    ASSERT_TRUE(output1.ok());
    ASSERT_EQ(output1.value().generate_outputs.size(), 1u);
    const auto& first = output1.value().generate_outputs[0];
    ASSERT_TRUE(first.logits.has_value());
    ASSERT_TRUE(first.hidden_states.has_value());
    EXPECT_EQ(toVec<float>(*first.logits), (std::vector<float>{8, 7, 6, 5}));
    EXPECT_EQ(toVec<float>(*first.hidden_states), (std::vector<float>{90, 91}));
    const auto& first_probs = first.aux_info.all_probs;
    ASSERT_TRUE(first_probs.has_value());
    ASSERT_EQ(first_probs->sizes().vec(), (std::vector<int64_t>{1, 5, 4}));
    EXPECT_EQ(toVec<float>(*first_probs),
              (std::vector<float>{0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, .25f, .25f, .25f, .25f}));
    auto output2 = stream2->nextOutput(100);
    ASSERT_TRUE(output2.ok());
    ASSERT_EQ(output2.value().generate_outputs.size(), 1u);
    EXPECT_FALSE(output2.value().generate_outputs[0].logits.has_value());
    EXPECT_FALSE(output2.value().generate_outputs[0].hidden_states.has_value());
    EXPECT_FALSE(output2.value().generate_outputs[0].aux_info.all_probs.has_value());
    // Device-state real_seq_len publication moved to the executor layer
    // (MtpExecutor::publishSyncMtpDeviceState); dispatchDecode itself only
    // performs host bookkeeping now, so no device-state asserts here.
}

TEST_F(MtpBatchStreamProcessorTest, EmptyDSpARKDecodeRoundPreservesEmptyEngramHistory) {
    ModelConfig model_config;
    model_config.max_seq_len                         = 128;
    model_config.vocab_size                          = 8;
    model_config.num_layers                          = 1;
    model_config.attn_config.v41_kv_source_layer_ids = {0};
    SpeculativeExecutionConfig sp_config;
    sp_config.type              = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle = 3;
    MtpBatchStreamProcessor processor(model_config, {}, {}, makeProcessorCacheConfig(), sp_config, false);
    TensorHolder            holder;
    StreamGroups            empty(std::list<GenerateStreamPtr>{});
    auto                    input = processor.gatherDecodeModelInput(empty, holder);
    ASSERT_TRUE(input.ok());
    EXPECT_EQ(input->combo_tokens.numel(), 0);
    ASSERT_TRUE(input->engram_token_windows.defined());
    EXPECT_EQ(input->engram_token_windows.sizes().vec(), (std::vector<int64_t>{0, 4}));
}

TEST_F(MtpBatchStreamProcessorTest, testGatherDecodeModelInput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

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

    auto processor = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input = processor.gatherDecodeModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input.ok());
    const std::vector<std::string> expected_tags = {cache_config.topology().groups().front().tag};
    EXPECT_EQ(model_input->kv_cache_group_tags, expected_tags);
    auto async_input = model_input.value();
    model_input->kv_cache_group_tags.clear();
    EXPECT_EQ(async_input.kv_cache_group_tags, expected_tags);
    EXPECT_TRUE(torch::equal(async_input.kv_cache_block_id, model_input->kv_cache_block_id));
    EXPECT_TRUE(torch::equal(async_input.kv_cache_kernel_block_id, model_input->kv_cache_kernel_block_id));

    auto          last_hidden_states        = model_input.value().last_hidden_states;
    auto          last_hidden_states_h      = last_hidden_states.cpu().clone();
    vector<float> expect_last_hidden_states = {0.1, 0.2, 1.1, 1.2};
    EXPECT_EQ(expect_last_hidden_states, toVec<float>(last_hidden_states_h));
}

TEST_F(MtpBatchStreamProcessorTest, testGatherDecodeModelInputKeepsEmptyTpFollowerPositionPlaceholder) {
    const StreamGroups empty_stream_groups({});
    for (const bool device_input : {false, true}) {
        autil::EnvGuard device_input_env("RTP_LLM_DEVICE_INPUT", device_input ? "1" : "0");
        for (int position_mode = 0; position_mode < 3; ++position_mode) {
            ModelConfig model_config;
            model_config.num_layers                            = 1;
            model_config.has_positional_encoding               = position_mode == 1;
            model_config.mm_model_config.mm_position_ids_style = position_mode == 2 ? MROPE : DEFAULT;
            model_config.attn_config.rope_config.index_factor  = position_mode == 2 ? 3 : 1;
            for (const auto spec_type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
                for (const int propose_step : {1, 3}) {
                    SCOPED_TRACE(testing::Message()
                                 << "device_input=" << device_input << " position_mode=" << position_mode
                                 << " spec_type=" << spec_type << " propose_step=" << propose_step);
                    SpeculativeExecutionConfig sp_config;
                    sp_config.type              = spec_type;
                    sp_config.gen_num_per_cycle = propose_step;
                    MtpBatchStreamProcessor processor(model_config,
                                                      PDSepConfig{},
                                                      ProfilingDebugLoggingConfig{},
                                                      makeProcessorCacheConfig(),
                                                      sp_config,
                                                      false);
                    TensorHolder            holder;
                    // TP followers gather no local streams before receiving
                    // the driver's input. An empty allocated tensor is still
                    // a valid broadcast placeholder, unlike an undefined one.
                    auto result = processor.gatherDecodeModelInput(empty_stream_groups, holder);
                    ASSERT_TRUE(result.ok()) << result.status();
                    ASSERT_TRUE(result->combo_tokens.defined());
                    EXPECT_EQ(result->combo_tokens.numel(), 0);
                    ASSERT_TRUE(result->input_lengths.defined());
                    EXPECT_EQ(result->input_lengths.numel(), 0);
                    const auto& positions = result->combo_position_ids;
                    ASSERT_EQ(positions.defined(), position_mode != 0);
                    if (positions.defined()) {
                        EXPECT_EQ(positions.dim(), 1);
                        EXPECT_EQ(positions.numel(), 0);
                        EXPECT_EQ(positions.scalar_type(), torch::kInt32);
                        EXPECT_TRUE(positions.device().is_cpu());
                    }
                }
            }
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testGatherDecodeModelInputUsesHostMtpAndDeviceDSparkPositionBases) {
    for (const bool device_input : {false, true}) {
        autil::EnvGuard device_input_env("RTP_LLM_DEVICE_INPUT", device_input ? "1" : "0");
        for (const bool mrope : {false, true}) {
            for (const auto spec_type : {SP_TYPE_MTP, SP_TYPE_DSPARK}) {
                for (const int propose_step : {1, 3}) {
                    SCOPED_TRACE(testing::Message() << "device_input=" << device_input << " mrope=" << mrope
                                                    << " spec_type=" << spec_type << " propose_step=" << propose_step);
                    ModelConfig model_config;
                    model_config.max_seq_len                           = 2048;
                    model_config.vocab_size                            = 256;
                    model_config.num_layers                            = 1;
                    model_config.has_positional_encoding               = !mrope;
                    model_config.mm_model_config.mm_position_ids_style = mrope ? MROPE : DEFAULT;
                    model_config.attn_config.rope_config.index_factor  = mrope ? 3 : 1;
                    RuntimeConfig              runtime_config;
                    ResourceContext            resource_context;
                    SpeculativeExecutionConfig sp_config;
                    sp_config.type              = spec_type;
                    sp_config.gen_num_per_cycle = propose_step;
                    auto steady = createContextStream(model_config, runtime_config, resource_context, {10, 11}, 1);
                    auto fresh  = createContextStream(model_config, runtime_config, resource_context, {20, 21, 22}, 2);
                    steady->setIsContextStream(false);
                    fresh->setIsContextStream(false);
                    steady->setContextPositionIds(
                        torch::tensor(mrope ? vector<int>{1, 1, 1, 2, 2, 2} : vector<int>{0, 1}, torch::kInt32));
                    fresh->setContextPositionIds(torch::tensor(
                        mrope ? vector<int>{100, 101, 102, 110, 111, 112, 120, 121, 122} : vector<int>{0, 1, 2},
                        torch::kInt32));
                    steady->getSPOutputBuffer()->hidden_states = torch::tensor({{1.0f, 2.0f}});
                    fresh->getSPOutputBuffer()->hidden_states  = torch::tensor({{3.0f, 4.0f}});
                    // Only DSpARK consumes the published device position.
                    // Ordinary MTP uses each request's host position state.
                    GenerateStream::MtpAsyncDeviceState state;
                    state.next_position_ids_gpu =
                        torch::tensor(mrope ? vector<int>{30, 40, 50} : vector<int>{30}, torch::kInt32)
                            .to(torch::kCUDA);
                    steady->setMtpAsyncDeviceState(std::move(state));
                    StreamGroups            stream_groups({steady, fresh});
                    MtpBatchStreamProcessor processor(model_config,
                                                      PDSepConfig{},
                                                      ProfilingDebugLoggingConfig{},
                                                      makeProcessorCacheConfig(),
                                                      sp_config,
                                                      false);
                    TensorHolder            holder;
                    auto                    result = processor.gatherDecodeModelInput(stream_groups, holder);
                    ASSERT_TRUE(result.ok()) << result.status();
                    ASSERT_TRUE(result->combo_position_ids.defined());
                    EXPECT_EQ(result->combo_position_ids.is_cuda(), spec_type == SP_TYPE_DSPARK);
                    EXPECT_EQ(result->combo_position_ids.dim(), 1);
                    EXPECT_EQ(result->combo_position_ids.scalar_type(), torch::kInt32);
                    // Gather publishes one position row per request. Draft
                    // and target-verify expansion happens after TP broadcast.
                    const auto expected = spec_type == SP_TYPE_DSPARK ?
                                              (mrope ? vector<int>{30, 40, 50, 122, 122, 122} : vector<int>{30, 2}) :
                                              (mrope ? vector<int>{2, 2, 2, 122, 122, 122} : vector<int>{1, 2});
                    EXPECT_EQ(expected, toVec<int>(result->combo_position_ids));
                }
            }
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testPrepareOneStepSpecDecodeModelInput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

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

    stream1->getSPOutputBuffer()->tokens = torch::tensor(propose_tokens_1, torch::kInt32).reshape({1, 2});
    stream2->getSPOutputBuffer()->tokens = torch::tensor(propose_tokens_2, torch::kInt32).reshape({1, 2});

    auto stream_groups = StreamGroups({stream1, stream2});

    auto processor = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input_status = processor.gatherDecodeModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input            = model_input_status.value();
    model_input.sequence_lengths = torch::tensor({1, 2}, torch::kInt32);
    const auto payload_tags      = model_input.kv_cache_group_tags;
    const auto physical_table    = model_input.kv_cache_block_id;
    const auto kernel_table      = model_input.kv_cache_kernel_block_id;

    processor.prepareOneStepSpecDecodeModelInput(stream_groups, model_input, holder);
    EXPECT_EQ(model_input.kv_cache_group_tags, payload_tags);
    EXPECT_TRUE(torch::equal(model_input.kv_cache_block_id, physical_table));
    EXPECT_TRUE(torch::equal(model_input.kv_cache_kernel_block_id, kernel_table));

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
    setenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "1", 1);
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

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

    auto processor = MtpBatchStreamProcessor(
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
    unsetenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE");
}

TEST_F(MtpBatchStreamProcessorTest, testprepareDecodeDraftModelInput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

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

    stream1->getSPOutputBuffer()->tokens        = torch::tensor(propose_tokens_1, torch::kInt32).reshape({1, 2});
    stream2->getSPOutputBuffer()->tokens        = torch::tensor(propose_tokens_2, torch::kInt32).reshape({1, 2});
    stream1->getSPOutputBuffer()->hidden_states = torch::tensor({{0.1f, 0.2f}});
    stream2->getSPOutputBuffer()->hidden_states = torch::tensor({{1.1f, 1.2f}});

    auto stream_groups = StreamGroups({stream1, stream2});

    auto processor = MtpBatchStreamProcessor(
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

    auto expect_positions =
        [](const GptModelInputs& input, const vector<int>& expected_prefix, const vector<int>& expected_sequence) {
            EXPECT_TRUE(input.prefix_lengths.is_cuda());
            EXPECT_TRUE(input.sequence_lengths.is_cuda());
            EXPECT_EQ(torch::kInt32, input.prefix_lengths.scalar_type());
            EXPECT_EQ(torch::kInt32, input.sequence_lengths.scalar_type());
            EXPECT_EQ(expected_prefix, toVec<int>(input.prefix_lengths));
            EXPECT_EQ(expected_sequence, toVec<int>(input.sequence_lengths));
            EXPECT_EQ(expected_sequence, toVec<int>(input.prefix_lengths));
        };
    expect_positions(model_input, {1, 2}, {1, 2});

    // Legacy GPU propose-token path receives the normal decode position.
    stream1->getSPOutputBuffer()->propose_tokens_gpu = torch::tensor({{3}}, torch::kInt32).to(torch::kCUDA);
    stream2->getSPOutputBuffer()->propose_tokens_gpu = torch::tensor({{1}}, torch::kInt32).to(torch::kCUDA);
    model_input.sequence_lengths                     = torch::tensor({4, 5}, torch::kInt32);
    processor.prepareDecodeDraftModelInput(stream_groups, model_input, holder);

    expect_positions(model_input, {4, 5}, {4, 5});

    // Device state includes the carried target token. Draft KV and target
    // verification both start one slot before that committed length.
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

    expect_positions(model_input, {6, 3}, {6, 3});
}

TEST_F(MtpBatchStreamProcessorTest, DraftCacheWritesRemainContiguousAcrossPreparationPaths) {
    // Synchronous updates retain the verify rectangle; accepted length alone
    // advances the next logical cache position.
    autil::EnvGuard             stream_async("RTP_LLM_STREAM_ASYNC", "0");
    autil::EnvGuard             device_state("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "0");
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig logging_config;
    SpeculativeExecutionConfig  sp_config;
    model_config.max_seq_len                           = 4096;
    model_config.vocab_size                            = 16;
    model_config.num_layers                            = 1;
    model_config.mm_model_config.mm_position_ids_style = MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    sp_config.gen_num_per_cycle                        = 4;
    auto                    cache_config               = test::makeSimpleMhaCacheConfig(1, 4, 2048, rtp_llm::TYPE_FP16);
    MtpBatchStreamProcessor processor(model_config, pd_sep_config, logging_config, cache_config, sp_config, false);

    for (int prompt_len : {465, 2048, 2051}) {
        SCOPED_TRACE(prompt_len);
        ResourceContext resource_context;
        auto            stream =
            createContextStream(model_config, runtime_config, resource_context, vector<int>(prompt_len, 1), 1);
        stream->getSPOutputBuffer()->tokens = torch::tensor({{2, 3}}, torch::kInt32);
        StreamGroups   groups({stream});
        TensorHolder   holder;
        GptModelInputs prefill;
        prefill.combo_tokens   = torch::ones({prompt_len}, torch::kInt32);
        prefill.input_lengths  = torch::tensor({prompt_len}, torch::kInt32);
        prefill.prefix_lengths = torch::tensor({0}, torch::kInt32);
        GptModelOutputs prefill_output;
        prefill_output.all_hidden_states = torch::zeros({prompt_len, 2});
        SamplerOutput sampled;
        sampled.token_ids = torch::tensor({{2}}, torch::kInt32);
        processor.updatePrefillPostDraftModelInput(groups, prefill, prefill_output, sampled, holder);
        ASSERT_EQ((vector<int>{prompt_len}), toVec<int>(prefill.input_lengths));
        ASSERT_EQ((vector<int>{0}), toVec<int>(prefill.prefix_lengths));
        EXPECT_EQ(2, toVec<int>(prefill.combo_tokens).back());
        const int first_unwritten_slot = toVec<int>(prefill.prefix_lengths)[0] + toVec<int>(prefill.input_lengths)[0];

        // Supply the same logical position through CPU, legacy GPU, incomplete
        // GPU state, and complete committed-length state. Physical block IDs
        // and explicit positional coordinates must not be rewritten.
        for (int path = 0; path < 4; ++path) {
            SCOPED_TRACE(path);
            stream->setMtpAsyncDeviceState({});
            stream->getSPOutputBuffer()->propose_tokens_gpu = torch::Tensor();
            GptModelInputs input;
            input.input_lengths            = torch::tensor({prompt_len}, torch::kInt32);
            input.sequence_lengths         = torch::tensor({first_unwritten_slot}, torch::kInt32);
            input.combo_position_ids       = torch::tensor({prompt_len, prompt_len + 1, prompt_len + 2}, torch::kInt32);
            input.kv_cache_block_id        = torch::tensor({{{17, 23}}}, torch::kInt32);
            input.kv_cache_kernel_block_id = input.kv_cache_block_id.clone();
            const auto positions           = input.combo_position_ids.clone();
            const auto block_ids           = input.kv_cache_block_id.clone();
            if (path == 1) {
                stream->getSPOutputBuffer()->propose_tokens_gpu = torch::tensor({{3}}, torch::kInt32).to(torch::kCUDA);
            } else if (path >= 2) {
                GenerateStream::MtpAsyncDeviceState state;
                state.propose_tokens_gpu = torch::tensor({{3}}, torch::kInt32).to(torch::kCUDA);
                if (path == 3) {
                    state.next_seq_len_gpu = torch::tensor({first_unwritten_slot + 1}, torch::kInt32).to(torch::kCUDA);
                    input.sequence_lengths = torch::tensor({99}, torch::kInt32);
                }
                stream->setMtpAsyncDeviceState(std::move(state));
            }
            processor.prepareDecodeDraftModelInput(groups, input, holder);
            EXPECT_EQ((vector<int>{first_unwritten_slot}), toVec<int>(input.sequence_lengths));
            EXPECT_EQ((vector<int>{first_unwritten_slot}), toVec<int>(input.prefix_lengths));
            EXPECT_EQ((vector<int>{3}), toVec<int>(input.combo_tokens));
            EXPECT_TRUE(torch::equal(positions, input.combo_position_ids));
            EXPECT_TRUE(torch::equal(block_ids, input.kv_cache_block_id));
            EXPECT_TRUE(torch::equal(block_ids, input.kv_cache_kernel_block_id));
            GptModelOutputs draft_output;
            draft_output.all_hidden_states = torch::zeros({1, 2});
            processor.updateDecodeDraftModelInput(input, draft_output, torch::tensor({{4}}, torch::kInt32), holder);
            EXPECT_EQ((vector<int>{first_unwritten_slot + 1}), toVec<int>(input.sequence_lengths));
            EXPECT_EQ((vector<int>{first_unwritten_slot}), toVec<int>(input.prefix_lengths));
            EXPECT_EQ(toVec<int>(positions + 1), toVec<int>(input.combo_position_ids));
        }

        stream->setMtpAsyncDeviceState({});
        stream->getSPOutputBuffer()->propose_tokens_gpu = torch::Tensor();
        for (int accepted : {1, 4, 5}) {
            SCOPED_TRACE(accepted);
            GptModelInputs post;
            post.prefix_lengths = torch::tensor({first_unwritten_slot}, torch::kInt32);
            post.input_lengths  = torch::tensor({5}, torch::kInt32);
            speculative::SpeculativeSamplerOutput rejection;
            rejection.accept_len_cpu    = torch::tensor({accepted}, torch::kInt32);
            rejection.accept_tokens_cpu = torch::tensor({{2, 3, 4, 5, 6}}, torch::kInt32);
            rejection.accept_len        = rejection.accept_len_cpu.to(torch::kCUDA);
            rejection.accept_tokens     = rejection.accept_tokens_cpu.to(torch::kCUDA);
            GptModelOutputs verified;
            verified.all_hidden_states = torch::zeros({5, 2});
            torch::Tensor selected_hidden;
            processor.updateDecodePostDraftModelInput(post, verified, rejection, 1, selected_hidden, holder);
            ASSERT_EQ((vector<int>{5}), toVec<int>(post.input_lengths));
            ASSERT_EQ((vector<int>{first_unwritten_slot}), toVec<int>(post.prefix_lengths));
            EXPECT_EQ((vector<int>{accepted - 1}), toVec<int>(post.lm_output_indexes));
            const int      next_slot = toVec<int>(post.prefix_lengths)[0] + accepted;
            GptModelInputs next;
            next.input_lengths    = torch::tensor({prompt_len}, torch::kInt32);
            next.sequence_lengths = torch::tensor({next_slot}, torch::kInt32);
            processor.prepareDecodeDraftModelInput(groups, next, holder);
            EXPECT_EQ((vector<int>{next_slot}), toVec<int>(next.sequence_lengths));
            EXPECT_EQ((vector<int>{next_slot}), toVec<int>(next.prefix_lengths));
            GenerateStream::MtpAsyncDeviceState state;
            state.propose_tokens_gpu = torch::tensor({{3}}, torch::kInt32).to(torch::kCUDA);
            state.next_seq_len_gpu   = torch::tensor({next_slot + 1}, torch::kInt32).to(torch::kCUDA);
            stream->setMtpAsyncDeviceState(std::move(state));
            next.sequence_lengths = torch::tensor({99}, torch::kInt32);
            processor.prepareDecodeDraftModelInput(groups, next, holder);
            EXPECT_EQ((vector<int>{next_slot}), toVec<int>(next.sequence_lengths));
            EXPECT_EQ((vector<int>{next_slot}), toVec<int>(next.prefix_lengths));
            stream->setMtpAsyncDeviceState({});
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkRuntimeGammaThreePrefillInputShapes) {
    constexpr int32_t gamma   = 3;
    constexpr int32_t mask_id = 12345;

    ModelConfig                 model_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig  sp_config;
    sp_config.type                    = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle       = gamma;
    sp_config.sp_dspark_mask_token_id = mask_id;

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelInputs model_input;
    model_input.input_lengths  = torch::tensor({3, 2}, torch::kInt32);
    model_input.prefix_lengths = torch::tensor({7, 4}, torch::kInt32);

    auto target_features =
        torch::arange(0, 60, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({5, 12});

    TensorHolder host_holder;
    model_input.last_hidden_states = target_features;
    // A well-formed commit input passes; the validator does not mutate it.
    EXPECT_NO_THROW(processor.validatePrefillDSparkCommitInput(model_input));

    // Missing aux features must refuse the commit.
    GptModelInputs no_features     = model_input;
    no_features.last_hidden_states = torch::Tensor();
    EXPECT_THROW(processor.validatePrefillDSparkCommitInput(no_features), std::exception);

    // A non-positive draft width must refuse the commit.
    SpeculativeExecutionConfig zero_width_config = sp_config;
    zero_width_config.gen_num_per_cycle          = 0;
    MtpBatchStreamProcessor zero_width(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, zero_width_config, false);
    EXPECT_THROW(zero_width.validatePrefillDSparkCommitInput(model_input), std::exception);

    // Anchors / committed ends come from stream state at the decode round
    // head; feed equivalent values here.
    torch::Tensor anchors        = torch::tensor({101, 202}, torch::kInt32);
    torch::Tensor committed_ends = torch::tensor({10, 6}, torch::kInt32);
    processor.buildDSparkProposeInput(model_input, anchors, committed_ends, host_holder);

    EXPECT_EQ((std::vector<int32_t>{101, mask_id, mask_id, 202, mask_id, mask_id}),
              toVec<int32_t>(model_input.combo_tokens));
    EXPECT_FALSE(model_input.last_hidden_states.defined());
    EXPECT_EQ((std::vector<int32_t>{10, 6}), toVec<int32_t>(model_input.prefix_lengths));
    EXPECT_EQ((std::vector<int32_t>{gamma, gamma}), toVec<int32_t>(model_input.input_lengths));
    EXPECT_EQ((std::vector<int32_t>{0, 1, 2, 3, 4, 5}), toVec<int32_t>(model_input.lm_output_indexes));
    // Fill-in DSpark keeps the anchor as conditioning only, so it queries one
    // extra row while still proposing gamma tokens.
    sp_config.sp_dspark_sample_from_anchor = false;
    MtpBatchStreamProcessor qwen_processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    qwen_processor.buildDSparkProposeInput(model_input, anchors, committed_ends, host_holder);
    EXPECT_EQ((std::vector<int32_t>{101, mask_id, mask_id, mask_id, 202, mask_id, mask_id, mask_id}),
              toVec<int32_t>(model_input.combo_tokens));
    EXPECT_EQ((std::vector<int32_t>{gamma + 1, gamma + 1}), toVec<int32_t>(model_input.input_lengths));
    EXPECT_EQ((std::vector<int32_t>{1, 2, 3, 5, 6, 7}), toVec<int32_t>(model_input.lm_output_indexes));
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkPrepareAndVerifyUsePerStreamDeviceState) {
    constexpr int32_t gamma = 3;

    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig  sp_config;
    model_config.max_seq_len                           = 2048;
    model_config.vocab_size                            = 256;
    model_config.num_layers                            = 1;
    model_config.mm_model_config.mm_position_ids_style = MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    sp_config.type                                     = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle                        = gamma;
    sp_config.sp_dspark_mask_token_id                  = 255;

    ResourceContext resource_context;

    auto steady_stream = createContextStream(model_config, runtime_config, resource_context, {10, 11}, 1);
    auto fresh_stream  = createContextStream(model_config, runtime_config, resource_context, {20, 21, 22}, 2);
    steady_stream->setIsContextStream(false);
    fresh_stream->setIsContextStream(false);
    steady_stream->setContextPositionIds(torch::tensor({1, 1, 1, 2, 2, 2}, torch::kInt32));
    fresh_stream->setContextPositionIds(torch::tensor({100, 101, 102, 110, 111, 112, 120, 121, 122}, torch::kInt32));

    // The steady stream's host token/length deliberately trail the state
    // published before its previous bookkeeping worker. The fresh stream has
    // no previous round and therefore legitimately falls back to host state.
    GenerateStream::MtpAsyncDeviceState steady_state;
    steady_state.accept_len_gpu        = torch::tensor({2}, torch::kInt32).to(torch::kCUDA);
    steady_state.accept_tokens_gpu     = torch::tensor({{101, 102, 0, 0}}, torch::kInt32).to(torch::kCUDA);
    steady_state.next_seq_len_gpu      = torch::tensor({10}, torch::kInt32).to(torch::kCUDA);
    steady_state.next_position_ids_gpu = torch::tensor({30, 40, 50}, torch::kInt32).to(torch::kCUDA);
    steady_stream->setMtpAsyncDeviceState(std::move(steady_state));

    StreamGroups   stream_groups({steady_stream, fresh_stream});
    GptModelInputs model_input;
    model_input.sequence_lengths = torch::tensor({1, 2}, torch::kInt32);

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder host_holder;
    auto         round_state    = processor.buildDSparkRoundState(stream_groups, model_input, host_holder);
    auto         proposal_input = model_input;
    processor.prepareDSparkProposeModelInput(round_state, proposal_input, host_holder);

    EXPECT_EQ((std::vector<int32_t>{102, 22}), toVec<int32_t>(round_state.anchors));
    EXPECT_EQ((std::vector<int32_t>{9, 2}), toVec<int32_t>(round_state.committed_ends));
    EXPECT_EQ((std::vector<int32_t>{102, 255, 255, 22, 255, 255}), toVec<int32_t>(proposal_input.combo_tokens));
    EXPECT_EQ((std::vector<int32_t>{gamma, gamma}), toVec<int32_t>(proposal_input.input_lengths));
    EXPECT_EQ((std::vector<int32_t>{9, 2}), toVec<int32_t>(proposal_input.prefix_lengths));
    EXPECT_EQ((std::vector<int32_t>{30, 40, 50, 31, 41, 51, 32, 42, 52, 122, 122, 122, 123, 123, 123, 124, 124, 124}),
              toVec<int32_t>(proposal_input.combo_position_ids));
    EXPECT_TRUE(proposal_input.combo_position_ids.is_cuda());
    // Proposal preparation must not mutate the target gather input.
    EXPECT_FALSE(model_input.combo_tokens.defined());
    EXPECT_FALSE(model_input.combo_position_ids.defined());

    auto target_prepare_input = proposal_input;
    processor.expandDSparkTargetVerifyPositionIdsFromProposal(target_prepare_input);
    EXPECT_EQ((std::vector<int32_t>{30,  40,  50,  31,  41,  51,  32,  42,  52,  33,  43,  53,
                                    122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125}),
              toVec<int32_t>(target_prepare_input.combo_position_ids));

    auto proposals = torch::tensor({{31, 32, 33}, {41, 42, 43}}, torch::kInt32).to(torch::kCUDA);
    processor.prepareDSparkTargetVerifyModelInput(round_state, model_input, proposals, host_holder);
    EXPECT_EQ((std::vector<int32_t>{102, 31, 32, 33, 22, 41, 42, 43}), toVec<int32_t>(model_input.combo_tokens));
    EXPECT_EQ((std::vector<int32_t>{gamma + 1, gamma + 1}), toVec<int32_t>(model_input.input_lengths));
    EXPECT_EQ((std::vector<int32_t>{9, 2}), toVec<int32_t>(model_input.prefix_lengths));
    EXPECT_EQ((std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}), toVec<int32_t>(model_input.lm_output_indexes));
    EXPECT_EQ((std::vector<int32_t>{30,  40,  50,  31,  41,  51,  32,  42,  52,  33,  43,  53,
                                    122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125}),
              toVec<int32_t>(model_input.combo_position_ids));
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkDecodeCommitPreservesDenseVerifyGeometry) {
    constexpr int32_t gamma = 3;

    ModelConfig                 model_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig  sp_config;
    sp_config.type              = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle = gamma;

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelInputs model_input;
    model_input.combo_tokens      = torch::tensor({11, 12, 13, 14, 21, 22, 23, 24}, torch::kInt32);
    model_input.input_lengths     = torch::tensor({gamma + 1, gamma + 1}, torch::kInt32);
    model_input.prefix_lengths    = torch::tensor({7, 15}, torch::kInt32);
    model_input.lm_output_indexes = torch::tensor({3, 7}, torch::kInt32);
    model_input.is_target_verify  = false;
    const auto combo_tokens       = model_input.combo_tokens.clone();
    const auto input_lengths      = model_input.input_lengths.clone();
    const auto prefix_lengths     = model_input.prefix_lengths.clone();
    const auto lm_output_indexes  = model_input.lm_output_indexes.clone();
    const auto target_features =
        torch::arange(0, 48, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({8, 6});

    processor.updateDecodePostDSparkCommitInput(model_input, target_features, 2);

    EXPECT_TRUE(torch::equal(model_input.combo_tokens, combo_tokens));
    EXPECT_TRUE(torch::equal(model_input.input_lengths, input_lengths));
    EXPECT_TRUE(torch::equal(model_input.prefix_lengths, prefix_lengths));
    EXPECT_TRUE(torch::equal(model_input.lm_output_indexes, lm_output_indexes));
    EXPECT_TRUE(torch::equal(model_input.last_hidden_states, target_features));
    EXPECT_TRUE(model_input.is_target_verify);
}

TEST_F(MtpBatchStreamProcessorTest, testUpdatePrefillPostDraftModelInput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

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
    GenerateStreamPtr stream3 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 3);
    GenerateStreamPtr middle_stream =
        createContextStream(model_config, runtime_config, resource_context, {1, 2, 3}, 4);
    middle_stream->setChunkSize(2);
    ASSERT_TRUE(middle_stream->isMiddleChunk());

    std::list<GenerateStreamPtr> streams{stream1, stream2, stream3, middle_stream};
    auto                         stream_groups = StreamGroups(streams);

    auto processor = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input_status = processor.gatherModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input = model_input_status.value();

    GptModelOutputs model_output;
    model_output.all_hidden_states = torch::zeros({7, 2}, torch::kFloat32);

    SamplerOutput sampler_output;
    sampler_output.token_ids =
        torch::tensor({1, -2, 2, 1, 2, 3, 3, 2, 1, 0, 1, 2}, torch::kInt32).reshape({4, 3});

    processor.updatePrefillPostDraftModelInput(stream_groups, model_input, model_output, sampler_output, holder);

    auto        combo_tokens        = model_input.combo_tokens;
    vector<int> expect_combo_tokens = {2, 2, 3, 2, 1, 2, 3};
    EXPECT_EQ(expect_combo_tokens, toVec<int>(combo_tokens));
}
TEST_F(MtpBatchStreamProcessorTest, testCudaPrefillShiftUsesPromptTailOnlyForMiddleChunks) {
    for (const bool explicit_positions : {false, true}) {
        for (const bool device_inputs : {false, true}) {
            for (const bool with_type_ids : {false, true}) {
                SCOPED_TRACE(testing::Message() << "explicit_positions=" << explicit_positions
                                               << ", device_inputs=" << device_inputs
                                               << ", with_type_ids=" << with_type_ids);
                ModelConfig config;
                config.max_seq_len = 64;
                config.vocab_size = 64;
                config.num_layers = 1;
                config.mm_model_config.mm_position_ids_style = DEFAULT;
                config.attn_config.rope_config.index_factor = 1;
                RuntimeConfig runtime;
                ResourceContext resources;
                auto middle = createContextStream(config, runtime, resources, {10, 11, 12, 13, 14, 15}, 1);
                auto final = createContextStream(config, runtime, resources, {20, 21, 22, 23, 24}, 2);
                for (const auto& stream : {middle, final}) {
                    stream->setReuseLength(2);
                    stream->setChunkSize(stream == middle ? 2 : 3);
                }
                ASSERT_TRUE(middle->isMiddleChunk());
                ASSERT_FALSE(final->isMiddleChunk());

                SpeculativeExecutionConfig spec;
                spec.gen_num_per_cycle = 1;
                MtpBatchStreamProcessor processor(config, PDSepConfig{}, ProfilingDebugLoggingConfig{},
                                                   makeProcessorCacheConfig(), spec, false);
                auto tensor_device = device_inputs ? torch::kCUDA : torch::kCPU;
                GptModelInputs input;
                input.combo_tokens = torch::tensor({12, 13, 22, 23, 24}, torch::kInt32).to(tensor_device);
                input.input_lengths = torch::tensor({2, 3}, torch::kInt32).to(tensor_device);
                if (explicit_positions) {
                    input.combo_position_ids = torch::tensor({2, 3, 2, 3, 4}, torch::kInt32).to(tensor_device);
                }
                if (with_type_ids) {
                    // Type IDs force the host fallback even without multimodal inputs.
                    input.combo_tokens_type_ids = torch::tensor({1, 2, 3, 4, 5}, torch::kInt32).to(tensor_device);
                }
                // Keep shared references so an in-place shift of the target input is observable.
                const auto original_tokens = input.combo_tokens;
                const auto original_positions = input.combo_position_ids;
                const auto original_type_ids = input.combo_tokens_type_ids;
                GptModelOutputs output;
                output.all_hidden_states = torch::zeros({5, 2});
                SamplerOutput sampled;
                sampled.token_ids = torch::tensor({{1, 2, 59}, {3, 4, 60}}, torch::kInt32).to(tensor_device);
                const auto original_samples = sampled.token_ids.clone();
                TensorHolder holder;
                processor.updatePrefillPostDraftModelInput(
                    StreamGroups({middle, final}), input, output, sampled, holder);

                // Shift each current window by one. The middle tail is prompt[4]=14;
                // the final tail is the sample, and its existing DEFAULT position stays 4.
                EXPECT_EQ((vector<int>{13, 14, 23, 24, 60}), toVec<int>(input.combo_tokens));
                EXPECT_EQ((vector<int>{12, 13, 22, 23, 24}), toVec<int>(original_tokens));
                if (explicit_positions) {
                    EXPECT_EQ((vector<int>{3, 4, 3, 4, 4}), toVec<int>(input.combo_position_ids));
                    EXPECT_EQ((vector<int>{2, 3, 2, 3, 4}), toVec<int>(original_positions));
                } else {
                    EXPECT_FALSE(input.combo_position_ids.defined());
                }
                if (with_type_ids) {
                    EXPECT_EQ((vector<int>{2, 0, 4, 5, 0}), toVec<int>(input.combo_tokens_type_ids));
                    EXPECT_EQ((vector<int>{1, 2, 3, 4, 5}), toVec<int>(original_type_ids));
                } else {
                    EXPECT_FALSE(input.combo_tokens_type_ids.defined());
                }
                EXPECT_TRUE(torch::equal(sampled.token_ids, original_samples));
                EXPECT_EQ(middle->reuseLength(), 2);
                EXPECT_EQ(final->reuseLength(), 2);
            }
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testDraftPrefillShiftsPositionsWithAndWithoutMultimodalInputs) {
    ModelConfig config;
    config.max_seq_len = 64;
    config.vocab_size = 64;
    config.num_layers = 1;
    config.attn_config.rope_config.index_factor = 1;
    RuntimeConfig runtime;
    ResourceContext resources;
    auto stream1 = createContextStream(config, runtime, resources, {10, 11, 12}, 1);
    auto stream2 = createContextStream(config, runtime, resources, {20, 21}, 2);
    auto groups = StreamGroups({stream1, stream2});
    SpeculativeExecutionConfig spec;
    spec.gen_num_per_cycle = 1;
    for (bool multimodal : {false, true}) {
        SCOPED_TRACE(testing::Message() << "multimodal=" << multimodal);
        MtpBatchStreamProcessor processor(config, PDSepConfig{}, ProfilingDebugLoggingConfig{},
                                           makeProcessorCacheConfig(), spec, false);
        // The non-MM case exercises CUDA-resident explicit positions and
        // tokens; the MM producer's existing full path uses host positions.
        auto device = multimodal ? torch::kCPU : torch::kCUDA;
        GptModelInputs input;
        input.combo_tokens = torch::tensor({10, 11, 12, 20, 21}, torch::kInt32).to(device);
        input.input_lengths = torch::tensor({3, 2}, torch::kInt32).to(device);
        input.combo_position_ids = torch::tensor({0, 1, 2, 0, 1}, torch::kInt32).to(device);
        if (multimodal) {
            input.text_tokens_mask = torch::tensor({1, 0, 1, 1, 0}, torch::kInt32);
            input.mm_features_locs = torch::tensor({1, 4}, torch::kInt32);
            input.multimodal_features = std::vector<torch::Tensor>{
                torch::full({1, 4}, 100.0), torch::full({1, 4}, 200.0)};
        }
        GptModelOutputs output;
        SamplerOutput sampled;
        sampled.token_ids = torch::tensor({{30}, {40}}, torch::kInt32).to(device);
        TensorHolder holder;
        processor.updatePrefillPostDraftModelInput(groups, input, output, sampled, holder);
        EXPECT_EQ((std::vector<int>{11, 12, 30, 21, 40}), toVec<int>(input.combo_tokens));
        EXPECT_EQ((std::vector<int>{1, 2, 2, 1, 1}), toVec<int>(input.combo_position_ids));
        if (multimodal) {
            EXPECT_EQ((std::vector<int>{0, 1, 1, 0, 1}), toVec<int>(input.text_tokens_mask));
            EXPECT_EQ((std::vector<int>{0, 3}), toVec<int>(input.mm_features_locs));
            ASSERT_EQ(input.multimodal_features->size(), 2);
            EXPECT_TRUE(torch::equal(input.multimodal_features->at(0), torch::full({1, 4}, 100.0)));
            EXPECT_TRUE(torch::equal(input.multimodal_features->at(1), torch::full({1, 4}, 200.0)));
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testUpdatePrefillPostDraftModelInputShiftsComboPositionIds) {
    ModelConfig                 model_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig  sp_config;
    model_config.max_seq_len                          = 2048;
    model_config.vocab_size                           = 4;
    model_config.num_layers                           = 1;
    model_config.mm_model_config.mm_position_ids_style = MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    sp_config.gen_num_per_cycle                       = 2;
    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;
    auto            stream1 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 1);
    auto            stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3}, 2);
    auto            middle_stream =
        createContextStream(model_config, runtime_config, resource_context, {1, 2, 3, 0}, 3);
    stream1->setContextPositionIds(torch::tensor({100, 101, 102, 110, 111, 112}, torch::kInt32));
    stream2->setContextPositionIds(torch::tensor({200, 201, 202, 210, 211, 212, 220, 221, 222}, torch::kInt32));
    middle_stream->setContextPositionIds(
        torch::tensor({300, 301, 302, 310, 311, 312, 320, 321, 322, 330, 331, 332}, torch::kInt32));
    middle_stream->setChunkSize(2);
    ASSERT_TRUE(middle_stream->isMiddleChunk());
    auto           stream_groups = StreamGroups({stream1, stream2, middle_stream});
    GptModelInputs model_input;
    model_input.input_lengths = torch::tensor({2, 3, 2}, torch::kInt32);
    model_input.combo_tokens  = torch::tensor({10, 11, 20, 21, 22, 30, 31}, torch::kInt32);
    model_input.combo_position_ids =
        torch::tensor({100, 101, 102, 110, 111, 112, 200, 201, 202, 210, 211, 212,
                       220, 221, 222, 300, 301, 302, 310, 311, 312},
                      torch::kInt32);
    GptModelOutputs model_output;
    model_output.all_hidden_states = torch::zeros({7, 2}, torch::kFloat32);
    SamplerOutput sampler_output;
    sampler_output.token_ids = torch::tensor({1, -2, 12, 1, 2, 23, 3, 2, 1}, torch::kInt32).reshape({3, 3});
    TensorHolder holder;
    processor.updatePrefillPostDraftModelInput(stream_groups, model_input, model_output, sampler_output, holder);
    EXPECT_EQ((vector<int>{11, 12, 21, 22, 23, 31, 3}), toVec<int>(model_input.combo_tokens));
    EXPECT_EQ((vector<int>{110, 111, 112, 112, 112, 112,
                           210, 211, 212, 220, 221, 222, 222, 222, 222,
                           310, 311, 312, 320, 321, 322}),
              toVec<int>(model_input.combo_position_ids));
}

TEST_F(MtpBatchStreamProcessorTest, testUpdateDecodePostDraftModelInput) {
    autil::EnvGuard stream_async("RTP_LLM_STREAM_ASYNC", "0");
    autil::EnvGuard device_state("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "0");
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

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

    auto processor = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input_status = processor.gatherModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input            = model_input_status.value();
    model_input.is_target_verify = true;
    model_input.input_lengths    = torch::tensor({3, 3}, torch::kInt32);

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

    auto        input_lengths        = model_input.input_lengths;
    vector<int> expect_input_lengths = {3, 3};
    EXPECT_EQ(expect_input_lengths, toVec<int>(input_lengths));

    auto        lm_output_indexes        = model_input.lm_output_indexes.cpu();
    vector<int> expect_lm_output_indexes = {2, 3};
    EXPECT_EQ(expect_lm_output_indexes, toVec<int>(lm_output_indexes));

    auto          last_hidden_states        = model_input.last_hidden_states;
    vector<float> expect_last_hidden_states = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                                              1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f};
    EXPECT_EQ(expect_last_hidden_states, toVec<float>(last_hidden_states));
    EXPECT_FALSE(model_input.is_target_verify);
    EXPECT_TRUE(model_input.is_mtp_draft_update);
}

TEST_F(MtpBatchStreamProcessorTest, testUpdateDecodePostDraftModelInputKeepsDenseDeviceState) {
    autil::EnvGuard device_state("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "1");
    ModelConfig                 model_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig  sp_config;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 2;

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    GptModelInputs                        model_input;
    speculative::SpeculativeSamplerOutput spec_decode_output;
    spec_decode_output.accept_len    = torch::tensor({3, 1}, torch::kInt32).to(torch::kCUDA);
    spec_decode_output.accept_tokens = torch::tensor({{2, 3, 1}, {2, 0, 0}}, torch::kInt32).to(torch::kCUDA);
    GptModelOutputs model_output;
    model_output.all_hidden_states =
        torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f}, torch::kFloat32)
            .reshape({6, 2})
            .to(torch::kCUDA);
    torch::Tensor hidden_states_d_t;
    TensorHolder  holder;

    processor.updateDecodePostDraftModelInput(
        model_input, model_output, spec_decode_output, 2, hidden_states_d_t, holder);

    EXPECT_TRUE(model_input.combo_tokens.is_cuda());
    EXPECT_EQ((vector<int>{2, 3, 1, 2, 0, 0}), toVec<int>(model_input.combo_tokens));
    EXPECT_TRUE(model_input.lm_output_indexes.is_cuda());
    EXPECT_EQ((vector<int>{2, 3}), toVec<int>(model_input.lm_output_indexes));
    EXPECT_EQ(6, model_input.last_hidden_states.size(0));
}

TEST_F(MtpBatchStreamProcessorTest, testUpdateDecodePostDraftModelInputPreservesComboPositionIds) {
    autil::EnvGuard stream_async("RTP_LLM_STREAM_ASYNC", "0");
    autil::EnvGuard device_state("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "0");
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

    model_config.max_seq_len                           = 2048;
    model_config.vocab_size                            = 4;
    model_config.num_layers                            = 1;
    model_config.mm_model_config.mm_position_ids_style = MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    sp_config.gen_num_per_cycle                        = 2;

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

    auto processor = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input_status = processor.gatherModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input              = model_input_status.value();
    model_input.is_target_verify   = true;
    model_input.input_lengths      = torch::tensor({3, 3, 3}, torch::kInt32);
    model_input.combo_position_ids = torch::tensor(
        {10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 91, 92},
        torch::kInt32);

    speculative::SpeculativeSamplerOutput spec_decode_output;
    spec_decode_output.accept_len_cpu    = torch::tensor({3, 2, 1}, torch::kInt32);
    spec_decode_output.accept_tokens_cpu = torch::tensor({{2, 3, 1}, {2, 3, 0}, {2, 0, 0}}, torch::kInt32);
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

    auto        combo_position_ids        = model_input.combo_position_ids;
    vector<int> expect_combo_position_ids = {10, 11, 12, 20, 21, 22, 30, 31, 32,
                                            40, 41, 42, 50, 51, 52, 60, 61, 62,
                                            70, 71, 72, 80, 81, 82, 90, 91, 92};
    EXPECT_EQ(expect_combo_position_ids, toVec<int>(combo_position_ids));
    EXPECT_EQ((vector<int>{3, 3, 3}), toVec<int>(model_input.input_lengths));
    EXPECT_EQ((vector<int>{2, 4, 6}), toVec<int>(model_input.lm_output_indexes));
    EXPECT_EQ(model_output.all_hidden_states.data_ptr(), model_input.last_hidden_states.data_ptr());
}

TEST_F(MtpBatchStreamProcessorTest, testUpdateDecodeDraftModelInputAdvancesComboPositionIds) {
    ModelConfig                 model_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig  sp_config;

    model_config.max_seq_len                           = 2048;
    model_config.vocab_size                            = 4;
    model_config.num_layers                            = 1;
    model_config.mm_model_config.mm_position_ids_style = MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    sp_config.gen_num_per_cycle                        = 2;

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    for (const auto device : {torch::kCPU, torch::kCUDA}) {
        const auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
        GptModelInputs model_input;
        model_input.combo_tokens       = torch::tensor({11, 21}, options);
        model_input.sequence_lengths   = torch::tensor({5, 7}, options);
        model_input.combo_position_ids = torch::tensor({5, 6, 7, 10, 11, 12}, options);
        const auto previous_lengths   = model_input.sequence_lengths;
        const auto previous_positions = model_input.combo_position_ids;

        GptModelOutputs model_output;
        model_output.all_hidden_states = torch::tensor({0.1f, 0.2f, 1.1f, 1.2f}, torch::kFloat32).reshape({2, 2});
        auto draft_token_ids           = torch::tensor({12, 22}, options).reshape({2, 1});

        TensorHolder holder;
        processor.updateDecodeDraftModelInput(model_input, model_output, draft_token_ids, holder);

        EXPECT_EQ((vector<int>{12, 22}), toVec<int>(model_input.combo_tokens));
        EXPECT_EQ((vector<int>{6, 8}), toVec<int>(model_input.sequence_lengths));
        EXPECT_EQ((vector<float>{0.1, 0.2, 1.1, 1.2}), toVec<float>(model_input.last_hidden_states));
        EXPECT_EQ((vector<int>{6, 7, 8, 11, 12, 13}), toVec<int>(model_input.combo_position_ids));
        EXPECT_EQ((vector<int>{5, 7}), toVec<int>(previous_lengths));
        EXPECT_EQ((vector<int>{5, 6, 7, 10, 11, 12}), toVec<int>(previous_positions));
    }
}

TEST_F(MtpBatchStreamProcessorTest, testExpandTargetVerifyPositionIdsFromDecodeKeepsPinnedHostOutput) {
    ModelConfig                 model_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig  sp_config;

    model_config.max_seq_len                           = 2048;
    model_config.vocab_size                            = 4;
    model_config.num_layers                            = 1;
    model_config.mm_model_config.mm_position_ids_style = MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    sp_config.gen_num_per_cycle                        = 2;

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    for (const auto device : {torch::kCPU, torch::kCUDA}) {
        GptModelInputs model_input;
        model_input.input_lengths = torch::tensor({3, 3}, torch::kInt32);
        model_input.combo_position_ids = torch::tensor({5, 6, 7, 10, 11, 12}, torch::kInt32).pin_memory().to(device);
        const auto original_positions = model_input.combo_position_ids;

        processor.expandTargetVerifyPositionIdsFromDecode(model_input);

        EXPECT_EQ(device, model_input.combo_position_ids.device().type());
        EXPECT_TRUE(model_input.combo_position_ids.is_contiguous());
        if (device == torch::kCPU) {
            EXPECT_TRUE(model_input.combo_position_ids.is_pinned());
        }
        EXPECT_EQ((vector<int>{5, 6, 7, 6, 7, 8, 7, 8, 9, 10, 11, 12, 11, 12, 13, 12, 13, 14}),
                  toVec<int>(model_input.combo_position_ids));
        EXPECT_EQ((vector<int>{5, 6, 7, 10, 11, 12}), toVec<int>(original_positions));
    }
}

TEST_F(MtpBatchStreamProcessorTest, testExpandTargetVerifyPositionIdsInitializesNonDriverPlaceholder) {
    ModelConfig                 model_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig  sp_config;

    model_config.max_seq_len                           = 2048;
    model_config.vocab_size                            = 4;
    model_config.num_layers                            = 1;
    model_config.mm_model_config.mm_position_ids_style = MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    sp_config.gen_num_per_cycle                        = 2;

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelInputs model_input;
    model_input.combo_position_ids = torch::tensor({5, 6, 7, 10, 11, 12}, torch::kInt32);
    auto empty_stream_groups       = StreamGroups(std::list<GenerateStreamPtr>{});

    processor.expandTargetVerifyPositionIds(empty_stream_groups, model_input);

    EXPECT_EQ(18, model_input.combo_position_ids.numel());
    EXPECT_EQ((vector<int>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
              toVec<int>(model_input.combo_position_ids));
}


TEST_F(MtpBatchStreamProcessorTest, testUpdateOneStepDraftSamplerOutput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

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
    stream1->getSPOutputBuffer()->tokens    = torch::tensor({1, 2}, torch::kInt32).reshape({1, 2});
    stream2->getSPOutputBuffer()->tokens    = torch::tensor({2, 3}, torch::kInt32).reshape({1, 2});

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
    setenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "1", 1);

    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

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
    stream1->getSPOutputBuffer()->tokens    = torch::tensor({1, 2}, torch::kInt32).reshape({1, 2});
    stream2->getSPOutputBuffer()->tokens    = torch::tensor({2, 3}, torch::kInt32).reshape({1, 2});

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

    unsetenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE");
}

TEST_F(MtpBatchStreamProcessorTest, updateMultiStepDraftSamplerOutput) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

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
    stream1->getSPOutputBuffer()->tokens    = torch::tensor({1, 2}, torch::kInt32).reshape({1, 2});
    stream2->getSPOutputBuffer()->tokens    = torch::tensor({2, 3}, torch::kInt32).reshape({1, 2});

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

TEST_F(MtpBatchStreamProcessorTest, testPrefillDispatchUsesDraftLastHiddenOverride) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();

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

TEST_F(MtpBatchStreamProcessorTest, testDSparkCommitOnlyPrefillDispatch) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    model_config.max_seq_len                 = 2048;
    model_config.vocab_size                  = 8;
    model_config.num_layers                  = 1;
    sp_config.type                           = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle              = 3;
    sp_config.sp_dspark_mask_token_id        = 0;

    ResourceContext resource_context;
    auto            stream = createContextStream(model_config, runtime_config, resource_context, {2}, 1);
    stream->getSPOutputBuffer()->propose_tokens_gpu = torch::tensor({9}, torch::kInt32).to(torch::kCUDA);
    StreamGroups            stream_groups({stream});
    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    MergedOutput target_output;
    target_output.sampler_output.token_ids = torch::tensor({3}, torch::kInt32).reshape({1, 1});
    MergedOutput commit_output;

    auto status = processor.dispatchPrefill(stream_groups, target_output, commit_output);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ((std::vector<int>{2, 3}), stream->getCompleteTokenIds()->completeTokenIdsVec(0));
    EXPECT_TRUE(stream->getProposeToken().empty());
    EXPECT_FALSE(stream->getSPOutputBuffer()->all_probs.defined());
    EXPECT_FALSE(stream->getSPOutputBuffer()->hidden_states.defined());
    EXPECT_FALSE(stream->getSPOutputBuffer()->propose_tokens_gpu.defined());
}

TEST_F(MtpBatchStreamProcessorTest, ReplaySnapshotsDoNotRequirePerProposalPages) {
    // Replay/FULL tables contain canonical pages only. A legacy permutation
    // would gather proposal slot 7 from this one-page table even if no group
    // subsequently used the permutation.
    auto table = torch::tensor({11, 21}, torch::kInt32).reshape({2, 1, 1}).to(torch::kCUDA);
    auto actual =
        MtpBatchStreamProcessor::advanceLinearCacheBlockTable(table,
                                                              torch::tensor({2}, torch::kInt32).to(torch::kCUDA),
                                                              torch::tensor({8}, torch::kInt32).to(torch::kCUDA),
                                                              {CacheGroupType::FULL, CacheGroupType::FULL},
                                                              /*group_spans=*/{0, 0});
    EXPECT_TRUE(torch::equal(actual.cpu(), table.cpu()));
    EXPECT_NE(actual.data_ptr(), table.data_ptr());
}

TEST_F(MtpBatchStreamProcessorTest, testAdvanceLinearCacheBlockTableMatchesHostCommitSemantics) {
    const std::vector<int32_t> previous_lengths = {1, 63, 64, 65, 127, 128, 129, 191};
    const std::vector<int32_t> accept_lengths   = {1, 2, 3, 4, 5, 6, 7, 8};
    constexpr int64_t          group_count      = 3;
    constexpr int64_t          block_count      = 12;
    const int64_t              batch_size       = static_cast<int64_t>(previous_lengths.size());

    auto current = torch::arange(group_count * batch_size * block_count, torch::kInt32)
                       .reshape({group_count, batch_size, block_count});
    auto actual = MtpBatchStreamProcessor::advanceLinearCacheBlockTable(
                      current.to(torch::kCUDA),
                      torch::tensor(previous_lengths, torch::kInt32).to(torch::kCUDA),
                      torch::tensor(accept_lengths, torch::kInt32).to(torch::kCUDA),
                      {CacheGroupType::LINEAR, CacheGroupType::FULL, CacheGroupType::LINEAR},
                      /*group_spans=*/{64, 0, 64})
                      .cpu();

    auto expected = current.clone();
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const int accept = accept_lengths[static_cast<size_t>(batch)];
        if (accept <= 1) {
            continue;
        }
        const int  current_cached = previous_lengths[static_cast<size_t>(batch)] - 1;
        const int  next_cached    = current_cached + accept;
        const auto cached_swap    = getCachedTokenBlockSwapIdx(current_cached, next_cached, 64);
        const auto final_swap     = getFinalTokenBlockSwapIdx(current_cached, next_cached, 64);
        for (const int64_t group : {0L, 2L}) {
            auto* row = expected.select(0, group).select(0, batch).data_ptr<int32_t>();
            std::swap(row[cached_swap.first], row[cached_swap.second]);
            std::swap(row[final_swap.first], row[final_swap.second]);
        }
    }

    EXPECT_TRUE(torch::equal(actual, expected));
    EXPECT_TRUE(torch::equal(actual.select(0, 1), current.select(0, 1)));
}

TEST_F(MtpBatchStreamProcessorTest, testAdvanceLinearCacheBlockTableMatchesEveryBlockOffsetAndAcceptLength) {
    constexpr int32_t    block_size  = 64;
    constexpr int32_t    max_accept  = 8;
    constexpr int64_t    group_count = 2;
    constexpr int64_t    block_count = 16;
    constexpr int64_t    offsets     = block_size;
    constexpr int64_t    batch_size  = offsets * max_accept;
    std::vector<int32_t> previous_lengths;
    std::vector<int32_t> accept_lengths;
    previous_lengths.reserve(batch_size);
    accept_lengths.reserve(batch_size);
    for (int32_t offset = 0; offset < block_size; ++offset) {
        for (int32_t accept = 1; accept <= max_accept; ++accept) {
            // previous_length - 1 covers every cached-token offset in a block.
            previous_lengths.push_back(block_size + offset + 1);
            accept_lengths.push_back(accept);
        }
    }

    auto current = torch::arange(group_count * batch_size * block_count, torch::kInt32)
                       .reshape({group_count, batch_size, block_count});
    auto actual = MtpBatchStreamProcessor::advanceLinearCacheBlockTable(
                      current.to(torch::kCUDA),
                      torch::tensor(previous_lengths, torch::kInt32).to(torch::kCUDA),
                      torch::tensor(accept_lengths, torch::kInt32).to(torch::kCUDA),
                      {CacheGroupType::LINEAR, CacheGroupType::FULL},
                      /*group_spans=*/{block_size, 0})
                      .cpu();

    auto expected = current.clone();
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const int accept = accept_lengths[static_cast<size_t>(batch)];
        if (accept <= 1) {
            continue;
        }
        const int  current_cached = previous_lengths[static_cast<size_t>(batch)] - 1;
        const int  next_cached    = current_cached + accept;
        const auto cached_swap    = getCachedTokenBlockSwapIdx(current_cached, next_cached, block_size);
        const auto final_swap     = getFinalTokenBlockSwapIdx(current_cached, next_cached, block_size);
        auto*      row            = expected.select(0, 0).select(0, batch).data_ptr<int32_t>();
        std::swap(row[cached_swap.first], row[cached_swap.second]);
        std::swap(row[final_swap.first], row[final_swap.second]);
    }

    EXPECT_TRUE(torch::equal(actual, expected));
}

TEST_F(MtpBatchStreamProcessorTest, testAdvanceLinearCacheBlockTableUsesEachLinearGroupsSpan) {
    const std::vector<int32_t> previous_lengths = {32767, 32768, 32769, 65535, 65536, 65537};
    const std::vector<int32_t> accept_lengths   = {4, 3, 1, 4, 3, 4};
    const std::vector<int64_t> group_spans      = {32768, 0, 65536};
    constexpr int64_t          group_count      = 3;
    constexpr int64_t          block_count      = 8;
    const int64_t              batch_size       = static_cast<int64_t>(previous_lengths.size());
    auto                       current          = torch::arange(group_count * batch_size * block_count, torch::kInt32)
                       .reshape({group_count, batch_size, block_count});
    auto actual = MtpBatchStreamProcessor::advanceLinearCacheBlockTable(
                      current.to(torch::kCUDA),
                      torch::tensor(previous_lengths, torch::kInt32).to(torch::kCUDA),
                      torch::tensor(accept_lengths, torch::kInt32).to(torch::kCUDA),
                      {CacheGroupType::LINEAR, CacheGroupType::FULL, CacheGroupType::LINEAR},
                      group_spans)
                      .cpu();

    auto expected = current.clone();
    for (const int64_t group : {0L, 2L}) {
        const int span = static_cast<int>(group_spans[static_cast<size_t>(group)]);
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const int accept = accept_lengths[static_cast<size_t>(batch)];
            if (accept <= 1) {
                continue;
            }
            const int  current_cached = previous_lengths[static_cast<size_t>(batch)] - 1;
            const int  next_cached    = current_cached + accept;
            const auto cached_swap    = getCachedTokenBlockSwapIdx(current_cached, next_cached, span);
            const auto final_swap     = getFinalTokenBlockSwapIdx(current_cached, next_cached, span);
            auto*      row            = expected.select(0, group).select(0, batch).data_ptr<int32_t>();
            std::swap(row[cached_swap.first], row[cached_swap.second]);
            std::swap(row[final_swap.first], row[final_swap.second]);
        }
    }
    EXPECT_TRUE(torch::equal(actual, expected));
    EXPECT_TRUE(torch::equal(actual.select(0, 1), current.select(0, 1)));
    // The same accepted window crosses the first group's boundary but not
    // the second's. Sharing one permutation would publish the wrong state.
    EXPECT_EQ(actual[0][0][0].item<int32_t>(), current[0][0][1].item<int32_t>());
    EXPECT_EQ(actual[2][0][0].item<int32_t>(), current[2][0][3].item<int32_t>());
}

TEST_F(MtpBatchStreamProcessorTest, testAdvanceLinearCacheBlockTableKeepsLongContextRowsAndPadding) {
    // The global cache-key span may be 2048 while this LINEAR group covers
    // 32768 tokens. Its real five-slot row is padded to the other request's
    // seven slots; only the active LINEAR slots may be permuted.
    auto current =
        torch::tensor({11, 12, 13, 14, 15, 0, 0, 21, 22, 23, 24, 25, 26, 27}, torch::kInt32).reshape({1, 2, 7});
    auto device_table = current.to(torch::kCUDA);
    auto actual       = MtpBatchStreamProcessor::advanceLinearCacheBlockTable(
                      device_table,
                      torch::tensor({65537, 131073}, torch::kInt32).to(torch::kCUDA),
                      torch::tensor({3, 3}, torch::kInt32).to(torch::kCUDA),
                      {CacheGroupType::LINEAR},
                      /*group_spans=*/{32768})
                      .cpu();
    auto expected =
        torch::tensor({11, 12, 15, 14, 13, 0, 0, 21, 22, 23, 24, 27, 26, 25}, torch::kInt32).reshape({1, 2, 7});
    EXPECT_TRUE(torch::equal(actual, expected));
    EXPECT_TRUE(torch::equal(device_table.cpu(), current));
}

TEST_F(MtpBatchStreamProcessorTest, testK3AcceptedStatePayloadUsesLogicalPrefix) {
    // Candidate i contains the state after the current token and i draft tokens.
    // Use payload identity as the oracle, independently of host swap helpers.
    constexpr int32_t    block_size = 4096;
    constexpr int64_t    blocks     = 8;
    std::vector<int32_t> previous, accepts;
    for (int32_t cached : {4094, 4095, 4096, 8190, 8191, 8192}) {
        for (int32_t accepted_drafts = 0; accepted_drafts <= 3; ++accepted_drafts) {
            previous.push_back(cached + 1);
            accepts.push_back(accepted_drafts + 1);
        }
    }
    const int64_t batch    = previous.size();
    auto          table    = (torch::arange(3 * batch * blocks, torch::kInt32) + 1).reshape({3, batch, blocks});
    auto          advanced = MtpBatchStreamProcessor::advanceLinearCacheBlockTable(
                        table.to(torch::kCUDA),
                        torch::tensor(previous, torch::kInt32).to(torch::kCUDA),
                        torch::tensor(accepts, torch::kInt32).to(torch::kCUDA),
                        {CacheGroupType::LINEAR, CacheGroupType::FULL, CacheGroupType::LINEAR},
                        /*group_spans=*/{block_size, 0, block_size})
                        .cpu();
    for (int64_t row = 0; row < batch; ++row) {
        const int32_t cached     = previous[row] - 1;
        const int32_t count      = accepts[row];
        const int32_t source     = cached / block_size + count - 1;
        const int32_t final_slot = (cached + count - 1) / block_size;
        for (int64_t group : {0L, 2L}) {
            EXPECT_EQ(advanced[group][row][final_slot].item<int32_t>(), table[group][row][source].item<int32_t>())
                << "cached=" << cached << " accepted_drafts=" << count - 1;
            // A completed boundary retains its own prefix checkpoint even if
            // the accepted final state has advanced into the following block.
            const int32_t until_boundary = block_size - cached % block_size;
            if (until_boundary < count) {
                const int32_t boundary_slot   = cached / block_size;
                const int32_t boundary_source = cached / block_size + until_boundary - 1;
                EXPECT_EQ(advanced[group][row][boundary_slot].item<int32_t>(),
                          table[group][row][boundary_source].item<int32_t>());
            }
        }
    }
    EXPECT_TRUE(torch::equal(table.select(0, 1), advanced.select(0, 1)));
}

TEST_F(MtpBatchStreamProcessorTest, testCacheSnapshotOverlayKeepsPhysicalKernelPairAndFreshRows) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    model_config.max_seq_len                 = 2048;
    model_config.vocab_size                  = 8;
    model_config.num_layers                  = 1;
    sp_config.gen_num_per_cycle              = 3;

    ResourceContext resource_context;
    auto            steady = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    auto            fresh  = createContextStream(model_config, runtime_config, resource_context, {2}, 2);
    steady->setIsContextStream(false);
    fresh->setIsContextStream(false);

    const auto                          cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    GenerateStream::MtpAsyncDeviceState state;
    state.next_kv_cache_block_id_gpu = torch::tensor({11, 12, 13}, torch::kInt32).reshape({1, 1, 3}).to(cuda_i32);
    state.next_kv_cache_kernel_block_id_gpu =
        torch::tensor({101, 102, 103, 104}, torch::kInt32).reshape({1, 1, 4}).to(cuda_i32);
    steady->setMtpAsyncDeviceState(std::move(state));

    GptModelInputs model_input;
    auto           pinned_i32            = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    model_input.kv_cache_block_id        = torch::zeros({1, 2, 5}, pinned_i32);
    model_input.kv_cache_kernel_block_id = torch::zeros({1, 2, 7}, pinned_i32);
    model_input.kv_cache_block_id.select(1, 1).copy_(torch::tensor({21, 22, 23, 24, 25}, torch::kInt32));
    model_input.kv_cache_kernel_block_id.select(1, 1).copy_(
        torch::tensor({201, 202, 203, 204, 205, 206, 207}, torch::kInt32));

    TestableMtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    processor.overlayMtpCacheSnapshots(StreamGroups({steady, fresh}), model_input, holder);

    ASSERT_TRUE(model_input.kv_cache_block_id.is_cuda());
    ASSERT_TRUE(model_input.kv_cache_kernel_block_id.is_cuda());
    EXPECT_EQ((std::vector<int32_t>{11, 12, 13, 0, 0}), toVec<int32_t>(model_input.kv_cache_block_id.select(1, 0)));
    EXPECT_EQ((std::vector<int32_t>{21, 22, 23, 24, 25}), toVec<int32_t>(model_input.kv_cache_block_id.select(1, 1)));
    EXPECT_EQ((std::vector<int32_t>{101, 102, 103, 104, 0, 0, 0}),
              toVec<int32_t>(model_input.kv_cache_kernel_block_id.select(1, 0)));
    EXPECT_EQ((std::vector<int32_t>{201, 202, 203, 204, 205, 206, 207}),
              toVec<int32_t>(model_input.kv_cache_kernel_block_id.select(1, 1)));

    auto too_narrow              = model_input;
    too_narrow.kv_cache_block_id = torch::zeros({1, 2, 2}, pinned_i32);
    EXPECT_ANY_THROW(processor.overlayMtpCacheSnapshots(StreamGroups({steady, fresh}), too_narrow, holder));
    too_narrow                          = model_input;
    too_narrow.kv_cache_kernel_block_id = torch::zeros({1, 2, 3}, pinned_i32);
    EXPECT_ANY_THROW(processor.overlayMtpCacheSnapshots(StreamGroups({steady, fresh}), too_narrow, holder));
}

TEST_F(MtpBatchStreamProcessorTest, testCacheSnapshotOverlayRestoresReplayPagesInMixedGroups) {
    ModelConfig model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 8;
    model_config.num_layers  = 2;
    RuntimeConfig              runtime_config;
    SpeculativeExecutionConfig sp_config;
    sp_config.gen_num_per_cycle = 3;
    CacheConfig cache_config;
    cache_config.layer_num = model_config.num_layers;
    cache_config.fromGroupedSpecs(
        {test::makeLinearSpec("linear", 1, TYPE_FP16, 1, 1), test::makeMhaSpec("full", 1, TYPE_FP16, 1, 1)},
        {{0}, {1}},
        {CacheGroupType::LINEAR, CacheGroupType::FULL},
        {"linear", "full"});
    cache_config.linear_replay.group_ids = {0};
    TestableMtpBatchStreamProcessor processor(model_config, {}, {}, cache_config, sp_config, false);

    auto steady = createContextStream(model_config, runtime_config, {}, {1}, 1);
    auto fresh  = createContextStream(model_config, runtime_config, {}, {2}, 2);
    steady->setIsContextStream(false);
    fresh->setIsContextStream(false);
    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    GenerateStream::MtpAsyncDeviceState state;
    state.next_kv_cache_block_id_gpu =
        torch::tensor({11, 12, 13, 31, 32, 33}, torch::kInt32).reshape({2, 1, 3}).to(cuda_i32);
    state.next_kv_cache_kernel_block_id_gpu =
        torch::tensor({101, 102, 103, 104, 301, 302, 303, 304}, torch::kInt32).reshape({2, 1, 4}).to(cuda_i32);
    steady->setMtpAsyncDeviceState(state);
    // The steady row is owned by the immutable GPU snapshot, even if its
    // mutable host resource is unavailable while bookkeeping runs.
    steady->setKVCache(BatchKVCacheResource{});

    const auto expected_physical = torch::tensor(
        {11, 12, 13, 0, 0, 21, 22, 23, 24, 25, 31, 32, 33, 0, 0, 41, 42, 43, 44, 45}, torch::kInt32)
                                       .reshape({2, 2, 5});
    const auto expected_kernel = torch::tensor(
        {101, 102, 103, 104, 0, 0, 0, 201, 202, 203, 204, 205, 206, 207,
         301, 302, 303, 304, 0, 0, 0, 401, 402, 403, 404, 405, 406, 407}, torch::kInt32)
                                     .reshape({2, 2, 7});
    for (bool device_tables : {false, true}) {
        SCOPED_TRACE(device_tables);
        GptModelInputs inputs;
        inputs.kv_cache_block_id = expected_physical.pin_memory();
        inputs.kv_cache_kernel_block_id = expected_kernel.pin_memory();
        // This is the gather contract for snapshot-backed streams: both
        // tables are zero until overlay, for every cache group.
        inputs.kv_cache_block_id.select(1, 0).zero_();
        inputs.kv_cache_kernel_block_id.select(1, 0).zero_();
        if (device_tables) {
            inputs.kv_cache_block_id = inputs.kv_cache_block_id.to(cuda_i32);
            inputs.kv_cache_kernel_block_id = inputs.kv_cache_kernel_block_id.to(cuda_i32);
        }
        TensorHolder holder;
        processor.overlayMtpCacheSnapshots(StreamGroups({steady, fresh}), inputs, holder);
        EXPECT_TRUE(torch::equal(inputs.kv_cache_block_id.cpu(), expected_physical));
        EXPECT_TRUE(torch::equal(inputs.kv_cache_kernel_block_id.cpu(), expected_kernel));
        EXPECT_TRUE(torch::equal(state.next_kv_cache_block_id_gpu.cpu(),
                                expected_physical.narrow(1, 0, 1).narrow(2, 0, 3)));
        EXPECT_TRUE(torch::equal(state.next_kv_cache_kernel_block_id_gpu.cpu(),
                                expected_kernel.narrow(1, 0, 1).narrow(2, 0, 4)));
    }
}

TEST_F(MtpBatchStreamProcessorTest, testPrefillShiftsMultimodalMaskAndPackedLocations) {
    ModelConfig model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 64;
    model_config.num_layers  = 1;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;
    auto            stream1 = createContextStream(model_config, runtime_config, resource_context, {10, 11, 12}, 1);
    auto            stream2 = createContextStream(model_config, runtime_config, resource_context, {20, 21}, 2);
    auto            groups  = StreamGroups({stream1, stream2});
    SpeculativeExecutionConfig speculative;
    speculative.gen_num_per_cycle = 1;
    MtpBatchStreamProcessor processor(
        model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, makeProcessorCacheConfig(), speculative, false);
    for (bool starts_at_boundary : {false, true}) {
        GptModelInputs input;
        input.combo_tokens          = torch::tensor({10, 11, 12, 20, 21}, torch::kInt32);
        input.input_lengths         = torch::tensor({3, 2}, torch::kInt32);
        input.text_tokens_mask      = torch::tensor({1, 0, 1, 1, 0}, torch::kInt32);
        input.combo_tokens_type_ids = torch::tensor({0, 1, 0, 0, 1}, torch::kInt32);
        input.mm_features_locs =
            torch::tensor(starts_at_boundary ? std::vector<int>{0, 3} : std::vector<int>{1, 4}, torch::kInt32);
        int rows                  = starts_at_boundary ? 2 : 1;
        input.multimodal_features = std::vector<torch::Tensor>{torch::ones({rows, 4}), torch::ones({rows, 4})};
        input.mm_extra_input      = std::vector<torch::Tensor>{torch::arange(rows * 8), torch::arange(rows * 8)};
        GptModelOutputs output;
        SamplerOutput   sampled;
        sampled.token_ids = torch::tensor({{30}, {40}}, torch::kInt32);
        TensorHolder holder;
        processor.updatePrefillPostDraftModelInput(groups, input, output, sampled, holder);
        EXPECT_EQ((std::vector<int>{11, 12, 30, 21, 40}), toVec<int>(input.combo_tokens));
        EXPECT_EQ((std::vector<int>{0, 1, 1, 0, 1}), toVec<int>(input.text_tokens_mask));
        EXPECT_EQ((std::vector<int>{1, 0, 0, 1, 0}), toVec<int>(input.combo_tokens_type_ids));
        EXPECT_EQ((std::vector<int>{0, 3}), toVec<int>(input.mm_features_locs));
        ASSERT_EQ(input.multimodal_features->size(), 2);
        EXPECT_EQ(input.multimodal_features->at(0).size(0), 1);
        EXPECT_EQ(input.multimodal_features->at(1).size(0), 1);
        EXPECT_EQ(input.mm_extra_input->at(0).numel(), 8);
        EXPECT_EQ(input.mm_extra_input->at(1).numel(), 8);
    }
}

void expectMultimodalRows(const GptModelInputs&                 input,
                          const vector<int>&                    expected_locs,
                          const vector<pair<int64_t, int64_t>>& expected_prompt_rows,
                          const torch::Tensor&                  expected_features,
                          const torch::Tensor&                  expected_deepstack) {
    ASSERT_TRUE(input.multimodal_features.has_value());
    ASSERT_TRUE(input.mm_extra_input.has_value());
    const auto& features = input.multimodal_features.value();
    const auto& extra    = input.mm_extra_input.value();
    ASSERT_FALSE(features.empty());
    ASSERT_EQ(features.size(), extra.size());
    ASSERT_EQ(input.mm_features_locs.numel(), features.size());
    ASSERT_EQ(input.mm_features_spans.dim(), 2);
    ASSERT_EQ(input.mm_features_spans.size(0), features.size());
    ASSERT_EQ(input.mm_features_spans.size(1), 3);
    const auto                     locs  = toVec<int>(input.mm_features_locs);
    const auto                     spans = toVec<int64_t>(input.mm_features_spans);
    vector<int>                    actual_locs;
    vector<pair<int64_t, int64_t>> actual_prompt_rows;
    vector<torch::Tensor>          feature_rows, deepstack_rows;
    for (size_t i = 0; i < features.size(); ++i) {
        const int64_t rows = features[i].size(0);
        ASSERT_GT(rows, 0);
        ASSERT_EQ(spans[i * 3 + 2] - spans[i * 3 + 1], rows);
        for (int64_t row = 0; row < rows; ++row) {
            actual_locs.push_back(locs[i] + row);
            actual_prompt_rows.emplace_back(spans[i * 3], spans[i * 3 + 1] + row);
        }
        feature_rows.push_back(features[i].cpu());
        deepstack_rows.push_back(extra[i].cpu().reshape({expected_deepstack.size(0), rows, expected_features.size(1)}));
    }
    EXPECT_EQ(actual_locs, expected_locs);
    EXPECT_EQ(actual_prompt_rows, expected_prompt_rows);
    EXPECT_TRUE(torch::equal(torch::cat(feature_rows), expected_features));
    EXPECT_TRUE(torch::equal(torch::cat(deepstack_rows, 1), expected_deepstack));
}

TEST_F(MtpBatchStreamProcessorTest, testChunkedPrefillDraftCarriesTheNextImageRow) {
    // The draft sees prompt[start + 1 : end + 1]. The appended prompt token
    // can be an image row even when the target chunk contains no image yet.
    constexpr int prompt_length = 12;
    constexpr int chunk_length  = 4;
    constexpr int hidden_size   = 4;
    // Host MROPE inputs exercise the fallback; device DEFAULT inputs also
    // exercise the CUDA shift when the image is outside the current window.
    for (const auto& [device_input, mrope] : {std::pair<bool, bool>{false, true}, {true, false}}) {
        autil::EnvGuard device_input_env("RTP_LLM_DEVICE_INPUT", device_input ? "1" : "0");
        for (const auto& [image_start, image_end] : {std::pair<int, int>{0, 1}, {0, 2}, {2, 9}, {4, 9}}) {
            SCOPED_TRACE(testing::Message() << "device_input=" << device_input << " mrope=" << mrope
                                            << " image_start=" << image_start << " image_end=" << image_end);
            const int   image_length = image_end - image_start;
            const int   factor       = mrope ? 3 : 1;
            ModelConfig model;
            model.max_seq_len                           = 64;
            model.vocab_size                            = 64;
            model.num_layers                            = 1;
            model.has_positional_encoding               = true;
            model.mm_model_config.is_multimodal         = true;
            model.mm_model_config.mm_position_ids_style = mrope ? MROPE : DEFAULT;
            model.attn_config.rope_config.index_factor  = factor;
            RuntimeConfig runtime;
            runtime.fifo_scheduler_config.prefill_chunk_size = chunk_length;
            ResourceContext resources;
            resources.role_type = RoleType::PDFUSION;

            auto query             = std::make_shared<GenerateInput>();
            query->generate_config = std::make_shared<GenerateConfig>();
            query->input_ids       = torch::arange(prompt_length, torch::kInt32);
            // Image tokens deliberately exceed the vocabulary. A lost
            // vision mask would send the draft to an invalid lookup.
            query->input_ids.slice(0, image_start, image_end).fill_(1000);
            query->text_tokens_mask = torch::ones({prompt_length}, torch::kInt32);
            query->text_tokens_mask->slice(0, image_start, image_end).zero_();
            query->mm_locs = torch::tensor({image_start}, torch::kInt32);
            auto full_feature =
                torch::arange(image_length * hidden_size, torch::kFloat32).reshape({image_length, hidden_size});
            auto full_extra =
                torch::arange(2 * image_length * hidden_size, torch::kFloat32).reshape({2, image_length, hidden_size});
            query->multimodal_features = std::vector<torch::Tensor>{full_feature};
            query->mm_extra_input      = std::vector<torch::Tensor>{full_extra.flatten()};
            auto stream = std::make_shared<NormalGenerateStream>(query, model, runtime, resources, nullptr);
            ASSERT_FALSE(stream->hasError()) << stream->stopReason();
            ASSERT_TRUE(stream->chunkedPrefillEnabled());
            stream->generate_status_->status = StreamState::RUNNING;
            stream->setNeedReleaseResource(false);
            // Use distinct full-prompt coordinates to catch a chunk-local
            // reset as well as an incorrect sampled-token tail position.
            auto full_positions =
                mrope ?
                    (torch::arange(prompt_length, torch::kInt32).unsqueeze(1) * 10 + torch::arange(3, torch::kInt32)) :
                    torch::arange(prompt_length, torch::kInt32);
            stream->setContextPositionIds(full_positions.flatten());

            SpeculativeExecutionConfig speculative;
            speculative.type              = SP_TYPE_MTP;
            speculative.gen_num_per_cycle = 1;
            MtpBatchStreamProcessor processor(
                model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, makeProcessorCacheConfig(), speculative, false);
            for (int chunk_start = 0; chunk_start < prompt_length; chunk_start += chunk_length) {
                SCOPED_TRACE(testing::Message() << "chunk_start=" << chunk_start);
                ASSERT_EQ(stream->reuseLength(), chunk_start);
                ASSERT_EQ(stream->contextLength(), chunk_length);
                const bool middle = chunk_start + chunk_length < prompt_length;
                ASSERT_EQ(stream->isMiddleChunk(), middle);
                StreamGroups groups({stream});
                TensorHolder holder;
                auto         gathered = processor.gatherModelInput(groups, holder);
                ASSERT_TRUE(gathered.ok()) << gathered.status();
                auto input = std::move(gathered.value());
                if (chunk_start == 0 && image_start == chunk_length) {
                    EXPECT_FALSE(input.multimodal_features.has_value() && !input.multimodal_features->empty());
                }
                GptModelOutputs output;
                output.all_hidden_states = torch::zeros({chunk_length, hidden_size});
                SamplerOutput sampled;
                sampled.token_ids = torch::tensor({{63}}, torch::kInt32).to(device_input ? torch::kCUDA : torch::kCPU);
                processor.updatePrefillPostDraftModelInput(groups, input, output, sampled, holder);

                const int shifted_start   = chunk_start + 1;
                const int shifted_end     = std::min(chunk_start + chunk_length + 1, prompt_length);
                auto      expected_tokens = query->input_ids.slice(0, shifted_start, shifted_end).clone();
                auto      expected_mask   = query->text_tokens_mask->slice(0, shifted_start, shifted_end).clone();
                if (!middle) {
                    expected_tokens = torch::cat({expected_tokens, torch::tensor({63}, torch::kInt32)});
                    expected_mask   = torch::cat({expected_mask, torch::ones({1}, torch::kInt32)});
                }
                EXPECT_EQ(toVec<int>(input.combo_tokens), toVec<int>(expected_tokens));
                EXPECT_EQ(toVec<int>(input.text_tokens_mask), toVec<int>(expected_mask));
                auto expected_positions =
                    full_positions.reshape({prompt_length, factor}).slice(0, shifted_start, shifted_end).clone();
                if (!middle) {
                    auto tail =
                        mrope ? torch::full({1, factor}, 112, torch::kInt32) : torch::tensor({{11}}, torch::kInt32);
                    expected_positions = torch::cat({expected_positions, tail}, 0);
                }
                EXPECT_EQ(toVec<int>(input.combo_position_ids), toVec<int>(expected_positions));

                const int overlap_start = std::max(shifted_start, image_start);
                const int overlap_end   = std::min(shifted_end, image_end);
                if (overlap_start < overlap_end) {
                    vector<int>                    expected_locs;
                    vector<pair<int64_t, int64_t>> expected_prompt_rows;
                    for (int position = overlap_start; position < overlap_end; ++position) {
                        expected_locs.push_back(position - shifted_start);
                        expected_prompt_rows.emplace_back(0, position);
                    }
                    expectMultimodalRows(input,
                                         expected_locs,
                                         expected_prompt_rows,
                                         full_feature.slice(0, overlap_start - image_start, overlap_end - image_start),
                                         full_extra.slice(1, overlap_start - image_start, overlap_end - image_start));
                } else {
                    EXPECT_TRUE(!input.multimodal_features.has_value() || input.multimodal_features->empty());
                    EXPECT_TRUE(!input.mm_extra_input.has_value() || input.mm_extra_input->empty());
                    EXPECT_EQ(input.mm_features_locs.numel(), 0);
                    EXPECT_EQ(input.mm_features_spans.numel(), 0);
                }
                EXPECT_TRUE(torch::equal(query->multimodal_features->at(0), full_feature));
                if (middle) {
                    stream->advanceChunk();
                }
            }
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testChunkedPrefillDraftKeepsMixedImageRowsInRequestOrder) {
    ModelConfig model;
    model.max_seq_len                   = 64;
    model.vocab_size                    = 64;
    model.num_layers                    = 1;
    model.mm_model_config.is_multimodal = true;
    RuntimeConfig runtime;
    runtime.fifo_scheduler_config.prefill_chunk_size = 2;
    ResourceContext resources;
    resources.role_type = RoleType::PDFUSION;
    auto make_stream    = [&](const vector<int>& tokens, int image_start, int image_length, int feature_base) {
        auto query             = std::make_shared<GenerateInput>();
        query->input_ids       = torch::tensor(tokens, torch::kInt32);
        query->generate_config = std::make_shared<GenerateConfig>();
        if (image_length > 0) {
            query->mm_locs          = torch::tensor({image_start}, torch::kInt32);
            query->text_tokens_mask = torch::ones({static_cast<int64_t>(tokens.size())}, torch::kInt32);
            query->text_tokens_mask->slice(0, image_start, image_start + image_length).zero_();
            query->multimodal_features = std::vector<torch::Tensor>{
                (torch::arange(image_length * 2, torch::kFloat32) + feature_base).reshape({image_length, 2})};
            query->mm_extra_input =
                std::vector<torch::Tensor>{torch::arange(2 * image_length * 2, torch::kFloat32) + feature_base + 1000};
        }
        auto stream = std::make_shared<NormalGenerateStream>(query, model, runtime, resources, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        stream->setNeedReleaseResource(false);
        return stream;
    };
    auto before_image = make_stream({1, 2, 1000, 1000, 1000, 1000}, 2, 4, 10);
    auto inside_image = make_stream({1000, 1000, 1000, 1000, 3, 4}, 0, 4, 20);
    auto final_text   = make_stream({5, 6}, 0, 0, 0);
    for (const auto& stream : {before_image, inside_image, final_text}) {
        ASSERT_FALSE(stream->hasError()) << stream->stopReason();
        ASSERT_TRUE(stream->chunkedPrefillEnabled());
    }
    ASSERT_TRUE(before_image->isMiddleChunk());
    ASSERT_TRUE(inside_image->isMiddleChunk());
    ASSERT_FALSE(final_text->isMiddleChunk());
    SpeculativeExecutionConfig speculative;
    speculative.type              = SP_TYPE_MTP;
    speculative.gen_num_per_cycle = 1;
    MtpBatchStreamProcessor processor(
        model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, makeProcessorCacheConfig(), speculative, false);
    StreamGroups groups({before_image, inside_image, final_text});
    TensorHolder holder;
    auto         gathered = processor.gatherModelInput(groups, holder);
    ASSERT_TRUE(gathered.ok()) << gathered.status();
    auto input = std::move(gathered.value());
    // Only request B has image rows in the target input. Request A's image
    // enters at the appended draft token, before B in packed input order.
    EXPECT_EQ(toVec<int>(input.mm_features_locs), (vector<int>{2}));
    GptModelOutputs output;
    output.all_hidden_states = torch::zeros({6, 2});
    SamplerOutput sampled;
    sampled.token_ids = torch::tensor({{63}, {63}, {63}}, torch::kInt32);
    processor.updatePrefillPostDraftModelInput(groups, input, output, sampled, holder);
    EXPECT_EQ(toVec<int>(input.combo_tokens), (vector<int>{2, 1000, 1000, 1000, 6, 63}));
    EXPECT_EQ(toVec<int>(input.text_tokens_mask), (vector<int>{1, 0, 0, 0, 1, 1}));
    expectMultimodalRows(
        input,
        {1, 2, 3},
        {{0, 2}, {1, 1}, {1, 2}},
        torch::tensor({10.f, 11.f, 22.f, 23.f, 24.f, 25.f}).reshape({3, 2}),
        torch::tensor({1010.f, 1011.f, 1022.f, 1023.f, 1024.f, 1025.f, 1018.f, 1019.f, 1030.f, 1031.f, 1032.f, 1033.f})
            .reshape({2, 3, 2}));
}

}  // namespace rtp_llm
