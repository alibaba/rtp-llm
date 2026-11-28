#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <limits>
#include <thread>
#include <stdexcept>
#include "torch/all.h"
#if USING_CUDA
#include "c10/cuda/CUDAGuard.h"
#include "c10/cuda/CUDAStream.h"
#endif
#include "gtest/gtest.h"
#include "autil/EnvUtil.h"

#if USING_CUDA
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#endif

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

#define private public
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#undef private
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/LinearBlocksUtil.h"

using namespace std;

namespace rtp_llm {

TEST(EngramHistoryTest, VerifyAndRejectedCandidatesDoNotPolluteNextHistory) {
    auto anchor   = torch::tensor({{10, 9, 8, 7}, {20, 19, 18, 17}}, torch::kInt32);
    auto verify   = torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kInt32);
    auto windows  = mtp::makeEngramVerifyWindows(anchor, verify);
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

// The ROCm compiler wrapper does not preserve -fno-access-control. Keep
// fixture-only stream state access in a derived test class on both platforms.
class ProcessorTestGenerateStream: public NormalGenerateStream {
public:
    using NormalGenerateStream::NormalGenerateStream;
    using GenerateStream::generate_status_;
    using GenerateStream::sampling_state_;
};

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
                                          bool                   return_logprobs       = false,
                                          int                    top_logprobs          = 0,
                                          bool                   in_think_mode         = false,
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
        query->generate_config->return_logprobs = return_logprobs;
        query->generate_config->top_logprobs    = top_logprobs;
        query->generate_config->in_think_mode   = in_think_mode || (!return_logprobs && !end_think_token_ids.empty());
        query->generate_config->num_return_sequences = num_return_sequences;
        auto stream =
            make_shared<ProcessorTestGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        if (!end_think_token_ids.empty()) {
            // Main's factory routes think constraints through grammar, so this
            // suite attaches the spec-verify think processor directly.
            auto think_processor = ThinkModeLogitsProcessor::fromGenerateInput(query, 1);
            if (think_processor != nullptr) {
                stream->sampling_state_.logits_processors.push_back(think_processor);
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
};

class TestableMtpBatchStreamProcessor: public MtpBatchStreamProcessor {
public:
    using MtpBatchStreamProcessor::MtpBatchStreamProcessor;
    using MtpBatchStreamProcessor::overlayMtpCacheSnapshots;
    using MtpBatchStreamProcessor::prepareDecodeSpecUpdateInfo;
    using MtpBatchStreamProcessor::preparePrefillSpecUpdateInfo;
};

#if USING_CUDA
namespace {

struct AsyncSamplerInputCase {
    std::unique_ptr<MtpBatchStreamProcessor> processor;
    std::list<GenerateStreamPtr>             streams;
    GptModelOutputs                          model_output;
    torch::Tensor                            draft_tokens;
};

AsyncSamplerInputCase makeAsyncSamplerInputCase(MtpBatchStreamProcessorTest& fixture) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_config;
    ResourceContext             resource_context;
    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 64;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 3;
    sp_config.type              = SP_TYPE_DSPARK;

    AsyncSamplerInputCase data;
    data.processor = std::make_unique<MtpBatchStreamProcessor>(
        model_config, pd_sep_config, profiling_config, fixture.makeProcessorCacheConfig(), sp_config, false);
    data.streams.push_back(fixture.createContextStream(model_config, runtime_config, resource_context, {5, 6, 7}, 1));
    data.streams.push_back(
        fixture.createContextStream(model_config, runtime_config, resource_context, {8, 9, 10, 11, 12}, 2));
    for (auto& stream : data.streams) {
        stream->setScoreLen(4);
        // Keep a history reader active to exercise the pinned snapshot and device history lifetime.
        stream->generateConfig()->repetition_penalty = 1.05f;
    }
    data.model_output.logits = torch::zeros({8, 64}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
    data.draft_tokens        = torch::full({2, 3}, 21, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    return data;
}

SamplerInputs gatherAsyncSamplerInput(AsyncSamplerInputCase& data) {
    auto result =
        data.processor->gatherSpecSamplerInput(StreamGroups(data.streams), data.model_output, data.draft_tokens);
    RTP_LLM_CHECK_WITH_INFO(result.ok(), "sampler input gather failed: %s", result.status().ToString().c_str());
    EXPECT_EQ(result.value().force_topk, 0);
    return std::move(result.value());
}

void expectAsyncSamplerHistory(const torch::Tensor&                 history,
                               const std::vector<std::vector<int>>& expected_prefixes,
                               int                                  proposal_token) {
    auto cpu = history.cpu();
    ASSERT_EQ(cpu.size(0), 8);
    ASSERT_EQ(cpu.size(1), 9);
    const auto* values = cpu.data_ptr<int32_t>();
    for (int batch = 0; batch < 2; ++batch) {
        const auto& prefix = expected_prefixes[batch];
        for (int position = 0; position < 4; ++position) {
            const auto* row = values + (batch * 4 + position) * cpu.size(1);
            for (size_t token = 0; token < prefix.size(); ++token) {
                EXPECT_EQ(row[token], prefix[token]);
            }
            for (int proposal = 0; proposal < 3; ++proposal) {
                EXPECT_EQ(row[prefix.size() + proposal], proposal_token);
            }
        }
    }
    // Columns beyond the populated history/proposals intentionally remain
    // uninitialized and are excluded from numerical comparisons.
}

}  // namespace

TEST_F(MtpBatchStreamProcessorTest, SamplerInputOwnsExpandedHistorySnapshot) {
    auto                         stream = cuda_graph::graphGetStreamFromPool(false);
    cuda_graph::GraphStreamGuard guard(stream);
    auto                         data   = makeAsyncSamplerInputCase(*this);
    auto                         inputs = gatherAsyncSamplerInput(data);
    ASSERT_TRUE(inputs.token_ids.is_cuda());
    ASSERT_EQ(inputs.token_ids.size(0), 8);
    for (auto& request : data.streams)
        request->completeTokenIds().fill_(-99);
    expectAsyncSamplerHistory(inputs.token_ids, {{5, 6, 7}, {8, 9, 10, 11, 12}}, 21);
}

TEST_F(MtpBatchStreamProcessorTest, SamplerInputsOwnHistoryStorageAcrossRounds) {
    auto                         stream = cuda_graph::graphGetStreamFromPool(false);
    cuda_graph::GraphStreamGuard guard(stream);
    auto                         data = makeAsyncSamplerInputCase(*this);
    std::vector<torch::Tensor>   histories;
    for (int round = 0; round < 4; ++round) {
        for (auto& request : data.streams)
            request->completeTokenIds().fill_(5 + round);
        data.draft_tokens.fill_(21 + round);
        auto inputs = gatherAsyncSamplerInput(data);
        histories.push_back(inputs.token_ids);
    }
    for (int round = 0; round < 4; ++round) {
        expectAsyncSamplerHistory(
            histories[round], {std::vector<int>(3, 5 + round), std::vector<int>(5, 5 + round)}, 21 + round);
    }
}

#endif

static void applySamplerMask(SamplerInputs& inputs, const SpecLogitsVerifyRunner::LaunchResult& mask) {
    SpecLogitsVerifyRunner::applyMaskToLogits(inputs.logits,
                                              mask,
                                              inputs.vocab_size,
                                              inputs.row_mapping ? inputs.row_mapping->dense_to_packed :
                                                                   torch::Tensor());
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
    // Position-specific committed history is only materialized when a reader
    // is active; a non-identity repetition penalty keeps this case on the
    // full-upload path instead of the dead-history elision.
    stream1->generateConfig()->repetition_penalty = 1.05f;
    stream2->generateConfig()->repetition_penalty = 1.05f;

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

    auto draft_token_ids       = torch::tensor({1, 2, 3, 3, 2, 1}, torch::kInt32).reshape({2, 3});
    auto sampler_inputs_status = processor.gatherSpecSamplerInput(stream_groups, model_output, draft_token_ids);
    ASSERT_TRUE(sampler_inputs_status.ok());

    const auto& sampler_inputs = sampler_inputs_status.value();
    EXPECT_EQ((std::vector<int64_t>{8, 4}), sampler_inputs.logits.sizes().vec());
    EXPECT_EQ(sampler_inputs.probability_output, SamplerProbabilityOutput::DISTRIBUTION);
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

// The gather exposes the per-round logits view; Sampler owns its preparation
// and chooses the probability representation.
TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputAliasesLogitsWhenNoWriterActive) {
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

    // Mirrors the production decode request: do_sample stays at its default
    // true, top_k=1 selects greedy, penalties stay at identity defaults.
    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {5}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {6, 7}, 2);
    for (const auto& stream : {stream1, stream2}) {
        stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
        stream->generateConfig()->top_k = 1;
    }

    auto stream_groups = StreamGroups({stream1, stream2});
    auto processor     = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelOutputs model_output;
    model_output.logits =
        torch::arange(32, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({8, 4});
    auto draft_token_ids = torch::tensor({1, 2, 3, 3, 2, 1}, torch::kInt32).reshape({2, 3});

    auto sampler_inputs_status = processor.gatherSpecSamplerInput(stream_groups, model_output, draft_token_ids);
    ASSERT_TRUE(sampler_inputs_status.ok());
    const auto& sampler_inputs = sampler_inputs_status.value();

    EXPECT_EQ((std::vector<int64_t>{8, 4}), sampler_inputs.logits.sizes().vec());
    EXPECT_TRUE(sampler_inputs.logits.is_contiguous());
    // The elided clone: the sampler consumes the model-output storage itself.
    EXPECT_EQ(model_output.logits.data_ptr(), sampler_inputs.logits.data_ptr());
    EXPECT_TRUE(torch::equal(model_output.logits, sampler_inputs.logits));
    // The caller requests a distribution; forward chooses its representation.
    EXPECT_TRUE(sampler_inputs.skip_logits_probs_restore);
    EXPECT_EQ(sampler_inputs.probability_output, SamplerProbabilityOutput::DISTRIBUTION);
#if USING_CUDA
    // Follow the real gather -> sampler path: unread history can remain
    // uninitialized, but every final-column token must be defined.
    Sampler sampler({8});
    auto    original_logits = model_output.logits.clone();
    auto    output          = sampler.forward(sampler_inputs);
    EXPECT_TRUE(output.token_ids_are_point_mass);
    EXPECT_TRUE(output.token_ids.select(1, output.token_ids.size(1) - 1).eq(3).all().item<bool>());
    EXPECT_TRUE(torch::equal(model_output.logits, original_logits));
#endif
}

TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputConsumesLogitsWithTemperatureAndPadding) {
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

    auto draft_token_ids = torch::tensor({1, 2, 3, 3, 2, 1}, torch::kInt32).reshape({2, 3});

    // Temperature may modify the shared per-round storage.
    {
        ResourceContext   resource_context;
        GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {5}, 1);
        GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {6, 7}, 2);
        for (const auto& stream : {stream1, stream2}) {
            stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
            stream->generateConfig()->top_k = 1;
        }
        stream1->generateConfig()->temperature = 0.5f;

        auto stream_groups = StreamGroups({stream1, stream2});
        auto processor     = MtpBatchStreamProcessor(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

        GptModelOutputs model_output;
        model_output.logits =
            torch::arange(32, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({8, 4});

        auto status = processor.gatherSpecSamplerInput(stream_groups, model_output, draft_token_ids);
        ASSERT_TRUE(status.ok());
        const auto& sampler_inputs = status.value();
        EXPECT_EQ(sampler_inputs.probability_output, SamplerProbabilityOutput::DISTRIBUTION);
        EXPECT_EQ(model_output.logits.data_ptr(), sampler_inputs.logits.data_ptr());
        EXPECT_TRUE(torch::equal(model_output.logits, sampler_inputs.logits));
        Sampler sampler({8});
        auto    output = sampler.forward(sampler_inputs);
        EXPECT_EQ(output.token_ids_are_point_mass, enableCudaSampling());
        EXPECT_TRUE(output.token_ids.select(1, sampler_inputs.step).eq(3).all().item<bool>());
    }

    // Padding is cropped with a view; the sampler chooses a compatible kernel.
    {
        ResourceContext   resource_context;
        GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {5}, 1);
        GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {6, 7}, 2);
        for (const auto& stream : {stream1, stream2}) {
            stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
            stream->generateConfig()->top_k = 1;
        }

        auto stream_groups = StreamGroups({stream1, stream2});
        auto processor     = MtpBatchStreamProcessor(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

        GptModelOutputs model_output;
        model_output.logits =
            torch::arange(56, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({8, 7});

        auto status = processor.gatherSpecSamplerInput(stream_groups, model_output, draft_token_ids);
        ASSERT_TRUE(status.ok());
        const auto& sampler_inputs = status.value();
        EXPECT_EQ((std::vector<int64_t>{8, 4}), sampler_inputs.logits.sizes().vec());
        EXPECT_EQ(model_output.logits.data_ptr(), sampler_inputs.logits.data_ptr());
        EXPECT_FALSE(sampler_inputs.logits.is_contiguous());
        EXPECT_TRUE(torch::equal(model_output.logits.narrow(1, 0, 4), sampler_inputs.logits));
    }
}

// Speculative masks do not read sampler history; penalties still do.
TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputSkipsMaskedHistoryUnlessPenaltyNeedsIt) {
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

    for (bool with_penalty : {false, true}) {
        SCOPED_TRACE(with_penalty);
        ResourceContext   resource_context;
        GenerateStreamPtr stream =
            createContextStream(model_config, runtime_config, resource_context, {1, 2}, 1, {7}, {8, 9});
        stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
        stream->generateConfig()->top_k            = 1;
        stream->generateConfig()->presence_penalty = with_penalty ? 0.5f : 0.0f;

        MtpBatchStreamProcessor processor(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
        StreamGroups stream_groups({stream});

        GptModelOutputs model_output;
        model_output.logits = torch::zeros({3, 16}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

        model_output.logits.select(1, 7).fill_(2.0f);
        model_output.logits.select(1, 3).fill_(1.0f);

        SpecLogitsVerifyRunner             runner;
        SpecLogitsVerifyRunner::LaunchTask task;
        task.total_streams = 1;
        task.propose_step  = static_cast<int>(sp_config.gen_num_per_cycle);
        task.vocab_size    = model_config.vocab_size;
        task.draft_tokens  = torch::tensor(std::vector<int32_t>{1, 2}, torch::kInt32).reshape({1, 2});
        for (const auto& processor_ptr : stream->getAllLogitsProcessorPtr()) {
            ASSERT_NE(processor_ptr, nullptr);
            task.active.push_back({processor_ptr, /*stream_idx=*/0});
        }
        ASSERT_EQ(1u, task.active.size());
        auto spec_result = runner.run(task);
        ASSERT_TRUE(spec_result.has_active_processor);

        auto status = processor.gatherSpecSamplerInput(stream_groups, model_output, task.draft_tokens);
        ASSERT_TRUE(status.ok());
        auto& sampler_inputs = status.value();
        applySamplerMask(sampler_inputs, spec_result);
        Sampler sampler({3});
        auto    output = sampler.forward(sampler_inputs);

#if USING_CUDA
        EXPECT_EQ(sampler_inputs.token_ids.defined(), with_penalty);
#endif
        if (with_penalty) {
            auto history = output.token_ids.narrow(1, 0, 4).cpu();
            EXPECT_TRUE(
                torch::equal(history, torch::tensor({{1, 2, 1, 2}, {1, 2, 1, 2}, {1, 2, 1, 2}}, torch::kInt32)));
        }

        // No clone is needed to preserve raw logits for logprobs.
        EXPECT_EQ(model_output.logits.data_ptr(), sampler_inputs.logits.data_ptr());
        EXPECT_TRUE(torch::equal(model_output.logits, sampler_inputs.logits));
        const float neg_inf = -std::numeric_limits<float>::max();
        for (int i = 0; i < 3; ++i) {
            EXPECT_EQ(neg_inf, sampler_inputs.logits[i][7].item<float>());
        }
#if USING_CUDA
        EXPECT_TRUE(output.token_ids.select(1, output.token_ids.size(1) - 1).eq(3).all().item<bool>());
#endif
    }
}

// Dead sampler-input elision: with production-equivalent defaults
// (return_all_probs DEFAULT as the MTP executor sets it, identity penalties,
// no ngram ban, single beam) no consumer reads the history columns on device.
// The gather leaves history undefined, and Sampler allocates the output matrix;
// unread history columns are never asserted here.
TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputSkipsHistoryMaterializationWhenNoReaderActive) {
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
    for (auto& stream : {stream1, stream2}) {
        stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
    }

    auto processor = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelOutputs model_output;
    model_output.logits = torch::zeros({8, 7}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto gather_and_check_dead =
        [&](const std::list<GenerateStreamPtr>& streams, const torch::Tensor& draft, int64_t expected_width) {
            auto status = processor.gatherSpecSamplerInput(StreamGroups(streams), model_output, draft);
            ASSERT_TRUE(status.ok());
            const auto& sampler_inputs = status.value();
            EXPECT_FALSE(sampler_inputs.token_ids.defined());
            Sampler sampler({});
            auto    output = sampler.forward(sampler_inputs);
            EXPECT_EQ(output.token_ids.sizes().vec(),
                      (std::vector<int64_t>{static_cast<int64_t>(streams.size()) * 4, expected_width}));
        };

    // Round 1: gate-on from a cold processor.
    auto draft1 = torch::tensor({1, 2, 3, 3, 2, 1}, torch::kInt32).reshape({2, 3});
    gather_and_check_dead({stream1, stream2}, draft1, /*expected_width=*/6);

    // Commit accepted tokens; the host stream state is the only truth the
    // next materialized round will read (the exposed matrix never feeds it).
    stream1->update({torch::tensor({1, 2}, torch::kInt32).reshape({1, 2}),
                     2,
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor()});
    stream2->update({torch::tensor({3}, torch::kInt32).reshape({1, 1}),
                     1,
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor(),
                     torch::Tensor()});

    // Round 2: gate-on again with grown lengths.
    auto draft2 = torch::tensor({2, 3, 1, 1, 3, 2}, torch::kInt32).reshape({2, 3});
    gather_and_check_dead({stream1, stream2}, draft2, /*expected_width=*/7);
}

// Each reader-side gate condition on its own must restore the materialized
// full-upload path with the current committed history and proposals.
TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputDeadHistoryGateFlipsRestoreMaterialization) {
    auto run_flip_case = [&](const char* name, const auto& tweak, bool expect_all_probs) {
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
        for (auto& stream : {stream1, stream2}) {
            stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
            tweak(stream);
        }

        auto processor = MtpBatchStreamProcessor(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

        GptModelOutputs model_output;
        model_output.logits = torch::zeros({8, 7}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto draft          = torch::tensor({1, 2, 3, 3, 2, 1}, torch::kInt32).reshape({2, 3});

        auto status = processor.gatherSpecSamplerInput(StreamGroups({stream1, stream2}), model_output, draft);
        ASSERT_TRUE(status.ok()) << name;
        const auto& sampler_inputs = status.value();
        EXPECT_EQ(sampler_inputs.probability_output != SamplerProbabilityOutput::NONE, expect_all_probs) << name;
        Sampler     sampler({});
        auto        cpu    = sampler.forward(sampler_inputs).token_ids.cpu();
        const auto* values = cpu.data_ptr<int32_t>();
        const auto  width  = cpu.size(1);
        ASSERT_EQ(width, 6) << name;
        const std::vector<int> prefix1 = {5, 1, 2, 3};
        const std::vector<int> prefix2 = {6, 7, 3, 2, 1};
        for (int64_t row = 0; row < 4; ++row) {
            for (size_t col = 0; col < prefix1.size(); ++col) {
                EXPECT_EQ(values[row * width + col], prefix1[col]) << name << " row=" << row << " col=" << col;
            }
        }
        for (int64_t row = 4; row < 8; ++row) {
            for (size_t col = 0; col < prefix2.size(); ++col) {
                EXPECT_EQ(values[row * width + col], prefix2[col]) << name << " row=" << row << " col=" << col;
            }
        }
    };

    run_flip_case(
        "repetition_penalty",
        [](const GenerateStreamPtr& s) { s->generateConfig()->repetition_penalty = 1.05f; },
        /*expect_all_probs=*/true);
    run_flip_case(
        "presence_penalty",
        [](const GenerateStreamPtr& s) { s->generateConfig()->presence_penalty = 0.5f; },
        /*expect_all_probs=*/true);
    run_flip_case(
        "frequency_penalty",
        [](const GenerateStreamPtr& s) { s->generateConfig()->frequency_penalty = 0.25f; },
        /*expect_all_probs=*/true);
    run_flip_case(
        "no_repeat_ngram_size",
        [](const GenerateStreamPtr& s) { s->generateConfig()->no_repeat_ngram_size = 3; },
        /*expect_all_probs=*/true);
    run_flip_case(
        "penalty_with_return_all_probs_none",
        [](const GenerateStreamPtr& s) {
            s->setReturnAllProbs(ReturnAllProbsMode::NONE);
            s->generateConfig()->presence_penalty = 0.5f;
        },
        /*expect_all_probs=*/true);
}

// Re-enabling a history reader must upload the current host history in full,
// including commits made during elided rounds and replacement requests.
TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputHistoryResumesAfterDeadHistoryRounds) {
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
    for (auto& stream : {stream1, stream2}) {
        stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
        stream->generateConfig()->repetition_penalty = 1.05f;
    }

    auto processor = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelOutputs model_output;
    model_output.logits = torch::zeros({8, 7}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // The single-block fixture KV resource reports an error from
    // updateKvCacheBlocks once a stream's total tokens exceed the block
    // capacity; the token history itself commits before that step. Force the
    // update so every round's append-only commit lands and the multi-round
    // history contract stays testable. Tokens must stay in-vocab
    // (vocab_size=4): update() rejects out-of-vocab ids.
    auto commit = [](const GenerateStreamPtr& stream, const std::vector<int32_t>& tokens) {
        stream->update({torch::tensor(tokens, torch::kInt32).reshape({1, static_cast<int64_t>(tokens.size())}),
                        static_cast<int>(tokens.size()),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        torch::Tensor(),
                        /*update_remote_generate=*/true,
                        /*force_update_info=*/true});
    };

    // Round 1: materialized history with sequence lengths {1, 2}.
    auto draft1 = torch::tensor({1, 2, 3, 3, 2, 1}, torch::kInt32).reshape({2, 3});
    {
        auto status = processor.gatherSpecSamplerInput(StreamGroups({stream1, stream2}), model_output, draft1);
        ASSERT_TRUE(status.ok());
        Sampler sampler({});
        auto    cpu = sampler.forward(status.value()).token_ids.cpu();
        ASSERT_EQ(cpu.size(1), 6);
        EXPECT_EQ(cpu[0][0].item<int32_t>(), 5);
        EXPECT_EQ(cpu[4][0].item<int32_t>(), 6);
        EXPECT_EQ(cpu[4][1].item<int32_t>(), 7);
    }
    commit(stream1, {1, 2});  // seq 3
    commit(stream2, {3});     // seq 3

    // Round 2 (dead sampler input): penalties identity -> gate on. The
    // exposed matrix's column contents are don't-care (history AND proposal
    // materialization elided; only geometry is defined).
    for (auto& stream : {stream1, stream2}) {
        stream->generateConfig()->repetition_penalty = 1.0f;
    }
    auto draft2 = torch::tensor({2, 3, 1, 1, 3, 2}, torch::kInt32).reshape({2, 3});
    {
        auto status = processor.gatherSpecSamplerInput(StreamGroups({stream1, stream2}), model_output, draft2);
        ASSERT_TRUE(status.ok());
        EXPECT_FALSE(status.value().token_ids.defined());
        Sampler sampler({});
        auto    cpu = sampler.forward(status.value()).token_ids.cpu();
        ASSERT_EQ(cpu.sizes().vec(), (std::vector<int64_t>{8, 7}));
    }
    commit(stream1, {3});     // seq 4
    commit(stream2, {1, 2});  // seq 5

    // Round 3: the full upload includes every newly committed column.
    for (auto& stream : {stream1, stream2}) {
        stream->generateConfig()->repetition_penalty = 1.05f;
    }
    auto draft3 = torch::tensor({0, 1, 2, 3, 0, 1}, torch::kInt32).reshape({2, 3});
    {
        auto status = processor.gatherSpecSamplerInput(StreamGroups({stream1, stream2}), model_output, draft3);
        ASSERT_TRUE(status.ok());
        Sampler sampler({});
        auto    cpu = sampler.forward(status.value()).token_ids.cpu();
        ASSERT_EQ(cpu.size(1), 9);
        const std::vector<int> prefix1 = {5, 1, 2, 3, 0, 1, 2};
        const std::vector<int> prefix2 = {6, 7, 3, 1, 2, 3, 0, 1};
        for (int64_t row = 0; row < 4; ++row) {
            for (size_t col = 0; col < prefix1.size(); ++col) {
                EXPECT_EQ(cpu[row][col].item<int32_t>(), prefix1[col]) << "row=" << row << " col=" << col;
            }
        }
        for (int64_t row = 4; row < 8; ++row) {
            for (size_t col = 0; col < prefix2.size(); ++col) {
                EXPECT_EQ(cpu[row][col].item<int32_t>(), prefix2[col]) << "row=" << row << " col=" << col;
            }
        }
    }

    // Round 4 (dead sampler input) with a composition change: stream2
    // leaves and a new request reuses its ID. Only geometry is defined.
    GenerateStreamPtr stream3 = createContextStream(model_config, runtime_config, resource_context, {2, 1}, 2);
    stream3->setScoreLen(sp_config.gen_num_per_cycle + 1);
    stream1->generateConfig()->repetition_penalty = 1.0f;
    auto draft4                                   = torch::tensor({1, 1, 2, 2, 3, 3}, torch::kInt32).reshape({2, 3});
    {
        auto status = processor.gatherSpecSamplerInput(StreamGroups({stream1, stream3}), model_output, draft4);
        ASSERT_TRUE(status.ok());
        Sampler sampler({});
        auto    cpu = sampler.forward(status.value()).token_ids.cpu();
        ASSERT_EQ(cpu.sizes().vec(), (std::vector<int64_t>{8, 8}));
    }

    // Round 5: the replacement request must get its own complete history.
    stream1->generateConfig()->repetition_penalty = 1.05f;
    stream3->generateConfig()->repetition_penalty = 1.05f;
    auto draft5                                   = torch::tensor({3, 2, 1, 1, 2, 3}, torch::kInt32).reshape({2, 3});
    {
        auto status = processor.gatherSpecSamplerInput(StreamGroups({stream1, stream3}), model_output, draft5);
        ASSERT_TRUE(status.ok());
        Sampler sampler({});
        auto    cpu = sampler.forward(status.value()).token_ids.cpu();
        ASSERT_EQ(cpu.size(1), 8);
        const std::vector<int> prefix1 = {5, 1, 2, 3, 3, 2, 1};
        const std::vector<int> prefix3 = {2, 1, 1, 2, 3};
        for (int64_t row = 0; row < 4; ++row) {
            for (size_t col = 0; col < prefix1.size(); ++col) {
                EXPECT_EQ(cpu[row][col].item<int32_t>(), prefix1[col]) << "row=" << row << " col=" << col;
            }
        }
        for (int64_t row = 4; row < 8; ++row) {
            for (size_t col = 0; col < prefix3.size(); ++col) {
                EXPECT_EQ(cpu[row][col].item<int32_t>(), prefix3[col]) << "row=" << row << " col=" << col;
            }
        }
    }
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
    model_output.logits = torch::zeros({3, 16}, torch::TensorOptions().device(torch::kCUDA));

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

    auto sampler_inputs_status = processor.gatherSpecSamplerInput(stream_groups, model_output, task.draft_tokens);
    ASSERT_TRUE(sampler_inputs_status.ok());
    auto sampler_inputs = sampler_inputs_status.value();
    applySamplerMask(sampler_inputs, spec_result);
    sampler_inputs.top_k.fill_(1);
    Sampler sampler({});
    sampler.forward(sampler_inputs);

    // Constrained/think processors ride the SpecLogitsVerifyRunner packed
    // allow-mask, which the sampler applies to the verify logits inline; the
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
    target_output.sampler_output.token_ids =
        torch::tensor({2, -1, 1, 1, 2, 3}, torch::kInt32).reshape({2, 3}).to(torch::kCUDA);
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
    class TrackingUpdateProcessor: public BaseLogitsProcessor {
    public:
        std::optional<ErrorInfo> process(const SamplerInputs&, size_t, size_t) override {
            return std::nullopt;
        }
        void                     updateMultiSeqStatus(const std::vector<int>&) override {}
        std::optional<ErrorInfo> updateStatus(const torch::Tensor&, int32_t count) override {
            accepted_tokens += count;
            return std::nullopt;
        }
        int accepted_tokens = 0;
    };
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
    stream1->setIsContextStream(false);
    stream2->setIsContextStream(false);
    auto logits_processor = std::make_shared<TrackingUpdateProcessor>();
    stream1->sampling_state_.logits_processors.push_back(logits_processor);

    auto stream_groups = StreamGroups({stream1, stream2});

    speculative::SpeculativeSamplerOutput spec_decode_output;
    spec_decode_output.accept_len_cpu             = torch::tensor({5, 1}, torch::kInt32);
    spec_decode_output.verified_draft_lengths_cpu = torch::tensor({4, 0}, torch::kInt32);
    spec_decode_output.accept_tokens_cpu          = torch::tensor({{2, 3, 1, 3, 2}, {2, 0, 0, 0, 0}}, torch::kInt32);
    spec_decode_output.accept_len                 = spec_decode_output.accept_len_cpu.to(torch::kCUDA);
    spec_decode_output.accept_tokens              = spec_decode_output.accept_tokens_cpu.to(torch::kCUDA);

    MergedOutput draft_prefill_output;
    draft_prefill_output.model_output.all_hidden_states =
        torch::tensor({0.2f, 0.02f, 0.3f, 0.03f, 0.4f, 0.04f, 0.5f, 0.05f, 0.6f, 0.06f, 1.3f, 0.13f}, torch::kFloat32)
            .reshape({6, 2});
    draft_prefill_output.sampler_output.token_ids = torch::tensor({0L, 3L}, torch::kInt64).reshape({2, 1});
    draft_prefill_output.sampler_output.all_probs =
        torch::tensor({0.2f, 0.1f, 0.3f, 0.5f, 0.3f, 0.1f, 0.4f, 0.2f}, torch::kFloat32).reshape({2, 4});

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    auto status = processor.dispatchDecode(stream_groups, spec_decode_output, std::move(draft_prefill_output));
    EXPECT_TRUE(status.ok());

    checkOutput(stream1, {1, 2, 3, 1, 3, 2}, {2, 0}, {0.2, 0.1, 0.3, 0.5}, {0.6, 0.06});
    checkOutput(stream2, {2, 1, 2}, {2, 3}, {0.3, 0.1, 0.4, 0.2}, {1.3, 0.13});
    EXPECT_EQ(logits_processor->accepted_tokens, 5);
    EXPECT_EQ(stream1->speculativeVerifiedDraftTokens(), 4);
    EXPECT_EQ(stream2->speculativeVerifiedDraftTokens(), 0);
    // Device-state real_seq_len publication moved to the executor layer
    // (MtpExecutor::publishSyncMtpDeviceState); dispatchDecode itself only
    // performs host bookkeeping now, so no device-state asserts here.
}

#if USING_CUDA
namespace {
// Same bounded host-callback gate as the executor tests. The callback owns its
// state until completion; release is idempotent and does not call CUDA APIs.
class ScopedSamplerStreamGate {
public:
    explicit ScopedSamplerStreamGate(cuda_graph::GraphStream stream): state_(std::make_shared<State>()) {
        auto*      callback_state = new std::shared_ptr<State>(state_);
        const auto status         = cudaLaunchHostFunc(
            stream.stream(),
            [](void* opaque) {
                std::unique_ptr<std::shared_ptr<State>> owner(static_cast<std::shared_ptr<State>*>(opaque));
                auto&                                   state = **owner;
                std::unique_lock<std::mutex>            lock(state.mutex);
                if (!state.cv.wait_for(lock, std::chrono::seconds(5), [&] { return state.released; })) {
                    state.timed_out.store(true);
                }
            },
            callback_state);
        if (status != cudaSuccess) {
            delete callback_state;
            throw std::runtime_error(cudaGetErrorString(status));
        }
    }

    ~ScopedSamplerStreamGate() {
        release();
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->released = true;
        }
        state_->cv.notify_all();
    }

    bool timedOut() const {
        return state_->timed_out.load();
    }

private:
    struct State {
        std::mutex              mutex;
        std::condition_variable cv;
        bool                    released = false;
        std::atomic<bool>       timed_out{false};
    };
    std::shared_ptr<State> state_;
};
}  // namespace

TEST_F(MtpBatchStreamProcessorTest, DecodeDispatchWaitsForFinalDraftBeforeUpdatingStream) {
    class TrackingUpdateProcessor: public BaseLogitsProcessor {
    public:
        std::shared_ptr<torch::Event> writer;
        int                           accepted_tokens{0};
        bool                          updated_before_writer{false};
        std::optional<ErrorInfo>      process(const SamplerInputs&, size_t, size_t) override {
            return std::nullopt;
        }
        void                     updateMultiSeqStatus(const std::vector<int>&) override {}
        std::optional<ErrorInfo> updateStatus(const torch::Tensor&, int32_t count) override {
            updated_before_writer |= !writer->query();
            accepted_tokens += count;
            return std::nullopt;
        }
    };
    ModelConfig model;
    model.max_seq_len = 128;
    model.vocab_size  = 8;
    model.num_layers  = 1;
    SpeculativeExecutionConfig sp;
    sp.type               = SP_TYPE_DSPARK;
    sp.gen_num_per_cycle  = 5;
    auto            cache = makeProcessorCacheConfig();
    ResourceContext resources;
    resources.cache_manager = std::make_shared<KVCacheManager>(cache);
    auto stream             = createContextStream(model, RuntimeConfig{}, resources, {1}, /*block_id=*/1);
    stream->generateConfig()->max_new_tokens = 8;
    stream->setIsContextStream(false);
    StreamGroups                          groups({stream});
    MtpBatchStreamProcessor               processor(model, {}, {}, cache, sp, false);
    speculative::SpeculativeSamplerOutput result;
    auto                                  tracker = std::make_shared<TrackingUpdateProcessor>();
    tracker->writer                               = result.transfer_done_event;
    stream->sampling_state_.logits_processors.push_back(tracker);

    // Poison the host views until a later final-draft event. A missing processor
    // wait must consume invalid/stale data or update before the event is ready.
    auto host = torch::full({8}, -1, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    auto device =
        torch::tensor({2, 2, 3, 0, 0, 0, 0, 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    result.accept_len_cpu             = host.narrow(0, 0, 1);
    result.accept_tokens_cpu          = host.narrow(0, 1, 6).reshape({1, 6});
    result.verified_draft_lengths_cpu = host.narrow(0, 7, 1);
    result.accept_len                 = device.narrow(0, 0, 1);
    result.accept_tokens              = device.narrow(0, 1, 6).reshape({1, 6});
    cuda_graph::graphGetCurrentStream().synchronize();
    auto                    producer = cuda_graph::graphGetStreamFromPool(false);
    ScopedSamplerStreamGate gate(producer);
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    dispatch_finished = false;
    std::thread             release_gate;
    std::exception_ptr      failure;
    absl::Status            status;
    try {
        {
            cuda_graph::GraphStreamGuard guard(producer);
            host.copy_(device, /*non_blocking=*/true);
            result.transfer_done_event->record(producer);
        }
        release_gate = std::thread([&] {
            std::unique_lock<std::mutex> lock(mutex);
            // Only a fuse for the intentionally blocked writer; elapsed time
            // is not an assertion or evidence of physical overlap/performance.
            cv.wait_for(lock, std::chrono::milliseconds(200), [&] { return dispatch_finished; });
            lock.unlock();
            gate.release();
        });
        status       = processor.dispatchDecode(groups, result, MergedOutput{});
    } catch (...) {
        failure = std::current_exception();
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        dispatch_finished = true;
    }
    cv.notify_all();
    gate.release();
    if (release_gate.joinable())
        release_gate.join();
    producer.synchronize();  // Test writer quiescence before any host/tensor teardown.
    if (failure)
        std::rethrow_exception(failure);
    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(gate.timedOut());
    EXPECT_FALSE(tracker->updated_before_writer);
    EXPECT_EQ(tracker->accepted_tokens, 2);
    EXPECT_EQ(stream->seqLength(), 3);
    EXPECT_EQ(stream->speculativeVerifiedDraftTokens(), 1);
}
#endif

TEST_F(MtpBatchStreamProcessorTest, DecodeVerifiedDraftLengthsRequireHostPayloadOrKnownDenseWidth) {
    ModelConfig model;
    model.max_seq_len = 128;
    model.vocab_size  = 8;
    model.num_layers  = 1;
    SpeculativeExecutionConfig sp;
    sp.type              = SP_TYPE_DSPARK;
    sp.gen_num_per_cycle = 5;
    MtpBatchStreamProcessor processor(model, {}, {}, makeProcessorCacheConfig(), sp, false);
    auto                    stream = createContextStream(model, RuntimeConfig{}, ResourceContext{}, {2}, 1);
    StreamGroups            groups({stream});
    speculative::SpeculativeSamplerOutput result;
    result.accept_len_cpu    = torch::ones({1}, torch::kInt32);
    result.accept_tokens_cpu = torch::zeros({1, 6}, torch::kInt32);
    MergedOutput      draft;
    MtpTargetLogprobs logprobs;
    for (const int known_width : {-1, 5}) {
        result.verified_draft_length = known_width;
        std::vector<StreamSpecUpdateInfo> updates;
        processor.prepareDecodeSpecUpdateInfo(groups, result, draft, logprobs, updates);
        ASSERT_EQ(updates.size(), 1);
        EXPECT_EQ(updates[0].verified_draft_tokens, known_width);
    }
    result.verified_draft_lengths_cpu = torch::zeros({1}, torch::kInt32);
    std::vector<StreamSpecUpdateInfo> updates;
    processor.prepareDecodeSpecUpdateInfo(groups, result, draft, logprobs, updates);
    ASSERT_EQ(updates.size(), 1);
    EXPECT_EQ(updates[0].verified_draft_tokens, 0);
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
    auto round_state = processor.buildDSparkRoundState(empty, input.value(), holder);
    processor.prepareDSparkProposeModelInput(round_state, input.value(), holder);
    processor.prepareDSparkTargetVerifyModelInput(round_state, input.value(), {}, holder);
    EXPECT_EQ(input->combo_tokens.numel(), 0);
    EXPECT_EQ(input->engram_token_windows.sizes().vec(), (std::vector<int64_t>{0, 4}));
}

TEST_F(MtpBatchStreamProcessorTest, DSparkGatherReusesEngramHistoryAndDefersReplacedMetadata) {
    autil::EnvGuard device_input("RTP_LLM_DEVICE_INPUT", "1");
    ModelConfig     model;
    model.max_seq_len                         = 2048;
    model.vocab_size                          = 256;
    model.num_layers                          = 1;
    model.attn_config.v41_kv_source_layer_ids = {0};
    SpeculativeExecutionConfig speculative;
    speculative.type                    = SP_TYPE_DSPARK;
    speculative.gen_num_per_cycle       = 3;
    speculative.sp_dspark_mask_token_id = 255;
    MtpBatchStreamProcessor processor(model, {}, {}, makeProcessorCacheConfig(), speculative, false);
    const auto              options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    for (int64_t batch : {1, 7, 65}) {
        auto history = torch::arange(batch * 4, options).reshape({batch, 4});
        for (bool reverse : {false, true}) {
            std::list<GenerateStreamPtr> streams;
            for (int64_t index = 0; index < batch; ++index) {
                const auto row    = reverse ? batch - 1 - index : index;
                auto       stream = createContextStream(model, {}, {}, {10}, index + 1);
                stream->setIsContextStream(false);
                GenerateStream::MtpAsyncDeviceState state;
                state.accept_len_gpu                    = torch::full({1}, 2, options);
                state.accept_tokens_gpu                 = torch::full({1, 4}, 11, options);
                state.next_seq_len_gpu                  = torch::full({1}, 12, options);
                state.next_seq_len_upper_bound          = 12;
                state.next_kv_cache_block_id_gpu        = torch::ones({1, 1, 3}, options);
                state.next_kv_cache_kernel_block_id_gpu = torch::ones({1, 1, 4}, options);
                state.engram_token_window_gpu           = history.narrow(0, row, 1);
                stream->setMtpAsyncDeviceState(std::move(state));
                streams.push_back(stream);
            }
            TensorHolder holder;
            StreamGroups groups(streams);
            auto         input = processor.gatherDecodeModelInput(groups, holder);
            ASSERT_TRUE(input.ok());
            EXPECT_TRUE(input->combo_tokens.device().is_cpu());
            EXPECT_TRUE(input->input_lengths.device().is_cpu());
            EXPECT_TRUE(input->lm_output_indexes.device().is_cpu());
            EXPECT_TRUE(torch::equal(input->engram_token_windows, reverse ? history.flip({0}) : history));
            if (!reverse) {
                EXPECT_EQ(input->engram_token_windows.data_ptr(), history.data_ptr());
            }
            auto round = processor.buildDSparkRoundState(groups, *input, holder);
            processor.prepareDSparkProposeModelInput(round, *input, holder);
            EXPECT_TRUE(input->combo_tokens.is_cuda());
            EXPECT_TRUE(input->input_lengths.is_cuda());
            EXPECT_TRUE(input->prefix_lengths.is_cuda());
            EXPECT_TRUE(input->sequence_lengths.is_cuda());
            EXPECT_TRUE(input->lm_output_indexes.is_cuda());
            EXPECT_EQ(input->lm_output_indexes.numel(), batch * 3);
        }
    }
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

TEST_F(MtpBatchStreamProcessorTest, testPrepareOneStepSpecDecodeModelInput) {
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
    sp_config.gen_num_per_cycle                        = 1;

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

    // Qwen4 target verification consumes two adjacent text positions per
    // stream. All three MRoPE components must advance together; otherwise the
    // sparse target and draft indexer disagree about the accepted row.
    auto position_ids = toVec<int>(model_input.combo_position_ids);
    ASSERT_EQ(position_ids.size(), 2 * 2 * 3);
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t dim = 0; dim < 3; ++dim) {
            const size_t first  = (batch * 2) * 3 + dim;
            const size_t second = (batch * 2 + 1) * 3 + dim;
            EXPECT_EQ(position_ids[first], expect_prefix_lengths[batch]);
            EXPECT_EQ(position_ids[second], expect_prefix_lengths[batch] + 1);
            EXPECT_EQ(position_ids[second], position_ids[first] + 1);
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testPrepareOneStepSpecDecodeModelInputFromDeviceState) {
    setenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "1", 1);
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
    sp_config.gen_num_per_cycle                        = 1;

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

    auto position_ids = toVec<int>(model_input.combo_position_ids);
    ASSERT_EQ(position_ids.size(), 2 * 2 * 3);
    EXPECT_TRUE(model_input.combo_position_ids.is_cuda());
    EXPECT_EQ(model_input.combo_position_ids.scalar_type(), torch::kInt32);
    EXPECT_TRUE(model_input.combo_position_ids.is_contiguous());
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t dim = 0; dim < 3; ++dim) {
            const size_t first  = (batch * 2) * 3 + dim;
            const size_t second = (batch * 2 + 1) * 3 + dim;
            EXPECT_EQ(position_ids[first], expect_prefix_lengths[batch]);
            EXPECT_EQ(position_ids[second], expect_prefix_lengths[batch] + 1);
            EXPECT_EQ(position_ids[second], position_ids[first] + 1);
        }
    }
    unsetenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE");
}

TEST_F(MtpBatchStreamProcessorTest, testExpandTargetVerifyPositionIdsPreservesMropeOffsetFromDevicePrefix) {
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
    sp_config.gen_num_per_cycle                        = 3;

    ResourceContext resource_context;
    auto            stream = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3, 4}, 1);
    // The final multimodal row has max position 12. Relative to the canonical
    // input-length anchor 3, decode positions therefore carry offset 9.
    stream->setContextPositionIds(torch::tensor({0, 0, 0, 1, 1, 1, 7, 9, 8, 10, 12, 11}, torch::kInt32));

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    GptModelInputs model_input;
    model_input.combo_position_ids = torch::zeros({6}, torch::kInt32);
    auto authoritative_prefix      = torch::tensor({20, 40}, torch::kInt32).to(torch::kCUDA);

    processor.expandTargetVerifyPositionIds(StreamGroups({stream, stream}), model_input, authoritative_prefix);

    EXPECT_TRUE(model_input.combo_position_ids.is_cuda());
    EXPECT_EQ(model_input.combo_position_ids.scalar_type(), torch::kInt32);
    EXPECT_TRUE(model_input.combo_position_ids.is_contiguous());
    EXPECT_EQ(
        (vector<int>{29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52}),
        toVec<int>(model_input.combo_position_ids));
}

TEST_F(MtpBatchStreamProcessorTest, testprepareDecodeDraftModelInput) {
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

    auto& model_input                = model_input_status.value();
    model_input.sequence_lengths     = torch::tensor({1, 2}, torch::kInt32);
    const auto original_position_ids = model_input.combo_position_ids.clone();

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
    ASSERT_EQ(model_input.combo_position_ids.numel(), 2 * 3);
    EXPECT_TRUE(torch::equal(model_input.combo_position_ids, original_position_ids));

    // Legacy GPU propose-token path receives the normal decode position.
    stream1->getSPOutputBuffer()->propose_tokens_gpu = torch::tensor({{3}}, torch::kInt32).to(torch::kCUDA);
    stream2->getSPOutputBuffer()->propose_tokens_gpu = torch::tensor({{1}}, torch::kInt32).to(torch::kCUDA);
    model_input.sequence_lengths                     = torch::tensor({4, 5}, torch::kInt32);
    processor.prepareDecodeDraftModelInput(stream_groups, model_input, holder);

    expect_positions(model_input, {4, 5}, {4, 5});
    EXPECT_TRUE(torch::equal(model_input.combo_position_ids, original_position_ids));

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
    EXPECT_TRUE(torch::equal(model_input.combo_position_ids, original_position_ids));
}

TEST_F(MtpBatchStreamProcessorTest, DraftPreparationPreservesProposalOnlyRowsAndDeviceLengths) {
    autil::EnvGuard stream_async("RTP_LLM_STREAM_ASYNC", "0");
    for (const auto* enabled : {"0", "1"}) {
        autil::EnvGuard device_state("RTP_LLM_MTP_ASYNC_DEVICE_STATE", enabled);
        SCOPED_TRACE(enabled);
        for (bool pending : {false, true}) {
            SCOPED_TRACE(pending);
            ModelConfig model_config;
            model_config.max_seq_len                           = 2048;
            model_config.vocab_size                            = 32;
            model_config.num_layers                            = 1;
            model_config.mm_model_config.mm_position_ids_style = MROPE;
            model_config.attn_config.rope_config.index_factor  = 3;
            SpeculativeExecutionConfig sp_config;
            sp_config.gen_num_per_cycle = 3;
            RuntimeConfig   runtime_config;
            ResourceContext resource_context;
            auto steady = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3, 4}, 1);
            auto idle   = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3}, 1);
            idle->setIsContextStream(false);
            idle->getSPOutputBuffer()->tokens = torch::tensor({{0, 2}}, torch::kInt32);
            const auto cuda_i32               = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            GenerateStream::MtpAsyncDeviceState steady_state;
            steady_state.propose_tokens_gpu                = torch::full({1, 1}, 11, cuda_i32);
            steady_state.next_seq_len_gpu                  = torch::full({1}, 6, cuda_i32);
            steady_state.next_seq_len_upper_bound          = 6;
            steady_state.next_kv_cache_block_id_gpu        = torch::ones({1, 1, 1}, cuda_i32);
            steady_state.next_kv_cache_kernel_block_id_gpu = steady_state.next_kv_cache_block_id_gpu.clone();
            steady->setMtpAsyncDeviceState(std::move(steady_state));
            if (pending) {
                steady->incPendingAsyncBookkeeping();
            }
            auto pending_guard = std::shared_ptr<void>(nullptr, [steady, pending](void*) {
                if (pending) {
                    steady->decPendingAsyncBookkeepingAndMaybeRelease();
                }
            });
            GenerateStream::MtpAsyncDeviceState idle_state;
            idle_state.propose_tokens_gpu = torch::full({1, 1}, 22, cuda_i32);
            idle->setMtpAsyncDeviceState(std::move(idle_state));
            MtpBatchStreamProcessor processor(model_config,
                                              PDSepConfig{},
                                              ProfilingDebugLoggingConfig{},
                                              makeProcessorCacheConfig(),
                                              sp_config,
                                              false);
            TensorHolder            holder;
            GptModelInputs          input;
            input.input_lengths = torch::tensor({1, 3}, torch::kInt32);
            // Snapshot rows have a shape-only zero from the generic gatherer.
            input.sequence_lengths   = torch::tensor({0, 2}, torch::kInt32);
            input.combo_position_ids = torch::tensor({0, 0, 0, 2, 2, 2}, torch::kInt32);
            processor.prepareDecodeDraftModelInput(StreamGroups({steady, idle}), input, holder);
            EXPECT_EQ((vector<int>{11, 22}), toVec<int>(input.combo_tokens));
            EXPECT_EQ((vector<int>{5, 2}), toVec<int>(input.sequence_lengths));
            EXPECT_EQ((vector<int>{5, 2}), toVec<int>(input.prefix_lengths));
            EXPECT_EQ((vector<int>{5, 5, 5, 2, 2, 2}), toVec<int>(input.combo_position_ids));
            EXPECT_EQ(pending, steady->hasPendingAsyncBookkeeping());

            GptModelInputs idle_input;
            idle_input.input_lengths      = torch::tensor({3}, torch::kInt32);
            idle_input.sequence_lengths   = torch::tensor({2}, torch::kInt32);
            idle_input.combo_position_ids = torch::tensor({2, 2, 2}, torch::kInt32);
            processor.prepareDecodeDraftModelInput(StreamGroups({idle}), idle_input, holder);
            // The published proposal wins over the deliberately different legacy token,
            // including when the device-state environment switch is off.
            EXPECT_EQ((vector<int>{22}), toVec<int>(idle_input.combo_tokens));
            EXPECT_EQ((vector<int>{2}), toVec<int>(idle_input.prefix_lengths));

            for (bool missing_length : {false, true}) {
                SCOPED_TRACE(missing_length);
                auto incomplete               = steady->getMtpAsyncDeviceState();
                incomplete.propose_tokens_gpu = missing_length ? torch::full({1, 1}, 11, cuda_i32) : torch::Tensor();
                incomplete.next_seq_len_gpu   = missing_length ? torch::Tensor() : torch::full({1}, 6, cuda_i32);
                steady->setMtpAsyncDeviceState(std::move(incomplete));
                GptModelInputs invalid_input;
                invalid_input.input_lengths      = torch::tensor({1}, torch::kInt32);
                invalid_input.sequence_lengths   = torch::tensor({0}, torch::kInt32);
                invalid_input.combo_position_ids = torch::zeros({3}, torch::kInt32);
                EXPECT_THROW(processor.prepareDecodeDraftModelInput(StreamGroups({steady}), invalid_input, holder),
                             std::runtime_error);
            }
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, DraftCacheWritesRemainContiguousAcrossPreparationPaths) {
    // This test exercises the legacy compact post-rejection path; device-state
    // preparation below is selected by explicitly supplied stream state.
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
            GptModelOutputs verified;
            verified.all_hidden_states = torch::zeros({5, 2});
            torch::Tensor selected_hidden;
            processor.updateDecodePostDraftModelInput(post, verified, rejection, 1, selected_hidden, holder);
            ASSERT_EQ((vector<int>{accepted}), toVec<int>(post.input_lengths));
            ASSERT_EQ((vector<int>{first_unwritten_slot}), toVec<int>(post.prefix_lengths));
            const int      next_slot = toVec<int>(post.prefix_lengths)[0] + toVec<int>(post.input_lengths)[0];
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

TEST_F(MtpBatchStreamProcessorTest, DSparkPrefillCommitRequiresTargetFeaturesAndPositiveWidth) {
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
    model_input.input_lengths  = torch::tensor({3, 2}, torch::kInt32);
    model_input.prefix_lengths = torch::tensor({7, 4}, torch::kInt32);

    auto target_features =
        torch::arange(0, 60, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({5, 12});

    model_input.last_hidden_states = target_features;
    EXPECT_NO_THROW(processor.validatePrefillDSparkCommitInput(model_input));

    GptModelInputs no_features     = model_input;
    no_features.last_hidden_states = torch::Tensor();
    EXPECT_THROW(processor.validatePrefillDSparkCommitInput(no_features), std::exception);

    SpeculativeExecutionConfig zero_width_config = sp_config;
    zero_width_config.gen_num_per_cycle          = 0;
    MtpBatchStreamProcessor zero_width(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, zero_width_config, false);
    EXPECT_THROW(zero_width.validatePrefillDSparkCommitInput(model_input), std::exception);
}

TEST_F(MtpBatchStreamProcessorTest, DSparkProposalSelectsGammaLogitsInBothAnchorModes) {
    constexpr int32_t           gamma   = 3;
    constexpr int32_t           mask_id = 12345;
    ModelConfig                 model_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig  sp_config;
    sp_config.type                    = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle       = gamma;
    sp_config.sp_dspark_mask_token_id = mask_id;
    mtp::DSparkRoundState round_state{
        torch::tensor({101, 202}, torch::kInt32), torch::tensor({10, 6}, torch::kInt32), {}, {}};
    for (bool sample_from_anchor : {true, false}) {
        SCOPED_TRACE(sample_from_anchor);
        sp_config.sp_dspark_sample_from_anchor = sample_from_anchor;
        MtpBatchStreamProcessor processor(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
        GptModelInputs model_input;
        model_input.last_hidden_states = torch::ones({2, 12}, torch::kFloat32);
        TensorHolder host_holder;
        processor.prepareDSparkProposeModelInput(round_state, model_input, host_holder);

        /** Conditioning-only anchors add a query row; both modes select exactly gamma logits per request. */
        EXPECT_EQ(toVec<int32_t>(model_input.combo_tokens),
                  sample_from_anchor ?
                      (std::vector<int32_t>{101, mask_id, mask_id, 202, mask_id, mask_id}) :
                      (std::vector<int32_t>{101, mask_id, mask_id, mask_id, 202, mask_id, mask_id, mask_id}));
        EXPECT_EQ(toVec<int32_t>(model_input.input_lengths),
                  sample_from_anchor ? (std::vector<int32_t>{3, 3}) : (std::vector<int32_t>{4, 4}));
        EXPECT_EQ(toVec<int32_t>(model_input.lm_output_indexes),
                  sample_from_anchor ? (std::vector<int32_t>{0, 1, 2, 3, 4, 5}) :
                                       (std::vector<int32_t>{1, 2, 3, 5, 6, 7}));
        EXPECT_EQ(toVec<int32_t>(model_input.prefix_lengths), (std::vector<int32_t>{10, 6}));
        EXPECT_FALSE(model_input.last_hidden_states.defined());
    }
}

TEST_F(MtpBatchStreamProcessorTest, testTorchSpecDSparkAnchorProposalAndVerifyWidthsFiveAndSeven) {
    for (const int32_t gamma : {5, 7}) {
        SCOPED_TRACE(gamma);
        ModelConfig                 model_config;
        PDSepConfig                 pd_sep_config;
        ProfilingDebugLoggingConfig logging_config;
        CacheConfig                 cache_config = makeProcessorCacheConfig();
        SpeculativeExecutionConfig  sp_config;
        sp_config.type                         = SP_TYPE_DSPARK;
        sp_config.gen_num_per_cycle            = gamma;
        sp_config.sp_dspark_mask_token_id      = 255;
        sp_config.sp_dspark_sample_from_anchor = true;
        MtpBatchStreamProcessor processor(model_config, pd_sep_config, logging_config, cache_config, sp_config, false);

        TensorHolder          holder;
        auto                  anchors   = torch::tensor({101, 202}, torch::kInt32).to(torch::kCUDA);
        auto                  ends      = torch::tensor({10, 20}, torch::kInt32).to(torch::kCUDA);
        auto                  positions = torch::tensor({10, 20}, torch::kInt32).reshape({2, 1}).to(torch::kCUDA);
        mtp::DSparkRoundState round{anchors, ends, positions};
        GptModelInputs        propose_input;
        mtp::DSparkProposeInputBuffers propose_buffers;
        mtp::prepareDSparkProposeInput(propose_input,
                                       round,
                                       /*propose_step=*/static_cast<size_t>(gamma),
                                       /*mask_token_id=*/255,
                                       /*sample_from_anchor=*/true,
                                       propose_buffers,
                                       holder);
        std::vector<int32_t> expected_propose;
        for (const int32_t anchor : {101, 202}) {
            expected_propose.push_back(anchor);
            expected_propose.insert(expected_propose.end(), gamma - 1, 255);
        }
        EXPECT_EQ(expected_propose, toVec<int32_t>(propose_input.combo_tokens));
        EXPECT_EQ((std::vector<int32_t>{gamma, gamma}), toVec<int32_t>(propose_input.input_lengths));
        EXPECT_EQ(gamma * 2, propose_input.lm_output_indexes.numel());

        auto           proposals = torch::arange(1, gamma * 2 + 1, torch::kInt32).reshape({2, gamma}).to(torch::kCUDA);
        GptModelInputs verify_input;
        processor.prepareDSparkTargetVerifyModelInput(round, verify_input, proposals, holder);
        std::vector<int32_t> expected_verify{101};
        for (int32_t token = 1; token <= gamma; ++token) {
            expected_verify.push_back(token);
        }
        expected_verify.push_back(202);
        for (int32_t token = gamma + 1; token <= gamma * 2; ++token) {
            expected_verify.push_back(token);
        }
        EXPECT_EQ(expected_verify, toVec<int32_t>(verify_input.combo_tokens));
        EXPECT_EQ((std::vector<int32_t>{gamma + 1, gamma + 1}), toVec<int32_t>(verify_input.input_lengths));
        EXPECT_EQ((gamma + 1) * 2, verify_input.lm_output_indexes.numel());
        EXPECT_TRUE(verify_input.is_target_verify);
    }
}

TEST_F(MtpBatchStreamProcessorTest, DFlashSevenProposalsExcludeAnchorAndPreserveCommitGeometry) {
    ModelConfig                 model_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig  sp_config;
    sp_config.type                         = SP_TYPE_DFLASH;
    sp_config.gen_num_per_cycle            = 7;
    sp_config.sp_dspark_mask_token_id      = 255;
    sp_config.sp_dspark_sample_from_anchor = false;
    MtpBatchStreamProcessor processor(model_config, pd_sep_config, logging_config, cache_config, sp_config, false);

    GptModelInputs        input;
    TensorHolder          holder;
    mtp::DSparkRoundState round{torch::tensor({101, 255}, torch::kInt32), torch::tensor({4096, 8192}, torch::kInt32)};
    mtp::DSparkProposeInputBuffers propose_buffers;
    mtp::prepareDSparkProposeInput(input,
                                   round,
                                   /*propose_step=*/7,
                                   /*mask_token_id=*/255,
                                   /*sample_from_anchor=*/false,
                                   propose_buffers,
                                   holder);
    EXPECT_EQ((std::vector<int32_t>{101, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}),
              toVec<int32_t>(input.combo_tokens));
    EXPECT_EQ((std::vector<int32_t>{8, 8}), toVec<int32_t>(input.input_lengths));
    EXPECT_EQ((std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15}),
              toVec<int32_t>(input.lm_output_indexes));
    EXPECT_EQ((std::vector<int32_t>{4096, 8192}), toVec<int32_t>(input.prefix_lengths));

    auto       features = torch::randn({16, 12}, torch::TensorOptions().device(torch::kCUDA));
    const auto tokens   = input.combo_tokens.clone();
    processor.updateDecodePostDSparkCommitInput(input, features, 2);
    EXPECT_TRUE(input.is_target_verify);
    EXPECT_TRUE(torch::equal(input.combo_tokens, tokens));
    EXPECT_TRUE(torch::equal(input.last_hidden_states, features));
    EXPECT_EQ((std::vector<int32_t>{8, 8}), toVec<int32_t>(input.input_lengths));
    EXPECT_EQ((std::vector<int32_t>{4096, 8192}), toVec<int32_t>(input.prefix_lengths));
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
    model_input.sequence_lengths     = torch::tensor({1, 2}, torch::kInt32);
    model_input.engram_token_windows = torch::tensor({{102, 101, 11, 10}, {22, 21, 20, -1}}, torch::kInt32);

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
    EXPECT_FALSE(proposal_input.engram_token_windows.defined());
    EXPECT_EQ(toVec<int32_t>(model_input.engram_token_windows),
              (std::vector<int32_t>{102, 101, 11, 10, 22, 21, 20, -1}));
    EXPECT_FALSE(model_input.combo_tokens.defined());
    EXPECT_FALSE(model_input.combo_position_ids.defined());

    auto target_prepare_input = proposal_input;
    processor.expandDSparkTargetVerifyPositionIdsFromProposal(target_prepare_input);
    EXPECT_EQ((std::vector<int32_t>{30,  40,  50,  31,  41,  51,  32,  42,  52,  33,  43,  53,
                                    122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125}),
              toVec<int32_t>(target_prepare_input.combo_position_ids));

    auto proposals = torch::tensor({{31, 32, 33}, {41, 42, 43}}, torch::kInt32).to(torch::kCUDA);
    prepareDSparkTargetVerifyModelInput(sp_config.gen_num_per_cycle, round_state, model_input, proposals, host_holder);
    EXPECT_EQ((std::vector<int32_t>{102, 31, 32, 33, 22, 41, 42, 43}), toVec<int32_t>(model_input.combo_tokens));
    EXPECT_EQ((std::vector<int32_t>{gamma + 1, gamma + 1}), toVec<int32_t>(model_input.input_lengths));
    EXPECT_EQ((std::vector<int32_t>{9, 2}), toVec<int32_t>(model_input.prefix_lengths));
    EXPECT_EQ((std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}), toVec<int32_t>(model_input.lm_output_indexes));
    EXPECT_EQ((std::vector<int32_t>{30,  40,  50,  31,  41,  51,  32,  42,  52,  33,  43,  53,
                                    122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125}),
              toVec<int32_t>(model_input.combo_position_ids));
    EXPECT_EQ(toVec<int32_t>(model_input.engram_token_windows),
              (std::vector<int32_t>{102, 101, 11, 10, 31, 102, 101, 11, 32, 31, 102, 101, 33, 32, 31, 102,
                                    22,  21,  20, -1, 41, 22,  21,  20, 42, 41, 22,  21,  43, 42, 41, 22}));
}

#if USING_CUDA
void expectDSparkBoundedVerifyInputsEqual(const GptModelInputs&           actual,
                                          const DSparkPackedTargetLayout& actual_layout,
                                          const GptModelInputs&           expected,
                                          const DSparkPackedTargetLayout& expected_layout) {
    auto same = [](const torch::Tensor& a, const torch::Tensor& b, const char* field) {
        SCOPED_TRACE(field);
        ASSERT_EQ(a.defined(), b.defined());
        if (a.defined() && b.defined()) {
            EXPECT_EQ(a.scalar_type(), b.scalar_type());
            EXPECT_EQ(a.sizes().vec(), b.sizes().vec());
            EXPECT_TRUE(torch::equal(a, b));
        }
    };
    same(actual.combo_tokens, expected.combo_tokens, "combo_tokens");
    same(actual.combo_position_ids, expected.combo_position_ids, "combo_position_ids");
    same(actual.input_lengths, expected.input_lengths, "input_lengths");
    same(actual.prefix_lengths, expected.prefix_lengths, "prefix_lengths");
    same(actual.sequence_lengths, expected.sequence_lengths, "sequence_lengths");
    same(actual.sequence_lengths_plus_1, expected.sequence_lengths_plus_1, "sequence_lengths_plus_1");
    same(actual.lm_output_indexes, expected.lm_output_indexes, "lm_output_indexes");
    same(actual.engram_token_windows, expected.engram_token_windows, "engram_token_windows");
    same(actual.kv_cache_block_id, expected.kv_cache_block_id, "kv_cache_block_id");
    same(actual_layout.dense_tokens, expected_layout.dense_tokens, "dense_tokens");
    same(actual_layout.dense_position_ids, expected_layout.dense_position_ids, "dense_position_ids");
    same(actual_layout.packed_to_dense, expected_layout.packed_to_dense, "packed_to_dense");
    same(actual_layout.dense_to_packed, expected_layout.dense_to_packed, "dense_to_packed");
    EXPECT_EQ(actual.is_target_verify, expected.is_target_verify);
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkBoundedVerifyCaptureReplaysCurrentDeviceInputs) {
    const c10::cuda::CUDAGuard       guard(0);
    const auto                       stream = at::cuda::getStreamFromPool();
    const c10::cuda::CUDAStreamGuard stream_guard(stream);
    ModelConfig                      model_config;
    PDSepConfig                      pd_sep_config;
    ProfilingDebugLoggingConfig      profiling_config;
    CacheConfig                      cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig       sp_config;
    sp_config.type              = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle = 5;
    MtpBatchStreamProcessor processor(model_config, pd_sep_config, profiling_config, cache_config, sp_config, false);
    const auto              ints  = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);
    constexpr int64_t       batch = 48, gamma = 5, budget = 144;
    DSparkRoundState        state;
    state.anchors                  = torch::arange(batch, ints) * 100 + 10;
    state.committed_ends           = torch::arange(batch, ints) * 7 + 15;
    state.position_bases           = torch::arange(batch * 3, ints).reshape({batch, 3}) * 11;
    state.engram_windows           = state.anchors.unsqueeze(1) - torch::arange(4, ints).unsqueeze(0);
    auto                 proposals = torch::arange(batch * gamma, ints).reshape({batch, gamma}) + 1000;
    auto                 rates     = torch::ones({batch, gamma}, ints.dtype(torch::kFloat32));
    std::vector<int64_t> order;
    for (int64_t row = batch - 1; row >= 0; --row) {
        order.push_back(row);
    }
    const auto     order_d = torch::tensor(order, ints.dtype(torch::kInt64));
    TensorHolder   holder;
    GptModelInputs captured;
    torch::Tensor  captured_lengths, captured_owned;
    auto           run = [&]() {
        return prepareDSparkBoundedTargetVerifyModelInput(sp_config.gen_num_per_cycle,
                                                          state,
                                                          captured,
                                                          proposals,
                                                          rates,
                                                          order,
                                                          budget,
                                                          holder,
                                                          captured_lengths,
                                                          &captured_owned);
    };
    run();
    stream.synchronize();
    at::cuda::CUDAGraph graph;
    graph.capture_begin();
    auto captured_layout = run();
    graph.capture_end();
    for (int round = 0; round < 3; ++round) {
        state.anchors.add_(1);
        state.committed_ends.add_(3);
        state.position_bases.add_(2);
        state.engram_windows.add_(1);
        proposals.add_(10);
        rates.fill_(1.0f);
        rates.narrow(0, round * 7, 7).zero_();
        graph.replay();
        auto expected_lengths = speculative::SpeculativeSampler::buildValidDraftLengths(rates, order_d, budget);
        GptModelInputs expected;
        auto           expected_owned  = proposals;
        auto           expected_layout = prepareDSparkPackedTargetVerifyModelInput(
            sp_config.gen_num_per_cycle, state, expected, proposals, expected_lengths, budget, holder, &expected_owned);
        EXPECT_TRUE(torch::equal(captured_lengths, expected_lengths));
        EXPECT_TRUE(torch::equal(captured_owned, expected_owned));
        expectDSparkBoundedVerifyInputsEqual(captured, captured_layout, expected, expected_layout);
    }
}
#endif

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
    /** Target forward clears this flag; the decode COMMIT wrapper must restore it. */
    model_input.is_target_verify = false;
    const auto combo_tokens      = model_input.combo_tokens.clone();
    const auto input_lengths     = model_input.input_lengths.clone();
    const auto prefix_lengths    = model_input.prefix_lengths.clone();
    const auto lm_output_indexes = model_input.lm_output_indexes.clone();
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

    GenerateStreamPtr stream1       = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    GenerateStreamPtr stream2       = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);
    GenerateStreamPtr stream3       = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 3);
    GenerateStreamPtr middle_stream = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3}, 4);
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
        torch::tensor({1, -2, 2, 1, 2, 3, 3, 2, 1, 0, 1, 2}, torch::kInt32).reshape({4, 3}).cuda();
    const auto sampled_before = sampler_output.token_ids.clone();

    processor.updatePrefillPostDraftModelInput(stream_groups, model_input, model_output, sampler_output, holder);

    auto        combo_tokens        = model_input.combo_tokens;
    vector<int> expect_combo_tokens = {2, 2, 3, 2, 1, 2, 3};
    EXPECT_EQ(expect_combo_tokens, toVec<int>(combo_tokens));
    EXPECT_TRUE(combo_tokens.is_cuda());
    EXPECT_TRUE(torch::equal(sampler_output.token_ids, sampled_before));
}
TEST_F(MtpBatchStreamProcessorTest, testUpdatePrefillPostDraftModelInputShiftsComboPositionIds) {
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
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;
    auto            stream1 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 1);
    auto            stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3}, 2);
    auto middle_stream      = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3, 0}, 3);
    stream1->setContextPositionIds(torch::tensor({100, 101, 102, 110, 111, 112}, torch::kInt32));
    stream2->setContextPositionIds(torch::tensor({200, 201, 202, 210, 211, 212, 220, 221, 222}, torch::kInt32));
    middle_stream->setContextPositionIds(
        torch::tensor({300, 301, 302, 310, 311, 312, 320, 321, 322, 330, 331, 332}, torch::kInt32));
    middle_stream->setChunkSize(2);
    ASSERT_TRUE(middle_stream->isMiddleChunk());
    auto           stream_groups = StreamGroups({stream1, stream2, middle_stream});
    GptModelInputs model_input;
    model_input.input_lengths      = torch::tensor({2, 3, 2}, torch::kInt32);
    model_input.combo_tokens       = torch::tensor({10, 11, 20, 21, 22, 30, 31}, torch::kInt32);
    model_input.combo_position_ids = torch::tensor(
        {100, 101, 102, 110, 111, 112, 200, 201, 202, 210, 211, 212, 220, 221, 222, 300, 301, 302, 310, 311, 312},
        torch::kInt32);
    GptModelOutputs model_output;
    model_output.all_hidden_states = torch::zeros({7, 2}, torch::kFloat32);
    SamplerOutput sampler_output;
    sampler_output.token_ids = torch::tensor({1, -2, 12, 1, 2, 23, 3, 2, 1}, torch::kInt32).reshape({3, 3});
    TensorHolder holder;
    processor.updatePrefillPostDraftModelInput(stream_groups, model_input, model_output, sampler_output, holder);
    EXPECT_EQ((vector<int>{11, 12, 21, 22, 23, 31, 3}), toVec<int>(model_input.combo_tokens));
    EXPECT_EQ((vector<int>{110, 111, 112, 112, 112, 112, 210, 211, 212, 220, 221,
                           222, 222, 222, 222, 310, 311, 312, 320, 321, 322}),
              toVec<int>(model_input.combo_position_ids));
}

TEST_F(MtpBatchStreamProcessorTest, testUpdateDecodePostDraftModelInput) {
    unsetenv("RTP_LLM_STREAM_ASYNC");
    unsetenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE");
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
        model_input, model_output, spec_decode_output, stream_groups.size(), hidden_states_d_t, holder);

    auto        combo_tokens        = model_input.combo_tokens.cpu();
    vector<int> expect_combo_tokens = {2, 3, 1, 2};
    EXPECT_EQ(expect_combo_tokens, toVec<int>(combo_tokens));

    auto        input_lengths        = model_input.input_lengths;
    vector<int> expect_input_lengths = {3, 1};
    EXPECT_EQ(expect_input_lengths, toVec<int>(input_lengths));

    auto        lm_output_indexes        = model_input.lm_output_indexes.cpu();
    vector<int> expect_lm_output_indexes = {2, 3};
    EXPECT_EQ(expect_lm_output_indexes, toVec<int>(lm_output_indexes));

    auto          last_hidden_states        = model_input.last_hidden_states;
    vector<float> expect_last_hidden_states = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 1.1f, 1.2f};
    EXPECT_EQ(expect_last_hidden_states, toVec<float>(last_hidden_states));
    EXPECT_FALSE(model_input.is_target_verify);
}

TEST_F(MtpBatchStreamProcessorTest, testUpdateDecodePostDraftModelInputKeepsDenseDeviceState) {
    setenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE", "1", 1);
    ModelConfig                 model_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig  sp_config;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 2;

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    GptModelInputs model_input;
    model_input.is_target_verify = true;
    model_input.input_lengths    = torch::tensor({3, 3}, torch::kInt32).to(torch::kCUDA);
    model_input.prefix_lengths   = torch::tensor({7, 11}, torch::kInt32).to(torch::kCUDA);
    model_input.sequence_lengths = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    model_input.combo_position_ids =
        torch::tensor({10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62}, torch::kInt32)
            .to(torch::kCUDA);
    const auto                            dense_position_ids = model_input.combo_position_ids.clone();
    speculative::SpeculativeSamplerOutput spec_decode_output;
    // Exercise both ends of the legal interval [1, gamma + 1]. Device state
    // deliberately keeps the verify rows dense; the output indexes carry the
    // per-stream accepted boundary into the next draft prefill.
    spec_decode_output.accept_len    = torch::tensor({1, 3}, torch::kInt32).to(torch::kCUDA);
    spec_decode_output.accept_tokens = torch::tensor({{2, 0, 0}, {2, 3, 1}}, torch::kInt32).to(torch::kCUDA);
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
    EXPECT_EQ((vector<int>{2, 0, 0, 2, 3, 1}), toVec<int>(model_input.combo_tokens));
    EXPECT_EQ((vector<int>{3, 3}), toVec<int>(model_input.input_lengths));
    EXPECT_EQ((vector<int>{7, 11}), toVec<int>(model_input.prefix_lengths));
    EXPECT_EQ(0, model_input.sequence_lengths.numel());
    EXPECT_TRUE(model_input.lm_output_indexes.is_cuda());
    EXPECT_EQ((vector<int>{0, 5}), toVec<int>(model_input.lm_output_indexes));
    EXPECT_TRUE(torch::equal(model_input.combo_position_ids, dense_position_ids));
    EXPECT_EQ(6, model_input.last_hidden_states.size(0));
    EXPECT_TRUE(torch::equal(hidden_states_d_t, model_output.all_hidden_states));
    EXPECT_FALSE(model_input.is_target_verify);
    unsetenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE");
}

TEST_F(MtpBatchStreamProcessorTest, testUpdateDecodePostDraftModelInputCompactsComboPositionIds) {
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
    model_input.prefix_lengths     = torch::tensor({7, 11, 13}, torch::kInt32);
    model_input.sequence_lengths   = torch::empty({0}, torch::kInt32);
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
        model_input, model_output, spec_decode_output, stream_groups.size(), hidden_states_d_t, holder);

    EXPECT_EQ((vector<int>{2, 3, 1, 2, 3, 2}), toVec<int>(model_input.combo_tokens));
    EXPECT_EQ((vector<int>{3, 2, 1}), toVec<int>(model_input.input_lengths));
    EXPECT_EQ((vector<int>{7, 11, 13}), toVec<int>(model_input.prefix_lengths));
    EXPECT_EQ(0, model_input.sequence_lengths.numel());
    EXPECT_EQ((vector<int>{2, 4, 5}), toVec<int>(model_input.lm_output_indexes));
    // PyWrappedModel prefixes this cumsum with zero to build cu_seqlens.
    EXPECT_EQ((vector<int>{3, 5, 6}), toVec<int>(model_input.input_lengths.cumsum(0, torch::kInt32)));
    EXPECT_EQ(6, model_input.last_hidden_states.size(0));
    auto        combo_position_ids        = model_input.combo_position_ids;
    vector<int> expect_combo_position_ids = {10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 70, 71, 72};
    EXPECT_EQ(expect_combo_position_ids, toVec<int>(combo_position_ids));
}

TEST_F(MtpBatchStreamProcessorTest, testUpdateDecodeDraftModelInputKeepsCudaLengthsAndPositionDevice) {
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
    sp_config.gen_num_per_cycle                        = 4;

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    for (bool cuda_lengths : {false, true}) {
        for (const std::string positions : {"none", "cpu", "cuda"}) {
            SCOPED_TRACE(::testing::Message() << "cuda_lengths=" << cuda_lengths << ", positions=" << positions);
            GptModelInputs model_input;
            model_input.combo_tokens = torch::tensor({11, 21}, torch::kInt32).cuda();
            model_input.sequence_lengths =
                torch::tensor({5, 7}, torch::kInt32).to(cuda_lengths ? torch::kCUDA : torch::kCPU);
            model_input.prefix_lengths = model_input.sequence_lengths;
            if (positions != "none") {
                model_input.combo_position_ids = torch::tensor({5, 6, 7, 10, 11, 12}, torch::kInt32)
                                                     .to(positions == "cuda" ? torch::kCUDA : torch::kCPU);
            }
            TensorHolder holder;
            for (int step = 1; step <= 2; ++step) {
                const auto      previous_lengths   = model_input.sequence_lengths;
                const auto      previous_positions = model_input.combo_position_ids;
                GptModelOutputs model_output;
                model_output.all_hidden_states = torch::full({2, 2}, static_cast<float>(step)).cuda();
                auto draft_token_ids = torch::tensor({11 + step, 21 + step}, torch::kInt32).reshape({2, 1}).cuda();

                processor.updateDecodeDraftModelInput(model_input, model_output, draft_token_ids, holder);

                EXPECT_EQ((vector<int>{11 + step, 21 + step}), toVec<int>(model_input.combo_tokens));
                EXPECT_TRUE(model_input.sequence_lengths.is_cuda());
                EXPECT_EQ(model_input.sequence_lengths.scalar_type(), torch::kInt32);
                EXPECT_EQ((vector<int64_t>{2}), model_input.sequence_lengths.sizes().vec());
                EXPECT_EQ((vector<int>{5 + step, 7 + step}), toVec<int>(model_input.sequence_lengths));
                EXPECT_EQ((vector<int>{4 + step, 6 + step}), toVec<int>(previous_lengths));
                EXPECT_EQ((vector<int>{5, 7}), toVec<int>(model_input.prefix_lengths));
                EXPECT_EQ(model_input.last_hidden_states.data_ptr(), model_output.all_hidden_states.data_ptr());
                EXPECT_TRUE(
                    torch::equal(model_input.last_hidden_states.cpu(), torch::full({2, 2}, static_cast<float>(step))));
                if (positions == "none") {
                    EXPECT_FALSE(model_input.combo_position_ids.defined());
                } else {
                    EXPECT_EQ(model_input.combo_position_ids.device(), previous_positions.device());
                    EXPECT_EQ(model_input.combo_position_ids.scalar_type(), torch::kInt32);
                    EXPECT_EQ((vector<int64_t>{6}), model_input.combo_position_ids.sizes().vec());
                    EXPECT_EQ(model_input.combo_position_ids.is_pinned(), positions == "cpu");
                    EXPECT_EQ((vector<int>{5 + step, 6 + step, 7 + step, 10 + step, 11 + step, 12 + step}),
                              toVec<int>(model_input.combo_position_ids));
                    EXPECT_EQ((vector<int>{4 + step, 5 + step, 6 + step, 9 + step, 10 + step, 11 + step}),
                              toVec<int>(previous_positions));
                }
            }
        }
    }
}

#if USING_CUDA
TEST_F(MtpBatchStreamProcessorTest, DraftPositionUpdatesStayOnCudaWithoutMutatingPreviousSteps) {
    ModelConfig model_config;
    model_config.max_seq_len                           = 2048;
    model_config.vocab_size                            = 32;
    model_config.num_layers                            = 1;
    model_config.mm_model_config.mm_position_ids_style = MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    SpeculativeExecutionConfig sp_config;
    sp_config.gen_num_per_cycle = 4;
    MtpBatchStreamProcessor processor(
        model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, makeProcessorCacheConfig(), sp_config, false);
    auto                       cuda_stream = c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard stream_guard(cuda_stream);
    GptModelInputs             input;
    input.combo_tokens                 = torch::tensor({11, 21}, torch::kInt32).cuda();
    input.sequence_lengths             = torch::tensor({5, 7}, torch::kInt32).cuda();
    input.prefix_lengths               = input.sequence_lengths;
    input.combo_position_ids           = torch::tensor({5, 6, 7, 10, 11, 12}, torch::kInt32).cuda();
    const auto      original_positions = input.combo_position_ids;
    GptModelOutputs output;
    output.all_hidden_states = torch::zeros({2, 2}, torch::TensorOptions().device(torch::kCUDA));
    auto         tokens      = torch::tensor({{12}, {22}}, torch::kInt32).cuda();
    TensorHolder holder;
    processor.updateDecodeDraftModelInput(input, output, tokens, holder);
    EXPECT_TRUE(input.combo_position_ids.is_cuda());
    const auto first_step_positions = input.combo_position_ids;
    processor.updateDecodeDraftModelInput(input, output, tokens, holder);
    EXPECT_EQ(original_positions.device(), input.combo_position_ids.device());
    EXPECT_EQ(torch::kInt32, input.combo_position_ids.scalar_type());
    EXPECT_EQ(6, input.combo_position_ids.numel());
    cuda_stream.synchronize();
    EXPECT_EQ((vector<int>{5, 6, 7, 10, 11, 12}), toVec<int>(original_positions));
    EXPECT_EQ((vector<int>{6, 7, 8, 11, 12, 13}), toVec<int>(first_step_positions));
    EXPECT_EQ((vector<int>{7, 8, 9, 12, 13, 14}), toVec<int>(input.combo_position_ids));
    EXPECT_EQ((vector<int>{7, 9}), toVec<int>(input.sequence_lengths));
    EXPECT_EQ((vector<int>{5, 7}), toVec<int>(input.prefix_lengths));
}
#endif

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

#if USING_CUDA
TEST_F(MtpBatchStreamProcessorTest, testDevicePrefixPositionUploadsSurvivePinnedReuseOnNondefaultStream) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config           = makeProcessorCacheConfig();
    model_config.max_seq_len                           = 2048;
    model_config.vocab_size                            = 4;
    model_config.num_layers                            = 1;
    model_config.mm_model_config.mm_position_ids_style = MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    sp_config.gen_num_per_cycle                        = 1;
    ResourceContext resource_context;
    auto            stream = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3, 4}, 1);
    stream->setContextPositionIds(torch::tensor({0, 0, 0, 1, 1, 1, 7, 9, 8, 10, 12, 11}, torch::kInt32));
    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    auto                  cuda_stream = c10::cuda::getStreamFromPool();
    vector<torch::Tensor> outputs;
    {
        c10::cuda::CUDAStreamGuard guard(cuda_stream);
        for (int step = 0; step < 32; ++step) {
            GptModelInputs input;
            input.combo_position_ids = torch::zeros({3}, torch::kInt32);
            auto prefix = torch::full({1}, 20 + step, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
            processor.expandTargetVerifyPositionIds(StreamGroups({stream}), input, prefix);
            outputs.push_back(input.combo_position_ids);
            // Reuse similarly sized pinned allocations after the function's
            // temporary owners die, before any test-side synchronization.
            for (int reuse = 0; reuse < 4; ++reuse) {
                auto scratch = torch::empty({6}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
                scratch.fill_(-999);
            }
        }
    }
    cuda_stream.synchronize();
    for (int step = 0; step < 32; ++step) {
        EXPECT_EQ((vector<int>{29 + step, 29 + step, 29 + step, 30 + step, 30 + step, 30 + step}),
                  toVec<int>(outputs[step]));
    }
}
#endif

TEST_F(MtpBatchStreamProcessorTest, testExpandTargetVerifyPositionIdsUsesDeviceReceiverForDeviceState) {
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
    auto           empty_stream_groups = StreamGroups(std::list<GenerateStreamPtr>{});
    GptModelInputs input;
    input.combo_position_ids = torch::tensor({5, 6, 7, 10, 11, 12}, torch::kInt32);
    auto prefix              = torch::tensor({19, 31}, torch::kInt32).to(torch::kCUDA);
    processor.expandTargetVerifyPositionIds(empty_stream_groups, input, prefix);
    ASSERT_TRUE(input.combo_position_ids.is_cuda());
    EXPECT_EQ(input.combo_position_ids.device(), prefix.device());
    EXPECT_EQ(input.combo_position_ids.scalar_type(), torch::kInt32);
    EXPECT_TRUE(input.combo_position_ids.is_contiguous());
    EXPECT_EQ(input.combo_position_ids.numel(), 18);
    EXPECT_EQ(vector<int>(18, 0), toVec<int>(input.combo_position_ids));
    // The whole receive buffer can be overwritten by the rank-0 CUDA payload.
    auto positions = torch::arange(18, prefix.options());
    input.combo_position_ids.copy_(positions);
    EXPECT_EQ(toVec<int>(positions), toVec<int>(input.combo_position_ids));

    input.combo_position_ids = torch::zeros({6}, torch::kInt32);
    EXPECT_ANY_THROW(processor.expandTargetVerifyPositionIds(
        empty_stream_groups, input, torch::tensor({1}, torch::kInt32).to(torch::kCUDA)));
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
                      /*seq_size_per_block=*/64)
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
                      block_size)
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

TEST_F(MtpBatchStreamProcessorTest, testSparseLinearSnapshotCommitsAcceptedStateAcrossEveryBlockBoundary) {
    constexpr int32_t block_size  = 64;
    constexpr int32_t verify_size = 8;
    constexpr int64_t group_count = 3;
    constexpr int64_t block_count = 16;
    constexpr int64_t batch_size  = block_size * verify_size;
    constexpr int32_t first_write = block_count - verify_size;
    constexpr int32_t first_tail  = first_write - 2;
    constexpr int32_t null_block  = -1;

    auto                 current = torch::full({group_count, batch_size, block_count}, null_block, torch::kInt32);
    std::vector<int32_t> previous_lengths;
    std::vector<int32_t> accept_lengths;
    // A physical state's payload is represented by the number of tokens it
    // incorporates. This checks the decode contract independently of the
    // host/GPU swap-index helpers, including rejected speculative states.
    std::vector<int32_t> state_tokens(group_count * batch_size * block_count + 1, -1);
    previous_lengths.reserve(batch_size);
    accept_lengths.reserve(batch_size);
    for (int32_t offset = 0; offset < block_size; ++offset) {
        for (int32_t accept = 1; accept <= verify_size; ++accept) {
            const int64_t batch  = offset * verify_size + accept - 1;
            const int32_t cached = first_write * block_size + offset;
            previous_lengths.push_back(cached + 1);
            accept_lengths.push_back(accept);
            for (int64_t group = 0; group < group_count; ++group) {
                auto*         row   = current.select(0, group).select(0, batch).data_ptr<int32_t>();
                const int32_t begin = group == 1 ? 0 : first_tail;
                for (int32_t slot = begin; slot < block_count; ++slot) {
                    row[slot] = static_cast<int32_t>((group * batch_size + batch) * block_count + slot + 1);
                    state_tokens[row[slot]] = (slot + 1) * block_size;
                }
                if (group == 1) {
                    continue;
                }
                // The current recurrent state may live in the previous block
                // at an exact 64-token boundary. Verify writes one state per
                // token, starting at the current block, never in old history.
                const int32_t read_slot = (cached - 1) / block_size;
                ASSERT_GT(row[read_slot], 0);
                state_tokens[row[read_slot]] = cached;
                for (int32_t token = 0; token < verify_size; ++token) {
                    ASSERT_GT(row[first_write + token], 0);
                    state_tokens[row[first_write + token]] = cached + token + 1;
                }
            }
        }
    }

    auto device_current = current.to(torch::kCUDA);
    auto actual         = MtpBatchStreamProcessor::advanceLinearCacheBlockTable(
                      device_current,
                      torch::tensor(previous_lengths, torch::kInt32).to(torch::kCUDA),
                      torch::tensor(accept_lengths, torch::kInt32).to(torch::kCUDA),
                      {CacheGroupType::LINEAR, CacheGroupType::FULL, CacheGroupType::LINEAR},
                      block_size)
                      .cpu();

    EXPECT_TRUE(torch::equal(device_current.cpu(), current));
    EXPECT_TRUE(torch::equal(actual.select(0, 1), current.select(0, 1)));
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const int32_t cached      = previous_lengths[batch] - 1;
        const int32_t accept      = accept_lengths[batch];
        const int32_t next_cached = cached + accept;
        SCOPED_TRACE(::testing::Message() << "cached=" << cached << " accept=" << accept);
        for (const int64_t group : {0L, 2L}) {
            const auto*   row       = actual.select(0, group).select(0, batch).data_ptr<int32_t>();
            const int32_t next_read = (next_cached - 1) / block_size;
            ASSERT_GT(row[next_read], 0);
            EXPECT_EQ(state_tokens[row[next_read]], next_cached);
            // Every completed boundary reached in this round must still be a
            // valid prefix checkpoint, even when a later candidate is rejected.
            const int32_t boundary = (cached / block_size + 1) * block_size;
            if (boundary <= next_cached) {
                const int32_t checkpoint = boundary / block_size - 1;
                ASSERT_GT(row[checkpoint], 0);
                EXPECT_EQ(state_tokens[row[checkpoint]], boundary);
            }
            if (cached % block_size == 0) {
                const int32_t checkpoint = cached / block_size - 1;
                ASSERT_GT(row[checkpoint], 0);
                EXPECT_EQ(state_tokens[row[checkpoint]], cached);
            }
            for (int32_t slot = 0; slot < first_tail; ++slot) {
                EXPECT_EQ(row[slot], null_block);
            }
            for (int32_t slot = first_tail; slot < block_count; ++slot) {
                EXPECT_GT(row[slot], 0);
            }
            if (accept == 1) {
                EXPECT_TRUE(
                    torch::equal(actual.select(0, group).select(0, batch), current.select(0, group).select(0, batch)));
            }
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testSparseSnapshotOverlayDoesNotResurrectHostHistory) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    model_config.max_seq_len                 = 2048;
    model_config.vocab_size                  = 8;
    model_config.num_layers                  = 1;
    sp_config.gen_num_per_cycle              = 7;

    ResourceContext resource_context;
    auto            steady = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    auto            fresh  = createContextStream(model_config, runtime_config, resource_context, {2}, 2);
    steady->setIsContextStream(false);
    fresh->setIsContextStream(false);

    constexpr int64_t block_count = 16;
    constexpr int64_t first_tail  = 6;
    auto              physical    = torch::arange(1, block_count + 1, torch::kInt32).reshape({1, 1, block_count});
    auto              kernel      = physical + 100;
    physical.narrow(2, 0, first_tail).fill_(-1);
    kernel.narrow(2, 0, first_tail).fill_(-1);
    GenerateStream::MtpAsyncDeviceState state;
    state.next_kv_cache_block_id_gpu        = physical.to(torch::kCUDA);
    state.next_kv_cache_kernel_block_id_gpu = kernel.to(torch::kCUDA);
    steady->setMtpAsyncDeviceState(std::move(state));

    GptModelInputs model_input;
    const auto     pinned_i32 = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    // The host view intentionally disagrees with the committed device snapshot.
    model_input.kv_cache_block_id        = torch::full({1, 2, block_count}, 71, pinned_i32);
    model_input.kv_cache_kernel_block_id = torch::full({1, 2, block_count}, 171, pinned_i32);
    const auto fresh_physical            = model_input.kv_cache_block_id.select(1, 1).clone();
    const auto fresh_kernel              = model_input.kv_cache_kernel_block_id.select(1, 1).clone();

    TestableMtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    processor.overlayMtpCacheSnapshots(StreamGroups({steady, fresh}), model_input, holder);
    // The asynchronous overlay owns its source tensors even after a later
    // allocator boundary invalidates the stream's published snapshot.
    steady->clearMtpCacheSnapshot();

    EXPECT_FALSE(steady->hasMtpCacheSnapshot());
    EXPECT_TRUE(torch::equal(model_input.kv_cache_block_id.select(1, 0).cpu(), physical.select(1, 0)));
    EXPECT_TRUE(torch::equal(model_input.kv_cache_kernel_block_id.select(1, 0).cpu(), kernel.select(1, 0)));
    EXPECT_TRUE(torch::equal(model_input.kv_cache_block_id.select(1, 1).cpu(), fresh_physical));
    EXPECT_TRUE(torch::equal(model_input.kv_cache_kernel_block_id.select(1, 1).cpu(), fresh_kernel));
}

TEST_F(MtpBatchStreamProcessorTest, testNonLinearCacheSnapshotOwnsUnchangedTables) {
    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    // Different physical/kernel widths also occur with compressed cache
    // groups. Large sequence lengths must not be used as table indices for
    // FULL/SWA groups; only the allocator decides whether they have capacity.
    for (const int64_t width : {3, 7}) {
        auto source   = torch::arange(2 * 2 * width, cuda_i32).reshape({2, 2, width});
        auto expected = source.cpu();
        auto snapshot =
            MtpBatchStreamProcessor::advanceLinearCacheBlockTable(source,
                                                                  torch::tensor({32768, 32771}, cuda_i32),
                                                                  torch::tensor({1, 6}, cuda_i32),
                                                                  {CacheGroupType::FULL, CacheGroupType::SWA},
                                                                  /*seq_size_per_block=*/64);
        ASSERT_NE(snapshot.data_ptr(), source.data_ptr());
        source.fill_(-1);
        EXPECT_TRUE(torch::equal(snapshot.cpu(), expected));
    }
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
    const auto                          cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    GenerateStream::MtpAsyncDeviceState state;
    state.next_kv_cache_block_id_gpu =
        torch::tensor({11, 12, 13, 31, 32, 33}, torch::kInt32).reshape({2, 1, 3}).to(cuda_i32);
    state.next_kv_cache_kernel_block_id_gpu =
        torch::tensor({101, 102, 103, 104, 301, 302, 303, 304}, torch::kInt32).reshape({2, 1, 4}).to(cuda_i32);
    steady->setMtpAsyncDeviceState(state);
    // The steady row is owned by the immutable GPU snapshot, even if its
    // mutable host resource is unavailable while bookkeeping runs.
    steady->setKVCache(BatchKVCacheResource{});

    const auto expected_physical =
        torch::tensor({11, 12, 13, 0, 0, 21, 22, 23, 24, 25, 31, 32, 33, 0, 0, 41, 42, 43, 44, 45}, torch::kInt32)
            .reshape({2, 2, 5});
    const auto expected_kernel = torch::tensor({101, 102, 103, 104, 0, 0, 0, 201, 202, 203, 204, 205, 206, 207,
                                                301, 302, 303, 304, 0, 0, 0, 401, 402, 403, 404, 405, 406, 407},
                                               torch::kInt32)
                                     .reshape({2, 2, 7});
    for (bool device_tables : {false, true}) {
        SCOPED_TRACE(device_tables);
        GptModelInputs inputs;
        inputs.kv_cache_block_id        = expected_physical.pin_memory();
        inputs.kv_cache_kernel_block_id = expected_kernel.pin_memory();
        // This is the gather contract for snapshot-backed streams: both
        // tables are zero until overlay, for every cache group.
        inputs.kv_cache_block_id.select(1, 0).zero_();
        inputs.kv_cache_kernel_block_id.select(1, 0).zero_();
        if (device_tables) {
            inputs.kv_cache_block_id        = inputs.kv_cache_block_id.to(cuda_i32);
            inputs.kv_cache_kernel_block_id = inputs.kv_cache_kernel_block_id.to(cuda_i32);
        }
        TensorHolder holder;
        processor.overlayMtpCacheSnapshots(StreamGroups({steady, fresh}), inputs, holder);
        EXPECT_TRUE(torch::equal(inputs.kv_cache_block_id.cpu(), expected_physical));
        EXPECT_TRUE(torch::equal(inputs.kv_cache_kernel_block_id.cpu(), expected_kernel));
        EXPECT_TRUE(
            torch::equal(state.next_kv_cache_block_id_gpu.cpu(), expected_physical.narrow(1, 0, 1).narrow(2, 0, 3)));
        EXPECT_TRUE(torch::equal(state.next_kv_cache_kernel_block_id_gpu.cpu(),
                                 expected_kernel.narrow(1, 0, 1).narrow(2, 0, 4)));
    }
}

TEST_F(MtpBatchStreamProcessorTest, testSpeculativeMasksPreserveForcedAndRejectedTokens) {
    ModelConfig model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 8;
    model_config.num_layers  = 1;
    RuntimeConfig                runtime_config;
    ResourceContext              resource_context;
    std::list<GenerateStreamPtr> streams;
    for (int row = 0; row < 3; ++row) {
        auto stream = createContextStream(model_config, runtime_config, resource_context, {1}, row + 1);
        stream->generateConfig()->top_k = 1;
        stream->generateConfig()->top_p = 1.0f;
        streams.push_back(stream);
    }
    const auto    cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    SamplerOutput draft, target;
    draft.token_ids                = torch::tensor({{1, 2}, {1, 2}, {1, 2}}, torch::kInt32).to(cuda_i32);
    draft.token_ids_are_point_mass = true;
    target.token_ids = torch::tensor({{3}, {4}, {5}, {1}, {2}, {5}, {3}, {4}, {5}}, torch::kInt32).to(cuda_i32);
    target.all_probs = torch::empty({3, 3, 8}, torch::TensorOptions().device(torch::kCUDA));
    target.token_ids_are_point_mass = true;
    speculative::SpeculativeSampler sampler(torch::Tensor(), 2);
    for (int stochastic_row : {-1, 0, 1, 2}) {
        for (const std::vector<bool>& forced :
             {std::vector<bool>{false, false, false}, {true, true, true}, {true, false, false}, {false, true, true}}) {
            int row = 0;
            for (const auto& stream : streams) {
                stream->generateConfig()->top_k           = row == stochastic_row ? 2 : 1;
                stream->generateConfig()->force_sp_accept = forced[row++];
            }
            auto                 output = sampler.forward(streams, draft, target);
            std::vector<int32_t> expected_tokens, expected_lengths;
            for (row = 0; row < 3; ++row) {
                const bool accept_all = forced[row] || row == 1;
                expected_lengths.push_back(accept_all ? 3 : 1);
                const std::vector<int32_t> tokens =
                    accept_all ? std::vector<int32_t>{1, 2, 5} : std::vector<int32_t>{3, 0, 0};
                expected_tokens.insert(expected_tokens.end(), tokens.begin(), tokens.end());
            }
            EXPECT_EQ(expected_tokens, toVec<int32_t>(output.accept_tokens));
            EXPECT_EQ(expected_lengths, toVec<int32_t>(output.accept_len));
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testCacheSnapshotFullBatchAndWidthFallbackPreserveEveryEntry) {
    ModelConfig model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 8;
    model_config.num_layers  = 1;
    RuntimeConfig   runtime_config;
    ResourceContext resource_context;
    auto            first  = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    auto            second = createContextStream(model_config, runtime_config, resource_context, {2}, 2);
    first->setIsContextStream(false);
    second->setIsContextStream(false);
    SpeculativeExecutionConfig speculative;
    speculative.gen_num_per_cycle = 3;
    TestableMtpBatchStreamProcessor processor(
        model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, makeProcessorCacheConfig(), speculative, false);
    const auto cuda_i32   = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const auto pinned_i32 = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);

    // Identical widths use direct device concatenation when enabled. Wider
    // destinations and distinct snapshot widths must retain all host padding.
    for (int mode : {0, 1, 2, 3}) {
        GptModelInputs input;
        input.kv_cache_block_id                          = torch::full({2, 2, mode == 0 ? 3 : 5}, -7, pinned_i32);
        input.kv_cache_kernel_block_id                   = torch::full({2, 2, mode == 0 ? 4 : 7}, -9, pinned_i32);
        auto                           expected_physical = input.kv_cache_block_id.clone();
        auto                           expected_kernel   = input.kv_cache_kernel_block_id.clone();
        std::vector<GenerateStreamPtr> streams{first, second};
        for (int row = 0; row < 2; ++row) {
            int  physical_width = mode >= 2 && row == 1 ? 4 : 3;
            int  kernel_width   = mode >= 2 && row == 1 ? 6 : 4;
            auto physical =
                (torch::arange(2 * physical_width, torch::kInt32) + 100 * (row + 1)).reshape({2, 1, physical_width});
            auto kernel =
                (torch::arange(2 * kernel_width, torch::kInt32) + 1000 * (row + 1)).reshape({2, 1, kernel_width});
            expected_physical.narrow(1, row, 1).narrow(2, 0, physical_width).copy_(physical);
            expected_kernel.narrow(1, row, 1).narrow(2, 0, kernel_width).copy_(kernel);
            GenerateStream::MtpAsyncDeviceState state;
            state.next_kv_cache_block_id_gpu        = physical.to(cuda_i32);
            state.next_kv_cache_kernel_block_id_gpu = kernel.to(cuda_i32);
            if (mode == 3) {
                auto strided = torch::empty({2, 1, 2 * physical_width}, cuda_i32).slice(2, 0, 2 * physical_width, 2);
                strided.copy_(state.next_kv_cache_block_id_gpu);
                state.next_kv_cache_block_id_gpu = strided;
            }
            streams[row]->setMtpAsyncDeviceState(std::move(state));
        }
        TensorHolder holder;
        processor.overlayMtpCacheSnapshots(StreamGroups({first, second}), input, holder);
        EXPECT_TRUE(torch::equal(input.kv_cache_block_id.cpu(), expected_physical)) << "mode=" << mode;
        EXPECT_TRUE(torch::equal(input.kv_cache_kernel_block_id.cpu(), expected_kernel)) << "mode=" << mode;
    }
}

TEST_F(MtpBatchStreamProcessorTest, CacheSnapshotGatherHandlesChunkedMixedRaggedBatches) {
    ModelConfig model;
    model.max_seq_len = 2048;
    model.vocab_size  = 8;
    model.num_layers  = 1;
    RuntimeConfig              runtime;
    ResourceContext            resources;
    SpeculativeExecutionConfig speculative;
    speculative.gen_num_per_cycle = 3;
    TestableMtpBatchStreamProcessor processor(
        model, PDSepConfig{}, ProfilingDebugLoggingConfig{}, makeProcessorCacheConfig(), speculative, false);
    const auto host_options = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    for (int batch : {7, 65, 129}) {
        GptModelInputs input;
        input.kv_cache_block_id                        = torch::full({2, batch, 5}, -7, host_options);
        input.kv_cache_kernel_block_id                 = torch::full({2, batch, 7}, -9, host_options);
        auto                         expected_physical = input.kv_cache_block_id.clone();
        auto                         expected_kernel   = input.kv_cache_kernel_block_id.clone();
        std::list<GenerateStreamPtr> streams;
        for (int row = 0; row < batch; ++row) {
            auto stream = createContextStream(model, runtime, resources, {1}, row + 1);
            stream->setIsContextStream(false);
            if (row % 5 != 0) {
                const int pw = 3 + row % 3, kw = 4 + row % 4;
                auto      physical = torch::full({2, 1, pw}, row + 100, host_options);
                auto      kernel   = torch::full({2, 1, kw}, row + 1000, host_options);
                expected_physical.narrow(1, row, 1).narrow(2, 0, pw).copy_(physical);
                expected_kernel.narrow(1, row, 1).narrow(2, 0, kw).copy_(kernel);
                GenerateStream::MtpAsyncDeviceState state;
                state.next_kv_cache_block_id_gpu        = physical.to(torch::kCUDA);
                state.next_kv_cache_kernel_block_id_gpu = kernel.to(torch::kCUDA);
                stream->setMtpAsyncDeviceState(std::move(state));
            }
            streams.push_back(stream);
        }
        TensorHolder holder;
        processor.overlayMtpCacheSnapshots(StreamGroups(streams), input, holder);
        EXPECT_TRUE(torch::equal(input.kv_cache_block_id.cpu(), expected_physical)) << batch;
        EXPECT_TRUE(torch::equal(input.kv_cache_kernel_block_id.cpu(), expected_kernel)) << batch;
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

TEST_F(MtpBatchStreamProcessorTest, testPrefillTargetLogprobsUseEmittedFirstToken) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config = makeProcessorCacheConfig();

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
    auto target_logprobs = captureMtpTargetLogprobs(raw_logits, 2, 5);
    finalizeSelectedMtpTargetLogprobs(target_logprobs, torch::tensor({2}, torch::kInt32), {0});
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
    cache_config                = makeProcessorCacheConfig();
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
    cache_config = makeProcessorCacheConfig();

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
    auto target_logprobs = captureMtpTargetLogprobs(target_logits, 3, 5);

    // Preparing stream updates must reuse scores that have already been finalized.
    processor.finalizeDecodeTargetLogprobs(stream_groups, spec_output, target_logprobs);
    ASSERT_TRUE(target_logprobs.finalized());
    auto finalized_token_logprobs = target_logprobs.token_logprobs.clone();
    auto finalized_top_ids        = target_logprobs.top_logprob_token_ids.clone();
    auto finalized_top_logprobs   = target_logprobs.top_logprobs.clone();

    std::vector<StreamSpecUpdateInfo> update_infos;
    processor.prepareDecodeSpecUpdateInfo(
        stream_groups, spec_output, draft_prefill_output, target_logprobs, update_infos);

    ASSERT_EQ(update_infos.size(), 3);
    ASSERT_TRUE(target_logprobs.finalized());
    EXPECT_TRUE(torch::equal(target_logprobs.token_logprobs, finalized_token_logprobs));
    EXPECT_TRUE(torch::equal(target_logprobs.top_logprob_token_ids, finalized_top_ids));
    EXPECT_TRUE(torch::equal(target_logprobs.top_logprobs, finalized_top_logprobs));
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
    cache_config                = makeProcessorCacheConfig();
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
    // The complete {3, 4} closing delimiter remains reasoning; only token 5 is content.
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
    auto target_logprobs = captureMtpTargetLogprobs(target_logits, 8, 8);
    processor.finalizeDecodeTargetLogprobs(stream_groups, spec_output, target_logprobs);

    ASSERT_TRUE(target_logprobs.finalized());
    // Dense selected rows are crossing[4] => 10, then content[0] => 12.
    ASSERT_EQ(target_logprobs.token_logprobs.size(0), 2);
    // The thinking-only k=20 stream contributes no row. The crossing stream
    // requests k=0 and the content stream requests k=1, so the shared compact
    // reduction must use width 1 rather than the capture-time/global width 8.
    EXPECT_EQ(target_logprobs.maxTopLogprobs(), 1);
    auto expected_rows   = torch::tensor({10, 12}, torch::kInt64);
    auto expected_logits = target_logits.index_select(0, expected_rows);
    auto expected_ids    = torch::tensor({5, 6}, torch::kInt64);
    auto expected        = torch::log_softmax(expected_logits, -1).gather(1, expected_ids.unsqueeze(1)).squeeze(1);
    EXPECT_TRUE(torch::allclose(target_logprobs.token_logprobs.cpu(), expected.cpu()));

    std::vector<StreamSpecUpdateInfo> update_infos;
    processor.prepareDecodeSpecUpdateInfo(stream_groups, spec_output, draft_output, target_logprobs, update_infos);
    ASSERT_EQ(update_infos.size(), 3);
    EXPECT_EQ(update_infos[0].logprobs_offset, 2);
    EXPECT_FALSE(update_infos[0].token_logprobs.defined());
    EXPECT_EQ(update_infos[1].logprobs_offset, 4);
    EXPECT_EQ(update_infos[1].token_logprobs.sizes(), (torch::IntArrayRef{1, 1}));
    EXPECT_EQ(update_infos[1].top_logprobs.sizes(), (torch::IntArrayRef{1, 1, 0}));
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
    cache_config                = makeProcessorCacheConfig();
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
    auto target_logprobs = captureMtpTargetLogprobs(dense_logits, 2, 5);
    EXPECT_EQ(target_logprobs.raw_logits.size(0), 4);
    EXPECT_EQ(target_logprobs.dense_row_count, 4);
    EXPECT_FALSE(target_logprobs.token_logprobs.defined());

    // The mixed decode capture stays zero-copy until acceptance. Early finalize
    // then reduces only the one accepted row belonging to the requesting
    // stream; the regular worker consumes the compact result idempotently.
    processor.finalizeDecodeTargetLogprobs(stream_groups, spec_output, target_logprobs);
    ASSERT_TRUE(target_logprobs.finalized());

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
    cache_config                = makeProcessorCacheConfig();
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
    auto target_logprobs = captureMtpTargetLogprobs(raw_logits, 2, 5);
    finalizeSelectedMtpTargetLogprobs(target_logprobs, target_output.sampler_output.token_ids.select(1, 1), {0});

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

    stream->specUpdate({.new_tokens            = new_tokens,
                        .num_new_tokens        = 3,
                        .draft_tokens          = torch::Tensor(),
                        .draft_token_gpu       = torch::Tensor(),
                        .token_logprobs        = selected,
                        .top_logprob_token_ids = top_ids,
                        .top_logprobs          = top_values});

    ASSERT_FALSE(stream->hasError());
    EXPECT_EQ(stream->seqLength() - stream->inputLength(), 2);
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
    capped_stream->specUpdate({.new_tokens            = new_tokens,
                               .num_new_tokens        = 3,
                               .draft_tokens          = torch::Tensor(),
                               .draft_token_gpu       = torch::Tensor(),
                               .token_logprobs        = selected,
                               .top_logprob_token_ids = top_ids,
                               .top_logprobs          = top_values});

    EXPECT_TRUE(capped_stream->hasEvent(StreamEvents::GenerateDone));
    EXPECT_EQ(capped_stream->moveToNext(), StreamState::FINISHED);
    EXPECT_TRUE(capped_stream->isFinished());
    EXPECT_FALSE(capped_stream->hasError());
    EXPECT_EQ(capped_stream->seqLength(), capped_stream->inputLength());
    EXPECT_TRUE(torch::equal(capped_stream->getSPOutputBuffer()->tokens, original_spec_tokens));
    EXPECT_FALSE(capped_stream->getTokenLogProbs().defined());
}

TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputRejectsUndersizedLogitsWithoutLogprobs) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config = makeProcessorCacheConfig();

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

    GptModelOutputs model_output;
    model_output.logits =
        torch::arange(0, 8, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({2, 4});
    auto original_logits = model_output.logits.clone();
    auto draft           = torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    EXPECT_ANY_THROW(processor.gatherSpecSamplerInput(stream_groups, model_output, draft));
    EXPECT_TRUE(torch::equal(model_output.logits, original_logits));
}

TEST_F(MtpBatchStreamProcessorTest, testGatherSpecSamplerInputCropsPaddedLogitsWithoutLogprobs) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config = makeProcessorCacheConfig();

    model_config.max_seq_len    = 32;
    model_config.vocab_size     = 5;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 1;

    ModelConfig small_vocab_model_config = model_config;
    small_vocab_model_config.vocab_size  = model_config.vocab_size;

    ResourceContext resource_context;
    auto small_vocab_stream = createContextStream(small_vocab_model_config, runtime_config, resource_context, {1}, 1);
    auto large_vocab_stream = createContextStream(model_config, runtime_config, resource_context, {2}, 2);
    small_vocab_stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
    large_vocab_stream->setScoreLen(sp_config.gen_num_per_cycle + 1);
    StreamGroups stream_groups({small_vocab_stream, large_vocab_stream});

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    GptModelOutputs model_output;
    model_output.logits =
        torch::arange(0, 32, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)).reshape({4, 8});
    auto original_logits = model_output.logits.clone();

    auto draft                 = torch::zeros({2, 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto sampler_inputs_status = processor.gatherSpecSamplerInput(stream_groups, model_output, draft);
    ASSERT_TRUE(sampler_inputs_status.ok());
    const auto& sampler_inputs = sampler_inputs_status.value();
    ASSERT_EQ(sampler_inputs.logits.sizes(), (torch::IntArrayRef{4, 5}));
    EXPECT_EQ(sampler_inputs.probability_output, SamplerProbabilityOutput::DISTRIBUTION);
    EXPECT_TRUE(torch::equal(model_output.logits, original_logits));
    EXPECT_TRUE(torch::equal(sampler_inputs.logits, original_logits.narrow(1, 0, 5)));
}

TEST_F(MtpBatchStreamProcessorTest, testSpecMaskExcludesPaddedSamplerLogitsWithoutLogprobs) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config = makeProcessorCacheConfig();

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

    GptModelOutputs model_output;
    // Real token logits are all negative while padded LM-head columns are zero.
    // Without suffix masking, greedy sampling would select OOV token 5.
    model_output.logits = torch::tensor(
        {{-5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 0.0f, 0.0f}, {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, 0.0f, 0.0f, 0.0f}},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto original_logits = model_output.logits.clone();

    SpecLogitsVerifyRunner::LaunchResult spec_mask_result;
    spec_mask_result.has_active_processor   = true;
    auto cuda_i32                           = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    spec_mask_result.packed_allow_mask_gpu  = torch::tensor({{15}, {30}}, cuda_i32);
    spec_mask_result.logits_row_indices_gpu = torch::tensor({0, 1}, cuda_i32);
    auto draft                              = torch::zeros({1, 1}, cuda_i32);
    auto sampler_inputs_status              = processor.gatherSpecSamplerInput(stream_groups, model_output, draft);
    ASSERT_TRUE(sampler_inputs_status.ok());
    auto sampler_inputs = std::move(sampler_inputs_status.value());
    applySamplerMask(sampler_inputs, spec_mask_result);

    sampler_inputs.top_k.fill_(1);
    Sampler sampler({});
    auto    output        = sampler.forward(sampler_inputs);
    auto    masked_logits = sampler_inputs.logits.cpu();

    EXPECT_EQ(masked_logits.index({0, 4}).item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(masked_logits.index({1, 0}).item<float>(), BaseLogitsProcessor::neg_inf);
    EXPECT_EQ(masked_logits.size(1), model_config.vocab_size);
    auto sampled_token_ids = output.token_ids.select(1, output.token_ids.size(1) - 1).cpu().to(torch::kLong);
    EXPECT_EQ(toVec<int64_t>(sampled_token_ids), (std::vector<int64_t>{3, 1}));
    EXPECT_TRUE(torch::all(sampled_token_ids < model_config.vocab_size).item<bool>());
    EXPECT_TRUE(torch::equal(model_output.logits.narrow(1, 5, 3), original_logits.narrow(1, 5, 3)));
    EXPECT_EQ(model_output.logits.data_ptr(), sampler_inputs.logits.data_ptr());
}

}  // namespace rtp_llm
