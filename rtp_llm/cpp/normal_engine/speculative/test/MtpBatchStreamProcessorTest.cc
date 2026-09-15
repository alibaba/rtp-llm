#include <cstdlib>
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
#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
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
        query->generate_config->num_return_sequences  = num_return_sequences;
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
        addr.setBatchBlocks(0, 0, {block_id});
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

    auto draft_token_ids       = torch::tensor({1, 2, 3, 3, 2, 1}, torch::kInt32).reshape({2, 3});
    auto sampler_inputs_status = processor.gatherSpecSamplerInput(stream_groups, model_output, {}, draft_token_ids);
    ASSERT_TRUE(sampler_inputs_status.ok());

    const auto& sampler_inputs = sampler_inputs_status.value();
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
    task.total_streams   = 1;
    task.propose_step    = static_cast<int>(sp_config.gen_num_per_cycle);
    task.vocab_size      = model_config.vocab_size;
    task.draft_tokens    = torch::tensor(std::vector<int32_t>{1, 2}, torch::kInt32).reshape({1, 2});
    for (const auto& processor_ptr : stream->getAllLogitsProcessorPtr()) {
        ASSERT_NE(processor_ptr, nullptr);
        ASSERT_EQ(processor_ptr->mtpCapability().mode, MtpProcessorMode::SPEC_VERIFY);
        task.active.push_back({processor_ptr, /*stream_idx=*/0});
    }
    ASSERT_EQ(1u, task.active.size());
    auto spec_result = runner.run(task);
    ASSERT_TRUE(spec_result.has_active_processor);

    auto sampler_inputs_status =
        processor.gatherSpecSamplerInput(stream_groups, model_output, spec_result, task.draft_tokens);
    ASSERT_TRUE(sampler_inputs_status.ok());
    auto sampler_inputs = sampler_inputs_status.value();

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

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    auto status = processor.dispatchDecode(stream_groups, spec_decode_output, std::move(draft_prefill_output));
    EXPECT_TRUE(status.ok());

    checkOutput(stream1, {1, 2, 3, 1, 3, 2}, {2, 0}, {0.2, 0.1, 0.3, 0.5}, {0.6, 0.06});
    checkOutput(stream2, {2, 1, 2}, {2, 3}, {0.3, 0.1, 0.4, 0.2}, {1.3, 0.13});
    // Device-state real_seq_len publication moved to the executor layer
    // (MtpExecutor::publishSyncMtpDeviceState); dispatchDecode itself only
    // performs host bookkeeping now, so no device-state asserts here.
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

    auto          last_hidden_states        = model_input.value().last_hidden_states;
    auto          last_hidden_states_h      = last_hidden_states.cpu().clone();
    vector<float> expect_last_hidden_states = {0.1, 0.2, 1.1, 1.2};
    EXPECT_EQ(expect_last_hidden_states, toVec<float>(last_hidden_states_h));
}

TEST_F(MtpBatchStreamProcessorTest, testDecodeTextMaskRebindPreservesMultimodalStorageAndUndefinedMask) {
    for (bool cuda_mask : {false, true}) {
        SCOPED_TRACE(cuda_mask);
        GptModelInputs input;
        input.combo_tokens = torch::arange(5, torch::kInt32).cuda();
        MtpBatchStreamProcessor::resetDecodeTextTokensMask(input);
        EXPECT_FALSE(input.text_tokens_mask.defined());

        const auto prefill_mask = torch::tensor({1, 0, 0, 1}, torch::kInt32);
        input.text_tokens_mask = cuda_mask ? prefill_mask.cuda() : prefill_mask;
        const auto original_mask = input.text_tokens_mask;
        const auto feature = torch::arange(8, torch::kFloat32).reshape({2, 4});
        input.multimodal_features = std::vector<torch::Tensor>{feature};
        input.mm_features_locs = torch::tensor({1}, torch::kInt32);
        const auto feature_locs = input.mm_features_locs;
        MtpBatchStreamProcessor::resetDecodeTextTokensMask(input);
        EXPECT_EQ((vector<int>{1, 1, 1, 1, 1}), toVec<int>(input.text_tokens_mask));
        EXPECT_EQ(input.text_tokens_mask.is_cuda(), cuda_mask);
        EXPECT_TRUE(input.text_tokens_mask.is_contiguous());
        EXPECT_EQ((vector<int>{1, 0, 0, 1}), toVec<int>(original_mask));
        EXPECT_EQ(input.multimodal_features.value()[0].data_ptr(), feature.data_ptr());
        EXPECT_EQ(input.mm_features_locs.data_ptr(), feature_locs.data_ptr());

        input.combo_tokens = torch::arange(2, torch::kInt32).cuda();
        MtpBatchStreamProcessor::resetDecodeTextTokensMask(input);
        EXPECT_EQ((vector<int>{1, 1}), toVec<int>(input.text_tokens_mask));
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

    model_input.text_tokens_mask = torch::ones({2}, torch::kInt32).pin_memory();
    const auto gathered_mask = model_input.text_tokens_mask;
    processor.prepareOneStepSpecDecodeModelInput(stream_groups, model_input, holder);
    EXPECT_EQ((vector<int>{1, 1, 1, 1}), toVec<int>(model_input.text_tokens_mask));
    EXPECT_TRUE(model_input.text_tokens_mask.is_pinned());
    EXPECT_EQ((vector<int>{1, 1}), toVec<int>(gathered_mask));

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

TEST_F(MtpBatchStreamProcessorTest, testOneStepDeviceStateExpandsMropePositionsAcrossRounds) {
    struct RestoreEnv {
        std::string name;
        bool        present;
        std::string value;
        explicit RestoreEnv(const char* key):
            name(key), present(std::getenv(key) != nullptr), value(present ? std::getenv(key) : "") {}
        ~RestoreEnv() {
            if (present) {
                setenv(name.c_str(), value.c_str(), 1);
            } else {
                unsetenv(name.c_str());
            }
        }
    } restore_stream_async("RTP_LLM_STREAM_ASYNC"), restore_device_state("RTP_LLM_MTP_ASYNC_DEVICE_STATE");

    // Both switches select the same device gather. Restore them even after an
    // ASSERT failure, because this target also contains host-pipeline tests.
    for (bool stream_async : {false, true}) {
        SCOPED_TRACE(stream_async);
        setenv("RTP_LLM_STREAM_ASYNC", stream_async ? "1" : "0", 1);
        setenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE", stream_async ? "0" : "1", 1);
        for (int position_style : {-1, static_cast<int>(DEFAULT), static_cast<int>(MMWITHTAG), static_cast<int>(MROPE)}) {
            SCOPED_TRACE(position_style);
            const bool with_positions = position_style >= 0;
            const bool with_mrope     = position_style == static_cast<int>(MROPE);
            const int  axes           = with_mrope ? 3 : 1;
            ModelConfig                model_config;
            RuntimeConfig              runtime_config;
            ResourceContext            resource_context;
            SpeculativeExecutionConfig sp_config;
            model_config.max_seq_len  = 2048;
            model_config.vocab_size   = 256;
            model_config.num_layers   = 1;
            if (with_positions) {
                model_config.mm_model_config.mm_position_ids_style = position_style;
                model_config.attn_config.rope_config.index_factor   = axes;
            }
            sp_config.type              = SP_TYPE_EAGLE;
            sp_config.model_type        = "qwen35_moe_mtp";
            sp_config.gen_num_per_cycle = 1;
            auto stream1 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 1);
            auto stream2 = createContextStream(model_config, runtime_config, resource_context, {3, 4, 5}, 2);
            stream1->setIsContextStream(false);
            stream2->setIsContextStream(false);
            if (with_mrope) {
                stream1->setContextPositionIds(torch::tensor({0, 0, 0, 4, 12, 7}, torch::kInt32));
                stream2->setContextPositionIds(torch::tensor({0, 0, 0, 1, 1, 1, 24, 20, 31}, torch::kInt32));
            } else if (position_style == static_cast<int>(MMWITHTAG)) {
                stream1->setContextPositionIds(torch::tensor({0, 12}, torch::kInt32));
                stream2->setContextPositionIds(torch::tensor({0, 1, 31}, torch::kInt32));
            }
            MtpBatchStreamProcessor processor(model_config,
                                               PDSepConfig{},
                                               ProfilingDebugLoggingConfig{},
                                               makeProcessorCacheConfig(),
                                               sp_config,
                                               false);
            StreamGroups stream_groups({stream1, stream2});
            const std::vector<std::vector<int>> sequence_lengths{{3, 4}, {5, 5}};
            const std::vector<std::vector<int>> host_sequence_lengths{{3, 4}, {3, 4}};
            const std::vector<std::vector<int>> accept_lengths{{2, 1}, {1, 2}};
            const std::vector<std::vector<int>> expected_tokens{{41, 80, 50, 90}, {60, 81, 71, 91}};
            // Independent, literal oracle: decode starts at max(last context
            // T/H/W) + device seq_len - context_len, then advances once for the
            // draft. Round 2 leaves host bookkeeping one accepted batch behind.
            std::vector<std::vector<int>> gathered_positions{{13, 13, 13, 32, 32, 32},
                                                              {13, 13, 13, 32, 32, 32}};
            std::vector<std::vector<int>> expected_positions{
                {13, 13, 13, 14, 14, 14, 32, 32, 32, 33, 33, 33},
                {15, 15, 15, 16, 16, 16, 33, 33, 33, 34, 34, 34}};
            if (!with_mrope) {
                if (position_style == static_cast<int>(MMWITHTAG)) {
                    gathered_positions = {{13, 32}, {13, 32}};
                    expected_positions = {{13, 14, 32, 33}, {15, 16, 33, 34}};
                } else {
                    gathered_positions = {{2, 3}, {2, 3}};
                    expected_positions = {{2, 3, 3, 4}, {4, 5, 4, 5}};
                }
            }
            for (size_t round = 0; round < sequence_lengths.size(); ++round) {
                SCOPED_TRACE(round);
                const std::vector<GenerateStreamPtr> streams{stream1, stream2};
                for (size_t request = 0; request < streams.size(); ++request) {
                    streams[request]->setSeqLength(host_sequence_lengths[round][request]);
                    const int first_token = 40 + static_cast<int>(round) * 20 + static_cast<int>(request) * 10;
                    GenerateStream::MtpAsyncDeviceState state;
                    state.accept_len_gpu = torch::tensor({accept_lengths[round][request]}, torch::kInt32).cuda();
                    state.accept_tokens_gpu = torch::tensor({first_token, first_token + 1}, torch::kInt32)
                                                  .reshape({1, 2})
                                                  .cuda();
                    state.propose_tokens_gpu =
                        torch::tensor({80 + static_cast<int>(request) * 10 + static_cast<int>(round)}, torch::kInt32)
                            .reshape({1, 1})
                            .cuda();
                    state.next_seq_len_gpu = torch::tensor({sequence_lengths[round][request]}, torch::kInt32).cuda();
                    streams[request]->setMtpAsyncDeviceState(std::move(state));
                }
                GptModelInputs inputs;
                inputs.combo_tokens      = torch::tensor({-10, -20}, torch::kInt32);
                inputs.input_lengths     = torch::tensor({1, 1}, torch::kInt32);
                inputs.sequence_lengths  = torch::tensor({99, 99}, torch::kInt32);
                if (with_positions) {
                    inputs.combo_position_ids = torch::tensor(gathered_positions[round], torch::kInt32);
                }
                if (with_mrope) {
                    inputs.text_tokens_mask = torch::ones({2}, torch::kInt32).cuda();
                }
                TensorHolder holder;
                processor.prepareOneStepSpecDecodeModelInput(stream_groups, inputs, holder);
                // Device state differs from host tokens and the deliberately
                // wrong host sequence lengths, so fallback cannot pass these.
                if (with_mrope) {
                    ASSERT_TRUE(inputs.text_tokens_mask.is_cuda());
                    EXPECT_EQ((vector<int>{1, 1, 1, 1}), toVec<int>(inputs.text_tokens_mask));
                } else {
                    EXPECT_FALSE(inputs.text_tokens_mask.defined());
                }
                ASSERT_TRUE(inputs.combo_tokens.is_cuda());
                ASSERT_EQ(inputs.combo_tokens.numel(), 4);
                EXPECT_EQ(expected_tokens[round], toVec<int>(inputs.combo_tokens));
                EXPECT_EQ((std::vector<int>{2, 2}), toVec<int>(inputs.input_lengths));
                EXPECT_EQ((std::vector<int>{sequence_lengths[round][0] - 1, sequence_lengths[round][1] - 1}),
                          toVec<int>(inputs.prefix_lengths));
                EXPECT_EQ((std::vector<int>{0, 1, 2, 3}), toVec<int>(inputs.lm_output_indexes));
                EXPECT_EQ(inputs.sequence_lengths.numel(), 0);
                if (with_positions) {
                    ASSERT_TRUE(inputs.combo_position_ids.is_cuda());
                    ASSERT_TRUE(inputs.combo_position_ids.is_contiguous());
                    ASSERT_EQ(inputs.combo_position_ids.numel(), inputs.combo_tokens.numel() * axes);
                    EXPECT_EQ(expected_positions[round], toVec<int>(inputs.combo_position_ids));
                } else {
                    EXPECT_FALSE(inputs.combo_position_ids.defined());
                }
            }
        }
    }
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
            EXPECT_EQ(expected_sequence, toVec<int>(input.prefix_lengths + 1));
        };
    expect_positions(model_input, {1, 2}, {2, 3});

    // Legacy GPU propose-token path receives the normal decode position.
    stream1->getSPOutputBuffer()->propose_tokens_gpu = torch::tensor({{3}}, torch::kInt32).to(torch::kCUDA);
    stream2->getSPOutputBuffer()->propose_tokens_gpu = torch::tensor({{1}}, torch::kInt32).to(torch::kCUDA);
    model_input.sequence_lengths                     = torch::tensor({4, 5}, torch::kInt32);
    processor.prepareDecodeDraftModelInput(stream_groups, model_input, holder);

    expect_positions(model_input, {4, 5}, {5, 6});

    // Device state publishes the committed length, which is already the draft
    // decode position and one greater than the target prefix.
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

    expect_positions(model_input, {6, 3}, {7, 4});
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
}

TEST_F(MtpBatchStreamProcessorTest, testDSparkPrepareAndVerifyUsePerStreamDeviceState) {
    constexpr int32_t gamma = 3;

    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config = makeProcessorCacheConfig();
    SpeculativeExecutionConfig  sp_config;
    model_config.max_seq_len          = 2048;
    model_config.vocab_size           = 256;
    model_config.num_layers           = 1;
    sp_config.type                    = SP_TYPE_DSPARK;
    sp_config.gen_num_per_cycle       = gamma;
    sp_config.sp_dspark_mask_token_id = 255;

    ResourceContext resource_context;

    auto steady_stream = createContextStream(model_config, runtime_config, resource_context, {10, 11}, 1);
    auto fresh_stream  = createContextStream(model_config, runtime_config, resource_context, {20, 21, 22}, 2);
    steady_stream->setIsContextStream(false);
    fresh_stream->setIsContextStream(false);

    // The steady stream's host token/length deliberately trail the state
    // published before its previous bookkeeping worker. The fresh stream has
    // no previous round and therefore legitimately falls back to host state.
    GenerateStream::MtpAsyncDeviceState steady_state;
    steady_state.accept_len_gpu    = torch::tensor({2}, torch::kInt32).to(torch::kCUDA);
    steady_state.accept_tokens_gpu = torch::tensor({{101, 102, 0, 0}}, torch::kInt32).to(torch::kCUDA);
    steady_state.next_seq_len_gpu  = torch::tensor({10}, torch::kInt32).to(torch::kCUDA);
    steady_stream->setMtpAsyncDeviceState(std::move(steady_state));

    StreamGroups   stream_groups({steady_stream, fresh_stream});
    GptModelInputs model_input;
    model_input.sequence_lengths = torch::tensor({1, 2}, torch::kInt32);
    model_input.text_tokens_mask = torch::ones({2}, torch::kInt32);

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder host_holder;
    auto         round_head = processor.prepareDSparkDraftModelInput(stream_groups, model_input, host_holder);

    EXPECT_FALSE(model_input.combo_position_ids.defined());
    EXPECT_EQ((std::vector<int32_t>{102, 22}), toVec<int32_t>(round_head.anchors));
    EXPECT_EQ((std::vector<int32_t>{9, 2}), toVec<int32_t>(round_head.committed_ends));
    EXPECT_EQ((std::vector<int32_t>{102, 255, 255, 22, 255, 255}), toVec<int32_t>(model_input.combo_tokens));
    EXPECT_EQ((std::vector<int32_t>{gamma, gamma}), toVec<int32_t>(model_input.input_lengths));
    EXPECT_EQ((std::vector<int32_t>{9, 2}), toVec<int32_t>(model_input.prefix_lengths));

    EXPECT_EQ((vector<int>{1, 1, 1, 1, 1, 1}), toVec<int>(model_input.text_tokens_mask));

    auto proposals = torch::tensor({{31, 32, 33}, {41, 42, 43}}, torch::kInt32).to(torch::kCUDA);
    processor.updateDSparkTargetVerifyModelInput(round_head, model_input, proposals, host_holder);
    EXPECT_EQ((vector<int>{1, 1, 1, 1, 1, 1, 1, 1}), toVec<int>(model_input.text_tokens_mask));
    EXPECT_FALSE(model_input.combo_position_ids.defined());
    EXPECT_EQ((std::vector<int32_t>{102, 31, 32, 33, 22, 41, 42, 43}), toVec<int32_t>(model_input.combo_tokens));
    EXPECT_EQ((std::vector<int32_t>{gamma + 1, gamma + 1}), toVec<int32_t>(model_input.input_lengths));
    EXPECT_EQ((std::vector<int32_t>{9, 2}), toVec<int32_t>(model_input.prefix_lengths));
    EXPECT_EQ((std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}), toVec<int32_t>(model_input.lm_output_indexes));
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

    auto stream_groups = StreamGroups({stream1, stream2});

    auto processor = MtpBatchStreamProcessor(
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

    processor.updatePrefillPostDraftModelInput(stream_groups, model_input, model_output, sampler_output, holder);

    auto        combo_tokens        = model_input.combo_tokens;
    vector<int> expect_combo_tokens = {2, 2, 3};
    EXPECT_EQ(expect_combo_tokens, toVec<int>(combo_tokens));
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
    stream1->setContextPositionIds(torch::tensor({100, 101, 102, 110, 111, 112}, torch::kInt32));
    stream2->setContextPositionIds(torch::tensor({200, 201, 202, 210, 211, 212, 220, 221, 222}, torch::kInt32));
    auto           stream_groups = StreamGroups({stream1, stream2});
    GptModelInputs model_input;
    model_input.input_lengths = torch::tensor({2, 3}, torch::kInt32);
    model_input.combo_tokens  = torch::tensor({10, 11, 20, 21, 22}, torch::kInt32);
    model_input.combo_position_ids =
        torch::tensor({100, 101, 102, 110, 111, 112, 200, 201, 202, 210, 211, 212, 220, 221, 222}, torch::kInt32);
    GptModelOutputs model_output;
    model_output.all_hidden_states =
        torch::tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f}, torch::kFloat32).reshape({5, 2});
    SamplerOutput sampler_output;
    sampler_output.token_ids = torch::tensor({1, -2, 12, 1, 2, 23}, torch::kInt32).reshape({2, 3});
    TensorHolder holder;
    processor.updatePrefillPostDraftModelInput(stream_groups, model_input, model_output, sampler_output, holder);
    EXPECT_EQ((vector<int>{11, 12, 21, 22, 23}), toVec<int>(model_input.combo_tokens));
    EXPECT_EQ((vector<int>{110, 111, 112, 112, 112, 112, 210, 211, 212, 220, 221, 222, 222, 222, 222}),
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

    model_input.text_tokens_mask = torch::ones({6}, torch::kInt32);
    processor.updateDecodePostDraftModelInput(
        model_input, model_output, spec_decode_output, 2, hidden_states_d_t, holder);
    EXPECT_EQ((vector<int>{1, 1, 1, 1}), toVec<int>(model_input.text_tokens_mask));

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

    model_input.text_tokens_mask = torch::ones({6}, torch::kInt32).cuda();
    processor.updateDecodePostDraftModelInput(
        model_input, model_output, spec_decode_output, 2, hidden_states_d_t, holder);
    EXPECT_EQ((vector<int>{1, 1, 1, 1, 1, 1}), toVec<int>(model_input.text_tokens_mask));

    EXPECT_TRUE(model_input.combo_tokens.is_cuda());
    EXPECT_EQ((vector<int>{2, 3, 1, 2, 0, 0}), toVec<int>(model_input.combo_tokens));
    EXPECT_TRUE(model_input.lm_output_indexes.is_cuda());
    EXPECT_EQ((vector<int>{2, 3}), toVec<int>(model_input.lm_output_indexes));
    EXPECT_EQ(6, model_input.last_hidden_states.size(0));
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

    model_input.text_tokens_mask = torch::ones({9}, torch::kInt32);
    processor.updateDecodePostDraftModelInput(
        model_input, model_output, spec_decode_output, 3, hidden_states_d_t, holder);
    EXPECT_EQ((vector<int>{1, 1, 1, 1, 1, 1}), toVec<int>(model_input.text_tokens_mask));

    auto        combo_position_ids        = model_input.combo_position_ids;
    vector<int> expect_combo_position_ids = {10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 70, 71, 72};
    EXPECT_EQ(expect_combo_position_ids, toVec<int>(combo_position_ids));
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

    GptModelInputs model_input;
    model_input.combo_tokens       = torch::tensor({11, 21}, torch::kInt32);
    model_input.sequence_lengths   = torch::tensor({5, 7}, torch::kInt32);
    model_input.combo_position_ids = torch::tensor({5, 6, 7, 10, 11, 12}, torch::kInt32);

    GptModelOutputs model_output;
    model_output.all_hidden_states = torch::tensor({0.1f, 0.2f, 1.1f, 1.2f}, torch::kFloat32).reshape({2, 2});
    auto draft_token_ids           = torch::tensor({12, 22}, torch::kInt32).reshape({2, 1});

    TensorHolder holder;
    processor.updateDecodeDraftModelInput(model_input, model_output, draft_token_ids, holder);

    EXPECT_EQ((vector<int>{12, 22}), toVec<int>(model_input.combo_tokens));
    EXPECT_EQ((vector<int>{6, 8}), toVec<int>(model_input.sequence_lengths));
    EXPECT_EQ((vector<float>{0.1, 0.2, 1.1, 1.2}), toVec<float>(model_input.last_hidden_states));
    EXPECT_EQ((vector<int>{6, 7, 8, 11, 12, 13}), toVec<int>(model_input.combo_position_ids));
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

namespace {

MtpBatchStreamProcessor makeMultimodalMtpProcessor(const std::string& draft_model_type = "qwen35_moe_mtp",
                                                  SpeculativeType sp_type = SP_TYPE_EAGLE) {
    ModelConfig model_config;
    model_config.model_type = "qwen35_moe";
    model_config.vocab_size = 32;
    model_config.max_seq_len = 4096;
    model_config.num_layers = 1;
    SpeculativeExecutionConfig sp_config;
    sp_config.type = sp_type;
    sp_config.model_type = draft_model_type;
    return MtpBatchStreamProcessor(model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{},
                                   MtpBatchStreamProcessorTest::makeProcessorCacheConfig(), sp_config, false);
}

GptModelInputs multimodalMtpInput(const vector<int>& lengths,
                                  const vector<int>& mask,
                                  const vector<torch::Tensor>& features,
                                  const vector<int>& locs) {
    GptModelInputs input;
    input.combo_tokens = torch::arange(mask.size(), torch::kInt32);
    input.input_lengths = torch::tensor(lengths, torch::kInt32);
    input.sequence_lengths = torch::empty({0}, torch::kInt32);
    input.prefix_lengths = torch::zeros({static_cast<int64_t>(lengths.size())}, torch::kInt32);
    input.text_tokens_mask = torch::tensor(mask, torch::kInt32);
    input.multimodal_features = features;
    input.mm_features_locs = torch::tensor(locs, torch::kInt32);
    return input;
}

}  // namespace

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalShiftsMaskWithinEachRequest) {
    auto processor = makeMultimodalMtpProcessor();
    auto input = multimodalMtpInput({4, 4}, {1, 0, 0, 1, 1, 1, 0, 1},
                                    {torch::ones({2, 2}), torch::ones({1, 2})}, {1, 6});
    const auto old_mask = input.text_tokens_mask;
    processor.alignPrefillMultimodalInputs(input, input.input_lengths);
    EXPECT_EQ((vector<int>{0, 0, 1, 1, 1, 0, 1, 1}), toVec<int>(input.text_tokens_mask));
    EXPECT_EQ((vector<int>{0, 5}), toVec<int>(input.mm_features_locs));
    EXPECT_EQ((vector<int>{1, 0, 0, 1, 1, 1, 0, 1}), toVec<int>(old_mask));
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalRejectsMaskLengthMismatch) {
    auto processor = makeMultimodalMtpProcessor();
    auto input = multimodalMtpInput({2, 2}, {1, 0, 1}, {torch::ones({1, 2})}, {1});
    EXPECT_ANY_THROW(processor.alignPrefillMultimodalInputs(input, input.input_lengths));
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalShiftsFeatureLocations) {
    auto processor = makeMultimodalMtpProcessor();
    const auto first = torch::arange(6).reshape({3, 2});
    const auto second = torch::arange(4).reshape({2, 2});
    auto input = multimodalMtpInput({4, 4}, {1, 0, 0, 0, 1, 1, 0, 0}, {first, second}, {1, 6});
    processor.alignPrefillMultimodalInputs(input, input.input_lengths);
    EXPECT_EQ((vector<int>{0, 5}), toVec<int>(input.mm_features_locs));
    ASSERT_EQ(input.multimodal_features->size(), 2);
    EXPECT_TRUE(torch::equal(input.multimodal_features.value()[0], first));
    EXPECT_TRUE(torch::equal(input.multimodal_features.value()[1], second));
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalDropsRowAtRequestBoundary) {
    auto processor = makeMultimodalMtpProcessor();
    const auto feature = torch::arange(6).reshape({3, 2});
    auto input = multimodalMtpInput({4, 4}, {1, 1, 1, 1, 0, 0, 0, 1}, {feature}, {4});
    processor.alignPrefillMultimodalInputs(input, input.input_lengths);
    EXPECT_EQ((vector<int>{4}), toVec<int>(input.mm_features_locs));
    EXPECT_TRUE(torch::equal(input.multimodal_features.value()[0], feature.slice(0, 1)));
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalCropsPartiallyReusedFeature) {
    auto processor = makeMultimodalMtpProcessor();
    const auto feature = torch::arange(3714 * 2).reshape({3714, 2});
    // Gatherer owns the prefix crop: original loc 1, reused prefix 1844.
    // Its canonical output is the last 1871 rows at local position zero.
    auto input = multimodalMtpInput({1871}, vector<int>(1871, 0), {feature.slice(0, 1843)}, {0});
    processor.alignPrefillMultimodalInputs(input, input.input_lengths);
    EXPECT_EQ((vector<int>{0}), toVec<int>(input.mm_features_locs));
    ASSERT_EQ(input.multimodal_features->size(), 1);
    EXPECT_EQ(input.multimodal_features.value()[0].size(0), 1870);
    EXPECT_TRUE(torch::equal(input.multimodal_features.value()[0], feature.slice(0, 1844)));
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalPartialReuseAcrossRequests) {
    auto processor = makeMultimodalMtpProcessor();
    const auto first = torch::arange(4).reshape({2, 2});
    const auto second = torch::arange(8).reshape({4, 2});
    // The second image has already lost two rows to prefix reuse in the gatherer.
    auto input = multimodalMtpInput({4, 4}, {1, 0, 0, 1, 0, 0, 1, 1}, {first, second.slice(0, 2)}, {1, 4});
    processor.alignPrefillMultimodalInputs(input, input.input_lengths);
    EXPECT_EQ((vector<int>{0, 4}), toVec<int>(input.mm_features_locs));
    EXPECT_TRUE(torch::equal(input.multimodal_features.value()[0], first));
    EXPECT_TRUE(torch::equal(input.multimodal_features.value()[1], second.slice(0, 3)));
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalRejectsFeatureMaskMismatch) {
    auto processor = makeMultimodalMtpProcessor();
    auto input = multimodalMtpInput({4}, {1, 1, 1, 1}, {torch::ones({2, 2})}, {1});
    EXPECT_ANY_THROW(processor.alignPrefillMultimodalInputs(input, input.input_lengths));
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalDropsLoneVisualTokenAtRequestStart) {
    auto processor = makeMultimodalMtpProcessor();
    auto input = multimodalMtpInput({2}, {0, 1}, {torch::ones({1, 2})}, {0});
    input.mm_extra_input = vector<torch::Tensor>{torch::ones({4})};
    processor.alignPrefillMultimodalInputs(input, input.input_lengths);
    EXPECT_TRUE(input.multimodal_features->empty());
    EXPECT_TRUE(input.mm_extra_input->empty());
    EXPECT_EQ(input.mm_features_locs.numel(), 0);
    EXPECT_EQ((vector<int>{1, 1}), toVec<int>(input.text_tokens_mask));
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalDropsEmptyFeatureAtRequestStart) {
    auto processor = makeMultimodalMtpProcessor();
    auto input = multimodalMtpInput({1}, {1}, {torch::empty({0, 2})}, {0});
    processor.alignPrefillMultimodalInputs(input, input.input_lengths);
    EXPECT_TRUE(input.multimodal_features->empty());
    EXPECT_EQ(input.mm_features_locs.numel(), 0);
    EXPECT_EQ((vector<int>{1}), toVec<int>(input.text_tokens_mask));
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalDeepstackUsesSameFeatureRows) {
    auto processor = makeMultimodalMtpProcessor();
    const auto feature = torch::arange(6).reshape({3, 2});
    const auto extra = torch::arange(12).reshape({2, 3, 2});
    auto input = multimodalMtpInput({4}, {0, 0, 0, 1}, {feature}, {0});
    input.mm_extra_input = vector<torch::Tensor>{extra.reshape({-1})};
    processor.alignPrefillMultimodalInputs(input, input.input_lengths);
    EXPECT_TRUE(torch::equal(input.multimodal_features.value()[0], feature.slice(0, 1)));
    ASSERT_EQ(input.mm_extra_input->size(), 1);
    EXPECT_TRUE(torch::equal(input.mm_extra_input.value()[0], extra.slice(1, 1).contiguous().reshape({-1})));
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalGateAcceptsQwenDraftsUnderEagleAndMtp) {
    for (const auto& draft : {"qwen35_moe_mtp", "qwen35_dense_mtp", "qwen3_next_mtp"}) {
        for (const auto type : {SP_TYPE_EAGLE, SP_TYPE_MTP}) {
            auto processor = makeMultimodalMtpProcessor(draft, type);
            auto input = multimodalMtpInput({2}, {0, 1}, {torch::ones({1, 2})}, {0});
            processor.alignPrefillMultimodalInputs(input, input.input_lengths);
            EXPECT_EQ((vector<int>{1, 1}), toVec<int>(input.text_tokens_mask));
            EXPECT_TRUE(input.multimodal_features->empty());
        }
    }
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalGateKeepsOtherDraftContracts) {
    const vector<std::pair<std::string, SpeculativeType>> cases = {
        {"qwen_3_moe_eagle3", SP_TYPE_EAGLE}, {"qwen35_moe_mtp", SP_TYPE_DSPARK},
        {"deepseek-v3-mtp", SP_TYPE_MTP}};
    for (const auto& [model_type, sp_type] : cases) {
        auto processor = makeMultimodalMtpProcessor(model_type, sp_type);
        auto input = multimodalMtpInput({2}, {0, 1}, {torch::ones({1, 2})}, {0});
        processor.alignPrefillMultimodalInputs(input, input.input_lengths);
        EXPECT_EQ((vector<int>{0, 1}), toVec<int>(input.text_tokens_mask));
        EXPECT_EQ(input.multimodal_features->size(), 1);
        EXPECT_EQ((vector<int>{0}), toVec<int>(input.mm_features_locs));
    }
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalCppShiftPrecedesCpAdjacentRuns) {
    auto processor = makeMultimodalMtpProcessor();
    vector<int> mask(16, 1);
    std::fill(mask.begin() + 2, mask.begin() + 14, 0);
    const auto feature = torch::arange(24).reshape({12, 2});
    auto input = multimodalMtpInput({16}, mask, {feature}, {2});
    processor.alignPrefillMultimodalInputs(input, input.input_lengths);
    // Token shift is covered by the public-entry test below; emulate its output.
    input.combo_tokens = torch::arange(1, 17, torch::kInt32);
    ParallelismConfig cp_config;
    cp_config.tp_size = 2;
    cp_config.tp_rank = 0;
    ZigZagProcessor cp(cp_config);
    torch_ext::PyContextParallelParams cp_params;
    cp.handleInputs(input, cp_params);
    EXPECT_EQ((vector<int>{0, 1, 2, 3, 12, 13, 14, 15}), toVec<int>(cp_params.prefill_shuffle_indices));
    EXPECT_EQ((vector<int>{1, 0, 0, 0, 0, 1, 1, 1}), toVec<int>(input.text_tokens_mask));
    EXPECT_EQ((vector<int>{1, 4}), toVec<int>(input.mm_features_locs));
    ASSERT_EQ(input.multimodal_features->size(), 2);
    EXPECT_TRUE(torch::equal(input.multimodal_features.value()[0].cpu(), feature.slice(0, 0, 3)));
    EXPECT_TRUE(torch::equal(input.multimodal_features.value()[1].cpu(), feature.slice(0, 11, 12)));
}

TEST_F(MtpBatchStreamProcessorTest, testMtpMultimodalCppShiftPreservesCpChunkBoundary) {
    auto processor = makeMultimodalMtpProcessor();
    vector<int> mask(16, 1);
    std::fill(mask.begin() + 2, mask.begin() + 5, 0);
    const auto feature = torch::arange(6).reshape({3, 2});
    auto input = multimodalMtpInput({16}, mask, {feature}, {2});
    processor.alignPrefillMultimodalInputs(input, input.input_lengths);
    input.combo_tokens = torch::arange(1, 17, torch::kInt32);
    ParallelismConfig cp_config;
    cp_config.tp_size = 2;
    cp_config.tp_rank = 0;
    ZigZagProcessor cp(cp_config);
    torch_ext::PyContextParallelParams cp_params;
    cp.handleInputs(input, cp_params);
    // Local shifting loses the visual token at index 3; global shifting keeps it.
    EXPECT_EQ((vector<int>{1, 0, 0, 0, 1, 1, 1, 1}), toVec<int>(input.text_tokens_mask));
    ASSERT_EQ(input.multimodal_features->size(), 1);
    EXPECT_TRUE(torch::equal(input.multimodal_features.value()[0].cpu(), feature));
}

TEST_F(MtpBatchStreamProcessorTest, testUpdatePrefillPostDraftModelInputAlignsMultimodalFieldsOnce) {
    auto processor = makeMultimodalMtpProcessor();
    ModelConfig model_config;
    model_config.vocab_size = 32;
    model_config.max_seq_len = 16;
    model_config.num_layers = 1;
    RuntimeConfig runtime_config;
    ResourceContext resource_context;
    auto stream1 = createContextStream(model_config, runtime_config, resource_context, {1, 2, 3, 4}, 1);
    auto stream2 = createContextStream(model_config, runtime_config, resource_context, {5, 6, 7, 8}, 2);
    const StreamGroups streams({stream1, stream2});
    const auto first = torch::tensor({20.0f, 21.0f, 22.0f, 23.0f}).reshape({2, 2});
    const auto second = torch::tensor({30.0f, 31.0f}).reshape({1, 2});
    auto input = multimodalMtpInput({4, 4}, {1, 0, 0, 1, 1, 1, 0, 1}, {first, second}, {1, 6});
    input.combo_tokens = torch::tensor({1, 1000, 1001, 2, 4, 5, 1002, 6}, torch::kInt32);
    GptModelOutputs output;
    output.all_hidden_states = torch::zeros({8, 2});
    SamplerOutput sampled;
    sampled.token_ids = torch::tensor({3, 7}, torch::kInt32).reshape({2, 1});
    TensorHolder holder;
    processor.updatePrefillPostDraftModelInput(streams, input, output, sampled, holder);
    EXPECT_EQ((vector<int>{1000, 1001, 2, 3, 5, 1002, 6, 7}), toVec<int>(input.combo_tokens));
    EXPECT_EQ((vector<int>{0, 0, 1, 1, 1, 0, 1, 1}), toVec<int>(input.text_tokens_mask));
    EXPECT_EQ((vector<int>{0, 5}), toVec<int>(input.mm_features_locs));
    EXPECT_TRUE(torch::equal(input.multimodal_features.value()[0], first));
    EXPECT_TRUE(torch::equal(input.multimodal_features.value()[1], second));
}
}  // namespace rtp_llm
