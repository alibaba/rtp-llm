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
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"
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

void fillScoreTokenIdsWithMemcpy(torch::Tensor&                     token_ids,
                                 const std::vector<torch::Tensor>&  complete_token_ids,
                                 const std::vector<int64_t>&        seq_lens,
                                 int64_t                            score_len) {
    int64_t batch_idx = 0;
    auto*   dst       = token_ids.data_ptr<int32_t>();
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

void fillScoreTokenIdsWithTorchCopy(torch::Tensor&                     token_ids,
                                    const std::vector<torch::Tensor>&  complete_token_ids,
                                    const std::vector<int64_t>&        seq_lens,
                                    int64_t                            score_len) {
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
    GenerateStreamPtr createContextStream(const ModelConfig&     model_config,
                                          const RuntimeConfig&   runtime_config,
                                          const ResourceContext& resource_context,
                                          const vector<int>&     input_ids,
                                          const int              block_id,
                                          const vector<int>&     begin_think_token_ids = {},
                                          const vector<int>&     end_think_token_ids   = {}) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids       = torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
        query->generate_config = make_shared<GenerateConfig>();
        query->generate_config->begin_think_token_ids = begin_think_token_ids;
        query->generate_config->end_think_token_ids   = end_think_token_ids;
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
        sp_output_buffer->tokens     = torch::tensor(propose_tokens, torch::kInt32).reshape({1, 2});
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

    auto memcpy_us =
        benchmarkUs([&]() { fillScoreTokenIdsWithMemcpy(dst_memcpy, complete_token_ids, seq_lens, score_len); },
                    iterations);
    auto torch_us =
        benchmarkUs([&]() { fillScoreTokenIdsWithTorchCopy(dst_torch, complete_token_ids, seq_lens, score_len); },
                    iterations);

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
    const int64_t logical_score_rows =
        static_cast<int64_t>(stream_groups.size() * (sp_config.gen_num_per_cycle + 1));
    model_output.logits = torch::arange(
                              0,
                              (logical_score_rows + 4) * 4,
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .reshape({logical_score_rows + 4, 4});

    auto sampler_inputs_status = processor.gatherSpecSamplerInput(stream_groups, model_inputs, model_output);
    ASSERT_TRUE(sampler_inputs_status.ok());

    const auto& sampler_inputs = sampler_inputs_status.value();
    auto        token_ids      = sampler_inputs.token_ids;
    auto stride    = token_ids.size(1);
    auto* data     = token_ids.data_ptr<int32_t>();

    for (int64_t row = 0; row < 4; ++row) {
        EXPECT_EQ(5, data[row * stride]);
    }
    for (int64_t row = 4; row < 8; ++row) {
        EXPECT_EQ(6, data[row * stride]);
        EXPECT_EQ(7, data[row * stride + 1]);
    }
    EXPECT_EQ(sampler_inputs.logits.size(0), logical_score_rows);
    EXPECT_EQ(sampler_inputs.logits[-1][-1].item<float>(), 31.0f);
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
    model_output.logits = torch::zeros({3, 16}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Stateful processors supply verify masks through the executor's runner;
    // score-batch gathering intentionally skips their ordinary process().
    SpecLogitsVerifyRunner             verify_runner;
    SpecLogitsVerifyRunner::LaunchTask verify_task;
    verify_task.total_streams = 1;
    verify_task.propose_step  = sp_config.gen_num_per_cycle;
    verify_task.vocab_size    = model_config.vocab_size;
    verify_task.draft_tokens  = torch::tensor({{1, 2}}, torch::kInt32);
    size_t processor_idx = 0;
    for (const auto& logits_processor : stream->getAllLogitsProcessorPtr()) {
        if (auto spec_processor = std::dynamic_pointer_cast<SpecLogitsProcessor>(logits_processor)) {
            verify_task.active.push_back({spec_processor,
                                          0,
                                          processor_idx,
                                          static_cast<uint64_t>(stream->streamId()),
                                          static_cast<int64_t>(stream->seqLength()),
                                          static_cast<int64_t>(stream->outputTokenLen())});
        }
        ++processor_idx;
    }
    ASSERT_FALSE(verify_task.active.empty());
    auto verify_result = verify_runner.buildInline(verify_task);
    ASSERT_TRUE(verify_result.has_active_processor);
    ASSERT_TRUE(verify_result.spec_vocab_mask_gpu.defined());

    auto sampler_inputs_status =
        processor.gatherSpecSamplerInput(stream_groups, model_input, model_output, verify_result);
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

TEST_F(MtpBatchStreamProcessorTest, testLinearReplayUsesPerGroupPhysicalBlockSize) {
    ModelConfig model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 128;
    model_config.num_layers  = 6;
    RuntimeConfig              runtime_config;
    SpeculativeExecutionConfig sp_config;
    sp_config.gen_num_per_cycle = 3;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    auto                        cache_config = test::makeSimpleHybridMhaCacheConfig(6, 64, 16, DataType::TYPE_FP16, 2);
    cache_config.kernel_seq_size_per_block   = 4;
    cache_config.linear_replay_group_ids     = {0, 1};
    cache_config.group_types[1]              = CacheGroupType::LINEAR;
    auto first_spec                 = std::dynamic_pointer_cast<LinearKVCacheSpec>(cache_config.cache_specs[0]);
    first_spec->seq_size_per_block  = 8;
    auto second_spec                = std::make_shared<LinearKVCacheSpec>(*first_spec);
    second_spec->seq_size_per_block = 4;
    cache_config.cache_specs[1]     = second_spec;

    ResourceContext resource_context;
    // This test supplies a published lease and page snapshot; it only executes metadata gathering.
    resource_context.cache_manager = std::make_shared<KVCacheManager>(
        test::makeSimpleMhaCacheConfig(1, 64, 16, DataType::TYPE_FP16), /*warmup=*/true);
    auto stream = createContextStream(model_config, runtime_config, resource_context, std::vector<int>(17, 1), 1);
    stream->setIsContextStream(false);
    auto lease                                                       = std::make_shared<LinearReplayLease>();
    lease->slot_id                                                   = 0;
    lease->generation                                                = 1;
    stream->stream_cache_resource_->linear_replay_lease_             = lease;
    stream->stream_cache_resource_->linear_replay_initial_block_ids_ = {12, 24, -1};

    BatchKVCacheResource pages;
    pages.resetBatchSize(1);
    pages.initGroups(3, 3, {0, 1, 2}, cache_config.kernelBlocksPerKvBlock(), cache_config.group_types);
    pages.setBatchBlocks(0, 0, {11, 12, 13});
    pages.setBatchBlocks(0, 1, {21, 22, 23, 24, 25});
    pages.setBatchBlocks(0, 2, {31, 32});
    EXPECT_EQ(pages.kernelBlocks(0, 0), pages.blocks(0, 0));
    EXPECT_EQ(pages.kernelBlocks(0, 1), pages.blocks(0, 1));
    EXPECT_EQ(pages.kernelBlocks(0, 2).size(), 8u);
    stream->setKVCache(pages);

    GptModelInputs inputs;
    inputs.kv_cache_kernel_block_id =
        torch::full({3, 1, 8}, -1, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    for (int group = 0; group < 3; ++group) {
        const auto& kernel_blocks = pages.kernelBlocks(0, group);
        std::memcpy(inputs.kv_cache_kernel_block_id.data_ptr<int32_t>() + group * 8,
                    kernel_blocks.data(),
                    kernel_blocks.size() * sizeof(int32_t));
    }
    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder                                   host_holder;
    std::vector<GenerateStream::LinearReplayRound> rounds;
    ASSERT_TRUE(
        processor.gatherLinearReplayInputs(StreamGroups({stream}), cache_config, inputs, host_holder, rounds).ok());
    ASSERT_TRUE(inputs.linear_replay.has_value());
    EXPECT_EQ(toVec<int32_t>(inputs.linear_replay->active_block_ids), (std::vector<int32_t>{13, 25, -1}));
    EXPECT_EQ(toVec<int32_t>(inputs.linear_replay->state_read_block_ids), (std::vector<int32_t>{12, 24, -1}));
    EXPECT_EQ(toVec<int32_t>(inputs.linear_replay->anchor_processed_lengths), (std::vector<int32_t>{16}));
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
    target_output.sampler_output.all_probs = torch::tensor({{0.0f, 0.75f, 0.25f, 0.0f}, {0.0f, 0.0f, 0.2f, 0.8f}});

    MergedOutput draft_output;
    draft_output.model_output.all_hidden_states =
        torch::tensor({0.3f, 0.4f, 1.5f, 1.6f, 1.7f, 1.8f}, torch::kFloat32).reshape({3, 2});
    draft_output.sampler_output.token_ids = torch::tensor({2L, 0L}, torch::kInt64).reshape({2, 1});
    draft_output.sampler_output.all_probs =
        torch::tensor({0.2f, 0.1f, 0.3f, 0.5f, 0.3f, 0.1f, 0.4f, 0.2f}, torch::kFloat32).reshape({2, 4});

    stream1->generateConfig()->return_all_probs = true;
    stream1->generateConfig()->is_streaming     = true;
    auto status = processor.dispatchPrefill(stream_groups, target_output, draft_output);
    EXPECT_TRUE(status.ok());
    draft_output.model_output.all_hidden_states.fill_(9.0f);

    ASSERT_TRUE(stream1->hasOutput());
    auto prefill_result = stream1->nextOutput();
    ASSERT_TRUE(prefill_result.ok());
    EXPECT_TRUE(torch::allclose(prefill_result.value().generate_outputs[0].aux_info.all_probs.value(),
                                target_output.sampler_output.all_probs.narrow(0, 0, 1)));
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
    // MtpExecutor publishes the next device state after this processor commits
    // stream output; dispatchDecode alone does not own that publication.
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

    stream1->getSPOutputBuffer()->tokens = torch::tensor(propose_tokens_1, torch::kInt32).reshape({1, 2});
    stream2->getSPOutputBuffer()->tokens = torch::tensor(propose_tokens_2, torch::kInt32).reshape({1, 2});

    auto stream_groups = StreamGroups({stream1, stream2});

    cache_config.group_types = {CacheGroupType::FULL};
    auto processor           = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input_status = processor.gatherDecodeModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input            = model_input_status.value();
    model_input.sequence_lengths = torch::tensor({1, 2}, torch::kInt32);
    model_input.sequence_lengths_host_for_log =
        torch::tensor({1, 2}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));

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
    ASSERT_TRUE(model_input.input_lengths_host_for_log.defined());
    EXPECT_EQ(expect_input_lengths, toVec<int>(model_input.input_lengths_host_for_log));
    ASSERT_TRUE(model_input.prefix_lengths_host_for_log.defined());
    EXPECT_EQ(expect_prefix_lengths, toVec<int>(model_input.prefix_lengths_host_for_log));
    EXPECT_FALSE(model_input.sequence_lengths_host_for_log.defined());

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

    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

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
    // Deliberately inconsistent with next_seq_len_gpu ({7, 4} -> prefix {6, 3}):
    // the device path must not republish this mirror.
    model_input.sequence_lengths_host_for_log =
        torch::tensor({98, 98}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    model_input.prefix_lengths_host_for_log = model_input.sequence_lengths_host_for_log;
    model_input.sequence_lengths_plus_1     = torch::full({2}, 100, cuda_i32);

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
    ASSERT_TRUE(model_input.input_lengths_host_for_log.defined());
    EXPECT_EQ(expect_input_lengths, toVec<int>(model_input.input_lengths_host_for_log));
    EXPECT_FALSE(model_input.prefix_lengths_host_for_log.defined());
    EXPECT_FALSE(model_input.sequence_lengths_host_for_log.defined());
    EXPECT_FALSE(model_input.sequence_lengths_plus_1.defined());

    // The next round must observe device updates without a host length mirror
    // or a stream bookkeeping update.
    auto next_seq_len_1 = stream1->getNextSeqLenGpu();
    auto next_seq_len_2 = stream2->getNextSeqLenGpu();
    next_seq_len_1.add_(2);
    next_seq_len_2.add_(1);
    processor.prepareOneStepSpecDecodeModelInput(stream_groups, model_input, holder);
    EXPECT_EQ(std::vector<int>({8, 4}), toVec<int>(model_input.prefix_lengths));
    EXPECT_EQ(expect_combo_tokens, toVec<int>(model_input.combo_tokens));
    EXPECT_FALSE(model_input.prefix_lengths_host_for_log.defined());
    EXPECT_FALSE(model_input.sequence_lengths_host_for_log.defined());
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

    stream1->getSPOutputBuffer()->tokens        = torch::tensor(propose_tokens_1, torch::kInt32).reshape({1, 2});
    stream2->getSPOutputBuffer()->tokens        = torch::tensor(propose_tokens_2, torch::kInt32).reshape({1, 2});
    stream1->getSPOutputBuffer()->hidden_states = torch::tensor({{0.1f, 0.2f}});
    stream2->getSPOutputBuffer()->hidden_states = torch::tensor({{1.1f, 1.2f}});

    auto stream_groups = StreamGroups({stream1, stream2});

    cache_config.group_types = {CacheGroupType::FULL};
    auto processor           = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;
    auto         model_input_status = processor.gatherDecodeModelInput(stream_groups, holder);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input                         = model_input_status.value();
    model_input.sequence_lengths              = torch::tensor({1, 2}, torch::kInt32);
    model_input.sequence_lengths_plus_1       = torch::tensor({2, 3}, torch::kInt32).to(torch::kCUDA);
    model_input.sequence_lengths_host_for_log = torch::tensor({1, 2}, torch::kInt32).pin_memory();

    processor.prepareDecodeDraftModelInput(stream_groups, model_input, holder);
    EXPECT_FALSE(model_input.sequence_lengths_plus_1.defined());

    auto        combo_tokens        = model_input.combo_tokens;
    vector<int> expect_combo_tokens = {3, 1};
    EXPECT_EQ(expect_combo_tokens, toVec<int>(combo_tokens));

    auto        lm_output_indexes        = model_input.lm_output_indexes;
    vector<int> expect_lm_output_indexes = {0, 1};
    EXPECT_TRUE(lm_output_indexes.is_cuda());
    EXPECT_EQ(expect_lm_output_indexes, toVec<int>(lm_output_indexes));

    auto expect_positions = [](const GptModelInputs& input,
                               const vector<int>&    expected_prefix,
                               const vector<int>&    expected_sequence,
                               bool                  has_host_mirror = true) {
        EXPECT_TRUE(input.prefix_lengths.is_cuda());
        EXPECT_TRUE(input.sequence_lengths.is_cuda());
        EXPECT_EQ(torch::kInt32, input.prefix_lengths.scalar_type());
        EXPECT_EQ(torch::kInt32, input.sequence_lengths.scalar_type());
        EXPECT_EQ(expected_prefix, toVec<int>(input.prefix_lengths));
        EXPECT_EQ(expected_sequence, toVec<int>(input.sequence_lengths));
        EXPECT_EQ(expected_sequence, toVec<int>(input.prefix_lengths + 1));
        if (has_host_mirror) {
            EXPECT_EQ(expected_prefix, toVec<int>(input.prefix_lengths_host_for_log));
            EXPECT_EQ(expected_sequence, toVec<int>(input.sequence_lengths_host_for_log));
            EXPECT_TRUE(input.prefix_lengths_host_for_log.is_pinned());
            EXPECT_TRUE(input.sequence_lengths_host_for_log.is_pinned());
        } else {
            EXPECT_FALSE(input.prefix_lengths_host_for_log.defined());
            EXPECT_FALSE(input.sequence_lengths_host_for_log.defined());
        }
    };
    expect_positions(model_input, {1, 2}, {2, 3});

    model_input.sequence_lengths_plus_1       = torch::tensor({2, 3}, torch::kInt32).to(torch::kCUDA);
    model_input.sequence_lengths_host_for_log = torch::tensor({2, 3}, torch::kInt32).pin_memory();
    auto original_sequence_lengths_host    = model_input.sequence_lengths_host_for_log;
    GptModelOutputs model_output;
    model_output.all_hidden_states = torch::zeros({2, 4}, torch::kFloat32).to(torch::kCUDA);
    processor.updateDecodeDraftModelInput(
        model_input, model_output, torch::tensor({1, 2}, torch::kInt32).to(torch::kCUDA), holder);
    EXPECT_FALSE(model_input.sequence_lengths_plus_1.defined());
    EXPECT_EQ(std::vector<int>({1, 2}), toVec<int>(model_input.prefix_lengths));
    EXPECT_EQ(std::vector<int>({1, 2}), toVec<int>(model_input.prefix_lengths_host_for_log));
    EXPECT_EQ(std::vector<int>({3, 4}), toVec<int>(model_input.sequence_lengths));
    EXPECT_EQ(std::vector<int>({3, 4}), toVec<int>(model_input.sequence_lengths_host_for_log));
    EXPECT_EQ(std::vector<int>({2, 3}), toVec<int>(original_sequence_lengths_host));
    EXPECT_TRUE(model_input.sequence_lengths_host_for_log.is_pinned());

    // Exercise the legacy CPU fallback as well. It must publish the new pinned
    // host mirror without mutating a snapshot retained by the previous step.
    model_input.sequence_lengths              = torch::tensor({3, 4}, torch::kInt32);
    model_input.sequence_lengths_plus_1       = torch::tensor({4, 5}, torch::kInt32).to(torch::kCUDA);
    model_input.sequence_lengths_host_for_log = torch::tensor({3, 4}, torch::kInt32).pin_memory();
    auto fallback_original_sequence_lengths_host = model_input.sequence_lengths_host_for_log;
    processor.updateDecodeDraftModelInput(
        model_input, model_output, torch::tensor({1, 2}, torch::kInt32).to(torch::kCUDA), holder);
    EXPECT_EQ(std::vector<int>({4, 5}), toVec<int>(model_input.sequence_lengths));
    EXPECT_EQ(std::vector<int>({4, 5}), toVec<int>(model_input.sequence_lengths_host_for_log));
    EXPECT_EQ(std::vector<int>({3, 4}), toVec<int>(fallback_original_sequence_lengths_host));
    EXPECT_TRUE(model_input.sequence_lengths_host_for_log.is_pinned());

    // Legacy GPU propose-token path receives the normal decode position.
    stream1->getSPOutputBuffer()->propose_tokens_gpu = torch::tensor({{3}}, torch::kInt32).to(torch::kCUDA);
    stream2->getSPOutputBuffer()->propose_tokens_gpu = torch::tensor({{1}}, torch::kInt32).to(torch::kCUDA);
    model_input.sequence_lengths              = torch::tensor({4, 5}, torch::kInt32);
    model_input.sequence_lengths_host_for_log = torch::tensor({4, 5}, torch::kInt32).pin_memory();
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

    model_input.sequence_lengths              = torch::tensor({99, 99}, torch::kInt32);
    model_input.sequence_lengths_host_for_log = torch::tensor({98, 98}, torch::kInt32).pin_memory();
    model_input.prefix_lengths_host_for_log   = model_input.sequence_lengths_host_for_log;
    model_input.sequence_lengths_plus_1       = torch::tensor({100, 100}, torch::kInt32).to(torch::kCUDA);
    processor.prepareDecodeDraftModelInput(stream_groups, model_input, holder);

    expect_positions(model_input, {6, 3}, {7, 4}, false);
    EXPECT_FALSE(model_input.sequence_lengths_plus_1.defined());

    processor.updateDecodeDraftModelInput(
        model_input, model_output, torch::tensor({1, 2}, torch::kInt32).to(torch::kCUDA), holder);
    EXPECT_EQ(std::vector<int>({6, 3}), toVec<int>(model_input.prefix_lengths));
    EXPECT_EQ(std::vector<int>({8, 5}), toVec<int>(model_input.sequence_lengths));
    EXPECT_FALSE(model_input.prefix_lengths_host_for_log.defined());
    EXPECT_FALSE(model_input.sequence_lengths_host_for_log.defined());
    EXPECT_FALSE(model_input.sequence_lengths_plus_1.defined());
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
        model_input, model_output, spec_decode_output, 2, 2, hidden_states_d_t, holder);

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

TEST_F(MtpBatchStreamProcessorTest, testUpdateDecodePostDraftModelInputPadsPhysicalBatch) {
    ModelConfig                 model_config;
    RuntimeConfig               runtime_config;
    SpeculativeExecutionConfig  sp_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    sp_config.gen_num_per_cycle = 3;

    auto processor = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    TensorHolder holder;

    GptModelInputs model_input;
    model_input.input_lengths           = torch::tensor({4}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    model_input.sequence_lengths        = torch::tensor({11}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    model_input.prefix_lengths          = torch::tensor({10}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    model_input.sequence_lengths_plus_1 = torch::tensor({12}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    model_input.kv_cache_kernel_block_id =
        torch::tensor({{{7, 8}}}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    speculative::SpeculativeSamplerOutput spec_decode_output;
    spec_decode_output.accept_len =
        torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    spec_decode_output.accept_tokens =
        torch::tensor({{31, 32, 0, 0}}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    GptModelOutputs model_output;
    model_output.all_hidden_states = torch::arange(
                                         0,
                                         16,
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                                         .reshape({8, 2});
    torch::Tensor hidden_states_d_t;

    processor.updateDecodePostDraftModelInput(
        model_input, model_output, spec_decode_output, 1, 2, hidden_states_d_t, holder);

    EXPECT_EQ(toVec<int>(model_input.combo_tokens.cpu()), (vector<int>{31, 32, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(toVec<int>(model_input.lm_output_indexes.cpu()), (vector<int>{1, 7}));
    EXPECT_EQ(toVec<int>(model_input.input_lengths.cpu()), (vector<int>{4, 4}));
    EXPECT_EQ(toVec<int>(model_input.sequence_lengths.cpu()), (vector<int>{11, 0}));
    EXPECT_EQ(toVec<int>(model_input.prefix_lengths.cpu()), (vector<int>{10, 0}));
    EXPECT_EQ(toVec<int>(model_input.sequence_lengths_plus_1.cpu()), (vector<int>{12, 1}));
    EXPECT_EQ(toVec<int>(model_input.kv_cache_kernel_block_id.cpu()), (vector<int>{7, 8, 0, 0}));
    EXPECT_EQ(model_input.last_hidden_states.size(0), 8);
    EXPECT_EQ(hidden_states_d_t.size(0), 8);
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

}  // namespace rtp_llm
