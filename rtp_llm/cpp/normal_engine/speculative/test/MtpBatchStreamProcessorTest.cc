#include <memory>
#include "torch/all.h"
#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

#define private public
#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#undef private
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;

namespace rtp_llm {

class MtpBatchStreamProcessorTest: public DeviceTestBase {
public:
    GenerateStreamPtr createContextStream(const ModelConfig&     model_config,
                                          const RuntimeConfig&   runtime_config,
                                          const ResourceContext& resource_context,
                                          const vector<int>&     input_ids,
                                          const int              block_id) {
        std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
        query->input_ids       = createBuffer<int32_t>({input_ids.size()}, input_ids, AllocationType::HOST);
        query->generate_config = make_shared<GenerateConfig>();
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
        sp_output_buffer->tokens     = createBuffer<int>({1, 2}, propose_tokens, AllocationType::HOST);
        stream->setReturnAllProbs(ReturnAllProbsMode::DEFAULT);
        stream->setSPOutputBuffer(sp_output_buffer);
        stream->setRunning();
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
        auto tokens_h         = device_->clone({*tokens, AllocationType::HOST});
        EXPECT_EQ(expect_propose_tokens, buffer2vector<int>(*tokens_h));

        auto all_probs   = sp_output_buffer->all_probs;
        auto all_probs_h = device_->clone({*all_probs, AllocationType::HOST});
        EXPECT_EQ(expect_all_probs, buffer2vector<float>(*all_probs_h));

        auto last_hidden_states   = sp_output_buffer->hidden_states;
        auto last_hidden_states_h = device_->clone({*last_hidden_states, AllocationType::HOST});
        EXPECT_EQ(expect_last_hidden_states, buffer2vector<float>(*last_hidden_states_h));
    }
};

TEST_F(MtpBatchStreamProcessorTest, testPrefillDispatch) {
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

    ResourceContext resource_context;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {2}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {1, 2}, 2);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    processor.setKVCacheGroupTypes({CacheGroupType::FULL});

    StreamGroups stream_groups(streams);

    MergedOutput target_output;
    target_output.model_output.all_hidden_states = createBuffer<float>({3, 2}, {0.1, 0.2, 1.1, 1.2, 1.3, 1.4});
    target_output.sampler_output.token_ids       = createBuffer<int>({2, 3}, {2, -1, 1, 1, 2, 3}, AllocationType::HOST);
    target_output.sampler_output.all_probs = createBuffer<float>({2, 2}, {0.1, 0.9, 0.2, 0.8}, AllocationType::HOST);

    MergedOutput draft_output;
    draft_output.model_output.all_hidden_states = createBuffer<float>({3, 2}, {0.3, 0.4, 1.5, 1.6, 1.7, 1.8});
    draft_output.sampler_output.token_ids       = createBuffer<int64_t>({2, 1}, {2, 0}, AllocationType::HOST);
    draft_output.sampler_output.all_probs =
        createBuffer<float>({2, 4}, {0.2, 0.1, 0.3, 0.5, 0.3, 0.1, 0.4, 0.2}, AllocationType::HOST);

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
    CacheConfig                 cache_config;

    model_config.max_seq_len    = 2048;
    model_config.vocab_size     = 4;
    model_config.vocab_size     = 4;
    model_config.num_layers     = 1;
    sp_config.gen_num_per_cycle = 4;

    ResourceContext resource_context;

    GenerateStreamPtr stream1 = createContextStream(model_config, runtime_config, resource_context, {1}, 1);
    GenerateStreamPtr stream2 = createContextStream(model_config, runtime_config, resource_context, {2, 1}, 2);

    auto stream_groups = StreamGroups({stream1, stream2});

    speculative::SpeculativeSamplerOutput spec_decode_output;
    spec_decode_output.accept_len = {5, 1};

    spec_decode_output.accept_tokens = {createBuffer<int>({1, 5}, {2, 3, 1, 3, 2}, AllocationType::HOST),
                                        createBuffer<int>({1, 1}, {2}, AllocationType::HOST)};

    MergedOutput draft_prefill_output;
    draft_prefill_output.model_output.all_hidden_states =
        createBuffer<float>({6, 2}, {0.2, 0.02, 0.3, 0.03, 0.4, 0.04, 0.5, 0.05, 0.6, 0.06, 1.3, 0.13});
    draft_prefill_output.sampler_output.token_ids = createBuffer<int64_t>({2, 1}, {0, 3}, AllocationType::HOST);
    draft_prefill_output.sampler_output.all_probs =
        createBuffer<float>({2, 4}, {0.2, 0.1, 0.3, 0.5, 0.3, 0.1, 0.4, 0.2}, AllocationType::HOST);

    MtpBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    processor.setKVCacheGroupTypes({CacheGroupType::FULL});

    auto status = processor.dispatchDecode(stream_groups, spec_decode_output, std::move(draft_prefill_output));
    EXPECT_TRUE(status.ok());

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
                                                          device_,
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

    stream1->getSPOutputBuffer()->hidden_states = createBuffer<float>({1, 2}, {0.1, 0.2});
    stream2->getSPOutputBuffer()->hidden_states = createBuffer<float>({1, 2}, {1.1, 1.2});

    auto stream_groups = StreamGroups({stream1, stream2});

    auto processor = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    processor.setKVCacheGroupTypes({CacheGroupType::FULL});
    auto model_input = processor.gatherDecodeModelInput(stream_groups);
    EXPECT_TRUE(model_input.ok());

    auto          last_hidden_states        = model_input.value().last_hidden_states;
    auto          last_hidden_states_h      = device_->clone({*last_hidden_states, AllocationType::HOST});
    vector<float> expect_last_hidden_states = {0.1, 0.2, 1.1, 1.2};
    EXPECT_EQ(expect_last_hidden_states, buffer2vector<float>(*last_hidden_states_h));
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
                                                          device_,
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

    BufferPtr context_token_1 = createBuffer<int>({1, 1}, {2}, AllocationType::HOST);
    BufferPtr context_token_2 = createBuffer<int>({1, 1}, {3}, AllocationType::HOST);

    stream1->update({context_token_1, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
    stream2->update({context_token_2, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});

    vector<int> propose_tokens_1 = {2, 3};
    vector<int> propose_tokens_2 = {3, 1};

    stream1->getSPOutputBuffer()->tokens = createBuffer<int>({1, 2}, propose_tokens_1, AllocationType::HOST);
    stream2->getSPOutputBuffer()->tokens = createBuffer<int>({1, 2}, propose_tokens_2, AllocationType::HOST);

    auto stream_groups = StreamGroups({stream1, stream2});

    auto processor = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    processor.setKVCacheGroupTypes({CacheGroupType::FULL});
    auto model_input_status = processor.gatherDecodeModelInput(stream_groups);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input            = model_input_status.value();
    model_input.sequence_lengths = createBuffer<int>({2}, {1, 2}, AllocationType::HOST);

    processor.prepareOneStepSpecDecodeModelInput(stream_groups, model_input);

    auto        combo_tokens        = model_input.combo_tokens;
    vector<int> expect_combo_tokens = {2, 3, 3, 1};
    EXPECT_EQ(expect_combo_tokens, buffer2vector<int>(*combo_tokens));

    auto        prefix_lengths        = model_input.prefix_lengths;
    vector<int> expect_prefix_lengths = {1, 2};
    EXPECT_EQ(expect_prefix_lengths, buffer2vector<int>(*prefix_lengths));

    auto        input_lengths        = model_input.input_lengths;
    vector<int> expect_input_lengths = {2, 2};
    EXPECT_EQ(expect_input_lengths, buffer2vector<int>(*input_lengths));

    auto sequence_lengths = model_input.sequence_lengths;
    EXPECT_EQ(0, sequence_lengths->shape()[0]);

    auto        lm_output_indexes        = model_input.lm_output_indexes;
    vector<int> expect_lm_output_indexes = {0, 1, 2, 3};
    EXPECT_EQ(expect_lm_output_indexes, buffer2vector<int>(*lm_output_indexes));
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
                                                          device_,
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

    BufferPtr context_token_1 = createBuffer<int>({1, 1}, {2}, AllocationType::HOST);
    BufferPtr context_token_2 = createBuffer<int>({1, 1}, {3}, AllocationType::HOST);

    stream1->update({context_token_1, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});
    stream2->update({context_token_2, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr});

    vector<int> propose_tokens_1 = {2, 3};
    vector<int> propose_tokens_2 = {3, 1};

    stream1->getSPOutputBuffer()->tokens        = createBuffer<int>({1, 2}, propose_tokens_1, AllocationType::HOST);
    stream2->getSPOutputBuffer()->tokens        = createBuffer<int>({1, 2}, propose_tokens_2, AllocationType::HOST);
    stream1->getSPOutputBuffer()->hidden_states = createBuffer<float>({1, 2}, {0.1, 0.2});
    stream2->getSPOutputBuffer()->hidden_states = createBuffer<float>({1, 2}, {1.1, 1.2});

    auto stream_groups = StreamGroups({stream1, stream2});

    auto processor = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);
    processor.setKVCacheGroupTypes({CacheGroupType::FULL});
    auto model_input_status = processor.gatherDecodeModelInput(stream_groups);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input            = model_input_status.value();
    model_input.sequence_lengths = createBuffer<int>({2}, {1, 2}, AllocationType::HOST);

    processor.prepareDecodeDraftModelInput(stream_groups, model_input);

    auto        combo_tokens        = model_input.combo_tokens;
    vector<int> expect_combo_tokens = {3, 1};
    EXPECT_EQ(expect_combo_tokens, buffer2vector<int>(*combo_tokens));

    auto        lm_output_indexes        = model_input.lm_output_indexes;
    vector<int> expect_lm_output_indexes = {0, 1};
    EXPECT_EQ(expect_lm_output_indexes, buffer2vector<int>(*lm_output_indexes));
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
                                                          device_,
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
    processor.setKVCacheGroupTypes({CacheGroupType::FULL});
    auto model_input_status = processor.gatherModelInput(stream_groups);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input            = model_input_status.value();
    model_input.sequence_lengths = createBuffer<int>({2}, {1, 2}, AllocationType::HOST);

    GptModelOutputs model_output;
    model_output.all_hidden_states = createBuffer<float>({3, 2}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, AllocationType::HOST);

    SamplerOutput sampler_output;
    sampler_output.token_ids = createBuffer<int>({2, 3}, {1, -2, 2, 1, 2, 3}, AllocationType::HOST);

    processor.updatePrefillPostDraftModelInput(model_input, model_output, sampler_output);

    auto        combo_tokens        = model_input.combo_tokens;
    vector<int> expect_combo_tokens = {2, 2, 3};
    EXPECT_EQ(expect_combo_tokens, buffer2vector<int>(*combo_tokens));
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
                                                          device_,
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
    processor.setKVCacheGroupTypes({CacheGroupType::FULL});
    auto model_input_status = processor.gatherModelInput(stream_groups);
    EXPECT_TRUE(model_input_status.ok());

    auto& model_input = model_input_status.value();

    speculative::SpeculativeSamplerOutput spec_decode_output;
    spec_decode_output.accept_len    = {3, 1};
    spec_decode_output.accept_tokens = {createBuffer<int>({1, 3}, {2, 3, 1}, AllocationType::HOST),
                                        createBuffer<int>({1, 1}, {2}, AllocationType::HOST)};

    torch::Tensor hidden_states_d_t;
    size_t        total_accept_len;

    GptModelOutputs model_output;
    model_output.all_hidden_states =
        createBuffer<float>({6, 2}, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6}, AllocationType::HOST);

    processor.updateDecodePostDraftModelInput(
        model_input, model_output, spec_decode_output, 2, hidden_states_d_t, total_accept_len);

    auto        combo_tokens        = model_input.combo_tokens;
    vector<int> expect_combo_tokens = {2, 3, 1, 2};
    EXPECT_EQ(expect_combo_tokens, buffer2vector<int>(*combo_tokens));

    auto        input_lengths        = model_input.input_lengths;
    vector<int> expect_input_lengths = {3, 1};
    EXPECT_EQ(expect_input_lengths, buffer2vector<int>(*input_lengths));

    auto        lm_output_indexes        = model_input.lm_output_indexes;
    vector<int> expect_lm_output_indexes = {2, 3};
    EXPECT_EQ(expect_lm_output_indexes, buffer2vector<int>(*lm_output_indexes));

    auto          last_hidden_states        = model_input.last_hidden_states;
    vector<float> expect_last_hidden_states = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.1, 1.2};
    EXPECT_EQ(expect_last_hidden_states, buffer2vector<float>(*last_hidden_states));
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
                                                          device_,
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

    stream1->getSPOutputBuffer()->all_probs = createBuffer<float>({1, 4}, {0.1, 0.2, 0.3, 0.4}, AllocationType::HOST);
    stream2->getSPOutputBuffer()->all_probs = createBuffer<float>({1, 4}, {0.5, 0.6, 0.7, 0.8}, AllocationType::HOST);
    stream1->getSPOutputBuffer()->tokens    = createBuffer<int>({1, 2}, {1, 2}, AllocationType::HOST);
    stream2->getSPOutputBuffer()->tokens    = createBuffer<int>({1, 2}, {2, 3}, AllocationType::HOST);

    auto stream_groups = StreamGroups({stream1, stream2});
    auto processor     = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    torch::Tensor draft_token_probs_d_t;
    SamplerOutput sampler_output;

    processor.updateOneStepDraftSamplerOutput(stream_groups, sampler_output, draft_token_probs_d_t);

    auto        token_ids        = sampler_output.token_ids;
    vector<int> expect_token_ids = {2, 3};
    EXPECT_EQ(expect_token_ids, buffer2vector<int>(*token_ids));

    auto          all_probs        = sampler_output.all_probs;
    vector<float> expect_all_probs = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    EXPECT_EQ(expect_all_probs, buffer2vector<float>(*all_probs));
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
                                                          device_,
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

    stream1->getSPOutputBuffer()->all_probs = createBuffer<float>({1, 4}, {0.1, 0.2, 0.3, 0.4}, AllocationType::HOST);
    stream2->getSPOutputBuffer()->all_probs = createBuffer<float>({1, 4}, {0.5, 0.6, 0.7, 0.8}, AllocationType::HOST);
    stream1->getSPOutputBuffer()->tokens    = createBuffer<int>({1, 2}, {1, 2}, AllocationType::HOST);
    stream2->getSPOutputBuffer()->tokens    = createBuffer<int>({1, 2}, {2, 3}, AllocationType::HOST);

    auto output_token_probs_1 =
        createBuffer<float>({2, 1, 4}, {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8}, AllocationType::HOST);
    auto output_token_probs_2 =
        createBuffer<float>({2, 1, 4}, {2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8}, AllocationType::HOST);

    auto draft_token_ids = createBuffer<int>({2, 4}, {2, 0, 1, 2, 3, 1, 2, 3}, AllocationType::HOST);

    auto stream_groups = StreamGroups({stream1, stream2});
    auto processor     = MtpBatchStreamProcessor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, sp_config, false);

    torch::Tensor              draft_token_probs_d_t;
    torch::Tensor              draft_token_ids_d_t = Buffer2torchTensor(draft_token_ids);
    torch::Tensor              spec_token_ids_d_t;
    std::vector<torch::Tensor> draft_token_probs_list;
    SamplerOutput              sampler_output;

    draft_token_probs_list.push_back(Buffer2torchTensor(output_token_probs_1));
    draft_token_probs_list.push_back(Buffer2torchTensor(output_token_probs_2));

    processor.updateMultiStepDraftSamplerOutput(stream_groups,
                                                sampler_output,
                                                draft_token_ids_d_t,
                                                spec_token_ids_d_t,
                                                draft_token_probs_d_t,
                                                draft_token_probs_list);

    auto        token_ids        = sampler_output.token_ids;
    vector<int> expect_token_ids = {0, 1, 2, 1, 2, 3};
    EXPECT_EQ(expect_token_ids, buffer2vector<int>(*token_ids));

    auto          all_probs        = sampler_output.all_probs;
    vector<float> expect_all_probs = {0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4,
                                      0.5, 0.6, 0.7, 0.8, 1.5, 1.6, 1.7, 1.8, 2.5, 2.6, 2.7, 2.8};
    EXPECT_EQ(expect_all_probs, buffer2vector<float>(*all_probs));
}

}  // namespace rtp_llm