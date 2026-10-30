#include "rtp_llm/cpp/observability/ExecutionRecorder.h"
#include <memory>
#include <optional>
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerUtils.h"
#include "torch/all.h"
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

using namespace std;

namespace rtp_llm {

template<typename T>
std::vector<T> toVec(const torch::Tensor& t) {
    auto c = t.is_cuda() ? t.cpu().contiguous() : t.contiguous();
    return std::vector<T>(c.data_ptr<T>(), c.data_ptr<T>() + c.numel());
}

static torch::Tensor hostIntBuffer(std::vector<int32_t> data) {
    return torch::tensor(data, torch::kInt32);
}

class NormalBatchStreamProcessorTest: public DeviceTestBase {
protected:
    static EngineInitParams makeExecutorParams() {
        EngineInitParams params;
        params.model_id                                   = 0;
        params.model_config_.max_seq_len                  = 128;
        params.model_config_.vocab_size                   = 128;
        params.model_config_.input_vocab_size             = 128;
        params.model_config_.num_layers                   = 1;
        params.model_config_.attn_config.head_num         = 2;
        params.model_config_.attn_config.kv_head_num      = 2;
        params.model_config_.attn_config.size_per_head    = 64;
        params.model_config_.hidden_size                  = 128;
        params.model_config_.attn_config.tokens_per_block = 2;
        params.py_model                                   = py::none();
        return params;
    }

    static ModelConfig makeLogprobsModelConfig(int vocab_size = 4) {
        ModelConfig model_config;
        model_config.max_seq_len                 = 16;
        model_config.vocab_size                  = vocab_size;
        model_config.num_layers                  = 1;
        model_config.special_tokens.eos_token_id = -1;
        return model_config;
    }

    static std::unique_ptr<NormalBatchStreamProcessor> makeLogprobsProcessor(const ModelConfig& model_config) {
        PDSepConfig                 pd_sep_config;
        ProfilingDebugLoggingConfig profiling_debug_logging_config;
        CacheConfig                 cache_config;
        cache_config.group_types = {CacheGroupType::FULL};
        return std::make_unique<NormalBatchStreamProcessor>(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    }

    static std::shared_ptr<NormalGenerateStream>
    makeLogprobsStream(const ModelConfig&                   model_config,
                       bool                                 return_logprobs,
                       int                                  top_logprobs,
                       int                                  max_new_tokens,
                       bool                                 is_streaming,
                       const std::vector<std::vector<int>>& stop_words_list     = {},
                       bool                                 in_think_mode       = false,
                       const std::vector<int>&              end_think_token_ids = {}) {
        auto query                 = std::make_shared<GenerateInput>();
        query->input_ids           = hostIntBuffer({0});
        query->generate_config     = std::make_shared<GenerateConfig>();
        auto& config               = *query->generate_config;
        config.return_logprobs     = return_logprobs;
        config.top_logprobs        = top_logprobs;
        config.max_new_tokens      = max_new_tokens;
        config.is_streaming        = is_streaming;
        config.ignore_eos          = true;
        config.stop_words_list     = stop_words_list;
        config.in_think_mode       = in_think_mode;
        config.end_think_token_ids = end_think_token_ids;
        ResourceContext resource_context;
        RuntimeConfig   runtime_config;
        auto            stream =
            std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    }

    static absl::Status dispatchLogprobsStep(const NormalBatchStreamProcessor&   processor,
                                             const std::list<GenerateStreamPtr>& streams,
                                             const torch::Tensor&                raw_logits,
                                             const std::vector<int32_t>&         sampled_tokens,
                                             std::shared_ptr<RecordedBatch>      recorded_batch = nullptr) {
        StreamGroups stream_groups(streams);
        MergedOutput outputs;
        outputs.recorded_batch      = std::move(recorded_batch);
        outputs.model_output.logits = raw_logits.to(torch::kCUDA);
        outputs.sampler_output.token_ids =
            torch::tensor(sampled_tokens, torch::kInt32).reshape({(int64_t)sampled_tokens.size(), 1});
        return processor.dispatch(stream_groups, outputs);
    }

    static void
    expectFloatTensorNear(const torch::Tensor& actual, const torch::Tensor& expected, float tolerance = 1e-5f) {
        auto actual_vec   = toVec<float>(actual);
        auto expected_vec = toVec<float>(expected);
        ASSERT_EQ(actual_vec.size(), expected_vec.size());
        for (size_t i = 0; i < actual_vec.size(); ++i) {
            EXPECT_NEAR(actual_vec[i], expected_vec[i], tolerance) << "element " << i;
        }
    }
};

class TestStatefulLogitsProcessor: public BaseLogitsProcessor {
public:
    explicit TestStatefulLogitsProcessor(bool async_device_state): async_device_state_(async_device_state) {}

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override {
        (void)inputs;
        (void)start_idx;
        (void)finish_idx;
    }

    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override {
        (void)src_batch_indices;
    }

    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override {
        (void)new_tokens;
        accepted_token_len_ += num_new_tokens;
    }

    bool isStateful() const override {
        return true;
    }

    bool supportsNormalAsyncDeviceState() const override {
        return async_device_state_;
    }

    int64_t acceptedTokenLen() const override {
        return accepted_token_len_;
    }

private:
    bool    async_device_state_;
    int64_t accepted_token_len_ = 0;
};

TEST_F(NormalBatchStreamProcessorTest, testBatchTtftCapturedAtOutputUpdate) {
    const auto config    = makeLogprobsModelConfig();
    auto       processor = makeLogprobsProcessor(config);
    // Non-streaming, aux_info disabled: TTFT must not wait for publication.
    auto              first  = makeLogprobsStream(config, false, 0, 3, false);
    auto              second = makeLogprobsStream(config, false, 0, 3, false);
    ExecutionRecorder recorder("", "");
    auto              make_batch = [&] {
        return std::make_shared<RecordedBatch>(
            recorder, ExecutionRecorder::Json{}, std::vector<ExecutionRecorder::Json>(2));
    };
    auto prefill = make_batch();
    auto logits  = torch::zeros({2, 4}, torch::kFloat32);
    ASSERT_TRUE(dispatchLogprobsStep(*processor, {first, second}, logits, {1, 2}, prefill).ok());
    for (const auto& timing : prefill->timings) {
        ASSERT_TRUE(timing.ttft_us.has_value());
    }
    EXPECT_EQ(*prefill->timings[0].ttft_us, first->complete_token_ids_->firstTokenLatencyUs());
    EXPECT_EQ(*prefill->timings[1].ttft_us, second->complete_token_ids_->firstTokenLatencyUs());
    auto decode = make_batch();
    ASSERT_TRUE(dispatchLogprobsStep(*processor, {first, second}, logits, {2, 1}, decode).ok());
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(decode->timings[i].ttft_us, prefill->timings[i].ttft_us);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testSimpleAssemble) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                = 2048;
    model_config.vocab_size                 = 2048;
    model_config.num_layers                 = 2;
    model_config.attn_config.kv_cache_dtype = KvCacheDataType::INT8;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};

    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = hostIntBuffer({1, 2});
    query1->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    query1->input_ids = hostIntBuffer({1});
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(1, 3, {0, 0, 0});
    addr1.setBatchBlocks(0, 0, {1, 2, 3, 4});
    stream1->setKVCache(addr1);
    stream1->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = hostIntBuffer({1, 2, 3});
    query2->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    query2->input_ids = hostIntBuffer({1, 2});
    BatchKVCacheResource addr2;
    addr2.resetBatchSize(1);
    addr2.initGroups(1, 3, {0, 0, 0});
    addr2.setBatchBlocks(0, 0, {5, 6, 7, 8});
    stream2->setKVCache(addr2);
    stream2->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query3 = make_shared<GenerateInput>();
    query3->input_ids                     = hostIntBuffer({1, 2, 3});
    query3->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream3 =
        make_shared<NormalGenerateStream>(query3, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.resetBatchSize(1);
    addr3.initGroups(1, 3, {0, 0, 0});
    addr3.setBatchBlocks(0, 0, {9, 10});
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4 = make_shared<GenerateInput>();
    query4->input_ids                     = hostIntBuffer({1, 2, 3, 4});
    query4->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream4 =
        make_shared<NormalGenerateStream>(query4, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr4;
    addr4.resetBatchSize(1);
    addr4.initGroups(1, 3, {0, 0, 0});
    addr4.setBatchBlocks(0, 0, {11, 12, 13, 14});
    stream4->setKVCache(addr4);
    stream4->setReuseLength(1);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);
    streams.emplace_back(stream3);
    streams.emplace_back(stream4);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }

    {
        StreamGroups stream_groups(streams);
        TensorHolder holder;

        auto merge_input_status = processor.gatherModelInput(stream_groups, holder, true);

        EXPECT_TRUE(merge_input_status.ok());
        auto&       model_input       = merge_input_status.value();
        vector<int> combo_tokens      = {2, 3, 1, 2, 3, 2, 3, 4};
        vector<int> input_lengths     = {1, 2, 3, 3};
        vector<int> sequence_lengths  = {1, 2};
        vector<int> prefix_lengths    = {0, 1};
        vector<int> kv_cache_block_id = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 11, 12, 13, 14};
        EXPECT_EQ(combo_tokens, toVec<int>(model_input.combo_tokens));
        EXPECT_EQ(input_lengths, toVec<int>(model_input.input_lengths));
        EXPECT_EQ(sequence_lengths, toVec<int>(model_input.sequence_lengths));
        EXPECT_EQ(prefix_lengths, toVec<int>(model_input.prefix_lengths));
        ASSERT_TRUE(model_input.record_lengths.has_value());
        EXPECT_EQ(model_input.record_lengths->q_tokens, (std::vector<int32_t>{1, 1, 3, 3}));
        EXPECT_EQ(model_input.record_lengths->kv_tokens_before, (std::vector<int32_t>{1, 2, 0, 1}));
        EXPECT_EQ(kv_cache_block_id, toVec<int>(model_input.kv_cache_block_id));
    }
    {
        MMModelConfig mm_model_config;
        model_config.mm_model_config = mm_model_config;
        NormalBatchStreamProcessor processor(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

        StreamGroups stream_groups(streams);
        TensorHolder holder;
        auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
        EXPECT_TRUE(merge_input_status.ok());
        auto& model_input = merge_input_status.value();
        EXPECT_FALSE(model_input.attention_mask.defined());
        EXPECT_FALSE(model_input.record_lengths.has_value());
    }
}

TEST_F(NormalBatchStreamProcessorTest, testRecordedDeviceLengthsUseCpuMirror) {
    const auto model_config = makeLogprobsModelConfig(128);
    auto       processor    = makeLogprobsProcessor(model_config);
    auto       stream       = makeLogprobsStream(model_config, false, 0, 8, true);
    stream->setIsContextStream(false);
    stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
        .last_sample_token_gpu = torch::tensor({42}, torch::kInt32).to(torch::kCUDA),
        .next_seq_len_gpu      = torch::tensor({4}, torch::kInt32).to(torch::kCUDA),
        .last_real_seq_len     = 3,
        .next_real_seq_len     = 4,
    });
    ASSERT_EQ(stream->seqLength(), 1);  // Async CPU bookkeeping has not caught up.
    StreamGroups groups({stream});
    TensorHolder holder;
    auto         result = processor->gatherModelInput(groups, holder, true);
    ASSERT_TRUE(result.ok()) << result.status();
    ASSERT_TRUE(result->record_lengths.has_value());
    EXPECT_EQ(result->record_lengths->q_tokens, (std::vector<int32_t>{1}));
    EXPECT_EQ(result->record_lengths->kv_tokens_before, (std::vector<int32_t>{3}));
    EXPECT_EQ(result->record_lengths->kv_tokens_before, toVec<int32_t>(result->sequence_lengths));
    auto next_state              = stream->getNormalAsyncDeviceState();
    next_state.next_real_seq_len = 5;
    stream->setNormalAsyncDeviceState(std::move(next_state));
    EXPECT_EQ(result->record_lengths->kv_tokens_before, (std::vector<int32_t>{3}));
}

TEST_F(NormalBatchStreamProcessorTest, testDeviceLengthsAllowMissingCpuMirror) {
    const auto model_config = makeLogprobsModelConfig(128);
    auto       processor    = makeLogprobsProcessor(model_config);
    auto       stream       = makeLogprobsStream(model_config, false, 0, 8, true);
    stream->setIsContextStream(false);
    for (int mirror : {-1, 0}) {
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .last_sample_token_gpu = torch::tensor({42}, torch::kInt32).to(torch::kCUDA),
            .next_seq_len_gpu      = torch::tensor({4}, torch::kInt32).to(torch::kCUDA),
            .next_real_seq_len     = mirror,
        });
        for (bool record : {false, true}) {
            StreamGroups groups({stream});
            TensorHolder holder;
            auto         result = processor->gatherModelInput(groups, holder, record);
            ASSERT_TRUE(result.ok()) << result.status();
            EXPECT_EQ(toVec<int32_t>(result->sequence_lengths), (std::vector<int32_t>{3}));
            ASSERT_EQ(result->record_lengths.has_value(), record);
            if (record) {
                EXPECT_EQ(result->record_lengths->q_tokens, (std::vector<int32_t>{1}));
                EXPECT_EQ(result->record_lengths->kv_tokens_before, (std::vector<int32_t>{-1}));
            }
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, testDeviceStateFastPathWaitsForBlockingLogitsProcessorState) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 128;
    model_config.vocab_size  = 128900;
    RuntimeConfig runtime_config;

    std::shared_ptr<GenerateInput> query          = make_shared<GenerateInput>();
    query->input_ids                              = hostIntBuffer({1, 2, 3});
    query->generate_config                        = make_shared<GenerateConfig>();
    query->generate_config->in_think_mode         = true;
    query->generate_config->max_thinking_tokens   = 10;
    query->generate_config->begin_think_token_ids = {128821};
    query->generate_config->end_think_token_ids   = {128822};

    GenerateStreamPtr stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(false);
    stream->generate_status_->status = StreamState::RUNNING;

    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
        .last_sample_token_gpu = torch::full({1}, 42, cuda_i32),
        .next_seq_len_gpu      = torch::full({1}, 4, cuda_i32),
        .last_real_seq_len     = 3,
        .next_real_seq_len     = 4,
    });

    std::list<GenerateStreamPtr> streams{stream};
    StreamGroups                 stream_groups(streams);

    EngineInitParams params;
    params.model_config_ = model_config;
    params.py_model      = py::none();
    NormalExecutor executor(params, nullptr, true);

    EXPECT_TRUE(executor.gatherCanUseDeviceState(stream_groups));
    stream->logits_processor_list_.push_back(std::make_shared<TestStatefulLogitsProcessor>(false));
    stream->incPendingAsyncBookkeeping();
    EXPECT_FALSE(executor.gatherCanUseDeviceState(stream_groups));
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
}

TEST_F(NormalBatchStreamProcessorTest, testDeviceStateFastPathAllowsAsyncLogitsProcessorState) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 128;
    model_config.vocab_size  = 128900;
    RuntimeConfig runtime_config;

    std::shared_ptr<GenerateInput> query = make_shared<GenerateInput>();
    query->input_ids                     = hostIntBuffer({1, 2, 3});
    query->generate_config               = make_shared<GenerateConfig>();

    GenerateStreamPtr stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(false);
    stream->generate_status_->status = StreamState::RUNNING;

    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
        .last_sample_token_gpu = torch::full({1}, 42, cuda_i32),
        .next_seq_len_gpu      = torch::full({1}, 4, cuda_i32),
        .last_real_seq_len     = 3,
        .next_real_seq_len     = 4,
    });
    stream->logits_processor_list_.push_back(std::make_shared<TestStatefulLogitsProcessor>(true));

    std::list<GenerateStreamPtr> streams{stream};
    StreamGroups                 stream_groups(streams);

    EngineInitParams params;
    params.model_config_ = model_config;
    params.py_model      = py::none();
    NormalExecutor executor(params, nullptr, true);

    stream->incPendingAsyncBookkeeping();
    EXPECT_TRUE(executor.gatherCanUseDeviceState(stream_groups));
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
}

TEST_F(NormalBatchStreamProcessorTest, testSoftmaxProbs) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2;
    model_config.num_layers  = 2;

    PDSepConfig                    pd_sep_config;
    ProfilingDebugLoggingConfig    profiling_debug_logging_config;
    CacheConfig                    cache_config;
    RuntimeConfig                  runtime_config;
    std::shared_ptr<GenerateInput> query1         = make_shared<GenerateInput>();
    query1->input_ids                             = hostIntBuffer({1});
    query1->generate_config                       = make_shared<GenerateConfig>();
    query1->generate_config->return_softmax_probs = true;
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(1, 3, {0, 0, 0});
    addr1.setBatchBlocks(0, 0, {1});
    stream1->setKVCache(addr1);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }
    cache_config.group_types = {CacheGroupType::FULL};
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    StreamGroups stream_groups(streams);
    TensorHolder holder;
    auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
    EXPECT_TRUE(merge_input_status.ok());

    SamplerInputs sampler_inputs;
    MergedOutput  merge_outputs;
    auto          hidden_tensor                = torch::tensor({1.0f, 2.0f}).reshape({1, 2}).to(torch::kCUDA);
    auto          logits_tensor                = torch::tensor({1.0f, 2.0f}).reshape({1, 2}).to(torch::kCUDA);
    merge_outputs.model_output.hidden_states   = hidden_tensor;
    merge_outputs.model_output.logits          = logits_tensor;
    merge_outputs.sampler_output.token_ids     = torch::tensor({0, 1}, torch::kInt32).reshape({1, 2});
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f}).to(torch::kCUDA);
    auto status                                = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());

    auto softmax_probs = stream1->getSoftmaxProbs();
    EXPECT_TRUE(softmax_probs.defined());
    EXPECT_EQ(2048, softmax_probs.numel());
    EXPECT_NEAR(0.731058, softmax_probs.data_ptr<float>()[1], 0.0001);
}

TEST_F(NormalBatchStreamProcessorTest, testCompactRawLogprobsAccumulateForNonStreamingOutput) {
    auto model_config = makeLogprobsModelConfig();
    auto processor    = makeLogprobsProcessor(model_config);
    auto stream       = makeLogprobsStream(model_config, true, 2, 2, false);
    ASSERT_FALSE(stream->getTokenLogProbs().defined());

    auto step1_logits = torch::tensor({0.0f, 1.0f, 2.0f, 3.0f}, torch::kFloat32).reshape({1, 4});
    auto step2_logits = torch::tensor({4.0f, 1.0f, 0.0f, -1.0f}, torch::kFloat32).reshape({1, 4});
    ASSERT_TRUE(dispatchLogprobsStep(*processor, {stream}, step1_logits, {0}).ok());
    EXPECT_FALSE(stream->hasOutput());
    ASSERT_TRUE(dispatchLogprobsStep(*processor, {stream}, step2_logits, {1}).ok());

    auto output_result = stream->nextOutput();
    ASSERT_TRUE(output_result.ok());
    ASSERT_EQ(output_result.value().generate_outputs.size(), 1);
    const auto& output = output_result.value().generate_outputs[0];
    ASSERT_TRUE(output.token_logprobs.has_value());
    ASSERT_TRUE(output.top_logprob_token_ids.has_value());
    ASSERT_TRUE(output.top_logprobs.has_value());
    EXPECT_EQ(output.logprobs_offset, 0);
    EXPECT_EQ(output.logprobs_count, 2);

    auto step1_logprobs    = torch::log_softmax(step1_logits, -1);
    auto step2_logprobs    = torch::log_softmax(step2_logits, -1);
    auto expected_selected = torch::stack({step1_logprobs[0][0], step2_logprobs[0][1]});
    expectFloatTensorNear(output.token_logprobs.value(), expected_selected);

    // Step 1 deliberately sampled token 0 although raw top-2 are tokens 3 and
    // 2. The selected token logprob must still be returned independently.
    EXPECT_EQ(toVec<int32_t>(output.top_logprob_token_ids.value()), (std::vector<int32_t>{3, 2, 0, 1}));
    auto step1_topk            = step1_logprobs.topk(2, -1, true, true);
    auto step2_topk            = step2_logprobs.topk(2, -1, true, true);
    auto expected_top_logprobs = torch::cat({std::get<0>(step1_topk), std::get<0>(step2_topk)}, 0);
    expectFloatTensorNear(output.top_logprobs.value(), expected_top_logprobs);
}

TEST_F(NormalBatchStreamProcessorTest, testCompactRawLogprobsKZeroReturnsSelectedOnly) {
    auto model_config = makeLogprobsModelConfig();
    auto processor    = makeLogprobsProcessor(model_config);
    auto stream       = makeLogprobsStream(model_config, true, 0, 1, true);
    auto logits       = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::kFloat32).reshape({1, 4});
    ASSERT_TRUE(dispatchLogprobsStep(*processor, {stream}, logits, {1}).ok());

    auto output_result = stream->nextOutput();
    ASSERT_TRUE(output_result.ok());
    const auto& output = output_result.value().generate_outputs[0];
    ASSERT_TRUE(output.token_logprobs.has_value());
    ASSERT_TRUE(output.top_logprob_token_ids.has_value());
    ASSERT_TRUE(output.top_logprobs.has_value());
    EXPECT_EQ(output.top_logprob_token_ids.value().sizes(), (torch::IntArrayRef{1, 0}));
    EXPECT_EQ(output.top_logprobs.value().sizes(), (torch::IntArrayRef{1, 0}));
    expectFloatTensorNear(output.token_logprobs.value(), torch::log_softmax(logits, -1)[0][1].reshape({1}));
}

TEST_F(NormalBatchStreamProcessorTest, testThinkingSkipsRawLogprobsUntilAfterFirstCloseToken) {
    auto model_config = makeLogprobsModelConfig();
    auto processor    = makeLogprobsProcessor(model_config);
    auto stream       = makeLogprobsStream(
        model_config, true, 0, 4, true, /*stop_words_list=*/{}, /*in_think_mode=*/true, /*end=*/{2, 3});
    GptModelInputs model_inputs;

    const auto run_step = [&](const torch::Tensor& logits, int32_t sampled_token, bool expect_raw_snapshot) {
        StreamGroups    stream_groups({stream});
        GptModelOutputs model_output;
        model_output.logits = logits.to(torch::kCUDA);
        auto sampler_input  = processor->gatherSamplerInput(stream_groups, model_inputs, model_output);
        ASSERT_TRUE(sampler_input.ok());
        EXPECT_EQ(sampler_input->raw_logprobs_logits.defined(), expect_raw_snapshot);
        EXPECT_EQ(sampler_input->raw_logprobs_row_indices.defined(), expect_raw_snapshot);

        MergedOutput outputs;
        outputs.model_output                            = model_output;
        outputs.sampler_output.token_ids                = torch::tensor({{sampled_token}}, torch::kInt32);
        outputs.sampler_output.raw_logprobs_logits      = sampler_input->raw_logprobs_logits;
        outputs.sampler_output.raw_logprobs_row_indices = sampler_input->raw_logprobs_row_indices;
        ASSERT_TRUE(processor->dispatch(stream_groups, outputs).ok());
    };

    auto reasoning_logits = torch::tensor({4.0f, 3.0f, 2.0f, 1.0f}, torch::kFloat32).reshape({1, 4});
    run_step(reasoning_logits, 1, false);
    auto reasoning_result = stream->nextOutput();
    ASSERT_TRUE(reasoning_result.ok());
    const auto& reasoning_output = reasoning_result.value().generate_outputs[0];
    EXPECT_EQ(toVec<int32_t>(reasoning_output.output_ids), (std::vector<int32_t>{1}));
    EXPECT_EQ(reasoning_output.logprobs_offset, 1);
    EXPECT_EQ(reasoning_output.logprobs_count, 0);
    EXPECT_FALSE(reasoning_output.token_logprobs.has_value());
    EXPECT_FALSE(stream->getTokenLogProbs().defined());
    EXPECT_FALSE(stream->hasLogprobsContentStarted());

    auto close_logits = torch::tensor({1.0f, 2.0f, 4.0f, 3.0f}, torch::kFloat32).reshape({1, 4});
    run_step(close_logits, 2, false);
    auto close_result = stream->nextOutput();
    ASSERT_TRUE(close_result.ok());
    const auto& close_output = close_result.value().generate_outputs[0];
    EXPECT_EQ(toVec<int32_t>(close_output.output_ids), (std::vector<int32_t>{2}));
    EXPECT_EQ(close_output.logprobs_offset, 1);
    EXPECT_EQ(close_output.logprobs_count, 0);
    EXPECT_FALSE(close_output.token_logprobs.has_value());
    EXPECT_FALSE(stream->getTokenLogProbs().defined());
    EXPECT_TRUE(stream->hasLogprobsContentStarted());

    // The next decode row is captured. It is the tail of the textual close
    // delimiter, but DashScope's content logprobs begin immediately after the
    // first close token.
    auto content_logits = torch::tensor({0.0f, 1.0f, 2.0f, 4.0f}, torch::kFloat32).reshape({1, 4});
    run_step(content_logits, 3, true);
    auto content_result = stream->nextOutput();
    ASSERT_TRUE(content_result.ok());
    const auto& content_output = content_result.value().generate_outputs[0];
    EXPECT_EQ(toVec<int32_t>(content_output.output_ids), (std::vector<int32_t>{3}));
    EXPECT_EQ(content_output.logprobs_offset, 0);
    EXPECT_EQ(content_output.logprobs_count, 1);
    ASSERT_TRUE(content_output.token_logprobs.has_value());
    ASSERT_TRUE(content_output.top_logprob_token_ids.has_value());
    ASSERT_TRUE(content_output.top_logprobs.has_value());
    EXPECT_EQ(content_output.top_logprobs.value().sizes(), (torch::IntArrayRef{1, 0}));
    expectFloatTensorNear(content_output.token_logprobs.value(),
                          torch::log_softmax(content_logits, -1)[0][3].reshape({1}));
    ASSERT_TRUE(stream->getTokenLogProbs().defined());
    EXPECT_EQ(stream->logprobs_history_size_, 1);
}

TEST_F(NormalBatchStreamProcessorTest, testThinkingBoundaryPacketReturnsCompactContentLogprobs) {
    auto model_config = makeLogprobsModelConfig();
    auto stream       = makeLogprobsStream(
        model_config, true, 0, 4, false, /*stop_words_list=*/{}, /*in_think_mode=*/true, /*end=*/{2, 3});

    auto token_logprobs        = torch::tensor({{-0.3f, -0.4f}}, torch::kFloat32);
    auto top_logprob_token_ids = torch::empty({1, 2, 0}, torch::kInt32);
    auto top_logprobs          = torch::empty({1, 2, 0}, torch::kFloat32);
    stream->update({torch::tensor({{1, 2, 3, 0}}, torch::kInt32),
                    4,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    true,
                    false,
                    token_logprobs,
                    top_logprob_token_ids,
                    top_logprobs,
                    -1,
                    2});

    auto output_result = stream->nextOutput();
    ASSERT_TRUE(output_result.ok());
    const auto& output = output_result.value().generate_outputs[0];
    EXPECT_EQ(toVec<int32_t>(output.output_ids), (std::vector<int32_t>{1, 2, 3, 0}));
    EXPECT_EQ(output.logprobs_offset, 2);
    EXPECT_EQ(output.logprobs_count, 2);
    ASSERT_TRUE(output.token_logprobs.has_value());
    ASSERT_TRUE(output.top_logprob_token_ids.has_value());
    ASSERT_TRUE(output.top_logprobs.has_value());
    expectFloatTensorNear(output.token_logprobs.value(), token_logprobs.reshape({2}));
    EXPECT_EQ(output.top_logprobs.value().sizes(), (torch::IntArrayRef{2, 0}));
    EXPECT_TRUE(stream->hasLogprobsContentStarted());
}

TEST_F(NormalBatchStreamProcessorTest, testCudaLogprobsRowReductionKZeroMixedBatchAndPaddedVocab) {
    for (const auto dtype : {torch::kFloat16, torch::kBFloat16}) {
        SCOPED_TRACE(c10::toString(dtype));
        auto         model_config   = makeLogprobsModelConfig(5);
        auto         processor      = makeLogprobsProcessor(model_config);
        auto         plain_stream   = makeLogprobsStream(model_config, false, 0, 1, true);
        auto         logprob_stream = makeLogprobsStream(model_config, true, 0, 1, true);
        StreamGroups stream_groups({plain_stream, logprob_stream});

        // The last three columns simulate TP-alignment padding and deliberately
        // dominate the real vocabulary. Only the requested stream's second row
        // should be snapshotted and reduced, in the original half/bfloat dtype.
        auto padded_logits_fp32 = torch::tensor({-2.0f,
                                                 0.5f,
                                                 1.5f,
                                                 3.0f,
                                                 -1.0f,
                                                 100.0f,
                                                 99.0f,
                                                 98.0f,
                                                 4.0f,
                                                 -2.0f,
                                                 0.0f,
                                                 1.0f,
                                                 2.0f,
                                                 80.0f,
                                                 70.0f,
                                                 60.0f},
                                                torch::kFloat32)
                                      .reshape({2, 8});
        GptModelInputs  model_inputs;
        GptModelOutputs model_output;
        model_output.logits = padded_logits_fp32.to(dtype).to(torch::kCUDA);

        auto sampler_input = processor->gatherSamplerInput(stream_groups, model_inputs, model_output);
        ASSERT_TRUE(sampler_input.ok());
        ASSERT_TRUE(sampler_input->raw_logprobs_logits.defined());
        EXPECT_EQ(sampler_input->raw_logprobs_logits.scalar_type(), dtype);
        EXPECT_EQ(sampler_input->raw_logprobs_logits.sizes(), (torch::IntArrayRef{1, 5}));
        EXPECT_EQ(toVec<int64_t>(sampler_input->raw_logprobs_row_indices), (std::vector<int64_t>{1}));

        MergedOutput outputs;
        outputs.model_output                            = model_output;
        outputs.sampler_output.token_ids                = torch::tensor({0, 1}, torch::kInt32).reshape({2, 1});
        outputs.sampler_output.raw_logprobs_logits      = sampler_input->raw_logprobs_logits;
        outputs.sampler_output.raw_logprobs_row_indices = sampler_input->raw_logprobs_row_indices;
        ASSERT_TRUE(processor->dispatch(stream_groups, outputs).ok());

        auto plain_result   = plain_stream->nextOutput();
        auto logprob_result = logprob_stream->nextOutput();
        ASSERT_TRUE(plain_result.ok());
        ASSERT_TRUE(logprob_result.ok());
        EXPECT_FALSE(plain_result.value().generate_outputs[0].token_logprobs.has_value());

        const auto& output = logprob_result.value().generate_outputs[0];
        ASSERT_TRUE(output.token_logprobs.has_value());
        ASSERT_TRUE(output.top_logprob_token_ids.has_value());
        ASSERT_TRUE(output.top_logprobs.has_value());
        EXPECT_EQ(output.top_logprob_token_ids.value().sizes(), (torch::IntArrayRef{1, 0}));
        EXPECT_EQ(output.top_logprobs.value().sizes(), (torch::IntArrayRef{1, 0}));

        auto expected =
            torch::log_softmax(padded_logits_fp32.index({1, torch::indexing::Slice(0, 5)}), -1).index({1}).reshape({1});
        expectFloatTensorNear(output.token_logprobs.value(), expected, 2e-3f);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testCompactRawLogprobsMixedBatchOnlyReturnsForRequestedStream) {
    auto model_config  = makeLogprobsModelConfig();
    auto processor     = makeLogprobsProcessor(model_config);
    auto requested     = makeLogprobsStream(model_config, true, 1, 1, true);
    auto not_requested = makeLogprobsStream(model_config, false, 0, 1, true);
    auto logits = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f}, torch::kFloat32).reshape({2, 4});
    ASSERT_TRUE(dispatchLogprobsStep(*processor, {requested, not_requested}, logits, {0, 3}).ok());

    auto requested_output = requested->nextOutput();
    auto plain_output     = not_requested->nextOutput();
    ASSERT_TRUE(requested_output.ok());
    ASSERT_TRUE(plain_output.ok());
    EXPECT_TRUE(requested_output.value().generate_outputs[0].token_logprobs.has_value());
    EXPECT_TRUE(requested_output.value().generate_outputs[0].top_logprobs.has_value());
    EXPECT_FALSE(plain_output.value().generate_outputs[0].token_logprobs.has_value());
    EXPECT_FALSE(plain_output.value().generate_outputs[0].top_logprob_token_ids.has_value());
    EXPECT_FALSE(plain_output.value().generate_outputs[0].top_logprobs.has_value());
    EXPECT_FALSE(not_requested->getTokenLogProbs().defined());
}

TEST_F(NormalBatchStreamProcessorTest, testSamplerInputSnapshotsOnlyRequestedRowsAndRealVocab) {
    auto           model_config = makeLogprobsModelConfig();
    auto           processor    = makeLogprobsProcessor(model_config);
    GptModelInputs model_inputs;

    auto requested     = makeLogprobsStream(model_config, true, 2, 2, true);
    auto not_requested = makeLogprobsStream(model_config, false, 0, 2, true);
    // Width 6 simulates two LM-head padding columns beyond the real vocab 4.
    GptModelOutputs model_output;
    model_output.logits =
        torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 50.0f, 60.0f, 6.0f, 7.0f, 8.0f, 9.0f, 70.0f, 80.0f}, torch::kFloat32)
            .reshape({2, 6})
            .to(torch::kCUDA);
    auto original_logits = model_output.logits.clone();
    auto sampler         = processor->gatherSamplerInput({{requested, not_requested}}, model_inputs, model_output);
    ASSERT_TRUE(sampler.ok());

    // Logprob capture must not change the sampler distribution, including
    // padded LM-head columns. The independent raw-logprob snapshot contains
    // only requested model row 0 and only the four real vocabulary columns.
    EXPECT_EQ(sampler.value().logits.data_ptr<float>(), model_output.logits.data_ptr<float>());
    EXPECT_TRUE(torch::equal(sampler.value().logits, original_logits));
    ASSERT_TRUE(sampler.value().raw_logprobs_logits.defined());
    EXPECT_EQ(sampler.value().raw_logprobs_row_indices.is_pinned(), model_output.logits.is_cuda());
    EXPECT_EQ(sampler.value().raw_logprobs_logits.sizes(), (torch::IntArrayRef{1, 4}));
    EXPECT_EQ(toVec<int64_t>(sampler.value().raw_logprobs_row_indices), (std::vector<int64_t>{0}));
    EXPECT_EQ(toVec<float>(sampler.value().raw_logprobs_logits), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));

    // The compact snapshot must not alias logits that processors mutate.
    sampler.value().logits.fill_(-100.0f);
    EXPECT_EQ(toVec<float>(sampler.value().raw_logprobs_logits), (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));

    // Dispatch must prefer the compact pre-sampler snapshot over the now
    // mutated model_output logits, and map only the requested stream's row.
    MergedOutput outputs;
    outputs.model_output                       = model_output;
    outputs.sampler_output.token_ids           = torch::tensor({0, 3}, torch::kInt32).reshape({2, 1}).to(torch::kCUDA);
    outputs.sampler_output.raw_logprobs_logits = sampler.value().raw_logprobs_logits;
    outputs.sampler_output.raw_logprobs_row_indices = sampler.value().raw_logprobs_row_indices;
    ASSERT_TRUE(processor->dispatch({{requested, not_requested}}, outputs).ok());

    auto requested_output = requested->nextOutput();
    auto plain_output     = not_requested->nextOutput();
    ASSERT_TRUE(requested_output.ok());
    ASSERT_TRUE(plain_output.ok());
    const auto& logprobs_output           = requested_output.value().generate_outputs[0];
    auto        original_requested_logits = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::kFloat32).reshape({1, 4});
    auto        expected_logprobs         = torch::log_softmax(original_requested_logits, -1);
    expectFloatTensorNear(logprobs_output.token_logprobs.value(), expected_logprobs[0][0].reshape({1}));
    EXPECT_EQ(toVec<int32_t>(logprobs_output.top_logprob_token_ids.value()), (std::vector<int32_t>{3, 2}));
    EXPECT_FALSE(plain_output.value().generate_outputs[0].token_logprobs.has_value());
}

TEST_F(NormalBatchStreamProcessorTest, testSamplerInputPreservesNarrowLogitsWithoutLogprobs) {
    auto           model_config = makeLogprobsModelConfig(100);
    auto           processor    = makeLogprobsProcessor(model_config);
    auto           stream       = makeLogprobsStream(model_config, false, 0, 1, true);
    GptModelInputs model_inputs;

    GptModelOutputs model_output;
    model_output.logits =
        torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
            .reshape({1, 4});
    auto original_logits = model_output.logits.clone();

    auto sampler = processor->gatherSamplerInput({{stream}}, model_inputs, model_output);
    ASSERT_TRUE(sampler.ok());
    EXPECT_EQ(sampler.value().logits.sizes(), (torch::IntArrayRef{1, 4}));
    EXPECT_TRUE(torch::equal(sampler.value().logits, original_logits));
    EXPECT_FALSE(sampler.value().raw_logprobs_logits.defined());
}

TEST_F(NormalBatchStreamProcessorTest, testSamplerInputPreservesPaddedLogitsWithoutLogprobs) {
    auto           model_config = makeLogprobsModelConfig(4);
    auto           processor    = makeLogprobsProcessor(model_config);
    auto           stream       = makeLogprobsStream(model_config, false, 0, 1, true);
    GptModelInputs model_inputs;

    GptModelOutputs model_output;
    model_output.logits = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 50.0f, 60.0f},
                                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                              .reshape({1, 6});
    auto original_logits = model_output.logits.clone();

    auto sampler = processor->gatherSamplerInput({{stream}}, model_inputs, model_output);
    ASSERT_TRUE(sampler.ok());
    EXPECT_EQ(sampler.value().logits.data_ptr<float>(), model_output.logits.data_ptr<float>());
    EXPECT_TRUE(torch::equal(sampler.value().logits, original_logits));
    EXPECT_TRUE(torch::equal(model_output.logits, original_logits));
    EXPECT_FALSE(sampler.value().raw_logprobs_logits.defined());
}

TEST_F(NormalBatchStreamProcessorTest, testLogprobsExcludePaddedVocabColumns) {
    auto model_config = makeLogprobsModelConfig(4);
    auto processor    = makeLogprobsProcessor(model_config);
    auto stream       = makeLogprobsStream(model_config, true, 2, 1, true);
    // Padded columns deliberately dominate every real logit. They must affect
    // neither normalization nor Top-K token IDs.
    auto padded_logits = torch::tensor({0.0f, 1.0f, 2.0f, 3.0f, 100.0f, 99.0f}, torch::kFloat32).reshape({1, 6});
    ASSERT_TRUE(dispatchLogprobsStep(*processor, {stream}, padded_logits, {0}).ok());

    auto output_result = stream->nextOutput();
    ASSERT_TRUE(output_result.ok());
    const auto& output        = output_result.value().generate_outputs[0];
    auto        real_logprobs = torch::log_softmax(padded_logits.narrow(1, 0, 4), -1);
    expectFloatTensorNear(output.token_logprobs.value(), real_logprobs[0][0].reshape({1}));
    EXPECT_EQ(toVec<int32_t>(output.top_logprob_token_ids.value()), (std::vector<int32_t>{3, 2}));
    expectFloatTensorNear(output.top_logprobs.value(), std::get<0>(real_logprobs.topk(2, -1, true, true)));
}

TEST_F(NormalBatchStreamProcessorTest, testLogprobsPreserveSmallNegativeNormalizationCorrection) {
    auto model_config = makeLogprobsModelConfig(4);
    auto processor    = makeLogprobsProcessor(model_config);
    auto stream       = makeLogprobsStream(model_config, true, 1, 1, true);
    auto logits       = torch::tensor({80.0f, 66.0f, 65.0f, 64.0f}, torch::kFloat32).reshape({1, 4});

    ASSERT_TRUE(dispatchLogprobsStep(*processor, {stream}, logits, {0}).ok());
    auto output_result = stream->nextOutput();
    ASSERT_TRUE(output_result.ok());
    const auto& output   = output_result.value().generate_outputs[0];
    const float expected = torch::log_softmax(logits, -1).index({0, 0}).item<float>();
    ASSERT_LT(expected, 0.0f);
    EXPECT_LT(output.token_logprobs.value().item<float>(), 0.0f);
    EXPECT_NEAR(output.token_logprobs.value().item<float>(), expected, 1e-9f);
    EXPECT_LT(output.top_logprobs.value().index({0, 0}).item<float>(), 0.0f);
}

TEST_F(NormalBatchStreamProcessorTest, testLogprobsRespectMaxTokenTruncation) {
    auto model_config = makeLogprobsModelConfig(5);
    auto stream       = makeLogprobsStream(model_config, true, 1, 2, false);
    auto new_tokens   = torch::tensor({1, 2, 3}, torch::kInt32).reshape({1, 3});
    auto selected     = torch::tensor({{-0.1f, -0.2f, -0.3f}}, torch::kFloat32);
    auto top_ids      = torch::tensor({1, 2, 3}, torch::kInt32).reshape({1, 3, 1});
    auto top_values   = torch::tensor({-0.01f, -0.02f, -0.03f}, torch::kFloat32).reshape({1, 3, 1});
    stream->update({new_tokens,
                    3,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    true,
                    false,
                    selected,
                    top_ids,
                    top_values});

    auto output_result = stream->nextOutput();
    ASSERT_TRUE(output_result.ok());
    const auto& output = output_result.value().generate_outputs[0];
    EXPECT_EQ(toVec<int32_t>(output.output_ids), (std::vector<int32_t>{1, 2}));
    expectFloatTensorNear(output.token_logprobs.value(), selected.narrow(1, 0, 2));
    EXPECT_EQ(toVec<int32_t>(output.top_logprob_token_ids.value()), (std::vector<int32_t>{1, 2}));
}

TEST_F(NormalBatchStreamProcessorTest, testLogprobsRespectStopTokenTruncation) {
    auto model_config = makeLogprobsModelConfig(5);
    auto stream       = makeLogprobsStream(model_config, true, 1, 5, false, {{2}});
    auto new_tokens   = torch::tensor({1, 2, 3}, torch::kInt32).reshape({1, 3});
    auto selected     = torch::tensor({{-0.1f, -0.2f, -0.3f}}, torch::kFloat32);
    auto top_ids      = torch::tensor({1, 2, 3}, torch::kInt32).reshape({1, 3, 1});
    auto top_values   = torch::tensor({-0.01f, -0.02f, -0.03f}, torch::kFloat32).reshape({1, 3, 1});
    stream->update({new_tokens,
                    3,
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    true,
                    false,
                    selected,
                    top_ids,
                    top_values});

    auto output_result = stream->nextOutput();
    ASSERT_TRUE(output_result.ok());
    const auto& output = output_result.value().generate_outputs[0];
    EXPECT_EQ(toVec<int32_t>(output.output_ids), (std::vector<int32_t>{1, 2}));
    expectFloatTensorNear(output.token_logprobs.value(), selected.narrow(1, 0, 2));
    EXPECT_EQ(toVec<int32_t>(output.top_logprob_token_ids.value()), (std::vector<int32_t>{1, 2}));
}

TEST_F(NormalBatchStreamProcessorTest, testLoss) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2048;
    model_config.num_layers  = 2;
    PDSepConfig                    pd_sep_config;
    ProfilingDebugLoggingConfig    profiling_debug_logging_config;
    CacheConfig                    cache_config;
    RuntimeConfig                  runtime_config;
    std::shared_ptr<GenerateInput> query1   = make_shared<GenerateInput>();
    query1->input_ids                       = hostIntBuffer({1});
    query1->generate_config                 = make_shared<GenerateConfig>();
    query1->generate_config->calculate_loss = 1;
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(1, 3, {0, 0, 0});
    addr1.setBatchBlocks(0, 0, {1});
    stream1->setKVCache(addr1);

    std::shared_ptr<GenerateInput> query3   = make_shared<GenerateInput>();
    query3->input_ids                       = hostIntBuffer({0, 1});
    query3->generate_config                 = make_shared<GenerateConfig>();
    query3->generate_config->calculate_loss = 2;
    GenerateStreamPtr stream3 =
        make_shared<NormalGenerateStream>(query3, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.resetBatchSize(1);
    addr3.initGroups(1, 3, {0, 0, 0});
    addr3.setBatchBlocks(0, 0, {9});
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4   = make_shared<GenerateInput>();
    query4->input_ids                       = hostIntBuffer({0, 1, 0});
    query4->generate_config                 = make_shared<GenerateConfig>();
    query4->generate_config->calculate_loss = 1;
    GenerateStreamPtr stream4 =
        make_shared<NormalGenerateStream>(query4, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr4;
    addr4.resetBatchSize(1);
    addr4.initGroups(1, 3, {0, 0, 0});
    addr4.setBatchBlocks(0, 0, {11, 12});
    stream4->setKVCache(addr4);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream3);
    streams.emplace_back(stream4);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }
    cache_config.group_types = {CacheGroupType::FULL};
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    StreamGroups stream_groups(streams);
    TensorHolder holder;
    auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
    EXPECT_TRUE(merge_input_status.ok());
    EXPECT_TRUE(merge_input_status.value().need_all_logits);

    SamplerInputs sampler_inputs;
    MergedOutput  merge_outputs;
    auto loss_hidden_tensor = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).reshape({3, 2}).to(torch::kCUDA);
    auto loss_logits_tensor = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).reshape({3, 2}).to(torch::kCUDA);
    auto loss_all_logits_tensor =
        torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f})
            .reshape({6, 2})
            .to(torch::kCUDA);
    merge_outputs.model_output.hidden_states = loss_hidden_tensor;
    merge_outputs.model_output.logits        = loss_logits_tensor;
    merge_outputs.model_output.all_logits    = loss_all_logits_tensor;
    merge_outputs.sampler_output.token_ids =
        torch::tensor({0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1}, torch::kInt32).reshape({3, 4});
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f, 2.0f, 3.0f}).to(torch::kCUDA);
    auto status                                = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(stream1->getLoss().defined());
    EXPECT_TRUE(stream3->getLoss().defined());
    auto loss3 = stream3->getLoss();
    EXPECT_EQ(1, loss3.numel());
    EXPECT_NEAR(0.31326, loss3.data_ptr<float>()[0], 0.0001);
    EXPECT_TRUE(stream4->getLoss().defined());
    auto loss4 = stream4->getLoss();
    EXPECT_EQ(2, loss4.numel());
    EXPECT_NEAR(2.25525, *(torch::mean(loss4).exp().data_ptr<float>()), 0.0001);
}

TEST_F(NormalBatchStreamProcessorTest, testMultimodalGatherBatch) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                   = 2048;
    model_config.vocab_size                    = 2048;
    model_config.num_layers                    = 2;
    model_config.attn_config.kv_cache_dtype    = KvCacheDataType::INT8;
    model_config.mm_model_config.is_multimodal = true;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};
    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = hostIntBuffer({1, -1, -1, -1, 2});
    query1->generate_config               = make_shared<GenerateConfig>();
    query1->mm_locs                       = torch::tensor({1}, torch::kInt32);
    query1->text_tokens_mask              = torch::tensor({1, 0, 0, 0, 1}, torch::kInt32);
    query1->multimodal_features           = {torch::rand({3, 10}, torch::kFloat16)};
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    stream1->setIsContextStream(true);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = hostIntBuffer({3, 4, 5});
    query2->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    stream2->setIsContextStream(true);

    std::shared_ptr<GenerateInput> query3 = make_shared<GenerateInput>();
    query3->input_ids                     = hostIntBuffer({6, 7, -1, -1, 8});
    query3->generate_config               = make_shared<GenerateConfig>();
    query3->mm_locs                       = torch::tensor({2}, torch::kInt32);
    query3->text_tokens_mask              = torch::tensor({1, 1, 0, 0, 1}, torch::kInt32);
    query3->multimodal_features           = {torch::rand({2, 10}, torch::kFloat16)};
    GenerateStreamPtr stream3 =
        make_shared<NormalGenerateStream>(query3, model_config, runtime_config, resource_context, nullptr);
    stream3->setIsContextStream(true);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);
    streams.emplace_back(stream3);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }

    {
        StreamGroups stream_groups(streams);
        TensorHolder holder;

        auto merge_input_status = processor.gatherModelInput(stream_groups, holder);
        EXPECT_TRUE(merge_input_status.ok());

        auto&       model_input      = merge_input_status.value();
        vector<int> combo_tokens     = {1, -1, -1, -1, 2, 3, 4, 5, 6, 7, -1, -1, 8};
        vector<int> input_lengths    = {5, 3, 5};
        vector<int> text_tokens_mask = {1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1};
        vector<int> mm_features_locs = {1, 10};

        EXPECT_EQ(combo_tokens, toVec<int>(model_input.combo_tokens));
        EXPECT_EQ(input_lengths, toVec<int>(model_input.input_lengths));
        EXPECT_EQ(text_tokens_mask, toVec<int>(model_input.text_tokens_mask));
        EXPECT_EQ(mm_features_locs, toVec<int>(model_input.mm_features_locs));

        EXPECT_EQ(model_input.multimodal_features.value().size(), 2);
        EXPECT_EQ(model_input.multimodal_features.value()[0].numel(), 3 * 10);
        EXPECT_EQ(model_input.multimodal_features.value()[1].numel(), 2 * 10);
    }
}

TEST_F(NormalBatchStreamProcessorTest, gatherExecutionModesPreserveFlagsAndErrors) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len      = 128;
    model_config.vocab_size       = 128;
    model_config.input_vocab_size = 128;
    model_config.num_layers       = 1;
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    auto make_stream = [&](std::vector<int32_t> tokens, int max_new_tokens) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = hostIntBuffer(std::move(tokens));
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = max_new_tokens;
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, true);

    struct StreamMode {
        bool context;
        int  max_new_tokens;
    };
    struct TestCase {
        const char*             name;
        std::vector<StreamMode> modes;
        bool                    prefill_only;
        bool                    mixed;
        bool                    existing_error = false;
    };
    const std::vector<TestCase> cases = {
        {"empty", {}, false, false},
        {"decode-generation", {{false, 1}, {false, 2}}, false, false},
        {"context-generation", {{true, 1}, {true, 2}}, false, false},
        {"context-prefill-only", {{true, 0}, {true, 0}}, true, false},
        {"decode-prefill-only", {{false, 0}}, true, false},
        {"both-containers-generation", {{true, 1}, {false, 1}}, false, false},
        {"both-containers-prefill-only", {{true, 0}, {false, 0}}, true, false},
        {"context-mixed-prefill-first", {{true, 0}, {true, 1}}, true, true},
        {"context-mixed-generation-first", {{true, 1}, {true, 0}}, false, true},
        {"decode-mixed-prefill-first", {{false, 0}, {false, 1}}, true, true},
        {"decode-mixed-generation-first", {{false, 1}, {false, 0}}, false, true},
        {"cross-container-decode-generation-first", {{false, 1}, {true, 0}}, false, true},
        {"cross-container-context-prefill-first", {{true, 0}, {false, 1}}, false, true},
        {"cross-container-decode-prefill-first", {{false, 0}, {true, 1}}, true, true},
        {"cross-container-context-generation-first", {{true, 1}, {false, 0}}, true, true},
        {"mixed-context-behind-generation-decode", {{false, 1}, {true, 1}, {true, 0}}, false, true},
        {"mixed-decode-ahead-of-generation-context", {{true, 1}, {false, 1}, {false, 0}}, false, true},
        {"existing-error-does-not-define-mode", {{true, 0}, {true, 1}}, false, false, true},
        {"existing-error-in-cross-container-batch", {{true, 0}, {false, 1}}, false, false, true},
        {"existing-error-before-mixed-context", {{false, 1}, {true, 0}, {true, 1}}, true, true, true},
        {"existing-error-in-homogeneous-batch", {{true, 1}, {false, 1}}, false, false, true},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        std::list<GenerateStreamPtr> streams;
        int64_t                      expected_tokens = 0;
        for (const auto& mode : test_case.modes) {
            auto stream = make_stream({1, 2}, mode.max_new_tokens);
            stream->setIsContextStream(mode.context);
            streams.push_back(stream);
            expected_tokens += mode.context ? 2 : 1;
        }
        if (test_case.existing_error) {
            streams.front()->reportError(ErrorCode::EXECUTION_EXCEPTION, "existing request error");
        }
        StreamGroups stream_groups(streams);
        TensorHolder holder;
        auto         model_input = processor.gatherModelInput(stream_groups, holder);
        // Mixed batches still reach TP sync; flags follow the execution mode
        // derived from schedulable (non-errored) streams.
        ASSERT_TRUE(model_input.ok());
        EXPECT_EQ(model_input->combo_tokens.numel(), expected_tokens);
        EXPECT_EQ(model_input->skip_lm_head, test_case.prefill_only);
        EXPECT_FALSE(model_input->capture_hidden_states);  // No capture reservation during warmup.
        for (const auto& stream : streams) {
            if (test_case.existing_error && stream == streams.front()) {
                EXPECT_EQ(stream->statusInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
                EXPECT_EQ(stream->stopReason(), "existing request error");
            } else if (test_case.mixed) {
                EXPECT_TRUE(stream->hasError());
                EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
                EXPECT_EQ(stream->stopReason(), kMixedExecutionModeBatchError);
            } else {
                EXPECT_FALSE(stream->hasError());
            }
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, prefillOnlySkipsLmHeadAndDispatchesOneEmptyOutput) {
    class RecordingGenerateStream: public NormalGenerateStream {
    public:
        using NormalGenerateStream::NormalGenerateStream;

        void updateOutput(const StreamUpdateInfo& update_info) override {
            update_infos.push_back(update_info);
            NormalGenerateStream::updateOutput(update_info);
        }

        std::vector<StreamUpdateInfo> update_infos;
    };

    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len      = 128;
    model_config.vocab_size       = 128;
    model_config.input_vocab_size = 128;
    model_config.num_layers       = 1;
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;

    auto make_stream = [&](std::vector<int32_t> tokens) {
        auto query                             = make_shared<GenerateInput>();
        query->input_ids                       = hostIntBuffer(std::move(tokens));
        query->generate_config                 = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens = 0;
        query->generate_config->aux_info       = true;
        auto stream =
            make_shared<RecordingGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };

    auto                                                                           stream1  = make_stream({1, 2});
    auto                                                                           stream2  = make_stream({3, 4, 5});
    const std::vector<std::pair<std::shared_ptr<RecordingGenerateStream>, size_t>> expected = {
        {stream1, stream1->seqLength()}, {stream2, stream2->seqLength()}};
    StreamGroups               stream_groups({stream1, stream2});
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, true);

    TensorHolder holder;
    auto         model_input = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(model_input.ok());
    EXPECT_TRUE(model_input->skip_lm_head);
    EXPECT_FALSE(model_input->capture_hidden_states);  // Zero-token execution alone does not grant capture.
    EXPECT_EQ(model_input->combo_tokens.numel(), 5);

    for (int i = 0; i < 2; ++i) {
        ASSERT_TRUE(processor.dispatchPrefillOnly(stream_groups).ok());
    }
    for (const auto& [stream, input_length] : expected) {
        ASSERT_EQ(stream->update_infos.size(), 1);
        const auto& update_info = stream->update_infos.front();
        EXPECT_EQ(update_info.num_new_tokens, 0);
        EXPECT_EQ(update_info.new_tokens.sizes(), (torch::IntArrayRef{1, 0}));
        EXPECT_FALSE(update_info.update_remote_generate);
        EXPECT_FALSE(update_info.force_update_info);

        auto output = stream->nextOutput();
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs.size(), 1);
        EXPECT_TRUE(output.value().generate_outputs[0].finished);
        EXPECT_EQ(output.value().generate_outputs[0].output_ids.sizes(), (torch::IntArrayRef{1, 0}));
        EXPECT_EQ(stream->seqLength(), input_length);
        EXPECT_FALSE(stream->hasOutput());

        stream->moveToNext();
        auto finished = stream->nextOutput();
        ASSERT_FALSE(finished.ok());
        EXPECT_EQ(finished.status().code(), ErrorCode::FINISHED);
    }
}

TEST_F(NormalBatchStreamProcessorTest, CaptureUsesOnlyFullyReservedCompleteContextBatches) {
    auto                           params = makeExecutorParams();
    NormalModelInputGathererConfig config;
    config.vocab_size = config.input_vocab_size = 128;
    config.num_layers                           = 1;
    config.role_type                            = RoleType::PDFUSION;
    struct Case {
        const char*                                   name;
        bool                                          capture;
        std::function<void(const GenerateStreamPtr&)> mutate;
    };
    const std::vector<Case> cases = {
        {"normal-generation", true, [](const auto&) {}},
        {"ordinary-peer", false, [](const auto& s) { s->generateConfig()->capture_hidden_states = false; }},
        {"reuse-conflict", false, [](const auto& s) { s->generateConfig()->reuse_cache = true; }},
        {"missing-budget", false, [](const auto& s) { s->cancelHiddenStateCapture(); }},
        {"decode-lane", false, [](const auto& s) { s->setIsContextStream(false); }},
        {"fake-lane", false, [](const auto& s) { s->setIsFakeStream(true); }},
        {"errored-lane", false, [](const auto& s) { s->reportError(ErrorCode::CANCELLED, "cancelled"); }},
        {"beam", false, [](const auto& s) { s->generateConfig()->num_beams = 2; }},
        {"multiple-sequences", false, [](const auto& s) { s->generateConfig()->num_return_sequences = 2; }},
        {"reused-prefix", false, [](const auto& s) { s->setReuseLength(1); }},
        {"repeated-context", false, [](const auto& s) { s->step(); }},
        {"generated-token-in-context",
         false,
         [](const auto& s) {
             s->completeTokenIds().template data_ptr<int32_t>()[2] = 3;
             s->setSeqLength(3);
         }},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        auto                         cancelled = std::make_shared<std::vector<int>>(2, 0);
        std::list<GenerateStreamPtr> streams;
        for (int i = 0; i < 2; ++i) {
            auto query                                    = std::make_shared<GenerateInput>();
            query->request_id                             = i + 1;
            query->input_ids                              = hostIntBuffer({1, 2});
            query->generate_config                        = std::make_shared<GenerateConfig>();
            query->generate_config->max_new_tokens        = 4;
            query->generate_config->capture_hidden_states = true;
            query->generate_config->reuse_cache           = false;
            if (i == 1 && std::string(test_case.name) == "multiple-sequences") {
                query->generate_config->num_return_sequences = 2;
            }
            if (i == 1 && std::string(test_case.name) == "beam") {
                query->generate_config->num_beams = 2;
            }
            auto stream = std::make_shared<NormalGenerateStream>(
                query, params.model_config_, params.runtime_config, ResourceContext{}, nullptr);
            stream->generate_status_->status = StreamState::RUNNING;
            stream->setHiddenStateCaptureReservation([cancelled, i] { ++cancelled->at(i); });
            streams.push_back(stream);
        }
        test_case.mutate(streams.back());
        TensorHolder holder;
        auto         result = NormalModelInputGatherer(config).gather(StreamGroups(streams), holder);
        ASSERT_TRUE(result.ok());
        EXPECT_EQ(result->capture_hidden_states, test_case.capture);
        EXPECT_FALSE(result->skip_lm_head);
        for (const auto& stream : streams) {
            EXPECT_EQ(stream->hasHiddenStateCaptureReservation(), test_case.capture);
        }
        EXPECT_EQ(*cancelled, (std::vector<int>(2, test_case.capture ? 0 : 1)));
        EXPECT_FALSE(streams.front()->hasError());
        if (std::string(test_case.name) != "errored-lane") {
            EXPECT_FALSE(streams.back()->hasError());
        } else {
            EXPECT_EQ(streams.back()->statusInfo().code(), ErrorCode::CANCELLED);
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, CaptureLegacyWarmupRoleAndGatherFailureBoundaries) {
    auto params = makeExecutorParams();
    for (int tokens : {0, 4}) {
        for (bool warmup : {false, true}) {
            for (RoleType role : {RoleType::PREFILL, RoleType::PDFUSION, RoleType::DECODE}) {
                SCOPED_TRACE(::testing::Message()
                             << "tokens=" << tokens << " warmup=" << warmup << " role=" << static_cast<int>(role));
                NormalModelInputGathererConfig config;
                config.num_layers = 1;
                config.vocab_size = config.input_vocab_size   = 128;
                config.role_type                              = role;
                config.warm_up                                = warmup;
                auto query                                    = std::make_shared<GenerateInput>();
                query->input_ids                              = hostIntBuffer({1, 2});
                query->generate_config                        = std::make_shared<GenerateConfig>();
                query->generate_config->max_new_tokens        = tokens;
                query->generate_config->capture_hidden_states = tokens > 0;
                query->generate_config->reuse_cache           = tokens == 0;  // Preserve legacy implicit capture.
                ResourceContext resources;
                resources.role_type = role;
                auto stream         = std::make_shared<NormalGenerateStream>(
                    query, params.model_config_, params.runtime_config, resources, nullptr);
                stream->generate_status_->status = StreamState::RUNNING;
                auto cancelled                   = std::make_shared<int>(0);
                stream->setHiddenStateCaptureReservation([cancelled] { ++*cancelled; });
                TensorHolder holder;
                auto         result = NormalModelInputGatherer(config).gather(StreamGroups({stream}), holder);
                ASSERT_TRUE(result.ok());
                const bool capture = !warmup && role != RoleType::DECODE;
                EXPECT_EQ(result->capture_hidden_states, capture);
                EXPECT_EQ(result->skip_lm_head, tokens == 0);
                EXPECT_EQ(*cancelled, capture ? 0 : 1);
                EXPECT_FALSE(stream->hasError());
            }
        }
    }
    NormalModelInputGathererConfig config;
    config.vocab_size = config.input_vocab_size   = 128;
    config.num_layers                             = 1;
    auto query                                    = std::make_shared<GenerateInput>();
    query->input_ids                              = hostIntBuffer({1, 129});
    query->generate_config                        = std::make_shared<GenerateConfig>();
    query->generate_config->capture_hidden_states = true;
    query->generate_config->reuse_cache           = false;
    auto stream                                   = std::make_shared<NormalGenerateStream>(
        query, params.model_config_, params.runtime_config, ResourceContext{}, nullptr);
    auto cancelled = std::make_shared<int>(0);
    stream->setHiddenStateCaptureReservation([cancelled] { ++*cancelled; });
    TensorHolder holder;
    auto         result = NormalModelInputGatherer(config).gather(StreamGroups({stream}), holder);
    EXPECT_FALSE(result.ok());
    EXPECT_FALSE(stream->hasHiddenStateCaptureReservation());
    EXPECT_EQ(*cancelled, 1);
}

TEST_F(NormalBatchStreamProcessorTest, ReservedCaptureKeepsNormalSamplingAndRealForwardErrors) {
    class CaptureModel: public ModelBase {
    public:
        bool tryReserveHiddenStateCapture(int64_t request_id, int64_t tokens) override {
            EXPECT_EQ(request_id, 42);
            EXPECT_EQ(tokens, 2);
            ++reserve_count;
            return grant;
        }
        void cancelHiddenStateCapture(int64_t request_id) override {
            EXPECT_EQ(request_id, 42);
            ++cancel_count;
        }
        GptModelOutputs forward(const GptModelInputs& inputs) override {
            EXPECT_EQ(inputs.capture_hidden_states, grant);
            EXPECT_FALSE(inputs.skip_lm_head);
            EXPECT_EQ(toVec<int32_t>(inputs.combo_tokens), (std::vector<int32_t>{1, 2}));
            EXPECT_EQ(toVec<int32_t>(inputs.prefix_lengths), (std::vector<int32_t>{0}));
            if (fail_forward) {
                throw std::runtime_error("real model forward failed");
            }
            GptModelOutputs output;
            output.logits = torch::zeros({1, 128}, torch::TensorOptions(torch::kFloat32).device(torch::kCUDA));
            output.logits.index_put_({0, 7}, 100.f);
            return output;
        }
        std::optional<std::string> takeDeferredHiddenStateCaptureError() override {
            return fail_capture ? std::make_optional<std::string>("training publication failed") : std::nullopt;
        }
        bool grant         = true;
        bool fail_forward  = false;
        bool fail_capture  = false;
        int  reserve_count = 0;
        int  cancel_count  = 0;
    };
    for (int failure : {0, 1, 2, 3}) {
        SCOPED_TRACE(failure);
        auto  params        = makeExecutorParams();
        auto  executor      = std::make_shared<NormalExecutor>(params, nullptr);
        auto  model         = std::make_unique<CaptureModel>();
        auto* observed      = model.get();
        model->fail_capture = failure == 1;
        model->fail_forward = failure == 2;
        model->grant        = failure != 3;
        executor->setModel(std::move(model));
        auto query                                    = std::make_shared<GenerateInput>();
        query->request_id                             = 42;
        query->input_ids                              = hostIntBuffer({1, 2});
        query->generate_config                        = std::make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens        = 1;
        query->generate_config->capture_hidden_states = true;
        query->generate_config->reuse_cache           = false;
        query->generate_config->top_k                 = 1;
        query->generate_config->ignore_eos            = true;
        auto stream                                   = std::make_shared<NormalGenerateStream>(
            query, params.model_config_, params.runtime_config, ResourceContext{}, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        const bool granted               = executor->tryReserveHiddenStateCapture(42, 2);
        ASSERT_EQ(granted, failure != 3);
        if (granted) {
            const std::weak_ptr<NormalExecutor> owner = executor;
            stream->setHiddenStateCaptureReservation([owner] {
                if (auto executor = owner.lock()) {
                    executor->cancelHiddenStateCapture(42);
                }
            });
        }
        if (failure == 2) {
            EXPECT_THROW(executor->process({stream}), std::runtime_error);
        } else {
            ASSERT_TRUE(executor->process({stream}).ok());
            EXPECT_FALSE(stream->hasError());
            auto output = stream->nextOutput();
            ASSERT_TRUE(output.ok());
            ASSERT_EQ(output.value().generate_outputs.size(), 1u);
            EXPECT_TRUE(output.value().generate_outputs[0].finished);
            EXPECT_EQ(toVec<int32_t>(output.value().generate_outputs[0].output_ids), (std::vector<int32_t>{7}));
        }
        stream->cancelHiddenStateCapture();
        EXPECT_EQ(observed->reserve_count, 1);
        EXPECT_EQ(observed->cancel_count, granted ? 1 : 0);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testExecutorUsesGatheredModeAndPreservesCaptureErrors) {
    class CaptureModel: public ModelBase {
    public:
        GptModelOutputs forward(const GptModelInputs& inputs) override {
            ++forward_count;
            EXPECT_EQ(inputs.skip_lm_head, expected_prefill_only);
            EXPECT_FALSE(inputs.capture_hidden_states);  // This fixture uses an unreserved warmup gatherer.
            GptModelOutputs output;
            // No logits at all for target-only prefill: entering the sampler is a regression.
            if (!inputs.skip_lm_head) {
                output.logits = torch::zeros({inputs.lm_output_indexes.size(0), 128},
                                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            }
            return output;
        }

        std::optional<std::string> takeDeferredHiddenStateCaptureError() override {
            return fail_capture ? std::make_optional<std::string>("capture publication failed") : std::nullopt;
        }

        bool expected_prefill_only = true;
        bool fail_capture          = false;
        int  forward_count         = 0;
    };

    struct TestCase {
        const char* name;
        int         first_max_new_tokens;
        int         second_max_new_tokens;
        bool        first_has_error;
        bool        second_is_decode;
        bool        expected_prefill_only;
        bool        fail_capture;
    };
    const std::vector<TestCase> cases = {
        {"prefill-only", 0, 0, false, false, true, false},
        {"failed-generation-before-prefill", 1, 0, true, false, true, false},
        {"failed-prefill-before-generation", 0, 1, true, false, false, false},
        {"failed-prefill-before-decode-generation", 0, 1, true, true, false, false},
        {"decode-prefill-precedes-context-generation", 1, 0, false, true, true, false},
        {"decode-generation-precedes-context-prefill", 0, 1, false, true, false, false},
        {"capture-failure", 0, 0, false, false, true, true},
        {"generation-capture-failure", 1, 1, false, false, false, true},
        {"capture-failure-preserves-existing-error", 1, 0, true, false, true, true},
    };
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        auto           params = makeExecutorParams();
        NormalExecutor executor(params, nullptr);
        // Reuse the warmup gatherer's fake KV layout, but run normal executor dispatch.
        executor.setBatchProcessor(std::make_unique<NormalBatchStreamProcessor>(
            params.model_config_, params.pd_sep_config, params.profiling_debug_logging_config, CacheConfig(), true));
        auto  model                  = std::make_unique<CaptureModel>();
        auto* observed_model         = model.get();
        model->expected_prefill_only = test_case.expected_prefill_only;
        model->fail_capture          = test_case.fail_capture;
        executor.setModel(std::move(model));

        ResourceContext resource_context;
        auto            make_stream = [&](int max_new_tokens) {
            auto query                             = make_shared<GenerateInput>();
            query->input_ids                       = hostIntBuffer({1, 2});
            query->generate_config                 = make_shared<GenerateConfig>();
            query->generate_config->max_new_tokens = max_new_tokens;
            query->generate_config->top_k          = 1;
            query->generate_config->aux_info       = true;
            auto stream                            = make_shared<NormalGenerateStream>(
                query, params.model_config_, params.runtime_config, resource_context, nullptr);
            stream->generate_status_->status = StreamState::RUNNING;
            return stream;
        };
        auto first  = make_stream(test_case.first_max_new_tokens);
        auto second = make_stream(test_case.second_max_new_tokens);
        if (test_case.first_has_error) {
            first->reportError(ErrorCode::INVALID_PARAMS, "original request error");
        }
        second->setIsContextStream(!test_case.second_is_decode);
        ASSERT_TRUE(executor.process({first, second}).ok());
        EXPECT_EQ(observed_model->forward_count, 1);
        EXPECT_FALSE(first->hasPendingAsyncBookkeeping());
        EXPECT_FALSE(second->hasPendingAsyncBookkeeping());
        if (test_case.first_has_error) {
            EXPECT_EQ(first->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(first->stopReason(), "original request error");
        }
        if (test_case.second_is_decode && !test_case.first_has_error) {
            // Mixed batches still forward for collective alignment, but may not publish success.
            EXPECT_EQ(first->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(second->statusInfo().code(), ErrorCode::INVALID_PARAMS);
            EXPECT_EQ(second->stopReason(), kMixedExecutionModeBatchError);
            EXPECT_FALSE(first->hasOutput());
            EXPECT_FALSE(second->hasOutput());
        } else {
            ASSERT_FALSE(second->hasError());
            auto output = second->nextOutput();
            ASSERT_TRUE(output.ok());
            ASSERT_EQ(output.value().generate_outputs.size(), 1);
            EXPECT_TRUE(output.value().generate_outputs[0].finished);
            EXPECT_EQ(second->outputTokenLen(), test_case.second_max_new_tokens);
            EXPECT_EQ(second->seqLength(), 2 + test_case.second_max_new_tokens);
            if (test_case.expected_prefill_only) {
                EXPECT_EQ(output.value().generate_outputs[0].output_ids.sizes(), (torch::IntArrayRef{1, 0}));
            }
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, prefillOnlyRejectsNonEmptyOrMalformedTokenUpdates) {
    const auto config = makeLogprobsModelConfig();
    for (const auto& shape : std::vector<std::vector<int64_t>>{{0}, {2, 0}, {1, 1}}) {
        auto             stream = makeLogprobsStream(config, false, 0, 0, false);
        StreamUpdateInfo update{torch::empty(shape, torch::kInt32), 0};
        const bool       core_dump_on_exception      = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception = false;
        EXPECT_THROW(stream->update(update), std::runtime_error);
        StaticConfig::user_ft_core_dump_on_exception = core_dump_on_exception;
        EXPECT_EQ(stream->outputTokenLen(), 0);
        EXPECT_FALSE(stream->hasOutput());
        EXPECT_FALSE(stream->hasEvent(StreamEvents::GenerateDone));
    }
}

TEST_F(NormalBatchStreamProcessorTest, capturePolicyIsPassedToModelConstruction) {
    class EmptyModel: public ModelBase {
    public:
        GptModelOutputs forward(const GptModelInputs&) override {
            return {};
        }
    };
    struct RestoreFactory {
        ~RestoreFactory() {
            NormalExecutor::test_model_factory = nullptr;
        }
    } restore_factory;
    for (auto role : {RoleType::PREFILL, RoleType::PDFUSION, RoleType::DECODE}) {
        for (bool warm_up : {false, true}) {
            for (bool draft : {false, true}) {
                auto params                                         = makeExecutorParams();
                params.pd_sep_config.role_type                      = role;
                params.model_config_.hidden_state_capture_layer_ids = {0};
                bool constructed                                    = false;
                NormalExecutor::test_model_factory                  = [&](const GptModelInitParams& init) {
                    constructed        = true;
                    const bool capture = !warm_up && !draft && role != RoleType::DECODE;
                    EXPECT_EQ(init.hidden_state_capture_layer_ids,
                              capture ? std::vector<int64_t>{0} : std::vector<int64_t>{});
                    EXPECT_EQ(init.hidden_state_capture_dtype, params.model_config_.hidden_state_capture_dtype);
                    return std::make_unique<EmptyModel>();
                };
                NormalExecutor executor(params, nullptr, warm_up, draft);
                EXPECT_TRUE(constructed);
            }
        }
    }
}

}  // namespace rtp_llm
