#include <memory>
#include <numeric>
#include "torch/all.h"
#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/DSV41CacheState.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"

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

class NormalBatchStreamProcessorTest: public DeviceTestBase {};

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

        auto merge_input_status = processor.gatherModelInput(stream_groups, holder);

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

        auto&           model_input       = merge_input_status.value();
        vector<int>     combo_tokens      = {1, -1, -1, -1, 2, 3, 4, 5, 6, 7, -1, -1, 8};
        vector<int>     input_lengths     = {5, 3, 5};
        vector<int>     text_tokens_mask  = {1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1};
        vector<int>     mm_features_locs  = {1, 10};
        vector<int64_t> mm_features_spans = {0, 1, 4, 2, 2, 4};

        EXPECT_EQ(combo_tokens, toVec<int>(model_input.combo_tokens));
        EXPECT_EQ(input_lengths, toVec<int>(model_input.input_lengths));
        EXPECT_EQ(text_tokens_mask, toVec<int>(model_input.text_tokens_mask));
        EXPECT_EQ(mm_features_locs, toVec<int>(model_input.mm_features_locs));
        EXPECT_EQ(mm_features_spans, toVec<int64_t>(model_input.mm_features_spans));

        EXPECT_EQ(model_input.multimodal_features.value().size(), 2);
        EXPECT_EQ(model_input.multimodal_features.value()[0].numel(), 3 * 10);
        EXPECT_EQ(model_input.multimodal_features.value()[1].numel(), 2 * 10);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testMultimodalGatherSlicesReusedPrefix) {
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

    auto query                 = make_shared<GenerateInput>();
    query->input_ids           = hostIntBuffer({1, -1, -1, -1, 2});
    query->generate_config     = make_shared<GenerateConfig>();
    query->mm_locs             = torch::tensor({1}, torch::kInt32);
    query->text_tokens_mask    = torch::tensor({1, 0, 0, 0, 1}, torch::kInt32);
    auto full_feature          = torch::arange(30, torch::kFloat16).reshape({3, 10});
    query->multimodal_features = {full_feature};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(true);
    stream->setReuseLength(2);
    stream->generate_status_->status = StreamState::RUNNING;

    std::list<GenerateStreamPtr> stream_list;
    stream_list.emplace_back(stream);
    StreamGroups streams(stream_list);
    TensorHolder holder;
    auto         gathered = processor.gatherModelInput(streams, holder);
    ASSERT_TRUE(gathered.ok());
    auto& model_input = gathered.value();

    EXPECT_EQ(toVec<int>(model_input.combo_tokens), std::vector<int>({-1, -1, 2}));
    EXPECT_EQ(toVec<int>(model_input.text_tokens_mask), std::vector<int>({0, 0, 1}));
    EXPECT_EQ(toVec<int>(model_input.mm_features_locs), std::vector<int>({0}));
    EXPECT_EQ(toVec<int64_t>(model_input.mm_features_spans), std::vector<int64_t>({0, 1, 4}));
    ASSERT_TRUE(model_input.multimodal_features.has_value());
    ASSERT_EQ(model_input.multimodal_features->size(), 1);
    EXPECT_TRUE(torch::equal(model_input.multimodal_features->at(0).cpu(), full_feature.slice(0, 1, 3)));
}

class V41BatchStreamProcessorTest: public NormalBatchStreamProcessorTest {
protected:
    void SetUp() override {
        NormalBatchStreamProcessorTest::SetUp();
        model_config_.max_seq_len = 4096;
        model_config_.vocab_size = model_config_.input_vocab_size = 129280;
        model_config_.num_layers                                  = 40;
        model_config_.attn_config.tokens_per_block                = 128;
        model_config_.attn_config.dsv41_cache_layout_version      = 1;
        cache_config_.seq_size_per_block = cache_config_.kernel_seq_size_per_block = 128;
        cache_config_.layer_num = cache_config_.layer_all_num = 40;
        cache_config_.group_types                             = {CacheGroupType::FULL};
        cache_config_.layer_to_group_id.assign(40, 0);
        processor_ = std::make_unique<NormalBatchStreamProcessor>(
            model_config_, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache_config_, false);
    }

    GenerateStreamPtr
    makeStream(const std::vector<int32_t>& tokens, int64_t id, int choices = 1, bool canonical = true) {
        auto input                                   = std::make_shared<GenerateInput>();
        input->input_ids                             = hostIntBuffer(tokens);
        input->request_id                            = id;
        input->need_release_resource                 = false;
        input->fake_query                            = !canonical;
        input->generate_config                       = std::make_shared<GenerateConfig>();
        input->generate_config->max_new_tokens       = 8;
        input->generate_config->num_return_sequences = choices;
        input->generate_config->ignore_eos           = true;
        if (canonical) {
            auto prepared         = std::make_shared<V41RequestInputs>();
            prepared->token_types = torch::full({static_cast<int64_t>(tokens.size())}, -1, torch::kInt32);
            prepared->image_mask  = torch::zeros({static_cast<int64_t>(tokens.size())}, torch::kBool);
            input->v41_inputs     = prepared;
        }
        auto stream =
            std::make_shared<NormalGenerateStream>(input, model_config_, RuntimeConfig{}, ResourceContext{}, nullptr);
        BatchKVCacheResource resource;
        resource.resetBatchSize(choices);
        resource.initGroups(1, 40, cache_config_.layer_to_group_id);
        for (int batch = 0; batch < choices; ++batch) {
            resource.setBatchBlocks(batch, 0, {1 + 4 * batch, 2 + 4 * batch, 3 + 4 * batch, 4 + 4 * batch});
            resource.cacheResource(batch).setDsv41CacheState(std::make_shared<DSV41CacheState>(
                DSV41CacheIdentity{std::string(40, 'a'), std::string(64, 'b'), DSV41ReplayMode::FULL, 1, 128, 128}));
        }
        stream->setKVCache(resource);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    }

    static std::shared_ptr<DSV41CacheState> state(const GenerateStreamPtr& stream, int batch = 0) {
        return stream->kvCachePtr()->cacheResource(batch).dsv41CacheState();
    }

    static MergedOutput sampled(const std::vector<int32_t>& tokens) {
        MergedOutput output;
        output.sampler_output.token_ids = hostIntBuffer(tokens).reshape({static_cast<int64_t>(tokens.size()), 1});
        return output;
    }

    ModelConfig                                 model_config_;
    CacheConfig                                 cache_config_;
    std::unique_ptr<NormalBatchStreamProcessor> processor_;
};

TEST_F(V41BatchStreamProcessorTest, PrefillDispatchMakesEverySequenceReadyForItsNextDecode) {
    const int64_t                first_id  = (1LL << 40) + 17;
    const int64_t                second_id = (1LL << 40) + 23;
    auto                         first     = makeStream({11, 12, 13}, first_id, 2);
    auto                         second    = makeStream({21, 22}, second_id);
    std::list<GenerateStreamPtr> streams{first, second};
    StreamGroups                 prefill(streams);
    TensorHolder                 holder;
    auto                         gathered = processor_->gatherModelInput(prefill, holder);
    ASSERT_TRUE(gathered.ok());
    EXPECT_EQ(toVec<int64_t>(gathered->v41_request_id), (std::vector<int64_t>{first_id, first_id, second_id}));
    EXPECT_EQ(toVec<int>(gathered->input_lengths), (std::vector<int>{3, 3, 2}));
    EXPECT_EQ(toVec<bool>(gathered->v41_state_ready), (std::vector<bool>{true, true, true}));
    EXPECT_EQ(toVec<bool>(gathered->v41_is_fake), (std::vector<bool>{false, false, false}));
    EXPECT_EQ(toVec<int>(gathered->combo_tokens), (std::vector<int>{11, 12, 13, 11, 12, 13, 21, 22}));
    EXPECT_EQ(state(first, 0)->view().target_ready_end, 0);
    EXPECT_EQ(state(first, 1)->view().target_ready_end, 0);

    ASSERT_TRUE(processor_->dispatch(prefill, sampled({31, 32, 41})).ok());
    EXPECT_FALSE(first->isContextStream());
    EXPECT_EQ(first->seqLength(), 4);
    EXPECT_EQ(second->seqLength(), 3);
    EXPECT_EQ(state(first, 0)->view().target_ready_end, 3);
    EXPECT_EQ(state(first, 1)->view().target_ready_end, 3);
    EXPECT_EQ(state(second)->view().target_ready_end, 2);
    StreamGroups decode(streams);
    gathered = processor_->gatherModelInput(decode, holder);
    ASSERT_TRUE(gathered.ok());
    EXPECT_EQ(gathered->request_id.numel(), 0);
    EXPECT_EQ(toVec<int64_t>(gathered->v41_request_id), (std::vector<int64_t>{first_id, first_id, second_id}));
    EXPECT_EQ(toVec<int>(gathered->sequence_lengths), (std::vector<int>{3, 3, 2}));
    EXPECT_EQ(toVec<int>(gathered->combo_tokens), (std::vector<int>{31, 32, 41}));
    EXPECT_EQ(toVec<bool>(gathered->v41_state_ready), (std::vector<bool>{true, true, true}));
    EXPECT_EQ(toVec<int>(gathered->engram_history_ids), (std::vector<int>{11, 12, 13, 11, 12, 13, 0, 21, 22}));
    EXPECT_EQ(toVec<bool>(gathered->engram_history_valid),
              (std::vector<bool>{true, true, true, true, true, true, false, true, true}));

    ASSERT_TRUE(processor_->dispatch(decode, sampled({33, 34, 42})).ok());
    EXPECT_EQ(state(first, 0)->view().target_ready_end, 4);
    EXPECT_EQ(state(first, 1)->view().target_ready_end, 4);
    EXPECT_EQ(state(second)->view().target_ready_end, 3);
    first->kvCachePtr()->cacheResource(1).setDsv41CacheState(nullptr);
    StreamGroups next_decode(streams);
    gathered = processor_->gatherModelInput(next_decode, holder);
    ASSERT_TRUE(gathered.ok());
    EXPECT_EQ(toVec<bool>(gathered->v41_state_ready), (std::vector<bool>{true, false, true}));
}

TEST_F(V41BatchStreamProcessorTest, FakeRankRowsStayInvalidAndDoNotPublishTargetState) {
    auto fake = makeStream({7}, 0, 1, false);
    fake->setIsFakeStream(true);
    fake->update({.new_tokens = hostIntBuffer({0}).reshape({1, 1}), .num_new_tokens = 1});
    state(fake)->markTargetReady(1);
    std::list<GenerateStreamPtr> streams{fake};
    StreamGroups                 groups(streams);
    TensorHolder                 holder;
    auto                         gathered = processor_->gatherModelInput(groups, holder);
    ASSERT_TRUE(gathered.ok());
    ASSERT_TRUE(gathered->is_fake_stream);
    EXPECT_EQ(toVec<int>(gathered->sequence_lengths), (std::vector<int>{1}));
    EXPECT_EQ(toVec<int64_t>(gathered->v41_request_id), (std::vector<int64_t>{0}));
    EXPECT_EQ(toVec<bool>(gathered->v41_is_fake), (std::vector<bool>{true}));
    EXPECT_EQ(toVec<bool>(gathered->v41_state_ready), (std::vector<bool>{false}));
    EXPECT_EQ(toVec<int>(gathered->v41_token_types), (std::vector<int>{-1}));
    EXPECT_EQ(toVec<bool>(gathered->v41_token_valid), (std::vector<bool>{false}));
    EXPECT_EQ(toVec<int>(gathered->engram_history_ids), (std::vector<int>{0, 0, 0}));
    EXPECT_EQ(toVec<bool>(gathered->engram_history_valid), (std::vector<bool>{false, false, false}));
    ASSERT_TRUE(processor_->dispatch(groups, sampled({9})).ok());
    EXPECT_EQ(state(fake)->view().target_ready_end, 1);
    EXPECT_EQ(state(fake)->view().encoder_materialized_end, 1);
}

TEST_F(V41BatchStreamProcessorTest, CacheDataHitsCannotAdvancePastRestoredTargetState) {
    std::vector<int32_t> tokens(300);
    std::iota(tokens.begin(), tokens.end(), 1);
    auto  stream   = makeStream(tokens, 101);
    auto& resource = stream->streamCacheResource();
    // Only block geometry is needed here; no allocator or physical KV pool is initialized.
    resource.resource_context_.cache_manager = std::make_shared<KVCacheManager>(cache_config_, true);
    auto restored = std::make_shared<KVCacheResource>(stream->kvCachePtr()->cacheResource(0));
    restored->setDeviceReuseBlockNum(1);
    restored->setMemoryReuseBlockNum(1);
    auto matches = std::make_shared<FusedAsyncContext>(std::vector<std::shared_ptr<AsyncContext>>{});
    auto loaded  = std::make_shared<FusedAsyncReadContext>(matches, restored, nullptr);
    loaded->setFusedReadContext(nullptr);
    resource.waitLoadCacheDone(loaded);
    EXPECT_EQ(restored->reuseBlockNum(), 2);
    EXPECT_EQ(stream->reuseLength(), 0);
    EXPECT_EQ(stream->initialReuseLength(), 0);

    restored->setDsv41CacheState(nullptr);
    resource.waitLoadCacheDone(loaded);
    EXPECT_EQ(stream->reuseLength(), 0);
    EXPECT_EQ(stream->initialReuseLength(), 0);
    std::list<GenerateStreamPtr> streams{stream};
    TensorHolder                 holder;
    StreamGroups                 unrecovered(streams);
    auto                         gathered = processor_->gatherModelInput(unrecovered, holder);
    ASSERT_TRUE(gathered.ok());
    EXPECT_EQ(toVec<int>(gathered->prefix_lengths), (std::vector<int>{0}));
    EXPECT_EQ(toVec<int>(gathered->combo_tokens), tokens);

    restored->setDsv41CacheState(state(stream));
    state(stream)->markTargetReady(128);
    resource.waitLoadCacheDone(loaded);
    EXPECT_EQ(restored->reuseBlockNum(), 2);
    EXPECT_EQ(stream->reuseLength(), 128);
    EXPECT_EQ(stream->initialReuseLength(), 128);
    StreamGroups recovered(streams);
    gathered = processor_->gatherModelInput(recovered, holder);
    ASSERT_TRUE(gathered.ok());
    EXPECT_EQ(toVec<int>(gathered->prefix_lengths), (std::vector<int>{128}));
    EXPECT_EQ(toVec<int>(gathered->combo_tokens), (std::vector<int32_t>(tokens.begin() + 128, tokens.end())));
    EXPECT_EQ(toVec<bool>(gathered->v41_state_ready), (std::vector<bool>{true}));
    EXPECT_EQ(toVec<int>(gathered->engram_history_ids.narrow(0, 0, 1)), (std::vector<int>{126, 127, 128}));
}

}  // namespace rtp_llm
