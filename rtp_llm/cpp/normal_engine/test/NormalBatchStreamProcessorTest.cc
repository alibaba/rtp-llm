#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include "torch/all.h"
#include "gtest/gtest.h"

#define private public
#define protected public
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/normal_engine/PromptScoringRows.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

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

static void initFullCacheConfig(CacheConfig& cache_config, int layer_num) {
    auto spec = std::make_shared<MHAKVCacheSpec>();
    spec->tag = "default";
    std::vector<int> layer_ids(static_cast<size_t>(layer_num));
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    cache_config.layer_num     = static_cast<uint32_t>(layer_num);
    cache_config.layer_all_num = static_cast<uint32_t>(layer_num);
    cache_config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {"default"});
}

class NormalBatchStreamProcessorTest: public DeviceTestBase {
protected:
    static ModelConfig makeOutputVocabModelConfig(std::vector<int64_t> output_vocab_ids = {0, 2, 7},
                                                  int64_t              padded_size      = 0) {
        ModelConfig model_config;
        model_config.max_seq_len      = 8;
        model_config.vocab_size       = 10;
        model_config.num_layers       = 1;
        model_config.output_vocab_ids = std::move(output_vocab_ids);
        model_config.output_vocab_padded_size =
            padded_size > 0 ? padded_size : static_cast<int64_t>(model_config.output_vocab_ids.size());
        return model_config;
    }
};

TEST_F(NormalBatchStreamProcessorTest, testWarmUpWithoutCacheManager) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len      = 2048;
    model_config.vocab_size       = 2048;
    model_config.input_vocab_size = 2048;
    model_config.num_layers       = 1;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    RuntimeConfig               runtime_config;

    auto query             = make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({1, 2, 3});
    query->generate_config = make_shared<GenerateConfig>();
    GenerateStreamPtr stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;
    StreamGroups stream_groups({stream});

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, true);

    EXPECT_EQ(processor.model_input_gatherer_config_.kv_cache_group_nums, 0);
    EXPECT_TRUE(processor.model_input_gatherer_config_.kv_cache_group_types.empty());
    ASSERT_EQ(stream->kvCache().groupNums(), 1);
    EXPECT_EQ(stream->kvCache().cacheResource().soleGroupTagForLayer(0), "__warmup__");
    TensorHolder holder;
    auto         model_input = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(model_input.ok());
    EXPECT_FALSE(model_input->kv_cache_block_id.defined());
    EXPECT_FALSE(model_input->kv_cache_kernel_block_id.defined());
}

TEST_F(NormalBatchStreamProcessorTest, testCacheKeyWidthIndependentOfBlockTable) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2048;
    model_config.num_layers  = 1;

    PDSepConfig pd_sep_config;
    pd_sep_config.role_type = RoleType::PREFILL;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({1, 2, 3});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->num_return_sequences = 2;
    GenerateStreamPtr stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    BatchKVCacheResource resource;
    resource.resetBatchSize(2);
    resource.initGroups(cache_config.topologyPtr());
    resource.setBatchBlocks(0, 0, {1, 2});
    resource.setBatchBlocks(1, 0, {3, 4});
    resource.setBatchCacheKeys(0, CacheKeysType{101, 102, 103});
    resource.setBatchCacheKeys(1, CacheKeysType{201, 202, 203, 204, 205});
    stream->setKVCache(resource);
    stream->generate_status_->status = StreamState::RUNNING;

    StreamGroups stream_groups({stream});
    EXPECT_EQ(stream_groups.curBlocksNum(), 2);
    EXPECT_EQ(stream_groups.maxCacheKeysNum(), 5);

    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    TensorHolder holder;
    auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(merge_input_status.ok());
    EXPECT_TRUE(merge_input_status.value().pd_separation);
    const auto& cache_keys = merge_input_status.value().cache_keys;
    ASSERT_TRUE(cache_keys.defined());
    EXPECT_EQ(cache_keys.size(0), 2);
    EXPECT_EQ(cache_keys.size(1), 5);
    EXPECT_EQ(toVec<int64_t>(cache_keys), (std::vector<int64_t>{101, 102, 103, 0, 0, 201, 202, 203, 204, 205}));
}

class TestStatefulLogitsProcessor: public BaseLogitsProcessor {
public:
    explicit TestStatefulLogitsProcessor(bool async_device_state): async_device_state_(async_device_state) {}

    std::optional<ErrorInfo> process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override {
        (void)inputs;
        (void)start_idx;
        (void)finish_idx;
        return std::nullopt;
    }

    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override {
        (void)src_batch_indices;
    }

    std::optional<ErrorInfo> updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override {
        (void)new_tokens;
        accepted_token_len_ += num_new_tokens;
        return std::nullopt;
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
    model_config.attn_config.kv_cache_dtype = KvCacheDataType::FP8;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    cache_config.kv_block_stride_bytes = 4096;
    cache_config.kv_scale_stride_bytes = 256;

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
    addr1.initGroups(cache_config.topologyPtr());
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
    addr2.initGroups(cache_config.topologyPtr());
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
    addr3.initGroups(cache_config.topologyPtr());
    addr3.setBatchBlocks(0, 0, {9, 10});
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4 = make_shared<GenerateInput>();
    query4->input_ids                     = hostIntBuffer({1, 2, 3, 4});
    query4->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream4 =
        make_shared<NormalGenerateStream>(query4, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr4;
    addr4.resetBatchSize(1);
    addr4.initGroups(cache_config.topologyPtr());
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

    for (const auto* device_input : {"0", "1"}) {
        autil::EnvGuard device_input_guard("RTP_LLM_DEVICE_INPUT", device_input);
        StreamGroups    stream_groups(streams);
        TensorHolder    holder;

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
        EXPECT_EQ(model_input.input_lengths.device(), model_input.prefix_lengths.device());
#if USING_CUDA
        EXPECT_TRUE(model_input.input_lengths.is_cpu());
        EXPECT_TRUE(model_input.prefix_lengths.is_cpu());
        EXPECT_TRUE(model_input.prefix_lengths.is_pinned());
#endif
        EXPECT_EQ(kv_cache_block_id, toVec<int>(model_input.kv_cache_block_id));
        EXPECT_EQ(model_input.kv_block_stride_bytes, cache_config.kv_block_stride_bytes);
        EXPECT_EQ(model_input.kv_scale_stride_bytes, cache_config.kv_scale_stride_bytes);
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

TEST_F(NormalBatchStreamProcessorTest, testAsyncMropeGatherUsesPublishedTokenAndPosition) {
    autil::EnvGuard device_input("RTP_LLM_DEVICE_INPUT", "1");
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                           = 128;
    model_config.vocab_size                            = 128;
    model_config.num_layers                            = 1;
    model_config.mm_model_config.mm_position_ids_style = PositionIdsStyle::MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    CacheConfig cache_config;
    initFullCacheConfig(cache_config, 1);
    RuntimeConfig runtime_config;
    auto          query    = make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({1, 2, 3});
    query->generate_config = make_shared<GenerateConfig>();
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->context_position_ids_ = hostIntBuffer({0, 0, 0, 3, 5, 7, 8, 10, 12});
    stream->setIsContextStream(false);
    stream->generate_status_->status = StreamState::RUNNING;
    BatchKVCacheResource resource;
    resource.resetBatchSize(1);
    resource.initGroups(cache_config.topologyPtr());
    resource.setBatchBlocks(0, 0, {1, 2, 3, 4});
    stream->setKVCache(resource);
    NormalBatchStreamProcessor processor(
        model_config, PDSepConfig{}, ProfilingDebugLoggingConfig{}, cache_config, false);
    const auto device_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    // Deliberately leave host tokens/length behind, as an unfinished dispatch
    // worker would. Both token ids and all three mRoPE axes must use this step.
    for (int next_length : {5, 6}) {
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .last_sample_token_gpu = torch::full({1}, 42 + next_length, device_i32),
            .next_seq_len_gpu      = torch::full({1}, next_length, device_i32),
            .last_real_seq_len     = next_length - 1,
            .next_real_seq_len     = next_length,
        });
        StreamGroups groups({stream});
        TensorHolder holder;
        auto         result = processor.gatherModelInput(groups, holder);
        ASSERT_TRUE(result.ok()) << result.status();
        EXPECT_TRUE(result->combo_tokens.is_cuda());
#if USING_CUDA
        EXPECT_TRUE(result->sequence_lengths.is_cpu());
        EXPECT_TRUE(result->sequence_lengths.is_pinned());
        EXPECT_TRUE(result->input_lengths.is_cpu());
        EXPECT_TRUE(result->kv_cache_kernel_block_id.is_cpu());
#else
        EXPECT_TRUE(result->sequence_lengths.is_cuda());
#endif
        EXPECT_EQ(toVec<int32_t>(result->combo_tokens), (vector<int32_t>{42 + next_length}));
        EXPECT_EQ(toVec<int32_t>(result->sequence_lengths), (vector<int32_t>{next_length - 1}));
        EXPECT_EQ(toVec<int32_t>(result->combo_position_ids), (vector<int32_t>(3, 12 + next_length - 3)));
        EXPECT_EQ(stream->seqLength(), 3);
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

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig                  runtime_config;
    std::shared_ptr<GenerateInput> query1         = make_shared<GenerateInput>();
    query1->input_ids                             = hostIntBuffer({1});
    query1->generate_config                       = make_shared<GenerateConfig>();
    query1->generate_config->return_softmax_probs = true;
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(cache_config.topologyPtr());
    addr1.setBatchBlocks(0, 0, {1});
    stream1->setKVCache(addr1);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }
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

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabMapsGreedyTokenBeforeStreamUpdate) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig();
    RuntimeConfig   runtime_config;

    auto query             = make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({2});
    query->generate_config = make_shared<GenerateConfig>();
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 2}, torch::kInt32).reshape({1, 2});

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabSamplerFailureDoesNotBlockPeerStream) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig();
    RuntimeConfig   runtime_config;

    auto make_stream = [&]() {
        auto query             = make_shared<GenerateInput>();
        query->input_ids       = hostIntBuffer({2});
        query->generate_config = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };
    auto failed_stream  = make_stream();
    auto healthy_stream = make_stream();

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups stream_groups({failed_stream, healthy_stream});
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 99, 2, 2}, torch::kInt32).reshape({2, 2});
    merge_outputs.sampler_output.success   = torch::tensor({false, true}, torch::kBool);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_TRUE(failed_stream->hasError());
    EXPECT_EQ(failed_stream->completeTokenIdsVec(0), (std::vector<int>{2}));
    EXPECT_FALSE(healthy_stream->hasError());
    EXPECT_EQ(healthy_stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testInvalidCompactTokenDoesNotIndexProbabilitiesOrBlockPeer) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig();
    RuntimeConfig   runtime_config;

    auto make_stream = [&]() {
        auto query                                   = make_shared<GenerateInput>();
        query->input_ids                             = hostIntBuffer({2});
        query->generate_config                       = make_shared<GenerateConfig>();
        query->generate_config->max_new_tokens       = 1;
        query->generate_config->return_softmax_probs = true;
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };
    auto failed_stream  = make_stream();
    auto healthy_stream = make_stream();

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups stream_groups({failed_stream, healthy_stream});
    MergedOutput merge_outputs;
    merge_outputs.model_output.logits =
        torch::tensor({0.0f, 1.0f, 9.0f, 0.0f, 1.0f, 9.0f}, torch::kFloat32).reshape({2, 3}).to(torch::kCUDA);
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 99, 2, 2}, torch::kInt32).reshape({2, 2});
    merge_outputs.sampler_output.success   = torch::tensor({true, true}, torch::kBool);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_TRUE(failed_stream->hasError());
    EXPECT_EQ(failed_stream->completeTokenIdsVec(0), (std::vector<int>{2}));
    EXPECT_FALSE(healthy_stream->hasError());
    EXPECT_EQ(healthy_stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testDynamicBeamRejectsParentOutsidePreviousBatch) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig({0, 1, 2, 4, 7, 9});
    RuntimeConfig   runtime_config;

    auto query                                 = make_shared<GenerateInput>();
    query->input_ids                           = hostIntBuffer({2});
    query->generate_config                     = make_shared<GenerateConfig>();
    query->generate_config->variable_num_beams = {2};
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_FALSE(stream->hasError());
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids  = torch::tensor({2, 1, 2, 1}, torch::kInt32).reshape({2, 2});
    merge_outputs.sampler_output.beam_index = torch::tensor({0, 1}, torch::kInt32);
    merge_outputs.sampler_output.success    = torch::tensor({true}, torch::kBool);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabRestoresOnlyCurrentBeamToken) {
    NormalOutputDispatcher dispatcher({0, 2, 4, 7, 9});
    auto batch_token_ids   = torch::tensor({100, 3, 101, 200, 4, 201}, torch::kInt32).reshape({2, 3}).contiguous();
    auto current_token_ids = torch::tensor({3, 4}, torch::kInt32).reshape({2, 1}).contiguous();
    GenerateStreamPtr unused_stream;

    ASSERT_TRUE(dispatcher.restoreCurrentTokenIds(unused_stream, batch_token_ids, current_token_ids, 1));
    EXPECT_EQ(toVec<int32_t>(batch_token_ids), (std::vector<int32_t>{100, 7, 101, 200, 9, 201}));
    EXPECT_EQ(toVec<int32_t>(current_token_ids), (std::vector<int32_t>{7, 9}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabSelectedProbabilityUsesCompactToken) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig();
    RuntimeConfig   runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({2});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens       = 1;
    query->generate_config->return_softmax_probs = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    merge_outputs.model_output.logits =
        torch::tensor({0.0f, 1.0f, 9.0f}, torch::kFloat32).reshape({1, 3}).to(torch::kCUDA);
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 2}, torch::kInt32).reshape({1, 2});

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
    const auto expected_probability = torch::softmax(torch::tensor({0.0f, 1.0f, 9.0f}), -1)[2].item<float>();
    EXPECT_NEAR(stream->getSoftmaxProbs()[0][1].item<float>(), expected_probability, 1e-6);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabClampsPositiveTopKToLogitsWidth) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig({0, 7});
    RuntimeConfig   runtime_config;

    auto query                    = make_shared<GenerateInput>();
    query->input_ids              = hostIntBuffer({2});
    query->generate_config        = make_shared<GenerateConfig>();
    query->generate_config->top_k = 8;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_output;
    model_output.logits = torch::zeros({1, 2}, torch::kFloat32).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_output);
    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_EQ(sampler_inputs->top_k.data_ptr<int32_t>()[0], 2);
}

TEST_F(NormalBatchStreamProcessorTest, testDisabledOutputVocabPreservesTopK) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;

    auto query                    = make_shared<GenerateInput>();
    query->input_ids              = hostIntBuffer({2});
    query->generate_config        = make_shared<GenerateConfig>();
    query->generate_config->top_k = 8;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_output;
    model_output.logits = torch::zeros({1, 2}, torch::kFloat32).to(torch::kCUDA);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_output);
    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_EQ(sampler_inputs->top_k.data_ptr<int32_t>()[0], 8);
}

TEST_F(NormalBatchStreamProcessorTest, testPaddedSizeLargerThanKKeepsDispatchAndSamplingOnK) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig({0, 2, 7}, /*padded_size=*/8);
    RuntimeConfig   runtime_config;

    auto query                    = make_shared<GenerateInput>();
    query->input_ids              = hostIntBuffer({2});
    query->generate_config        = make_shared<GenerateConfig>();
    query->generate_config->top_k = 8;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;
    EXPECT_EQ(stream->outputVocabSize(), 3u);

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups stream_groups({stream});

    // top_k is clamped to the K-wide logits, not to the padded width P
    GptModelOutputs model_output;
    model_output.logits = torch::zeros({1, 3}, torch::kFloat32).to(torch::kCUDA);
    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_output);
    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_EQ(sampler_inputs->vocab_size, 3u);
    EXPECT_EQ(sampler_inputs->top_k.data_ptr<int32_t>()[0], 3);

    // dispatch consumes K-wide results; compact-id restoration is insensitive to P
    MergedOutput merge_outputs;
    merge_outputs.sampler_output.token_ids = torch::tensor({2, 2}, torch::kInt32).reshape({1, 2});
    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{2, 7}));
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabPassesCompactEosOnlyToMultiSeqProcessor) {
    ResourceContext resource_context;
    auto            model_config             = makeOutputVocabModelConfig({0, 2, 4, 7, 9});
    model_config.special_tokens.eos_token_id = 7;
    RuntimeConfig runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({2});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->num_return_sequences = 2;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    ASSERT_FALSE(stream->hasError());
    ASSERT_EQ(stream->logits_processor_list_.size(), 1);
    ASSERT_NE(std::dynamic_pointer_cast<MultiSeqLogitsProcessor>(stream->logits_processor_list_[0]), nullptr);

    SamplerInputs inputs;
    inputs.logits        = torch::zeros({2, 5}, torch::kFloat32).to(torch::kCUDA);
    inputs.finished_mask = torch::tensor({false, true}, torch::kBool);
    stream->logits_processor_list_[0]->process(inputs, 0, 2);

    auto processed_logits = inputs.logits.cpu();
    for (int token_id = 0; token_id < 5; ++token_id) {
        if (token_id == 3) {
            EXPECT_FLOAT_EQ(processed_logits[1][token_id].item<float>(), 0.0f);
        } else {
            EXPECT_EQ(processed_logits[1][token_id].item<float>(), -std::numeric_limits<float>::infinity());
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabRejectsUnsupportedRequestOnCurrentStream) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig();
    RuntimeConfig   runtime_config;

    auto penalty_query                                 = make_shared<GenerateInput>();
    penalty_query->input_ids                           = hostIntBuffer({2});
    penalty_query->generate_config                     = make_shared<GenerateConfig>();
    penalty_query->generate_config->repetition_penalty = 1.1f;
    auto penalty_stream =
        make_shared<NormalGenerateStream>(penalty_query, model_config, runtime_config, resource_context, nullptr);
    EXPECT_TRUE(penalty_stream->hasError());
    EXPECT_EQ(penalty_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);

    auto beam_query                        = make_shared<GenerateInput>();
    beam_query->input_ids                  = hostIntBuffer({2});
    beam_query->generate_config            = make_shared<GenerateConfig>();
    beam_query->generate_config->num_beams = 2;
    auto beam_stream =
        make_shared<NormalGenerateStream>(beam_query, model_config, runtime_config, resource_context, nullptr);
    EXPECT_TRUE(beam_stream->hasError());
    EXPECT_EQ(beam_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);

    auto think_query                                  = make_shared<GenerateInput>();
    think_query->input_ids                            = hostIntBuffer({2});
    think_query->generate_config                      = make_shared<GenerateConfig>();
    think_query->generate_config->in_think_mode       = true;
    think_query->generate_config->max_thinking_tokens = 1;
    think_query->generate_config->end_think_token_ids = {7};
    auto think_stream =
        make_shared<NormalGenerateStream>(think_query, model_config, runtime_config, resource_context, nullptr);
    EXPECT_TRUE(think_stream->hasError());
    EXPECT_EQ(think_stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabRejectsEachUnsupportedConfigItem) {
    ResourceContext resource_context;
    auto            model_config = makeOutputVocabModelConfig();
    RuntimeConfig   runtime_config;

    struct RejectCase {
        std::string                          message_keyword;
        std::function<void(GenerateConfig&)> mutate;
    };
    std::vector<RejectCase> cases = {
        {"repetition", [](GenerateConfig& c) { c.repetition_penalty = 1.1f; }},
        {"presence", [](GenerateConfig& c) { c.presence_penalty = 0.1f; }},
        {"frequency", [](GenerateConfig& c) { c.frequency_penalty = 0.1f; }},
        {"no_repeat_ngram_size", [](GenerateConfig& c) { c.no_repeat_ngram_size = 2; }},
        {"full-vocabulary logits", [](GenerateConfig& c) { c.return_logits = true; }},
        {"full-vocabulary logits", [](GenerateConfig& c) { c.return_prompt_logits = true; }},
        {"full-vocabulary logits", [](GenerateConfig& c) { c.return_all_probs = ReturnAllProbsMode::DEFAULT; }},
        {"full-vocabulary logits", [](GenerateConfig& c) { c.calculate_loss = 1; }},
        {"full-vocabulary logits", [](GenerateConfig& c) { c.select_tokens_id = {2}; }},
        {"full-vocabulary logits", [](GenerateConfig& c) { c.select_tokens_str = {"a"}; }},
        {"think mode", [](GenerateConfig& c) { c.in_think_mode = true; }},
    };
    for (const auto& reject_case : cases) {
        auto query             = make_shared<GenerateInput>();
        query->input_ids       = hostIntBuffer({2});
        query->generate_config = make_shared<GenerateConfig>();
        reject_case.mutate(*query->generate_config);
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        EXPECT_TRUE(stream->hasError()) << "keyword=" << reject_case.message_keyword;
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS) << "keyword=" << reject_case.message_keyword;
        EXPECT_NE(stream->statusInfo().ToString().find(reject_case.message_keyword), std::string::npos)
            << "keyword=" << reject_case.message_keyword;
    }
}

TEST_F(NormalBatchStreamProcessorTest, testOutputVocabRejectsMissingPrimaryEos) {
    ResourceContext resource_context;
    // Default special_tokens.eos_token_id is 0; this vocabulary does not contain it.
    auto          model_config = makeOutputVocabModelConfig({2, 4, 7});
    RuntimeConfig runtime_config;

    auto query             = make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({2});
    query->generate_config = make_shared<GenerateConfig>();
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_NE(stream->statusInfo().ToString().find("EOS"), std::string::npos);
}

TEST_F(NormalBatchStreamProcessorTest, testDynamicBeamDispatchReordersAndPlacesTokenAtSeqLength) {
    ResourceContext resource_context;
    ModelConfig     model_config;  // no output vocab: beam layout is orthogonal to pruning
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;

    auto query                                   = make_shared<GenerateInput>();
    query->input_ids                             = hostIntBuffer({5});
    query->generate_config                       = make_shared<GenerateConfig>();
    query->generate_config->variable_num_beams   = {2, 2};
    query->generate_config->max_new_tokens       = 2;
    query->generate_config->return_hidden_states = true;
    query->generate_config->return_logits        = true;
    query->generate_config->return_softmax_probs = true;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    ASSERT_FALSE(stream->hasError());

    // Advance one beam step so currentBatchSize == nextBatchSize == 2 and the
    // stream uses the beam token layout (seqLength 1 -> 2).
    int  error_token_id = -1;
    auto first_tokens   = torch::tensor({5, 1, 5, 2}, torch::kInt32).reshape({2, 2});
    ASSERT_TRUE(
        stream->complete_token_ids_->update(first_tokens, 0, 1, 1, 8, 10, true, stream->streamId(), error_token_id));
    stream->generate_status_->status = StreamState::RUNNING;
    ASSERT_EQ(stream->currentBatchSize(), 2);
    ASSERT_EQ(stream->nextBatchSize(), 2);

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups stream_groups({stream});
    MergedOutput merge_outputs;
    // Per-output-beam rows; the new token sits at column seqLength()==2 while the
    // trailing column holds a different token, so a wrong token_position (last column)
    // would be observable instead of silently passing.
    merge_outputs.sampler_output.token_ids  = torch::tensor({5, 1, 2, 1, 5, 2, 3, 1}, torch::kInt32).reshape({2, 4});
    merge_outputs.sampler_output.beam_index = torch::tensor({1, 0}, torch::kInt32);
    merge_outputs.sampler_output.success    = torch::tensor({true, true}, torch::kBool);
    // Distinct per-row values so parent reordering is observable.
    merge_outputs.model_output.hidden_states =
        torch::tensor({10.0f, 10.0f, 20.0f, 20.0f}).reshape({2, 2}).to(torch::kCUDA);
    merge_outputs.model_output.logits =
        torch::tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}).reshape({2, 4}).to(torch::kCUDA);

    ASSERT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    ASSERT_FALSE(stream->hasError()) << "code=" << static_cast<int>(stream->statusInfo().code())
                                     << " msg=" << stream->statusInfo().ToString();

    // (1) New tokens land in the seqLength column (index 2), not the last column,
    // and each beam keeps its own parent history.
    EXPECT_EQ(stream->completeTokenIdsVec(0), (std::vector<int>{5, 1, 2}));
    EXPECT_EQ(stream->completeTokenIdsVec(1), (std::vector<int>{5, 2, 3}));

    // (2) Hidden states and logits follow beam_index: output row 0 <- parent row 1,
    // output row 1 <- parent row 0.
    auto outputs_status = stream->nextOutput();
    ASSERT_TRUE(outputs_status.ok());
    auto outputs = std::move(outputs_status.value());
    ASSERT_EQ(outputs.generate_outputs.size(), 2u);
    ASSERT_TRUE(outputs.generate_outputs[0].hidden_states.has_value());
    ASSERT_TRUE(outputs.generate_outputs[1].hidden_states.has_value());
    ASSERT_TRUE(outputs.generate_outputs[0].logits.has_value());
    ASSERT_TRUE(outputs.generate_outputs[1].logits.has_value());
    EXPECT_EQ(toVec<float>(*outputs.generate_outputs[0].hidden_states), (std::vector<float>{20.0f, 20.0f}));
    EXPECT_EQ(toVec<float>(*outputs.generate_outputs[1].hidden_states), (std::vector<float>{10.0f, 10.0f}));
    EXPECT_EQ(toVec<float>(*outputs.generate_outputs[0].logits), (std::vector<float>{4.0f, 5.0f, 6.0f, 7.0f}));
    EXPECT_EQ(toVec<float>(*outputs.generate_outputs[1].logits), (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f}));

    // (3) Softmax probabilities are gathered from the parent's raw logits row:
    // beam 0 <- raw row 1 at token 2, beam 1 <- raw row 0 at token 3.
    auto probs = stream->getSoftmaxProbs();
    ASSERT_TRUE(probs.defined());
    const auto row1_softmax  = torch::softmax(torch::tensor({4.0f, 5.0f, 6.0f, 7.0f}), -1);
    const auto row0_softmax  = torch::softmax(torch::tensor({0.0f, 1.0f, 2.0f, 3.0f}), -1);
    bool       beam0_matched = false, beam1_matched = false;
    for (int pos = 0; pos < probs.size(1); ++pos) {
        if (std::abs(probs[0][pos].item<float>() - row1_softmax[2].item<float>()) < 1e-5) {
            beam0_matched = true;
        }
        if (std::abs(probs[1][pos].item<float>() - row0_softmax[3].item<float>()) < 1e-5) {
            beam1_matched = true;
        }
    }
    EXPECT_TRUE(beam0_matched);
    EXPECT_TRUE(beam1_matched);
}

TEST_F(NormalBatchStreamProcessorTest, testLoss) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2048;
    model_config.num_layers  = 2;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig                  runtime_config;
    std::shared_ptr<GenerateInput> query1   = make_shared<GenerateInput>();
    query1->input_ids                       = hostIntBuffer({1});
    query1->generate_config                 = make_shared<GenerateConfig>();
    query1->generate_config->calculate_loss = 1;
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(cache_config.topologyPtr());
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
    addr3.initGroups(cache_config.topologyPtr());
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
    addr4.initGroups(cache_config.topologyPtr());
    addr4.setBatchBlocks(0, 0, {11, 12});
    stream4->setKVCache(addr4);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream3);
    streams.emplace_back(stream4);

    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    StreamGroups stream_groups(streams);
    TensorHolder holder;
    auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
    EXPECT_TRUE(merge_input_status.ok());
    EXPECT_TRUE(merge_input_status.value().need_all_logits);
    // Concrete gatherer output (not just the downstream remap): stream1 (ts=1) contributes
    // no loss rows, stream3 (ts=2, offset 1) -> row 1, stream4 (ts=3, offset 3) -> rows 3,4.
    const auto& row_indexes = merge_input_status.value().all_logits_row_indexes;
    ASSERT_TRUE(row_indexes.defined());
    EXPECT_EQ(toVec<int32_t>(row_indexes), (std::vector<int32_t>{1, 3, 4}));
    // Labels are the next execute-token for every selected row: stream3's row reads token[1]=1,
    // stream4's rows read token[1]=1 and token[2]=0. No stream asks for prompt top-k here.
    const auto& row_labels = merge_input_status.value().all_logits_row_labels;
    ASSERT_TRUE(row_labels.defined());
    EXPECT_EQ(row_labels.size(0), row_indexes.size(0));
    EXPECT_EQ(toVec<int32_t>(row_labels), (std::vector<int32_t>{1, 1, 0}));
    EXPECT_EQ(merge_input_status.value().all_logits_top_k, 0);

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

TEST_F(NormalBatchStreamProcessorTest, testAllLogitsTopKIgnoresEmptyPromptWindows) {
    // A prompt-logits request whose clamped window is empty consumes no top-k rows, so
    // its prompt_logits_top_k must not set the batch-wide merged top-k width: the forward
    // sizes the [lm prefix ++ scoring rows, k] tables from this value and unconsumed
    // placeholder rows would otherwise scale with the largest requested k.
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2048;
    model_config.num_layers  = 2;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    int  next_block  = 1;
    auto make_stream = [&](std::vector<int32_t> ids,
                           bool                 decode,
                           int                  loss,
                           bool                 prompt_logits,
                           int                  prompt_start,
                           int                  prompt_end,
                           int                  top_k) {
        auto query                                   = make_shared<GenerateInput>();
        query->input_ids                             = hostIntBuffer(std::move(ids));
        query->generate_config                       = make_shared<GenerateConfig>();
        query->generate_config->calculate_loss       = loss;
        query->generate_config->return_prompt_logits = prompt_logits;
        query->generate_config->prompt_logits_start  = prompt_start;
        query->generate_config->prompt_logits_end    = prompt_end;
        query->generate_config->prompt_logits_top_k  = top_k;
        GenerateStreamPtr stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        BatchKVCacheResource addr;
        addr.resetBatchSize(1);
        addr.initGroups(cache_config.topologyPtr());
        addr.setBatchBlocks(0, 0, {next_block, next_block + 1, next_block + 2});
        next_block += 3;
        stream->setKVCache(addr);
        if (decode) {
            stream->setIsContextStream(false);
        }
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };

    // Mixed batch: a decode neighbor (lm row only), the single-token context request
    // with an empty [0, 0) window and a large accepted k, a loss-only neighbor
    // (scoring rows without top-k demand), and one real prompt window with a small k.
    std::list<GenerateStreamPtr> streams = {
        make_stream({7}, /*decode=*/true, 0, false, -1, -1, 0),
        make_stream({9}, /*decode=*/false, 0, true, 0, 0, 300000),
        make_stream({1, 2, 3}, /*decode=*/false, 1, false, -1, -1, 0),
        make_stream({5, 6, 7, 8}, /*decode=*/false, 0, true, 1, 3, 5),
    };
    StreamGroups stream_groups(streams);
    TensorHolder holder;
    auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(merge_input_status.ok());
    const auto& model_input = merge_input_status.value();
    EXPECT_TRUE(model_input.need_all_logits);

    // Scoring rows in allStreams() order (decode first): decode -> none; the empty
    // window contributes none; the loss stream selects [0, 2) at token offset 2;
    // the prompt window selects [1, 3) at offset 5.
    const auto& row_indexes = model_input.all_logits_row_indexes;
    ASSERT_TRUE(row_indexes.defined());
    EXPECT_EQ(toVec<int32_t>(row_indexes), (std::vector<int32_t>{2, 3, 6, 7}));
    const auto& row_labels = model_input.all_logits_row_labels;
    ASSERT_TRUE(row_labels.defined());
    EXPECT_EQ(toVec<int32_t>(row_labels), (std::vector<int32_t>{2, 3, 7, 8}));
    // The empty-window k=300000 is excluded; only the real window's k survives.
    EXPECT_EQ(model_input.all_logits_top_k, 5);

    // The exact F1 shape: only the empty-window request asks for prompt logits, so the
    // batch keeps a defined-but-empty row set and a merged top-k width of 0.
    std::list<GenerateStreamPtr> empty_only = {
        make_stream({7}, /*decode=*/true, 0, false, -1, -1, 0),
        make_stream({9}, /*decode=*/false, 0, true, 0, 0, 300000),
    };
    StreamGroups empty_groups(empty_only);
    TensorHolder empty_holder;
    auto         empty_status = processor.gatherModelInput(empty_groups, empty_holder);
    ASSERT_TRUE(empty_status.ok());
    EXPECT_TRUE(empty_status.value().need_all_logits);
    ASSERT_TRUE(empty_status.value().all_logits_row_indexes.defined());
    EXPECT_EQ(empty_status.value().all_logits_row_indexes.numel(), 0);
    EXPECT_EQ(empty_status.value().all_logits_top_k, 0);
}

TEST_F(NormalBatchStreamProcessorTest, testAllLogitsRowSelectedLayout) {
    // Row-selected all_logits layout (all_logits_lm_rows >= 0): the first all_logits_lm_rows rows
    // are the lm_output_indexes rows, then each stream's promptScoringRowRange slice in
    // allStreams() order (decode streams first). The dispatcher remap must hand every consumer
    // the same rows the legacy token-aligned layout would have served, so this test dispatches an
    // identical mixed batch under both layouts and requires identical losses and prompt logits.
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 16;
    model_config.num_layers  = 2;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    const int64_t vocab = model_config.vocab_size;

    struct StreamSpec {
        std::vector<int32_t> ids;
        bool                 decode;
        int                  loss;
        bool                 prompt_logits;
        int                  prompt_start;
        int                  prompt_end;
    };
    // Cover the flag matrix plus boundary cases a wrong offset/remap would corrupt:
    //   - a decode stream whose prompt window clamps to its single executed token
    //   - decode/context streams with no scoring consumers (must contribute zero rows)
    //   - loss-only, prompt-only, and loss+prompt (contiguous union) ranges
    //   - an empty prompt window that selects no rows but keeps the flag set
    const std::vector<StreamSpec> specs = {
        {{5, 6}, true, 0, true, -1, -1},             // decode prompt rows: [0, 1)
        {{7, 8}, true, 0, false, -1, -1},            // decode, no consumers: no rows
        {{1, 2, 3, 4}, false, 2, false, -1, -1},     // loss only: [0, ts-1) = 3 rows
        {{1, 2, 3, 4, 5, 6}, false, 0, true, 2, 5},  // prompt window: [2, 5) = 3 rows
        {{2, 3, 4, 5, 6}, false, 2, true, 1, 4},     // loss + window: union [0, 4) = 4 rows
        {{9, 9, 9}, false, 0, false, -1, -1},        // context, no consumers: no rows
        {{3, 4, 5, 6}, false, 0, true, 4, 4},        // boundary: empty window, no rows
    };
    const int64_t lm_rows = static_cast<int64_t>(specs.size());

    int  next_block  = 1;
    auto make_stream = [&](const StreamSpec& spec) {
        auto query                                   = make_shared<GenerateInput>();
        query->input_ids                             = hostIntBuffer(spec.ids);
        query->generate_config                       = make_shared<GenerateConfig>();
        query->generate_config->calculate_loss       = spec.loss;
        query->generate_config->return_prompt_logits = spec.prompt_logits;
        query->generate_config->prompt_logits_start  = spec.prompt_start;
        query->generate_config->prompt_logits_end    = spec.prompt_end;
        query->generate_config->prompt_logits_top_k  = 4;
        // updateOutput only enqueues outputs for streaming or finished streams; prompt logits
        // ride on the emitted output, so these test streams must stream.
        query->generate_config->is_streaming = true;
        GenerateStreamPtr stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        BatchKVCacheResource addr;
        addr.resetBatchSize(1);
        addr.initGroups(cache_config.topologyPtr());
        addr.setBatchBlocks(0, 0, {next_block, next_block + 1, next_block + 2});
        next_block += 3;
        stream->setKVCache(addr);
        stream->setIsContextStream(!spec.decode);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };

    // Token-aligned reference: every executed token owns one row whose values are unique enough
    // that any slice offset error is observable.
    int total_tokens = 0;
    for (const auto& spec : specs) {
        total_tokens += spec.decode ? 1 : static_cast<int>(spec.ids.size());
    }
    auto aligned = torch::arange(total_tokens * vocab, torch::kFloat32).reshape({total_tokens, vocab});
    aligned += torch::arange(total_tokens, torch::kFloat32).reshape({total_tokens, 1}) * 100.0f;
    aligned = aligned.to(torch::kCUDA);

    // Build the row-selected tensor exactly like the gatherer + model would: lm_rows leading
    // rows (never consumed by the scoring paths; poisoned so an off-by-lm_rows bug is visible),
    // then each stream's [a, b) slice of its own execute-token range in allStreams() order.
    std::vector<torch::Tensor> pieces       = {torch::full({lm_rows, vocab}, -1e30f, aligned.options())};
    int                        token_offset = 0;
    for (const auto& spec : specs) {
        const int token_size = spec.decode ? 1 : static_cast<int>(spec.ids.size());
        const auto [a, b] =
            promptScoringRowRange(token_size, spec.loss != 0, spec.prompt_logits, spec.prompt_start, spec.prompt_end);
        if (b > a) {
            pieces.push_back(aligned.narrow(0, token_offset + a, b - a));
        }
        token_offset += token_size;
    }
    auto selected = torch::cat(pieces, 0);

    auto run_dispatch =
        [&](const std::list<GenerateStreamPtr>& streams, const torch::Tensor& all_logits, int64_t all_logits_lm_rows) {
            StreamGroups stream_groups(streams);
            MergedOutput merge_outputs;
            merge_outputs.model_output.hidden_states = torch::zeros({lm_rows, 4}, torch::kFloat32).to(torch::kCUDA);
            merge_outputs.model_output.logits        = torch::zeros({lm_rows, vocab}, torch::kFloat32).to(torch::kCUDA);
            merge_outputs.model_output.all_logits    = all_logits;
            merge_outputs.model_output.all_logits_lm_rows = all_logits_lm_rows;
            merge_outputs.sampler_output.token_ids        = torch::ones({lm_rows, 4}, torch::kInt32);
            merge_outputs.sampler_output.cum_log_probs    = torch::zeros({lm_rows}, torch::kFloat32).to(torch::kCUDA);
            EXPECT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
        };

    std::list<GenerateStreamPtr> streams_selected;
    std::list<GenerateStreamPtr> streams_aligned;
    for (const auto& spec : specs) {
        streams_selected.emplace_back(make_stream(spec));
        streams_aligned.emplace_back(make_stream(spec));
    }
    run_dispatch(streams_selected, selected, lm_rows);
    run_dispatch(streams_aligned, aligned, -1);

    auto it_sel = streams_selected.begin();
    auto it_ref = streams_aligned.begin();
    for (size_t i = 0; i < specs.size(); ++i, ++it_sel, ++it_ref) {
        SCOPED_TRACE("stream " + std::to_string(i));
        const auto& sel = *it_sel;
        const auto& ref = *it_ref;

        auto loss_sel = sel->getLoss();
        auto loss_ref = ref->getLoss();
        EXPECT_EQ(loss_sel.defined(), loss_ref.defined());
        if (loss_sel.defined() && loss_ref.defined()) {
            EXPECT_TRUE(torch::equal(loss_sel.cpu(), loss_ref.cpu()));
        }

        const int token_size = specs[i].decode ? 1 : static_cast<int>(specs[i].ids.size());
        const auto [wa, wb]  = promptScoringRowRange(
            token_size, false, specs[i].prompt_logits, specs[i].prompt_start, specs[i].prompt_end);

        auto out_sel = sel->nextOutput(5000);
        auto out_ref = ref->nextOutput(5000);
        ASSERT_TRUE(out_sel.ok());
        ASSERT_TRUE(out_ref.ok());
        auto& go_sel = out_sel.value().generate_outputs;
        auto& go_ref = out_ref.value().generate_outputs;
        ASSERT_EQ(go_sel.size(), 1u);
        ASSERT_EQ(go_ref.size(), 1u);
        const bool expect_prompt_logits = specs[i].prompt_logits && wb > wa;
        EXPECT_EQ(go_sel[0].prompt_logits.has_value(), expect_prompt_logits);
        ASSERT_EQ(go_sel[0].prompt_logits.has_value(), go_ref[0].prompt_logits.has_value());
        if (expect_prompt_logits) {
            const auto& pl_sel = *go_sel[0].prompt_logits;
            const auto& pl_ref = *go_ref[0].prompt_logits;
            EXPECT_EQ(pl_sel.start_pos, wa);
            EXPECT_EQ(pl_sel.end_pos, wb);
            EXPECT_EQ(pl_sel.start_pos, pl_ref.start_pos);
            EXPECT_EQ(pl_sel.end_pos, pl_ref.end_pos);
            EXPECT_TRUE(torch::equal(pl_sel.topk_logprobs, pl_ref.topk_logprobs));
            EXPECT_TRUE(torch::equal(pl_sel.topk_token_ids, pl_ref.topk_token_ids));
            EXPECT_EQ(pl_sel.target_logprobs.defined(), pl_ref.target_logprobs.defined());
            if (pl_sel.target_logprobs.defined() && pl_ref.target_logprobs.defined()) {
                EXPECT_TRUE(torch::equal(pl_sel.target_logprobs, pl_ref.target_logprobs));
            }
        }
    }
}

TEST_F(NormalBatchStreamProcessorTest, testShardedScoringOutputsLayout) {
    // The tp_size>1 sharded reduction emits per-row lse / target logit / merged top-k instead of
    // all_logits. Feeding the merged tensors derived from the same selected rows must drive the
    // dispatcher to the same outputs the gathered layout produces: top-k ids/values bitwise,
    // logprobs bitwise (same lse value is subtracted), loss equal within reduction-order ULPs.
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 16;
    model_config.num_layers  = 2;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    const int64_t vocab = model_config.vocab_size;
    const int64_t top_k = 4;

    struct StreamSpec {
        std::vector<int32_t> ids;
        bool                 decode;
        int                  loss;
        bool                 prompt_logits;
        int                  prompt_start;
        int                  prompt_end;
    };
    const std::vector<StreamSpec> specs = {
        {{5, 6}, true, 0, true, -1, -1},             // decode prompt rows: [0, 1)
        {{7, 8}, true, 0, false, -1, -1},            // decode, no consumers
        {{1, 2, 3, 4}, false, 2, false, -1, -1},     // loss only: [0, 3)
        {{1, 2, 3, 4, 5, 6}, false, 0, true, 2, 5},  // prompt window: [2, 5)
        {{2, 3, 4, 5, 6}, false, 2, true, 1, 4},     // loss + window: union [0, 4)
        {{9, 9, 9}, false, 0, false, -1, -1},        // context, no consumers
        {{3, 4, 5, 6}, false, 0, true, 4, 4},        // empty window, no rows
    };
    const int64_t lm_rows = static_cast<int64_t>(specs.size());

    int  next_block  = 1;
    auto make_stream = [&](const StreamSpec& spec) {
        auto query                                    = make_shared<GenerateInput>();
        query->input_ids                              = hostIntBuffer(spec.ids);
        query->generate_config                        = make_shared<GenerateConfig>();
        query->generate_config->calculate_loss        = spec.loss;
        query->generate_config->return_prompt_logits  = spec.prompt_logits;
        query->generate_config->prompt_logits_start   = spec.prompt_start;
        query->generate_config->prompt_logits_end     = spec.prompt_end;
        query->generate_config->prompt_logits_top_k   = top_k;
        query->generate_config->return_target_logprob = true;
        query->generate_config->is_streaming          = true;
        GenerateStreamPtr stream =
            make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        BatchKVCacheResource addr;
        addr.resetBatchSize(1);
        addr.initGroups(cache_config.topologyPtr());
        addr.setBatchBlocks(0, 0, {next_block, next_block + 1, next_block + 2});
        next_block += 3;
        stream->setKVCache(addr);
        stream->setIsContextStream(!spec.decode);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    };

    int total_tokens = 0;
    for (const auto& spec : specs) {
        total_tokens += spec.decode ? 1 : static_cast<int>(spec.ids.size());
    }
    auto aligned = torch::arange(total_tokens * vocab, torch::kFloat32).reshape({total_tokens, vocab});
    aligned += torch::arange(total_tokens, torch::kFloat32).reshape({total_tokens, 1}) * 100.0f;
    aligned = aligned.to(torch::kCUDA);

    // Dense row-selected [lm ++ per-stream [a,b)] layout for the gathered reference arm; the
    // label list covers only the scoring rows (the gatherer's all_logits_row_labels contract).
    std::vector<torch::Tensor> pieces       = {torch::full({lm_rows, vocab}, -1e30f, aligned.options())};
    std::vector<int32_t>       label_vec;
    int                        token_offset = 0;
    for (const auto& spec : specs) {
        const int token_size = spec.decode ? 1 : static_cast<int>(spec.ids.size());
        const auto [a, b] =
            promptScoringRowRange(token_size, spec.loss != 0, spec.prompt_logits, spec.prompt_start, spec.prompt_end);
        for (int r = a; r < b; ++r) {
            label_vec.push_back(r + 1 < token_size ? spec.ids[r + 1] : -1);
        }
        if (b > a) {
            pieces.push_back(aligned.narrow(0, token_offset + a, b - a));
        }
        token_offset += token_size;
    }
    auto selected = torch::cat(pieces, 0);

    // Merged sharded outputs the model would emit for these rows: compact — the scoring rows
    // only, no lm prefix. Exact full-vocab lse/top-k (the merge is value-exact for top-k; lse
    // is subtracted identically downstream) and the raw target logit per row (-inf where the
    // row has no label).
    auto selected_scoring = selected.narrow(0, lm_rows, selected.size(0) - lm_rows);
    auto lse              = selected_scoring.logsumexp(-1);
    auto topk             = selected_scoring.topk(top_k, -1);
    auto lbl              = torch::tensor(label_vec, torch::kInt32).to(torch::kCUDA).to(torch::kLong);
    auto tcol             = lbl.clamp(0, vocab - 1).unsqueeze(1);
    auto tvals            = selected_scoring.gather(1, tcol).squeeze(1);
    auto tmask            = lbl < 0;
    auto target           = tvals.masked_fill(tmask, -std::numeric_limits<float>::infinity());

    auto run_dispatch = [&](const std::list<GenerateStreamPtr>& streams, bool sharded) {
        StreamGroups stream_groups(streams);
        MergedOutput merge_outputs;
        merge_outputs.model_output.hidden_states   = torch::zeros({lm_rows, 4}, torch::kFloat32).to(torch::kCUDA);
        merge_outputs.model_output.logits          = torch::zeros({lm_rows, vocab}, torch::kFloat32).to(torch::kCUDA);
        merge_outputs.sampler_output.token_ids     = torch::ones({lm_rows, 4}, torch::kInt32);
        merge_outputs.sampler_output.cum_log_probs = torch::zeros({lm_rows}, torch::kFloat32).to(torch::kCUDA);
        merge_outputs.model_output.all_logits_lm_rows = lm_rows;
        if (sharded) {
            merge_outputs.model_output.all_logits_lse           = lse;
            merge_outputs.model_output.all_logits_target_logits = target;
            merge_outputs.model_output.all_logits_topk_values   = std::get<0>(topk);
            merge_outputs.model_output.all_logits_topk_ids      = std::get<1>(topk).to(torch::kInt32);
        } else {
            merge_outputs.model_output.all_logits = selected;
        }
        EXPECT_TRUE(processor.dispatch(stream_groups, merge_outputs).ok());
    };

    std::list<GenerateStreamPtr> streams_sharded;
    std::list<GenerateStreamPtr> streams_ref;
    for (const auto& spec : specs) {
        streams_sharded.emplace_back(make_stream(spec));
        streams_ref.emplace_back(make_stream(spec));
    }
    run_dispatch(streams_sharded, true);
    run_dispatch(streams_ref, false);

    auto it_sel = streams_sharded.begin();
    auto it_ref = streams_ref.begin();
    for (size_t i = 0; i < specs.size(); ++i, ++it_sel, ++it_ref) {
        SCOPED_TRACE("stream " + std::to_string(i));
        const auto& sel = *it_sel;
        const auto& ref = *it_ref;

        auto loss_sel = sel->getLoss();
        auto loss_ref = ref->getLoss();
        EXPECT_EQ(loss_sel.defined(), loss_ref.defined());
        if (loss_sel.defined() && loss_ref.defined()) {
            // CE kernel vs lse-minus-target: same value up to reduction order — check the
            // relative ULP scale, not an absolute epsilon, since logits here are O(1e3).
            auto diff =
                ((loss_sel.cpu() - loss_ref.cpu()).abs() / loss_ref.cpu().abs().clamp_min(1)).max().item<float>();
            EXPECT_LT(diff, 1e-5f);
        }

        const int token_size = specs[i].decode ? 1 : static_cast<int>(specs[i].ids.size());
        const auto [wa, wb]  = promptScoringRowRange(
            token_size, false, specs[i].prompt_logits, specs[i].prompt_start, specs[i].prompt_end);

        auto out_sel = sel->nextOutput(5000);
        auto out_ref = ref->nextOutput(5000);
        ASSERT_TRUE(out_sel.ok());
        ASSERT_TRUE(out_ref.ok());
        auto& go_sel = out_sel.value().generate_outputs;
        auto& go_ref = out_ref.value().generate_outputs;
        ASSERT_EQ(go_sel.size(), 1u);
        ASSERT_EQ(go_ref.size(), 1u);
        const bool expect_prompt_logits = specs[i].prompt_logits && wb > wa;
        EXPECT_EQ(go_sel[0].prompt_logits.has_value(), expect_prompt_logits);
        ASSERT_EQ(go_sel[0].prompt_logits.has_value(), go_ref[0].prompt_logits.has_value());
        if (expect_prompt_logits) {
            const auto& pl_sel = *go_sel[0].prompt_logits;
            const auto& pl_ref = *go_ref[0].prompt_logits;
            EXPECT_EQ(pl_sel.start_pos, pl_ref.start_pos);
            EXPECT_EQ(pl_sel.end_pos, pl_ref.end_pos);
            // Same lse is subtracted on both paths, so these must match bitwise.
            EXPECT_TRUE(torch::equal(pl_sel.topk_logprobs, pl_ref.topk_logprobs));
            EXPECT_TRUE(torch::equal(pl_sel.topk_token_ids, pl_ref.topk_token_ids));
            EXPECT_EQ(pl_sel.target_logprobs.defined(), pl_ref.target_logprobs.defined());
            if (pl_sel.target_logprobs.defined() && pl_ref.target_logprobs.defined()) {
                // Same raw target logit minus the same lse: bitwise equal.
                EXPECT_TRUE(torch::equal(pl_sel.target_logprobs, pl_ref.target_logprobs));
            }
        }
    }
}

namespace {

// Shared fixture for the dispatcher-local scoring-OOM guard tests: one calculate_loss stream,
// one prompt-logits stream and one normal stream dispatched in a single batch.
struct ScoringOomFixture {
    ResourceContext              resource_context;
    ModelConfig                  model_config;
    PDSepConfig                  pd_sep_config;
    ProfilingDebugLoggingConfig  profiling_debug_logging_config;
    CacheConfig                  cache_config;
    RuntimeConfig                runtime_config;
    GenerateStreamPtr            loss_stream;
    GenerateStreamPtr            prompt_stream;
    GenerateStreamPtr            normal_stream;
    MergedOutput                 merge_outputs;
    std::list<GenerateStreamPtr> streams;

    GenerateStreamPtr makeStream(const std::vector<int32_t>& ids, int loss, bool prompt_logits, int& next_block) {
        auto query                                   = make_shared<GenerateInput>();
        query->input_ids                             = hostIntBuffer(ids);
        query->generate_config                       = make_shared<GenerateConfig>();
        query->generate_config->calculate_loss       = loss;
        query->generate_config->return_prompt_logits = prompt_logits;
        query->generate_config->prompt_logits_top_k  = 2;
        query->generate_config->is_streaming         = true;
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        BatchKVCacheResource addr;
        addr.resetBatchSize(1);
        addr.initGroups(cache_config.topologyPtr());
        addr.setBatchBlocks(0, 0, {next_block, next_block + 1, next_block + 2});
        next_block += 3;
        stream->setKVCache(addr);
        stream->setIsContextStream(true);
        stream->generate_status_->status = StreamState::RUNNING;
        return stream;
    }

    explicit ScoringOomFixture(size_t vocab = 16) {
        model_config.max_seq_len = 2048;
        model_config.vocab_size  = vocab;
        model_config.num_layers  = 2;
        initFullCacheConfig(cache_config, model_config.num_layers);

        int next_block = 1;
        loss_stream    = makeStream({1, 2, 3}, /*loss=*/1, /*prompt_logits=*/false, next_block);
        prompt_stream  = makeStream({4, 5, 6, 7}, /*loss=*/0, /*prompt_logits=*/true, next_block);
        normal_stream  = makeStream({8, 9}, /*loss=*/0, /*prompt_logits=*/false, next_block);
        streams        = {loss_stream, prompt_stream, normal_stream};

        // Legacy token-aligned layout: 3 + 4 + 2 = 9 executed-token rows.
        merge_outputs.model_output.logits      = torch::zeros({3, (int64_t)vocab}, torch::kFloat32).to(torch::kCUDA);
        merge_outputs.model_output.all_logits  = torch::randn({9, (int64_t)vocab}, torch::kFloat32).to(torch::kCUDA);
        merge_outputs.sampler_output.token_ids = torch::ones({3, 4}, torch::kInt32);
        merge_outputs.sampler_output.cum_log_probs = torch::zeros({3}, torch::kFloat32).to(torch::kCUDA);
    }
};

struct ScoringHookResetGuard {
    ~ScoringHookResetGuard() {
        std::atomic_store(&NormalOutputDispatcher::test_scoring_failure_hook,
                          std::shared_ptr<std::function<void(const GenerateStreamPtr&)>>(nullptr));
    }
};

void installScoringHook(std::function<void(const GenerateStreamPtr&)> hook) {
    std::atomic_store(&NormalOutputDispatcher::test_scoring_failure_hook,
                      std::make_shared<std::function<void(const GenerateStreamPtr&)>>(std::move(hook)));
}

}  // namespace

TEST_F(NormalBatchStreamProcessorTest, testDispatchScoringOomFailsOnlyThatStream) {
    ScoringOomFixture          fixture;
    NormalBatchStreamProcessor processor(fixture.model_config,
                                         fixture.pd_sep_config,
                                         fixture.profiling_debug_logging_config,
                                         fixture.cache_config,
                                         /*warm_up=*/false);

    // Inject a dispatch-stage OOM inside each scoring stream's own output computation.
    std::vector<const GenerateStream*> hooked;
    installScoringHook([&hooked](const GenerateStreamPtr& s) {
        hooked.push_back(s.get());
        throw c10::OutOfMemoryError("CUDA out of memory");
    });
    ScoringHookResetGuard hook_guard;

    StreamGroups stream_groups(fixture.streams);
    EXPECT_TRUE(processor.dispatch(stream_groups, fixture.merge_outputs).ok());

    // Both scoring streams fail request-locally with the non-retryable execution error.
    for (const auto& stream : {fixture.loss_stream, fixture.prompt_stream}) {
        EXPECT_TRUE(stream->hasError());
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
        EXPECT_NE(stream->statusInfo().ToString().find("out-of-memory"), std::string::npos);
    }
    // The normal stream in the same batch is dispatched and produces output.
    EXPECT_FALSE(fixture.normal_stream->hasError());
    auto normal_out = fixture.normal_stream->nextOutput(5000);
    ASSERT_TRUE(normal_out.ok());
    // The hook only fires for streams that consume scoring rows (covers loss and prompt
    // logits), never for the normal stream.
    EXPECT_EQ(hooked, (std::vector<const GenerateStream*>{fixture.loss_stream.get(), fixture.prompt_stream.get()}));
}

TEST_F(NormalBatchStreamProcessorTest, testDispatchScoringNonOomRethrows) {
    ScoringOomFixture          fixture;
    NormalBatchStreamProcessor processor(fixture.model_config,
                                         fixture.pd_sep_config,
                                         fixture.profiling_debug_logging_config,
                                         fixture.cache_config,
                                         /*warm_up=*/false);

    installScoringHook([](const GenerateStreamPtr&) { throw std::runtime_error("some non-OOM dispatch failure"); });
    ScoringHookResetGuard hook_guard;

    StreamGroups stream_groups(fixture.streams);
    EXPECT_THROW(processor.dispatch(stream_groups, fixture.merge_outputs), std::runtime_error);
}

TEST_F(NormalBatchStreamProcessorTest, testDispatchScoringOomRethrowsOnWarmUp) {
    ScoringOomFixture fixture;
    // A warm-up processor must never convert a scoring OOM into a request failure: the
    // exception propagates so the warm-up/sizing step fails.
    NormalBatchStreamProcessor processor(fixture.model_config,
                                         fixture.pd_sep_config,
                                         fixture.profiling_debug_logging_config,
                                         fixture.cache_config,
                                         /*warm_up=*/true);

    installScoringHook([](const GenerateStreamPtr&) { throw c10::OutOfMemoryError("CUDA out of memory"); });
    ScoringHookResetGuard hook_guard;

    StreamGroups stream_groups(fixture.streams);
    EXPECT_THROW(processor.dispatch(stream_groups, fixture.merge_outputs), c10::OutOfMemoryError);
}

TEST_F(NormalBatchStreamProcessorTest, testDispatchScoringWrappedOomMessageFailsStream) {
    ScoringOomFixture          fixture;
    NormalBatchStreamProcessor processor(fixture.model_config,
                                         fixture.pd_sep_config,
                                         fixture.profiling_debug_logging_config,
                                         fixture.cache_config,
                                         /*warm_up=*/false);

    // Allocator retry paths surface OOM as a wrapped message, not c10::OutOfMemoryError;
    // the substring catch must still fail only the scoring streams.
    installScoringHook([](const GenerateStreamPtr&) {
        throw std::runtime_error("allocator retry failed during scoring reduction: CUDA out of memory");
    });
    ScoringHookResetGuard hook_guard;

    StreamGroups stream_groups(fixture.streams);
    EXPECT_TRUE(processor.dispatch(stream_groups, fixture.merge_outputs).ok());
    for (const auto& stream : {fixture.loss_stream, fixture.prompt_stream}) {
        EXPECT_TRUE(stream->hasError());
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::EXECUTION_EXCEPTION);
    }
    EXPECT_FALSE(fixture.normal_stream->hasError());
}

TEST_F(NormalBatchStreamProcessorTest, testDispatchShardedScoringOomFailsOnlyThatStream) {
    ScoringOomFixture          fixture;
    NormalBatchStreamProcessor processor(fixture.model_config,
                                         fixture.pd_sep_config,
                                         fixture.profiling_debug_logging_config,
                                         fixture.cache_config,
                                         /*warm_up=*/false);

    // tp>1 sharded layout: the model emits merged per-row partials (lse / target /
    // top-k) and all_logits_lm_rows >= 0 instead of all_logits. Build synthetic
    // tensors in exactly that merged layout for this batch: compact scoring rows only,
    // no lm prefix — [loss_stream rows [0,2); prompt_stream window [0,4)] = 6 rows. The
    // hook fails ONLY the loss stream; the neighboring prompt stream must still succeed
    // and slice its own rows at the right offset (cumulative scoring rows).
    const int64_t vocab    = fixture.model_config.vocab_size;
    const int64_t top_k    = fixture.prompt_stream->generateConfig()->prompt_logits_top_k;
    auto          selected = torch::arange(6 * vocab, torch::kFloat32).reshape({6, vocab}).to(torch::kCUDA);
    selected += torch::arange(6, torch::kFloat32).reshape({6, 1}).to(torch::kCUDA) * 100.0f;

    std::vector<int32_t> label_vec = {2, 3, 5, 6, 7, -1};
    auto                 lbl       = torch::tensor(label_vec, torch::kInt32).to(torch::kCUDA).to(torch::kLong);
    auto                 tcol      = lbl.clamp(0, vocab - 1).unsqueeze(1);
    auto                 tvals     = selected.gather(1, tcol).squeeze(1);
    auto                 tmask     = lbl < 0;
    auto                 target    = tvals.masked_fill(tmask, -std::numeric_limits<float>::infinity());
    auto                 topk      = selected.topk(top_k, -1);

    fixture.merge_outputs.model_output.all_logits               = torch::Tensor();
    fixture.merge_outputs.model_output.all_logits_lm_rows       = 3;
    fixture.merge_outputs.model_output.all_logits_lse           = selected.logsumexp(-1);
    fixture.merge_outputs.model_output.all_logits_target_logits = target;
    fixture.merge_outputs.model_output.all_logits_topk_values   = std::get<0>(topk);
    fixture.merge_outputs.model_output.all_logits_topk_ids      = std::get<1>(topk).to(torch::kInt32);

    std::vector<const GenerateStream*> hooked;
    installScoringHook([&hooked](const GenerateStreamPtr& s) {
        hooked.push_back(s.get());
        if (s->generateConfig()->calculate_loss) {
            throw c10::OutOfMemoryError("CUDA out of memory");
        }
    });
    ScoringHookResetGuard hook_guard;

    StreamGroups stream_groups(fixture.streams);
    EXPECT_TRUE(processor.dispatch(stream_groups, fixture.merge_outputs).ok());

    // The hook fires once per scoring consumer stream; only the loss stream threw.
    EXPECT_EQ(hooked, (std::vector<const GenerateStream*>{fixture.loss_stream.get(), fixture.prompt_stream.get()}));
    EXPECT_TRUE(fixture.loss_stream->hasError());
    EXPECT_EQ(fixture.loss_stream->statusInfo().code(), ErrorCode::EXECUTION_EXCEPTION);

    // The prompt stream after the failed one still reads its own slice — proving the
    // successor stream's row offset does not shift when a neighbor fails mid-batch.
    EXPECT_FALSE(fixture.prompt_stream->hasError());
    auto prompt_out = fixture.prompt_stream->nextOutput(5000);
    ASSERT_TRUE(prompt_out.ok());
    const auto& go = prompt_out.value().generate_outputs;
    ASSERT_GE(go.size(), 1u);
    ASSERT_TRUE(go[0].prompt_logits.has_value());
    const auto& pl = *go[0].prompt_logits;
    EXPECT_EQ(pl.start_pos, 0);
    EXPECT_EQ(pl.end_pos, 4);
    // Rows 2..5 of the merged fields are this stream's window [0,4): topk ids/values
    // bitwise, logprobs equal (same lse is subtracted downstream).
    auto expect_topk_v = std::get<0>(topk).narrow(0, 2, 4);
    auto expect_topk_i = std::get<1>(topk).narrow(0, 2, 4);
    auto expect_lse    = selected.logsumexp(-1).narrow(0, 2, 4);
    EXPECT_TRUE(torch::equal(pl.topk_token_ids.cpu(), expect_topk_i.cpu().to(torch::kInt32)));
    EXPECT_TRUE(torch::equal(pl.topk_logprobs.cpu(), (expect_topk_v - expect_lse.unsqueeze(1)).cpu()));

    EXPECT_FALSE(fixture.normal_stream->hasError());
    auto normal_out = fixture.normal_stream->nextOutput(5000);
    ASSERT_TRUE(normal_out.ok());
}

TEST_F(NormalBatchStreamProcessorTest, testMultimodalGatherBatch) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                   = 2048;
    model_config.vocab_size                    = 2048;
    model_config.num_layers                    = 2;
    model_config.attn_config.kv_cache_dtype    = KvCacheDataType::FP8;
    model_config.mm_model_config.is_multimodal = true;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
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

TEST_F(NormalBatchStreamProcessorTest, testPartiallyReusedMultimodalFeatureIsNormalizedWithinItsStream) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                   = 2048;
    model_config.vocab_size                    = 2048;
    model_config.num_layers                    = 2;
    model_config.attn_config.kv_cache_dtype    = KvCacheDataType::FP8;
    model_config.mm_model_config.is_multimodal = true;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    auto query1             = make_shared<GenerateInput>();
    query1->input_ids       = hostIntBuffer({11, 12, 13, 14});
    query1->generate_config = make_shared<GenerateConfig>();
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    stream1->setIsContextStream(true);

    auto fully_reused_feature   = torch::arange(2, torch::kFloat32).reshape({1, 2});
    auto fully_reused_deepstack = torch::arange(4, torch::kFloat32).reshape({2, 1, 2});
    auto feature                = torch::arange(12, torch::kFloat32).reshape({6, 2});
    auto deepstack              = torch::arange(24, torch::kFloat32).reshape({2, 6, 2});
    auto query2                 = make_shared<GenerateInput>();
    query2->input_ids           = hostIntBuffer({-1, -1, -1, -1, -1, -1, -1, 21});
    query2->generate_config     = make_shared<GenerateConfig>();
    query2->mm_locs             = torch::tensor({0, 1}, torch::kInt32);
    query2->text_tokens_mask    = torch::tensor({0, 0, 0, 0, 0, 0, 0, 1}, torch::kInt32);
    query2->multimodal_features = {fully_reused_feature, feature};
    query2->mm_extra_input      = std::vector<torch::Tensor>{fully_reused_deepstack.flatten(), deepstack.flatten()};
    GenerateStreamPtr stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    stream2->setIsContextStream(true);
    stream2->setReuseLength(3);

    std::list<GenerateStreamPtr> streams{stream1, stream2};
    for (const auto& stream : streams) {
        stream->generate_status_->status = StreamState::RUNNING;
    }

    StreamGroups stream_groups(streams);
    TensorHolder holder;
    auto         merge_input_status = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(merge_input_status.ok());

    auto& model_input = merge_input_status.value();
    EXPECT_EQ(toVec<int>(model_input.combo_tokens), (vector<int>{11, 12, 13, 14, -1, -1, -1, -1, 21}));
    EXPECT_EQ(toVec<int>(model_input.input_lengths), (vector<int>{4, 5}));
    EXPECT_EQ(toVec<int>(model_input.mm_features_locs), (vector<int>{4}));
    EXPECT_EQ(toVec<int>(model_input.text_tokens_mask), (vector<int>{1, 1, 1, 1, 0, 0, 0, 0, 1}));

    ASSERT_TRUE(model_input.multimodal_features.has_value());
    ASSERT_EQ(model_input.multimodal_features.value().size(), 1);
    EXPECT_TRUE(torch::equal(model_input.multimodal_features.value()[0].cpu(), feature.slice(0, 2, 6)));

    ASSERT_TRUE(model_input.mm_extra_input.has_value());
    ASSERT_EQ(model_input.mm_extra_input.value().size(), 1);
    EXPECT_TRUE(torch::equal(model_input.mm_extra_input.value()[0].cpu().reshape({2, 4, 2}), deepstack.slice(1, 2, 6)));

    // The stream retains the complete ViT output for later reuse decisions.
    ASSERT_EQ(stream2->multimodalFeatures().size(), 2);
    EXPECT_TRUE(torch::equal(stream2->multimodalFeatures()[0], fully_reused_feature));
    EXPECT_TRUE(torch::equal(stream2->multimodalFeatures()[1], feature));
}

TEST_F(NormalBatchStreamProcessorTest, testMisalignedMultimodalExtraInputIsRejected) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                   = 2048;
    model_config.vocab_size                    = 2048;
    model_config.num_layers                    = 2;
    model_config.attn_config.kv_cache_dtype    = KvCacheDataType::FP8;
    model_config.mm_model_config.is_multimodal = true;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    initFullCacheConfig(cache_config, model_config.num_layers);
    RuntimeConfig              runtime_config;
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    auto feature               = torch::arange(12, torch::kFloat32).reshape({6, 2});
    auto query                 = make_shared<GenerateInput>();
    query->input_ids           = hostIntBuffer({-1, -1, -1, -1, -1, -1, 21});
    query->generate_config     = make_shared<GenerateConfig>();
    query->mm_locs             = torch::tensor({0}, torch::kInt32);
    query->text_tokens_mask    = torch::tensor({0, 0, 0, 0, 0, 0, 1}, torch::kInt32);
    query->multimodal_features = {feature};
    query->mm_extra_input      = std::vector<torch::Tensor>{torch::arange(3, torch::kFloat32)};
    GenerateStreamPtr stream =
        make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(true);
    stream->generate_status_->status = StreamState::RUNNING;

    std::list<GenerateStreamPtr> streams{stream};
    StreamGroups                 stream_groups(streams);
    TensorHolder                 holder;
    bool                         threw = false;
    try {
        (void)processor.gatherModelInput(stream_groups, holder);
    } catch (const std::runtime_error& e) {
        threw = true;
        EXPECT_NE(std::string(e.what()).find("not divisible"), std::string::npos);
    }
    EXPECT_TRUE(threw);
}

}  // namespace rtp_llm
