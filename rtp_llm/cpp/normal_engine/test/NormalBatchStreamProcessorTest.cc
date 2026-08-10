#include <memory>
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

static torch::Tensor normalStateTokenView(const GenerateStream::NormalAsyncDeviceState& state) {
    return state.last_sample_token_gpu.defined() ?
               state.last_sample_token_gpu :
               state.batched_last_sample_tokens_gpu.narrow(0, state.device_batch_index, 1);
}

static torch::Tensor normalStateNextSeqLenView(const GenerateStream::NormalAsyncDeviceState& state) {
    return state.next_seq_len_gpu.defined() ? state.next_seq_len_gpu :
                                              state.batched_next_seq_lens_gpu.narrow(0, state.device_batch_index, 1);
}

static GenerateStreamPtr makeAsyncDecodeStream(const ModelConfig&   model_config,
                                               const RuntimeConfig& runtime_config,
                                               ResourceContext&     resource_context,
                                               int                  block_id) {
    auto query             = std::make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({1, 2, 3});
    query->generate_config = std::make_shared<GenerateConfig>();
    auto stream =
        std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(false);
    stream->generate_status_->status = StreamState::RUNNING;

    BatchKVCacheResource kv_cache;
    kv_cache.resetBatchSize(1);
    kv_cache.initGroups(1, 1, {0});
    kv_cache.setBatchBlocks(0, 0, {block_id});
    stream->setKVCache(kv_cache);
    return stream;
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

TEST_F(NormalBatchStreamProcessorTest, testDeviceStateGatherPreservesHostSequenceLengths) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 128;
    model_config.vocab_size  = 128;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    auto query             = std::make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({1, 2, 3});
    query->generate_config = std::make_shared<GenerateConfig>();
    GenerateStreamPtr stream =
        std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(false);
    stream->generate_status_->status = StreamState::RUNNING;

    BatchKVCacheResource kv_cache;
    kv_cache.resetBatchSize(1);
    kv_cache.initGroups(1, 1, {0});
    kv_cache.setBatchBlocks(0, 0, {1});
    stream->setKVCache(kv_cache);

    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
        .last_sample_token_gpu = torch::full({1}, 42, cuda_i32),
        .next_seq_len_gpu      = torch::full({1}, 4, cuda_i32),
        .last_real_seq_len     = 3,
        .next_real_seq_len     = 4,
    });

    std::list<GenerateStreamPtr> streams{stream};
    StreamGroups                 stream_groups(streams);
    TensorHolder                 holder;
    auto                         status = processor.gatherModelInput(stream_groups, holder);

    ASSERT_TRUE(status.ok()) << status.status();
    const auto& model_input = status.value();
    EXPECT_TRUE(model_input.combo_tokens.is_cuda());
    EXPECT_TRUE(model_input.sequence_lengths.is_cuda());
    ASSERT_TRUE(model_input.sequence_lengths_host_for_log.defined());
    EXPECT_FALSE(model_input.sequence_lengths_host_for_log.is_cuda());
    EXPECT_EQ(std::vector<int32_t>({42}), toVec<int32_t>(model_input.combo_tokens));
    EXPECT_EQ(std::vector<int32_t>({3}), toVec<int32_t>(model_input.sequence_lengths));
    EXPECT_EQ(std::vector<int32_t>({3}), toVec<int32_t>(model_input.sequence_lengths_host_for_log));
}

TEST_F(NormalBatchStreamProcessorTest, testDeviceStateGatherBatchesSequenceLengthArithmetic) {
    constexpr int batch_size = 128;

    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 4096;
    model_config.vocab_size  = 4096;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    const auto                   cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    std::list<GenerateStreamPtr> streams;
    std::vector<int32_t>         expected_tokens;
    std::vector<int32_t>         expected_device_seq_lens;
    std::vector<int32_t>         expected_host_seq_lens;
    expected_tokens.reserve(batch_size);
    expected_device_seq_lens.reserve(batch_size);
    expected_host_seq_lens.reserve(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        auto stream = makeAsyncDecodeStream(model_config, runtime_config, resource_context, i + 1);
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .last_sample_token_gpu = torch::full({1}, 100 + i, cuda_i32),
            .next_seq_len_gpu      = torch::full({1}, 1000 + i, cuda_i32),
            .last_real_seq_len     = 1999 + i,
            .next_real_seq_len     = 2000 + i,
        });
        streams.push_back(stream);
        expected_tokens.push_back(100 + i);
        // Deliberately keep the device and host mirrors different. The batched
        // path must preserve the original contract: model input comes from
        // device state, while sequence_lengths_host_for_log uses the mirror.
        expected_device_seq_lens.push_back(999 + i);
        expected_host_seq_lens.push_back(1999 + i);
    }

    StreamGroups stream_groups(streams);
    TensorHolder holder;
    auto         status = processor.gatherModelInput(stream_groups, holder);

    ASSERT_TRUE(status.ok()) << status.status();
    const auto& model_input = status.value();
    EXPECT_EQ(expected_tokens, toVec<int32_t>(model_input.combo_tokens));
    EXPECT_EQ(expected_device_seq_lens, toVec<int32_t>(model_input.sequence_lengths));
    EXPECT_EQ(expected_host_seq_lens, toVec<int32_t>(model_input.sequence_lengths_host_for_log));
}

TEST_F(NormalBatchStreamProcessorTest, testDeviceStateGatherReusesSharedBatchBacking) {
    constexpr int batch_size = 128;

    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 4096;
    model_config.vocab_size  = 4096;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    const auto cuda_i32      = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto       tokens        = torch::arange(100, 100 + batch_size, cuda_i32);
    auto       next_seq_lens = torch::arange(1000, 1000 + batch_size, cuda_i32);

    std::list<GenerateStreamPtr> streams;
    for (int i = 0; i < batch_size; ++i) {
        auto stream = makeAsyncDecodeStream(model_config, runtime_config, resource_context, i + 1);
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .batched_last_sample_tokens_gpu = tokens,
            .batched_next_seq_lens_gpu      = next_seq_lens,
            .device_batch_index             = i,
            .last_real_seq_len              = 1999 + i,
            .next_real_seq_len              = 2000 + i,
        });
        streams.push_back(stream);
    }

    StreamGroups stream_groups(streams);
    TensorHolder holder;
    auto         status = processor.gatherModelInput(stream_groups, holder);

    ASSERT_TRUE(status.ok()) << status.status();
    const auto& model_input = status.value();
    // No per-stream views/cat are needed for tokens: the original backing
    // Tensor is installed directly in GptModelInputs.
    EXPECT_EQ(tokens.unsafeGetTensorImpl(), model_input.combo_tokens.unsafeGetTensorImpl());
    EXPECT_EQ(toVec<int32_t>(tokens), toVec<int32_t>(model_input.combo_tokens));
    auto expected_sequence_lengths = next_seq_lens - 1;
    EXPECT_EQ(toVec<int32_t>(expected_sequence_lengths), toVec<int32_t>(model_input.sequence_lengths));
}

TEST_F(NormalBatchStreamProcessorTest, testDeviceStateGatherFallsBackForReorderedBatchBacking) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 4096;
    model_config.vocab_size  = 4096;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    const auto cuda_i32      = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto       tokens        = torch::tensor({10, 20}, cuda_i32);
    auto       next_seq_lens = torch::tensor({100, 200}, cuda_i32);

    std::list<GenerateStreamPtr> streams;
    for (int i = 0; i < 2; ++i) {
        auto      stream        = makeAsyncDecodeStream(model_config, runtime_config, resource_context, i + 1);
        const int backing_index = 1 - i;
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .batched_last_sample_tokens_gpu = tokens,
            .batched_next_seq_lens_gpu      = next_seq_lens,
            .device_batch_index             = backing_index,
            .last_real_seq_len              = 299 + i,
            .next_real_seq_len              = 300 + i,
        });
        streams.push_back(stream);
    }

    StreamGroups stream_groups(streams);
    TensorHolder holder;
    auto         status = processor.gatherModelInput(stream_groups, holder);

    ASSERT_TRUE(status.ok()) << status.status();
    const auto& model_input = status.value();
    EXPECT_NE(tokens.unsafeGetTensorImpl(), model_input.combo_tokens.unsafeGetTensorImpl());
    EXPECT_EQ(std::vector<int32_t>({20, 10}), toVec<int32_t>(model_input.combo_tokens));
    EXPECT_EQ(std::vector<int32_t>({199, 99}), toVec<int32_t>(model_input.sequence_lengths));
    EXPECT_EQ(std::vector<int32_t>({299, 300}), toVec<int32_t>(model_input.sequence_lengths_host_for_log));
}

TEST_F(NormalBatchStreamProcessorTest, testPublishNormalDeviceStateBatchesExistingDeviceState) {
    constexpr int batch_size = 128;

    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 4096;
    model_config.vocab_size  = 4096;
    RuntimeConfig runtime_config;

    const auto                   cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    std::list<GenerateStreamPtr> streams;
    for (int i = 0; i < batch_size; ++i) {
        auto stream = makeAsyncDecodeStream(model_config, runtime_config, resource_context, i + 1);
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .last_sample_token_gpu = torch::full({1}, i, cuda_i32),
            .next_seq_len_gpu      = torch::full({1}, 1000 + i, cuda_i32),
            .last_real_seq_len     = 1999 + i,
            .next_real_seq_len     = 2000 + i,
        });
        streams.push_back(stream);
    }

    EngineInitParams params;
    params.model_config_ = model_config;
    params.py_model      = py::none();
    NormalExecutor executor(params, nullptr, true);

    // SamplerOutput is not guaranteed to be int32. The batched publish path
    // must preserve the original per-stream contract and store int32 tokens.
    const auto cuda_i64  = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto       token_ids = torch::arange(500, 500 + batch_size, cuda_i64).reshape({batch_size, 1});
    executor.publishNormalDeviceState(StreamGroups(streams), SamplerOutput{token_ids});

    at::TensorImpl* shared_token_impl = nullptr;
    at::TensorImpl* shared_seq_impl   = nullptr;
    int             index             = 0;
    for (const auto& stream : streams) {
        const auto& state = stream->getNormalAsyncDeviceState();
        EXPECT_FALSE(state.last_sample_token_gpu.defined());
        EXPECT_FALSE(state.next_seq_len_gpu.defined());
        EXPECT_EQ(torch::kInt32, state.batched_last_sample_tokens_gpu.scalar_type());
        EXPECT_EQ(index, state.device_batch_index);
        if (index == 0) {
            shared_token_impl = state.batched_last_sample_tokens_gpu.unsafeGetTensorImpl();
            shared_seq_impl   = state.batched_next_seq_lens_gpu.unsafeGetTensorImpl();
        }
        EXPECT_EQ(shared_token_impl, state.batched_last_sample_tokens_gpu.unsafeGetTensorImpl());
        EXPECT_EQ(shared_seq_impl, state.batched_next_seq_lens_gpu.unsafeGetTensorImpl());
        EXPECT_EQ(std::vector<int32_t>({500 + index}), toVec<int32_t>(normalStateTokenView(state)));
        EXPECT_EQ(std::vector<int32_t>({1001 + index}), toVec<int32_t>(normalStateNextSeqLenView(state)));
        EXPECT_EQ(2000 + index, state.last_real_seq_len);
        EXPECT_EQ(2001 + index, state.next_real_seq_len);
        ++index;
    }

    // A second publish exercises the steady-state reuse path: the current
    // sequence-length batch comes directly from the shared backing tensor.
    auto next_token_ids = torch::arange(900, 900 + batch_size, cuda_i64);
    executor.publishNormalDeviceState(StreamGroups(streams), SamplerOutput{next_token_ids});
    index = 0;
    for (const auto& stream : streams) {
        const auto& state = stream->getNormalAsyncDeviceState();
        EXPECT_EQ(std::vector<int32_t>({900 + index}), toVec<int32_t>(normalStateTokenView(state)));
        EXPECT_EQ(std::vector<int32_t>({1002 + index}), toVec<int32_t>(normalStateNextSeqLenView(state)));
        EXPECT_EQ(2001 + index, state.last_real_seq_len);
        EXPECT_EQ(2002 + index, state.next_real_seq_len);
        ++index;
    }
}

TEST_F(NormalBatchStreamProcessorTest, testPublishNormalDeviceStateBatchesHostFallback) {
    constexpr int batch_size = 4;

    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 4096;
    model_config.vocab_size  = 4096;
    RuntimeConfig runtime_config;

    std::list<GenerateStreamPtr> streams;
    for (int i = 0; i < batch_size; ++i) {
        auto stream = makeAsyncDecodeStream(model_config, runtime_config, resource_context, i + 1);
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .last_real_seq_len = 1999 + i,
            .next_real_seq_len = 2000 + i,
        });
        streams.push_back(stream);
    }

    EngineInitParams params;
    params.model_config_ = model_config;
    params.py_model      = py::none();
    NormalExecutor executor(params, nullptr, true);

    const auto cuda_i32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto       token_ids = torch::arange(700, 700 + batch_size, cuda_i32);
    executor.publishNormalDeviceState(StreamGroups(streams), SamplerOutput{token_ids});

    at::TensorImpl* shared_token_impl = nullptr;
    at::TensorImpl* shared_seq_impl   = nullptr;
    int             index             = 0;
    for (const auto& stream : streams) {
        const auto& state = stream->getNormalAsyncDeviceState();
        if (index == 0) {
            shared_token_impl = state.batched_last_sample_tokens_gpu.unsafeGetTensorImpl();
            shared_seq_impl   = state.batched_next_seq_lens_gpu.unsafeGetTensorImpl();
        }
        EXPECT_EQ(shared_token_impl, state.batched_last_sample_tokens_gpu.unsafeGetTensorImpl());
        EXPECT_EQ(shared_seq_impl, state.batched_next_seq_lens_gpu.unsafeGetTensorImpl());
        EXPECT_EQ(index, state.device_batch_index);
        EXPECT_EQ(std::vector<int32_t>({700 + index}), toVec<int32_t>(normalStateTokenView(state)));
        EXPECT_EQ(std::vector<int32_t>({2001 + index}), toVec<int32_t>(normalStateNextSeqLenView(state)));
        EXPECT_EQ(2000 + index, state.last_real_seq_len);
        EXPECT_EQ(2001 + index, state.next_real_seq_len);
        ++index;
    }
}

TEST_F(NormalBatchStreamProcessorTest, testPublishNormalDeviceStateBatchesMixedSequenceLengthSources) {
    constexpr int batch_size = 4;

    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 4096;
    model_config.vocab_size  = 4096;
    RuntimeConfig runtime_config;

    const auto                   cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    std::list<GenerateStreamPtr> streams;
    for (int i = 0; i < batch_size; ++i) {
        auto stream = makeAsyncDecodeStream(model_config, runtime_config, resource_context, i + 1);
        GenerateStream::NormalAsyncDeviceState state{
            .last_real_seq_len = 1999 + i,
            .next_real_seq_len = 2000 + i,
        };
        if (i % 2 == 0) {
            state.next_seq_len_gpu = torch::full({1}, 1000 + i, cuda_i32);
        }
        stream->setNormalAsyncDeviceState(std::move(state));
        streams.push_back(stream);
    }

    EngineInitParams params;
    params.model_config_ = model_config;
    params.py_model      = py::none();
    NormalExecutor executor(params, nullptr, true);

    auto token_ids = torch::arange(800, 800 + batch_size, cuda_i32);
    executor.publishNormalDeviceState(StreamGroups(streams), SamplerOutput{token_ids});

    const std::vector<int32_t> expected_next_seq_lens{1001, 2002, 1003, 2004};
    int                        index = 0;
    for (const auto& stream : streams) {
        const auto& state = stream->getNormalAsyncDeviceState();
        EXPECT_EQ(std::vector<int32_t>({800 + index}), toVec<int32_t>(normalStateTokenView(state)));
        EXPECT_EQ(std::vector<int32_t>({expected_next_seq_lens[index]}),
                  toVec<int32_t>(normalStateNextSeqLenView(state)));
        EXPECT_EQ(2000 + index, state.last_real_seq_len);
        EXPECT_EQ(2001 + index, state.next_real_seq_len);
        ++index;
    }
}

TEST_F(NormalBatchStreamProcessorTest, testPublishNormalDeviceStateSelectsLastTokenColumnOnceWithPadding) {
    constexpr int batch_size  = 4;
    constexpr int token_width = 3;
    constexpr int padded_rows = 2;

    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 4096;
    model_config.vocab_size  = 4096;
    RuntimeConfig runtime_config;

    const auto                   cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    std::list<GenerateStreamPtr> streams;
    for (int i = 0; i < batch_size; ++i) {
        auto stream = makeAsyncDecodeStream(model_config, runtime_config, resource_context, i + 1);
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .last_sample_token_gpu = torch::full({1}, i, cuda_i32),
            .next_seq_len_gpu      = torch::full({1}, 1000 + i, cuda_i32),
            .last_real_seq_len     = 1999 + i,
            .next_real_seq_len     = 2000 + i,
        });
        streams.push_back(stream);
    }

    EngineInitParams params;
    params.model_config_ = model_config;
    params.py_model      = py::none();
    NormalExecutor executor(params, nullptr, true);

    auto token_ids = torch::arange(0, (batch_size + padded_rows) * token_width, cuda_i32)
                         .reshape({batch_size + padded_rows, token_width});
    executor.publishNormalDeviceState(StreamGroups(streams), SamplerOutput{token_ids});

    int index = 0;
    for (const auto& stream : streams) {
        const auto& state = stream->getNormalAsyncDeviceState();
        EXPECT_EQ(std::vector<int32_t>({index * token_width + token_width - 1}),
                  toVec<int32_t>(normalStateTokenView(state)));
        EXPECT_TRUE(state.batched_last_sample_tokens_gpu.is_contiguous());
        EXPECT_EQ(batch_size, state.batched_last_sample_tokens_gpu.size(0));
        EXPECT_EQ(std::vector<int32_t>({1001 + index}), toVec<int32_t>(normalStateNextSeqLenView(state)));
        ++index;
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

}  // namespace rtp_llm
