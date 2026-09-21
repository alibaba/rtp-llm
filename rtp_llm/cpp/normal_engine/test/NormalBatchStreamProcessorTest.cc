#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include "torch/all.h"
#include "gtest/gtest.h"

#define private public
#define protected public
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
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

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value): name_(name) {
        const char* old_value = std::getenv(name_);
        if (old_value != nullptr) {
            old_value_ = old_value;
            had_value_ = true;
        }
        setenv(name_, value, 1);
    }

    ~ScopedEnvVar() {
        if (had_value_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    std::string old_value_;
    bool        had_value_ = false;
};

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

TEST_F(NormalBatchStreamProcessorTest, testDeviceStateFastPathWaitsForBlockingLogitsProcessorState) {
    ScopedEnvVar    device_input_env("RTP_LLM_DEVICE_INPUT", "1");
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
    PDSepConfig                  pd_sep_config;
    ProfilingDebugLoggingConfig  profiling_debug_logging_config;
    CacheConfig                  cache_config;
    NormalBatchStreamProcessor   processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    EXPECT_EQ(processor.prepareDeviceStateInputs(streams).size(), 1);
    stream->logits_processor_list_.push_back(std::make_shared<TestStatefulLogitsProcessor>(false));
    stream->incPendingAsyncBookkeeping();
    EXPECT_TRUE(processor.prepareDeviceStateInputs(streams).empty());
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
}

TEST_F(NormalBatchStreamProcessorTest, testDeviceStateFastPathAllowsAsyncLogitsProcessorState) {
    ScopedEnvVar    device_input_env("RTP_LLM_DEVICE_INPUT", "1");
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
    PDSepConfig                  pd_sep_config;
    ProfilingDebugLoggingConfig  profiling_debug_logging_config;
    CacheConfig                  cache_config;
    NormalBatchStreamProcessor   processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    stream->incPendingAsyncBookkeeping();
    EXPECT_EQ(processor.prepareDeviceStateInputs(streams).size(), 1);
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
}

TEST_F(NormalBatchStreamProcessorTest, testDeviceStateFastPathRejectsPendingKvCacheMapping) {
    ScopedEnvVar    device_input_env("RTP_LLM_DEVICE_INPUT", "1");
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 128;
    model_config.vocab_size  = 128900;
    RuntimeConfig runtime_config;

    auto query             = make_shared<GenerateInput>();
    query->input_ids       = hostIntBuffer({1, 2, 3});
    query->generate_config = make_shared<GenerateConfig>();
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(false);
    stream->generate_status_->status = StreamState::RUNNING;
    const auto cuda_i32              = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
        .last_sample_token_gpu = torch::full({1}, 42, cuda_i32),
        .next_seq_len_gpu      = torch::full({1}, 4, cuda_i32),
        .last_real_seq_len     = 3,
        .next_real_seq_len     = 4,
    });

    std::list<GenerateStreamPtr> streams{stream};
    PDSepConfig                  pd_sep_config;
    ProfilingDebugLoggingConfig  profiling_debug_logging_config;
    CacheConfig                  cache_config;
    NormalBatchStreamProcessor   processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    ASSERT_EQ(processor.prepareDeviceStateInputs(streams).size(), 1);

    auto pending_state                    = stream->getNormalAsyncDeviceState();
    pending_state.kv_cache_update_pending = true;
    stream->setNormalAsyncDeviceState(std::move(pending_state));
    EXPECT_TRUE(processor.prepareDeviceStateInputs(streams).empty());
}

TEST_F(NormalBatchStreamProcessorTest, testDeviceStatePlanRejectsIneligibleBatch) {
    enum class InvalidState {
        MISSING_HOST_LENGTH,
        BAD_DTYPE,
        BAD_NUMEL
    };
    struct Case {
        const char*                 name;
        int                         num_beams;
        int                         num_return_sequences;
        std::optional<InvalidState> invalid_state;
    };
    const std::vector<Case> cases = {
        {"beam", 2, 1, std::nullopt},
        {"multiple_return_sequences", 1, 2, std::nullopt},
        {"missing_host_length", 1, 1, InvalidState::MISSING_HOST_LENGTH},
        {"malformed_dtype", 1, 1, InvalidState::BAD_DTYPE},
        {"malformed_numel", 1, 1, InvalidState::BAD_NUMEL},
    };

    ScopedEnvVar    device_input_env("RTP_LLM_DEVICE_INPUT", "1");
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 128;
    model_config.vocab_size  = 128900;
    RuntimeConfig               runtime_config;
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    auto make_valid_stream = [&] {
        auto query             = make_shared<GenerateInput>();
        query->input_ids       = hostIntBuffer({1, 2, 3});
        query->generate_config = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(false);
        stream->generate_status_->status = StreamState::RUNNING;
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .last_sample_token_gpu = torch::full({1}, 42, cuda_i32),
            .next_seq_len_gpu      = torch::full({1}, 4, cuda_i32),
            .last_real_seq_len     = 3,
            .next_real_seq_len     = 4,
        });
        return std::pair{query, stream};
    };

    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        auto [query, stream]                         = make_valid_stream();
        query->generate_config->num_beams            = test_case.num_beams;
        query->generate_config->num_return_sequences = test_case.num_return_sequences;
        if (test_case.invalid_state.has_value()) {
            auto state = stream->getNormalAsyncDeviceState();
            switch (*test_case.invalid_state) {
                case InvalidState::MISSING_HOST_LENGTH:
                    state.next_real_seq_len = -1;
                    break;
                case InvalidState::BAD_DTYPE:
                    state.last_sample_token_gpu =
                        torch::full({1}, 42.0f, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                    break;
                case InvalidState::BAD_NUMEL:
                    state.next_seq_len_gpu = torch::full({2}, 4, cuda_i32);
                    break;
            }
            stream->setNormalAsyncDeviceState(std::move(state));
        }
        EXPECT_TRUE(processor.prepareDeviceStateInputs({stream}).empty());
    }

    // The selected map is all-or-none. A valid row cannot overlap when a
    // peer has an unsafe mapping state, because StreamGroups must use one
    // coherent source for the entire batch.
    auto [valid_query, valid_stream]     = make_valid_stream();
    auto [invalid_query, invalid_stream] = make_valid_stream();
    (void)valid_query;
    (void)invalid_query;
    auto invalid_state                    = invalid_stream->getNormalAsyncDeviceState();
    invalid_state.kv_cache_update_pending = true;
    invalid_stream->setNormalAsyncDeviceState(std::move(invalid_state));
    EXPECT_TRUE(processor.prepareDeviceStateInputs({valid_stream, invalid_stream}).empty());

    {
        ScopedEnvVar disabled_device_input("RTP_LLM_DEVICE_INPUT", "0");
        EXPECT_TRUE(processor.prepareDeviceStateInputs({valid_stream}).empty());
    }
}

TEST_F(NormalBatchStreamProcessorTest, testDeviceStateFastPathFallsBackWhenPositionSnapshotIsUnavailable) {
    struct PositionIdCase {
        const char*      name;
        PositionIdsStyle position_style;
        bool             has_positional_encoding;
        int              position_id_len_factor;
    };
    // Keep one genuinely ineligible position-id case.  The MRoPE case below
    // has an immutable anchor and a complete normal-device snapshot, so it is
    // deliberately tested as an async-capable decode rather than folded into
    // this conservative fallback check.
    const PositionIdCase cases[] = {{"absolute_position_encoding", PositionIdsStyle::DEFAULT, true, 1}};

    ScopedEnvVar device_input_env("RTP_LLM_DEVICE_INPUT", "1");
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        ResourceContext resource_context;
        ModelConfig     model_config;
        model_config.max_seq_len                           = 128;
        model_config.vocab_size                            = 128900;
        model_config.mm_model_config.mm_position_ids_style = test_case.position_style;
        model_config.has_positional_encoding               = test_case.has_positional_encoding;
        model_config.attn_config.rope_config.index_factor  = test_case.position_id_len_factor;
        RuntimeConfig runtime_config;

        auto query             = make_shared<GenerateInput>();
        query->input_ids       = hostIntBuffer({1, 2, 3});
        query->generate_config = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(false);
        stream->generate_status_->status = StreamState::RUNNING;
        const auto cuda_i32              = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .last_sample_token_gpu = torch::full({1}, 42, cuda_i32),
            .next_seq_len_gpu      = torch::full({1}, 4, cuda_i32),
            .last_real_seq_len     = 3,
            .next_real_seq_len     = 4,
        });
        stream->incPendingAsyncBookkeeping();

        std::list<GenerateStreamPtr> streams{stream};

        // This case has a valid device snapshot, but gather still needs
        // host token/length/position state. The pending worker must therefore
        // be joined before processDecodeStreams reads those accessors.
        PDSepConfig                 pd_sep_config;
        ProfilingDebugLoggingConfig profiling_debug_logging_config;
        CacheConfig                 cache_config;
        NormalBatchStreamProcessor  processor(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
        EXPECT_TRUE(processor.prepareDeviceStateInputs(streams).empty());
        StreamGroups stream_groups(streams);
        TensorHolder holder;
        auto         model_input_status = processor.gatherModelInput(stream_groups, holder);
        EXPECT_TRUE(model_input_status.ok());
        if (model_input_status.ok()) {
            // The gatherer publishes CPU-derived metadata to CUDA at its final
            // boundary. Check the values, not the output device: host token 3
            // and seq_len 3 differ from the pending device snapshot (42, 4).
            const auto& model_input = model_input_status.value();
            EXPECT_EQ(toVec<int>(model_input.combo_tokens), std::vector<int>({3}));
            EXPECT_EQ(toVec<int>(model_input.sequence_lengths), std::vector<int>({2}));
        }
        stream->decPendingAsyncBookkeepingAndMaybeRelease();
    }
}

TEST_F(NormalBatchStreamProcessorTest, testMropeDeviceSnapshotKeepsDecodeRowsCoherentAcrossStreams) {
    // Host histories are one output step behind (token 3), while the normal
    // device snapshots already describe the next decode rows. The selected
    // plan supplies one epoch to both StreamGroups and gather.
    ScopedEnvVar    device_input_env("RTP_LLM_DEVICE_INPUT", "1");
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                           = 128;
    model_config.vocab_size                            = 128900;
    model_config.mm_model_config.mm_position_ids_style = PositionIdsStyle::MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    RuntimeConfig runtime_config;
    const auto    cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    auto make_stream = [&](int token, int next_seq_len, int next_real_seq_len, const torch::Tensor& anchor) {
        auto query             = make_shared<GenerateInput>();
        query->input_ids       = hostIntBuffer({1, 2, 3});  // intentionally stale worker-owned history
        query->generate_config = make_shared<GenerateConfig>();
        auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
        stream->setIsContextStream(false);
        stream->generate_status_->status = StreamState::RUNNING;
        stream->setContextPositionIds(anchor);
        stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
            .last_sample_token_gpu = torch::full({1}, token, cuda_i32),
            .next_seq_len_gpu      = torch::full({1}, next_seq_len, cuda_i32),
            .last_real_seq_len     = next_real_seq_len - 1,
            .next_real_seq_len     = next_real_seq_len,
        });
        stream->incPendingAsyncBookkeeping();
        return stream;
    };

    // First row is ordinary pure-text MRoPE: context [0, 1, 2], next length
    // 4, therefore position [3, 3, 3].  The second has a nontrivial tail
    // anchor [12, 14, 13]; next length 7 must yield [18, 18, 18], proving the
    // formula uses the immutable anchor and the post-sample length rather
    // than the stale host seqLength()==3.
    auto first = make_stream(42, 4, 4, torch::tensor({0, 0, 0, 1, 1, 1, 2, 2, 2}, torch::kInt32).reshape({3, 3}));
    auto second =
        make_stream(43, 7, 7, torch::tensor({10, 11, 9, 11, 12, 10, 12, 14, 13}, torch::kInt32).reshape({3, 3}));
    std::list<GenerateStreamPtr> streams{first, second};
    PDSepConfig                  pd_sep_config;
    ProfilingDebugLoggingConfig  profiling_debug_logging_config;
    CacheConfig                  cache_config;
    NormalBatchStreamProcessor   processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    auto normal_states = processor.prepareDeviceStateInputs(streams);
    ASSERT_EQ(normal_states.size(), 2);

    // Freeze the selected epoch, then replace the live stream state before
    // groups/gather read it.  Both consumers must keep using the original
    // plan: token 42 / length 4, never this later token 99 / length 9.
    first->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
        .last_sample_token_gpu = torch::full({1}, 99, cuda_i32),
        .next_seq_len_gpu      = torch::full({1}, 9, cuda_i32),
        .last_real_seq_len     = 8,
        .next_real_seq_len     = 9,
    });
    // The plan also owns the validated anchor. Replacing the live stream
    // anchor after selection must not pair token 42 with another round's
    // position basis.
    first->setContextPositionIds(
        torch::tensor({100, 100, 100, 101, 101, 101, 102, 102, 102}, torch::kInt32).reshape({3, 3}));
    StreamGroups stream_groups(streams, std::move(normal_states));

    // StreamGroups is constructed before gather.  Its capacity must come
    // from the selected snapshot epoch: host history remains length 3 and
    // the later live state above must not change its capacity.
    EXPECT_EQ(stream_groups.totalDecodeBatchSize(), 2);
    EXPECT_EQ(stream_groups.modelExecuteTokenSize(), 2);
    EXPECT_EQ(stream_groups.maxSeqLen(), 7);
    TensorHolder holder;
    auto         model_input_status = processor.gatherModelInput(stream_groups, holder);
    ASSERT_TRUE(model_input_status.ok());
    const auto& model_input = model_input_status.value();

    // `next_seq_len` is the committed length after the sampled token.  The
    // decode model input uses its zero-based predecessor, so 4 -> 3 and
    // 7 -> 6.  Token and position values must belong to that same snapshot.
    EXPECT_EQ(toVec<int>(model_input.combo_tokens), (std::vector<int>{42, 43}));
    EXPECT_EQ(toVec<int>(model_input.sequence_lengths), (std::vector<int>{3, 6}));
    EXPECT_EQ(toVec<int>(model_input.combo_position_ids), (std::vector<int>{3, 3, 3, 18, 18, 18}));
    // The normal snapshot route is allowed to make only this published
    // position payload device-resident.  The shared ordinary/MTP gather path
    // retains its host layout, because some MTP consumers dereference it.
    EXPECT_TRUE(model_input.combo_position_ids.is_cuda());

    first->decPendingAsyncBookkeepingAndMaybeRelease();
    second->decPendingAsyncBookkeepingAndMaybeRelease();
}

struct ProcessOverlapProbe {
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    forward_entered    = false;
    bool                    sampler_entered    = false;
    bool                    worker_started     = false;
    bool                    release_worker     = false;
    int                     worker_iter_count  = -1;
    int                     sampler_iter_count = -1;
    std::vector<int>        combo_tokens;
    std::vector<int>        sequence_lengths;
    std::vector<int>        combo_position_ids;
};

class SnapshotRecordingModel final: public ModelBase {
public:
    SnapshotRecordingModel(std::shared_ptr<ProcessOverlapProbe> probe, size_t vocab_size):
        probe_(std::move(probe)), vocab_size_(vocab_size) {}

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        {
            std::lock_guard<std::mutex> lock(probe_->mutex);
            probe_->combo_tokens       = toVec<int>(inputs.combo_tokens);
            probe_->sequence_lengths   = toVec<int>(inputs.sequence_lengths);
            probe_->combo_position_ids = toVec<int>(inputs.combo_position_ids);
            probe_->forward_entered    = true;
        }
        probe_->cv.notify_all();
        GptModelOutputs outputs;
        const auto      row_count = inputs.lm_output_indexes.defined() ? inputs.lm_output_indexes.numel() : 1;
        outputs.logits            = torch::zeros({std::max<int64_t>(1, row_count), static_cast<int64_t>(vocab_size_)},
                                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        return outputs;
    }

private:
    std::shared_ptr<ProcessOverlapProbe> probe_;
    size_t                               vocab_size_;
};

class StopAtSamplerProcessor final: public NormalBatchStreamProcessor {
public:
    StopAtSamplerProcessor(const ModelConfig&                   model_config,
                           const PDSepConfig&                   pd_sep_config,
                           const ProfilingDebugLoggingConfig&   profiling_debug_logging_config,
                           const CacheConfig&                   cache_config,
                           std::shared_ptr<ProcessOverlapProbe> probe):
        NormalBatchStreamProcessor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false),
        probe_(std::move(probe)) {}

    absl::StatusOr<SamplerInputs> gatherSamplerInput(const StreamGroups& stream_groups,
                                                     const GptModelInputs&,
                                                     const GptModelOutputs&) const override {
        {
            std::lock_guard<std::mutex> lock(probe_->mutex);
            probe_->sampler_entered    = true;
            probe_->sampler_iter_count = stream_groups.allStreams().empty() ?
                                             -1 :
                                             static_cast<int>(stream_groups.allStreams().front()->iterCount());
        }
        probe_->cv.notify_all();
        // This is a controlled terminal point. It proves process passed the
        // real gather/model-forward path without running a sampler or sending
        // dispatch work that could obscure the previous-worker ordering.
        return absl::AbortedError(kStoppedAtSamplerMarker);
    }

private:
    static constexpr const char*         kStoppedAtSamplerMarker = "test stops after pre-sampler ordering check";
    std::shared_ptr<ProcessOverlapProbe> probe_;
};

TEST_F(NormalBatchStreamProcessorTest, testProcessOverlapsMropeSnapshotButJoinsBeforeSampler) {
    // A real previous-worker task is held on NormalExecutor's AsyncRunner.
    // Snapshot decode may reach model forward before release; it still must
    // join at the pre-sampler boundary, where host sampler metadata is read.
    ScopedEnvVar    device_input_env("RTP_LLM_DEVICE_INPUT", "1");
    ScopedEnvVar    stream_async_env("RTP_LLM_STREAM_ASYNC", "1");
    ScopedEnvVar    drop_broad_sync_env("RTP_LLM_DROP_BROAD_SYNC", "1");
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                           = 128;
    model_config.vocab_size                            = 128900;
    model_config.mm_model_config.mm_position_ids_style = PositionIdsStyle::MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    RuntimeConfig runtime_config;

    auto query                             = make_shared<GenerateInput>();
    query->input_ids                       = hostIntBuffer({1, 2, 3});
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 2;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(false);
    stream->generate_status_->status = StreamState::RUNNING;
    stream->setContextPositionIds(torch::tensor({0, 0, 0, 1, 1, 1, 2, 2, 2}, torch::kInt32).reshape({3, 3}));
    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
        .last_sample_token_gpu = torch::full({1}, 42, cuda_i32),
        .next_seq_len_gpu      = torch::full({1}, 4, cuda_i32),
        .last_real_seq_len     = 3,
        .next_real_seq_len     = 4,
    });
    stream->incPendingAsyncBookkeeping();

    EngineInitParams params;
    params.model_config_ = model_config;
    params.py_model      = py::none();
    NormalExecutor executor(params, nullptr, false);
    auto           probe = std::make_shared<ProcessOverlapProbe>();
    executor.setModel(std::make_unique<SnapshotRecordingModel>(probe, model_config.vocab_size));
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    executor.setBatchProcessor(std::make_unique<StopAtSamplerProcessor>(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, probe));

    executor.dispatch_runner_.launch([&] {
        std::unique_lock<std::mutex> lock(probe->mutex);
        probe->worker_started = true;
        probe->cv.notify_all();
        // Wait until gather/model forward has consumed the selected snapshot.
        // If the implementation takes the conservative fallback, the main
        // thread is waiting in sync; release_worker breaks that test deadlock.
        probe->cv.wait(lock, [&] { return probe->forward_entered || probe->release_worker; });
        if (probe->forward_entered) {
            probe->worker_iter_count = static_cast<int>(stream->iterCount());
        }
        probe->cv.wait(lock, [&] { return probe->release_worker; });
    });
    {
        std::unique_lock<std::mutex> lock(probe->mutex);
        EXPECT_TRUE(probe->cv.wait_for(lock, std::chrono::seconds(2), [&] { return probe->worker_started; }));
    }

    absl::Status       process_status = absl::UnknownError("process did not return");
    std::exception_ptr process_exception;
    std::thread        process_thread([&] {
        try {
            process_status = executor.process({stream});
        } catch (...) {
            process_exception = std::current_exception();
        }
    });
    bool               forward_before_release = false;
    bool               sampler_before_release = false;
    {
        std::unique_lock<std::mutex> lock(probe->mutex);
        forward_before_release =
            probe->cv.wait_for(lock, std::chrono::seconds(2), [&] { return probe->forward_entered; });
        // A single immediate read could race a missing pre-sampler join. Hold
        // a bounded observation window while the previous worker remains
        // blocked; sampler must not consume its host metadata in that window.
        sampler_before_release =
            forward_before_release
            && probe->cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return probe->sampler_entered; });
    }

    // Always release before joining, including when an expectation above
    // fails. This prevents a failing test from leaving the executor worker
    // blocked and contaminating a later GPU test.
    {
        std::lock_guard<std::mutex> lock(probe->mutex);
        probe->release_worker = true;
    }
    probe->cv.notify_all();
    process_thread.join();

    EXPECT_TRUE(forward_before_release);
    EXPECT_FALSE(sampler_before_release);
    EXPECT_EQ(process_exception, nullptr);
    EXPECT_EQ(process_status.code(), absl::StatusCode::kAborted);
    EXPECT_EQ(process_status.message(), "test stops after pre-sampler ordering check");
    {
        std::lock_guard<std::mutex> lock(probe->mutex);
        // Snapshot gather must not call step() while the previous worker may
        // still inspect the old round.  The executor performs that advance
        // only after its pre-sampler join.
        EXPECT_EQ(probe->worker_iter_count, 0);
        EXPECT_EQ(probe->combo_tokens, (std::vector<int>{42}));
        EXPECT_EQ(probe->sequence_lengths, (std::vector<int>{3}));
        EXPECT_EQ(probe->combo_position_ids, (std::vector<int>{3, 3, 3}));
        EXPECT_TRUE(probe->sampler_entered);
        EXPECT_EQ(probe->sampler_iter_count, 1);
    }
    stream->decPendingAsyncBookkeepingAndMaybeRelease();
}

TEST_F(NormalBatchStreamProcessorTest, testProcessJoinsBeforeFallbackGatherWithoutMropeAnchor) {
    // The no-anchor case cannot derive a safe snapshot position. It must not
    // enter model forward while the previous worker is blocked; after release
    // it follows ordinary host gather and reaches the controlled sampler stop.
    ScopedEnvVar    device_input_env("RTP_LLM_DEVICE_INPUT", "1");
    ScopedEnvVar    stream_async_env("RTP_LLM_STREAM_ASYNC", "1");
    ScopedEnvVar    drop_broad_sync_env("RTP_LLM_DROP_BROAD_SYNC", "1");
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len                           = 128;
    model_config.vocab_size                            = 128900;
    model_config.mm_model_config.mm_position_ids_style = PositionIdsStyle::MROPE;
    model_config.attn_config.rope_config.index_factor  = 3;
    RuntimeConfig runtime_config;

    auto query                             = make_shared<GenerateInput>();
    query->input_ids                       = hostIntBuffer({1, 2, 3});
    query->generate_config                 = make_shared<GenerateConfig>();
    query->generate_config->max_new_tokens = 2;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->setIsContextStream(false);
    stream->generate_status_->status = StreamState::RUNNING;
    // This is a valid normal device state except for the MRoPE anchor.  It
    // distinguishes the anchor fallback from a generic no-state fallback.
    const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
        .last_sample_token_gpu = torch::full({1}, 42, cuda_i32),
        .next_seq_len_gpu      = torch::full({1}, 4, cuda_i32),
        .last_real_seq_len     = 3,
        .next_real_seq_len     = 4,
    });
    stream->incPendingAsyncBookkeeping();

    EngineInitParams params;
    params.model_config_ = model_config;
    params.py_model      = py::none();
    NormalExecutor executor(params, nullptr, false);
    auto           probe = std::make_shared<ProcessOverlapProbe>();
    executor.setModel(std::make_unique<SnapshotRecordingModel>(probe, model_config.vocab_size));
    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    executor.setBatchProcessor(std::make_unique<StopAtSamplerProcessor>(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, probe));

    executor.dispatch_runner_.launch([&] {
        std::unique_lock<std::mutex> lock(probe->mutex);
        probe->worker_started = true;
        probe->cv.notify_all();
        probe->cv.wait(lock, [&] { return probe->forward_entered || probe->release_worker; });
        // This is the previous worker's commit point in the test. Once it is
        // released, ordinary host gather has a valid anchor and cannot write
        // uninitialized MRoPE positions.
        if (probe->release_worker) {
            stream->setContextPositionIds(torch::tensor({0, 0, 0, 1, 1, 1, 2, 2, 2}, torch::kInt32).reshape({3, 3}));
        }
        probe->cv.wait(lock, [&] { return probe->release_worker; });
    });
    {
        std::unique_lock<std::mutex> lock(probe->mutex);
        EXPECT_TRUE(probe->cv.wait_for(lock, std::chrono::seconds(2), [&] { return probe->worker_started; }));
    }

    absl::Status       process_status = absl::UnknownError("process did not return");
    std::exception_ptr process_exception;
    std::thread        process_thread([&] {
        try {
            process_status = executor.process({stream});
        } catch (...) {
            process_exception = std::current_exception();
        }
    });
    bool               forward_before_release = false;
    {
        std::unique_lock<std::mutex> lock(probe->mutex);
        forward_before_release =
            probe->cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return probe->forward_entered; });
    }

    {
        std::lock_guard<std::mutex> lock(probe->mutex);
        probe->release_worker = true;
    }
    probe->cv.notify_all();
    process_thread.join();

    EXPECT_FALSE(forward_before_release);
    EXPECT_EQ(process_exception, nullptr);
    EXPECT_EQ(process_status.code(), absl::StatusCode::kAborted);
    EXPECT_EQ(process_status.message(), "test stops after pre-sampler ordering check");
    {
        std::lock_guard<std::mutex> lock(probe->mutex);
        EXPECT_TRUE(probe->forward_entered);
        EXPECT_TRUE(probe->sampler_entered);
        EXPECT_EQ(probe->sampler_iter_count, 1);
    }
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

TEST_F(NormalBatchStreamProcessorTest, testDisabledOutputVocabMasksPaddedLogits) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 8;
    model_config.vocab_size  = 10;
    model_config.num_layers  = 1;
    RuntimeConfig runtime_config;

    auto query                               = make_shared<GenerateInput>();
    query->input_ids                         = hostIntBuffer({2});
    query->generate_config                   = make_shared<GenerateConfig>();
    query->generate_config->top_k            = 8;
    query->generate_config->return_all_probs = ReturnAllProbsMode::DEFAULT;
    auto stream = make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    stream->generate_status_->status = StreamState::RUNNING;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);
    StreamGroups    stream_groups({stream});
    GptModelOutputs model_output;
    model_output.logits = torch::zeros({1, 16}, torch::kFloat32).to(torch::kCUDA);
    model_output.logits.narrow(1, 10, 6).fill_(100.0f);

    auto sampler_inputs = processor.gatherSamplerInput(stream_groups, GptModelInputs(), model_output);
    ASSERT_TRUE(sampler_inputs.ok());
    EXPECT_EQ(sampler_inputs->top_k.data_ptr<int32_t>()[0], 8);
    EXPECT_EQ(sampler_inputs->all_probs.size(1), 16);
    EXPECT_TRUE(torch::isneginf(sampler_inputs->logits.narrow(1, 10, 6)).all().item<bool>());
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
    // CUDA masks with -inf and ROCm with the lowest finite float. Verify the
    // sampling contract instead: a finished row can select only compact EOS.
    auto probabilities = processed_logits.softmax(-1);
    for (int token_id = 0; token_id < 5; ++token_id) {
        EXPECT_FLOAT_EQ(processed_logits[0][token_id].item<float>(), 0.0f);
        if (token_id == 3) {
            EXPECT_FLOAT_EQ(processed_logits[1][token_id].item<float>(), 0.0f);
            EXPECT_FLOAT_EQ(probabilities[1][token_id].item<float>(), 1.0f);
        } else {
            EXPECT_FLOAT_EQ(probabilities[1][token_id].item<float>(), 0.0f);
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
