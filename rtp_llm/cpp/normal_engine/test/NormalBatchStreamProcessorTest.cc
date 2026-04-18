#include <memory>
#include "torch/all.h"
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

using namespace std;

namespace rtp_llm {

template<typename T>
std::vector<T> toVec(const torch::Tensor& t) {
    auto c = t.contiguous();
    return std::vector<T>(c.data_ptr<T>(), c.data_ptr<T>() + c.numel());
}

static torch::Tensor hostIntBuffer(std::vector<int32_t> data) {
    return torch::tensor(data, torch::kInt32);
}

class NormalBatchStreamProcessorTest: public DeviceTestBase {};

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
        stream->setRunning();
    }

    {
        StreamGroups stream_groups(streams);

        auto merge_input_status = processor.gatherModelInput(stream_groups);

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
        auto         merge_input_status = processor.gatherModelInput(stream_groups);
        EXPECT_TRUE(merge_input_status.ok());
        auto& model_input = merge_input_status.value();
        EXPECT_FALSE(model_input.attention_mask.defined());
    }
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
        stream->setRunning();
    }
    cache_config.group_types = {CacheGroupType::FULL};
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    StreamGroups stream_groups(streams);
    auto         merge_input_status = processor.gatherModelInput(stream_groups);
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
        stream->setRunning();
    }
    cache_config.group_types = {CacheGroupType::FULL};
    NormalBatchStreamProcessor processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    StreamGroups stream_groups(streams);
    auto         merge_input_status = processor.gatherModelInput(stream_groups);
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
        stream->setRunning();
    }

    {
        StreamGroups stream_groups(streams);

        auto merge_input_status = processor.gatherModelInput(stream_groups);
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

// Test NaN flag detection and stream status marking
TEST_F(NormalBatchStreamProcessorTest, testNanFlagDetection) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2;
    model_config.num_layers  = 2;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};
    RuntimeConfig               runtime_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    // Create decode stream (1 token per batch)
    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = torch::tensor({1}, torch::kInt32);
    query1->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(1, model_config.num_layers, {0, 0});
    addr1.setBatchBlocks(0, 0, {1});
    stream1->setKVCache(addr1);
    stream1->setIsContextStream(false);

    // Create another decode stream (1 token per batch)
    // Note: token ID must be < vocab_size (2), so we use {0}
    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = torch::tensor({0}, torch::kInt32);
    query2->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr2;
    addr2.resetBatchSize(1);
    addr2.initGroups(1, model_config.num_layers, {0, 0});
    addr2.setBatchBlocks(0, 0, {2});
    stream2->setKVCache(addr2);
    stream2->setIsContextStream(false);

    // Create context stream (multiple tokens per batch)
    // Note: token IDs must be < vocab_size (2), so we use {0, 1, 0}
    std::shared_ptr<GenerateInput> query3 = make_shared<GenerateInput>();
    query3->input_ids                     = torch::tensor({0, 1, 0}, torch::kInt32);
    query3->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream3 =
        make_shared<NormalGenerateStream>(query3, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.resetBatchSize(1);
    addr3.initGroups(1, model_config.num_layers, {0, 0});
    addr3.setBatchBlocks(0, 0, {3, 4});
    stream3->setKVCache(addr3);
    stream3->setIsContextStream(true);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);
    streams.emplace_back(stream3);

    for (const auto& stream : streams) {
        stream->setRunning();
    }

    StreamGroups stream_groups(streams);
    auto         merge_input_status = processor.gatherModelInput(stream_groups);
    EXPECT_TRUE(merge_input_status.ok());

    // Prepare MergedOutput with nan_flag
    // NOTE: NormalBatchStreamProcessor expects nan_flag shape == [total_batch_size],
    // i.e. one flag per batch (not per token).
    // Batch order: decode streams first (stream1, stream2), then context streams (stream3).
    // nan_flag: [1, 0, 1] means:
    //   - stream1 batch: has NaN
    //   - stream2 batch: no NaN
    //   - stream3 batch: has NaN
    std::vector<float> nan_flag_data = {1, 0, 1};  // 3 batches

    MergedOutput merge_outputs;
    // dispatch() only uses nan_flag (shape [total_batch_size]) and sampler_output.token_ids (must be < vocab_size)
    merge_outputs.model_output.hidden_states   = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).reshape({3, 2});
    merge_outputs.model_output.logits          = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).reshape({3, 2});
    merge_outputs.model_output.nan_flag        = torch::tensor(nan_flag_data, torch::kFloat32);
    merge_outputs.sampler_output.token_ids     = torch::tensor({0, 1, 0}, torch::kInt32).reshape({3, 1});
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f, 2.0f, 3.0f});

    auto status = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());

    // Verify stream1 (has NaN) is stopped
    EXPECT_TRUE(stream1->stopped()) << "Stream1 should be stopped due to NaN";
    EXPECT_EQ(stream1->statusInfo().code(), ErrorCode::NAN_DETECTED) << "Stream1 error code should be NAN_DETECTED";
    std::string error_msg1 = stream1->statusInfo().ToString();
    EXPECT_NE(error_msg1.find("NaN detected"), std::string::npos)
        << "Stream1 error message should contain 'NaN detected'";

    // Verify stream2 (no NaN) is still running
    EXPECT_FALSE(stream2->stopped()) << "Stream2 should not be stopped (no NaN)";
    EXPECT_TRUE(stream2->running()) << "Stream2 should still be running";

    // Verify stream3 (has NaN in one token) is stopped
    EXPECT_TRUE(stream3->stopped()) << "Stream3 should be stopped due to NaN";
    EXPECT_EQ(stream3->statusInfo().code(), ErrorCode::NAN_DETECTED) << "Stream3 error code should be NAN_DETECTED";
    std::string error_msg3 = stream3->statusInfo().ToString();
    EXPECT_NE(error_msg3.find("NaN detected"), std::string::npos)
        << "Stream3 error message should contain 'NaN detected'";
}

// Test NaN flag detection for decode stream only
TEST_F(NormalBatchStreamProcessorTest, testNanFlagDetection_DecodeStream) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2;
    model_config.num_layers  = 2;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};
    RuntimeConfig               runtime_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    // Create decode stream
    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = torch::tensor({1}, torch::kInt32);
    query1->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(1, model_config.num_layers, {0, 0});
    addr1.setBatchBlocks(0, 0, {1});
    stream1->setKVCache(addr1);
    stream1->setIsContextStream(false);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);

    for (const auto& stream : streams) {
        stream->setRunning();
    }

    StreamGroups stream_groups(streams);
    auto         merge_input_status = processor.gatherModelInput(stream_groups);
    EXPECT_TRUE(merge_input_status.ok());

    // nan_flag: [1] means token 0 has NaN
    std::vector<float> nan_flag_data      = {1};
    std::vector<int32_t> input_lengths_data = {1};

    MergedOutput merge_outputs;
    merge_outputs.model_output.hidden_states   = torch::tensor({1.0f, 2.0f}).reshape({1, 2});
    merge_outputs.model_output.logits          = torch::tensor({1.0f, 2.0f}).reshape({1, 2});
    merge_outputs.model_output.nan_flag        = torch::tensor(nan_flag_data, torch::kFloat32);
    merge_outputs.sampler_output.token_ids     = torch::tensor({0}, torch::kInt32).reshape({1, 1});
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f});

    auto status = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());

    // Verify stream is stopped
    EXPECT_TRUE(stream1->stopped()) << "Stream should be stopped due to NaN";
    EXPECT_EQ(stream1->statusInfo().code(), ErrorCode::NAN_DETECTED) << "Error code should be NAN_DETECTED";
}

// Test NaN flag detection for context stream only
TEST_F(NormalBatchStreamProcessorTest, testNanFlagDetection_ContextStream) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2;
    model_config.num_layers  = 2;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};
    RuntimeConfig               runtime_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    // Create context stream with 4 tokens
    // Note: token IDs must be < vocab_size (2), so we use {0, 1, 0, 1}
    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = torch::tensor({0, 1, 0, 1}, torch::kInt32);
    query1->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(1, model_config.num_layers, {0, 0});
    addr1.setBatchBlocks(0, 0, {1, 2});
    stream1->setKVCache(addr1);
    stream1->setIsContextStream(true);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);

    for (const auto& stream : streams) {
        stream->setRunning();
    }

    StreamGroups stream_groups(streams);
    auto         merge_input_status = processor.gatherModelInput(stream_groups);
    EXPECT_TRUE(merge_input_status.ok());

    // NOTE: NormalBatchStreamProcessor expects nan_flag shape == [total_batch_size],
    // i.e. one flag per batch (not per token).
    // nan_flag: [1] means this (single) batch has NaN.
    std::vector<float> nan_flag_data = {1};

    MergedOutput merge_outputs;
    merge_outputs.model_output.hidden_states   = torch::tensor({1.0f, 2.0f}).reshape({1, 2});
    merge_outputs.model_output.logits          = torch::tensor({1.0f, 2.0f}).reshape({1, 2});
    merge_outputs.model_output.nan_flag        = torch::tensor(nan_flag_data, torch::kFloat32);
    merge_outputs.sampler_output.token_ids     = torch::tensor({0}, torch::kInt32).reshape({1, 1});
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f});

    auto status = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());

    // Verify stream is stopped
    EXPECT_TRUE(stream1->stopped()) << "Stream should be stopped due to NaN";
    EXPECT_EQ(stream1->statusInfo().code(), ErrorCode::NAN_DETECTED) << "Error code should be NAN_DETECTED";
    std::string error_msg = stream1->statusInfo().ToString();
    EXPECT_NE(error_msg.find("NaN detected"), std::string::npos) << "Error message should contain 'NaN detected'";
}

// Test multiple streams with mixed NaN cases
TEST_F(NormalBatchStreamProcessorTest, testNanFlagDetection_MultipleStreams) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2;
    model_config.num_layers  = 2;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};
    RuntimeConfig               runtime_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    // Create 3 decode streams
    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = torch::tensor({1}, torch::kInt32);
    query1->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resetBatchSize(1);
    addr1.initGroups(1, model_config.num_layers, {0, 0});
    addr1.setBatchBlocks(0, 0, {1});
    stream1->setKVCache(addr1);
    stream1->setIsContextStream(false);

    // Note: token IDs must be < vocab_size (2), so we use {0} and {1}
    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = torch::tensor({0}, torch::kInt32);
    query2->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream2 =
        make_shared<NormalGenerateStream>(query2, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr2;
    addr2.resetBatchSize(1);
    addr2.initGroups(1, model_config.num_layers, {0, 0});
    addr2.setBatchBlocks(0, 0, {2});
    stream2->setKVCache(addr2);
    stream2->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query3 = make_shared<GenerateInput>();
    query3->input_ids                     = torch::tensor({1}, torch::kInt32);
    query3->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream3 =
        make_shared<NormalGenerateStream>(query3, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.resetBatchSize(1);
    addr3.initGroups(1, model_config.num_layers, {0, 0});
    addr3.setBatchBlocks(0, 0, {3});
    stream3->setKVCache(addr3);
    stream3->setIsContextStream(false);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream2);
    streams.emplace_back(stream3);

    for (const auto& stream : streams) {
        stream->setRunning();
    }

    StreamGroups stream_groups(streams);
    auto         merge_input_status = processor.gatherModelInput(stream_groups);
    EXPECT_TRUE(merge_input_status.ok());

    // nan_flag: [1, 0, 1] means:
    //   - stream1 (token 0): has NaN
    //   - stream2 (token 1): no NaN
    //   - stream3 (token 2): has NaN
    std::vector<float> nan_flag_data      = {1, 0, 1};
    std::vector<int32_t> input_lengths_data = {1, 1, 1};

    MergedOutput merge_outputs;
    merge_outputs.model_output.hidden_states   = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).reshape({3, 2});
    merge_outputs.model_output.logits          = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).reshape({3, 2});
    merge_outputs.model_output.nan_flag        = torch::tensor(nan_flag_data, torch::kFloat32);
    merge_outputs.sampler_output.token_ids     = torch::tensor({0, 1, 2}, torch::kInt32).reshape({3, 1});
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f, 2.0f, 3.0f});

    auto status = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());

    // Verify stream1 (has NaN) is stopped
    EXPECT_TRUE(stream1->stopped()) << "Stream1 should be stopped due to NaN";
    EXPECT_EQ(stream1->statusInfo().code(), ErrorCode::NAN_DETECTED);

    // Verify stream2 (no NaN) is still running
    EXPECT_FALSE(stream2->stopped()) << "Stream2 should not be stopped (no NaN)";
    EXPECT_TRUE(stream2->running()) << "Stream2 should still be running";

    // Verify stream3 (has NaN) is stopped
    EXPECT_TRUE(stream3->stopped()) << "Stream3 should be stopped due to NaN";
    EXPECT_EQ(stream3->statusInfo().code(), ErrorCode::NAN_DETECTED);
}

// Test NaN flag detection for beam_search scenario
TEST_F(NormalBatchStreamProcessorTest, testNanFlagDetection_BeamSearch) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2;
    model_config.num_layers  = 2;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};
    RuntimeConfig               runtime_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    // Create decode stream with beam_search (num_beams = 3)
    // Note: In the first step (output_len=0), currentBatchSize() = 1 even with num_beams > 1
    // In subsequent steps (output_len > 0), currentBatchSize() = num_beams = 3
    // This test verifies NaN detection works correctly with beam_search configuration
    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = torch::tensor({1}, torch::kInt32);
    query1->generate_config               = make_shared<GenerateConfig>();
    query1->generate_config->num_beams    = 3;
    query1->generate_config->do_sample    = false;  // beam_search typically uses do_sample=false
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    // In first step of beam_search, batch_size = 1 (even though num_beams = 3)
    addr1.resetBatchSize(1);
    addr1.initGroups(1, model_config.num_layers, {0, 0});
    addr1.setBatchBlocks(0, 0, {1});
    stream1->setKVCache(addr1);
    stream1->setIsContextStream(false);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);

    for (const auto& stream : streams) {
        stream->setRunning();
    }

    StreamGroups stream_groups(streams);
    auto         merge_input_status = processor.gatherModelInput(stream_groups);
    EXPECT_TRUE(merge_input_status.ok());

    // In first step of beam_search, currentBatchSize() = 1, but nextBatchSize() = num_beams = 3
    // nan_flag shape should be [1] (one flag for the single batch in first step)
    // nan_flag: [1] means this batch has NaN
    std::vector<float> nan_flag_data = {1};

    MergedOutput merge_outputs;
    merge_outputs.model_output.hidden_states = torch::tensor({1.0f, 2.0f}).reshape({1, 2});
    merge_outputs.model_output.logits        = torch::tensor({1.0f, 2.0f}).reshape({1, 2});
    merge_outputs.model_output.nan_flag      = torch::tensor(nan_flag_data, torch::kFloat32);
    // nextBatchSize() = num_beams = 3, so token_ids shape should be [3, 1]
    merge_outputs.sampler_output.token_ids     = torch::tensor({0, 1, 0}, torch::kInt32).reshape({3, 1});
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f, 2.0f, 3.0f});
    // beam_index is required for beam_search
    merge_outputs.sampler_output.beam_index = torch::tensor({0, 1, 2}, torch::kInt32);

    auto status = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());

    // Verify stream is stopped due to NaN
    EXPECT_TRUE(stream1->stopped()) << "Stream should be stopped due to NaN in beam_search";
    EXPECT_EQ(stream1->statusInfo().code(), ErrorCode::NAN_DETECTED) << "Error code should be NAN_DETECTED";
    std::string error_msg = stream1->statusInfo().ToString();
    EXPECT_NE(error_msg.find("NaN detected"), std::string::npos) << error_msg;
}

// Test NaN flag detection for num_return_sequences > 1 scenario
TEST_F(NormalBatchStreamProcessorTest, testNanFlagDetection_NumReturnSequences) {
    ResourceContext resource_context;
    ModelConfig     model_config;
    model_config.max_seq_len = 2048;
    model_config.vocab_size  = 2;
    model_config.num_layers  = 2;

    PDSepConfig                 pd_sep_config;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    CacheConfig                 cache_config;
    cache_config.group_types = {CacheGroupType::FULL};
    RuntimeConfig               runtime_config;
    NormalBatchStreamProcessor  processor(
        model_config, pd_sep_config, profiling_debug_logging_config, cache_config, false);

    // Create decode stream with num_return_sequences = 3 (no beam_search)
    std::shared_ptr<GenerateInput> query1         = make_shared<GenerateInput>();
    query1->input_ids                             = torch::tensor({1}, torch::kInt32);
    query1->generate_config                       = make_shared<GenerateConfig>();
    query1->generate_config->num_beams            = 1;  // No beam_search
    query1->generate_config->num_return_sequences = 3;
    query1->generate_config->do_sample            = true;
    GenerateStreamPtr stream1 =
        make_shared<NormalGenerateStream>(query1, model_config, runtime_config, resource_context, nullptr);
    BatchKVCacheResource addr1;
    // With num_return_sequences = 3, batch_size = 3
    addr1.resetBatchSize(3);
    addr1.initGroups(3, model_config.num_layers, {0, 0});
    addr1.setBatchBlocks(0, 0, {1});
    addr1.setBatchBlocks(1, 0, {2});
    addr1.setBatchBlocks(2, 0, {3});
    stream1->setKVCache(addr1);
    stream1->setIsContextStream(false);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);

    for (const auto& stream : streams) {
        stream->setRunning();
    }

    StreamGroups stream_groups(streams);
    auto         merge_input_status = processor.gatherModelInput(stream_groups);
    EXPECT_TRUE(merge_input_status.ok());

    // With num_return_sequences = 3, currentBatchSize() = 3
    // nan_flag shape should be [3] (one flag per return sequence)
    // nan_flag: [1, 0, 1] means:
    //   - sequence 0: has NaN
    //   - sequence 1: no NaN
    //   - sequence 2: has NaN
    std::vector<float> nan_flag_data = {1, 0, 1};

    MergedOutput merge_outputs;
    merge_outputs.model_output.hidden_states   = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).reshape({3, 2});
    merge_outputs.model_output.logits          = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}).reshape({3, 2});
    merge_outputs.model_output.nan_flag        = torch::tensor(nan_flag_data, torch::kFloat32);
    merge_outputs.sampler_output.token_ids     = torch::tensor({0, 1, 0}, torch::kInt32).reshape({3, 1});
    merge_outputs.sampler_output.cum_log_probs = torch::tensor({1.0f, 2.0f, 3.0f});

    auto status = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());

    // Verify stream is stopped due to NaN in some sequences
    EXPECT_TRUE(stream1->stopped()) << "Stream should be stopped due to NaN with num_return_sequences > 1";
    EXPECT_EQ(stream1->statusInfo().code(), ErrorCode::NAN_DETECTED) << "Error code should be NAN_DETECTED";
    std::string error_msg = stream1->statusInfo().ToString();
    EXPECT_NE(error_msg.find("NaN detected"), std::string::npos) << error_msg;
}

}  // namespace rtp_llm
