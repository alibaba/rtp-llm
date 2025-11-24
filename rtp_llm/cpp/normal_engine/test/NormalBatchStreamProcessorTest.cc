#include <memory>
#include "torch/all.h"
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

using namespace std;

namespace rtp_llm {

class NormalBatchStreamProcessorTest: public DeviceTestBase {};

TEST_F(NormalBatchStreamProcessorTest, testSimpleAssemble) {
    ResourceContext  resource_context;
    GptInitParameter param;
    param.max_seq_len_        = 2048;
    param.vocab_size_         = 2048;
    param.num_layers_         = 2;
    param.kv_cache_data_type_ = DataType::TYPE_INT8;
    NormalBatchStreamProcessor     processor(param, CacheConfig(), false);
    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = createBuffer<int32_t>({2}, {1, 2}, AllocationType::HOST);
    query1->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream1             = make_shared<NormalGenerateStream>(query1, param, resource_context, nullptr);
    query1->input_ids                     = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    BatchKVCacheResource addr1;
    addr1.resize(1);
    addr1.initGroups(1);
    addr1.batch_resource[0].group_block_ids[0]->block_indices = {1, 2, 3, 4};
    stream1->setKVCache(addr1);
    stream1->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = createBuffer<int32_t>({3}, {1, 2, 3}, AllocationType::HOST);
    query2->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream2             = make_shared<NormalGenerateStream>(query2, param, resource_context, nullptr);
    query2->input_ids                     = createBuffer<int32_t>({2}, {1, 2}, AllocationType::HOST);
    BatchKVCacheResource addr2;
    addr2.resize(1);
    addr2.initGroups(1);
    addr2.batch_resource[0].group_block_ids[0]->block_indices = {5, 6, 7, 8};
    stream2->setKVCache(addr2);
    stream2->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query3 = make_shared<GenerateInput>();
    query3->input_ids                     = createBuffer<int32_t>({3}, {1, 2, 3}, AllocationType::HOST);
    query3->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr    stream3          = make_shared<NormalGenerateStream>(query3, param, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.resize(1);
    addr3.initGroups(1);
    addr3.batch_resource[0].group_block_ids[0]->block_indices = {9, 10};
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4 = make_shared<GenerateInput>();
    query4->input_ids                     = createBuffer<int32_t>({4}, {1, 2, 3, 4}, AllocationType::HOST);
    query4->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr    stream4          = make_shared<NormalGenerateStream>(query4, param, resource_context, nullptr);
    BatchKVCacheResource addr4;
    addr4.resize(1);
    addr4.initGroups(1);
    addr4.batch_resource[0].group_block_ids[0]->block_indices = {11, 12, 13, 14};
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
        EXPECT_EQ(combo_tokens, buffer2vector<int>(*model_input.combo_tokens));
        EXPECT_EQ(input_lengths, buffer2vector<int>(*model_input.input_lengths));
        EXPECT_EQ(sequence_lengths, buffer2vector<int>(*model_input.sequence_lengths));
        EXPECT_EQ(prefix_lengths, buffer2vector<int>(*model_input.prefix_lengths));
        EXPECT_EQ(kv_cache_block_id, buffer2vector<int>(*model_input.kv_cache_block_id));
    }
    {
        NormalBatchStreamProcessor processor(param, CacheConfig(), false);
        StreamGroups               stream_groups(streams);
        auto                       merge_input_status = processor.gatherModelInput(stream_groups);
        EXPECT_TRUE(merge_input_status.ok());
        auto& model_input = merge_input_status.value();
        EXPECT_EQ(model_input.attention_mask.get(), nullptr);
    }
}

TEST_F(NormalBatchStreamProcessorTest, testSoftmaxProbs) {
    ResourceContext  resource_context;
    GptInitParameter param;
    param.max_seq_len_                            = 2048;
    param.vocab_size_                             = 2;
    param.num_layers_                             = 2;
    std::shared_ptr<GenerateInput> query1         = make_shared<GenerateInput>();
    query1->input_ids                             = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query1->generate_config                       = make_shared<GenerateConfig>();
    query1->generate_config->return_softmax_probs = true;
    // query1->generate_config->is_streaming   = true;
    GenerateStreamPtr    stream1 = make_shared<NormalGenerateStream>(query1, param, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resize(1);
    addr1.initGroups(1);
    addr1.batch_resource[0].group_block_ids[0]->block_indices = {1};
    stream1->setKVCache(addr1);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);

    for (const auto& stream : streams) {
        stream->setRunning();
    }
    NormalBatchStreamProcessor processor(param, CacheConfig(), false);
    StreamGroups               stream_groups(streams);
    auto                       merge_input_status = processor.gatherModelInput(stream_groups);
    EXPECT_TRUE(merge_input_status.ok());

    SamplerInputs sampler_inputs;
    MergedOutput  merge_outputs;
    merge_outputs.model_output.hidden_states   = createBuffer<float>({1, 2}, {1, 2});
    merge_outputs.model_output.logits          = createBuffer<float>({1, 2}, {1, 2});
    merge_outputs.sampler_output.token_ids     = createBuffer<int>({1, 2}, {0, 1}, AllocationType::HOST);
    merge_outputs.sampler_output.cum_log_probs = createBuffer<float>({1}, {1});
    auto status                                = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());

    auto softmax_probs = stream1->getSoftmaxProbs();
    EXPECT_TRUE(softmax_probs);
    EXPECT_EQ(2048, softmax_probs->size());
    EXPECT_NEAR(0.731058, *(softmax_probs->dataWithOffset<float>(1)), 0.0001);
}

TEST_F(NormalBatchStreamProcessorTest, testLoss) {
    ResourceContext  resource_context;
    GptInitParameter param;
    param.max_seq_len_                      = 2048;
    param.vocab_size_                       = 2048;
    param.num_layers_                       = 2;
    std::shared_ptr<GenerateInput> query1   = make_shared<GenerateInput>();
    query1->input_ids                       = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query1->generate_config                 = make_shared<GenerateConfig>();
    query1->generate_config->calculate_loss = 1;
    GenerateStreamPtr    stream1 = make_shared<NormalGenerateStream>(query1, param, resource_context, nullptr);
    BatchKVCacheResource addr1;
    addr1.resize(1);
    addr1.initGroups(1);
    addr1.batch_resource[0].group_block_ids[0]->block_indices = {1};
    stream1->setKVCache(addr1);

    std::shared_ptr<GenerateInput> query3   = make_shared<GenerateInput>();
    query3->input_ids                       = createBuffer<int32_t>({2}, {0, 1}, AllocationType::HOST);
    query3->generate_config                 = make_shared<GenerateConfig>();
    query3->generate_config->calculate_loss = 2;
    GenerateStreamPtr    stream3 = make_shared<NormalGenerateStream>(query3, param, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.resize(1);
    addr3.initGroups(1);
    addr3.batch_resource[0].group_block_ids[0]->block_indices = {9};
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4   = make_shared<GenerateInput>();
    query4->input_ids                       = createBuffer<int32_t>({3}, {0, 1, 0}, AllocationType::HOST);
    query4->generate_config                 = make_shared<GenerateConfig>();
    query4->generate_config->calculate_loss = 1;
    GenerateStreamPtr    stream4 = make_shared<NormalGenerateStream>(query4, param, resource_context, nullptr);
    BatchKVCacheResource addr4;
    addr4.resize(1);
    addr4.initGroups(1);
    addr4.batch_resource[0].group_block_ids[0]->block_indices = {11, 12};
    stream4->setKVCache(addr4);

    std::list<GenerateStreamPtr> streams;
    streams.emplace_back(stream1);
    streams.emplace_back(stream3);
    streams.emplace_back(stream4);

    for (const auto& stream : streams) {
        stream->setRunning();
    }
    NormalBatchStreamProcessor processor(param, CacheConfig(), false);
    StreamGroups               stream_groups(streams);
    auto                       merge_input_status = processor.gatherModelInput(stream_groups);
    EXPECT_TRUE(merge_input_status.ok());
    EXPECT_TRUE(merge_input_status.value().need_all_logits);

    SamplerInputs sampler_inputs;
    MergedOutput  merge_outputs;
    merge_outputs.model_output.hidden_states = createBuffer<float>({3, 2}, {1, 2, 3, 4, 5, 6});
    merge_outputs.model_output.logits        = createBuffer<float>({3, 2}, {1, 2, 3, 4, 5, 6});
    merge_outputs.model_output.all_logits    = createBuffer<float>({6, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    merge_outputs.sampler_output.token_ids =
        createBuffer<int>({3, 4}, {0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1}, AllocationType::HOST);
    merge_outputs.sampler_output.cum_log_probs = createBuffer<float>({3}, {1, 2, 3});
    auto status                                = processor.dispatch(stream_groups, merge_outputs);
    EXPECT_TRUE(status.ok());
    EXPECT_FALSE(stream1->getLoss());
    EXPECT_TRUE(stream3->getLoss());
    auto loss3 = stream3->getLoss();
    EXPECT_EQ(1, loss3->size());
    EXPECT_NEAR(0.31326, *(loss3->data<float>()), 0.0001);
    EXPECT_TRUE(stream4->getLoss());
    auto loss4 = stream4->getLoss();
    EXPECT_EQ(2, loss4->size());
    EXPECT_NEAR(2.25525, *(torch::mean(rtp_llm::Buffer2torchTensor(*loss4)).exp().data_ptr<float>()), 0.0001);
}

TEST_F(NormalBatchStreamProcessorTest, testMultimodalGatherBatch) {
    ResourceContext  resource_context;
    GptInitParameter param;
    param.max_seq_len_        = 2048;
    param.vocab_size_         = 2048;
    param.num_layers_         = 2;
    param.kv_cache_data_type_ = DataType::TYPE_INT8;
    param.is_multimodal_      = true;
    NormalBatchStreamProcessor     processor(param, CacheConfig(), false);
    std::shared_ptr<GenerateInput> query1 = make_shared<GenerateInput>();
    query1->input_ids                     = createBuffer<int32_t>({5}, {1, -1, -1, -1, 2}, AllocationType::HOST);
    query1->generate_config               = make_shared<GenerateConfig>();
    query1->mm_locs                       = createBuffer<int32_t>({1}, {1}, AllocationType::HOST);
    query1->text_tokens_mask              = createBuffer<int32_t>({5}, {1, 0, 0, 0, 1}, AllocationType::HOST);
    query1->multimodal_features           = {torch::rand({3, 10}, torch::kFloat16)};
    GenerateStreamPtr stream1             = make_shared<NormalGenerateStream>(query1, param, resource_context, nullptr);
    stream1->setIsContextStream(true);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = createBuffer<int32_t>({3}, {3, 4, 5}, AllocationType::HOST);
    query2->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream2             = make_shared<NormalGenerateStream>(query2, param, resource_context, nullptr);
    stream2->setIsContextStream(true);

    std::shared_ptr<GenerateInput> query3 = make_shared<GenerateInput>();
    query3->input_ids                     = createBuffer<int32_t>({5}, {6, 7, -1, -1, 8}, AllocationType::HOST);
    query3->generate_config               = make_shared<GenerateConfig>();
    query3->mm_locs                       = createBuffer<int32_t>({1}, {2}, AllocationType::HOST);
    query3->text_tokens_mask              = createBuffer<int32_t>({5}, {1, 1, 0, 0, 1}, AllocationType::HOST);
    query3->multimodal_features           = {torch::rand({2, 10}, torch::kFloat16)};
    GenerateStreamPtr stream3             = make_shared<NormalGenerateStream>(query3, param, resource_context, nullptr);
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

        EXPECT_EQ(combo_tokens, buffer2vector<int>(*model_input.combo_tokens));
        EXPECT_EQ(input_lengths, buffer2vector<int>(*model_input.input_lengths));
        EXPECT_EQ(text_tokens_mask, buffer2vector<int>(*model_input.text_tokens_mask));
        EXPECT_EQ(mm_features_locs, buffer2vector<int>(*model_input.mm_features_locs));

        EXPECT_EQ(model_input.multimodal_features.value().size(), 2);
        EXPECT_EQ(model_input.multimodal_features.value()[0]->size(), 3 * 10);
        EXPECT_EQ(model_input.multimodal_features.value()[1]->size(), 2 * 10);
    }
}

}  // namespace rtp_llm
