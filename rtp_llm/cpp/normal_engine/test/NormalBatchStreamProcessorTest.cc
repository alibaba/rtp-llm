#include <memory>
#include <cmath>
#include <set>
#include "torch/all.h"
#include "gtest/gtest.h"

#define private public
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"

using namespace std;

namespace rtp_llm {

class NormalBatchStreamProcessorTest: public DeviceTestBase {};

class CsrVariableBeamTest: public DeviceTestBase {
protected:
    // A complete layered trie; tokens occupy disjoint ranges at each depth.
    // Keep snapshots local to these tests so other stream tests stay unconstrained.
    ConstraintTreeCsrSnapshotPtr makeTree(const std::vector<int>& fanouts, int vocab) {
        auto tree             = std::make_shared<ConstraintTreeCsrSnapshot>();
        tree->version_        = 1;
        tree->start_token_id_ = vocab - 2;
        tree->end_token_id_   = vocab - 1;
        tree->row_ptr_        = {0};
        int level_size = 1, next_state = 1, token_begin = 10;
        for (int fanout : fanouts) {
            for (int node = 0; node < level_size; ++node) {
                for (int child = 0; child < fanout; ++child) {
                    tree->col_idx_.push_back(token_begin + child);
                    tree->next_state_.push_back(next_state++);
                }
                tree->row_ptr_.push_back(tree->col_idx_.size());
            }
            level_size *= fanout;
            token_begin += fanout;
        }
        tree->sid_count_           = level_size;
        tree->terminal_mask_state_ = tree->row_ptr_.size() - 1;
        for (int node = 0; node < level_size; ++node) {
            tree->col_idx_.push_back(tree->endTokenId());
            tree->next_state_.push_back(-1);
            tree->row_ptr_.push_back(tree->col_idx_.size());
        }
        tree->device_row_ptr_ = device_->clone({*vector2Buffer(tree->row_ptr_), AllocationType::DEVICE});
        tree->device_col_idx_ = device_->clone({*vector2Buffer(tree->col_idx_), AllocationType::DEVICE});
        return tree;
    }

    GenerateStreamPtr
    makeStream(const ConstraintTreeCsrSnapshotPtr& tree, const std::vector<int>& schedule, int steps, int vocab) {
        auto input                                 = std::make_shared<GenerateInput>();
        input->input_ids                           = vector2Buffer(std::vector<int>{1, 2});
        input->generate_config                     = std::make_shared<GenerateConfig>();
        input->generate_config->variable_num_beams = schedule;
        input->generate_config->num_beams          = *std::max_element(schedule.begin(), schedule.end());
        input->generate_config->max_new_tokens     = steps;
        input->generate_config->is_streaming       = true;
        input->generate_config->do_sample          = false;
        input->generate_config->top_k              = 1;
        GptInitParameter params;
        params.max_seq_len_                  = 32;
        params.vocab_size_                   = vocab;
        params.special_tokens_.eos_token_id_ = vocab - 1;
        params.seq_size_per_block_           = 8;
        ResourceContext resources;
        resources.cache_manager =
            std::make_shared<CacheManager>(CacheConfig(KVCacheParam{1, 20000, 1, 8, 8, DataType::TYPE_FP32}), device_);
        auto stream = std::make_shared<NormalGenerateStream>(input, params, resources, nullptr);
        RTP_LLM_CHECK(stream->initKVBlock(params.max_seq_len_).ok());
        stream->tree_logits_processor_ptr_ = std::make_shared<TreeLogitsProcessor>(
            device_, std::vector<StreamTreeInfo>{StreamTreeInfo(true, 2, 0, true, tree)});
        stream->initializeLogitsProcessorList();
        stream->setRunning();
        return stream;
    }

    void step(const std::list<GenerateStreamPtr>& streams, int vocab) {
        GptInitParameter params;
        params.vocab_size_ = vocab;
        NormalBatchStreamProcessor processor(params, CacheConfig(), false);
        StreamGroups               groups(streams);
        MergedOutput               output;
        output.model_output.logits = device_->allocateBuffer(
            {DataType::TYPE_FP32, {groups.totalSamplerBatchSizeIn(), (size_t)vocab}, AllocationType::DEVICE});
        // Non-uniform deterministic scores exercise ranking/parent reordering.
        auto logits = Buffer2torchTensor(*output.model_output.logits, false);
        logits.copy_(torch::arange(vocab, logits.options()).remainder(97).mul_(0.01));
        auto inputs = processor.gatherSamplerInput(groups, GptModelInputs{}, output.model_output);
        ASSERT_TRUE(inputs.ok());
        Sampler sampler({device_, vocab - 1, groups.totalSamplerBatchSizeOut()});
        output.sampler_output = sampler.forward(inputs.value());
        device_->syncDeviceStream(DeviceStream::DEFAULT);
        ASSERT_TRUE(processor.dispatch(groups, output).ok());
    }

    void expectValidOutput(const GenerateStreamPtr& stream, const ConstraintTreeCsrSnapshotPtr& tree, int width) {
        ASSERT_FALSE(stream->stopped()) << stream->statusInfo().ToString();
        // Existing beam search emits only the final result, even when the
        // request asks for streaming. Still verify every intermediate state.
        if (stream->finished()) {
            ASSERT_TRUE(stream->hasOutput());
            auto output = stream->nextOutput();
            ASSERT_TRUE(output.ok());
            ASSERT_EQ(width, output.value().generate_outputs.size());
        } else {
            EXPECT_FALSE(stream->hasOutput());
        }
        ASSERT_EQ(width, stream->getTreeLogitsProcessor()->size());
        std::set<std::vector<int>> unique;
        for (int beam = 0; beam < width; ++beam) {
            auto tokens = stream->completeTokenIdsVec(beam);
            int  state  = 0;
            for (size_t position = 2; position < tokens.size(); ++position) {
                state = tree->transition(state, tokens[position]);
                ASSERT_NE(ConstraintTreeCsrSnapshot::INVALID_TRANSITION, state);
            }
            EXPECT_TRUE(unique.insert(tokens).second);
            EXPECT_TRUE(std::isfinite(stream->cumLogProbs()->data<float>()[beam]));
        }
    }
};

TEST_F(CsrVariableBeamTest, GrowShrinkHoldOneAndGrowAgain) {
    const int vocab = 128;
    auto      tree  = makeTree({4, 4, 4, 4, 4}, vocab);
    for (const auto& schedule : std::vector<std::vector<int>>{{2, 4, 1, 1, 3, 3}, {1, 1, 3, 2, 4, 4}, {3}}) {
        auto stream = makeStream(tree, schedule, 6, vocab);
        for (int index = 0; index < 6; ++index) {
            ASSERT_NO_THROW(step({stream}, vocab));
            expectValidOutput(stream, tree, schedule[std::min(index, (int)schedule.size() - 1)]);
            ASSERT_FALSE(stream->stopped());
        }
        EXPECT_TRUE(stream->finished());
    }
}

TEST_F(CsrVariableBeamTest, BusinessWidth512To3500AndEos) {
    const int vocab  = 8192;
    auto      tree   = makeTree({512, 8}, vocab);
    auto      stream = makeStream(tree, {512, 3500}, 3, vocab);
    for (int width : {512, 3500, 3500}) {
        ASSERT_NO_THROW(step({stream}, vocab));
        expectValidOutput(stream, tree, width);
        ASSERT_FALSE(stream->stopped());
    }
    EXPECT_TRUE(stream->finished());
}

TEST_F(CsrVariableBeamTest, FinalOutputUsesCurrentRatherThanNextScheduledWidth) {
    auto tree   = makeTree({4, 4, 4}, 128);
    auto stream = makeStream(tree, {2, 3, 4}, 2, 128);
    ASSERT_NO_THROW(step({stream}, 128));
    expectValidOutput(stream, tree, 2);
    ASSERT_NO_THROW(step({stream}, 128));
    expectValidOutput(stream, tree, 3);
    EXPECT_TRUE(stream->finished());
}

TEST_F(CsrVariableBeamTest, InsufficientCandidatesRejectOnlyAffectedRequest) {
    const int vocab = 128;
    auto      tree  = makeTree({2, 2}, vocab);
    auto      bad   = makeStream(tree, {2, 5}, 2, vocab);
    auto      good  = makeStream(tree, {2, 3}, 2, vocab);
    ASSERT_NO_THROW(step({bad, good}, vocab));
    expectValidOutput(bad, tree, 2);
    expectValidOutput(good, tree, 2);
    ASSERT_NO_THROW(step({bad, good}, vocab));
    EXPECT_TRUE(bad->stopped());
    EXPECT_FALSE(bad->hasOutput());
    expectValidOutput(good, tree, 3);
    EXPECT_TRUE(good->finished());
}

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
    addr1.batch_block_id = {{1, 2, 3, 4}};
    stream1->setKVCache(addr1);
    stream1->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query2 = make_shared<GenerateInput>();
    query2->input_ids                     = createBuffer<int32_t>({3}, {1, 2, 3}, AllocationType::HOST);
    query2->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr stream2             = make_shared<NormalGenerateStream>(query2, param, resource_context, nullptr);
    query2->input_ids                     = createBuffer<int32_t>({2}, {1, 2}, AllocationType::HOST);
    BatchKVCacheResource addr2;
    addr2.batch_block_id = {{5, 6, 7, 8}};
    stream2->setKVCache(addr2);
    stream2->setIsContextStream(false);

    std::shared_ptr<GenerateInput> query3 = make_shared<GenerateInput>();
    query3->input_ids                     = createBuffer<int32_t>({3}, {1, 2, 3}, AllocationType::HOST);
    query3->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr    stream3          = make_shared<NormalGenerateStream>(query3, param, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.batch_block_id = {{9, 10}};
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4 = make_shared<GenerateInput>();
    query4->input_ids                     = createBuffer<int32_t>({4}, {1, 2, 3, 4}, AllocationType::HOST);
    query4->generate_config               = make_shared<GenerateConfig>();
    GenerateStreamPtr    stream4          = make_shared<NormalGenerateStream>(query4, param, resource_context, nullptr);
    BatchKVCacheResource addr4;
    addr4.batch_block_id = {{11, 12, 13, 14}};
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
    addr1.batch_block_id = {{1}};
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
    addr1.batch_block_id = {{1}};
    stream1->setKVCache(addr1);

    std::shared_ptr<GenerateInput> query3   = make_shared<GenerateInput>();
    query3->input_ids                       = createBuffer<int32_t>({2}, {0, 1}, AllocationType::HOST);
    query3->generate_config                 = make_shared<GenerateConfig>();
    query3->generate_config->calculate_loss = 2;
    GenerateStreamPtr    stream3 = make_shared<NormalGenerateStream>(query3, param, resource_context, nullptr);
    BatchKVCacheResource addr3;
    addr3.batch_block_id = {{9}};
    stream3->setKVCache(addr3);

    std::shared_ptr<GenerateInput> query4   = make_shared<GenerateInput>();
    query4->input_ids                       = createBuffer<int32_t>({3}, {0, 1, 0}, AllocationType::HOST);
    query4->generate_config                 = make_shared<GenerateConfig>();
    query4->generate_config->calculate_loss = 1;
    GenerateStreamPtr    stream4 = make_shared<NormalGenerateStream>(query4, param, resource_context, nullptr);
    BatchKVCacheResource addr4;
    addr4.batch_block_id = {{11, 12}};
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
