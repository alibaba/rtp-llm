
#include "gtest/gtest.h"

#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include <unistd.h>

using namespace std;

namespace rtp_llm {

class SamplerDataBuilder {
public:
    SamplerDataBuilder(): device_(rtp_llm::DeviceFactory::getDefaultDevice()) {};

    struct Config {
        size_t            batch_size;
        size_t            vocab_size;
        size_t            max_length;
        rtp_llm::DataType logits_type = rtp_llm::DataType::TYPE_FP32;
    };

    BaseLogitsProcessorPtr generateLogitsProcessor(bool in_tree_mode, size_t batch_size, std::string file_path) {
        std::vector<StreamTreeInfo> tree_infos;

        PrefixToCandidateTokens::instance()->reloadPrefixDict(file_path);

        for (size_t i = 0; i < batch_size; i++) {
            auto tree_info =
                StreamTreeInfo(in_tree_mode,
                               0,
                               0,
                               0,
                               std::make_shared<TreeDFA<std::string, int>>(PrefixToCandidateTokens::instance()));
            tree_infos.push_back(tree_info);
        }

        BaseLogitsProcessorPtr processor_ptr = std::make_shared<TreeLogitsProcessor>(device_, tree_infos);
        return processor_ptr;
    }

    SamplerInputs allocate(Config config, std::vector<BaseLogitsProcessorPtr> processors, std::vector<size_t> nums) {
        SamplerInputs sampler_inputs;

        sampler_inputs.step                = config.max_length;
        sampler_inputs.batch_size          = config.batch_size;
        sampler_inputs.batch_size_out      = config.batch_size;
        sampler_inputs.vocab_size          = config.vocab_size;
        LogitsProcessorStatesPtr state_ptr = std::make_shared<LogitsProcessorStates>();
        for (size_t i = 0, idx = 0; i < processors.size(); i++) {
            state_ptr->insert(processors[i], idx, idx + nums[i]);
            idx += nums[i];
        }
        sampler_inputs.logits_processor_states_ptr = state_ptr;
        sampler_inputs.logits                      = device_->allocateBuffer(
            {config.logits_type, {config.batch_size, config.vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
        sampler_inputs.sequence_lengths = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.input_lengths = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.num_beams_in = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_UINT64, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.num_beams_out = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_UINT64, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.top_k = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_UINT32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.top_p = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.temperature = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.random_seeds = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_UINT64, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.repetition_penalty = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.min_lengths = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.cum_log_probs = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {config.batch_size}, rtp_llm::AllocationType::HOST}, {});
        sampler_inputs.token_ids = device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                                            {config.batch_size, sampler_inputs.step + 1},
                                                            rtp_llm::AllocationType::HOST},
                                                           {});
        device_->bufMemset(*sampler_inputs.logits, 0);
        device_->bufMemset(*sampler_inputs.token_ids, 0);
        return sampler_inputs;
    };

    void setTokenIds(SamplerInputs& sampler_inputs, std::vector<std::vector<int>>& token_ids) {
        RTP_LLM_CHECK(token_ids.size() == sampler_inputs.batch_size);
        RTP_LLM_CHECK(token_ids[0].size() == sampler_inputs.step + 1);
        for (auto i = 0; i < sampler_inputs.batch_size; i++) {
            auto tensor = Buffer2torchTensor(*sampler_inputs.token_ids->index(i), false);
            for (auto j = 0; j < sampler_inputs.step + 1; j++) {
                tensor[j] = token_ids[i][j];
            }
        }
    }

    rtp_llm::DeviceBase* device_;
};

class TreeLogitsProcessorTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
    }

    void TearDown() override {
        DeviceTestBase::TearDown();
    }

    rtp_llm::BufferPtr randint(int start, int end, std::vector<int64_t> shape, bool is_host) {
        auto tensor  = torch::randint(start, end, shape, at::TensorOptions().dtype(at::ScalarType::Int));
        auto alloc_t = is_host ? AllocationType::HOST : AllocationType::DEVICE;
        return tensorToBuffer(tensor, alloc_t);
    }

    rtp_llm::BufferPtr rand(std::vector<int64_t> shape, bool is_host) {
        auto tensor  = torch::rand(torch::IntArrayRef(shape));
        auto alloc_t = is_host ? AllocationType::HOST : AllocationType::DEVICE;
        return tensorToBuffer(tensor, alloc_t);
    }
};

#define EXPECT_SIMILAR(vec1, vec2, eps)                                                                                \
    do {                                                                                                               \
        bool similar = true;                                                                                           \
        if (vec1.size() != vec2.size()) {                                                                              \
            similar = false;                                                                                           \
        } else {                                                                                                       \
            for (size_t i = 0; i < vec1.size(); ++i) {                                                                 \
                if (std::fabs(vec1[i] - vec2[i]) >= eps) {                                                             \
                    similar = false;                                                                                   \
                    break;                                                                                             \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        EXPECT_TRUE(similar) << "Vectors are not similar";                                                             \
    } while (0)

TEST_F(TreeLogitsProcessorTest, testGenerateVocabMask) {
    SamplerDataBuilder     builder;
    std::string            file_path  = "./rtp_llm/cpp/models/logits_processor/test/gir_prefix_dict.json";
    size_t                 batch_size = 4;
    size_t                 vocab_size = 1024;
    size_t                 max_length = 1024;
    BaseLogitsProcessorPtr processor  = builder.generateLogitsProcessor(true, batch_size, file_path);
    SamplerInputs sampler_inputs = builder.allocate({batch_size, vocab_size, max_length}, {processor}, {batch_size});
    std::vector<std::vector<size_t>> batch_candidate_token_ids = {{}, {1}, {2, 3, 4}, {1, 3, 5}};
    rtp_llm::BufferPtr vocab_mask = processor->generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids);

    std::vector<std::vector<int32_t>> expect_vocab_mask(batch_size, std::vector<int32_t>(vocab_size, 1));
    expect_vocab_mask[1][1] = 0;
    expect_vocab_mask[2][2] = 0;
    expect_vocab_mask[2][3] = 0;
    expect_vocab_mask[2][4] = 0;
    expect_vocab_mask[3][1] = 0;
    expect_vocab_mask[3][3] = 0;
    expect_vocab_mask[3][5] = 0;

    auto vocab_mask_hosts = getBufferValues<uint8_t>(*vocab_mask);
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < vocab_size; j++) {
            ASSERT_TRUE(vocab_mask_hosts[i * vocab_size + j] == expect_vocab_mask[i][j]);
        }
    }
}

template<typename Dtype>
void setBuffer(rtp_llm::BufferPtr buf, std::vector<std::vector<Dtype>> content) {
    RTP_LLM_CHECK(buf->shape().size() == 2);
    RTP_LLM_CHECK(buf->shape()[0] == content.size());
    RTP_LLM_CHECK(buf->shape()[1] == content[0].size());
    for (auto i = 0; i < buf->shape()[0]; i++) {
        auto tensor = Buffer2torchTensor(*buf->index(i), false);
        for (auto j = 0; j < buf->shape()[1]; j++) {
            tensor[j] = content[i][j];
        }
    }
}

TEST_F(TreeLogitsProcessorTest, testUpdateStatus) {
    {
        SamplerDataBuilder     builder;
        std::string            file_path  = "./rtp_llm/cpp/models/logits_processor/test/gir_prefix_dict.json";
        size_t                 batch_size = 4;
        size_t                 vocab_size = 1024;
        size_t                 max_length = 10;
        BaseLogitsProcessorPtr processor  = builder.generateLogitsProcessor(true, batch_size, file_path);
        SamplerInputs          sampler_inputs =
            builder.allocate({batch_size, vocab_size, max_length}, {processor}, {batch_size});

        rtp_llm::BufferPtr new_token = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {batch_size, 1}, rtp_llm::AllocationType::HOST}, {});
        std::vector<std::vector<int>> new_token_ids = {{64000}, {64003}, {64006}, {64008}};
        setBuffer(new_token, new_token_ids);

        processor->updateStatus(new_token, 1);

        auto                     proc        = std::dynamic_pointer_cast<TreeLogitsProcessor>(processor);
        std::vector<std::string> status_list = proc->getStatus();
        EXPECT_EQ("225_64000", status_list[0]);
        EXPECT_EQ("225_64003", status_list[1]);
        EXPECT_EQ("225_64006", status_list[2]);
        EXPECT_EQ("225_64008", status_list[3]);

        std::vector<std::vector<int>> token_ids_2 = {{64001}, {64001}, {64004}, {64001}};
        setBuffer(new_token, token_ids_2);

        processor->updateStatus(new_token, 1);

        status_list = proc->getStatus();
        EXPECT_EQ("225_64000_64001", status_list[0]);
        EXPECT_EQ("225_64003_64001", status_list[1]);
        EXPECT_EQ("225_64006_64004", status_list[2]);
        EXPECT_EQ("225_64008_64001", status_list[3]);

        std::vector<std::vector<int>> token_ids_3 = {{2}, {2}, {2}, {2}};
        setBuffer(new_token, token_ids_3);

        processor->updateStatus(new_token, 1);

        status_list = proc->getStatus();
        EXPECT_EQ("225_64000_64001_2", status_list[0]);
        EXPECT_EQ("225_64003_64001_2", status_list[1]);
        EXPECT_EQ("225_64006_64004_2", status_list[2]);
        EXPECT_EQ("225_64008_64001_2", status_list[3]);

        std::vector<std::vector<int>> token_ids_4 = {{1}, {1}, {1}, {1}};
        setBuffer(new_token, token_ids_4);

        processor->updateStatus(new_token, 1);

        status_list = proc->getStatus();
        EXPECT_EQ("225_64000_64001_2", status_list[0]);
        EXPECT_EQ("225_64003_64001_2", status_list[1]);
        EXPECT_EQ("225_64006_64004_2", status_list[2]);
        EXPECT_EQ("225_64008_64001_2", status_list[3]);
    }
}

TEST_F(TreeLogitsProcessorTest, testProcess) {
    {
        SamplerDataBuilder     builder;
        std::string            file_path  = "./rtp_llm/cpp/models/logits_processor/test/gir_prefix_dict.json";
        size_t                 batch_size = 4;
        size_t                 vocab_size = 100000;
        size_t                 max_length = 10;
        BaseLogitsProcessorPtr processor  = builder.generateLogitsProcessor(true, batch_size, file_path);
        SamplerInputs          sampler_inputs =
            builder.allocate({batch_size, vocab_size, max_length}, {processor}, {batch_size});

        std::vector<std::vector<int>> token_ids = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51},
                                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9}};
        builder.setTokenIds(sampler_inputs, token_ids);

        std::vector<std::vector<float>> logits_list;
        std::vector<std::vector<float>> logits_index_list = {{64000}, {64003, 64006, 64008}, {64011}, {64001}};
        for (size_t i = 0; i < batch_size; i++) {
            auto logits = sampler_inputs.logits->index(i);
            auto tensor = Buffer2torchTensor(*logits, false);
            tensor.fill_(0);
            for (auto index : logits_index_list[i]) {
                tensor[index] = 1;
            }
        }
        processor->process(sampler_inputs, 0, batch_size);

        auto logits       = sampler_inputs.logits->index(0);
        auto logits_hosts = getBufferValues<float>(*logits);
        ASSERT_EQ(logits_hosts[64000], 1);
        ASSERT_EQ(logits_hosts[64003], 0);
        ASSERT_EQ(logits_hosts[64011], 0);
        ASSERT_TRUE(logits_hosts[64001] == -INFINITY);
    }
}

TEST_F(TreeLogitsProcessorTest, testGenerateVocabWeight) {
    SamplerDataBuilder     builder;
    std::string            file_path  = "./rtp_llm/cpp/models/logits_processor/test/gir_prefix_with_weight_dict.json";
    size_t                 batch_size = 4;
    size_t                 vocab_size = 1024;
    size_t                 max_length = 1024;
    BaseLogitsProcessorPtr processor  = builder.generateLogitsProcessor(true, batch_size, file_path);
    SamplerInputs sampler_inputs = builder.allocate({batch_size, vocab_size, max_length}, {processor}, {batch_size});

    std::vector<TokenWeights> token_weights(4);
    token_weights[0].token_ids = {1, 5};
    token_weights[0].weights   = {0.1f, 0.3f};
    token_weights[1].token_ids = {2, 3, 4};
    token_weights[1].weights   = {0.1f, 0.2f, 0.3f};
    token_weights[2].token_ids = {1, 3, 5};
    token_weights[2].weights   = {0.1f, 0.2f, 0.3f};
    token_weights[3].token_ids = {1, 3};
    token_weights[3].weights   = {0.1f, 0.2f};

    std::vector<const TokenWeights*> batch_candidate_token_weights = {
        &token_weights[0], &token_weights[1], &token_weights[2], &token_weights[3]};
    std::vector<rtp_llm::BufferPtr> vocab_weight =
        processor->generateVocabWeight(batch_size, vocab_size, batch_candidate_token_weights);

    std::vector<int>   expect_batch_idx = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<int>   expect_token_idx = {1, 5, 2, 3, 4, 1, 3, 5, 1, 3};
    std::vector<float> expect_weight    = {0.1f, 0.3f, 0.1f, 0.2f, 0.3f, 0.1f, 0.2f, 0.3f, 0.1f, 0.2f};

    auto h_batch_indices = getBufferValues<int>(*vocab_weight[0]);
    auto h_token_indices = getBufferValues<int>(*vocab_weight[1]);
    auto h_weights       = getBufferValues<float>(*vocab_weight[2]);
    for (size_t i = 0; i < h_batch_indices.size(); i++) {
        EXPECT_EQ(expect_batch_idx[i], h_batch_indices[i]);
        EXPECT_EQ(expect_token_idx[i], h_token_indices[i]);
        EXPECT_EQ(expect_weight[i], h_weights[i]);
    }
}

TEST_F(TreeLogitsProcessorTest, testWeightProcess) {
    SamplerDataBuilder     builder;
    std::string            file_path  = "./rtp_llm/cpp/models/logits_processor/test/gir_prefix_with_weight_dict.json";
    size_t                 batch_size = 4;
    size_t                 vocab_size = 100000;
    size_t                 max_length = 10;
    BaseLogitsProcessorPtr processor  = builder.generateLogitsProcessor(true, batch_size, file_path);
    SamplerInputs sampler_inputs = builder.allocate({batch_size, vocab_size, max_length}, {processor}, {batch_size});

    std::vector<std::vector<int>> token_ids = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                                               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51},
                                               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9}};
    builder.setTokenIds(sampler_inputs, token_ids);

    std::vector<std::vector<float>> logits_list;
    std::vector<std::vector<float>> logits_index_list = {{64000}, {64003, 64006, 64008}, {64011}, {64001}};
    for (size_t i = 0; i < batch_size; i++) {
        auto logits = sampler_inputs.logits->index(i);
        auto tensor = Buffer2torchTensor(*logits, false);
        tensor.fill_(0);
        int idx = 1;
        for (auto index : logits_index_list[i]) {
            tensor[index] = i + 1 + 0.1f * idx;
            idx++;
        }
    }
    processor->process(sampler_inputs, 0, batch_size);

    auto logits       = sampler_inputs.logits->index(0);
    auto logits_hosts = getBufferValues<float>(*logits);
    ASSERT_FLOAT_EQ(logits_hosts[64000], 1.2f);
    ASSERT_FLOAT_EQ(logits_hosts[64003], 0.15f);
    ASSERT_FLOAT_EQ(logits_hosts[64006], 0.12f);
    ASSERT_FLOAT_EQ(logits_hosts[64008], 0.2f);
    ASSERT_FLOAT_EQ(logits_hosts[64011], 0.3f);
    ASSERT_TRUE(logits_hosts[64001] == -INFINITY);

    logits       = sampler_inputs.logits->index(1);
    logits_hosts = getBufferValues<float>(*logits);
    ASSERT_FLOAT_EQ(logits_hosts[64000], 0.1f);
    ASSERT_FLOAT_EQ(logits_hosts[64003], 2.25f);
    ASSERT_FLOAT_EQ(logits_hosts[64006], 2.32f);
    ASSERT_FLOAT_EQ(logits_hosts[64008], 2.5f);
    ASSERT_FLOAT_EQ(logits_hosts[64011], 0.3f);
    ASSERT_TRUE(logits_hosts[64001] == -INFINITY);

    logits       = sampler_inputs.logits->index(2);
    logits_hosts = getBufferValues<float>(*logits);
    ASSERT_FLOAT_EQ(logits_hosts[64000], 0.1f);
    ASSERT_FLOAT_EQ(logits_hosts[64003], 0.15f);
    ASSERT_FLOAT_EQ(logits_hosts[64006], 0.12f);
    ASSERT_FLOAT_EQ(logits_hosts[64008], 0.2f);
    ASSERT_FLOAT_EQ(logits_hosts[64011], 3.4f);
    ASSERT_TRUE(logits_hosts[64001] == -INFINITY);

    logits       = sampler_inputs.logits->index(3);
    logits_hosts = getBufferValues<float>(*logits);
    ASSERT_FLOAT_EQ(logits_hosts[64000], 0.1f);
    ASSERT_FLOAT_EQ(logits_hosts[64003], 0.15f);
    ASSERT_FLOAT_EQ(logits_hosts[64006], 0.12f);
    ASSERT_FLOAT_EQ(logits_hosts[64008], 0.2f);
    ASSERT_FLOAT_EQ(logits_hosts[64011], 0.3f);
    ASSERT_TRUE(logits_hosts[64001] == -INFINITY);
}

#undef EXPECT_SIMILAR

}  // namespace rtp_llm
