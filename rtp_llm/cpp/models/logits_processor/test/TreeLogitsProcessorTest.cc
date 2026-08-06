
#include "gtest/gtest.h"

#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"
#include <unistd.h>

using namespace std;

namespace rtp_llm {

class SamplerDataBuilder {
public:
    SamplerDataBuilder() = default;

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

        BaseLogitsProcessorPtr processor_ptr = std::make_shared<TreeLogitsProcessor>(tree_infos);
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
        sampler_inputs.logits =
            torch::empty({(int64_t)config.batch_size, (int64_t)config.vocab_size},
                         torch::TensorOptions().dtype(dataTypeToTorchType(config.logits_type)).device(torch::kCUDA));
        sampler_inputs.sequence_lengths   = torch::empty({(int64_t)config.batch_size}, torch::kInt32);
        sampler_inputs.input_lengths      = torch::empty({(int64_t)config.batch_size}, torch::kInt32);
        sampler_inputs.num_beams_in       = torch::empty({(int64_t)config.batch_size}, torch::kLong);
        sampler_inputs.num_beams_out      = torch::empty({(int64_t)config.batch_size}, torch::kLong);
        sampler_inputs.top_k              = torch::empty({(int64_t)config.batch_size}, torch::kInt32);
        sampler_inputs.top_p              = torch::empty({(int64_t)config.batch_size}, torch::kFloat32);
        sampler_inputs.temperature        = torch::empty({(int64_t)config.batch_size}, torch::kFloat32);
        sampler_inputs.repetition_penalty = torch::empty({(int64_t)config.batch_size}, torch::kFloat32);
        sampler_inputs.cum_log_probs      = torch::empty({(int64_t)config.batch_size}, torch::kFloat32);
        sampler_inputs.token_ids =
            torch::empty({(int64_t)config.batch_size, (int64_t)(sampler_inputs.step + 1)}, torch::kInt32);
        sampler_inputs.logits.zero_();
        sampler_inputs.token_ids.zero_();
        return sampler_inputs;
    };

    void setTokenIds(SamplerInputs& sampler_inputs, std::vector<std::vector<int>>& token_ids) {
        RTP_LLM_CHECK(token_ids.size() == sampler_inputs.batch_size);
        RTP_LLM_CHECK(token_ids[0].size() == sampler_inputs.step + 1);
        for (auto i = 0; i < sampler_inputs.batch_size; i++) {
            auto tensor = sampler_inputs.token_ids[i];
            for (auto j = 0; j < sampler_inputs.step + 1; j++) {
                tensor[j] = token_ids[i][j];
            }
        }
    }
};

class TreeLogitsProcessorTest: public DeviceTestBase {};

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
    auto vocab_mask     = processor->generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids);
    auto vocab_mask_cpu = vocab_mask.cpu().contiguous();

    std::vector<std::vector<int32_t>> expect_vocab_mask(batch_size, std::vector<int32_t>(vocab_size, 1));
    expect_vocab_mask[1][1] = 0;
    expect_vocab_mask[2][2] = 0;
    expect_vocab_mask[2][3] = 0;
    expect_vocab_mask[2][4] = 0;
    expect_vocab_mask[3][1] = 0;
    expect_vocab_mask[3][3] = 0;
    expect_vocab_mask[3][5] = 0;

    auto vocab_mask_hosts = vocab_mask_cpu.data_ptr<uint8_t>();
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < vocab_size; j++) {
            ASSERT_TRUE(vocab_mask_hosts[i * vocab_size + j] == expect_vocab_mask[i][j]);
        }
    }
}

template<typename Dtype>
void setTensor(torch::Tensor& tensor, std::vector<std::vector<Dtype>> content) {
    RTP_LLM_CHECK(tensor.dim() == 2);
    RTP_LLM_CHECK((size_t)tensor.size(0) == content.size());
    RTP_LLM_CHECK((size_t)tensor.size(1) == content[0].size());
    for (int64_t i = 0; i < tensor.size(0); i++) {
        for (int64_t j = 0; j < tensor.size(1); j++) {
            tensor[i][j] = content[i][j];
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

        auto new_token = torch::tensor({{64000}, {64003}, {64006}, {64008}}, torch::kInt32);

        processor->updateStatus(new_token, 1);

        auto                     proc        = std::dynamic_pointer_cast<TreeLogitsProcessor>(processor);
        std::vector<std::string> status_list = proc->getStatus();
        EXPECT_EQ("225_64000", status_list[0]);
        EXPECT_EQ("225_64003", status_list[1]);
        EXPECT_EQ("225_64006", status_list[2]);
        EXPECT_EQ("225_64008", status_list[3]);

        std::vector<std::vector<int>> token_ids_2 = {{64001}, {64001}, {64004}, {64001}};
        setTensor(new_token, token_ids_2);

        processor->updateStatus(new_token, 1);

        status_list = proc->getStatus();
        EXPECT_EQ("225_64000_64001", status_list[0]);
        EXPECT_EQ("225_64003_64001", status_list[1]);
        EXPECT_EQ("225_64006_64004", status_list[2]);
        EXPECT_EQ("225_64008_64001", status_list[3]);

        std::vector<std::vector<int>> token_ids_3 = {{2}, {2}, {2}, {2}};
        setTensor(new_token, token_ids_3);

        processor->updateStatus(new_token, 1);

        status_list = proc->getStatus();
        EXPECT_EQ("225_64000_64001_2", status_list[0]);
        EXPECT_EQ("225_64003_64001_2", status_list[1]);
        EXPECT_EQ("225_64006_64004_2", status_list[2]);
        EXPECT_EQ("225_64008_64001_2", status_list[3]);

        std::vector<std::vector<int>> token_ids_4 = {{1}, {1}, {1}, {1}};
        setTensor(new_token, token_ids_4);

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
            auto tensor = sampler_inputs.logits[i];
            tensor.fill_(0);
            for (auto index : logits_index_list[i]) {
                tensor[index] = 1;
            }
        }
        processor->process(sampler_inputs, 0, batch_size);

        auto logits_tensor = sampler_inputs.logits[0].cpu();
        auto logits_hosts  = std::vector<float>(logits_tensor.data_ptr<float>(),
                                               logits_tensor.data_ptr<float>() + logits_tensor.numel());
        ASSERT_EQ(logits_hosts[64000], 1);
        ASSERT_EQ(logits_hosts[64003], 0);
        ASSERT_EQ(logits_hosts[64011], 0);
        ASSERT_TRUE(logits_hosts[64001] == -INFINITY);
    }
}

#undef EXPECT_SIMILAR

}  // namespace rtp_llm
