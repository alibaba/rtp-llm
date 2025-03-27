
#include "gtest/gtest.h"

#include "src/fastertransformer/devices/testing/TestBase.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/models/ThinkModeLogitsProcessor.h"
#include "src/fastertransformer/core/BufferHelper.h"

using namespace std;

namespace ft = fastertransformer;

namespace rtp_llm {

class SamplerDataBuilder {
public:

    SamplerDataBuilder() :
        device_(ft::DeviceFactory::getDefaultDevice()) {};

    struct Config {
        size_t batch_size;
        size_t vocab_size;
        size_t max_length;
        ft::DataType logits_type = ft::DataType::TYPE_FP32;
    };

    BaseLogitsProcessorPtr generateLogitsProcessor(bool think_mode, std::vector<int> max_thinking_tokens, std::vector<int> end_think_token_ids, std::vector<int> think_status) {
        std::vector<StreamThinkInfo> think_infos;

        size_t batch_size = max_thinking_tokens.size();
        for (size_t i = 0; i < batch_size; i++) {
            auto think_info = StreamThinkInfo(
                think_mode, max_thinking_tokens[i], end_think_token_ids, std::make_shared<StringContainDFA<size_t, int>>(end_think_token_ids)
            );
            think_info.think_end_status_dfa_ptr->forceSetStatus(think_status[i]);
            think_infos.push_back(think_info);
        }

        BaseLogitsProcessorPtr processor_ptr = std::make_shared<ThinkModeLogitsProcessor>(device_, think_infos);
        return processor_ptr;
    }

    SamplerInputs allocate(Config config, std::vector<BaseLogitsProcessorPtr> grammars) {
        SamplerInputs sampler_inputs;

        sampler_inputs.step                 = config.max_length;
        sampler_inputs.batch_size           = config.batch_size;
        sampler_inputs.vocab_size           = config.vocab_size;
        sampler_inputs.grammars.clear();
        sampler_inputs.grammars.insert(sampler_inputs.grammars.end(), grammars.begin(), grammars.end());
        sampler_inputs.logits               = device_->allocateBuffer({config.logits_type, {config.batch_size, config.vocab_size}, ft::AllocationType::HOST}, {});
        sampler_inputs.sequence_lengths     = device_->allocateBuffer({ft::DataType::TYPE_INT32, {config.batch_size}, ft::AllocationType::HOST}, {});
        sampler_inputs.input_lengths        = device_->allocateBuffer({ft::DataType::TYPE_INT32, {config.batch_size}, ft::AllocationType::HOST}, {});
        sampler_inputs.num_beams            = device_->allocateBuffer({ft::DataType::TYPE_UINT64, {config.batch_size}, ft::AllocationType::HOST}, {});
        sampler_inputs.top_k                = device_->allocateBuffer({ft::DataType::TYPE_UINT32, {config.batch_size}, ft::AllocationType::HOST}, {});
        sampler_inputs.top_p                = device_->allocateBuffer({ft::DataType::TYPE_FP32, {config.batch_size}, ft::AllocationType::HOST}, {});
        sampler_inputs.temperature          = device_->allocateBuffer({ft::DataType::TYPE_FP32, {config.batch_size}, ft::AllocationType::HOST}, {});
        sampler_inputs.random_seeds         = device_->allocateBuffer({ft::DataType::TYPE_UINT64, {config.batch_size}, ft::AllocationType::HOST}, {});
        sampler_inputs.repetition_penalty   = device_->allocateBuffer({ft::DataType::TYPE_FP32, {config.batch_size}, ft::AllocationType::HOST}, {});
        sampler_inputs.min_lengths          = device_->allocateBuffer({ft::DataType::TYPE_INT32, {config.batch_size}, ft::AllocationType::HOST}, {});
        sampler_inputs.cum_log_probs        = device_->allocateBuffer({ft::DataType::TYPE_FP32, {config.batch_size}, ft::AllocationType::HOST}, {});
        sampler_inputs.token_ids            = device_->allocateBuffer({ft::DataType::TYPE_INT32, {config.batch_size, sampler_inputs.step + 1}, ft::AllocationType::HOST}, {});
        device_->bufMemset(*sampler_inputs.logits, 0);
        device_->bufMemset(*sampler_inputs.token_ids, 0);
        return sampler_inputs;
    };

    void setSequenceLengths(SamplerInputs& sampler_inputs, std::vector<int>& sequence_lengths) {
        FT_CHECK(sequence_lengths.size() == sampler_inputs.batch_size);
        sampler_inputs.sequence_lengths = ft::vector2Buffer(sequence_lengths);
    };

    void setTokenIds(SamplerInputs& sampler_inputs, std::vector<std::vector<int>>& token_ids) {
        FT_CHECK(token_ids.size() == sampler_inputs.batch_size);
        FT_CHECK(token_ids[0].size() == sampler_inputs.step + 1);
        for (auto i = 0; i < sampler_inputs.batch_size; i++) {
            auto tensor = Buffer2torchTensor(*sampler_inputs.token_ids->index(i), false);
            for (auto j = 0; j < sampler_inputs.step + 1; j++) {
                tensor[j] = token_ids[i][j];
            }
        }
    }

    ft::DeviceBase* device_;
};


class SamplerTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
    }

    void TearDown() override {
        DeviceTestBase::TearDown();
    }
    
    ft::BufferPtr randint(int start, int end, std::vector<int64_t> shape, bool is_host) {
        auto tensor  = torch::randint(start, end, shape, at::TensorOptions().dtype(at::ScalarType::Int));
        auto alloc_t = is_host ? AllocationType::HOST : AllocationType::DEVICE;
        return tensorToBuffer(tensor, alloc_t);
    }

    ft::BufferPtr rand(std::vector<int64_t> shape, bool is_host) {
        auto tensor  = torch::rand(torch::IntArrayRef(shape));
        auto alloc_t = is_host ? AllocationType::HOST : AllocationType::DEVICE;
        return tensorToBuffer(tensor, alloc_t);
    }
};

TEST_F(SamplerTest, testMemFill) {
    SamplerDataBuilder builder;

    std::vector<int> end_think_token_ids = {101, 102};
    std::vector<int> max_thinking_tokens = {3, 4, 5, 4};
    std::vector<int> think_status = {0, 0, 0, 0};
    BaseLogitsProcessorPtr processor = builder.generateLogitsProcessor(true, max_thinking_tokens, end_think_token_ids, think_status);
    
    SamplerInputs sampler_inputs = builder.allocate({4, 1024, 1024}, {processor});
    std::vector<int> sequence_lengths = {1, 2, 3, 4};
    builder.setSequenceLengths(sampler_inputs, sequence_lengths);
    EXPECT_EQ(buffer2vector<int>(*sampler_inputs.sequence_lengths), std::vector<int>({1, 2, 3, 4}));

    torch::Tensor tensor = torch::tensor({{2, 2, 2, 2, 2},
                                          {2, 2, 2, 2, 2},
                                          {2, 2, 2, 2, 2},
                                          {2, 2, 2, 2, 2}}, 
                                        torch::dtype(torch::kInt));
    auto logits = torchTensor2Buffer(tensor);
    processor->memFill(logits->index(0), 5, 0);
    processor->memFill(logits->index(1), 5, 1);
    processor->memFill(logits->index(2), 5, 2);
    processor->memFill(logits->index(3), 5, 3);

    EXPECT_EQ(buffer2vector<int>(*logits->index(0)), std::vector<int>({1, 0, 0, 0, 0}));
    EXPECT_EQ(buffer2vector<int>(*logits->index(1)), std::vector<int>({0, 1, 0, 0, 0}));
    EXPECT_EQ(buffer2vector<int>(*logits->index(2)), std::vector<int>({0, 0, 1, 0, 0}));
    EXPECT_EQ(buffer2vector<int>(*logits->index(3)), std::vector<int>({0, 0, 0, 1, 0}));


    torch::Tensor tensor2 = torch::tensor({{2, 2, 2, 2, 2},
                                          {2, 2, 2, 2, 2},
                                          {2, 2, 2, 2, 2},
                                          {2, 2, 2, 2, 2}}, 
                                        torch::dtype(torch::kDouble));
    auto logits2 = torchTensor2Buffer(tensor2);
    processor->memFill(logits2->index(0), 5, 0);
    processor->memFill(logits2->index(1), 5, 1);
    processor->memFill(logits2->index(2), 5, 2);
    processor->memFill(logits2->index(3), 5, 3);

    EXPECT_EQ(buffer2vector<double>(*logits2->index(0)), std::vector<double>({1, 0, 0, 0, 0}));
    EXPECT_EQ(buffer2vector<double>(*logits2->index(1)), std::vector<double>({0, 1, 0, 0, 0}));
    EXPECT_EQ(buffer2vector<double>(*logits2->index(2)), std::vector<double>({0, 0, 1, 0, 0}));
    EXPECT_EQ(buffer2vector<double>(*logits2->index(3)), std::vector<double>({0, 0, 0, 1, 0}));
}

TEST_F(SamplerTest, testUpdateStatus) {
    {
        SamplerDataBuilder builder;
        size_t batch_size = 4;
        size_t vocab_size = 10;
        size_t max_length = 10;
        std::vector<int> end_think_token_ids = {5};
        std::vector<int> max_thinking_tokens = {3, 3, 3, 3};
        std::vector<int> think_status = {0, 0, 0, 0};
        BaseLogitsProcessorPtr processor = builder.generateLogitsProcessor(true, max_thinking_tokens, end_think_token_ids, think_status);
        
        SamplerInputs sampler_inputs = builder.allocate({batch_size, vocab_size, max_length}, {processor});
        std::vector<int> sequence_lengths = {1, 2, 3, 4};
        builder.setSequenceLengths(sampler_inputs, sequence_lengths);
        EXPECT_EQ(buffer2vector<int>(*sampler_inputs.sequence_lengths), std::vector<int>({1, 2, 3, 4}));

        std::vector<std::vector<int>> token_ids = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9}
        };
        builder.setTokenIds(sampler_inputs, token_ids);
        
        processor->updateStatus(sampler_inputs);

        auto proc = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processor);
        std::vector<size_t> think_end_tokens_status = proc->thinkEndTokensStatus();
        EXPECT_EQ(0, think_end_tokens_status[0]);
        EXPECT_EQ(0, think_end_tokens_status[1]);
        EXPECT_EQ(1, think_end_tokens_status[2]);
        EXPECT_EQ(0, think_end_tokens_status[3]);
    }

    {
        SamplerDataBuilder builder;
        size_t batch_size = 4;
        size_t vocab_size = 10;
        size_t max_length = 10;
        std::vector<int> end_think_token_ids = {5, 5};
        std::vector<int> max_thinking_tokens = {3, 3, 3, 3};
        std::vector<int> think_status = {0, 0, 1, 1};
        BaseLogitsProcessorPtr processor = builder.generateLogitsProcessor(true, max_thinking_tokens, end_think_token_ids, think_status);
        
        SamplerInputs sampler_inputs = builder.allocate({batch_size, vocab_size, max_length}, {processor});
        std::vector<int> sequence_lengths = {1, 2, 3, 4};
        builder.setSequenceLengths(sampler_inputs, sequence_lengths);
        EXPECT_EQ(buffer2vector<int>(*sampler_inputs.sequence_lengths), std::vector<int>({1, 2, 3, 4}));

        std::vector<std::vector<int>> token_ids = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9}
        };
        builder.setTokenIds(sampler_inputs, token_ids);

        processor->updateStatus(sampler_inputs);

        auto proc = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processor);
        std::vector<size_t> think_end_tokens_status = proc->thinkEndTokensStatus();
        EXPECT_EQ(0, think_end_tokens_status[0]);
        EXPECT_EQ(0, think_end_tokens_status[1]);
        EXPECT_EQ(2, think_end_tokens_status[2]);
        EXPECT_EQ(0, think_end_tokens_status[3]);
    }

    {
        SamplerDataBuilder builder;
        size_t batch_size = 4;
        size_t vocab_size = 10;
        size_t max_length = 10;
        std::vector<int> end_think_token_ids = {5, 6};
        std::vector<int> max_thinking_tokens = {3, 3, 3, 3};
        std::vector<int> think_status = {0, 0, 1, 1};
        BaseLogitsProcessorPtr processor = builder.generateLogitsProcessor(true, max_thinking_tokens, end_think_token_ids, think_status);
        
        SamplerInputs sampler_inputs = builder.allocate({batch_size, vocab_size, max_length}, {processor});
        std::vector<int> sequence_lengths = {1, 2, 3, 4};
        builder.setSequenceLengths(sampler_inputs, sequence_lengths);
        EXPECT_EQ(buffer2vector<int>(*sampler_inputs.sequence_lengths), std::vector<int>({1, 2, 3, 4}));

        std::vector<std::vector<int>> token_ids = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6}
        };
        builder.setTokenIds(sampler_inputs, token_ids);

        processor->updateStatus(sampler_inputs);

        auto proc = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(processor);
        std::vector<size_t> think_end_tokens_status = proc->thinkEndTokensStatus();
        EXPECT_EQ(1, think_end_tokens_status[0]);
        EXPECT_EQ(0, think_end_tokens_status[1]);
        EXPECT_EQ(1, think_end_tokens_status[2]);
        EXPECT_EQ(2, think_end_tokens_status[3]);
    }
}


std::string tensorToString(const at::Tensor& tensor, size_t size) {
    std::ostringstream oss;

    if (tensor.dim() != 1 || tensor.size(0) != size) {
        return "Error: Tensor must be one-dimensional with size 10.";
    }

    oss << "Tensor values: [";

    for (size_t i = 0; i < tensor.size(0); ++i) {
        oss << tensor[i].item<int>();
        if (i < tensor.size(0) - 1) {
            oss << ", ";
        }
    }

    oss << "]";
    return oss.str();
}

TEST_F(SamplerTest, testSetVocabMask) {
    {
        SamplerDataBuilder builder;
        size_t batch_size = 4;
        size_t vocab_size = 10;
        size_t max_length = 10;
        std::vector<int> end_think_token_ids = {5, 6};
        std::vector<int> max_thinking_tokens = {3, 3, 3, 3};
        std::vector<int> think_status = {0, 0, 1, 1};
        BaseLogitsProcessorPtr processor = builder.generateLogitsProcessor(true, max_thinking_tokens, end_think_token_ids, think_status);
        
        SamplerInputs sampler_inputs = builder.allocate({batch_size, vocab_size, max_length}, {processor});
        std::vector<int> sequence_lengths = {1, 2, 3, 4};
        builder.setSequenceLengths(sampler_inputs, sequence_lengths);
        EXPECT_EQ(buffer2vector<int>(*sampler_inputs.sequence_lengths), std::vector<int>({1, 2, 3, 4}));

        EXPECT_EQ((size_t) 1, sampler_inputs.grammars.size());
        auto grammar = std::dynamic_pointer_cast<ThinkModeLogitsProcessor>(sampler_inputs.grammars[0]);
        
        for (size_t i = 0; i < batch_size; i++) {
            grammar->setVocabMask(grammar->think_infos_[i].think_end_status_dfa_ptr,
                sampler_inputs.logits->index(i), 1, end_think_token_ids,
                vocab_size, i % 2 == 0 ? true : false);
        }

        string expect_tensor_string_0 = "Tensor values: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]";
        string expect_tensor_string_1 = "Tensor values: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]";
        string expect_tensor_string_2 = "Tensor values: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]";
        EXPECT_EQ(expect_tensor_string_1, tensorToString(Buffer2torchTensor(*sampler_inputs.logits->index(0), false), 10));
        EXPECT_EQ(expect_tensor_string_0, tensorToString(Buffer2torchTensor(*sampler_inputs.logits->index(1), false), 10));
        EXPECT_EQ(expect_tensor_string_2, tensorToString(Buffer2torchTensor(*sampler_inputs.logits->index(2), false), 10));
        EXPECT_EQ(expect_tensor_string_0, tensorToString(Buffer2torchTensor(*sampler_inputs.logits->index(3), false), 10));
    }
}

}  // namespace rtp_llm
