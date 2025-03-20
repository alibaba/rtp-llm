
#include "gtest/gtest.h"

#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/stream/CompleteTokenIds.h"
#include "maga_transformer/cpp/normal_engine/NormalGenerateStream.h"
#include "src/fastertransformer/devices/testing/TestBase.h"

using namespace std;

namespace rtp_llm {

typedef std::shared_ptr<CompleteTokenIds> CompleteTokenIdsPtr;

class CompleteTokenIdsBuilder {
public:

    CompleteTokenIdsBuilder(ft::GptInitParameter params) :
        params_(params), device_(ft::DeviceFactory::getDefaultDevice()),
        generate_input_(new GenerateInput()),
        generate_config_(new GenerateConfig()) {
        params_.max_seq_len_ = 2048;
    }

    CompleteTokenIdsPtr createCompleteTokenIds(std::vector<int> input_ids, int max_thinking_tokens) {
        generate_config_->in_think_mode = true;
        generate_config_->max_thinking_tokens = max_thinking_tokens;
        generate_config_->end_think_token_ids.push_back(101);
        generate_config_->end_think_token_ids.push_back(102);
        generate_config_->num_beams = 1;
        generate_config_->num_return_sequences = 1;
        generate_config_->max_new_tokens = 200;
        generate_input_->generate_config = generate_config_;
        generate_input_->input_ids = ft::vector2Buffer(input_ids);
        auto tileNum = std::max((int)generate_config_->num_beams, (int)generate_config_->num_return_sequences);

        auto complete_token_ids = std::make_shared<CompleteTokenIds>(device_, tileNum, params_.max_seq_len_, params_.seq_size_per_block_,
            generate_config_->in_think_mode, generate_config_->max_thinking_tokens, 0, generate_config_->end_think_token_ids);
        complete_token_ids->init(generate_input_);
        return complete_token_ids;
    };

private:
    ft::GptInitParameter params_;
    ft::DeviceBase* device_;

public:
    std::shared_ptr<GenerateInput> generate_input_;
    std::shared_ptr<GenerateConfig> generate_config_;
    ResourceContext resource_context_;
    
};

class CompleteTokenIdsTest: public DeviceTestBase {
protected:
};

TEST_F(CompleteTokenIdsTest, testUpdateWithMaxThinkingTokens) {
    ft::GptInitParameter params;
    params.vocab_size_ = 200;
    params.max_seq_len_ = 100;
    auto builder = CompleteTokenIdsBuilder(params);
    auto complete_token_ids = builder.createCompleteTokenIds({1, 2, 3, 4, 5}, 1);

    torch::Tensor tensor = torch::tensor({{6, 7, 8}}, torch::dtype(torch::kInt));
    auto new_tokens_ptr = torchTensor2Buffer(tensor);

    int64_t begin_time_us = 0;
    int num_new_tokens = 3;
    int max_token_num = std::min((int)params.max_seq_len_, (int)builder.generate_config_->max_new_tokens + builder.generate_input_->inputLength());
    int input_length = builder.generate_input_->inputLength();
    int num_beams = builder.generate_input_->generate_config->num_beams;
    int64_t stream_id = builder.generate_input_->request_id;
    int error_token_id = 0;
    ASSERT_EQ(complete_token_ids->seq_length_, 5);
    
    bool ret = complete_token_ids->update(new_tokens_ptr, begin_time_us, 
        num_new_tokens, input_length, max_token_num, params.vocab_size_, 
        num_beams, stream_id, error_token_id);
    ASSERT_EQ(ret, true);
    ASSERT_EQ(complete_token_ids->seq_length_, 8);

    std::vector<size_t> think_end_tokens_status = complete_token_ids->thinkEndTokensStatus();
    ASSERT_EQ((size_t)1, think_end_tokens_status.size());
    ASSERT_EQ(2, think_end_tokens_status[0]);

}


TEST_F(CompleteTokenIdsTest, testUpdateWithMaxThinkingTokensStepbyStep) {
    ft::GptInitParameter params;
    params.vocab_size_ = 200;
    params.max_seq_len_ = 100;
    auto builder = CompleteTokenIdsBuilder(params);
    auto complete_token_ids = builder.createCompleteTokenIds({1, 2, 3, 4, 5}, 1);
    int input_len = 5;

    torch::Tensor tensor = torch::tensor({{6}}, torch::dtype(torch::kInt));
    auto new_tokens_ptr = torchTensor2Buffer(tensor);

    int64_t begin_time_us = 0;
    int num_new_tokens = 1;
    int max_token_num = std::min((int)params.max_seq_len_, (int)builder.generate_config_->max_new_tokens + builder.generate_input_->inputLength());
    int num_beams = builder.generate_input_->generate_config->num_beams;
    int64_t stream_id = builder.generate_input_->request_id;
    int error_token_id = 0;
    ASSERT_EQ(complete_token_ids->seq_length_, 5);
    
    bool ret = complete_token_ids->update(new_tokens_ptr, begin_time_us, 
        num_new_tokens, input_len, max_token_num, params.vocab_size_, 
        num_beams, stream_id, error_token_id);
    ASSERT_EQ(ret, true);
    ASSERT_EQ(complete_token_ids->seq_length_, 6);
    printf("--%s\n", complete_token_ids->toString(0).c_str());
    std::vector<size_t> think_end_tokens_status = complete_token_ids->thinkEndTokensStatus();
    ASSERT_EQ((size_t)1, think_end_tokens_status.size());
    ASSERT_EQ(1, think_end_tokens_status[0]);



    torch::Tensor tensor2 = torch::tensor({{7}}, torch::dtype(torch::kInt));
    auto new_tokens_ptr2 = torchTensor2Buffer(tensor2);
    ret = complete_token_ids->update(new_tokens_ptr2, begin_time_us, 
        num_new_tokens, input_len, max_token_num, params.vocab_size_, 
        num_beams, stream_id, error_token_id);
    ASSERT_EQ(ret, true);
    ASSERT_EQ(complete_token_ids->seq_length_, 7);
    printf("--%s\n", complete_token_ids->toString(0).c_str());
    think_end_tokens_status = complete_token_ids->thinkEndTokensStatus();
    ASSERT_EQ((size_t)1, think_end_tokens_status.size());
    ASSERT_EQ(2, think_end_tokens_status[0]);

}


TEST_F(CompleteTokenIdsTest, testUpdateWithMaxThinkingTokensStepbyStepWithEarlyThinkEnd) {
    ft::GptInitParameter params;
    params.vocab_size_ = 200;
    params.max_seq_len_ = 100;
    auto builder = CompleteTokenIdsBuilder(params);
    auto complete_token_ids = builder.createCompleteTokenIds({1}, 5);
    int input_len = 1;

    torch::Tensor tensor = torch::tensor({{101}}, torch::dtype(torch::kInt));
    auto new_tokens_ptr = torchTensor2Buffer(tensor);

    int64_t begin_time_us = 0;
    int num_new_tokens = 1;
    int max_token_num = std::min((int)params.max_seq_len_, (int)builder.generate_config_->max_new_tokens + builder.generate_input_->inputLength());
    int num_beams = builder.generate_input_->generate_config->num_beams;
    int64_t stream_id = builder.generate_input_->request_id;
    int error_token_id = 0;
    ASSERT_EQ(complete_token_ids->seq_length_, 1);
    
    bool ret = complete_token_ids->update(new_tokens_ptr, begin_time_us, 
        num_new_tokens, input_len, max_token_num, params.vocab_size_, 
        num_beams, stream_id, error_token_id);
    ASSERT_EQ(ret, true);
    ASSERT_EQ(complete_token_ids->seq_length_, 2);
    printf("%s\n", complete_token_ids->toString(0).c_str());
    std::vector<size_t> think_end_tokens_status = complete_token_ids->thinkEndTokensStatus();
    ASSERT_EQ((size_t)1, think_end_tokens_status.size());
    ASSERT_EQ(1, think_end_tokens_status[0]);



    torch::Tensor tensor2 = torch::tensor({{2}}, torch::dtype(torch::kInt));
    auto new_tokens_ptr2 = torchTensor2Buffer(tensor2);
    ret = complete_token_ids->update(new_tokens_ptr2, begin_time_us, 
        num_new_tokens, input_len, max_token_num, params.vocab_size_, 
        num_beams, stream_id, error_token_id);
    ASSERT_EQ(ret, true);
    ASSERT_EQ(complete_token_ids->seq_length_, 3);
    printf("%s\n", complete_token_ids->toString(0).c_str());
    think_end_tokens_status = complete_token_ids->thinkEndTokensStatus();
    ASSERT_EQ((size_t)1, think_end_tokens_status.size());
    ASSERT_EQ(0, think_end_tokens_status[0]);



    torch::Tensor tensor3 = torch::tensor({{101}}, torch::dtype(torch::kInt));
    auto new_tokens_ptr3 = torchTensor2Buffer(tensor3);
    ret = complete_token_ids->update(new_tokens_ptr3, begin_time_us, 
        num_new_tokens, input_len, max_token_num, params.vocab_size_, 
        num_beams, stream_id, error_token_id);
    ASSERT_EQ(ret, true);
    ASSERT_EQ(complete_token_ids->seq_length_, 4);
    printf("%s\n", complete_token_ids->toString(0).c_str());
    think_end_tokens_status = complete_token_ids->thinkEndTokensStatus();
    ASSERT_EQ((size_t)1, think_end_tokens_status.size());
    ASSERT_EQ(1, think_end_tokens_status[0]);



    torch::Tensor tensor4 = torch::tensor({{102}}, torch::dtype(torch::kInt));
    auto new_tokens_ptr4 = torchTensor2Buffer(tensor4);
    ret = complete_token_ids->update(new_tokens_ptr4, begin_time_us, 
        num_new_tokens, input_len, max_token_num, params.vocab_size_, 
        num_beams, stream_id, error_token_id);
    ASSERT_EQ(ret, true);
    ASSERT_EQ(complete_token_ids->seq_length_, 5);
    printf("%s\n", complete_token_ids->toString(0).c_str());
    think_end_tokens_status = complete_token_ids->thinkEndTokensStatus();
    ASSERT_EQ((size_t)1, think_end_tokens_status.size());
    ASSERT_EQ(2, think_end_tokens_status[0]);


    torch::Tensor tensor5 = torch::tensor({{6}}, torch::dtype(torch::kInt));
    auto new_tokens_ptr5 = torchTensor2Buffer(tensor5);
    ret = complete_token_ids->update(new_tokens_ptr5, begin_time_us, 
        num_new_tokens, input_len, max_token_num, params.vocab_size_, 
        num_beams, stream_id, error_token_id);
    ASSERT_EQ(ret, true);
    ASSERT_EQ(complete_token_ids->seq_length_, 6);
    printf("%s\n", complete_token_ids->toString(0).c_str());
    think_end_tokens_status = complete_token_ids->thinkEndTokensStatus();
    ASSERT_EQ((size_t)1, think_end_tokens_status.size());
    ASSERT_EQ(2, think_end_tokens_status[0]);

}

}  // namespace rtp_llm
