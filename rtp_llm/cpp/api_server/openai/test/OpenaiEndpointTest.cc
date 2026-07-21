#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/test/mock/MockTokenizer.h"
#include "rtp_llm/cpp/api_server/test/mock/MockChatRender.h"
#include "rtp_llm/cpp/api_server/openai/OpenaiEndpoint.h"

using namespace ::testing;
namespace rtp_llm {

class OpenaiEndpointTest: public ::testing::Test {
public:
    OpenaiEndpointTest()           = default;
    ~OpenaiEndpointTest() override = default;

protected:
    void SetUp() override {
        mock_tokenizer_ = std::make_shared<MockTokenizer>();
        tokenizer_      = std::dynamic_pointer_cast<Tokenizer>(mock_tokenizer_);

        mock_render_ = std::make_shared<MockChatRender>();
        render_      = std::dynamic_pointer_cast<ChatRender>(mock_render_);
    }
    void TearDown() override {}

protected:
    std::shared_ptr<MockTokenizer> mock_tokenizer_;
    std::shared_ptr<Tokenizer>     tokenizer_;

    std::shared_ptr<MockChatRender> mock_render_;
    std::shared_ptr<ChatRender>     render_;
};

TEST_F(OpenaiEndpointTest, Constructor_TokenizerIsNull) {
    ModelConfig model_config;
    auto        openai_endpoint = std::make_shared<OpenaiEndpoint>(nullptr, nullptr, model_config);
    EXPECT_TRUE(openai_endpoint->stop_word_ids_list_.empty());
    EXPECT_TRUE(openai_endpoint->stop_words_list_.empty());
}

TEST_F(OpenaiEndpointTest, Constructor_IsPreTrainedTokenizer) {
    ModelConfig model_config;
    model_config.special_tokens.eos_token_id = 5;

    EXPECT_CALL(*mock_tokenizer_, isPreTrainedTokenizer).WillOnce(Return(true));
    EXPECT_CALL(*mock_tokenizer_, getEosTokenId).WillOnce(Return(10));

    auto openai_endpoint = std::make_shared<OpenaiEndpoint>(tokenizer_, nullptr, model_config);
    EXPECT_EQ(openai_endpoint->eos_token_id_, 10);
    EXPECT_TRUE(openai_endpoint->stop_word_ids_list_.empty());
    EXPECT_TRUE(openai_endpoint->stop_words_list_.empty());
}

TEST_F(OpenaiEndpointTest, Constructor_IsNotPreTrainedTokenizer) {
    ModelConfig model_config;
    model_config.special_tokens.eos_token_id = 5;

    EXPECT_CALL(*mock_tokenizer_, isPreTrainedTokenizer).WillOnce(Return(false));
    EXPECT_CALL(*mock_tokenizer_, getEosTokenId).Times(0);

    auto openai_endpoint = std::make_shared<OpenaiEndpoint>(tokenizer_, nullptr, model_config);
    EXPECT_EQ(openai_endpoint->eos_token_id_, 5);
    EXPECT_TRUE(openai_endpoint->stop_word_ids_list_.empty());
    EXPECT_TRUE(openai_endpoint->stop_words_list_.empty());
}

TEST_F(OpenaiEndpointTest, Constructor_RenderIsNotNull) {
    ModelConfig model_config;
    model_config.special_tokens.eos_token_id       = 5;
    model_config.special_tokens.stop_words_id_list = {{1, 2, 3}, {4, 5, 6}};

    EXPECT_CALL(*mock_tokenizer_, isPreTrainedTokenizer).WillOnce(Return(true));
    EXPECT_CALL(*mock_tokenizer_, getEosTokenId).WillOnce(Return(10));

    std::vector<std::vector<int>> render_ids_list = {{7, 8}, {9, 10}};
    EXPECT_CALL(*mock_render_, get_all_extra_stop_word_ids_list).WillOnce(Return(render_ids_list));

    std::vector<std::vector<int>> ids_list;
    {
        for (const auto& list : model_config.special_tokens.stop_words_id_list) {
            std::vector<int> temp(list.size());
            std::transform(list.begin(), list.end(), temp.begin(), [](int64_t val) { return static_cast<int>(val); });
            ids_list.push_back(temp);
        }
        ids_list.insert(ids_list.begin(), render_ids_list.begin(), render_ids_list.end());
    }
    std::vector<std::string> word_list;
    for (int i = 0; i < ids_list.size(); ++i) {
        std::string word = "test" + std::to_string(i);
        EXPECT_CALL(*mock_tokenizer_, decode(ids_list.at(i))).WillOnce(Return(word));
        word_list.push_back(word);
    }

    auto openai_endpoint = std::make_shared<OpenaiEndpoint>(tokenizer_, render_, model_config);
    EXPECT_EQ(openai_endpoint->eos_token_id_, 10);
    EXPECT_EQ(openai_endpoint->stop_word_ids_list_, ids_list);
    EXPECT_EQ(openai_endpoint->stop_words_list_, word_list);
}

TEST_F(OpenaiEndpointTest, ExtractGenerationConfig) {
    EXPECT_CALL(*mock_tokenizer_, isPreTrainedTokenizer).WillOnce(Return(false));
    EXPECT_CALL(*mock_render_, get_all_extra_stop_word_ids_list).WillOnce(Return(std::vector<std::vector<int>>()));

    ModelConfig model_config;
    auto        openai_endpoint = std::make_shared<OpenaiEndpoint>(tokenizer_, render_, model_config);

    ChatCompletionRequest req;
    req.stream       = false;
    req.temperature  = 52.1;
    req.top_p        = 12.34;
    req.top_k        = 1;
    req.max_tokens   = 100;
    req.stop         = "hello world";
    req.seed         = 10;
    req.logprobs     = true;
    req.top_logprobs = 5;

    std::vector<std::string>      stop_words_list    = {std::get<std::string>(req.stop.value())};
    std::vector<std::vector<int>> tokenize_words_res = {{1, 2, 3}};
    EXPECT_CALL(*mock_render_, tokenize_words(stop_words_list)).WillOnce(Return(tokenize_words_res));

    auto config = openai_endpoint->extract_generation_config(req);
    EXPECT_TRUE(config != nullptr);
    EXPECT_EQ(req.stream.value(), false);
    EXPECT_EQ(config->is_streaming, true);  // always true
    EXPECT_NEAR(config->temperature, req.temperature.value(), 1e-6);
    EXPECT_NEAR(config->top_p, req.top_p.value(), 1e-6);
    EXPECT_EQ(config->top_k, req.top_k.value());
    EXPECT_EQ(config->max_new_tokens, req.max_tokens.value());
    EXPECT_EQ(config->stop_words_str, stop_words_list);
    EXPECT_EQ(config->stop_words_list, tokenize_words_res);
    EXPECT_EQ(config->random_seed, req.seed.value());
    EXPECT_TRUE(config->return_logprobs);
    EXPECT_EQ(config->top_logprobs, 5);
    EXPECT_FALSE(config->return_all_probs);

    ChatCompletionRequest effective_extra_logprobs;
    GenerateConfig        effective_extra_config;
    effective_extra_config.return_logprobs = true;
    effective_extra_config.top_logprobs    = 2;
    effective_extra_logprobs.extra_configs = effective_extra_config;
    effective_extra_logprobs.top_logprobs  = 4;
    EXPECT_CALL(*mock_render_, tokenize_words(std::vector<std::string>()))
        .WillOnce(Return(std::vector<std::vector<int>>()));
    auto effective_config = openai_endpoint->extract_generation_config(effective_extra_logprobs);
    EXPECT_TRUE(effective_config->return_logprobs);
    EXPECT_EQ(effective_config->top_logprobs, 4);

    ChatCompletionRequest mixed_top_logprobs;
    GenerateConfig        mixed_extra_config;
    mixed_extra_config.top_logprobs  = 5;
    mixed_top_logprobs.logprobs      = true;
    mixed_top_logprobs.extra_configs = mixed_extra_config;
    EXPECT_CALL(*mock_render_, tokenize_words(std::vector<std::string>()))
        .WillOnce(Return(std::vector<std::vector<int>>()));
    auto mixed_config = openai_endpoint->extract_generation_config(mixed_top_logprobs);
    EXPECT_TRUE(mixed_config->return_logprobs);
    EXPECT_EQ(mixed_config->top_logprobs, 5);

    ChatCompletionRequest top_without_logprobs;
    top_without_logprobs.top_logprobs = 1;
    EXPECT_THROW(openai_endpoint->extract_generation_config(top_without_logprobs), std::exception);

    ChatCompletionRequest zero_top_without_logprobs;
    zero_top_without_logprobs.top_logprobs = 0;
    EXPECT_THROW(openai_endpoint->extract_generation_config(zero_top_without_logprobs), std::exception);

    ChatCompletionRequest invalid_top_logprobs;
    invalid_top_logprobs.logprobs     = true;
    invalid_top_logprobs.top_logprobs = 21;
    EXPECT_THROW(openai_endpoint->extract_generation_config(invalid_top_logprobs), std::exception);

    ChatCompletionRequest invalid_extra_top_logprobs;
    GenerateConfig        invalid_extra_config;
    invalid_extra_config.return_logprobs     = true;
    invalid_extra_config.top_logprobs        = 21;
    invalid_extra_top_logprobs.extra_configs = invalid_extra_config;
    EXPECT_THROW(openai_endpoint->extract_generation_config(invalid_extra_top_logprobs), std::exception);

    ChatCompletionRequest invalid_extra_n;
    GenerateConfig        invalid_extra_n_config;
    invalid_extra_n_config.return_logprobs      = true;
    invalid_extra_n_config.num_return_sequences = 2;
    invalid_extra_n.extra_configs               = invalid_extra_n_config;
    EXPECT_THROW(openai_endpoint->extract_generation_config(invalid_extra_n), std::exception);

    ChatCompletionRequest extra_top_without_logprobs;
    GenerateConfig        extra_top_without_logprobs_config;
    extra_top_without_logprobs_config.top_logprobs = 1;
    extra_top_without_logprobs.extra_configs       = extra_top_without_logprobs_config;
    EXPECT_THROW(openai_endpoint->extract_generation_config(extra_top_without_logprobs), std::exception);

    ChatCompletionRequest thinking_logprobs;
    GenerateConfig        thinking_logprobs_config;
    thinking_logprobs_config.return_logprobs = true;
    thinking_logprobs_config.in_think_mode   = true;
    thinking_logprobs.extra_configs          = thinking_logprobs_config;
    EXPECT_THROW(openai_endpoint->extract_generation_config(thinking_logprobs), std::exception);
}

TEST_F(OpenaiEndpointTest, GetChatRender) {
    {
        ModelConfig model_config;
        auto        openai_endpoint = std::make_shared<OpenaiEndpoint>(nullptr, nullptr, model_config);
        EXPECT_EQ(openai_endpoint->getChatRender(), nullptr);
    }
    {
        EXPECT_CALL(*mock_render_, get_all_extra_stop_word_ids_list).WillOnce(Return(std::vector<std::vector<int>>()));
        ModelConfig model_config;
        auto        openai_endpoint = std::make_shared<OpenaiEndpoint>(nullptr, render_, model_config);
        EXPECT_EQ(openai_endpoint->getChatRender(), render_);
    }
}

}  // namespace rtp_llm
