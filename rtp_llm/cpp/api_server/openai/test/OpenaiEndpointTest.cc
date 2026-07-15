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
    auto openai_endpoint = std::make_shared<OpenaiEndpoint>(nullptr, nullptr, model_config);
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
    auto                      openai_endpoint = std::make_shared<OpenaiEndpoint>(tokenizer_, render_, model_config);

    ChatCompletionRequest req;
    req.stream      = false;
    req.temperature = 52.1;
    req.top_p       = 12.34;
    req.max_tokens  = 100;
    req.stop        = "hello world";
    req.seed        = 10;

    std::vector<std::string>      stop_words_list    = {std::get<std::string>(req.stop.value())};
    std::vector<std::vector<int>> tokenize_words_res = {{1, 2, 3}};
    EXPECT_CALL(*mock_render_, tokenize_words(stop_words_list)).WillOnce(Return(tokenize_words_res));

    auto config = openai_endpoint->extract_generation_config(req);
    EXPECT_TRUE(config != nullptr);
    EXPECT_EQ(req.stream.value(), false);
    EXPECT_EQ(config->is_streaming, true);  // always true
    EXPECT_NEAR(config->temperature, req.temperature.value(), 1e-6);
    EXPECT_NEAR(config->top_p, req.top_p.value(), 1e-6);
    EXPECT_EQ(config->max_new_tokens, req.max_tokens.value());
    EXPECT_EQ(config->stop_words_str, stop_words_list);
    EXPECT_EQ(config->stop_words_list, tokenize_words_res);
    EXPECT_EQ(config->random_seed, req.seed.value());
}

TEST_F(OpenaiEndpointTest, GetChatRender) {
    {
        ModelConfig model_config;
        auto openai_endpoint = std::make_shared<OpenaiEndpoint>(nullptr, nullptr, model_config);
        EXPECT_EQ(openai_endpoint->getChatRender(), nullptr);
    }
    {
        EXPECT_CALL(*mock_render_, get_all_extra_stop_word_ids_list).WillOnce(Return(std::vector<std::vector<int>>()));
        ModelConfig model_config;
        auto openai_endpoint = std::make_shared<OpenaiEndpoint>(nullptr, render_, model_config);
        EXPECT_EQ(openai_endpoint->getChatRender(), render_);
    }
}

// When chat_render_ is null but a tokenizer is available, request stop words must
// still be encoded (via the tokenizer) instead of being silently dropped.
TEST_F(OpenaiEndpointTest, ExtractGenerationConfig_NullRender_UsesTokenizerForStopWords) {
    EXPECT_CALL(*mock_tokenizer_, isPreTrainedTokenizer).WillOnce(Return(false));

    ModelConfig model_config;
    auto        openai_endpoint = std::make_shared<OpenaiEndpoint>(tokenizer_, nullptr, model_config);

    ChatCompletionRequest req;
    req.stop = "foo";

    std::vector<int> foo_ids = {1, 2, 3};
    EXPECT_CALL(*mock_tokenizer_, encode(std::string("foo"))).WillOnce(Return(foo_ids));

    auto config = openai_endpoint->extract_generation_config(req);
    ASSERT_TRUE(config != nullptr);
    // stop constraint preserved through the tokenizer fallback, not silently dropped.
    EXPECT_EQ(config->stop_words_str, std::vector<std::string>{"foo"});
    EXPECT_EQ(config->stop_words_list, std::vector<std::vector<int>>{foo_ids});
}

// Stop words requested but neither renderer nor tokenizer available: fail fast.
TEST_F(OpenaiEndpointTest, ExtractGenerationConfig_NullRenderAndTokenizer_StopWords_Throws) {
    ModelConfig model_config;
    auto        openai_endpoint = std::make_shared<OpenaiEndpoint>(nullptr, nullptr, model_config);

    ChatCompletionRequest req;
    req.stop = "foo";

    EXPECT_THROW(openai_endpoint->extract_generation_config(req), std::runtime_error);
}

// select_tokens_str requested but tokenizer is null: fail fast (no equivalent path).
TEST_F(OpenaiEndpointTest, ExtractGenerationConfig_NullTokenizer_SelectTokens_Throws) {
    ModelConfig model_config;
    auto        openai_endpoint = std::make_shared<OpenaiEndpoint>(nullptr, nullptr, model_config);

    GenerateConfig gc;
    gc.select_tokens_str = {"a"};
    ChatCompletionRequest req;
    req.extra_configs = gc;

    EXPECT_THROW(openai_endpoint->extract_generation_config(req), std::runtime_error);
}

// sp_advice_prompt requested but tokenizer is null: fail fast (no equivalent path).
TEST_F(OpenaiEndpointTest, ExtractGenerationConfig_NullTokenizer_SpAdvice_Throws) {
    ModelConfig model_config;
    auto        openai_endpoint = std::make_shared<OpenaiEndpoint>(nullptr, nullptr, model_config);

    GenerateConfig gc;
    gc.sp_advice_prompt = "hint";
    ChatCompletionRequest req;
    req.extra_configs = gc;

    EXPECT_THROW(openai_endpoint->extract_generation_config(req), std::runtime_error);
}

}  // namespace rtp_llm
