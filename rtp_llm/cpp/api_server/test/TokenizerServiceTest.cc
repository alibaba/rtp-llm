#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/ErrorResponse.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/api_server/TokenizerService.h"
#include "rtp_llm/cpp/api_server/test/mock/MockEngineBase.h"
#include "rtp_llm/cpp/api_server/test/mock/MockHttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/test/mock/MockTokenProcessor.h"

using namespace ::testing;
namespace rtp_llm {

class TokenizerServiceTest: public ::testing::Test {
public:
    TokenizerServiceTest()           = default;
    ~TokenizerServiceTest() override = default;

protected:
    void SetUp() override {
        mock_writer_          = std::make_unique<http_server::MockHttpResponseWriter>();
        mock_token_processor_ = std::make_shared<MockTokenProcessor>();
        auto token_processor  = std::dynamic_pointer_cast<TokenProcessor>(mock_token_processor_);
        tokenizer_service_    = std::make_shared<TokenizerService>(token_processor);
    }
    void TearDown() override {}

    std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>
    CreateHttpPacket(const std::string& body) {
        auto packet = new ::anet::HTTPPacket();
        packet->setBody(body.c_str(), body.size());
        return std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>(
            packet, [](::anet::HTTPPacket* packet) { packet->free(); });
    }

    void SetToMaster() {
        auto& parallel_info = ParallelInfo::globalParallelInfo();
        parallel_info.setTpSize(1);
        parallel_info.setPpSize(1);
        parallel_info.setWorldRank(0);
        parallel_info.setWorldSize(1);
        parallel_info.setLocalWorldSize(1);
        parallel_info.setDpSize(1);
        parallel_info.setEpSize(1);
    }
    void SetToWorker() {
        auto& parallel_info = ParallelInfo::globalParallelInfo();
        parallel_info.setTpSize(1);
        parallel_info.setPpSize(2);
        parallel_info.setWorldRank(1);
        parallel_info.setWorldSize(2);
        parallel_info.setLocalWorldSize(1);
        parallel_info.setDpSize(1);
        parallel_info.setEpSize(1);
    }

protected:
    std::unique_ptr<http_server::MockHttpResponseWriter> mock_writer_;
    std::shared_ptr<MockTokenProcessor>                  mock_token_processor_;
    std::shared_ptr<TokenizerService>                    tokenizer_service_;
};

TEST_F(TokenizerServiceTest, Constructor) {
    EXPECT_NE(tokenizer_service_, nullptr);
    EXPECT_NE(tokenizer_service_->token_processor_, nullptr);
}

TEST_F(TokenizerServiceTest, TokenizerEncode_IsNotMaster) {
    // 模拟不是 master 的情况
    SetToWorker();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_FALSE(parallel_info.isMaster());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        ErrorResponse error_response;
        autil::legacy::FromJsonString(error_response, data);
        EXPECT_EQ(error_response.error_code, 515);
        return true;
    }));

    tokenizer_service_->tokenizerEncode(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(TokenizerServiceTest, TokenizerEncode_ParseRequestBodyFailed) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isMaster());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    // request body 为空, 模拟解析 request body 失败的情况
    http_server::HttpRequest request;

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        ErrorResponse error_response;
        autil::legacy::FromJsonString(error_response, data);
        EXPECT_EQ(error_response.error_code, 514);
        return true;
    }));

    tokenizer_service_->tokenizerEncode(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, 500);
    EXPECT_EQ(writer_ptr->_statusMessage, "Internal Server Error");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(TokenizerServiceTest, TokenizerEncode_RequestBodyHasNoPrompt) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isMaster());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    // 模拟 request body 中没有 prompt 字段的情况
    http_server::HttpRequest request;
    const std::string        body = R"del({
    "no_prompt": "hello, what is your age",
    "generate_config": {
        "max_new_tokens": 20
    }
})del";
    request._request              = CreateHttpPacket(body);

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        ErrorResponse error_response;
        autil::legacy::FromJsonString(error_response, data);
        EXPECT_EQ(error_response.error_code, 514);
        return true;
    }));

    tokenizer_service_->tokenizerEncode(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, 500);
    EXPECT_EQ(writer_ptr->_statusMessage, "Internal Server Error");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(TokenizerServiceTest, TokenizerEncode_HasOffsetMappingButTokenizerResponseIsNull) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isMaster());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    const std::string        body = R"del({
    "prompt": "hello, what is your age",
    "generate_config": {
        "max_new_tokens": 20
    },
    "return_offsets_mapping": true
})del";
    request._request              = CreateHttpPacket(body);

    // 模拟 token_processor tokenizer 失败
    EXPECT_CALL(*mock_token_processor_, tokenizer("hello, what is your age")).WillOnce(Return(nullptr));

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        ErrorResponse error_response;
        autil::legacy::FromJsonString(error_response, data);
        EXPECT_EQ(error_response.error_code, 514);
        return true;
    }));

    tokenizer_service_->tokenizerEncode(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, 500);
    EXPECT_EQ(writer_ptr->_statusMessage, "Internal Server Error");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(TokenizerServiceTest, TokenizerEncode_HasOffsetMapping) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isMaster());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    const std::string        body = R"del({
    "prompt": "hello, what is your age",
    "generate_config": {
        "max_new_tokens": 20
    },
    "return_offsets_mapping": true
})del";
    request._request              = CreateHttpPacket(body);

    // 模拟 token_processor tokenizer 的返回值
    auto tokenizer_response            = std::make_shared<TokenizerEncodeResponse>();
    tokenizer_response->offset_mapping = {{1, 2, 3}, {4, 5, 6}};
    tokenizer_response->tokens         = {"hello", "what", "is", "your", "age"};
    tokenizer_response->token_ids      = {1, 2, 3, 4, 5};
    tokenizer_response->error          = "none";
    EXPECT_CALL(*mock_token_processor_, tokenizer("hello, what is your age")).WillOnce(Return(tokenizer_response));

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([tokenizer_response](const std::string& data) {
        TokenizerEncodeResponse response;
        autil::legacy::FromJsonString(response, data);
        EXPECT_EQ(response.offset_mapping, tokenizer_response->offset_mapping);
        EXPECT_EQ(response.tokens, tokenizer_response->tokens);
        EXPECT_EQ(response.token_ids, tokenizer_response->token_ids);
        EXPECT_EQ(response.error, tokenizer_response->error);
        return true;
    }));

    tokenizer_service_->tokenizerEncode(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, 200);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(TokenizerServiceTest, TokenizerEncode_WithoutOffsetMapping) {
    // 模拟是 master 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_TRUE(parallel_info.isMaster());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    const std::string        body = R"del({
    "prompt": "hello, what is your age",
    "generate_config": {
        "max_new_tokens": 20
    }
})del";
    request._request              = CreateHttpPacket(body);

    // 没有 return_offsets_mapping 字段所以不会调用 token_processor tokenizer
    EXPECT_CALL(*mock_token_processor_, tokenizer(_)).Times(0);

    std::vector<int>         token_ids = {1, 2, 3, 4, 5};
    std::vector<std::string> tokens    = {"hello", "what", "is", "your", "age"};
    EXPECT_EQ(token_ids.size(), tokens.size());
    EXPECT_CALL(*mock_token_processor_, encode("hello, what is your age")).WillOnce(Return(token_ids));
    for (int i = 0; i < token_ids.size(); ++i) {
        auto id = token_ids[i];
        EXPECT_CALL(*mock_token_processor_, decode(std::vector<int>{id})).WillOnce(Return(tokens[i]));
    }

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([tokens, token_ids](const std::string& data) {
        TokenizerEncodeResponse response;
        autil::legacy::FromJsonString(response, data);
        EXPECT_EQ(response.tokens, tokens);
        EXPECT_EQ(response.token_ids, token_ids);
        EXPECT_TRUE(response.offset_mapping.empty());
        EXPECT_TRUE(response.error.empty());
        return true;
    }));

    tokenizer_service_->tokenizerEncode(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, 200);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(TokenizerServiceTest, TokenizerEncodeRequest) {
    TokenizerEncodeRequest req;
    req.prompt                 = "hello";
    req.return_offsets_mapping = true;

    auto json_str = autil::legacy::ToJsonString(req);
    EXPECT_EQ(json_str.empty(), false);

    TokenizerEncodeRequest req2;
    EXPECT_NO_THROW(autil::legacy::FromJsonString(req2, json_str));
    EXPECT_EQ(req2.prompt, req.prompt);
    EXPECT_TRUE(req2.return_offsets_mapping);
    EXPECT_TRUE(req.return_offsets_mapping);
}

}  // namespace rtp_llm
