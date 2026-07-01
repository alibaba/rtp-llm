#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "autil/legacy/json.h"
#include "rtp_llm/cpp/api_server/Exception.h"
#include "rtp_llm/cpp/api_server/ChatService.h"
#include "rtp_llm/cpp/api_server/test/mock/MockApiServerMetricReporter.h"
#include "rtp_llm/cpp/api_server/test/mock/MockChatRender.h"
#include "rtp_llm/cpp/api_server/test/mock/MockEngineBase.h"
#include "rtp_llm/cpp/api_server/test/mock/MockHttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/test/mock/MockOpenaiEndpoint.h"
#include "rtp_llm/cpp/api_server/test/mock/MockTokenizer.h"
#include "rtp_llm/cpp/api_server/test/mock/MockGenerateStream.h"

using namespace ::testing;
namespace rtp_llm {

class ChatServiceTest: public ::testing::Test {
public:
    ChatServiceTest()           = default;
    ~ChatServiceTest() override = default;

protected:
    void SetUp() override {
        data_ = std::vector<int>{1, 2, 3, 4, 5};

        mock_engine_ = std::make_shared<MockEngineBase>();
        auto engine  = std::dynamic_pointer_cast<EngineBase>(mock_engine_);

        auto request_counter = std::make_shared<autil::AtomicCounter>();

        mock_tokenizer_ = std::make_shared<MockTokenizer>();
        auto tokenizer  = std::dynamic_pointer_cast<Tokenizer>(mock_tokenizer_);

        mock_render_ = std::make_shared<MockChatRender>();
        auto render  = std::dynamic_pointer_cast<ChatRender>(mock_render_);

        mock_metric_reporter_ = std::make_shared<MockApiServerMetricReporter>();
        auto metric_reporter  = std::dynamic_pointer_cast<ApiServerMetricReporter>(mock_metric_reporter_);

        // ChatService 构造函数中会初始化 OpenaiEndpoint, OpenaiEndpoint 构造函数中会使用 tokenizer 和 render ,
        // 所以需要 mock
        MockWhenConstructOpenaiEndPoint();

        ModelConfig model_config;
        chat_service_ = std::make_shared<ChatService>(
            engine, nullptr, request_counter, tokenizer, render, model_config, metric_reporter);

        // mock OpenaiEndpoint 方便测试
        mock_openai_endpoint_           = std::make_shared<MockOpenaiEndpoint>(tokenizer, render, model_config);
        auto openai_endpoint            = std::dynamic_pointer_cast<OpenaiEndpoint>(mock_openai_endpoint_);
        chat_service_->openai_endpoint_ = openai_endpoint;

        mock_writer_ = std::make_unique<http_server::MockHttpResponseWriter>();
        auto writer  = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
        ASSERT_TRUE(writer != nullptr);
        writer_ = std::unique_ptr<http_server::HttpResponseWriter>(writer);
    }
    void TearDown() override {
        // 需要手动释放 unique_ptr 的所有权, 避免 double free
        writer_.release();
        mock_writer_.reset();
    }

    void MockWhenConstructOpenaiEndPoint() {
        ON_CALL(*mock_tokenizer_, isPreTrainedTokenizer).WillByDefault(Return(false));
        EXPECT_CALL(*mock_tokenizer_, isPreTrainedTokenizer).Times(AnyNumber());

        ON_CALL(*mock_render_, get_all_extra_stop_word_ids_list).WillByDefault(Return(std::vector<std::vector<int>>()));
        EXPECT_CALL(*mock_render_, get_all_extra_stop_word_ids_list).Times(AnyNumber());
    }

    std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>
    CreateHttpPacket(const std::string& body) {
        auto packet = new ::anet::HTTPPacket();
        packet->setBody(body.c_str(), body.size());
        return std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>(
            packet, [](::anet::HTTPPacket* packet) { packet->free(); });
    }

    std::shared_ptr<MockGenerateStream> CreateMockGenerateStream() {
        auto input             = std::make_shared<GenerateInput>();
        input->generate_config = std::make_shared<GenerateConfig>();

        input->input_ids = torch::tensor(std::vector<int32_t>(data_.begin(), data_.end()), torch::kInt32);

        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = data_.size();

        auto mock_stream = std::make_shared<MockGenerateStream>(input, model_config, runtime_config);
        return mock_stream;
    }

    torch::Tensor CreateOutputIdsTensor() {
        return torch::from_blob(data_.data(), {(int64_t)data_.size()}, torch::kInt32).clone();
    }

protected:
    std::shared_ptr<MockEngineBase>                      mock_engine_;
    std::unique_ptr<http_server::HttpResponseWriter>     writer_;
    std::shared_ptr<MockTokenizer>                       mock_tokenizer_;
    std::shared_ptr<MockChatRender>                      mock_render_;
    std::shared_ptr<MockApiServerMetricReporter>         mock_metric_reporter_;
    std::shared_ptr<ChatService>                         chat_service_;
    std::shared_ptr<MockOpenaiEndpoint>                  mock_openai_endpoint_;
    std::unique_ptr<http_server::MockHttpResponseWriter> mock_writer_;
    std::vector<int>                                     data_;
};

TEST_F(ChatServiceTest, ChatCompletions_ThrowException) {
    http_server::HttpRequest request;
    const std::string        body = R"del({
    "messages": [
        {
            "role": "user",
            "content": "who are you?"
        }
    ],
    "temperature": 52.1,
    "top_p": 0.1,
    "stream": true,
    "source": "test_source"
})del";
    request._request              = CreateHttpPacket(body);

    auto generate_config = std::make_shared<GenerateConfig>();
    EXPECT_CALL(*mock_openai_endpoint_, extract_generation_config)
        .WillRepeatedly(Invoke([generate_config](const ChatCompletionRequest& req) {
            EXPECT_EQ(req.messages.size(), 1);
            EXPECT_EQ(req.messages[0].role, "user");
            EXPECT_TRUE(std::holds_alternative<std::string>(req.messages[0].content));
            EXPECT_EQ(std::get<std::string>(req.messages[0].content), "who are you?");

            EXPECT_TRUE(req.temperature.has_value());
            EXPECT_NEAR(req.temperature.value(), 52.1, 1e-3);

            EXPECT_TRUE(req.stream.has_value());
            EXPECT_EQ(req.stream.value(), true);

            EXPECT_TRUE(req.top_p.has_value());
            EXPECT_NEAR(req.top_p.value(), 0.1, 1e-6);
            return generate_config;
        }));

    EXPECT_CALL(*mock_metric_reporter_, reportFTInputTokenLengthMetric).Times(1);
    EXPECT_CALL(*mock_metric_reporter_, reportFTNumBeansMetric).Times(1);

    RenderedInputs rendered_inputs{std::vector<int>(), std::vector<MultimodalInput>(), std::string()};
    EXPECT_CALL(*mock_render_, render_chat_request).WillOnce(Return(rendered_inputs));

    auto mock_stream = CreateMockGenerateStream();
    auto stream      = std::dynamic_pointer_cast<GenerateStream>(mock_stream);
    EXPECT_CALL(*mock_engine_, enqueue(Matcher<const std::shared_ptr<GenerateInput>&>(_))).WillOnce(Return(stream));

    auto mock_ctx = std::make_shared<MockRenderContext>();
    auto ctx      = std::dynamic_pointer_cast<RenderContext>(mock_ctx);
    EXPECT_CALL(*mock_render_, getRenderContext).WillOnce(Return(ctx));
    EXPECT_CALL(*mock_ctx, init).WillOnce(Return());

    // outputs.generate_outputs 为空, 模拟抛出异常的情况
    GenerateOutputs outputs;
    EXPECT_CALL(*mock_stream, nextOutput()).WillOnce(Return(ErrorResult<GenerateOutputs>(std::move(outputs))));

    // EXPECT_CALL(*mock_metric_reporter_, reportErrorQpsMetric)
    //     .WillOnce(Invoke([](const std::string& source, int error_code) {
    //         EXPECT_EQ(source, "test_source");
    //         EXPECT_EQ(error_code, HttpApiServerException::UNKNOWN_ERROR);
    //     }));

    try {
        chat_service_->chatCompletions(writer_, request, 10086);
    } catch (const std::runtime_error& e) {
        EXPECT_EQ(typeid(e), typeid(RTPException));
    }
}

TEST_F(ChatServiceTest, ChatCompletions) {
    http_server::HttpRequest request;
    const std::string        body = R"del({
    "messages": [
        {
            "role": "user",
            "content": "who are you?"
        }
    ],
    "temperature": 52.1,
    "top_p": 0.1,
    "stream": true,
    "source": "test_source"
})del";
    request._request              = CreateHttpPacket(body);

    auto generate_config = std::make_shared<GenerateConfig>();
    EXPECT_CALL(*mock_openai_endpoint_, extract_generation_config)
        .WillRepeatedly(Invoke([generate_config](const ChatCompletionRequest& req) {
            EXPECT_EQ(req.messages.size(), 1);
            EXPECT_EQ(req.messages[0].role, "user");
            EXPECT_TRUE(std::holds_alternative<std::string>(req.messages[0].content));
            EXPECT_EQ(std::get<std::string>(req.messages[0].content), "who are you?");

            EXPECT_TRUE(req.temperature.has_value());
            EXPECT_NEAR(req.temperature.value(), 52.1, 1e-3);

            EXPECT_TRUE(req.stream.has_value());
            EXPECT_EQ(req.stream.value(), true);

            EXPECT_TRUE(req.top_p.has_value());
            EXPECT_NEAR(req.top_p.value(), 0.1, 1e-6);
            return generate_config;
        }));

    EXPECT_CALL(*mock_metric_reporter_, reportFTInputTokenLengthMetric).Times(1);
    EXPECT_CALL(*mock_metric_reporter_, reportFTNumBeansMetric).Times(1);

    RenderedInputs rendered_inputs{std::vector<int>(), std::vector<MultimodalInput>(), std::string()};
    EXPECT_CALL(*mock_render_, render_chat_request).WillOnce(Return(rendered_inputs));

    auto mock_stream = CreateMockGenerateStream();
    auto stream      = std::dynamic_pointer_cast<GenerateStream>(mock_stream);
    EXPECT_CALL(*mock_engine_, enqueue(Matcher<const std::shared_ptr<GenerateInput>&>(_))).WillOnce(Return(stream));

    auto mock_ctx = std::make_shared<MockRenderContext>();
    auto ctx      = std::dynamic_pointer_cast<RenderContext>(mock_ctx);
    EXPECT_CALL(*mock_render_, getRenderContext).WillOnce(Return(ctx));
    EXPECT_CALL(*mock_ctx, init).WillOnce(Return());

    const std::string json_response = "this is a test json response";
    EXPECT_CALL(*mock_ctx, render_stream_response_first)
        .WillOnce(Invoke([generate_config, json_response](int n, std::string debug_info) {
            EXPECT_EQ(generate_config->num_return_sequences, n);
            return json_response;
        }));
    EXPECT_CALL(*mock_ctx, render_stream_response)
        .WillOnce(Invoke([generate_config, data = data_, json_response](const GenerateOutputs&                 resp,
                                                                        const std::shared_ptr<GenerateConfig>& config,
                                                                        bool is_streaming) {
            EXPECT_EQ(config, generate_config);
            EXPECT_EQ(is_streaming, true);
            auto             output_ids_tensor = resp.generate_outputs[0].output_ids.contiguous();
            std::vector<int> output_tokens_list(output_ids_tensor.data_ptr<int>(),
                                                output_ids_tensor.data_ptr<int>() + output_ids_tensor.numel());
            EXPECT_EQ(output_tokens_list.size(), data.size());
            for (int i = 0; i < output_tokens_list.size(); ++i) {
                EXPECT_EQ(output_tokens_list.at(i), data.at(i));
            }
            return json_response;
        }));
    EXPECT_CALL(*mock_ctx, render_stream_response_flush)
        .WillOnce(Invoke([generate_config, data = data_, json_response](const GenerateOutputs&                 resp,
                                                                        const std::shared_ptr<GenerateConfig>& config,
                                                                        bool is_streaming) {
            EXPECT_EQ(config, generate_config);
            EXPECT_EQ(is_streaming, true);
            auto             output_ids_tensor = resp.generate_outputs[0].output_ids.contiguous();
            std::vector<int> output_tokens_list(output_ids_tensor.data_ptr<int>(),
                                                output_ids_tensor.data_ptr<int>() + output_ids_tensor.numel());
            EXPECT_EQ(output_tokens_list.size(), data.size());
            for (int i = 0; i < output_tokens_list.size(); ++i) {
                EXPECT_EQ(output_tokens_list.at(i), data.at(i));
            }
            return json_response;
        }));
    EXPECT_CALL(*mock_ctx, render_stream_response_final)
        .WillOnce(Invoke([data = data_, json_response](const GenerateOutputs& resp) {
            auto             output_ids_tensor = resp.generate_outputs[0].output_ids.contiguous();
            std::vector<int> output_tokens_list(output_ids_tensor.data_ptr<int>(),
                                                output_ids_tensor.data_ptr<int>() + output_ids_tensor.numel());
            EXPECT_EQ(output_tokens_list.size(), data.size());
            for (int i = 0; i < output_tokens_list.size(); ++i) {
                EXPECT_EQ(output_tokens_list.at(i), data.at(i));
            }
            return json_response;
        }));
    EXPECT_CALL(*mock_writer_, Write).WillRepeatedly(Invoke([json_response](const std::string& data) {
        EXPECT_EQ(data, ChatService::sseResponse(json_response));
        return true;
    }));

    GenerateOutput output;
    output.output_ids = CreateOutputIdsTensor();
    GenerateOutputs outputs;
    outputs.generate_outputs.push_back(output);

    // nextOutput 第一次正常返回, 第二次返回错误
    EXPECT_CALL(*mock_stream, nextOutput())
        .WillOnce(Return(ErrorResult<GenerateOutputs>(std::move(outputs))))
        .WillOnce(Return(ErrorResult<GenerateOutputs>(ErrorCode::OUTPUT_QUEUE_IS_EMPTY, "output queue is empty")));

    EXPECT_CALL(*mock_metric_reporter_, reportResponseFirstTokenLatencyMs).WillOnce(Invoke([](double val) {
        EXPECT_TRUE(val >= 0);
    }));
    EXPECT_CALL(*mock_metric_reporter_, reportResponseIterateLatencyMs).WillOnce(Invoke([](double val) {
        EXPECT_TRUE(val >= 0);
    }));

    EXPECT_CALL(*mock_writer_, WriteDone).WillOnce(Return(true));

    EXPECT_CALL(*mock_metric_reporter_, reportSuccessQpsMetric).WillOnce(Invoke([](const std::string& source) {
        EXPECT_EQ(source, "test_source");
    }));
    EXPECT_CALL(*mock_metric_reporter_, reportResponseIterateCountMetric).WillOnce(Invoke([](int32_t val) {
        EXPECT_EQ(val, 1);
    }));
    EXPECT_CALL(*mock_metric_reporter_, reportFTIterateCountMetric).WillOnce(Invoke([](double val) {
        EXPECT_NEAR(val, 1, 1e-6);
    }));
    EXPECT_CALL(*mock_metric_reporter_, reportFTOutputTokenLengthMetric).WillOnce(Invoke([](double val) {
        EXPECT_TRUE(val >= 0);
    }));
    EXPECT_CALL(*mock_metric_reporter_, reportResponseLatencyMs).WillOnce(Invoke([](double val) {
        EXPECT_TRUE(val >= 0);
    }));

    chat_service_->chatCompletions(writer_, request, 10086);

    EXPECT_EQ(writer_->_type, http_server::HttpResponseWriter::WriteType::Stream);
    EXPECT_EQ(writer_->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_->_headers.at("Content-Type"), "text/event-stream");
}

TEST_F(ChatServiceTest, ChatCompletions_StreamErrorJsonEscaped) {
    http_server::HttpRequest request;
    const std::string        body = R"del({
    "messages": [
        {
            "role": "user",
            "content": "who are you?"
        }
    ],
    "stream": true,
    "source": "test_source"
})del";
    request._request              = CreateHttpPacket(body);

    auto generate_config = std::make_shared<GenerateConfig>();
    EXPECT_CALL(*mock_openai_endpoint_, extract_generation_config).WillRepeatedly(Return(generate_config));

    RenderedInputs rendered_inputs{std::vector<int>(), std::vector<MultimodalInput>(), std::string()};
    EXPECT_CALL(*mock_render_, render_chat_request).WillOnce(Return(rendered_inputs));

    auto mock_stream = CreateMockGenerateStream();
    mock_stream->reportError(ErrorCode::UNKNOWN_ERROR, "bad \"quote\"\nline");
    auto stream = std::dynamic_pointer_cast<GenerateStream>(mock_stream);
    EXPECT_CALL(*mock_engine_, enqueue(Matcher<const std::shared_ptr<GenerateInput>&>(_))).WillOnce(Return(stream));

    auto mock_ctx = std::make_shared<MockRenderContext>();
    auto ctx      = std::dynamic_pointer_cast<RenderContext>(mock_ctx);
    EXPECT_CALL(*mock_render_, getRenderContext).WillOnce(Return(ctx));
    EXPECT_CALL(*mock_ctx, init).WillOnce(Return());

    const std::string first_json = R"({"first":true})";
    const std::string chunk_json = R"({"chunk":"ok"})";
    EXPECT_CALL(*mock_ctx, render_stream_response_first).WillOnce(Return(first_json));
    EXPECT_CALL(*mock_ctx, render_stream_response).WillOnce(Return(chunk_json));

    GenerateOutput output;
    output.output_ids = CreateOutputIdsTensor();
    GenerateOutputs outputs;
    outputs.generate_outputs.push_back(output);

    EXPECT_CALL(*mock_stream, hasError())
        .WillOnce(Return(false))
        .WillOnce(Return(false))
        .WillOnce(Return(true))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(*mock_stream, getStatus()).WillOnce(Return(StreamState::RUNNING));
    EXPECT_CALL(*mock_stream, nextOutput()).WillOnce(Return(ErrorResult<GenerateOutputs>(std::move(outputs))));

    std::vector<std::string> writes;
    EXPECT_CALL(*mock_writer_, Write).WillRepeatedly(Invoke([&writes](const std::string& data) {
        writes.push_back(data);
        return true;
    }));
    EXPECT_CALL(*mock_writer_, WriteDone()).WillOnce(Return(true));
    EXPECT_CALL(*mock_metric_reporter_, reportErrorQpsMetric(Eq("test_source"), _)).Times(1);

    EXPECT_NO_THROW(chat_service_->chatCompletions(writer_, request, 10086));

    ASSERT_EQ(writes.size(), 3u);
    EXPECT_EQ(writes[0], "data: " + first_json + "\n\n");
    EXPECT_EQ(writes[1], "data: " + chunk_json + "\n\n");
    EXPECT_THAT(writes[2], StartsWith("data: "));

    const std::string error_json = writes[2].substr(6, writes[2].size() - 8);
    auto              json_map =
        autil::legacy::AnyCast<autil::legacy::json::JsonMap>(autil::legacy::json::ParseJson(error_json));
    auto              error_map =
        autil::legacy::AnyCast<autil::legacy::json::JsonMap>(json_map["error"]);
    EXPECT_EQ(autil::legacy::AnyCast<std::string>(error_map["message"]), "bad \"quote\"\nline");
    EXPECT_EQ(autil::legacy::AnyCast<std::string>(error_map["type"]), "stream_error");
}

}  // namespace rtp_llm
