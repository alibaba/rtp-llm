#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/Exception.h"
#include "rtp_llm/cpp/api_server/ErrorResponse.h"
#include "rtp_llm/cpp/api_server/InferenceService.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/api_server/test/mock/MockApiServerMetricReporter.h"
#include "rtp_llm/cpp/api_server/test/mock/MockEngineBase.h"
#include "rtp_llm/cpp/api_server/test/mock/MockGenerateStream.h"
#include "rtp_llm/cpp/api_server/test/mock/MockGenerateStreamWrapper.h"
#include "rtp_llm/cpp/api_server/test/mock/MockHttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/test/mock/MockTokenProcessor.h"

using namespace ::testing;
namespace rtp_llm {

class InferenceServiceTest: public ::testing::Test {
public:
    InferenceServiceTest()           = default;
    ~InferenceServiceTest() override = default;

protected:
    void SetUp() override {
        mock_writer_ = std::make_unique<http_server::MockHttpResponseWriter>();

        mock_engine_ = std::make_shared<MockEngineBase>();
        auto engine  = std::dynamic_pointer_cast<EngineBase>(mock_engine_);

        mock_token_processor_ = std::make_shared<MockTokenProcessor>();
        auto token_processor  = std::dynamic_pointer_cast<TokenProcessor>(mock_token_processor_);

        mock_metric_reporter_ = std::make_shared<MockApiServerMetricReporter>();
        auto metric_reporter  = std::dynamic_pointer_cast<ApiServerMetricReporter>(mock_metric_reporter_);

        auto                      request_counter = std::make_shared<autil::AtomicCounter>();
        auto                      controller      = std::make_shared<ConcurrencyController>(1, false);

        inference_service_ = std::make_shared<InferenceService>(
            engine, nullptr, request_counter, token_processor, controller, model_config_, metric_reporter);
    }
    void TearDown() override {}

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

        data_                     = std::vector<int>{1, 2, 3, 4, 5};
        std::vector<size_t> shape = {data_.size()};
        // 由于 Buffer 内部不负责管理传入的地址数据(只是使用), 所以数据必须具有较久的生命周期
        input->input_ids = std::make_shared<rtp_llm::Buffer>(
            rtp_llm::MemoryType::MEMORY_CPU, rtp_llm::DataType::TYPE_INT32, shape, data_.data());

        ModelConfig model_config;
        RuntimeConfig runtime_config;
        model_config.max_seq_len = data_.size();

        auto mock_stream = std::make_shared<MockGenerateStream>(input, model_config, runtime_config);
        return mock_stream;
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
    std::shared_ptr<MockEngineBase>                      mock_engine_;
    std::shared_ptr<MockTokenProcessor>                  mock_token_processor_;
    std::shared_ptr<MockApiServerMetricReporter>         mock_metric_reporter_;
    std::shared_ptr<InferenceService>                    inference_service_;
    std::vector<int>                                     data_;
    ModelConfig                                          model_config_;
};

TEST_F(InferenceServiceTest, Inference_IsInternal_IsNotWorker) {
    // 模拟不是 worker 的情况
    SetToMaster();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_FALSE(parallel_info.isWorker());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    EXPECT_CALL(*mock_writer_, isConnected()).WillOnce(Return(true));
    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        ErrorResponse error_response;
        autil::legacy::FromJsonString(error_response, data);
        EXPECT_EQ(error_response.error_code, HttpApiServerException::UNSUPPORTED_OPERATION);
        return true;
    }));

    inference_service_->inference(writer_ptr, request, true);

    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, HttpApiServerException::UNSUPPORTED_OPERATION);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(InferenceServiceTest, Inference_IsNotInternal_IsNotMaster) {
    // 模拟不是 master 的情况
    SetToWorker();
    auto& parallel_info = ParallelInfo::globalParallelInfo();
    EXPECT_FALSE(parallel_info.isMaster());

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    EXPECT_CALL(*mock_writer_, isConnected()).WillOnce(Return(true));
    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        ErrorResponse error_response;
        autil::legacy::FromJsonString(error_response, data);
        EXPECT_EQ(error_response.error_code, HttpApiServerException::UNSUPPORTED_OPERATION);
        return true;
    }));

    inference_service_->inference(writer_ptr, request, false);

    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, HttpApiServerException::UNSUPPORTED_OPERATION);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(InferenceServiceTest, InferResponseFailed_ControllerIsNull) {
    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    // 模拟 controller 为空
    inference_service_->controller_ = nullptr;
    try {
        inference_service_->inferResponse(10086, writer_ptr, request);
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::UNKNOWN_ERROR);
        EXPECT_EQ(e.getMessage(), "infer response failed, concurrency controller is null");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::UNKNOWN_ERROR instead of std::exception";
    }

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(InferenceServiceTest, InferResponseFailed_ControllerIncrementFailed) {
    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    const std::string        body = R"({"prompt": "hello"})";
    request._request              = CreateHttpPacket(body);

    EXPECT_CALL(*mock_metric_reporter_, reportQpsMetric(StrEq("unknown")));
    EXPECT_CALL(*mock_metric_reporter_, reportConflictQpsMetric());

    // 将 max_concurrency_ 置为 0, 模拟 controller increment 失败
    auto saved                                        = inference_service_->controller_->get_available_concurrency();
    inference_service_->controller_->max_concurrency_ = 0;
    try {
        inference_service_->inferResponse(10086, writer_ptr, request);
        FAIL() << "should throw HttpApiServerException::CONCURRENCY_LIMIT_ERROR";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::CONCURRENCY_LIMIT_ERROR);
        EXPECT_EQ(e.getMessage(), "Too Many Requests");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::CONCURRENCY_LIMIT_ERROR instead of std::exception";
    }
    inference_service_->controller_->max_concurrency_ = saved;

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(InferenceServiceTest, InferResponseFailed_ParseRequestBodyFailed) {
    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    // request body 为空, 在解析时会抛异常
    http_server::HttpRequest request;
    EXPECT_THROW(inference_service_->inferResponse(10086, writer_ptr, request), std::exception);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(InferenceServiceTest, InferResponseFailed_NoPromptInRequest) {
    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    // 模拟请求中没有 prompt 字段
    http_server::HttpRequest request;
    const std::string        body = R"del({
    "no_prompt": "hello, what is your age",
    "generate_config": {
        "max_new_tokens": 20
    },
    "source": "test_source"
    })del";
    request._request              = CreateHttpPacket(body);

    try {
        inference_service_->inferResponse(10086, writer_ptr, request);
        FAIL() << "should throw HttpApiServerException::NO_PROMPT_ERROR";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::NO_PROMPT_ERROR);
        EXPECT_EQ(e.getMessage(), "no prompt in request!");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::NO_PROMPT_ERROR instead of std::exception";
    }

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(InferenceServiceTest, InferResponseSuccess) {
    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    const std::string        body = R"del(
        {
            "prompt": "hello request",
            "generate_config": {
                "is_streaming": true,
                "max_new_tokens": 20
            },
            "source": "test_source"
        }
    )del";
    request._request              = CreateHttpPacket(body);

    // engine
    auto                           mock_stream = CreateMockGenerateStream();
    auto                           stream      = std::dynamic_pointer_cast<GenerateStream>(mock_stream);
    std::vector<GenerateStreamPtr> streams({stream});
    EXPECT_CALL(*mock_engine_, batchEnqueue(Matcher<const std::vector<std::shared_ptr<GenerateInput>>&>(_)))
        .WillOnce(Return(streams));

    // stream
    GenerateOutputs outputs;
    outputs.generate_outputs.emplace_back(GenerateOutput());
    EXPECT_CALL(*mock_stream, nextOutput())
        .WillOnce(Return(ErrorResult<GenerateOutputs>(std::move(outputs))))
        .WillOnce(Return(ErrorResult<GenerateOutputs>(ErrorCode::OUTPUT_QUEUE_IS_EMPTY, "output queue is empty")));
    EXPECT_CALL(*mock_stream, finished()).Times(2);

    // writer
    EXPECT_CALL(*mock_writer_, isConnected()).WillOnce(Return(true));
    EXPECT_CALL(*mock_writer_, WriteDone()).Times(1);
    {
        InSequence s;
        EXPECT_CALL(*mock_writer_, Write(HasSubstr("hello response"))).WillOnce(Return(true));
        auto done_res = inference_service_->doneResponse();
        EXPECT_CALL(*mock_writer_, Write(StrEq(done_res))).WillOnce(Return(true));
    }

    // metric
    EXPECT_CALL(*mock_metric_reporter_, reportQpsMetric(Eq("test_source")));
    EXPECT_CALL(*mock_metric_reporter_, reportSuccessQpsMetric(Eq("test_source")));
    EXPECT_CALL(*mock_metric_reporter_, reportResponseIterateCountMetric(Eq(1)));
    EXPECT_CALL(*mock_metric_reporter_, reportResponseLatencyMs(Ge(0)));
    EXPECT_CALL(*mock_metric_reporter_, reportFTInputTokenLengthMetric(Eq(0)));
    EXPECT_CALL(*mock_metric_reporter_, reportFTNumBeansMetric(Eq(1)));
    EXPECT_CALL(*mock_metric_reporter_, reportFTPreTokenProcessorRtMetric(Gt(0)));
    EXPECT_CALL(*mock_metric_reporter_, reportFTPostTokenProcessorRtMetric(Gt(0)));
    EXPECT_CALL(*mock_metric_reporter_, reportResponseFirstTokenLatencyMs(Ge(0)));
    EXPECT_CALL(*mock_metric_reporter_, reportResponseIterateQpsMetric());

    // token processor
    std::vector<std::string> texts;
    texts.emplace_back("hello response");
    EXPECT_CALL(*mock_token_processor_, getTokenProcessorCtx(_, _, _)).WillOnce(Return(nullptr));
    EXPECT_CALL(*mock_token_processor_, decodeTokens(_, _, _, _)).WillOnce(Return(texts));
    EXPECT_CALL(*mock_token_processor_, encode(StrEq("hello request")));

    EXPECT_NO_THROW(inference_service_->inferResponse(10086, writer_ptr, request));
    EXPECT_EQ(inference_service_->controller_->current_concurrency_, 0);

    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Stream);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "text/event-stream");
    EXPECT_EQ(writer_ptr->_statusCode, 200);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(InferenceServiceTest, fillGenerateInput) {

    EXPECT_CALL(*mock_metric_reporter_, reportFTInputTokenLengthMetric(_)).Times(1);
    EXPECT_CALL(*mock_metric_reporter_, reportFTNumBeansMetric(_)).Times(1);
    EXPECT_CALL(*mock_metric_reporter_, reportFTPreTokenProcessorRtMetric(_)).Times(1);

    std::vector<int> token_ids = {1, 2, 3, 4, 5};
    EXPECT_CALL(*mock_token_processor_, encode(_)).WillOnce(Return(token_ids));

    int64_t                  request_id = 10086;
    std::string              text;
    std::vector<std::string> urls;
    auto                     generate_config = std::make_shared<GenerateConfig>();
    auto                     input = inference_service_->fillGenerateInput(request_id, text, urls, generate_config);

    EXPECT_EQ(input->request_id, request_id);
    EXPECT_TRUE(input->generate_config != nullptr);

    auto now = autil::TimeUtility::currentTimeInMicroSeconds();
    EXPECT_NEAR(input->begin_time_us, now, 5000);

    EXPECT_EQ(input->input_ids->type(), rtp_llm::DataType::TYPE_INT32);
    EXPECT_EQ(input->input_ids->size(), token_ids.size());
    EXPECT_EQ(input->input_ids->sizeBytes(), token_ids.size() * sizeof(int));
    EXPECT_TRUE(std::memcmp(input->input_ids->data(), token_ids.data(), input->input_ids->sizeBytes()) == 0);
}

TEST_F(InferenceServiceTest, iterateStreams) {

    EXPECT_CALL(*mock_metric_reporter_, reportResponseFirstTokenLatencyMs(_)).Times(1);
    EXPECT_CALL(*mock_metric_reporter_, reportResponseIterateLatencyMs(_)).Times(1);
    EXPECT_CALL(*mock_metric_reporter_, reportResponseIterateQpsMetric()).Times(2);

    auto mock_stream_wrapper = std::make_shared<MockGenerateStreamWrapper>(nullptr, nullptr);
    auto stream_ptr          = std::dynamic_pointer_cast<GenerateStreamWrapper>(mock_stream_wrapper);
    std::vector<std::shared_ptr<GenerateStreamWrapper>> streams;
    streams.push_back(stream_ptr);
    EXPECT_CALL(*mock_stream_wrapper, generateResponse)
        .WillOnce(Return(std::make_pair(MultiSeqsResponse(), false)))
        .WillOnce(Return(std::make_pair(MultiSeqsResponse(), false)))
        .WillOnce(Return(std::make_pair(MultiSeqsResponse(), true)));

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);  // remember to release
    EXPECT_CALL(*mock_writer_, Write).WillRepeatedly(Return(true));

    autil::StageTime  timer;
    const std::string body = R"({"prompt": "hello"})";
    auto              req  = InferenceParsedRequest::extractRequest(body, model_config_, nullptr);
    auto [cnt, res]        = inference_service_->iterateStreams(streams, writer_ptr, req, timer);

    ASSERT_EQ(cnt, 2);
    ASSERT_EQ(res.size(), 2);

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(InferenceServiceTest, iterateStreams_CancelError) {

    EXPECT_CALL(*mock_metric_reporter_, reportCancelQpsMetric(StrEq("unknown")));

    auto mock_stream_wrapper = std::make_shared<MockGenerateStreamWrapper>(nullptr, nullptr);
    auto stream_ptr          = std::dynamic_pointer_cast<GenerateStreamWrapper>(mock_stream_wrapper);
    std::vector<std::shared_ptr<GenerateStreamWrapper>> streams;
    streams.push_back(stream_ptr);
    EXPECT_CALL(*mock_stream_wrapper, generateResponse).WillOnce(Return(std::make_pair(MultiSeqsResponse(), false)));

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);  // remember to release
    EXPECT_CALL(*mock_writer_, Write).WillOnce(Return(false));

    autil::StageTime  timer;
    const std::string body = R"({"prompt": "hello"})";
    auto              req  = InferenceParsedRequest::extractRequest(body, model_config_, nullptr);
    try {
        auto [cnt, res] = inference_service_->iterateStreams(streams, writer_ptr, req, timer);
        FAIL() << "should throw HttpApiServerException::CANCELLED_ERROR";
    } catch (const HttpApiServerException& e) {
        EXPECT_EQ(e.getType(), HttpApiServerException::CANCELLED_ERROR);
        EXPECT_EQ(e.getMessage(), "client disconnects");
    } catch (const std::exception& e) {
        FAIL() << "should throw HttpApiServerException::CANCELLED_ERROR instead of std::exception";
    }

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(InferenceServiceTest, StreamingResponse_MultiSeqsResponse) {
    MultiSeqsResponse var = MultiSeqsResponse();
    std::string       res = inference_service_->streamingResponse(var);
    ASSERT_TRUE(res.find(R"("response":)") != std::string::npos);
    ASSERT_TRUE(res.find(R"("finished":)") != std::string::npos);
    ASSERT_TRUE(res.find(R"("aux_info":)") != std::string::npos);
    ASSERT_TRUE(res.find(R"("loss":)") == std::string::npos);
    ASSERT_TRUE(res.find(R"("logits":)") == std::string::npos);
    ASSERT_TRUE(res.find(R"("hidden_states":)") == std::string::npos);
    ASSERT_TRUE(res.find(R"("output_ids":)") == std::string::npos);
    ASSERT_TRUE(res.find(R"("input_ids":)") == std::string::npos);
}

TEST_F(InferenceServiceTest, StreamingResponse_BatchResponse) {
    std::vector<MultiSeqsResponse> batch_state;
    batch_state.push_back(MultiSeqsResponse());
    batch_state.push_back(MultiSeqsResponse());
    BatchResponse batch(batch_state);
    std::string   res = inference_service_->streamingResponse(batch);
    ASSERT_TRUE(res.find(R"("response_batch":[)") != std::string::npos);
}

TEST_F(InferenceServiceTest, StreamingResponse_InvalidResponse) {
    std::string invalid_type;
    std::string res = inference_service_->streamingResponse(invalid_type);
    ASSERT_EQ(res, "data:\n\n");
}

TEST_F(InferenceServiceTest, FormatResponse) {
    std::vector<MultiSeqsResponse> batch_state;
    batch_state.push_back(MultiSeqsResponse());
    {
        auto res = inference_service_->formatResponse(batch_state, /*batch_infer=*/true);
        ASSERT_EQ(res.type(), typeid(BatchResponse));
    }
    {
        auto res = inference_service_->formatResponse(batch_state, /*batch_infer=*/false);
        ASSERT_EQ(res.type(), typeid(MultiSeqsResponse));
    }
}

TEST_F(InferenceServiceTest, ExtractRequest_PromptBatch) {
    std::string jsonStr = R"({"prompt_batch": ["prompt1", "prompt2", "prompt3"]})";
    auto        req     = InferenceParsedRequest::extractRequest(jsonStr, model_config_, nullptr);
    ASSERT_EQ(req.batch_infer, true);
    ASSERT_EQ(req.is_streaming, false);
    ASSERT_EQ(req.input_texts.size(), 3);
    ASSERT_EQ(req.input_urls.size(), 3);
    ASSERT_EQ(req.generate_configs.size(), 3);
}

TEST_F(InferenceServiceTest, ExtractRequest_Prompt) {
    std::string jsonStr = R"({"prompt": "prompt1"})";
    auto        req     = InferenceParsedRequest::extractRequest(jsonStr, model_config_, nullptr);
    ASSERT_EQ(req.batch_infer, false);
    ASSERT_EQ(req.is_streaming, false);
    ASSERT_EQ(req.input_texts.size(), 1);
    ASSERT_EQ(req.input_urls.size(), 1);
    ASSERT_EQ(req.generate_configs.size(), 1);
}

TEST_F(InferenceServiceTest, ExtractRequest_AdapterName) {
    std::string jsonStr =
        R"({"prompt_batch": ["prompt1", "prompt2", "prompt3"], "generate_config": {"adapter_name": ["test0", "test1", "test2"]}})";
    auto req = InferenceParsedRequest::extractRequest(jsonStr, model_config_, nullptr);
    ASSERT_EQ(req.batch_infer, true);
    ASSERT_EQ(req.is_streaming, false);
    ASSERT_EQ(req.input_texts.size(), 3);
    ASSERT_EQ(req.input_urls.size(), 3);
    ASSERT_EQ(req.generate_configs.size(), 3);
    ASSERT_EQ(req.generate_configs[0]->adapter_name, "test0");
    ASSERT_EQ(req.generate_configs[1]->adapter_name, "test1");
    ASSERT_EQ(req.generate_configs[2]->adapter_name, "test2");
}

}  // namespace rtp_llm
