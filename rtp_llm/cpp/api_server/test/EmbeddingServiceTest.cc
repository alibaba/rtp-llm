#include "rtp_llm/cpp/api_server/EmbeddingService.h"

#include "rtp_llm/cpp/api_server/Exception.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/api_server/test/mock/MockApiServerMetricReporter.h"
#include "rtp_llm/cpp/api_server/test/mock/MockEmbeddingEndpoint.h"
#include "rtp_llm/cpp/api_server/test/mock/MockHttpResponseWriter.h"

using namespace ::testing;
namespace rtp_llm {

class EmbeddingServiceTest: public ::testing::Test {
public:
    EmbeddingServiceTest()           = default;
    ~EmbeddingServiceTest() override = default;

protected:
    void SetUp() override {
        mock_embedding_endpoint_ = std::make_shared<MockEmbeddingEndpoint>();
        auto embedding_endpoint  = std::dynamic_pointer_cast<EmbeddingEndpoint>(mock_embedding_endpoint_);

        // mock_metric_reporter_ = std::make_shared<MockApiServerMetricReporter>();
        // auto metric_reporter  = std::dynamic_pointer_cast<ApiServerMetricReporter>(mock_metric_reporter_);
        //  TODO: mock kmonitor::MetricsReporterPtr
        kmonitor::MetricsReporterPtr metric_reporter;

        auto request_counter = std::make_shared<autil::AtomicCounter>();
        auto controller      = std::make_shared<ConcurrencyController>(1, false);

        SetToMaster();
        embedding_service_ =
            std::make_shared<EmbeddingService>(embedding_endpoint, request_counter, controller, metric_reporter);

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
    std::shared_ptr<MockEmbeddingEndpoint>               mock_embedding_endpoint_;
    std::shared_ptr<MockApiServerMetricReporter>         mock_metric_reporter_;
    std::shared_ptr<EmbeddingService>                    embedding_service_;
    std::unique_ptr<http_server::MockHttpResponseWriter> mock_writer_;
    std::unique_ptr<http_server::HttpResponseWriter>     writer_;
};

TEST_F(EmbeddingServiceTest, Embedding_ParseJsonFailed) {
    http_server::HttpRequest request;
    const std::string        body = R"({"private_request": "invalid_bool"})";
    request._request              = CreateHttpPacket(body);

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        EXPECT_FALSE(data.empty());
        return true;
    }));
    // EXPECT_CALL(*mock_metric_reporter_, reportErrorQpsMetric(_,_));
    // EXPECT_CALL(*mock_metric_reporter_,
    //         reportErrorQpsMetric(StrEq("unknown"), HttpApiServerException::ERROR_INPUT_FORMAT_ERROR));

    embedding_service_->embedding(writer_, request);

    EXPECT_EQ(writer_->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_->_statusCode, 400);
    EXPECT_EQ(writer_->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_->_headers.at("Content-Type"), "application/json");
}

TEST_F(EmbeddingServiceTest, Embedding_EmbeddingEndpointIsNull) {
    embedding_service_->embedding_endpoint_ = nullptr;

    http_server::HttpRequest request;
    const std::string        body = R"({"source": "test_source"})";
    request._request              = CreateHttpPacket(body);

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        EXPECT_FALSE(data.empty());
        return true;
    }));
    // EXPECT_CALL(*mock_metric_reporter_,
    //         reportErrorQpsMetric(StrEq("test_source"), HttpApiServerException::UNKNOWN_ERROR));

    embedding_service_->embedding(writer_, request);

    EXPECT_EQ(writer_->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_->_statusCode, 501);
    EXPECT_EQ(writer_->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_->_headers.at("Content-Type"), "application/json");
}

TEST_F(EmbeddingServiceTest, Embedding_Success) {
    http_server::HttpRequest request;
    const std::string        body = R"({"source": "test_source"})";
    request._request              = CreateHttpPacket(body);

    // EXPECT_CALL(*mock_metric_reporter_, reportQpsMetric(StrEq("test_source")));

    std::string handle_response = "hello world";
    EXPECT_CALL(*mock_embedding_endpoint_, handle)
        .WillOnce(
            Invoke([handle_response](const std::string&                              body,
                                     std::optional<EmbeddingEndpoint::EmbeddingType> type,
                                     const kmonitor::MetricsReporterPtr&             metrics_reporter,
                                     int64_t start_time_us) { return std::make_pair(handle_response, std::nullopt); }));
    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([handle_response](const std::string& data) {
        EXPECT_EQ(data, handle_response);
        return true;
    }));
    // EXPECT_CALL(*mock_metric_reporter_, reportSuccessQpsMetric(StrEq("test_source")));

    embedding_service_->embedding(writer_, request);

    EXPECT_EQ(writer_->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_->_statusCode, 200);
    EXPECT_EQ(writer_->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_->_headers.at("Content-Type"), "application/json");
}

}  // namespace rtp_llm
