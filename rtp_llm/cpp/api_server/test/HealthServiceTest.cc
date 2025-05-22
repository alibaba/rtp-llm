#include "rtp_llm/cpp/api_server/common/HealthService.h"
#include "rtp_llm/cpp/api_server/test/mock/MockHttpResponseWriter.h"

using namespace ::testing;
namespace rtp_llm {

class HealthServiceTest: public ::testing::Test {
public:
    HealthServiceTest()           = default;
    ~HealthServiceTest() override = default;

protected:
    void SetUp() override {
        mock_writer_    = std::make_unique<http_server::MockHttpResponseWriter>();
        health_service_ = std::make_shared<HealthService>();
    }
    void TearDown() override {}

private:
    std::unique_ptr<http_server::MockHttpResponseWriter> mock_writer_;
    std::shared_ptr<HealthService>                       health_service_;
};

TEST_F(HealthServiceTest, HealthCheck_ServerNotStopped) {
    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        EXPECT_EQ(data, R"("ok")");
        return true;
    }));

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;
    health_service_->healthCheck(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(HealthServiceTest, HealthCheck_ServerStopped) {
    health_service_->stop();
    EXPECT_TRUE(health_service_->is_stopped_.load());

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        EXPECT_EQ(data, R"({"detail":"this server has been shutdown"})");
        return true;
    }));

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;
    health_service_->healthCheck(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(mock_writer_->_statusCode, 503);
    EXPECT_EQ(mock_writer_->_statusMessage, "Service Unavailable");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(HealthServiceTest, HealthCheck2_ServerNotStopped) {
    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        EXPECT_EQ(data, R"({"status":"home"})");
        return true;
    }));

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;
    health_service_->healthCheck2(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(HealthServiceTest, HealthCheck2_ServerStopped) {
    health_service_->stop();
    EXPECT_TRUE(health_service_->is_stopped_.load());

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        EXPECT_EQ(data, R"({"detail":"this server has been shutdown"})");
        return true;
    }));

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;
    health_service_->healthCheck2(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(mock_writer_->_statusCode, 503);
    EXPECT_EQ(mock_writer_->_statusMessage, "Service Unavailable");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

}  // namespace rtp_llm