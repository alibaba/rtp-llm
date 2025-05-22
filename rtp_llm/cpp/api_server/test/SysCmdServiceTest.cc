#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "rtp_llm/cpp/api_server/SysCmdService.h"
#include "rtp_llm/cpp/api_server/test/mock/MockHttpResponseWriter.h"
#include "rtp_llm/cpp/utils/Logger.h"

using namespace ::testing;
namespace rtp_llm {

class SysCmdServiceTest: public ::testing::Test {
protected:
    void SetUp() override {
        mock_writer_ = std::make_unique<http_server::MockHttpResponseWriter>();
        cmd_service_ = std::make_shared<SysCmdService>();
    }
    void TearDown() override {}

    std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>
    CreateHttpPacket(const std::string& body) {
        auto packet = new ::anet::HTTPPacket();
        packet->setBody(body.c_str(), body.size());
        return std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>(
            packet, [](::anet::HTTPPacket* packet) { packet->free(); });
    }

protected:
    std::unique_ptr<http_server::MockHttpResponseWriter> mock_writer_;
    std::shared_ptr<SysCmdService>                       cmd_service_;
};

TEST_F(SysCmdServiceTest, SetLogLevelFailed_NoLogLevelInRequest) {
    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        EXPECT_NE(data, R"({"status":"ok"})");
        return true;
    }));

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;
    cmd_service_->setLogLevel(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(SysCmdServiceTest, SetLogLevelFailed_TorchExtSetLogLevelFailed) {
    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        EXPECT_NE(data, R"({"status":"ok"})");
        return true;
    }));

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    const std::string        body = R"del({
    "log_level": "test"
})del";
    request._request              = CreateHttpPacket(body);

    cmd_service_->setLogLevel(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(SysCmdServiceTest, SetLogLevelSuccess) {
    EXPECT_CALL(*mock_writer_, Write).Times(3).WillRepeatedly(Invoke([](const std::string& data) {
        EXPECT_EQ(data, R"({"status":"ok"})");
        return true;
    }));

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    auto& logger = Logger::getEngineLogger();

    // set log level to INFO
    {
        http_server::HttpRequest request;
        const std::string        body = R"del(
{
    "log_level": "INFO"
})del";
        request._request              = CreateHttpPacket(body);

        cmd_service_->setLogLevel(writer_ptr, request);
        EXPECT_EQ(logger.getLevelfromstr("FAKE_ENV_NAME"), alog::LOG_LEVEL_INFO);
    }

    // set log level to DEBUG
    {
        http_server::HttpRequest request;
        const std::string        body = R"del(
{
    "log_level": "DEBUG"
})del";
        request._request              = CreateHttpPacket(body);

        cmd_service_->setLogLevel(writer_ptr, request);
        EXPECT_EQ(logger.getLevelfromstr("FAKE_ENV_NAME"), alog::LOG_LEVEL_DEBUG);
    }

    // set log level to TRACE
    {
        http_server::HttpRequest request;
        const std::string        body = R"del(
{
    "log_level": "TRACE"
})del";
        request._request              = CreateHttpPacket(body);

        cmd_service_->setLogLevel(writer_ptr, request);
        EXPECT_EQ(logger.getLevelfromstr("FAKE_ENV_NAME"), alog::LOG_LEVEL_TRACE1);
    }

    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

}  // namespace rtp_llm