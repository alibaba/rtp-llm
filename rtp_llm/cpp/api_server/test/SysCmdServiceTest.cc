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

TEST_F(SysCmdServiceTest, StartProfileParametersAndDefaults) {
    for (const std::string body :
         {std::string(""), std::string(R"({"trace_name":"embedding","start_step":2,"num_steps":4,"all_tp":true})")}) {
        bool called = false;
        EXPECT_CALL(*mock_writer_, Write(R"({"status":"ok"})")).WillOnce(Return(true));
        std::unique_ptr<http_server::HttpResponseWriter> writer(mock_writer_.release());
        http_server::HttpRequest                         request;
        request._request = CreateHttpPacket(body);
        cmd_service_->startProfile(writer, request, [&](const std::string& name, int start, int steps, bool all_rank) {
            called = true;
            EXPECT_EQ(name, body.empty() ? "" : "embedding");
            EXPECT_EQ(start, body.empty() ? 0 : 2);
            EXPECT_EQ(steps, body.empty() ? 0 : 4);
            EXPECT_EQ(all_rank, !body.empty());
        });
        EXPECT_TRUE(called);
        mock_writer_.reset(static_cast<http_server::MockHttpResponseWriter*>(writer.release()));
    }
}

TEST_F(SysCmdServiceTest, StartProfileRejectsMalformedRequest) {
    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& body) {
        EXPECT_NE(body.find("error"), std::string::npos);
        return true;
    }));
    std::unique_ptr<http_server::HttpResponseWriter> writer(mock_writer_.release());
    http_server::HttpRequest                         request;
    request._request = CreateHttpPacket(R"({"num_steps":"invalid"})");
    cmd_service_->startProfile(
        writer, request, [](const std::string&, int, int, bool) { FAIL() << "unexpected start"; });
    EXPECT_EQ(writer->_statusCode, 400);
}

}  // namespace rtp_llm