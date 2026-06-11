#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "autil/NetUtil.h"

#include "rtp_llm/cpp/api_server/FreezeService.h"
#include "rtp_llm/cpp/api_server/HttpApiServer.h"
#include "rtp_llm/cpp/engine_base/freeze/FreezeLifecycleController.h"

#include "rtp_llm/cpp/api_server/test/mock/MockEngineBase.h"
#include "rtp_llm/cpp/api_server/test/mock/MockHttpResponseWriter.h"

using namespace ::testing;

namespace rtp_llm {

class FreezeServiceTest: public ::testing::Test {
protected:
    void SetUp() override {
        mock_writer_    = std::make_unique<http_server::MockHttpResponseWriter>();
        mock_engine_    = std::make_shared<MockEngineBase>();
        freeze_service_ = std::make_shared<FreezeService>(mock_engine_);
    }
    void TearDown() override {}

    std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>
    CreateHttpPacket(const std::string& body) {
        auto packet = new ::anet::HTTPPacket();
        packet->setBody(body.c_str(), body.size());
        return std::unique_ptr<::anet::HTTPPacket, std::function<void(::anet::HTTPPacket*)>>(
            packet, [](::anet::HTTPPacket* packet) { packet->free(); });
    }

    // Captures the next Write() payload into written_.
    void ExpectWriteOnce() {
        EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([this](const std::string& data) {
            written_ = data;
            return true;
        }));
    }

protected:
    std::unique_ptr<http_server::MockHttpResponseWriter> mock_writer_;
    std::shared_ptr<MockEngineBase>                      mock_engine_;
    std::shared_ptr<FreezeService>                       freeze_service_;
    std::string                                          written_;
};

// -------------------------- POST /admin/freeze --------------------------

TEST_F(FreezeServiceTest, FreezeSuccess_EmptyBody) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    freeze_service_->freeze(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_THAT(written_, HasSubstr(R"("status":"ok")"));
    EXPECT_THAT(written_, HasSubstr(R"("state":"FROZEN")"));
    EXPECT_THAT(written_, HasSubstr(R"("freeze_epoch":1)"));
    EXPECT_EQ(mock_engine_->freezeController().state(), FreezeState::FROZEN);

    writer_ptr.release();
}

TEST_F(FreezeServiceTest, FreezeSuccess_WithBody) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    const std::string        body = R"({"mode":"force","drain_timeout_ms":1234,"reason":"unit test"})";
    request._request              = CreateHttpPacket(body);

    freeze_service_->freeze(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_THAT(written_, HasSubstr(R"("status":"ok")"));
    EXPECT_THAT(written_, HasSubstr(R"("state":"FROZEN")"));
    EXPECT_EQ(mock_engine_->freezeController().freezeEpoch(), 1);

    writer_ptr.release();
}

TEST_F(FreezeServiceTest, FreezeFailed_InvalidJsonBody) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    request._request = CreateHttpPacket("{not a json");

    freeze_service_->freeze(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 400);
    EXPECT_THAT(written_, HasSubstr("error"));
    // state machine untouched
    EXPECT_EQ(mock_engine_->freezeController().state(), FreezeState::RUNNING);

    writer_ptr.release();
}

TEST_F(FreezeServiceTest, FreezeFailed_InvalidMode) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    request._request = CreateHttpPacket(R"({"mode":"whatever"})");

    freeze_service_->freeze(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 400);
    EXPECT_THAT(written_, HasSubstr("error"));
    EXPECT_EQ(mock_engine_->freezeController().state(), FreezeState::RUNNING);

    writer_ptr.release();
}

TEST_F(FreezeServiceTest, FreezeFailed_DrainTimeoutKeepsDraining) {
    // M3 drain hook reports "not drained": freeze stays in DRAINING and the
    // endpoint reports a conflict instead of releasing GPU.
    FreezeHooks hooks;
    hooks.drain = [](const FreezeOptions&) { return false; };
    mock_engine_->freezeController().setHooks(hooks);

    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    freeze_service_->freeze(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 409);
    EXPECT_THAT(written_, HasSubstr("error"));
    EXPECT_EQ(mock_engine_->freezeController().state(), FreezeState::DRAINING);

    writer_ptr.release();
}

// -------------------------- POST /admin/resume --------------------------

TEST_F(FreezeServiceTest, ResumeSuccess_FromFrozen) {
    auto freeze_result = mock_engine_->freezeController().freeze(FreezeOptions());
    ASSERT_TRUE(freeze_result.ok);
    ASSERT_EQ(mock_engine_->freezeController().state(), FreezeState::FROZEN);

    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    freeze_service_->resume(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_THAT(written_, HasSubstr(R"("status":"ok")"));
    EXPECT_THAT(written_, HasSubstr(R"("state":"RUNNING")"));
    EXPECT_THAT(written_, HasSubstr(R"("freeze_epoch":1)"));
    EXPECT_EQ(mock_engine_->freezeController().state(), FreezeState::RUNNING);

    writer_ptr.release();
}

TEST_F(FreezeServiceTest, ResumeWhileDrainingAbortsFreeze) {
    FreezeHooks hooks;
    hooks.drain = [](const FreezeOptions&) { return false; };
    mock_engine_->freezeController().setHooks(hooks);
    mock_engine_->freezeController().freeze(FreezeOptions());
    ASSERT_EQ(mock_engine_->freezeController().state(), FreezeState::DRAINING);

    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    freeze_service_->resume(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_THAT(written_, HasSubstr(R"("status":"ok")"));
    EXPECT_THAT(written_, HasSubstr(R"("state":"RUNNING")"));
    EXPECT_EQ(mock_engine_->freezeController().state(), FreezeState::RUNNING);

    writer_ptr.release();
}

// -------------------------- GET /admin/freeze_status --------------------------

TEST_F(FreezeServiceTest, FreezeStatus_InitialSchema) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    freeze_service_->freezeStatus(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    // all 8 fields of FreezeStatusResponsePB must be present
    EXPECT_THAT(written_, HasSubstr(R"("state":"RUNNING")"));
    EXPECT_THAT(written_, HasSubstr(R"("freeze_epoch":0)"));
    EXPECT_THAT(written_, HasSubstr(R"("kv_memory_state":"ACTIVE")"));
    EXPECT_THAT(written_, HasSubstr(R"("device_kv_cache_valid":true)"));
    EXPECT_THAT(written_, HasSubstr(R"("active_request_count":0)"));
    EXPECT_THAT(written_, HasSubstr(R"("active_cache_transfer_count":0)"));
    EXPECT_THAT(written_, HasSubstr(R"("gpu_resource_state":"ACTIVE")"));
    EXPECT_THAT(written_, HasSubstr(R"("last_error":"")"));

    writer_ptr.release();
}

TEST_F(FreezeServiceTest, FreezeStatus_AfterFreeze) {
    auto freeze_result = mock_engine_->freezeController().freeze(FreezeOptions());
    ASSERT_TRUE(freeze_result.ok);

    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    freeze_service_->freezeStatus(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_THAT(written_, HasSubstr(R"("state":"FROZEN")"));
    EXPECT_THAT(written_, HasSubstr(R"("freeze_epoch":1)"));
    EXPECT_THAT(written_, HasSubstr(R"("device_kv_cache_valid":false)"));
    EXPECT_THAT(written_, HasSubstr(R"("gpu_resource_state":"RELEASED")"));

    writer_ptr.release();
}

// -------------------------- engine is null --------------------------

TEST_F(FreezeServiceTest, AllEndpoints_EngineNull) {
    auto service = std::make_shared<FreezeService>(nullptr);

    EXPECT_CALL(*mock_writer_, Write).Times(3).WillRepeatedly(Invoke([](const std::string& data) {
        EXPECT_THAT(data, HasSubstr("error"));
        return true;
    }));

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    service->freeze(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_statusCode, 503);
    service->resume(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_statusCode, 503);
    service->freezeStatus(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_statusCode, 503);

    writer_ptr.release();
}

// -------------------------- route registration --------------------------

TEST_F(FreezeServiceTest, RegisterFreezeService_Success) {
    const auto        port = autil::NetUtil::randomPort();
    const std::string addr = "tcp:0.0.0.0:" + std::to_string(port);
    EngineInitParams  params;
    HttpApiServer     server(nullptr, nullptr, addr, params, py::none());
    // start() runs registerServices() which already includes registerFreezeService()
    ASSERT_TRUE(server.start());
    EXPECT_NE(server.freeze_service_, nullptr);
    // re-registration goes through the same route table without error
    EXPECT_TRUE(server.registerFreezeService());
    server.stop();
}

TEST_F(FreezeServiceTest, RegisterFreezeService_HttpServerIsNull) {
    const auto        port = autil::NetUtil::randomPort();
    const std::string addr = "tcp:0.0.0.0:" + std::to_string(port);
    EngineInitParams  params;
    HttpApiServer     server(nullptr, nullptr, addr, params, py::none());
    // not started: http_server_ is null
    EXPECT_FALSE(server.registerFreezeService());
}

}  // namespace rtp_llm
