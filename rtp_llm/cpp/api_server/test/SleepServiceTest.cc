#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "autil/NetUtil.h"

#include "rtp_llm/cpp/api_server/SleepService.h"
#include "rtp_llm/cpp/api_server/HttpApiServer.h"
#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"

#include "rtp_llm/cpp/api_server/test/mock/MockEngineBase.h"
#include "rtp_llm/cpp/api_server/test/mock/MockHttpResponseWriter.h"

using namespace ::testing;

namespace rtp_llm {

class SleepServiceTest: public ::testing::Test {
protected:
    void SetUp() override {
        mock_writer_ = std::make_unique<http_server::MockHttpResponseWriter>();
        mock_engine_ = std::make_shared<MockEngineBase>();
        mock_engine_->sleepController().setEnabled(true);
        sleep_service_ = std::make_shared<SleepService>(mock_engine_);
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
    std::shared_ptr<SleepService>                        sleep_service_;
    std::string                                          written_;
};

// -------------------------- POST /sleep --------------------------

TEST_F(SleepServiceTest, SleepSuccess_EmptyBody) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    sleep_service_->sleep(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_THAT(written_, HasSubstr(R"("status":"ok")"));
    EXPECT_THAT(written_, Not(HasSubstr(R"("state")")));
    EXPECT_THAT(written_, Not(HasSubstr(R"("sleep_epoch")")));
    EXPECT_EQ(mock_engine_->sleepController().state(), SleepState::SLEEPING);

    writer_ptr.release();
}

TEST_F(SleepServiceTest, SleepSuccess_WithBody) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    const std::string        body = R"({"mode":"abort","timeout_ms":1234,"reason":"unit test"})";
    request._request              = CreateHttpPacket(body);

    sleep_service_->sleep(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_THAT(written_, HasSubstr(R"("status":"ok")"));
    EXPECT_THAT(written_, Not(HasSubstr(R"("state")")));
    EXPECT_THAT(written_, Not(HasSubstr(R"("sleep_epoch")")));
    EXPECT_EQ(mock_engine_->sleepController().sleepEpoch(), 1);

    writer_ptr.release();
}

TEST_F(SleepServiceTest, SleepFailed_InvalidJsonBody) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    request._request = CreateHttpPacket("{not a json");

    sleep_service_->sleep(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 400);
    EXPECT_THAT(written_, HasSubstr("error"));
    // state machine untouched
    EXPECT_EQ(mock_engine_->sleepController().state(), SleepState::RUNNING);

    writer_ptr.release();
}

TEST_F(SleepServiceTest, SleepFailed_InvalidMode) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    request._request = CreateHttpPacket(R"({"mode":"whatever"})");

    sleep_service_->sleep(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 400);
    EXPECT_THAT(written_, HasSubstr("error"));
    EXPECT_EQ(mock_engine_->sleepController().state(), SleepState::RUNNING);

    writer_ptr.release();
}

TEST_F(SleepServiceTest, SleepFailed_Disabled) {
    mock_engine_->sleepController().setEnabled(false);
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    sleep_service_->sleep(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 501);
    EXPECT_THAT(written_, HasSubstr("sleep mode is disabled"));
    EXPECT_EQ(mock_engine_->sleepController().state(), SleepState::RUNNING);

    writer_ptr.release();
}

TEST_F(SleepServiceTest, SleepFailed_UnsupportedLevel) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    request._request = CreateHttpPacket(R"({"level":2})");

    sleep_service_->sleep(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 400);
    EXPECT_THAT(written_, HasSubstr("unknown level=2"));

    writer_ptr.release();
}

TEST_F(SleepServiceTest, SleepFailed_LevelZeroUnimplemented) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    request._request = CreateHttpPacket(R"({"level":0})");

    sleep_service_->sleep(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 501);
    EXPECT_THAT(written_, HasSubstr("level=0"));
    EXPECT_THAT(written_, HasSubstr("not implemented"));
    EXPECT_EQ(mock_engine_->sleepController().state(), SleepState::RUNNING);

    writer_ptr.release();
}

TEST_F(SleepServiceTest, SleepFailed_EmptyTag) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    request._request = CreateHttpPacket(R"({"tags":["kv_cache",""]})");

    sleep_service_->sleep(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 400);
    EXPECT_THAT(written_, HasSubstr("tags must be non-empty strings"));

    writer_ptr.release();
}

TEST_F(SleepServiceTest, SleepFailed_ExternalPrepareOnlyRejected) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    request._request = CreateHttpPacket(R"({"prepare_only":true})");

    sleep_service_->sleep(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 400);
    EXPECT_THAT(written_, HasSubstr("prepare_only is unsupported"));
    EXPECT_EQ(mock_engine_->sleepController().state(), SleepState::RUNNING);

    writer_ptr.release();
}

TEST_F(SleepServiceTest, SleepFailed_DrainTimeoutKeepsDraining) {
    // M3 drain hook reports "not drained": sleep stays in DRAINING and the
    // endpoint reports a conflict instead of releasing GPU.
    SleepHooks hooks;
    hooks.drain = [](const SleepOptions&) { return false; };
    mock_engine_->sleepController().setHooks(hooks);

    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    sleep_service_->sleep(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 409);
    EXPECT_THAT(written_, HasSubstr("error"));
    EXPECT_EQ(mock_engine_->sleepController().state(), SleepState::DRAINING);

    writer_ptr.release();
}

// -------------------------- POST /wake_up --------------------------

TEST_F(SleepServiceTest, WakeUpSuccess_FromSleeping) {
    auto sleep_result = mock_engine_->sleepController().sleep(SleepOptions());
    ASSERT_TRUE(sleep_result.ok);
    ASSERT_EQ(mock_engine_->sleepController().state(), SleepState::SLEEPING);

    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    sleep_service_->wakeUp(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_THAT(written_, HasSubstr(R"("status":"ok")"));
    EXPECT_THAT(written_, Not(HasSubstr(R"("state")")));
    EXPECT_THAT(written_, Not(HasSubstr(R"("sleep_epoch")")));
    EXPECT_EQ(mock_engine_->sleepController().state(), SleepState::RUNNING);

    writer_ptr.release();
}

TEST_F(SleepServiceTest, WakeUpWhileDrainingAbortsSleep) {
    SleepHooks hooks;
    hooks.drain = [](const SleepOptions&) { return false; };
    mock_engine_->sleepController().setHooks(hooks);
    mock_engine_->sleepController().sleep(SleepOptions());
    ASSERT_EQ(mock_engine_->sleepController().state(), SleepState::DRAINING);

    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    sleep_service_->wakeUp(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_THAT(written_, HasSubstr(R"("status":"ok")"));
    EXPECT_THAT(written_, Not(HasSubstr(R"("state")")));
    EXPECT_THAT(written_, Not(HasSubstr(R"("sleep_epoch")")));
    EXPECT_EQ(mock_engine_->sleepController().state(), SleepState::RUNNING);

    writer_ptr.release();
}

TEST_F(SleepServiceTest, WakeUpFailed_ExternalCommitOnlyRejected) {
    auto sleep_result = mock_engine_->sleepController().sleep(SleepOptions());
    ASSERT_TRUE(sleep_result.ok);
    ASSERT_EQ(mock_engine_->sleepController().state(), SleepState::SLEEPING);

    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);

    http_server::HttpRequest request;
    request._request = CreateHttpPacket(R"({"commit_only":true})");

    sleep_service_->wakeUp(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 400);
    EXPECT_THAT(written_, HasSubstr("commit_only is unsupported"));
    EXPECT_EQ(mock_engine_->sleepController().state(), SleepState::SLEEPING);

    writer_ptr.release();
}

// -------------------------- GET /sleep_status --------------------------

TEST_F(SleepServiceTest, SleepStatus_InitialSchema) {
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    sleep_service_->sleepStatus(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    // All SleepStatusResponsePB fields must be present.
    EXPECT_THAT(written_, HasSubstr(R"("sleep_mode_enabled":true)"));
    EXPECT_THAT(written_, HasSubstr(R"("effective":true)"));
    EXPECT_THAT(written_, HasSubstr(R"("supported_levels":[1])"));
    EXPECT_THAT(written_, HasSubstr(R"("supported_modes":["wait","abort"])"));
    EXPECT_THAT(written_, HasSubstr(R"("disabled_reason":"")"));
    EXPECT_THAT(written_, HasSubstr(R"("state":"RUNNING")"));
    EXPECT_THAT(written_, HasSubstr(R"("sleep_epoch":0)"));
    EXPECT_THAT(written_, HasSubstr(R"("kv_memory_state":"ACTIVE")"));
    EXPECT_THAT(written_, HasSubstr(R"("device_kv_cache_valid":true)"));
    EXPECT_THAT(written_, HasSubstr(R"("active_request_count":0)"));
    EXPECT_THAT(written_, HasSubstr(R"("active_cache_transfer_count":0)"));
    EXPECT_THAT(written_, HasSubstr(R"("gpu_resource_state":"ACTIVE")"));
    EXPECT_THAT(written_, HasSubstr(R"("last_error":"")"));

    writer_ptr.release();
}

TEST_F(SleepServiceTest, SleepStatus_AfterSleep) {
    auto sleep_result = mock_engine_->sleepController().sleep(SleepOptions());
    ASSERT_TRUE(sleep_result.ok);

    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    sleep_service_->sleepStatus(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_THAT(written_, HasSubstr(R"("state":"SLEEPING")"));
    EXPECT_THAT(written_, HasSubstr(R"("sleep_epoch":1)"));
    EXPECT_THAT(written_, HasSubstr(R"("device_kv_cache_valid":false)"));
    EXPECT_THAT(written_, HasSubstr(R"("gpu_resource_state":"RELEASED")"));

    writer_ptr.release();
}

TEST_F(SleepServiceTest, IsSleeping_ReportsCapabilityAndState) {
    auto sleep_result = mock_engine_->sleepController().sleep(SleepOptions());
    ASSERT_TRUE(sleep_result.ok);

    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    sleep_service_->isSleeping(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_THAT(written_, HasSubstr(R"("is_sleeping":true)"));
    EXPECT_THAT(written_, HasSubstr(R"("sleep_mode_enabled":true)"));
    EXPECT_THAT(written_, HasSubstr(R"("effective":true)"));
    EXPECT_THAT(written_, HasSubstr(R"("state":"SLEEPING")"));

    writer_ptr.release();
}

TEST_F(SleepServiceTest, SleepStatus_DisabledReportsFallbackSignal) {
    mock_engine_->sleepController().setEnabled(false);
    ExpectWriteOnce();

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    sleep_service_->sleepStatus(writer_ptr, request);

    EXPECT_EQ(writer_ptr->_statusCode, 200);
    EXPECT_THAT(written_, HasSubstr(R"("sleep_mode_enabled":false)"));
    EXPECT_THAT(written_, HasSubstr(R"("effective":false)"));
    EXPECT_THAT(written_, HasSubstr(R"("supported_levels":[])"));
    EXPECT_THAT(written_, HasSubstr(R"("disabled_reason":"sleep mode is disabled")"));

    writer_ptr.release();
}

// -------------------------- engine is null --------------------------

TEST_F(SleepServiceTest, AllEndpoints_EngineNull) {
    auto service = std::make_shared<SleepService>(nullptr);

    EXPECT_CALL(*mock_writer_, Write).Times(4).WillRepeatedly(Invoke([](const std::string& data) {
        EXPECT_THAT(data, HasSubstr("error"));
        return true;
    }));

    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    service->sleep(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_statusCode, 503);
    service->wakeUp(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_statusCode, 503);
    service->isSleeping(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_statusCode, 503);
    service->sleepStatus(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_statusCode, 503);

    writer_ptr.release();
}

// -------------------------- route registration --------------------------

TEST_F(SleepServiceTest, RegisterSleepService_Success) {
    const auto        port = autil::NetUtil::randomPort();
    const std::string addr = "tcp:0.0.0.0:" + std::to_string(port);
    EngineInitParams  params;
    HttpApiServer     server(nullptr, nullptr, addr, params, py::none());
    // start() runs registerServices() which already includes registerSleepService()
    ASSERT_TRUE(server.start());
    EXPECT_NE(server.sleep_service_, nullptr);
    // re-registration goes through the same route table without error
    EXPECT_TRUE(server.registerSleepService());
    server.stop();
}

TEST_F(SleepServiceTest, RegisterSleepService_HttpServerIsNull) {
    const auto        port = autil::NetUtil::randomPort();
    const std::string addr = "tcp:0.0.0.0:" + std::to_string(port);
    EngineInitParams  params;
    HttpApiServer     server(nullptr, nullptr, addr, params, py::none());
    // not started: http_server_ is null
    EXPECT_FALSE(server.registerSleepService());
}

}  // namespace rtp_llm
