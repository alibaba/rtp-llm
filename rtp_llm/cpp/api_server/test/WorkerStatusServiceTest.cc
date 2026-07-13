#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "autil/EnvUtil.h"

#include "rtp_llm/cpp/api_server/WorkerStatusService.h"
#include "rtp_llm/cpp/api_server/ConcurrencyControllerUtil.h"

#include "rtp_llm/cpp/api_server/test/mock/MockEngineBase.h"
#include "rtp_llm/cpp/api_server/test/mock/MockHttpResponseWriter.h"

using namespace ::testing;

namespace rtp_llm {

class WorkerStatusServiceTest: public ::testing::Test {
public:
    WorkerStatusServiceTest()           = default;
    ~WorkerStatusServiceTest() override = default;

protected:
    void SetUp() override {
        mock_writer_           = std::make_unique<http_server::MockHttpResponseWriter>();
        mock_engine_base_      = std::make_shared<MockEngineBase>();
        controller_            = std::make_shared<ConcurrencyController>();
        worker_status_service_ = std::make_shared<WorkerStatusService>(mock_engine_base_, controller_);
    }
    void TearDown() override {}

protected:
    std::unique_ptr<http_server::MockHttpResponseWriter> mock_writer_;
    std::shared_ptr<MockEngineBase>                      mock_engine_base_;
    std::shared_ptr<ConcurrencyController>               controller_;
    std::shared_ptr<WorkerStatusService>                 worker_status_service_;
};

TEST_F(WorkerStatusServiceTest, Constructor) {
    WorkerStatusService worker_status_service(nullptr, nullptr);
    EXPECT_EQ(worker_status_service.engine_, nullptr);
    EXPECT_EQ(worker_status_service.controller_, nullptr);
}

TEST_F(WorkerStatusServiceTest, WorkerStatus_AlreadyStopped) {
    auto writer = dynamic_cast<http_server::HttpResponseWriter*>(mock_writer_.get());
    ASSERT_TRUE(writer != nullptr);
    std::unique_ptr<http_server::HttpResponseWriter> writer_ptr(writer);
    http_server::HttpRequest                         request;

    EXPECT_CALL(*mock_writer_, Write).WillOnce(Invoke([](const std::string& data) {
        EXPECT_EQ(data, R"({"detail":"this server has been shutdown"})");
        return true;
    }));

    worker_status_service_->stop();
    worker_status_service_->workerStatus(writer_ptr, request);
    EXPECT_EQ(writer_ptr->_type, http_server::HttpResponseWriter::WriteType::Normal);
    EXPECT_EQ(writer_ptr->_headers.count("Content-Type"), 1);
    EXPECT_EQ(writer_ptr->_headers.at("Content-Type"), "application/json");
    EXPECT_EQ(writer_ptr->_statusCode, 503);
    EXPECT_EQ(writer_ptr->_statusMessage, "Service Unavailable");

    // 需要手动释放 unique_ptr 的所有权, 避免 double free
    writer_ptr.release();
}

TEST_F(WorkerStatusServiceTest, Stop) {
    WorkerStatusService worker_status_service(nullptr, nullptr);
    EXPECT_FALSE(worker_status_service.is_stopped_.load());

    worker_status_service.stop();
    EXPECT_TRUE(worker_status_service.is_stopped_.load());
}

}  // namespace rtp_llm
