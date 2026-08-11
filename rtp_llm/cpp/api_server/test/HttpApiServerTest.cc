#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/api_server/HttpApiServer.h"
#include "rtp_llm/cpp/api_server/common/HealthService.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRouter.h"
#include "rtp_llm/cpp/api_server/test/mock/MockChatRender.h"

#include "rtp_llm/cpp/api_server/http_server/http_client/SimpleHttpClient.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequestWorkItem.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpServerAdapter.h"
#include "autil/NetUtil.h"

namespace rtp_llm {

class HttpApiServerTest: public ::testing::Test {
public:
    HttpApiServerTest()           = default;
    ~HttpApiServerTest() override = default;

protected:
    void SetUp() override {
        const auto        port    = autil::NetUtil::randomPort();
        const std::string address = "tcp:0.0.0.0:" + std::to_string(port);
        EngineInitParams  params;
        server_ = std::make_shared<HttpApiServer>(nullptr, nullptr, address, params, py::none());
        EXPECT_TRUE(server_->start());
    }
    void TearDown() override {
        server_.reset();
    }

private:
    std::shared_ptr<HttpApiServer> server_;
};

TEST_F(HttpApiServerTest, testApiServerStart) {
    const auto        port = autil::NetUtil::randomPort();
    const std::string addr = "tcp:0.0.0.0:" + std::to_string(port);
    EngineInitParams  params;
    py::object        token_processor;
    HttpApiServer     server(nullptr, nullptr, addr, params, token_processor);
    ASSERT_TRUE(server.start());
    ASSERT_FALSE(server.isStoped());
    ASSERT_EQ(server.getListenAddr(), addr);
    server.stop();
    ASSERT_TRUE(server.isStoped());
}

TEST_F(HttpApiServerTest, MissingRendererDoesNotExposeChatRoutes) {
    ASSERT_EQ(server_->render_, nullptr);
    EXPECT_EQ(server_->chat_service_, nullptr);

    const auto& router = server_->http_server_->_router;
    ASSERT_NE(router, nullptr);
    EXPECT_FALSE(router->FindRoute("POST", "/chat/completions").has_value());
    EXPECT_FALSE(router->FindRoute("POST", "/v1/chat/completions").has_value());
    EXPECT_FALSE(router->FindRoute("POST", "/chat/render").has_value());
    EXPECT_FALSE(router->FindRoute("POST", "/v1/chat/render").has_value());

    EXPECT_TRUE(router->FindRoute("GET", "/health").has_value());
    EXPECT_TRUE(router->FindRoute("POST", "/tokenizer/encode").has_value());
    EXPECT_TRUE(router->FindRoute("POST", "/").has_value());
    EXPECT_TRUE(router->FindRoute("POST", "/inference_internal").has_value());
}

TEST(HttpApiServerRendererTest, ConfiguredRendererExposesChatRoutes) {
    const auto        port = autil::NetUtil::randomPort();
    const std::string addr = "tcp:0.0.0.0:" + std::to_string(port);
    EngineInitParams  params;
    HttpApiServer     server(nullptr, nullptr, addr, params, py::none());
    server.render_ = std::make_shared<::testing::NiceMock<MockChatRender>>();

    ASSERT_TRUE(server.start());
    ASSERT_NE(server.chat_service_, nullptr);
    const auto& router = server.http_server_->_router;
    ASSERT_NE(router, nullptr);
    EXPECT_TRUE(router->FindRoute("POST", "/chat/completions").has_value());
    EXPECT_TRUE(router->FindRoute("POST", "/v1/chat/completions").has_value());
    EXPECT_TRUE(router->FindRoute("POST", "/chat/render").has_value());
    EXPECT_TRUE(router->FindRoute("POST", "/v1/chat/render").has_value());
    server.stop();
}

TEST(HttpApiServerLifetimeTest, OwnsModelConfigAfterInitParamsLifetimeEnds) {
    std::unique_ptr<HttpApiServer> server;
    {
        EngineInitParams params;
        params.model_config_.max_seq_len = 4096;
        server = std::make_unique<HttpApiServer>(nullptr, nullptr, "", params, py::none());
    }

    EXPECT_EQ(server->model_config_.max_seq_len, 4096);
}

TEST_F(HttpApiServerTest, testApiServerStop) {
    const auto        port = autil::NetUtil::randomPort();
    const std::string addr = "tcp:0.0.0.0:" + std::to_string(port);
    EngineInitParams  params;
    auto              server = std::make_shared<HttpApiServer>(nullptr, nullptr, addr, params, py::none());
    EXPECT_TRUE(server->start());
    auto permit = server->request_admission_gate_->tryAcquire();
    ASSERT_TRUE(permit);
    server->beginDrain();
    EXPECT_FALSE(server->waitForDrain(std::chrono::steady_clock::now()));

    std::thread releaser([permit = std::move(permit)]() mutable {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        permit.reset();
    });
    EXPECT_TRUE(server->waitForDrain(std::chrono::steady_clock::now() + std::chrono::seconds(1)));
    EXPECT_TRUE(server->finishStop());
    EXPECT_TRUE(server->isStoped());
    releaser.join();
}

TEST_F(HttpApiServerTest, ForceStopTransportDoesNotWaitForActiveRequest) {
    const auto        port = autil::NetUtil::randomPort();
    const std::string addr = "tcp:0.0.0.0:" + std::to_string(port);
    EngineInitParams  params;
    auto              server = std::make_shared<HttpApiServer>(nullptr, nullptr, addr, params, py::none());
    ASSERT_TRUE(server->start());

    auto permit = server->request_admission_gate_->tryAcquire();
    ASSERT_TRUE(permit);
    auto stopped = std::async(std::launch::async, [&server]() { server->forceStopTransport(); });

    EXPECT_EQ(stopped.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    stopped.get();
    EXPECT_TRUE(server->isStoped());
    EXPECT_FALSE(server->waitForDrain(std::chrono::steady_clock::now()));

    permit.reset();
    EXPECT_TRUE(server->waitForDrain(std::chrono::steady_clock::now()));
}

TEST_F(HttpApiServerTest, QueuedAdmissionCoversHandlerExecution) {
    ASSERT_TRUE(server_->http_server_->_serverAdapter->_requestAdmissionHandler);
    auto admission_token = server_->http_server_->_serverAdapter->_requestAdmissionHandler();
    ASSERT_NE(admission_token, nullptr);

    std::atomic<size_t> calls{0};
    auto request = std::make_shared<http_server::HttpRequest>();
    http_server::ResponseHandler handler = [this, &calls](std::unique_ptr<http_server::HttpResponseWriter>,
                                                          const http_server::HttpRequest&) {
        EXPECT_FALSE(server_->waitForDrain(std::chrono::steady_clock::now()));
        ++calls;
    };
    http_server::HttpRequestWorkItem queued_work(handler, nullptr, std::move(request), std::move(admission_token));

    server_->beginDrain();
    EXPECT_FALSE(server_->waitForDrain(std::chrono::steady_clock::now()));
    queued_work.process();
    EXPECT_EQ(calls.load(), 1);
    EXPECT_TRUE(server_->waitForDrain(std::chrono::steady_clock::now()));
    EXPECT_TRUE(server_->finishStop());
}

TEST_F(HttpApiServerTest, HandlerExceptionReleasesQueuedAdmission) {
    auto admission_token = server_->http_server_->_serverAdapter->_requestAdmissionHandler();
    ASSERT_NE(admission_token, nullptr);

    auto request = std::make_shared<http_server::HttpRequest>();
    http_server::ResponseHandler handler = [](std::unique_ptr<http_server::HttpResponseWriter>,
                                              const http_server::HttpRequest&) {
        throw std::runtime_error("handler failed");
    };
    http_server::HttpRequestWorkItem queued_work(handler, nullptr, std::move(request), std::move(admission_token));

    server_->beginDrain();
    EXPECT_FALSE(server_->waitForDrain(std::chrono::steady_clock::now()));
    EXPECT_THROW(queued_work.process(), std::runtime_error);
    EXPECT_TRUE(server_->waitForDrain(std::chrono::steady_clock::now()));
    EXPECT_TRUE(server_->finishStop());
}

TEST_F(HttpApiServerTest, DrainingRequestIsRejectedBeforeEnteringTheWorkerQueue) {
    server_->beginDrain();

    auto response = std::make_shared<std::promise<std::pair<bool, std::string>>>();
    auto future   = response->get_future();
    http_server::HttpCallBack callback = [response](bool ok, const std::string& body) {
        response->set_value({ok, body});
    };

    http_server::SimpleHttpClient client;
    ASSERT_TRUE(client.post(server_->getListenAddr(), "/tokenizer/encode", "{}", std::move(callback)));
    ASSERT_EQ(future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    const auto [ok, body] = future.get();
    EXPECT_FALSE(ok);
    EXPECT_EQ(body, R"({"detail":"server is draining"})");
    EXPECT_TRUE(server_->waitForDrain(std::chrono::steady_clock::now()));
    EXPECT_TRUE(server_->finishStop());
}

TEST_F(HttpApiServerTest, IsEmbedding_InferenceService) {
    const auto        port = autil::NetUtil::randomPort();
    const std::string addr = "tcp:0.0.0.0:" + std::to_string(port);
    EngineInitParams  params;
    py::object        token_processor;
    HttpApiServer     server(nullptr, nullptr, addr, params, token_processor);
    ASSERT_TRUE(server.start());
    ASSERT_NE(server.inference_service_, nullptr);
    ASSERT_EQ(server.embedding_service_, nullptr);
    server.stop();
}

TEST_F(HttpApiServerTest, IsEmbedding_EmbeddingService) {
    const auto        port = autil::NetUtil::randomPort();
    const std::string addr = "tcp:0.0.0.0:" + std::to_string(port);
    EngineInitParams  params;
    py::object        py_render;
    HttpApiServer     server(nullptr, nullptr, params, py_render);
    ASSERT_TRUE(server.start(addr));
    ASSERT_EQ(server.inference_service_, nullptr);
    ASSERT_NE(server.embedding_service_, nullptr);
    server.stop();
}

// -------------------------- HealthService Test --------------------------

TEST_F(HttpApiServerTest, testRegisterHealthServiceFailed_HttpServerIsNull) {
    server_->http_server_ = nullptr;
    EXPECT_FALSE(server_->registerHealthService());
}

TEST_F(HttpApiServerTest, testRegisterHealthServiceFailed_RegisterRouteFailed) {
    // 将 http server 的 router 置空, 模拟 RegisterRoute 失败
    server_->http_server_->_router = nullptr;
    EXPECT_FALSE(server_->registerHealthService());
}

TEST_F(HttpApiServerTest, testRegisterHealthServiceSuccess) {
    EXPECT_TRUE(server_->registerHealthService());
}

// -------------------------- WorkerStatusService Test --------------------------

TEST_F(HttpApiServerTest, testRegisterWorkerStatusServiceFailed_HttpServerIsNull) {
    server_->http_server_ = nullptr;
    EXPECT_FALSE(server_->registerWorkerStatusService());
}

TEST_F(HttpApiServerTest, testRegisterWorkerStatusServiceFailed_RegisterRouteFailed) {
    // 将 http server 的 router 置空, 模拟 RegisterRoute 失败
    server_->http_server_->_router = nullptr;
    EXPECT_FALSE(server_->registerWorkerStatusService());
}

TEST_F(HttpApiServerTest, testRegisterWorkerStatusServiceSuccess) {
    EXPECT_TRUE(server_->registerWorkerStatusService());
}

TEST_F(HttpApiServerTest, testStop) {
    EXPECT_FALSE(server_->isStoped());
    EXPECT_TRUE(server_->registerHealthService());
    EXPECT_TRUE(server_->health_service_ != nullptr);
    EXPECT_FALSE(server_->health_service_->is_stopped_);

    server_->stop();
    EXPECT_TRUE(server_->health_service_->is_stopped_);
}

// -------------------------- ModelStatusService Test --------------------------

TEST_F(HttpApiServerTest, testRegisterModelStatusServiceFailed_HttpServerIsNull) {
    server_->http_server_ = nullptr;
    EXPECT_FALSE(server_->registerModelStatusService());
}

TEST_F(HttpApiServerTest, testRegisterModelStatusServiceFailed_RegisterRouteFailed) {
    // 将 http server 的 router 置空, 模拟 RegisterRoute 失败
    server_->http_server_->_router = nullptr;
    EXPECT_FALSE(server_->registerModelStatusService());
}

TEST_F(HttpApiServerTest, testRegisterModelStatusServiceSuccess) {
    EXPECT_TRUE(server_->registerModelStatusService());
}

// -------------------------- SysCmdService Test --------------------------

TEST_F(HttpApiServerTest, testRegisterSysCmdServiceFailed_HttpServerIsNull) {
    server_->http_server_ = nullptr;
    EXPECT_FALSE(server_->registerSysCmdService());
}

TEST_F(HttpApiServerTest, testRegisterSysCmdServiceFailed_RegisterRouteFailed) {
    // 将 http server 的 router 置空, 模拟 RegisterRoute 失败
    server_->http_server_->_router = nullptr;
    EXPECT_FALSE(server_->registerSysCmdService());
}

TEST_F(HttpApiServerTest, testRegisterSysCmdServiceSuccess) {
    EXPECT_TRUE(server_->registerSysCmdService());
}

// -------------------------- TokenizerService Test --------------------------

TEST_F(HttpApiServerTest, testRegisterTokenizerServiceFailed_HttpServerIsNull) {
    server_->http_server_ = nullptr;
    EXPECT_FALSE(server_->registerTokenizerService());
}

TEST_F(HttpApiServerTest, testRegisterTokenizerServiceFailed_RegisterRouteFailed) {
    // 将 http server 的 router 置空, 模拟 RegisterRoute 失败
    server_->http_server_->_router = nullptr;
    EXPECT_FALSE(server_->registerTokenizerService());
}

TEST_F(HttpApiServerTest, testRegisterTokenizerServiceSuccess) {
    EXPECT_TRUE(server_->registerTokenizerService());
}

}  // namespace rtp_llm
