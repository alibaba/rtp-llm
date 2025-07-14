#include "gtest/gtest.h"
#include "rtp_llm/cpp/api_server/HttpApiServer.h"
#include "rtp_llm/cpp/api_server/common/HealthService.h"
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

TEST_F(HttpApiServerTest, testApiServerStop) {
    const auto        port = autil::NetUtil::randomPort();
    const std::string addr = "tcp:0.0.0.0:" + std::to_string(port);
    EngineInitParams  params;
    auto              server = std::make_shared<HttpApiServer>(nullptr, nullptr, addr, params, py::none());
    EXPECT_TRUE(server->start());
    server->active_request_count_->inc();
    auto runnable = [server]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        server->active_request_count_->dec();
    };
    std::thread t(runnable);
    server->stop();
    EXPECT_TRUE(server->isStoped());
    t.join();
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
