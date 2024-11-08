#include "gtest/gtest.h"
#include "maga_transformer/cpp/api_server/HttpApiServer.h"
#include "autil/NetUtil.h"

namespace rtp_llm {

class HttpApiServerTest: public ::testing::Test {
public:
    HttpApiServerTest()           = default;
    ~HttpApiServerTest() override = default;

protected:
    void SetUp() override {
        const auto           port    = autil::NetUtil::randomPort();
        const std::string    address = "tcp:0.0.0.0:" + std::to_string(port);
        ft::GptInitParameter params;
        server_ = std::make_shared<HttpApiServer>(nullptr, address, params, py::none());
        EXPECT_TRUE(server_->start());
    }
    void TearDown() override {
        server_.reset();
    }

private:
    std::shared_ptr<HttpApiServer> server_;
};

TEST_F(HttpApiServerTest, testStart) {
    ft::GptInitParameter params;
    py::object           token_processor;
    HttpApiServer        server(nullptr, "tcp:0.0.0.0:9999", params, token_processor);
    ASSERT_TRUE(server.start());
    ASSERT_FALSE(server.isStoped());
    ASSERT_EQ(server.getListenAddr(), "tcp:0.0.0.0:9999");
    server.stop();
    ASSERT_TRUE(server.isStoped());
}

// -------------------------- Health Service Test --------------------------

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

// -------------------------- Worker Status Service Test --------------------------

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

}  // namespace rtp_llm
