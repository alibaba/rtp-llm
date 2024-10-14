#include "gtest/gtest.h"
#include "maga_transformer/cpp/api_server/HttpApiServer.h"

namespace rtp_llm {

class HttpApiServerTest: public ::testing::Test {
};

TEST_F(HttpApiServerTest, testStart) {
    ft::GptInitParameter params;
    py::object token_processor;
    HttpApiServer server(nullptr, "tcp:0.0.0.0:9999", params, token_processor);
    ASSERT_TRUE(server.start());
    ASSERT_FALSE(server.isStoped());
    ASSERT_EQ(server.getListenAddr(), "tcp:0.0.0.0:9999");
    server.stop();
    ASSERT_TRUE(server.isStoped());
}

} // namespace rtp_llm
