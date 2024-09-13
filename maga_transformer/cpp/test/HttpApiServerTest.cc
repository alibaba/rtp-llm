#include "maga_transformer/cpp/HttpApiServer.h"
#include "maga_transformer/cpp/http_server/http_server/HttpRouter.h"
#include "aios/network/anet/httppacket.h"

#include <gtest/gtest.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

using namespace http_server;

namespace rtp_llm {

class HttpApiServerTest: public ::testing::Test {
public:
    void SetUp() override {
        AUTIL_ROOT_LOG_CONFIG();
        server_ = std::make_shared<HttpApiServer>(nullptr, ft::GptInitParameter(), py::object());
    }
    void TearDown() override {}

    std::shared_ptr<HttpRequest> CreateHttpRequest(const std::string& body) {
        // packet will be released in HttpRequest
        ::anet::HTTPPacket* packet = new ::anet::HTTPPacket();
        packet->setURI("/test?a=b");
        packet->setMethod(::anet::HTTPPacket::HM_GET);
        packet->setBody(body.c_str(), body.length());

        auto request    = std::make_shared<HttpRequest>();
        auto http_error = request->Parse(packet);
        EXPECT_TRUE(http_error.IsOK());
        return request;
    }

    void MockRouteCallback(const std::string&                  method,
                           const std::string&                  endpoint,
                           const std::shared_ptr<HttpRequest>& request) {
        auto callback_opt = server_->http_server_._router->FindRoute(method, endpoint);
        ASSERT_TRUE(callback_opt.has_value());
        auto callback = callback_opt.value();
        auto writer   = std::make_unique<HttpResponseWriter>(nullptr);
        callback(std::move(writer), *request);
    }

private:
    std::shared_ptr<HttpApiServer> server_;
};

TEST_F(HttpApiServerTest, testRegisterRoot) {
    EXPECT_TRUE(server_->registerRoot());
    auto request = CreateHttpRequest("");
    MockRouteCallback("GET", "/", request);
}

TEST_F(HttpApiServerTest, testRegisterHealth) {
    EXPECT_TRUE(server_->registerHealth());
    auto request = CreateHttpRequest("");
    MockRouteCallback("GET", "/health", request);
    MockRouteCallback("POST", "/health", request);
    MockRouteCallback("GET", "/GraphService/cm2_status", request);
    MockRouteCallback("POST", "/GraphService/cm2_status", request);
    MockRouteCallback("GET", "/SearchService/cm2_status", request);
    MockRouteCallback("POST", "/SearchService/cm2_status", request);
    MockRouteCallback("GET", "/status", request);
    MockRouteCallback("POST", "/status", request);
    MockRouteCallback("POST", "/health_check", request);
}

TEST_F(HttpApiServerTest, testRegisterV1Model) {
    EXPECT_TRUE(server_->registerV1Model());
    auto request = CreateHttpRequest("");
    MockRouteCallback("GET", "/v1/models", request);
}

TEST_F(HttpApiServerTest, testRegisterSetDebugLog) {
    EXPECT_TRUE(server_->registerSetDebugLog());
    const std::string body    = R"del(
{
    "debug": true
}
)del";
    auto              request = CreateHttpRequest(body);
    MockRouteCallback("POST", "/set_debug_log", request);
}

TEST_F(HttpApiServerTest, testRegisterSetDebugPrint) {
    EXPECT_TRUE(server_->registerSetDebugPrint());
    const std::string body    = R"del(
{
    "debug": true
}
)del";
    auto              request = CreateHttpRequest(body);
    MockRouteCallback("POST", "/set_debug_print", request);
}

TEST_F(HttpApiServerTest, testRegisterTokenizerEncode) {
    EXPECT_TRUE(server_->registerTokenizerEncode());
    // json 中没有 prompt 字段, 这样就不会调用到 python 方法, 否则程序会崩溃
    const std::string body    = R"del(
{
    "no_prompt": "what is your name",
    "return_offsets_mapping": false
}
)del";
    auto              request = CreateHttpRequest(body);
    MockRouteCallback("POST", "/tokenizer/encode", request);
}

TEST_F(HttpApiServerTest, testRegisterInferenceInternal) {
    EXPECT_TRUE(server_->registerInferenceInternal());
    const std::string body    = R"del(
{
    "prompt": "hello, what is your age",
    "generate_config": {
        "max_new_tokens": 20
    }
}
)del";
    auto              request = CreateHttpRequest(body);
    MockRouteCallback("POST", "/inference_internal", request);
}

}  // namespace rtp_llm