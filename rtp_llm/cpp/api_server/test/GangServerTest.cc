#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "autil/EnvUtil.h"
#include "autil/NetUtil.h"
#include "autil/legacy/jsonizable.h"
#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/api_server/GangServer.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpServer.h"

using namespace ::testing;
namespace rtp_llm {

class GangServerTest: public Test {
public:
    GangServerTest() = default;
    ~GangServerTest() override {
        if (http_server_) {
            http_server_->Stop();
        }
    }

public:
    bool StartLocalHttpServer(int port) {
        std::string address = "tcp:127.0.0.1:" + std::to_string(port);
        http_server_        = std::make_shared<http_server::HttpServer>();
        if (!http_server_->Start(address)) {
            printf("start local http server failed!\n");
            return false;
        }
        return true;
    }

    bool RegisterPost(const std::string& route, const http_server::HttpServer::ResponseHandler& handler) const {
        if (!http_server_) {
            return false;
        }
        return http_server_->RegisterRoute("POST", route, handler);
    }

protected:
    void SetUp() override {
        gang_server_ = std::make_shared<GangServer>(py::none());
    }
    void TearDown() override {}

protected:
    std::shared_ptr<GangServer>              gang_server_;
    std::shared_ptr<http_server::HttpServer> http_server_;
};

TEST_F(GangServerTest, Constructor) {
    EXPECT_TRUE(gang_server_->thread_pool_ != nullptr);
    EXPECT_EQ(gang_server_->thread_pool_->getName(), "GangServerThreadPool");
    EXPECT_EQ(gang_server_->thread_pool_->getThreadNum(), ParallelInfo::globalParallelInfo().getWorldSize());
    EXPECT_EQ(gang_server_->thread_pool_->getQueueSize(), 100);
}

TEST_F(GangServerTest, RequestWorkers_PushWorkItemFailed) {
    const int                                worker_num = 5;
    std::vector<std::pair<std::string, int>> workers;
    for (int i = 0; i < worker_num; ++i) {
        workers.emplace_back(std::make_pair("127.0.0.1", 55555));
    }
    gang_server_->workers_ = workers;

    // 停止线程池, 模拟 push work item 失败的情况
    EXPECT_TRUE(gang_server_->thread_pool_ != nullptr);
    gang_server_->thread_pool_->stop();

    std::map<std::string, std::string> body_map = {
        {"name", "alibaba"},
        {"hello", "world"},
    };
    gang_server_->requestWorkers(body_map, "test_route", true);
}

TEST_F(GangServerTest, RequestWorkers_Success) {
    // 启动 http server
    const int         port  = autil::NetUtil::randomPort();
    const std::string route = "test/route";
    StartLocalHttpServer(port);

    // 模拟 worker 数量
    const int                                worker_num = 5;
    std::vector<std::pair<std::string, int>> workers;
    for (int i = 0; i < worker_num; ++i) {
        workers.emplace_back(std::make_pair("127.0.0.1", port));
    }
    gang_server_->workers_ = workers;

    // 注册路由, 模拟请求转发到 worker 时的处理
    int request_count = 0;
    RegisterPost("/" + route,
                 [&request_count](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                  const http_server::HttpRequest&                  request) {
                     ++request_count;

                     auto body_json_str = request.GetBody();
                     EXPECT_FALSE(body_json_str.empty());
                     std::map<std::string, std::string> body_map;
                     ::autil::legacy::FromJsonString(body_map, body_json_str);
                     EXPECT_TRUE(body_map.count("name"));
                     EXPECT_EQ(body_map.at("name"), "alibaba");
                     EXPECT_TRUE(body_map.count("hello"));
                     EXPECT_EQ(body_map.at("hello"), "world");

                     writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
                     writer->AddHeader("Content-Type", "application/json");
                     writer->Write("success");
                     return;
                 });

    std::map<std::string, std::string> body_map = {
        {"name", "alibaba"},
        {"hello", "world"},
    };
    gang_server_->requestWorkers(body_map, route, true);

    // worker 数量应该与转发到 worker 的请求数一致
    EXPECT_EQ(worker_num, request_count);
}

}  // namespace rtp_llm
