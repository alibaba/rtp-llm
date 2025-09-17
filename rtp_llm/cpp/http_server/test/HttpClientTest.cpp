#include "gtest/gtest.h"

#include "autil/NetUtil.h"
#include "autil/legacy/jsonizable.h"

#include "http_server/HttpServer.h"
#include "http_client/SimpleHttpClient.h"

namespace http_server {

class WorkerStatusResponse: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("available_concurrency", available_concurrency);
        json.Jsonize("alive", alive);
    }

public:
    int  available_concurrency;
    bool alive;
};

class HttpClientTest: public ::testing::Test {

public:
    void SetUp() override {
        for (int i = 0; i < connection_size; i++) {
            std::string address = "tcp:0.0.0.0:" + std::to_string(autil::NetUtil::randomPort());
            auto        server  = initServer(address);
            http_servers_.insert(std::make_pair(address, server));
        }
    }
    void TearDown() override {
        for (auto& it : http_servers_) {
            if (it.second) {
                it.second->Stop();
                it.second.reset();
            }
        }
    }
    int                                                             connection_size = 20;
    std::map<std::string, std::shared_ptr<http_server::HttpServer>> http_servers_;
    std::shared_ptr<http_server::HttpServer>                        initServer(const std::string& address);
};
std::shared_ptr<http_server::HttpServer> HttpClientTest::initServer(const std::string& address) {
    auto server_ = std::make_shared<http_server::HttpServer>();
    auto route = [](std::unique_ptr<http_server::HttpResponseWriter> writer, const http_server::HttpRequest& request) {
        std::string response = R"del(
        {
            "available_concurrency": 32,
            "available_kv_cache": 18416,
            "total_kv_cache": 18416,
            "alive": true
        })del";
        writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
        writer->AddHeader("Content-Type", "application/json");
        writer->Write(response);
    };
    server_->RegisterRoute("GET", "/worker_status", route);
    server_->Start(address);
    return server_;
}

void processResponse(const std::string& response_body) {
    try {
        WorkerStatusResponse worker_status_response;
        autil::legacy::FromJsonString(worker_status_response, response_body);
        ASSERT_TRUE(worker_status_response.available_concurrency == 32);
    } catch (...) {
        ASSERT_TRUE(0);
    }
}

TEST_F(HttpClientTest, testRequestSuccess) {
    auto http_client_ = std::make_shared<SimpleHttpClient>();
    ASSERT_TRUE(http_client_ != nullptr);

    std::atomic_int finish_cnt       = 0;
    int             http_request_num = 5;
    for (auto& it : http_servers_) {
        for (int i = 0; i < http_request_num; i++) {
            HttpCallBack http_call_back = [&finish_cnt](bool ok, const std::string& response_body) {
                ASSERT_TRUE(ok);
                processResponse(response_body);
                ++finish_cnt;
            };
            ASSERT_TRUE(http_client_->get(it.first, "/worker_status", "", std::move(http_call_back)));
        }
    }
    while (true) {
        if (finish_cnt == http_servers_.size() * http_request_num) {
            break;
        }
    }
}

TEST_F(HttpClientTest, testRequestInvalidAddress) {
    auto http_client_ = std::make_shared<SimpleHttpClient>();
    ASSERT_TRUE(http_client_ != nullptr);

    std::mutex mutex;
    mutex.lock();
    HttpCallBack http_call_back = [&mutex](bool ok, const std::string& response_body) {
        ASSERT_FALSE(ok);
        ASSERT_TRUE(response_body == "");
        mutex.unlock();
    };

    std::string address = "tcp:0.0.0.0:" + std::to_string(autil::NetUtil::randomPort());  // invalid address
    ASSERT_TRUE(http_client_->get(address, "/worker_status", "", std::move(http_call_back)));
    mutex.lock();
    mutex.unlock();
}

TEST_F(HttpClientTest, testRequestInvalidRoute) {
    auto http_client_ = std::make_shared<SimpleHttpClient>();
    ASSERT_TRUE(http_client_ != nullptr);

    auto address = http_servers_.begin()->first;

    std::mutex mutex;
    mutex.lock();
    HttpCallBack http_call_back = [&mutex](bool ok, const std::string& response_body) {
        ASSERT_FALSE(ok);
        ASSERT_TRUE(response_body == std::string("http route not found: [GET: /worker_status_invalid]"));
        mutex.unlock();
    };

    ASSERT_TRUE(http_client_->get(address, "/worker_status_invalid", "", std::move(http_call_back)));
    mutex.lock();
    mutex.unlock();
}

TEST_F(HttpClientTest, testRecycleConnectionInHandlePacket) {
    auto http_client_ = std::make_shared<SimpleHttpClient>();
    ASSERT_TRUE(http_client_ != nullptr);

    auto       address = http_servers_.begin()->first;
    std::mutex mutex;
    mutex.lock();
    HttpCallBack http_call_back = [&mutex](bool ok, const std::string& response_body) {
        sleep(2);
        ASSERT_TRUE(ok);
        processResponse(response_body);
        mutex.unlock();
    };
    ASSERT_TRUE(http_client_->get(address, "/worker_status", "", std::move(http_call_back)));
    ASSERT_TRUE(http_client_->connection_pool_->busy_connection_pool_[address]->size() == 1);
    ASSERT_TRUE(http_client_->connection_pool_->idle_connection_pool_[address]->size() == 0);
    mutex.lock();
    mutex.unlock();
    ASSERT_TRUE(http_client_->connection_pool_->busy_connection_pool_[address]->size() == 0);
    ASSERT_TRUE(http_client_->connection_pool_->idle_connection_pool_[address]->size() == 1);
}

}  // namespace http_server