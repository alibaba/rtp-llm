#include <atomic>
#include <chrono>
#include <thread>

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

// Returns whether the body parsed into the expected worker status; callers
// record the outcome and assert on the test thread, because the callback runs
// on a network thread where gtest assertions are not allowed.
bool processResponse(const std::string& response_body) {
    try {
        WorkerStatusResponse worker_status_response;
        autil::legacy::FromJsonString(worker_status_response, response_body);
        return worker_status_response.available_concurrency == 32;
    } catch (...) {
        return false;
    }
}

// Bounded wait for an asynchronous condition: a lost callback must fail the
// test instead of hanging the whole gtest binary.
template <typename Predicate>
bool waitUntil(Predicate done, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!done() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return done();
}

TEST_F(HttpClientTest, testRequestSuccess) {
    auto http_client_ = std::make_shared<SimpleHttpClient>();
    ASSERT_TRUE(http_client_ != nullptr);

    const int        http_request_num = 5;
    const int        expected_total   = static_cast<int>(http_servers_.size()) * http_request_num;
    std::atomic<int> finish_cnt{0};
    std::atomic<int> failures{0};
    for (auto& it : http_servers_) {
        for (int i = 0; i < http_request_num; i++) {
            HttpCallBack http_call_back = [&finish_cnt, &failures](bool ok, const std::string& response_body) {
                if (!ok || !processResponse(response_body)) {
                    ++failures;
                }
                ++finish_cnt;
            };
            ASSERT_TRUE(http_client_->get(it.first, "/worker_status", "", std::move(http_call_back)));
        }
    }
    ASSERT_TRUE(waitUntil([&] { return finish_cnt.load() >= expected_total; }, std::chrono::seconds(30)))
        << "only " << finish_cnt.load() << " of " << expected_total << " callbacks arrived";
    ASSERT_EQ(0, failures.load());
}

TEST_F(HttpClientTest, testRequestInvalidAddress) {
    auto http_client_ = std::make_shared<SimpleHttpClient>();
    ASSERT_TRUE(http_client_ != nullptr);

    // Record the callback outcome and assert on the test thread: gtest
    // assertions may only run in the thread that owns the test.
    std::atomic<bool> callback_invoked{false};
    std::atomic<bool> callback_ok{false};
    std::atomic<bool> response_empty{false};
    HttpCallBack      http_call_back = [&](bool ok, const std::string& response_body) {
        callback_ok.store(ok);
        response_empty.store(response_body.empty());
        callback_invoked.store(true);
    };

    // Keep the invalid address out of the ports the fixture already bound:
    // an address that actually listens is not an invalid address at all.
    std::string address;
    do {
        address = "tcp:0.0.0.0:" + std::to_string(autil::NetUtil::randomPort());
    } while (http_servers_.count(address) != 0);

    // An invalid address must never be reported as a successful transfer.
    // Two observations are legitimate: the submission is refused outright
    // (get() returns false because the connection already failed before the
    // packet could be posted) or the submission is accepted and the failure
    // arrives through the callback.  Only a silent success is a bug.
    const bool submitted = http_client_->get(address, "/worker_status", "", std::move(http_call_back));
    if (!submitted) {
        ASSERT_FALSE(callback_ok.load()) << "a refused submission must not report success";
        return;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!callback_invoked.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(callback_invoked.load()) << "no callback for an invalid address";
    ASSERT_FALSE(callback_ok.load());
    ASSERT_TRUE(response_empty.load());
}

TEST_F(HttpClientTest, testRequestInvalidRoute) {
    auto http_client_ = std::make_shared<SimpleHttpClient>();
    ASSERT_TRUE(http_client_ != nullptr);

    auto              address = http_servers_.begin()->first;
    std::atomic<bool> callback_invoked{false};
    std::atomic<bool> callback_ok{true};
    std::atomic<bool> body_matches{false};
    HttpCallBack      http_call_back = [&](bool ok, const std::string& response_body) {
        callback_ok.store(ok);
        body_matches.store(response_body == std::string("http route not found: [GET: /worker_status_invalid]"));
        callback_invoked.store(true);
    };

    ASSERT_TRUE(http_client_->get(address, "/worker_status_invalid", "", std::move(http_call_back)));
    ASSERT_TRUE(waitUntil([&] { return callback_invoked.load(); }, std::chrono::seconds(10)))
        << "no callback for an invalid route";
    ASSERT_FALSE(callback_ok.load());
    ASSERT_TRUE(body_matches.load());
}

TEST_F(HttpClientTest, testRecycleConnectionInHandlePacket) {
    auto http_client_ = std::make_shared<SimpleHttpClient>();
    ASSERT_TRUE(http_client_ != nullptr);

    auto              address = http_servers_.begin()->first;
    std::atomic<bool> callback_invoked{false};
    std::atomic<bool> callback_ok{false};
    std::atomic<bool> parsed{false};
    HttpCallBack      http_call_back = [&](bool ok, const std::string& response_body) {
        // Slow consumer on purpose: the connection must already be recycled
        // before the callback runs, so a slow consumer cannot pin it.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        callback_ok.store(ok);
        parsed.store(processResponse(response_body));
        callback_invoked.store(true);
    };

    ASSERT_TRUE(http_client_->get(address, "/worker_status", "", std::move(http_call_back)));
    ASSERT_TRUE(waitUntil([&] { return callback_invoked.load(); }, std::chrono::seconds(10)))
        << "no callback for a valid request";
    ASSERT_TRUE(callback_ok.load());
    ASSERT_TRUE(parsed.load());
    // Assert the settled pool state instead of racing the network thread right
    // after submission: recycle happens before the callback, so by the time the
    // callback returned the connection must be back in the idle list.
    ASSERT_TRUE(http_client_->connection_pool_->busy_connection_pool_[address]->size() == 0);
    ASSERT_TRUE(http_client_->connection_pool_->idle_connection_pool_[address]->size() == 1);
}

}  // namespace http_server