#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/load_balancer/WRRLoadBalancer.h"
#include "rtp_llm/cpp/http_server/http_client/SimpleHttpClient.h"
#include "aios/network/anet/connection.h"
#include "rtp_llm/cpp/http_server/http_server/HttpServer.h"
#include "autil/NetUtil.h"
#include "autil/StringUtil.h"

namespace rtp_llm {
class WRRLoadBalancerTest: public ::testing::Test {
public:
    struct ServerInfo {
        std::string                              address;
        std::shared_ptr<http_server::HttpServer> server;
    };
    std::vector<ServerInfo>          http_servers_;
    std::shared_ptr<WRRLoadBalancer> load_balancer_;

public:
    void initTest() {
        auto config    = makeConfig();
        load_balancer_ = std::make_shared<WRRLoadBalancer>();
        ASSERT_TRUE(load_balancer_->init(config));
    }
    LoadBalancerInitParams makeConfig();
    void                   initServer(const std::string& spec, int available_kv_cache);
};
LoadBalancerInitParams WRRLoadBalancerTest::makeConfig() {
    LocalSubscribeServiceConfig local_config;
    for (int i = 0; i < 10; i++) {
        uint32_t http_port = autil::NetUtil::randomPort();
        local_config.nodes.emplace_back(
            LocalNodeJsonize("test-biz", "0.0.0.0", autil::NetUtil::randomPort(), http_port));
        initServer("tcp:0.0.0.0:" + std::to_string(http_port), 200);
    }
    for (int i = 10; i < 30; i++) {
        uint32_t http_port = autil::NetUtil::randomPort();
        local_config.nodes.emplace_back(
            LocalNodeJsonize("test-biz", "0.0.0.0", autil::NetUtil::randomPort(), http_port));
        initServer("tcp:0.0.0.0:" + std::to_string(http_port), 400);
    }
    SubscribeServiceConfig config;
    config.local_configs.push_back(local_config);
    LoadBalancerInitParams params;
    params.update_interval_ms = 100;
    params.sync_status_interval_ms = 10;
    params.subscribe_config = config;
    return params;
}

void WRRLoadBalancerTest::initServer(const std::string& spec, int available_kv_cache) {
    auto server = std::make_shared<http_server::HttpServer>();
    auto route  = [available_kv_cache](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                      const http_server::HttpRequest&                  request) {
        std::string response_format = R"del(
        {
            "available_concurrency": 32,
            "available_kv_cache": %s,
            "total_kv_cache": 18416,
            "step_latency_ms": 29.23,
            "step_per_minute": 2052,
            "onflight_requests": 0,
            "iterate_count": 1,
            "version": 0,
            "alive": true
        })del";
        auto response = autil::StringUtil::formatString(response_format, std::to_string(available_kv_cache).c_str());
        writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
        writer->AddHeader("Content-Type", "application/json");
        writer->Write(response);
    };
    server->RegisterRoute("GET", "/worker_status", route);
    ASSERT_TRUE(server->Start(spec));
    ServerInfo server_info;
    server_info.address = spec;
    server_info.server  = server;
    http_servers_.push_back(server_info);
}

TEST_F(WRRLoadBalancerTest, testSyncWorkStatus) {
    initTest();
    sleep(5);
    {
        std::shared_lock<std::shared_mutex> lock(load_balancer_->host_load_balance_info_map_mutex_);
        ASSERT_EQ(load_balancer_->host_load_balance_info_map_.size(), 30);
        for (int i = 0; i < 10; i++) {
            auto server_info = http_servers_[i];
            ASSERT_TRUE(load_balancer_->host_load_balance_info_map_[server_info.address].load_balance_info.available_kv_cache == 200);
        }
        for (int i = 10; i < 30; i++) {
            auto server_info = http_servers_[i];
            ASSERT_TRUE(load_balancer_->host_load_balance_info_map_[server_info.address].load_balance_info.available_kv_cache == 400);
        }
    }
}

TEST_F(WRRLoadBalancerTest, testChooseHost) {
    initTest();
    sleep(5);
    int count_200 = 0, count_400 = 0;
    for (int i = 0; i < 1000; i++) {
        auto ret_host = load_balancer_->chooseHost("test-biz");
        ASSERT_TRUE(ret_host != nullptr);
        auto address = "tcp:" + ret_host->ip + ":" + std::to_string(ret_host->http_port);
        if (load_balancer_->host_load_balance_info_map_[address].load_balance_info.available_kv_cache == 200) {
            count_200++;
        }
        if (load_balancer_->host_load_balance_info_map_[address].load_balance_info.available_kv_cache == 400) {
            count_400++;
        }
    }
    ASSERT_TRUE(count_200 + count_400 == 1000);
    double count_200_rate = (double)count_200 / 1000;
    ASSERT_TRUE(count_200_rate >= 0.15 && count_200_rate <= 0.25);
}

TEST_F(WRRLoadBalancerTest, testChangeWeight) {
    // 10 host* 200 available kv cache, 20 host* 400 available kv cache
    autil::EnvUtil::setEnv("WRR_AVAILABLE_RATIO", "0");
    initTest();

    sleep(5);
    int count_200 = 0, count_400 = 0;
    for (int i = 0; i < 1000; i++) {
        auto ret_host = load_balancer_->chooseHost("test-biz");
        ASSERT_TRUE(ret_host != nullptr);
        auto address = "tcp:" + ret_host->ip + ":" + std::to_string(ret_host->http_port);
        if (load_balancer_->host_load_balance_info_map_[address].load_balance_info.available_kv_cache == 200) {
            count_200++;
        }
        if (load_balancer_->host_load_balance_info_map_[address].load_balance_info.available_kv_cache == 400) {
            count_400++;
        }
    }
    ASSERT_TRUE(count_200 + count_400 == 1000) << count_200 << count_400;
    double count_200_rate = (double)count_200 / 1000;
    //(10*200)/(10*200+20*400)=0.2
    ASSERT_TRUE(count_200_rate >= 0.15 && count_200_rate <= 0.25);

    // stop 20 host* 400 vailable kv cache server
    for (int i = 10; i < 30; i++) {
        auto server_info = http_servers_[i];
        ASSERT_TRUE(server_info.server->Stop());
    }
    sleep(5);
    // all in 10 host* 200 vailable kv cache server
    count_200 = 0;
    for (int i = 0; i < 1000; i++) {
        auto ret_host = load_balancer_->chooseHost("test-biz");
        ASSERT_TRUE(ret_host != nullptr);
        auto address = "tcp:" + ret_host->ip + ":" + std::to_string(ret_host->http_port);
        if (load_balancer_->host_load_balance_info_map_[address].load_balance_info.available_kv_cache == 200) {
            count_200++;
        }
    }
    ASSERT_TRUE(count_200 == 1000);

    // 10 host* 200 available kv cache, 20 host* 300 available kv cache
    for (int i = 10; i < 30; i++) {
        auto server_info = http_servers_[i];
        initServer(server_info.address, 300);
    }
    sleep(5);

    int count_300 = 0;
    count_200     = 0;
    for (int i = 0; i < 1000; i++) {
        auto ret_host = load_balancer_->chooseHost("test-biz");
        ASSERT_TRUE(ret_host != nullptr);
        auto address = "tcp:" + ret_host->ip + ":" + std::to_string(ret_host->http_port);
        if (load_balancer_->host_load_balance_info_map_[address].load_balance_info.available_kv_cache == 200) {
            count_200++;
        }
        if (load_balancer_->host_load_balance_info_map_[address].load_balance_info.available_kv_cache == 300) {
            count_300++;
        }
    }
    ASSERT_TRUE(count_200 + count_300 == 1000);
    count_200_rate = (double)count_200 / 1000;
    //(10*200)/(10*200+20*300)=0.25
    ASSERT_TRUE(count_200_rate >= 0.20 && count_200_rate <= 0.30) << count_200_rate;

    // stop all server
    for (auto& server_info : http_servers_) {
        ASSERT_TRUE(server_info.server->Stop());
    }
    sleep(5);
    ASSERT_TRUE(load_balancer_->host_load_balance_info_map_.size() == 0);
    // choose host by rr
    for (int i = 0; i < 1000; i++) {
        auto ret_host = load_balancer_->chooseHost("test-biz");
        ASSERT_TRUE(ret_host != nullptr);
    }
}

}  // namespace rtp_llm