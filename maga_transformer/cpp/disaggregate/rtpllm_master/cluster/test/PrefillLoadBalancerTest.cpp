#include "src/fastertransformer/devices/testing/TestBase.h"
#include "maga_transformer/cpp/disaggregate/rtpllm_master/cluster/PrefillLoadBalancer.h"
#include "maga_transformer/cpp/disaggregate/rtpllm_master/cluster/test/FakeServer.h"
#include "maga_transformer/cpp/http_server/http_server/HttpServer.h"
#include "autil/NetUtil.h"
#include "autil/StringUtil.h"

using namespace rtp_llm::rtp_llm_master;

namespace rtp_llm {
class PrefillLoadBalancerTest: public EngineBaseTest {
public:
    struct ServerInfo {
        std::string                              address;
        std::shared_ptr<http_server::HttpServer> server;
    };
    std::vector<ServerInfo>              http_servers_;
    std::shared_ptr<PrefillLoadBalancer> load_balancer_;

public:
    void SetUp() override {
        EngineBaseTest::SetUp();
        auto config                     = makeConfig();
        auto estimator_config           = EstimatorConfig();
        estimator_config.estimator_type = "local";
        load_balancer_                  = std::make_shared<PrefillLoadBalancer>(20000);
        ASSERT_TRUE(load_balancer_->initWithEstimator(config, estimator_config));
    }
    LoadBalancerInitParams makeConfig();
    void
    initServer(const std::string& spec, const std::string& running_task_list, const std::string& finished_task_list);
};
LoadBalancerInitParams PrefillLoadBalancerTest::makeConfig() {
    LocalSubscribeServiceConfig   local_config;
    std::vector<std::vector<int>> input_length_list  = {{}};
    std::vector<std::vector<int>> prefix_length_list = {{}};
    for (int i = 0; i < 5; i++) {
        std::string running_task_list  = R"([{"prefix_length": 0, "input_length": 10000, "request_id": 1111}])";
        std::string finished_task_list = R"([{"prefix_length": 0, "input_length": 10000, "request_id": 2222}])";
        uint32_t    http_port          = autil::NetUtil::randomPort();
        local_config.nodes.emplace_back(
            LocalNodeJsonize("test-biz", "0.0.0.0", autil::NetUtil::randomPort(), http_port));
        initServer("tcp:0.0.0.0:" + std::to_string(http_port), running_task_list, finished_task_list);
    }
    for (int i = 5; i < 15; i++) {
        std::string running_task_list  = R"([{"prefix_length": 0, "input_length": 20000, "request_id": 3333}])";
        std::string finished_task_list = R"([{"prefix_length": 0, "input_length": 20000, "request_id": 4444}])";
        uint32_t    http_port          = autil::NetUtil::randomPort();
        local_config.nodes.emplace_back(
            LocalNodeJsonize("test-biz", "0.0.0.0", autil::NetUtil::randomPort(), http_port));
        initServer("tcp:0.0.0.0:" + std::to_string(http_port), running_task_list, finished_task_list);
    }
    SubscribeServiceConfig config;
    config.local_configs.push_back(local_config);
    LoadBalancerInitParams params;
    params.update_interval_ms      = 100;
    params.sync_status_interval_ms = 10;
    params.subscribe_config        = config;
    return params;
}

void PrefillLoadBalancerTest::initServer(const std::string& spec,
                                         const std::string& running_task_list,
                                         const std::string& finished_task_list) {
    auto server = FakeServer::initServer(spec, running_task_list, finished_task_list);
    ServerInfo server_info;
    server_info.address = spec;
    server_info.server  = server;
    http_servers_.push_back(server_info);
}

TEST_F(PrefillLoadBalancerTest, testSyncWorkStatus) {
    sleep(3);
    {
        std::unique_lock<std::shared_mutex> lock(load_balancer_->host_load_balance_info_map_mutex_);
        ASSERT_EQ(load_balancer_->worker_map_.size(), 15);
        for (int i = 0; i < 5; i++) {
            auto server_info = http_servers_[i];
            ASSERT_TRUE(load_balancer_->worker_map_[server_info.address].running_task_list().size() == 1);
            ASSERT_TRUE(load_balancer_->worker_map_[server_info.address].running_task_list()[0].input_length == 10000);
        }
        for (int i = 5; i < 15; i++) {
            auto server_info = http_servers_[i];
            ASSERT_TRUE(load_balancer_->worker_map_[server_info.address].running_task_list().size() == 1);
            ASSERT_TRUE(load_balancer_->worker_map_[server_info.address].running_task_list()[0].input_length == 20000);
        }
    }
}

TEST_F(PrefillLoadBalancerTest, testChooseHost) {
    {
        std::unique_lock<std::shared_mutex> lock(load_balancer_->host_load_balance_info_map_mutex_);
        load_balancer_->worker_map_.clear();
    }
    sleep(3);
    for (int i = 0; i < 6; i++) {
        auto ret_host = load_balancer_->chooseHost("test-biz");
        ASSERT_TRUE(ret_host != nullptr);
        auto                                address = "tcp:" + ret_host->ip + ":" + std::to_string(ret_host->http_port);
        std::unique_lock<std::shared_mutex> lock(load_balancer_->host_load_balance_info_map_mutex_);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list()[0].input_length, 10000);
    }
}

TEST_F(PrefillLoadBalancerTest, testChangeWeight) {
    load_balancer_->worker_map_.clear();
    sleep(3);
    for (int i = 0; i < 5; i++) {
        TaskDescription task({std::to_string(i), 0, 20000, {}, {}});
        auto            ret_host = load_balancer_->chooseHostWithTask("test-biz", task);
        ASSERT_TRUE(ret_host.status() == absl::OkStatus());
        auto host    = ret_host.value().host;
        auto address = "tcp:" + host->ip + ":" + std::to_string(host->http_port);
        FT_LOG_INFO("i : %d, addr: %s", i, address.c_str());
        std::unique_lock<std::shared_mutex> lock(load_balancer_->host_load_balance_info_map_mutex_);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list().size(), 1);
        ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list().size(), 1);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list()[0].input_length, 10000);
        ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list()[0].input_length, 20000);
    }
    for (int i = 5; i < 15; i++) {
        TaskDescription task({std::to_string(i), 0, 30000, {}, {}});
        auto            ret_host = load_balancer_->chooseHostWithTask("test-biz", task);
        ASSERT_TRUE(ret_host.status() == absl::OkStatus());
        auto                                host    = ret_host.value().host;
        auto                                address = "tcp:" + host->ip + ":" + std::to_string(host->http_port);
        std::unique_lock<std::shared_mutex> lock(load_balancer_->host_load_balance_info_map_mutex_);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list().size(), 1);
        ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list().size(), 1);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list()[0].input_length, 20000);
        ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list()[0].input_length, 30000);
    }
}

TEST_F(PrefillLoadBalancerTest, testPendingTaskInherit) {
    load_balancer_->worker_map_.clear();
    load_balancer_->pending_task_timeout_ms_ = 20000;
    sleep(3);
    TaskDescription task({"00", 0, 50000, {}, {}});
    auto            ret = load_balancer_->chooseHostWithTask("test-biz", task);
    ASSERT_TRUE(ret.status() == absl::OkStatus());
    auto host    = ret.value().host;
    auto address = "tcp:" + host->ip + ":" + std::to_string(host->http_port);
    {
        std::unique_lock<std::shared_mutex> lock(load_balancer_->host_load_balance_info_map_mutex_);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list().size(), 1);
        ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list().size(), 1);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list()[0].input_length, 10000);
        ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list()[0].input_length, 50000);
    }
    sleep(3);
    {
        std::unique_lock<std::shared_mutex> lock(load_balancer_->host_load_balance_info_map_mutex_);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list().size(), 1);
        ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list().size(), 1);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list()[0].input_length, 10000);
        ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list()[0].input_length, 50000);
    }
}

TEST_F(PrefillLoadBalancerTest, testPendingTaskExpire) {
    load_balancer_->worker_map_.clear();
    load_balancer_->pending_task_timeout_ms_ = 1000;
    sleep(3);
    TaskDescription task({"0", 0, 20000, {}, {}});
    auto            ret = load_balancer_->chooseHostWithTask("test-biz", task);
    ASSERT_TRUE(ret.status() == absl::OkStatus());
    auto host    = ret.value().host;
    auto address = "tcp:" + host->ip + ":" + std::to_string(host->http_port);
    {
        std::unique_lock<std::shared_mutex> lock(load_balancer_->host_load_balance_info_map_mutex_);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list().size(), 1);
        ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list().size(), 1);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list()[0].input_length, 10000);
        ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list()[0].input_length, 20000);
    }
    sleep(3);
    {
        std::unique_lock<std::shared_mutex> lock(load_balancer_->host_load_balance_info_map_mutex_);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list().size(), 1);
        ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list().size(), 0);
        ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list()[0].input_length, 10000);
    }
}

TEST_F(PrefillLoadBalancerTest, testPendingTaskRemove) {
    load_balancer_->worker_map_.clear();
    load_balancer_->pending_task_timeout_ms_ = 20000;
    sleep(3);
    // expire with running task
    {
        TaskDescription task({"1111", 0, 20000, {}, {}});
        auto            ret = load_balancer_->chooseHostWithTask("test-biz", task);
        ASSERT_TRUE(ret.status() == absl::OkStatus());
        auto host    = ret.value().host;
        auto address = "tcp:" + host->ip + ":" + std::to_string(host->http_port);
        {
            std::shared_lock<std::shared_mutex> lock(load_balancer_->host_load_balance_info_map_mutex_);
            ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list().size(), 1);
            ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list().size(), 1);
        }
        sleep(3);
        {
            std::shared_lock<std::shared_mutex> lock(load_balancer_->host_load_balance_info_map_mutex_);
            ASSERT_EQ(load_balancer_->worker_map_[address].running_task_list().size(), 1);
            ASSERT_EQ(load_balancer_->worker_map_[address].pending_task_list().size(), 0);
        }
    }
}

}  // namespace rtp_llm