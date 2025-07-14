#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/cluster/PrefillLoadBalancer.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/tokenize/RemoteTokenizeModule.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/cluster/test/FakeServer.h"
#include "rtp_llm/cpp/http_server/http_server/HttpServer.h"
#include "autil/NetUtil.h"
#include "autil/StringUtil.h"

using namespace rtp_llm::rtp_llm_master;

namespace rtp_llm {
namespace rtp_llm_master {

class RemoteTokenizeModuleTest: public EngineBaseTest {
public:
    struct ServerInfo {
        std::string                              address;
        std::shared_ptr<http_server::HttpServer> server;
    };

public:
    std::pair<LoadBalancerInitParams, std::shared_ptr<http_server::HttpServer>>
    makeConfig(const std::vector<int>& token_ids) {
        LocalSubscribeServiceConfig local_config;
        uint32_t                    http_port = 8088;
        local_config.nodes.emplace_back(LocalNodeJsonize("test-biz", "0.0.0.0", http_port, http_port));
        auto                   server = initServer("tcp:0.0.0.0:" + std::to_string(http_port), token_ids);
        SubscribeServiceConfig config;
        config.local_configs.push_back(local_config);
        LoadBalancerInitParams params;
        params.update_interval_ms      = 100;
        params.sync_status_interval_ms = 10;
        params.subscribe_config        = config;
        return {params, server};
    }

    std::shared_ptr<http_server::HttpServer> initServer(const std::string& spec, const std::vector<int>& token_ids) {
        auto server = FakeServer::initServer(spec, "[]", "[]", token_ids);
        return server;
    }
};

TEST_F(RemoteTokenizeModuleTest, testSimple) {
    auto [config, server]                              = makeConfig({1, 2, 3, 4, 5});
    auto estimator_config                              = EstimatorConfig();
    estimator_config.estimator_type                    = "local";
    std::shared_ptr<PrefillLoadBalancer> load_balancer = std::make_shared<PrefillLoadBalancer>(20000);
    ASSERT_TRUE(load_balancer->initWithEstimator(config, estimator_config));
    RemoteTokenizeModule module;
    ASSERT_TRUE(module.init(load_balancer));
    std::string request = "{\"prompt\": \"hello\"}";
    sleep(3);
    auto res = module.encodeRequest(request, "test-biz");
    ASSERT_TRUE(res.ok());
    ASSERT_EQ(res.value()->token_ids, std::vector<int>({1, 2, 3, 4, 5}));
    ASSERT_EQ(res.value()->input_length, 5);
}

}  // namespace rtp_llm_master
}  // namespace rtp_llm