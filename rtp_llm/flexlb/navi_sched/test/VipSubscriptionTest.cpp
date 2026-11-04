#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <memory>
#include <type_traits>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/SubscribeServiceFactory.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/VIPSubscribeService.h"

namespace {
class TestClient final : public multi_call::VipSubscriptionClient {
public:
    bool init(const std::string &domain, bool queryFullApi) override {
        initializedDomain = domain;
        fullApi = queryFullApi;
        return initSucceeds;
    }
    bool queryAllIp(const std::string &domain, std::vector<multi_call::VipHost> &output,
                    uint32_t timeoutMs) override {
        queriedDomain = domain;
        timeout = timeoutMs;
        if (!querySucceeds) return false;
        output = hosts;
        return true;
    }
    bool initSucceeds = true;
    bool querySucceeds = true;
    bool fullApi = false;
    uint32_t timeout = 0;
    std::string initializedDomain;
    std::string queriedDomain;
    std::vector<multi_call::VipHost> hosts;
};
}

int main() {
    using namespace multi_call;
    static_assert(std::is_abstract_v<VipSubscriptionClient>);
    VipConfig config;
    config.jmenvDom = "test-bootstrap";
    config.queryFullApi = true;
    config.clusters = {"test-domain"};
    SubscribeServiceFactory::setVipClientFactory({});
    auto missing = SubscribeServiceFactory::createVipSub(config);
    assert(missing && !missing->init());
    SubscribeServiceFactory::setVipClientFactory([] { return nullptr; });
    assert(!SubscribeServiceFactory::createVipSub(config)->init());

    auto client = std::make_shared<TestClient>();
    client->hosts = {{"127.0.0.1", 1234, 25, true}, {"127.0.0.2", 2345, 35, false}};
    SubscribeServiceFactory::setVipClientFactory([client] { return client; });
    auto service = SubscribeServiceFactory::createVipSub(config);
    assert(service->init());
    assert(client->initializedDomain == config.jmenvDom && client->fullApi);
    TopoNodeVec nodes;
    HeartbeatSpecMap heartbeats;
    assert(service->getClusterInfoMap(nodes, heartbeats));
    assert(client->queriedDomain == "test-domain" && client->timeout == 10000);
    assert(nodes.size() == 2);
    assert(nodes[0].spec.ip == "127.0.0.1");
    assert(nodes[0].spec.ports[MC_PROTOCOL_GRPC_STREAM] == 1234);
    assert(nodes[0].weight == 25 && nodes[0].isValid);
    assert(nodes[1].weight == 35 && !nodes[1].isValid);

    client->querySucceeds = false;
    nodes.clear();
    assert(!service->getClusterInfoMap(nodes, heartbeats));
    assert(nodes.empty());
    client->querySucceeds = true;
    client->hosts.clear();
    assert(service->getClusterInfoMap(nodes, heartbeats) && nodes.empty());
    client->initSucceeds = false;
    assert(!SubscribeServiceFactory::createVipSub(config)->init());
    SubscribeServiceFactory::setVipClientFactory({});
}
