#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/NacosSubscribeService.h"
#include "maga_transformer/cpp/utils/Logger.h"

using namespace nacos;

namespace rtp_llm {

bool NacosSubscribeService::init(const NacosSubscribeServiceConfig& config) {
    if (inited_) {
        RTP_LLM_LOG_WARNING("nacos subscribe service init failed, service is already inited");
        return false;
    }

    Properties configProps;
    configProps[PropertyKeyConst::SERVER_ADDR] = config.server_host;
    factory_.reset(NacosFactoryFactory::getNacosFactory(configProps));
    if (factory_ == nullptr) {
        RTP_LLM_LOG_WARNING("nacos subscribe service init failed, factory is null");
        return false;
    }

    naming_service_.reset(factory_->CreateNamingService());
    if (naming_service_ == nullptr) {
        RTP_LLM_LOG_WARNING("nacos subscribe service init failed, naming service is null");
        return false;
    }

    config_ = config;
    inited_ = true;
    return true;
}

bool NacosSubscribeService::isReady() {
    return inited_;
}

bool NacosSubscribeService::getTopoNodes(std::vector<std::shared_ptr<const TopoNode>>& topo_nodes) {
    for (auto cluster : config_.clusters) {
        auto instances = naming_service_->getAllInstances(cluster);
        RTP_LLM_LOG_WARNING("nacos get instance count %u, cluster %s", instances.size(), cluster.c_str());
        for (auto instance : instances) {
            if (instance.enabled && instance.healthy) {
                topo_nodes.push_back(std::make_shared<TopoNode>(cluster, instance.ip, instance.port));
            } else {
                RTP_LLM_LOG_DEBUG("nacos subscribe service get topo nodes failed, instance is not enabled or healthy, instance is [%s:%d]", instance.ip.c_str(), instance.port);
            }
        }
    }
    return true;
} 

}
