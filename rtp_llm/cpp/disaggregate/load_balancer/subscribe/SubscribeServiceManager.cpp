#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceManager.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceCreator.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

SubscribeServiceManager::~SubscribeServiceManager() {}

bool SubscribeServiceManager::init(const SubscribeServiceConfig& config) {
    if (!config.validate()) {
        RTP_LLM_LOG_ERROR("subscribe service config is invalid, config is [%s]",
                     autil::legacy::ToJsonString(config).c_str());
        return false;
    }

    for (auto& cm2_config : config.cm2_configs) {
        auto service = createInstanceFromCm2Config(cm2_config);
        if (!service) {
            return false;
        }
        subscribe_service_vec_.push_back(service);
    }

    for (auto& local_config : config.local_configs) {
        auto service = createInstanceFromLocalConfig(local_config);
        if (!service) {
            return false;
        }
        subscribe_service_vec_.push_back(service);
    }

    for (auto& nacos_config : config.nacos_configs) {
        auto service = createInstanceFromNacosConfig(nacos_config);
        if (!service) {
            return false;
        }
        subscribe_service_vec_.push_back(service);
    }

    for (auto& vip_config : config.vip_configs) {
        auto service = createInstanceFromVIPConfig(vip_config);
        if (!service) {
            return false;
        }
        subscribe_service_vec_.push_back(service);
    }
    RTP_LLM_LOG_INFO("subscribe service manager init success, config is [%s]",
                autil::legacy::ToJsonString(config, true).c_str());
    return true;
}

bool SubscribeServiceManager::isAllReady() {
    for (auto& service : subscribe_service_vec_) {
        if (!service->isReady()) {
            return false;
        }
    }
    return true;
}

bool SubscribeServiceManager::getTopoNodes(std::vector<std::shared_ptr<const TopoNode>>& topo_nodes) {
    for (auto& service : subscribe_service_vec_) {
        if (!service->getTopoNodes(topo_nodes)) {
            return false;
        }
    }
    return true;
}

}  // namespace rtp_llm