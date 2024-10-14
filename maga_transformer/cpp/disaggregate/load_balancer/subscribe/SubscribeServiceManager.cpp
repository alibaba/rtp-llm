#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceManager.h"

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, SubscribeServiceManager);

SubscribeServiceManager::~SubscribeServiceManager() {}

bool SubscribeServiceManager::init(const SubscribeServiceConfig& config) {
    if (!config.validate()) {
        AUTIL_LOG(
            ERROR, "subscribe service config is invalid, config is [%s]", autil::legacy::ToJsonString(config).c_str());
        return false;
    }

    for (auto& cm2_config : config.cm2_configs) {
        auto service = SubscribeService::createInstance(cm2_config);
        if (!service) {
            return false;
        }
        subscribe_service_vec_.push_back(service);
    }

    for (auto& local_config : config.local_configs) {
        auto service = SubscribeService::createInstance(local_config);
        if (!service) {
            return false;
        }
        subscribe_service_vec_.push_back(service);
    }

    AUTIL_LOG(INFO,
              "subscribe service manager init success, config is [%s]",
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