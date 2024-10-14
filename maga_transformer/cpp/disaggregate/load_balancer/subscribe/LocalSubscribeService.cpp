#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/LocalSubscribeService.h"

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, LocalSubscribeService);

bool LocalSubscribeService::init(const LocalSubscribeServiceConfig& config) {
    for (auto& node_config : config.nodes) {
        if (!node_config.validate()) {
            AUTIL_LOG(WARN,
                      "local subscribe service init failed, node config is invalid, config is [%s]",
                      autil::legacy::ToJsonString(node_config).c_str());
            return false;
        }

        auto node = std::make_shared<const TopoNode>(node_config.biz, node_config.ip, node_config.arpc_port);
        nodes_.push_back(node);
    }
    inited_ = true;
    return true;
}

bool LocalSubscribeService::isReady() {
    return inited_;
}

bool LocalSubscribeService::getTopoNodes(std::vector<std::shared_ptr<const TopoNode>>& topo_nodes) {
    for (auto& node : nodes_) {
        topo_nodes.push_back(node);
    }
    return true;
}

}  // namespace rtp_llm