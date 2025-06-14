#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/LocalSubscribeService.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool LocalSubscribeService::init(const LocalSubscribeServiceConfig& config) {
    for (auto& node_config : config.nodes) {
        if (!node_config.validate()) {
            RTP_LLM_LOG_WARNING("local subscribe service init failed, node config is invalid, config is [%s]",
                           autil::legacy::ToJsonString(node_config).c_str());
            return false;
        }

        auto node = std::make_shared<const TopoNode>(
            node_config.biz, node_config.ip, node_config.rpc_port, node_config.http_port);
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