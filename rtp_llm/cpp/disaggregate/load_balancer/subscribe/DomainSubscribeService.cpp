#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/DomainSubscribeService.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/NetUtil.h"

namespace rtp_llm {

bool DomainSubscribeService::init(const DomainSubscribeServiceConfig& config) {
    domain_    = config.domain;
    http_port_ = config.http_port;
    rpc_port_  = config.rpc_port;
    inited_    = true;
    return true;
}

bool DomainSubscribeService::isReady() {
    return inited_;
}

bool DomainSubscribeService::getTopoNodes(std::vector<std::shared_ptr<const TopoNode>>& topo_nodes) {
    int                      err;
    std::vector<std::string> ips;
    err = getAddressByName(domain_, ips);
    if (err == 0) {
        for (auto& ip : ips) {
            auto node = std::make_shared<const TopoNode>(domain_, ip, rpc_port_, http_port_);
            topo_nodes.push_back(node);
        }
    } else {
        RTP_LLM_LOG_WARNING("get ips by domain [%s] error, error code: %d", domain_.c_str(), err);
    }
    return true;
}

}  // namespace rtp_llm