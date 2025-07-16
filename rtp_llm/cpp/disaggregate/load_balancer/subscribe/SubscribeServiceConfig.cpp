#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceConfig.h"

namespace rtp_llm {

void VIPSubscribeServiceConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) {
    json.Jsonize("jmenv_dom", jmenv_dom, jmenv_dom);
    json.Jsonize("services", clusters, clusters);
}

bool VIPSubscribeServiceConfig::validate() const {
    return !jmenv_dom.empty();
}

void NacosSubscribeServiceConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) {
    json.Jsonize("server_host", server_host, server_host);
    json.Jsonize("clusters", clusters, clusters);
}

bool NacosSubscribeServiceConfig::validate() const {
    return !server_host.empty();
}

void CM2SubscribeServiceConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) {
    json.Jsonize("zk_host", zk_host, zk_host);
    json.Jsonize("zk_path", zk_path, zk_path);
    json.Jsonize("zk_timeout_ms", zk_timeout_ms, zk_timeout_ms);
    json.Jsonize("clusters", clusters, clusters);
}

bool CM2SubscribeServiceConfig::validate() const {
    return !zk_host.empty() && !zk_path.empty() && zk_timeout_ms > 0;
}

LocalNodeJsonize::LocalNodeJsonize(const std::string& biz_,
                                   const std::string& ip_,
                                   uint32_t           rpc_port_,
                                   uint32_t           http_port_):
    biz(biz_), ip(ip_), rpc_port(rpc_port_), http_port(http_port_) {}

void LocalNodeJsonize::Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) {
    json.Jsonize("biz", biz, biz);
    json.Jsonize("ip", ip, ip);
    json.Jsonize("rpc_port", rpc_port, rpc_port);
    json.Jsonize("http_port", http_port, http_port);
}

bool LocalNodeJsonize::validate() const {
    return !biz.empty() && !ip.empty() && rpc_port > 0;
}

void LocalSubscribeServiceConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) {
    json.Jsonize("nodes", nodes, nodes);
}

bool LocalSubscribeServiceConfig::validate() const {
    for (auto& node : nodes) {
        if (!node.validate()) {
            return false;
        }
    }
    return true;
}

void DomainSubscribeServiceConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) {
    json.Jsonize("domain", domain, domain);
    json.Jsonize("httpPort", http_port, http_port);
    json.Jsonize("rpcPort", rpc_port, rpc_port);
}

bool DomainSubscribeServiceConfig::validate() const {
    return !domain.empty() && http_port > 0 && rpc_port > 0;
}

void SubscribeServiceConfig::Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) {
    json.Jsonize("cm2", cm2_configs, cm2_configs);
    json.Jsonize("local", local_configs, local_configs);
    json.Jsonize("nacos", nacos_configs, nacos_configs);
    json.Jsonize("vip", vip_configs, vip_configs);
    json.Jsonize("domain", domain_configs, domain_configs);
}

bool SubscribeServiceConfig::validate() const {
    for (auto& config : cm2_configs) {
        if (!config.validate()) {
            return false;
        }
    }
    for (auto& config : local_configs) {
        if (!config.validate()) {
            return false;
        }
    }
    for (auto& config : nacos_configs) {
        if (!config.validate()) {
            return false;
        }
    }
    for (auto& config : vip_configs) {
        if (!config.validate()) {
            return false;
        }
    }
    for (auto& config : domain_configs) {
        if (!config.validate()) {
            return false;
        }
    }
    return true;
}

}  // namespace rtp_llm