#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/TopoNode.h"

namespace rtp_llm {

TopoNode::TopoNode(const std::string& biz_, const std::string& ip_, uint32_t rpc_port_, uint32_t http_port_):
    biz(biz_), ip(ip_), rpc_port(rpc_port_), http_port(http_port_) {}

std::ostream& operator<<(std::ostream& os, const TopoNode& node) {
    os << node.biz << std::string("_") << node.ip << std::string("_") << std::to_string(node.rpc_port);
    return os;
}

}  // namespace rtp_llm