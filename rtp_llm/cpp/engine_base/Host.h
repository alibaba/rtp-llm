#pragma once

namespace rtp_llm {

struct Host {
    std::string ip;
    uint32_t    rpc_port;
    uint32_t    http_port = 0;

    Host(const std::string& ip_, uint32_t rpc_port_, uint32_t http_port_):
        ip(ip_), rpc_port(rpc_port_), http_port(http_port_) {}
    Host(const std::string& ip_, uint32_t rpc_port_): ip(ip_), rpc_port(rpc_port_) {}
};

}  // namespace rtp_llm
