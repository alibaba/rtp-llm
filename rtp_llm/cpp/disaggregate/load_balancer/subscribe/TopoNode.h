#pragma once

#include <string>

namespace rtp_llm {

class TopoNode {
public:
    TopoNode(const std::string& biz_, const std::string& ip_, uint32_t rpc_port_, uint32_t http_port_ = 0);

public:
    friend std::ostream& operator<<(std::ostream& os, const TopoNode& node);

public:
    std::string biz;
    std::string ip;
    uint32_t    rpc_port{0};
    uint32_t    http_port{0};
};

}  // namespace rtp_llm