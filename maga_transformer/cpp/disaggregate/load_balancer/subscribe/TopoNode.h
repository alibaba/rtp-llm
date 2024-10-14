#pragma once

#include <string>

namespace rtp_llm {

class TopoNode {
public:
    TopoNode(const std::string& biz_, const std::string& ip_, uint32_t arpc_port_);

public:
    friend std::ostream& operator<<(std::ostream& os, const TopoNode& node);

public:
    std::string biz;
    std::string ip;
    uint32_t    arpc_port{0};
};

}  // namespace rtp_llm