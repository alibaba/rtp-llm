#pragma once
#include <string>

namespace rtp_llm {

enum RoleType {
    PDFUSION = 0,
    PREFILL  = 1,
    DECODE   = 2,
    VIT      = 3,
    FRONTEND = 4
};

enum VitSeparation {
    VIT_SEPARATION_LOCAL  = 0,  // Local multimodal processing
    VIT_SEPARATION_ROLE   = 1,  // VIT role (separated VIT process)
    VIT_SEPARATION_REMOTE = 2   // Remote multimodal processing
};

class RoleAddr {
public:
    RoleType    role;
    std::string ip;
    int         http_port;
    int         grpc_port;

    RoleAddr(RoleType type, std::string ip, int http_port, int grpc_port):
        role(type), ip(ip), http_port(http_port), grpc_port(grpc_port) {}
};

}  // namespace rtp_llm

