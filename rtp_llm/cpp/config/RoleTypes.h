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

