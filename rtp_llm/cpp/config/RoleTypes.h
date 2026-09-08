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

inline std::string roleTypeToString(RoleType role) {
    switch (role) {
        case PDFUSION:
            return "PDFUSION";
        case PREFILL:
            return "PREFILL";
        case DECODE:
            return "DECODE";
        case VIT:
            return "VIT";
        case FRONTEND:
            return "FRONTEND";
        default:
            return "UNKNOWN";
    }
}

inline RoleType stringToRoleType(const std::string& role_str) {
    if (role_str == "PDFUSION") {
        return PDFUSION;
    } else if (role_str == "PREFILL") {
        return PREFILL;
    } else if (role_str == "DECODE") {
        return DECODE;
    } else if (role_str == "VIT") {
        return VIT;
    } else if (role_str == "FRONTEND") {
        return FRONTEND;
    }
    return PDFUSION;
}

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
