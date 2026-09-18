#pragma once

#include <cstdint>
#include <string>

namespace rtp_llm {

inline bool isValidGrpcPort(int64_t port) {
    return port >= 1 && port <= 65535;
}

inline std::string formatGrpcHostPort(const std::string& host, int64_t port) {
    if (host.empty() || !isValidGrpcPort(port)) {
        return "";
    }
    if (host.front() == '[' || host.back() == ']') {
        if (host.front() == '[' && host.back() == ']') {
            return host + ":" + std::to_string(port);
        }
        return "";
    }
    if (host.find(':') != std::string::npos) {
        return "[" + host + "]:" + std::to_string(port);
    }
    return host + ":" + std::to_string(port);
}

}  // namespace rtp_llm
