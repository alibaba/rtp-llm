#pragma once

#include <stdexcept>
#include <string>

namespace rtp_llm {

inline constexpr const char* kMMTransportModeGrpc = "grpc";
inline constexpr const char* kMMTransportModeRdma = "rdma";

inline std::string validateMMTransportMode(std::string mode) {
    if (mode != kMMTransportModeGrpc && mode != kMMTransportModeRdma) {
        throw std::invalid_argument("invalid mm transport mode '" + mode + "'; expected 'grpc' or 'rdma'");
    }
    return mode;
}

}  // namespace rtp_llm
