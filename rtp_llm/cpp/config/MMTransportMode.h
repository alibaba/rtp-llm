#pragma once

#include <stdexcept>
#include <string>

namespace rtp_llm {

inline constexpr const char* kMMTransportModeGrpc = "grpc";
inline constexpr const char* kMMTransportModeRdma = "rdma";
inline constexpr const char* kMMTransportModeKvcm = "kvcm";

inline std::string validateMMTransportMode(std::string mode) {
    if (mode != kMMTransportModeGrpc && mode != kMMTransportModeRdma && mode != kMMTransportModeKvcm) {
        throw std::invalid_argument("invalid mm transport mode '" + mode + "'; expected 'grpc', 'rdma' or 'kvcm'");
    }
    return mode;
}

}  // namespace rtp_llm
