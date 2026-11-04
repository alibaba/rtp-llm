#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace navi {

// Runtime settings shared by services that create a single Navi biz.
struct NaviVipSubscription {
    std::string jmenvDomain;
    std::vector<std::string> services;
    std::string serviceKey;
    int32_t grpcPortDiff = 0;
};

struct NaviRuntimeConfig {
    size_t threadNum = 64;
    size_t queueSize = 1000;
    size_t processingSize = 100000000;
    bool disablePerf = true;
    bool disableSymbolTable = true;
    NaviVipSubscription vipSubscription;
};

} // namespace navi
