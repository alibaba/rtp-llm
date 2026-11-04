#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace multi_call {

struct VipHost {
    std::string ip;
    uint16_t port = 0;
    int32_t weight = 0;
    bool valid = false;
};

// Implemented and supplied by the embedding application; no transport is included.
class VipSubscriptionClient {
public:
    virtual ~VipSubscriptionClient() = default;
    virtual bool init(const std::string &jmenvDomain, bool queryFullApi) = 0;
    virtual bool queryAllIp(const std::string &domain, std::vector<VipHost> &hosts,
                            uint32_t timeoutMs) = 0;
};

using VipSubscriptionClientPtr = std::shared_ptr<VipSubscriptionClient>;

} // namespace multi_call
