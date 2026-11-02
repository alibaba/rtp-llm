#pragma once

#include "aios/navi_sched/common/NaviSchedCommon.h"
#include "autil/legacy/jsonizable.h"
#include <map>
#include <string>
#include <vector>

namespace navi_sched {

class NaviSchedConfigBuilder {
public:
    using NamedConfigs = std::map<std::string, std::string>;

    static std::string build(const NaviSchedRuntimeConfig &config,
                             const std::vector<std::string> &kernelList,
                             bool createGigClient, const NamedConfigs &kernelConfigs = {},
                             const NamedConfigs &resourceConfigs = {},
                             const std::vector<std::string> &bizResourceList = {},
                             size_t grpcClientIoThreadNum = 0,
                             const std::map<std::string, size_t> &extraTaskQueueThreads = {});

private:
    static autil::legacy::json::JsonArray makeStringArray(
        const std::vector<std::string> &values);
    static autil::legacy::json::JsonArray makeNamedConfigArray(
        const NamedConfigs &configs);
    static autil::legacy::json::JsonMap makeLocalNode(const NaviSchedLocalNode &node);
    static autil::legacy::json::JsonMap makeCm2Subscription(
        const NaviSchedCm2Subscription &config);
    static autil::legacy::json::JsonMap makeSubscribeConfig(
        const NaviSchedRuntimeConfig &config);
};

}
