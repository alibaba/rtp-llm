#pragma once

#include "aios/network/arpc/arpc/CommonMacros.h"

namespace accl {
namespace barex {
class XStatsReporter;
}
}  // namespace accl

ARPC_BEGIN_NAMESPACE(arpc);

class ServerMetricReporter;

struct RdmaServerConfig {
    size_t                                       rdmaIoThreadNum{1};
    size_t                                       rdmaWorkerThreadNum{1};
    std::shared_ptr<ServerMetricReporter>        rpcMetricReporter;
    std::shared_ptr<accl::barex::XStatsReporter> acclMetricReporter;
};

ARPC_END_NAMESPACE(arpc);