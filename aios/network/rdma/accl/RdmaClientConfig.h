#pragma once

#include "aios/network/arpc/arpc/CommonMacros.h"

namespace accl {
namespace barex {
class XStatsReporter;
}
}  // namespace accl

ARPC_BEGIN_NAMESPACE(arpc);

struct RdmaClientConfig {
    size_t                                         rdmaIoThreadNum{1};
    size_t                                         rdmaWorkerThreadNum{1};
    std::shared_ptr<::accl::barex::XStatsReporter> statsReporter;
};

ARPC_END_NAMESPACE(arpc);