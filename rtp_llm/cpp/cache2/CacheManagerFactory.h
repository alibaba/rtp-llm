#pragma once

#include <memory>

#include "ProxyCacheManager.h"

namespace rtp_llm {

class CacheManagerFactory {
public:
    static std::shared_ptr<CacheManager>
    createCacheManager(const CacheConfig&                 config,
                       rtp_llm::DeviceBase*               device,
                       bool                               warmup           = false,
                       const kmonitor::MetricsReporterPtr metrics_reporter = nullptr,
                       const GptInitParameter&            params           = GptInitParameter{});
};

}  // namespace rtp_llm