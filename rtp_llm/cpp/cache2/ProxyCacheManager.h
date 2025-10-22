#pragma once

#include "rtp_llm/cpp/cache/CacheManager.h"

namespace rtp_llm {

class ProxyDistKvCache;

class ProxyCacheManager: public CacheManager {
public:
    ProxyCacheManager(const CacheConfig&                 config,
                      rtp_llm::DeviceBase*               device,
                      bool                               warmup           = false,
                      const kmonitor::MetricsReporterPtr metrics_reporter = nullptr,
                      const GptInitParameter&            params           = GptInitParameter{});
    ~ProxyCacheManager() override = default;

    bool initDistKvCache() override;

private:
    std::map<std::string, std::string> genExtraMeta(const std::string& adapter_name) const override;
};

}  // namespace rtp_llm