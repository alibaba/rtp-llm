#pragma once

#include <map>
#include <string>

namespace rtp_llm {

struct KmonParam {
public:
    KmonParam();

public:
    bool init();

private:
    static bool parseKMonitorTags(const std::string& tagsStr, std::map<std::string, std::string>& tagsMap);

public:
    std::string serviceName;
    std::string amonitorPath;
    std::string roleType;
    std::string partId;
    std::string hippoSlaveIp;

    std::string                        kmonitorPort;
    std::string                        kmonitorServiceName;
    std::string                        kmonitorSinkAddress;
    bool                               kmonitorEnableLogFileSink    = false;
    bool                               kmonitorEnablePrometheusSink = false;
    bool                               kmonitorManuallyMode         = false;
    std::string                        kmonitorTenant;
    std::string                        kmonitorMetricsPrefix;
    std::string                        kmonitorGlobalTableMetricsPrefix;
    std::string                        kmonitorTableMetricsPrefix;
    std::string                        kmonitorMetricsReporterCacheLimit;
    std::map<std::string, std::string> kmonitorTags;
    int                                kmonitorNormalSamplePeriod;
};

}  // namespace rtp_llm
