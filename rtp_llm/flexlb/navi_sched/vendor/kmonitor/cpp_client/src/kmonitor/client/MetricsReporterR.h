#pragma once

#include "kmonitor/client/MetricsReporter.h"
#include "navi/engine/Resource.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

class MetricsReporterR : public navi::ExternalResource {
public:
    MetricsReporterR();
    MetricsReporterR(const MetricsReporterPtr &metricsReporter);
    ~MetricsReporterR();
    MetricsReporterR(const MetricsReporterR &) = delete;
    MetricsReporterR &operator=(const MetricsReporterR &) = delete;

public:
    const MetricsReporterPtr &getReporter() const;
    MetricsReporterPtr getQueryMetricsReporter(const MetricsTags &tags, const std::string &subPath);

public:
    static const std::string RESOURCE_ID;

private:
    MetricsReporterPtr _reporter;
};
NAVI_TYPEDEF_PTR(MetricsReporterR);

END_KMONITOR_NAMESPACE(kmonitor);
