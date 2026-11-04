#include "kmonitor/client/MetricsReporterR.h"

#include "autil/StringUtil.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

const std::string MetricsReporterR::RESOURCE_ID = "metrics_reporter_r";

MetricsReporterR::MetricsReporterR() : navi::ExternalResource(RESOURCE_ID, navi::RS_EXTERNAL) {}

MetricsReporterR::MetricsReporterR(const MetricsReporterPtr &metricsReporter)
    : navi::ExternalResource(RESOURCE_ID, navi::RS_EXTERNAL), _reporter(metricsReporter) {}

MetricsReporterR::~MetricsReporterR() {}

const MetricsReporterPtr &MetricsReporterR::getReporter() const { return _reporter; }

MetricsReporterPtr MetricsReporterR::getQueryMetricsReporter(const MetricsTags &tags, const std::string &subPath) {
    if (tags.Size() == 0 && subPath.empty()) {
        return _reporter;
    } else {
        return _reporter->getSubReporter(subPath, tags);
    }
}

REGISTER_RESOURCE(MetricsReporterR);

END_KMONITOR_NAMESPACE(kmonitor);
