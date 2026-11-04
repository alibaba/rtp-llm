#pragma once

#include <sstream>
#include <string>

#include "autil/Log.h"
#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/core/MetricsConfig.h"
#include "kmonitor/client/core/MetricsRecord.h"
#include "kmonitor/client/net/HttpClientAsync.h"
#include "kmonitor/client/net/LogFileAgentClient.h"
#include "kmonitor/client/sink/Sink.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);

class PrometheusGatewaySink : public Sink {
public:
    static const char *NAME;
    explicit PrometheusGatewaySink(const std::string &addr,
                                   const std::string &jobName,
                                   bool enableLogFileSink = false,
                                   const MetricsNameMapping *nameMapping = nullptr);
    ~PrometheusGatewaySink() override;
    bool Init() override;
    void PutMetrics(MetricsRecord *record) override;
    void Flush() override;
    bool ReConnect();

private:
    PrometheusGatewaySink(const PrometheusGatewaySink &);
    PrometheusGatewaySink &operator=(const PrometheusGatewaySink &);
    std::string getDefaultSinkAddr() const;
    void getTagStringAndPath(const std::map<std::string, std::string> &tagMap, std::string &tagStr, std::string &path);
    void parseMetricMeta();
    bool validateMetricName(const std::string &name);
    bool validateAndParseValue(const std::string &value, float &outValue);
    void addPrometheusMetric(const std::string &metricName,
                             const MetricsValue *value,
                             const std::string &tagString,
                             const std::string &path,
                             float outputVal);

private:
    std::string gatewayAddr;
    std::string jobName;
    std::string instanceTagName;
    std::string reportPath;
    std::unique_ptr<HttpClientAsync> httpClient;
    std::stringstream stringstream;
    std::vector<std::pair<std::string, std::string>> promRecords;
    std::vector<std::string> whiteMetricPatterns;
    std::vector<std::string> blackMetricPatterns;
    std::set<std::string> labelTags;
    std::set<std::string> ignoreTags;
    LogFileAgentClient *logFileClient = nullptr;
    const MetricsNameMapping *nameMapping_ = nullptr;

private:
    static const int HTTP_CLIENT_THREAD_NUM;
    static const int HTTP_CLIENT_THREAD_QUEUE_SIZE;
    static const int HTTP_CLIENT_TIMEOUT_MS;
    static const int HTTP_CLIENT_BATCH_SIZE;

    static const std::string PROMETHEUS_GATEWAY_PORT;

private:
    AUTIL_LOG_DECLARE();
};

END_KMONITOR_NAMESPACE(kmonitor);
