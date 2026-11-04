#include "kmonitor/client/sink/PrometheusGatewaySink.h"

#include <iostream>
#include <map>
#include <string>

#include "autil/EnvUtil.h"
#include "autil/Log.h"
#include "autil/StringUtil.h"
#include "kmonitor/client/common/Common.h"
#include "kmonitor/client/core/MetricsTags.h"
#include "kmonitor/client/core/MetricsValue.h"
#include "kmonitor/client/metric/Metric.h"

BEGIN_KMONITOR_NAMESPACE(kmonitor);
AUTIL_LOG_SETUP(kmonitor, PrometheusGatewaySink);

using namespace autil;
using namespace std;

const char *PrometheusGatewaySink::NAME = "PrometheusGatewaySink";
const string PrometheusGatewaySink::PROMETHEUS_GATEWAY_PORT("9091");

const int PrometheusGatewaySink::HTTP_CLIENT_THREAD_NUM = 1;
const int PrometheusGatewaySink::HTTP_CLIENT_THREAD_QUEUE_SIZE = 100;
const int PrometheusGatewaySink::HTTP_CLIENT_TIMEOUT_MS = 1000;

PrometheusGatewaySink::PrometheusGatewaySink(const string &addr,
                                             const string &name,
                                             bool enableLogFileSink,
                                             const MetricsNameMapping *nameMapping)
    : Sink(PrometheusGatewaySink::NAME), nameMapping_(nameMapping) {
    gatewayAddr = addr;
    jobName = name;
    if (gatewayAddr.empty()) {
        gatewayAddr = getDefaultSinkAddr();
    }
    httpClient = make_unique<HttpClientAsync>(
        EnvUtil::getEnv("KMONITOR_HTTP_CLIENT_THREAD_NUM", HTTP_CLIENT_THREAD_NUM),
        EnvUtil::getEnv("KMONITOR_HTTP_CLIENT_THREAD_QUEUE_SIZE", HTTP_CLIENT_THREAD_QUEUE_SIZE),
        EnvUtil::getEnv("KMONITOR_HTTP_CLIENT_TIMEOUT_MS", HTTP_CLIENT_TIMEOUT_MS));
    parseMetricMeta();
    if (enableLogFileSink) {
        AUTIL_LOG(INFO, "enable output metrics to log file");
        logFileClient = new LogFileAgentClient();
    } else {
        AUTIL_LOG(INFO, "disable output metrics to log file");
        logFileClient = nullptr;
    }
    AUTIL_LOG(INFO, "Init PrometheusGatewaySink with gatewayAddr: %s", gatewayAddr.c_str())
}

string PrometheusGatewaySink::getDefaultSinkAddr() const { return "localhost:" + PROMETHEUS_GATEWAY_PORT; }

PrometheusGatewaySink::~PrometheusGatewaySink() {
    if (logFileClient != nullptr) {
        delete logFileClient;
        logFileClient = nullptr;
    }
}

void PrometheusGatewaySink::parseMetricMeta() {
    reportPath = "/metrics/job/" + jobName;
    instanceTagName = EnvUtil::getEnv("KMONITOR_PROM_INSTANCE_TAG_NAME", "host");
    whiteMetricPatterns = StringUtil::split(EnvUtil::getEnv("KMONITOR_PROM_WHITE_METRIC_PATTERNS"), ",");
    blackMetricPatterns = StringUtil::split(EnvUtil::getEnv("KMONITOR_PROM_BLACK_METRIC_PATTERNS"), ",");
    vector<string> tags = StringUtil::split(EnvUtil::getEnv("KMONITOR_PROM_LABEL_TAGS"), ",");
    labelTags.insert(tags.begin(), tags.end());
    tags = StringUtil::split(EnvUtil::getEnv("KMONITOR_PROM_IGNORE_TAGS"), ",");
    ignoreTags.insert(tags.begin(), tags.end());
}

bool PrometheusGatewaySink::Init() {
    if (logFileClient != nullptr && !logFileClient->Init()) {
        return false;
    }
    return true;
}

void PrometheusGatewaySink::PutMetrics(MetricsRecord *record) {
    auto tagMap = record->Tags()->GetTagsMap();
    string tagString, path;
    getTagStringAndPath(tagMap, tagString, path);

    for (const auto &value : record->Values()) {
        if (!validateMetricName(value->Name())) {
            continue;
        }
        float outputVal = 0;
        if (!validateAndParseValue(value->Value(), outputVal)) {
            continue;
        }

        // 总是汇报原始指标名
        addPrometheusMetric(value->Name(), value, tagString, path, outputVal);

        // 如果有指标映射，同时汇报映射后的指标名
        if (nameMapping_ != nullptr) {
            const string &mappedName = nameMapping_->getMappedName(value->Name());
            // 只有当映射后的名称与原始名称不同时才汇报
            if (mappedName != value->Name()) {
                addPrometheusMetric(mappedName, value, tagString, path, outputVal);
            }
        }
    }
}

void PrometheusGatewaySink::addPrometheusMetric(
    const string &metricName, const MetricsValue *value, const string &tagString, const string &path, float outputVal) {
    string metric = metricName;
    StringUtil::replaceAll(metric, ".", "_");
    stringstream << "# TYPE " << metric << " gauge\n";
    stringstream << metric;
    stringstream << tagString;
    stringstream << " " << outputVal << "\n";
    promRecords.emplace_back(make_pair<string, string>(reportPath + path, stringstream.str()));
    stringstream.str("");
    stringstream.clear();
}

bool PrometheusGatewaySink::validateMetricName(const string &name) {
    if (!whiteMetricPatterns.empty()) {
        for (const auto &pattern : whiteMetricPatterns) {
            if (name.find(pattern) != string::npos) {
                return true;
            }
        }
        return false;
    }
    if (!blackMetricPatterns.empty()) {
        for (const auto &pattern : blackMetricPatterns) {
            if (name.find(pattern) != string::npos) {
                return false;
            }
        }
        return true;
    }
    return true;
}

bool PrometheusGatewaySink::validateAndParseValue(const string &valueStr, float &outValue) {
    if (valueStr.find('{') == 0) {
        return false;
    }
    auto splitVal = StringUtil::split(valueStr, "_");
    if (splitVal.empty() || splitVal[0].empty()) {
        return false;
    }
    if (!StringUtil::fromString<float>(splitVal[0], outValue)) {
        return false;
    }
    if (std::isnan(outValue)) {
        outValue = 0;
    }
    return true;
}

void PrometheusGatewaySink::getTagStringAndPath(const map<string, string> &tagMap, string &tagStr, string &path) {
    if (tagMap.empty()) {
        tagStr = "{}";
        return;
    }
    tagStr.reserve(tagMap.size() * 32);
    tagStr.push_back('{');
    for (const auto &tag : tagMap) {
        if (tag.first == instanceTagName) {
            path += "/instance/" + tag.second;
        }
        if (ignoreTags.count(tag.first) != 0) {
            continue;
        }
        if (labelTags.count(tag.first) != 0) {
            if (tag.second.empty()) {
                path += "/" + tag.first + "@base64/=";
            } else {
                path += "/" + tag.first + "/" + tag.second;
            }
            continue;
        }
        tagStr.append(tag.first).append("=\"").append(tag.second).append("\",");
    }
    if (tagStr.size() > 0 && tagStr[tagStr.size() - 1] == ',') {
        tagStr.erase(tagStr.size() - 1);
    }
    tagStr.push_back('}');
}

void PrometheusGatewaySink::Flush() {
    if (promRecords.empty()) {
        return;
    }
    if (logFileClient) {
        logFileClient->AppendBatch(promRecords);
    }
    for (auto &record : promRecords) {
        AUTIL_LOG(DEBUG, "async post: %s %s", record.first.c_str(), record.second.c_str());
        httpClient->asyncPost(gatewayAddr, std::move(record.first), std::move(record.second));
    }
    promRecords.clear();
}

END_KMONITOR_NAMESPACE(kmonitor);
