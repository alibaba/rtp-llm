#include "rtp_llm/cpp/metrics/KmonParam.h"

#include "autil/EnvUtil.h"
#include "autil/StringUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

using namespace std;
using namespace autil;

namespace rtp_llm {

// for kmon client
static const std::string KMONITOR_PORT("kmonitorPort");
static const std::string KMONITOR_SINK_ADDRESS("kmonitorSinkAddress");
static const std::string KMONITOR_ENABLE_LOGFILE_SINK("kmonitorEnableLogFileSink");
static const std::string KMONITOR_ENABLE_PROMETHEUS_SINK("kmonitorEnablePrometheusSink");
static const std::string KMONITOR_MANUALLY_MODE("kmonitorManuallyMode");
static const std::string KMONITOR_SERVICE_NAME("kmonitorServiceName");
static const std::string KMONITOR_TENANT("kmonitorTenant");
static const std::string KMONITOR_METRICS_PREFIX("kmonitorMetricsPrefix");
static const std::string KMONITOR_GLOBAL_TABLE_METRICS_PREFIX("kmonitorGlobalTableMetricsPrefix");
static const std::string KMONITOR_TABLE_METRICS_PREFIX("kmonitorTableMetricsPrefix");
static const std::string KMONITOR_METRICS_REPORTER_CACHE_LIMIT("kmonitorMetricsReporterCacehLimit");
static const std::string KMONITOR_TAGS("kmonitorTags");
static const std::string KMONITOR_NORMAL_SAMPLE_PERIOD("kmonitorNormalSamplePeriod");
static const std::string KMONITOR_KEYVALUE_SEP("^");
static const std::string KMONITOR_MULTI_SEP("@");
static const std::string KMONITOR_TABLE_NAME_WHITELIST("kmonitorTableNameWhiteList");

static const std::string SERVICE_NAME("serviceName");
static const std::string AMONITOR_PATH("amonitorPath");
static const std::string PART_ID("partId");
static const std::string HIPPO_SLAVE_IP("HIPPO_SLAVE_IP");
static const std::string ROLE_TYPE("roleType");

KmonParam::KmonParam(): kmonitorEnableLogFileSink(false), kmonitorManuallyMode(false), kmonitorNormalSamplePeriod(1) {}

bool KmonParam::parseKMonitorTags(const string& tagsStr, map<string, string>& tagsMap) {
    auto tagVec = StringUtil::split(tagsStr, KMONITOR_MULTI_SEP);
    for (const auto& tags : tagVec) {
        auto kvVec = StringUtil::split(tags, KMONITOR_KEYVALUE_SEP);
        if (kvVec.size() != 2) {
            RTP_LLM_LOG_ERROR("parse kmonitor tags [%s] failed.", tags.c_str());
            return false;
        }
        StringUtil::trim(kvVec[0]);
        StringUtil::trim(kvVec[1]);
        tagsMap[kvVec[0]] = kvVec[1];
    }
    return true;
}

bool KmonParam::init() {
    serviceName  = autil::EnvUtil::getEnv(SERVICE_NAME, "suez_service");
    amonitorPath = autil::EnvUtil::getEnv(AMONITOR_PATH, "");

    // compatible with ha3 qrs && searcher
    partId       = autil::EnvUtil::getEnv(PART_ID, "");
    hippoSlaveIp = autil::EnvUtil::getEnv(HIPPO_SLAVE_IP, "127.0.0.1");
    roleType     = autil::EnvUtil::getEnv(ROLE_TYPE, "");

    /***
        for kmon
    ***/
    kmonitorPort        = autil::EnvUtil::getEnv(KMONITOR_PORT, "4141");
    kmonitorServiceName = autil::EnvUtil::getEnv(KMONITOR_SERVICE_NAME, "");
    kmonitorSinkAddress =
        autil::EnvUtil::getEnv(KMONITOR_SINK_ADDRESS, autil::EnvUtil::getEnv(HIPPO_SLAVE_IP, "127.0.0.1"));
    kmonitorEnableLogFileSink = autil::EnvUtil::getEnv(KMONITOR_ENABLE_LOGFILE_SINK, kmonitorEnableLogFileSink);
    kmonitorEnablePrometheusSink =
        autil::EnvUtil::getEnv(KMONITOR_ENABLE_PROMETHEUS_SINK, kmonitorEnablePrometheusSink);
    kmonitorManuallyMode              = autil::EnvUtil::getEnv(KMONITOR_MANUALLY_MODE, kmonitorManuallyMode);
    kmonitorTenant                    = autil::EnvUtil::getEnv(KMONITOR_TENANT, "default");
    kmonitorMetricsPrefix             = autil::EnvUtil::getEnv(KMONITOR_METRICS_PREFIX, "");
    kmonitorGlobalTableMetricsPrefix  = autil::EnvUtil::getEnv(KMONITOR_GLOBAL_TABLE_METRICS_PREFIX, "");
    kmonitorTableMetricsPrefix        = autil::EnvUtil::getEnv(KMONITOR_TABLE_METRICS_PREFIX, "");
    kmonitorMetricsReporterCacheLimit = autil::EnvUtil::getEnv(KMONITOR_METRICS_REPORTER_CACHE_LIMIT, "");
    string kmonitorTagsStr            = autil::EnvUtil::getEnv(KMONITOR_TAGS, "");
    if (!kmonitorTagsStr.empty() && !parseKMonitorTags(kmonitorTagsStr, kmonitorTags)) {
        return false;
    }
    kmonitorNormalSamplePeriod = autil::EnvUtil::getEnv(KMONITOR_NORMAL_SAMPLE_PERIOD, 1);

    return true;
}

}  // namespace rtp_llm
