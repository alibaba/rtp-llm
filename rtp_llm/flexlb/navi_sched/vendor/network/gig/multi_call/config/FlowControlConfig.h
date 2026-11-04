#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/ControllerParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "autil/legacy/jsonizable.h"

namespace multi_call {

class FlowControlConfig;
MULTI_CALL_TYPEDEF_PTR(FlowControlConfig);

typedef std::map<std::string, FlowControlConfigPtr> FlowControlConfigMap;
MULTI_CALL_TYPEDEF_PTR(FlowControlConfigMap);

class FlowControlConfig : public autil::legacy::Jsonizable
{
public:
    FlowControlConfig()
        : probePercent(DEFAULT_PROBE_PERCENT)
        , errorRatioLimit(MAX_PERCENT)
        , latencyUpperLimitPercent(DEFAULT_LATENCY_UPPER_LIMIT_PERCENT)
        , latencyUpperLimitMs(DEFAULT_LATENCY_UPPER_LIMIT_MS)
        , beginServerDegradeLatency(DEFAULT_DEGRADE_LATENCY)
        , beginDegradeLatency(DEFAULT_DEGRADE_LATENCY)
        , fullDegradeLatency(DEFAULT_DEGRADE_LATENCY)
        , etTriggerPercent(MAX_PERCENT)
        , etWaitTimeFactor(DEFAULT_WAIT_TIME_FACTOR)
        , etMinWaitTime(0)
        , retryTriggerPercent(MAX_PERCENT)
        , retryWaitTimeFactor(DEFAULT_WAIT_TIME_FACTOR)
        , retryMinWaitTime(0)
        , retryMinProviderWeight(0)
        , retryLimitPerSecond(DEFAULT_RETRY_LIMIT_PER_SECOND)
        , latencyTimeWindowSize(DEFAULT_LATENCY_TIME_WINDOW_SIZE)
        , beginServerDegradeErrorRatio(MAX_PERCENT)
        , beginDegradeErrorRatio(MAX_PERCENT)
        , fullDegradeErrorRatio(MAX_PERCENT)
        , minWeight(MIN_WEIGHT_FLOAT)
        , overrideClientConfig(false) {
    }
    ~FlowControlConfig() {
    }

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper &json) {
        json.Jsonize("probe_percent", probePercent, probePercent);
        json.Jsonize("error_ratio_limit", errorRatioLimit, errorRatioLimit);
        json.Jsonize("latency_upper_limit_percent", latencyUpperLimitPercent,
                     latencyUpperLimitPercent);
        json.Jsonize("latency_upper_limit_ms", latencyUpperLimitMs, latencyUpperLimitMs);
        json.Jsonize("full_degrade_latency", fullDegradeLatency, fullDegradeLatency);
        json.Jsonize("begin_degrade_latency", beginDegradeLatency, fullDegradeLatency);
        json.Jsonize("begin_server_degrade_latency", beginServerDegradeLatency,
                     beginDegradeLatency);
        json.Jsonize("et_trigger_percent", etTriggerPercent, etTriggerPercent);
        json.Jsonize("et_wait_time_factor", etWaitTimeFactor, etWaitTimeFactor);
        json.Jsonize("et_min_wait_time", etMinWaitTime, etMinWaitTime);

        json.Jsonize("retry_trigger_percent", retryTriggerPercent, retryTriggerPercent);
        json.Jsonize("retry_wait_time_factor", retryWaitTimeFactor, retryWaitTimeFactor);
        json.Jsonize("retry_min_wait_time", retryMinWaitTime, retryMinWaitTime);
        json.Jsonize("retry_min_provider_weight", retryMinProviderWeight, retryMinProviderWeight);
        json.Jsonize("retry_limit_per_second", retryLimitPerSecond, retryLimitPerSecond);
        json.Jsonize("latency_time_window_size", latencyTimeWindowSize, latencyTimeWindowSize);

        json.Jsonize("full_degrade_error_ratio", fullDegradeErrorRatio, fullDegradeErrorRatio);
        json.Jsonize("begin_degrade_error_ratio", beginDegradeErrorRatio, fullDegradeErrorRatio);
        json.Jsonize("begin_server_degrade_error_ratio", beginServerDegradeErrorRatio,
                     beginDegradeErrorRatio);

        json.Jsonize("min_weight", minWeight, minWeight);
        json.Jsonize("cooldown_after_reset_ms", cooldownAfterResetMs, cooldownAfterResetMs);
        json.Jsonize("override_client_config", overrideClientConfig, overrideClientConfig);

        json.Jsonize("request_info_field", compatibleFieldInfo.requestInfoField,
                     compatibleFieldInfo.requestInfoField);
        json.Jsonize("ec_field", compatibleFieldInfo.ecField, compatibleFieldInfo.ecField);
        json.Jsonize("response_info_field", compatibleFieldInfo.responseInfoField,
                     compatibleFieldInfo.responseInfoField);

        if (json.GetMode() == FROM_JSON) {
            FlowControlConfigMapPtr tmpConfigMap;
            json.Jsonize("sub_config_map", tmpConfigMap, tmpConfigMap);
            if (tmpConfigMap && !tmpConfigMap->empty()) {
                subConfigMap = tmpConfigMap;
            }
            validate();
        } else {
            if (subConfigMap) {
                json.Jsonize("sub_config_map", subConfigMap, subConfigMap);
            }
        }
    }
    bool etEnabled() const {
        return MAX_PERCENT != etTriggerPercent;
    }
    bool retryEnabled() const {
        return MAX_PERCENT != retryTriggerPercent;
    }
    bool singleRetryEnabled() const {
        return latencyTimeWindowSize > DEFAULT_LATENCY_TIME_WINDOW_SIZE;
    }
    FlowControlConfig *clone();
    void validate();

public:
    float probePercent; // percent
    float errorRatioLimit;
    float latencyUpperLimitPercent;     // percent
    float latencyUpperLimitMs;          // ms
    uint32_t beginServerDegradeLatency; // ms
    uint32_t beginDegradeLatency;       // ms
    uint32_t fullDegradeLatency;        // ms

    float etTriggerPercent;
    uint32_t etWaitTimeFactor;
    uint32_t etMinWaitTime; // ms

    float retryTriggerPercent;
    uint32_t retryWaitTimeFactor;
    uint32_t retryMinWaitTime; // ms
    int32_t retryMinProviderWeight;
    int32_t retryLimitPerSecond;
    int32_t latencyTimeWindowSize;

    float beginServerDegradeErrorRatio;
    float beginDegradeErrorRatio;
    float fullDegradeErrorRatio;

    float minWeight;
    int64_t cooldownAfterResetMs = 0; // ms, 0 means disabled
    bool overrideClientConfig = false;

    CompatibleFieldInfo compatibleFieldInfo;
    // for heartbeat publish
    FlowControlConfigMapPtr subConfigMap;

private:
    static float getValidPercent(float in, const std::string &name) {
        if (in < 0.0) {
            AUTIL_LOG(WARN, "%s [%f] is less than 0", name.c_str(), in);
            in = 0.0;
        } else if (in > MAX_PERCENT) {
            AUTIL_LOG(WARN, "%s [%f] is great than 1.0", name.c_str(), in);
            in = MAX_PERCENT;
        }
        return in;
    }
    int32_t getValidInteger(int32_t in, const std::string &name) {
        if (in < 0) {
            in = 0;
            AUTIL_LOG(WARN, "%s integer [%d] is less than 0", name.c_str(), in);
        }
        return in;
    }

private:
    AUTIL_LOG_DECLARE();
};

} // namespace multi_call
