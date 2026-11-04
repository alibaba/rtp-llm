#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/FlowConfigSnapshot.h"

using namespace std;
using namespace autil::legacy::json;
using namespace autil::legacy;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, FlowConfigSnapshot);

static const std::string DEFAULT_FLOW_CONFIG_NAME = "default_flow_control_strategy";
FlowConfigSnapshot::FlowConfigSnapshot() {
    _configMap = std::make_shared<FlowControlConfigMap>();
    _heartbeatConfigMap = std::make_shared<FlowControlConfigMap>();
    _defaultConfig = std::make_shared<FlowControlConfig>();
}

FlowConfigSnapshot::~FlowConfigSnapshot() {
}

void FlowConfigSnapshot::updateDefaultFlowConfig(const FlowControlConfigPtr &flowControlConfig) {
    if (flowControlConfig) {
        _defaultConfig = flowControlConfig;
    } else {
        _defaultConfig = std::make_shared<FlowControlConfig>();
    }
}

void FlowConfigSnapshot::updateFlowConfig(const std::string &strategy,
                                          const FlowControlConfigPtr &flowControlConfig) {
    if (!flowControlConfig) {
        _configMap->erase(strategy);
    } else {
        if (DEFAULT_FLOW_CONFIG_NAME == strategy) {
            _defaultConfig = flowControlConfig;
        } else {
            (*_configMap)[strategy] = flowControlConfig;
        }
    }
}

void FlowConfigSnapshot::updateHeartbeatConfig(
    const std::shared_ptr<BizFlowControlInfoMap> &bizFlowControlInfoMap) {
    _heartbeatConfigMap->clear();
    if (!bizFlowControlInfoMap || bizFlowControlInfoMap->empty()) {
        return;
    }
    auto &flowControlMap = *_heartbeatConfigMap;
    for (const auto &pair : *bizFlowControlInfoMap) {
        const auto &configPtr = pair.second.config;
        if (!configPtr) {
            continue;
        }
        const auto &name = pair.first;
        if (configPtr->subConfigMap) {
            for (const auto &subPair : *configPtr->subConfigMap) {
                if (subPair.second) {
                    flowControlMap.emplace(getSubFlowControlStrategy(name, subPair.first),
                                           subPair.second);
                }
            }
        }
        flowControlMap.emplace(name, configPtr);
    }
}

FlowConfigSnapshotPtr FlowConfigSnapshot::clone() {
    auto snapshot = std::make_shared<FlowConfigSnapshot>();
    for (auto iter = _configMap->begin(); iter != _configMap->end(); ++iter) {
        FlowControlConfigPtr configPtr(iter->second->clone());
        (*(snapshot->_configMap))[iter->first] = configPtr;
    }
    for (auto iter = _heartbeatConfigMap->begin(); iter != _heartbeatConfigMap->end(); ++iter) {
        FlowControlConfigPtr configPtr(iter->second->clone());
        (*(snapshot->_heartbeatConfigMap))[iter->first] = configPtr;
    }
    snapshot->_defaultConfig = FlowControlConfigPtr(_defaultConfig->clone());
    return snapshot;
}

FlowControlConfigPtr FlowConfigSnapshot::getFlowControlConfig(const std::string &strategy) const {
    return getFlowControlConfig(strategy, strategy);
}

FlowControlConfigPtr FlowConfigSnapshot::getFlowControlConfig(const std::string &bizName,
                                                              const std::string &strategy) const {
    FlowControlConfigPtr config;
    if (getFlowControlConfig(bizName, strategy, config)) {
        return config;
    } else {
        return _defaultConfig;
    }
}

bool FlowConfigSnapshot::getFlowControlConfig(const std::string &strategy,
                                              FlowControlConfigPtr &config) const {
    return getFlowControlConfig(strategy, strategy, config);
}

bool FlowConfigSnapshot::getFlowControlConfig(const std::string &bizName,
                                              const std::string &strategy,
                                              FlowControlConfigPtr &config) const {
    FlowControlConfigPtr configHeartbeat;
    {
        auto it = _heartbeatConfigMap->find(bizName);
        if (it != _heartbeatConfigMap->end()) {
            configHeartbeat = it->second;
        }
    }
    if (configHeartbeat && configHeartbeat->overrideClientConfig) {
        config = std::move(configHeartbeat);
        return true;
    }
    {
        auto it = _heartbeatConfigMap->find(strategy);
        if (it != _heartbeatConfigMap->end()) {
            configHeartbeat = it->second;
        }
    }
    if (configHeartbeat && configHeartbeat->overrideClientConfig) {
        config = std::move(configHeartbeat);
        return true;
    }
    FlowControlConfigPtr configClient;
    {
        auto it = _configMap->find(strategy);
        if (it != _configMap->end()) {
            configClient = it->second;
        }
    }
    if (configClient) {
        config = std::move(configClient);
        return true;
    } else if (configHeartbeat) {
        config = std::move(configHeartbeat);
        return true;
    }
    return false;
}

void FlowConfigSnapshot::getFlowControlSwitch(const vector<string> &bizNameVec,
                                              const vector<string> &strategyVec,
                                              bool &earlyTermination, bool &retry,
                                              bool &singleRetry) const {
    assert(bizNameVec.size() == strategyVec.size());
    earlyTermination = false;
    retry = false;
    singleRetry = false;
    auto count = bizNameVec.size();
    for (size_t i = 0; i < count; i++) {
        const auto &bizName = bizNameVec[i];
        const auto &strategy = strategyVec[i];
        const auto &configPtr = getFlowControlConfig(bizName, strategy);
        if (configPtr) {
            earlyTermination = earlyTermination || configPtr->etEnabled();
            retry = retry || configPtr->retryEnabled();
            singleRetry = singleRetry || configPtr->singleRetryEnabled();
        }
    }
}

std::string FlowConfigSnapshot::toString() const {
    autil::legacy::json::JsonMap jsonMap;
    jsonMap["user"] = ToJson(_configMap);
    jsonMap["heartbeat"] = ToJson(_heartbeatConfigMap);
    jsonMap["default"] = ToJson(_defaultConfig);
    return ToJsonString(jsonMap, true);
}

} // namespace multi_call
