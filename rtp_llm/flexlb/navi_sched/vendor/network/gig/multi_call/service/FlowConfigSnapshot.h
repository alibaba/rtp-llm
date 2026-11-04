#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/WorkerNodeInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/FlowControlConfig.h"
#include "autil/legacy/json.h"

namespace multi_call {

class FlowConfigSnapshot
{
public:
    FlowConfigSnapshot();
    ~FlowConfigSnapshot();

private:
    FlowConfigSnapshot(const FlowConfigSnapshot &);
    FlowConfigSnapshot &operator=(const FlowConfigSnapshot &);

public: // only for create snapshot for old snapshot
    void updateDefaultFlowConfig(const FlowControlConfigPtr &flowControlConfig);
    void updateHeartbeatConfig(const std::shared_ptr<BizFlowControlInfoMap> &bizFlowControlInfoMap);
    void updateFlowConfig(const std::string &strategy,
                          const FlowControlConfigPtr &flowControlConfig);
    std::shared_ptr<FlowConfigSnapshot> clone();

public:
    FlowControlConfigPtr getFlowControlConfig(const std::string &bizName,
                                              const std::string &strategy) const;
    bool getFlowControlConfig(const std::string &bizName, const std::string &strategy,
                              FlowControlConfigPtr &config) const;
    void getFlowControlSwitch(const std::vector<std::string> &bizNameVec,
                              const std::vector<std::string> &strategyVec, bool &earlyTermination,
                              bool &retry, bool &singleRetry) const;
    const FlowControlConfigMap &getConfigMap() const {
        return *_configMap;
    }
    std::string toString() const;

private:
    FlowControlConfigPtr getFlowControlConfig(const std::string &strategy) const;
    bool getFlowControlConfig(const std::string &strategy, FlowControlConfigPtr &config) const;

private:
    FlowControlConfigMapPtr _configMap;
    FlowControlConfigMapPtr _heartbeatConfigMap;
    FlowControlConfigPtr _defaultConfig;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(FlowConfigSnapshot);
} // namespace multi_call
