#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ControllerFeedBack.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/Filter.h"

namespace multi_call {

class AverageLatency;

class ControllerBase
{
public:
    ControllerBase(const std::string &controllerName);
    virtual ~ControllerBase();

private:
    ControllerBase(const ControllerBase &);
    ControllerBase &operator=(const ControllerBase &);

public:
    virtual float controllerOutput() const = 0;

public:
    bool isValid() const;
    int64_t count() const;

    bool hasServerAgent() const;
    void setHasServerAgent(bool value);

    bool serverFilterReady() const;
    void setServerFilterReady(bool value);

    void setServerValue(float value);
    void setServerValueVersion(int64_t version);
    bool serverValueReady() const;
    float serverValue() const;
    int64_t serverValueVersion() const;
    float serverValueDelayMs() const;
    void clearServerValue();

    void setLoadBalanceServerValue(float value);
    float loadBalanceServerValue() const;

    void setServerAvgWeight(float weight);
    float serverAvgWeight() const;

    void updateLocal(float value);
    float localValue() const;

    bool needProbe() const;
    CompareFlag compare(const ControllerBase &rhs, float allow = 0.0f) const;
    bool invalidServerVersion(const AverageLatency &avgLatency) const;

public:
    // for ut
    void setCurrent(float current) {
        _localFilter.setCurrent(current);
    }
    void setFilterCounter(int64_t counter) {
        _localFilter.setCounter(counter);
    }

private:
    std::string _controllerName;
    bool _hasServerAgent;
    bool _serverFilterReady;
    Filter _localFilter;
    int64_t _serverValueVersion;
    float _serverValueDelay;
    float _serverValue;
    float _loadBalanceServerValue;
    float _serverAvgWeight;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ControllerBase);

} // namespace multi_call

