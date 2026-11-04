#pragma once

#include <string>
#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/WorkerNodeInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

class SubscribeService
{
public:
    SubscribeService() : _enabled(true) {
    }
    virtual ~SubscribeService() {
    }

private:
    SubscribeService(const SubscribeService &);
    SubscribeService &operator=(const SubscribeService &);

public:
    virtual bool init() = 0;
    virtual SubscribeType getType() = 0;
    virtual bool clusterInfoNeedUpdate() {
        return true;
    }
    virtual bool getClusterInfoMap(TopoNodeVec &topNodeVec, HeartbeatSpecMap &heartbeatSpecs) = 0;
    virtual bool addSubscribe(const std::vector<std::string> &names) {
        return false;
    }
    virtual bool deleteSubscribe(const std::vector<std::string> &names) {
        return false;
    }
    virtual bool checkSubscribeWork() {
        return true;
    }
    virtual bool stopSubscribe() {
        return false;
    }
    virtual void setEnable(bool enable) {
        _enabled = enable;
    }
    virtual bool isEnable() {
        return _enabled;
    }

    virtual std::string getServiceKey() const = 0;

private:
    bool _enabled;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(SubscribeService);

} // namespace multi_call

