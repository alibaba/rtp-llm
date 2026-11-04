#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/FlowControlConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceProvider.h"
#include "autil/Lock.h"
namespace multi_call {

struct BizStatistic {
    BizStatistic()
        : expectNum(0)
        , resultNum(0)
        , etThreshold(0)
        , retryThreshold(0)
        , needRetry(false)
        , needSingleRetry(false) {
    }

    uint32_t expectNum;
    uint32_t resultNum;
    uint32_t etThreshold;
    uint32_t retryThreshold;
    bool needRetry;
    bool needSingleRetry;
    void setEtThreshold(uint32_t num) {
        if (num < 1) {
            num = 1;
        } else if (num > expectNum) {
            num = expectNum;
        }
        etThreshold = num;
    }

    void setRetryThreshold(uint32_t num) {
        if (num < 1) {
            num = 1;
        } else if (num > expectNum) {
            num = expectNum;
        }
        retryThreshold = num;
    }
};

class CallDelegationStatistic
{
public:
    typedef std::map<std::string, BizStatistic> CallDelegationStatisticResultMap;

public:
    CallDelegationStatistic();
    ~CallDelegationStatistic();

public:
    void collectStatistic(const std::string &bizName, size_t providerCount,
                          const FlowControlConfigPtr &flowControlConfig, bool disableRetry = false);
    bool hasEnoughResultForEt() const;
    bool hasEnoughResultForRetry(const std::string &clusterName) const;
    bool IsSingleResultNeedRetry(const std::string &bizName) const;
    inline bool hasEnoughResultForRetry(const BizStatistic &stat) const {
        return stat.needRetry && stat.resultNum >= stat.retryThreshold &&
               stat.resultNum != stat.expectNum;
    }
    inline bool IsSingleResultNeedRetry(const BizStatistic &stat) const {
        return stat.needSingleRetry && 1 == stat.expectNum && 0 == stat.resultNum;
    }

    uint32_t getClusterResultNum(const std::string &clusterName) const {
        autil::ScopedReadLock lock(_lock);
        auto it = _callDelegationResultMap.find(clusterName);
        return it == _callDelegationResultMap.end() ? 0 : it->second.resultNum;
    }

    BizStatistic getBizStatistic(const std::string &clusterName) const {
        autil::ScopedReadLock lock(_lock);
        auto it = _callDelegationResultMap.find(clusterName);
        return it == _callDelegationResultMap.end() ? BizStatistic() : it->second;
    }
    void addClusterResult(const std::string &clusterName, uint32_t resultNum = 1) {
        auto it = _callDelegationResultMap.find(clusterName);
        assert(it != _callDelegationResultMap.end());
        autil::ScopedWriteLock lock(_lock);
        it->second.resultNum += resultNum;
        AUTIL_LOG(DEBUG, "cluster %s, result %u, et %u, retry %u", clusterName.c_str(),
                  it->second.resultNum, it->second.etThreshold, it->second.retryThreshold);
    }
    void prepareCollectStatistic(const std::vector<std::string> &bizNameVec);
    uint32_t getEtWaitTimeFactor() const {
        return _etWaitTimeFactor;
    }
    int64_t getEtMinWaitTimeInMicroSeconds() const {
        return _etMinWaitTime * 1000;
    }

private:
    void collectProviderStatistic(const std::string &bizName, size_t providerCount);
    void collectProviderStatistic(const SearchServiceProviderVector &providerVec,
                                  const std::string &bizName);

public:
    // for test
    CallDelegationStatisticResultMap &getStatisticResultMap() {
        return _callDelegationResultMap;
    }
    void setEnableEt(bool flag) {
        _enableEt = flag;
    }
    void setEtWaitTimeFactor(uint32_t factor) {
        _etWaitTimeFactor = factor;
    }

private:
    CallDelegationStatistic(const CallDelegationStatistic &);
    CallDelegationStatistic &operator=(const CallDelegationStatistic &);

private:
    CallDelegationStatisticResultMap _callDelegationResultMap;
    bool _enableEt;
    uint32_t _etWaitTimeFactor;
    uint32_t _etMinWaitTime;
    mutable autil::ReadWriteLock _lock;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(CallDelegationStatistic);

} // namespace multi_call

