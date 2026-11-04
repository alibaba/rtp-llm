#pragma once

#include <unordered_map>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/WarmUpStrategy.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "autil/Lock.h"
#include "autil/LoopThread.h"

namespace multi_call {

class BizStat;

class GigStatistic
{
public:
    GigStatistic(const std::string &logPrefix = "", const std::string &snapshotLogDir = "");
    ~GigStatistic();
    std::shared_ptr<BizStat> getBizStat(const std::string &bizName,
                                        PartIdTy partId = INVALID_PART_ID);
    WarmUpStrategy *getWarmUpStrategy(const std::string &warmUpStrategy);
    void updateWarmUpStrategy(const std::string &warmUpStrategy, int64_t timeoutInSecond,
                              int64_t queryCountLimit);
    void init();
    void start(int64_t smoothIntervalUs);
    void stop(int64_t smoothIntervalUs);
    bool stopped();
    void start(const std::string &bizName, PartIdTy partId = INVALID_PART_ID,
               int64_t smoothIntervalUs = INVALID_SMOOTH_WEIGHT_INTERVAL);
    void stop(const std::string &bizName, PartIdTy partId = INVALID_PART_ID,
              int64_t smoothIntervalUs = INVALID_SMOOTH_WEIGHT_INTERVAL);
    bool stopped(const std::string &bizName, PartIdTy partId = INVALID_PART_ID);
    bool longTimeNoQuery(int64_t second);
    bool longTimeNoQuery(const std::string &bizName, int64_t second);
    const std::string &getLogPrefix() const;
    const std::string &getSnapshotLogDir() const;

private:
    std::shared_ptr<BizStat> newBizStat();
    void updateStartStatAll(bool started, int64_t smoothIntervalUs);
    void updateStartStat(const std::string &bizName, PartIdTy partId, bool started,
                         int64_t smoothIntervalUs);
    void logThread();
    std::vector<std::shared_ptr<BizStat>> getBizStatVec(const std::string &bizName) const;

private:
    static bool longTimeNoQuery(const std::vector<std::shared_ptr<BizStat>> &statVec,
                                int64_t second);
    typedef std::unordered_map<std::string, std::unordered_map<PartIdTy, std::shared_ptr<BizStat>>>
        BizStatMap;

private:
    std::string _logPrefix;
    std::string _snapshotLogDir;
    mutable autil::ReadWriteLock _lock;
    // started stat for nonexist biz
    bool _otherBizStarted;
    int64_t _otherSmoothIntervalUs;
    BizStatMap _statMap;
    autil::ReadWriteLock _warmUpStrategyMapLock;
    std::unordered_map<std::string, WarmUpStrategy *> _warmUpStrategyMap;
    autil::LoopThreadPtr _logThreadPtr;
    std::string _logStr;

private:
    static const uint32_t UPDATE_TIME_INTERVAL = 2 * 1000 * 1000; // 2s
    static const size_t LOG_COUNT = 3000;
};

} // namespace multi_call
