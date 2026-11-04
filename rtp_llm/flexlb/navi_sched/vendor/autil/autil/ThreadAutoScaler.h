#pragma once

#include <cstddef>
#include <string>

#include "autil/Lock.h"
#include "autil/Thread.h"

namespace autil {

class ThreadAutoScaler {
public:
    ThreadAutoScaler();
    ~ThreadAutoScaler();

public:
    bool init(const std::string &threadName);
    void stop();
    void wait(size_t index);
    size_t getMaxThreadNum() const;
    size_t getMinThreadNum() const;
    size_t getActiveThreadNum() const;

private:
    bool initByEnv();
    void autoScaleThread();
    bool updateActiveThreadCount();
    void signal();

private:
    static bool calcActiveThreadCount(size_t &activeCount);

private:
    size_t mMinActiveThreadNum;
    size_t mMaxActiveThreadNum;
    volatile size_t mActiveThreadCount;
    autil::ThreadCond *mCpuScaleNotifier;
    bool mStopped;
    autil::ThreadPtr mAutoScaleThread;

private:
    static constexpr size_t DEFAULT_THREAD_NUM_MIN = 20;
    static constexpr size_t DEFAULT_THREAD_NUM_MAX = 96;
    static const std::string THREAD_NUM_RANGE_SEPERATOR;

public:
    static const std::string THREAD_NUM_RANGE_ENV;
};

} // namespace autil
