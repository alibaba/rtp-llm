#include "autil/LoopThread.h"

#include "autil/DailyRunMode.h"
#include "autil/Log.h"
#include "autil/TimeUtility.h"

namespace autil {
AUTIL_DECLARE_AND_SETUP_LOGGER(autil, LoopThread);

bool LoopThread::_disableLoopFun = false;
LoopThread::LoopThread(bool strictMode) {
    _run = false;
    _loopInterval = 0;
    _nextTime = 0;
    _strictMode = strictMode;
}

LoopThread::~LoopThread() { stop(); }

LoopThreadPtr
LoopThread::createLoopThread(const std::function<void()> &loopFunction, int64_t loopInterval, bool strictMode) {
    return createLoopThread(loopFunction, loopInterval, std::string(), strictMode);
}

LoopThreadPtr LoopThread::createLoopThread(const std::function<void()> &loopFunction,
                                           int64_t loopInterval /*us*/,
                                           const char *name,
                                           bool strictMode) {
    return createLoopThread(loopFunction, loopInterval, std::string(name), strictMode);
}

LoopThreadPtr LoopThread::createLoopThread(const std::function<void()> &loopFunction,
                                           int64_t loopInterval /*us*/,
                                           const std::string &name,
                                           bool strictMode) {
    if (DailyRunMode::enable()) {
        loopInterval = 300 * 1000;
    }
    LoopThreadPtr ret(new LoopThread(strictMode));

    ret->_name = name;
    ret->_loopFunction = loopFunction;
    ret->_loopInterval = loopInterval;
    ret->_run = true;

    ret->_threadPtr = Thread::createThread(std::bind(&LoopThread::loop, ret.get()), name);

    if (!ret->_threadPtr) {
        AUTIL_LOG(ERROR, "create loop thread fail");
        return LoopThreadPtr();
    }

    return ret;
}

void LoopThread::stop() {
    {
        ScopedLock lock(_cond);
        if (!_threadPtr) {
            return;
        }

        _run = false;
        _cond.signal();
    }

    _threadPtr->join();
    _threadPtr.reset();
}

void LoopThread::runOnce() {
    ScopedLock lock(_cond);
    _nextTime = -1;
    _cond.signal();
}

void LoopThread::loop() {
    AUTIL_LOG(INFO, "loop started for thread[%s] loopInterval[%ld]", _name.c_str(), _loopInterval);
    while (true) {
        int64_t nowTime;
        {
            ScopedLock lock(_cond);
            if (!_run) {
                break;
            }
            nowTime = TimeUtility::monotonicTimeUs();
            if (nowTime < _nextTime) {
                _cond.wait(_nextTime - nowTime);
                if (!_run) {
                    break;
                }
            }

            if (_nextTime == -1) {
                _nextTime = 0;
            }

            nowTime = TimeUtility::monotonicTimeUs();
            if (nowTime < _nextTime) {
                continue;
            }
        }
        if (!_disableLoopFun) {
            _loopFunction();
        }
        {
            // in strictMode, we reset the nowTime after loopFunction
            if (_strictMode) {
                nowTime = TimeUtility::monotonicTimeUs();
            }
            ScopedLock lock(_cond);
            if (_nextTime != -1) {
                _nextTime = nowTime + _loopInterval;
            }
        }
    }
}

} // namespace autil
