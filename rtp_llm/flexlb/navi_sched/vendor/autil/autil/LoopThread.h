#pragma once

#include <functional>
#include <memory>
#include <stdint.h>
#include <string>

#include "autil/Lock.h"
#include "autil/Thread.h"

namespace autil {

class LoopThread;

typedef std::shared_ptr<LoopThread> LoopThreadPtr;

class LoopThread {
private:
    explicit LoopThread(bool strictMode = false);

public:
    ~LoopThread();

private:
    LoopThread(const LoopThread &);
    LoopThread &operator=(const LoopThread &);

public:
    void stop();

public:
    // if create loop thread fail, return null LoopThreadPtr
    static LoopThreadPtr
    createLoopThread(const std::function<void()> &loopFunction, int64_t loopInterval /*us*/, bool strictMode = false);
    // add this function to avoid the cast from (const char *) to (bool)
    static LoopThreadPtr createLoopThread(const std::function<void()> &loopFunction,
                                          int64_t loopInterval /*us*/,
                                          const char *name,
                                          bool strictMode = false);
    static LoopThreadPtr createLoopThread(const std::function<void()> &loopFunction,
                                          int64_t loopInterval /*us*/,
                                          const std::string &name,
                                          bool strictMode = false);
    void runOnce();

public:
    static void disableLoopFuncForTest() { _disableLoopFun = true; }
    static void enableLoopFuncForTest() { _disableLoopFun = false; }

private:
    void loop();

private:
    volatile bool _run;
    ThreadCond _cond;
    ThreadPtr _threadPtr;
    std::string _name;
    std::function<void()> _loopFunction;
    int64_t _loopInterval;
    int64_t _nextTime;
    bool _strictMode;
    static bool _disableLoopFun;
};

} // namespace autil
