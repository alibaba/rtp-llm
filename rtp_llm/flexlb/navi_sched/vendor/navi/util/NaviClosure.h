#pragma once

#include "autil/Lock.h"
#include "navi/common.h"

namespace navi {

class NaviClosure {
public:
    NaviClosure() = default;
    virtual ~NaviClosure() = default;
    NaviClosure(const NaviClosure &) = delete;
    NaviClosure &operator=(const NaviClosure &) = delete;

public:
    virtual void run() = 0;
};

class WaitClosure {
public:
    WaitClosure();
    virtual ~WaitClosure();

private:
    WaitClosure(const WaitClosure &);
    WaitClosure &operator=(const WaitClosure &);

public:
    void wait();
    void notify();
    bool isFinish() const;

protected:
    volatile bool _isFinished;
    autil::ThreadCond _cond;
};

}
