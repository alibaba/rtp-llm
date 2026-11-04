#pragma once

#include "navi/engine/Data.h"
#include "navi/engine/NaviResult.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"

namespace navi {

struct NaviUserData {
public:
    NaviUserData()
        : partId(INVALID_NAVI_PART_ID)
        , index(0)
        , streamId(0)
        , eof(false)
    {
    }
public:
    std::string name;
    NaviPartId partId;
    size_t index;
    size_t streamId;
    DataPtr data;
    bool eof;
};

class ReadyBitMap;
class GraphDomainUser;
class GraphDomainClient;
class Port;
class NaviWorkerBase;
class Graph;
class NaviSession;
class NaviSnapshot;
class GraphResult;
class NaviHostInfo;
class ComputeScope;

template <typename T>
class DomainHolder;

struct PortDataInfo {
public:
    PortDataInfo(Port *port_, NaviPartId partId_)
        : port(port_)
        , partId(partId_)
        , streamId(0)
    {
    }
public:
    Port *port;
    NaviPartId partId;
    size_t streamId;
};

class CondWaiter {
public:
    void wait() {
        autil::ScopedLock lock(_cond);
        while (!_finished) {
            _cond.wait();
        }
    }
    void notify() {
        autil::ScopedLock lock(_cond);
        if (_finished) {
            _cond.signal();
            return;
        }
        _finished = true;
        _cond.signal();
    }
private:
    autil::ThreadCond _cond;
    bool _finished = false;
};

NAVI_TYPEDEF_PTR(CondWaiter);

class NaviUserResult : public std::enable_shared_from_this<NaviUserResult>
{
public:
    NaviUserResult();
    virtual ~NaviUserResult();
private:
    NaviUserResult(const NaviUserResult &);
    NaviUserResult &operator=(const NaviUserResult &);
public:
    // return true if data is valid
    // eof always indicates the end of data
    virtual bool nextData(NaviUserData &data, bool &eof) = 0;
    // call this if eof return true
    NaviResultPtr getNaviResult();
    void abort(ErrorCode ec = EC_ABORT);
    size_t outputCount() const;
    void waitInit() const;
private:
    NaviResultPtr getUnfinishedNaviResult();
    void setGraphResult(const std::shared_ptr<GraphResult> &result);
    const std::shared_ptr<GraphResult> &getGraphResult() const;
    NaviResultPtr getNaviResultWithoutCheck();
    bool bindDomain(bool collectMalloc,
                    const std::shared_ptr<GraphDomainUser> &domain);
    virtual void notify(bool finish) = 0;
    void notifyData();
    bool hasOutput() const;
    void setWorkerInfo(const std::shared_ptr<NaviHostInfo> &hostInfo, NaviWorkerBase *worker);
    virtual CondWaiterPtr getDestructWaiter() const {
        return CondWaiterPtr();
    }
    virtual void initDestructWaiter() {
    }
private:
    bool collectPort(const std::shared_ptr<GraphDomainUser> &domain);
    void initFinishMap(uint32_t portCount);
    void updateError(ErrorCode ec);
private:
    friend class KernelTester;
    friend class ArpcGraphClosure;
    friend class WorkerDestructItem;
protected:
    bool nextPort(NaviUserData &userData, bool &eof, ErrorCode &ec);
    void terminate(ErrorCode ec);
protected:
    DECLARE_LOGGER();
    mutable autil::ThreadCond _cond;
    bool _inited;
    bool _finished;
    std::shared_ptr<GraphResult> _graphResult;
    volatile atomic64_t _dataCount;
    mutable autil::ThreadMutex _resultMutex;
    NaviResultPtr _naviResult;
private:
    DomainHolder<GraphDomainUser> *_domainHolder;
    std::vector<PortDataInfo> _portVec;
    ReadyBitMap *_finishMap;
    size_t _loopIndex;
    std::atomic<bool> _terminated;
    // do not use _worker, gdb only
    NaviWorkerBase *_gdbPtr;
    std::shared_ptr<NaviHostInfo> _hostInfo;
private:
    friend class GraphDomainUser;
    friend class GraphDomainClient;
    friend class NaviWorkerBase;
    friend class Graph;
    friend class NaviSession;
    friend class NaviSnapshot;
    friend class DelaySendDestructClosure;
};

NAVI_TYPEDEF_PTR(NaviUserResult);

class NaviUserResultClosure {
public:
    NaviUserResultClosure() = default;
    virtual ~NaviUserResultClosure() = default;
public:
    virtual void run(NaviUserResultPtr result,
                     const std::shared_ptr<ComputeScope> &computeScope) = 0;
public:
    bool waitDestruct = false;
};

class DefaultDestructClosure : public NaviUserResultClosure {
public:
    void wait() {
        autil::ScopedLock lock(_cond);
        while (!_finished) {
            _cond.wait();
        }
    }
    void run(NaviUserResultPtr result,
             const std::shared_ptr<ComputeScope> &computeScope) override {
        autil::ScopedLock lock(_cond);
        if (_finished) {
            _cond.signal();
            return;
        }
        _finished = true;
        _cond.signal();
    }

private:
    autil::ThreadCond _cond;
    bool _finished = false;
};

class NaviSyncUserResult : public NaviUserResult {
public:
    virtual bool nextData(NaviUserData &data, bool &eof) override;
private:
    virtual void notify(bool finish) override;
    void wait();
    virtual void initDestructWaiter() override;
    virtual CondWaiterPtr getDestructWaiter() const override {
        return _destructWaiter;
    }
    void waitDestruct() const;
private:
    CondWaiterPtr _destructWaiter;
};

class NaviAsyncUserResult : public NaviUserResult {
public:
    NaviAsyncUserResult(NaviUserResultClosure *closure);
public:
    virtual bool nextData(NaviUserData &data, bool &eof) override;
private:
    virtual void notify(bool finish) override;
    bool processData(NaviUserData &data, bool &eof, ErrorCode &retEc);
private:
    bool _cleared = false;
    NaviUserResultClosure *_resultClosure;
    std::list<NaviUserData> _resultData;
    atomic64_t _pendingNotify;
    autil::ThreadCond _notifyCond;
};


}
