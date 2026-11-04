#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Request.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/CallBack.h"
#include "autil/LockFreeThreadPool.h"

namespace autil {
class ThreadPool;
}

namespace multi_call {

class Connection
{
public:
    Connection(const std::string &spec, ProtocolType type, size_t queueSize, uint32_t queueTimeout,
               int64_t queueConsumeTimeout);
    virtual ~Connection();

private:
    Connection(const Connection &);
    Connection &operator=(const Connection &);

public:
    virtual void post(const RequestPtr &request, const CallBackPtr &callBack) = 0;

public:
    const std::string &getSpec() const {
        return _spec;
    }
    std::string getHost() const {
        return _spec.substr(4); // remove "tcp:"
    }
    std::string getHostIp() const {
        size_t end = _spec.rfind(':');
        return _spec.substr(4, end - 4); // remove "tcp:" and end ":port"
    }
    ProtocolType getType() const {
        return _type;
    }

    void setCallBackThreadPool(autil::LockFreeThreadPool *callBackThreadPool);

private:
    static size_t genAllocId();

protected:
    size_t _allocId;
    std::string _spec;
    ProtocolType _type;
    size_t _queueSize;
    uint32_t _queueTimeout;
    int64_t _queueConsumeTimeout;
    autil::LockFreeThreadPool *_callBackThreadPool;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(Connection);

} // namespace multi_call

