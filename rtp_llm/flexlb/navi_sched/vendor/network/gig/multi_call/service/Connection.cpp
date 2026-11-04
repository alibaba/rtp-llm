#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/Connection.h"

#include "autil/StringUtil.h"

using namespace std;
using namespace autil;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, Connection);

Connection::Connection(const std::string &spec, ProtocolType type, size_t queueSize,
                       uint32_t queueTimeout, int64_t queueConsumeTimeout)
    : _allocId(genAllocId())
    , _spec(spec)
    , _type(type)
    , _queueSize(queueSize)
    , _queueTimeout(queueTimeout)
    , _queueConsumeTimeout(queueConsumeTimeout)
    , _callBackThreadPool(nullptr) {
}

Connection::~Connection() {
}

void Connection::setCallBackThreadPool(autil::LockFreeThreadPool *callBackThreadPool) {
    _callBackThreadPool = callBackThreadPool;
}



size_t Connection::genAllocId() {
    static std::atomic<size_t> count = {0};
    return count.fetch_add(1);
}

} // namespace multi_call
