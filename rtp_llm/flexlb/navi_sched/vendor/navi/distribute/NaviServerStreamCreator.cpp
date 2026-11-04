#include "navi/distribute/NaviServerStreamCreator.h"
#include "navi/distribute/NaviServerStream.h"
#include "navi/engine/Navi.h"
#include "navi/engine/NaviSnapshot.h"

using namespace autil;

namespace navi {

NaviServerStreamCreator::NaviServerStreamCreator(const NaviLoggerPtr &logger, Navi *navi, const std::string &name)
    : _navi(navi)
{
    atomic_set(&_queryCount, 0);
    _logger.logger = logger;
    _logger.object = this;
    _logger.prefix = name;
}

NaviServerStreamCreator::~NaviServerStreamCreator() {
}

std::string NaviServerStreamCreator::methodName() const {
    return NAVI_RPC_METHOD_NAME;
}

multi_call::GigServerStreamPtr NaviServerStreamCreator::create(const multi_call::StreamRpcParam &param) {
    auto snapshot = incQueryCount();
    if (snapshot) {
        return std::make_shared<NaviServerStream>(param, this, snapshot);
    } else {
        return nullptr;
    }
}

std::shared_ptr<NaviSnapshot> NaviServerStreamCreator::incQueryCount() {
    ScopedReadLock lock(_rwLock);
    if (!_navi) {
        NAVI_LOG(SCHEDULE2, "creator is stopped, queryCount [%lld]", atomic_read(&_queryCount));
        return nullptr;
    }
    auto snapshot = _navi->getSnapshot();
    if (!snapshot) {
        NAVI_LOG(SCHEDULE2, "navi snapshot is null, navi: [%p], queryCount [%lld]", _navi, atomic_read(&_queryCount));
        return nullptr;
    }
    NAVI_LOG(SCHEDULE2, "inc queryCount from [%lld]", atomic_read(&_queryCount));
    atomic_inc(&_queryCount);
    return snapshot;
}

void NaviServerStreamCreator::decQueryCount() {
    NAVI_LOG(SCHEDULE2, "dec queryCount from [%lld]", atomic_read(&_queryCount));
    atomic_dec(&_queryCount);
}

void NaviServerStreamCreator::stop() {
    NAVI_LOG(INFO, "start stop navi server stream");
    {
        ScopedWriteLock lock(_rwLock);
        _navi = nullptr;
    }
    size_t sleepCount = 0;
    while (true) {
        auto queryCount = atomic_read(&_queryCount);
        if (queryCount <= 0) {
            NAVI_LOG(INFO, "wait queryCount succeed, sleepCount [%lu]", sleepCount);
            break;
        }
        if (sleepCount++ % 200 == 0) {
            NAVI_LOG(INFO, "still has queryCount [%lld], sleep", queryCount);
        }
        usleep(10 * 1000);
    }
    NAVI_LOG(INFO, "finish stop navi server stream");
}

}

