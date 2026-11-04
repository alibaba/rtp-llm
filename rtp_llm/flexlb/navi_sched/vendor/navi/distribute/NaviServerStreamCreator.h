#pragma once

#include "navi/common.h"
#include "navi/log/NaviLogger.h"
#include <autil/Lock.h>
#include <multi_call/stream/GigServerStreamCreator.h>

namespace navi {

class Navi;
class NaviSnapshot;

class NaviServerStreamCreator : public multi_call::GigServerStreamCreator
{
public:
    NaviServerStreamCreator(const NaviLoggerPtr &logger, Navi *navi, const std::string &name);
    ~NaviServerStreamCreator();
private:
    NaviServerStreamCreator(const NaviServerStreamCreator &);
    NaviServerStreamCreator &operator=(const NaviServerStreamCreator &);
public:
    std::string methodName() const override;
    multi_call::GigServerStreamPtr create(const multi_call::StreamRpcParam &param) override;
public:
    void decQueryCount();
    void stop();
private:
    std::shared_ptr<NaviSnapshot> incQueryCount();
private:
    autil::ReadWriteLock _rwLock;
    Navi *_navi = nullptr;
    atomic64_t _queryCount;
private:
    DECLARE_LOGGER();
};

NAVI_TYPEDEF_PTR(NaviServerStreamCreator);

}
