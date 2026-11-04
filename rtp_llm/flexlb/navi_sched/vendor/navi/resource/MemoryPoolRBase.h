#pragma once

#include "autil/Lock.h"
#include "autil/LockFreeQueue.h"

namespace kmonitor {
class MetricsTags;
} // namespace kmonitor

namespace autil {
namespace mem_pool {
class Pool;
}
}

namespace navi {

class MemoryPoolRBase
{
public:
    MemoryPoolRBase();
    ~MemoryPoolRBase();
private:
    MemoryPoolRBase(const MemoryPoolRBase &);
    MemoryPoolRBase &operator=(const MemoryPoolRBase &);
public:
    size_t getPoolFromCache(autil::mem_pool::Pool *&pool) const;
    size_t putPoolToCache(autil::mem_pool::Pool *pool) const;
    size_t size() {
        return _poolQueue.Size();
    }
protected:
    mutable autil::LockFreeQueue<autil::mem_pool::Pool *> _poolQueue;
};

}
