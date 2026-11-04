#include "kmonitor/client/core/MetricsTags.h"
#include "navi/resource/MemoryPoolRBase.h"
#include "autil/mem_pool/Pool.h"

using namespace kmonitor;
namespace navi {

MemoryPoolRBase::MemoryPoolRBase() {
}

MemoryPoolRBase::~MemoryPoolRBase() {
    while (true) {
        autil::mem_pool::Pool *pool = nullptr;
        getPoolFromCache(pool);
        if (pool) {
            delete pool;
        } else {
            break;
        }
    }
}

size_t MemoryPoolRBase::getPoolFromCache(autil::mem_pool::Pool *&pool) const {
    if (!_poolQueue.Pop(&pool)) {
        pool = nullptr;
    } else {
        pool->reset();
    }
    return _poolQueue.Size();
}

size_t MemoryPoolRBase::putPoolToCache(autil::mem_pool::Pool *pool) const {
    _poolQueue.Push(pool);
    return _poolQueue.Size();
}
}
