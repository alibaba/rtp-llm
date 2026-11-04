#pragma once

#include <memory>
#include <stddef.h>

#define POOL_NEW_CLASS(pool, type, args...) (new (pool->allocate(sizeof(type))) type(args))

#define POOL_NEW(pool, type) ((type *)(pool->allocate(sizeof(type))))

#define POOL_NEW_VECTOR(pool, type, num) ((type *)(pool->allocate(sizeof(type) * num)))

template <typename T>
inline void POOL_DELETE_CLASS(T *value) {
    if (value) {
        value->~T();
    }
}

#define POOL_COMPATIBLE_NEW_CLASS(pool, type, args...) ((pool) ? POOL_NEW_CLASS(pool, type, args) : new type(args))

#define POOL_COMPATIBLE_DELETE_CLASS(pool, value) (pool) ? POOL_DELETE_CLASS(value) : (delete (value))

#define POOL_COMPATIBLE_NEW_VECTOR(pool, type, num) (pool) ? POOL_NEW_VECTOR(pool, type, num) : new type[num]

#define POOL_COMPATIBLE_DELETE_VECTOR(pool, p)                                                                         \
    if (!pool) {                                                                                                       \
        delete[] p;                                                                                                    \
    }

namespace autil {
namespace mem {
class QuotaController;
}
} // namespace autil

namespace autil {
namespace mem_pool {

class PoolBase {
public:
    virtual ~PoolBase() {}

public:
    virtual void *allocate(size_t numBytes) = 0;

    virtual void *allocate(size_t numBytes, size_t alignment) { return allocate(numBytes); }

    virtual void *allocateUnsafe(size_t numBytes) { return allocate(numBytes); }

    virtual void *allocateUnsafe(size_t numBytes, size_t alignment) { return allocate(numBytes, alignment); }

    virtual void deallocate(void *ptr, size_t size) = 0;
    virtual void release() = 0;
    virtual size_t reset() = 0;

public:
    virtual bool isInPool(const void *ptr) const = 0;

    virtual size_t getUsedBytes() const { return 0; }

    virtual size_t getTotalBytes() const { return 0; }

    virtual void bindQuotaController(std::shared_ptr<mem::QuotaController> quotaController) {}

    virtual std::shared_ptr<mem::QuotaController> getQuotaController() const { return nullptr; }
};

} // namespace mem_pool
} // namespace autil
