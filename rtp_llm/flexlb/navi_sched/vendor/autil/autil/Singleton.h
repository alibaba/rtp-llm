#pragma once
#include <memory>

#include "autil/Lock.h"
#include "autil/Log.h"

namespace autil {
class LazyInstantiation {
protected:
    template <typename T>
    static void create(T *&ptr) {
        T *tmp = new T;
        __asm__ __volatile__("" ::: "memory");
        ptr = tmp;
        static std::shared_ptr<T> destroyer(ptr);
    }
};

template <typename T, typename InstPolicy = LazyInstantiation>
class Singleton : private InstPolicy {
protected:
    Singleton(const Singleton &) {}
    Singleton() {}

public:
    ~Singleton() {}

public:
    static T *getInstance();

private:
    AUTIL_LOG_DECLARE();
};

template <typename T, typename InstPolicy>
inline T *Singleton<T, InstPolicy>::getInstance() {
    static T *ptr = 0;
    static autil::RecursiveThreadMutex gLock;

    if (unlikely(!ptr)) {
        autil::ScopedLock sl(gLock);
        if (!ptr) {
            InstPolicy::create(ptr);
        }
    }
    return const_cast<T *>(ptr);
}

} // namespace autil
