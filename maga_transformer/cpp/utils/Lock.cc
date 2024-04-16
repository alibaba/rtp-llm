#include "maga_transformer/cpp/utils/Lock.h"

namespace rtp_llm {

class RecursivePthreadMutexattr {
public:
    RecursivePthreadMutexattr() {
        pthread_mutexattr_init(&_mta);
        pthread_mutexattr_settype(&_mta, PTHREAD_MUTEX_RECURSIVE);
    }

    ~RecursivePthreadMutexattr() { pthread_mutexattr_destroy(&_mta); }

public:
    pthread_mutexattr_t _mta;
};

static const RecursivePthreadMutexattr sRecursivePthreadMutexattr = RecursivePthreadMutexattr();

const pthread_mutexattr_t *RecursiveThreadMutex::RECURSIVE_PTHREAD_MUTEXATTR_PTR = &sRecursivePthreadMutexattr._mta;
const int Notifier::EXITED = (1 << 16) + 1;

} // namespace autil
