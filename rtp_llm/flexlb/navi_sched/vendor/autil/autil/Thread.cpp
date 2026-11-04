#include "autil/Thread.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <pthread.h>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

#include "autil/Log.h"
#include "autil/ThreadNameScope.h"

using namespace std;

namespace autil {
AUTIL_DECLARE_AND_SETUP_LOGGER(autil, Thread);

struct ThreadFunctionWrapperArg {
    function<void()> threadFunction;
    string name;
    int niceValue = 0;
};

void *Thread::threadWrapperFunction(void *arg) {
    unique_ptr<ThreadFunctionWrapperArg> p(static_cast<ThreadFunctionWrapperArg *>(arg));
    AUTIL_LOG(INFO, "thread started for name[%s]", p->name.c_str());
    niceThreadFunc(p->niceValue, p->name);
    p->threadFunction();
    return NULL;
}

void Thread::setName(const std::string &name) {
    if (_id) {
        pthread_setname_np(_id, name.c_str());
    }
}

std::string Thread::getName() const {
    if (_id) {
        char buf[1024];
        pthread_getname_np(_id, buf, 1024);
        return string(buf);
    }
    return "";
}

void Thread::join() {
    if (_id) {
        int ret = pthread_join(_id, NULL);
        (void)ret;
        assert(ret == 0);
    }
    _id = 0;
}

bool Thread::niceThreadFunc(int niceValue, const string &name) {
    if (niceValue != 0) {
        int ret = nice(niceValue);
        if (ret == -1) {
            int64_t tid = (long)syscall(SYS_gettid);
            char buf[512];
            snprintf(buf, 512, "sudo renice -n %d %ld", niceValue, tid);
            ret = system(buf);
            AUTIL_LOG(INFO, "renice thread name [%s], cmd [%s], ret[%d].", name.c_str(), buf, ret);
            if (ret != 0) {
                AUTIL_LOG(ERROR, "inc thread name [%s] nice value [%d] falied.", name.c_str(), niceValue);
                return false;
            }
        } else {
            AUTIL_LOG(
                INFO, "inc thread name [%s] nice value [%d] success, new nice [%d].", name.c_str(), niceValue, ret);
        }
    }
    return true;
}

ThreadPtr Thread::createThread(const function<void()> &threadFunction, const string &name, int niceValue) {

    ThreadPtr threadPtr(new Thread);
    ThreadFunctionWrapperArg *arg = new ThreadFunctionWrapperArg;
    arg->threadFunction = threadFunction;
    arg->niceValue = niceValue;
    arg->name = name;

    unique_ptr<ThreadNameScope> scope;
    if (!name.empty()) {
        scope.reset(new ThreadNameScope(name));
    }
    int rtn = pthread_create(&threadPtr->_id, NULL, &Thread::threadWrapperFunction, arg);

    if (rtn != 0) {
        delete arg;
        threadPtr->_id = 0;
        threadPtr.reset();
        AUTIL_LOG(ERROR, "Create thread error [%d]", rtn);
    }
    return threadPtr;
}

} // namespace autil
