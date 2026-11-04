#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <string>

namespace autil {

class Thread;

typedef std::shared_ptr<Thread> ThreadPtr;
typedef pthread_t ThreadId;

class Thread {
public:
    ~Thread() { join(); }

public:
    void setName(const std::string &name);
    std::string getName() const;
    void join();
    pthread_t getId() const { return _id; }

public:
    // when create thread fail, return null ThreadPtr
    // need hold return value else will hang to the created thread terminate
    // nicevalue: renice thread priority, 0 do nonthing, -1 need authorition
    static ThreadPtr
    createThread(const std::function<void()> &threadFunction, const std::string &name = "", int niceValue = 0);
    // nice thread priority, try use nice() syscall, if no cap_sys_nice will use renice cmd by system call with sudo
    static bool niceThreadFunc(int niceValue, const std::string &name);

private:
    static void *threadWrapperFunction(void *arg);

private:
    Thread() { _id = 0; }
    pthread_t _id;
};

} // namespace autil
