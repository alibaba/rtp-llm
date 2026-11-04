#pragma once
#include <pthread.h>
#include <stddef.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/runnable.h"

//#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/log.h"
namespace anet {
class Runnable;

class Thread {

public:
    /**
     * 构造函数
     */
    Thread() { tid = 0; }

    /**
     * 起一个线程，开始运行
     */
    void start(Runnable *r, void *a) {
        runnable = r;
        args = a;
        pthread_create(&tid, NULL, Thread::hook, this);
    }

    void setName(const char *name) { 
        if (tid) {
            pthread_setname_np(tid, name);
        }
    }

    /**
     * 等待线程退出
     */
    void join() {
        if (tid) {
            pthread_join(tid, NULL);
        }
        tid = 0;
    }

    /**
     * 得到Runnable对象
     *
     * @return Runnable
     */
    Runnable *getRunnable() { return runnable; }

    /**
     * 得到回调参数
     *
     * @return args
     */
    void *getArgs() { return args; }

    /**
     * 线程的回调函数
     *
     */

    static void *hook(void *arg) {
        Thread *thread = (Thread *)arg;

        if (thread->getRunnable()) {
            thread->getRunnable()->run(thread, thread->getArgs());
        }

        return (void *)NULL;
    }

private:
    pthread_t tid;
    Runnable *runnable;
    void *args;
};

} // namespace anet

