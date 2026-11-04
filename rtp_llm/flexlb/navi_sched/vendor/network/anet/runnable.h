#pragma once

namespace anet {
class Thread;
class Runnable {

public:
    /*
     * 析构
     */
    virtual ~Runnable() {}
    /*
     * 运行入口函数
     */
    virtual void run(Thread *thread, void *arg) = 0;
};
} // namespace anet

