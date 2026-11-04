#pragma once

#include <map>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <string>

#include "autil/ThreadPool.h"

namespace autil {

class ThreadPoolManager {
public:
    ThreadPoolManager();
    ~ThreadPoolManager();

private:
    ThreadPoolManager(const ThreadPoolManager &);
    ThreadPoolManager &operator=(const ThreadPoolManager &);

public:
    // format:  taskQueueName|queueSize|threadNum[;taskQueueName|queueSize|threadNum}]
    // example: search_queue|500|24;summary_queue|500|10
    bool addThreadPool(const std::string &threadPoolsConfigStr);
    bool addThreadPool(const std::string &poolName, int32_t queueSize, int32_t threadNum);
    bool start();
    void stop(autil::ThreadPool::STOP_TYPE stopType = autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
    autil::ThreadPool *getThreadPool(const std::string &poolName);
    size_t getItemCount() const;
    size_t getTotalThreadNum() const;
    size_t getTotalQueueSize() const;
    std::map<std::string, size_t> getQueueSizeMap() const;
    std::map<std::string, size_t> getItemCountMap() const;
    std::map<std::string, size_t> getActiveThreadCount() const;
    std::map<std::string, size_t> getTotalThreadCount() const;
    const std::map<std::string, autil::ThreadPool *> &getPools() const { return _pools; }

private:
    std::map<std::string, autil::ThreadPool *> _pools;

private:
    friend class ThreadPoolManagerTest;
};

typedef std::shared_ptr<ThreadPoolManager> ThreadPoolManagerPtr;

} // namespace autil
