#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace navi {

class ThreadPoolScheduleQueue {
public:
    ThreadPoolScheduleQueue();
    ~ThreadPoolScheduleQueue();
    ThreadPoolScheduleQueue(const ThreadPoolScheduleQueue &) = delete;
    ThreadPoolScheduleQueue &operator=(const ThreadPoolScheduleQueue &) = delete;

    bool insert(size_t keyH, size_t keyL, uint64_t value);
    uint64_t deleteMin(size_t &keyH, size_t &keyL);
    size_t size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace navi
