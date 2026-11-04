// Keep TBB isolated from the legacy atomic macros in Navi headers.
#include "tbb/concurrent_priority_queue.h"

#include "navi/engine/ThreadPoolScheduleQueue.h"

namespace navi {

struct ThreadPoolScheduleQueue::Impl {
    struct Item {
        size_t keyH;
        size_t keyL;
        uint64_t value;
    };
    struct Later {
        bool operator()(const Item &left, const Item &right) const {
            return left.keyH > right.keyH || (left.keyH == right.keyH && left.keyL > right.keyL);
        }
    };
    tbb::concurrent_priority_queue<Item, Later> queue;
};

ThreadPoolScheduleQueue::ThreadPoolScheduleQueue() : _impl(new Impl) {}
ThreadPoolScheduleQueue::~ThreadPoolScheduleQueue() = default;

bool ThreadPoolScheduleQueue::insert(size_t keyH, size_t keyL, uint64_t value) {
    _impl->queue.push({keyH, keyL, value});
    return true;
}

uint64_t ThreadPoolScheduleQueue::deleteMin(size_t &keyH, size_t &keyL) {
    // Avoid joining the combining queue when idle workers already observe no work.
    if (_impl->queue.empty()) {
        return 0;
    }
    Impl::Item item;
    if (!_impl->queue.try_pop(item)) {
        return 0;
    }
    keyH = item.keyH;
    keyL = item.keyL;
    return item.value;
}

size_t ThreadPoolScheduleQueue::size() const { return _impl->queue.size(); }

} // namespace navi
