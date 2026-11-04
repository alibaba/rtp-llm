#include "navi/memleak/MemLeakCollector.h"
#include "navi/util/CommonUtil.h"
#include "navi/proto/GraphVis.pb.h"
#include "navi/engine/CollectFlags.h"
#include "navi/engine/StackTable.h"
#include <execinfo.h>

namespace navi {

ThreadMemLeakResult::ThreadMemLeakResult(int64_t mallocTraceId)
    : _mallocTraceId(mallocTraceId)
{
}

ThreadMemLeakResult::~ThreadMemLeakResult() {
    for (auto entry : _entrys) {
        delete entry;
    }
    _entrys.clear();
}

void ThreadMemLeakResult::collect(int type,
                                  size_t size,
                                  void *ptr,
                                  uint64_t time,
                                  const MemLeakStackPtr &stack)
{
    auto entry = new MemLeakEntry();
    entry->time = time;
    entry->type = type;
    entry->size = size;
    entry->ptr = ptr;
    entry->stack = stack;
    if (0 == type) {
        auto mallocId = _mallocId++;
        entry->mallocId = mallocId;
        if (mallocId == _mallocTraceId) {
            NAVI_KERNEL_LOG(
                INFO, "global malloc trace ptr [%p] size [%lu]", ptr, size);
        }
    }
    _entrys.push_back(entry);
}

void ThreadMemLeakResult::fillProto(KernelMallocDef *mallocDef, StackTable &stackTable) const {
    for (auto entry : _entrys) {
        auto event = mallocDef->add_events();
        event->set_time(entry->time);
        event->set_is_free(entry->type != 0);
        event->set_size(entry->size);
        event->set_ptr((size_t)entry->ptr);
        if (0 == entry->type) {
            event->set_malloc_id(entry->mallocId);
        }
        if (entry->globalTraced) {
            event->set_global_traced(entry->globalTraced);
        }
        const auto &stack = entry->stack;
        if (!stack) {
            continue;
        }
        stackTable.addStack(stack->hash, stack->count, stack->addrs);
        event->set_stack_hash(stack->hash);
    }
}

ThreadMemLeakCollector::ThreadMemLeakCollector(const MemLeakCollectorPtr &memLeakCollector, int64_t mallocTraceId)
    : _memLeakCollector(memLeakCollector)
    , _mallocTraceId(mallocTraceId)
{
    assert(_memLeakCollector);
}

ThreadMemLeakCollector::~ThreadMemLeakCollector() {
}



void ThreadMemLeakCollector::mount(ThreadMemLeakDeduper &deduper) {
    _deduper = &deduper;
}

void ThreadMemLeakCollector::unmount() {
    if (_deduper) {
        _deduper->report(getResult());
    }
    _deduper = nullptr;
}

void ThreadMemLeakCollector::stop() {
    _stopped = true;
}

ThreadMemLeakDeduperScope::ThreadMemLeakDeduperScope(
    const std::shared_ptr<ThreadMemLeakCollector> &collector)
    : _collector(collector)
{
    if (_collector) {
        _collector->mount(_deduper);
    }
}

ThreadMemLeakDeduperScope::~ThreadMemLeakDeduperScope() {
    if (_collector) {
        _collector->unmount();
    }
}

ThreadMemLeakDeduper::~ThreadMemLeakDeduper() {
    auto stateMap = std::move(_stateMap);
    (void)stateMap;
}

void ThreadMemLeakDeduper::collect(
    const std::shared_ptr<MemLeakCollector> &memLeakCollector,
    MemState type,
    size_t size,
    void *ptr)
{
    auto it = _stateMap.find(ptr);
    if (_stateMap.end() == it) {
        auto &state = _stateMap[ptr];
        state.state = type;
        if (MS_FREE == type) {
            state.remainState = MS_FREE;
            state.time = CommonUtil::getTimelineTimeNs();
            state.stack = memLeakCollector->getStack(type, size);
        }
    } else {
        auto &state = it->second;
        if (MS_MALLOC == state.state && MS_FREE == type) {
            if (MS_UNKNOWN == state.remainState) {
                _stateMap.erase(it);
            } else {
                state.state = MS_FREE;
            }
        } else if (MS_FREE == state.state && MS_MALLOC == type) {
            state.state = MS_MALLOC;
        } else if (MS_FREE == state.state && MS_FREE == type) {
        } else if (MS_MALLOC == state.state && MS_MALLOC == type) {
        }
    }
}

void ThreadMemLeakDeduper::report(const ThreadMemLeakResultPtr &result) {
    if (!result) {
        return;
    }
    for (const auto &pair : _stateMap) {
        const auto &state = pair.second;
        if (MS_FREE == state.remainState) {
            assert(0 != state.time);
            result->collect(
                state.remainState, 0, pair.first, state.time, state.stack);
        }
    }
}

MemLeakCollector::MemLeakCollector() {
}

MemLeakCollector::~MemLeakCollector() {
}

ThreadMemLeakCollectorPtr MemLeakCollector::getThreadCollector(int64_t mallocTraceId) {
    return std::make_shared<ThreadMemLeakCollector>(shared_from_this(),
                                                    mallocTraceId);
}

MemLeakStackPtr MemLeakCollector::getStack(int type, size_t size) {
    constexpr int MAX_STACK_COUNT = 256;
    void *stackArray[MAX_STACK_COUNT];
    int32_t btCount = 0;
    void *stackVec = nullptr;
    if (0 == type) {
        stackArray[0] = (void *)size;
        btCount = backtrace(&stackArray[1], MAX_STACK_COUNT - 1) + 1;
        stackVec = &stackArray[0];
        if (btCount > 3) {
            stackVec = &stackArray[3];
            btCount -= 3;
            stackArray[3] = (void *)size;
        }
    } else {
        btCount = backtrace(&stackArray[0], MAX_STACK_COUNT);
        stackVec = &stackArray[0];
        if (btCount > 3) {
            stackVec = &stackArray[3];
            btCount -= 3;
        }
    }
    uint64_t hashKey = StackTable::getStackHash(btCount, (uint64_t *)stackVec);
    MemLeakStackPtr stack;
    if (0 == _stackMapLock.tryrdlock()) {
        auto it = _stackMap.find(hashKey);
        if (_stackMap.end() != it) {
            stack = it->second;
        }
        _stackMapLock.unlock();
    }
    if (!stack) {
        void *ret = malloc((2 + btCount) * 8);
        stack.reset((MemLeakStack *)ret, [](MemLeakStack *stack) { free(stack); });
        stack->hash = hashKey;
        stack->count = btCount;
        memcpy(&stack->addrs, stackVec, btCount * 8);
        if (0 == _stackMapLock.trywrlock()) {
            auto it = _stackMap.find(hashKey);
            if (_stackMap.end() == it) {
                _stackMap[hashKey] = stack;
            }
            _stackMapLock.unlock();
        }
    }
    return stack;
}

}
