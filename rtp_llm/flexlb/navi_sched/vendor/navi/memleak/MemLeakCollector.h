#pragma once

#include "navi/common.h"

namespace navi {

class MemLeakCollector;
class KernelMallocDef;
class StackTable;
class ThreadMemLeakDeduper;

struct MemLeakStack {
    size_t hash;
    size_t count;
    size_t addrs[0];
};

NAVI_TYPEDEF_PTR(MemLeakStack);

struct MemLeakEntry {
    uint64_t time;
    bool globalTraced = false;
    int type = 0;
    size_t size = 0;
    void *ptr = nullptr;
    MemLeakStackPtr stack;
    int64_t mallocId = -1;
};

class ThreadMemLeakResult {
public:
    ThreadMemLeakResult(int64_t mallocTraceId);
    ~ThreadMemLeakResult();
public:
    void collect(int type,
                 size_t size,
                 void *ptr,
                 uint64_t time,
                 const MemLeakStackPtr &stack);
    void fillProto(KernelMallocDef *mallocDef, StackTable &stackTable) const;
public:
    int64_t _mallocTraceId = -1;
    int64_t _mallocId = 0;
    std::vector<MemLeakEntry *> _entrys;
};

NAVI_TYPEDEF_PTR(ThreadMemLeakResult);

class ThreadMemLeakCollector {
public:
    ThreadMemLeakCollector(
        const std::shared_ptr<MemLeakCollector> &memLeakCollector,
        int64_t mallocTraceId);
    ~ThreadMemLeakCollector();
public:
    void collect(int type, size_t size, void *ptr);
    void mount(ThreadMemLeakDeduper &deduper);
    void unmount();
    void stop();
    const ThreadMemLeakResultPtr &getResult() {
        if (!_result) {
            _result = std::make_shared<ThreadMemLeakResult>(_mallocTraceId);
        }
        return _result;
    }
    void fillProto(KernelMallocDef *mallocDef, StackTable &stackTable) const;
public:
private:
    volatile bool _stopped = false;
    std::shared_ptr<MemLeakCollector> _memLeakCollector;
    int64_t _mallocTraceId;
    ThreadMemLeakResultPtr _result;
    ThreadMemLeakDeduper *_deduper = nullptr;
};

NAVI_TYPEDEF_PTR(ThreadMemLeakCollector);

enum MemState {
    MS_MALLOC = 0,
    MS_FREE = 1,
    MS_UNKNOWN = 2,
};

struct MemDedupState {
public:
    MemState remainState = MS_UNKNOWN;
    uint64_t time = 0;
    MemState state = MS_UNKNOWN;
    MemLeakStackPtr stack;
};

class ThreadMemLeakDeduper {
public:
    ~ThreadMemLeakDeduper();
public:
    void collect(const std::shared_ptr<MemLeakCollector> &memLeakCollector,
                 MemState type,
                 size_t size,
                 void *ptr);
    void report(const ThreadMemLeakResultPtr &result);
public:
    NaviUnorderedMap<void *, MemDedupState> _stateMap;
};

class ThreadMemLeakDeduperScope {
public:
    ThreadMemLeakDeduperScope(const std::shared_ptr<ThreadMemLeakCollector> &collector);
    ~ThreadMemLeakDeduperScope();
private:
    ThreadMemLeakCollectorPtr _collector;
    ThreadMemLeakDeduper _deduper;
};

class MemLeakCollector : public std::enable_shared_from_this<MemLeakCollector> {
public:
    MemLeakCollector();
    ~MemLeakCollector();
public:
    ThreadMemLeakCollectorPtr getThreadCollector(int64_t mallocTraceId);
    MemLeakStackPtr getStack(int type, size_t size);
private:
    mutable autil::ReadWriteLock _stackMapLock;
    NaviUnorderedMap<size_t, MemLeakStackPtr> _stackMap;
};

NAVI_TYPEDEF_PTR(MemLeakCollector);

}
