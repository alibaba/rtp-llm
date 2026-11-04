#pragma once

#include <list>

#include "navi/common.h"
#include "navi/engine/Data.h"
#include "navi/proto/GraphState.pb.h"

namespace navi {

class Node;
class ReadyBitMap;

class AsyncPipe {
public:
    AsyncPipe(Node *node, ActivateStrategy activateStrategy);
    AsyncPipe(const AsyncPipe &) = delete;
    AsyncPipe &operator=(const AsyncPipe &) = delete;
    virtual ~AsyncPipe();

public:
    virtual ErrorCode setData(const DataPtr &data); // virtual for test
    virtual ErrorCode setEof();                     // virtual for test
    virtual ErrorCode setDataOverride(
        const DataPtr &data,
        bool overrideData,
        bool ignoreData = false);
    virtual size_t getDataItemCnt() const;
    virtual bool getData(DataPtr &data, bool &eof); // virtual for test
    virtual void terminate();                       // virtual for test

private:
    struct PeriodicInfo {
        PeriodicInfo(int64_t periodMs, int64_t nextTriggerMs, int64_t repeat);
        void reset(int64_t periodMs, int64_t nextTriggerMs, int64_t repeat);
        void snapshot(int64_t &activatedCount,
                      int64_t &remainingRepeat,
                      int64_t &periodMs,
                      int64_t &nextTriggerMs) const;
        bool stop();
        int64_t checkTimeout(int64_t endTimeMs, bool &activated);
        bool isActive() const;

        mutable autil::ThreadMutex mutex;
        std::atomic<int64_t> periodMs{0};
        std::atomic<int64_t> nextTriggerMs{0};
        std::atomic<int64_t> activatedCount{0};
        std::atomic<int64_t> remainingRepeat{0};
        std::atomic<bool> active{false};
        bool stopped = false;
    };

    void switchBitMap(ReadyBitMap *newReadyMap, ReadyBitMap *newConsumedMap, size_t newPipeIndex);
    ActivateStrategy getActivateStrategy() const { return _activateStrategy; }
    bool canRecycle() const;
    bool registerPeriodic(int64_t periodMs, int64_t nextTriggerMs, int64_t repeat);
    bool getPeriodicActivationCount(int64_t &activatedCount,
                                    int64_t &remainingCount,
                                    int64_t &periodMs,
                                    int64_t &nextTriggerMs) const;
    bool stopPeriodic();
    int64_t checkPeriodicTimeout(int64_t endTimeMs, bool &activated);
    bool periodicActive() const;
    size_t getIndex() const { return _pipeIndex; }
    void decItemCount();
    void incItemCount();
    void scheduleNode() const;
    void fillState(KernelSingleAsyncPipeStateDef *state) const;

private:
    friend class NodeAsyncInfo;
private:
    Node *_node;
    ReadyBitMap *_nodeReadyMap = nullptr;
    ReadyBitMap *_nodeConsumedMap = nullptr;
    size_t _pipeIndex = 0;
    ActivateStrategy _activateStrategy;
    bool _terminated : 1;
    bool _eof        : 1;
    size_t _setCount = 0;
    size_t _getCount = 0;
    std::atomic<PeriodicInfo *> _periodicInfo{nullptr};
    mutable autil::ThreadMutex _mutex;
    std::list<DataPtr> _dataList;
};

NAVI_TYPEDEF_PTR(AsyncPipe);

} // namespace navi
