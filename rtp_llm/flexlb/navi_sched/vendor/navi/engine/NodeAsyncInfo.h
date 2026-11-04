#pragma once

#include "navi/util/ReadyBitMap.h"
#include "navi/engine/AsyncPipe.h"

namespace navi {

class Node;

class NodeAsyncInfo;
struct NodeAsync {
    mutable autil::ThreadMutex asyncMutex;
    std::shared_ptr<NodeAsyncInfo> asyncInfo;
};

class NodeAsyncInfo
{
public:
    NodeAsyncInfo(Node *node, const std::shared_ptr<NodeAsyncInfo> &old);
    ~NodeAsyncInfo();
    NodeAsyncInfo(const NodeAsyncInfo &) = delete;
    NodeAsyncInfo &operator=(const NodeAsyncInfo &) = delete;
public:
    AsyncPipePtr addOne(ActivateStrategy activateStrategy);
    bool consolidate();
    void terminate();
    bool registerPeriodicPipe(
        const AsyncPipePtr &asyncPipe,
        int64_t periodMs,
        int64_t nextTriggerMs,
        int64_t repeat) const;
    bool getPeriodicPipeActivationCount(
        const AsyncPipePtr &asyncPipe,
        int64_t &activatedCount,
        int64_t &remainingCount,
        int64_t &periodMs,
        int64_t &nextTickTimeMs) const;
    bool stopPeriodicAsyncPipe(const AsyncPipePtr &asyncPipe) const;
    int64_t checkTimeoutAsyncPipes(
        int64_t endTimeMs,
        std::vector<AsyncPipePtr> &activatedPipes) const;
    bool hasActivePeriodicPipe() const;
public:
    bool independentReady() const;
    bool isOk() const;
    bool hasReady() const;
    bool isFinish() const;
    bool isFinishConsumed() const;
    void fillState(KernelAsyncPipeStatesDef *state) const;
private:
    void refresh();
private:
    Node *_node;
    std::vector<AsyncPipePtr> _pipes;
    std::vector<AsyncPipePtr> _independentPipes;
    ReadyBitMap *_readyMap = nullptr;
    ReadyBitMap *_consumedMap = nullptr;
};

NAVI_TYPEDEF_PTR(NodeAsyncInfo);

}
