#include "navi/engine/NodeAsyncInfo.h"
#include "navi/engine/Graph.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/Node.h"

namespace navi {

NodeAsyncInfo::NodeAsyncInfo(Node *node, const NodeAsyncInfoPtr &old)
    : _node(node)
{
    if (old) {
        _pipes = old->_pipes;
    }
}

NodeAsyncInfo::~NodeAsyncInfo() {
    if (_readyMap) {
        ReadyBitMap::freeReadyBitMap(_readyMap);
    }
    if (_consumedMap) {
        ReadyBitMap::freeReadyBitMap(_consumedMap);
    }
}

AsyncPipePtr NodeAsyncInfo::addOne(ActivateStrategy activateStrategy) {
    auto newPipe = std::make_shared<AsyncPipe>(_node, activateStrategy);
    _pipes.push_back(newPipe);
    consolidate();
    return newPipe;
}

bool NodeAsyncInfo::consolidate() {
    std::vector<AsyncPipePtr> newPipes;
    std::swap(newPipes, _pipes);
    _independentPipes.clear();
    for (const auto &pipe : newPipes) {
        if (pipe->canRecycle()) {
            continue;
        }
        _pipes.push_back(pipe);
        if (AS_INDEPENDENT == pipe->getActivateStrategy()) {
            _independentPipes.push_back(pipe);
        }
    }
    refresh();
    return !_pipes.empty();
}

void NodeAsyncInfo::refresh() {
    auto bitMapSize = _pipes.size();
    _readyMap = ReadyBitMap::createReadyBitMap(bitMapSize);
    _consumedMap = ReadyBitMap::createReadyBitMap(bitMapSize);
    for (size_t index = 0; index < bitMapSize; index++) {
        const auto &pipe = _pipes[index];
        pipe->switchBitMap(_readyMap, _consumedMap, index);
    }
}

void NodeAsyncInfo::terminate() {
    for (const auto &pipe : _pipes) {
        pipe->terminate();
    }
}

bool NodeAsyncInfo::registerPeriodicPipe(
        const AsyncPipePtr &asyncPipe,
        int64_t periodMs,
        int64_t nextTriggerMs,
        int64_t repeat) const
{
    if (!asyncPipe) {
        return false;
    }
    return asyncPipe->registerPeriodic(periodMs, nextTriggerMs, repeat);
}

bool NodeAsyncInfo::getPeriodicPipeActivationCount(
        const AsyncPipePtr &asyncPipe,
        int64_t &activatedCount,
        int64_t &remainingCount,
        int64_t &periodMs,
        int64_t &nextTickTimeMs) const
{
    if (!asyncPipe) {
        return false;
    }
    int64_t relativeNextTriggerMs = -1;
    if (!asyncPipe->getPeriodicActivationCount(
            activatedCount,
            remainingCount,
            periodMs,
            relativeNextTriggerMs)) {
        return false;
    }
    nextTickTimeMs = relativeNextTriggerMs >= 0
        ? _node->getGraph()->getGraph()->getTimeoutChecker()->beginTime() /
            1000 + relativeNextTriggerMs
        : -1;
    return true;
}

bool NodeAsyncInfo::stopPeriodicAsyncPipe(const AsyncPipePtr &asyncPipe) const
{
    if (!asyncPipe) {
        return false;
    }
    return asyncPipe->stopPeriodic();
}

int64_t NodeAsyncInfo::checkTimeoutAsyncPipes(
        int64_t endTimeMs,
        std::vector<AsyncPipePtr> &activatedPipes) const
{
    int64_t nextTimeoutMs = -1;
    for (const auto &pipe : _pipes) {
        bool activated = false;
        auto pipeTimeoutMs =
            pipe->checkPeriodicTimeout(endTimeMs, activated);
        if (activated) {
            activatedPipes.emplace_back(pipe);
        }
        if (pipeTimeoutMs >= 0 &&
            (nextTimeoutMs < 0 || pipeTimeoutMs < nextTimeoutMs)) {
            nextTimeoutMs = pipeTimeoutMs;
        }
    }
    return nextTimeoutMs;
}

bool NodeAsyncInfo::hasActivePeriodicPipe() const
{
    for (const auto &pipe : _pipes) {
        if (pipe->periodicActive()) {
            return true;
        }
    }
    return false;
}

bool NodeAsyncInfo::independentReady() const {
    for (const auto &pipe : _independentPipes) {
        auto index = pipe->getIndex();
        if (_readyMap->isReady(index)) {
            return true;
        }
    }
    return false;
}

bool NodeAsyncInfo::isOk() const {
    return _readyMap->isOk();
}

bool NodeAsyncInfo::hasReady() const {
    return _readyMap->hasReady();
}

bool NodeAsyncInfo::isFinish() const {
    return _readyMap->isFinish();
}

bool NodeAsyncInfo::isFinishConsumed() const {
    return _consumedMap->isFinish();
}

void NodeAsyncInfo::fillState(KernelAsyncPipeStatesDef *state) const {
    state->set_ok(isOk());
    state->set_independent_ready(independentReady());
    state->set_has_ready(hasReady());
    state->set_finish(isFinish());
    state->set_finish_consumed(isFinishConsumed());

    for (const auto &pipe: _pipes) {
        if (AS_INDEPENDENT != pipe->getActivateStrategy()) {
            pipe->fillState(state->add_pipes());
        }
    }
    for (const auto &pipe: _independentPipes) {
        pipe->fillState(state->add_independent_pipes());
    }
}

}
