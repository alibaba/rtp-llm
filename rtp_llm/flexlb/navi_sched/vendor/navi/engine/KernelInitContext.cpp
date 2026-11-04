#include "navi/engine/KernelInitContext.h"

#include "navi/engine/Graph.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/NaviWorkerBase.h"
#include "navi/engine/Node.h"

namespace navi {

KernelInitContext::KernelInitContext(Node *node)
    : _node(node)
    , _subGraph(node->getGraph())
    , _forkError(false)
    , _hasFork(false)
    , _hasOutput(false)
{}

const TimeoutChecker *KernelInitContext::getTimeoutChecker() const {
    return _node->getTimeoutChecker();
}

bool KernelInitContext::getInputNode(PortIndex index, std::string &node) {
    return _node->getInputNode(index, node);
}

bool KernelInitContext::getIORange(IoType ioType, const std::string &name, PortIndex &start, PortIndex &end) const {
    return _node->getIORange(ioType, name, start, end);
}

bool KernelInitContext::getInputGroupSize(IndexType group, size_t &size) {
    auto groupCount = _node->getInputGroupCount();
    if (group >= groupCount) {
        NAVI_KERNEL_LOG(ERROR,
                        "group index [%u] overflow, node [%s] kernel [%s] "
                        "input group count [%lu]",
                        group,
                        _node->getName().c_str(),
                        _node->getKernelName().c_str(),
                        groupCount);
        return false;
    }
    return _node->getInputGroupSize(group, size);
}

bool KernelInitContext::getOutputGroupSize(IndexType group, size_t &size) {
    return _node->getOutputGroupSize(group, size);
}

bool KernelInitContext::inputConnected(PortIndex index) {
    return _node->inputConnected(index);
}

bool KernelInitContext::outputConnected(PortIndex index) {
    return _node->outputConnected(index);
}

const std::string &KernelInitContext::getConfigPath() const {
    return _node->getConfigPath();
}

const std::string &KernelInitContext::getThisBizName() const {
    return _node->getBizName();
}

NaviPartId KernelInitContext::getThisPartId() const {
    return _node->getGraph()->getPartId();
}

NaviPartId KernelInitContext::getPartCount() const {
    return _node->getGraph()->getPartCount();
}

int32_t KernelInitContext::getNodeIndex() const {
    return _node->getId();
}

ErrorCode KernelInitContext::fork(GraphDef *graphDef, const ForkGraphParam &param) {
    if (_hasOutput) {
        NAVI_KERNEL_LOG(ERROR, "can't do fork and setOutput in one compute");
        return EC_DATA_OVERRIDE;
    }
    auto ec = _node->fork(graphDef, param);
    if (EC_NONE != ec) {
        _forkError = true;
    }
    _hasFork = true;
    return ec;
}

bool KernelInitContext::forkError() const {
    return _forkError;
}

AsyncPipePtr KernelInitContext::createAsyncPipe(ActivateStrategy activateStrategy) {
    return _node->createAsyncPipe(activateStrategy);
}

bool KernelInitContext::registerPeriodicPipe(
        const AsyncPipePtr &asyncPipe,
        int64_t periodMs,
        int64_t repeat)
{
    return _node->registerPeriodicPipe(asyncPipe, periodMs, repeat);
}

bool KernelInitContext::getPeriodicPipeActivationCount(
        const AsyncPipePtr &asyncPipe,
        int64_t &activatedCount,
        int64_t &remainingCount,
        int64_t &periodMs,
        int64_t &nextTickTimeMs) const
{
    return _node->getPeriodicPipeActivationCount(
        asyncPipe,
        activatedCount,
        remainingCount,
        periodMs,
        nextTickTimeMs);
}

bool KernelInitContext::stopPeriodicAsyncPipe(
        const AsyncPipePtr &asyncPipe)
{
    return _node->stopPeriodicAsyncPipe(asyncPipe);
}

KernelConfigContext KernelInitContext::getConfigContext() const {
    return _node->getConfigContext();
}

bool KernelInitContext::getBizPartInfo(const std::string &bizName,
                                       NaviPartId &partCount,
                                       std::vector<NaviPartId> &partIds) const
{
    return _node->getBizPartInfo(bizName, partCount, partIds);
}

int64_t KernelInitContext::getSessionBeginTime() const {
    return _subGraph->getWorker()->getSessionBeginTime();
}

int64_t KernelInitContext::getRunGraphBeginTime() const {
    return _subGraph->getWorker()->getInitScheduleTime();
}

ResourcePtr
KernelInitContext::buildR(const std::string &name, KernelConfigContext ctx, ResourceMap &inputResourceMap) const {
    return _node->buildR(name, &ctx, inputResourceMap);
}

void KernelInitContext::cacheExpensive(bool value) {
    _node->cacheExpensive(value);
}

} // namespace navi
