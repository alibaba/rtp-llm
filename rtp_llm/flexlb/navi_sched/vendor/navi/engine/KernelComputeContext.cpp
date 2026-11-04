#include "navi/engine/KernelComputeContext.h"
#include "navi/engine/Node.h"

namespace navi {

GroupDatas::GroupDatas()
    : _begin(nullptr)
    , _end(nullptr)
{
}

GroupDatas::GroupDatas(InputSnapshot *begin, InputSnapshot *end)
    : _begin(begin)
    , _end(end)
{
}

GroupDatas::Iterator::Iterator()
    : _ptr(nullptr)
{
}

GroupDatas::Iterator::Iterator(InputSnapshot *ptr)
    : _ptr(ptr)
{
}

GroupDatas::Iterator GroupDatas::Iterator::operator+(size_t offset) const {
    auto it = *this;
    it._ptr += offset;
    return it;
}

GroupDatas::Iterator GroupDatas::Iterator::operator++() {
    ++_ptr;
    return *this;
}

bool GroupDatas::Iterator::operator!=(const Iterator &rhs) const {
    return _ptr != rhs._ptr;
}

const StreamData &GroupDatas::Iterator::operator*() {
    return _ptr->data;
}

GroupDatas::Iterator GroupDatas::begin() const {
    return Iterator(_begin);
}

GroupDatas::Iterator GroupDatas::end() const {
    return Iterator(_end);
}

size_t GroupDatas::size() const {
    return _end - _begin;
}

const StreamData &GroupDatas::operator[](size_t index) const {
    return _begin[index].data;
}

bool GroupDatas::eof() const {
    for (const auto &streamData : *this) {
        if (!streamData.eof) {
            return false;
        }
    }
    return true;
}

InputPeekScope::InputPeekScope(KernelComputeContext &ctx, PortIndex index)
    : _ctx(ctx)
    , _index(index)
{
    _ok = _ctx.peekInput(_index, _data);
}

InputPeekScope::~InputPeekScope() {
    _ctx.clearInput(_index);
}

GroupPeekScope::GroupPeekScope(KernelComputeContext &ctx, IndexType group)
    : _ctx(ctx)
    , _group(group)
{
    _ok = _ctx.peekGroupInput(_group, _datas);
}

GroupPeekScope::~GroupPeekScope() {
    _ctx.clearGroupInput(_group);
}

KernelComputeContext::KernelComputeContext(Node *node)
    : KernelInitContext(node)
    , _deadLock(true)
    , _ignoreDeadlock(false)
{
    assert(_node);
}

bool KernelComputeContext::getInput(PortIndex index, DataPtr &data, bool &eof) {
    StreamData streamData;
    auto ret = getInput(index, streamData);
    data = std::move(streamData.data);
    eof = streamData.eof;
    return ret;
}

bool KernelComputeContext::getInput(PortIndex index, StreamData &streamData, bool clear) {
    auto ret = _node->getInput(index, EGT_GET, streamData, clear);
    if (ret) {
        _deadLock = false;
    }
    return ret;
}

bool KernelComputeContext::getGroupInput(IndexType group, GroupDatas &datas) {
    auto groupCount = _node->getInputGroupCount();
    if (group >= groupCount) {
        NAVI_KERNEL_LOG(ERROR,
                        "group index [%u] overflow, node [%s] kernel [%s] "
                        "input group count [%lu]",
                        group, _node->getName().c_str(),
                        _node->getKernelName().c_str(), groupCount);
        return false;
    }
    return doGetGroupInput(group, EGT_GET, datas);
}

bool KernelComputeContext::doGetGroupInput(IndexType group, EdgeGetType type, GroupDatas &datas) {
    InputSnapshot *begin = nullptr;
    InputSnapshot *end = nullptr;
    if (!_node->snapshotGroupInput(group, type, begin, end)) {
        return false;
    }
    datas = GroupDatas(begin, end);
    _deadLock = false;
    return true;
}

bool KernelComputeContext::peekInput(PortIndex index, DataPtr &data, bool &eof) {
    StreamData streamData;
    auto ret = peekInput(index, streamData);
    data = std::move(streamData.data);
    eof = streamData.eof;
    return ret;
}

bool KernelComputeContext::peekInput(PortIndex index, StreamData &streamData) {
    auto ret = _node->getInput(index, EGT_PEEK, streamData, false);
    if (ret) {
        _deadLock = false;
    }
    return ret;
}

bool KernelComputeContext::peekGroupInput(IndexType group, GroupDatas &datas) {
    auto groupCount = _node->getInputGroupCount();
    if (group >= groupCount) {
        NAVI_KERNEL_LOG(ERROR,
                        "group index [%u] overflow, node [%s] kernel [%s] "
                        "input group count [%lu]",
                        group, _node->getName().c_str(),
                        _node->getKernelName().c_str(), groupCount);
        return false;
    }
    return doGetGroupInput(group, EGT_PEEK, datas);
}

bool KernelComputeContext::clearInput(PortIndex index) {
    StreamData streamData;
    auto ret = _node->getInput(index, EGT_CLEAR, streamData, false);
    if (ret) {
        _deadLock = false;
    }
    return ret;
}

bool KernelComputeContext::clearGroupInput(IndexType group) {
    auto groupCount = _node->getInputGroupCount();
    if (group >= groupCount) {
        NAVI_KERNEL_LOG(ERROR,
                        "group index [%u] overflow, node [%s] kernel [%s] "
                        "input group count [%lu]",
                        group, _node->getName().c_str(),
                        _node->getKernelName().c_str(), groupCount);
        return false;
    }
    GroupDatas datas;
    return doGetGroupInput(group, EGT_CLEAR, datas);
}

bool KernelComputeContext::setInputEof(PortIndex index) {
    auto ret = _node->setInputEof(index);
    if (ret) {
        _deadLock = false;
    }
    return ret;
}

// index can be group index
bool KernelComputeContext::setOutput(PortIndex index,
                                     const DataPtr &data, bool eof)
{
    if (_hasFork) {
        NAVI_KERNEL_LOG(ERROR, "can't do fork and setOutput in one compute");
        return false;
    }
    _hasOutput = true;
    auto ret = _node->setOutput(index, data, nullptr, eof);
    if (ret) {
        _deadLock = false;
    }
    return ret;
}

// index can be group index
bool KernelComputeContext::setOutput(PortIndex index,
                                     Data *data, bool eof)
{
    if (_hasFork) {
        delete data;
        NAVI_KERNEL_LOG(ERROR, "can't do fork and setOutput in one compute");
        return false;
    }
    _hasOutput = true;
    auto ret = _node->setOutput(index, DataPtr(), data, eof);
    if (ret) {
        _deadLock = false;
    }
    return ret;
}

bool KernelComputeContext::deadLock() const {
    return (!_ignoreDeadlock) && _deadLock;
}

void KernelComputeContext::setIgnoreDeadlock() {
    _node->incNodeSnapshot();
    _ignoreDeadlock = true;
}

void KernelComputeContext::stopSchedule() {
    _node->stopSchedule();
}

ErrorCode KernelComputeContext::fork(GraphDef *graphDef, const ForkGraphParam &param) {
    _deadLock = false;
    return KernelInitContext::fork(graphDef, param);
}

int64_t KernelComputeContext::getScheduleCount() const {
    return _node->getScheduleCount();
}

void KernelComputeContext::updateTraceLevel(const std::string &levelStr) const {
    _node->updateTraceLevel(levelStr);
}

bool KernelComputeContext::updateTimeoutMs(int64_t timeoutMs) const {
    return _node->updateTimeoutMs(timeoutMs);
}

void KernelComputeContext::updateThreadLimit(uint32_t threadLimit) const {
    _node->updateThreadLimit(threadLimit);
}

void KernelComputeContext::updateCollect(bool collectMetric, bool collectPerf) const {
    _node->updateCollect(collectMetric, collectPerf);
}

void KernelComputeContext::collectTrace(std::vector<std::string> &traceVec) const {
    _node->collectTrace(traceVec);
}

std::shared_ptr<multi_call::GigStreamRpcInfoMap> KernelComputeContext::getRpcInfoMap() const {
    return _node->getRpcInfoMap();
}

NaviErrorPtr KernelComputeContext::getScopeError() const {
    return _node->getScopeError();
}

ErrorHandleStrategy KernelComputeContext::getScopeErrorHandleStrategy() const {
    return _node->getScopeErrorHandleStrategy();
}

void KernelComputeContext::reportScopeError(const NaviErrorPtr &error) {
    return _node->reportError(error);
}

void KernelComputeContext::reportScopeError(ErrorCode ec) {
    _node->setErrorCode(ec, nullptr);
}

int64_t KernelComputeContext::getRemainTimeMs() const {
    return _node->getRemainTimeMs();
}

}
