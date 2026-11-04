#include "navi/engine/PartState.h"

#include "navi/util/ReadyBitMap.h"

namespace navi {

MessageCounter::MessageCounter() {
    atomic_set(&_messageCount, 0);
}

void MessageCounter::incMessageCount() {
    atomic_inc(&_messageCount);
}

void MessageCounter::decMessageCount() {
    atomic_dec(&_messageCount);
}

int64_t MessageCounter::messageCount() const {
    return atomic_read(&_messageCount);
}

BorderPartState::BorderPartState(size_t edgeCount) {
    _portEofMap = InlineReadyBitMap::createReadyBitMap(edgeCount);
    _portEofMap.setFinish(false);
}

BorderPartState::~BorderPartState() {
    InlineReadyBitMap::freeReadyBitMap(_portEofMap);
}

void BorderPartState::setEof(size_t edgeId) {
    _portEofMap.setFinish(edgeId, true);
}

bool BorderPartState::eof() const {
    return _portEofMap.isFinish();
}

BorderState::BorderState() {}

bool BorderState::init(const PartInfo &partInfo, size_t edgeCount) {
    if (_partStates) {
        return false;
    }
    // TODO sub border part states?
    auto partCount = partInfo.getFullPartCount();
    auto partStates = (BorderPartState *)malloc(sizeof(BorderPartState) * partCount);
    for (int32_t i = 0; i < partCount; i++) {
        auto mem = &partStates[i];
        new (mem) BorderPartState(edgeCount);
    }
    _partInfo = partInfo;
    _partStates = partStates;
    return true;
}

BorderState::~BorderState() {
    if (_partStates) {
        for (NaviPartId partId = 0; partId < _partInfo.getFullPartCount(); partId++) {
            _partStates[partId].~BorderPartState();
        }
        free(_partStates);
    }
    _partStates = nullptr;
}

void BorderState::setEof(NaviPartId partId, size_t edgeId) {
    assert(_partInfo.isUsed(partId));
    _partStates[partId].setEof(edgeId);
}

bool BorderState::eof(NaviPartId partId) const {
    assert(_partInfo.isUsed(partId));
    return _partStates[partId].eof();
}

bool BorderState::eof() const {
    bool retEof = true;
    for (auto partId : _partInfo) {
        retEof = retEof && eof(partId);
    }
    return retEof;
}

void BorderState::incMessageCount(NaviPartId partId) {
    assert(_partInfo.isUsed(partId));
    _partStates[partId].incMessageCount();
}

void BorderState::decMessageCount(NaviPartId partId) {
    assert(_partInfo.isUsed(partId));
    _partStates[partId].decMessageCount();
}

int64_t BorderState::messageCount(NaviPartId partId) const {
    assert(_partInfo.isUsed(partId));
    return _partStates[partId].messageCount();
}

StreamPartState::StreamPartState()
    : _sending(false)
    , _collected(false)
    , _inited(false)
{
}

StreamPartState::~StreamPartState() {
}

bool StreamPartState::inited() const {
    return _inited;
}

void StreamPartState::setInited() {
    _inited = true;
}

StreamState::StreamState()
    : _partCount(1)
    , _partStates(nullptr)
{
}

StreamState::~StreamState() {
    if (_partStates) {
        for (NaviPartId partId = 0; partId < _partCount; partId++) {
            _partStates[partId].~StreamPartState();
        }
        free(_partStates);
    }
    InlineReadyBitMap::freeReadyBitMap(_partSendEofMap);
    InlineReadyBitMap::freeReadyBitMap(_partReceiveEofMap);
    InlineReadyBitMap::freeReadyBitMap(_partCancelMap);
}

bool StreamState::init() {
    if (_partStates) {
        return false;
    }
    auto allocSize = sizeof(StreamPartState) * _partCount;
    auto mem = (StreamPartState *)malloc(allocSize);
    for (NaviPartId partId = 0; partId < _partCount; partId++) {
        new (&mem[partId]) StreamPartState();
    }
    _partStates = mem;

    _partSendEofMap = InlineReadyBitMap::createReadyBitMap(_partCount);
    _partSendEofMap.setFinish(false);

    _partReceiveEofMap = InlineReadyBitMap::createReadyBitMap(_partCount);
    _partReceiveEofMap.setFinish(false);

    _partCancelMap = InlineReadyBitMap::createReadyBitMap( _partCount);
    _partCancelMap.setFinish(false);
    return true;
}

bool StreamState::streamInited() const {
    return nullptr != _partStates;
}

void StreamState::lock(NaviPartId partId) {
    _partStates[partId]._lock.lock();
}

void StreamState::unlock(NaviPartId partId) {
    _partStates[partId]._lock.unlock();
}

StreamPartState &StreamState::getPartState(NaviPartId partId) {
    return _partStates[partId];
}

void StreamState::setPartCount(NaviPartId partCount) {
    _partCount = partCount;
}

NaviPartId StreamState::getPartCount() const {
    return _partCount;
}

bool StreamState::inited(NaviPartId partId) const {
    return _partStates[partId].inited();
}

void StreamState::setInited(NaviPartId partId) {
    _partStates[partId].setInited();
}

void StreamState::incMessageCount(NaviPartId partId) {
    _partStates[partId].incMessageCount();
}

void StreamState::decMessageCount(NaviPartId partId) {
    _partStates[partId].decMessageCount();
}

bool StreamState::noMessage(NaviPartId partId) const {
    auto &partStat = _partStates[partId];
    return partStat.messageCount() <= 0 && partStat.inited();
}

bool StreamState::setCollected(NaviPartId partId) {
    auto &partStat = _partStates[partId];
    autil::ScopedLock scope(partStat._lock);
    if (partStat._collected) {
        return false;
    }
    partStat._collected = true;
    return true;
}

void StreamState::setSendEof(NaviPartId partId) {
    _partSendEofMap.setFinish(partId, true);
}

bool StreamState::sendEof(NaviPartId partId) const {
    return _partSendEofMap.isFinish(partId);
}

bool StreamState::sendEof() const {
    return _partSendEofMap.isFinish();
}

void StreamState::setReceiveEof(NaviPartId partId) {
    _partReceiveEofMap.setFinish(partId, true);
}

bool StreamState::receiveEof(NaviPartId partId) const {
    return _partReceiveEofMap.isFinish(partId);
}

bool StreamState::receiveEof() const {
    return _partReceiveEofMap.isFinish();
}

void StreamState::receiveCancel(NaviPartId partId) {
    _partCancelMap.setFinish(partId, true);
}

bool StreamState::cancelled(NaviPartId partId) const {
    return _partCancelMap.isFinish(partId);
}

}
