#include "navi/engine/LocalOutputPort.h"
#include "navi/engine/Node.h"
#include "navi/engine/TypeContext.h"
#include "navi/engine/BackPressureData.h"

namespace navi {

BPOutputStates::BPOutputStates() {
}

BPOutputStates::~BPOutputStates() {
}

void BPOutputStates::init(NaviPartId usedCount,
                          bool localMode,
                          const BackPressureQueueDef &queueDef,
                          bool backPressureEofStop)
{
    _backPressureEofStop = backPressureEofStop;
    _states.resize(usedCount);
    for (auto &state : _states) {
        state.maxDataCount = localMode ? 1 : queueDef.queue_high();
        atomic_set(&state.updateCount, 0);
    }
}

void BPOutputStates::updateTotalCount(NaviPartId index, int64_t totalCount) {
    auto &state = _states[index];
    state.dataCount = std::max((int64_t)state.dataCount, totalCount);
}

void BPOutputStates::updateMaxDataCount(NaviPartId index, int64_t maxDataCount, bool eof) {
    auto &state = _states[index];
    state.maxDataCount = std::max((int64_t)state.maxDataCount, maxDataCount);
    state.eof = eof;
    atomic_inc(&state.updateCount);
}

bool BPOutputStates::isReady(NaviPartId index) const {
    auto &state = _states[index];
    if (state.eof) {
        return !_backPressureEofStop;
    } else {
        return (state.maxDataCount < 0 || state.maxDataCount > state.dataCount);
    }
}

bool BPOutputStates::isFinish(NaviPartId index) const {
    auto &state = _states[index];
    return state.eof && _backPressureEofStop;
}

int64_t BPOutputStates::getTotalDataCount(NaviPartId index) const {
    auto &state = _states[index];
    return state.dataCount;
}

int64_t BPOutputStates::getMaxDataCount(NaviPartId index) const {
    auto &state = _states[index];
    return state.maxDataCount;
}

void BPOutputStates::fillState(const PartInfo &partInfo, KernelOutputPortStateDef *state) const {
    auto count = size();
    for (size_t i = 0; i < count; i++) {
        const auto &partState = _states[i];
        auto stateDef = state->add_bp_states();
        stateDef->set_part_id(partInfo.getUsedPartId(i));
        stateDef->set_update_count(atomic_read(&partState.updateCount));
        stateDef->set_data_count(partState.dataCount);
        stateDef->set_max_data_count(partState.maxDataCount);
        stateDef->set_eof(partState.eof);
    }
}

LocalOutputPort::LocalOutputPort(const EdgeDef &def, SubGraphBorder *border)
    : LocalPortBase(def.input(), def.output(), IOT_OUTPUT, border)
{
    _logger.addPrefix("LocalOutputPort");
}

LocalOutputPort::~LocalOutputPort() {
    DELETE_AND_SET_NULL(_bpOutputStates);
}

void LocalOutputPort::setCallback(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) {
    NAVI_LOG(SCHEDULE2, "set callback part [%d] eof [%d] node [%p] nodeReadyMap [%p]",
             partId, eof, _node, &_nodeReadyMap);
    if (!_node) {
        return;
    }
    NaviPartId index = INVALID_NAVI_PART_ID;
    NaviPartId usedCount = INVALID_NAVI_PART_COUNT;
    if (_bpOutputStates) {
        index = getPartInfo().getUsedPartIndex(partId);
        usedCount = _bpOutputStates->size();
        if (index < 0 || index >= usedCount) {
            NAVI_LOG(ERROR, "invalid back pressure partId: %d, total used count: %d, index: %d", partId, usedCount, index);
            return;
        }
        _bpOutputStates->updateTotalCount(index, totalDataCount);
    }
    bool ready = true;
    {
        autil::ScopedLock lock(_lock);
        if (_bpOutputStates) {
            ready = _bpOutputStates->isReady(index);
        }
        bool finish = eof;
        _nodeReadyMap.setReadyFinish(partId, ready, finish);
    }
    if (_bpOutputStates) {
        NAVI_LOG(SCHEDULE2,
                 "back pressure set callback,  partId: %d, total used count: %d, index: %d, eof: %d, totalDataCount: "
                 "%ld, queueSize: %ld, isReady: %d, isFinish: %d, maxDataCount: %ld, allReady: %d",
                 partId,
                 usedCount,
                 index,
                 eof,
                 totalDataCount,
                 queueSize,
                 _bpOutputStates->isReady(index),
                 _bpOutputStates->isFinish(index),
                 _bpOutputStates->getMaxDataCount(index),
                 _nodeReadyMap.isReady());
    }

    NAVI_LOG(SCHEDULE1,
             "update total data count to %ld, partId: %d, portId: %d, ready: %d, eof: %d, index: %d",
             totalDataCount,
             partId,
             getPortId(),
             ready,
             eof,
             index);
    scheduleNode();
}

void LocalOutputPort::getCallback(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) {
    NAVI_LOG(SCHEDULE2, "get callback part [%d] eof [%d] node [%p] nodeReadyMap [%p]",
             partId, eof, _node, &_nodeReadyMap);
    if (!_node) {
        return;
    }
    NaviPartId index = INVALID_NAVI_PART_ID;
    NaviPartId usedCount = INVALID_NAVI_PART_COUNT;
    if (_bpOutputStates) {
        index = getPartInfo().getUsedPartIndex(partId);
        usedCount = _bpOutputStates->size();
        if (index < 0 || index >= usedCount) {
            NAVI_LOG(ERROR, "invalid back pressure partId: %d, total used count: %d, index: %d", partId, usedCount, index);
            return;
        }
    }
    {
        autil::ScopedLock lock(_lock);
        bool ready = true;
        if (_bpOutputStates) {
            ready = _bpOutputStates->isReady(index);
        }
        _nodeReadyMap.setReady(partId, ready);
    }
    if (_bpOutputStates) {
        NAVI_LOG(SCHEDULE2,
                 "back pressure get callback,  partId: %d, total used count: %d, index: %d, eof: %d, totalDataCount: "
                 "%ld, queueSize: %ld, isReady: %d, isFinish: %d, maxDataCount: %ld, allReady: %d",
                 partId,
                 usedCount,
                 index,
                 eof,
                 totalDataCount,
                 queueSize,
                 _bpOutputStates->isReady(index),
                 _bpOutputStates->isFinish(index),
                 _bpOutputStates->getMaxDataCount(index),
                 _nodeReadyMap.isReady());
    }
    scheduleNode();
}

void LocalOutputPort::fillInputState(KernelOutputPortStateDef *state) const {
    fillState(state->mutable_port_state());
    state->set_ready(_nodeReadyMap.isReady());
    state->set_finish(_nodeReadyMap.isFinish());
    if (_bpOutputStates) {
        _bpOutputStates->fillState(_partInfo, state);
    }
}

void LocalOutputPort::initBPNotify() {
    auto localMode = (PQT_LOCAL_LOCAL_3 == _dataQueues->getQueueType());
    auto usedCount = getPartInfo().getUsedPartCount();
    _bpOutputStates = new BPOutputStates();
    _bpOutputStates->init(usedCount, localMode, getQueueDef()->queue(), _backPressureEofStop);
}

void LocalOutputPort::receiveBackPressure(NaviPartId partId,
                                          const DataPtr &data,
                                          bool eof)
{
    if (!_node) {
        return;
    }
    if (!_bpOutputStates) {
        NAVI_LOG(ERROR, "back pressure state not inited");
        return;
    }
    int64_t maxDataCount = -1;
    auto dataTyped = dynamic_cast<BackPressureData *>(data.get());
    if (dataTyped) {
        maxDataCount = dataTyped->maxDataCount;
    }
    auto index = getPartInfo().getUsedPartIndex(partId);
    auto usedCount = _bpOutputStates->size();
    if (index < 0 || index >= usedCount) {
        NAVI_LOG(ERROR, "invalid back pressure partId: %d, total used count: %d, index: %d", partId, usedCount, index);
        return;
    }
    _bpOutputStates->updateMaxDataCount(index, maxDataCount, eof);
    bool ready = false;
    bool finish = false;
    {
        autil::ScopedLock lock(_lock);
        ready = _bpOutputStates->isReady(index);
        finish = _bpOutputStates->isFinish(index);
        _nodeReadyMap.setReadyFinish(partId, ready, finish);
    }
    NAVI_LOG(
        SCHEDULE1,
        "receive back pressure, target [%s:%s] portId: %u, partId: %d, index: %d, ready: %d, allReady: %d, finish: %d, "
        "eof: %d, update max count to: %ld, totalDataCount: %ld, low: %ld, high: %ld, portId: %u, peerPortId: %u",
        getNode().c_str(),
        getPort().c_str(),
        _portId,
        partId,
        index,
        ready,
        _nodeReadyMap.isReady(),
        finish,
        eof,
        maxDataCount,
        _bpOutputStates->getTotalDataCount(index),
        getQueueDef()->queue().queue_low(),
        getQueueDef()->queue().queue_high(),
        _portId,
        _peerPortId);
    scheduleNode();
}
}
