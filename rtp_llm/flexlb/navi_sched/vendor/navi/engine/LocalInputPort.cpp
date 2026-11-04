#include "navi/engine/LocalInputPort.h"
#include "navi/engine/LocalOutputPort.h"
#include "navi/engine/Node.h"
#include "navi/engine/SubGraphBorder.h"
#include "navi/engine/TypeContext.h"
#include "navi/engine/BackPressureData.h"
#include "navi/util/ReadyBitMap.h"

namespace navi {

BPInputStates::BPInputStates() {
}

BPInputStates::~BPInputStates() {
}

void BPInputStates::init(NaviPartId usedCount, bool localMode) {
    _localMode = localMode;
    _states.resize(usedCount);
    for (auto &state : _states) {
        atomic_set(&state.updateCount, 0);
    }
}

int64_t BPInputStates::updateMaxCount(NaviPartId index,
                                      int64_t totalDataCount,
                                      int64_t queueSize,
                                      const BackPressureQueueDef &queueDef)
{
    auto queueLow = _localMode ? 0 : queueDef.queue_low();
    if (totalDataCount < queueLow || queueSize > queueLow) {
        return -1;
    }
    auto queueHigh = _localMode ? 1 : queueDef.queue_high();
    auto consumed = totalDataCount - queueSize;
    auto newMaxCount = consumed + queueHigh;
    auto &state = _states[index];
    while (true) {
        auto oldValue = state.notifyMaxCount;
        if (newMaxCount <= oldValue) {
            return -1;
        }
        if (cmpxchg(&state.notifyMaxCount, oldValue, newMaxCount)) {
            atomic_inc(&state.updateCount);
            return newMaxCount;
        }
    }
    return -1;
}

void BPInputStates::fillState(const PartInfo &partInfo, KernelInputPortStateDef *state) const {
    auto count = size();
    for (size_t i = 0; i < count; i++) {
        const auto &partState = _states[i];
        auto stateDef = state->add_bp_states();
        stateDef->set_part_id(partInfo.getUsedPartId(i));
        stateDef->set_update_count(atomic_read(&partState.updateCount));
        stateDef->set_notify_max_count(partState.notifyMaxCount);
    }
}

LocalInputPort::LocalInputPort(const EdgeDef &def,
                               SubGraphBorder *border)
    : LocalPortBase(def.output(), def.input(), IOT_INPUT, border)
{
    _logger.addPrefix("LocalInputPort");
}

LocalInputPort::~LocalInputPort() {
    DELETE_AND_SET_NULL(_bpInputStates);
}

void LocalInputPort::setCallback(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) {
    if (_bpPort) {
        sendBackPressure(partId, eof, totalDataCount, queueSize);
    }
    if (_bpNotifyPort) {
        notifyBP(partId);
    }
    NAVI_LOG(SCHEDULE2, "set callback part [%d] eof [%d] node [%p] nodeReadyMap [%p]",
             partId, eof, _node, &_nodeReadyMap);
    _border->notifySend(partId);
    if (!_node) {
        return;
    }
    doCallBack(true, partId, eof);
}

void LocalInputPort::getCallback(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) {
    if (_bpPort) {
        sendBackPressure(partId, eof, totalDataCount, queueSize);
    }
    NAVI_LOG(SCHEDULE2,
             "get callback part [%d] eof [%d] node [%p]",
             partId,
             eof,
             _node);
    if (!_node) {
        return;
    }
    doCallBack(false, partId, eof);
}

void LocalInputPort::doCallBack(bool isSet, NaviPartId partId, bool eof) {
    {
        autil::ScopedLock lock(_lock);
        auto queueSize = getQueueSize(partId);
        bool ready = queueSize > 0;
        bool finish = eof && (queueSize == 0);
        NAVI_LOG(SCHEDULE2, "do callback part [%d] ready [%d] finish [%d] optional [%d]",
                 partId, ready, finish,
                 _nodeReadyMap.isOptional(partId));
        _nodeReadyMap.setReadyFinish(partId, ready, finish);
    }
    NAVI_LOG(SCHEDULE2, "after set nodeReadyMap [%s]",
             autil::StringUtil::toString(_nodeReadyMap).c_str());
    scheduleNode();
}

void LocalInputPort::fillOutputState(KernelInputPortStateDef *state) const {
    fillState(state->mutable_port_state());
    state->set_ready(_nodeReadyMap.isReady());
    state->set_finish(_nodeReadyMap.isFinish());
    if (_bpInputStates) {
        _bpInputStates->fillState(_partInfo, state);
    }
}

bool LocalInputPort::setBPPort(Port *port) {
    if (!port->getQueueDef()) {
        NAVI_LOG(ERROR, "back pressure port has no queue def");
        return false;
    }
    auto localMode = (PQT_LOCAL_LOCAL_3 == _dataQueues->getQueueType());
    setQueueDef(port->getQueueDef(), false);
    _bpPort = port;
    _bpInputStates = new BPInputStates();
    _bpInputStates->init(getPartInfo().getUsedPartCount(), localMode);
    return true;
}

bool LocalInputPort::setBPNotifyPort(Port *port) {
    auto portTyped = dynamic_cast<LocalOutputPort *>(port);
    if (!portTyped) {
        NAVI_LOG(ERROR, "bp notify port is not local output port, port type: %s", typeid(*port).name());
        return false;
    }
    portTyped->setQueueDef(getQueueDef(), false);
    portTyped->initBPNotify();
    _bpNotifyPort = portTyped;
    return true;
}

void LocalInputPort::sendBackPressure(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) {
    if (eof) {
        _bpPort->setData(partId, nullptr, eof);
        return;
    }
    if (!_bpInputStates) {
        NAVI_LOG(ERROR, "back pressure state not inited");
        return;
    }
    auto index = getPartInfo().getUsedPartIndex(partId);
    auto usedCount = _bpInputStates->size();
    if (index < 0 || index >= usedCount) {
        NAVI_LOG(ERROR, "invalid back pressure partId: %d, total used count: %d, index: %d", partId, usedCount, index);
        return;
    }
    const auto &queueDef = _bpPort->getQueueDef()->queue();
    auto newMaxCount = _bpInputStates->updateMaxCount(index, totalDataCount, queueSize, queueDef);
    NAVI_LOG(SCHEDULE1,
             "send back pressure, target [%s:%s] portId [%u], bpPortId [%u], bpPeerPortId [%u], partId: %d, update max "
             "count to: %ld, totalDataCount: %ld, queueSize: %ld, low: %ld, high: %ld, portId: %u, peerPortId: %u",
             getPeerNode().c_str(),
             getPeerPort().c_str(),
             _peerPortId,
             _bpPort->getPortId(),
             _bpPort->getPeerPortId(),
             partId,
             newMaxCount,
             totalDataCount,
             queueSize,
             queueDef.queue_low(),
             queueDef.queue_high(),
             _portId,
             _peerPortId);
    if (newMaxCount < 0) {
        return;
    }
    auto data = std::make_shared<BackPressureData>();
    data->maxDataCount = newMaxCount;
    _bpPort->setData(partId, data, eof);
}

void LocalInputPort::notifyBP(NaviPartId partId) {
    DataPtr data;
    bool eof = false;
    auto ec = getData(partId, data, eof);
    if (EC_NONE != ec && EC_NO_DATA != ec) {
        return;
    }
    _bpNotifyPort->receiveBackPressure(partId, data, eof);
}

}
