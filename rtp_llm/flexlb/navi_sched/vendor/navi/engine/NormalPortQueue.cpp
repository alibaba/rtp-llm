#include "navi/engine/NormalPortQueue.h"
#include "navi/engine/Port.h"

namespace navi {

NormalPortQueue::NormalPortQueue() {
    _logger.addPrefix("normal");
}

NormalPortQueue::~NormalPortQueue() {}

ErrorCode NormalPortQueue::setEof(NaviPartId fromPartId) {
    return doSetEof(0, fromPartId);
}

ErrorCode NormalPortQueue::push(NaviPartId toPartId, NaviPartId fromPartId,
                                const DataPtr &data, bool eof, bool &retEof)
{
    assert(_partInfo.isUsed(toPartId));

    NaviPartId toBitMap;
    NaviPartId fromBitMap;
    NaviPartId toCallBack;
    NaviPartId fromCallBack;
    if (!convertPartId(_queueType, toPartId, fromPartId, toBitMap, fromBitMap,
                       toCallBack, fromCallBack))
    {
        return EC_UNKNOWN;
    }
    if (!checkPartId(toBitMap, fromBitMap)) {
        return EC_PART_ID_OVERFLOW;
    }
    int64_t totalDataCount = 0;
    int64_t queueSize = 0;
    bool forceEof = false;
    {
        auto &portDataQueue = _partQueues[toBitMap];
        autil::ScopedLock lock(portDataQueue.lock);
        if (portDataQueue.eofMap.isFinish(fromBitMap)) {
            return EC_DATA_AFTER_EOF;
        }
        portDataQueue.eofMap.setFinish(fromBitMap, eof);
        retEof = portDataQueue.eofMap.isFinish();
        forceEof = portDataQueue.forceEof;
        if (!forceEof) {
            portDataQueue.dataList.push_back(data);
            portDataQueue.dataCount++;
            portDataQueue.totalDataCount++;
        }
        totalDataCount = portDataQueue.totalDataCount;
        queueSize = portDataQueue.dataList.size();
    }
    if (!forceEof) {
        _downStream->incBorderMessageCount(toBitMap);
        _downStream->setCallback(toCallBack, retEof, totalDataCount, queueSize);
        _upStream->setCallback(fromCallBack, eof, totalDataCount, queueSize);
    } else {
        _downStream->setCallback(toCallBack, true, totalDataCount, queueSize);
        _upStream->setCallback(fromCallBack, true, totalDataCount, queueSize);
    }
    return EC_NONE;
}

ErrorCode NormalPortQueue::pop(NaviPartId toPartId, DataPtr &data, bool &eof) {
    if (!checkPartId(toPartId, 0)) {
        return EC_PART_ID_OVERFLOW;
    }
    int64_t totalDataCount = 0;
    int64_t queueSize = 0;
    {
        auto &portDataQueue = _partQueues[toPartId];
        autil::ScopedLock lock(portDataQueue.lock);
        if (0 == portDataQueue.dataCount) {
            eof = (portDataQueue.eofMap.isFinish() || portDataQueue.forceEof);
            data.reset();
            return EC_NO_DATA;
        }
        portDataQueue.dataCount--;
        eof = (portDataQueue.eofMap.isFinish() || portDataQueue.forceEof) && (portDataQueue.dataCount == 0);
        totalDataCount = portDataQueue.totalDataCount;
        auto &dataList = portDataQueue.dataList;
        assert(!dataList.empty());
        data = dataList.front();
        dataList.pop_front();
        queueSize = dataList.size();
    }
    _downStream->decBorderMessageCount(toPartId);
    _downStream->getCallback(toPartId, eof, totalDataCount, queueSize);
    _upStream->getCallback(toPartId, eof, totalDataCount, queueSize);
    NAVI_KERNEL_LOG(SCHEDULE2,
                    "partId: %d, pop data [%p] eof [%d], totalDataCount [%ld], queueSize [%ld]",
                    toPartId,
                    data.get(),
                    eof,
                    totalDataCount,
                    queueSize);
    return EC_NONE;
}

}
