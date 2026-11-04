#include "navi/engine/PortDataQueue.h"

#include "navi/engine/Port.h"
#include "navi/log/NaviLogger.h"
#include "navi/util/CommonUtil.h"

namespace navi {

PortDataQueue::PortDataQueue()
    : dataCount(0)
    , totalDataCount(0)
    , forceEof(false)
{
}

PortDataQueue::~PortDataQueue() {
    InlineReadyBitMap::freeReadyBitMap(eofMap);
}

void PortDataQueue::init(const PartInfo &eofPartInfo) {
    eofMap = eofPartInfo.createReadyBitMap();
    eofMap.setFinish(false);
}

PortDataQueues::PortDataQueues()
    : _partQueues(nullptr)
    , _upStream(nullptr)
    , _downStream(nullptr)
    , _storeType(PST_UNKNOWN)
    , _queueType(PQT_UNKNOWN)
{
}

PortDataQueues::~PortDataQueues() {
    if (_partQueues) {
        for (NaviPartId partId = 0; partId < _partInfo.getFullPartCount(); partId++) {
            _partQueues[partId].~PortDataQueue();
        }
        free(_partQueues);
        _partQueues = nullptr;
    }
}

bool PortDataQueues::init(const std::shared_ptr<NaviLogger> &logger,
                          const PartInfo &partInfo,
                          const PartInfo &eofPartInfo,
                          Port *upStream,
                          Port *downStream,
                          PortStoreType storeType,
                          PortQueueType queueType)
{
    assert(partInfo.getFullPartCount() > 0 && "full part count must be non-zero");
    // TODO queues size = UsedPartCount?
    size_t allocSize = sizeof(PortDataQueue) * partInfo.getFullPartCount();
    auto mem = (PortDataQueue *)malloc(allocSize);
    for (NaviPartId partId = 0; partId < partInfo.getFullPartCount(); partId++) {
        auto partQueue = &mem[partId];
        new (partQueue) PortDataQueue();
        partQueue->init(eofPartInfo);
    }
    _logger.object = this;
    _logger.logger = logger;
    _logger.addPrefix(
        "%s %s %p->%p %d %d", CommonUtil::getPortQueueType(queueType),
        CommonUtil::getStoreType(storeType),
        upStream, downStream,
        partInfo.getFullPartCount(), eofPartInfo.getFullPartCount());
    _partInfo = partInfo;
    _eofPartInfo = eofPartInfo;
    _partQueues = mem;
    _upStream = upStream;
    _downStream = downStream;
    _storeType = storeType;
    _queueType = queueType;
    NAVI_LOG(SCHEDULE2, "port data queues init, partInfo [%s] eofPartInfo [%s] "
             "storeType [%s] queueType [%s]",
             partInfo.ShortDebugString().c_str(), eofPartInfo.ShortDebugString().c_str(),
             CommonUtil::getStoreType(storeType),
             CommonUtil::getPortQueueType(queueType));
    return true;
}

PortStoreType PortDataQueues::getStoreType() const {
    return _storeType;
}

PortQueueType PortDataQueues::getQueueType() const {
    return _queueType;
}

ErrorCode PortDataQueues::doSetEof(NaviPartId toPartId, NaviPartId fromPartId) {
    bool retEof;
    if (PST_DATA == _storeType) {
        return push(toPartId, fromPartId, DataPtr(), true, retEof);
    } else {
        return pushSerialize(toPartId, fromPartId, (NaviPortData *)nullptr,
                             true, retEof);
    }
}

bool PortDataQueues::checkPartId(NaviPartId toPartId,
                                 NaviPartId fromPartId) const
{
    auto fullPartCount = _partInfo.getFullPartCount();
    auto eofPartCount = _eofPartInfo.getFullPartCount();
    auto ret = toPartId < fullPartCount && fromPartId < eofPartCount;
    if (!ret) {
        NAVI_LOG(ERROR,
                 "partId overflow, toPartId: %d, fromPartId: %d, partCount: %d, "
                 "eofPartCount: %d",
                 toPartId, fromPartId, fullPartCount, eofPartCount);
    }
    return ret;
}

int64_t PortDataQueues::size(NaviPartId toPartId) const {
    PortDataQueue &portDataQueue = _partQueues[toPartId];
    autil::ScopedLock lock(portDataQueue.lock);
    return portDataQueue.dataCount;
}

int64_t PortDataQueues::totalDataCount(NaviPartId toPartId) const {
    PortDataQueue &portDataQueue = _partQueues[toPartId];
    autil::ScopedLock lock(portDataQueue.lock);
    return portDataQueue.totalDataCount;
}

bool PortDataQueues::convertPartId(PortQueueType queueType, NaviPartId to,
                                   NaviPartId from, NaviPartId &toBitMap,
                                   NaviPartId &fromBitMap,
                                   NaviPartId &toCallBack,
                                   NaviPartId &fromCallBack)
{
    switch (queueType) {
    case PQT_LOCAL_REMOTE_SERVER_1:
    case PQT_LOCAL_REMOTE_CLIENT_2:
    case PQT_LOCAL_LOCAL_3:
        fromBitMap = 0;
        toBitMap = to;
        fromCallBack = to;
        toCallBack = to;
        break;
    case PQT_REMOTE_LOCAL_SERVER_4:
        fromBitMap = 0;
        toBitMap = from;
        fromCallBack = 0;
        toCallBack = from;
        break;
    case PQT_REMOTE_LOCAL_CLIENT_5:
        fromBitMap = 0;
        toBitMap = from;
        fromCallBack = from;
        toCallBack = from;
        break;
    case PQT_REMOTE_REMOTE_CLIENT_6:
        fromBitMap = from;
        toBitMap = to;
        fromCallBack = from;
        toCallBack = to;
        break;
    case PQT_UNKNOWN:
    default:
        assert(false);
        fromBitMap = from;
        toBitMap = to;
        fromCallBack = from;
        toCallBack = to;
        return false;
    }
    return true;
}

ErrorCode PortDataQueues::pushSerialize(NaviPartId toPartId,
                                        NaviPartId fromPartId,
                                        NaviPortData *data,
                                        bool eof,
                                        bool &retEof)
{
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
        NAVI_LOG(ERROR, "partId overflow, input toPartId: %d, "
                        "input fromPartId: %d, eof: %d",
                 toPartId, fromPartId, eof);
        return EC_PART_ID_OVERFLOW;
    }
    assert(_partInfo.isUsed(toBitMap));
    assert(_eofPartInfo.isUsed(fromBitMap));

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
            NaviPortData temp;
            if (data) {
                temp.Swap(data);
            } else {
                if (!SerializeData::serialize(_logger, toPartId, fromPartId, nullptr, true, nullptr, temp)) {
                    return EC_SERIALIZE;
                }
            }
            portDataQueue.serializeDataList.emplace_back(std::move(temp));
            portDataQueue.dataCount++;
            portDataQueue.totalDataCount++;
        }
        totalDataCount = portDataQueue.totalDataCount;
        // Old-ABI std::list::size() scans the backlog while holding this lock.
        queueSize = portDataQueue.dataCount;
        NAVI_LOG(SCHEDULE2,
                 "upstream [%p,%s], downstream [%p,%s], portId: %d, portKey: %s, from: %d, to: %d, eof: %d, portEof: "
                 "%d, dataCount: %ld, totalDataCount: %ld, retEof: %d, forceEof: %d",
                 _upStream,
                 typeid(*_upStream).name(),
                 _downStream,
                 typeid(*_downStream).name(),
                 _upStream->getPortId(),
                 _upStream->getPortKey().c_str(),
                 fromPartId,
                 toPartId,
                 eof,
                 portDataQueue.eofMap.isFinish(),
                 portDataQueue.dataCount,
                 portDataQueue.totalDataCount,
                 retEof,
                 portDataQueue.forceEof);
    }
    if (!forceEof) {
        _upStream->setCallback(fromCallBack, eof, totalDataCount, queueSize);
        _downStream->incBorderMessageCount(toBitMap);
        _downStream->setCallback(toCallBack, retEof, totalDataCount, queueSize);
    }
    return EC_NONE;
}

ErrorCode PortDataQueues::popSerialize(NaviPartId toPartId, NaviPortData *data,
                                       bool &eof)
{
    if (!checkPartId(toPartId, 0)) {
        NAVI_LOG(ERROR, "partId overflow, toPartId: %d", toPartId);
        return EC_PART_ID_OVERFLOW;
    }
    int64_t totalDataCount = 0;
    int64_t queueSize = 0;
    {
        auto &portDataQueue = _partQueues[toPartId];
        autil::ScopedLock lock(portDataQueue.lock);
        if (0 == portDataQueue.dataCount) {
            eof = (portDataQueue.eofMap.isFinish() || portDataQueue.forceEof);
            return EC_NO_DATA;
        }
        portDataQueue.dataCount--;
        eof = (portDataQueue.eofMap.isFinish() || portDataQueue.forceEof) && (portDataQueue.dataCount == 0);
        totalDataCount = portDataQueue.totalDataCount;
        auto &dataList = portDataQueue.serializeDataList;
        assert(!dataList.empty());
        auto &listData = dataList.front();
        assert(data);
        data->Swap(&listData);
        dataList.pop_front();
        queueSize = portDataQueue.dataCount;
    }
    _downStream->decBorderMessageCount(toPartId);
    _downStream->getCallback(toPartId, eof, totalDataCount, queueSize);
    _upStream->getCallback(toPartId, eof, totalDataCount, queueSize);
    return EC_NONE;
}

// for LocalInputPort
ErrorCode PortDataQueues::forceEof(NaviPartId toPartId, NaviPartId fromPartId) {
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
        NAVI_LOG(ERROR, "partId overflow, input toPartId: %d, input fromPartId: %d", toPartId, fromPartId);
        return EC_PART_ID_OVERFLOW;
    }
    assert(_partInfo.isUsed(toBitMap));
    assert(_eofPartInfo.isUsed(fromBitMap));

    std::list<DataPtr> dataList;
    std::list<NaviPortData> serializeDataList;
    int64_t totalDataCount = 0;
    int64_t queueSize = 0;
    bool alreadyEof = false;
    {
        auto &portDataQueue = _partQueues[toBitMap];
        autil::ScopedLock lock(portDataQueue.lock);
        portDataQueue.forceEof = true;
        alreadyEof = portDataQueue.eofMap.isFinish();
        totalDataCount = portDataQueue.totalDataCount;
        queueSize = portDataQueue.dataCount;
        dataList = std::move(portDataQueue.dataList);
        serializeDataList = std::move(portDataQueue.serializeDataList);
        portDataQueue.dataCount = 0;
        NAVI_LOG(SCHEDULE2,
                 "force eof, upstream [%p,%s], downstream [%p,%s], portId: %d, portKey: %s, from: %d, to: %d, portEof: "
                 "%d, dataCount: %ld, totalDataCount: %ld",
                 _upStream,
                 typeid(*_upStream).name(),
                 _downStream,
                 typeid(*_downStream).name(),
                 _upStream->getPortId(),
                 _upStream->getPortKey().c_str(),
                 fromPartId,
                 toPartId,
                 portDataQueue.eofMap.isFinish(),
                 portDataQueue.dataCount,
                 portDataQueue.totalDataCount);
    }
    dataList.clear();
    serializeDataList.clear();
    if (!alreadyEof) {
        _upStream->setCallback(fromCallBack, true, totalDataCount, queueSize);
        _downStream->incBorderMessageCount(toBitMap);
        _downStream->setCallback(toCallBack, true, totalDataCount, queueSize);
    }
    return EC_NONE;
}

void PortDataQueues::fillInputState(PortQueueStateDef *state) const {
    state->set_full_part_count(_partInfo.getFullPartCount());
    state->set_store_type((PortStoreTypeDef)_storeType);
    state->set_queue_type((PortQueueTypeDef)_queueType);
    for (auto partId : _partInfo) {
        auto partDef = state->add_parts();
        partDef->set_part_id(partId);
        auto &portDataQueue = _partQueues[partId];
        autil::ScopedLock lock(portDataQueue.lock);
        partDef->set_total_data_count(portDataQueue.totalDataCount);
        partDef->set_data_count(portDataQueue.dataCount);
        partDef->set_eof(portDataQueue.eofMap.isFinish());
        partDef->set_force_eof(portDataQueue.forceEof);
    }
}

std::ostream& operator<<(std::ostream& os, const PortDataQueues& portDataQueues) {
    return os << &portDataQueues << ","
              << typeid(portDataQueues).name() << ","
              << portDataQueues._partInfo.ShortDebugString() << ","
              << portDataQueues._eofPartInfo.ShortDebugString();
}

}
