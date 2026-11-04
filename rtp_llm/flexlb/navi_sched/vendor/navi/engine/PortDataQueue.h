#pragma once

#include <list>
#include "navi/common.h"
#include "navi/engine/Data.h"
#include "navi/engine/PartInfo.h"
#include "navi/engine/SerializeData.h"
#include "navi/proto/NaviStream.pb.h"
#include "navi/util/ReadyBitMap.h"


namespace navi {

class NaviLogger;
class PartInfo;

class PortDataQueue {
public:
    PortDataQueue();
    ~PortDataQueue();
public:
    void init(const PartInfo &eofPartInfo);
public:
    autil::ThreadMutex lock;
    int64_t dataCount;
    int64_t totalDataCount;
    std::list<DataPtr> dataList;
    std::list<NaviPortData> serializeDataList;
    InlineReadyBitMap eofMap;
    bool forceEof;
};

class Port;

class PortDataQueues {
public:
    PortDataQueues();
    virtual ~PortDataQueues();
public:
    bool init(const std::shared_ptr<NaviLogger> &logger,
              const PartInfo &partInfo,
              const PartInfo &eofPartInfo,
              Port *upStream,
              Port *downStream,
              PortStoreType storeType,
              PortQueueType queueType);
    PortStoreType getStoreType() const;
    PortQueueType getQueueType() const;
    int64_t size(NaviPartId toPartId) const;
    int64_t totalDataCount(NaviPartId toPartId) const;
public:
    virtual ErrorCode setEof(NaviPartId fromPartId) = 0;
    virtual ErrorCode push(NaviPartId toPartId, NaviPartId fromPartId,
                           const DataPtr &data, bool eof, bool &retEof) = 0;
    virtual ErrorCode pop(NaviPartId toPartId, DataPtr &data, bool &eof) = 0;
    ErrorCode pushSerialize(NaviPartId toPartId, NaviPartId fromPartId,
                            NaviPortData *data, bool eof, bool &retEof);
    ErrorCode popSerialize(NaviPartId toPartId, NaviPortData *data, bool &eof);
    ErrorCode forceEof(NaviPartId toPartId, NaviPartId fromPartId);
    void fillInputState(PortQueueStateDef *state) const;
protected:
    ErrorCode doSetEof(NaviPartId toPartId, NaviPartId fromPartId);
    bool checkPartId(NaviPartId toPartId, NaviPartId fromPartId) const;
public:
    static bool convertPartId(PortQueueType queueType, NaviPartId to,
                              NaviPartId from, NaviPartId &toBitMap,
                              NaviPartId &fromBitMap, NaviPartId &toCallBack,
                              NaviPartId &fromCallBack);
    friend std::ostream& operator<<(std::ostream& os, const PortDataQueues& portDataQueues);
protected:
    DECLARE_LOGGER();
    PartInfo _partInfo;
    PartInfo _eofPartInfo;
    PortDataQueue *_partQueues;
    Port *_upStream;
    Port *_downStream;
    PortStoreType _storeType;
    PortQueueType _queueType;
};

}
