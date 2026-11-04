#pragma once

#include "navi/common.h"
#include "navi/engine/BorderConnectInfo.h"
#include "navi/engine/Data.h"
#include "navi/engine/GraphParam.h"
#include "navi/engine/NormalPortQueue.h"
#include "navi/engine/ShufflePortQueue.h"
#include "navi/engine/Type.h"

namespace navi {

class SubGraphBorder;
class PortDataQueues;

class Port {
public:
    Port(const std::string &node,
         const std::string &port,
         int32_t nodeIdex,
         const std::string &peerNode,
         const std::string &peerPort,
         IoType ioType,
         PortStoreType storeType,
         SubGraphBorder *border);
    Port(const NodePortDef &nodePort,
         const NodePortDef &peerPort,
         IoType ioType,
         PortStoreType storeType,
         SubGraphBorder *border);
    virtual ~Port();
public:
    const std::string &getNode() const;
    int32_t getNodeIndex() const;
    const std::string &getPort() const;
    const std::string &getPortKey() const;
    const std::string &getPeerNode() const;
    const std::string &getPeerPort() const;
    IoType getIoType() const;
    bool isLocalPort() const;
    void setPortInfo(PortId portId,
                     PortId peerPortId,
                     const std::string &portNodeName,
                     int32_t portNodeId,
                     bool backPressureEofStop);
    PortId getPortId() const;
    PortId getPeerPortId() const;
    const std::string &getPortNodeName() const;
    int32_t getPortNodeId() const;
    bool hasUpStream() const;
    bool hasDownStream() const;
    bool setDownStream(Port *port, const BorderConnectInfo &connectInfo);
    void bindDataType(const Type *dataType);
    const Type *getDataType() const;
    void setPartInfo(const PartInfo &partInfo);
    const PartInfo &getPartInfo() const;
    SubGraphBorder *getPeerBorder() const;
    void setPortBorderIndex(size_t index);
    size_t getPortBorderIndex() const;
    SubGraphBorder *getBorder() const;
    void setLogger(const NaviObjectLogger &logger, bool needLogPrefix);
    void setQueueDef(const EdgeQueueDef *queue, bool backPressurePort);
    const EdgeQueueDef *getQueueDef() const;
    bool isBackPressurePort() const;
public:
    friend std::ostream& operator<<(std::ostream& os, const Port& port);
public:
    ErrorCode setData(std::vector<DataPtr> &dataVec, bool eof);
    ErrorCode setData(NaviPartId partId, const DataPtr &data, bool eof);
    ErrorCode setData(NaviPortData &data, bool &eof);
    ErrorCode setEof(NaviPartId fromPartId);
    ErrorCode setEofAll();
    ErrorCode setForceEofAll();
private:
    ErrorCode setDataToData(NaviPartId toPartId, NaviPartId fromPartId,
                            const DataPtr &data, bool eof, bool &retEof);
    ErrorCode setDataToSerialize(NaviPartId toPartId, NaviPartId fromPartId,
                                 const DataPtr &data, bool eof);
    ErrorCode setSerializeToSerialize(NaviPartId toPartId,
                                      NaviPartId fromPartId, NaviPortData &data,
                                      bool &retEof);
    ErrorCode setSerializeToData(const NaviPortData &data, bool &retEof);
public:
    ErrorCode getData(NaviPartId toPartId, DataPtr &data, bool &eof);
    ErrorCode getData(NaviPartId toPartId, NaviPortData &data, bool &eof);
private:
    ErrorCode getDataToData(NaviPartId partId, DataPtr &data, bool &eof);
    ErrorCode getSerializeToData(NaviPartId partId, DataPtr &data, bool &eof);
    ErrorCode getSerializeToSerialize(NaviPartId partId, NaviPortData &data,
                                      bool &eof);
    ErrorCode getDataToSerialize(NaviPartId partId,
                                 NaviPortData &serializeData, bool &eof);
public:
    virtual void setCallback(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) {
    }
    virtual void getCallback(NaviPartId partId, bool eof, int64_t totalDataCount, int64_t queueSize) {
    }
    virtual void fillInputState(KernelOutputPortStateDef *state) const {
    }
    virtual void fillOutputState(KernelInputPortStateDef *state) const {
    }
public:
    void incBorderMessageCount(NaviPartId partId);
    void decBorderMessageCount(NaviPartId partId);
protected:
    int64_t getQueueSize(NaviPartId partId);
    void fillState(PortStateDef *state) const;
private:
    bool initDataType(const NaviPortData &data);
    bool initDataType(const DataPtr &data);
    bool initDataTypeByTypeStr(const std::string &typeStr);
    bool link(Port *downStream, const BorderConnectInfo &connectInfo);
    NaviPartId getFromPartId() const;
protected:
    DECLARE_LOGGER();
    std::string _node;
    std::string _port;
    int32_t _nodeIdex;
    std::string _peerNode;
    std::string _peerPort;
    std::string _portKey;
    IoType _ioType;
    Port *_upStream;
    Port *_downStream;
    const Type *_dataType;
    PortStoreType _storeType;
    PortId _portId;
    PortId _peerPortId;
    std::string _portNodeName;
    int32_t _portNodeId;
    PartInfo _partInfo;
    size_t _borderIndex;
    PortDataQueues *_dataQueues;
    SubGraphBorder *_border;
    GraphParam *_param;
    const EdgeQueueDef *_queue;
    bool _backPressurePort;
    bool _backPressureEofStop;
};

}
