#pragma once

#include "navi/common.h"
#include "navi/builder/BorderId.h"
#include "navi/engine/PartState.h"
#include "navi/engine/Port.h"
#include <vector>

namespace navi {

class GraphDomain;
class SubGraphBorder;
NAVI_TYPEDEF_PTR(SubGraphBorder);

typedef NaviUnorderedMap<BorderId, SubGraphBorderPtr, BorderIdHash> SubGraphBorderMap;
typedef std::shared_ptr<SubGraphBorderMap> SubGraphBorderMapPtr;

class SubGraphBorder {
private:
    struct SamePortVec {
        uint32_t linkIndex = 0;
        NaviVector<Port *, 2> ports;
    };
public:
    SubGraphBorder(const NaviLoggerPtr &logger, GraphParam *param,
                   NaviPartId partId, const BorderId &borderId);
    virtual ~SubGraphBorder();
public:
    bool init(const PartInfoDef &partInfo, const BorderDef &borderDef,
              GraphDomain *domain, bool backPressureEofStop);
    GraphParam *getGraphParam() const;
    Port *getPort(const std::string &node, const std::string &port);
    Port *getPort(PortId portId) const;
    bool linkTo(SubGraphBorder &toBorder);
    bool linked() const;
    const BorderId &getBorderId() const;
    std::string getBorderIdStr() const;
    const NaviVector<Port *, 4> &getPortVec() const;
    bool initBackPressure(const SubGraphBorderMapPtr &borderMap);
public:
    void notifySend(NaviPartId partId);
    bool collectBorderMessage(NaviPartId partId, NaviBorderData &borderMessage);
    bool receive(NaviBorderData &borderData);
    void incMessageCount(NaviPartId partId);
    void decMessageCount(NaviPartId partId);
private:
    void initBorderInfo(const PartInfoDef &partInfo, GraphId peer,
                        GraphDomain *domain);
    void initPartInfo(const PartInfoDef &partInfoDef, GraphDomain *domain);
    Port *createPort(IoType ioType, const BorderEdgeDef &borderEdgeDef, bool backPressureEofStop);
    Port *doCreateLocalPort(IoType ioType, const EdgeDef &def);
    Port *doCreateRemotePort(IoType ioType, const EdgeDef &def);
    void initPort(Port *port, const BorderEdgeDef &borderEdgeDef, bool backPressureEofStop);
    bool addPort(Port *port);
    void initPortIndexMap();
    bool initBorderState(NaviPartId partCount);
    bool doCollectBorderMessage(NaviPartId partId,
                                NaviBorderData &borderMessage);
    bool collect(NaviPartId partId, NaviBorderData &borderMessage);
    ErrorCode collectPortData(Port *port, NaviPartId partId,
                              NaviBorderData &borderMessage);
    void fillBorderId(NaviBorderData &borderData);
    bool getBorderConnectInfo(const SubGraphBorder &toBorder,
                              BorderConnectInfo &connectInfo) const;
    bool initBorderState(const BorderConnectInfo &connectInfo,
                         SubGraphBorder &toBorder);
    bool doInitBorderState(const PartInfo &partInfo);
    void initQueueType(const BorderConnectInfo &connectInfo,
                       SubGraphBorder &toBorder);
    bool doLink(const BorderConnectInfo &connectInfo, SubGraphBorder &toBorder);
    bool bindBPPort(Port *port, const SubGraphBorderMapPtr &borderMap);
    bool isOnClient(const SubGraphBorder &toBorder) const;
    bool setSendEof(NaviPartId from, NaviPartId to, size_t edgeId);
    bool setReceiveEof(NaviPartId from, NaviPartId to, size_t edgeId);
public:
    void setBorderDomainIndex(size_t index);
    size_t getBorderDomainIndex() const;
    const PartInfo &getPartInfo() const;
    NaviPartId getPartId() const;
    bool isLocalBorder() const;
    bool eof() const;
    bool eof(NaviPartId partId) const;
    ErrorCode setEof(NaviPartId fromPartId);
public:
    friend std::ostream& operator<<(std::ostream& os, const SubGraphBorder& subBorder);
private:
    DECLARE_LOGGER();
    GraphParam *_param;
    NaviPartId _partId;
    BorderId _borderId;
    bool _localBorder;
    size_t _domainIndex;
    PartInfo _partInfo;
    GraphDomain *_domain;
    NaviUnorderedMap<std::string, SamePortVec> _portNameMap;
    NaviUnorderedMap<PortId, Port *> _portIdMap;
    NaviVector<Port *, 4> _portVec;
    BorderState _borderState;
    PortQueueType _queueType;
    bool _linked;
};

}
