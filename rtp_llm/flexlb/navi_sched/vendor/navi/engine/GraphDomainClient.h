#pragma once

#include "navi/engine/GraphDomainRemoteBase.h"
#include <multi_call/interface/SearchService.h>

namespace navi {

class NaviClientStream;

class GraphDomainClient : public GraphDomainRemoteBase
{
public:
    GraphDomainClient(Graph *graph,
                      const std::string &bizName);
    ~GraphDomainClient();
private:
    GraphDomainClient(const GraphDomainClient &);
    GraphDomainClient &operator=(const GraphDomainClient &);
private:
    ErrorCode doPreInit() override;
    bool postInit() override;
    bool run() override;
    bool receive(const multi_call::GigStreamMessage &message) override;
    bool receiveCancel(const multi_call::GigStreamMessage &message, multi_call::MultiCallErrorCode ec) override;
    bool doSend(NaviPartId gigPartId, NaviMessage &naviMessage) override;
    NaviPartId getStreamPartId(NaviPartId partId) override;
    void tryCancel(const multi_call::GigStreamBasePtr &stream) override;
    bool onSendFailure(NaviPartId gigPartId) override;
    void collectMetric(GraphMetric *graphMetric) override;
public:
    const std::string &getBizName() const {
        return _bizName;
    }
    const PartInfo &getPartInfo() const;
    multi_call::GigStreamRpcInfoVec stealStreamRpcInfo();
private:
    void doRun(const std::vector<NaviPartId> &partIds);
    void finishPart(NaviPartId gigPartId, const multi_call::GigStreamBasePtr &stream, const NaviErrorPtr &error);
    void addResult(NaviPartId partId, NaviMessage *naviMessage,
                   bool withRpcInfo, const NaviErrorPtr &error);
    bool collectRpcInfo(NaviPartId partId, multi_call::GigStreamRpcInfo &info);
    void receiveCallBack(NaviPartId gigPartId, bool forceStop);
    bool initTimeout(const SubGraphOption &option);
    bool initGigStream(const LocationDef &location);
    bool initDirectLocation(
        const LocationDef &location,
        const std::shared_ptr<NaviClientStream> &stream);
    void addTags(const std::shared_ptr<NaviClientStream> &stream, const GigInfoDef &gigInfo) const;
    ErrorCode fillInitMessage(NaviPartId partId, NaviMessage &naviMessage) const;
    bool sendEof(NaviPartId partId) const;
    bool receiveEof() const;
    void notifyFinishAsync(ErrorCode ec);
    std::shared_ptr<NaviClientStream> createClientStream();
    void initStreamRpcParam(int64_t globalStartTimeUs, multi_call::StreamRpcParam &streamRpcParam);
    bool clientReceive(const multi_call::GigStreamMessage &message, const NaviErrorPtr &error);
    void receivePartResult(NaviMessage *naviMessage);
private:
    friend class DomainRunItem;
private:
    std::string _bizName;
    PartInfo _partInfo;
    std::string _subGraphStr;
    NaviCompressType _compressType = CT_NONE;
    RpcGraphResultPtr _rpcResult;
    int64_t _rpcTimeoutMs = 0;
    int64_t _subGraphTimeoutMs = 0;
    mutable autil::ThreadMutex _metricLock;
    std::vector<KernelMetricPtr> _metrics;
    std::shared_ptr<TopoNodeParam> _topoParam;
    // for collect malloc
    multi_call::GigStreamBasePtr _clientStream;
};

}
