#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamBase.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamRpcInfo.h"
#include "navi/distribute/StreamMetric.h"
#include "navi/engine/GraphDomain.h"
#include "navi/engine/NaviWorkerBase.h"
#include "navi/engine/PartState.h"
#include "navi/proto/NaviStream.pb.h"
#include "navi/util/ReadyBitMap.h"

namespace navi {

class NaviStreamBase;

class GraphDomainRemoteBase
    : public GraphDomain,
      public std::enable_shared_from_this<GraphDomainRemoteBase>
{
public:
    GraphDomainRemoteBase(Graph *graph, GraphDomainType type);
    ~GraphDomainRemoteBase();
private:
    GraphDomainRemoteBase(const GraphDomainRemoteBase &);
    GraphDomainRemoteBase &operator=(const GraphDomainRemoteBase &);
public:
    void addSubBorder(const SubGraphBorderPtr &border);
public:
    void notifySend(NaviPartId partId) override;
    void setStream(const multi_call::GigStreamBasePtr &stream);
public:
    virtual bool receive(const multi_call::GigStreamMessage &message) = 0;
    virtual bool receiveCancel(const multi_call::GigStreamMessage &message, multi_call::MultiCallErrorCode ec) = 0;
private:
    virtual bool doSend(NaviPartId gigPartId, NaviMessage &naviMessage) = 0;
    virtual NaviPartId getStreamPartId(NaviPartId partId) = 0;
    virtual void tryCancel(const multi_call::GigStreamBasePtr &stream) = 0;
    virtual bool onSendFailure(NaviPartId gigPartId) = 0;
public:
    bool preInit() override;
    virtual ErrorCode doPreInit();
    void doFinish(ErrorCode ec) override;
    void incMessageCount(NaviPartId partId) override;
    void decMessageCount(NaviPartId partId) override;
protected:
    bool bindStream();
    void unBindStream();
    multi_call::GigStreamBasePtr getStream() const;
    StreamMetricPtr stealStreamMetric();
    bool doReceive(const multi_call::GigStreamMessage &message);
private:
    NaviStreamBase *getStreamBase();
    void send(NaviPartId partId);
    bool collectData(NaviPartId partId, NaviMessage &naviMessage);
    bool borderReceive(NaviBorderData &borderData);
    SubGraphBorder *getOutputBorder(const BorderId &borderId) const;
protected:
    mutable autil::ReadWriteLock _streamLock;
    multi_call::GigStreamBasePtr _stream;
    StreamState _streamState;
    StreamMetricPtr _streamMetric;
};

NAVI_TYPEDEF_PTR(GraphDomainRemoteBase);

}
