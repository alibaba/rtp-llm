#pragma once

#include "navi/engine/GraphDomainRemoteBase.h"

namespace navi {

class GraphDomainServer : public GraphDomainRemoteBase
{
public:
    GraphDomainServer(Graph *graph);
    ~GraphDomainServer();
private:
    GraphDomainServer(const GraphDomainServer &);
    GraphDomainServer &operator=(const GraphDomainServer &);
public:
    ErrorCode doPreInit() override;
    bool run() override;
    bool receive(const multi_call::GigStreamMessage &message) override;
    bool receiveCancel(const multi_call::GigStreamMessage &message, multi_call::MultiCallErrorCode ec) override;
    bool doSend(NaviPartId gigPartId, NaviMessage &naviMessage) override;
    NaviPartId getStreamPartId(NaviPartId partId) override;
    void tryCancel(const multi_call::GigStreamBasePtr &stream) override;
    bool onSendFailure(NaviPartId gigPartId) override;
    void collectMetric(GraphMetric *graphMetric) override;
    void fillTrace(NaviMessage &naviMessage);
    void fillFiniMessage(bool delaySend, NaviMessage &naviMessage);
private:
    bool sendEof() const;
};

}
