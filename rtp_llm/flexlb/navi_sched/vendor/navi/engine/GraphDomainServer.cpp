#include "navi/engine/GraphDomainServer.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/FileRecorder.h"
#include "navi/distribute/NaviStreamBase.h"
#include "navi/distribute/NaviServerStream.h"
#include "navi/engine/Graph.h"
#include "navi/engine/RemoteInputPort.h"
#include "navi/engine/RemoteOutputPort.h"
#include "navi/proto/GraphVis.pb.h"

namespace navi {

class DelaySendDestructClosure : public NaviUserResultClosure {
public:
    DelaySendDestructClosure(bool cancel_)
        : cancel(cancel_)
    {
        waitDestruct = true;
    }
    ~DelaySendDestructClosure() {
    }
public:
    void run(NaviUserResultPtr result, const ComputeScopePtr &computeScope) override {
        if (result) {
            auto naviResult = result->getNaviResultWithoutCheck();
            result.reset();
            if (naviResult) {
                naviResult->toProto(false, naviMessage, computeScope);
            }
        }
        if (!cancel) {
            stream->send(0, true, &naviMessage);
        } else {
            stream->sendCancel(0, &naviMessage);
        }
        delete this;
    }

public:
    bool cancel = false;
    NaviMessage naviMessage;
    multi_call::GigStreamBasePtr stream;
};

GraphDomainServer::GraphDomainServer(Graph *graph)
    : GraphDomainRemoteBase(graph, GDT_SERVER)
{
}

GraphDomainServer::~GraphDomainServer() {
}

ErrorCode GraphDomainServer::doPreInit() {
    _logger.addPrefix("onBiz: %s part: %d",
                      _graph->getLocalBizName().c_str(),
                      _graph->getLocalPartId());
    if (!_streamState.init()) {
        NAVI_LOG(ERROR, "init domain part state failed");
        return EC_ABORT;
    }
    _streamState.setInited(0);
    return EC_NONE;
}

bool GraphDomainServer::run() {
    return true;
}

bool GraphDomainServer::doSend(NaviPartId gigPartId, NaviMessage &naviMessage) {
    if (0 == naviMessage.border_datas_size()) {
        return true;
    }
    naviMessage.set_msg_id(CommonUtil::random64());
    bool eof = sendEof();
    NAVI_LOG(SCHEDULE1, "start do send, eof [%d]", eof);
    auto localPartId = _graph->getLocalPartId();
    bool delaySend = _param->runParams.getCollectFlags().collectMalloc();
    if (eof) {
        NAVI_LOG(SCHEDULE1, "send eof from localPartId [%d]", localPartId);
        fillFiniMessage(delaySend, naviMessage);
    } else if (_streamState.sendEof(gigPartId)) {
        NAVI_LOG(SCHEDULE1, "part [%d] already finished, msg [%s]", gigPartId,
                 naviMessage.DebugString().c_str());
        return false;
    }
    if (!eof) {
        fillTrace(naviMessage);
    }
    NAVI_LOG(SCHEDULE2, "message data send [%08x] partId: %d, msg len: %d, eof: %d",
             naviMessage.msg_id(), localPartId,
             naviMessage.ByteSize(), eof);
    NAVI_LOG(SCHEDULE3, "message data send [%08x] partId: %d, eof [%d] msg content [%s]",
             naviMessage.msg_id(), localPartId,
             eof, naviMessage.DebugString().c_str());
    {
        auto stream = getStream();
        if (!stream) {
            NAVI_LOG(SCHEDULE1, "part [%d] null stream, msg [%s]", gigPartId,
                     naviMessage.DebugString().c_str());
            return false;
        }
        if (!delaySend || !eof) {
            if (!stream->send(gigPartId, eof, &naviMessage)) {
                NAVI_LOG(SCHEDULE1,
                         "part [%d] gig stream send data failed, msg [%s]",
                         gigPartId,
                         naviMessage.DebugString().c_str());
                return false;
            }
        } else {

            auto delayClosure = new DelaySendDestructClosure(false);
            delayClosure->naviMessage.ParseFromString(naviMessage.SerializeAsString());
            delayClosure->stream = stream;
            _param->worker->setDestructClosure(delayClosure);
        }
    }
    if (eof) {
        _streamState.setSendEof(gigPartId);
    }
    if (eof) {
        notifyFinish(EC_NONE, true);
    }
    return true;
}

bool GraphDomainServer::receiveCancel(const multi_call::GigStreamMessage &message, multi_call::MultiCallErrorCode ec) {
    NAVI_LOG(DEBUG,
             "server receive cancel, handler: [%p], ec [%s], partId: %d",
             (void *)message.handlerId,
             multi_call::translateErrorCode(ec),
             message.partId);
    auto partId = message.partId;
    bool cancelled;
    {
        StreamStatePartScopedLock lock(_streamState, partId);
        cancelled = _streamState.cancelled(partId);
        _streamState.receiveCancel(partId);
    }
    if (cancelled) {
        NAVI_LOG(SCHEDULE2, "ignore cancel after stream state cancelled, partId [%d]", partId);
        return true;
    }
    if (message.message) {
        NAVI_LOG(SCHEDULE2, "cancel msg len [%d]",
                 message.message->ByteSize());
        NAVI_LOG(SCHEDULE3, "cancel msg content [%s]",
                 message.message->DebugString().c_str());
        if (!doReceive(message)) {
            notifyFinish(EC_STREAM_RPC, true);
            return false;
        }
    }
    notifyFinish(EC_ABORT, true);
    return true;
}

bool GraphDomainServer::receive(const multi_call::GigStreamMessage &message) {
    if (message.eof) {
        _streamState.setReceiveEof(message.partId);
    }
    if (!doReceive(message)) {
        NAVI_LOG(ERROR, "receive msg failed");
        notifyFinish(EC_STREAM_RECEIVE, true);
        return false;
    }
    if (_streamState.cancelled(message.partId)) {
        notifyFinish(EC_ABORT, true);
    }
    return true;
}

void GraphDomainServer::fillTrace(NaviMessage &naviMessage) {
    NAVI_LOG(SCHEDULE1, "start fill trace message [%08x]", naviMessage.msg_id());
    _graphResult->toProto(true, naviMessage, nullptr);
    NAVI_LOG(SCHEDULE1, "end fill trace message [%08x]", naviMessage.msg_id());
}

void GraphDomainServer::fillFiniMessage(bool delaySend, NaviMessage &naviMessage) {
    NAVI_LOG(SCHEDULE1, "start fill finish message [%08x]", naviMessage.msg_id());
    _graph->collectMetric();
    _graphResult->setFinish();

    if (!delaySend) {
        _graphResult->toProto(false, naviMessage, nullptr);
        NAVI_LOG(
            SCHEDULE1, "end fill finish message [%08x]", naviMessage.msg_id());
    }
}

NaviPartId GraphDomainServer::getStreamPartId(NaviPartId partId) {
    return 0;
}

void GraphDomainServer::tryCancel(const multi_call::GigStreamBasePtr &stream) {
    NaviPartId partId = 0;
    auto inited = _streamState.streamInited();
    if (inited) {
        if (_streamState.cancelled(partId)) {
            NAVI_LOG(SCHEDULE3, "cancel ignore, already cancelled");
            return;
        }
        if (_streamState.sendEof(partId)) {
            NAVI_LOG(SCHEDULE3, "cancel ignore, already send eof");
            return;
        }
        _streamState.lock(partId);
    }
    NaviMessage naviMessage;
    naviMessage.set_msg_id(CommonUtil::random64());
    if (unlikely(_param && _param->runParams.getCollectFlags().collectMalloc())) {
        fillFiniMessage(true, naviMessage);

        auto delayClosure = new DelaySendDestructClosure(true);
        delayClosure->naviMessage.ParseFromString(naviMessage.SerializeAsString());
        delayClosure->stream = stream;
        _param->worker->setDestructClosure(delayClosure);
    } else {
        fillFiniMessage(false, naviMessage);
        stream->sendCancel(partId, &naviMessage);
    }
    if (inited) {
        _streamState.unlock(partId);
    }
    NAVI_LOG(SCHEDULE3, "send cancel data [%s]", naviMessage.DebugString().c_str());
}

bool GraphDomainServer::onSendFailure(NaviPartId gigPartId) {
    // trigger run graph error
    return true;
}

bool GraphDomainServer::sendEof() const {
    bool eof = true;
    for (const auto &pair : _remoteInputBorderMap) {
        eof = eof && pair.second->eof();
    }
    return eof;
}

void GraphDomainServer::collectMetric(GraphMetric *graphMetric) {
    auto streamMetric = stealStreamMetric();
    if (streamMetric) {
        streamMetric->collectMetric(_graph->getLocalBizName(), _graphId, graphMetric);
    }
}

}
