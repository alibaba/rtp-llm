#include "navi/engine/PartInfoDefOverride.h"
#include "navi/engine/GraphDomainClient.h"

#include "multi_call/interface/QuerySession.h"
#include "navi/distribute/NaviClientStream.h"
#include "navi/engine/Graph.h"
#include "navi/engine/ParamTopoTreeNode.h"
#include "navi/resource/QuerySessionR.h"
#include "autil/CompressionUtil.h"

using namespace autil;

namespace navi {

class NaviClientStreamDestructItem : public NaviThreadPoolItemBase
{
public:
    NaviClientStreamDestructItem(NaviClientStream *stream)
        : _stream(stream)
    {
    }
public:
    void process() override {
        NaviClientStream *stream = nullptr;
        std::swap(stream, _stream);
        if (stream) {
            delete stream;
        }
    }
    void destroy() override {
        process();
        delete this;
    }
private:
    NaviClientStream *_stream = nullptr;
};

class NotifyFinishItem : public NaviThreadPoolItemBase
{
public:
    NotifyFinishItem(GraphDomainClient *domain, ErrorCode ec)
        : _domain(domain)
        , _ec(ec)
    {
    }
public:
    void process() override {
        GraphDomainClient *domain = nullptr;
        std::swap(domain, _domain);
        if (domain) {
            domain->notifyFinish(_ec, false);
            auto *worker = domain->release(GDHR_NOTIFY_FINISH_ASYNC);
            if (worker) {
                worker->decItemCount();
            }
        }
    }
    void destroy() override {
        process();
        delete this;
    }
private:
    GraphDomainClient *_domain = nullptr;
    ErrorCode _ec;
};

class DomainRunItem : public NaviThreadPoolItemBase
{
public:
    DomainRunItem(GraphDomainClient *domain,
                  NaviWorkerBase *worker,
                  CollectFlags collectFlags,
                  const std::shared_ptr<TopoNodeParam> &topoParam,
                  GraphId graphId,
                  size_t &shardId)
        : _worker(worker)
        , _collectFlags(collectFlags)
        , _domain(domain)
        , _topoParam(topoParam)
    {
        if (collectFlags.needCollect()) {

            _metric = std::make_shared<KernelMetric>(
                graphId, "navi.domain_client@" + domain->getBizName() + "@" + std::to_string(shardId++), "");
        }
    }

public:
    const KernelMetricPtr &getMetric() const {
        return _metric;
    }
    bool addPartId(NaviPartId partId) {
        if (_partIds.size() >= 4) {
            return false;
        }
        _partIds.push_back(partId);
        return true;
    }
public:
    void process() override {
        GraphDomainClient *domain = nullptr;
        std::swap(domain, _domain);
        if (domain) {
            PerfEnableScope perfEnableScope(_worker->getNaviPerf(), _collectFlags);
            NaviLoggerScope loggerScope(_worker->getLogger());
            ComputeScopePtr scope;
            if (_metric) {
                scope = std::make_shared<ComputeScope>(_metric.get(),
                                                       nullptr,
                                                       0,
                                                       perfEnableScope.getPerfThread(),
                                                       _worker->getMemLeakCollector().get(),
                                                       _schedInfo,
                                                       _collectFlags,
                                                       _topoParam ? &_topoParam->getMallocTraceParam() : nullptr);
                scope->setType(GET_DOMAIN_CLIENT_RUN);
            }
            domain->doRun(_partIds);
            auto *worker = domain->release(GDHR_NOTIFY_FINISH_ASYNC);
            if (worker) {
                worker->decItemCount();
            }
        }
    }
    void destroy() override {
        process();
        delete this;
    }
private:
    NaviWorkerBase *_worker = nullptr;
    CollectFlags _collectFlags;
    GraphDomainClient *_domain = nullptr;
    KernelMetricPtr _metric;
    std::shared_ptr<TopoNodeParam> _topoParam;
    std::vector<NaviPartId> _partIds;
};

GraphDomainClient::GraphDomainClient(Graph *graph,
                                     const std::string &bizName)
    : GraphDomainRemoteBase(graph, GDT_CLIENT)
    , _bizName(bizName)
{
    _logger.addPrefix("biz: %s", bizName.c_str());
}

GraphDomainClient::~GraphDomainClient() {
}

ErrorCode GraphDomainClient::doPreInit() {
    NaviLoggerScope scope(_logger);
    assert(_subGraphDef);
    if (!initTimeout(_subGraphDef->option())) {
        NAVI_LOG(
            ERROR,
            "domain timeout, rpc timeout [%ld] ms, sub graph timeout [%ld] ms",
            _rpcTimeoutMs, _subGraphTimeoutMs);
        return EC_TIMEOUT;
    }
    if (!initGigStream(_subGraphDef->location())) {
        NAVI_LOG(ERROR, "init gig stream failed, biz: %s", _bizName.c_str());
        return EC_UNKNOWN;
    }
    _rpcResult = std::make_shared<RpcGraphResult>();
    _rpcResult->init(_graph->getLocalBizName(), _graph->getLocalPartId(),
                     _bizName, _partInfo, _subGraphDef->graph_id());
    _graphResult->addSubResult(_rpcResult);
    return EC_NONE;
}

bool GraphDomainClient::initTimeout(const SubGraphOption &option) {
    auto needCollect = _param->runParams.getCollectFlags().needCollect();
    double decayFactor = needCollect ? 0.7 : RPC_TIMEOUT_DECAY_FACTOR;
    int64_t decayMs = needCollect ? 1000 : SUBGRAPH_TIMEOUT_DECAY_MS;
    auto remainTimeMs = _graph->getRemainTimeMs();
    _subGraphTimeoutMs = remainTimeMs * decayFactor;
    if (option.timeout_ms() > 0) {
        _subGraphTimeoutMs = std::min(_subGraphTimeoutMs, option.timeout_ms());
    }
    _rpcTimeoutMs = std::min(_subGraphTimeoutMs + decayMs, remainTimeMs);
    NAVI_LOG(SCHEDULE1,
             "remainTime: [%ld]ms, rpc timeout [%ld]ms, sub graph timeout [%ld]ms, decayFactor [%lf], decayMs [%ld]ms",
             remainTimeMs,
             _rpcTimeoutMs,
             _subGraphTimeoutMs,
             decayFactor,
             decayMs);
    return _subGraphTimeoutMs > 0 && _rpcTimeoutMs > 0;
}

bool GraphDomainClient::initGigStream(const LocationDef &location) {
    const auto &querySessionR = _param->querySessionR;
    if (!querySessionR) {
        NAVI_LOG(ERROR, "get query session failed for client biz [%s]", _bizName.c_str());
        return false;
    }
    const auto &querySession = querySessionR->getQuerySession();
    auto stream = createClientStream();
    const auto &partInfoDef = location.part_info();
    stream->reserveEnablePartId(partInfoDef.part_ids_size());
    for (const auto &partId : partInfoDef.part_ids()) {
        stream->enablePartId(partId);
    }
    stream->setDisableProbe(!location.gig_info().enable_probe());
    if (_param->runParams.disableGigDegrade()) {
        stream->setDisableDegrade(true);
    }
    if (NAVI_COLLECT_MALLOC()) {
        stream->setDisableDetection(true);
    }
    bool allowLack = (EHS_ERROR_AS_EOF == getErrorStrategy());
    stream->setAllowLack(allowLack);
    addTags(stream, location.gig_info());
    stream->setTimeout(_rpcTimeoutMs);
    stream->setFlowControlStrategy(querySessionR->getQueryStrategy(_bizName));
    stream->setUserRequestType(querySession->sessionRequestType());
    auto gigStream =
        std::dynamic_pointer_cast<multi_call::GigClientStream>(stream);
    if (!initDirectLocation(location, stream)) {
        return false;
    }
    if (!querySession->bind(gigStream)) {
        NAVI_LOG(WARN, "bind gig stream failed, maybe biz not exist or full degrade");
        return false;
    }
    NaviPartId partCount = INVALID_NAVI_PART_COUNT;
    auto streamPartCount = stream->getPartCount();
    if (streamPartCount > 0) {
        partCount = streamPartCount;
    } else {
        partCount = 1;
        for (const auto &id : partInfoDef.part_ids()) {
            partCount = std::max(partCount, id + 1);
        }
    }
    if (!_partInfo.init(partCount, partInfoDef)) {
        NAVI_LOG(ERROR, "init partInfo failed, partCount: %d", partCount);;
        return false;
    }
    _streamState.setPartCount(_partInfo.getFullPartCount());
    if (!_streamState.init()) {
        NAVI_LOG(ERROR, "init domain part state failed");
        return false;
    }
    const auto &topoParam  = _param->runParams.getTopoParam();
    stream->initMetric(_param->worker->getNaviPerf(),
                       _param->runParams.getCollectFlags(),
                       topoParam ? &topoParam->getMallocTraceParam() : nullptr,
                       _param->memLeakCollector);
    setStream(gigStream);
    NAVI_LOG(SCHEDULE1, "init gig stream success [%p]", stream.get());
    return true;
}

bool GraphDomainClient::initDirectLocation(
        const LocationDef &location,
        const std::shared_ptr<NaviClientStream> &stream)
{
    if (!location.has_direct_location()) {
        return true;
    }
    const auto &directLocation = location.direct_location();
    if (directLocation.host().empty() ||
        directLocation.grpc_stream_port() <= 0 ||
        directLocation.grpc_stream_port() > 65535) {
        NAVI_LOG(ERROR, "invalid direct location [%s]",
                 directLocation.DebugString().c_str());
        return false;
    }
    multi_call::Spec spec;
    spec.ip = directLocation.host();
    spec.ports[multi_call::MC_PROTOCOL_GRPC_STREAM] =
        directLocation.grpc_stream_port();
    stream->setSpec(spec);
    return true;
}

void GraphDomainClient::addTags(const NaviClientStreamPtr &stream, const GigInfoDef &gigInfo) const {
    const auto &tags = gigInfo.tags();
    for (const auto &tag : tags) {
        stream->addMatchTag(tag.first, (multi_call::TagMatchType)tag.second);
    }
}

bool GraphDomainClient::postInit() {
    auto partCount = _partInfo.getFullPartCount();
    auto location = _subGraphDef->mutable_location();
    auto overridePartInfo = PartInfoDefOverride::replacePartInfo(location);
    auto partInfo = overridePartInfo->setTlsValue();
    if (!partInfo) {
        NAVI_LOG(ERROR, "set tls value failed");
        return false;
    }
    auto inputPartCount = partInfo->part_count();
    if (INVALID_NAVI_PART_COUNT != inputPartCount && partCount != inputPartCount) {
        NAVI_LOG(ERROR, "partCount mismatch, expect [%d], actual[%d]", inputPartCount, partCount);
        return false;
    }
    if (partInfo->part_count() != partCount) {
        partInfo->set_part_count(partCount);
    }
    // for test
    overridePartInfo->initCopy(*partInfo);
    NAVI_LOG(SCHEDULE1, "graph: %d, location: %s", _subGraphDef->graph_id(), location->DebugString().c_str());
    if (!initPeerInfo()) {
        return false;
    }
    _subGraphStr = _subGraphDef->SerializeAsString();
    if (_subGraphStr.empty()) {
        NAVI_LOG(ERROR, "empty sub graph");
        return false;
    }
    {
        NAVI_LOG(SCHEDULE1, "before compress, graph size: %lu", _subGraphStr.size());
        std::string compressedResult;
        auto compressed =
            autil::CompressionUtil::compress(_subGraphStr, autil::CompressType::LZ4, compressedResult, nullptr);
        if (compressed) {
            _compressType = CT_LZ4;
            std::swap(_subGraphStr, compressedResult);
            NAVI_LOG(SCHEDULE1, "after compress, graph size: %lu", _subGraphStr.size());
        } else {
            NAVI_LOG(SCHEDULE1, "compress failed");
        }
    }
    return true;
}

bool GraphDomainClient::run() {
    if (finished()) {
        return true;
    }
    bool collectMalloc = NAVI_COLLECT_MALLOC();
    if (unlikely(collectMalloc)) {
        for (auto partId : _partInfo) {
            notifySend(partId);
        }
        return true;
    }
    if (!acquire(GDHR_NOTIFY_SEND)) {
        return true;
    }
    auto collectFlags = _param->runParams.getCollectFlags();
    auto globalStartTimeUs = _param->runParams.getGlobalStartTimeUs();
    auto graphId = _subGraphDef->graph_id();
    size_t shardId = 0;
    auto *item = new DomainRunItem(this, _param->worker, collectFlags, _topoParam, graphId, shardId);
    const auto &metric = item->getMetric();
    if (unlikely(nullptr != metric)) {
        autil::ScopedLock lock(_metricLock);
        _metrics.push_back(metric);
    }
    for (auto partId : _partInfo) {
        if (!item->addPartId(partId)) {
            acquire(GDHR_NOTIFY_FINISH_ASYNC);
            _param->worker->getDefaultThreadPool()->push(
                globalStartTimeUs, 0, item, collectMalloc);
            item = new DomainRunItem(this, _param->worker, collectFlags, _topoParam, graphId, shardId);
            const auto &metric = item->getMetric();
            if (unlikely(nullptr != metric)) {
                autil::ScopedLock lock(_metricLock);
                _metrics.push_back(metric);
            }
            item->addPartId(partId);
        }
    }
    _param->worker->getDefaultThreadPool()->push(
        globalStartTimeUs, 0, item, collectMalloc);
    return true;
}

void GraphDomainClient::doRun(const std::vector<NaviPartId> &partIds) {
    if (finished()) {
        return;
    }
    for (auto partId : partIds) {
        notifySend(partId);
    }
}

bool GraphDomainClient::doSend(NaviPartId gigPartId, NaviMessage &naviMessage) {
    if (!_streamState.inited(gigPartId)) {
        NAVI_LOG(SCHEDULE2, "domain client init gigPartId[%d]", gigPartId);
        auto ec = fillInitMessage(gigPartId, naviMessage);
        if (EC_NONE != ec) {
            NAVI_LOG(ERROR, "fill init message failed! gig partId: %d",
                     gigPartId);
            notifyFinish(ec, true);
            return false;
        }
        _streamState.setInited(gigPartId);
    } else if (0 == naviMessage.border_datas_size()) {
        return true;
    }
    auto stream = getStream();
    if (!stream) {
        NAVI_LOG(SCHEDULE1, "send message failed, null stream! gig partId: %d",
                 gigPartId);
        return true;
    }
    if (_streamState.sendEof(gigPartId)) {
        NAVI_LOG(ERROR, "message after eof, gig partId: %d, message: %s",
                 gigPartId, naviMessage.DebugString().c_str());
        notifyFinish(EC_STREAM_SEND, true);
        return false;
    }
    bool eof = sendEof(gigPartId);
    naviMessage.set_msg_id(CommonUtil::random64());
    NAVI_LOG(SCHEDULE2, "message data send [%08x] toPartId: %d, msg len: %d, eof: %d",
             naviMessage.msg_id(), gigPartId,
             naviMessage.ByteSize(), eof);
    NAVI_LOG(SCHEDULE3, "message data send [%08x] toPartId: %d, eof [%d] msg content [%s]",
             naviMessage.msg_id(), gigPartId,
             eof, naviMessage.DebugString().c_str());
    if (!stream->send(gigPartId, eof, &naviMessage)) {
        auto *clientStream = (NaviClientStream *)stream.get();
        NAVI_LOG(WARN, "stream send message failed, gigPartId [%d], rpc info [%s]",
                 gigPartId,
                 StringUtil::toString(clientStream->getStreamRpcInfo(gigPartId)).c_str());
        return false;
    }
    if (eof) {
        _streamState.setSendEof(gigPartId);
    }
    return true;
}

ErrorCode GraphDomainClient::fillInitMessage(NaviPartId partId,
                                             NaviMessage &naviMessage) const
{
    assert(partId < _partInfo.getFullPartCount());
    auto domainGraph = naviMessage.mutable_domain_graph();
    domainGraph->set_sub_graph(_subGraphStr);
    domainGraph->set_this_part_id(partId);
    domainGraph->set_compress_type(_compressType);
    if (_counterInfo) {
        domainGraph->mutable_counter_info()->CopyFrom(*_counterInfo);
    }
    auto graphId = _subGraphDef->graph_id();
    auto remainTimeMs = _subGraphTimeoutMs;
    NaviLoggerScope scope(_logger);
    auto paramTopoTreeNode = _param->runParams.getParamNode(graphId, partId, std::string());
    return RunGraphParams::toProto(_param->runParams, graphId,
                                   _param->creatorManager,
                                   partId,
                                   remainTimeMs,
                                   paramTopoTreeNode,
                                   *naviMessage.mutable_params());
}

void GraphDomainClient::tryCancel(const multi_call::GigStreamBasePtr &stream) {
    auto partCount = _streamState.getPartCount();
    for (NaviPartId partId = 0; partId < partCount; partId++) {
        if (!_partInfo.isUsed(partId)) {
            continue;
        }
        finishPart(partId, stream, nullptr);
    }
}

void GraphDomainClient::finishPart(NaviPartId gigPartId,
                                   const multi_call::GigStreamBasePtr &stream,
                                   const NaviErrorPtr &error)
{
    assert(_partInfo.isUsed(gigPartId));
    auto inited = _streamState.streamInited();
    NAVI_LOG(DEBUG, "finish part [%d], inited [%d]", gigPartId, inited);
    if (!inited) {
        if (stream) {
            stream->sendCancel(gigPartId, nullptr);
        }
        return;
    }
    if (stream) {
        _streamState.lock(gigPartId);
        stream->sendCancel(gigPartId, nullptr);
        _streamState.unlock(gigPartId);
    }
    addResult(gigPartId, nullptr, true, error);
    receiveCallBack(gigPartId, true);
}

bool GraphDomainClient::onSendFailure(NaviPartId gigPartId) {
    assert(_partInfo.isUsed(gigPartId));
    if (EHS_ERROR_AS_FATAL == getErrorStrategy()) {
        NAVI_LOG(ERROR, "send failure, partId [%d]", gigPartId);
    }
    auto sendError = std::make_shared<NaviError>();
    sendError->id = _logger.logger->getLoggerId();
    sendError->ec = EC_STREAM_SEND;
    sendError->errorEvent = _logger.logger->firstErrorEvent();
    sendError->result = _graphResult.get();
    finishPart(gigPartId, getStream(), sendError);
    // do not trigger run graph error
    return false;
}

const PartInfo &GraphDomainClient::getPartInfo() const {
    return _partInfo;
}

bool GraphDomainClient::receiveCancel(const multi_call::GigStreamMessage &message, multi_call::MultiCallErrorCode ec) {
    NAVI_LOG(DEBUG,
             "client receive cancel, handler: [%p], gig ec [%s], partId: %d",
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
    auto cancelError = std::make_shared<NaviError>();
    cancelError->id = _logger.logger->getLoggerId();
    cancelError->ec = EC_PART_CANCEL;
    cancelError->result = _graphResult.get();
    if (message.message) {
        NAVI_LOG(SCHEDULE2, "cancel msg len [%d]",
                 message.message->ByteSize());
        NAVI_LOG(SCHEDULE3, "cancel msg content [%s]",
                 message.message->DebugString().c_str());
        if (!clientReceive(message, cancelError)) {
            notifyFinish(EC_STREAM_RPC, true);
            return false;
        }
    } else {
        addResult(partId, nullptr, true, cancelError);
    }
    receiveCallBack(partId, false);
    return true;
}

bool GraphDomainClient::receive(const multi_call::GigStreamMessage &message) {
    auto partId = message.partId;
    if (message.eof) {
        _streamState.setReceiveEof(partId);
    }
    if (!clientReceive(message, nullptr)) {
        NAVI_LOG(ERROR, "receive msg failed");
        notifyFinish(EC_STREAM_RECEIVE, true);
        return false;
    }
    receiveCallBack(partId, false);
    return true;
}

bool GraphDomainClient::clientReceive(const multi_call::GigStreamMessage &message, const NaviErrorPtr &error) {
    auto naviMessage = dynamic_cast<NaviMessage *>(message.message);
    if (!naviMessage) {
        NAVI_LOG(ERROR, "receive error, null msg, partId[%d], this type: %s",
                 message.partId, typeid(*this).name());
        return false;
    }
    addResult(message.partId, naviMessage, message.eof, error);
    return doReceive(message);
}

void GraphDomainClient::addResult(NaviPartId partId, NaviMessage *naviMessage,
                                  bool withRpcInfo, const NaviErrorPtr &error)
{
    NaviErrorPtr resultError;
    if (withRpcInfo) {
        multi_call::GigStreamRpcInfo rpcInfo;
        if (collectRpcInfo(partId, rpcInfo)) {
            resultError = _rpcResult->addResult(partId, naviMessage, &rpcInfo, error);
        } else {
            if (naviMessage) {
                resultError = _rpcResult->addResult(partId, naviMessage, nullptr, error);
            }
        }
    } else if (naviMessage) {
        resultError = _rpcResult->addResult(partId, naviMessage, nullptr, error);
    }
    if (resultError) {
        if (EHS_ERROR_AS_FATAL == getErrorStrategy()) {
            NAVI_LOG(ERROR, "result has error [%s], error as fatal",
                     CommonUtil::getErrorString(resultError->ec));
            notifyFinish(resultError, true);
        } else {
            NAVI_LOG(SCHEDULE1, "result has error [%s], error as eof",
                     CommonUtil::getErrorString(resultError->ec));
        }
    }
}

bool GraphDomainClient::collectRpcInfo(NaviPartId partId, multi_call::GigStreamRpcInfo &info) {
    if (!_streamState.streamInited()) {
        return false;
    }
    if (!_streamState.setCollected(partId)) {
        return false;
    }
    NAVI_LOG(SCHEDULE2, "try collect stream rpc info part [%d]", partId);
    auto stream = getStream();
    if (!stream) {
        NAVI_LOG(SCHEDULE2, "collect rpc info failed, null stream, part [%d]", partId);
        return false;
    }
    auto *clientStream = (NaviClientStream *)stream.get();
    info = clientStream->getStreamRpcInfo(partId);
    NAVI_LOG(SCHEDULE2, "rpc info collect success, part [%d] biz [%s]", partId,
             _bizName.c_str());
    return true;
}

void GraphDomainClient::receiveCallBack(NaviPartId gigPartId, bool forceStop) {
    assert(_partInfo.isUsed(gigPartId));
    NAVI_LOG(SCHEDULE1,
             "receive callback, partId [%d], part receiveEof[%d], part cancel [%d], total eof [%d], forceStop [%d]",
             gigPartId,
             _streamState.receiveEof(gigPartId),
             _streamState.cancelled(gigPartId),
             receiveEof(),
             forceStop);
    if (!_streamState.receiveEof(gigPartId) && (forceStop || _streamState.cancelled(gigPartId))) {
        NAVI_LOG(DEBUG, "client receive cancel, finish all port, partId [%d]",
                 gigPartId);
        for (const auto &pair : _outputBorderMap) {
            pair.second->setEof(gigPartId);
        }
    }
    if (receiveEof()) {
        NAVI_LOG(SCHEDULE1, "receive all part eof, current partId [%d], partInfo [%s]",
                 gigPartId, _partInfo.ShortDebugString().c_str());
        notifyFinishAsync(EC_NONE);
    }
}

NaviPartId GraphDomainClient::getStreamPartId(NaviPartId partId) {
    return partId;
}

bool GraphDomainClient::sendEof(NaviPartId partId) const {
    bool eof = true;
    for (const auto &pair : _remoteInputBorderMap) {
        eof = eof && pair.second->eof(partId);
    }
    return eof;
}

bool GraphDomainClient::receiveEof() const {
    bool eof = true;
    for (const auto &pair : _remoteOutputBorderMap) {
        eof = eof && pair.second->eof();
    }
    return eof;
}

void GraphDomainClient::notifyFinishAsync(ErrorCode ec) {
    if (finished()) {
        return;
    }
    if (unlikely(NAVI_COLLECT_MALLOC())) {
        notifyFinish(ec, false);
        return;
    }
    if (!acquire(GDHR_NOTIFY_FINISH_ASYNC)) {
        // TODO: assert
        NAVI_LOG(ERROR, "fatal: acquire failed");
        return;
    }
    auto *item = new NotifyFinishItem(this, ec);
    _param->worker->getDefaultThreadPool()->push(
        _param->runParams.getGlobalStartTimeUs(),
        0,
        item,
        _param->runParams.getCollectFlags().collectMalloc());
}

NaviClientStreamPtr GraphDomainClient::createClientStream() {
    auto globalStartTimeUs = _param->runParams.getGlobalStartTimeUs();
    bool collectMalloc = _param->runParams.getCollectFlags().collectMalloc();
    multi_call::StreamRpcParam streamRpcParam;
    initStreamRpcParam(globalStartTimeUs, streamRpcParam);
    const auto &defaultThreadPool = _param->worker->getDefaultThreadPool();
    NaviClientStreamPtr stream;
    if (unlikely(collectMalloc)) {
        stream = std::make_shared<NaviClientStream>(streamRpcParam, _bizName, _logger.logger);
        _clientStream = stream;
    } else {
        stream = NaviClientStreamPtr(
            new NaviClientStream(streamRpcParam, _bizName, _logger.logger),
            [defaultThreadPool, globalStartTimeUs, collectMalloc](
                NaviClientStream *stream) {
                auto *item = new NaviClientStreamDestructItem(stream);
                defaultThreadPool->push(globalStartTimeUs + DESTRUCT_DELAY_US,
                                        0,
                                        item,
                                        collectMalloc);
            });
    }
    stream->setThreadPool(defaultThreadPool);
    return stream;
}

void GraphDomainClient::initStreamRpcParam(int64_t globalStartTimeUs, multi_call::StreamRpcParam &streamRpcParam) {
    streamRpcParam.asyncMode = _param->runParams.asyncIo();
    streamRpcParam.globalStartTimeUs = globalStartTimeUs;
    auto flags = _param->runParams.getCollectFlags(nullptr).getFlags();
    auto paramTopoTree = _param->runParams.getParamTopoTree();
    if (unlikely(0 != flags || nullptr != paramTopoTree)) {
        auto graphId = _subGraphDef->graph_id();
        streamRpcParam.collectFlagsFunc = [graphId, flags, paramTopoTree] (multi_call::PartIdTy partId) {
            auto paramNode = paramTopoTree ? paramTopoTree->getNode(graphId, partId, std::string()) : nullptr;
            auto param = paramNode ? paramNode->getParam() : nullptr;
            auto thisPartFlags = flags;
            if (param) {
                thisPartFlags = param->_collectFlags.getFlags() | thisPartFlags;
            }
            return thisPartFlags;
        };
    }
    if (paramTopoTree) {
        _topoParam = paramTopoTree->getParam();
        if (_topoParam) {
            const auto &mallocTraceParam = _topoParam->getMallocTraceParam();
            if (mallocTraceParam.isValid()) {
                streamRpcParam.mallocTraceStr = mallocTraceParam.toString();
            }
        }
    }
}

void GraphDomainClient::collectMetric(GraphMetric *graphMetric) {
    {
        autil::ScopedLock lock(_metricLock);
        for (const auto &metric : _metrics) {
            graphMetric->addKernelMetricPtr(metric);
        }
    }
    auto streamMetric = stealStreamMetric();
    if (streamMetric) {
        streamMetric->collectMetric(_bizName, _graphId, graphMetric);
    }
}

}
