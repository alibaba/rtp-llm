#include "navi/engine/QueryHistory.h"
#include "navi/engine/NaviResult.h"
#include "navi/rpc_server/NaviArpcRequestData.h"
#include "navi/rpc_server/ArpcRegistryParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/HTTPRPCServerClosure.h"

namespace navi {

QueryHistory::QueryHistory() {
}

QueryHistory::~QueryHistory() {
    QuerySample *sample = nullptr;
    while (_queryQueue.Pop(&sample)) {
        delete sample;
    }
}

void QueryHistory::updateConfig(const QueryHistoryConfig &config) {
    _sampleIntervalUs = config.sampleIntervalUs;
    _normalHistory.updateConfig(config.normal);
    _errorHistory.updateConfig(config.error);
}

void QueryHistory::toProto(bool shortQueryHistory, QueryHistoryDef &historyDef, SymbolTableDef &symbolTableDef) const {
    fillConfig(*historyDef.mutable_config());
    _normalHistory.toProto(shortQueryHistory, *historyDef.mutable_normal(), symbolTableDef);
    _errorHistory.toProto(shortQueryHistory, *historyDef.mutable_error(), symbolTableDef);
}

void QueryHistory::fromProto(const QueryHistoryDef &historyDef, const SymbolTableDef &symbolTableDef) {
    parseConfig(historyDef.config());
    _normalHistory.fromProto(historyDef.normal(), symbolTableDef);
    _errorHistory.fromProto(historyDef.error(), symbolTableDef);
}

void QueryHistory::parseConfig(const QueryHistoryConfigDef &configDef) {
    _sampleIntervalUs = configDef.sample_interval_us();
    _normalHistory.parseConfig(configDef.normal());
    _errorHistory.parseConfig(configDef.error());
}

void QueryHistory::fillConfig(QueryHistoryConfigDef &configDef) const {
    configDef.set_sample_interval_us(_sampleIntervalUs);
    _normalHistory.fillConfig(*configDef.mutable_normal());
    _errorHistory.fillConfig(*configDef.mutable_error());
}

bool QueryHistory::checkTimeInterval(int64_t beginTimeUs, bool hasError, int64_t &prevLogTimeUs) const {
    prevLogTimeUs = (!hasError) ? _prevLogTimeUs : _prevErrorLogTimeUs;
    return beginTimeUs - prevLogTimeUs >= _sampleIntervalUs;
}

bool QueryHistory::updateLogTime(int64_t beginTimeUs, bool hasError, int64_t prevLogTimeUs) {
    volatile int64_t *addr = (!hasError) ? &_prevLogTimeUs : &_prevErrorLogTimeUs;
    return cmpxchg(addr, prevLogTimeUs, beginTimeUs);
}

void QueryHistory::addQuery(const std::shared_ptr<NaviArpcRequestData> &requestData,
                            const RegistryArpcParam &arpcParam,
                            http_arpc::HTTPRPCServerClosure *httpClosure,
                            const std::shared_ptr<NaviResult> &naviResult,
                            const std::string &queryKey,
                            google::protobuf::Message *response,
                            bool needDebug)
{
    if (!requestData || !response) {
        return;
    }
    auto beginTimeUs = autil::TimeUtility::currentTime();
    int64_t prevLogTimeUs = 0;
    bool hasError = (!naviResult || naviResult->error);
    if (!checkTimeInterval(beginTimeUs, hasError, prevLogTimeUs)) {
        return;
    }
    if (_queryQueue.Size() > 10) {
        return;
    }
    auto request = requestData->getRequest();
    if (!request) {
        return;
    }
    http_arpc::ProtoJsonizer *protoJsonizer = nullptr;
    if (httpClosure) {
        protoJsonizer = httpClosure->getProtoJsonizer().get();
    } else {
        protoJsonizer = arpcParam.protoJsonizer.get();
    }
    if (!protoJsonizer) {
        return;
    }
    if (!updateLogTime(beginTimeUs, hasError, prevLogTimeUs)) {
        return;
    }
    auto querySample = new QuerySample();
    if (httpClosure) {
        querySample->httpPath = httpClosure->getServiceAndMethod();
    } else {
        if (!arpcParam.httpAliasVec.empty()) {
            querySample->httpPath = arpcParam.httpAliasVec[0];
        }
    }
    querySample->arpcMethod = arpcParam.method;
    querySample->queryKey = queryKey;
    querySample->query = protoJsonizer->toJson(*request);
    querySample->clientIp = requestData->getClientIp();
    querySample->naviWebQuery = needDebug;
    querySample->isHttp = (httpClosure != nullptr);
    querySample->responseSize = response ? response->ByteSizeLong() : -1;
    if (naviResult) {
        querySample->session = naviResult->id;
        querySample->error = naviResult->error;
        querySample->beginTime = naviResult->beginTime;
        querySample->endTime = naviResult->endTime;
        querySample->queueUs = naviResult->queueUs;
        querySample->computeUs = naviResult->computeUs;
    }
    _queryQueue.Push(querySample);
}

void QueryHistory::workLoop() {
    QuerySample *sample = nullptr;
    while (_queryQueue.Pop(&sample)) {
        if (sample->error) {
            _errorHistory.collect(sample);
        } else {
            _normalHistory.collect(sample);
        }
    }
}

}

