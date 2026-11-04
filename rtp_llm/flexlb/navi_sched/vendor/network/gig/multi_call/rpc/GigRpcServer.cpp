#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc/GigRpcServer.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCServer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/KMonitorANetServerMetricReporter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/GigAgent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/arpc/ServiceWrapper.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/TopoInfoBuilder.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/server/GrpcServerWorker.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HeartbeatServerManager.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HeartbeatStreamCreator.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/SharedPtrUtil.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/HTTPRPCServer.h"
#include "autil/NetUtil.h"
#include "autil/legacy/base64.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, GigRpcServer);

GigRpcServer::GigRpcServer()
    : _autoDelete(true)
    , _agent(new GigAgent())
    , _heartbeatManager(new HeartbeatServerManager())
    , _heartbeatThreadNum(5)
    , _tcpPort(INVALID_PORT)
{
}

GigRpcServer::GigRpcServer(arpc::ANetRPCServer *arpcServer)
    : _arpcServer(arpcServer, [](arpc::ANetRPCServer *) {})
    , _autoDelete(false)
    , _agent(new GigAgent())
    , _heartbeatManager(new HeartbeatServerManager())
    , _heartbeatThreadNum(5)
    , _tcpPort(INVALID_PORT)
{
}

GigRpcServer::~GigRpcServer() {
    release();
    _agent.reset();
    _wrappers.clear();
    AUTIL_LOG(INFO, "gig rpc server stopped");
}

void GigRpcServer::release() {
    stopHeartbeat();
    stopGrpcServer();
    stopHttpArpcServer();
    stopArpcServer();
    AUTIL_LOG(INFO, "gig rpc server release");
}

bool GigRpcServer::init(const GigRpcServerConfig &serverConfig) {
    if (serverConfig.heartbeatThreadNum == 0) {
        AUTIL_LOG(ERROR, "heartbeat thread count must be positive");
        return false;
    }
    _heartbeatThreadNum = serverConfig.heartbeatThreadNum;
    if (!_agent->init(serverConfig.logPrefix, serverConfig.enableAgentStat,
                      serverConfig.snapshotLogDir)) {
        AUTIL_LOG(ERROR, "init gig agent failed!");
        return false;
    }
    _agent->stop();
    return true;
}

bool GigRpcServer::startArpcServer(ArpcServerDescription &desc) {
    if (_arpcServer) {
        AUTIL_LOG(ERROR, "arpc server already started");
        return false;
    }
    _arpcDesc = desc;
    if (desc.ioThreadNum <= 1) {
        _arpcServer.reset(new arpc::ANetRPCServer(NULL, desc.threadNum, desc.queueSize));
        if (!_arpcServer->StartPrivateTransport()) {
            AUTIL_LOG(ERROR, "arpc server start private transport failed");
            return false;
        }
    } else {
        _arpcServerTransport.reset(new anet::Transport(desc.ioThreadNum, anet::SHARE_THREAD));
        _arpcServer.reset(
            new arpc::ANetRPCServer(_arpcServerTransport.get(), desc.threadNum, desc.queueSize));
        if (!_arpcServerTransport->start()) {
            AUTIL_LOG(ERROR, "arpc server start public transport failed");
            return false;
        }
        _arpcServerTransport->setName(desc.name);
        AUTIL_LOG(INFO, "arpc server start public transport success, name [%s] thread [%d]",
                  desc.name.c_str(), desc.ioThreadNum);
    }

    if (desc.enableANetMetric || desc.enableArpcMetric) {
        arpc::KMonitorANetMetricReporterConfig metricConfig;
        metricConfig.anetConfig.enableANetMetric = desc.enableANetMetric;
        metricConfig.arpcConfig.enableArpcMetric = desc.enableArpcMetric;
        auto metricReporter =
            std::make_shared<arpc::KMonitorANetServerMetricReporter>(metricConfig);
        if (metricReporter->init(_arpcServer->GetTransport())) {
            AUTIL_LOG(INFO, "arpc server set metric reporter");
            _arpcServer->SetMetricReporter(metricReporter);
        } else {
            AUTIL_LOG(WARN, "anet server metric reporter init failed");
        }
    }

    std::string spec("tcp:0.0.0.0:" + desc.port);
    if (!_arpcServer->Listen(spec)) {
        AUTIL_LOG(ERROR, "arpc server listen on %s failed", spec.c_str());
        return false;
    }
    AUTIL_LOG(INFO, "arpc server listen on %s success", spec.c_str());
    if (_heartbeatManager) {
        _heartbeatManager->updateSpec(getRpcSpec());
    }
    return true;
}

void GigRpcServer::stopArpcServer() {
    if (_arpcServer && _autoDelete) {
        AUTIL_LOG(INFO, "auto delete arpc server");
        _arpcServer->Close();
        _arpcServer->StopPrivateTransport();
    }
    if (_arpcServerTransport) {
        _arpcServerTransport->stop();
        _arpcServerTransport->wait();
    }
    _arpcServer.reset();
    _arpcServerTransport.reset();
    AUTIL_LOG(INFO, "arpc server stoped");
}

bool GigRpcServer::startHttpArpcServer(HttpArpcServerDescription &desc) {
    if (_httpArpcServer) {
        AUTIL_LOG(ERROR, "http arpc server already started");
        return false;
    }
    if (!_arpcServer) {
        AUTIL_LOG(ERROR, "http arpc server start failed, arpc server not started");
        return false;
    }
    _httpArpcServerTransport.reset(new anet::Transport(desc.ioThreadNum, anet::SHARE_THREAD));
    _httpArpcServer.reset(new http_arpc::HTTPRPCServer(
        _arpcServer.get(), _httpArpcServerTransport.get(), desc.threadNum, desc.queueSize,
        desc.decodeUri, desc.haCompatible, desc.passThrough));
    _httpArpcServer->setDefaultGigHeader(getDefaultHeader());
    if (!_httpArpcServerTransport->start()) {
        AUTIL_LOG(ERROR, "http arpc transport start failed");
        return false;
    }
    _httpArpcServerTransport->setName(desc.name);

    if (!_httpArpcServer->StartThreads()) {
        AUTIL_LOG(ERROR, "http arpc server start threads failed");
        return false;
    }
    std::string spec("tcp:0.0.0.0:" + desc.port);
    if (!_httpArpcServer->Listen(spec)) {
        AUTIL_LOG(ERROR, "http arpc server listen on %s failed", spec.c_str());
        return false;
    }
    AUTIL_LOG(INFO, "http arpc server listen on %s success", spec.c_str());
    if (_heartbeatManager) {
        _heartbeatManager->updateSpec(getRpcSpec());
    }
    return true;
}

std::string GigRpcServer::getDefaultHeader() {
    GigResponseInfo responseInfo;
    responseInfo.set_gig_response_checksum(GIG_RESPONSE_CHECKSUM);
    responseInfo.set_latency_ms(0);
    responseInfo.set_ec(MULTI_CALL_ERROR_NO_RESPONSE);
    responseInfo.set_target_weight(MIN_WEIGHT);
    auto encoded = autil::legacy::Base64EncodeFast(responseInfo.SerializeAsString());
    AUTIL_LOG(INFO, "gig default header: [%s], encoded: [%s]", responseInfo.DebugString().c_str(),
              encoded.c_str());
    return encoded;
}

void GigRpcServer::stopHttpArpcServer() {
    if (_httpArpcServer) {
        _httpArpcServer->Close();
        _httpArpcServer->StopPrivateTransport();
    }
    if (_httpArpcServerTransport) {
        _httpArpcServerTransport->stop();
        _httpArpcServerTransport->wait();
    }
    _httpArpcServer.reset();
    _httpArpcServerTransport.reset();
    AUTIL_LOG(INFO, "http arpc server stoped");
}

bool GigRpcServer::startGrpcServer(const GrpcServerDescription &desc) {
    auto worker = createGrpcServer(desc);
    if (!worker) {
        AUTIL_LOG(ERROR, "init grpc server failed, port [%s]", desc.port.c_str());
        return false;
    }
    _grpcWorker = worker;
    if (!_heartbeatManager) {
        return true;
    }
    if (!_heartbeatManager->init(_agent, getRpcSpec(), _heartbeatThreadNum)) {
        AUTIL_LOG(ERROR, "init heartbeat manager failed");
        return false;
    }
    _heartbeatManager->updateSpec(getRpcSpec());
    if (!registerHeartbeat()) {
        AUTIL_LOG(ERROR, "register grpc heartbeat failed, port [%s]", desc.port.c_str());
        return false;
    }
    return true;
}

GrpcServerWorkerPtr GigRpcServer::createGrpcServer(const GrpcServerDescription &desc) {
    GrpcServerWorkerPtr grpcWorker(new GrpcServerWorker(*this));
    if (!grpcWorker->init(desc)) {
        return GrpcServerWorkerPtr();
    } else {
        return grpcWorker;
    }
}

void GigRpcServer::stopGrpcServer() {
    _grpcWorker.reset();
    AUTIL_LOG(INFO, "grpc server stoped");
}





void GigRpcServer::setTcpPort(int32_t tcpPort) {
    _tcpPort = tcpPort;
    if (_heartbeatManager) {
        _heartbeatManager->updateSpec(getRpcSpec());
    }
}



bool GigRpcServer::registerHeartbeat() {
    HeartbeatStreamCreatorPtr creator = std::make_shared<HeartbeatStreamCreator>();
    creator->setManager(_heartbeatManager);
    auto creatorBase = std::dynamic_pointer_cast<multi_call::GigServerStreamCreator>(creator);
    if (!registerGrpcStreamService(creatorBase)) {
        AUTIL_LOG(ERROR, "register server heartbeat failed");
        return false;
    }
    _heartbeatCreator = creatorBase;
    return true;
}

void GigRpcServer::stopHeartbeat() {
    if (_heartbeatCreator) {
        unRegisterGrpcStreamService(_heartbeatCreator);
        int64_t waitTime = 5 * 1000 * 1000;
        if (!SharedPtrUtil::waitUseCount(_heartbeatCreator, 1, waitTime)) {
            AUTIL_LOG(WARN, "wait heartbeat creator delete timeout");
        }
        _heartbeatCreator.reset();
    }
    if (_heartbeatManager) {
        _heartbeatManager->stop();
        AUTIL_LOG(INFO, "heartbeat stopped, manager [%p], this [%p]", _heartbeatManager.get(),
                  this);
        _heartbeatManager.reset();
    }
}

bool GigRpcServer::publish(const ServerBizTopoInfo &info, SignatureTy &signature) {
    if (_heartbeatManager) {
        return _heartbeatManager->publish(info, signature);
    } else {
        AUTIL_LOG(ERROR, "publish topo failed, server stopped");
        return false;
    }
}

bool GigRpcServer::publish(const std::vector<ServerBizTopoInfo> &infoVec,
                           std::vector<SignatureTy> &signatureVec) {
    if (_heartbeatManager) {
        return _heartbeatManager->publish(infoVec, signatureVec);
    } else {
        AUTIL_LOG(ERROR, "publish topo failed, server stopped");
        return false;
    }
}

bool GigRpcServer::publishGroup(PublishGroupTy group, const std::vector<ServerBizTopoInfo> &infoVec,
                                std::vector<SignatureTy> &signatureVec) {
    if (INVALID_PUBLISH_GROUP == group) {
        AUTIL_LOG(ERROR, "invalid publish group id [%lu]", group);
        return false;
    }
    if (_heartbeatManager) {
        return _heartbeatManager->publishGroup(group, infoVec, signatureVec);
    } else {
        AUTIL_LOG(ERROR, "publish topo failed, group id [%lu], server stopped", group);
        return false;
    }
}

bool GigRpcServer::publishGroup(PublishGroupTy group, const std::vector<BizTopoInfo> &topoVec,
                                std::vector<SignatureTy> &signatureVec) {
    std::vector<multi_call::ServerBizTopoInfo> infoVec;
    for (const auto &topo : topoVec) {
        multi_call::ServerBizTopoInfo info;
        info.bizName = topo.bizName;
        info.partCount = topo.partCnt;
        info.partId = topo.partId;
        info.version = topo.version;
        info.protocolVersion = topo.protocolVersion;
        info.targetWeight = topo.targetWeight;
        info.flowControlConfig = topo.flowControlConfig;
        infoVec.push_back(info);
    }
    return publishGroup(group, infoVec, signatureVec);
}

bool GigRpcServer::updateVolatileInfo(SignatureTy signature, const BizVolatileInfo &info) {
    if (_heartbeatManager) {
        return _heartbeatManager->updateVolatileInfo(signature, info);
    }
    {
        AUTIL_LOG(ERROR, "update topo failed [%lu], call publish first", signature);
        return false;
    }
}

bool GigRpcServer::unpublish(SignatureTy signature) {
    if (_heartbeatManager) {
        return _heartbeatManager->unpublish(signature);
    } else {
        AUTIL_LOG(ERROR, "unpublish topo failed [%lu], call publish first", signature);
        return false;
    }
}

bool GigRpcServer::unpublish(const std::vector<SignatureTy> &signatureVec) {
    if (_heartbeatManager) {
        return _heartbeatManager->unpublish(signatureVec);
    } else {
        AUTIL_LOG(ERROR, "unpublish topo vec failed, count [%lu], call publish first",
                  signatureVec.size());
        return false;
    }
}

std::vector<ServerBizTopoInfo> GigRpcServer::getPublishInfos() const {
    if (_heartbeatManager) {
        return _heartbeatManager->getPublishInfos();
    } else {
        return {};
    }
}

std::string GigRpcServer::getTopoInfoStr() const {
    if (_heartbeatManager) {
        return _heartbeatManager->getTopoInfoStr();
    } else {
        return TopoInfoBuilder().build();
    }
}

Spec GigRpcServer::getRpcSpec() {
    Spec spec;
    spec.ip = autil::NetUtil::getBindIp();
    if (_arpcServer) {
        auto arpcPort = _arpcServer->GetListenPort();
        if (arpcPort < 0) {
            arpcPort = INVALID_PORT;
        }
        spec.ports[MC_PROTOCOL_ARPC] = arpcPort;
    }
    if (_httpArpcServer) {
        auto httpPort = _httpArpcServer->getPort();
        if (httpPort < 0) {
            httpPort = INVALID_PORT;
        }
        spec.ports[MC_PROTOCOL_HTTP] = httpPort;
    }
    auto grpcPort = getGrpcPort();
    spec.ports[MC_PROTOCOL_GRPC] = grpcPort;
    spec.ports[MC_PROTOCOL_GRPC_STREAM] = grpcPort;
    
    spec.ports[MC_PROTOCOL_TCP] = _tcpPort;
    return spec;
}

void GigRpcServer::setSearchService(const SearchServicePtr &searchService) {
    autil::ScopedSpinLock scoped(_spinLock);
    _searchService = searchService;
}

SearchServicePtr GigRpcServer::getSearchService() {
    autil::ScopedSpinLock scoped(_spinLock);
    return _searchService;
}

bool GigRpcServer::registerGrpcService(const std::string &methodName,
                                       google::protobuf::Message *request,
                                       google::protobuf::Message *response,
                                       const CompatibleFieldInfo &compatibleInfo,
                                       const GigRpcMethod &method) {
    if (!_grpcWorker) {
        AUTIL_LOG(INFO, "grpc server has not been initialized");
        return false;
    }
    if (!request || !response || !method) {
        DELETE_AND_SET_NULL(request);
        DELETE_AND_SET_NULL(response);
        return false;
    }
    GigRpcMethodArgPtr arg(new GigRpcMethodArg());
    arg->request = request;
    arg->response = response;
    arg->method = method;
    arg->compatibleInfo = compatibleInfo;
    return _grpcWorker->addMethod(methodName, arg);
}

/**
 * _arpcServer->RegisterService(wrapper(_agent, rpcService, ...), ...)
 * append to _wrapper if success
 */
bool GigRpcServer::registerArpcService(google::protobuf::Service *rpcService,
                                       const CompatibleFieldInfo &compatibleInfo,
                                       const arpc::ThreadPoolDescriptor &threadPoolDescriptor) {
    return registerArpcService<arpc::ThreadPoolDescriptor>(rpcService, compatibleInfo,
                                                           threadPoolDescriptor);
}

bool GigRpcServer::registerArpcService(google::protobuf::Service *rpcService,
                                       const CompatibleFieldInfo &compatibleInfo,
                                       const autil::ThreadPoolBasePtr &pool) {
    return registerArpcService<autil::ThreadPoolBasePtr>(rpcService, compatibleInfo, pool);
}

bool GigRpcServer::registerArpcService(google::protobuf::Service *rpcService,
                                       const CompatibleFieldInfo &compatibleInfo,
                                       const arpc::ThreadPoolDescriptor &threadPoolDescriptor,
                                       const std::map<std::string, std::string> &aliasMap) {
    if (!registerArpcService<arpc::ThreadPoolDescriptor>(rpcService, compatibleInfo,
                                                         threadPoolDescriptor)) {
        return false;
    }
    if (_httpArpcServer) {
        _httpArpcServer->addAlias(aliasMap);
    }
    return true;
}

template <typename ThreadPoolRep>
bool GigRpcServer::registerArpcService(google::protobuf::Service *rpcService,
                                       const CompatibleFieldInfo &compatibleInfo,
                                       const ThreadPoolRep &poolRep) {
    if (!_arpcServer) {
        AUTIL_LOG(INFO, "gig arpc server disabled");
        return false;
    }
    auto wrapper = std::make_shared<ServiceWrapper>(*this, rpcService, compatibleInfo);
    auto ret = _arpcServer->RegisterService(wrapper.get(), poolRep);
    if (ret) {
        _wrappers.insert(wrapper);
    } else {
        AUTIL_LOG(ERROR, "register service:[%s] wrapper failed!",
                  rpcService->GetDescriptor()->DebugString().c_str());
        wrapper.reset();
    }
    if (_httpArpcServer) {
        _httpArpcServer->registerService();
    }
    
    return ret;
}

void GigRpcServer::unRegisterArpcService(
    const std::shared_ptr<google::protobuf::Service> &rpcService) {
    if (!_arpcServer) {
        return;
    }
    std::shared_ptr<ServiceWrapper> found;
    for (auto wrapper : _wrappers) {
        if (rpcService == wrapper->getRpcServicePtr()) {
            found = wrapper;
            break;
        }
    }
    if (!found) {
        return;
    }
    _arpcServer->unRegisterService(found);
    if (_httpArpcServer) {
        _httpArpcServer->registerService();
    }
    _wrappers.erase(found);
}

bool GigRpcServer::registerArpcService(const std::shared_ptr<google::protobuf::Service> &rpcService,
                                       const std::string &methodName,
                                       const CompatibleFieldInfo &compatibleInfo,
                                       const arpc::ThreadPoolDescriptor &threadPoolDescriptor) {
    if (!_arpcServer) {
        AUTIL_LOG(INFO, "gig arpc server disabled");
        return false;
    }
    auto wrapper = std::make_shared<ServiceWrapper>(*this, rpcService.get(), compatibleInfo);
    wrapper->setRpcServicePtr(rpcService);
    auto ret = _arpcServer->RegisterService(wrapper, methodName, threadPoolDescriptor);
    if (ret) {
        _wrappers.insert(wrapper);
    } else {
        AUTIL_LOG(ERROR, "register service:[%s] wrapper failed!",
                  rpcService->GetDescriptor()->DebugString().c_str());
        wrapper.reset();
    }
    if (_httpArpcServer) {
        _httpArpcServer->registerService();
    }
    
    return ret;
}

bool GigRpcServer::registerGrpcStreamService(const GigServerStreamCreatorPtr &creator) {
    if (!_grpcWorker) {
        AUTIL_LOG(ERROR, "grpc server has not been initialized");
        return false;
    }
    return _grpcWorker->registerStreamService(creator);
}

bool GigRpcServer::unRegisterGrpcStreamService(const GigServerStreamCreatorPtr &creator) {
    if (!_grpcWorker) {
        AUTIL_LOG(ERROR, "grpc server has not been initialized");
        return false;
    }
    return _grpcWorker->unRegisterStreamService(creator);
}

arpc::ANetRPCServer *GigRpcServer::getArpcServer() const {
    return _arpcServer.get();
}

http_arpc::HTTPRPCServer *GigRpcServer::getHttpArpcServer() const {
    return _httpArpcServer.get();
}



bool GigRpcServer::hasGrpc() const {
    return _grpcWorker.get();
}

int32_t GigRpcServer::getGrpcPort() const {
    if (!_grpcWorker) {
        return INVALID_PORT;
    } else {
        return _grpcWorker->getPort();
    }
}

std::vector<int32_t> GigRpcServer::getGrpcIoPids() const {
    if (!_grpcWorker) {
        return {};
    } else {
        return _grpcWorker->getIoPids();
    }
}

GigAgentPtr &GigRpcServer::getAgent() {
    return _agent;
}

void GigRpcServer::startAgent(int64_t smoothIntervalUs) {
    _agent->start(smoothIntervalUs);
    if (_heartbeatManager) {
        _heartbeatManager->startAllBiz();
    }
}

void GigRpcServer::stopAgent(int64_t smoothIntervalUs) {
    _agent->stop(smoothIntervalUs);
    if (_heartbeatManager) {
        _heartbeatManager->stopAllBiz();
    }
    if (smoothIntervalUs > 0) {
        usleep(smoothIntervalUs);
    }
    if (_heartbeatManager) {
        _heartbeatManager->notify();
    }
}

bool GigRpcServer::waitUntilNoQuery(int64_t noQueryTimeInSecond, int64_t timeoutInSecond,
                                    bool disableAllReplicaStopped) {
    AUTIL_LOG(
        INFO,
        "begin wait until no query, noQueryTime[%ld]s timeout[%ld]s, disableAllReplicaStopped [%d]",
        noQueryTimeInSecond, timeoutInSecond, disableAllReplicaStopped);
    int64_t waitTime = 0;
    while (waitTime < timeoutInSecond) {
        if (!disableAllReplicaStopped && _heartbeatManager &&
            _heartbeatManager->isAllReplicaStopped()) {
            AUTIL_LOG(WARN, "all replica stopped, ignore no query wait");
            return true;
        }
        if (_agent->longTimeNoQuery(noQueryTimeInSecond)) {
            break;
        }
        usleep(1 * 1000 * 1000);
        waitTime += 1;
    }
    if (waitTime >= timeoutInSecond) {
        AUTIL_LOG(WARN, "wait timeout %ld", timeoutInSecond);
        return false;
    }
    AUTIL_LOG(INFO, "end wait until no query, wait[%ld]s", waitTime);
    return true;
}

template bool GigRpcServer::registerArpcService<arpc::ThreadPoolDescriptor>(
    google::protobuf::Service *, const CompatibleFieldInfo &, const arpc::ThreadPoolDescriptor &);
template bool GigRpcServer::registerArpcService<autil::ThreadPoolBasePtr>(
    google::protobuf::Service *, const CompatibleFieldInfo &, const autil::ThreadPoolBasePtr &);

} // namespace multi_call
