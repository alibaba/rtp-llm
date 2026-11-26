#include "aios/navi_sched/resource/TairKvcmGrpcClientR.h"
#include "aios/network/gig/multi_call/grpc/client/GrpcUnaryStatusReceiver.h"
#include "aios/network/gig/multi_call/interface/GrpcRequest.h"
#include "aios/network/gig/multi_call/interface/GrpcResponse.h"
#include "aios/network/gig/multi_call/interface/RequestGenerator.h"
#include "aios/network/gig/multi_call/interface/QuerySession.h"
#include "aios/network/gig/multi_call/service/GrpcConnectionPool.h"
#include "aios/network/gig/multi_call/service/SearchServiceManager.h"
#include "autil/TimeUtility.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/engine/ResourceConfigContext.h"
#include <algorithm>
#include <charconv>
#include <limits>
#include <set>

namespace navi_sched {
namespace {
const std::string META_SERVICE = "/kv_cache_manager.proto.meta.MetaService/";
const std::string COMPLETION_TYPE = "navi_sched.tair_kvcm_grpc_completion";
class QueryResponse final : public multi_call::GrpcResponse, public multi_call::GrpcUnaryStatusReceiver {
public:
    void onGrpcStatus(const grpc::Status &value, bool ok) override {
        status = value; eventOk = ok; statusReceived = true;
    }
    grpc::Status status;
    bool eventOk = false;
    bool statusReceived = false;
};
class QueryRequest final : public multi_call::GrpcRequest {
public:
    QueryRequest(const std::string &method, const std::shared_ptr<google::protobuf::Arena> &arena,
                 TairKvcmGrpcClientR::MessagePtr payload, size_t bytes)
        : GrpcRequest(method, arena), _payload(std::move(payload)), _bytes(bytes) {
        setMessage(_payload.get());
    }
    ~QueryRequest() override { setMessage(nullptr); } // The shared owner releases the protobuf.
    multi_call::ResponsePtr newResponse() override { return std::make_shared<QueryResponse>(); }
    size_t size() const override { return _bytes; }
private:
    TairKvcmGrpcClientR::MessagePtr _payload;
    size_t _bytes;
};
class QueryGenerator final : public multi_call::RequestGenerator {
public:
    QueryGenerator(const std::string &target, const std::string &method,
                   const TairKvcmGrpcClientR::MessagePtr &payload, size_t bytes, uint32_t timeoutMs)
        : RequestGenerator("navi_sched_tair_kvcm_unary") {
        const auto split = target.rfind(':');
        multi_call::Spec spec; spec.ip = target.substr(0, split);
        spec.ports[multi_call::MC_PROTOCOL_GRPC] = std::stoi(target.substr(split + 1));
        setSpec(spec); setDisableRetry(true); setDisableProbe(true);
        _request = std::make_shared<QueryRequest>(method, getProtobufArena(), payload, bytes);
        _request->setTimeout(timeoutMs);
    }
    void generate(multi_call::PartIdTy, multi_call::PartRequestMap &output) override {
        output.emplace(0, _request);
    }
private:
    multi_call::GrpcRequestPtr _request;
};
struct PendingCall {
    std::shared_ptr<TairKvcmGrpcCompletionData> completion;
    navi::AsyncPipePtr pipe;
    multi_call::ReplyPtr reply;
};
class QueryClosure final : public multi_call::Closure {
public:
    explicit QueryClosure(std::shared_ptr<PendingCall> call) : _call(std::move(call)) {}
    void Run() override {
        std::unique_ptr<QueryClosure> destroy(this);
        _call->completion->completedUs = autil::TimeUtility::monotonicTimeUs();
        _call->completion->reply = std::move(_call->reply);
        _call->pipe->setData(_call->completion);
    }
private:
    std::shared_ptr<PendingCall> _call;
};
}

const std::string TairKvcmGrpcClientR::RESOURCE_ID = "navi_sched.tair_kvcm_grpc_client_r";
TairKvcmGrpcCompletionData::TairKvcmGrpcCompletionData() : navi::Data(COMPLETION_TYPE) {}

void TairKvcmGrpcCompletionData::unpack()
{
    if (_unpacked) return;
    _unpacked = true;
    if (!reply) return; // Allows synthetic completions in deterministic tests.
    size_t lack = 0;
    const auto replies = reply->getResponses(lack);
    auto actual = lack == 0 && replies.size() == 1
        ? std::dynamic_pointer_cast<QueryResponse>(replies.front()) : nullptr;
    if (!actual || !actual->statusReceived) {
        eventOk = false;
        status = grpc::Status(grpc::StatusCode::UNKNOWN, "GIG unary reply unavailable");
        return;
    }
    eventOk = actual->eventOk;
    status = actual->status;
    if (actual->isFailed() && status.ok())
        status = grpc::Status(grpc::StatusCode::UNKNOWN, "GIG unary request failed");
    if (auto buffer = actual->getMessageBuffer()) response = *buffer;
    if (response.Length() > maxResponseBytes)
        status = grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "KVCM query reply exceeds byte limit");
}

bool TairKvcmGrpcClientR::configureTransport()
{
    if (_transportReady) return true;
    if (!_gigClientR || !_gigClientR->getSearchService()) return false;
    const auto manager = _gigClientR->getSearchService()->getSearchServiceManager();
    if (!manager || !manager->getConnectionManager()) return false;
    const auto pool = std::dynamic_pointer_cast<multi_call::GrpcConnectionPool>(
        manager->getConnectionManager()->getConnectionPool(multi_call::MC_PROTOCOL_GRPC));
    _transportReady = pool && pool->configureMaxMessageBytes(MAX_TRANSPORT_BYTES);
    return _transportReady;
}

bool TairKvcmGrpcClientR::validTarget(const std::string &target)
{
    const auto split = target.rfind(':');
    if (split == std::string::npos || !split || split + 1 == target.size() ||
        target.find_first_of(" /\r\n\t?#@") != std::string::npos ||
        target.find('\0') != std::string::npos) return false;
    // Current Spectrum/KVCM endpoints use IPv4 or DNS host names.
    if (target.substr(0, split).find(':') != std::string::npos) return false;
    int port = 0;
    const auto raw = target.data() + split + 1;
    const auto result = std::from_chars(raw, target.data() + target.size(), port);
    return result.ec == std::errc() && result.ptr == target.data() + target.size() &&
           port > 0 && port <= 65535;
}

bool TairKvcmGrpcClientR::Topology::valid() const
{
    if (!threadNum) return false;
    if (!configured()) return true;
    if (port <= 0 || port > 65535) return false;
    if (discovery.type == "direct")
        return validTarget(address + ":" + std::to_string(port));
    if (discovery.isSpectrum()) return address.find_first_not_of(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") == std::string::npos;
    if (address.find_first_of(" \r\n\t") != std::string::npos) return false;
    if (discovery.type == "vipserver") return true;
    if (discovery.type == "static-env") {
        if (discovery.hosts.empty()) return false;
        for (const auto &host : discovery.hosts) if (!validTarget(host)) return false;
        return true;
    }
    return false;
}

void TairKvcmGrpcClientR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ)
        .dependOn(navi::GigClientR::RESOURCE_ID, true, BIND_RESOURCE_TO(_gigClientR));
}

bool TairKvcmGrpcClientR::config(navi::ResourceConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "address", _topology.address, _topology.address);
    NAVI_JSONIZE(ctx, "port", _topology.port, _topology.port);
    NAVI_JSONIZE(ctx, "discovery", _topology.discovery, _topology.discovery);
    NAVI_JSONIZE(ctx, "namespace", _topology.instanceNamespace, _topology.instanceNamespace);
    NAVI_JSONIZE(ctx, "thread_num", _topology.threadNum, _topology.threadNum);
    return _topology.valid();
}

bool TairKvcmGrpcClientR::makeSubscription(const multi_call::MultiCallConfig &parent,
                                           multi_call::SubscribeConfig &subscription) const
{
    subscription = multi_call::SubscribeConfig();
    if (!_topology.configured() || !_topology.valid()) return false;
    const auto &discovery = _topology.discovery;
    const auto key = "navi_sched_tair_grpc:" + _topology.address;
    if (discovery.isSpectrum()) {
        subscription.spectrumConfig.serviceId = _topology.address;
        subscription.spectrumConfig.serviceKey = key;
        subscription.spectrumConfig.supportHeartbeat = false;
    } else if (discovery.type == "vipserver") {
        // Use the same GIG platform configuration as the existing subscriptions.
        subscription.vipConfig = parent.subConf.vipConfig;
        // The native VIP client has one process-wide environment. A second
        // subscription must not silently switch to another environment.
        if (!subscription.vipConfig.jmenvDom.empty() && !discovery.jmenvDom.empty() &&
            subscription.vipConfig.jmenvDom != discovery.jmenvDom) return false;
        if (!discovery.jmenvDom.empty()) subscription.vipConfig.jmenvDom = discovery.jmenvDom;
        if (subscription.vipConfig.jmenvDom.empty()) return false;
        subscription.vipConfig.clusters = {_topology.address};
        subscription.vipConfig.serviceKey = key;
        subscription.vipConfig.supportHeartbeat = false;
        subscription.vipConfig.grpcPortDiff = 0;
    } else if (discovery.type == "static-env") {
        std::set<std::string> ips;
        for (const auto &host : discovery.hosts) {
            const auto split = host.rfind(':');
            if (!ips.insert(host.substr(0, split)).second) continue;
            multi_call::LocalJsonize node;
            node._bizName = _topology.address; node._clusterName = _topology.address;
            node._partCnt = 1; node._partId = 0; node._version = multi_call::DEFAULT_VERSION_ID;
            node._weight = multi_call::MAX_WEIGHT; node._meta.targetWeight = multi_call::MAX_WEIGHT;
            node._ip = host.substr(0, split); node._httpPort = std::stoi(host.substr(split + 1));
            node._grpcPort = _topology.port; node._supportHeartbeat = false; node._serviceKey = key;
            subscription.localConfig.nodes.emplace_back(std::move(node));
        }
    } else return false;
    subscription.allowEmptySub = false;
    return true;
}

bool TairKvcmGrpcClientR::subscribe()
{
    if (!_gigClientR || !_gigClientR->getSearchService()) return false;
    const auto &service = _gigClientR->getSearchService();
    const auto manager = service->getSearchServiceManager();
    if (!manager) return false;
    multi_call::SubscribeConfig subscription;
    if (!makeSubscription(service->getMultiCallConfig(), subscription)) return false;
    _subscriptionReady = manager->addSubscribeService(subscription);
    return _subscriptionReady;
}

navi::ErrorCode TairKvcmGrpcClientR::init(navi::ResourceInitContext &)
{
    if (!_topology.configured()) return navi::EC_NONE;
    if (!configureTransport()) return navi::EC_INIT_RESOURCE;
    if (_topology.discovery.type != "direct" && !subscribe())
        NAVI_KERNEL_LOG(WARN, "KVCM discovery subscription unavailable, type [%s]",
                        _topology.discovery.type.c_str());
    return navi::EC_NONE;
}

std::vector<std::string> TairKvcmGrpcClientR::bootstrapTargets() const
{
    if (!_topology.configured()) return {};
    if (_topology.discovery.type == "direct")
        return {_topology.address + ":" + std::to_string(_topology.port)};
    if (!_subscriptionReady || !_gigClientR || !_gigClientR->getSearchService()) return {};
    const auto manager = _gigClientR->getSearchService()->getSearchServiceManager();
    const auto snapshot = manager ? manager->getSearchServiceSnapshot() : nullptr;
    if (!snapshot) return {};
    std::set<std::string> targets;
    for (const auto &entry : snapshot->getProviderMap()) {
        const auto &provider = entry.second.first;
        if (provider && provider->getClusterName() == _topology.address) {
            auto target = provider->getSpec().ip + ":" + std::to_string(_topology.port);
            if (validTarget(target)) targets.insert(std::move(target));
        }
    }
    return {targets.begin(), targets.end()};
}

bool TairKvcmGrpcClientR::call(const std::string &target, const std::string &method,
    const MessagePtr &payload, int64_t deadlineUs, size_t maxMessageBytes,
    const navi::AsyncPipePtr &pipe, size_t queryIndex, uint64_t round,
    bool discovery, const std::shared_ptr<void> &queryToken)
{
    const auto now = autil::TimeUtility::monotonicTimeUs();
    const auto remainingUs = deadlineUs - now;
    if (!_transportReady || !pipe || !payload || !validTarget(target) || remainingUs < 1000 ||
        !maxMessageBytes || maxMessageBytes > MAX_TRANSPORT_BYTES ||
        (method != META_SERVICE + "GetClusterInfo" &&
         method != META_SERVICE + "GetCacheLocationsByBackend")) return false;
    const size_t bytes = payload->ByteSizeLong();
    if (bytes > maxMessageBytes) return false;
    const auto service = _gigClientR->getSearchService();
    if (!service) return false;
    auto call = std::make_shared<PendingCall>();
    call->pipe = pipe;
    call->completion = std::make_shared<TairKvcmGrpcCompletionData>();
    auto &completion = *call->completion;
    completion.queryIndex = queryIndex; completion.round = round; completion.discovery = discovery;
    completion.queryToken = queryToken; completion.startedUs = now; completion.deadlineUs = deadlineUs;
    completion.requestBytes = bytes; completion.maxResponseBytes = maxMessageBytes;
    multi_call::SearchParam param;
    const auto timeoutMs = static_cast<uint32_t>(std::min<int64_t>(remainingUs / 1000,
        std::numeric_limits<uint32_t>::max()));
    param.generatorVec.push_back(std::make_shared<QueryGenerator>(target, method, payload, bytes, timeoutMs));
    param.closure = new QueryClosure(call);
    service->search(param, call->reply);
    return true;
}

// Shared GIG discovery snapshots own connection retention. A Resource must not
// sweep another user's connections or stop the shared SearchService.
REGISTER_RESOURCE(TairKvcmGrpcClientR);
} // namespace navi_sched
