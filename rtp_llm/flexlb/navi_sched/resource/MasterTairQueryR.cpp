#include "aios/navi_sched/resource/MasterTairQueryR.h"
#include "autil/TimeUtility.h"
#include "autil/Scope.h"
#include "autil/legacy/json.h"
#include "navi/engine/NaviRegisterMacro.h"
#include <algorithm>

namespace navi_sched {
namespace {
bool sameQueryPolicy(const NaviSchedTairKvcmConfig &a, const NaviSchedTairKvcmConfig &b)
{
    if (a.instanceId != b.instanceId ||
        a.hashBlockSize != b.hashBlockSize || a.nodeCount != b.nodeCount ||
        a.groups.size() != b.groups.size()) return false;
    for (size_t i = 0; i < a.groups.size(); ++i) {
        const auto &x = a.groups[i];
        const auto &y = b.groups[i];
        if (x.groupId != y.groupId || x.hashBlockScale != y.hashBlockScale ||
            x.tail != y.tail || x.objectSizeBytes != y.objectSizeBytes ||
            x.locationSpecNames != y.locationSpecNames) return false;
    }
    return true;
}
}



const std::string MasterTairQueryR::RESOURCE_ID = "navi_sched.master_tair_query_r";
void MasterTairQueryR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ)
        .dependOn(TairKvcmGrpcClientR::RESOURCE_ID, true, BIND_RESOURCE_TO(_grpcClientR))
        .dependOn(kmonitor::MetricsReporterR::RESOURCE_ID, false,
                  BIND_RESOURCE_TO(_metricsReporterR));
}
navi::ErrorCode MasterTairQueryR::init(navi::ResourceInitContext &ctx)
{
    // Native capability controls use. Initializing this resource sends no traffic.
    return navi::EC_NONE;
}


std::shared_ptr<const NaviSchedTairKvcmConfig> MasterTairQueryR::canonicalPolicy(
    const std::shared_ptr<const NaviSchedTairKvcmConfig> &policy,
    const std::string &instanceId)
{
    std::lock_guard<std::mutex> lock(_cacheMutex);
    const auto key = std::make_pair(policy.get(), instanceId);
    const auto found = _policyAliases.find(key);
    if (found != _policyAliases.end()) return found->second.canonical;
    // Compare fixed layouts only when a new Agent configuration is connected.
    // Live rounds use the cached pointer; identical namespaces share one lookup.
    std::shared_ptr<const NaviSchedTairKvcmConfig> canonical;
    auto effective = std::make_shared<NaviSchedTairKvcmConfig>(*policy);
    effective->instanceId = instanceId;
    for (const auto &entry : _policyAliases) {
        if (sameQueryPolicy(*effective, *entry.second.canonical)) {
            canonical = entry.second.canonical;
            break;
        }
    }
    if (!canonical) canonical = std::move(effective);
    _policyAliases.emplace(key, PolicyAlias{policy, canonical});
    return canonical;
}


void MasterTairQueryR::buildQueries(MasterScheduleContextData &data,
    const std::vector<NaviSchedScheduleItem> &requests, uint64_t scheduleId,
    const SchedulerConfigR::Config &config)
{
    if (!data.tairKvcmQueries.empty()) return;
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    data.tairKvcmNodeIndexes.clear();
    if (beginUs >= data.context->getScheduleDeadlineUs()) return;
    {
        std::lock_guard<std::mutex> lock(_cacheMutex);
        for (auto it = _policyAliases.begin(); it != _policyAliases.end();) {
            if (it->second.source.expired()) it = _policyAliases.erase(it);
            else ++it;
        }
    }
    // Match FlexLB: a scheduling pool uses its first effective Worker's
    // registered instance ID. Other IDs do not create additional namespaces.
    std::string instanceId;
    for (size_t n = 0; n < data.context->nodeCount(); ++n) {
        const auto source = data.context->getNode(n).getTairKvcmConfig();
        if (source && !source->instanceId.empty()) {
            instanceId = _grpcClientR && !_grpcClientR->topology().instanceNamespace.empty()
                ? _grpcClientR->topology().instanceNamespace + "_" + std::to_string(source->hashBlockSize)
                : source->instanceId;
            break;
        }
    }
    if (instanceId.empty()) return;
    std::map<const NaviSchedTairKvcmConfig *, size_t> queryIndexes;
    for (size_t n = 0; n < data.context->nodeCount(); ++n) {
        const auto source = data.context->getNode(n).getTairKvcmConfig();
        if (!source) continue;
        const auto policy = canonicalPolicy(source, instanceId);
        const auto found = queryIndexes.find(policy.get());
        if (found != queryIndexes.end()) {
            data.tairKvcmNodeIndexes[found->second].push_back(n);
            continue;
        }
        TairKvcmQueryInput input;
        prepareTairKvcmBatchQuery(policy, requests, input, config.tairQueryMaxObjects);

        input.traceId = std::to_string(scheduleId);
        queryIndexes.emplace(policy.get(), data.tairKvcmQueries.size());
        data.tairKvcmQueries.emplace_back(std::move(input));
        data.tairKvcmNodeIndexes.push_back({n});
    }
    data.tairKvcmPrepareUs = autil::TimeUtility::monotonicTimeUs() - beginUs;
}


MasterTairQueryR::BatchPtr MasterTairQueryR::startBatch(
    MasterScheduleContextDataPtr data, const std::vector<NaviSchedScheduleItem> &requests,
    uint64_t scheduleId, const navi::AsyncPipePtr &pipe, uint64_t round)
{
    auto state = std::make_shared<Batch>();
    auto &batch = *state;
    batch.data = std::move(data);
    batch.round = round;
    batch.completionPipe = pipe;
    batch.config = batch.data->schedulerConfig ? batch.data->schedulerConfig->config : SchedulerConfigR::Config();
    if (_grpcClientR && _grpcClientR->topology().configured())
        buildQueries(*batch.data, requests, scheduleId, batch.config);
    batch.queries.assign(batch.data->tairKvcmQueries.size(), Query());
    batch.data->tairKvcmResults.assign(batch.queries.size(), TairKvcmQueryResult());
    batch.remaining = batch.queries.size();
    for (size_t i = 0; i < batch.queries.size(); ++i) {
        auto &query = batch.queries[i];
        query.startedUs = autil::TimeUtility::monotonicTimeUs();
        auto &input = batch.data->tairKvcmQueries[i];
        batch.data->tairKvcmResults[i].remoteHits.assign(input.objects.size(), 0);
        if (!input.policy || !_grpcClientR || !_grpcClientR->topology().configured()) {
            finish(batch, i, TairKvcmQueryStatus::NOT_CONFIGURED);
            continue;
        }
        if (input.exceedsObjectBudget || input.objects.size() > batch.config.tairQueryMaxObjects) {
            finish(batch, i, TairKvcmQueryStatus::TOO_LARGE);
            continue;
        }
        if (input.objects.empty()) {
            finish(batch, i, TairKvcmQueryStatus::EMPTY);
            continue;
        }
        if (query.startedUs >= batch.data->context->getScheduleDeadlineUs()) {
            finish(batch, i, TairKvcmQueryStatus::WINDOW_EXPIRED);
            continue;
        }
        query.body = makeTairKvcmGrpcQueryMessage(input, batch.config.tairQueryMaxHttpBytes);
        if (!query.body) {
            finish(batch, i, TairKvcmQueryStatus::TOO_LARGE);
            continue;
        }
        query.token = trackTairKvcmQuery();
        auto service = _service;
        query.service = service;
        bool cachedLeader = false;
        {
            std::lock_guard<std::mutex> lock(service->mutex);
            query.leader = service->leader;
            query.serviceRevision = service->revision;
            cachedLeader = query.leader.valid() && !service->refreshRequired;
        }
        if (cachedLeader) {
            startLookup(batch, i);
        } else {
            startDiscovery(batch, i);
        }
    }
    return state;
}


void MasterTairQueryR::startDiscovery(Batch &batch, size_t index)
{
    auto &query = batch.queries[index];
    const auto targets = _grpcClientR->bootstrapTargets();
    if (targets.empty()) {
        finish(batch, index, TairKvcmQueryStatus::RPC_ERROR);
        return;
    }
    // Try another subscribed bootstrap after a failed discovery when available.
    const auto &target = targets[query.discoveryAttempts++ % targets.size()];
    const auto split = target.rfind(':');
    const TairKvcmGrpcEndpoint seed{target.substr(0, split), std::stoi(target.substr(split + 1))};
    send(batch, index, RpcStage::LEADER, seed, TAIR_KVCM_GRPC_GET_CLUSTER_INFO,
         makeTairKvcmGrpcLeaderMessage(batch.data->tairKvcmQueries[index].traceId));
}


bool MasterTairQueryR::retry(Batch &batch, size_t index, const char *reason)
{
    auto &query = batch.queries[index];
    // One recovery cycle per admitted logical query, with the original token
    // and common schedule deadline. No parallel requests or independent budget.
    if (query.retryCount != 0 || autil::TimeUtility::monotonicTimeUs() >=
        batch.data->context->getScheduleDeadlineUs()) return false;
    ++query.retryCount;
    if (_metricsReporterR && _metricsReporterR->getReporter()) {
        kmonitor::MetricsTags tags;
        tags.AddTag("stage", query.stage == RpcStage::LEADER ? "leader" : "lookup");
        tags.AddTag("reason", reason);
        _metricsReporterR->getReporter()->report(1, "master.tair_query_retry_qps",
            kmonitor::MetricType::QPS, &tags, kmonitor::MetricLevel::FATAL);
    }
    NAVI_KERNEL_LOG(INFO, "[navi-master-tair-retry] trace_id=%s query_index=%lu "
        "stage=%s reason=%s retry=%lu remaining_us=%ld", batch.data->tairKvcmQueries[index].traceId.c_str(),
        index, query.stage == RpcStage::LEADER ? "leader" : "lookup", reason, query.retryCount,
        batch.data->context->getScheduleDeadlineUs() - autil::TimeUtility::monotonicTimeUs());
    startDiscovery(batch, index);
    return true;
}


void MasterTairQueryR::startLookup(Batch &batch, size_t index)
{
    send(batch, index, RpcStage::LOOKUP, batch.queries[index].leader,
         TAIR_KVCM_GRPC_GET_REMOTE_LOCATIONS, batch.queries[index].body);
}


void MasterTairQueryR::send(Batch &batch, size_t index, RpcStage stage,
                                 const TairKvcmGrpcEndpoint &endpoint,
                                 const std::string &method, const TairKvcmGrpcMessagePtr &body)
{
    const size_t bodyBytes = body ? static_cast<size_t>(body->GetCachedSize()) : 0;
    if (!body || bodyBytes > batch.config.tairQueryMaxHttpBytes) {
        finish(batch, index, TairKvcmQueryStatus::TOO_LARGE);
        return;
    }
    const auto remainingUs = batch.data->context->getScheduleDeadlineUs() -
                             autil::TimeUtility::monotonicTimeUs();
    if (remainingUs < 1000) {
        finish(batch, index, TairKvcmQueryStatus::WINDOW_EXPIRED);
        return;
    }
    const auto &input = batch.data->tairKvcmQueries[index];
    if (!_grpcClientR || !endpoint.valid()) {
        finish(batch, index, TairKvcmQueryStatus::RPC_ERROR);
        return;
    }
    batch.queries[index].stage = stage;
    if (!_grpcClientR->call(endpoint.target(), method, body,
            batch.data->context->getScheduleDeadlineUs(), batch.config.tairQueryMaxHttpBytes,
            batch.completionPipe, index, batch.round, stage == RpcStage::LEADER,
            batch.queries[index].token)) {
        finish(batch, index, TairKvcmQueryStatus::RPC_ERROR);
        return;
    }
    batch.queries[index].sent = true;
    ++batch.queries[index].rpcCount;
    if (_metricsReporterR && _metricsReporterR->getReporter()) {
        const auto &reporter = _metricsReporterR->getReporter();
        kmonitor::MetricsTags tags;
        tags.AddTag("stage", stage == RpcStage::LEADER ? "leader" : "lookup");
        reporter->report(1, "master.tair_grpc_rpc_qps", kmonitor::MetricType::QPS, &tags,
                         kmonitor::MetricLevel::FATAL);
        reporter->report(bodyBytes, "master.tair_grpc_request_bytes_qps", kmonitor::MetricType::QPS,
                         &tags, kmonitor::MetricLevel::FATAL);
        if (stage == RpcStage::LOOKUP)
            reporter->report(input.objects.size(), "master.tair_lookup_object_qps", kmonitor::MetricType::QPS,
                             nullptr, kmonitor::MetricLevel::FATAL);
    }

}


void MasterTairQueryR::invalidateLeader(const Query &query)
{
    if (!query.service) return;
    std::lock_guard<std::mutex> lock(query.service->mutex);
    // A late failure from an older RPC must not invalidate a newer discovery.
    if (query.service->revision == query.serviceRevision)
        query.service->refreshRequired = true;
}


void MasterTairQueryR::completed(TairKvcmGrpcCompletionData &completion,
                                 const BatchPtr &state)
{
    completion.unpack();
    autil::ScopeGuard releaseCompletedRpc([&] {
        completion.queryToken.reset();
        if (_metricsReporterR && _metricsReporterR->getReporter())
            _metricsReporterR->getReporter()->report(tairKvcmQueriesInFlight(),
                "master.tair_query_inflight", kmonitor::MetricType::GAUGE,
                nullptr, kmonitor::MetricLevel::FATAL);
    });
    const bool active = state && completion.round == state->round &&
        completion.queryIndex < state->queries.size() &&
        state->queries[completion.queryIndex].stage != RpcStage::DONE;
    std::string body;
    TairKvcmQueryStatus status;
    size_t responseBytes = 0;
    responseBytes = completion.response.Length();
    const bool expired = completion.completedUs >= completion.deadlineUs ||
        completion.status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED;
    bool succeeded = completion.eventOk && completion.status.ok() && !expired;
    status = expired ? TairKvcmQueryStatus::WINDOW_EXPIRED : TairKvcmQueryStatus::RPC_ERROR;
    if (responseBytes > completion.maxResponseBytes ||
        completion.status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED) {
        status = TairKvcmQueryStatus::TOO_LARGE;
        succeeded = false;
    }
    if (succeeded) {
        std::vector<grpc::Slice> slices;
        succeeded = completion.response.Dump(&slices).ok();
        if (succeeded) {
            body.reserve(responseBytes);
            for (const auto &slice : slices)
                body.append(reinterpret_cast<const char *>(slice.begin()), slice.size());
            status = TairKvcmQueryStatus::OK;
        } else status = TairKvcmQueryStatus::INVALID_RESPONSE;
    }
    namespace proto = kv_cache_manager::proto::meta;
    auto errorCode = proto::OK;
    TairKvcmGrpcEndpoint discoveredLeader;
    TairKvcmQueryResult parsed;
    if (succeeded && active) {
        if (completion.discovery) {
            succeeded = readTairKvcmGrpcLeader(body, discoveredLeader, &errorCode);
            status = succeeded ? TairKvcmQueryStatus::OK : errorCode != proto::OK
                ? TairKvcmQueryStatus::BUSINESS_ERROR : TairKvcmQueryStatus::INVALID_RESPONSE;
        } else {
            parsed = readTairKvcmGrpcQueryResponse(
                state->data->tairKvcmQueries[completion.queryIndex], body, &errorCode);
            status = parsed.status;
            succeeded = status == TairKvcmQueryStatus::OK;
        }
    }
    reportRpcCompletion(completion, status, responseBytes, !active, errorCode);
    if (!active) return;
    auto &batch = *state;
    const auto index = completion.queryIndex;
    auto &query = batch.queries[index];
    auto &result = batch.data->tairKvcmResults[index];
    result.requestBytes += completion.requestBytes;
    result.responseBytes += responseBytes;
    if (!succeeded) {
        // FlexLB's business error lives in header.status.code, independently
        // of the gRPC transport status. Only a not-leader lookup is retried.
        const bool notLeader = query.stage == RpcStage::LOOKUP &&
            status == TairKvcmQueryStatus::BUSINESS_ERROR &&
            errorCode == proto::SERVER_NOT_LEADER;
        if (notLeader || status == TairKvcmQueryStatus::RPC_ERROR)
            invalidateLeader(query);
        if (notLeader && retry(batch, index, "not_leader")) return;
        finish(batch, index, status);
        return;
    }
    if (autil::TimeUtility::monotonicTimeUs() >= batch.data->context->getScheduleDeadlineUs()) {
        finish(batch, index, TairKvcmQueryStatus::WINDOW_EXPIRED);
        return;
    }
    if (query.stage == RpcStage::LEADER) {
        query.leader = std::move(discoveredLeader);
        bool changed = false;
        {
            std::lock_guard<std::mutex> lock(query.service->mutex);
            if (query.startedUs >= query.service->discoveryStartedUs) {
                changed = query.service->leader.valid() &&
                    query.service->leader.target() != query.leader.target();
                query.service->leader = query.leader;
                query.service->discoveryStartedUs = query.startedUs;
                query.service->refreshRequired = false;
                query.serviceRevision = ++query.service->revision;
            } else {
                query.serviceRevision = 0;
            }
        }
        if (_metricsReporterR && _metricsReporterR->getReporter()) {
            const auto &reporter = _metricsReporterR->getReporter();
            reporter->report(1, "master.tair_leader_refresh_qps", kmonitor::MetricType::QPS,
                             nullptr, kmonitor::MetricLevel::FATAL);
            if (changed) reporter->report(1, "master.tair_leader_change_qps", kmonitor::MetricType::QPS,
                                         nullptr, kmonitor::MetricLevel::FATAL);
        }
        NAVI_KERNEL_LOG(INFO, "[navi-master-tair-leader] trace_id=%s query_index=%lu "
            "leader=%s changed=%d retry=%lu", batch.data->tairKvcmQueries[index].traceId.c_str(), index,
            query.leader.target().c_str(), int(changed), query.retryCount);
        // Connection retention belongs to the shared GIG discovery snapshot.
        startLookup(batch, index);
    } else {
        result.remoteHits = std::move(parsed.remoteHits);
        finish(batch, index, parsed.status);
    }
}


void MasterTairQueryR::finish(Batch &batch, size_t index, TairKvcmQueryStatus status)
{
    // An outstanding gRPC callback retains the token after a window timeout.
    batch.queries[index].token.reset();
    auto &result = batch.data->tairKvcmResults[index];
    result.status = status;
    result.elapsedUs = autil::TimeUtility::monotonicTimeUs() - batch.queries[index].startedUs;
    if (status != TairKvcmQueryStatus::OK)
        result.remoteHits.assign(batch.data->tairKvcmQueries[index].objects.size(), 0);
    batch.queries[index].stage = RpcStage::DONE;
    --batch.remaining;
}


void MasterTairQueryR::expirePending(const BatchPtr &state)
{
    if (!state) return;
    auto &batch = *state;
    for (size_t i = 0; i < batch.queries.size(); ++i)
        if (batch.queries[i].stage != RpcStage::DONE)
            finish(batch, i, TairKvcmQueryStatus::WINDOW_EXPIRED);
}


void MasterTairQueryR::reportRpcCompletion(
    const TairKvcmGrpcCompletionData &completion, TairKvcmQueryStatus status,
    size_t responseBytes, bool late, kv_cache_manager::proto::meta::ErrorCode errorCode) const
{
    if (!_metricsReporterR || !_metricsReporterR->getReporter()) return;
    const auto &reporter = _metricsReporterR->getReporter();
    kmonitor::MetricsTags tags;
    tags.AddTag("stage", completion.discovery ? "leader" : "lookup");
    tags.AddTag("status", tairKvcmQueryStatusName(status));
    tags.AddTag("grpc_code", std::to_string(static_cast<int>(completion.status.error_code())));
    tags.AddTag("kvcm_code", std::to_string(static_cast<int>(errorCode)));
    reporter->report(1, "master.tair_grpc_completion_qps", kmonitor::MetricType::QPS, &tags,
                     kmonitor::MetricLevel::FATAL);
    reporter->report(std::max<int64_t>(0, completion.completedUs - completion.startedUs) / 1000.0,
                     "master.tair_grpc_rpc_ms", kmonitor::MetricType::GAUGE, &tags,
                     kmonitor::MetricLevel::FATAL);
    reporter->report(responseBytes, "master.tair_grpc_response_bytes_qps", kmonitor::MetricType::QPS,
                     &tags, kmonitor::MetricLevel::FATAL);
    if (late)
        reporter->report(1, "master.tair_late_reply_qps", kmonitor::MetricType::QPS,
                         nullptr, kmonitor::MetricLevel::FATAL);
}


void MasterTairQueryR::reportBatch(const Batch &batch) const
{
    if (!_metricsReporterR || !_metricsReporterR->getReporter()) return;
    const auto &reporter = _metricsReporterR->getReporter();
    size_t ok = 0;
    for (const auto &r : batch.data->tairKvcmResults) ok += r.status == TairKvcmQueryStatus::OK;
    const char *outcome = batch.data->tairKvcmResults.empty() ? "NO_QUERY"
        : ok == batch.data->tairKvcmResults.size() ? "ALL_OK" : ok ? "PARTIAL_OK" : "NO_OK";
    kmonitor::MetricsTags frameTags;
    frameTags.AddTag("outcome", outcome);
    const auto gauge = [&](const char *name, double value) {
        reporter->report(value, name, kmonitor::MetricType::GAUGE, nullptr,
                         kmonitor::MetricLevel::FATAL);
    };
    reporter->report(1, "master.tair_query_frame_qps", kmonitor::MetricType::QPS, &frameTags,
                     kmonitor::MetricLevel::FATAL);
    reporter->report(batch.data->context->requestCount(), "master.tair_query_frame_request_qps",
                     kmonitor::MetricType::QPS, &frameTags, kmonitor::MetricLevel::FATAL);
    gauge("master.tair_query_inflight", tairKvcmQueriesInFlight());
    gauge("master.tair_query_max_objects", batch.config.tairQueryMaxObjects);
    gauge("master.tair_query_max_http_bytes", batch.config.tairQueryMaxHttpBytes);
    reporter->report(batch.data->tairKvcmPrepareUs / 1000.0,
                     "master.tair_query_prepare_ms", kmonitor::MetricType::GAUGE,
                     nullptr, kmonitor::MetricLevel::FATAL);
    for (size_t i = 0; i < batch.data->tairKvcmResults.size(); ++i) {
        const auto &result = batch.data->tairKvcmResults[i];
        kmonitor::MetricsTags tags;
        tags.AddTag("status", tairKvcmQueryStatusName(result.status));
        const auto report = [&](const char *name, double value, kmonitor::MetricType type) {
            reporter->report(value, name, type, &tags, kmonitor::MetricLevel::FATAL);
        };
        report("master.tair_query_qps", 1, kmonitor::MetricType::QPS);
        if (result.status != TairKvcmQueryStatus::OK &&
            result.status != TairKvcmQueryStatus::EMPTY &&
            result.status != TairKvcmQueryStatus::NOT_CONFIGURED)
            report("master.tair_query_failed_qps", 1, kmonitor::MetricType::QPS);
        if (i < batch.queries.size()) {
            report("master.tair_query_rpc_count", batch.queries[i].rpcCount, kmonitor::MetricType::GAUGE);
            if (batch.queries[i].retryCount)
                report("master.tair_query_retry_result_qps", 1, kmonitor::MetricType::QPS);
        }
        report("master.tair_query_ms", result.elapsedUs / 1000.0, kmonitor::MetricType::GAUGE);
        report("master.tair_query_request_bytes", result.requestBytes, kmonitor::MetricType::GAUGE);
        report("master.tair_query_response_bytes", result.responseBytes, kmonitor::MetricType::GAUGE);
        size_t hits = 0;
        for (const auto hit : result.remoteHits) hits += hit;
        report("master.tair_query_remote_hit_objects", hits, kmonitor::MetricType::GAUGE);
    }
}


REGISTER_RESOURCE(MasterTairQueryR);

} // namespace navi_sched
