#include "aios/navi_sched/scheduler/TairKvcmGrpcProtocol.h"

namespace navi_sched {
namespace proto = kv_cache_manager::proto::meta;
const std::string TAIR_KVCM_GRPC_GET_CLUSTER_INFO =
    "/kv_cache_manager.proto.meta.MetaService/GetClusterInfo";
const std::string TAIR_KVCM_GRPC_GET_REMOTE_LOCATIONS =
    "/kv_cache_manager.proto.meta.MetaService/GetCacheLocationsByBackend";

bool TairKvcmGrpcEndpoint::valid() const
{
    return !host.empty() && host.find_first_of(" :/\r\n\t?#@") == std::string::npos &&
           host.find('\0') == std::string::npos && port > 0 && port <= 65535;
}

TairKvcmGrpcMessagePtr makeTairKvcmGrpcLeaderMessage(const std::string &traceId)
{
    auto request = std::make_shared<proto::GetClusterInfoRequest>();
    request->set_trace_id(traceId);
    request->ByteSizeLong();
    // Same as FlexLB: Leader discovery is service-wide, without instance_id.
    return request;
}

TairKvcmGrpcMessagePtr makeTairKvcmGrpcQueryMessage(const TairKvcmQueryInput &input,
                                                   size_t maxBytes)
{
    if (!input.policy || input.policy->instanceId.empty()) return {};
    auto request = std::make_shared<proto::GetCacheLocationsByBackendRequest>();
    request->set_trace_id(input.traceId);
    request->set_instance_id(input.policy->instanceId);
    request->set_query_type(proto::QT_BATCH_GET);
    for (const auto &object : input.objects) {
        if (object.locationSpecName.empty()) return {};
        request->add_block_keys(object.blockKey);
        request->add_location_spec_names(object.locationSpecName);
        request->mutable_block_mask()->mutable_bool_masks()->add_values(false);
    }
    for (const auto backend : {proto::ST_TAIRMEMPOOL, proto::ST_NFS}) {
        auto selector = request->add_backend_selectors();
        selector->set_backend_type(backend);
        selector->set_strategy(proto::LSS_WEIGHTED_RANDOM);
    }
    // No L2 selector: KVCM does not need to calculate per-engine local hits.
    if (request->ByteSizeLong() > maxBytes) return {};
    return request;
}

std::string makeTairKvcmGrpcLeaderRequest(const std::string &traceId)
{
    return makeTairKvcmGrpcLeaderMessage(traceId)->SerializeAsString();
}

bool makeTairKvcmGrpcQueryRequest(const TairKvcmQueryInput &input,
                                 std::string &payload, size_t maxBytes)
{
    payload.clear();
    auto request = makeTairKvcmGrpcQueryMessage(input, maxBytes);
    return request && request->SerializeToString(&payload);
}


bool readTairKvcmGrpcLeader(const std::string &payload, TairKvcmGrpcEndpoint &endpoint,
                            proto::ErrorCode *errorCode)
{
    endpoint = {};
    if (errorCode) *errorCode = proto::OK;
    proto::GetClusterInfoResponse response;
    if (!response.ParseFromString(payload) || !response.has_header() ||
        !response.header().has_status()) return false;
    if (errorCode) *errorCode = response.header().status().code();
    if (response.header().status().code() != proto::OK || !response.has_leader_endpoint()) return false;
    TairKvcmGrpcEndpoint leader{response.leader_endpoint().host(),
                               response.leader_endpoint().meta_rpc_port()};
    if (!leader.valid()) return false;
    endpoint = std::move(leader);
    return true;
}

TairKvcmQueryResult readTairKvcmGrpcQueryResponse(const TairKvcmQueryInput &input,
                                                const std::string &payload,
                                                proto::ErrorCode *errorCode)
{
    if (errorCode) *errorCode = proto::OK;
    TairKvcmQueryResult result;
    result.status = TairKvcmQueryStatus::INVALID_RESPONSE;
    result.remoteHits.assign(input.objects.size(), 0);
    proto::GetCacheLocationsByBackendResponse response;
    if (!response.ParseFromString(payload) || !response.has_header() ||
        !response.header().has_status()) return result;
    if (errorCode) *errorCode = response.header().status().code();
    if (response.header().status().code() != proto::OK) {
        result.status = TairKvcmQueryStatus::BUSINESS_ERROR;
        return result;
    }
    if (static_cast<size_t>(response.key_locations_size()) != input.objects.size()) return result;
    std::vector<uint8_t> hits(input.objects.size(), 0);
    for (size_t i = 0; i < input.objects.size(); ++i) {
        for (const auto &location : response.key_locations(i).locations()) {
            if (location.type() != proto::ST_TAIRMEMPOOL && location.type() != proto::ST_NFS) continue;
            for (const auto &spec : location.location_specs())
                if (spec.name() == input.objects[i].locationSpecName && !spec.uri().empty()) hits[i] = 1;
        }
    }
    result.status = TairKvcmQueryStatus::OK;
    result.remoteHits = std::move(hits);
    return result;
}
} // namespace navi_sched
