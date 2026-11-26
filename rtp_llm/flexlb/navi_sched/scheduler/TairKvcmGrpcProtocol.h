#pragma once
#include <memory>

#include "aios/navi_sched/scheduler/TairKvcmQuery.h"
#include "aios/navi_sched/proto/TairKvcmMetaService.pb.h"

namespace navi_sched {

struct TairKvcmGrpcEndpoint {
    std::string host;
    int32_t port = 0;
    bool valid() const;
    std::string target() const { return host + ":" + std::to_string(port); }
};

extern const std::string TAIR_KVCM_GRPC_GET_CLUSTER_INFO;
extern const std::string TAIR_KVCM_GRPC_GET_REMOTE_LOCATIONS;
// Immutable after publication; retries reuse the same typed request payload.
using TairKvcmGrpcMessagePtr = std::shared_ptr<google::protobuf::Message>;
TairKvcmGrpcMessagePtr makeTairKvcmGrpcLeaderMessage(const std::string &traceId);
TairKvcmGrpcMessagePtr makeTairKvcmGrpcQueryMessage(const TairKvcmQueryInput &input,
                                                   size_t maxBytes);
std::string makeTairKvcmGrpcLeaderRequest(const std::string &traceId);
bool makeTairKvcmGrpcQueryRequest(const TairKvcmQueryInput &input,
                                 std::string &payload, size_t maxBytes);
bool readTairKvcmGrpcLeader(const std::string &payload, TairKvcmGrpcEndpoint &endpoint,
                                                                        kv_cache_manager::proto::meta::ErrorCode *errorCode = nullptr);
TairKvcmQueryResult readTairKvcmGrpcQueryResponse(const TairKvcmQueryInput &input,
                                                const std::string &payload,
                                                kv_cache_manager::proto::meta::ErrorCode *errorCode = nullptr);

} // namespace navi_sched
