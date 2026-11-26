#pragma once

#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "aios/navi_sched/scheduler/ScheduleContextInternal.h"

namespace navi_sched {

struct TairKvcmObjectQuery {
    int64_t blockKey = 0;
    std::string locationSpecName;
};

// A logical V6D group is usable only when every physical PP/PCP object hits.
struct TairKvcmLogicalGroupQuery {
    size_t groupIndex = 0;
    size_t physicalBegin = 0;
    NaviSchedBlockHashes blockHashes;
};

struct TairKvcmQueryInput {
    std::shared_ptr<const NaviSchedTairKvcmConfig> policy;
    std::string traceId;
    // One entry per physical group/node spec. The caller retains the mapping
    // to logical Full/Mamba blocks and requires all PP parts before matching.
    std::vector<TairKvcmObjectQuery> objects;
    std::vector<TairKvcmLogicalGroupQuery> logicalGroups;
    bool exceedsObjectBudget = false;
};

enum class TairKvcmQueryStatus {
    NOT_CONFIGURED, EMPTY, OK, INVALID_REQUEST, WINDOW_EXPIRED,
    RPC_ERROR, INVALID_RESPONSE, BUSINESS_ERROR,
    RATE_LIMITED, IN_FLIGHT, BACKOFF, TOO_LARGE,
};

// Size bounds apply to one deduplicated lookup. The token only observes its
// outstanding RPC lifetime; it never rejects or delays a query.
constexpr size_t TAIR_KVCM_MAX_QUERY_OBJECTS = 32768;
constexpr size_t TAIR_KVCM_MAX_RPC_BYTES = 4 * 1024 * 1024;
std::shared_ptr<void> trackTairKvcmQuery();
size_t tairKvcmQueriesInFlight();

struct TairKvcmQueryResult {
    TairKvcmQueryStatus status = TairKvcmQueryStatus::NOT_CONFIGURED;
    // Zero for every failed lookup; only nonempty remote URIs count as hits.
    std::vector<uint8_t> remoteHits;
    int64_t elapsedUs = 0;
    size_t requestBytes = 0;
    size_t responseBytes = 0;
};

// Batch all requests for one connected V6D instance. Repeated group keys are
// queried once, with exactly the spec names registered by Native V6D.
void prepareTairKvcmBatchQuery(
    const std::shared_ptr<const NaviSchedTairKvcmConfig> &config,
    const std::vector<NaviSchedScheduleItem> &requests, TairKvcmQueryInput &input,
    size_t maxObjects = TAIR_KVCM_MAX_QUERY_OBJECTS);
NaviSchedExternalCacheReadyBlocks getTairKvcmReadyBlocks(const TairKvcmQueryInput &input,
                                         const TairKvcmQueryResult &result);

const char *tairKvcmQueryStatusName(TairKvcmQueryStatus status);


} // namespace navi_sched
