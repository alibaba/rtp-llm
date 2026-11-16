#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/normal_engine/DecodeReplicaCoordinator.h"

namespace rtp_llm {

inline bool isDsv4DedicatedDecodeTopology(
    RoleType role, int pp, int tp, int dp, int ep, int world, int local_world, bool local_cp) {
    return role == RoleType::DECODE && pp == 1 && tp == 1 && (dp == 4 || dp == 8) && ep == dp && world == dp
           && local_world == world && !local_cp;
}

inline bool isDsv4DecodeRequestSupported(
    bool is_fake, bool is_context, bool cache_ready, int num_beams, int num_sequences, bool force_disable_speculation) {
    return is_fake
           || (!is_context && cache_ready && num_beams == 1 && num_sequences == 1 && !force_disable_speculation);
}

inline constexpr int     kDsv4DecodeMaxBuckets      = 32;
inline constexpr int64_t kDsv4DecodeProtocolVersion = 3;

inline bool validDsv4DecodeCaptureBuckets(const std::vector<int>& buckets, int world, int width) {
    return !buckets.empty() && buckets.size() <= kDsv4DecodeMaxBuckets && world > 0 && width > 0
           && std::is_sorted(buckets.begin(), buckets.end())
           && std::adjacent_find(buckets.begin(), buckets.end()) == buckets.end() && buckets.front() > 0
           && buckets.back() <= std::numeric_limits<int>::max() / world / width;
}

enum class Dsv4DecodeControlVote : int64_t {
    RUN,
    PAUSE,
    STOP
};

inline DecodeReplicaRound classifyDsv4DecodeControl(const std::vector<int64_t>& records) {
    if (std::find(records.begin(), records.end(), static_cast<int64_t>(Dsv4DecodeControlVote::STOP)) != records.end()) {
        return DecodeReplicaRound::STOP;
    }
    return std::find(records.begin(), records.end(), static_cast<int64_t>(Dsv4DecodeControlVote::PAUSE))
                   != records.end() ?
               DecodeReplicaRound::PAUSED :
               DecodeReplicaRound::EXECUTE;
}

inline DecodeReplicaRound classifyDsv4DecodeRound(const std::vector<int64_t>& records) {
    if (records.empty() || *std::min_element(records.begin(), records.end()) < 0) {
        return DecodeReplicaRound::INVALID;
    }
    return *std::max_element(records.begin(), records.end()) > 0 ? DecodeReplicaRound::EXECUTE :
                                                                   DecodeReplicaRound::IDLE;
}

enum class Dsv4DecodeControl : int64_t {
    MODEL_INITIALIZE,
    CACHE_CAPACITY,
    STARTUP,
    CAPTURE,
    CAPTURE_INPUTS_READY,
    CAPTURE_WARMUP_READY,
    CAPTURE_COMPLETE,
    ROUND_CONTROL,
    ROUND,
    PLAN_SPECULATIVE,
    PLAN_EAGER_WARMUP,
    PREPARE_GRAPH,
    PREPARE_EAGER,
    SPECULATIVE_STATUS,
};

struct Dsv4DecodeRecord {
    int64_t              sequence;
    Dsv4DecodeControl    control;
    DecodeExecutionPhase phase;
    int64_t              value;
    int64_t              bucket           = 0;
    int64_t              width            = 0;
    int64_t              protocol_version = kDsv4DecodeProtocolVersion;
};

inline bool matchingDsv4DecodeInvocation(const Dsv4DecodeRecord& left, const Dsv4DecodeRecord& right) {
    return left.sequence == right.sequence && left.control == right.control && left.phase == right.phase
           && left.width == right.width && left.bucket == right.bucket
           && left.protocol_version == right.protocol_version;
}

inline int dsv4DecodeRequestBucket(int64_t max_local_requests, const std::vector<int>& buckets) {
    const auto bucket = std::lower_bound(buckets.begin(), buckets.end(), max_local_requests);
    return max_local_requests > 0 && bucket != buckets.end() ? *bucket : 0;
}

inline DecodeExecution dsv4DecodeCandidate(DecodeExecutionPhase    phase,
                                           int                     token_width,
                                           int64_t                 max_local_requests,
                                           const std::vector<int>& buckets) {
    const auto bucket = dsv4DecodeRequestBucket(max_local_requests, buckets);
    return {bucket ? DecodeExecutionMode::GRAPH : DecodeExecutionMode::EAGER, phase, bucket, token_width};
}

inline bool allDsv4DecodePeersReady(const std::vector<int64_t>& votes) {
    return !votes.empty() && std::all_of(votes.begin(), votes.end(), [](int64_t vote) { return vote == 1; });
}

inline DecodePreparationStatus combineDsv4DecodeReadiness(DecodePreparationStatus left, DecodePreparationStatus right) {
    return std::max(left, right);
}

}  // namespace rtp_llm
