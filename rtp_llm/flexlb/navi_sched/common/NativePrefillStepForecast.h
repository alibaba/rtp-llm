#pragma once

#include "aios/navi_sched/common/TPUPrefillExecutionPolicy.h"
#include "aios/navi_sched/data/NaviSchedLatencySample.h"
#include "autil/legacy/jsonizable.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <unordered_set>

namespace navi_sched {

// Master-local frozen inputs, also retained in schedule replay JSON.
struct NativePrefillStepRequest {
    uint64_t requestId = 0;
    TPUPrefillRequestInput input;
    bool executing = false;
    bool reported = false;
};

class NativePrefillStepForecast {
public:
    static bool declaredBy(const std::string &fingerprint)
    {
        if (fingerprint.find("\"unified_capacity_batch_count_unit\"") == std::string::npos)
            return false;
        try {
            autil::legacy::json::JsonMap values;
            autil::legacy::FromJsonString(values, fingerprint);
            const auto found = values.find("unified_capacity_batch_count_unit");
            return found != values.end() &&
                   autil::legacy::AnyCast<std::string>(found->second) ==
                       "native_positive_prefill_step_v1";
        } catch (const std::exception &) {
            return false;
        }
    }

    // An all-cache-resolved forecast, never a runtime scheduling instruction.
    // Unknown readiness/metadata is handled by the caller before invoking this.
    // Exact volume parity prevents an incomplete request list from dropping work.
    static bool prepare(NaviSchedLatencyFeatures &features,
                        const std::vector<NativePrefillStepRequest> &requests,
                        const TPUPrefillExecutionPolicyConfig &policy)
    {
        if (features.getExpectedKVWorkload() || features.getWaitingBatchSize() != 0 ||
            features.getRunningDecodeRequestCount() != 0) return false;
        long double count = 0, hit = 0, miss = 0, square = 0, product = 0;
        size_t bound = 0;
        std::unordered_set<uint64_t> seen;
        std::vector<TPUPrefillRequestInput> ordered;
        ordered.reserve(requests.size());
        for (const auto &request : requests) {
            const auto &input = request.input;
            if (!request.requestId || !seen.insert(request.requestId).second ||
                input.computedTokens < 0 ||
                input.promptTokens <= input.computedTokens ||
                input.sharedPrefixBoundary < 0 ||
                input.sharedPrefixBoundary > input.promptTokens ||
                input.prefillEndTokens != 0 || input.numTokens != 0) return false;
            const auto n = input.promptTokens - input.computedTokens;
            if (static_cast<uint64_t>(n) > std::numeric_limits<size_t>::max() - bound)
                return false;
            bound += static_cast<size_t>(n);
            ++count;
            hit += input.computedTokens;
            miss += n;
            square += static_cast<long double>(n) * n;
            product += static_cast<long double>(input.computedTokens) * n;
            ordered.push_back(input);
        }
        const auto equal = [](long double lhs, long double rhs) {
            return std::isfinite(lhs) && std::isfinite(rhs) &&
                   std::abs(lhs - rhs) <= 1e-10L * std::max(1.0L, std::abs(rhs));
        };
        if (!equal(count, features.getRunningBatchSize()) ||
            !equal(hit, features.getRunningCacheHitTokenCount()) ||
            !equal(miss, features.getRunningCacheMissTokenCount()) ||
            !equal(square, features.getRunningCacheMissTokenSquareSum()) ||
            !equal(product, features.getRunningCacheHitMissTokenProductSum())) return false;
        TPUPrefillExecutionProjection projection;
        if (!TPUPrefillExecutionPolicy::validate(policy) ||
            (!ordered.empty() &&
             !TPUPrefillExecutionPolicy::projectCacheResolvedBatchSummary(
                 ordered, policy, bound, projection))) return false;
        return features.setBatchCount(projection.stepCount);
    }
};

} // namespace navi_sched
