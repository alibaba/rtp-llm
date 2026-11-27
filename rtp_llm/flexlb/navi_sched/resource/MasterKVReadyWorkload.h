#pragma once
#include "aios/navi_sched/common/NaviSchedCommon.h"
#include "aios/navi_sched/data/NaviSchedLatencySample.h"
#include "aios/navi_sched/data/NaviSchedKVPreparation.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_set>
#include <vector>

namespace navi_sched {

// A Master cost projection, never a replacement for observed workload accounting
// or Agent training features. Unknown/incomplete observations return no projection.
class MasterKVReadyWorkload {
public:
    static std::optional<NaviSchedLatencyFeatures> project(
        const NaviSchedWorkloadSnapshot &workload,
        const std::vector<NaviSchedKVPreparation> &requests, bool enabled) {
        if (!workload.valid()) return std::nullopt;
        if (!enabled) {
            NaviSchedLatencyFeatures unchanged;
            if (!unchanged.appendPrefillWorkload(workload)) return std::nullopt;
            return unchanged;
        }
        if (!workload.prefillCapacitySplit ||
            requests.size() != static_cast<size_t>(workload.requestCount)) return std::nullopt;
        std::unordered_set<uint64_t> identities;
        std::unordered_set<int64_t> allBatches, readyBatches;
        identities.reserve(requests.size());
        allBatches.reserve(requests.size());
        readyBatches.reserve(requests.size());
        Totals allRunning, allWaiting, readyRunning, readyWaiting;
        int64_t readyRequests = 0;
        for (const auto &request : requests) {
            if (!request.valid() || !identities.insert(request.requestId).second) return std::nullopt;
            if (request.localBatchId > 0) allBatches.insert(request.localBatchId);
            auto &all = request.runningCapacity ? allRunning : allWaiting;
            if (request.computeTokens > 0 && !all.append(request)) return std::nullopt;
            if (!request.executable()) continue;
            ++readyRequests;
            if (request.localBatchId > 0) readyBatches.insert(request.localBatchId);
            auto &ready = request.runningCapacity ? readyRunning : readyWaiting;
            if (request.computeTokens > 0 && !ready.append(request)) return std::nullopt;
        }
        if (allBatches.size() != static_cast<size_t>(workload.batchCount) ||
            !allRunning.matches(workload.runningCapacityPrefillRequestCount,
                workload.runningCapacityPrefillContextTokenCount, workload.runningCapacityPrefillTokenCount,
                workload.runningCapacityPrefillTokenSquareSum,
                workload.runningCapacityContextRemainingPrefillTokenProductSum) ||
            !allWaiting.matches(workload.waitingCapacityPrefillRequestCount,
                workload.waitingCapacityPrefillContextTokenCount, workload.waitingCapacityPrefillTokenCount,
                workload.waitingCapacityPrefillTokenSquareSum,
                workload.waitingCapacityContextRemainingPrefillTokenProductSum)) return std::nullopt;
        NaviSchedLatencyFeatures features;
        if (!features.appendCapacitySplitAggregate(
                readyBatches.size(), readyRunning.count, readyRunning.context, readyRunning.compute,
                double(readyRunning.squares), double(readyRunning.products),
                readyWaiting.count, readyWaiting.context, readyWaiting.compute,
                double(readyWaiting.squares), double(readyWaiting.products)) ||
            !features.appendPrecedingRequestCount(readyRequests) ||
            !features.appendRunningElapsedTimeMs(workload.runningElapsedTimeMs) ||
            !features.appendRunningDecodeProgress(workload.runningDecodeRequestCount,
                workload.runningDecodeGeneratedTokenCount, workload.runningDecodeRemainingTokenBudget))
            return std::nullopt;
        return features;
    }
private:
    struct Totals {
        int64_t count = 0, context = 0, compute = 0;
        unsigned __int128 squares = 0, products = 0;
        bool append(const NaviSchedKVPreparation &request) {
            const auto maximum = std::numeric_limits<int64_t>::max();
            if (count == maximum || request.contextTokens > maximum - context ||
                request.computeTokens > maximum - compute) return false;
            ++count;
            context += request.contextTokens;
            compute += request.computeTokens;
            squares += static_cast<unsigned __int128>(request.computeTokens) * request.computeTokens;
            products += static_cast<unsigned __int128>(request.contextTokens) * request.computeTokens;
            return true;
        }
        bool matches(int64_t n, int64_t hit, int64_t miss, double sq, double prod) const {
            const auto same = [](double a, double b) {
                return std::isfinite(a) && std::isfinite(b) &&
                    std::abs(a - b) <= 1e-12 * std::max({1.0, std::abs(a), std::abs(b)});
            };
            return count == n && context == hit && compute == miss &&
                same(double(squares), sq) && same(double(products), prod);
        }
    };
};
}
