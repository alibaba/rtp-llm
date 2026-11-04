#include "rtp_llm/flexlb/navi_sched/resource/AgentWorkloadOwnership.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace navi_sched {

void AgentWorkloadOwnership::PrefillTotals::appendRequest(int64_t hitTokens, int64_t missTokens)
{
    ++count;
    hit += hitTokens;
    miss += missTokens;
    squares += static_cast<unsigned __int128>(missTokens) * missTokens;
    products += static_cast<unsigned __int128>(hitTokens) * missTokens;
}

void AgentWorkloadOwnership::PrefillTotals::append(const PrefillTotals &other)
{
    count += other.count;
    hit += other.hit;
    miss += other.miss;
    squares += other.squares;
    products += other.products;
}

bool AgentWorkloadOwnership::PrefillTotals::subtract(const PrefillTotals &other)
{
    if (count < other.count || hit < other.hit || miss < other.miss ||
        squares < other.squares || products < other.products) return false;
    count -= other.count;
    hit -= other.hit;
    miss -= other.miss;
    squares -= other.squares;
    products -= other.products;
    return true;
}

AgentWorkloadOwnership::PreparedBatch AgentWorkloadOwnership::prepareBatch(
    const std::vector<NaviSchedRequestStats> &requests)
{
    return prepareRequests(requests.data(), requests.size());
}

AgentWorkloadOwnership::PreparedBatch AgentWorkloadOwnership::prepareRequest(
    const NaviSchedRequestStats &request)
{
    return prepareRequests(&request, 1);
}

AgentWorkloadOwnership::PreparedBatch AgentWorkloadOwnership::prepareRequests(
    const NaviSchedRequestStats *requests, size_t count)
{
    if (count == 0) return nullptr;
    auto batch = std::make_shared<Batch>();
    batch->byRequestId.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto &request = requests[i];
        if (!request.valid() || batch->byRequestId.count(request.requestId) ||
            request.totalTokenCount > std::numeric_limits<int64_t>::max() - batch->promptTokens) {
            return nullptr;
        }
        const auto miss = request.totalTokenCount - request.cacheHitTokenCount;
        batch->requests.push_back({request.cacheHitTokenCount, miss});
        batch->byRequestId.emplace(request.requestId, std::prev(batch->requests.end()));
        batch->promptTokens += request.totalTokenCount;
        if (miss > 0) batch->prefillFeatures.appendRequest(request.cacheHitTokenCount, miss);
    }
    return batch;
}

bool AgentWorkloadOwnership::addBatch(
    int64_t localBatchId, const std::vector<NaviSchedRequestStats> &requests)
{
    return addBatch(localBatchId, prepareBatch(requests));
}

bool AgentWorkloadOwnership::addBatch(int64_t localBatchId, const PreparedBatch &batch)
{
    if (localBatchId <= 0 || !batch || batch->requests.empty() || batch->localBatchId != 0 ||
        _batches.count(localBatchId)) {
        return false;
    }
    size_t promotedCount = 0;
    int64_t promotedPromptTokens = 0;
    if (_unbatched) {
        for (const auto &entry : batch->byRequestId) {
            const auto found = _unbatched->byRequestId.find(entry.first);
            if (found == _unbatched->byRequestId.end()) continue;
            const auto &old = *found->second;
            const auto &ready = *entry.second;
            if (old.cacheHitTokens != ready.cacheHitTokens ||
                old.cacheMissTokens != ready.cacheMissTokens) return false;
            ++promotedCount;
            promotedPromptTokens += old.cacheHitTokens + old.cacheMissTokens;
        }
    }
    if (batch->requests.size() - promotedCount >
            static_cast<size_t>(std::numeric_limits<int32_t>::max()) - _requestCount ||
        batch->promptTokens - promotedPromptTokens >
            std::numeric_limits<int64_t>::max() - _promptTokens) return false;
    const auto inserted = _batches.emplace(localBatchId, batch);
    try {
        _batchOrder.emplace_back(batch);
    } catch (...) {
        _batches.erase(inserted.first);
        throw;
    }
    batch->localBatchId = localBatchId;
    batch->position = std::prev(_batchOrder.end());
    if (_unbatched) {
        for (const auto &entry : batch->byRequestId) {
            const auto found = _unbatched->byRequestId.find(entry.first);
            if (found != _unbatched->byRequestId.end()) eraseRequest(*_unbatched, found);
        }
        if (_unbatched->requests.empty()) {
            _batchOrder.erase(_unbatched->position);
            _unbatched.reset();
        }
    }
    _prefillFeatures.append(batch->prefillFeatures);
    _promptTokens += batch->promptTokens;
    _requestCount += batch->requests.size();
    return true;
}

bool AgentWorkloadOwnership::addUnbatchedRequest(uint64_t requestId, const PreparedBatch &request)
{
    if (!request || request->localBatchId != 0 || request->requests.size() != 1 ||
        request->byRequestId.count(requestId) != 1 ||
        (_unbatched && _unbatched->byRequestId.count(requestId)) ||
        _requestCount == static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        request->promptTokens > std::numeric_limits<int64_t>::max() - _promptTokens) return false;
    if (!_unbatched) {
        auto pending = std::make_shared<Batch>();
        _batchOrder.emplace_back(pending);
        pending->position = std::prev(_batchOrder.end());
        _unbatched = std::move(pending);
    }
    const auto position = request->requests.begin();
    _unbatched->byRequestId.emplace(requestId, position);
    _unbatched->requests.splice(_unbatched->requests.end(), request->requests);
    _unbatched->prefillFeatures.append(request->prefillFeatures);
    _unbatched->promptTokens += request->promptTokens;
    _prefillFeatures.append(request->prefillFeatures);
    _promptTokens += request->promptTokens;
    ++_requestCount;
    return true;
}

bool AgentWorkloadOwnership::removeUnbatchedRequests(const std::vector<uint64_t> &requestIds)
{
    if (!_unbatched) return true;
    for (auto id : requestIds) {
        const auto found = _unbatched->byRequestId.find(id);
        if (found != _unbatched->byRequestId.end() && !eraseRequest(*_unbatched, found)) return false;
    }
    if (_unbatched->requests.empty()) {
        _batchOrder.erase(_unbatched->position);
        _unbatched.reset();
    }
    return true;
}

bool AgentWorkloadOwnership::eraseRequest(
    Batch &batch, BatchRequests::iterator request)
{
    const auto contribution = request->second;
    if (contribution->cacheMissTokens > 0) {
        PrefillTotals removed;
        removed.appendRequest(contribution->cacheHitTokens, contribution->cacheMissTokens);
        auto global = _prefillFeatures;
        auto local = batch.prefillFeatures;
        if (!global.subtract(removed) || !local.subtract(removed)) {
            return false;
        }
        _prefillFeatures = global;
        batch.prefillFeatures = local;
    }
    const auto promptTokens = contribution->cacheHitTokens + contribution->cacheMissTokens;
    _promptTokens -= promptTokens;
    batch.promptTokens -= promptTokens;
    --_requestCount;
    batch.requests.erase(contribution);
    batch.byRequestId.erase(request);
    if (batch.prefillFeatures.getBatchSize() == 0) {
        batch.prefillFeatures = PrefillTotals();
    }
    if (_prefillFeatures.getBatchSize() == 0) {
        _prefillFeatures = PrefillTotals();
    }
    return true;
}

bool AgentWorkloadOwnership::removeRequests(
    int64_t localBatchId, const std::vector<uint64_t> &requestIds)
{
    const auto batch = _batches.find(localBatchId);
    if (batch == _batches.end()) {
        return true;
    }
    for (auto requestId : requestIds) {
        const auto request = batch->second->byRequestId.find(requestId);
        if (request != batch->second->byRequestId.end() && !eraseRequest(*batch->second, request)) {
            return false;
        }
    }
    if (batch->second->byRequestId.empty()) {
        _batchOrder.erase(batch->second->position);
        _batches.erase(batch);
    }
    return true;
}

bool AgentWorkloadOwnership::handoff(
    const std::vector<int64_t> &localBatchIds, RetiredBatches &retired)
{
    retired.reserve(retired.size() + localBatchIds.size());
    for (auto localBatchId : localBatchIds) {
        const auto batch = _batches.find(localBatchId);
        if (batch == _batches.end()) {
            continue;
        }
        const auto &owner = batch->second;
        if (!_prefillFeatures.subtract(owner->prefillFeatures)) {
            return false;
        }
        _requestCount -= owner->requests.size();
        _promptTokens -= owner->promptTokens;
        _batchOrder.erase(owner->position);
        retired.emplace_back(std::move(batch->second));
        _batches.erase(batch);
    }
    if (_prefillFeatures.getBatchSize() == 0) {
        _prefillFeatures = PrefillTotals();
    }
    return true;
}

bool AgentWorkloadOwnership::combine(
    const NaviSchedWorkloadSnapshot &observed, NaviSchedWorkloadSnapshot &combined,
    const std::vector<int64_t> &coveredBatchIds) const
{
    const std::unordered_set<int64_t> covered(coveredBatchIds.begin(), coveredBatchIds.end());
    auto prefillFeatures = _prefillFeatures;
    auto requestCount = _requestCount;
    auto batchCount = _batches.size();
    auto promptTokens = _promptTokens;
    for (auto id : covered) {
        const auto batch = _batches.find(id);
        if (batch == _batches.end()) continue;
        if (!prefillFeatures.subtract(batch->second->prefillFeatures)) return false;
        requestCount -= batch->second->requests.size();
        --batchCount;
        promptTokens -= batch->second->promptTokens;
    }
    const auto largestTokenCount = std::max({observed.totalPromptTokenCount,
        observed.remainingTotalContextTokenCount, observed.remainingTotalTokenCount,
        observed.remainingPrefillContextTokenCount, observed.remainingPrefillTokenCount,
        observed.runningCapacityPrefillContextTokenCount, observed.runningCapacityPrefillTokenCount,
        observed.waitingCapacityPrefillContextTokenCount, observed.waitingCapacityPrefillTokenCount});
    if (!observed.valid() || !observed.prefillCapacitySplit ||
        requestCount > static_cast<size_t>(std::numeric_limits<int32_t>::max() -
                                              observed.requestCount) ||
        batchCount > static_cast<size_t>(std::numeric_limits<int32_t>::max() -
                                             observed.batchCount) ||
        promptTokens > std::numeric_limits<int64_t>::max() - largestTokenCount ||
        requestCount > static_cast<size_t>(std::numeric_limits<int32_t>::max() - observed.waitingRequestCount)) {
        return false;
    }
    combined = observed;
    combined.batchCount += batchCount;
    combined.requestCount += requestCount;
    combined.waitingRequestCount += requestCount;
    combined.totalPromptTokenCount += promptTokens;
    PrefillTotals running;
    auto slots = std::max(0, observed.maxRunningRequestCount - observed.runningCapacitySlotCount);
    // Numeric totals are incremental; inspect only the available capacity slots.
    for (const auto &batch : _batchOrder) {
        if (batch->localBatchId > 0 && covered.count(batch->localBatchId)) continue;
        if (slots <= 0) {
            break;
        }
        for (const auto &request : batch->requests) {
            if (slots-- <= 0) {
                break;
            }
            ++combined.runningCapacitySlotCount;
            if (request.cacheMissTokens > 0) {
                running.appendRequest(request.cacheHitTokens, request.cacheMissTokens);
            }
        }
    }
    combined.runningCapacityPrefillRequestCount += running.getBatchSize();
    combined.runningCapacityPrefillContextTokenCount += running.getCacheHitTokenCount();
    combined.runningCapacityPrefillTokenCount += running.getCacheMissTokenCount();
    combined.runningCapacityPrefillTokenSquareSum += running.getCacheMissTokenSquareSum();
    combined.runningCapacityContextRemainingPrefillTokenProductSum +=
        running.getCacheHitMissTokenProductSum();
    auto waiting = prefillFeatures;
    if (!waiting.subtract(running)) return false;
    combined.waitingCapacityPrefillRequestCount += waiting.getBatchSize();
    combined.waitingCapacityPrefillContextTokenCount +=
        waiting.getCacheHitTokenCount();
    combined.waitingCapacityPrefillTokenCount +=
        waiting.getCacheMissTokenCount();
    combined.waitingCapacityPrefillTokenSquareSum +=
        waiting.getCacheMissTokenSquareSum();
    combined.waitingCapacityContextRemainingPrefillTokenProductSum +=
        waiting.getCacheHitMissTokenProductSum();
    combined.remainingPrefillRequestCount = combined.runningCapacityPrefillRequestCount +
                                           combined.waitingCapacityPrefillRequestCount;
    combined.remainingPrefillContextTokenCount = combined.runningCapacityPrefillContextTokenCount +
                                               combined.waitingCapacityPrefillContextTokenCount;
    combined.remainingPrefillTokenCount = combined.runningCapacityPrefillTokenCount +
                                        combined.waitingCapacityPrefillTokenCount;
    combined.remainingPrefillTokenSquareSum = combined.runningCapacityPrefillTokenSquareSum +
                                            combined.waitingCapacityPrefillTokenSquareSum;
    combined.contextRemainingPrefillTokenProductSum =
        combined.runningCapacityContextRemainingPrefillTokenProductSum +
        combined.waitingCapacityContextRemainingPrefillTokenProductSum;
    combined.remainingTotalContextTokenCount += prefillFeatures.getCacheHitTokenCount();
    combined.remainingTotalTokenCount += prefillFeatures.getCacheMissTokenCount();
    combined.remainingTotalTokenSquareSum += prefillFeatures.getCacheMissTokenSquareSum();
    combined.contextRemainingTotalTokenProductSum += prefillFeatures.getCacheHitMissTokenProductSum();
    return combined.valid();
}

} // namespace navi_sched
