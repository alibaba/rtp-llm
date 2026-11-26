#include "aios/navi_sched/scheduler/TairKvcmQuery.h"
#include <algorithm>
#include <atomic>
#include <cstring>
namespace navi_sched {
namespace {
std::shared_ptr<std::atomic<size_t>> queryCounter()
{
    static auto count = std::make_shared<std::atomic<size_t>>(0);
    return count;
}
struct QueryToken {
    explicit QueryToken(std::shared_ptr<std::atomic<size_t>> count)
        : counter(std::move(count))
    {
        counter->fetch_add(1, std::memory_order_relaxed);
    }
    ~QueryToken() { counter->fetch_sub(1, std::memory_order_relaxed); }
    std::shared_ptr<std::atomic<size_t>> counter;
};
} // namespace

std::shared_ptr<void> trackTairKvcmQuery()
{
    return std::make_shared<QueryToken>(queryCounter());
}

size_t tairKvcmQueriesInFlight()
{
    return queryCounter()->load(std::memory_order_relaxed);
}

void prepareTairKvcmBatchQuery(
    const std::shared_ptr<const NaviSchedTairKvcmConfig> &config,
    const std::vector<NaviSchedScheduleItem> &requests, TairKvcmQueryInput &input,
    size_t maxObjects)
{
    input.policy = config;
    input.objects.clear();
    input.logicalGroups.clear();
    input.exceedsObjectBudget = false;
    if (!config || !config->configured()) return;
    // Layout validation belongs to the connection handshake, not this hot path.
    for (size_t g = 0; g < config->groups.size(); ++g) {
        const auto &group = config->groups[g];
        const auto scale = static_cast<size_t>(group.hashBlockScale);
        absl::flat_hash_set<uint64_t> queriedKeys;
        for (const auto &item : requests) {
            const auto &request = item.request;
            if (request.blockSize != config->hashBlockSize || !request.blockHashes ||
                request.promptTokenCount <= 0) continue;
            const auto &hashes = *request.blockHashes;
            const auto limit = std::min(hashes.size(),
                static_cast<size_t>(request.promptTokenCount / request.blockSize));
            for (size_t start = 0; start < limit && scale <= limit - start; start += scale) {
                const auto key = hashes[start + scale - 1];
                if (!queriedKeys.insert(key).second) continue;
                if (group.locationSpecNames.size() >
                    maxObjects - input.objects.size()) {
                    input.objects.clear();
                    input.logicalGroups.clear();
                    input.exceedsObjectBudget = true;
                    return;
                }
                TairKvcmLogicalGroupQuery logical;
                logical.groupIndex = g;
                logical.physicalBegin = input.objects.size();
                if (group.tail) logical.blockHashes.push_back(key);
                else logical.blockHashes.assign(hashes.begin() + start,
                                                 hashes.begin() + start + scale);
                int64_t signedKey;
                static_assert(sizeof(signedKey) == sizeof(key));
                std::memcpy(&signedKey, &key, sizeof(key));
                for (const auto &spec : group.locationSpecNames)
                    input.objects.push_back({signedKey, spec});
                input.logicalGroups.emplace_back(std::move(logical));
            }
        }
    }
}

NaviSchedExternalCacheReadyBlocks getTairKvcmReadyBlocks(const TairKvcmQueryInput &input,
                                         const TairKvcmQueryResult &result)
{
    NaviSchedExternalCacheReadyBlocks ready;
    if (!input.policy || result.status != TairKvcmQueryStatus::OK ||
        result.remoteHits.size() != input.objects.size()) return ready;
    const auto &groups = input.policy->groups;
    std::vector<absl::flat_hash_set<uint64_t>> groupHits(groups.size());
    for (const auto &logical : input.logicalGroups) {
        if (logical.groupIndex >= groups.size()) return {};
        const auto count = groups[logical.groupIndex].locationSpecNames.size();
        if (logical.physicalBegin > result.remoteHits.size() || count == 0 ||
            count > result.remoteHits.size() - logical.physicalBegin) return {};
        const auto begin = result.remoteHits.begin() + logical.physicalBegin;
        if (!std::all_of(begin, begin + count, [](uint8_t hit) { return hit != 0; }))
            continue;
        groupHits[logical.groupIndex].insert(logical.blockHashes.begin(),
                                              logical.blockHashes.end());
    }
    bool fullStarted = false, tailStarted = false;
    for (size_t g = 0; g < groups.size(); ++g) {
        auto &combined = groups[g].tail ? ready.tail : ready.full;
        auto &started = groups[g].tail ? tailStarted : fullStarted;
        if (!started) {
            combined = std::move(groupHits[g]);
            started = true;
        } else {
            // Full and Mamba are separate requirements. Every group of a given
            // kind must contain a block before the common cache matcher sees it.
            for (auto it = combined.begin(); it != combined.end();) {
                if (!groupHits[g].count(*it)) combined.erase(it++);
                else ++it;
            }
        }
    }
    return ready;
}

const char *tairKvcmQueryStatusName(TairKvcmQueryStatus status)
{
    switch (status) {
#define TAIR_STATUS(name) case TairKvcmQueryStatus::name: return #name
        TAIR_STATUS(NOT_CONFIGURED); TAIR_STATUS(EMPTY); TAIR_STATUS(OK);
        TAIR_STATUS(INVALID_REQUEST); TAIR_STATUS(WINDOW_EXPIRED); TAIR_STATUS(RPC_ERROR);
        TAIR_STATUS(INVALID_RESPONSE); TAIR_STATUS(BUSINESS_ERROR);
        TAIR_STATUS(RATE_LIMITED); TAIR_STATUS(IN_FLIGHT); TAIR_STATUS(BACKOFF);
        TAIR_STATUS(TOO_LARGE);
#undef TAIR_STATUS
    }
    return "INVALID_STATUS";
}

} // namespace navi_sched
