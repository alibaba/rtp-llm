#include "rtp_llm/flexlb/navi_sched/resource/MasterPendingCacheR.h"
#include "rtp_llm/flexlb/navi_sched/resource/MasterBatchReservation.h"

#include "rtp_llm/flexlb/navi_sched/data/BlockHashUtil.h"
#include "rtp_llm/flexlb/navi_sched/debug/NaviSchedScheduleReplayJson.h"
#include "absl/container/flat_hash_set.h"
#include "autil/Scope.h"
#include "autil/legacy/jsonizable.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <time.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace navi_sched {
namespace {

constexpr uint64_t AGENT_ID_HASH = 1;
constexpr size_t REQUEST_COUNT = 64;
constexpr size_t BLOCK_COUNT = 128;
constexpr size_t MATCH_REPEAT_COUNT = 20000;

int64_t elapsedNs(const std::chrono::steady_clock::time_point &begin)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - begin)
        .count();
}

std::vector<NaviSchedScheduleItem> makeRequests(
    const NaviSchedBlockHashesConstPtr &blockHashes)
{
    std::vector<NaviSchedScheduleItem> requests(REQUEST_COUNT);
    for (size_t index = 0; index < requests.size(); ++index) {
        requests[index].request.requestId = index + 1;
        requests[index].request.promptTokenCount = BLOCK_COUNT;
        requests[index].request.blockSize = 1;
        requests[index].request.blockHashes = blockHashes;
    }
    return requests;
}

int64_t baselineMatch(
    const NaviSchedBlockHashes &blockHashes,
    const std::unordered_set<uint64_t> &fullReadyBlockHashes,
    const std::unordered_set<uint64_t> &tailReadyBlockHashes,
    size_t matchableBlockCount)
{
    int64_t matchedBlockCount = 0;
    const auto limit = std::min(blockHashes.size(), matchableBlockCount);
    for (size_t blockIndex = 0; blockIndex < limit; ++blockIndex) {
        const auto blockHash = blockHashes[blockIndex];
        if (fullReadyBlockHashes.end() == fullReadyBlockHashes.find(blockHash)) {
            break;
        }
        ++matchedBlockCount;
    }
    while (matchedBlockCount > 0 &&
           tailReadyBlockHashes.end() == tailReadyBlockHashes.find(
               blockHashes[static_cast<size_t>(matchedBlockCount - 1)])) {
        --matchedBlockCount;
    }
    return matchedBlockCount;
}

int64_t threadCpuNs()
{
    timespec value{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) {
        throw std::runtime_error("thread CPU clock failed");
    }
    return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
}

void benchmarkPendingStats(MasterPendingCacheR &cache, uint64_t agentId,
                           const std::string &name,
                           const autil::legacy::json::JsonMap &pending)
{
    using autil::legacy::FromJson;
    using autil::legacy::json::JsonMap;
    std::unordered_set<int64_t> batchIds;
    std::unordered_set<uint64_t> fullHashes, tailHashes;
    auto restore = [&](const char *key, PendingCacheState &output) {
        std::vector<JsonMap> records;
        FromJson(records, pending.at(key));
        std::vector<NaviSchedPendingCacheReservation> reservations;
        for (const auto &record : records) {
            NaviSchedPendingCacheReservation entry;
            FromJson(entry.batchId, record.at("batch_id"));
            FromJson(entry.scheduleBatchId, record.at("schedule_batch_id"));
            FromJson(entry.fullReadyBlockHashes, record.at("full_ready_block_hashes"));
            FromJson(entry.tailReadyBlockHashes, record.at("tail_ready_block_hashes"));
            batchIds.insert(entry.batchId);
            fullHashes.insert(entry.fullReadyBlockHashes.begin(),
                              entry.fullReadyBlockHashes.end());
            tailHashes.insert(entry.tailReadyBlockHashes.begin(),
                              entry.tailReadyBlockHashes.end());
            reservations.emplace_back(std::move(entry));
        }
        if (!output.replace(reservations)) {
            throw std::runtime_error("reference pending state restore failed");
        }
    };
    PendingCacheState scheduled, agent;
    restore("scheduled", scheduled);
    restore("agent", agent);
    const auto scheduledStats = scheduled.getStats();
    const auto agentStats = agent.getStats();
    auto check = [&](const auto &stats) {
        if (stats.batchCount != batchIds.size() ||
            stats.uniqueBlockCount != fullHashes.size() ||
            stats.uniqueTailReadyBlockCount != tailHashes.size()) {
            throw std::runtime_error("pending statistics disagree with independent union");
        }
        return stats.batchCount + stats.uniqueBlockCount + stats.uniqueTailReadyBlockCount;
    };
    // Reproduce the old query algorithm with current containers, outside production.
    auto scannedStats = [&] {
        PendingCacheStateStats stats;
        stats.batchCount = scheduledStats.batchCount + agentStats.batchCount -
                           scheduled.countBatchOverlap(agent);
        stats.uniqueBlockCount =
            scheduledStats.uniqueBlockCount + agentStats.uniqueBlockCount -
            scheduled.countFullReadyOverlap(agent);
        stats.uniqueTailReadyBlockCount = scheduledStats.uniqueTailReadyBlockCount +
            agentStats.uniqueTailReadyBlockCount - scheduled.countTailReadyOverlap(agent);
        return stats;
    };
    const auto expected = check(scannedStats());
    check(cache.getStats(agentId));
    constexpr size_t QUERIES_PER_ROUND = 30;
    constexpr size_t ROUNDS = 32;
    std::vector<int64_t> scanTimes, queryTimes;
    for (size_t trial = 0; trial < 5; ++trial) {
        uint64_t scanSum = 0, querySum = 0;
        auto scan = [&] {
            const auto begin = threadCpuNs();
            for (size_t i = 0; i < QUERIES_PER_ROUND * ROUNDS; ++i) {
                const auto stats = scannedStats();
                scanSum += stats.batchCount + stats.uniqueBlockCount +
                           stats.uniqueTailReadyBlockCount;
            }
            scanTimes.push_back(threadCpuNs() - begin);
        };
        auto query = [&] {
            const auto begin = threadCpuNs();
            for (size_t i = 0; i < QUERIES_PER_ROUND * ROUNDS; ++i) {
                const auto stats = cache.getStats(agentId);
                querySum += stats.batchCount + stats.uniqueBlockCount +
                            stats.uniqueTailReadyBlockCount;
            }
            queryTimes.push_back(threadCpuNs() - begin);
        };
        if (trial % 2 == 0) {
            scan();
            query();
        } else {
            query();
            scan();
        }
        if (scanSum != expected * QUERIES_PER_ROUND * ROUNDS || querySum != scanSum) {
            throw std::runtime_error("timed pending statistics mismatch");
        }
    }
    std::sort(scanTimes.begin(), scanTimes.end());
    std::sort(queryTimes.begin(), queryTimes.end());
    std::cout << "pending_stats_node=" << name
              << " scheduled_batches=" << scheduledStats.batchCount
              << " agent_batches=" << agentStats.batchCount
              << " scheduled_blocks=" << scheduledStats.uniqueBlockCount
              << " agent_blocks=" << agentStats.uniqueBlockCount
              << " union_blocks=" << fullHashes.size()
              << " scanned_full_blocks_per_query="
              << std::min(scheduledStats.uniqueBlockCount, agentStats.uniqueBlockCount)
              << " scanned_tail_blocks_per_query="
              << std::min(scheduledStats.uniqueTailReadyBlockCount,
                          agentStats.uniqueTailReadyBlockCount)
              << " scanned_batch_ids_per_query="
              << std::min(scheduledStats.batchCount, agentStats.batchCount)
              << " queries_per_round=" << QUERIES_PER_ROUND
              << " scan_cpu_ns_per_round=" << scanTimes[2] / ROUNDS
              << " current_cpu_ns_per_round=" << queryTimes[2] / ROUNDS << std::endl;
}

template <typename HashSet>
void matchPhysicalBlocks(const std::vector<NaviSchedScheduleItem> &requests,
                         int64_t blockSize, const HashSet &cached,
                         const HashSet &full, const HashSet &tail, bool hybrid,
                         std::vector<int64_t> &output)
{
    const NaviSchedBlockHashes *previous = nullptr;
    size_t previousFullCount = 0;
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto &request = requests[i].request;
        if (blockSize <= 0 || request.blockSize != blockSize) {
            output[i] = 0;
            previous = nullptr;
            previousFullCount = 0;
            continue;
        }
        const auto &hashes = BlockHashUtil::get(request.blockHashes);
        const auto limit = std::min(hashes.size(), BlockHashUtil::getMatchableBlockCount(
            request.promptTokenCount, request.blockSize, request.blockHashes));
        size_t count = 0;
        if (previous) {
            const auto sharedLimit = std::min(previousFullCount, limit);
            if (previous == &hashes) {
                count = sharedLimit;
            } else {
                while (count < sharedLimit && (*previous)[count] == hashes[count]) ++count;
            }
        }
        const auto &prefix = hybrid ? full : cached;
        while (count < limit && prefix.find(hashes[count]) != prefix.end()) ++count;
        previous = &hashes;
        previousFullCount = count;
        if (hybrid) {
            while (count > 0 && tail.find(hashes[count - 1]) == tail.end()) --count;
        }
        output[i] = count;
    }
}

std::pair<int64_t, int64_t> benchmarkPhysicalIndex(
    const std::vector<NaviSchedScheduleItem> &requests, int64_t blockSize,
    const MasterPendingCacheR::PhysicalBlockSet &cached,
    const MasterPendingCacheR::PhysicalBlockSet &full,
    const MasterPendingCacheR::PhysicalBlockSet &tail, bool hybrid,
    const std::vector<int64_t> &expected, bool churn)
{
    std::unordered_set<uint64_t> nodeCached(cached.begin(), cached.end());
    std::unordered_set<uint64_t> nodeFull(full.begin(), full.end());
    std::unordered_set<uint64_t> nodeTail(tail.begin(), tail.end());
    absl::flat_hash_set<uint64_t> flatCached(cached.begin(), cached.end());
    absl::flat_hash_set<uint64_t> flatFull(full.begin(), full.end());
    absl::flat_hash_set<uint64_t> flatTail(tail.begin(), tail.end());
    std::vector<int64_t> output(requests.size());
    if (churn) {
        struct Changes {
            std::vector<uint64_t> removed;
            std::vector<uint64_t> added;
        };
        const auto makeChanges = [](const auto &original, size_t epoch) {
            Changes changes;
            const std::vector<uint64_t> keys(original.begin(), original.end());
            const auto count = std::min<size_t>(64, keys.size());
            std::unordered_set<uint64_t> added;
            for (size_t i = 0; i < count; ++i) {
                const auto key = keys[(epoch * 64 + i) % keys.size()];
                auto temporary = key ^ ((epoch + 1) * 0x9e3779b97f4a7c15ULL);
                while (original.count(temporary) || !added.insert(temporary).second) ++temporary;
                changes.removed.push_back(key);
                changes.added.push_back(temporary);
            }
            return changes;
        };
        const auto apply = [](auto &set, const Changes &changes, bool restore) {
            const auto &removed = restore ? changes.added : changes.removed;
            const auto &added = restore ? changes.removed : changes.added;
            for (auto key : removed) {
                if (set.erase(key) != 1) throw std::runtime_error("churn erased missing key");
            }
            for (auto key : added) {
                if (!set.insert(key).second) throw std::runtime_error("churn inserted duplicate key");
            }
        };
        int64_t nodeUpdateNs = 0, flatUpdateNs = 0;
        uint64_t updates = 0;
        std::vector<int64_t> reference(requests.size());
        constexpr size_t EPOCHS = 128;
        for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
            const auto c = makeChanges(cached, epoch);
            const auto f = makeChanges(full, epoch);
            const auto t = makeChanges(tail, epoch);
            for (const bool restore : {false, true}) {
                const auto update = [&](auto &cSet, auto &fSet, auto &tSet, int64_t &time) {
                    const auto begin = threadCpuNs();
                    apply(cSet, c, restore);
                    apply(fSet, f, restore);
                    apply(tSet, t, restore);
                    time += threadCpuNs() - begin;
                };
                if (epoch % 2 == 0) {
                    update(nodeCached, nodeFull, nodeTail, nodeUpdateNs);
                    update(flatCached, flatFull, flatTail, flatUpdateNs);
                } else {
                    update(flatCached, flatFull, flatTail, flatUpdateNs);
                    update(nodeCached, nodeFull, nodeTail, nodeUpdateNs);
                }
                matchPhysicalBlocks(requests, blockSize, nodeCached, nodeFull,
                                    nodeTail, hybrid, reference);
                matchPhysicalBlocks(requests, blockSize, flatCached, flatFull,
                                    flatTail, hybrid, output);
                if (reference != output || (restore && output != expected)) {
                    throw std::runtime_error("churn query mismatch");
                }
                updates += 2 * (c.removed.size() + f.removed.size() + t.removed.size());
            }
        }
        const auto verify = [](const auto &original, const auto &actual) {
            if (original.size() != actual.size() ||
                !std::all_of(original.begin(), original.end(),
                             [&](uint64_t key) { return actual.count(key) == 1; })) {
                throw std::runtime_error("churn final membership mismatch");
            }
        };
        verify(cached, nodeCached);
        verify(full, nodeFull);
        verify(tail, nodeTail);
        verify(cached, flatCached);
        verify(full, flatFull);
        verify(tail, flatTail);
        std::cout << "physical_index_churn_epochs=" << EPOCHS
                  << " updates=" << updates << " query_cuts=" << 2 * EPOCHS
                  << " node_update_cpu_ns=" << nodeUpdateNs
                  << " flat_update_cpu_ns=" << flatUpdateNs << std::endl;
    }
    std::vector<int64_t> nodeTimes, flatTimes;
    constexpr size_t REPEATS = 50;
    auto measure = [&](const auto &c, const auto &f, const auto &t,
                       std::vector<int64_t> &times) {
        matchPhysicalBlocks(requests, blockSize, c, f, t, hybrid, output);
        if (output != expected) throw std::runtime_error("physical index match mismatch");
        const auto begin = threadCpuNs();
        for (size_t repeat = 0; repeat < REPEATS; ++repeat) {
            // Keep repeated immutable lookups inside the measured region.
            asm volatile("" ::: "memory");
            matchPhysicalBlocks(requests, blockSize, c, f, t, hybrid, output);
        }
        times.push_back(threadCpuNs() - begin);
        if (output != expected) throw std::runtime_error("timed physical index match mismatch");
    };
    for (size_t trial = 0; trial < 5; ++trial) {
        if (trial % 2 == 0) {
            measure(nodeCached, nodeFull, nodeTail, nodeTimes);
            measure(flatCached, flatFull, flatTail, flatTimes);
        } else {
            measure(flatCached, flatFull, flatTail, flatTimes);
            measure(nodeCached, nodeFull, nodeTail, nodeTimes);
        }
    }
    std::sort(nodeTimes.begin(), nodeTimes.end());
    std::sort(flatTimes.begin(), flatTimes.end());
    return {nodeTimes[2] / REPEATS, flatTimes[2] / REPEATS};
}

void benchmarkPendingExclusions(const autil::legacy::json::JsonMap &pending,
                                const NaviSchedScheduleReplay &replay,
                                const std::string &name, MasterPendingCacheR &cache,
                                uint64_t agentId, int64_t blockSize, bool hybrid,
                                const MasterPendingCacheR::PhysicalBlockSet &cached,
                                const MasterPendingCacheR::PhysicalBlockSet &full,
                                const MasterPendingCacheR::PhysicalBlockSet &tail)
{
    using autil::legacy::FromJson;
    using autil::legacy::json::JsonMap;
    struct Baseline {
        std::unordered_map<int64_t, NaviSchedPendingCacheReservation> reservations;
        std::unordered_map<uint64_t, size_t> fullRefs, tailRefs;

        // Algorithm from 648100e5: every positive ref lookup scans every batch.
        bool contains(uint64_t hash, int64_t excluded, bool tail) const
        {
            const auto &refs = tail ? tailRefs : fullRefs;
            const auto ref = refs.find(hash);
            if (ref == refs.end()) return false;
            size_t excludedCount = 0;
            for (const auto &[id, reservation] : reservations) {
                (void)id;
                const auto &hashes = tail ? reservation.tailReadyBlockHashes
                                          : reservation.fullReadyBlockHashes;
                if (reservation.scheduleBatchId == excluded &&
                    std::binary_search(hashes.begin(), hashes.end(), hash)) {
                    ++excludedCount;
                }
            }
            return ref->second > excludedCount;
        }
    };
    std::vector<Baseline> baselines(2);
    std::vector<PendingCacheState> indexedStates(2);
    size_t sourceIndex = 0;
    int64_t excludedSchedule = 0;
    uint64_t totalScanNs = 0, totalIndexNs = 0, totalVisits = 0, totalProbes = 0;
    for (const char *key : {"scheduled", "agent"}) {
        auto &baseline = baselines[sourceIndex++];
        std::vector<JsonMap> records;
        FromJson(records, pending.at(key));
        if (records.empty()) continue;
        std::vector<NaviSchedPendingCacheReservation> reservations;
        for (const auto &record : records) {
            NaviSchedPendingCacheReservation entry;
            FromJson(entry.batchId, record.at("batch_id"));
            FromJson(entry.scheduleBatchId, record.at("schedule_batch_id"));
            FromJson(entry.fullReadyBlockHashes, record.at("full_ready_block_hashes"));
            FromJson(entry.tailReadyBlockHashes, record.at("tail_ready_block_hashes"));
            for (auto hash : entry.fullReadyBlockHashes) ++baseline.fullRefs[hash];
            for (auto hash : entry.tailReadyBlockHashes) ++baseline.tailRefs[hash];
            if (!baseline.reservations.emplace(entry.batchId, entry).second) {
                throw std::runtime_error("duplicate replay pending batch");
            }
            reservations.emplace_back(std::move(entry));
        }
        if (excludedSchedule == 0) excludedSchedule = reservations.front().scheduleBatchId;
        auto &indexed = indexedStates[sourceIndex - 1];
        if (!indexed.replace(reservations)) {
            throw std::runtime_error("exclusion pending state restore failed");
        }
        struct Probe { uint64_t hash; int64_t excluded; bool tail; };
        std::vector<Probe> probes;
        const auto samples = std::min<size_t>(32, replay.requests.size());
        for (size_t i = 0; i < samples; ++i) {
            const auto &request = replay.requests[i * replay.requests.size() / samples].request;
            const auto &hashes = BlockHashUtil::get(request.blockHashes);
            for (auto hash : hashes) {
                for (auto excluded : {reservations.front().scheduleBatchId, int64_t(-1)}) {
                    probes.push_back({hash, excluded, false});
                }
            }
        }
        // Include actual pending ownership even when sampled incoming prefixes miss.
        for (size_t i = 0; i < std::min<size_t>(32, reservations.size()); ++i) {
            const auto &reservation = reservations[i];
            if (!reservation.fullReadyBlockHashes.empty()) {
                probes.push_back({reservation.fullReadyBlockHashes.front(),
                                  reservation.scheduleBatchId, false});
            }
            if (!reservation.tailReadyBlockHashes.empty()) {
                probes.push_back({reservation.tailReadyBlockHashes.back(),
                                  reservation.scheduleBatchId, true});
            }
        }
        uint64_t expectedSum = 0, visits = 0;
        for (const auto &probe : probes) {
            const bool match = baseline.contains(probe.hash, probe.excluded, probe.tail);
            const bool actual = probe.tail
                                    ? indexed.containsTailReadyExcludingSchedule(probe.hash, probe.excluded)
                                    : indexed.containsFullReadyExcludingSchedule(probe.hash, probe.excluded);
            if (actual != match) throw std::runtime_error("exclusion baseline/index mismatch");
            expectedSum += match;
            const auto &refs = probe.tail ? baseline.tailRefs : baseline.fullRefs;
            if (refs.count(probe.hash)) visits += reservations.size();
        }
        std::vector<int64_t> scanTimes, indexTimes;
        auto measure = [&](bool scan) {
            asm volatile("" ::: "memory");
            const auto begin = threadCpuNs();
            uint64_t sum = 0;
            for (const auto &probe : probes) {
                sum += scan ? baseline.contains(probe.hash, probe.excluded, probe.tail)
                            : probe.tail
                                  ? indexed.containsTailReadyExcludingSchedule(probe.hash, probe.excluded)
                                  : indexed.containsFullReadyExcludingSchedule(probe.hash, probe.excluded);
            }
            const auto elapsed = threadCpuNs() - begin;
            if (sum != expectedSum) throw std::runtime_error("timed exclusion checksum mismatch");
            (scan ? scanTimes : indexTimes).push_back(elapsed);
        };
        for (size_t trial = 0; trial < 5; ++trial) {
            measure(trial % 2 == 0);
            measure(trial % 2 != 0);
        }
        std::sort(scanTimes.begin(), scanTimes.end());
        std::sort(indexTimes.begin(), indexTimes.end());
        totalScanNs += scanTimes[2];
        totalIndexNs += indexTimes[2];
        totalVisits += visits;
        totalProbes += probes.size();
        std::cout << "exclusion_node=" << name << " source=" << key
                  << " batches=" << reservations.size() << " probes=" << probes.size()
                  << " matches=" << expectedSum << " baseline_batch_visits=" << visits
                  << " scan_cpu_ns=" << scanTimes[2] << " index_cpu_ns=" << indexTimes[2]
                  << std::endl;
    }
    std::cout << "exclusion_total_node=" << name << " probes=" << totalProbes
              << " baseline_batch_visits=" << totalVisits << " scan_cpu_ns=" << totalScanNs
              << " index_cpu_ns=" << totalIndexNs << std::endl;
    if (excludedSchedule == 0) return;
    const auto oldMatch = [&](const NaviSchedScheduleRequest &request) {
        if (blockSize <= 0 || request.blockSize != blockSize) return int64_t(0);
        const auto &hashes = BlockHashUtil::get(request.blockHashes);
        const auto limit = BlockHashUtil::getMatchableBlockCount(
            request.promptTokenCount, request.blockSize, request.blockHashes);
        const auto &prefix = hybrid ? full : cached;
        size_t matched = 0;
        while (matched < limit &&
               (prefix.count(hashes[matched]) ||
                baselines[0].contains(hashes[matched], excludedSchedule, false) ||
                baselines[1].contains(hashes[matched], excludedSchedule, false))) ++matched;
        while (hybrid && matched > 0 && !tail.count(hashes[matched - 1]) &&
               !baselines[0].contains(hashes[matched - 1], excludedSchedule, true) &&
               !baselines[1].contains(hashes[matched - 1], excludedSchedule, true)) --matched;
        return static_cast<int64_t>(matched);
    };
    const auto indexedMatch = [&](const NaviSchedScheduleRequest &request) {
        if (blockSize <= 0 || request.blockSize != blockSize) return int64_t(0);
        const auto &hashes = BlockHashUtil::get(request.blockHashes);
        const auto limit = BlockHashUtil::getMatchableBlockCount(
            request.promptTokenCount, request.blockSize, request.blockHashes);
        const auto &prefix = hybrid ? full : cached;
        size_t matched = 0;
        while (matched < limit &&
               (prefix.count(hashes[matched]) ||
                indexedStates[0].containsFullReadyExcludingSchedule(hashes[matched], excludedSchedule) ||
                indexedStates[1].containsFullReadyExcludingSchedule(hashes[matched], excludedSchedule))) ++matched;
        while (hybrid && matched > 0 && !tail.count(hashes[matched - 1]) &&
               !indexedStates[0].containsTailReadyExcludingSchedule(hashes[matched - 1], excludedSchedule) &&
               !indexedStates[1].containsTailReadyExcludingSchedule(hashes[matched - 1], excludedSchedule)) --matched;
        return static_cast<int64_t>(matched);
    };
    uint64_t expectedSum = 0;
    for (const auto &item : replay.requests) {
        const auto expected = oldMatch(item.request);
        const auto &request = item.request;
        const auto nativeMatch = blockSize <= 0 || request.blockSize != blockSize ? int64_t(0)
            : cache.getMatchedBlockCountExcludingSchedule(
                  agentId, excludedSchedule, BlockHashUtil::get(request.blockHashes),
                  cached, full, tail, hybrid, BlockHashUtil::getMatchableBlockCount(
                      request.promptTokenCount, request.blockSize, request.blockHashes));
        if (expected != indexedMatch(request) || expected != nativeMatch) {
            throw std::runtime_error("full exclusion KV match mismatch");
        }
        expectedSum += expected;
    }
    std::vector<int64_t> scanTimes, indexTimes;
    const auto measure = [&](bool scan) {
        asm volatile("" ::: "memory");
        const auto begin = threadCpuNs();
        uint64_t sum = 0;
        for (const auto &item : replay.requests) {
            sum += scan ? oldMatch(item.request) : indexedMatch(item.request);
        }
        const auto elapsed = threadCpuNs() - begin;
        if (sum != expectedSum) throw std::runtime_error("timed full KV checksum mismatch");
        (scan ? scanTimes : indexTimes).push_back(elapsed);
    };
    for (size_t trial = 0; trial < 5; ++trial) {
        measure(trial % 2 == 0);
        measure(trial % 2 != 0);
    }
    std::sort(scanTimes.begin(), scanTimes.end());
    std::sort(indexTimes.begin(), indexTimes.end());
    std::cout << "exclusion_kv_node=" << name << " requests=" << replay.requests.size()
              << " matched_blocks=" << expectedSum << " scan_cpu_ns=" << scanTimes[2]
              << " index_cpu_ns=" << indexTimes[2] << std::endl;
}

void benchmarkSnapshotContention(
    MasterPendingCacheR &cache, uint64_t agentId, const std::string &name,
    const NaviSchedScheduleReplay &replay, int64_t blockSize, bool hybrid,
    const MasterPendingCacheR::PhysicalBlockSet &cached,
    const MasterPendingCacheR::PhysicalBlockSet &full,
    const MasterPendingCacheR::PhysicalBlockSet &tail)
{
    const auto snapshot = cache.captureReplaySnapshot(agentId);
    const auto expectedStats = cache.getStats(agentId);
    if (snapshot.agent.empty()) {
        throw std::runtime_error("snapshot contention needs Agent pending entries");
    }
    AgentPendingCacheToMasterData update;
    update.fullSnapshot = true;
    update.revision = snapshot.agentRevision;
    update.addedReservations = snapshot.agent;
    update.batchCount = snapshot.agentStats.batchCount;
    update.uniqueBlockCount = snapshot.agentStats.uniqueBlockCount;
    update.uniqueTailReadyBlockCount = snapshot.agentStats.uniqueTailReadyBlockCount;
    update.checksum = snapshot.agentStats.checksum;
    std::vector<int64_t> expected(replay.requests.size());
    cache.getMatchedBlockCounts(agentId, replay.requests, blockSize,
                                cached, full, tail, hybrid, expected.data());
    constexpr size_t UPDATES = 8;
    constexpr size_t MAX_SAMPLES = 100000;
    std::atomic<bool> started{false}, done{false};
    std::vector<int64_t> waits;
    waits.reserve(MAX_SAMPLES);
    size_t queries = 0;
    int64_t maxWaitUs = 0, totalWaitUs = 0;
    std::exception_ptr readerError, writerError;
    std::thread reader([&]() {
        try {
            std::vector<int64_t> actual(expected.size());
            while (!done.load(std::memory_order_acquire)) {
                MasterPendingCacheR::BatchMatchTiming timing;
                cache.getMatchedBlockCounts(agentId, replay.requests, blockSize,
                    cached, full, tail, hybrid, actual.data(), &timing);
                if (actual != expected) {
                    throw std::runtime_error("snapshot update changed KV matching");
                }
                ++queries;
                maxWaitUs = std::max(maxWaitUs, timing.lockWaitUs);
                totalWaitUs += timing.lockWaitUs;
                if (waits.size() < MAX_SAMPLES) waits.push_back(timing.lockWaitUs);
                started.store(true, std::memory_order_release);
                std::this_thread::yield();
            }
        } catch (...) {
            readerError = std::current_exception();
            started.store(true, std::memory_order_release);
        }
    });
    autil::ScopeGuard stopReader([&]() {
        done.store(true, std::memory_order_release);
        if (reader.joinable()) reader.join();
    });
    while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
    const auto begin = std::chrono::steady_clock::now();
    const auto beginCpu = threadCpuNs();
    try {
        for (size_t index = 0; index < UPDATES; ++index) {
            ++update.revision;
            const auto applied = cache.applyAgentUpdate(agentId, update);
            if (!applied.applied || applied.staleSnapshotIgnored ||
                applied.stats.agentRevision != update.revision ||
                applied.stats.agentChecksum != update.checksum ||
                applied.stats.agentBatchCount != snapshot.agentStats.batchCount ||
                applied.stats.batchCount != expectedStats.batchCount ||
                applied.stats.uniqueBlockCount != expectedStats.uniqueBlockCount ||
                applied.stats.uniqueTailReadyBlockCount != expectedStats.uniqueTailReadyBlockCount ||
                applied.stats.scheduledBatchCount != expectedStats.scheduledBatchCount ||
                !applied.stats.agentSnapshotReady || applied.stateMismatch || applied.revisionGap) {
                throw std::runtime_error("snapshot update state mismatch");
            }
        }
    } catch (...) {
        writerError = std::current_exception();
    }
    const auto writerCpuNs = threadCpuNs() - beginCpu;
    const auto writerWallNs = elapsedNs(begin);
    done.store(true, std::memory_order_release);
    reader.join();
    if (writerError) std::rethrow_exception(writerError);
    if (readerError) std::rethrow_exception(readerError);
    if (waits.empty()) throw std::runtime_error("snapshot contention reader did not run");
    std::sort(waits.begin(), waits.end());
    const auto stats = cache.getStats(agentId);
    std::cout << "snapshot_contention_node=" << name
              << " updates=" << UPDATES << " requests_per_query=" << expected.size()
              << " agent_batches=" << snapshot.agentStats.batchCount
              << " agent_blocks=" << snapshot.agentStats.uniqueBlockCount
              << " scheduled_batches=" << snapshot.scheduledStats.batchCount
              << " scheduled_blocks=" << snapshot.scheduledStats.uniqueBlockCount
              << " union_blocks=" << stats.uniqueBlockCount
              << " writer_wall_ns=" << writerWallNs << " writer_cpu_ns=" << writerCpuNs
              << " queries=" << queries << " sampled_queries=" << waits.size()
              << " sample_truncated=" << (queries > MAX_SAMPLES)
              << " reader_wait_p50_us=" << waits[(waits.size()-1)/2]
              << " reader_wait_p99_us=" << waits[(waits.size()-1)*99/100]
              << " reader_wait_max_us=" << maxWaitUs
              << " reader_wait_sum_us=" << totalWaitUs << std::endl;
}

void benchmarkEmptyWaitingSnapshot(
    const std::vector<autil::legacy::json::JsonMap> &states)
{
    std::vector<std::unique_ptr<MasterBatchReservation>> reservations;
    size_t requestCount = 0;
    for (const auto &state : states) {
        std::vector<autil::legacy::json::JsonMap> records;
        autil::legacy::FromJson(records, state.at("reservations"));
        for (const auto &record : records) {
            auto reservation = std::make_unique<MasterBatchReservation>();
            if (!reservation->restoreReplayState(record) ||
                reservation->getWaitingRequestCount() != 0) {
                throw std::runtime_error("expected valid reservation without waiting requests");
            }
            requestCount += reservation->getRequestCount();
            reservations.emplace_back(std::move(reservation));
        }
    }
    if (reservations.empty() || requestCount == 0) {
        throw std::runtime_error("empty reservation snapshot benchmark");
    }
    constexpr size_t REPEATS = 1000;
    std::vector<int64_t> samples;
    for (size_t trial = 0; trial < 7; ++trial) {
        const auto begin = threadCpuNs();
        for (size_t repeat = 0; repeat < REPEATS; ++repeat) {
            NaviSchedLatencyFeatures workload, reserved;
            int32_t workloadSlots = 64, reservedSlots = 0;
            for (const auto &reservation : reservations) {
                if (!reservation->appendWaitingCapacityFeatures(workload, 128, workloadSlots) ||
                    !reservation->appendWaitingCapacityFeatures(reserved, 128, reservedSlots)) {
                    throw std::runtime_error("empty waiting capacity append failed");
                }
            }
            if (workloadSlots != 64 || reservedSlots != 0 ||
                workload.getBatchSize() != 0 || reserved.getBatchSize() != 0 ||
                workload.getBatchCount() != 0 || reserved.getBatchCount() != 0 ||
                workload.usesCapacitySplit() || reserved.usesCapacitySplit()) {
                throw std::runtime_error("empty waiting capacity append changed snapshot");
            }
        }
        samples.push_back((threadCpuNs() - begin) / REPEATS);
    }
    std::sort(samples.begin(), samples.end());
    std::cout << "empty_waiting_snapshot nodes=" << states.size()
              << " reservations=" << reservations.size() << " requests=" << requestCount
              << " appends_per_round=" << 2 * reservations.size()
              << " repeats_per_trial=" << REPEATS << " trials=" << samples.size()
              << " min_round_cpu_ns=" << samples.front()
              << " median_round_cpu_ns=" << samples[samples.size() / 2]
              << " max_round_cpu_ns=" << samples.back() << std::endl;
}

int benchmarkReplay(const std::string &path, bool statsOnly = false,
                    bool physicalIndexOnly = false, bool churn = false,
                    bool exclusionOnly = false, bool snapshotContentionOnly = false,
                    bool emptyWaitingOnly = false, size_t contentionNodeIndex = 0)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() > 768 * 1024 * 1024) {
        throw std::runtime_error("replay missing or larger than 768 MiB");
    }
    input.seekg(0);
    const std::string json{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
    NaviSchedScheduleReplay replay;
    NaviSchedScheduleReplayJson::fromJsonString(json, replay);
    using autil::legacy::json::JsonMap;
    std::vector<JsonMap> states;
    autil::legacy::FromJson(states, replay.schedulerInfo.at("pending_state"));
    if (states.size() != replay.nodes.size() || replay.requests.empty()) {
        throw std::runtime_error("incomplete pending replay");
    }
    if (snapshotContentionOnly && contentionNodeIndex >= states.size()) {
        throw std::runtime_error("snapshot contention node index out of range");
    }
    if (emptyWaitingOnly) {
        benchmarkEmptyWaitingSnapshot(states);
        return 0;
    }
    const auto value = [](const JsonMap &state, const std::string &key, auto &out) {
        autil::legacy::FromJson(out, state.at(key));
    };
    constexpr size_t REPEATS = 50;
    uint64_t recordedMismatches = 0;
    uint64_t totalBlocks = 0;
    for (const auto &item : replay.requests) totalBlocks += BlockHashUtil::size(item.request.blockHashes);
    int64_t totalScalarNs = 0, totalBatchNs = 0, totalInstrumentedNs = 0;
    int64_t totalNodeIndexNs = 0, totalFlatIndexNs = 0;
    size_t physicalIndexNodes = 0;
    for (size_t nodeIndex = 0; nodeIndex < states.size(); ++nodeIndex) {
        if (snapshotContentionOnly && nodeIndex != contentionNodeIndex) continue;
        const auto &state = states[nodeIndex];
        uint64_t agentId = 0;
        int64_t blockSize = 0;
        bool hybrid = false;
        value(state, "agent_id_hash", agentId);
        value(state, "block_size", blockSize);
        value(state, "hybrid_tail_required", hybrid);
        auto restoreSet = [&](const std::string &key) {
            std::vector<uint64_t> hashes;
            value(state, key, hashes);
            return MasterPendingCacheR::PhysicalBlockSet(hashes.begin(), hashes.end());
        };
        const auto cached = restoreSet("cached_block_hashes");
        const auto full = restoreSet("full_ready_block_hashes");
        const auto tail = restoreSet("tail_ready_block_hashes");
        JsonMap pending;
        value(state, "pending_cache", pending);
        MasterPendingCacheR cache;
        if (!cache.restoreReplayState(agentId, pending)) {
            throw std::runtime_error("pending state restore failed");
        }
        if (snapshotContentionOnly) {
            benchmarkSnapshotContention(cache, agentId, replay.nodes[nodeIndex].agentId,
                                        replay, blockSize, hybrid, cached, full, tail);
            continue;
        }
        if (exclusionOnly) {
            benchmarkPendingExclusions(pending, replay, replay.nodes[nodeIndex].agentId,
                                       cache, agentId, blockSize, hybrid, cached, full, tail);
            continue;
        }
        if (statsOnly) {
            benchmarkPendingStats(cache, agentId, replay.nodes[nodeIndex].agentId, pending);
            continue;
        }
        std::vector<int64_t> expected(replay.requests.size()), actual(replay.requests.size());
        auto scalar = [&] {
            for (size_t i = 0; i < replay.requests.size(); ++i) {
                const auto &request = replay.requests[i].request;
                expected[i] = blockSize <= 0 || request.blockSize != blockSize ? 0
                    : cache.getMatchedBlockCount(agentId, BlockHashUtil::get(request.blockHashes),
                        cached, full, tail, hybrid, BlockHashUtil::getMatchableBlockCount(
                            request.promptTokenCount, request.blockSize, request.blockHashes));
            }
        };
        scalar();
        cache.getMatchedBlockCounts(agentId, replay.requests, blockSize,
                                    cached, full, tail, hybrid, actual.data());
        if (expected != actual) throw std::runtime_error("scalar/batch match mismatch");
        if (physicalIndexOnly) {
            if (cache.getStats(agentId).batchCount != 0) continue;
            const auto times = benchmarkPhysicalIndex(
                replay.requests, blockSize, cached, full, tail, hybrid, expected, churn);
            ++physicalIndexNodes;
            totalNodeIndexNs += times.first;
            totalFlatIndexNs += times.second;
            std::cout << "physical_index_node=" << replay.nodes[nodeIndex].agentId
                      << " requests=" << replay.requests.size()
                      << " physical_blocks=" << cached.size() << " hybrid=" << hybrid
                      << " node_cpu_ns=" << times.first
                      << " flat_cpu_ns=" << times.second << std::endl;
            continue;
        }
        uint64_t checksum = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            checksum += actual[i];
            if (i >= replay.nodeMatches.size() || nodeIndex >= replay.nodeMatches[i].size()) {
                throw std::runtime_error("incomplete recorded match matrix");
            }
            const auto recorded = replay.nodeMatches[i][nodeIndex].matchedBlockCount;
            if (actual[i] != recorded) {
                if (recordedMismatches < 32) {
                    std::cout << "match_mismatch node_index=" << nodeIndex
                              << " agent=" << replay.nodes[nodeIndex].agentId
                              << " request_index=" << i
                              << " request_id=" << replay.requests[i].request.requestId
                              << " recorded=" << recorded << " actual=" << actual[i]
                              << std::endl;
                }
                ++recordedMismatches;
            }
        }
        std::vector<int64_t> scalarTimes, batchTimes, instrumentedTimes;
        for (size_t trial = 0; trial < 3; ++trial) {
            auto begin = std::chrono::steady_clock::now();
            for (size_t repeat = 0; repeat < REPEATS; ++repeat) scalar();
            scalarTimes.push_back(elapsedNs(begin));
            begin = std::chrono::steady_clock::now();
            for (size_t repeat = 0; repeat < REPEATS; ++repeat) {
                cache.getMatchedBlockCounts(agentId, replay.requests, blockSize,
                                            cached, full, tail, hybrid, actual.data());
            }
            batchTimes.push_back(elapsedNs(begin));
            begin = std::chrono::steady_clock::now();
            for (size_t repeat = 0; repeat < REPEATS; ++repeat) {
                MasterPendingCacheR::BatchMatchTiming timing;
                cache.getMatchedBlockCounts(agentId, replay.requests, blockSize,
                                            cached, full, tail, hybrid, actual.data(), &timing);
            }
            instrumentedTimes.push_back(elapsedNs(begin));
        }
        if (expected != actual) throw std::runtime_error("timed scalar/batch match mismatch");
        std::sort(scalarTimes.begin(), scalarTimes.end());
        std::sort(batchTimes.begin(), batchTimes.end());
        std::sort(instrumentedTimes.begin(), instrumentedTimes.end());
        totalScalarNs += scalarTimes[1] / REPEATS;
        totalBatchNs += batchTimes[1] / REPEATS;
        totalInstrumentedNs += instrumentedTimes[1] / REPEATS;
        const auto stats = cache.getStats(agentId);
        std::cout << "node=" << replay.nodes[nodeIndex].agentId
                  << " physical_blocks=" << cached.size()
                  << " pending_blocks=" << stats.uniqueBlockCount
                  << " pending_batches=" << stats.batchCount
                  << " hybrid=" << hybrid << " matched_blocks=" << checksum
                  << " scalar_ns=" << scalarTimes[1] / REPEATS
                  << " batch_ns=" << batchTimes[1] / REPEATS
                  << " instrumented_ns=" << instrumentedTimes[1] / REPEATS << std::endl;
    }
    if (statsOnly) {
        std::cout << "pending_stats_replay_nodes=" << states.size() << std::endl;
        return 0;
    }
    if (exclusionOnly || snapshotContentionOnly) return 0;
    if (physicalIndexOnly) {
        if (physicalIndexNodes == 0) throw std::runtime_error("no pending-free nodes in replay");
        std::cout << "physical_index_nodes=" << physicalIndexNodes
                  << " skipped_pending_nodes=" << states.size() - physicalIndexNodes
                  << " requests=" << replay.requests.size()
                  << " serial_node_cpu_ns=" << totalNodeIndexNs
                  << " serial_flat_cpu_ns=" << totalFlatIndexNs << std::endl;
        return 0;
    }
    std::cout << "replay_requests=" << replay.requests.size()
              << " replay_nodes=" << states.size() << " request_blocks=" << totalBlocks
              << " recorded_match_mismatches=" << recordedMismatches
              << " serial_scalar_ns=" << totalScalarNs
              << " serial_batch_ns=" << totalBatchNs
              << " serial_instrumented_ns=" << totalInstrumentedNs << std::endl;
    return 0;
}

} // namespace
} // namespace navi_sched

int main(int argc, char **argv)
{
    using namespace navi_sched;
    if ((argc == 3 || (argc == 4 && std::string(argv[1]) == "--snapshot-contention-replay")) &&
                     (std::string(argv[1]) == "--replay" ||
                      std::string(argv[1]) == "--stats-replay" ||
                      std::string(argv[1]) == "--physical-index-replay" ||
                      std::string(argv[1]) == "--exclusion-replay" ||
                      std::string(argv[1]) == "--snapshot-contention-replay" ||
                      std::string(argv[1]) == "--empty-waiting-replay" ||
                      std::string(argv[1]) == "--physical-index-churn-replay")) {
        try {
            const bool churn = std::string(argv[1]) == "--physical-index-churn-replay";
            return benchmarkReplay(argv[2], std::string(argv[1]) == "--stats-replay",
                                   std::string(argv[1]) == "--physical-index-replay" || churn,
                                   churn, std::string(argv[1]) == "--exclusion-replay",
                                   std::string(argv[1]) == "--snapshot-contention-replay",
                                   std::string(argv[1]) == "--empty-waiting-replay",
                                   argc == 4 ? std::stoul(argv[3]) : 0);
        } catch (const std::exception &error) {
            std::cerr << error.what() << std::endl;
            return 1;
        }
    }
    if (argc != 1) {
        std::cerr << "usage: " << argv[0]
                  << " [--replay FILE | --stats-replay FILE | --physical-index-replay FILE"
                  << " | --physical-index-churn-replay FILE | --exclusion-replay FILE"
                  << " | --snapshot-contention-replay FILE [NODE_INDEX] | --empty-waiting-replay FILE]"
                  << std::endl;
        return 2;
    }
    auto blockHashes = std::make_shared<const NaviSchedBlockHashes>();
    auto mutableBlockHashes = std::make_shared<NaviSchedBlockHashes>();
    mutableBlockHashes->reserve(BLOCK_COUNT);
    for (size_t index = 0; index < BLOCK_COUNT; ++index) {
        mutableBlockHashes->push_back(index + 1);
    }
    blockHashes = mutableBlockHashes;

    MasterPendingCacheR cache;
    MasterPendingCacheR::ReferencedRequest referenced{
        blockHashes.get(), BLOCK_COUNT, 1};
    if (!cache.reserve(AGENT_ID_HASH, 1, 1, {referenced},
                       NaviSchedCachePolicy{1, 16, true})) {
        return 1;
    }
    if (!cache.reserve(AGENT_ID_HASH + 1, 1, 1, {referenced},
                       NaviSchedCachePolicy{1, 16, true}) ||
        !cache.retire(AGENT_ID_HASH + 1, 1)) {
        return 1;
    }
    auto requests = makeRequests(blockHashes);
    const auto matchableBlockCount = BlockHashUtil::getMatchableBlockCount(
        requests.front().request.promptTokenCount,
        requests.front().request.blockSize,
        requests.front().request.blockHashes);
    std::vector<int64_t> matchedBlockCounts(REQUEST_COUNT);
    const MasterPendingCacheR::PhysicalBlockSet empty;
    const std::unordered_set<uint64_t> fullReady(blockHashes->begin(),
                                                  blockHashes->end());
    const std::unordered_set<uint64_t> tailReady = {blockHashes->back()};
    const MasterPendingCacheR::PhysicalBlockSet indexedFull(fullReady.begin(), fullReady.end());
    const MasterPendingCacheR::PhysicalBlockSet indexedTail(tailReady.begin(), tailReady.end());
    uint64_t checksum = 0;

    auto begin = std::chrono::steady_clock::now();
    for (size_t repeat = 0; repeat < MATCH_REPEAT_COUNT; ++repeat) {
        for (const auto &request : requests) {
            checksum += baselineMatch(*request.request.blockHashes, fullReady,
                                      tailReady, matchableBlockCount);
        }
    }
    const auto baselineMatchNs = elapsedNs(begin);

    begin = std::chrono::steady_clock::now();
    for (size_t repeat = 0; repeat < MATCH_REPEAT_COUNT; ++repeat) {
        checksum += cache.getMatchedBlockCounts(
            AGENT_ID_HASH + 1, requests, 1, empty, indexedFull, indexedTail, true,
            matchedBlockCounts.data());
        checksum += matchedBlockCounts.back();
    }
    const auto noPendingBatchMatchNs = elapsedNs(begin);

    begin = std::chrono::steady_clock::now();
    for (size_t repeat = 0; repeat < MATCH_REPEAT_COUNT; ++repeat) {
        checksum += cache.getMatchedBlockCounts(
            AGENT_ID_HASH, requests, 1, empty, empty, empty, true,
            matchedBlockCounts.data());
        checksum += matchedBlockCounts.back();
    }
    const auto batchMatchNs = elapsedNs(begin);

    begin = std::chrono::steady_clock::now();
    for (size_t repeat = 0; repeat < MATCH_REPEAT_COUNT; ++repeat) {
        for (const auto &request : requests) {
            checksum += cache.getMatchedBlockCount(
                AGENT_ID_HASH, *request.request.blockHashes, empty, empty,
                empty, true, matchableBlockCount);
        }
    }
    const auto repeatedSingleMatchNs = elapsedNs(begin);

    const auto matchOperations = MATCH_REPEAT_COUNT * REQUEST_COUNT;
    std::cout << "requests_per_batch=" << REQUEST_COUNT
              << " blocks_per_request=" << BLOCK_COUNT
              << " match_operations=" << matchOperations
              << " baseline_match_ns_per_request="
              << baselineMatchNs / matchOperations
              << " no_pending_match_ns_per_request="
              << noPendingBatchMatchNs / matchOperations
              << " pending_batch_match_ns_per_request="
              << batchMatchNs / matchOperations
              << " repeated_single_match_ns_per_request="
              << repeatedSingleMatchNs / matchOperations
              << " batch_speedup="
              << static_cast<double>(repeatedSingleMatchNs) /
                     std::max<int64_t>(1, batchMatchNs)
              << " checksum=" << checksum << std::endl;
    return 0;
}
