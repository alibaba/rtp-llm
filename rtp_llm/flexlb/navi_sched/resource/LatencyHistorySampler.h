#pragma once

#include "rtp_llm/flexlb/navi_sched/data/NaviSchedLatencySample.h"
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace autil {
class DataBuffer;
}

namespace navi_sched {

class LatencyHistorySampler {
public:
    using History = std::deque<NaviSchedLatencySample>;
    using HistoryPtr = std::shared_ptr<History>;
    using HistoryConstPtr = std::shared_ptr<const History>;

    struct Entry {
        uint64_t sequence = 0;
        NaviSchedLatencySample sample;
    };
    using Bucket = std::deque<Entry>;
    using Buckets = std::map<int32_t, Bucket>;

public:
    explicit LatencyHistorySampler(uint64_t randomSeed = 0);

    void configure(size_t historyLimit, size_t bucketCapacity);
    void clear();
    void append(const NaviSchedLatencySample &sample);
    std::vector<NaviSchedLatencySample> getFitBatch(size_t fitBatchSize) const;
    std::vector<NaviSchedLatencySample> getOnlineFitBatch() const;
    std::vector<std::vector<NaviSchedLatencySample>>
    getOnlineFitBatches(size_t currentSampleCount) const;
    std::vector<std::vector<NaviSchedLatencySample>>
    getOnlineFitBatches(size_t currentSampleCount,
                        size_t currentBucketSampleCount) const;
    HistoryConstPtr getHistory() const;
    size_t historySize() const;
    size_t sessionSampleCount() const;
    size_t bucketCount() const;
    size_t bucketSampleCount(int32_t bucketIndex) const;
    size_t sampleCountAtMost(double latencyMs) const;
    void serialize(autil::DataBuffer &dataBuffer) const;
    bool deserialize(autil::DataBuffer &dataBuffer);

    static int32_t getBucketIndex(double latencyMs);

private:
    void appendBucketSamples(
        const std::unordered_set<uint64_t> &excludedSequences,
        std::optional<int32_t> excludedBucket,
        std::vector<NaviSchedLatencySample> &result,
        uint64_t selectionContext,
        std::vector<uint64_t> *selectedSequences,
        size_t samplesPerBucket = 1,
        std::optional<int32_t> onlyBucket = std::nullopt) const;
    uint64_t makeSelectionKey(uint64_t selectionContext,
                              int32_t bucketIndex) const;
    static uint64_t mix(uint64_t value);
    static std::string formatSequenceList(const std::vector<uint64_t> &sequences);
    static void writeSample(autil::DataBuffer &dataBuffer,
                            const NaviSchedLatencySample &sample);
    static void readSample(autil::DataBuffer &dataBuffer, NaviSchedLatencySample &sample);
    bool deserializeUnchecked(autil::DataBuffer &dataBuffer);

private:
    HistoryPtr _history = std::make_shared<History>();
    std::deque<uint64_t> _historySequences;
    Buckets _buckets;
    size_t _historyLimit = 2000;
    size_t _bucketCapacity = 64;
    uint64_t _nextSequence = 1;
    size_t _sessionSampleCount = 0;
    uint64_t _randomSeed = 0;
};

} // namespace navi_sched
