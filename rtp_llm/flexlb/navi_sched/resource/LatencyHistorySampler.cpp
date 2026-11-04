#include "rtp_llm/flexlb/navi_sched/resource/LatencyHistorySampler.h"

#include "autil/DataBuffer.h"
#include "navi/log/NaviLogger.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace navi_sched {
namespace {
constexpr uint32_t MAX_SERIALIZED_BUCKET_COUNT = 2048;
constexpr uint32_t MAX_SERIALIZED_SAMPLE_COUNT = 1U << 20;
constexpr uint64_t DEFAULT_HISTORY_RANDOM_SEED = 0x4e41564948495354ULL;

bool validAggregate(int64_t batchSize, int64_t cacheHitTokenCount,
                    int64_t cacheMissTokenCount, double cacheMissTokenSquareSum,
                    double cacheHitMissTokenProductSum)
{
    return batchSize >= 0 && cacheHitTokenCount >= 0 && cacheMissTokenCount >= 0 &&
           std::isfinite(cacheMissTokenSquareSum) && cacheMissTokenSquareSum >= 0.0 &&
           std::isfinite(cacheHitMissTokenProductSum) &&
           cacheHitMissTokenProductSum >= 0.0;
}

bool validSample(const NaviSchedLatencySample &sample)
{
    const auto &features = sample.features;
    return std::isfinite(sample.latencyMs) && sample.latencyMs >= 0.0 &&
           features.getBatchCount() >= 0 && features.getPrecedingRequestCount() >= 0 &&
           validAggregate(features.getRunningBatchSize(),
                          features.getRunningCacheHitTokenCount(),
                          features.getRunningCacheMissTokenCount(),
                          features.getRunningCacheMissTokenSquareSum(),
                          features.getRunningCacheHitMissTokenProductSum()) &&
           validAggregate(features.getWaitingBatchSize(),
                          features.getWaitingCacheHitTokenCount(),
                          features.getWaitingCacheMissTokenCount(),
                          features.getWaitingCacheMissTokenSquareSum(),
                          features.getWaitingCacheHitMissTokenProductSum()) &&
           (features.usesCapacitySplit() || features.getWaitingBatchSize() == 0);
}
} // namespace

LatencyHistorySampler::LatencyHistorySampler(uint64_t randomSeed)
    : _randomSeed(randomSeed != 0 ? randomSeed : DEFAULT_HISTORY_RANDOM_SEED)
{
}

void LatencyHistorySampler::configure(size_t historyLimit, size_t bucketCapacity)
{
    _historyLimit = std::max<size_t>(1, historyLimit);
    _bucketCapacity = std::max<size_t>(1, bucketCapacity);
    if (!_history.unique()) {
        _history = std::make_shared<History>(*_history);
    }
    while (_history->size() > _historyLimit) {
        _history->pop_front();
        _historySequences.pop_front();
    }
    for (auto &[unused, bucket] : _buckets) {
        while (bucket.size() > _bucketCapacity) {
            bucket.pop_front();
        }
    }
}

void LatencyHistorySampler::clear()
{
    _history = std::make_shared<History>();
    _historySequences.clear();
    _buckets.clear();
    _nextSequence = 1;
    _sessionSampleCount = 0;
}

void LatencyHistorySampler::append(const NaviSchedLatencySample &sample)
{
    if (!_history.unique()) {
        _history = std::make_shared<History>(*_history);
    }
    const auto sequence = _nextSequence++;
    ++_sessionSampleCount;
    _history->emplace_back(sample);
    _historySequences.emplace_back(sequence);
    while (_history->size() > _historyLimit) {
        _history->pop_front();
        _historySequences.pop_front();
    }
    auto &bucket = _buckets[getBucketIndex(sample.latencyMs)];
    bucket.push_back({sequence, sample});
    while (bucket.size() > _bucketCapacity) {
        bucket.pop_front();
    }
    NAVI_KERNEL_LOG(
        INFO,
        "[latency-history-append] sequence=%lu bucket=%d local_batch_id=%ld "
        "event_time_us=%ld latency_ms=%.9f history_size=%lu bucket_size=%lu",
        sequence, getBucketIndex(sample.latencyMs), sample.localBatchId,
        sample.eventTimeUs, sample.latencyMs, _history->size(), bucket.size());
}

std::vector<NaviSchedLatencySample> LatencyHistorySampler::getFitBatch(
    size_t fitBatchSize) const
{
    const auto begin = _history->size() > fitBatchSize ? _history->end() - fitBatchSize
                                                       : _history->begin();
    std::vector<uint64_t> selectedSequences;
    const auto firstIndex = static_cast<size_t>(begin - _history->begin());
    selectedSequences.reserve(_history->size() - firstIndex);
    for (size_t index = firstIndex; index < _historySequences.size(); ++index) {
        selectedSequences.emplace_back(_historySequences[index]);
    }
    NAVI_KERNEL_LOG(INFO, "[latency-history-fit] mode=recent sample_count=%lu "
                    "sequence_ids=%s",
                    selectedSequences.size(),
                    formatSequenceList(selectedSequences).c_str());
    return {begin, _history->end()};
}

std::vector<NaviSchedLatencySample> LatencyHistorySampler::getOnlineFitBatch() const
{
    std::vector<NaviSchedLatencySample> result;
    std::vector<uint64_t> selectedSequences;
    const auto selectionContext =
        _historySequences.empty() ? 0 : _historySequences.back();
    appendBucketSamples({}, std::nullopt, result, selectionContext,
                        &selectedSequences);
    NAVI_KERNEL_LOG(INFO, "[latency-history-fit] mode=one_per_bucket "
                    "selection_context=%lu sample_count=%lu sequence_ids=%s",
                    selectionContext, selectedSequences.size(),
                    formatSequenceList(selectedSequences).c_str());
    return result;
}

std::vector<std::vector<NaviSchedLatencySample>>
LatencyHistorySampler::getOnlineFitBatches(size_t currentSampleCount) const
{
    return getOnlineFitBatches(currentSampleCount, 0);
}

std::vector<std::vector<NaviSchedLatencySample>>
LatencyHistorySampler::getOnlineFitBatches(
    size_t currentSampleCount, size_t currentBucketSampleCount) const
{
    std::vector<std::vector<NaviSchedLatencySample>> batches;
    currentSampleCount = std::min(currentSampleCount, _history->size());
    if (currentSampleCount == 0) {
        return batches;
    }

    std::unordered_set<uint64_t> currentSequences;
    currentSequences.reserve(currentSampleCount);
    const auto currentBegin = _historySequences.size() - currentSampleCount;
    for (size_t index = currentBegin; index < _historySequences.size(); ++index) {
        currentSequences.emplace(_historySequences[index]);
    }

    batches.reserve(currentSampleCount);
    const auto historyBegin = _history->size() - currentSampleCount;
    for (size_t index = 0; index < currentSampleCount; ++index) {
        auto &batch = batches.emplace_back();
        const auto &current = (*_history)[historyBegin + index];
        const auto currentBucket = getBucketIndex(current.latencyMs);
        std::vector<uint64_t> selectedSequences;
        const auto currentSequence = _historySequences[historyBegin + index];
        appendBucketSamples(currentSequences, currentBucket, batch,
                            currentSequence, &selectedSequences, 1);
        if (currentBucketSampleCount > 0) {
            appendBucketSamples(currentSequences, std::nullopt, batch,
                                currentSequence, &selectedSequences,
                                currentBucketSampleCount, currentBucket);
        }
        batch.emplace_back(current);
        selectedSequences.emplace_back(currentSequence);
        NAVI_KERNEL_LOG(
            INFO,
            "[latency-history-fit] mode=online_batch index=%lu "
            "current_sequence=%lu current_bucket=%d sample_count=%lu "
            "sequence_ids=%s",
            index, currentSequence, currentBucket, selectedSequences.size(),
            formatSequenceList(selectedSequences).c_str());
        // Keep the current sample excluded while selecting history, then make
        // it eligible for later batches as an already-processed sample.
        currentSequences.erase(currentSequence);
    }
    return batches;
}

void LatencyHistorySampler::appendBucketSamples(
    const std::unordered_set<uint64_t> &excludedSequences,
    std::optional<int32_t> excludedBucket, std::vector<NaviSchedLatencySample> &result,
    uint64_t selectionContext, std::vector<uint64_t> *selectedSequences,
    size_t samplesPerBucket, std::optional<int32_t> onlyBucket) const
{
    result.reserve(result.size() + _buckets.size());
    samplesPerBucket = std::max<size_t>(1, samplesPerBucket);
    for (const auto &[bucketIndex, bucket] : _buckets) {
        if (onlyBucket && bucketIndex != *onlyBucket) {
            continue;
        }
        if (bucket.empty() || (excludedBucket && bucketIndex == *excludedBucket)) {
            continue;
        }
        const auto selectionKey = makeSelectionKey(selectionContext, bucketIndex);
        const auto entryOffset = static_cast<size_t>(selectionKey % bucket.size());
        size_t appended = 0;
        std::vector<uint64_t> bucketSelections;
        for (size_t offset = 0; offset < bucket.size(); ++offset) {
            const auto &entry = bucket[(entryOffset + offset) % bucket.size()];
            if (excludedSequences.find(entry.sequence) == excludedSequences.end()) {
                result.emplace_back(entry.sample);
                bucketSelections.emplace_back(entry.sequence);
                if (selectedSequences) {
                    selectedSequences->emplace_back(entry.sequence);
                }
                if (++appended == samplesPerBucket) {
                    break;
                }
            }
        }
        NAVI_KERNEL_LOG(
            INFO,
            "[latency-history-bucket] context=%lu key=%lu bucket=%d "
            "bucket_size=%lu entry_offset=%lu samples_per_bucket=%lu "
            "excluded_bucket=%d only_bucket=%d selected_sequences=%s",
            selectionContext, selectionKey, bucketIndex, bucket.size(), entryOffset,
            samplesPerBucket, excludedBucket.value_or(std::numeric_limits<int32_t>::min()),
            onlyBucket.value_or(std::numeric_limits<int32_t>::min()),
            formatSequenceList(bucketSelections).c_str());
    }
}

uint64_t LatencyHistorySampler::makeSelectionKey(uint64_t selectionContext,
                                                  int32_t bucketIndex) const
{
    const auto bucketKey = static_cast<uint64_t>(
        static_cast<int64_t>(bucketIndex));
    return mix(_randomSeed ^ mix(selectionContext + 0x9e3779b97f4a7c15ULL) ^
               mix(bucketKey + 0xd1b54a32d192ed03ULL));
}

uint64_t LatencyHistorySampler::mix(uint64_t value)
{
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::string LatencyHistorySampler::formatSequenceList(
    const std::vector<uint64_t> &sequences)
{
    std::ostringstream output;
    output << '[';
    for (size_t index = 0; index < sequences.size(); ++index) {
        if (index > 0) {
            output << ',';
        }
        output << sequences[index];
    }
    output << ']';
    return output.str();
}

LatencyHistorySampler::HistoryConstPtr LatencyHistorySampler::getHistory() const
{
    return _history;
}

size_t LatencyHistorySampler::historySize() const { return _history->size(); }

size_t LatencyHistorySampler::sessionSampleCount() const
{
    return _sessionSampleCount;
}

size_t LatencyHistorySampler::bucketCount() const { return _buckets.size(); }

size_t LatencyHistorySampler::bucketSampleCount(int32_t bucketIndex) const
{
    const auto iterator = _buckets.find(bucketIndex);
    return iterator == _buckets.end() ? 0 : iterator->second.size();
}

size_t LatencyHistorySampler::sampleCountAtMost(double latencyMs) const
{
    if (!std::isfinite(latencyMs) || latencyMs < 0.0) {
        return 0;
    }
    size_t count = 0;
    for (const auto &[unused, bucket] : _buckets) {
        for (const auto &entry : bucket) {
            if (entry.sample.latencyMs <= latencyMs) {
                ++count;
            }
        }
    }
    return count;
}

void LatencyHistorySampler::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(_nextSequence);
    dataBuffer.write(uint32_t(_history->size()));
    for (size_t index = 0; index < _history->size(); ++index) {
        dataBuffer.write(_historySequences[index]);
        writeSample(dataBuffer, (*_history)[index]);
    }
    dataBuffer.write(uint32_t(_buckets.size()));
    for (const auto &[bucketIndex, bucket] : _buckets) {
        dataBuffer.write(bucketIndex);
        dataBuffer.write(uint32_t(bucket.size()));
        for (const auto &entry : bucket) {
            dataBuffer.write(entry.sequence);
            writeSample(dataBuffer, entry.sample);
        }
    }
}

bool LatencyHistorySampler::deserialize(autil::DataBuffer &dataBuffer)
{
    try {
        return deserializeUnchecked(dataBuffer);
    } catch (const autil::DataBufferCorruptedException &) {
        return false;
    }
}

bool LatencyHistorySampler::deserializeUnchecked(autil::DataBuffer &dataBuffer)
{
    uint64_t nextSequence = 0;
    uint32_t historySize = 0;
    dataBuffer.read(nextSequence);
    dataBuffer.read(historySize);
    if (nextSequence == 0 || historySize > MAX_SERIALIZED_SAMPLE_COUNT) {
        return false;
    }

    History loadedHistory;
    std::deque<uint64_t> loadedHistorySequences;
    uint64_t maxSequence = 0;
    uint64_t previousHistorySequence = 0;
    for (uint32_t index = 0; index < historySize; ++index) {
        uint64_t sequence = 0;
        NaviSchedLatencySample sample;
        dataBuffer.read(sequence);
        readSample(dataBuffer, sample);
        if (sequence <= previousHistorySequence || !validSample(sample)) {
            return false;
        }
        previousHistorySequence = sequence;
        loadedHistory.emplace_back(std::move(sample));
        loadedHistorySequences.emplace_back(sequence);
        maxSequence = std::max(maxSequence, sequence);
    }

    uint32_t bucketCount = 0;
    dataBuffer.read(bucketCount);
    if (bucketCount > MAX_SERIALIZED_BUCKET_COUNT) {
        return false;
    }
    Buckets loadedBuckets;
    uint64_t bucketSampleCount = 0;
    std::unordered_set<uint64_t> bucketSequences;
    for (uint32_t bucketNumber = 0; bucketNumber < bucketCount; ++bucketNumber) {
        int32_t bucketIndex = 0;
        uint32_t entryCount = 0;
        dataBuffer.read(bucketIndex);
        dataBuffer.read(entryCount);
        bucketSampleCount += entryCount;
        if (entryCount == 0 || bucketSampleCount > MAX_SERIALIZED_SAMPLE_COUNT) {
            return false;
        }
        auto [bucketIterator, inserted] = loadedBuckets.emplace(bucketIndex, Bucket());
        if (!inserted) {
            return false;
        }
        auto &bucket = bucketIterator->second;
        uint64_t previousBucketSequence = 0;
        for (uint32_t entryNumber = 0; entryNumber < entryCount; ++entryNumber) {
            Entry entry;
            dataBuffer.read(entry.sequence);
            readSample(dataBuffer, entry.sample);
            if (entry.sequence <= previousBucketSequence || !validSample(entry.sample) ||
                getBucketIndex(entry.sample.latencyMs) != bucketIndex ||
                !bucketSequences.emplace(entry.sequence).second) {
                return false;
            }
            previousBucketSequence = entry.sequence;
            maxSequence = std::max(maxSequence, entry.sequence);
            bucket.emplace_back(std::move(entry));
        }
    }
    if (nextSequence <= maxSequence) {
        return false;
    }

    while (loadedHistory.size() > _historyLimit) {
        loadedHistory.pop_front();
        loadedHistorySequences.pop_front();
    }
    for (auto &[unused, bucket] : loadedBuckets) {
        while (bucket.size() > _bucketCapacity) {
            bucket.pop_front();
        }
    }
    _history = std::make_shared<History>(std::move(loadedHistory));
    _historySequences = std::move(loadedHistorySequences);
    _buckets = std::move(loadedBuckets);
    _nextSequence = nextSequence;
    _sessionSampleCount = 0;
    return true;
}

int32_t LatencyHistorySampler::getBucketIndex(double latencyMs)
{
    if (!std::isfinite(latencyMs) || latencyMs <= 1.0) {
        return 0;
    }
    const auto value = std::floor(std::log2(latencyMs));
    return static_cast<int32_t>(
        std::min(value, static_cast<double>(std::numeric_limits<int32_t>::max())));
}

void LatencyHistorySampler::writeSample(autil::DataBuffer &dataBuffer,
                                        const NaviSchedLatencySample &sample)
{
    sample.features.serialize(dataBuffer);
    dataBuffer.write(sample.latencyMs);
    dataBuffer.write(sample.eventTimeUs);
}

void LatencyHistorySampler::readSample(autil::DataBuffer &dataBuffer,
                                       NaviSchedLatencySample &sample)
{
    sample.features.deserialize(dataBuffer);
    dataBuffer.read(sample.latencyMs);
    dataBuffer.read(sample.eventTimeUs);
}

} // namespace navi_sched
