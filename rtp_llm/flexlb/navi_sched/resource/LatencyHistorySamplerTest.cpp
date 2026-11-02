#include "aios/navi_sched/resource/LatencyHistorySampler.h"

#include "autil/DataBuffer.h"
#include "unittest/unittest.h"
#include <algorithm>
#include <cmath>

namespace navi_sched {
namespace {

NaviSchedLatencySample makeSample(double latencyMs, int64_t eventTimeUs)
{
    NaviSchedLatencySample sample;
    sample.features.appendRequest(eventTimeUs, eventTimeUs + 1);
    sample.latencyMs = latencyMs;
    sample.eventTimeUs = eventTimeUs;
    return sample;
}

void writeSample(autil::DataBuffer &buffer, const NaviSchedLatencySample &sample)
{
    sample.features.serialize(buffer);
    buffer.write(sample.latencyMs);
    buffer.write(sample.eventTimeUs);
}

void writeNegativeBatchSample(autil::DataBuffer &buffer)
{
    buffer.write(int64_t(1));
    buffer.write(int64_t(-1));
    buffer.write(int64_t(0));
    buffer.write(int64_t(0));
    buffer.write(0.0);
    buffer.write(0.0);
    buffer.write(false);
    buffer.write(int64_t(0));
    buffer.write(int64_t(0));
    buffer.write(int64_t(0));
    buffer.write(0.0);
    buffer.write(0.0);
    buffer.write(int64_t(0));
    buffer.write(1.0);
    buffer.write(int64_t(1));
}

}

class LatencyHistorySamplerTest : public TESTBASE {};

TEST_F(LatencyHistorySamplerTest, testLog2LatencyBucketBoundaries)
{
    EXPECT_EQ(0, LatencyHistorySampler::getBucketIndex(0.0));
    EXPECT_EQ(0, LatencyHistorySampler::getBucketIndex(1.0));
    EXPECT_EQ(0, LatencyHistorySampler::getBucketIndex(1.999));
    EXPECT_EQ(1, LatencyHistorySampler::getBucketIndex(2.0));
    EXPECT_EQ(1, LatencyHistorySampler::getBucketIndex(3.999));
    EXPECT_EQ(2, LatencyHistorySampler::getBucketIndex(4.0));
    EXPECT_EQ(10, LatencyHistorySampler::getBucketIndex(1024.0));
}

TEST_F(LatencyHistorySamplerTest, testSamplesEveryNonEmptyBucketOnce)
{
    LatencyHistorySampler sampler(17);
    sampler.configure(128, 64);
    int64_t sequence = 1;
    for (int32_t bucket = 0; bucket < 16; ++bucket) {
        const auto latency = bucket == 0 ? 1.0 : std::pow(2.0, bucket);
        sampler.append(makeSample(latency, sequence++));
        sampler.append(makeSample(latency + 0.25, sequence++));
    }

    const auto samples = sampler.getOnlineFitBatch();
    ASSERT_EQ(16, samples.size());
    for (size_t index = 0; index < samples.size(); ++index) {
        EXPECT_EQ(static_cast<int32_t>(index),
                  LatencyHistorySampler::getBucketIndex(samples[index].latencyMs));
    }
}

TEST_F(LatencyHistorySamplerTest, testBuildsOneHistoricalBatchPerCurrentSample)
{
    LatencyHistorySampler sampler(19);
    sampler.configure(128, 64);
    int64_t sequence = 1;
    for (int32_t bucket = 0; bucket < 16; ++bucket) {
        const auto latency = bucket == 0 ? 1.0 : std::pow(2.0, bucket);
        sampler.append(makeSample(latency, sequence++));
    }
    sampler.append(makeSample(8.25, sequence++));
    sampler.append(makeSample(32.25, sequence++));
    sampler.append(makeSample(8.5, sequence++));
    sampler.append(makeSample(128.25, sequence++));

    const auto batches = sampler.getOnlineFitBatches(4);
    ASSERT_EQ(4, batches.size());
    for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
        const auto &batch = batches[batchIndex];
        ASSERT_EQ(16, batch.size());
        EXPECT_EQ(static_cast<int64_t>(17 + batchIndex), batch.back().eventTimeUs);
        std::vector<int32_t> buckets;
        buckets.reserve(batch.size());
        for (const auto &sample : batch) {
            buckets.emplace_back(
                LatencyHistorySampler::getBucketIndex(sample.latencyMs));
        }
        std::sort(buckets.begin(), buckets.end());
        EXPECT_EQ((std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7,
                                        8, 9, 10, 11, 12, 13, 14, 15}),
                  buckets);
    }
}

TEST_F(LatencyHistorySamplerTest, testSelectionIsDeterministicForSameHistory)
{
    LatencyHistorySampler first(1234);
    LatencyHistorySampler second(1234);
    first.configure(128, 8);
    second.configure(128, 8);
    for (int64_t index = 1; index <= 32; ++index) {
        const auto latency = std::pow(2.0, index % 8) + index / 100.0;
        auto sample = makeSample(latency, index);
        sample.localBatchId = index * 10;
        first.append(sample);
        second.append(sample);
    }

    const auto firstBatches = first.getOnlineFitBatches(8, 2);
    const auto secondBatches = second.getOnlineFitBatches(8, 2);
    ASSERT_EQ(firstBatches.size(), secondBatches.size());
    for (size_t batchIndex = 0; batchIndex < firstBatches.size(); ++batchIndex) {
        ASSERT_EQ(firstBatches[batchIndex].size(), secondBatches[batchIndex].size());
        for (size_t sampleIndex = 0;
             sampleIndex < firstBatches[batchIndex].size(); ++sampleIndex) {
            const auto &left = firstBatches[batchIndex][sampleIndex];
            const auto &right = secondBatches[batchIndex][sampleIndex];
            EXPECT_EQ(left.eventTimeUs, right.eventTimeUs);
            EXPECT_EQ(left.localBatchId, right.localBatchId);
            EXPECT_DOUBLE_EQ(left.latencyMs, right.latencyMs);
        }
    }
}

TEST_F(LatencyHistorySamplerTest, testCurrentBucketExtraSamplesExcludeCurrentSample)
{
    LatencyHistorySampler sampler(20);
    sampler.configure(128, 64);
    for (int64_t index = 1; index <= 4; ++index) {
        sampler.append(makeSample(1.0 + index / 100.0, index));
    }
    sampler.append(makeSample(8.0, 5));
    sampler.append(makeSample(1.05, 6));

    const auto batches = sampler.getOnlineFitBatches(1, 2);
    ASSERT_EQ(1, batches.size());
    ASSERT_EQ(4, batches[0].size());
    EXPECT_EQ(6, batches[0].back().eventTimeUs);
    size_t sameBucketHistoricalCount = 0;
    for (const auto &sample : batches[0]) {
        if (LatencyHistorySampler::getBucketIndex(sample.latencyMs) == 0) {
            if (sample.eventTimeUs != 6) {
                ++sameBucketHistoricalCount;
            }
        }
    }
    EXPECT_EQ(2, sameBucketHistoricalCount);
}

TEST_F(LatencyHistorySamplerTest,
       testCurrentBucketExtraSamplesKeepOtherPressureBucket)
{
    LatencyHistorySampler sampler(24);
    sampler.configure(128, 64);
    sampler.append(makeSample(1.0, 1));
    sampler.append(makeSample(1024.0, 2));
    sampler.append(makeSample(1025.0, 3));
    sampler.append(makeSample(1026.0, 4));

    const auto batches = sampler.getOnlineFitBatches(1, 2);
    ASSERT_EQ(1, batches.size());
    ASSERT_EQ(4, batches[0].size());

    size_t lowPressureCount = 0;
    size_t highPressureCount = 0;
    for (const auto &sample : batches[0]) {
        if (LatencyHistorySampler::getBucketIndex(sample.latencyMs) == 0) {
            ++lowPressureCount;
        } else {
            ++highPressureCount;
        }
    }
    EXPECT_EQ(1, lowPressureCount);
    EXPECT_EQ(3, highPressureCount);
    EXPECT_EQ(4, batches[0].back().eventTimeUs);
}

TEST_F(LatencyHistorySamplerTest, testOnlineFitBatchesClampCurrentSampleCount)
{
    LatencyHistorySampler sampler(21);
    sampler.configure(16, 64);
    sampler.append(makeSample(8.0, 1));
    sampler.append(makeSample(16.0, 2));

    const auto batches = sampler.getOnlineFitBatches(100);
    ASSERT_EQ(2, batches.size());
    ASSERT_EQ(1, batches[0].size());
    ASSERT_EQ(2, batches[1].size());
    EXPECT_EQ(1, batches[0].back().eventTimeUs);
    EXPECT_EQ(2, batches[1].back().eventTimeUs);
    EXPECT_EQ(1, batches[1].front().eventTimeUs);
    EXPECT_TRUE(sampler.getOnlineFitBatches(0).empty());
}

TEST_F(LatencyHistorySamplerTest, testSingleBucketReturnsOneSample)
{
    LatencyHistorySampler sampler(23);
    sampler.configure(16, 64);
    for (int64_t index = 1; index <= 4; ++index) {
        sampler.append(makeSample(8.0, index));
    }
    const auto samples = sampler.getOnlineFitBatch();
    ASSERT_EQ(1, samples.size());
    EXPECT_GE(samples[0].eventTimeUs, 1);
    EXPECT_LE(samples[0].eventTimeUs, 4);
}

TEST_F(LatencyHistorySamplerTest, testBucketCapacityEvictsOldestOnly)
{
    LatencyHistorySampler sampler(29);
    sampler.configure(32, 3);
    for (int64_t index = 1; index <= 10; ++index) {
        sampler.append(makeSample(16.0 + index / 100.0, index));
    }
    EXPECT_EQ(1, sampler.bucketCount());
    EXPECT_EQ(3, sampler.bucketSampleCount(4));
    EXPECT_EQ(10, sampler.historySize());
}

TEST_F(LatencyHistorySamplerTest, testCountsRetainedSamplesByLatency)
{
    LatencyHistorySampler sampler(29);
    sampler.configure(2, 4);
    sampler.append(makeSample(32.0, 1));
    sampler.append(makeSample(128.0, 2));
    sampler.append(makeSample(512.0, 3));
    EXPECT_EQ(2, sampler.sampleCountAtMost(128.0));
    EXPECT_EQ(3, sampler.sampleCountAtMost(512.0));
    EXPECT_EQ(0, sampler.sampleCountAtMost(16.0));
}

TEST_F(LatencyHistorySamplerTest, testSerializationKeepsOldBucketsWithoutTtl)
{
    LatencyHistorySampler sampler(31);
    sampler.configure(4, 8);
    sampler.append(makeSample(1.0, 1));
    for (int64_t index = 2; index <= 10; ++index) {
        sampler.append(makeSample(1024.0, index));
    }
    ASSERT_EQ(4, sampler.historySize());
    ASSERT_EQ(1, sampler.bucketSampleCount(0));

    autil::DataBuffer buffer;
    sampler.serialize(buffer);
    autil::DataBuffer input(buffer.getData(), buffer.getDataLen());
    LatencyHistorySampler restored(31);
    restored.configure(4, 8);
    ASSERT_TRUE(restored.deserialize(input));
    EXPECT_EQ(4, restored.historySize());
    EXPECT_EQ(1, restored.bucketSampleCount(0));
    const auto expected = sampler.getOnlineFitBatch();
    const auto samples = restored.getOnlineFitBatch();
    ASSERT_EQ(expected.size(), samples.size());
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_EQ(expected[index].eventTimeUs, samples[index].eventTimeUs);
        EXPECT_DOUBLE_EQ(expected[index].latencyMs, samples[index].latencyMs);
    }
}

TEST_F(LatencyHistorySamplerTest, testHistorySnapshotUsesCopyOnWrite)
{
    LatencyHistorySampler sampler(41);
    sampler.append(makeSample(1.0, 1));
    const auto first = sampler.getHistory();
    sampler.append(makeSample(2.0, 2));
    const auto second = sampler.getHistory();
    ASSERT_EQ(1, first->size());
    ASSERT_EQ(2, second->size());
    EXPECT_NE(first.get(), second.get());
    EXPECT_EQ(1, first->front().eventTimeUs);
    EXPECT_EQ(2, second->back().eventTimeUs);
}

TEST_F(LatencyHistorySamplerTest, testDeserializeRejectsDuplicateHistorySequenceAtomically)
{
    LatencyHistorySampler sampler(43);
    sampler.append(makeSample(4.0, 99));

    autil::DataBuffer buffer;
    buffer.write(uint64_t(3));
    buffer.write(uint32_t(2));
    buffer.write(uint64_t(1));
    writeSample(buffer, makeSample(1.0, 1));
    buffer.write(uint64_t(1));
    writeSample(buffer, makeSample(2.0, 2));
    buffer.write(uint32_t(0));

    autil::DataBuffer input(buffer.getData(), buffer.getDataLen());
    EXPECT_FALSE(sampler.deserialize(input));
    const auto history = sampler.getHistory();
    ASSERT_EQ(1, history->size());
    EXPECT_EQ(99, history->front().eventTimeUs);
}

TEST_F(LatencyHistorySamplerTest, testDeserializeRejectsDuplicateBucketAtomically)
{
    LatencyHistorySampler sampler(47);
    sampler.append(makeSample(4.0, 99));

    autil::DataBuffer buffer;
    buffer.write(uint64_t(3));
    buffer.write(uint32_t(0));
    buffer.write(uint32_t(2));
    for (uint64_t sequence = 1; sequence <= 2; ++sequence) {
        buffer.write(int32_t(0));
        buffer.write(uint32_t(1));
        buffer.write(sequence);
        writeSample(buffer, makeSample(1.0, sequence));
    }

    autil::DataBuffer input(buffer.getData(), buffer.getDataLen());
    EXPECT_FALSE(sampler.deserialize(input));
    const auto history = sampler.getHistory();
    ASSERT_EQ(1, history->size());
    EXPECT_EQ(99, history->front().eventTimeUs);
}

TEST_F(LatencyHistorySamplerTest, testDeserializeRejectsInvalidFeaturesAtomically)
{
    LatencyHistorySampler sampler(53);
    sampler.append(makeSample(4.0, 99));

    autil::DataBuffer buffer;
    buffer.write(uint64_t(2));
    buffer.write(uint32_t(1));
    buffer.write(uint64_t(1));
    writeNegativeBatchSample(buffer);
    buffer.write(uint32_t(0));

    autil::DataBuffer input(buffer.getData(), buffer.getDataLen());
    EXPECT_FALSE(sampler.deserialize(input));
    const auto history = sampler.getHistory();
    ASSERT_EQ(1, history->size());
    EXPECT_EQ(99, history->front().eventTimeUs);
}

}
