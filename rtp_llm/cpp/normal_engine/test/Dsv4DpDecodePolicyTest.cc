#include "rtp_llm/cpp/normal_engine/Dsv4DpDecodePolicy.h"

#include <gtest/gtest.h>
#include <list>

namespace rtp_llm {
namespace {

TEST(Dsv4DpDecodePolicy, DedicatedTopologyRequiresSingleHostDpAndEp) {
    for (const int world : {4, 8}) {
        EXPECT_TRUE(isDsv4DedicatedDecodeTopology(RoleType::DECODE, 1, 1, world, world, world, world, false));
        EXPECT_FALSE(isDsv4DedicatedDecodeTopology(RoleType::PREFILL, 1, 1, world, world, world, world, false));
        EXPECT_FALSE(isDsv4DedicatedDecodeTopology(RoleType::DECODE, 1, 1, world, world, world, world, true));
        EXPECT_FALSE(isDsv4DedicatedDecodeTopology(RoleType::DECODE, 1, 1, world, world, world, world / 2, false));
        EXPECT_FALSE(isDsv4DedicatedDecodeTopology(RoleType::DECODE, 1, 2, world / 2, world, world, world, false));
    }
    EXPECT_FALSE(isDsv4DedicatedDecodeTopology(RoleType::DECODE, 1, 1, 2, 2, 2, 2, false));
}

TEST(Dsv4DpDecodePolicy, AdmissionRequiresPdTransitionAndOneSequence) {
    EXPECT_FALSE(isDsv4DecodeRequestSupported(false, true, true, 1, 1, false));
    EXPECT_TRUE(isDsv4DecodeRequestSupported(false, false, true, 1, 1, false));
    EXPECT_FALSE(isDsv4DecodeRequestSupported(false, false, false, 1, 1, false));
    EXPECT_FALSE(isDsv4DecodeRequestSupported(false, false, true, 2, 1, false));
    EXPECT_FALSE(isDsv4DecodeRequestSupported(false, false, true, 1, 2, false));
    EXPECT_FALSE(isDsv4DecodeRequestSupported(false, false, true, 1, 1, true));
    EXPECT_TRUE(isDsv4DecodeRequestSupported(true, false, false, 1, 1, false));
}

TEST(Dsv4DpDecodePolicy, AllIdleAndLocallyIdleRoundsDiffer) {
    EXPECT_EQ(classifyDsv4DecodeRound({0, 0, 0, 0}), DecodeReplicaRound::IDLE);
    EXPECT_EQ(classifyDsv4DecodeRound({0, 0, 0, 2}), DecodeReplicaRound::EXECUTE);
    EXPECT_EQ(classifyDsv4DecodeRound({0, 0, 0, 0, 0, 1, 0, 0}), DecodeReplicaRound::EXECUTE);
    EXPECT_EQ(classifyDsv4DecodeRound({0, -1, 0, 2}), DecodeReplicaRound::INVALID);
}

TEST(Dsv4DpDecodePolicy, RoundControlGatesAllSchedulerMutation) {
    using Vote            = Dsv4DecodeControlVote;
    int        schedules  = 0;
    int        agreements = 0;
    const auto round      = [&](std::initializer_list<Vote> votes) {
        const std::vector<int64_t> records = [&] {
            std::vector<int64_t> values;
            for (const auto vote : votes) {
                values.push_back(static_cast<int64_t>(vote));
            }
            return values;
        }();
        return scheduleDecodeReplicaRound(
            classifyDsv4DecodeControl(records),
            [&] {
                ++schedules;
                return 2;
            },
            [&](int64_t count) {
                ++agreements;
                return classifyDsv4DecodeRound({count, 0});
            });
    };
    EXPECT_EQ(round({Vote::RUN, Vote::RUN}), DecodeReplicaRound::EXECUTE);
    EXPECT_EQ(round({Vote::RUN, Vote::PAUSE}), DecodeReplicaRound::PAUSED);
    EXPECT_EQ(round({Vote::PAUSE, Vote::RUN}), DecodeReplicaRound::PAUSED);
    EXPECT_EQ(schedules, 1);
    EXPECT_EQ(agreements, 1);
    EXPECT_EQ(round({Vote::RUN, Vote::RUN}), DecodeReplicaRound::EXECUTE);
    EXPECT_EQ(round({Vote::PAUSE, Vote::STOP}), DecodeReplicaRound::STOP);
    EXPECT_EQ(schedules, 2);
    EXPECT_EQ(agreements, 2);
}

TEST(Dsv4DpDecodePolicy, RemoveFakeStreamsOnlyWhenRealRequestsCanExecute) {
    struct Stream {
        bool fake    = false;
        bool context = false;
        int  batch   = 1;
        bool isFakeStream() const {
            return fake;
        }
        bool isContextStream() const {
            return context;
        }
        int nextBatchSize() const {
            return batch;
        }
    };
    const Stream             fake{true};
    const Stream             real{};
    const Stream             context{false, true};
    const Stream             beams{false, false, 2};
    std::list<const Stream*> streams{&fake};
    EXPECT_EQ(countScheduledDecodeRequests(streams), 0);
    EXPECT_EQ(streams.size(), 1u);
    streams.push_back(&real);
    EXPECT_EQ(countScheduledDecodeRequests(streams), 1);
    EXPECT_EQ(streams.size(), 1u);
    EXPECT_EQ(streams.front(), &real);
    streams.push_back(&context);
    EXPECT_EQ(countScheduledDecodeRequests(streams), -1);
    streams.back() = &beams;
    EXPECT_EQ(countScheduledDecodeRequests(streams), -1);
}

TEST(Dsv4DpDecodePolicy, CommonBucketBoundsWorstCaseRows) {
    EXPECT_TRUE(validDsv4DecodeCaptureBuckets({1, 2, 4, 8}, 8, 4));
    EXPECT_FALSE(validDsv4DecodeCaptureBuckets({}, 8, 4));
    EXPECT_FALSE(validDsv4DecodeCaptureBuckets({1, 1, 2}, 8, 4));
    EXPECT_FALSE(validDsv4DecodeCaptureBuckets({1, std::numeric_limits<int>::max()}, 8, 4));
    EXPECT_EQ(dsv4DecodeRequestBucket(3, {1, 2, 4, 8}), 4);
    EXPECT_EQ(dsv4DecodeRequestBucket(9, {1, 2, 4, 8}), 0);
    EXPECT_EQ(dsv4DecodeRequestBucket(0, {1, 2, 4, 8}), 0);
}

TEST(Dsv4DpDecodePolicy, EagerWarmupNeedsNoCapturedCatalog) {
    const auto warmup = dsv4DecodeCandidate(DecodeExecutionPhase::TARGET, 1, 4, {});
    EXPECT_EQ(warmup.mode, DecodeExecutionMode::EAGER);
    EXPECT_EQ(warmup.bucket, 0);
    const auto serving = dsv4DecodeCandidate(DecodeExecutionPhase::TARGET_VERIFY, 4, 4, {1, 2, 4});
    EXPECT_EQ(serving.mode, DecodeExecutionMode::GRAPH);
    EXPECT_EQ(serving.bucket, 4);
    EXPECT_TRUE(allDsv4DecodePeersReady({1, 1, 1, 1}));
    EXPECT_FALSE(allDsv4DecodePeersReady({1, 0, 1, 1}));
}

TEST(Dsv4DpDecodePolicy, PhaseSequenceCapacityAndReadinessMustAgree) {
    Dsv4DecodeRecord target{12, Dsv4DecodeControl::PREPARE_GRAPH, DecodeExecutionPhase::TARGET_VERIFY, 0, 4, 4};
    auto             peer = target;
    peer.value            = 1;
    EXPECT_TRUE(matchingDsv4DecodeInvocation(target, peer));
    peer.phase = DecodeExecutionPhase::DSPARK_COMMIT;
    EXPECT_FALSE(matchingDsv4DecodeInvocation(target, peer));
    peer = target;
    peer.sequence++;
    EXPECT_FALSE(matchingDsv4DecodeInvocation(target, peer));
    peer        = target;
    peer.bucket = 8;
    EXPECT_FALSE(matchingDsv4DecodeInvocation(target, peer));
    peer       = target;
    peer.width = 3;
    EXPECT_FALSE(matchingDsv4DecodeInvocation(target, peer));
    peer         = target;
    peer.control = Dsv4DecodeControl::PREPARE_EAGER;
    EXPECT_FALSE(matchingDsv4DecodeInvocation(target, peer));
    peer = target;
    ++peer.protocol_version;
    EXPECT_FALSE(matchingDsv4DecodeInvocation(target, peer));

    using Status = DecodePreparationStatus;
    EXPECT_EQ(combineDsv4DecodeReadiness(Status::READY, Status::GRAPH_UNAVAILABLE), Status::GRAPH_UNAVAILABLE);
    EXPECT_EQ(combineDsv4DecodeReadiness(Status::GRAPH_UNAVAILABLE, Status::INVALID), Status::INVALID);
}

}  // namespace
}  // namespace rtp_llm
