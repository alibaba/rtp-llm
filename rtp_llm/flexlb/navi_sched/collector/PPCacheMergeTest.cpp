#include "aios/navi_sched/collector/PPCacheMergeR.h"
#include "aios/navi_sched/collector/PPCacheShardUpdateData.h"
#include "navi/tester/KernelTesterBuilder.h"
#include "unittest/unittest.h"
#include <future>
#include <thread>
#include <set>

using namespace navi_sched;
class PPCacheMergeTest : public TESTBASE {};

namespace {
PPCacheShardUpdateDataPtr wire(const PPCacheShardUpdate &value)
{
    auto data = std::make_shared<PPCacheShardUpdateData>();
    data->update = value;
    return data;
}
std::shared_ptr<navi::KernelTester> makeShard(const std::shared_ptr<PPCacheMergeR> &merge,
                                              uint32_t shard, uint64_t generation = 1)
{
    navi::KernelTesterBuilder builder;
    return builder.kernel("navi_sched.pp_cache_merge_k")
        .input("update")
        .output("finished")
        .attrs("{\"pp_group\":\"pp-a\",\"shard_id\":" + std::to_string(shard) +
               ",\"shard_count\":4,\"stream_generation\":" + std::to_string(generation) +
               "}")
        .resource(merge)
        .build();
}
PPCacheShardUpdate update(uint32_t shard, uint64_t sequence, bool snapshot = false)
{
    PPCacheShardUpdate result;
    result.shardId = shard;
    result.streamGeneration = 1;
    result.sequence = sequence;
    result.fullSnapshot = snapshot;
    return result;
}
}

TEST_F(PPCacheMergeTest, testFourNativeGraphsIntersectAndRetractOnEof)
{
    std::vector<PPCacheCollectorChanges> events;
    auto merge = std::make_shared<PPCacheMergeR>();
    ASSERT_TRUE(
        merge->initialize("pp-a", 4, [&](const auto &c) { events.push_back(c); }));
    std::vector<std::shared_ptr<navi::KernelTester>> shards;
    for (uint32_t i = 0; i < 4; ++i) {
        auto tester = makeShard(merge, i);
        ASSERT_TRUE(tester);
        ASSERT_EQ(navi::EC_NONE, tester->getErrorCode());
        auto snapshot = update(i, 1, true);
        snapshot.addedFull = {8, 9};
        snapshot.addedTail = {20};
        ASSERT_TRUE(tester->setInput("update", wire(snapshot)));
        ASSERT_TRUE(tester->compute());
        shards.push_back(tester);
        ASSERT_EQ(i == 3 ? 1 : 0, events.size());
    }
    ASSERT_EQ(std::set<uint64_t>({8, 9}),
              std::set<uint64_t>(events.back().addedFull.begin(),
                                 events.back().addedFull.end()));
    ASSERT_EQ(std::vector<uint64_t>({20}), events.back().addedTail);
    ASSERT_TRUE(shards[2]->setInputEof("update"));
    ASSERT_TRUE(shards[2]->compute());
    ASSERT_EQ(2, events.size());
    EXPECT_EQ(2, events.back().removedFull.size());
    EXPECT_EQ(std::vector<uint64_t>({20}), events.back().removedTail);
    navi::DataPtr data;
    bool eof = false;
    ASSERT_TRUE(shards[2]->getOutput("finished", data, eof));
    ASSERT_TRUE(eof);
}

TEST_F(PPCacheMergeTest, testFailureAndCancelFinishOnlyAfterEveryShard)
{
    std::vector<PPCacheCollectorChanges> events;
    auto merge = std::make_shared<PPCacheMergeR>();
    ASSERT_TRUE(
        merge->initialize("pp-a", 4, [&](const auto &c) { events.push_back(c); }));
    std::vector<std::shared_ptr<navi::KernelTester>> shards;
    for (uint32_t i = 0; i < 4; ++i) {
        auto tester = makeShard(merge, i);
        ASSERT_TRUE(tester);
        ASSERT_EQ(navi::EC_NONE, tester->getErrorCode());
        ASSERT_TRUE(tester->setInput("update", wire(update(i, 1, true))));
        ASSERT_TRUE(tester->compute());
        shards.push_back(tester);
    }
    ASSERT_TRUE(merge->beginRequest({10, 1}).accepted);
    for (uint32_t i = 0; i < 4; ++i) {
        auto terminal = update(i, 2);
        terminal.terminals = {{{10, 1},
                               i == 1   ? PPShardRequestStatus::FAILED
                               : i == 2 ? PPShardRequestStatus::CANCELLED
                                        : PPShardRequestStatus::SUCCEEDED}};
        ASSERT_TRUE(shards[i]->setInput("update", wire(terminal)));
        ASSERT_TRUE(shards[i]->compute());
        ASSERT_EQ(i == 3 ? 1 : 0, events.size());
    }
    ASSERT_EQ(1, events.back().completedRequests.size());
    EXPECT_FALSE(events.back().completedRequests[0].succeeded);
    EXPECT_EQ(PPShardRequestStatus::FAILED,
              events.back().completedRequests[0].shardStatuses[1]);
    EXPECT_EQ(PPShardRequestStatus::CANCELLED,
              events.back().completedRequests[0].shardStatuses[2]);
}

TEST_F(PPCacheMergeTest, testGapAbortsGraphAndRequestsFullResync)
{
    auto merge = std::make_shared<PPCacheMergeR>();
    ASSERT_TRUE(merge->initialize("pp-a", 4, [](const auto &) {}));
    auto tester = makeShard(merge, 0);
    ASSERT_TRUE(tester);
    ASSERT_EQ(navi::EC_NONE, tester->getErrorCode());
    ASSERT_TRUE(tester->setInput("update", wire(update(0, 1, true))));
    ASSERT_TRUE(tester->compute());
    ASSERT_TRUE(tester->setInput("update", wire(update(0, 3))));
    tester->compute();
    ASSERT_EQ(navi::EC_ABORT, tester->getErrorCode());
    tester.reset();
    auto replacement = makeShard(merge, 0, 2);
    ASSERT_TRUE(replacement);
    ASSERT_EQ(navi::EC_NONE, replacement->getErrorCode());
}

TEST_F(PPCacheMergeTest, testConcurrentMutationCannotOvertakeBlockedCallback)
{
    auto merge = std::make_shared<PPCacheMergeR>();
    std::promise<void> entered, release;
    auto ready = entered.get_future();
    auto gate = release.get_future();
    std::vector<PPCacheCollectorChanges> events;
    ASSERT_TRUE(merge->initialize("pp-a", 1, [&](const auto &c) {
        events.push_back(c);
        if (!c.addedFull.empty()) {
            entered.set_value();
            gate.wait();
        }
    }));
    ASSERT_TRUE(merge->connect("pp-a", 0, 1, 1).accepted);
    auto snapshot = update(0, 1, true);
    snapshot.addedFull = {8};
    std::thread writer([&]() { merge->apply(snapshot); });
    const bool callbackEntered =
        ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    auto removal = update(0, 2);
    removal.removedFull = {8};
    const auto result = merge->apply(removal);
    release.set_value();
    writer.join();
    ASSERT_TRUE(callbackEntered);
    ASSERT_TRUE(result.accepted);
    ASSERT_EQ(2, events.size());
    EXPECT_EQ(std::vector<uint64_t>({8}), events[0].addedFull);
    EXPECT_EQ(std::vector<uint64_t>({8}), events[1].removedFull);
}
