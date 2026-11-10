#include <gtest/gtest.h>

#include <future>
#include <limits>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"

namespace rtp_llm {
namespace {

using Registry = TransferTaskRegistry;
using State    = Registry::State;

TEST(TransferTaskRegistryTest, RejectsEmptyAndDuplicateExecutionIds) {
    Registry registry;
    EXPECT_EQ(registry.size(), 0u);
    EXPECT_FALSE(registry.begin(""));
    EXPECT_EQ(registry.size(), 0u);
    EXPECT_TRUE(registry.begin("first"));
    EXPECT_FALSE(registry.begin("first"));
    EXPECT_TRUE(registry.begin("second"));
    EXPECT_EQ(registry.size(), 2u);
    EXPECT_EQ(registry.query(registry.epoch(), {"first", "second", ""}),
              (std::vector<State>{State::RUNNING, State::RUNNING, State::UNKNOWN}));
}

TEST(TransferTaskRegistryTest, ExplicitSafetyProofCompletesEvenAfterUnsafeFailure) {
    Registry registry;
    ASSERT_TRUE(registry.begin("failed-execution"));
    ASSERT_TRUE(registry.finish("failed-execution", false));
    EXPECT_EQ(registry.query(registry.epoch(), {"failed-execution"}), (std::vector<State>{State::RUNNING}));
    // Business failure remains a failure; a later successful drain only proves
    // that the failed execution no longer accesses its buffers.
    EXPECT_TRUE(registry.finish("failed-execution", true));
    EXPECT_EQ(registry.query(registry.epoch(), {"failed-execution"}), (std::vector<State>{State::BUFFER_SAFE}));
    EXPECT_FALSE(registry.begin("failed-execution"));
}

TEST(TransferTaskRegistryTest, SafeCompletionIsMonotonicAndIdempotent) {
    Registry registry;
    ASSERT_TRUE(registry.begin("execution"));
    ASSERT_TRUE(registry.finish("execution", true));
    EXPECT_TRUE(registry.finish("execution", true));
    EXPECT_TRUE(registry.finish("execution", false));
    EXPECT_EQ(registry.query(registry.epoch(), {"execution"}), (std::vector<State>{State::BUFFER_SAFE}));
}

TEST(TransferTaskRegistryTest, MissingEvidenceDoesNotCreateOrCompleteAnExecution) {
    Registry registry;
    EXPECT_FALSE(registry.finish("missing", true));
    EXPECT_FALSE(registry.finish("missing", false));
    EXPECT_FALSE(registry.finish("", true));
    EXPECT_EQ(registry.size(), 0u);
    ASSERT_TRUE(registry.begin("unknown-completion"));
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(registry.finish("unknown-completion", false));
        EXPECT_EQ(registry.query(registry.epoch(), {"missing", "unknown-completion", ""}),
                  (std::vector<State>{State::UNKNOWN, State::RUNNING, State::UNKNOWN}));
    }
    EXPECT_TRUE(registry.begin("missing"));
}

TEST(TransferTaskRegistryTest, QueryRequiresCurrentWorkerEpochAndPreservesOrder) {
    Registry registry;
    Registry restarted;
    ASSERT_FALSE(registry.epoch().empty());
    ASSERT_NE(registry.epoch(), restarted.epoch());
    ASSERT_TRUE(registry.begin("safe"));
    ASSERT_TRUE(registry.finish("safe", true));
    ASSERT_TRUE(registry.begin("running"));
    ASSERT_TRUE(restarted.begin("safe"));
    ASSERT_TRUE(restarted.finish("safe", true));

    const std::vector<std::string> ids{"running", "missing", "safe", "safe", ""};
    EXPECT_EQ(registry.query(registry.epoch(), ids),
              (std::vector<State>{
                  State::RUNNING, State::UNKNOWN, State::BUFFER_SAFE, State::BUFFER_SAFE, State::UNKNOWN}));
    for (const auto& wrong_epoch : {std::string{}, restarted.epoch(), registry.epoch() + "-invalid"}) {
        EXPECT_EQ(registry.query(wrong_epoch, ids), std::vector<State>(ids.size(), State::UNKNOWN));
    }
    EXPECT_EQ(restarted.query(registry.epoch(), {"safe"}), (std::vector<State>{State::UNKNOWN}));
    EXPECT_TRUE(registry.query(registry.epoch(), std::vector<std::string>{}).empty());
}

TEST(TransferTaskRegistryTest, LocalStringQueryPreservesOrderWithoutCreatingRecords) {
    Registry registry;
    ASSERT_TRUE(registry.begin("safe"));
    ASSERT_TRUE(registry.finish("safe", true));
    ASSERT_TRUE(registry.begin("running"));
    const Registry& local = registry;
    const std::vector<std::string> ids{"running", "missing", "safe", "safe", ""};
    const std::vector<State> expected{
        State::RUNNING, State::UNKNOWN, State::BUFFER_SAFE, State::BUFFER_SAFE, State::UNKNOWN};
    EXPECT_EQ(local.queryLocal(ids), expected);
    EXPECT_EQ(local.queryLocal(ids), expected);
    EXPECT_EQ(local.size(), 2u);
    EXPECT_TRUE(local.queryLocal(std::vector<std::string>{}).empty());
    EXPECT_TRUE(registry.begin("missing"));
}

TEST(TransferTaskRegistryTest, LocalNumericQueryRejectsZeroAndObservesMonotonicProof) {
    Registry registry;
    const uint64_t largest = std::numeric_limits<uint64_t>::max();
    ASSERT_TRUE(registry.begin(uint64_t{42}));
    ASSERT_TRUE(registry.begin(largest));
    ASSERT_TRUE(registry.begin("0"));
    ASSERT_TRUE(registry.finish("0", true));
    const std::vector<uint64_t> ids{0, 42, 43, largest, 42};
    EXPECT_EQ(registry.queryLocal(ids),
              (std::vector<State>{State::UNKNOWN, State::RUNNING, State::UNKNOWN, State::RUNNING, State::RUNNING}));
    ASSERT_TRUE(registry.finish(uint64_t{42}, true));
    ASSERT_TRUE(registry.finish(uint64_t{42}, false));
    EXPECT_EQ(registry.queryLocal(ids),
              (std::vector<State>{
                  State::UNKNOWN, State::BUFFER_SAFE, State::UNKNOWN, State::RUNNING, State::BUFFER_SAFE}));
    EXPECT_EQ(registry.size(), 3u);
    EXPECT_TRUE(registry.queryLocal(std::vector<uint64_t>{}).empty());
    EXPECT_TRUE(registry.begin(uint64_t{43}));
}

TEST(TransferTaskRegistryTest, LocalQueryUsesThisInstanceWithoutWeakeningEpochChecks) {
    Registry first;
    Registry second;
    ASSERT_TRUE(first.begin("execution"));
    ASSERT_TRUE(first.finish("execution", true));
    ASSERT_TRUE(second.begin("execution"));
    const std::vector<std::string> ids{"execution"};
    EXPECT_EQ(first.queryLocal(ids), (std::vector<State>{State::BUFFER_SAFE}));
    EXPECT_EQ(second.queryLocal(ids), (std::vector<State>{State::RUNNING}));
    EXPECT_EQ(first.query(first.epoch(), ids), first.queryLocal(ids));
    EXPECT_EQ(second.query(second.epoch(), ids), second.queryLocal(ids));
    EXPECT_EQ(first.query(second.epoch(), ids), (std::vector<State>{State::UNKNOWN}));
    EXPECT_EQ(second.query(first.epoch(), ids), (std::vector<State>{State::UNKNOWN}));
    EXPECT_EQ(first.size(), 1u);
    EXPECT_EQ(second.size(), 1u);
}

TEST(TransferTaskRegistryTest, GrowthBeyondFormerCapacityRetainsEveryExecution) {
    Registry registry;
    constexpr size_t count = 8193;
    std::vector<uint64_t> ids;
    std::vector<State>    expected;
    ids.reserve(count);
    expected.reserve(count);
    for (uint64_t id = 1; id <= count; ++id) {
        ASSERT_TRUE(registry.begin(id)) << id;
        const bool safe = id % 2 == 0;
        ASSERT_TRUE(registry.finish(id, safe)) << id;
        ids.push_back(id);
        expected.push_back(safe ? State::BUFFER_SAFE : State::RUNNING);
        EXPECT_EQ(registry.size(), id);
    }
    EXPECT_EQ(registry.query(registry.epoch(), ids), expected);
    for (const auto id : ids) {
        EXPECT_FALSE(registry.begin(id)) << id;
    }
    EXPECT_EQ(registry.size(), count);
    EXPECT_EQ(registry.query(registry.epoch(), ids), expected);
}

TEST(TransferTaskRegistryTest, ConcurrentDuplicateSubmissionHasExactlyOneOwner) {
    Registry                      registry;
    std::promise<void>             start;
    const auto                    ready = start.get_future().share();
    std::vector<std::future<bool>> attempts;
    for (int i = 0; i < 16; ++i) {
        attempts.push_back(std::async(std::launch::async, [&] {
            ready.wait();
            return registry.begin("shared-execution");
        }));
    }
    start.set_value();
    size_t accepted = 0;
    for (auto& attempt : attempts) {
        accepted += attempt.get();
    }
    EXPECT_EQ(accepted, 1u);
    EXPECT_EQ(registry.size(), 1u);
    EXPECT_EQ(registry.query(registry.epoch(), {"shared-execution"}), (std::vector<State>{State::RUNNING}));
}

TEST(TransferTaskRegistryTest, ConcurrentIndependentSubmissionsAllRemainQueryable) {
    Registry                      registry;
    std::promise<void>             start;
    const auto                    ready = start.get_future().share();
    std::vector<std::future<bool>> attempts;
    std::vector<std::string>       ids;
    for (int i = 0; i < 16; ++i) {
        ids.push_back("execution-" + std::to_string(i));
        attempts.push_back(std::async(std::launch::async, [&, id = ids.back()] {
            ready.wait();
            const bool accepted = registry.begin(id);
            EXPECT_GE(registry.size(), 1u);
            EXPECT_LE(registry.size(), 16u);
            return accepted;
        }));
    }
    start.set_value();
    const auto states = [&] {
        for (size_t i = 0; i < attempts.size(); ++i) {
            const bool accepted = attempts[i].get();
            EXPECT_TRUE(accepted);
            EXPECT_EQ(registry.query(registry.epoch(), {ids[i]}),
                      (std::vector<State>{accepted ? State::RUNNING : State::UNKNOWN}));
        }
        return registry.query(registry.epoch(), ids);
    }();
    size_t running = 0;
    for (const auto state : states) {
        running += state == State::RUNNING;
    }
    EXPECT_EQ(running, ids.size());
    EXPECT_EQ(registry.size(), ids.size());
}

TEST(TransferTaskRegistryTest, ConcurrentFinishesPreserveProofAndRecordCount) {
    Registry registry;
    ASSERT_TRUE(registry.begin("execution"));
    std::promise<void>             start;
    const auto                    ready = start.get_future().share();
    std::vector<std::future<bool>> attempts;
    for (int i = 0; i < 16; ++i) {
        attempts.push_back(std::async(std::launch::async, [&, safe = i % 2 == 0] {
            ready.wait();
            const bool present = registry.finish("execution", safe);
            EXPECT_EQ(registry.size(), 1u);
            return present;
        }));
    }
    start.set_value();
    for (auto& attempt : attempts) {
        EXPECT_TRUE(attempt.get());
    }
    EXPECT_EQ(registry.query(registry.epoch(), {"execution"}), (std::vector<State>{State::BUFFER_SAFE}));
    EXPECT_FALSE(registry.begin("execution"));
    EXPECT_EQ(registry.size(), 1u);
}

TEST(TransferTaskRegistryTest, NumericTracesReuseExistingIdentityAndRejectZero) {
    Registry       registry;
    const uint64_t trace_id = 42;
    const uint64_t largest  = std::numeric_limits<uint64_t>::max();
    ASSERT_TRUE(registry.begin(trace_id));
    EXPECT_FALSE(registry.begin(trace_id));
    EXPECT_FALSE(registry.begin(std::to_string(trace_id)));
    ASSERT_TRUE(registry.begin(largest));
    ASSERT_TRUE(registry.finish(trace_id, false));
    EXPECT_EQ(registry.query(registry.epoch(), std::vector<uint64_t>{trace_id}),
              (std::vector<State>{State::RUNNING}));
    ASSERT_TRUE(registry.finish(trace_id, true));
    EXPECT_FALSE(registry.finish(uint64_t{43}, true));

    // The numeric overload must never use the string record "0" as evidence.
    ASSERT_TRUE(registry.begin("0"));
    ASSERT_TRUE(registry.finish("0", true));
    EXPECT_FALSE(registry.begin(uint64_t{0}));
    EXPECT_FALSE(registry.finish(uint64_t{0}, true));
    EXPECT_EQ(registry.query(registry.epoch(), std::vector<uint64_t>{0, trace_id, 43, largest, trace_id}),
              (std::vector<State>{
                  State::UNKNOWN, State::BUFFER_SAFE, State::UNKNOWN, State::RUNNING, State::BUFFER_SAFE}));
    EXPECT_EQ(registry.query("wrong-epoch", std::vector<uint64_t>{trace_id, largest}),
              (std::vector<State>{State::UNKNOWN, State::UNKNOWN}));
    EXPECT_TRUE(registry.query(registry.epoch(), std::vector<uint64_t>{}).empty());
}

TEST(TransferTaskRegistryTest, NumericAndKvcmIdentitiesUseIndependentRegistries) {
    Registry transfer;
    Registry kvcm;
    ASSERT_TRUE(transfer.begin(uint64_t{42}));
    ASSERT_TRUE(kvcm.begin("42"));
    ASSERT_TRUE(transfer.finish(uint64_t{42}, true));
    EXPECT_EQ(transfer.query(transfer.epoch(), std::vector<uint64_t>{42}),
              (std::vector<State>{State::BUFFER_SAFE}));
    EXPECT_EQ(kvcm.query(kvcm.epoch(), std::vector<std::string>{"42"}),
              (std::vector<State>{State::RUNNING}));
}

}  // namespace
}  // namespace rtp_llm
