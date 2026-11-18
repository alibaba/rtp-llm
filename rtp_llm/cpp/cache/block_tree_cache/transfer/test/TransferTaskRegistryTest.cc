#include <gtest/gtest.h>

#include <future>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"

namespace rtp_llm {
namespace {

using Registry     = TransferTaskRegistry;
using State        = Registry::State;
using ForgetResult = Registry::ForgetResult;

TEST(TransferTaskRegistryTest, MissingAdmissionFenceSurvivesForgetUntilLateArrivalOrExpiry) {
    auto now = Registry::Clock::time_point{};
    Registry registry([&] { return now; });
    const std::vector<std::string> ids{"missing", "late"};
    EXPECT_EQ(registry.queryAndBlockMissing(ids), (std::vector<State>{State::REJECTED, State::REJECTED}));
    now += std::chrono::minutes(30);
    registry.queryAndBlockMissing(ids);
    EXPECT_EQ(registry.forget("late"), ForgetResult::DONE);
    EXPECT_FALSE(registry.begin("late"));
    now += std::chrono::minutes(89);
    EXPECT_EQ(registry.queryLocal(ids), (std::vector<State>{State::REJECTED, State::UNKNOWN}));
    EXPECT_EQ(registry.forget("missing"), ForgetResult::DONE);
    EXPECT_EQ(registry.cleanupExpiredMissing(), 0u);
    now += std::chrono::minutes(1);
    EXPECT_EQ(registry.cleanupExpiredMissing(), 1u);
    EXPECT_EQ(registry.queryLocal(ids).front(), State::UNKNOWN);
    for (uint64_t trace_id = 1; trace_id <= 32; ++trace_id) {
        SCOPED_TRACE(trace_id);
        std::promise<void> start;
        const auto ready = start.get_future().share();
        auto admission = std::async(std::launch::async, [&] {
            ready.wait();
            return registry.begin(trace_id);
        });
        auto query = std::async(std::launch::async, [&] {
            ready.wait();
            return registry.queryAndBlockMissing(std::vector<uint64_t>{trace_id}).front();
        });
        start.set_value();
        const bool admitted = admission.get();
        const auto observed = query.get();
        EXPECT_EQ(observed, admitted ? State::RUNNING : State::REJECTED);
        EXPECT_EQ(registry.queryLocal(std::vector<uint64_t>{trace_id}),
                  (std::vector<State>{admitted ? State::RUNNING : State::UNKNOWN}));
        if (admitted) {
            EXPECT_FALSE(registry.begin(trace_id));
        }
        EXPECT_EQ(registry.forget(trace_id), admitted ? ForgetResult::RUNNING : ForgetResult::DONE);
    }
}

}  // namespace
}  // namespace rtp_llm
