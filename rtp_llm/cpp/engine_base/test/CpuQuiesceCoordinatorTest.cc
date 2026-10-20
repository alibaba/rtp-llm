#include "rtp_llm/cpp/engine_base/CpuQuiesceCoordinator.h"
#include "rtp_llm/cpp/engine_base/sleep/SleepRoundFence.h"

#include <future>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/csrc/utils/pybind.h>

namespace rtp_llm {
namespace {
namespace py = pybind11;
using namespace std::chrono_literals;

class CpuQuiesceCoordinatorTest: public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!Py_IsInitialized()) {
            interpreter = std::make_unique<py::scoped_interpreter>();
        }
        py::module_::import("torch.distributed");
    }
    static void TearDownTestSuite() {
        interpreter.reset();
    }

    std::vector<std::unique_ptr<CpuQuiesceCoordinator>> makeGroup(int size) {
        // Real TCPStore + Gloo sockets, not a fake reduction. Python is only
        // used to bootstrap native Backends (as in BackendManager startup).
        py::dict scope;
        scope["size"] = size;
        py::exec(R"PY(
import threading
from datetime import timedelta
import torch.distributed as dist
store = dist.TCPStore('127.0.0.1', 0, None, True,
                      timedelta(seconds=5), wait_for_workers=False)
groups = [None] * size
errors = []
def create(rank):
    try:
        # Each process has its own store client in production. Sharing one
        # client here serializes a waiting get against another rank's set.
        client = dist.TCPStore('127.0.0.1', store.port, None, False,
                               timedelta(seconds=5), wait_for_workers=False)
        groups[rank] = dist.ProcessGroupGloo(client, rank, size, timedelta(seconds=5))
    except BaseException as e:
        errors.append(e)
threads = [threading.Thread(target=create, args=(r,)) for r in range(size)]
for t in threads: t.start()
for t in threads: t.join(6)
if any(t.is_alive() for t in threads): raise RuntimeError('Gloo bootstrap hung')
if errors: raise errors[0]
)PY",
                 scope);
        bootstrap_store_ = scope["store"];
        std::vector<std::unique_ptr<CpuQuiesceCoordinator>> result;
        for (auto group : scope["groups"]) {
            result.push_back(std::make_unique<CpuQuiesceCoordinator>(group.cast<c10::intrusive_ptr<c10d::Backend>>()));
        }
        return result;
    }

    std::vector<absl::StatusOr<uint64_t>> reduce(std::vector<std::unique_ptr<CpuQuiesceCoordinator>>& group,
                                                 const std::vector<std::string>&                      tokens,
                                                 const std::vector<uint64_t>&                         rounds) {
        std::vector<std::future<absl::StatusOr<uint64_t>>> pending;
        for (size_t rank = 0; rank < group.size(); ++rank) {
            pending.push_back(std::async(std::launch::async, [&, rank] {
                return group[rank]->targetRound(tokens[rank], rounds[rank], std::chrono::steady_clock::now() + 3s);
            }));
        }
        std::vector<absl::StatusOr<uint64_t>> result;
        // Deliberately KEEP the GIL while waiting. Native quiesce must succeed
        // even if a model forward owns the interpreter on a different thread.
        EXPECT_TRUE(PyGILState_Check());
        for (auto& future : pending) {
            result.push_back(future.get());
        }
        return result;
    }

    static std::unique_ptr<py::scoped_interpreter> interpreter;
    py::object                                     bootstrap_store_;
};
std::unique_ptr<py::scoped_interpreter> CpuQuiesceCoordinatorTest::interpreter;

TEST_F(CpuQuiesceCoordinatorTest, FourRanksAgreeOnMaxWithoutGilOrCuda) {
    auto group = makeGroup(4);
    for (auto& value : reduce(group, {"sleep/1", "sleep/1", "sleep/1", "sleep/1"}, {11, 13, 10, 12})) {
        ASSERT_TRUE(value.ok()) << value.status();
        EXPECT_EQ(*value, 13);
    }
}

TEST_F(CpuQuiesceCoordinatorTest, ZeroRoundAndRepeatedCycles) {
    auto group = makeGroup(2);
    for (size_t cycle = 0; cycle < 5; ++cycle) {
        const auto token = "sleep/" + std::to_string(cycle);
        for (auto& value : reduce(group, {token, token}, {cycle, 2 * cycle})) {
            ASSERT_TRUE(value.ok()) << value.status();
            EXPECT_EQ(*value, 2 * cycle);
        }
        // Only one rank retries: it must use its cached decision rather than
        // launch an unmatched second collective and poison the control group.
        auto repeated = group[0]->targetRound(token, cycle, std::chrono::steady_clock::now() + 100ms);
        ASSERT_TRUE(repeated.ok()) << repeated.status();
        EXPECT_EQ(*repeated, 2 * cycle);
    }
}

TEST_F(CpuQuiesceCoordinatorTest, MismatchedOperationIsRejectedByEveryRank) {
    auto group = makeGroup(2);
    for (auto& value : reduce(group, {"sleep/old", "sleep/new"}, {3, 9})) {
        EXPECT_EQ(value.status().code(), absl::StatusCode::kFailedPrecondition);
    }
    // Identity rejection did not fail transport. A later matching operation
    // can succeed, with no stale decision leaked from the rejected one.
    for (auto& value : reduce(group, {"sleep/next", "sleep/next"}, {4, 10})) {
        ASSERT_TRUE(value.ok()) << value.status();
        EXPECT_EQ(*value, 10);
    }
}

TEST_F(CpuQuiesceCoordinatorTest, TokenLengthIsPartOfIdentity) {
    auto group = makeGroup(2);
    for (auto& value : reduce(group, {"epoch", std::string("epoch\0", 6)}, {2, 3})) {
        EXPECT_EQ(value.status().code(), absl::StatusCode::kFailedPrecondition);
    }
}

TEST_F(CpuQuiesceCoordinatorTest, LargestValidRoundAndFullTokenArePreserved) {
    auto              group = makeGroup(2);
    const std::string token(256, static_cast<char>(255));
    for (auto& value : reduce(group, {token, token}, {0, INT64_MAX})) {
        ASSERT_TRUE(value.ok()) << value.status();
        EXPECT_EQ(*value, INT64_MAX);
    }
}

TEST_F(CpuQuiesceCoordinatorTest, SimultaneousRetryUsesOneDecision) {
    auto       group    = makeGroup(2);
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    auto       first    = std::async(std::launch::async, [&] { return group[0]->targetRound("sleep", 3, deadline); });
    auto       retry    = std::async(std::launch::async, [&] { return group[0]->targetRound("sleep", 3, deadline); });
    auto       peer     = std::async(std::launch::async, [&] { return group[1]->targetRound("sleep", 9, deadline); });
    for (auto* future : {&first, &retry, &peer}) {
        auto value = future->get();
        ASSERT_TRUE(value.ok()) << value.status();
        EXPECT_EQ(*value, 9);
    }
}

TEST_F(CpuQuiesceCoordinatorTest, InvalidInputNeverEntersCollective) {
    auto group    = makeGroup(2);
    auto deadline = std::chrono::steady_clock::now() + 100ms;
    EXPECT_EQ(group[0]->targetRound("", 1, deadline).status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(group[0]->targetRound(std::string(257, 'x'), 1, deadline).status().code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(group[0]->targetRound("x", UINT64_MAX, deadline).status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(group[0]->targetRound("x", 1, std::chrono::steady_clock::now() - 1ms).status().code(),
              absl::StatusCode::kDeadlineExceeded);
}

TEST_F(CpuQuiesceCoordinatorTest, ChangedLocalRoundCannotReuseDecision) {
    auto group  = makeGroup(2);
    auto values = reduce(group, {"sleep", "sleep"}, {3, 8});
    ASSERT_TRUE(values[0].ok());
    EXPECT_EQ(group[0]->targetRound("sleep", 4, std::chrono::steady_clock::now() + 100ms).status().code(),
              absl::StatusCode::kFailedPrecondition);
}

TEST_F(CpuQuiesceCoordinatorTest, MissingRankTimesOutAndFailedTransportIsNotReused) {
    auto       group   = makeGroup(2);
    const auto started = std::chrono::steady_clock::now();
    auto       result  = group[0]->targetRound("missing", 3, started + 200ms);
    EXPECT_EQ(result.status().code(), absl::StatusCode::kUnavailable);
    EXPECT_LT(std::chrono::steady_clock::now() - started, 2s);
    const auto retried = std::chrono::steady_clock::now();
    EXPECT_EQ(group[0]->targetRound("next", 3, retried + 1s).status().code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_LT(std::chrono::steady_clock::now() - retried, 100ms);
}

TEST_F(CpuQuiesceCoordinatorTest, RealCpuDecisionDrivesExistingRoundFenceCatchup) {
    auto            group = makeGroup(2);
    SleepRoundFence fences[2];
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ(fences[0].next().action, SleepRoundFence::Action::RUN);
    }
    for (int i = 0; i < 7; ++i) {
        ASSERT_EQ(fences[1].next().action, SleepRoundFence::Action::RUN);
    }
    auto targets = reduce(group, {"sleep", "sleep"}, {fences[0].freeze(), fences[1].freeze()});
    for (int rank = 0; rank < 2; ++rank) {
        ASSERT_TRUE(targets[rank].ok());
        ASSERT_TRUE(fences[rank].setTarget(*targets[rank]));
        for (int i = 0; i < (rank == 0 ? 4 : 0); ++i) {
            EXPECT_EQ(fences[rank].next().action, SleepRoundFence::Action::RUN);
        }
        auto permit = fences[rank].next();
        EXPECT_EQ(permit.action, SleepRoundFence::Action::QUIESCE);
        fences[rank].finishQuiesce(permit.generation);
        EXPECT_TRUE(fences[rank].wait(100ms));
        EXPECT_TRUE(fences[rank].resume());
        EXPECT_EQ(fences[rank].next().action, SleepRoundFence::Action::RUN);
    }
}

}  // namespace
}  // namespace rtp_llm
