#include "rtp_llm/cpp/engine_base/schedulers/PDFusionScheduleCoordinator.h"
#include "gtest/gtest.h"
#include <chrono>
#include <csignal>
#include <sys/wait.h>
#include <thread>
#include <vector>
#include <unistd.h>
using namespace rtp_llm;
namespace {
using Clock = std::chrono::steady_clock;
struct Child {
    pid_t    pid     = -1;
    int      read_fd = -1;
    int      status  = -1;
    uint64_t digest  = 0;
};
class ProcessGroup {
public:
    ~ProcessGroup() {
        for (auto& c : children) {
            if (c.pid > 0 && c.status == -1) {
                ::kill(c.pid, SIGKILL);
                ::waitpid(c.pid, nullptr, 0);
            }
            if (c.read_fd >= 0) {
                ::close(c.read_fd);
            }
        }
    }
    std::vector<Child> children;
};
uint64_t runRank(int rank, const std::string& id, const std::string& scenario) {
    if (scenario == "late" && rank == 3) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    PDFusionScheduleCoordinator channel(id, rank, scenario == "mismatch" && rank == 3 ? 3 : 2, 150, 600);
    PDFusionGlobalCadence       cadence(2);
    uint64_t                    digest = 0;
    for (int64_t epoch = 1; epoch <= 100; ++epoch) {
        PDFusionPreparedState state;
        if (epoch % 5 == 1) {
            state.ready_prefill = rank == 0;
        }
        if (epoch % 5 == 2) {
            state.ready_prefill = rank == 0;
            state.ready_decode  = rank != 0;
        }
        if (epoch % 5 == 3) {
            state.ready_prefill = 1;
            state.ready_decode  = 10;
        }
        if (epoch % 5 == 4) {
            state.ready_decode = 10;
        }
        if (epoch == 4 && scenario == "stopped" && rank == 3) {
            state.stopped = 1;
        }
        if (epoch == 4 && ((scenario == "peer_exit" && rank == 2) || (scenario == "leader_exit" && rank == 0))) {
            _exit(23);
        }
        if (epoch == 4 && scenario == "timeout" && rank == 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        auto states = channel.exchange(epoch == 4 && scenario == "old_epoch" && rank == 2 ? 3 : epoch,
                                       PDFusionScheduleCoordinator::Phase::PREPARE,
                                       state);
        auto                  plan = cadence.choose(states);
        PDFusionPreparedState committed;
        if (plan == PDFusionPlan::PREFILL) {
            committed.ready_prefill = state.ready_prefill;
        }
        if (plan == PDFusionPlan::DECODE) {
            committed.ready_decode = state.ready_decode;
        }
        if (epoch % 7 == 0) {
            committed = {};
        }  // cancellation after reservation
        if (epoch == 4 && scenario == "commit_exit" && rank == 2) {
            _exit(23);
        }
        if (epoch == 4 && scenario == "commit_timeout" && rank == 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        if (epoch == 4 && scenario == "negative_state" && rank == 2) {
            committed.ready_decode = -1;
        }
        if (epoch == 4 && scenario == "wrong_phase" && rank == 2) {
            channel.exchange(epoch, PDFusionScheduleCoordinator::Phase::PREPARE, committed);
        }
        auto    commits = channel.exchange(epoch, PDFusionScheduleCoordinator::Phase::COMMIT, committed);
        int64_t total   = 0;
        for (const auto& s : commits) {
            total += s.ready_prefill + s.ready_decode;
        }
        cadence.finish(plan, total);
        digest = digest * 31 + static_cast<int64_t>(plan) * 100 + total;
    }
    return digest;
}
void exercise(const std::string& scenario, bool success) {
    ProcessGroup group;
    const auto   id =
        "coord-test-" + std::to_string(::getpid()) + "-" + std::to_string(Clock::now().time_since_epoch().count());
    const int count = scenario == "missing" ? 3 : 4;
    for (int i = 0; i < count; ++i) {
        int pipefd[2];
        ASSERT_EQ(::pipe(pipefd), 0);
        const int   rank = scenario == "duplicate" && i == 3 ? 2 : i;
        const pid_t pid  = ::fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            ::close(pipefd[0]);
            try {
                const auto digest  = runRank(rank, id + (scenario == "wrong_run" && i == 3 ? "-other" : ""), scenario);
                const auto written = ::write(pipefd[1], &digest, sizeof(digest));
                _exit(written == static_cast<ssize_t>(sizeof(digest)) ? 0 : 99);
            } catch (...) {
                _exit(2);
            }
        }
        ::close(pipefd[1]);
        group.children.push_back({pid, pipefd[0]});
    }
    const auto deadline  = Clock::now() + std::chrono::seconds(5);
    size_t     remaining = group.children.size();
    while (remaining && Clock::now() < deadline) {
        for (auto& child : group.children) {
            if (child.status != -1) {
                continue;
            }
            int        status = 0;
            const auto result = ::waitpid(child.pid, &status, WNOHANG);
            ASSERT_GE(result, 0);
            if (result == child.pid) {
                child.status = status;
                --remaining;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    ASSERT_EQ(remaining, 0) << "protocol exceeded bounded group shutdown";
    for (auto& child : group.children) {
        ASSERT_TRUE(WIFEXITED(child.status));
        const auto code = WEXITSTATUS(child.status);
        if (success) {
            ASSERT_EQ(code, 0);
            ASSERT_EQ(::read(child.read_fd, &child.digest, sizeof(child.digest)), sizeof(child.digest));
            EXPECT_EQ(child.digest, group.children.front().digest);
        } else {
            EXPECT_TRUE(code == 2 || code == 23);
        }
    }
}
TEST(PDFusionScheduleCoordinatorTest, FourRankPlanAndCancellationAgreement) {
    exercise("normal", true);
}
TEST(PDFusionScheduleCoordinatorTest, BoundedLateRank) {
    exercise("late", true);
}
TEST(PDFusionScheduleCoordinatorTest, FailuresCloseTheGroup) {
    for (const auto& scenario : {"old_epoch",
                                 "stopped",
                                 "peer_exit",
                                 "leader_exit",
                                 "timeout",
                                 "mismatch",
                                 "duplicate",
                                 "missing",
                                 "wrong_run",
                                 "commit_exit",
                                 "commit_timeout",
                                 "negative_state",
                                 "wrong_phase"}) {
        SCOPED_TRACE(scenario);
        exercise(scenario, false);
    }
}
TEST(PDFusionScheduleCoordinatorTest, GlobalCadenceDoesNotStarveReadyWhenNoDecode) {
    PDFusionGlobalCadence                cadence(16);
    std::array<PDFusionPreparedState, 4> states{};
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::IDLE);
    states[0].waiting = 1;
    states[0].loading = 1;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::IDLE);
    states[0].ready_prefill = 1;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
    cadence.finish(PDFusionPlan::PREFILL, 1);
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
    states[3].ready_decode = 1;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::DECODE);
    cadence.finish(PDFusionPlan::PREFILL, 0);
    EXPECT_EQ(cadence.decodeSincePrefill(), 0);
    for (int i = 0; i < 16; ++i) {
        cadence.finish(PDFusionPlan::DECODE, 1);
    }
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
}
}  // namespace
