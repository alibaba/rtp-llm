#include "rtp_llm/cpp/engine_base/schedulers/PDFusionScheduleCoordinator.h"
#include "gtest/gtest.h"
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <stdexcept>
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
uint64_t runRank(int rank, const std::string& id, const std::string& scenario, int rank_count = 4, int mtp_tokens = 0) {
    if (scenario == "late" && rank == rank_count - 1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    PDFusionCadenceConfig config;
    if (scenario.find("adaptive") == 0) {
        config.adaptive = 1;
    }
    auto peer_config = config;
    if (rank == rank_count - 1) {
        if (scenario == "adaptive_policy_mismatch")
            peer_config.adaptive = 0;
        if (scenario == "adaptive_window_mismatch")
            peer_config.decode_window = 32;
        if (scenario == "adaptive_water_mismatch")
            peer_config.low_water_pct = 60;
        if (scenario == "adaptive_wait_mismatch")
            peer_config.prefill_max_wait_ms = 500;
        if (scenario == "adaptive_share_mismatch")
            peer_config.min_decode_share_pct = 30;
    }
    PDFusionScheduleCoordinator channel(
        id,
        rank,
        scenario == "adaptive_legacy_ratio_ignored" ? rank - 1 :
                                                      (scenario == "mismatch" && rank == rank_count - 1 ? 3 : 2),
        150,
        600,
        scenario == "rank_count_mismatch" && rank == rank_count - 1 ? (rank_count == 2 ? 4 : 2) : rank_count,
        scenario == "mtp_mismatch" && rank == rank_count - 1 ? (mtp_tokens + 1) % 5 : mtp_tokens,
        peer_config);
    PDFusionGlobalCadence cadence(2, rank_count, config);
    uint64_t              digest = 0;
    for (int64_t epoch = 1; epoch <= 100; ++epoch) {
        PDFusionPreparedState state;
        state.last_model_us = 1000 + rank * 100 + (epoch % 5) * 100;
        if (epoch == 4 && scenario == "negative_model_cost" && rank == rank_count - 1)
            state.last_model_us = -1;
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
        if (scenario == "adaptive_drain" && epoch >= 51) {
            state.ready_decode  = 10;
            state.ready_prefill = epoch % 5 == 1 && rank == 0;
            state.draining      = rank == rank_count - 1;
        }
        if (epoch == 4 && scenario == "stopped" && rank == rank_count - 1) {
            state.stopped = 1;
        }
        if (epoch == 4
            && ((scenario == "peer_exit" && rank == rank_count - 1) || (scenario == "leader_exit" && rank == 0))) {
            _exit(23);
        }
        if (epoch == 4 && scenario == "timeout" && rank == rank_count - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        auto states = channel.exchange(epoch == 4 && scenario == "old_epoch" && rank == rank_count - 1 ? 3 : epoch,
                                       PDFusionScheduleCoordinator::Phase::PREPARE,
                                       state);
        auto plan = cadence.choose(states);
        if (scenario == "adaptive_drain" && epoch >= 51) {
            if (cadence.decision().drain_mask != (1 << (rank_count - 1))
                || (epoch % 5 == 1 && plan != PDFusionPlan::PREFILL)) {
                throw std::runtime_error("drain signal failed cross-rank propagation");
            }
        }
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
        if (epoch == 4 && scenario == "commit_exit" && rank == rank_count - 1) {
            _exit(23);
        }
        if (epoch == 4 && scenario == "commit_timeout" && rank == rank_count - 1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        if (epoch == 4 && scenario == "negative_state" && rank == rank_count - 1) {
            committed.ready_decode = -1;
        }
        if (epoch == 4 && scenario == "wrong_phase" && rank == rank_count - 1) {
            channel.exchange(epoch, PDFusionScheduleCoordinator::Phase::PREPARE, committed);
        }
        auto    commits = channel.exchange(epoch, PDFusionScheduleCoordinator::Phase::COMMIT, committed);
        int64_t total   = 0;
        for (const auto& s : commits) {
            total += s.ready_prefill + s.ready_decode;
        }
        cadence.finish(plan, commits);
        digest = digest * 31 + static_cast<int64_t>(plan) * 100 + total + cadence.decision().decode_debt_us
                 + cadence.prefillAdmissionAllowed();
    }
    return digest;
}
void exercise(const std::string& scenario, bool success, int rank_count = 4, int mtp_tokens = 0) {
    ProcessGroup group;
    const auto   id =
        "coord-test-" + std::to_string(::getpid()) + "-" + std::to_string(Clock::now().time_since_epoch().count());
    const int count = rank_count - (scenario == "missing" ? 1 : 0);
    for (int i = 0; i < count; ++i) {
        int pipefd[2];
        ASSERT_EQ(::pipe(pipefd), 0);
        const int   rank = scenario == "duplicate" && i == rank_count - 1 ? (rank_count == 2 ? 0 : rank_count - 2) : i;
        const pid_t pid  = ::fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            ::close(pipefd[0]);
            try {
                const auto digest  = runRank(rank,
                                            id + (scenario == "wrong_run" && i == rank_count - 1 ? "-other" : ""),
                                            scenario,
                                            rank_count,
                                            mtp_tokens);
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
                                 "negative_model_cost",
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

TEST(PDFusionScheduleCoordinatorTest, TwoRankMtpPlanCancellationAndLateArrival) {
    exercise("normal", true, 2, 1);
    exercise("late", true, 2, 1);
}
TEST(PDFusionScheduleCoordinatorTest, TwoRankMtpFailuresCloseTheGroup) {
    for (const auto& scenario : {"old_epoch",
                                 "stopped",
                                 "peer_exit",
                                 "leader_exit",
                                 "timeout",
                                 "mismatch",
                                 "rank_count_mismatch",
                                 "mtp_mismatch",
                                 "duplicate",
                                 "missing",
                                 "wrong_run",
                                 "commit_exit",
                                 "commit_timeout",
                                 "negative_state",
                                 "wrong_phase"}) {
        SCOPED_TRACE(scenario);
        exercise(scenario, false, 2, 1);
    }
}
TEST(PDFusionScheduleCoordinatorTest, InvalidRankCountAndMtpWidthFailBeforeConnect) {
    EXPECT_THROW(PDFusionScheduleCoordinator("invalid", 0, 7, 150, 600, 3, 1), std::runtime_error);
    EXPECT_THROW(PDFusionScheduleCoordinator("invalid", 2, 7, 150, 600, 2, 1), std::runtime_error);
    EXPECT_THROW(PDFusionScheduleCoordinator("invalid", 0, 7, 150, 600, 2, 5), std::runtime_error);
}

PDFusionCadenceConfig adaptiveConfig(int64_t window = 64) {
    return {1, window, 50, 1000};
}
std::array<PDFusionPreparedState, 4> decodeStates(int64_t count) {
    std::array<PDFusionPreparedState, 4> states{};
    for (auto& state : states)
        state.ready_decode = count;
    return states;
}
TEST(PDFusionScheduleCoordinatorTest, AdaptiveAllReadyBypassesRatioAndUsesActiveRanks) {
    for (int ranks : {2, 4}) {
        PDFusionGlobalCadence cadence(1000000, ranks, adaptiveConfig());
        cadence.finish(PDFusionPlan::PREFILL, 1);
        auto states = decodeStates(128);
        for (int rank = 0; rank < ranks; ++rank)
            states[rank].ready_prefill = 1;
        EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
        EXPECT_STREQ(cadence.decision().reason, "all_ready");
        EXPECT_EQ(cadence.decision().low_decode_mask, 0);
        EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
    }
}
TEST(PDFusionScheduleCoordinatorTest, AdaptivePartialPrefillAccumulatesBeyondRatio) {
    PDFusionGlobalCadence cadence(7, 4, adaptiveConfig());
    auto                  committed = decodeStates(128);
    auto                  states    = committed;
    states[0].ready_prefill         = 1;
    states[1].waiting               = 5;
    states[2].loading               = 1;
    for (int i = 0; i < 80; ++i) {
        ASSERT_EQ(cadence.choose(states), PDFusionPlan::DECODE);
        EXPECT_STREQ(cadence.decision().reason, "accumulate");
        cadence.finish(PDFusionPlan::DECODE, committed);
    }
}
TEST(PDFusionScheduleCoordinatorTest, UntimedPrefillKeepsWaitingDespiteAgeUntilAllActiveRanksReady) {
    for (int ranks : {2, 4}) {
        auto config                = adaptiveConfig();
        config.prefill_max_wait_ms = 0;
        PDFusionGlobalCadence cadence(7, ranks, config);
        auto                  states = decodeStates(128);
        states[0].ready_prefill      = 1;
        for (int epoch = 0; epoch < 200; ++epoch) {
            states[0].oldest_ready_us = 1000000LL * (epoch + 1);
            EXPECT_EQ(cadence.choose(states), PDFusionPlan::DECODE);
            EXPECT_STREQ(cadence.decision().reason, "accumulate");
            cadence.finish(PDFusionPlan::DECODE, decodeStates(128));
        }
        for (int rank = 1; rank < ranks; ++rank)
            states[rank].ready_prefill = 1;
        EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
        EXPECT_STREQ(cadence.decision().reason, "all_ready");
    }
}
TEST(PDFusionScheduleCoordinatorTest, UntimedPrefillFlushesOnLowDecodeOrExplicitDrainNotEmptyWaitingQueues) {
    for (int ranks : {2, 4}) {
        PDFusionCadenceConfig config;
        config.adaptive = 1;
        EXPECT_EQ(config.prefill_max_wait_ms, 0);
        PDFusionGlobalCadence cadence(7, ranks, config);
        auto                  states = decodeStates(128);
        cadence.finish(PDFusionPlan::DECODE, states);
        states[0].ready_prefill   = 1;
        states[0].oldest_ready_us = 600000000;
        EXPECT_EQ(cadence.choose(states), PDFusionPlan::DECODE);
        states[ranks - 1].draining = 1;
        EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
        EXPECT_STREQ(cadence.decision().reason, "drain");
        EXPECT_EQ(cadence.decision().drain_mask, 1 << (ranks - 1));
        EXPECT_EQ(cadence.decision().low_decode_mask, 0);
        states[ranks - 1].draining = 0;
        states[1].ready_decode     = 64;
        EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
        EXPECT_STREQ(cadence.decision().reason, "decode_low");
        states                     = {};
        states[ranks - 1].draining = 1;
        EXPECT_EQ(cadence.choose(states), PDFusionPlan::IDLE);
        states[0].ready_prefill = 1;
        EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
        states[0].draining = 2;
        EXPECT_THROW(cadence.choose(states), std::runtime_error);
    }
}
TEST(PDFusionScheduleCoordinatorTest, UntimedDrainProtocolAgreesForTwoFourRanksAndMtp) {
    exercise("adaptive_drain", true);
    for (int width = 0; width <= 4; ++width)
        exercise("adaptive_drain", true, 2, width);
}
TEST(PDFusionScheduleCoordinatorTest, AdaptiveLowWaterIncludesEqualityAndAnyRank) {
    PDFusionGlobalCadence cadence(7, 4, adaptiveConfig());
    auto                  states = decodeStates(128);
    cadence.finish(PDFusionPlan::DECODE, states);
    states[0].ready_prefill = 1;
    states[2].ready_decode  = 65;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::DECODE);
    states[2].ready_decode = 64;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
    EXPECT_EQ(cadence.decision().low_decode_mask, 4);
    EXPECT_STREQ(cadence.decision().reason, "decode_low");
    states[2].ready_decode = 0;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
    states[0].ready_prefill = 0;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::DECODE);
}
TEST(PDFusionScheduleCoordinatorTest, AdaptiveTimeoutBoundaryAndColdStart) {
    PDFusionGlobalCadence cadence(7, 4, adaptiveConfig());
    auto                  states = decodeStates(8);
    states[0].ready_prefill      = 1;
    states[0].oldest_ready_us    = 999999;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::DECODE);
    EXPECT_EQ(cadence.decision().decode_peak[0], 8);
    states[0].oldest_ready_us = 1000000;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
    EXPECT_STREQ(cadence.decision().reason, "prefill_timeout");
    states            = {};
    states[0].waiting = 1;
    states[0].loading = 1;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::IDLE);
    states[0].ready_prefill = 1;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
}
TEST(PDFusionScheduleCoordinatorTest, AdaptiveWindowCountsCompletedRealDecodeAndPreservesPrefill) {
    PDFusionGlobalCadence cadence(7, 4, adaptiveConfig());
    cadence.finish(PDFusionPlan::DECODE, decodeStates(128));
    auto low = decodeStates(32);
    for (int i = 0; i < 63; ++i)
        cadence.finish(PDFusionPlan::DECODE, low);
    low[0].ready_prefill = 1;
    EXPECT_EQ(cadence.choose(low), PDFusionPlan::PREFILL);
    EXPECT_EQ(cadence.decision().decode_peak[0], 128);
    std::array<PDFusionPreparedState, 4> cancelled{};
    cadence.finish(PDFusionPlan::DECODE, cancelled);
    cancelled[0].ready_prefill = 1;
    cadence.finish(PDFusionPlan::PREFILL, cancelled);
    EXPECT_EQ(cadence.choose(low), PDFusionPlan::PREFILL);
    EXPECT_EQ(cadence.decision().decode_peak[0], 128);
    cadence.finish(PDFusionPlan::DECODE, decodeStates(32));
    EXPECT_EQ(cadence.choose(low), PDFusionPlan::DECODE);
    EXPECT_EQ(cadence.decision().decode_peak[0], 32);
}
TEST(PDFusionScheduleCoordinatorTest, AdaptiveFakeRanksContributeZeroAndReservationsDoNotBecomeHistory) {
    PDFusionGlobalCadence cadence(7, 4, adaptiveConfig(1));
    auto                  states = decodeStates(128);
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::DECODE);
    auto committed            = decodeStates(4);
    committed[3].ready_decode = 0;
    cadence.finish(PDFusionPlan::DECODE, committed);
    states                  = decodeStates(3);
    states[0].ready_prefill = 1;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::DECODE);
    EXPECT_EQ(cadence.decision().decode_peak[0], 4);
    EXPECT_EQ(cadence.decision().decode_peak[3], 0);
    states[3].ready_decode = 0;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
}
class CadenceEnvScope {
public:
    explicit CadenceEnvScope(const char* name):
        name_(name), present_(std::getenv(name) != nullptr), value_(present_ ? std::getenv(name) : "") {
        ::unsetenv(name);
    }
    ~CadenceEnvScope() {
        if (present_)
            ::setenv(name_, value_.c_str(), 1);
        else
            ::unsetenv(name_);
    }

private:
    const char* name_;
    bool        present_;
    std::string value_;
};
TEST(PDFusionScheduleCoordinatorTest, AdaptiveEnvironmentDefaultsAndValidation) {
    CadenceEnvScope flag("PDFUSION_CADENCE_ADAPTIVE"), window("PDFUSION_CADENCE_DECODE_WINDOW"),
        water("PDFUSION_CADENCE_LOW_WATER_PCT"), wait("PDFUSION_CADENCE_PREFILL_MAX_WAIT_MS"),
        share("PDFUSION_CADENCE_MIN_DECODE_SHARE_PCT");
    EXPECT_EQ(PDFusionCadenceConfig::fromEnv().adaptive, 0);
    ::setenv("PDFUSION_CADENCE_DECODE_WINDOW", "invalid", 1);
    EXPECT_EQ(PDFusionCadenceConfig::fromEnv().adaptive, 0);
    ::setenv("PDFUSION_CADENCE_ADAPTIVE", "1", 1);
    EXPECT_THROW(PDFusionCadenceConfig::fromEnv(), std::runtime_error);
    ::setenv("PDFUSION_CADENCE_DECODE_WINDOW", "32", 1);
    EXPECT_EQ(PDFusionCadenceConfig::fromEnv().decode_window, 32);
    EXPECT_EQ(PDFusionCadenceConfig::fromEnv().min_decode_share_pct, 25);
    for (const auto& name : {"PDFUSION_CADENCE_ADAPTIVE",
                             "PDFUSION_CADENCE_DECODE_WINDOW",
                             "PDFUSION_CADENCE_LOW_WATER_PCT",
                             "PDFUSION_CADENCE_PREFILL_MAX_WAIT_MS",
                             "PDFUSION_CADENCE_MIN_DECODE_SHARE_PCT"}) {
        const std::string old = std::getenv(name) ? std::getenv(name) : "";
        for (const auto& value : {"", "-1", "1.5", "99999999999999999999999999"}) {
            ::setenv(name, value, 1);
            EXPECT_THROW(PDFusionCadenceConfig::fromEnv(), std::runtime_error) << name << '=' << value;
        }
        if (old.empty())
            ::unsetenv(name);
        else
            ::setenv(name, old.c_str(), 1);
    }
}
TEST(PDFusionScheduleCoordinatorTest, AdaptiveRankAgreementMtpAndConfigurationMismatch) {
    exercise("adaptive", true);
    for (int width = 0; width <= 4; ++width)
        exercise("adaptive", true, 2, width);
    for (const auto& scenario : {"adaptive_policy_mismatch",
                                 "adaptive_window_mismatch",
                                 "adaptive_water_mismatch",
                                 "adaptive_wait_mismatch",
                                 "adaptive_share_mismatch"}) {
        SCOPED_TRACE(scenario);
        exercise(scenario, false);
        exercise(scenario, false, 2, 1);
    }
}

TEST(PDFusionScheduleCoordinatorTest, AdaptiveAdmissionPacingRepaysActualGlobalCostBeforeNewAdmission) {
    for (int ranks : {2, 4}) {
        PDFusionGlobalCadence cadence(1000000, ranks, adaptiveConfig());
        auto                  states  = decodeStates(128);
        auto                  prefill = std::array<PDFusionPreparedState, 4>{};
        for (int i = 0; i < ranks; ++i)
            prefill[i].ready_prefill = 1;
        cadence.finish(PDFusionPlan::PREFILL, prefill);
        ASSERT_FALSE(cadence.prefillAdmissionAllowed());
        for (int i = 0; i < ranks; ++i)
            states[i].last_model_us = 700000 - i * 10000;
        ASSERT_EQ(cadence.choose(states), PDFusionPlan::DECODE);
        EXPECT_EQ(cadence.decision().decode_debt_us, 233333);
        EXPECT_FALSE(cadence.prefillAdmissionAllowed());
        for (int round = 0; round < 6; ++round) {
            cadence.finish(PDFusionPlan::DECODE, states);
            for (int i = 0; i < ranks; ++i)
                states[i].last_model_us = 40000 - i * 1000;
            ASSERT_EQ(cadence.choose(states), PDFusionPlan::DECODE);
            EXPECT_EQ(cadence.prefillAdmissionAllowed(), round >= 4);
            EXPECT_EQ(cadence.decision().admit_credit_after_scaled, 3000000);
        }
        // A prepared reservation is NEVER suppressed by the admission gate.
        cadence.finish(PDFusionPlan::PREFILL, prefill);
        for (int i = 0; i < ranks; ++i) {
            states[i].ready_prefill = 1;
            states[i].last_model_us = 700000;
        }
        EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
        EXPECT_STREQ(cadence.decision().reason, "all_ready");
        EXPECT_FALSE(cadence.prefillAdmissionAllowed());
        EXPECT_EQ(cadence.decision().admit_credit_before_scaled, 0);
        EXPECT_EQ(cadence.decision().admit_credit_after_scaled, 0);
    }
}

TEST(PDFusionScheduleCoordinatorTest, AdmissionLookaheadIsOneGlobalDecodeQuantumAndKeepsActualDebt) {
    for (int ranks : {2, 4}) {
        PDFusionGlobalCadence                cadence(7, ranks, adaptiveConfig());
        auto                                 states = decodeStates(128);
        std::array<PDFusionPreparedState, 4> prefill{};
        for (int i = 0; i < ranks; ++i)
            prefill[i].ready_prefill = 1;
        cadence.finish(PDFusionPlan::PREFILL, prefill);
        for (int i = 0; i < ranks; ++i)
            states[i].last_model_us = 700000;
        cadence.choose(states);
        for (int n = 0; n < 5; ++n) {
            cadence.finish(PDFusionPlan::DECODE, states);
            for (int i = 0; i < ranks; ++i)
                states[i].last_model_us = 40000 - i * 1000;
            cadence.choose(states);
        }
        // PREPARE for the next Decode has already happened: admission in the
        // following loop can overlap its still-unreported completed cost.
        EXPECT_TRUE(cadence.prefillAdmissionAllowed());
        EXPECT_EQ(cadence.decision().debt_after_scaled, 2500000);
        EXPECT_EQ(cadence.decision().admit_credit_after_scaled, 3000000);
        cadence.finish(PDFusionPlan::DECODE, states);
        for (int i = 0; i < ranks; ++i) {
            states[i].last_model_us = 10000;
            states[i].ready_prefill = 1;
        }
        EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
        // A shorter actual round cannot erase unpaid debt using the forecast.
        EXPECT_EQ(cadence.decision().debt_after_scaled, 1750000);
        EXPECT_EQ(cadence.decision().admit_credit_after_scaled, 750000);
        EXPECT_FALSE(cadence.prefillAdmissionAllowed());
        cadence.finish(PDFusionPlan::PREFILL, prefill);
        for (int i = 0; i < ranks; ++i) {
            states[i].last_model_us = 700000;
            states[i].ready_prefill = 0;
        }
        cadence.choose(states);
        EXPECT_EQ(cadence.decision().debt_after_scaled, 19250000);
        EXPECT_EQ(cadence.decision().admit_credit_before_scaled, 0);
        EXPECT_FALSE(cadence.prefillAdmissionAllowed());
    }
}

TEST(PDFusionScheduleCoordinatorTest, AdaptiveAdmissionBudgetDoesNotBankIdleOrCountCancelledWork) {
    PDFusionGlobalCadence cadence(7, 4, adaptiveConfig());
    auto                  states = decodeStates(128);
    for (auto& s : states)
        s.last_model_us = 40000;
    for (int i = 0; i < 1000; ++i) {
        cadence.finish(PDFusionPlan::DECODE, states);
        cadence.choose(states);
    }
    auto prefill             = std::array<PDFusionPreparedState, 4>{};
    prefill[0].ready_prefill = 1;  // Matching fakes still execute the same global round.
    cadence.finish(PDFusionPlan::PREFILL, prefill);
    for (auto& s : states)
        s.last_model_us = 700000;
    cadence.choose(states);
    EXPECT_EQ(cadence.decision().decode_debt_us, 193333);
    const auto debt = cadence.decision().decode_debt_us;
    cadence.finish(PDFusionPlan::PREFILL, 0);
    cadence.choose(states);
    EXPECT_EQ(cadence.decision().decode_debt_us, debt);
    EXPECT_FALSE(cadence.prefillAdmissionAllowed());
    states = {};
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::IDLE);
    EXPECT_TRUE(cadence.prefillAdmissionAllowed());
    states[0].ready_prefill = 1;
    EXPECT_EQ(cadence.choose(states), PDFusionPlan::PREFILL);
}

TEST(PDFusionScheduleCoordinatorTest, AdmissionPacingCanBeDisabledAndLegacyIsUnchanged) {
    auto config                 = adaptiveConfig();
    config.min_decode_share_pct = 0;
    PDFusionGlobalCadence adaptive(7, 4, config), legacy(7);
    for (auto* cadence : {&adaptive, &legacy}) {
        cadence->finish(PDFusionPlan::PREFILL, 4);
        auto states = decodeStates(128);
        for (auto& s : states)
            s.last_model_us = 700000;
        cadence->choose(states);
        EXPECT_TRUE(cadence->prefillAdmissionAllowed());
        EXPECT_EQ(cadence->decision().decode_debt_us, 0);
    }
}
}  // namespace

TEST(PDFusionScheduleCoordinatorTest, AllTwoRankMtpWidthsAgreeOnPlan) {
    for (int width = 0; width <= 4; ++width) {
        SCOPED_TRACE(width);
        exercise("normal", true, 2, width);
        exercise("late", true, 2, width);
        exercise("mtp_mismatch", false, 2, width);
    }
}

TEST(PDFusionScheduleCoordinatorTest, AdaptiveSelectsCoordinatorWithoutLegacyModeOrRatio) {
    CadenceEnvScope flag("PDFUSION_CADENCE_ADAPTIVE");
    auto            config = PDFusionCadenceConfig::fromEnv();
    EXPECT_EQ(config.adaptive, 0);
    EXPECT_FALSE(config.usesCoordinator("off", ""));
    EXPECT_FALSE(config.usesCoordinator("off", "ratio"));
    EXPECT_TRUE(config.usesCoordinator("cadence", "ratio"));
    EXPECT_THROW(config.usesCoordinator("cadence", ""), std::runtime_error);
    EXPECT_EQ(config.decodeSteps("7"), 7);
    EXPECT_THROW(config.decodeSteps("1/7"), std::invalid_argument);
    ::setenv("PDFUSION_CADENCE_ADAPTIVE", "1", 1);
    config = PDFusionCadenceConfig::fromEnv();
    for (const auto& mode : {"", "ratio"}) {
        EXPECT_TRUE(config.usesCoordinator("off", mode));
        EXPECT_TRUE(config.usesCoordinator("cadence", mode));
    }
    for (const auto& ratio : {"", "0", "7", "1/7", "not-a-ratio", "9999999999999999999999"}) {
        EXPECT_EQ(config.decodeSteps(ratio), 1);
    }
    EXPECT_THROW(config.usesCoordinator("unknown", ""), std::runtime_error);
}

TEST(PDFusionScheduleCoordinatorTest, AdaptiveHandshakeIgnoresUnusedLegacyNForAllRanksAndMtpWidths) {
    exercise("adaptive_legacy_ratio_ignored", true);
    for (int width = 0; width <= 4; ++width) {
        exercise("adaptive_legacy_ratio_ignored", true, 2, width);
    }
    exercise("mismatch", false);
    exercise("mismatch", false, 2, 1);
}

TEST(PDFusionScheduleCoordinatorTest, AdaptivePlanIgnoresLegacyNWhileDisabledCadenceRetainsIt) {
    auto states                = decodeStates(128);
    states[0].ready_prefill    = 1;
    auto config                = adaptiveConfig();
    config.prefill_max_wait_ms = 0;
    for (int64_t unused : {-1, 0, 7, 1000001}) {
        PDFusionGlobalCadence cadence(unused, 4, config);
        EXPECT_EQ(cadence.choose(states), PDFusionPlan::DECODE);
        EXPECT_STREQ(cadence.decision().reason, "accumulate");
    }
    PDFusionGlobalCadence legacy(7);
    for (int i = 0; i < 7; ++i) {
        legacy.finish(PDFusionPlan::DECODE, decodeStates(128));
    }
    EXPECT_EQ(legacy.choose(states), PDFusionPlan::PREFILL);
    EXPECT_THROW(PDFusionGlobalCadence(0), std::runtime_error);
}
