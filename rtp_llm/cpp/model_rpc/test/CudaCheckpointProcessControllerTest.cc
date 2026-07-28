#include "rtp_llm/cpp/model_rpc/CudaCheckpointProcessController.h"

#include <deque>
#include <memory>
#include <string>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

constexpr int kRunning      = static_cast<int>(CudaCheckpointProcessState::RUNNING);
constexpr int kLocked       = static_cast<int>(CudaCheckpointProcessState::LOCKED);
constexpr int kCheckpointed = static_cast<int>(CudaCheckpointProcessState::CHECKPOINTED);

class FakeCudaCheckpointDriver final : public CudaCheckpointDriver {
public:
    bool ensureLoaded(std::string* error) override {
        ++load_calls;
        if (!loaded && error != nullptr) {
            *error = load_error;
        }
        return loaded;
    }

    int getState(int pid, int* output_state) override {
        ++get_state_calls;
        last_pid = pid;
        const int result = popResult(get_state_results);
        if (result == 0) {
            *output_state = state;
        }
        return result;
    }

    int lock(int pid, uint32_t timeout_ms) override {
        ++lock_calls;
        last_pid         = pid;
        last_timeout_ms  = timeout_ms;
        const int result = lock_result;
        if (result == 0 && transition_state) {
            state = kLocked;
        }
        return result;
    }

    int checkpoint(int pid) override {
        ++checkpoint_calls;
        last_pid         = pid;
        const int result = checkpoint_result;
        if (result == 0 && transition_state) {
            state = kCheckpointed;
        }
        return result;
    }

    int restore(int pid) override {
        ++restore_calls;
        last_pid         = pid;
        const int result = restore_result;
        if (result == 0 && transition_state) {
            state = kLocked;
        }
        return result;
    }

    int unlock(int pid) override {
        ++unlock_calls;
        last_pid         = pid;
        const int result = unlock_result;
        if (result == 0 && transition_state) {
            state = kRunning;
        }
        return result;
    }

    static int popResult(std::deque<int>& results) {
        if (results.empty()) {
            return 0;
        }
        const int result = results.front();
        results.pop_front();
        return result;
    }

    bool            loaded{true};
    std::string     load_error{"driver unavailable"};
    int             state{kRunning};
    bool            transition_state{true};
    std::deque<int> get_state_results;
    int             lock_result{0};
    int             checkpoint_result{0};
    int             restore_result{0};
    int             unlock_result{0};
    int             load_calls{0};
    int             get_state_calls{0};
    int             lock_calls{0};
    int             checkpoint_calls{0};
    int             restore_calls{0};
    int             unlock_calls{0};
    int             last_pid{-1};
    uint32_t        last_timeout_ms{0};
};

class CudaCheckpointProcessControllerTest : public ::testing::Test {
protected:
    CudaCheckpointCommandResult execute(const std::string& action,
                                        const std::string& transaction_id  = "transaction-7",
                                        int64_t            sleep_epoch     = 7,
                                        uint32_t           timeout_ms      = 25000,
                                        bool               backend_sleeping = true,
                                        int64_t            backend_epoch    = 7) {
        return controller.execute(
            CudaCheckpointCommand{action, transaction_id, sleep_epoch, timeout_ms},
            kPid,
            backend_epoch,
            backend_sleeping);
    }

    static constexpr int kPid = 4321;
    std::shared_ptr<FakeCudaCheckpointDriver> driver = std::make_shared<FakeCudaCheckpointDriver>();
    CudaCheckpointProcessController           controller{driver};
};

TEST_F(CudaCheckpointProcessControllerTest, RunsFullLifecycleAndMakesRetriesIdempotent) {
    auto result = execute("GET_STATE", "", -1);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.state, "RUNNING");
    EXPECT_EQ(result.transaction_id, "");
    EXPECT_EQ(result.sleep_epoch, -1);

    result = execute("LOCK");
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.state, "LOCKED");
    EXPECT_EQ(result.transaction_id, "transaction-7");
    EXPECT_EQ(result.sleep_epoch, 7);
    EXPECT_EQ(driver->last_timeout_ms, 25000);

    ASSERT_TRUE(execute("LOCK").success);
    ASSERT_TRUE(execute("CHECKPOINT").success);
    ASSERT_TRUE(execute("CHECKPOINT").success);
    ASSERT_TRUE(execute("RESTORE").success);
    ASSERT_TRUE(execute("RESTORE").success);
    ASSERT_TRUE(execute("UNLOCK").success);
    result = execute("UNLOCK");
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.state, "RUNNING");

    EXPECT_EQ(driver->lock_calls, 1);
    EXPECT_EQ(driver->checkpoint_calls, 1);
    EXPECT_EQ(driver->restore_calls, 1);
    EXPECT_EQ(driver->unlock_calls, 1);
    EXPECT_EQ(driver->last_pid, kPid);
}

TEST_F(CudaCheckpointProcessControllerTest, UsesDefaultLockTimeoutWhenRequestUsesZero) {
    const auto result = execute("LOCK", "transaction-7", 7, 0);
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(driver->last_timeout_ms, 10000);
}

TEST_F(CudaCheckpointProcessControllerTest, RejectsInvalidOrUnsafeMutationsBeforeCallingDriverOperation) {
    auto result = execute("INVALID");
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error.find("invalid CUDA checkpoint action"), std::string::npos);

    result = execute("LOCK", "", 7);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error.find("require transaction_id"), std::string::npos);

    result = execute("LOCK", "transaction-7", 7, 25000, false);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error.find("backend state SLEEPING"), std::string::npos);

    result = execute("LOCK", "transaction-7", 6, 25000, true, 7);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error.find("sleep_epoch differs"), std::string::npos);

    EXPECT_EQ(driver->lock_calls, 0);
    EXPECT_EQ(driver->checkpoint_calls, 0);
    EXPECT_EQ(driver->restore_calls, 0);
    EXPECT_EQ(driver->unlock_calls, 0);
}

TEST_F(CudaCheckpointProcessControllerTest, RejectsAStaleTransactionAfterLockOwnsRank) {
    ASSERT_TRUE(execute("LOCK").success);

    const auto result = execute("CHECKPOINT", "stale-transaction");
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.error.find("does not own this backend rank"), std::string::npos);
    EXPECT_EQ(result.transaction_id, "transaction-7");
    EXPECT_EQ(result.sleep_epoch, 7);
    EXPECT_EQ(driver->checkpoint_calls, 0);
    EXPECT_EQ(driver->state, kLocked);
}

TEST_F(CudaCheckpointProcessControllerTest, PreservesNewTransactionIdentityWhenLockFails) {
    driver->lock_result = 801;

    auto result = execute("LOCK");
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.cuda_result, 801);
    EXPECT_EQ(result.transaction_id, "transaction-7");
    EXPECT_EQ(result.sleep_epoch, 7);
    EXPECT_NE(result.error.find("LOCK failed"), std::string::npos);
    EXPECT_EQ(driver->state, kRunning);

    driver->lock_result = 0;
    result              = execute("LOCK");
    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(driver->lock_calls, 2);
    EXPECT_EQ(driver->state, kLocked);
}

TEST_F(CudaCheckpointProcessControllerTest, ReportsDriverLoadAndGetStateFailures) {
    driver->loaded     = false;
    driver->load_error = "missing checkpoint symbols";
    auto result        = execute("GET_STATE");
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.cuda_result, -1);
    EXPECT_EQ(result.state, "UNKNOWN");
    EXPECT_EQ(result.error, "missing checkpoint symbols");

    driver->loaded = true;
    driver->get_state_results.push_back(999);
    result = execute("GET_STATE");
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.cuda_result, 999);
    EXPECT_NE(result.error.find("cuCheckpointProcessGetState failed"), std::string::npos);
}

TEST_F(CudaCheckpointProcessControllerTest, ReportsPostActionQueryAndStateTransitionFailures) {
    driver->get_state_results = {0, 777};
    auto result               = execute("LOCK");
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.cuda_result, 777);
    EXPECT_NE(result.error.find("post-action cuCheckpointProcessGetState failed"), std::string::npos);

    auto second_driver              = std::make_shared<FakeCudaCheckpointDriver>();
    second_driver->transition_state = false;
    CudaCheckpointProcessController second_controller{second_driver};
    result = second_controller.execute(
        CudaCheckpointCommand{"LOCK", "transaction-8", 8, 10000}, kPid, 8, true);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.cuda_result, 0);
    EXPECT_EQ(result.state, "RUNNING");
    EXPECT_NE(result.error.find("without reaching the expected"), std::string::npos);
}

TEST(CudaCheckpointProcessStateTest, MapsKnownAndUnknownStates) {
    EXPECT_STREQ(CudaCheckpointProcessController::stateName(kRunning), "RUNNING");
    EXPECT_STREQ(CudaCheckpointProcessController::stateName(kLocked), "LOCKED");
    EXPECT_STREQ(CudaCheckpointProcessController::stateName(kCheckpointed), "CHECKPOINTED");
    EXPECT_STREQ(CudaCheckpointProcessController::stateName(3), "FAILED");
    EXPECT_STREQ(CudaCheckpointProcessController::stateName(99), "UNKNOWN");
}

}  // namespace
}  // namespace rtp_llm
